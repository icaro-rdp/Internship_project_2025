import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import numpy as np
import gc
import time
import re
import json
import shutil
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Sequence, Tuple, Optional

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor,
)
from main.Utils.explainability import GradCAM, MultiscalePixelMasking
from main.Utils.cleanup import clear_gpu_memory, cleanup_model_and_data
from main.Utils.logger import info, warn, error, set_level
from main.Utils.config import get_xai_config, get_data_config, get_model_order
from main.Utils.comparisons import (
    compare_heatmaps,
    uniform_heatmaps,
)
from main.Utils.visualization import (
    visualize_similarity_matrix,
    visualize_similarity_distribution,
    visualize_violin_distribution,
)
from main.data import IMAGENET_DATASET, DENSENET_DATASET

# Load config
_data_cfg = get_data_config()
SINGLE_BATCH_SIZE = _data_cfg["single_batch_size"]
NUM_WORKERS = _data_cfg["num_workers"]

DIRS = {
    "output": Path("Outputs/Experiment_2_variants"),
    "weights": Path("Outputs/Experiment_1_variants/Weights"),
}
DIRS["maps"] = DIRS["output"] / "XAI_Maps"
DIRS["gradcam"] = DIRS["maps"] / "GradCAM"
DIRS["mpm"] = DIRS["maps"] / "Multiscale_Pixel_Masking"
DIRS["plots"] = DIRS["output"] / "Plots"

MODEL_REGISTRY = {
    "vgg16": {
        "class": VGG16AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "target_layer": "features.28",
    },
    "vgg19": {
        "class": VGG19AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "target_layer": "features.34",
    },
    "resnet152": {
        "class": ResNet152AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "target_layer": "features.7.2.conv3",
    },
    "densenet161": {
        "class": DenseNet161AuthenticityPredictor,
        "dataset": DENSENET_DATASET,
        "target_layer": "features.denseblock4.denselayer24.conv2",
    },
    "efficientnetb3": {
        "class": EfficientNetB3AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "target_layer": "features.8.0",
    },
    "barlowtwins": {
        "class": BarlowTwinsAuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "target_layer": "features.7.2.conv3",
    },
}

# XAI parameters - loaded from config
XAI_PARAMS = get_xai_config()

# Model order for visualization - loaded from config
MODEL_ORDER = get_model_order()


def setup_directories():
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 2. JSON ENCODER (for numpy types)
# ============================================================================


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# ============================================================================
# 3. GENERATION LOGIC
# ============================================================================


def get_weight_files(models_filter, variants_filter):
    if not DIRS["weights"].exists():
        return {}
    all_files = sorted(DIRS["weights"].glob("*.pth"))
    grouped = defaultdict(list)

    if isinstance(variants_filter, str):
        variants_filter = {variants_filter}
    req_vars = {str(v).lower() for v in variants_filter}
    include_all = "all" in req_vars

    for p in all_files:
        match = re.match(r"^([A-Za-z0-9_]+)_exp1", p.name)
        if not match:
            continue
        m_name = match.group(1)
        if models_filter and m_name not in models_filter:
            continue
        if m_name not in MODEL_REGISTRY:
            continue

        tag = "orig"
        if "greedy" in str(p):
            tag = re.search(r"exp1b_variant\d+_greedy_pruned", str(p)).group(0)
        elif "negative" in str(p):
            tag = re.search(r"exp1b_variant\d+_negative_pruned", str(p)).group(0)
        elif "variant" in str(p):
            tag = re.search(r"exp1a_variant\d+", str(p)).group(0)

        keep = include_all
        if not keep:
            if "greedy" in req_vars and "greedy" in tag:
                keep = True
            elif "negative" in req_vars and "negative" in tag:
                keep = True
            elif "orig" in req_vars and "orig" in tag:
                keep = True
            elif "base" in req_vars and ("orig" in tag or "exp1a" in tag):
                keep = True

        if keep:
            grouped[m_name].append(p)
    return grouped


def run_generation_loop(
    model, loader, device, model_name, variant_tag, method, save_maps
):
    if not save_maps:
        return
    is_gc = method == "gradcam"
    out_dir = DIRS["gradcam"] if is_gc else DIRS["mpm"]
    stem = f"{model_name}_{variant_tag}"
    final_path = out_dir / f"{stem}_maps.npy"

    if final_path.exists():
        info(f"Map {final_path} exists. Skipping.")
        return

    temp_dir = out_dir / f"temp_{stem}"
    temp_dir.mkdir(exist_ok=True)

    batch_maps = []
    chunk_idx = 0
    flush_int = XAI_PARAMS["gc_interval"] if is_gc else XAI_PARAMS["mpm_interval"]

    try:
        info(f"Generating {method} for {stem}...")
        for i, (img, _) in enumerate(loader):
            img = img.to(device)
            if is_gc:
                cam = GradCAM(model, MODEL_REGISTRY[model_name]["target_layer"], False)
                res = cam.generate_map(img, 0)
                cam.cleanup()
            else:
                mpm = MultiscalePixelMasking(
                    model, XAI_PARAMS["sigma"], XAI_PARAMS["px_batch"], 0
                )
                res = mpm.generate_map(img, 0)

            if isinstance(res, torch.Tensor):
                res = res.cpu().numpy()
            batch_maps.append(res)

            if len(batch_maps) >= flush_int:
                np.save(temp_dir / f"part_{chunk_idx:04d}.npy", np.array(batch_maps))
                batch_maps = []
                chunk_idx += 1
                gc.collect()

        if batch_maps:
            np.save(temp_dir / f"part_{chunk_idx:04d}.npy", np.array(batch_maps))

        parts = sorted(temp_dir.glob("*.npy"))
        if parts:
            full_arr = np.concatenate([np.load(p) for p in parts], axis=0)
            np.save(final_path, full_arr)
            info(f"Saved {final_path} shape: {full_arr.shape}")
            shutil.rmtree(temp_dir)

    except Exception as e:
        error(f"Gen Error {stem}: {e}")


# ============================================================================
# 4. COMPARISON LOGIC
# ============================================================================


def load_and_resize_map(path, target_res):
    try:
        arr = np.load(path)
        resized = uniform_heatmaps(arr, target_res[0], target_res[1])
        return resized
    except Exception as e:
        error(f"Failed to load/resize {path}: {e}")
        return None


def save_plots_for_result(comp_res, labels, method, scope_name, metrics):
    """Save visualization plots with consistent model ordering."""
    # Sort labels according to MODEL_ORDER
    ordered_labels = sorted(
        labels,
        key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else len(MODEL_ORDER),
    )

    for metric in metrics:
        if metric not in comp_res["summary"]:
            continue
        try:
            fig_mat = visualize_similarity_matrix(
                comp_res, ordered_labels, metric=metric, add_title=False
            )
            if fig_mat:
                out_name = f"{method}_{scope_name}_{metric}_matrix"
                fig_mat.savefig(
                    DIRS["plots"] / f"{out_name}.svg", bbox_inches="tight", format="svg"
                )
                fig_mat.savefig(
                    DIRS["plots"] / f"{out_name}.png", bbox_inches="tight", format="png"
                )

                plt.close(fig_mat)

            fig_dist = visualize_similarity_distribution(
                comp_res, metric=metric, add_title=False
            )
            if fig_dist and scope_name not in [
                "between_model_architectures",
                "cross_methods",
            ]:
                out_name = f"{method}_{scope_name}_{metric}_distribution"
                fig_dist.savefig(
                    DIRS["plots"] / f"{out_name}.svg", bbox_inches="tight", format="svg"
                )
                fig_dist.savefig(
                    DIRS["plots"] / f"{out_name}.png", bbox_inches="tight", format="png"
                )
                plt.close(fig_dist)
        except Exception as e:
            warn(f"Could not save plots for {method} {scope_name} ({metric}): {e}")


def run_comparisons(methods, kinds, metrics, target_res, models_filter, save_json):
    results = {}

    def get_file_map(method):
        d = DIRS["gradcam"] if method == "gradcam" else DIRS["mpm"]
        if not d.exists():
            return {}
        g = defaultdict(dict)
        for f in d.glob("*_maps.npy"):
            stem = f.stem.replace("_maps", "")
            if "_" not in stem:
                continue
            parts = stem.split("_", 1)
            if len(parts) < 2:
                continue
            m_name, var = parts
            if models_filter and m_name not in models_filter:
                continue
            g[m_name][var] = f
        return g

    for method in methods:
        groups = get_file_map(method)
        if not groups:
            continue

        prototypes = {}
        valid_protos = []
        within_res = {}

        # Container for violin plot data: {metric: {model_name: [values]}}
        within_model_stats = defaultdict(lambda: defaultdict(list))

        # --- 1. Intra-Model & Prototype Creation ---
        for m_name, variants_dict in groups.items():
            var_names = list(variants_dict.keys())
            var_paths = list(variants_dict.values())

            info(f"[{method}] Loading {len(var_paths)} variants for {m_name}...")
            loaded_vars = []
            for p in var_paths:
                arr = load_and_resize_map(p, target_res)
                if arr is not None:
                    loaded_vars.append(arr)

            if not loaded_vars:
                continue

            # A. Within-Model Variants
            if "within_model_variants" in kinds and len(loaded_vars) >= 2:
                info(f"[{method}] Comparing variants for {m_name}...")
                comp_res = compare_heatmaps(loaded_vars, metrics=metrics)
                comp_res["variants"] = var_names
                within_res[m_name] = comp_res

                # Collect data for violin plots
                for metric in metrics:
                    if metric in comp_res["summary"]:
                        # Extract 'mean' values from pairwise comparisons
                        values = [
                            d["mean"]
                            for d in comp_res["summary"][metric].values()
                            if "mean" in d
                        ]
                        within_model_stats[metric][m_name].extend(values)

                save_plots_for_result(
                    comp_res, var_names, method, f"within_{m_name}", metrics
                )

            # B. Prototype
            if "between_model_architectures" in kinds:
                stack = np.stack(loaded_vars)
                proto = np.mean(stack, axis=0)
                prototypes[m_name] = proto
                valid_protos.append(m_name)

            del loaded_vars
            gc.collect()

        # Generate Violin Plots for Intra-Model Analysis
        if within_model_stats:
            for metric, model_data in within_model_stats.items():
                try:
                    fig_violin = visualize_violin_distribution(
                        model_data,
                        metric=metric,
                        add_title=False,
                        custom_model_order=[
                            "barlowtwins",
                            "resnet152",
                            "densenet161",
                            "efficientnetb3",
                            "vgg19",
                            "vgg16",
                        ],
                    )
                    if fig_violin:
                        out_name = f"{method}_within_model_{metric}_violin"
                        fig_violin.savefig(
                            DIRS["plots"] / f"{out_name}.svg",
                            bbox_inches="tight",
                            format="svg",
                        )
                        fig_violin.savefig(
                            DIRS["plots"] / f"{out_name}.png",
                            bbox_inches="tight",
                            format="png",
                        )
                        plt.close(fig_violin)
                        info(f"Saved violin plot: {out_name}")
                except Exception as e:
                    warn(f"Could not save violin plot for {method} ({metric}): {e}")

        if within_res:
            results[f"{method}_within_model_variants"] = within_res

        # --- 2. Between-Model Architectures Comparison ---
        if "between_model_architectures" in kinds and len(prototypes) > 1:
            info(f"[{method}] Comparing Prototypes...")
            # Sort valid_protos according to MODEL_ORDER
            valid_protos_sorted = sorted(
                valid_protos,
                key=lambda x: (
                    MODEL_ORDER.index(x) if x in MODEL_ORDER else len(MODEL_ORDER)
                ),
            )
            proto_list = [prototypes[m] for m in valid_protos_sorted]

            comp_res = compare_heatmaps(proto_list, metrics=metrics)
            comp_res["models"] = valid_protos_sorted
            results[f"{method}_between_model_architectures"] = comp_res

            save_plots_for_result(
                comp_res,
                valid_protos_sorted,
                method,
                "between_model_architectures",
                metrics,
            )

            del prototypes, proto_list
            gc.collect()

    # --- 3. Cross-Method Comparison ---
    if "cross_methods" in kinds and "gradcam" in methods and "mpm" in methods:
        info("[Cross-Methods] Comparing GradCAM vs MPM...")
        gc_files = get_file_map("gradcam")
        mpm_files = get_file_map("mpm")
        cross_res = {}

        for m_name in gc_files:
            if m_name not in mpm_files:
                continue
            for var in gc_files[m_name]:
                if var not in mpm_files[m_name]:
                    continue

                p1 = gc_files[m_name][var]
                p2 = mpm_files[m_name][var]

                arr1 = load_and_resize_map(p1, target_res)
                arr2 = load_and_resize_map(p2, target_res)

                if arr1 is not None and arr2 is not None:
                    res = compare_heatmaps([arr1, arr2], metrics=metrics)
                    cross_res[f"{m_name}_{var}"] = res

                del arr1, arr2
                gc.collect()

        if cross_res:
            results["cross_methods"] = cross_res

    # Save JSON
    if save_json:
        out_path = DIRS["output"] / "experiment_2b_comparison.json"
        try:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
            info(f"Saved results to {out_path}")
        except Exception as e:
            error(f"JSON Save Failed: {e}")
            import traceback

            traceback.print_exc()


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================


def run_experiment_2(
    models=None,
    save_maps=True,
    variants="all",
    xai_methods="both",
    comparison_only=False,
    run_comparison=False,
    comparison_kinds=("between_model_architectures",),
    comparison_metrics=("correlation",),
    comparison_target_resolution=(224, 224),
    save_comparison_json=True,
):
    start = time.time()
    setup_directories()
    if isinstance(models, str):
        models = [models]

    methods = ["gradcam", "mpm"] if xai_methods in ["both", "all"] else [xai_methods]

    # --- Part A: Generation ---
    if not comparison_only:
        info(">>> STARTING GENERATION")
        queue = get_weight_files(models, variants)
        for m_name, paths in queue.items():
            info(f"Processing {m_name}")
            m_cls = MODEL_REGISTRY[m_name]["class"]

            for w_path in paths:
                tag_match = re.search(
                    r"exp1a_variant\d+|exp1b_variant\d+_greedy_pruned|exp1b_variant\d+_negative_pruned|orig",
                    str(w_path),
                )
                tag = tag_match.group(0) if tag_match else "orig"

                model = m_cls(freeze_backbone=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.load_state_dict(
                    torch.load(w_path, map_location=device, weights_only=True)
                )
                model.to(device).eval()

                ds = MODEL_REGISTRY[m_name]["dataset"]["test"]
                loader = DataLoader(
                    ds,
                    batch_size=SINGLE_BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )

                for meth in methods:
                    run_generation_loop(
                        model, loader, device, m_name, tag, meth, save_maps
                    )

                cleanup_model_and_data(model)

    # --- Part B: Comparison ---
    # Run comparison if comparison_only=True OR if run_comparison=True (generation + comparison)
    if comparison_only or run_comparison:
        info(">>> STARTING COMPARISONS ")
        run_comparisons(
            methods,
            comparison_kinds,
            comparison_metrics,
            comparison_target_resolution,
            models,
            save_comparison_json,
        )

    info(f"Done. Time: {time.time()-start:.2f}s")


if __name__ == "__main__":
    set_level("INFO")
    run_experiment_2(
        models=[
            "barlowtwins",
            "resnet152",
            "densenet161",
            "efficientnetb3",
            "vgg16",
            "vgg19",
        ],
        xai_methods="both",
        comparison_only=True,
        comparison_kinds=("between_model_architectures", "within_model_variants"),
        comparison_metrics=(
            "correlation",
            "top_percent_iou_5",
            "top_percent_iou_15",
            "top_percent_iou_25",
        ),
        save_comparison_json=True,
    )
