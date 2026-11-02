import gc
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from main.data import CLIP_DATASET, DENSENET_DATASET, IMAGENET_DATASET
from main.Models import (
    BarlowTwinsAuthenticityPredictor,
    CLIPViTAuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
)
from main.Utils.cleanup import cleanup_model_and_data
from main.Utils.comparisons import (
    compare_heatmaps,
    compare_lime_beta_collections,
    uniform_heatmaps,
)
from main.Utils.config import get_data_config, get_model_order, get_xai_config
from main.Utils.explainability import (
    LIME,
    GradCAM,
    MultiscalePixelMasking,
    compute_slic_segments,
)
from main.Utils.logger import error, info, set_level, warn
from main.Utils.visualization import (
    plot_lime_r2_distributions,
    visualize_similarity_distribution,
    visualize_similarity_matrix,
    visualize_violin_distribution,
)

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
DIRS["lime"] = DIRS["maps"] / "LIME"
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
    "clip_vit_b16": {
        "class": CLIPViTAuthenticityPredictor,
        "dataset": CLIP_DATASET,
        "target_layer": "vision_model.encoder.layers.11.layer_norm1",
    },
}

# XAI parameters - loaded from config
XAI_PARAMS = get_xai_config()

# Model order for visualization - loaded from config
MODEL_ORDER = get_model_order()


def save_test_samples_manifest() -> Path:
    """Save an immutable manifest linking test split indices to image filenames and scores."""
    manifest_path = DIRS["maps"] / "test_samples_manifest.json"
    if manifest_path.exists():
        return manifest_path

    test_ds = CLIP_DATASET["test"]
    manifest = []
    for test_idx in range(len(test_ds)):
        global_idx = int(test_ds.indices[test_idx])
        row = test_ds.dataset.data.iloc[global_idx]
        img_raw = str(row.iloc[3])
        manifest.append(
            {
                "test_idx": test_idx,
                "global_csv_idx": global_idx,
                "image_path": img_raw,
                "image_name": Path(img_raw).name,
                "authenticity_score": float(row.iloc[1]),
            }
        )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    info(f"Saved test samples manifest: {manifest_path} ({len(manifest)} items)")
    return manifest_path


def setup_directories():
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)
    save_test_samples_manifest()


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
        return super().default(obj)


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
        if not keep and (
            (tag.lower() in req_vars)
            or ("greedy" in req_vars and "greedy" in tag)
            or ("negative" in req_vars and "negative" in tag)
            or ("orig" in req_vars and "orig" in tag)
            or (
                ("base" in req_vars or "unpruned" in req_vars)
                and ("orig" in tag or "exp1a" in tag)
            )
        ):
            keep = True

        if keep:
            grouped[m_name].append(p)
    return grouped


def _resolve_methods(
    xai_methods: str | list[str] | tuple[str, ...] | set[str],
) -> list[str]:
    """Normalize xai_methods into an ordered list of canonical method names."""
    raw = [xai_methods] if isinstance(xai_methods, str) else list(xai_methods)
    resolved: list[str] = []
    for item in raw:
        token = str(item).lower()
        if token == "both":
            for m in ("gradcam", "mpm"):
                if m not in resolved:
                    resolved.append(m)
        elif token == "all":
            for m in ("gradcam", "mpm", "lime"):
                if m not in resolved:
                    resolved.append(m)
        elif token in ("gradcam", "mpm", "lime"):
            if token not in resolved:
                resolved.append(token)
        else:
            raise ValueError(
                f"Unknown XAI method '{item}'. Expected one of: gradcam, mpm, lime, both, all."
            )
    return resolved


def _method_dir(method: str) -> Path:
    """Return the XAI map directory corresponding to ``method``."""
    if method == "gradcam":
        return DIRS["gradcam"]
    if method == "mpm":
        return DIRS["mpm"]
    if method == "lime":
        return DIRS["lime"]
    raise ValueError(f"Unsupported XAI method: {method}")


def _map_output_path(model_name: str, variant_tag: str, method: str) -> Path:
    """Return the final .npy output path for a (model, variant, method) tuple."""
    out_dir = _method_dir(method)
    return out_dir / f"{model_name}_{variant_tag}_maps.npy"


def _lime_r2_output_path(model_name: str, variant_tag: str) -> Path:
    """Return the per-image weighted R2 .npy path for a LIME (model, variant) run."""
    return DIRS["lime"] / f"{model_name}_{variant_tag}_r2.npy"


def _lime_betas_output_path(model_name: str, variant_tag: str) -> Path:
    """Return the per-image superpixel beta .npz path for a LIME (model, variant) run."""
    return DIRS["lime"] / f"{model_name}_{variant_tag}_betas.npz"


def prepare_or_load_slic_segments(
    n_segments: int | None = None,
    compactness: float | None = None,
) -> np.ndarray:
    """Precompute or load the shared canonical 224x224 SLIC tessellation for the test split.

    Matches ``prepare_or_load_tessellation_store`` in ``lime_run.py`` by applying
    ``Resize(256) + CenterCrop(224)`` and ``skimage.segmentation.slic`` on
    ``img_canon_np.astype(np.float32) / 255.0``.

    Args:
        n_segments: Target number of superpixels (defaults to config ``lime_n_segments``).
        compactness: SLIC compactness parameter (defaults to config ``lime_compactness``).

    Returns:
        3D ``int16`` array of shape ``(N_test, 224, 224)``.
    """
    n_seg = int(
        n_segments if n_segments is not None else XAI_PARAMS.get("lime_n_segments", 150)
    )
    comp = float(
        compactness
        if compactness is not None
        else XAI_PARAMS.get("lime_compactness", 10.0)
    )

    DIRS["lime"].mkdir(parents=True, exist_ok=True)
    seg_path = DIRS["lime"] / "slic_segments.npy"
    meta_path = DIRS["lime"] / "tessellation_manifest.json"

    test_subset = IMAGENET_DATASET["test"]
    base_ds = test_subset.dataset
    test_indices = list(test_subset.indices)
    canon_size = 224

    if seg_path.exists() and meta_path.exists():
        loaded = np.load(seg_path)
        if loaded.shape == (len(test_indices), canon_size, canon_size):
            info(f"Loaded precomputed SLIC tessellation from {seg_path}")
            return loaded.astype(np.int16)

    info(
        f"Building canonical {canon_size}x{canon_size} SLIC tessellation for {len(test_indices)} test images..."
    )
    canon_crop = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(canon_size),
        ]
    )

    tess = np.zeros((len(test_indices), canon_size, canon_size), dtype=np.int16)
    stems: list[str] = []

    for i, idx in enumerate(test_indices):
        img_raw = str(base_ds.data.iloc[int(idx), 3])
        img_path = (
            (base_ds.project_root / Path(img_raw)).resolve()
            if not Path(img_raw).is_absolute()
            else Path(img_raw)
        )
        stems.append(img_path.stem)
        img_pil = Image.open(img_path).convert("RGB")
        img_canon_np = np.array(canon_crop(img_pil), dtype=np.uint8)
        tess[i] = compute_slic_segments(
            img_canon_np, n_segments=n_seg, compactness=comp
        )

    np.save(seg_path, tess)
    meta = {
        "N": len(test_indices),
        "H": canon_size,
        "W": canon_size,
        "dtype": "int16",
        "segments_file": str(seg_path),
        "image_stems_in_order": stems,
        "canon_crop": "Resize(256)+CenterCrop(224)",
        "n_segments_requested": n_seg,
        "compactness": comp,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    info(f"Saved precomputed SLIC tessellation to {seg_path}")
    return tess


def run_generation_loop(
    model,
    loader,
    device,
    model_name,
    variant_tag,
    method,
    save_maps,
    cluster_computing=False,
    slic_segments: np.ndarray | None = None,
):
    if not save_maps:
        return

    loader_batch_size = getattr(loader, "batch_size", None)
    if loader_batch_size is not None and loader_batch_size != 1:
        raise ValueError(
            f"run_generation_loop requires loader with batch_size=1, got {loader_batch_size}"
        )

    is_gc = method == "gradcam"
    is_lime = method == "lime"
    out_dir = _method_dir(method)
    stem = f"{model_name}_{variant_tag}"
    final_path = _map_output_path(model_name, variant_tag, method)

    if final_path.exists():
        info(f"Map {final_path} exists. Skipping.")
        return

    temp_dir = out_dir / f"temp_{stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    existing_parts = sorted(temp_dir.glob("part_[0-9][0-9][0-9][0-9].npy"))
    completed_images = 0
    for part_path in existing_parts:
        completed_images += int(np.load(part_path, mmap_mode="r").shape[0])

    batch_maps: list[np.ndarray] = []
    batch_r2: list[float] = []
    batch_betas: dict[str, np.ndarray] = {}
    chunk_idx = len(existing_parts)
    if is_gc:
        flush_int = XAI_PARAMS["gc_interval"]
    elif is_lime:
        flush_int = XAI_PARAMS.get("lime_interval", 25)
    else:
        flush_int = XAI_PARAMS["mpm_interval"]

    if completed_images > 0:
        info(
            f"  [{method}] {stem}: resuming from checkpoint {chunk_idx} "
            f"({completed_images}/{len(loader)} images already saved)"
        )

    px_batch = 512 if cluster_computing else XAI_PARAMS["px_batch"]
    lime_batch = 512 if cluster_computing else XAI_PARAMS.get("lime_batch_size", 64)
    use_amp = cluster_computing
    amp_dtype = torch.bfloat16 if cluster_computing else torch.float32

    if cluster_computing:
        torch.backends.cudnn.benchmark = True

    if is_gc:
        explainer = GradCAM(model, MODEL_REGISTRY[model_name]["target_layer"], False)
    elif is_lime:
        norm_mean = (
            (0.48145466, 0.4578275, 0.40821073)
            if model_name == "clip_vit_b16"
            else (0.485, 0.456, 0.406)
        )
        norm_std = (
            (0.26862954, 0.26130258, 0.27577711)
            if model_name == "clip_vit_b16"
            else (0.229, 0.224, 0.225)
        )
        model_input_size = 300 if model_name == "densenet161" else None
        explainer = LIME(
            model,
            n_segments=XAI_PARAMS.get("lime_n_segments", 150),
            compactness=XAI_PARAMS.get("lime_compactness", 10.0),
            num_perturbations=XAI_PARAMS.get("lime_num_perturbations", 1200),
            keep_prob=XAI_PARAMS.get("lime_keep_prob", 0.7),
            kernel_width=XAI_PARAMS.get("lime_kernel_width", 0.25),
            ridge_alpha=XAI_PARAMS.get("lime_ridge_alpha", 1.0),
            seed=XAI_PARAMS.get("lime_seed", 0),
            batch_size=lime_batch,
            model_input_size=model_input_size,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    else:
        explainer = MultiscalePixelMasking(
            model,
            XAI_PARAMS["sigma"],
            px_batch,
            XAI_PARAMS.get("mask_val", 0),
            stride=XAI_PARAMS.get("stride", 1),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

    failed = False
    try:
        info(f"Generating {method} for {stem}...")
        for _i, (img, _) in enumerate(loader):
            if _i < completed_images:
                continue
            if img.dim() == 4 and img.shape[0] != 1:
                raise ValueError(
                    f"run_generation_loop expects single-image batches (B=1), got shape {img.shape}"
                )
            img = img.to(device, non_blocking=cluster_computing)
            if is_lime and isinstance(explainer, LIME):
                seg_i = (
                    slic_segments[_i]
                    if (
                        slic_segments is not None
                        and _i < slic_segments.shape[0]
                        and slic_segments.shape[1:] == (img.shape[-2], img.shape[-1])
                    )
                    else None
                )
                res = explainer.generate_map(img, 0, segments=seg_i)
                batch_r2.append(
                    float(explainer.last_r2)
                    if explainer.last_r2 is not None
                    else float("nan")
                )
                if explainer.last_beta is not None:
                    batch_betas[f"img_{_i:04d}"] = explainer.last_beta
            else:
                res = explainer.generate_map(img, 0)

            if isinstance(res, torch.Tensor):
                res = res.cpu().numpy()
            batch_maps.append(res)

            if len(batch_maps) >= flush_int:
                np.save(temp_dir / f"part_{chunk_idx:04d}.npy", np.array(batch_maps))
                if is_lime:
                    np.save(
                        temp_dir / f"part_{chunk_idx:04d}_r2.npy",
                        np.array(batch_r2, dtype=np.float32),
                    )
                    np.savez_compressed(
                        temp_dir / f"part_{chunk_idx:04d}_betas.npz", **batch_betas
                    )
                    batch_r2 = []
                    batch_betas = {}
                batch_maps = []
                chunk_idx += 1
                info(
                    f"  [{method}] {stem}: saved checkpoint {chunk_idx} "
                    f"({_i + 1}/{len(loader)} images)"
                )
                gc.collect()

        if batch_maps:
            np.save(temp_dir / f"part_{chunk_idx:04d}.npy", np.array(batch_maps))
            if is_lime:
                np.save(
                    temp_dir / f"part_{chunk_idx:04d}_r2.npy",
                    np.array(batch_r2, dtype=np.float32),
                )
                np.savez_compressed(
                    temp_dir / f"part_{chunk_idx:04d}_betas.npz", **batch_betas
                )

        parts = sorted(temp_dir.glob("part_[0-9][0-9][0-9][0-9].npy"))
        if parts:
            full_arr = np.concatenate([np.load(p) for p in parts], axis=0)
            np.save(final_path, full_arr)
            info(f"Saved {final_path} shape: {full_arr.shape}")

            if is_lime:
                r2_parts = sorted(temp_dir.glob("part_*_r2.npy"))
                if r2_parts:
                    full_r2 = np.concatenate([np.load(p) for p in r2_parts], axis=0)
                    np.save(_lime_r2_output_path(model_name, variant_tag), full_r2)
                beta_parts = sorted(temp_dir.glob("part_*_betas.npz"))
                if beta_parts:
                    merged_betas: dict[str, np.ndarray] = {}
                    for bp in beta_parts:
                        with np.load(bp) as data:
                            for k in data.files:
                                merged_betas[k] = data[k]
                    np.savez_compressed(
                        _lime_betas_output_path(model_name, variant_tag),
                        **merged_betas,
                    )

            shutil.rmtree(temp_dir)

    except Exception as e:
        failed = True
        error(f"Gen Error {stem}: {e}")
        raise
    finally:
        explainer.cleanup(failed=failed)


# ============================================================================
# 4. COMPARISON LOGIC
# ============================================================================


def load_and_resize_map(path, target_res):
    try:
        arr = np.load(path)
        resized = uniform_heatmaps(arr, target_res[0], target_res[1])
        return resized
    except (OSError, ValueError, RuntimeError) as e:
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
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
            warn(f"Could not save plots for {method} {scope_name} ({metric}): {e}")


def _parse_model_and_variant(stem: str) -> tuple[str | None, str | None]:
    """Split a file stem like ``resnet152_exp1a_variant1`` into ``(model, variant)``."""
    if "_" not in stem:
        return None, None
    known_models = sorted(MODEL_REGISTRY.keys(), key=len, reverse=True)
    for km in known_models:
        if stem.startswith(f"{km}_"):
            return km, stem[len(km) + 1 :]
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def run_comparisons(
    methods,
    kinds,
    metrics,
    target_res,
    models_filter,
    save_json,
    save_plots: bool = False,
):
    results = {}
    methods = _resolve_methods(methods)

    def get_file_map(method: str) -> dict[str, dict[str, Path]]:
        d = _method_dir(method)
        if not d.exists():
            return {}
        g: dict[str, dict[str, Path]] = defaultdict(dict)
        for f in sorted(d.glob("*_maps.npy")):
            stem = f.stem.replace("_maps", "")
            matched_m, var = _parse_model_and_variant(stem)
            if not matched_m or not var:
                continue
            if models_filter and matched_m not in models_filter:
                continue
            g[matched_m][var] = f
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

                if save_plots:
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
        if save_plots and within_model_stats:
            for metric, model_data in within_model_stats.items():
                try:
                    fig_violin = visualize_violin_distribution(
                        model_data,
                        metric=metric,
                        add_title=False,
                        custom_model_order=MODEL_ORDER,
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
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
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

            if save_plots:
                save_plots_for_result(
                    comp_res,
                    valid_protos_sorted,
                    method,
                    "between_model_architectures",
                    metrics,
                )

            del prototypes, proto_list
            gc.collect()

    # --- 2b. LIME-Specific Surrogate R2 & 1D Superpixel Beta Summaries ---
    if "lime" in methods and DIRS["lime"].exists():
        # Surrogate Fidelity R2 Summary (Paper Appendix Table A.2)
        r2_by_model: dict[str, list[np.ndarray]] = defaultdict(list)
        for r2_file in sorted(DIRS["lime"].glob("*_r2.npy")):
            stem = r2_file.stem.replace("_r2", "")
            matched_m, _var = _parse_model_and_variant(stem)
            if not matched_m:
                continue
            if models_filter and matched_m not in models_filter:
                continue
            try:
                arr = np.load(r2_file).astype(float).ravel()
                r2_by_model[matched_m].append(arr)
            except OSError as e:
                warn(f"Could not load LIME R2 file {r2_file}: {e}")

        if r2_by_model:
            r2_summary: dict[str, dict[str, float | int]] = {}
            r2_concat: dict[str, np.ndarray] = {}
            for m_name, arr_list in r2_by_model.items():
                flat = np.concatenate(arr_list)
                finite = flat[np.isfinite(flat)]
                r2_concat[m_name] = finite
                if finite.size > 0:
                    r2_summary[m_name] = {
                        "mean": float(np.mean(finite)),
                        "std": float(np.std(finite)),
                        "median": float(np.median(finite)),
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "n_variants": len(arr_list),
                        "n_fits": int(finite.size),
                    }
            if r2_summary:
                results["lime_r2_summary"] = r2_summary
                if save_plots:
                    plot_lime_r2_distributions(r2_concat, DIRS["plots"])

        # 1D Superpixel Beta Consistency (Paper Table 5 & Figure 3C)
        beta_files_by_model: dict[str, dict[str, Path]] = defaultdict(dict)
        for beta_file in sorted(DIRS["lime"].glob("*_betas.npz")):
            stem = beta_file.stem.replace("_betas", "")
            matched_m, var = _parse_model_and_variant(stem)
            if not matched_m or not var:
                continue
            if models_filter and matched_m not in models_filter:
                continue
            beta_files_by_model[matched_m][var] = beta_file

        if beta_files_by_model:
            sp_within_res = {}
            sp_protos: dict[str, dict[str, np.ndarray]] = {}
            sp_valid_models: list[str] = []

            for m_name, var_map in beta_files_by_model.items():
                var_names = list(var_map.keys())
                loaded_beta_dicts: list[dict[str, np.ndarray]] = []
                for bp in var_map.values():
                    with np.load(bp) as npz_data:
                        loaded_beta_dicts.append(
                            {
                                k: np.asarray(npz_data[k], dtype=np.float64)
                                for k in npz_data.files
                            }
                        )

                if "within_model_variants" in kinds and len(loaded_beta_dicts) >= 2:
                    sp_comp = compare_lime_beta_collections(loaded_beta_dicts)
                    sp_comp["variants"] = var_names
                    sp_within_res[m_name] = sp_comp

                if "between_model_architectures" in kinds and loaded_beta_dicts:
                    common_keys = sorted(
                        set.intersection(*(set(d.keys()) for d in loaded_beta_dicts))
                    )
                    proto_dict: dict[str, np.ndarray] = {}
                    for img_key in common_keys:
                        vecs = [d[img_key] for d in loaded_beta_dicts]
                        if len({v.size for v in vecs}) == 1:
                            proto_dict[img_key] = np.mean(
                                np.stack(vecs, axis=0), axis=0
                            )
                    if proto_dict:
                        sp_protos[m_name] = proto_dict
                        sp_valid_models.append(m_name)

            if sp_within_res:
                results["lime_superpixel_within_model_variants"] = sp_within_res

            if "between_model_architectures" in kinds and len(sp_protos) > 1:
                sp_models_sorted = sorted(
                    sp_valid_models,
                    key=lambda x: (
                        MODEL_ORDER.index(x) if x in MODEL_ORDER else len(MODEL_ORDER)
                    ),
                )
                sp_proto_list = [sp_protos[m] for m in sp_models_sorted]
                sp_between = compare_lime_beta_collections(sp_proto_list)
                sp_between["models"] = sp_models_sorted
                results["lime_superpixel_between_model_architectures"] = sp_between
                if save_plots:
                    save_plots_for_result(
                        sp_between,
                        sp_models_sorted,
                        "lime_superpixel",
                        "between_model_architectures",
                        ("spearman", "correlation"),
                    )

    # --- 3. Cross-Method Comparison ---
    if "cross_methods" in kinds and len(methods) >= 2:
        file_maps = {m: get_file_map(m) for m in methods}
        cross_pairs_res: dict[str, dict[str, dict]] = {}

        for m1, m2 in combinations(methods, 2):
            pair_label = f"{m1}_vs_{m2}"
            info(f"[Cross-Methods] Comparing {m1} vs {m2}...")
            files1 = file_maps[m1]
            files2 = file_maps[m2]
            pair_res = {}

            for m_name in files1:
                if m_name not in files2:
                    continue
                for var in files1[m_name]:
                    if var not in files2[m_name]:
                        continue

                    arr1 = load_and_resize_map(files1[m_name][var], target_res)
                    arr2 = load_and_resize_map(files2[m_name][var], target_res)

                    if arr1 is not None and arr2 is not None:
                        res = compare_heatmaps([arr1, arr2], metrics=metrics)
                        pair_res[f"{m_name}_{var}"] = res

                    del arr1, arr2
                    gc.collect()

            if pair_res:
                cross_pairs_res[pair_label] = pair_res

        if "gradcam_vs_mpm" in cross_pairs_res:
            results["cross_methods"] = cross_pairs_res["gradcam_vs_mpm"]
        elif len(cross_pairs_res) == 1:
            results["cross_methods"] = next(iter(cross_pairs_res.values()))

        if cross_pairs_res:
            results["cross_methods_pairs"] = cross_pairs_res

    # Save JSON
    if save_json:
        out_path = DIRS["output"] / "experiment_2b_comparison.json"
        try:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
            info(f"Saved results to {out_path}")
        except (OSError, TypeError, ValueError) as e:
            error(f"JSON Save Failed: {e}")
            import traceback

            traceback.print_exc()

    return results


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
    save_plots=False,
    cluster_computing=False,
):
    start = time.time()
    setup_directories()
    if isinstance(models, str):
        models = [models]

    methods = _resolve_methods(xai_methods)

    # --- Part A: Generation ---
    if not comparison_only:
        if SINGLE_BATCH_SIZE != 1:
            raise ValueError(
                f"Experiment 2 requires single_batch_size=1, found {SINGLE_BATCH_SIZE} in data config"
            )
        info(">>> STARTING GENERATION")
        slic_segments = (
            prepare_or_load_slic_segments() if "lime" in methods and save_maps else None
        )
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

                if save_maps and all(
                    _map_output_path(m_name, tag, meth).exists() for meth in methods
                ):
                    info(f"All requested maps exist for {m_name}_{tag}. Skipping.")
                    continue

                model = m_cls(freeze_backbone=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.load_state_dict(
                    torch.load(w_path, map_location=device, weights_only=True)
                )
                model.to(device).eval()
                model.requires_grad_(False)

                for meth in methods:
                    if save_maps and _map_output_path(m_name, tag, meth).exists():
                        info(
                            f"Map {_map_output_path(m_name, tag, meth)} exists. Skipping."
                        )
                        continue

                    if meth == "lime" and m_name != "clip_vit_b16":
                        ds = IMAGENET_DATASET["test"]
                    else:
                        ds = MODEL_REGISTRY[m_name]["dataset"]["test"]

                    if cluster_computing:
                        loader = DataLoader(
                            ds,
                            batch_size=SINGLE_BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True,
                        )
                    else:
                        loader = DataLoader(
                            ds,
                            batch_size=SINGLE_BATCH_SIZE,
                            shuffle=False,
                            num_workers=0,
                        )

                    run_generation_loop(
                        model,
                        loader,
                        device,
                        m_name,
                        tag,
                        meth,
                        save_maps,
                        cluster_computing=cluster_computing,
                        slic_segments=slic_segments,
                    )
                    del loader

                model.cpu()
                del model
                cleanup_model_and_data(None)

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
            save_plots=save_plots,
        )

    info(f"Done. Time: {time.time() - start:.2f}s")


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
            "clip_vit_b16",
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
