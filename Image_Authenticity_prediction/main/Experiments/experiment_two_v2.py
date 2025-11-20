import torch
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import gc
import time
import re
import json
import shutil
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple, List

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
from main.Utils.comparisons import (
    compare_heatmaps,
    uniform_heatmaps,
    visualize_similarity_matrix,
    visualize_similarity_distribution,
)
from main.data import IMAGENET_DATASET, DENSENET_DATASET, SINGLE_BATCH_SIZE, NUM_WORKERS

DIRS = {
    "output": Path("Outputs/Experiment_2_variants"),
    "weights": Path("Outputs/Experiment_1_variants/Weights"),
}
DIRS["maps"] = DIRS["output"] / "XAI_Maps"
DIRS["gradcam"] = DIRS["maps"] / "GradCAM"
DIRS["mpm"] = DIRS["maps"] / "Multiscale_Pixel_Masking"
DIRS["prototypes"] = DIRS["maps"] / "Prototypes"
DIRS["plots"] = DIRS["output"] / "Plots"
DIRS["plots_gradcam"] = DIRS["plots"] / "GradCAM"
DIRS["plots_mpm"] = DIRS["plots"] / "MPM"

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

XAI_PARAMS = {
    "sigma": [3, 17, 65],
    "mask_val": 0,
    "px_batch": 256,
    "gc_interval": 50,  # Save chunk every 50 images
    "mpm_interval": 10,  # Save chunk every 10 images
}


def setup_directories():
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 2. GENERATION HELPERS (CHUNKED I/O)
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

        tag_m = re.search(
            r"exp1a_variant\d+|exp1b_variant\d+_greedy_pruned|exp1b_variant\d+_negative_pruned|orig",
            str(p),
        )
        tag = tag_m.group(0) if tag_m else "orig"

        keep = False
        if include_all:
            keep = True
        elif "greedy" in req_vars and "greedy" in tag:
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

    temp_chunk_dir = out_dir / f"temp_chunks_{stem}"
    temp_chunk_dir.mkdir(exist_ok=True)

    if final_path.exists():
        info(f"Map {final_path} already exists. Skipping.")
        return

    batch_maps = []
    chunk_idx = 0
    flush_interval = XAI_PARAMS["gc_interval"] if is_gc else XAI_PARAMS["mpm_interval"]

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
                    model,
                    XAI_PARAMS["sigma"],
                    XAI_PARAMS["px_batch"],
                    XAI_PARAMS["mask_val"],
                )
                res = mpm.generate_map(img, 0)

            if isinstance(res, torch.Tensor):
                res = res.cpu().numpy()

            batch_maps.append(res)

            # Save chunk
            if len(batch_maps) >= flush_interval:
                np.save(
                    temp_chunk_dir / f"chunk_{chunk_idx:05d}.npy", np.array(batch_maps)
                )
                batch_maps = []
                chunk_idx += 1
                torch.cuda.empty_cache()
                gc.collect()

        # Save remaining
        if batch_maps:
            np.save(temp_chunk_dir / f"chunk_{chunk_idx:05d}.npy", np.array(batch_maps))

        # --- Merge Chunks via Memory Mapping ---
        info(f"Merging chunks for {stem}...")
        chunk_files = sorted(temp_chunk_dir.glob("chunk_*.npy"))
        if not chunk_files:
            warn("No maps generated.")
            if temp_chunk_dir.exists():
                shutil.rmtree(temp_chunk_dir)
            return

        # Get Metadata
        first = np.load(chunk_files[0])
        dtype = first.dtype
        H, W = first.shape[1], first.shape[2]
        total_N = sum(np.load(f, mmap_mode="r").shape[0] for f in chunk_files)

        # Create output file on disk
        fp = np.memmap(final_path, dtype=dtype, mode="w+", shape=(total_N, H, W))

        cursor = 0
        for cf in chunk_files:
            data = np.load(cf)
            n = data.shape[0]
            fp[cursor : cursor + n] = data
            cursor += n
            del data

        fp.flush()
        del fp

        shutil.rmtree(temp_chunk_dir)
        info(f"Saved {final_path} with shape ({total_N}, {H}, {W})")

    except Exception as e:
        error(f"Error generating {stem}: {e}")
        import traceback

        traceback.print_exc()


# ============================================================================
# 3. COMPARISON HELPERS (BATCHED & MEMORY-MAPPED)
# ============================================================================


def _get_mmap(path):
    """Returns a memory-mapped array in read mode."""
    try:
        return np.load(path, mmap_mode="r")
    except Exception as e:
        error(f"Failed to mmap {path}: {e}")
        return None


def _resize_batch(batch_arrays, target_res):
    """Resizes a batch of numpy arrays to target resolution."""
    resized = []
    h, w = target_res
    for arr in batch_arrays:
        # Assuming uniform_heatmaps handles (B, H, W) or (H, W)
        # If the array is already the correct size, skip
        if arr.shape[-2:] == (h, w):
            resized.append(arr)
        else:
            # If uniform_heatmaps doesn't handle batches, loop here
            # Assuming it handles single images or batches safely
            try:
                r = uniform_heatmaps(arr, h, w)
                resized.append(r)
            except:
                # Fallback loop
                sub_batch = np.stack([uniform_heatmaps(x, h, w) for x in arr])
                resized.append(sub_batch)
    return resized


def create_prototype_on_disk(variant_paths, save_path, target_res):
    """
    Computes the mean of multiple variants batch-by-batch and saves
    result to disk to avoid RAM explosion.
    """
    if save_path.exists():
        return _get_mmap(save_path)

    mmaps = [_get_mmap(p) for p in variant_paths]
    mmaps = [m for m in mmaps if m is not None]
    if not mmaps:
        return None

    N = min(m.shape[0] for m in mmaps)
    dtype = np.float32

    # Create output memmap
    fp = np.memmap(
        save_path, dtype=dtype, mode="w+", shape=(N, target_res[0], target_res[1])
    )

    batch_size = 100

    info(f"Generating prototype: {save_path.name}")
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)

        # Load raw batch
        raw_batch = [m[i:end] for m in mmaps]

        # Resize
        resized = _resize_batch(raw_batch, target_res)

        # Stack and Mean
        stack = np.stack(resized)  # (Num_Vars, B, H, W)
        mean_batch = np.mean(stack, axis=0)  # (B, H, W)

        fp[i:end] = mean_batch

    fp.flush()
    # Return read-only mmap of the new file
    return np.load(save_path, mmap_mode="r")


def batch_process_metrics(map_paths, metrics, target_res, batch_size=100):
    """
    Calculates similarity metrics batch-wise.
    map_paths: List of paths to .npy files
    """
    mmaps = [_get_mmap(p) for p in map_paths]
    mmaps = [m for m in mmaps if m is not None]
    if len(mmaps) < 2:
        return {}

    N = min(m.shape[0] for m in mmaps)

    accumulators = defaultdict(float)
    count = 0

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        current_batch_size = end - i

        # Load slice
        raw_batch = [m[i:end] for m in mmaps]

        # Resize
        resized_batch = _resize_batch(raw_batch, target_res)

        # Compare
        # compare_heatmaps expects list of arrays [Map1, Map2, ...]
        batch_results = compare_heatmaps(resized_batch, metrics=metrics)

        for m_key, val in batch_results.items():
            # Weight average by batch size
            # Ensure val is aggregated if it's an array
            if hasattr(val, "__len__") and len(val) == current_batch_size:
                accumulators[m_key] += np.sum(val)
            else:
                # Scalar average returned
                accumulators[m_key] += val * current_batch_size

        count += current_batch_size

        # Periodic GC
        if i % (batch_size * 10) == 0:
            gc.collect()

    final_metrics = {k: v / count for k, v in accumulators.items()}
    return final_metrics


def run_comparisons(
    methods,
    kinds,
    metrics,
    target_res,
    models_filter,
    save_json,
    show_plots,
    save_plots,
):
    results = {}

    def get_groups(method_name):
        d = DIRS["gradcam"] if method_name == "gradcam" else DIRS["mpm"]
        if not d.exists():
            return {}
        g = defaultdict(dict)
        for f in d.glob("*_maps.npy"):
            stem = f.stem.replace("_maps", "")
            if "_" not in stem:
                continue
            m_name, var = stem.split("_", 1)
            if models_filter and m_name not in models_filter:
                continue
            g[m_name][var] = f
        return g

    # 1. Inter-Model (Prototypes) & Intra-Model
    for method in methods:
        groups = get_groups(method)
        if not groups:
            continue

        # --- Inter-Model (Prototype vs Prototype) ---
        if "inter_model" in kinds:
            info(f"[{method}] Running Inter-Model Comparison...")

            # Generate Prototypes first
            prototype_paths = []
            valid_model_names = []

            for m_name, variants in groups.items():
                v_paths = list(variants.values())
                proto_name = f"{method}_{m_name}_prototype.npy"
                proto_path = DIRS["prototypes"] / proto_name

                # This function handles mmap and batching internally
                res_mmap = create_prototype_on_disk(v_paths, proto_path, target_res)

                if res_mmap is not None:
                    prototype_paths.append(proto_path)
                    valid_model_names.append(m_name)

            if len(prototype_paths) > 1:
                # Compare prototypes using batched metric calculator
                comp = batch_process_metrics(prototype_paths, metrics, target_res)
                comp["models"] = valid_model_names
                results[f"{method}_inter_model"] = comp

                # Plots (Requires full matrix, which visualize_similarity_matrix usually builds internally)
                # Note: If visualize_similarity_matrix re-runs comparisons, it needs to be batch-aware.
                # Assuming here we only save aggregate stats to JSON.
                # If plots are needed, one would need to load the specific pairwise data.

        # --- Intra-Model (Variant vs Variant) ---
        if "intra_model_variants" in kinds:
            info(f"[{method}] Running Intra-Model Comparison...")
            intra_res = {}
            for m_name, variants in groups.items():
                if len(variants) < 2:
                    continue

                paths = list(variants.values())
                names = list(variants.keys())

                comp = batch_process_metrics(paths, metrics, target_res)
                comp["variants"] = names
                intra_res[m_name] = comp

            results[f"{method}_intra_model"] = intra_res

    # 2. Cross-Methods (GradCAM vs MPM)
    if "cross_methods" in kinds and "gradcam" in methods and "mpm" in methods:
        info("[Cross-Methods] Comparing GradCAM vs MPM...")
        gc_groups = get_groups("gradcam")
        mpm_groups = get_groups("mpm")

        cross_res = {}

        for m_name in gc_groups:
            if m_name not in mpm_groups:
                continue

            for var in gc_groups[m_name]:
                if var not in mpm_groups[m_name]:
                    continue

                p_gc = gc_groups[m_name][var]
                p_mpm = mpm_groups[m_name][var]

                # Binary comparison
                comp = batch_process_metrics([p_gc, p_mpm], metrics, target_res)
                cross_res[f"{m_name}_{var}"] = comp

        if cross_res:
            results["cross_methods"] = cross_res

    # Save Results
    if save_json:
        out = DIRS["output"] / "experiment_2b_comparison.json"

        # Convert numpy types for JSON
        def np_encoder(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError

        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=np_encoder)
        info(f"Comparison results saved to {out}")

    return results


# ============================================================================
# 4. MAIN FUNCTION
# ============================================================================


def run_experiment_2(
    models=None,
    save_plots=False,
    show_plots=False,
    save_maps=True,
    variants="all",
    xai_methods="both",
    comparison_only=False,
    comparison_kinds: Sequence[str] = (
        "cross_methods",
        "inter_model",
        "intra_model_variants",
    ),
    comparison_metrics: Sequence[str] = ("mse", "correlation", "cosine", "ssim", "emd"),
    comparison_target_resolution: Optional[Tuple[int, int]] = (224, 224),
    save_comparison_json: bool = True,
    show_comparison_plots: bool = False,
):
    start_time = time.time()
    setup_directories()

    if isinstance(models, str):
        models = [models]

    method_list = []
    if xai_methods in ["both", "all"]:
        method_list = ["gradcam", "mpm"]
    elif xai_methods in ["mpm", "masking"]:
        method_list = ["mpm"]
    elif xai_methods in ["gradcam"]:
        method_list = ["gradcam"]
    else:
        method_list = [xai_methods]

    # --- Experiment 2A: Generation ---
    if not comparison_only:
        info(">>> STARTING EXPERIMENT 2A: MAP GENERATION")

        work_queue = get_weight_files(models, variants)

        for model_name, file_paths in work_queue.items():
            info(f"Processing Model: {model_name}")
            model_cls = MODEL_REGISTRY[model_name]["class"]

            for weights_path in file_paths:
                m_match = re.search(
                    r"exp1a_variant\d+|exp1b_variant\d+_greedy_pruned|exp1b_variant\d+_negative_pruned|orig",
                    str(weights_path),
                )
                variant_tag = m_match.group(0) if m_match else "orig"

                # Init Model
                model = model_cls(freeze_backbone=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.load_state_dict(
                    torch.load(weights_path, map_location=device, weights_only=True)
                )
                model.to(device).eval()

                dataset = MODEL_REGISTRY[model_name]["dataset"]["test"]
                loader = DataLoader(
                    dataset,
                    batch_size=SINGLE_BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )

                for method in method_list:
                    run_generation_loop(
                        model,
                        loader,
                        device,
                        model_name,
                        variant_tag,
                        method,
                        save_maps,
                    )

                cleanup_model_and_data(model=model)
                del loader
                clear_gpu_memory()

    # --- Experiment 2B: Comparisons ---
    if save_maps or comparison_only:
        info(">>> STARTING EXPERIMENT 2B: COMPARISONS")
        run_comparisons(
            methods=method_list,
            kinds=comparison_kinds,
            metrics=comparison_metrics,
            target_res=comparison_target_resolution,
            models_filter=models,
            save_json=save_comparison_json,
            show_plots=show_comparison_plots or show_plots,
            save_plots=save_plots,
        )
    else:
        warn("Skipping comparisons.")

    info(f"Experiment 2 Finished. Total Time: {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    set_level("DEBUG")

    run_experiment_2(
        models=[
            "vgg16",
            "resnet152",
            "efficientnetb3",
            "vgg19",
            "barlowtwins",
            "densenet161",
        ],
        xai_methods="both",
        comparison_only=True,
        comparison_kinds=(
            "inter_model",
            "intra_model_variants",
        ),
        comparison_metrics=("correlation", "mse", "cosine"),
        save_comparison_json=True,
        show_comparison_plots=False,
        show_plots=False,
        save_plots=False,
    )
