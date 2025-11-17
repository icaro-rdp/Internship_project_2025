import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import numpy as np
import gc
import time
import re
import traceback
import json
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple


# Add main package to path - go up from  Experiments/ -> main/ -> Image_Authenticity_prediction/
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import models
from main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor
)

# Import utilities
from main.Utils.explainability import GradCAM, MultiscalePixelMasking
from main.Utils.cleanup import clear_gpu_memory, cleanup_model_and_data
from main.Utils.logger import info, warn, error, debug, set_level
from main.Utils.comparisons import compare_heatmaps, uniform_heatmaps
from main.Utils.visualization import visualize_similarity_matrix
from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    SINGLE_BATCH_SIZE,
    NUM_WORKERS,
)


# ============================================================================
# Configuration
# ============================================================================

# Model registry with their configurations
MODEL_REGISTRY = {
    'vgg16': {
        'class': VGG16AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.28',  # Last conv layer
        'input_size': 224
    },
    'vgg19': {
        'class': VGG19AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.34',  # Last conv layer
        'input_size': 224
    },
    'resnet152': {
        'class': ResNet152AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.7.2.conv3',  # Last residual block
        'input_size': 224
    },
    'densenet161': {
        'class': DenseNet161AuthenticityPredictor,
        'dataset': DENSENET_DATASET,
        'target_layer': 'features.denseblock4.denselayer24.conv2',  # Last dense block conv layer
        'input_size': 300
    },
    'efficientnetb3': {
        'class': EfficientNetB3AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.8.0',  # Last conv2d of the last block
        'input_size': 224
    },
    'barlowtwins': {
        'class': BarlowTwinsAuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.7.2.conv3',  # Last layer before avgpool
        'input_size': 224
    }
}

# XAI method specific configurations
CONFIG = {
    'sigma_values' : [3, 5, 9, 17, 33, 65],
    'sigma_values_test' : [3, 17, 65],
    'mask_value':0,
    'pixel_batch_size': 512
}
# Training hyperparameters

# Output directories
OUTPUT_DIR = Path('Outputs/Experiment_2_variants')
XAI_MAPS_OUTPUT = OUTPUT_DIR / 'XAI_Maps'
GRADCAM_OUTPUT = XAI_MAPS_OUTPUT / 'GradCAM'
MPM_OUTPUT = XAI_MAPS_OUTPUT / 'Multiscale_Pixel_Masking'
WEIGHTS_DIR = Path('Outputs/Experiment_1_variants/Weights')


# ============================================================================
# Experiment 2A: Explainability Methods / GradCAM - Multiscale Pixel Masking
# ============================================================================

def generate_explainability_maps(
    variants="all",
    xai_method="both",
    save_maps=True,
    verbose=True,
    models_to_test=None,
):
    """
    Conducts Experiment 2A using GradCAM for explainability.
        
    Args:
        models_to_test (list or None): List of model names to test. If None, tests all models in MODEL_REGISTRY.
        model_name (str): Name of the model to use.
        config (dict): Configuration dictionary for the model.
        save_maps (bool): Whether to save explainability maps.
    variants (str|Sequence[str]): Which variants to test ('all' | 'base' | 'greedy' | 'negative' | 'orig').
    
    Returns:
        None
    """
    print("=" * 80)
    info("EXPERIMENT 2A: GRADCAM-BASED EXPLAINABILITY")
    print("=" * 80)

    #create output directories
    XAI_MAPS_OUTPUT.mkdir(parents=True, exist_ok=True)
    GRADCAM_OUTPUT.mkdir(parents=True, exist_ok=True)
    MPM_OUTPUT.mkdir(parents=True, exist_ok=True)

    if isinstance(variants, str):
        requested_variants = {variants.lower()}
    else:
        requested_variants = {str(v).lower() for v in variants}

    if not requested_variants:
        requested_variants = {"all"}

    valid_variant_keys = {"all", "base", "greedy", "negative", "orig"}
    invalid_variant_keys = requested_variants - valid_variant_keys
    if invalid_variant_keys:
        warn(
            "Unknown variant selection %s. Supported options are %s."
            % (sorted(invalid_variant_keys), sorted(valid_variant_keys))
        )
        requested_variants -= invalid_variant_keys

    if not requested_variants:
        warn("No valid variants requested. Aborting explainability run.")
        return {}

    include_all_variants = "all" in requested_variants
    selected_variants = requested_variants - {"all"}

    valid_methods = {"gradcam", "multiscale_pixel_masking", "both"}
    if xai_method not in valid_methods:
        warn(
            "Unknown explainability method '%s'. Choose from %s."
            % (xai_method, sorted(valid_methods))
        )
        return {}

    run_gradcam = xai_method in {"gradcam", "both"}
    run_masking = xai_method in {"multiscale_pixel_masking", "both"}

    def variant_matches(tag: str) -> bool:
        tag_lower = tag.lower()
        if include_all_variants:
            return True
        if "base" in selected_variants and (
            tag_lower == "orig" or tag_lower.startswith("exp1a_variant")
        ):
            return True
        if "orig" in selected_variants and tag_lower == "orig":
            return True
        if "greedy" in selected_variants and "greedy_pruned" in tag_lower:
            return True
        if "negative" in selected_variants and "negative_pruned" in tag_lower:
            return True
        return False

    # Verify weights directory exists
    if not WEIGHTS_DIR.exists():
        error(f"Weights directory not found: {WEIGHTS_DIR}")
        return {}
    
    # Collect all .pth files 
    all_pth_files = sorted(WEIGHTS_DIR.glob('*.pth'))
    print(f"Found {len(all_pth_files)} weight files in {WEIGHTS_DIR}")

    if not all_pth_files:
        error(f"No .pth files found in {WEIGHTS_DIR}. Please run Experiment 1A and 1B first.")
        return {}
    
    # Group by model name extracted from filename prefix like 'vgg16_exp1a...'
    modelname_re = re.compile(r'^([A-Za-z0-9_]+)_exp1')
    weights_files_by_model = {}
    skipped_files = []
    
    for p in all_pth_files:
        m = modelname_re.match(p.name)
        if not m:
            # if filename does not follow naming convention, skip but record for inspection
            skipped_files.append(p)
            continue
            
        mn = m.group(1)
        
        # If user asked to test only specific models, skip others
        if models_to_test is not None and mn not in models_to_test:
            continue
            
        # Only include files for known models (we need config to instantiate the model)
        if mn not in MODEL_REGISTRY:
            warn(f"Found weights for unknown model '{mn}' -> skipping file {p.name}")
            skipped_files.append(p)
            continue
            
        weights_files_by_model.setdefault(mn, []).append(p)
        
    if not weights_files_by_model:
        error(f"No valid model weight files found in {WEIGHTS_DIR} (checked {len(all_pth_files)} files).")
        if skipped_files:
            info("Skipped files:")
            for s in skipped_files:
                info(f" - {s.name}")
        return {}
        
    total_files_found = sum(len(v) for v in weights_files_by_model.values())
    info(f"Found {total_files_found} weight file(s) across {len(weights_files_by_model)} model type(s) in {WEIGHTS_DIR}.")
    
    if verbose:
        info("Per-model file counts:")
        for mn, fls in sorted(weights_files_by_model.items()):
            info(f" - {mn}: {len(fls)} file(s)")
    
    # Results storage for numpy maps 
    results = {}
    
    # run explainability on each model (and its variant files)
    model_items = list(weights_files_by_model.items())
    
    for idx, (model_name, weight_file_list) in enumerate(model_items, 1):
        info(f"[{idx}/{len(model_items)}] XAI on {model_name.upper()}")
        print("-" * 80)

        try:
            config = MODEL_REGISTRY[model_name]
            
            dataset = config['dataset']
            test_loader = DataLoader(
                dataset['test'],
                batch_size=SINGLE_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS
            )
            
            # Prepare results container for this model
            results[model_name] = {}
            
            # Iterate over all weight files (variants) for this model
            for weights_path in weight_file_list:
                # prepare the naming convention for outputs
                m = re.search(r"exp1a_variant\d+|exp1b_variant\d+_greedy_pruned|exp1b_variant\d+_negative_pruned|orig", str(weights_path))
                variant_tag = m.group(0) if m else 'orig'
                model_name_output = f"{model_name}_{variant_tag}"

                if not variant_matches(variant_tag):
                    debug(
                        "Skipping variant '%s' for model '%s' based on selection %s."
                        % (variant_tag, model_name, sorted(selected_variants) if selected_variants else ['all'])
                    )
                    continue

                if verbose:
                    info(f"Loading model from {weights_path}...")

                # Instantiate model and load weights
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = config['class'](freeze_backbone=False)
                model.load_state_dict(torch.load(weights_path, weights_only=True))
                model.to(device)

                variant_result = results[model_name].setdefault(variant_tag, {})

                if run_gradcam:
                    print("Running GradCAM...")
                    # for each image in test set, generate GradCAM map
                    maps = []
                    for img_idx, (img, label) in enumerate(test_loader):
                        img = img.to(device)
                        label = label.to(device)

                        gradcam = GradCAM(
                            model=model,
                            target_layer=config['target_layer'],
                            relu=False
                        )
                        gradcam_map = gradcam.generate_map(img, target_index=0)
                        gradcam.cleanup()
                        print (f"GradCAM map shape: {gradcam_map.shape}")
                        maps.append(gradcam_map)

                    if save_maps:
                        # Save the numpy array to disk
                        print(f"Saving GradCAM map for {model_name} with shape {np.array(maps).shape}...")
                        maps_array = np.array(maps)
                        gradcam_map_path = GRADCAM_OUTPUT / f"{model_name_output}_maps.npy"
                        np.save(gradcam_map_path, maps_array)
                        variant_result['gradcam_map_path'] = str(gradcam_map_path)
                    variant_result['gradcam_sample_count'] = len(maps)

                    

                if run_masking:
                    print("Running Multiscale Pixel Masking...")
                    # for each image in test set, generate MPM map
                    maps = []
                    for img_idx, (img, label) in enumerate(test_loader):
                        img = img.to(device)
                        label = label.to(device)
                        info(f"Generating MPM map for image {img_idx+1} with model {model_name} {variant_tag}...")
                        mpm = MultiscalePixelMasking(
                            model=model,
                            sigma_list=CONFIG['sigma_values_test'],
                            pixel_batch_size=CONFIG['pixel_batch_size'],
                            mask_value=CONFIG['mask_value'])
                        
                        mpm_map = mpm.generate_map(img, target_index=0)
                        maps.append(mpm_map)
                        
                    if save_maps:
                        print(f"Saving Multiscale Pixel Masking map for {model_name})...")
                        maps_array = np.array(maps)
                        mpm_map_path = MPM_OUTPUT / f"{model_name_output}_maps.npy"
                        np.save(mpm_map_path, maps_array)
                        variant_result['mpm_map_path'] = str(mpm_map_path)
                    variant_result['mpm_sample_count'] = len(maps)
                         
        except Exception as e:
            error(f"Error testing {model_name}: {e}")
            error(traceback.format_exc())
                 
        finally:
            # Clean up memory after each model (success or failure)
            info(f"Cleaning up {model_name} from memory...")
            cleanup_model_and_data(
                model=locals().get('model'),
                dataloaders=locals().get('test_loader'),
                optimizer=None
            )
            clear_gpu_memory()
            info(f"✓ {model_name} memory cleaned")

    return results     

# ============================================================================
# Experiment 2B: Explainability Comparisons
# ============================================================================

def _load_heatmap(path: Path) -> np.ndarray:
    """Load heatmap array from disk and normalise shape to (N, H, W)."""
    arr = np.load(path)
    info(f"Loaded heatmap array from {path} with shape {arr.shape}")
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise ValueError(f"Heatmap array at {path} has 4 dims but channel dimension != 1: {arr.shape}")
    elif arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Heatmap array at {path} must be 3D after processing. Found shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _split_model_variant(stem: str) -> Tuple[str, str]:
    """Split a stored heatmap stem into (model, variant)."""
    if "_" not in stem:
        warn(f"Could not split stem '{stem}' into model and variant; defaulting variant to 'default'.")
        return stem, "default"
    model_name, variant = stem.split("_", 1)
    return model_name, variant


def _summarise(values: Sequence[float]) -> Dict[str, float]:
    """Compute summary statistics while dropping NaNs."""
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "median": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def _labels_from_summary(summary: Dict[str, Dict[str, float]]) -> Sequence[str]:
    """Derive sorted labels from pairwise summary keys like 'a_vs_b'."""
    label_set = set()
    for pair_label in summary.keys():
        parts = pair_label.split("_vs_")
        if len(parts) == 2:
            label_set.update(parts)
    return sorted(label_set)


def _reindex_summary_for_visualization(
    summary: Dict[str, Dict[str, float]],
    labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Map human-readable pair labels onto numerical indices for plotting."""
    index_lookup = {label: str(idx) for idx, label in enumerate(labels)}
    reindexed: Dict[str, Dict[str, float]] = {}
    for pair_label, stats in summary.items():
        parts = pair_label.split("_vs_")
        if len(parts) != 2:
            warn(f"Unexpected pair label format '{pair_label}' for visualization. Skipping.")
            continue
        left, right = parts
        if left not in index_lookup or right not in index_lookup:
            warn(
                "Pair '%s' references labels not present in the provided label list %s. Skipping." %
                (pair_label, labels)
            )
            continue
        indexed_key = f"{index_lookup[left]}_vs_{index_lookup[right]}"
        reindexed[indexed_key] = stats
    return reindexed


def _compare_context(
    label_to_path: Dict[str, Path],
    metrics: Sequence[str],
    target_resolution: Optional[Tuple[int, int]],
    overall_accumulator: Dict[str, Dict[str, list]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compare heatmaps for a specific grouping and update global accumulators."""

    if len(label_to_path) < 2:
        warn("Comparison context skipped because fewer than two heatmaps were provided.")
        return {}

    labels = []
    arrays = []
    for label, path in sorted(label_to_path.items()):
        arr = _load_heatmap(path)
        labels.append(label)
        arrays.append(arr)

    min_images = min(arr.shape[0] for arr in arrays)
    if min_images == 0:
        warn("Comparison context skipped because at least one heatmap has zero images.")
        return {}

    if target_resolution is None:
        target_h = max(arr.shape[1] for arr in arrays)
        target_w = max(arr.shape[2] for arr in arrays)
    else:
        target_h, target_w = target_resolution

    aligned_arrays = [
        uniform_heatmaps(arr, height=target_h, width=target_w, num_images=min_images)
        for arr in arrays
    ]

    comparison = compare_heatmaps(aligned_arrays, metrics=metrics)

    context_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric in metrics:
        metric_summary: Dict[str, Dict[str, float]] = {}
        summary_block = comparison["summary"].get(metric, {})
        per_image_block = comparison["per_image"].get(metric, {})
        for pair_key, stats in summary_block.items():
            i_str, j_str = pair_key.split("_vs_")
            pair_label = f"{labels[int(i_str)]}_vs_{labels[int(j_str)]}"
            metric_summary[pair_label] = {k: float(v) for k, v in stats.items()}

            values = per_image_block.get(pair_key)
            if values is not None:
                accumulator = overall_accumulator.setdefault(metric, {})
                accumulator.setdefault(pair_label, []).extend(
                    float(v) for v in values if not np.isnan(v)
                )
        if metric_summary:
            context_summary[metric] = metric_summary

    return context_summary


def compare_explainability_maps(
    methods: Optional[Sequence[str]] = None,
    metrics: Sequence[str] = ("mse", "correlation", "cosine", "ssim", "emd"),
    target_resolution: Optional[Tuple[int, int]] = (224, 224),
    save_json: bool = True,
    comparison_kinds: Sequence[str] = ("cross_methods", "inter_model", "intra_model_variants"),
    show_comparison_plots: bool = True,
) -> Dict[str, Any]:
    """Conduct Experiment 2B by comparing explainability maps saved on disk.

    Typical usage patterns:
        # Compare GradCAM maps across different models and variants
        compare_explainability_maps(
            methods=("gradcam",),
            comparison_kinds=("inter_model", "intra_model_variants"),
        )

        # Check GradCAM vs MPM agreement (cosine similarity only)
        compare_explainability_maps(
            methods=("gradcam", "multiscale_pixel_masking"),
            metrics=("cosine",),
            comparison_kinds=("cross_methods",),
        )
    """

    print("=" * 80)
    info("EXPERIMENT 2B: EXPLAINABILITY MAPS COMPARISON")
    print("=" * 80)

    method_dirs = {
        "gradcam": GRADCAM_OUTPUT,
        "multiscale_pixel_masking": MPM_OUTPUT,
    }

    if methods is None:
        requested_methods = list(method_dirs.keys())
    else:
        requested_methods = [m.strip().lower() for m in methods]

    available_methods = [m for m in requested_methods if method_dirs.get(m, Path()).exists()]
    if not available_methods:
        warn("None of the requested explainability methods have saved maps. Aborting comparisons.")
        return {}

    metrics = tuple(dict.fromkeys(metrics))
    requested_kinds = set()
    if isinstance(comparison_kinds, str):
        requested_kinds.add(comparison_kinds.lower())
    else:
        requested_kinds.update(kind.lower() for kind in comparison_kinds)

    valid_kinds = {"cross_methods", "inter_model", "intra_model_variants"}
    invalid_kinds = requested_kinds - valid_kinds
    if invalid_kinds:
        warn(f"Unknown comparison kinds {sorted(invalid_kinds)}. Supported kinds: {sorted(valid_kinds)}")
        requested_kinds -= invalid_kinds
    if not requested_kinds:
        requested_kinds = {"cross_methods"}

    # Collect map files per method keyed by model+variant stem
    method_files: Dict[str, Dict[str, Path]] = {}
    method_variant_groups: Dict[str, Dict[str, Dict[str, Path]]] = {}
    method_model_groups: Dict[str, Dict[str, Dict[str, Path]]] = {}

    for method in available_methods:
        method_path = method_dirs[method]
        files = list(method_path.glob("*_maps.npy"))
        if not files:
            warn(f"No saved maps found for method '{method}' in {method_path}.")
            continue

        stem_map: Dict[str, Path] = {}
        variant_groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
        model_groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
        for f in files:
            stem = f.stem.replace("_maps", "")
            stem_map[stem] = f
            model_name, variant = _split_model_variant(stem)
            variant_groups[variant][model_name] = f
            model_groups[model_name][variant] = f

        method_files[method] = stem_map
        method_variant_groups[method] = {variant: dict(model_map) for variant, model_map in variant_groups.items()}
        method_model_groups[method] = {model: dict(variant_map) for model, variant_map in model_groups.items()}

    if not method_files:
        warn("No comparison data available after scanning directories.")
        return {}

    results_payload: Dict[str, Any] = {}
    figures: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Cross-method comparisons (same model variant, different methods)
    # ------------------------------------------------------------------
    if "cross_methods" in requested_kinds:
        if len(method_files) < 2:
            warn("Cross-method comparison requires at least two methods with saved maps. Skipping this kind.")
        else:
            common_stems = set.intersection(*(set(files.keys()) for files in method_files.values()))
            if not common_stems:
                warn("No common model variants found across the selected methods for cross-method comparison.")
            else:
                cross_overall: Dict[str, Dict[str, list]] = {}
                per_variant_results: Dict[str, Dict[str, Any]] = {}
                for stem in sorted(common_stems):
                    label_to_path = {method: method_files[method][stem] for method in method_files}
                    context_summary = _compare_context(
                        label_to_path=label_to_path,
                        metrics=metrics,
                        target_resolution=target_resolution,
                        overall_accumulator=cross_overall,
                    )
                    if context_summary:
                        per_variant_results[stem] = {
                            "summary": context_summary,
                            "source_files": {method: str(method_files[method][stem]) for method in method_files},
                        }

                if per_variant_results:
                    overall_summary = {
                        metric: {pair: _summarise(values) for pair, values in pair_dict.items()}
                        for metric, pair_dict in cross_overall.items()
                        if pair_dict
                    }
                    block = {
                        "per_variant": per_variant_results,
                        "overall": overall_summary,
                        "methods_compared": sorted(method_files.keys()),
                        "metrics": list(metrics),
                    }
                    results_payload["cross_methods"] = block
                    if show_comparison_plots:
                        for metric in metrics:
                            matrix = overall_summary.get(metric)
                            if matrix:
                                labels = _labels_from_summary(matrix)
                                if labels:
                                    indexed_matrix = _reindex_summary_for_visualization(matrix, labels)
                                    if not indexed_matrix:
                                        continue
                                    figures[f"cross_methods_{metric}"] = visualize_similarity_matrix(
                                        results={"summary": {metric: indexed_matrix}},
                                        model_names=labels,
                                        metric=metric,
                                        stat="mean",
                                        annotate=True,
                                    )
                else:
                    warn("Cross-method comparison produced no results.")

    # ------------------------------------------------------------------
    # Inter-model comparisons (same method, different models for each variant)
    # ------------------------------------------------------------------
    if "inter_model" in requested_kinds:
        inter_model_payload: Dict[str, Any] = {}
        for method, variant_map in method_variant_groups.items():
            method_overall: Dict[str, Dict[str, list]] = {}
            variant_results: Dict[str, Dict[str, Any]] = {}
            for variant, model_paths in sorted(variant_map.items()):
                if len(model_paths) < 2:
                    continue
                context_summary = _compare_context(
                    label_to_path=model_paths,
                    metrics=metrics,
                    target_resolution=target_resolution,
                    overall_accumulator=method_overall,
                )
                if context_summary:
                    variant_results[variant] = {
                        "summary": context_summary,
                        "source_files": {model: str(path) for model, path in model_paths.items()},
                    }

            if variant_results:
                overall_summary = {
                    metric: {pair: _summarise(values) for pair, values in pair_dict.items()}
                    for metric, pair_dict in method_overall.items()
                    if pair_dict
                }
                block = {
                    "per_variant": variant_results,
                    "overall": overall_summary,
                    "metrics": list(metrics),
                }
                inter_model_payload[method] = block
                if show_comparison_plots:
                    for metric in metrics:
                        matrix = overall_summary.get(metric)
                        if matrix:
                            labels = _labels_from_summary(matrix)
                            if labels:
                                indexed_matrix = _reindex_summary_for_visualization(matrix, labels)
                                if not indexed_matrix:
                                    continue
                                figures[f"inter_model_{method}_{metric}"] = visualize_similarity_matrix(
                                    results={"summary": {metric: indexed_matrix}},
                                    model_names=labels,
                                    metric=metric,
                                    stat="mean",
                                    annotate=True,
                                )

        if inter_model_payload:
            results_payload["inter_model"] = inter_model_payload
        else:
            warn("Inter-model comparison produced no results.")

    # ------------------------------------------------------------------
    # Intra-model comparisons (same method and model, different variants)
    # ------------------------------------------------------------------
    if "intra_model_variants" in requested_kinds:
        intra_model_payload: Dict[str, Any] = {}
        for method, model_map in method_model_groups.items():
            method_overall: Dict[str, Dict[str, list]] = {}
            model_results: Dict[str, Dict[str, Any]] = {}
            for model, variant_paths in sorted(model_map.items()):
                if len(variant_paths) < 2:
                    continue
                context_summary = _compare_context(
                    label_to_path=variant_paths,
                    metrics=metrics,
                    target_resolution=target_resolution,
                    overall_accumulator=method_overall,
                )
                if context_summary:
                    model_results[model] = {
                        "summary": context_summary,
                        "source_files": {variant: str(path) for variant, path in variant_paths.items()},
                    }

            if model_results:
                overall_summary = {
                    metric: {pair: _summarise(values) for pair, values in pair_dict.items()}
                    for metric, pair_dict in method_overall.items()
                    if pair_dict
                }
                block = {
                    "per_model": model_results,
                    "overall": overall_summary,
                    "metrics": list(metrics),
                }
                intra_model_payload[method] = block
                if show_comparison_plots:
                    for metric in metrics:
                        matrix = overall_summary.get(metric)
                        if matrix:
                            labels = _labels_from_summary(matrix)
                            if labels:
                                indexed_matrix = _reindex_summary_for_visualization(matrix, labels)
                                if not indexed_matrix:
                                    continue
                                figures[f"intra_model_{method}_{metric}"] = visualize_similarity_matrix(
                                    results={"summary": {metric: indexed_matrix}},
                                    model_names=labels,
                                    metric=metric,
                                    stat="mean",
                                    annotate=True,
                                )

        if intra_model_payload:
            results_payload["intra_model_variants"] = intra_model_payload
        else:
            warn("Intra-model variant comparison produced no results.")

    if not results_payload:
        warn("No comparison results were generated. Ensure the requested data exists on disk.")
        return {}

    if save_json:
        output_path = OUTPUT_DIR / "experiment_2b_comparison.json"
        try:
            with output_path.open("w", encoding="utf-8") as fp:
                json.dump(results_payload, fp, indent=2)
            info(f"Experiment 2B comparison summary saved to {output_path}")
        except Exception as exc:
            warn(f"Could not write Experiment 2B summary to disk: {exc}")

    if show_comparison_plots and figures:
        info("Comparison figures generated: %s" % sorted(figures.keys()))

    return {"results": results_payload, "figures": figures} if show_comparison_plots else results_payload

# ============================================================================
# Full pipeline execution for Experiment 2
# ============================================================================

def run_experiment_2(
    models=None,
    save_plots=False,
    show_plots=False,
    save_maps=True,
    variants='all',
    xai_methods='both',
    comparison_only=False,
    comparison_kinds: Sequence[str] = ("cross_methods", "inter_model", "intra_model_variants"),
    comparison_metrics: Sequence[str] = ("mse", "correlation", "cosine", "ssim", "emd"),
    comparison_target_resolution: Optional[Tuple[int, int]] = (224, 224),
    save_comparison_json: bool = True,
    show_comparison_plots: bool = False,
):
    """
    Runs the full pipeline for Experiment 2, handling different explainability methods.
    Args:
        models (str|Sequence[str]|None): Model(s) to test. If None, tests all models.
        save_plots (bool): Whether to save plots.
        show_plots (bool): Whether to display plots.
        save_maps (bool): Whether to save explainability maps.
        variants (str|Sequence[str]): Which variants to test ('all' | 'base' | 'greedy' | 'negative' | 'orig').
        xai_methods (str|None): Explainability method to use ('gradcam' | 'mpm' | 'both'). If None, defaults to 'both'.
        comparison_only (bool): If True, skips map generation and only runs comparisons using existing maps.
        comparison_kinds (Sequence[str]): Kinds of comparisons to perform.
        comparison_metrics (Sequence[str]): Metrics to use for comparisons.
        comparison_target_resolution (Tuple[int, int]|None): Target resolution for heatmap alignment during comparison.
        save_comparison_json (bool): Whether to save comparison results as JSON.
        show_comparison_plots (bool): If True, also create Matplotlib figures summarising comparison results.
    Returns:
        Dict[str, Any] | None: Comparison results (and figures if requested) when comparisons run.
    """
    comparison_result: Optional[Any] = None
    # Create output directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    XAI_MAPS_OUTPUT.mkdir(parents=True, exist_ok=True)

    if isinstance(xai_methods, str):
        method_key = xai_methods.strip().lower()
    elif xai_methods is None:
        method_key = 'both'
    else:
        warn("xai_methods must be a string identifier. Defaulting to 'both'.")
        method_key = 'both'

    # Normalise aliases to match available explainability methods
    alias_map = {
        'gradcam': 'gradcam',
        'both': 'both',
        'all': 'both',
        'mpm': 'multiscale_pixel_masking',
        'multiscale_pixel_masking': 'multiscale_pixel_masking',
        'masking': 'multiscale_pixel_masking',
    }

    if method_key not in alias_map:
        warn(
            "Unknown explainability method '%s'. Choose from ['gradcam', 'mpm', 'both']."
            % xai_methods
        )
        return

    normalised_method = alias_map[method_key]

    run_gradcam = normalised_method in ['gradcam', 'both']
    run_masking = normalised_method in ['multiscale_pixel_masking', 'both']

    methods_to_compare: Sequence[str]
    if normalised_method == 'both':
        methods_to_compare = ('gradcam', 'multiscale_pixel_masking')
    elif normalised_method == 'gradcam':
        methods_to_compare = ('gradcam',)
    else:
        methods_to_compare = ('multiscale_pixel_masking',)

    if not run_gradcam and not run_masking:
        warn(f"No explainability method selected after normalisation for '{xai_methods}'.")
        return

    if comparison_only:
        info("Skipping Experiment 2A generation; running comparisons using existing maps.")
        comparison_result = compare_explainability_maps(
            methods=methods_to_compare,
            metrics=comparison_metrics,
            target_resolution=comparison_target_resolution,
            save_json=save_comparison_json,
            comparison_kinds=comparison_kinds,
            show_comparison_plots=show_comparison_plots,
        )
        info("--- Experiment 2 Complete (comparison only) ---")
        return comparison_result

    if isinstance(models, str):
        models_to_test = [models]
    elif models is None:
        models_to_test = None
    else:
        models_to_test = list(models)

    if models_to_test is not None:
        unknown_models = [m for m in models_to_test if m not in MODEL_REGISTRY]
        if unknown_models:
            warn(
                "Unknown model(s) requested %s. Available models are %s."
                % (unknown_models, sorted(MODEL_REGISTRY))
            )
            models_to_test = [m for m in models_to_test if m in MODEL_REGISTRY]
        if not models_to_test:
            warn("No valid models requested. Aborting Experiment 2 run.")
            return

    if run_gradcam:
        info(f"Starting GradCAM explainability maps generation...")
        generate_explainability_maps(
            xai_method='gradcam',
            save_maps=save_maps,
            variants=variants,
            models_to_test=models_to_test,
        )

    if run_masking:
        info(f"Starting Multiscale Pixel Masking explainability maps generation...")
        generate_explainability_maps(
            xai_method='multiscale_pixel_masking',
            save_maps=save_maps,
            variants=variants,
            models_to_test=models_to_test,
        )

    if save_maps:
        info("Starting Experiment 2B: explainability map comparison...")
        comparison_result = compare_explainability_maps(
            methods=methods_to_compare,
            metrics=comparison_metrics,
            target_resolution=comparison_target_resolution,
            save_json=save_comparison_json,
            comparison_kinds=comparison_kinds,
            show_comparison_plots=show_comparison_plots,
        )
    else:
        info("Skipping Experiment 2B comparison because maps were not saved.")

    info("--- Experiment 2 Complete ---")
    if save_maps:
        info(f"All explainability maps saved in {XAI_MAPS_OUTPUT}")
    if show_plots:
        info("Plots were displayed during the run.")
    if save_plots:
        info(f"All plots were saved in {OUTPUT_DIR}")
    return comparison_result
        



if __name__ == '__main__':
    # Example run: adjust parameters as needed

    # Start timer
    start_time = time.time()

    # Example: run full Experiment 2 for all models, generating maps and all comparison kinds
    # run_experiment_2()

    # Example: evaluate only GradCAM explainability for DenseNet variants and compute cross-method comparisons later
    # run_experiment_2(models=('densenet161',), xai_methods='gradcam', variants=('orig', 'greedy'))

    # Example: Run only the comparison step using existing maps for gradcam intra model and inter model variants.
    # # set_level('DEBUG') # Uncomment to enable debug logging
    run_experiment_2(
    variants='greedy',
    xai_methods='mpm', 
    )

    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format elapsed time as H:M:S
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    info(f"\nTotal execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")