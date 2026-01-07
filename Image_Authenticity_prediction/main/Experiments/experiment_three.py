# ! WIP - Experiment 3: Bagging Ensemble

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import sys
from pathlib import Path
import numpy as np
import time
import re
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple

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
from main.Utils.cleanup import clear_gpu_memory, cleanup_model_and_data
from main.Utils.logger import info, warn, error, debug, set_level
from main.Utils.config import get_ensemble_config, get_data_config


from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    SEED,
)
from main.train import train_model

# Load data config for NUM_WORKERS
_data_cfg = get_data_config()
NUM_WORKERS = _data_cfg["num_workers"]

# ============================================================================
# 1.1 DIRECTORIES
# ============================================================================
DIRS = {
    "base_weights": Path(
        "Outputs/Experiment_1_variants/Weights"
    ),  # Pre-trained from exp1
    "results": Path("Outputs/Experiment_3_ensemble/Results"),
}

# ============================================================================
# 1.2 MODEL REGISTRY
# ============================================================================
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

# ============================================================================
# 1.3 ENSEMBLE CONFIGURATION
# ============================================================================
# Ensemble config - loaded from config file
ENSEMBLE_CONFIG = get_ensemble_config()


def setup_directories():
    """Create all required directories."""
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


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
# 2. MODEL LOADING
# ============================================================================


def load_model_with_weights(
    model_name: str,
    weights_path: Path,
    device: str = "cuda",
    freeze_backbone: bool = True,  # Backbone frozen by default
) -> nn.Module:
    """Load a model with pre-trained weights."""
    model_cls = MODEL_REGISTRY[model_name]["class"]
    model = model_cls(freeze_backbone=freeze_backbone)

    if weights_path.exists():
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        debug(f"Loaded weights: {weights_path.name}")
    else:
        warn(f"Weights not found: {weights_path}, using initialized model")

    return model.to(device)


def get_predictions(
    model: nn.Module, dataloader: DataLoader, device: str = "cuda"
) -> torch.Tensor:
    """Get predictions from a model."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs, _ = model(images)
            all_preds.append(outputs.cpu())

    return torch.cat(all_preds, dim=0)


def get_labels(dataloader: DataLoader) -> torch.Tensor:
    """Extract labels from a dataloader."""
    all_labels = []
    for _, labels in dataloader:
        all_labels.append(labels.squeeze())
    return torch.cat(all_labels)


# ============================================================================
# 4. BAGGING ENSEMBLE
# ============================================================================


def check_bagging_variants(
    models_filter: List[str] = None,
    num_variants: int = 10,
) -> Dict[str, Any]:
    """
    Check which greedy pruned model variants exist (without loading them).

    Args:
        models_filter: List of model names to include (None = all)
        num_variants: Number of variants per model (default 10)

    Returns:
        Dict with available models and variant counts
    """
    info("=" * 60)
    info("BAGGING ENSEMBLE - Checking Available Variants")
    info("=" * 60)

    results = {"models": [], "variants_available": {}, "weight_paths": []}
    models_to_check = models_filter or list(MODEL_REGISTRY.keys())

    for m_name in models_to_check:
        if m_name not in MODEL_REGISTRY:
            warn(f"Model {m_name} not in registry, skipping")
            continue

        info(f"\n--- Checking {m_name} variants ---")
        variants_found = 0

        for variant_idx in range(1, num_variants + 1):
            weights_filename = f"{m_name}_exp1b_variant{variant_idx}_greedy_pruned.pth"
            weights_path = DIRS["base_weights"] / weights_filename

            if weights_path.exists():
                results["weight_paths"].append((m_name, weights_path))
                variants_found += 1
                info(f"  ✓ Found variant {variant_idx}: {weights_filename}")
            else:
                warn(f"  ✗ Not found: {weights_filename}")

        results["variants_available"][m_name] = variants_found
        if variants_found > 0:
            results["models"].append(m_name)

        info(f"  Total variants found for {m_name}: {variants_found}")

    total_variants = len(results["weight_paths"])
    info(f"\n--- Total variants available: {total_variants} ---")

    if total_variants == 0:
        error("No models found! Check that weights exist in base_weights directory.")

    results["total_variants"] = total_variants
    return results


# ============================================================================
# 6. EVALUATION
# ============================================================================


def evaluate_ensemble(
    models_filter: List[str] = None,
    device: str = "cuda",
    weight_paths: List[Tuple[str, Path]] = None,
) -> Dict[str, float]:
    """Evaluate bagging ensemble on test data (memory-efficient: one model at a time)."""
    from scipy.stats import pearsonr, spearmanr, kendalltau

    info(f"\n--- Evaluating BAGGING Ensemble ---")

    models_to_eval = models_filter or list(MODEL_REGISTRY.keys())

    # Prepare test loaders for each model
    test_loaders = {}
    for m_name in models_to_eval:
        if m_name not in MODEL_REGISTRY:
            continue
        dataset_dict = MODEL_REGISTRY[m_name]["dataset"]
        test_loaders[m_name] = DataLoader(
            dataset_dict["test"],
            batch_size=ENSEMBLE_CONFIG["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    # Get ground truth (use first loader)
    first_loader = next(iter(test_loaders.values()))
    y_true = get_labels(first_loader).numpy()

    # Memory-efficient: load one model at a time, get predictions, unload
    if weight_paths is None:
        # Build weight paths if not provided
        weight_paths = []
        num_variants = 10
        for m_name in models_to_eval:
            for variant_idx in range(1, num_variants + 1):
                wp = (
                    DIRS["base_weights"]
                    / f"{m_name}_exp1b_variant{variant_idx}_greedy_pruned.pth"
                )
                if wp.exists():
                    weight_paths.append((m_name, wp))

    info(f"  Processing {len(weight_paths)} model variants (one at a time)...")

    # Accumulate predictions on CPU
    all_preds = []
    for idx, (m_name, weights_path) in enumerate(weight_paths):
        info(f"    [{idx+1}/{len(weight_paths)}] {weights_path.name}")

        # Load model
        model = load_model_with_weights(m_name, weights_path, device)

        # Get predictions
        debug(f"Making predictions on test set for {m_name}")
        preds = get_predictions(model, test_loaders[m_name], device)
        all_preds.append(preds.squeeze().cpu())

        # Unload model and clear GPU memory
        del model
        clear_gpu_memory()

    y_pred = torch.mean(torch.stack(all_preds), dim=0).numpy()
    info(f"  Averaged predictions from {len(all_preds)} models")

    # Compute metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    plcc, plcc_p = pearsonr(y_pred, y_true)
    srcc, srcc_p = spearmanr(y_pred, y_true)
    krcc, krcc_p = kendalltau(y_pred, y_true)

    results = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "plcc": float(plcc),
        "srcc": float(srcc),
        "krcc": float(krcc),
        "plcc_p_value": float(plcc_p),
        "srcc_p_value": float(srcc_p),
        "krcc_p_value": float(krcc_p),
    }

    info(f"  MSE:  {mse:.4f}")
    info(f"  RMSE: {rmse:.4f}")
    info(f"  MAE:  {mae:.4f}")
    info(f"  PLCC: {plcc:.4f} (p={plcc_p:.2e})")
    info(f"  SRCC: {srcc:.4f} (p={srcc_p:.2e})")
    info(f"  KRCC: {krcc:.4f} (p={krcc_p:.2e})")

    return results


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================


def run_experiment_3(
    models: List[str] = None,
    evaluate: bool = True,
    save_results: bool = True,
):
    """
    Run Experiment 3: Bagging Ensemble.

    Args:
        models: List of model names to include (None = all)
        evaluate: Whether to evaluate the bagging ensemble
        save_results: Whether to save results to JSON

    Note:
        Bagging uses pre-trained greedy pruned models from Experiment 1
        (all 10 variants per model). No training is performed for bagging.
    """
    start = time.time()
    setup_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device: {device}")

    if isinstance(models, str):
        models = [models]

    results = {}

    # Check which greedy pruned variants exist (no loading yet)
    bagging_results = check_bagging_variants(models)
    results["bagging_info"] = {
        "models": bagging_results["models"],
        "variants_available": bagging_results.get("variants_available", {}),
        "total_variants": bagging_results.get("total_variants", 0),
    }
    if evaluate and bagging_results.get("total_variants", 0) > 0:
        # Pass weight_paths so evaluate_ensemble loads one model at a time
        results["bagging_evaluation"] = evaluate_ensemble(
            models, device, weight_paths=bagging_results["weight_paths"]
        )

    # --- Save Results ---
    if save_results and results:
        out_path = DIRS["results"] / "experiment_3_results.json"
        DIRS["results"].mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        info(f"\nSaved results to {out_path}")

    elapsed = time.time() - start
    info(f"\nExperiment 3 completed in {elapsed:.2f}s")

    return results


if __name__ == "__main__":
    set_level("DEBUG")
    run_experiment_3(
        models=[
            "barlowtwins",
            "resnet152",
            "densenet161",
            "efficientnetb3",
            "vgg16",
            "vgg19",
        ],
        evaluate=True,
        save_results=True,
    )
