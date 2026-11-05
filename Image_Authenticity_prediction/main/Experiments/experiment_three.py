"""Experiment 3: Variant-Aligned Unpruned 7-Model Ensembles and MPM Attribution.

Implements ADR-0005:
- Reuses the 70 unpruned Experiment 1A checkpoints across all 7 architectures
  (vgg16, vgg19, resnet152, densenet161, efficientnetb3, barlowtwins, clip_vit_b16).
- Constructs 10 variant-aligned 7-model ensembles (k = 1..10) so each ensemble
  variant k shares an identical training split (959 images) and disjoint
  validation split val_k (136 images, seed = 42 + k).
- Fits StackingMetaLearner weights via closed-form Ridge regression (alpha = 1.0)
  strictly on val_k and evaluates both Bagging and Stacking on the untouched
  273-image global test set.
- Generates full-dataset (273, 224, 224) Multiscale Pixel Masking (MPM) attribution
  maps per variant and architectural prototype heatmaps for Bagging and Stacking,
  supporting both fast Experiment 2 map reuse and live GPU MPM computation.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main.data import (
    CLIP_DATASET,
    DENSENET_DATASET,
    IMAGENET_DATASET,
    clip_dataset,
    denseNet_dataset,
    imageNet_dataset,
)
from main.Models import (
    BarlowTwinsAuthenticityPredictor,
    CLIPViTAuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
)
from main.train import test_model, train_model
from main.Utils import FeatureMapsPruner
from main.Utils.cleanup import cleanup_model_and_data, clear_gpu_memory
from main.Utils.comparisons import compare_heatmaps, uniform_heatmaps
from main.Utils.config import (
    get_data_config,
    get_ensemble_config,
    get_model_order,
    get_pruning_config,
    get_training_config,
    get_xai_config,
)
from main.Utils.explainability import MultiscalePixelMasking
from main.Utils.logger import debug, error, info, set_level, warn
from main.Utils.normalization import normalize_data
from main.Utils.visualization import (
    visualize_and_save_saliency,
    visualize_similarity_distribution,
    visualize_similarity_matrix,
)

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

_data_cfg = get_data_config()
NUM_WORKERS: int = int(_data_cfg["num_workers"])
BATCH_SIZE: int = int(_data_cfg["batch_size"])

TRAINING_CONFIG: dict[str, Any] = get_training_config()
PRUNING_CONFIG: dict[str, Any] = get_pruning_config()
ENSEMBLE_CONFIG: dict[str, Any] = get_ensemble_config()

NUM_VARIANTS: int = 10
HUMAN_AUTHENTICITY_RELIABILITY: float = 0.67

CANONICAL_MODEL_ORDER: tuple[str, ...] = (
    "vgg16",
    "vgg19",
    "resnet152",
    "densenet161",
    "efficientnetb3",
    "barlowtwins",
    "clip_vit_b16",
)

OUTPUT_DIR: Path = Path(
    os.environ.get("EXP3_OUTPUT_DIR", "Outputs/Experiment_3_ensemble")
)
EXP1_WEIGHTS_DIR: Path = Path(
    os.environ.get("EXP1_WEIGHTS_DIR", "Outputs/Experiment_1_variants/Weights")
)
EXP2_MPM_DIR: Path = Path(
    os.environ.get(
        "EXP2_MPM_DIR",
        "Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking",
    )
)

DIRS: dict[str, Path] = {
    "output": OUTPUT_DIR,
    "exp1_weights": EXP1_WEIGHTS_DIR,
    "exp2_mpm": EXP2_MPM_DIR,
    "weights": OUTPUT_DIR / "Weights",
    "results": OUTPUT_DIR / "Results",
    "maps": OUTPUT_DIR / "XAI_Maps",
    "mpm": OUTPUT_DIR / "XAI_Maps" / "Multiscale_Pixel_Masking",
    "plots": OUTPUT_DIR / "Plots",
    "heatmaps": OUTPUT_DIR / "Heatmaps",
    "rankings": OUTPUT_DIR / "Ranking_arrays",
    "ranking_plots": OUTPUT_DIR / "Ranking_Plots",
    "training_plots": OUTPUT_DIR / "Training_Plots",
    "training_history": OUTPUT_DIR / "Training_History",
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "vgg16": {
        "class": VGG16AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.28",
        "input_size": 224,
    },
    "vgg19": {
        "class": VGG19AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.34",
        "input_size": 224,
    },
    "resnet152": {
        "class": ResNet152AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.7.2.conv3",
        "input_size": 224,
    },
    "densenet161": {
        "class": DenseNet161AuthenticityPredictor,
        "dataset": DENSENET_DATASET,
        "backbone_dataset": denseNet_dataset,
        "target_layer": "features.denseblock4.denselayer24.conv2",
        "input_size": 300,
    },
    "efficientnetb3": {
        "class": EfficientNetB3AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.8.0",
        "input_size": 224,
    },
    "barlowtwins": {
        "class": BarlowTwinsAuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.7.2.conv3",
        "input_size": 224,
    },
    "clip_vit_b16": {
        "class": CLIPViTAuthenticityPredictor,
        "dataset": CLIP_DATASET,
        "backbone_dataset": clip_dataset,
        "target_layer": "visual_projection",
        "input_size": 224,
    },
}

BASE_WEIGHT_RE = re.compile(r"^([a-z0-9_]+)_(?:exp3a|exp1a)_variant(\d+)_best\.pth$")
LEGACY_PRUNED_WEIGHT_RE = re.compile(
    r"^([a-z0-9_]+)_(?:exp3b|exp1b)_variant(\d+)_greedy_pruned(?:_best)?\.pth$"
)


def _effective_num_workers(device: str | torch.device) -> int:
    """Determine a safe DataLoader worker count respecting SLURM CPU allocations."""
    dev_str = str(device)
    if dev_str.startswith("cpu"):
        return 0
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None and slurm_cpus.isdigit():
        return max(1, min(NUM_WORKERS, int(slurm_cpus)))
    return NUM_WORKERS


def setup_directories() -> None:
    """Create all required Experiment 3 output directories."""
    skip_keys = {"exp1_weights", "exp2_mpm"}
    for key, path in DIRS.items():
        if key not in skip_keys:
            path.mkdir(parents=True, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    """JSON encoder supporting NumPy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        """Convert NumPy types to standard Python serializable equivalents."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# 2. DATA SPLITTING & ORDERING UTILITIES
# ============================================================================


def get_ordered_models(models_filter: Sequence[str] | None = None) -> list[str]:
    """Return model names strictly ordered according to CANONICAL_MODEL_ORDER.

    Args:
        models_filter: Optional subset of model names to retain.

    Returns:
        Ordered list of registered model identifiers.
    """
    if models_filter is None:
        return [m for m in CANONICAL_MODEL_ORDER if m in MODEL_REGISTRY]
    requested = {str(m) for m in models_filter}
    ordered = [
        m for m in CANONICAL_MODEL_ORDER if m in requested and m in MODEL_REGISTRY
    ]
    for extra in models_filter:
        if extra in MODEL_REGISTRY and extra not in ordered:
            ordered.append(extra)
    return ordered


def parse_variant_indices(
    variants: str | Sequence[int] | int | None = "all",
    num_variants: int = NUM_VARIANTS,
) -> list[int]:
    """Parse a variant specification into a sorted list of 1-based variant indices.

    Args:
        variants: Variant selector ('all', comma-separated string '1,2,3',
            integer count, or sequence of ints).
        num_variants: Total number of variants when 'all' or None is specified.

    Returns:
        Sorted list of unique 1-based variant indices.

    Raises:
        ValueError: If no valid positive variant indices are parsed.
    """
    if variants is None:
        return list(range(1, num_variants + 1))
    if isinstance(variants, int):
        if variants <= 0:
            raise ValueError(f"Variant count must be positive, got {variants}")
        return list(range(1, variants + 1))
    if isinstance(variants, str):
        cleaned = variants.strip().lower()
        if cleaned in {"all", ""}:
            return list(range(1, num_variants + 1))
        parsed = [int(tok.strip()) for tok in cleaned.split(",") if tok.strip()]
    else:
        parsed = [int(v) for v in variants]

    unique_sorted = sorted({v for v in parsed if v > 0})
    if not unique_sorted:
        raise ValueError(f"No valid variant indices parsed from: {variants!r}")
    return unique_sorted


def create_global_test_indices(
    dataset_size: int, test_fraction: float = 0.2, seed: int = 42
) -> list[int]:
    """Create global test indices that remain constant across all variants.

    Args:
        dataset_size: Total number of samples in the dataset.
        test_fraction: Fraction of data reserved for the global test set.
        seed: Random seed for reproducible permutation.

    Returns:
        List of integer indices forming the global test partition.
    """
    test_size = int(test_fraction * dataset_size)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=gen).tolist()
    return perm[:test_size]


def create_variant_split(
    backbone_dataset: Dataset,
    global_test_indices: Sequence[int],
    variant_idx: int,
    val_fraction: float = 0.125,
) -> tuple[Subset, Subset, Subset, list[int]]:
    """Create deterministic train/val/test split for a specific variant index.

    Args:
        backbone_dataset: Full dataset instance to partition.
        global_test_indices: Pre-computed global test set indices.
        variant_idx: Variant index (1..10) seeding the train/val shuffle (42 + k).
        val_fraction: Fraction of non-test samples allocated to validation
            (0.125 of 80% remaining = 10% of total dataset = 136 images).

    Returns:
        Tuple of (train_ds, val_ds, test_ds, val_indices).
    """
    total_size = len(backbone_dataset)  # type: ignore[arg-type]
    test_indices_set = set(global_test_indices)

    remaining_indices = [i for i in range(total_size) if i not in test_indices_set]

    gen = torch.Generator().manual_seed(42 + variant_idx)
    perm = torch.randperm(len(remaining_indices), generator=gen).tolist()
    shuffled_remaining = [remaining_indices[i] for i in perm]

    val_size = int(val_fraction * len(shuffled_remaining))
    train_size = len(shuffled_remaining) - val_size

    train_indices = shuffled_remaining[:train_size]
    val_indices = shuffled_remaining[train_size:]

    train_ds = Subset(backbone_dataset, train_indices)
    val_ds = Subset(backbone_dataset, val_indices)
    test_ds = Subset(backbone_dataset, list(global_test_indices))

    return train_ds, val_ds, test_ds, val_indices


def _build_default_global_test_indices() -> dict[str, list[int]]:
    """Build the default global test indices dictionary across all dataset types."""
    return {
        "imagenet": create_global_test_indices(len(imageNet_dataset)),
        "densenet": create_global_test_indices(len(denseNet_dataset)),
        "clip": create_global_test_indices(len(clip_dataset)),
    }


def _dataset_key_for_model(model_name: str) -> str:
    """Return the canonical dataset key ('imagenet', 'densenet', or 'clip') for a model."""
    ds = MODEL_REGISTRY[model_name]["dataset"]
    if ds is CLIP_DATASET:
        return "clip"
    if ds is DENSENET_DATASET:
        return "densenet"
    return "imagenet"


# ============================================================================
# 3. MODEL LOADING, META-LEARNER & METRIC UTILITIES
# ============================================================================


def reset_regression_head(model: nn.Module) -> None:
    """Reinitialize linear layer parameters in the model's regression head.

    Args:
        model: Predictor module containing a ``regression_head`` submodule.
    """
    regression_head = getattr(model, "regression_head", None)
    if isinstance(regression_head, nn.Module):
        for layer in regression_head.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


def load_model_with_weights(
    model_name: str,
    weights_path: Path,
    device: str = "cuda",
    freeze_backbone: bool = True,
) -> nn.Module:
    """Instantiate a registered architecture and load checkpoint weights.

    Args:
        model_name: Key in ``MODEL_REGISTRY``.
        weights_path: Path to the saved ``state_dict`` ``.pth`` file.
        device: Target compute device ('cuda' or 'cpu').
        freeze_backbone: Whether backbone parameters are frozen.

    Returns:
        Model loaded with weights and moved to ``device``.
    """
    model_cls = MODEL_REGISTRY[model_name]["class"]
    model: nn.Module = model_cls(freeze_backbone=freeze_backbone)

    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        debug(f"Loaded weights: {weights_path.name}")
    else:
        warn(f"Weights not found: {weights_path}, using initialized model")

    return model.to(device)


def get_predictions(
    model: nn.Module, dataloader: DataLoader, device: str = "cuda"
) -> torch.Tensor:
    """Run forward inference over ``dataloader`` and concatenate scalar predictions.

    Args:
        model: Evaluated PyTorch model returning ``(outputs, features)``.
        dataloader: DataLoader yielding ``(images, labels)`` batches.
        device: Target compute device.

    Returns:
        Concatenated prediction tensor on CPU.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs, _ = model(images)
            all_preds.append(outputs.detach().cpu())

    return torch.cat(all_preds, dim=0)


def get_labels(dataloader: DataLoader) -> torch.Tensor:
    """Extract and concatenate ground-truth scalar labels from ``dataloader``.

    Args:
        dataloader: DataLoader yielding ``(images, labels)`` batches.

    Returns:
        1D tensor of ground-truth target scores.
    """
    all_labels: list[torch.Tensor] = []
    for _, labels in dataloader:
        squeezed = labels.squeeze()
        if squeezed.ndim == 0:
            squeezed = squeezed.unsqueeze(0)
        all_labels.append(squeezed.cpu())
    return torch.cat(all_labels, dim=0)


class StackingMetaLearner(nn.Module):
    """Level-1 linear combiner for stacking ensembles."""

    def __init__(self, num_base_models: int) -> None:
        """Initialize the linear meta-learner.

        Args:
            num_base_models: Number of base architectures in the ensemble.
        """
        super().__init__()
        self.fc = nn.Linear(num_base_models, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute linear combination of base model predictions.

        Args:
            x: Base prediction feature matrix of shape ``(B, num_base_models)``.

        Returns:
            Stacked prediction tensor of shape ``(B, 1)``.
        """
        return self.fc(x)


def fit_ridge_stacking_meta(
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 1.0,
) -> tuple[StackingMetaLearner, np.ndarray, float, float, float]:
    """Fit unconstrained Ridge regression meta-learner in closed form on validation data.

    Solves:
        w_ext = (X_ext^T X_ext + alpha * I)^{-1} X_ext^T y_val
    where X_ext = [X_val, 1] includes the intercept column.

    Args:
        x_val: Validation base-model prediction matrix of shape ``(N_val, M)``.
        y_val: Validation ground-truth vector of shape ``(N_val,)``.
        alpha: L2 regularization penalty (default 1.0).

    Returns:
        Tuple of ``(meta_learner, weights, intercept, val_mse, val_r2)``.
    """
    x_mat = np.asarray(x_val, dtype=np.float64)
    y_vec = np.asarray(y_val, dtype=np.float64).ravel()
    n_samples, num_models = x_mat.shape

    ones = np.ones((n_samples, 1), dtype=np.float64)
    x_ext = np.hstack([x_mat, ones])

    reg_eye = alpha * np.eye(num_models + 1, dtype=np.float64)
    gram = x_ext.T @ x_ext + reg_eye
    rhs = x_ext.T @ y_vec

    try:
        theta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(gram, rhs, rcond=None)[0]

    weights = theta[:num_models].astype(np.float64)
    intercept = float(theta[num_models])

    y_val_pred = x_mat @ weights + intercept
    val_mse = float(np.mean((y_val_pred - y_vec) ** 2))
    ss_res = float(np.sum((y_vec - y_val_pred) ** 2))
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    val_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    meta_model = StackingMetaLearner(num_base_models=num_models)
    with torch.no_grad():
        meta_model.fc.weight.copy_(
            torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
        )
        meta_model.fc.bias.copy_(torch.tensor([intercept], dtype=torch.float32))

    return meta_model, weights, intercept, val_mse, val_r2


def compute_regression_metrics(
    y_pred: np.ndarray, y_true: np.ndarray
) -> dict[str, float]:
    """Compute MSE, RMSE, MAE, PLCC, SRCC, and KRCC between predictions and targets.

    Args:
        y_pred: 1D array of predicted scores.
        y_true: 1D array of ground-truth scores.

    Returns:
        Dictionary of scalar evaluation metrics.
    """
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    true = np.asarray(y_true, dtype=np.float64).ravel()

    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - true)))

    if pred.size >= 2 and np.std(pred) > 1e-12 and np.std(true) > 1e-12:
        plcc_val, _ = pearsonr(pred, true)
        srcc_val, _ = spearmanr(pred, true)
        krcc_val, _ = kendalltau(pred, true)
        plcc = float(plcc_val)
        srcc = float(srcc_val)
        krcc = float(krcc_val)
    else:
        plcc = 0.0
        srcc = 0.0
        krcc = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "plcc": plcc,
        "srcc": srcc,
        "krcc": krcc,
    }


def compute_prediction_reliability(
    prediction_vectors: Sequence[np.ndarray],
    human_reliability: float = HUMAN_AUTHENTICITY_RELIABILITY,
) -> tuple[float, float]:
    """Compute Model Prediction Reliability and Combined Correlation Ceiling.

    Model Prediction Reliability is the mean pairwise Pearson correlation between
    prediction vectors generated across the 10 variants. The Combined Correlation
    Ceiling is sqrt(r_human * r_model).

    Args:
        prediction_vectors: Sequence of 1D test-set prediction arrays across variants.
        human_reliability: Spearman-Brown split-half human reliability (r_human).

    Returns:
        Tuple of ``(prediction_reliability, combined_correlation_ceiling)``.
    """
    if len(prediction_vectors) < 2:
        return 0.0, 0.0

    pairwise_corrs: list[float] = []
    for vec_a, vec_b in combinations(prediction_vectors, 2):
        a = np.asarray(vec_a, dtype=np.float64).ravel()
        b = np.asarray(vec_b, dtype=np.float64).ravel()
        if a.size >= 2 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
            r_val, _ = pearsonr(a, b)
            if np.isfinite(r_val):
                pairwise_corrs.append(float(r_val))

    if not pairwise_corrs:
        return 0.0, 0.0

    r_model = float(np.mean(pairwise_corrs))
    ceiling = (
        float(np.sqrt(human_reliability * r_model))
        if r_model > 0.0 and human_reliability > 0.0
        else 0.0
    )
    return r_model, ceiling


def _summarize_variant_metrics(
    variant_entries: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean and std dictionaries across variant metric records."""
    metric_keys = ("mse", "rmse", "mae", "plcc", "srcc", "krcc")
    if not variant_entries:
        zeros = {k: 0.0 for k in metric_keys}
        return zeros, zeros.copy()

    mean_dict = {
        k: float(np.mean([float(entry[k]) for entry in variant_entries]))
        for k in metric_keys
    }
    std_dict = {
        k: float(np.std([float(entry[k]) for entry in variant_entries]))
        for k in metric_keys
    }
    return mean_dict, std_dict


def discover_base_checkpoints(
    models_filter: Sequence[str] | None = None,
    variant_indices: Sequence[int] | None = None,
) -> dict[str, dict[int, Path]]:
    """Discover unpruned base-model checkpoints across Experiment 1A / 3A weight dirs.

    Prefers ``DIRS['exp1_weights']`` (``Outputs/Experiment_1_variants/Weights``)
    when ``DIRS['weights']`` is at its default location, or searches ``DIRS['weights']``
    exclusively when ``DIRS['weights']`` has been overridden (e.g. in isolated tests).

    Args:
        models_filter: Optional subset of model names.
        variant_indices: Optional subset of variant indices (1..10).

    Returns:
        Mapping ``{model_name: {variant_idx: checkpoint_path}}``.
    """
    ordered_models = get_ordered_models(models_filter)
    allowed_models = set(ordered_models)
    allowed_variants = set(variant_indices) if variant_indices is not None else None

    weights_dir = DIRS["weights"]
    exp1_dir = DIRS.get("exp1_weights", EXP1_WEIGHTS_DIR)

    default_weights_dir = (OUTPUT_DIR / "Weights").resolve()
    if weights_dir.resolve() != default_weights_dir:
        candidate_dirs = [weights_dir]
    else:
        candidate_dirs = [exp1_dir, weights_dir]

    discovered: dict[str, dict[int, Path]] = defaultdict(dict)

    for search_dir in candidate_dirs:
        if not search_dir.exists():
            continue
        pth_files = sorted(search_dir.glob("*.pth"))
        for path in pth_files:
            match = BASE_WEIGHT_RE.match(path.name)
            if not match:
                continue
            model_name = match.group(1)
            variant_idx = int(match.group(2))
            if model_name not in allowed_models:
                continue
            if allowed_variants is not None and variant_idx not in allowed_variants:
                continue
            if variant_idx not in discovered[model_name]:
                discovered[model_name][variant_idx] = path

    # Fallback to legacy pruned checkpoints only if no unpruned checkpoints exist
    if not discovered:
        for search_dir in candidate_dirs:
            if not search_dir.exists():
                continue
            for path in sorted(search_dir.glob("*.pth")):
                match = LEGACY_PRUNED_WEIGHT_RE.match(path.name)
                if not match:
                    continue
                model_name = match.group(1)
                variant_idx = int(match.group(2))
                if model_name not in allowed_models:
                    continue
                if allowed_variants is not None and variant_idx not in allowed_variants:
                    continue
                if variant_idx not in discovered[model_name]:
                    discovered[model_name][variant_idx] = path

    return {
        m: dict(sorted(discovered[m].items()))
        for m in ordered_models
        if discovered.get(m)
    }


# ============================================================================
# 4. EXPERIMENT 3A: TRAIN ALL VARIANTS (LEGACY / OPTIONAL)
# ============================================================================


def experiment_3a_train_all_variants(
    models_to_train: list[str] | None = None,
    global_test_indices: dict[str, list[int]] | None = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[int, list[int]]]]:
    """Train 10 variants per model architecture (optional legacy stage).

    By default, Experiment 3 reuses the 70 unpruned checkpoints from Experiment 1A
    (ADR-0005). This function is retained for backward compatibility.

    Args:
        models_to_train: List of model names to train (None = all 7 models).
        global_test_indices: Pre-computed test indices per dataset type.
        save_plots: Whether to save training loss curves.
        verbose: Whether to log detailed per-variant progress.

    Returns:
        Tuple of ``(results, variant_val_indices)``.
    """
    info("=" * 80)
    info("EXPERIMENT 3A: TRAINING ALL VARIANTS")
    info("=" * 80)

    DIRS["weights"].mkdir(parents=True, exist_ok=True)
    DIRS["training_plots"].mkdir(parents=True, exist_ok=True)
    DIRS["training_history"].mkdir(parents=True, exist_ok=True)

    ordered_models = get_ordered_models(models_to_train)

    if global_test_indices is None:
        global_test_indices = _build_default_global_test_indices()
    else:
        global_test_indices.setdefault("clip", global_test_indices.get("imagenet", []))

    results: dict[str, Any] = {}
    variant_val_indices: dict[str, dict[int, list[int]]] = {}

    device = torch.device(TRAINING_CONFIG["device"])
    num_workers = _effective_num_workers(device)

    for idx, model_name in enumerate(ordered_models, 1):
        info(f"[{idx}/{len(ordered_models)}] Training {model_name.upper()}")
        info("-" * 80)

        try:
            config = MODEL_REGISTRY[model_name]
            backbone_dataset = config["backbone_dataset"]
            ds_key = _dataset_key_for_model(model_name)
            test_indices = global_test_indices[ds_key]

            results[model_name] = {}
            variant_val_indices[model_name] = {}

            for variant_idx in range(1, NUM_VARIANTS + 1):
                weights_path = (
                    DIRS["weights"]
                    / f"{model_name}_exp3a_variant{variant_idx}_best.pth"
                )
                if weights_path.exists():
                    info(
                        f"Variant {variant_idx}/{NUM_VARIANTS} for {model_name} "
                        "already exists, skipping..."
                    )
                    _, _, _, val_indices = create_variant_split(
                        backbone_dataset, test_indices, variant_idx
                    )
                    variant_val_indices[model_name][variant_idx] = val_indices
                    continue

                info(f"Variant {variant_idx}/{NUM_VARIANTS} for {model_name}")

                model = config["class"](
                    freeze_backbone=TRAINING_CONFIG["freeze_backbone"]
                )
                reset_regression_head(model)

                train_ds, val_ds, _test_ds, val_indices = create_variant_split(
                    backbone_dataset, test_indices, variant_idx
                )
                variant_val_indices[model_name][variant_idx] = val_indices

                train_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=num_workers,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=num_workers,
                )

                dataloaders = {"train": train_loader, "val": val_loader}
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=TRAINING_CONFIG["learning_rate"]
                )

                if verbose:
                    info(f"  Training on device: {device}")
                    info(f"  Train size: {len(train_ds)}, Val size: {len(val_ds)}")

                best_model, history = train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=TRAINING_CONFIG["max_epochs"],
                    device=TRAINING_CONFIG["device"],
                    patience=TRAINING_CONFIG["patience"],
                )

                torch.save(best_model.state_dict(), weights_path)
                info(f"Variant weights saved to: {weights_path}")

                history_path = (
                    DIRS["training_history"]
                    / f"{model_name}_exp3a_variant{variant_idx}_history.npy"
                )
                np.save(history_path, history)

                if save_plots:
                    try:
                        matplotlib.use("Agg")
                        plt.figure(figsize=(10, 6))
                        plt.plot(history["train_loss"], label="Train Loss")
                        plt.plot(history["val_loss"], label="Val Loss")
                        plt.xlabel("Epoch")
                        plt.ylabel("MSE Loss")
                        plt.title(f"{model_name} Variant {variant_idx} Training")
                        plt.legend()
                        plt.grid(True)

                        plot_path = (
                            DIRS["training_plots"]
                            / f"{model_name}_exp3a_variant{variant_idx}_training.png"
                        )
                        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                        plt.close()
                    except (OSError, RuntimeError, ValueError) as exc:
                        warn(f"Could not save plot: {exc}")

                results[model_name][f"variant{variant_idx}"] = {
                    "final_val_loss": history["val_loss"][-1],
                    "best_val_loss": min(history["val_loss"]),
                    "epochs_trained": len(history["train_loss"]),
                    "weights_path": str(weights_path),
                    "val_indices_count": len(val_indices),
                }

                del model, best_model, train_loader, val_loader
                clear_gpu_memory()

        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            error(f"Error training {model_name}: {exc}")
            error(traceback.format_exc())
            results[model_name] = {"error": str(exc)}

        finally:
            cleanup_model_and_data(
                model=locals().get("model"),
                dataloaders=locals().get("dataloaders"),
                optimizer=locals().get("optimizer"),
            )
            clear_gpu_memory()

    val_indices_path = DIRS["results"] / "variant_val_indices.json"
    DIRS["results"].mkdir(parents=True, exist_ok=True)
    serializable_indices = {
        model: {str(k): v for k, v in variants.items()}
        for model, variants in variant_val_indices.items()
    }
    with open(val_indices_path, "w") as f:
        json.dump(serializable_indices, f, cls=NpEncoder)

    info("=" * 80)
    info("EXPERIMENT 3A: TRAINING COMPLETE")
    info("=" * 80)

    return results, variant_val_indices


# ============================================================================
# 5. EXPERIMENT 3B: PRUNE ALL VARIANTS (LEGACY / OPTIONAL)
# ============================================================================


def experiment_3b_prune_all_variants(
    models_to_prune: list[str] | None = None,
    variant_val_indices: dict[str, dict[int, list[int]]] | None = None,
    global_test_indices: dict[str, list[int]] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Prune trained variants using greedy pruning on the validation set (legacy stage).

    Deprecated in ADR-0005 in favor of unpruned Experiment 1A base checkpoints;
    retained as an optional function for backward compatibility.

    Args:
        models_to_prune: List of model names to prune (None = all available).
        variant_val_indices: Mapping ``model_name -> {variant_idx -> val_indices}``.
        global_test_indices: Pre-computed test indices per dataset type.
        verbose: Whether to log detailed progress.

    Returns:
        Dictionary of pruning metrics per model and variant.
    """
    info("=" * 80)
    info("EXPERIMENT 3B: PRUNING ALL VARIANTS (ON VALIDATION SET)")
    info("=" * 80)

    DIRS["rankings"].mkdir(parents=True, exist_ok=True)
    DIRS["ranking_plots"].mkdir(parents=True, exist_ok=True)

    if global_test_indices is None:
        global_test_indices = _build_default_global_test_indices()
    else:
        global_test_indices.setdefault("clip", global_test_indices.get("imagenet", []))

    checkpoints = discover_base_checkpoints(models_filter=models_to_prune)
    if not checkpoints:
        error(f"No trained weights found in {DIRS['weights']}")
        return {}

    results: dict[str, Any] = {}
    device = torch.device(TRAINING_CONFIG["device"])
    num_workers = _effective_num_workers(device)
    criterion = nn.MSELoss()

    for idx, (model_name, variant_map) in enumerate(checkpoints.items(), 1):
        info(f"[{idx}/{len(checkpoints)}] Pruning {model_name.upper()}")
        info("-" * 80)

        try:
            config = MODEL_REGISTRY[model_name]
            backbone_dataset = config["backbone_dataset"]
            target_layer = config["target_layer"]
            ds_key = _dataset_key_for_model(model_name)
            test_indices = global_test_indices[ds_key]

            results[model_name] = {}

            for variant_idx, weights_path in variant_map.items():
                variant_tag = f"variant{variant_idx}"
                pruned_weights_path = (
                    DIRS["weights"]
                    / f"{model_name}_exp3b_{variant_tag}_greedy_pruned.pth"
                )
                if pruned_weights_path.exists():
                    info(f"{model_name} {variant_tag} already pruned, skipping...")
                    continue

                if (
                    variant_val_indices is not None
                    and model_name in variant_val_indices
                    and variant_idx in variant_val_indices[model_name]
                ):
                    val_indices = variant_val_indices[model_name][variant_idx]
                else:
                    _, _, _, val_indices = create_variant_split(
                        backbone_dataset, test_indices, variant_idx
                    )

                val_ds = Subset(backbone_dataset, val_indices)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=num_workers,
                )

                if verbose:
                    info(f"  Validation set size: {len(val_ds)}")

                model = config["class"](freeze_backbone=False)
                model.load_state_dict(
                    torch.load(weights_path, map_location=device, weights_only=True)
                )

                pruner = FeatureMapsPruner(
                    model=model,
                    dataloader=val_loader,
                    layer_name=target_layer,
                    criterion=criterion,
                    eval_function=test_model,
                    device=device,
                )

                importance_path = (
                    DIRS["rankings"]
                    / f"{model_name}_exp3b_{variant_tag}_importance.npy"
                )
                pruner.compute_importance_scores(
                    save_path=str(importance_path),
                    force_recompute=PRUNING_CONFIG["force_recompute"],
                )

                if pruner.baseline_mse is None:
                    pruner.baseline_mse, pruner.baseline_rmse = pruner._evaluate_model()

                try:
                    plot_path = (
                        DIRS["ranking_plots"]
                        / f"{model_name}_exp3b_{variant_tag}_importance.png"
                    )
                    pruner.plot_importance_scores(save_path=str(plot_path))
                except (OSError, RuntimeError, ValueError) as exc:
                    warn(f"Could not save importance plot: {exc}")

                pruning_results = pruner.greedy_pruning(
                    model_save_path=str(pruned_weights_path)
                )

                results[model_name][variant_tag] = {
                    "baseline_mse": pruning_results["baseline_mse"],
                    "baseline_rmse": pruning_results["baseline_rmse"],
                    "final_mse": pruning_results["final_mse"],
                    "final_rmse": pruning_results["final_rmse"],
                    "improvement_mse": pruning_results["improvement_mse"],
                    "improvement_rmse": pruning_results["improvement_rmse"],
                    "removed_features": pruning_results["removed_features"],
                    "num_removed": len(pruning_results["removed_features"]),
                    "reduction_percentage": pruning_results["reduction_percentage"],
                    "pruned_weights_path": str(pruned_weights_path),
                    "original_weights_path": str(weights_path),
                }

                del model, pruner, val_loader
                clear_gpu_memory()

        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            error(f"Error pruning {model_name}: {exc}")
            error(traceback.format_exc())
            results[model_name] = {"error": str(exc)}

        finally:
            cleanup_model_and_data(
                model=locals().get("model"),
                dataloaders=locals().get("val_loader"),
                optimizer=None,
            )
            clear_gpu_memory()

    DIRS["results"].mkdir(parents=True, exist_ok=True)
    pruning_results_path = DIRS["results"] / "experiment_3b_pruning_results.json"
    with open(pruning_results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    info("=" * 80)
    info("EXPERIMENT 3B: PRUNING COMPLETE")
    info("=" * 80)

    return results


# ============================================================================
# 6. EXPERIMENT 3C: EVALUATE VARIANT-ALIGNED ENSEMBLES
# ============================================================================


def experiment_3c_evaluate_ensemble(
    models_filter: list[str] | None = None,
    global_test_indices: dict[str, list[int]] | None = None,
    device: str = "cuda",
    ensemble_mode: Sequence[str] = ("bagging", "stacking"),
    variants: str | Sequence[int] | int | None = "all",
    ridge_alpha: float = 1.0,
    stacking_cv_folds: int = 5,
    stacking_cv_repeats: int = 1,
) -> dict[str, Any]:
    """Evaluate 10 variant-aligned 7-model Bagging and Stacking ensembles (Stage 3C).

    For each variant k in 1..10:
    - Reconstructs validation split val_k (136 images, seed = 42 + k).
    - Computes base predictions X_val^(k) in R^{136 x 7} and X_test^(k) in R^{273 x 7}
      strictly ordered by CANONICAL_MODEL_ORDER.
    - Evaluates Bagging (unweighted arithmetic mean) on the 273 global test images.
    - Fits StackingMetaLearner on (X_val^(k), y_val^(k)) via closed-form Ridge
      regression (alpha = 1.0), saves stacking_exp3_variant{k}_meta.pth and
      stacking_coefficients.json, and evaluates on the 273 global test images.

    Args:
        models_filter: Optional list of model names to include (None = all 7 models).
        global_test_indices: Optional pre-computed test indices per dataset type.
        device: Compute device ('cuda' or 'cpu').
        ensemble_mode: Ensemble strategies to run ('bagging', 'stacking').
        variants: Variant indices to evaluate ('all', '1,2,3', or sequence of ints).
        ridge_alpha: Ridge regularization parameter alpha for stacking meta-learner.
        stacking_cv_folds: Kept for backward signature compatibility.
        stacking_cv_repeats: Kept for backward signature compatibility.

    Returns:
        Dictionary of evaluation results adhering to the ADR-0005 JSON schema.
    """
    del stacking_cv_folds, stacking_cv_repeats
    info("=" * 80)
    info("EXPERIMENT 3C: EVALUATING VARIANT-ALIGNED ENSEMBLES (BAG & STACK)")
    info("=" * 80)

    requested_variants = parse_variant_indices(variants)
    ordered_models = get_ordered_models(models_filter)
    modes = [m.lower() for m in ensemble_mode]

    checkpoints_by_model = discover_base_checkpoints(
        models_filter=ordered_models,
        variant_indices=requested_variants,
    )
    if not checkpoints_by_model:
        error(
            f"No base model checkpoints found in {DIRS.get('exp1_weights')} "
            f"or {DIRS['weights']}"
        )
        return {}

    total_found = sum(len(v) for v in checkpoints_by_model.values())
    info(
        f"Discovered {total_found} checkpoints across "
        f"{len(checkpoints_by_model)} architectures for variants {requested_variants}"
    )

    if global_test_indices is None:
        global_test_indices = _build_default_global_test_indices()
    else:
        default_idx = next(iter(global_test_indices.values()))
        global_test_indices.setdefault("imagenet", default_idx)
        global_test_indices.setdefault("densenet", default_idx)
        global_test_indices.setdefault("clip", default_idx)

    num_workers = _effective_num_workers(device)

    datasets_by_key: dict[str, Dataset] = {
        "imagenet": imageNet_dataset,
        "densenet": denseNet_dataset,
        "clip": clip_dataset,
    }
    test_loaders: dict[str, DataLoader] = {}
    for ds_key, ds_obj in datasets_by_key.items():
        test_sub = Subset(ds_obj, global_test_indices[ds_key])
        test_loaders[ds_key] = DataLoader(
            test_sub,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
        )

    y_test = get_labels(test_loaders["imagenet"]).numpy().astype(np.float64)
    info(f"Global held-out test set size: {len(y_test)}")

    need_validation = "stacking" in modes
    val_loaders_by_variant: dict[int, dict[str, DataLoader]] = {}
    y_val_by_variant: dict[int, np.ndarray] = {}

    if need_validation:
        for var_idx in requested_variants:
            val_loaders_by_variant[var_idx] = {}
            for ds_key, ds_obj in datasets_by_key.items():
                _, val_ds, _, _ = create_variant_split(
                    ds_obj, global_test_indices[ds_key], var_idx
                )
                val_loaders_by_variant[var_idx][ds_key] = DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=num_workers,
                )
            y_val_by_variant[var_idx] = (
                get_labels(val_loaders_by_variant[var_idx]["imagenet"])
                .numpy()
                .astype(np.float64)
            )

    # -------------------------------------------------------------------------
    # 1. EVALUATE BASELINE ARCHITECTURES & CACHE PREDICTIONS PER VARIANT
    # -------------------------------------------------------------------------
    baseline_models: dict[str, Any] = {}
    test_preds_by_variant: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    val_preds_by_variant: dict[int, dict[str, np.ndarray]] = defaultdict(dict)

    for model_name in ordered_models:
        if model_name not in checkpoints_by_model:
            continue
        ds_key = _dataset_key_for_model(model_name)
        test_loader = test_loaders[ds_key]
        variant_records: list[dict[str, Any]] = []
        model_test_vectors: list[np.ndarray] = []

        for var_idx, weights_path in checkpoints_by_model[model_name].items():
            info(
                f"  Evaluating {model_name} variant {var_idx} ({weights_path.name})..."
            )
            model = load_model_with_weights(
                model_name, weights_path, device=device, freeze_backbone=False
            )

            test_preds = (
                get_predictions(model, test_loader, device=device)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            test_preds_by_variant[var_idx][model_name] = test_preds
            model_test_vectors.append(test_preds)

            if need_validation:
                val_loader = val_loaders_by_variant[var_idx][ds_key]
                val_preds = (
                    get_predictions(model, val_loader, device=device)
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                val_preds_by_variant[var_idx][model_name] = val_preds

            del model
            clear_gpu_memory()

            metrics = compute_regression_metrics(test_preds, y_test)
            variant_records.append(
                {
                    "variant_idx": var_idx,
                    "weights_path": str(weights_path),
                    **metrics,
                }
            )

        mean_metrics, std_metrics = _summarize_variant_metrics(variant_records)
        pred_rel, comb_ceil = compute_prediction_reliability(model_test_vectors)

        baseline_models[model_name] = {
            "variants": variant_records,
            "num_variants": len(variant_records),
            "mean": mean_metrics,
            "std": std_metrics,
            "prediction_reliability": pred_rel,
            "combined_correlation_ceiling": comb_ceil,
        }

    # -------------------------------------------------------------------------
    # 2. CONSTRUCT & EVALUATE 10 VARIANT-ALIGNED ENSEMBLES
    # -------------------------------------------------------------------------
    DIRS["weights"].mkdir(parents=True, exist_ok=True)
    DIRS["results"].mkdir(parents=True, exist_ok=True)

    bagging_variants: list[dict[str, Any]] = []
    bagging_test_vectors: list[np.ndarray] = []

    stacking_variants: list[dict[str, Any]] = []
    stacking_test_vectors: list[np.ndarray] = []
    stacking_coeff_records: list[dict[str, Any]] = []

    available_variants = sorted(test_preds_by_variant.keys())

    for var_idx in available_variants:
        active_models = [
            m for m in ordered_models if m in test_preds_by_variant[var_idx]
        ]
        if not active_models:
            continue

        x_test_k = np.column_stack(
            [test_preds_by_variant[var_idx][m] for m in active_models]
        )

        if "bagging" in modes:
            y_pred_bag = np.mean(x_test_k, axis=1)
            bagging_test_vectors.append(y_pred_bag)
            bag_metrics = compute_regression_metrics(y_pred_bag, y_test)
            bagging_variants.append(
                {
                    "variant_idx": var_idx,
                    "num_models": len(active_models),
                    "models": active_models,
                    **bag_metrics,
                }
            )

        if "stacking" in modes:
            x_val_k = np.column_stack(
                [val_preds_by_variant[var_idx][m] for m in active_models]
            )
            y_val_k = y_val_by_variant[var_idx]

            meta_model, weights_vec, intercept_val, val_mse, val_r2 = (
                fit_ridge_stacking_meta(x_val_k, y_val_k, alpha=ridge_alpha)
            )

            meta_path = DIRS["weights"] / f"stacking_exp3_variant{var_idx}_meta.pth"
            torch.save(meta_model.state_dict(), meta_path)

            y_pred_stack = x_test_k @ weights_vec + intercept_val
            stacking_test_vectors.append(y_pred_stack)
            stack_metrics = compute_regression_metrics(y_pred_stack, y_test)

            weights_list = [float(w) for w in weights_vec]
            weights_by_model = {
                m_name: float(w)
                for m_name, w in zip(active_models, weights_vec, strict=True)
            }

            stacking_variants.append(
                {
                    "variant_idx": var_idx,
                    "num_models": len(active_models),
                    "models": active_models,
                    "weights": weights_list,
                    "weights_by_model": weights_by_model,
                    "intercept": float(intercept_val),
                    "val_mse": float(val_mse),
                    "val_r2": float(val_r2),
                    **stack_metrics,
                }
            )

            stacking_coeff_records.append(
                {
                    "variant_idx": var_idx,
                    "models": active_models,
                    "weights": weights_list,
                    "weights_by_model": weights_by_model,
                    "intercept": float(intercept_val),
                    "val_mse": float(val_mse),
                    "val_r2": float(val_r2),
                    "meta_weights_path": str(meta_path),
                }
            )

    bagging_result: dict[str, Any] = {}
    if bagging_variants:
        bag_mean, bag_std = _summarize_variant_metrics(bagging_variants)
        bag_rel, bag_ceil = compute_prediction_reliability(bagging_test_vectors)
        bagging_result = {
            "variants": bagging_variants,
            "num_variants": len(bagging_variants),
            "mean": bag_mean,
            "std": bag_std,
            "prediction_reliability": bag_rel,
            "combined_correlation_ceiling": bag_ceil,
            **bag_mean,
        }

    stacking_result: dict[str, Any] = {}
    if stacking_variants:
        stack_mean, stack_std = _summarize_variant_metrics(stacking_variants)
        stack_rel, stack_ceil = compute_prediction_reliability(stacking_test_vectors)
        stacking_result = {
            "variants": stacking_variants,
            "num_variants": len(stacking_variants),
            "mean": stack_mean,
            "std": stack_std,
            "prediction_reliability": stack_rel,
            "combined_correlation_ceiling": stack_ceil,
            **stack_mean,
        }

        # Export stacking_coefficients.json
        mean_weights_by_model: dict[str, float] = {}
        std_weights_by_model: dict[str, float] = {}
        for m_name in ordered_models:
            m_vals = [
                rec["weights_by_model"][m_name]
                for rec in stacking_coeff_records
                if m_name in rec["weights_by_model"]
            ]
            if m_vals:
                mean_weights_by_model[m_name] = float(np.mean(m_vals))
                std_weights_by_model[m_name] = float(np.std(m_vals))

        intercepts = [rec["intercept"] for rec in stacking_coeff_records]
        coeff_summary = {
            "ridge_alpha": float(ridge_alpha),
            "model_order": ordered_models,
            "variants": stacking_coeff_records,
            "mean_weights": mean_weights_by_model,
            "std_weights": std_weights_by_model,
            "mean_intercept": float(np.mean(intercepts)) if intercepts else 0.0,
            "std_intercept": float(np.std(intercepts)) if intercepts else 0.0,
        }
        coeff_path = DIRS["weights"] / "stacking_coefficients.json"
        with open(coeff_path, "w") as f:
            json.dump(coeff_summary, f, indent=2, cls=NpEncoder)
        info(f"Saved stacking coefficients to: {coeff_path}")

    comparison_summary: dict[str, Any] = {
        "bagging": (
            {
                "mean": bagging_result["mean"],
                "std": bagging_result["std"],
                "num_variants": bagging_result["num_variants"],
                "prediction_reliability": bagging_result["prediction_reliability"],
                "combined_correlation_ceiling": bagging_result[
                    "combined_correlation_ceiling"
                ],
            }
            if bagging_result
            else None
        ),
        "stacking": (
            {
                "mean": stacking_result["mean"],
                "std": stacking_result["std"],
                "num_variants": stacking_result["num_variants"],
                "prediction_reliability": stacking_result["prediction_reliability"],
                "combined_correlation_ceiling": stacking_result[
                    "combined_correlation_ceiling"
                ],
            }
            if stacking_result
            else None
        ),
        "baseline_architectures": {
            m_name: {
                "mean": m_data["mean"],
                "std": m_data["std"],
                "num_variants": m_data["num_variants"],
                "prediction_reliability": m_data["prediction_reliability"],
                "combined_correlation_ceiling": m_data["combined_correlation_ceiling"],
            }
            for m_name, m_data in baseline_models.items()
        },
    }

    full_results: dict[str, Any] = {
        "bagging": bagging_result if bagging_result else None,
        "stacking": stacking_result if stacking_result else None,
        "baseline_models": baseline_models,
        "comparison_summary": comparison_summary,
        # Backward-compatible aliases
        "pruned_ensemble_bagging": bagging_result if bagging_result else None,
        "pruned_ensemble_stacking": stacking_result if stacking_result else None,
    }

    results_path = DIRS["results"] / "experiment_3c_ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, cls=NpEncoder)
    info(f"Saved ensemble evaluation results to: {results_path}")

    comparison_path = DIRS["results"] / "experiment_3c_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison_summary, f, indent=2, cls=NpEncoder)
    info(f"Saved ensemble comparison summary to: {comparison_path}")

    info("=" * 80)
    info("EXPERIMENT 3C: EVALUATION COMPLETE")
    info("=" * 80)

    return full_results


# ============================================================================
# 7. EXPERIMENT 3D: ENSEMBLE MPM HEATMAPS, PROTOTYPES & CONSISTENCY
# ============================================================================


def _load_stacking_weights_for_variant(
    variant_idx: int,
    ordered_models: Sequence[str],
) -> np.ndarray | None:
    """Load the fitted StackingMetaLearner weight vector for ``variant_idx``."""
    num_models = len(ordered_models)
    meta_path = DIRS["weights"] / f"stacking_exp3_variant{variant_idx}_meta.pth"
    if meta_path.exists():
        state_dict = torch.load(meta_path, map_location="cpu", weights_only=True)
        weight_tensor = state_dict.get("fc.weight")
        if weight_tensor is not None and weight_tensor.numel() == num_models:
            return weight_tensor.detach().cpu().numpy().astype(np.float64).ravel()

    coeff_path = DIRS["weights"] / "stacking_coefficients.json"
    if coeff_path.exists():
        try:
            with open(coeff_path) as f:
                coeff_data = json.load(f)
            for entry in coeff_data.get("variants", []):
                if int(entry.get("variant_idx", -1)) == variant_idx:
                    by_model = entry.get("weights_by_model", {})
                    if all(m in by_model for m in ordered_models):
                        return np.array(
                            [float(by_model[m]) for m in ordered_models],
                            dtype=np.float64,
                        )
                    weights = entry.get("weights", [])
                    if len(weights) == num_models:
                        return np.asarray(weights, dtype=np.float64)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warn(f"Could not read stacking_coefficients.json: {exc}")

    legacy_path = DIRS["weights"] / "stacking_meta_weights.pth"
    if legacy_path.exists():
        state_dict = torch.load(legacy_path, map_location="cpu", weights_only=True)
        weight_tensor = state_dict.get("fc.weight")
        if weight_tensor is not None and weight_tensor.numel() == num_models:
            return weight_tensor.detach().cpu().numpy().astype(np.float64).ravel()

    return None


def _normalize_3d_maps(raw_maps: np.ndarray) -> np.ndarray:
    """Normalize each 2D slice of a 3D ``(N, H, W)`` array to ``[-1.0, 1.0]`` float32."""
    n_images = raw_maps.shape[0]
    normalized = np.empty(raw_maps.shape, dtype=np.float32)
    for idx in range(n_images):
        normalized[idx] = normalize_data(
            raw_maps[idx].astype(np.float32), min_range=-1.0, max_range=1.0
        ).astype(np.float32)
    return normalized


def _exp2_mpm_path(model_name: str, variant_idx: int) -> Path:
    """Return the Experiment 2 MPM .npy path for ``(model_name, variant_idx)``."""
    mpm_dir = DIRS.get("exp2_mpm", EXP2_MPM_DIR)
    return mpm_dir / f"{model_name}_exp1a_variant{variant_idx}_maps.npy"


def experiment_3d_plot_ensemble_heatmaps(
    global_test_indices: dict[str, list[int]] | None = None,
    models_filter: list[str] | None = None,
    num_sample_images: int = 3,
    device: str = "cuda",
    save_plots: bool = True,
    best_variant_per_arch: bool = False,
    sigma_override: list[int] | None = None,
    pixel_batch_override: int | None = None,
    modes: Sequence[str] = ("bagging", "stacking"),
    variants: str | Sequence[int] | int | None = "all",
    reuse_exp2_maps: bool = True,
    live_mpm: bool = False,
    target_resolution: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """Generate per-variant and prototype 273-image MPM heatmaps for ensembles (Stage 3D).

    Supports two execution modes:
    1. Map Reuse Mode (``reuse_exp2_maps=True``, default):
       Loads the 7 precomputed Experiment 2 MPM arrays from
       ``Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking/``,
       bilinearly resizes DenseNet161 maps from 300x300 to 224x224 via
       ``uniform_heatmaps``, computes the linear combination across all 273 test
       images, and normalizes each 2D map to ``[-1.0, 1.0]``.
    2. Live MPM Mode (``live_mpm=True`` or fallback when Exp 2 maps are missing):
       Runs ``MultiscalePixelMasking(..., normalize=False)`` across the 7 base
       models on the held-out test set and accumulates raw prediction drops.

    Saves per-variant ``(273, 224, 224)`` arrays:
    - ``{mode}_exp3_variant{1..10}_maps.npy``
    and architectural prototype heatmaps:
    - ``{mode}_exp3_prototype_maps.npy``
    and evaluates Within-Ensemble Attribution Consistency and Prototype Agreement.

    Args:
        global_test_indices: Pre-computed test indices per dataset type.
        models_filter: Optional list of model names to include (None = all 7).
        num_sample_images: Number of quantile sample overlays to render.
        device: Compute device for live MPM mode ('cuda' or 'cpu').
        save_plots: Whether to save similarity matrices, distributions, and overlays.
        best_variant_per_arch: Kept for backward signature compatibility (ignored).
        sigma_override: Optional override for MPM occluder sizes.
        pixel_batch_override: Optional override for MPM pixel batch size.
        modes: Ensemble strategies to explain ('bagging', 'stacking').
        variants: Variant indices to process ('all', '1,2,3', or sequence of ints).
        reuse_exp2_maps: Whether to reuse precomputed Experiment 2 MPM maps.
        live_mpm: Force live end-to-end MPM forward passes.
        target_resolution: Spatial resolution ``(H, W)`` for output maps.

    Returns:
        Dictionary summarizing saved map paths, prototypes, and XAI consistency.
    """
    del best_variant_per_arch
    xai_cfg = get_xai_config()
    sigma = sigma_override if sigma_override is not None else xai_cfg["sigma"]
    px_batch = (
        pixel_batch_override
        if pixel_batch_override is not None
        else xai_cfg["px_batch"]
    )
    stride = xai_cfg.get("stride", 1)
    mask_val = float(xai_cfg.get("mask_val", 0.0))

    variant_indices = parse_variant_indices(variants)
    ordered_models = get_ordered_models(models_filter)
    active_modes = [m.lower() for m in modes if m.lower() in {"bagging", "stacking"}]

    info("=" * 80)
    info(f"EXPERIMENT 3D: ENSEMBLE MPM ATTRIBUTION MAPS FOR MODES: {active_modes}")
    info("=" * 80)

    mpm_out_dir = DIRS.get("mpm", OUTPUT_DIR / "XAI_Maps" / "Multiscale_Pixel_Masking")
    plots_dir = DIRS.get("plots", OUTPUT_DIR / "Plots")
    results_dir = DIRS["results"]
    mpm_out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve ensemble weights per variant and mode
    weights_by_mode: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    num_models = len(ordered_models)
    if num_models == 0:
        error("No valid models selected for Stage 3D.")
        return {}

    for var_idx in variant_indices:
        if "bagging" in active_modes:
            weights_by_mode["bagging"][var_idx] = np.ones(
                num_models, dtype=np.float64
            ) / float(num_models)
        if "stacking" in active_modes:
            stack_w = _load_stacking_weights_for_variant(var_idx, ordered_models)
            if stack_w is None:
                warn(
                    f"Stacking weights for variant {var_idx} not found; "
                    "running Stage 3C evaluation first to fit meta-learner..."
                )
                experiment_3c_evaluate_ensemble(
                    models_filter=ordered_models,
                    global_test_indices=global_test_indices,
                    device=str(device),
                    ensemble_mode=["stacking"],
                    variants=[var_idx],
                )
                stack_w = _load_stacking_weights_for_variant(var_idx, ordered_models)
            if stack_w is None:
                error(
                    f"Could not load or fit stacking meta-learner for variant {var_idx}."
                )
                return {}
            weights_by_mode["stacking"][var_idx] = stack_w

    # 2. Check whether Experiment 2 MPM maps are available for reuse
    can_reuse_exp2 = bool(reuse_exp2_maps and not live_mpm)
    if can_reuse_exp2:
        missing_exp2 = [
            _exp2_mpm_path(m_name, var_idx)
            for var_idx in variant_indices
            for m_name in ordered_models
            if not _exp2_mpm_path(m_name, var_idx).exists()
        ]
        if missing_exp2:
            warn(
                f"{len(missing_exp2)} Experiment 2 MPM map files missing "
                f"(e.g. {missing_exp2[0]}); falling back to live MPM mode."
            )
            can_reuse_exp2 = False

    height, width = target_resolution
    variant_maps_by_mode: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    saved_variant_paths: dict[str, dict[int, str]] = defaultdict(dict)

    # 3A. Fast Map Reuse Execution
    if can_reuse_exp2:
        info(
            f"Executing Stage 3D in Map Reuse Mode across {len(variant_indices)} "
            f"variants and {num_models} architectures..."
        )
        for var_idx in variant_indices:
            accumulators: dict[str, np.ndarray | None] = {m: None for m in active_modes}
            for m_idx, model_name in enumerate(ordered_models):
                map_path = _exp2_mpm_path(model_name, var_idx)
                raw_arr = np.load(map_path)
                resized_arr = uniform_heatmaps(
                    raw_arr, height=height, width=width
                ).astype(np.float64)

                for mode in active_modes:
                    w_val = float(weights_by_mode[mode][var_idx][m_idx])
                    if accumulators[mode] is None:
                        accumulators[mode] = w_val * resized_arr
                    else:
                        accumulators[mode] += w_val * resized_arr

            for mode in active_modes:
                combined_raw = accumulators[mode]
                if combined_raw is None:
                    continue
                norm_maps = _normalize_3d_maps(combined_raw)
                out_path = mpm_out_dir / f"{mode}_exp3_variant{var_idx}_maps.npy"
                np.save(out_path, norm_maps)
                variant_maps_by_mode[mode][var_idx] = norm_maps
                saved_variant_paths[mode][var_idx] = str(out_path)
                info(
                    f"  Saved {out_path.name} with shape {norm_maps.shape} "
                    f"(dtype={norm_maps.dtype})"
                )

    # 3B. Live MPM Execution
    else:
        info(
            f"Executing Stage 3D in Live MPM Mode on device={device} "
            f"(sigma={sigma}, px_batch={px_batch})..."
        )
        if global_test_indices is None:
            global_test_indices = _build_default_global_test_indices()
        else:
            default_idx = next(iter(global_test_indices.values()))
            global_test_indices.setdefault("imagenet", default_idx)
            global_test_indices.setdefault("densenet", default_idx)
            global_test_indices.setdefault("clip", default_idx)

        checkpoints_by_model = discover_base_checkpoints(
            models_filter=ordered_models,
            variant_indices=variant_indices,
        )
        test_indices_inet = global_test_indices["imagenet"]
        n_test = len(test_indices_inet)

        datasets_by_key = {
            "imagenet": imageNet_dataset,
            "densenet": denseNet_dataset,
            "clip": clip_dataset,
        }

        for var_idx in variant_indices:
            accumulators_live: dict[str, np.ndarray] = {
                mode: np.zeros((n_test, height, width), dtype=np.float64)
                for mode in active_modes
            }
            for m_idx, model_name in enumerate(ordered_models):
                if (
                    model_name not in checkpoints_by_model
                    or var_idx not in checkpoints_by_model[model_name]
                ):
                    error(f"Missing checkpoint for {model_name} variant {var_idx}")
                    return {}
                w_path = checkpoints_by_model[model_name][var_idx]
                ds_key = _dataset_key_for_model(model_name)
                ds_obj = datasets_by_key[ds_key]
                test_indices_model = global_test_indices[ds_key]

                info(
                    f"  [Variant {var_idx}] Running live MPM on {w_path.name} "
                    f"({n_test} images)..."
                )
                model = load_model_with_weights(
                    model_name, w_path, device=str(device), freeze_backbone=False
                )
                mpm = MultiscalePixelMasking(
                    model,
                    sigma_list=sigma,
                    pixel_batch_size=px_batch,
                    mask_value=mask_val,
                    use_tqdm=False,
                    stride=stride,
                )

                model_maps = np.zeros((n_test, height, width), dtype=np.float32)
                try:
                    for img_i, dataset_idx in enumerate(test_indices_model):
                        img_tensor, _ = ds_obj[dataset_idx]
                        saliency = mpm.generate_map(
                            img_tensor.unsqueeze(0),
                            target_index=0,
                            normalize=False,
                        )
                        if saliency.shape != (height, width):
                            saliency = uniform_heatmaps(
                                saliency[np.newaxis, ...].astype(np.float32),
                                height=height,
                                width=width,
                            )[0]
                        model_maps[img_i] = saliency.astype(np.float32)
                finally:
                    mpm.cleanup()
                    del model, mpm
                    clear_gpu_memory()
                    gc.collect()

                for mode in active_modes:
                    w_val = float(weights_by_mode[mode][var_idx][m_idx])
                    accumulators_live[mode] += w_val * model_maps.astype(np.float64)

            for mode in active_modes:
                norm_maps = _normalize_3d_maps(accumulators_live[mode])
                out_path = mpm_out_dir / f"{mode}_exp3_variant{var_idx}_maps.npy"
                np.save(out_path, norm_maps)
                variant_maps_by_mode[mode][var_idx] = norm_maps
                saved_variant_paths[mode][var_idx] = str(out_path)
                info(f"  Saved {out_path.name} with shape {norm_maps.shape}")

    # 4. Compute Architectural Prototype Heatmaps across variants
    prototype_paths: dict[str, str] = {}
    prototype_arrays: dict[str, np.ndarray] = {}

    for mode in active_modes:
        ordered_var_arrays = [
            variant_maps_by_mode[mode][v]
            for v in variant_indices
            if v in variant_maps_by_mode[mode]
        ]
        if not ordered_var_arrays:
            continue
        proto_arr = np.mean(np.stack(ordered_var_arrays, axis=0), axis=0).astype(
            np.float32
        )
        proto_path = mpm_out_dir / f"{mode}_exp3_prototype_maps.npy"
        np.save(proto_path, proto_arr)
        prototype_paths[mode] = str(proto_path)
        prototype_arrays[mode] = proto_arr
        info(
            f"Saved {mode} prototype heatmap to {proto_path} (shape={proto_arr.shape})"
        )

    # 5. Compute Within-Ensemble Attribution Consistency & Prototype Agreement
    consistency_metrics = (
        "correlation",
        "ssim",
        "top_percent_iou_5",
        "top_percent_iou_15",
        "top_percent_iou_25",
    )
    within_ensemble_consistency: dict[str, Any] = {}

    for mode in active_modes:
        ordered_var_arrays = [
            variant_maps_by_mode[mode][v]
            for v in variant_indices
            if v in variant_maps_by_mode[mode]
        ]
        var_labels = [
            f"variant{v}" for v in variant_indices if v in variant_maps_by_mode[mode]
        ]
        if len(ordered_var_arrays) >= 2:
            info(
                f"Computing within-ensemble MPM consistency for {mode} "
                f"across {len(ordered_var_arrays)} variants..."
            )
            comp_res = compare_heatmaps(ordered_var_arrays, metrics=consistency_metrics)
            comp_res["variants"] = var_labels  # type: ignore[typeddict-unknown-key]

            agg_summary: dict[str, dict[str, float]] = {}
            for m_key in consistency_metrics:
                pair_means = [
                    d["mean"] for d in comp_res["summary"].get(m_key, {}).values()
                ]
                per_img_vals = list(comp_res["per_image"].get(m_key, {}).values())
                if pair_means and per_img_vals:
                    per_image_mean = np.mean(np.stack(per_img_vals, axis=0), axis=0)
                    agg_summary[m_key] = {
                        "mean": float(np.mean(per_image_mean)),
                        "std": float(np.std(per_image_mean)),
                        "min": float(np.min(per_image_mean)),
                        "max": float(np.max(per_image_mean)),
                        "median": float(np.median(per_image_mean)),
                    }
            within_ensemble_consistency[mode] = {
                "aggregated": agg_summary,
                "variants": var_labels,
                "summary": comp_res["summary"],
            }

            if save_plots:
                for m_key in ("correlation", "top_percent_iou_15"):
                    if m_key not in comp_res["summary"]:
                        continue
                    try:
                        fig_mat = visualize_similarity_matrix(
                            comp_res, var_labels, metric=m_key, add_title=False
                        )
                        if fig_mat:
                            fig_mat.savefig(
                                plots_dir / f"mpm_within_{mode}_{m_key}_matrix.png",
                                bbox_inches="tight",
                                dpi=300,
                            )
                            plt.close(fig_mat)
                        fig_dist = visualize_similarity_distribution(
                            comp_res, metric=m_key, add_title=False
                        )
                        if fig_dist:
                            fig_dist.savefig(
                                plots_dir
                                / f"mpm_within_{mode}_{m_key}_distribution.png",
                                bbox_inches="tight",
                                dpi=300,
                            )
                            plt.close(fig_dist)
                    except (OSError, ValueError, RuntimeError, KeyError) as exc:
                        warn(f"Could not save within-ensemble plot: {exc}")

    # Compare ensemble prototypes against single-architecture prototypes if available
    between_prototypes_agreement: dict[str, Any] = {}
    if can_reuse_exp2 and prototype_arrays:
        single_arch_protos: dict[str, np.ndarray] = {}
        for model_name in ordered_models:
            arch_vars = [
                uniform_heatmaps(
                    np.load(_exp2_mpm_path(model_name, v)),
                    height=height,
                    width=width,
                ).astype(np.float32)
                for v in variant_indices
                if _exp2_mpm_path(model_name, v).exists()
            ]
            if arch_vars:
                single_arch_protos[model_name] = np.mean(
                    np.stack(arch_vars, axis=0), axis=0
                ).astype(np.float32)

        if single_arch_protos:
            vis_order = [m for m in get_model_order() if m in single_arch_protos]
            for m in single_arch_protos:
                if m not in vis_order:
                    vis_order.append(m)

            all_proto_labels = vis_order + [
                f"{m}_ensemble" for m in active_modes if m in prototype_arrays
            ]
            all_proto_list = [single_arch_protos[m] for m in vis_order] + [
                prototype_arrays[m] for m in active_modes if m in prototype_arrays
            ]

            if len(all_proto_list) >= 2:
                info(
                    "Comparing Ensemble Prototype Heatmaps against "
                    "Single-Architecture Prototypes..."
                )
                proto_comp = compare_heatmaps(
                    all_proto_list, metrics=consistency_metrics
                )
                between_prototypes_agreement = {
                    "models": all_proto_labels,
                    "summary": proto_comp["summary"],
                }
                if save_plots:
                    for m_key in ("correlation", "top_percent_iou_15"):
                        if m_key not in proto_comp["summary"]:
                            continue
                        try:
                            fig_mat = visualize_similarity_matrix(
                                proto_comp,
                                all_proto_labels,
                                metric=m_key,
                                add_title=False,
                            )
                            if fig_mat:
                                fig_mat.savefig(
                                    plots_dir
                                    / f"mpm_prototypes_ensemble_vs_single_{m_key}_matrix.png",
                                    bbox_inches="tight",
                                    dpi=300,
                                )
                                plt.close(fig_mat)
                        except (OSError, ValueError, RuntimeError, KeyError) as exc:
                            warn(f"Could not save prototype matrix plot: {exc}")

    # Save sample quantile overlay visualizations if requested
    if save_plots and num_sample_images > 0 and prototype_arrays:
        try:
            if global_test_indices is None:
                global_test_indices = _build_default_global_test_indices()
            inet_indices = global_test_indices["imagenet"]
            n_avail = min(
                len(inet_indices),
                next(iter(prototype_arrays.values())).shape[0],
            )
            if n_avail > 0:
                scores = np.array(
                    [
                        float(imageNet_dataset[inet_indices[i]][1].item())
                        for i in range(n_avail)
                    ]
                )
                quantiles = (
                    [0.25, 0.50, 0.75]
                    if num_sample_images == 3
                    else np.linspace(0.1, 0.9, num_sample_images).tolist()
                )
                q_vals = np.quantile(scores, quantiles)
                used_pos: set[int] = set()
                for q, qval in zip(quantiles, q_vals, strict=False):
                    order = np.argsort(np.abs(scores - qval))
                    chosen = next(
                        (int(c) for c in order if int(c) not in used_pos),
                        int(order[0]),
                    )
                    used_pos.add(chosen)
                    dataset_idx = inet_indices[chosen]
                    img_inet, label = imageNet_dataset[dataset_idx]
                    q_tag = f"q{q:.2f}_idx{dataset_idx}_auth{float(label.item()):.2f}"
                    for mode, proto_map in prototype_arrays.items():
                        map_01 = np.clip((proto_map[chosen] + 1.0) / 2.0, 0.0, 1.0)
                        mode_plot_dir = plots_dir / "Sample_Overlays" / mode
                        mode_plot_dir.mkdir(parents=True, exist_ok=True)
                        visualize_and_save_saliency(
                            image_tensor=img_inet,
                            saliency_map=map_01.astype(np.float32),
                            output_dir=str(mode_plot_dir),
                            filename_prefix=f"{mode}_prototype_{q_tag}",
                            cmap_name="bwr",
                        )
        except (OSError, RuntimeError, ValueError, IndexError) as exc:
            warn(f"Could not render sample overlay plots: {exc}")

    xai_summary = {
        "status": "success",
        "execution_mode": "reuse_exp2_maps" if can_reuse_exp2 else "live_mpm",
        "modes_processed": active_modes,
        "variants_processed": variant_indices,
        "models": ordered_models,
        "target_resolution": list(target_resolution),
        "variant_map_paths": {
            mode: {str(k): v for k, v in paths.items()}
            for mode, paths in saved_variant_paths.items()
        },
        "prototype_map_paths": prototype_paths,
        "within_ensemble_consistency": within_ensemble_consistency,
        "between_prototypes_agreement": between_prototypes_agreement,
    }

    xai_results_path = results_dir / "experiment_3d_xai_consistency.json"
    with open(xai_results_path, "w") as f:
        json.dump(xai_summary, f, indent=2, cls=NpEncoder)
    info(f"Saved Stage 3D XAI consistency results to: {xai_results_path}")

    info("=" * 80)
    info("EXPERIMENT 3D: HEATMAP GENERATION & CONSISTENCY COMPLETE")
    info("=" * 80)

    return xai_summary


# ============================================================================
# 8. COMPLETE EXPERIMENT 3 PIPELINE
# ============================================================================


def run_experiment_3(
    models: list[str] | None = None,
    run_training: bool = False,
    run_pruning: bool = False,
    run_evaluation: bool = True,
    run_heatmaps: bool = True,
    save_results: bool = True,
    ensemble_mode: Sequence[str] = ("bagging", "stacking"),
    variants: str | Sequence[int] | int | None = "all",
    reuse_exp2_maps: bool = True,
    live_mpm: bool = False,
    device: str | None = None,
    stacking_cv_folds: int = 5,
    stacking_cv_repeats: int = 1,
    heatmap_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the complete Experiment 3 pipeline (ADR-0005).

    By default, skips legacy base training (3A) and channel pruning (3B), directly
    reusing the 70 unpruned Experiment 1A checkpoints to evaluate the 10
    variant-aligned 7-architecture Bagging and Stacking ensembles (Stage 3C) and
    generate full-dataset 273-image MPM heatmaps and prototypes (Stage 3D).

    Args:
        models: Optional list of model names to process (None = all 7 models).
        run_training: Whether to run legacy base training (Stage 3A, default False).
        run_pruning: Whether to run legacy validation pruning (Stage 3B, default False).
        run_evaluation: Whether to run ensemble evaluation (Stage 3C, default True).
        run_heatmaps: Whether to generate ensemble MPM heatmaps (Stage 3D, default True).
        save_results: Whether to save combined results JSON.
        ensemble_mode: Sequence of ensemble strategies ('bagging', 'stacking').
        variants: Variant selector ('all', '1,2,3', or sequence of ints).
        reuse_exp2_maps: Whether to reuse precomputed Experiment 2 MPM maps in Stage 3D.
        live_mpm: Force live end-to-end MPM forward passes in Stage 3D.
        device: Optional device override ('cuda' or 'cpu').
        stacking_cv_folds: Kept for backward signature compatibility.
        stacking_cv_repeats: Kept for backward signature compatibility.
        heatmap_kwargs: Optional extra keyword arguments forwarded to Stage 3D.

    Returns:
        Dictionary aggregating outputs from all executed stages.
    """
    start = time.time()
    setup_directories()

    resolved_device = (
        device
        if device is not None and device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if isinstance(models, str):
        models = [models]

    ordered_models = get_ordered_models(models)
    variant_indices = parse_variant_indices(variants)

    info(f"Device: {resolved_device}")
    info(f"Models ({len(ordered_models)}): {ordered_models}")
    info(f"Variants ({len(variant_indices)}): {variant_indices}")
    info(f"Strategies: {list(ensemble_mode)}")

    global_test_indices = _build_default_global_test_indices()
    info(f"Global test set size: {len(global_test_indices['imagenet'])}")

    results: dict[str, Any] = {}
    variant_val_indices: dict[str, dict[int, list[int]]] | None = None

    if run_training:
        training_results, variant_val_indices = experiment_3a_train_all_variants(
            models_to_train=ordered_models,
            global_test_indices=global_test_indices,
            save_plots=True,
            verbose=True,
        )
        results["training"] = training_results

    if run_pruning:
        pruning_results = experiment_3b_prune_all_variants(
            models_to_prune=ordered_models,
            variant_val_indices=variant_val_indices,
            global_test_indices=global_test_indices,
            verbose=True,
        )
        results["pruning"] = pruning_results

    if run_evaluation:
        eval_results = experiment_3c_evaluate_ensemble(
            models_filter=ordered_models,
            global_test_indices=global_test_indices,
            device=str(resolved_device),
            ensemble_mode=ensemble_mode,
            variants=variant_indices,
            stacking_cv_folds=stacking_cv_folds,
            stacking_cv_repeats=stacking_cv_repeats,
        )
        results["evaluation"] = eval_results

    if run_heatmaps:
        extra_kwargs = dict(heatmap_kwargs) if heatmap_kwargs else {}
        heatmap_results = experiment_3d_plot_ensemble_heatmaps(
            global_test_indices=global_test_indices,
            models_filter=ordered_models,
            device=str(resolved_device),
            modes=ensemble_mode,
            variants=variant_indices,
            reuse_exp2_maps=reuse_exp2_maps,
            live_mpm=live_mpm,
            **extra_kwargs,
        )
        results["heatmaps"] = heatmap_results

    if save_results and results:
        out_path = DIRS["results"] / "experiment_3_complete_results.json"
        DIRS["results"].mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        info(f"Complete results saved to {out_path}")

    elapsed = time.time() - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    info(f"Experiment 3 completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")

    return results


if __name__ == "__main__":
    set_level("INFO")
    run_experiment_3()
