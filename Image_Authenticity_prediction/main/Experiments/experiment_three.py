# Experiment 3: Bagging Ensemble with Independent Training and Pruning
#
# This experiment trains 10 variants per architecture (like Experiment 1),
# but with a key difference in the data split:
# - Train on training set
# - Validate and PRUNE on validation set (same validation set used for training)
# - Test set is reserved ONLY for final ensemble evaluation
#
# This ensures the ensemble is evaluated on truly unseen data.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import sys
from pathlib import Path
import numpy as np
import time
import re
import json
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

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
from main.Utils import FeatureMapsPruner
from main.Utils.cleanup import clear_gpu_memory, cleanup_model_and_data
from main.Utils.logger import info, warn, error, debug, set_level
from main.Utils.config import (
    get_ensemble_config,
    get_data_config,
    get_training_config,
    get_pruning_config,
)

from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    imageNet_dataset,
    denseNet_dataset,
    SEED,
)
from main.train import train_model, test_model

# Load configs
_data_cfg = get_data_config()
NUM_WORKERS = _data_cfg["num_workers"]
BATCH_SIZE = _data_cfg["batch_size"]

TRAINING_CONFIG = get_training_config()
PRUNING_CONFIG = get_pruning_config()
ENSEMBLE_CONFIG = get_ensemble_config()

# Number of variants per model
NUM_VARIANTS = 10

# ============================================================================
# 1.1 DIRECTORIES
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parent / "Outputs" / "Experiment_3_ensemble"
DIRS = {
    "weights": OUTPUT_DIR / "Weights",
    "rankings": OUTPUT_DIR / "Ranking_arrays",
    "ranking_plots": OUTPUT_DIR / "Ranking_Plots",
    "training_plots": OUTPUT_DIR / "Training_Plots",
    "training_history": OUTPUT_DIR / "Training_History",
    "results": OUTPUT_DIR / "Results",
}

# ============================================================================
# 1.2 MODEL REGISTRY
# ============================================================================
MODEL_REGISTRY = {
    "vgg16": {
        "class": VGG16AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.28",
    },
    "vgg19": {
        "class": VGG19AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.34",
    },
    "resnet152": {
        "class": ResNet152AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.7.2.conv3",
    },
    "densenet161": {
        "class": DenseNet161AuthenticityPredictor,
        "dataset": DENSENET_DATASET,
        "backbone_dataset": denseNet_dataset,
        "target_layer": "features.denseblock4.denselayer24.conv2",
    },
    "efficientnetb3": {
        "class": EfficientNetB3AuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.8.0",
    },
    "barlowtwins": {
        "class": BarlowTwinsAuthenticityPredictor,
        "dataset": IMAGENET_DATASET,
        "backbone_dataset": imageNet_dataset,
        "target_layer": "features.7.2.conv3",
    },
}


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
# 2. DATA SPLITTING UTILITIES
# ============================================================================


def create_global_test_indices(
    dataset_size: int, test_fraction: float = 0.2, seed: int = 42
) -> List[int]:
    """
    Create global test indices that remain constant across all variants.

    Args:
        dataset_size: Total size of the dataset
        test_fraction: Fraction of data to use for test set
        seed: Random seed for reproducibility

    Returns:
        List of test indices
    """
    test_size = int(test_fraction * dataset_size)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=gen).tolist()
    return perm[:test_size]


def create_variant_split(
    backbone_dataset,
    global_test_indices: List[int],
    variant_idx: int,
    val_fraction: float = 0.125,  # 0.125 of remaining = 10% of total
) -> Tuple[Subset, Subset, Subset, List[int]]:
    """
    Create train/val/test split for a specific variant.

    The key difference from Experiment 1: we return val_indices so we can
    use the SAME validation set for pruning later.

    Args:
        backbone_dataset: The full dataset
        global_test_indices: Pre-computed test indices (constant across variants)
        variant_idx: Variant index (1-10) used to seed the train/val shuffle
        val_fraction: Fraction of remaining data to use for validation

    Returns:
        Tuple of (train_ds, val_ds, test_ds, val_indices)
    """
    total_size = len(backbone_dataset)
    test_indices = set(global_test_indices)

    # Remaining indices for training and validation
    remaining_indices = [i for i in range(total_size) if i not in test_indices]

    # Shuffle remaining indices per-variant to create different train/val splits
    gen = torch.Generator().manual_seed(42 + variant_idx)
    perm = torch.randperm(len(remaining_indices), generator=gen).tolist()
    shuffled_remaining = [remaining_indices[i] for i in perm]

    # Split remaining into train and val
    val_size = int(val_fraction * len(shuffled_remaining))
    train_size = len(shuffled_remaining) - val_size

    train_indices = shuffled_remaining[:train_size]
    val_indices = shuffled_remaining[train_size:]

    train_ds = Subset(backbone_dataset, train_indices)
    val_ds = Subset(backbone_dataset, val_indices)
    test_ds = Subset(backbone_dataset, list(global_test_indices))

    return train_ds, val_ds, test_ds, val_indices


# ============================================================================
# 3. MODEL LOADING & UTILITIES
# ============================================================================


def reset_regression_head(model: nn.Module):
    """Reinitialize regression head weights for a distinct starting state."""
    try:
        for layer in model.regression_head.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    except AttributeError:
        # If model does not have regression_head attribute, ignore
        pass


def load_model_with_weights(
    model_name: str,
    weights_path: Path,
    device: str = "cuda",
    freeze_backbone: bool = True,
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
# 4. EXPERIMENT 3A: TRAIN ALL VARIANTS
# ============================================================================


def experiment_3a_train_all_variants(
    models_to_train: List[str] = None,
    global_test_indices: Dict[str, List[int]] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict[int, List[int]]]]:
    """
    Train 10 variants per model architecture.

    Args:
        models_to_train: List of model names to train (None = all)
        global_test_indices: Pre-computed test indices per dataset type
        save_plots: Whether to save training plots
        verbose: Whether to print detailed progress

    Returns:
        Tuple of:
        - results: Training results per model
        - variant_val_indices: Dict mapping model_name -> {variant_idx -> val_indices}
          This is critical for using the same validation set during pruning.
    """
    info("=" * 80)
    info("EXPERIMENT 3A: TRAINING ALL VARIANTS")
    info("=" * 80)

    # Create output directories
    DIRS["weights"].mkdir(parents=True, exist_ok=True)
    DIRS["training_plots"].mkdir(parents=True, exist_ok=True)
    DIRS["training_history"].mkdir(parents=True, exist_ok=True)

    # Select models to train
    if models_to_train is None:
        models_to_train = list(MODEL_REGISTRY.keys())

    # Create global test indices if not provided
    if global_test_indices is None:
        global_test_indices = {}
        # For ImageNet-based models
        total_imagenet = len(imageNet_dataset)
        global_test_indices["imagenet"] = create_global_test_indices(total_imagenet)
        # For DenseNet-based models
        total_densenet = len(denseNet_dataset)
        global_test_indices["densenet"] = create_global_test_indices(total_densenet)

    results = {}
    variant_val_indices = {}  # Store val indices per model/variant for pruning

    device = torch.device(TRAINING_CONFIG["device"])

    for idx, model_name in enumerate(models_to_train, 1):
        info(f"[{idx}/{len(models_to_train)}] Training {model_name.upper()}")
        info("-" * 80)

        try:
            config = MODEL_REGISTRY[model_name]
            backbone_dataset = config["backbone_dataset"]

            # Determine which global test indices to use
            if config["dataset"] is IMAGENET_DATASET:
                test_indices = global_test_indices["imagenet"]
            else:
                test_indices = global_test_indices["densenet"]

            results[model_name] = {}
            variant_val_indices[model_name] = {}

            for variant_idx in range(1, NUM_VARIANTS + 1):
                # Check if this variant is already trained
                weights_path = (
                    DIRS["weights"]
                    / f"{model_name}_exp3a_variant{variant_idx}_best.pth"
                )
                if weights_path.exists():
                    info(
                        f"✓ Variant {variant_idx}/{NUM_VARIANTS} for {model_name} already exists, skipping..."
                    )
                    # Still need to store val_indices for pruning
                    _, _, _, val_indices = create_variant_split(
                        backbone_dataset, test_indices, variant_idx
                    )
                    variant_val_indices[model_name][variant_idx] = val_indices
                    continue

                info(f"Variant {variant_idx}/{NUM_VARIANTS} for {model_name}")

                # Initialize model
                model = config["class"](
                    freeze_backbone=TRAINING_CONFIG["freeze_backbone"]
                )
                reset_regression_head(model)

                # Create data split - CRITICAL: store val_indices for pruning
                train_ds, val_ds, test_ds, val_indices = create_variant_split(
                    backbone_dataset, test_indices, variant_idx
                )
                variant_val_indices[model_name][variant_idx] = val_indices

                # Create dataloaders
                train_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )

                dataloaders = {"train": train_loader, "val": val_loader}

                # Setup training
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=TRAINING_CONFIG["learning_rate"]
                )

                if verbose:
                    info(f"  Training on device: {device}")
                    info(f"  Train size: {len(train_ds)}, Val size: {len(val_ds)}")

                # Train the model
                best_model, history = train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=TRAINING_CONFIG["max_epochs"],
                    device=TRAINING_CONFIG["device"],
                    patience=TRAINING_CONFIG["patience"],
                )

                # Save model weights
                weights_path = (
                    DIRS["weights"]
                    / f"{model_name}_exp3a_variant{variant_idx}_best.pth"
                )
                torch.save(best_model.state_dict(), weights_path)
                info(f"✓ Variant weights saved to: {weights_path}")

                # Save training history
                history_path = (
                    DIRS["training_history"]
                    / f"{model_name}_exp3a_variant{variant_idx}_history.npy"
                )
                np.save(history_path, history)

                # Save training plots
                if save_plots:
                    try:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

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
                    except Exception as e:
                        warn(f"Could not save plot: {e}")

                # Store results
                results[model_name][f"variant{variant_idx}"] = {
                    "final_val_loss": history["val_loss"][-1],
                    "best_val_loss": min(history["val_loss"]),
                    "epochs_trained": len(history["train_loss"]),
                    "weights_path": str(weights_path),
                    "val_indices_count": len(val_indices),
                }

                info(f"✓ {model_name} variant {variant_idx} complete!")
                info(f"  Best Val Loss: {min(history['val_loss']):.4f}")

                # Cleanup after each variant
                del model, best_model, train_loader, val_loader
                clear_gpu_memory()

        except Exception as e:
            error(f"Error training {model_name}: {e}")
            error(traceback.format_exc())
            results[model_name] = {"error": str(e)}

        finally:
            cleanup_model_and_data(
                model=locals().get("model"),
                dataloaders=locals().get("dataloaders"),
                optimizer=locals().get("optimizer"),
            )
            clear_gpu_memory()

    # Save variant validation indices for later use
    val_indices_path = DIRS["results"] / "variant_val_indices.json"
    DIRS["results"].mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_indices = {
        model: {str(k): v for k, v in variants.items()}
        for model, variants in variant_val_indices.items()
    }
    with open(val_indices_path, "w") as f:
        json.dump(serializable_indices, f, cls=NpEncoder)
    info(f"✓ Validation indices saved to: {val_indices_path}")

    info("=" * 80)
    info("EXPERIMENT 3A: TRAINING COMPLETE")
    info("=" * 80)

    return results, variant_val_indices


# ============================================================================
# 5. EXPERIMENT 3B: PRUNE ALL VARIANTS (ON VALIDATION SET)
# ============================================================================


def experiment_3b_prune_all_variants(
    models_to_prune: List[str] = None,
    variant_val_indices: Dict[str, Dict[int, List[int]]] = None,
    global_test_indices: Dict[str, List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Prune trained variants using greedy pruning on the VALIDATION set.

    This is the key difference from Experiment 1: pruning is done on validation,
    not on test, so the test set remains completely unseen for ensemble evaluation.

    Args:
        models_to_prune: List of model names to prune (None = all trained)
        variant_val_indices: Dict mapping model_name -> {variant_idx -> val_indices}
        global_test_indices: Pre-computed test indices per dataset type
        verbose: Whether to print detailed progress

    Returns:
        Pruning results per model/variant
    """
    info("=" * 80)
    info("EXPERIMENT 3B: PRUNING ALL VARIANTS (ON VALIDATION SET)")
    info("=" * 80)

    # Create output directories
    DIRS["rankings"].mkdir(parents=True, exist_ok=True)
    DIRS["ranking_plots"].mkdir(parents=True, exist_ok=True)

    # Load variant validation indices if not provided
    if variant_val_indices is None:
        val_indices_path = DIRS["results"] / "variant_val_indices.json"
        if val_indices_path.exists():
            with open(val_indices_path, "r") as f:
                loaded_indices = json.load(f)
            # Convert string keys back to int
            variant_val_indices = {
                model: {int(k): v for k, v in variants.items()}
                for model, variants in loaded_indices.items()
            }
            info(f"Loaded validation indices from: {val_indices_path}")
        else:
            error(f"Validation indices not found: {val_indices_path}")
            error("Please run experiment_3a_train_all_variants first.")
            return {}

    # Create global test indices if not provided (for reference)
    if global_test_indices is None:
        global_test_indices = {}
        global_test_indices["imagenet"] = create_global_test_indices(
            len(imageNet_dataset)
        )
        global_test_indices["densenet"] = create_global_test_indices(
            len(denseNet_dataset)
        )

    # Find all trained weight files
    all_pth_files = sorted(DIRS["weights"].glob("*_exp3a_*.pth"))
    if not all_pth_files:
        error(f"No trained weights found in {DIRS['weights']}")
        error("Please run experiment_3a_train_all_variants first.")
        return {}

    # Group by model name
    weights_by_model = defaultdict(list)
    for p in all_pth_files:
        match = re.match(r"^([a-z0-9]+)_exp3a_variant(\d+)_best\.pth$", p.name)
        if match:
            model_name = match.group(1)
            if models_to_prune is None or model_name in models_to_prune:
                if model_name in MODEL_REGISTRY:
                    weights_by_model[model_name].append(p)

    if not weights_by_model:
        error("No valid model weights found to prune.")
        return {}

    info(f"Found weights for {len(weights_by_model)} models:")
    for mn, files in weights_by_model.items():
        info(f"  - {mn}: {len(files)} variants")

    results = {}
    device = torch.device(TRAINING_CONFIG["device"])
    criterion = nn.MSELoss()

    for idx, (model_name, weight_files) in enumerate(weights_by_model.items(), 1):
        info(f"[{idx}/{len(weights_by_model)}] Pruning {model_name.upper()}")
        info("-" * 80)

        try:
            config = MODEL_REGISTRY[model_name]
            backbone_dataset = config["backbone_dataset"]
            target_layer = config["target_layer"]

            results[model_name] = {}

            for weights_path in weight_files:
                # Extract variant index from filename
                match = re.search(r"variant(\d+)", weights_path.name)
                if not match:
                    continue
                variant_idx = int(match.group(1))
                variant_tag = f"variant{variant_idx}"

                # Check if this variant is already pruned
                pruned_weights_path = (
                    DIRS["weights"]
                    / f"{model_name}_exp3b_{variant_tag}_greedy_pruned.pth"
                )
                if pruned_weights_path.exists():
                    info(f"✓ {model_name} {variant_tag} already pruned, skipping...")
                    continue

                info(f"Processing {variant_tag}...")

                # Get validation indices for this variant
                if (
                    model_name not in variant_val_indices
                    or variant_idx not in variant_val_indices[model_name]
                ):
                    warn(
                        f"No validation indices for {model_name} {variant_tag}, skipping"
                    )
                    continue

                val_indices = variant_val_indices[model_name][variant_idx]

                # Create validation dataloader (SAME as used in training)
                val_ds = Subset(backbone_dataset, val_indices)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )

                if verbose:
                    info(f"  Validation set size: {len(val_ds)}")

                # Load the trained model
                model = config["class"](freeze_backbone=False)
                model.load_state_dict(torch.load(weights_path, weights_only=True))

                # Create pruner using VALIDATION loader (not test!)
                pruner = FeatureMapsPruner(
                    model=model,
                    dataloader=val_loader,  # KEY: prune on validation set
                    layer_name=target_layer,
                    criterion=criterion,
                    eval_function=test_model,
                    device=device,
                )

                # Compute importance scores
                importance_path = (
                    DIRS["rankings"]
                    / f"{model_name}_exp3b_{variant_tag}_importance.npy"
                )
                importance_scores = pruner.compute_importance_scores(
                    save_path=str(importance_path),
                    force_recompute=PRUNING_CONFIG["force_recompute"],
                )

                # Ensure baseline is computed (may be None if importance was loaded from file)
                if pruner.baseline_mse is None:
                    pruner.baseline_mse, pruner.baseline_rmse = pruner._evaluate_model()

                # Plot importance scores
                try:
                    plot_path = (
                        DIRS["ranking_plots"]
                        / f"{model_name}_exp3b_{variant_tag}_importance.png"
                    )
                    pruner.plot_importance_scores(save_path=str(plot_path))
                except Exception as e:
                    warn(f"Could not save importance plot: {e}")

                info(
                    f"  Baseline MSE: {pruner.baseline_mse:.4f}, RMSE: {pruner.baseline_rmse:.4f}"
                )

                # Perform greedy pruning
                pruned_weights_path = (
                    DIRS["weights"]
                    / f"{model_name}_exp3b_{variant_tag}_greedy_pruned.pth"
                )
                pruning_results = pruner.greedy_pruning(
                    model_save_path=str(pruned_weights_path)
                )

                # Store results
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

                info(f"✓ {model_name} {variant_tag} pruning complete!")
                info(f"  Final MSE: {pruning_results['final_mse']:.4f}")
                info(f"  Features removed: {len(pruning_results['removed_features'])}")
                info(f"  Reduction: {pruning_results['reduction_percentage']:.1f}%")

                # Cleanup
                del model, pruner, val_loader
                clear_gpu_memory()

        except Exception as e:
            error(f"Error pruning {model_name}: {e}")
            error(traceback.format_exc())
            results[model_name] = {"error": str(e)}

        finally:
            cleanup_model_and_data(
                model=locals().get("model"),
                dataloaders=locals().get("val_loader"),
                optimizer=None,
            )
            clear_gpu_memory()

    # Save pruning results
    pruning_results_path = DIRS["results"] / "experiment_3b_pruning_results.json"
    with open(pruning_results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    info(f"✓ Pruning results saved to: {pruning_results_path}")

    info("=" * 80)
    info("EXPERIMENT 3B: PRUNING COMPLETE")
    info("=" * 80)

    return results


# ============================================================================
# 6. EXPERIMENT 3C: EVALUATE ENSEMBLE (ON TEST SET ONLY)
# ============================================================================


def experiment_3c_evaluate_ensemble(
    models_filter: List[str] = None,
    global_test_indices: Dict[str, List[int]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate the bagging ensemble on the TEST set.

    This is the ONLY evaluation on test data - both training and pruning
    were done on train/val splits, so test is truly unseen.

    Args:
        models_filter: List of model names to include (None = all)
        global_test_indices: Pre-computed test indices per dataset type
        device: Device to use

    Returns:
        Ensemble evaluation metrics
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau

    info("=" * 80)
    info("EXPERIMENT 3C: EVALUATING ENSEMBLE ON TEST SET")
    info("=" * 80)

    # Create global test indices if not provided
    if global_test_indices is None:
        global_test_indices = {}
        global_test_indices["imagenet"] = create_global_test_indices(
            len(imageNet_dataset)
        )
        global_test_indices["densenet"] = create_global_test_indices(
            len(denseNet_dataset)
        )

    # Find all pruned weight files
    all_pruned_files = sorted(DIRS["weights"].glob("*_exp3b_*_greedy_pruned.pth"))
    if not all_pruned_files:
        error(f"No pruned weights found in {DIRS['weights']}")
        error("Please run experiment_3b_prune_all_variants first.")
        return {}

    # Group by model name
    weights_by_model = defaultdict(list)
    for p in all_pruned_files:
        match = re.match(r"^([a-z0-9]+)_exp3b_variant(\d+)_greedy_pruned\.pth$", p.name)
        if match:
            model_name = match.group(1)
            if models_filter is None or model_name in models_filter:
                if model_name in MODEL_REGISTRY:
                    weights_by_model[model_name].append(p)

    if not weights_by_model:
        error("No valid pruned model weights found.")
        return {}

    total_variants = sum(len(v) for v in weights_by_model.values())
    info(
        f"Found {total_variants} pruned variants across {len(weights_by_model)} models"
    )

    # Prepare test loaders for each dataset type
    test_loaders = {}

    # ImageNet test loader
    imagenet_test_ds = Subset(imageNet_dataset, global_test_indices["imagenet"])
    test_loaders["imagenet"] = DataLoader(
        imagenet_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # DenseNet test loader
    densenet_test_ds = Subset(denseNet_dataset, global_test_indices["densenet"])
    test_loaders["densenet"] = DataLoader(
        densenet_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Get ground truth labels (same for both since same indices, different transforms)
    y_true = get_labels(test_loaders["imagenet"]).numpy()

    info(f"Test set size: {len(y_true)}")

    # Collect predictions from all models (one at a time for memory efficiency)
    all_preds = []
    model_results = {}

    for model_name, weight_files in weights_by_model.items():
        config = MODEL_REGISTRY[model_name]

        # Determine which test loader to use
        if config["dataset"] is IMAGENET_DATASET:
            test_loader = test_loaders["imagenet"]
        else:
            test_loader = test_loaders["densenet"]

        model_results[model_name] = {"variants": []}

        for weights_path in weight_files:
            info(f"  Processing {weights_path.name}...")

            # Load model
            model = load_model_with_weights(
                model_name, weights_path, device, freeze_backbone=False
            )

            # Get predictions
            preds = get_predictions(model, test_loader, device)
            all_preds.append(preds.squeeze().cpu())

            # Also compute individual model metrics
            preds_np = preds.squeeze().cpu().numpy()
            individual_mse = np.mean((preds_np - y_true) ** 2)
            model_results[model_name]["variants"].append(
                {
                    "weights_path": str(weights_path),
                    "mse": float(individual_mse),
                    "rmse": float(np.sqrt(individual_mse)),
                }
            )

            # Cleanup
            del model
            clear_gpu_memory()

    # Compute ensemble prediction (simple average)
    y_pred = torch.mean(torch.stack(all_preds), dim=0).numpy()

    info(f"Averaged predictions from {len(all_preds)} model variants")

    # Compute ensemble metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    plcc, plcc_p = pearsonr(y_pred, y_true)
    srcc, srcc_p = spearmanr(y_pred, y_true)
    krcc, krcc_p = kendalltau(y_pred, y_true)

    ensemble_results = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "plcc": float(plcc),
        "srcc": float(srcc),
        "krcc": float(krcc),
        "plcc_p_value": float(plcc_p),
        "srcc_p_value": float(srcc_p),
        "krcc_p_value": float(krcc_p),
        "num_models": len(all_preds),
        "test_size": len(y_true),
    }

    info("=" * 60)
    info("ENSEMBLE RESULTS (on unseen test set)")
    info("=" * 60)
    info(f"  MSE:  {mse:.4f}")
    info(f"  RMSE: {rmse:.4f}")
    info(f"  MAE:  {mae:.4f}")
    info(f"  PLCC: {plcc:.4f} (p={plcc_p:.2e})")
    info(f"  SRCC: {srcc:.4f} (p={srcc_p:.2e})")
    info(f"  KRCC: {krcc:.4f} (p={krcc_p:.2e})")

    # Compute and report individual model average performance
    all_individual_mses = []
    for model_name, mdata in model_results.items():
        for v in mdata["variants"]:
            all_individual_mses.append(v["mse"])

    if all_individual_mses:
        avg_individual_mse = np.mean(all_individual_mses)
        info(f"\nAverage individual model MSE: {avg_individual_mse:.4f}")
        info(f"Ensemble improvement: {avg_individual_mse - mse:.4f}")

    # Save results
    full_results = {
        "ensemble": ensemble_results,
        "individual_models": model_results,
    }

    results_path = DIRS["results"] / "experiment_3c_ensemble_results.json"
    DIRS["results"].mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, cls=NpEncoder)
    info(f"\n✓ Results saved to: {results_path}")

    info("=" * 80)
    info("EXPERIMENT 3C: EVALUATION COMPLETE")
    info("=" * 80)

    return full_results


# ============================================================================
# 7. COMPLETE PIPELINE
# ============================================================================


def run_experiment_3(
    models: List[str] = None,
    run_training: bool = True,
    run_pruning: bool = True,
    run_evaluation: bool = True,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete Experiment 3 pipeline.

    This experiment trains 10 variants per architecture and prunes them,
    but with a key difference from Experiment 1:
    - Training: train on train set, validate on val set
    - Pruning: prune on val set (same val set used during training)
    - Evaluation: ensemble evaluated on test set (completely unseen)

    Args:
        models: List of model names to process (None = all)
        run_training: Whether to run training (Experiment 3A)
        run_pruning: Whether to run pruning (Experiment 3B)
        run_evaluation: Whether to run ensemble evaluation (Experiment 3C)
        save_results: Whether to save results to JSON

    Returns:
        Combined results from all stages
    """
    start = time.time()
    setup_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device: {device}")
    info(f"Models: {models if models else 'all'}")

    if isinstance(models, str):
        models = [models]

    # Create global test indices (shared across all stages)
    global_test_indices = {
        "imagenet": create_global_test_indices(len(imageNet_dataset)),
        "densenet": create_global_test_indices(len(denseNet_dataset)),
    }
    info(f"Global test set size (ImageNet): {len(global_test_indices['imagenet'])}")
    info(f"Global test set size (DenseNet): {len(global_test_indices['densenet'])}")

    results = {}
    variant_val_indices = None

    # Stage 3A: Training
    if run_training:
        training_results, variant_val_indices = experiment_3a_train_all_variants(
            models_to_train=models,
            global_test_indices=global_test_indices,
            save_plots=True,
            verbose=True,
        )
        results["training"] = training_results

    # Stage 3B: Pruning
    if run_pruning:
        pruning_results = experiment_3b_prune_all_variants(
            models_to_prune=models,
            variant_val_indices=variant_val_indices,
            global_test_indices=global_test_indices,
            verbose=True,
        )
        results["pruning"] = pruning_results

    # Stage 3C: Ensemble Evaluation
    if run_evaluation:
        eval_results = experiment_3c_evaluate_ensemble(
            models_filter=models,
            global_test_indices=global_test_indices,
            device=str(device),
        )
        results["evaluation"] = eval_results

    # Save combined results
    if save_results and results:
        out_path = DIRS["results"] / "experiment_3_complete_results.json"
        DIRS["results"].mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        info(f"\n✓ Complete results saved to {out_path}")

    elapsed = time.time() - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    info(
        f"\nExperiment 3 completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
    )

    return results


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Experiment 3: Bagging Ensemble with Independent Training and Pruning
    =====================================================================

    This experiment trains 10 variants per architecture and prunes them,
    with the key difference that:
    - Training uses train/val splits (different per variant)
    - Pruning uses the SAME validation set as training (not test!)
    - Final ensemble is evaluated on test set (completely unseen)

    Usage:
    ------
    cd Image_Authenticity_prediction/main/Experiments/
    conda activate <your_env>
    python experiment_three.py

    Configuration Examples:
    -----------------------
    # Run complete pipeline for all models
    run_experiment_3()

    # Run only training
    run_experiment_3(run_pruning=False, run_evaluation=False)

    # Run only pruning (requires trained models)
    run_experiment_3(run_training=False, run_evaluation=False)

    # Run only evaluation (requires trained and pruned models)
    run_experiment_3(run_training=False, run_pruning=False)

    # Run for specific models only
    run_experiment_3(models=['vgg16', 'resnet152'])
    """

    set_level("DEBUG")

    # Configure which parts of the experiment to run
    run_experiment_3(
        models=[
            "vgg16",
            "vgg19",
            "resnet152",
            "densenet161",
            "efficientnetb3",
            "barlowtwins",
        ],
        run_training=True,
        run_pruning=True,
        run_evaluation=True,
        save_results=True,
    )
