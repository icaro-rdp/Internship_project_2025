# ! WIP - Experiment 3: Ensemble Strategies (Bagging and Stacking)

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


from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    NUM_WORKERS,
    SEED,
)
from main.train import train_model

# ============================================================================
# 1.1 DIRECTORIES
# ============================================================================
DIRS = {
    "base_weights": Path(
        "Outputs/Experiment_1_variants/Weights"
    ),  # Pre-trained from exp1
    "stacking_weights": Path("Outputs/Experiment_3_ensemble/Weights/Stacking"),
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
ENSEMBLE_CONFIG = {
    "batch_size": 32,
    "num_epochs_base": 500,
    "num_epochs_meta": 40,
    "learning_rate": 0.001,
    "learning_rate_meta": 0.001,
    "n_splits": 7,  # K-Fold splits for stacking
    "patience": 15,  # Early stopping patience
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
# 5. STACKING ENSEMBLE
# ============================================================================


class StackingMetaLearner(nn.Module):
    """Linear meta-learner for stacking ensemble."""

    def __init__(self, num_base_models: int):
        super().__init__()
        self.fc = nn.Linear(num_base_models, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_ensemble(
    models_filter: List[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Train models using stacking strategy with K-Fold OOF predictions.
    """
    from sklearn.model_selection import KFold, train_test_split

    info("=" * 60)
    info("STACKING ENSEMBLE TRAINING")
    info("=" * 60)

    DIRS["stacking_weights"].mkdir(parents=True, exist_ok=True)

    models_to_train = models_filter or list(MODEL_REGISTRY.keys())
    results = {"models": [], "base_models": []}

    # -------------------------------------------------------------------------
    # Step 1: Load/Train base models on full training data
    # -------------------------------------------------------------------------
    info("\n--- Step 1: Loading/Training Base Models ---")
    base_models = []

    for m_name in models_to_train:
        if m_name not in MODEL_REGISTRY:
            continue

        save_path = DIRS["stacking_weights"] / f"{m_name}_stacking_base.pth"

        if save_path.exists():
            info(f"  Loading existing: {m_name}")
            model = load_model_with_weights(m_name, save_path, device)
        else:
            info(f"  Training: {m_name}")
            dataset_dict = MODEL_REGISTRY[m_name]["dataset"]
            train_loader = DataLoader(
                dataset_dict["train"],
                batch_size=ENSEMBLE_CONFIG["batch_size"],
                shuffle=True,
                num_workers=NUM_WORKERS,
            )
            val_loader = DataLoader(
                dataset_dict["val"],
                batch_size=ENSEMBLE_CONFIG["batch_size"],
                shuffle=False,
                num_workers=NUM_WORKERS,
            )

            dataloaders = {"train": train_loader, "val": val_loader}

            # Initialize model (freeze_backbone=True by default, only regression head trainable)
            model_cls = MODEL_REGISTRY[m_name]["class"]
            model = model_cls()  # Backbone frozen by default
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=ENSEMBLE_CONFIG["learning_rate"],
            )
            criterion = nn.MSELoss()

            # Train using train_model function with early stopping
            model, history = train_model(
                model=model,
                dataloaders=dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=ENSEMBLE_CONFIG["num_epochs_base"],
                device=device,
                patience=ENSEMBLE_CONFIG["patience"],
            )

            torch.save(model.state_dict(), save_path)
            info(f"    Saved: {save_path.name}")

        base_models.append((m_name, model))
        results["models"].append(m_name)
        clear_gpu_memory()

    # -------------------------------------------------------------------------
    # Step 2: Generate OOF predictions using K-Fold
    # -------------------------------------------------------------------------
    info(
        f"\n--- Step 2: Generating OOF Predictions ({ENSEMBLE_CONFIG['n_splits']}-Fold) ---"
    )

    # Use IMAGENET as reference for indices
    main_train_dataset = IMAGENET_DATASET["train"]
    n_samples = len(main_train_dataset)
    n_models = len(base_models)

    oof_predictions = torch.zeros(n_samples, n_models)
    oof_labels = torch.zeros(n_samples)

    kf = KFold(
        n_splits=ENSEMBLE_CONFIG["n_splits"],
        shuffle=True,
        random_state=SEED,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
        info(
            f"  Fold {fold_idx + 1}/{ENSEMBLE_CONFIG['n_splits']} ({len(val_idx)} val samples)"
        )

        # Get labels for this fold
        if isinstance(main_train_dataset, Subset):
            base_ds = main_train_dataset.dataset
            global_val_indices = [main_train_dataset.indices[i] for i in val_idx]
        else:
            base_ds = main_train_dataset
            global_val_indices = list(val_idx)

        val_subset = Subset(base_ds, global_val_indices)
        label_loader = DataLoader(
            val_subset, batch_size=ENSEMBLE_CONFIG["batch_size"], shuffle=False
        )
        fold_labels = get_labels(label_loader)
        oof_labels[val_idx] = fold_labels

        # Get predictions from each base model
        for model_idx, (m_name, model) in enumerate(base_models):
            dataset_dict = MODEL_REGISTRY[m_name]["dataset"]
            model_train_ds = dataset_dict["train"]

            if isinstance(model_train_ds, Subset):
                model_base = model_train_ds.dataset
                model_val_indices = [model_train_ds.indices[i] for i in val_idx]
            else:
                model_base = model_train_ds
                model_val_indices = list(val_idx)

            fold_val_subset = Subset(model_base, model_val_indices)
            fold_loader = DataLoader(
                fold_val_subset,
                batch_size=ENSEMBLE_CONFIG["batch_size"],
                shuffle=False,
                num_workers=NUM_WORKERS,
            )

            preds = get_predictions(model, fold_loader, device)
            oof_predictions[val_idx, model_idx] = preds.squeeze()

    # -------------------------------------------------------------------------
    # Step 3: Train Meta-Learner
    # -------------------------------------------------------------------------
    info("\n--- Step 3: Training Meta-Learner ---")

    meta_save_path = DIRS["stacking_weights"] / "meta_learner.pth"

    if meta_save_path.exists():
        info("  Loading existing meta-learner")
        meta_learner = StackingMetaLearner(n_models)
        meta_learner.load_state_dict(
            torch.load(meta_save_path, map_location=device, weights_only=True)
        )
        meta_learner.to(device)
    else:
        # Split OOF for meta-learner training
        X_train, X_val, y_train, y_val = train_test_split(
            oof_predictions,
            oof_labels,
            test_size=0.2,
            random_state=SEED,
        )

        meta_train_ds = TensorDataset(X_train, y_train)
        meta_val_ds = TensorDataset(X_val, y_val)
        meta_train_loader = DataLoader(
            meta_train_ds, batch_size=ENSEMBLE_CONFIG["batch_size"], shuffle=True
        )
        meta_val_loader = DataLoader(
            meta_val_ds, batch_size=ENSEMBLE_CONFIG["batch_size"], shuffle=False
        )

        meta_learner = StackingMetaLearner(n_models).to(device)
        optimizer = torch.optim.Adam(
            meta_learner.parameters(), lr=ENSEMBLE_CONFIG["learning_rate_meta"]
        )
        criterion = nn.MSELoss()

        for epoch in range(ENSEMBLE_CONFIG["num_epochs_meta"]):
            meta_learner.train()
            train_loss = 0.0
            for X_batch, y_batch in meta_train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = meta_learner(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                info(
                    f"  Epoch {epoch+1}/{ENSEMBLE_CONFIG['num_epochs_meta']}, Loss: {train_loss/len(meta_train_loader):.4f}"
                )

        torch.save(meta_learner.state_dict(), meta_save_path)
        info(f"  Saved: {meta_save_path.name}")

    results["base_models"] = base_models
    results["meta_learner"] = meta_learner
    return results


# ============================================================================
# 6. EVALUATION
# ============================================================================


def evaluate_ensemble(
    strategy: str,
    models_filter: List[str] = None,
    device: str = "cuda",
    weight_paths: List[Tuple[str, Path]] = None,
) -> Dict[str, float]:
    """Evaluate an ensemble on test data (memory-efficient: one model at a time)."""
    from scipy.stats import pearsonr, spearmanr, kendalltau

    info(f"\n--- Evaluating {strategy.upper()} Ensemble ---")

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

    if strategy == "bagging":
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

    elif strategy == "stacking":
        # Load base models and meta-learner
        base_models = []
        for m_name in models_to_eval:
            weights_path = DIRS["stacking_weights"] / f"{m_name}_stacking_base.pth"
            if weights_path.exists():
                model = load_model_with_weights(m_name, weights_path, device)
                base_models.append((m_name, model))

        meta_path = DIRS["stacking_weights"] / "meta_learner.pth"
        meta_learner = StackingMetaLearner(len(base_models))
        meta_learner.load_state_dict(
            torch.load(meta_path, map_location=device, weights_only=True)
        )
        meta_learner.to(device).eval()

        # Get base model predictions
        base_preds = []
        for m_name, model in base_models:
            preds = get_predictions(model, test_loaders[m_name], device)
            base_preds.append(preds.squeeze())

        X_meta = torch.stack(base_preds, dim=1).to(device)

        with torch.no_grad():
            y_pred = meta_learner(X_meta).squeeze().cpu().numpy()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

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
    strategy: str = "both",  # "bagging", "stacking", or "both"
    train: bool = True,
    evaluate: bool = True,
    save_results: bool = True,
):
    """
    Run Experiment 3: Ensemble Strategies.

    Args:
        models: List of model names to include (None = all)
        strategy: "bagging", "stacking", or "both"
        train: Whether to train stacking models (bagging uses pre-trained)
        evaluate: Whether to evaluate ensembles
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

    # --- Bagging ---
    if strategy in ["bagging", "both"]:
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
                "bagging", models, device, weight_paths=bagging_results["weight_paths"]
            )

    # --- Stacking ---
    if strategy in ["stacking", "both"]:
        if train:
            stacking_results = train_ensemble(models, device)
            results["stacking_training"] = {
                "models": stacking_results["models"],
            }
        if evaluate:
            results["stacking_evaluation"] = evaluate_ensemble(
                "stacking", models, device
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
        strategy="bagging",
        evaluate=True,
        save_results=True,
    )
