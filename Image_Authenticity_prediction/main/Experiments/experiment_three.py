# ! WIP - Experiment 3: Ensemble Strategies (Bagging and Stacking)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
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
from typing import Sequence, Tuple, Optional, List, Dict, Any
import random

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
from main.Utils.logger import info, warn, error, debug, set_level
from main.Utils.comparisons import (
    compare_heatmaps,
    uniform_heatmaps,
)
from main.Utils.visualization import (
    visualize_similarity_matrix,
    visualize_similarity_distribution,
    visualize_violin_distribution,
)
from main.data import IMAGENET_DATASET, DENSENET_DATASET, SINGLE_BATCH_SIZE, NUM_WORKERS

# ============================================================================
# 1.1 DIRECTORIES
# ============================================================================
DIRS = {
    "output": Path("Outputs/Experiment_3_ensemble"),
    "weights": Path("Outputs/Experiment_3_ensemble/Weights"),
    "base_weights": Path(
        "Outputs/Experiment_1_variants/Weights"
    ),  # Pre-trained from exp1
}
DIRS["bagging_weights"] = DIRS["weights"] / "Bagging"
DIRS["stacking_weights"] = DIRS["weights"] / "Stacking"
DIRS["maps"] = DIRS["output"] / "XAI_Maps"
DIRS["gradcam"] = DIRS["maps"] / "GradCAM"
DIRS["mpm"] = DIRS["maps"] / "Multiscale_Pixel_Masking"
DIRS["plots"] = DIRS["output"] / "Plots"
DIRS["results"] = DIRS["output"] / "Results"

# ============================================================================
# 1.2 MODEL REGISTRY (identical to experiment_two)
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

XAI_PARAMS = {
    "sigma": [3, 17, 65],
    "mask_val": 0,
    "px_batch": 256,
    "gc_interval": 50,
    "mpm_interval": 10,
}

MODEL_ORDER = [
    "barlowtwins",
    "resnet152",
    "densenet161",
    "efficientnetb3",
    "vgg16",
    "vgg19",
]

# ============================================================================
# 1.3 ENSEMBLE CONFIGURATION
# ============================================================================
ENSEMBLE_CONFIG = {
    "random_state": 42,
    "batch_size": 32,
    "num_epochs_base": 20,
    "num_epochs_meta": 40,
    "learning_rate": 0.001,
    "learning_rate_meta": 0.001,
    "n_splits": 7,  # K-Fold splits for stacking
}


def setup_directories():
    """Create all required directories."""
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 2. REPRODUCIBILITY & UTILITIES
# ============================================================================


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# ============================================================================
# 3. MODEL LOADING (mirrors experiment_two pattern)
# ============================================================================


def get_weight_files(models_filter, variants_filter):
    """Get weight files matching filters (identical to experiment_two)."""
    if not DIRS["base_weights"].exists():
        return {}
    all_files = sorted(DIRS["base_weights"].glob("*.pth"))
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


def load_model_with_weights(
    model_name: str,
    weights_path: Path,
    device: str = "cuda",
    freeze_backbone: bool = False,
) -> nn.Module:
    """Load a model with pre-trained weights."""
    model_cls = MODEL_REGISTRY[model_name]["class"]
    model = model_cls(freeze_backbone=freeze_backbone)

    if weights_path.exists():
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        info(f"Loaded weights: {weights_path.name}")
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


class BaggingEnsemble:
    """Bagging ensemble that averages predictions from multiple models."""

    def __init__(self, models: List[nn.Module], device: str = "cuda"):
        self.models = models
        self.device = device
        for model in self.models:
            model.to(device)
            model.eval()

    def predict(self, dataloaders: Dict[str, DataLoader]) -> torch.Tensor:
        """
        Get averaged predictions from all models.

        Args:
            dataloaders: Dict mapping model_name to its dataloader
        """
        all_model_predictions = []

        for model_name, model in self.models:
            loader = dataloaders.get(model_name)
            if loader is None:
                warn(f"No dataloader for {model_name}, skipping")
                continue
            preds = get_predictions(model, loader, self.device)
            all_model_predictions.append(preds)

        stacked = torch.stack(all_model_predictions, dim=0)
        return torch.mean(stacked, dim=0)


def create_bootstrap_indices(dataset_size: int, seed: int = None) -> np.ndarray:
    """Create bootstrap sample indices (sampling with replacement)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(dataset_size, size=dataset_size, replace=True)


def train_bagging_ensemble(
    models_filter: List[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Train models using bagging strategy.
    Each model is trained on a bootstrap sample.
    """
    info("=" * 60)
    info("BAGGING ENSEMBLE TRAINING")
    info("=" * 60)

    set_seed(ENSEMBLE_CONFIG["random_state"])
    DIRS["bagging_weights"].mkdir(parents=True, exist_ok=True)

    trained_models = []
    results = {"models": [], "bootstrap_info": {}}

    models_to_train = models_filter or list(MODEL_REGISTRY.keys())

    for m_name in models_to_train:
        if m_name not in MODEL_REGISTRY:
            warn(f"Model {m_name} not in registry, skipping")
            continue

        info(f"\n--- Training {m_name} with bootstrap sample ---")

        # Check if already trained
        save_path = DIRS["bagging_weights"] / f"{m_name}_bagging.pth"
        if save_path.exists():
            info(f"  Loading existing weights: {save_path.name}")
            model = load_model_with_weights(
                m_name, save_path, device, freeze_backbone=False
            )
            trained_models.append((m_name, model))
            results["models"].append(m_name)
            continue

        # Get dataset
        dataset_dict = MODEL_REGISTRY[m_name]["dataset"]
        train_dataset = dataset_dict["train"]

        # Create bootstrap sample
        bootstrap_seed = ENSEMBLE_CONFIG["random_state"] + hash(m_name) % 1000
        bootstrap_indices = create_bootstrap_indices(len(train_dataset), bootstrap_seed)

        # Create subset
        if isinstance(train_dataset, Subset):
            global_indices = [train_dataset.indices[i] for i in bootstrap_indices]
            bootstrap_subset = Subset(train_dataset.dataset, global_indices)
        else:
            bootstrap_subset = Subset(train_dataset, bootstrap_indices.tolist())

        results["bootstrap_info"][m_name] = {
            "seed": bootstrap_seed,
            "unique_samples": len(np.unique(bootstrap_indices)),
            "total_samples": len(bootstrap_indices),
        }

        train_loader = DataLoader(
            bootstrap_subset,
            batch_size=ENSEMBLE_CONFIG["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

        # Initialize model
        model_cls = MODEL_REGISTRY[m_name]["class"]
        model = model_cls(freeze_backbone=False).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=ENSEMBLE_CONFIG["learning_rate"]
        )
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(ENSEMBLE_CONFIG["num_epochs_base"]):
            model.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(images)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(train_loader)
                info(
                    f"  Epoch {epoch+1}/{ENSEMBLE_CONFIG['num_epochs_base']}, Loss: {avg_loss:.4f}"
                )

        # Save weights
        torch.save(model.state_dict(), save_path)
        info(f"  Saved: {save_path.name}")

        trained_models.append((m_name, model))
        results["models"].append(m_name)

        clear_gpu_memory()

    results["ensemble"] = BaggingEnsemble(trained_models, device)
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


def train_stacking_ensemble(
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

    set_seed(ENSEMBLE_CONFIG["random_state"])
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
            model = load_model_with_weights(
                m_name, save_path, device, freeze_backbone=False
            )
        else:
            info(f"  Training: {m_name}")
            dataset_dict = MODEL_REGISTRY[m_name]["dataset"]
            train_loader = DataLoader(
                dataset_dict["train"],
                batch_size=ENSEMBLE_CONFIG["batch_size"],
                shuffle=True,
                num_workers=NUM_WORKERS,
            )

            model_cls = MODEL_REGISTRY[m_name]["class"]
            model = model_cls(freeze_backbone=False).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=ENSEMBLE_CONFIG["learning_rate"]
            )
            criterion = nn.MSELoss()

            for epoch in range(ENSEMBLE_CONFIG["num_epochs_base"]):
                model.train()
                epoch_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs, _ = model(images)
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 5 == 0:
                    info(
                        f"    Epoch {epoch+1}/{ENSEMBLE_CONFIG['num_epochs_base']}, Loss: {epoch_loss/len(train_loader):.4f}"
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
        random_state=ENSEMBLE_CONFIG["random_state"],
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
            random_state=ENSEMBLE_CONFIG["random_state"],
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
) -> Dict[str, float]:
    """Evaluate an ensemble on test data."""
    from scipy.stats import pearsonr

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
        # Load bagging models
        models = []
        for m_name in models_to_eval:
            weights_path = DIRS["bagging_weights"] / f"{m_name}_bagging.pth"
            if weights_path.exists():
                model = load_model_with_weights(m_name, weights_path, device)
                models.append((m_name, model))

        # Average predictions
        all_preds = []
        for m_name, model in models:
            preds = get_predictions(model, test_loaders[m_name], device)
            all_preds.append(preds.squeeze())

        y_pred = torch.mean(torch.stack(all_preds), dim=0).numpy()

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
    corr, p_val = pearsonr(y_pred, y_true)

    results = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(corr),
        "p_value": float(p_val),
    }

    info(f"  MSE:  {mse:.4f}")
    info(f"  RMSE: {rmse:.4f}")
    info(f"  MAE:  {mae:.4f}")
    info(f"  Pearson r: {corr:.4f} (p={p_val:.2e})")

    return results


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================


def run_experiment_3(
    models: List[str] = None,
    strategy: str = "both",  # "bagging", "stacking", or "both"
    train_models: bool = True,
    evaluate: bool = True,
    save_results: bool = True,
):
    """
    Run Experiment 3: Ensemble Strategies.

    Args:
        models: List of model names to include (None = all)
        strategy: "bagging", "stacking", or "both"
        train_models: Whether to train models
        evaluate: Whether to evaluate ensembles
        save_results: Whether to save results to JSON
    """
    start = time.time()
    setup_directories()
    set_seed(ENSEMBLE_CONFIG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device: {device}")

    if isinstance(models, str):
        models = [models]

    results = {}

    # --- Bagging ---
    if strategy in ["bagging", "both"]:
        if train_models:
            bagging_results = train_bagging_ensemble(models, device)
            results["bagging_training"] = {
                "models": bagging_results["models"],
                "bootstrap_info": bagging_results.get("bootstrap_info", {}),
            }
        if evaluate:
            results["bagging_evaluation"] = evaluate_ensemble("bagging", models, device)

    # --- Stacking ---
    if strategy in ["stacking", "both"]:
        if train_models:
            stacking_results = train_stacking_ensemble(models, device)
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
    set_level("INFO")
    run_experiment_3(
        models=[
            "barlowtwins",
            "resnet152",
            "densenet161",
            "efficientnetb3",
            "vgg16",
            "vgg19",
        ],
        strategy="both",
        train_models=True,
        evaluate=True,
        save_results=True,
    )
