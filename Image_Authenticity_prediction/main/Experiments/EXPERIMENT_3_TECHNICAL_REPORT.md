# Experiment 3: Bagging Ensemble with Independent Training and Pruning

## Technical Report

**Version:** 2.1  
**Date:** February 2026  
**File:** `experiment_three.py`  
**Status:** Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Purpose & Motivation](#2-purpose--motivation)
3. [Methodology](#3-methodology)
4. [Configuration & Setup](#4-configuration--setup)
5. [Data Splitting Strategy](#5-data-splitting-strategy)
6. [Experiment 3A: Training](#6-experiment-3a-training)
7. [Experiment 3B: Pruning](#7-experiment-3b-pruning)
8. [Experiment 3C: Ensemble Evaluation](#8-experiment-3c-ensemble-evaluation)
9. [Complete Pipeline](#9-complete-pipeline)
10. [Output Files & Directory Structure](#10-output-files--directory-structure)
11. [Usage Examples](#11-usage-examples)
12. [Key Guarantees & Reproducibility](#12-key-guarantees--reproducibility)

---

## 1. Executive Summary

Experiment 3 implements a **multi-ensemble** approach for image authenticity prediction. The experiment trains **10 variants per model architecture** (60 total models across 6 architectures), supports bagging as the primary method, and also includes optional stacking and stacking-CV evaluations.

### Key Innovation

The critical methodological difference from Experiment 1 is the **separation of pruning and testing**:

| Stage            | Experiment 1                   | Experiment 3                            |
| ---------------- | ------------------------------ | --------------------------------------- |
| Training         | Train/Val split                | Train/Val split (different per variant) |
| Pruning          | Done on **Test set**           | Done on **Validation set**              |
| Final Evaluation | Test set (seen during pruning) | Test set (**completely unseen**)        |

This ensures the ensemble is evaluated on truly unseen data, providing a more rigorous assessment of generalization performance.

---

## 2. Purpose & Motivation

### 2.1 Research Questions

1. Does ensemble averaging of multiple pruned models improve prediction accuracy?
2. How does pruning on validation (vs. test) affect model generalization?
3. What is the benefit of model diversity through different train/val splits?

### 2.2 Bagging Strategy

**Bagging (Bootstrap Aggregating)** reduces variance by:

- Training multiple models on different data subsets
- Combining predictions through averaging
- Exploiting model diversity to improve robustness

In this implementation:

- Each variant sees a different train/val split (same test)
- Different random initialization of regression heads
- Same architecture, different learned features

---

## 3. Methodology

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT 3                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Dataset (N images)                                                 │
│       │                                                             │
│       ├── Test (20%) ─────────────────────────────┐                 │
│       │   [FIXED seed=42]                         │                 │
│       │                                           │                 │
│       └── Remaining (80%)                         │                 │
│               │                                   │                 │
│   ┌───────────┼───────────┐                       │                 │
│   ▼           ▼           ▼                       │                 │
│ Var1        Var2        Var10                     │                 │
│ seed=43     seed=44     seed=52                   │                 │
│   │           │           │                       │                 │
│   ▼           ▼           ▼                       │                 │
│ ┌─────┐    ┌─────┐    ┌─────┐                     │                 │
│ │Train│    │Train│    │Train│  ← 3A: Training     │                 │
│ │ Val │    │ Val │    │ Val │                     │                 │
│ └──┬──┘    └──┬──┘    └──┬──┘                     │                 │
│    │          │          │                        │                 │
│    ▼          ▼          ▼                        │                 │
│ ┌─────┐    ┌─────┐    ┌─────┐                     │                 │
│ │Prune│    │Prune│    │Prune│  ← 3B: Pruning      │                 │
│ │(Val)│    │(Val)│    │(Val)│    (on VAL set)     │                 │
│ └──┬──┘    └──┬──┘    └──┬──┘                     │                 │
│    │          │          │                        │                 │
│    └──────────┼──────────┘                        │                 │
│               │                                   │                 │
│               ▼                                   ▼                 │
│         ┌──────────┐                        ┌──────────┐            │
│         │ ENSEMBLE │ ───── evaluate on ───▶ │   TEST   │            │
│         │ (avg)    │                        │ (unseen) │            │
│         └──────────┘                        └──────────┘            │
│                                                                     │
│                              3C: Evaluation                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Architectures

| Model          | Backbone            | Target Pruning Layer                      | Input Size |
| -------------- | ------------------- | ----------------------------------------- | ---------- |
| VGG16          | ImageNet pretrained | `features.28`                             | 224×224    |
| VGG19          | ImageNet pretrained | `features.34`                             | 224×224    |
| ResNet152      | ImageNet pretrained | `features.7.2.conv3`                      | 224×224    |
| DenseNet161    | ImageNet pretrained | `features.denseblock4.denselayer24.conv2` | 300×300    |
| EfficientNetB3 | ImageNet pretrained | `features.8.0`                            | 224×224    |
| BarlowTwins    | Self-supervised     | `features.7.2.conv3`                      | 224×224    |

---

## 4. Configuration & Setup

### 4.1 Dependencies

```python
# Core
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Project modules
from main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor,
)
from main.Utils import FeatureMapsPruner
from main.train import train_model, test_model
from main.data import imageNet_dataset, denseNet_dataset
```

### 4.2 Configuration Loading

Configuration is loaded from `Configs/config.yaml`:

```python
TRAINING_CONFIG = get_training_config()
# Contains: learning_rate, max_epochs, patience, device, freeze_backbone

PRUNING_CONFIG = get_pruning_config()
# Contains: force_recompute

ENSEMBLE_CONFIG = get_ensemble_config()
# Contains: batch_size, aggregation method

DATA_CONFIG = get_data_config()
# Contains: batch_size, num_workers
```

### 4.3 Constants

```python
NUM_VARIANTS = 10  # Number of variants per model architecture
```

---

## 5. Data Splitting Strategy

### 5.1 Global Test Indices

The test set is **fixed across all variants** using a deterministic seed:

```python
def create_global_test_indices(
    dataset_size: int,
    test_fraction: float = 0.2,
    seed: int = 42
) -> List[int]:
    """
    Create global test indices that remain constant across all variants.
    """
    test_size = int(test_fraction * dataset_size)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=gen).tolist()
    return perm[:test_size]
```

**Key Properties:**

- Uses `seed=42` for reproducibility
- 20% of data reserved for testing
- Same indices used by ALL 60 model variants

### 5.2 Variant-Specific Train/Val Splits

Each variant gets a **different train/val split** while sharing the same test set:

```python
def create_variant_split(
    backbone_dataset,
    global_test_indices: List[int],
    variant_idx: int,
    val_fraction: float = 0.125,  # 0.125 of remaining ≈ 10% of total
) -> Tuple[Subset, Subset, Subset, List[int]]:
    """
    Create train/val/test split for a specific variant.
    Returns val_indices for reuse during pruning.
    """
```

**Algorithm:**

1. Exclude test indices from dataset
2. Shuffle remaining indices with `seed = 42 + variant_idx`
3. Split into train (87.5% of remaining) and val (12.5% of remaining)
4. Return `val_indices` for consistent pruning

**Split Proportions:**

| Split      | Fraction of Total |
| ---------- | ----------------- |
| Train      | ~70%              |
| Validation | ~10%              |
| Test       | 20% (fixed)       |

### 5.3 Variant Diversity Table

| Variant | Shuffle Seed | Train/Val Split | Test Set       |
| ------- | ------------ | --------------- | -------------- |
| 1       | 43           | Unique          | Same (seed 42) |
| 2       | 44           | Unique          | Same (seed 42) |
| 3       | 45           | Unique          | Same (seed 42) |
| ...     | ...          | ...             | Same (seed 42) |
| 10      | 52           | Unique          | Same (seed 42) |

---

## 6. Experiment 3A: Training

### 6.1 Function Signature

```python
def experiment_3a_train_all_variants(
    models_to_train: List[str] = None,
    global_test_indices: Dict[str, List[int]] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict[int, List[int]]]]:
```

### 6.2 Training Flow

```
For each model in [vgg16, vgg19, resnet152, densenet161, efficientnetb3, barlowtwins]:
    For variant_idx in 1..10:
        1. Create model instance with frozen backbone
        2. Reset regression head (random initialization)
        3. Create train/val/test split using create_variant_split()
        4. Store val_indices in variant_val_indices dict
        5. Create DataLoaders
        6. Setup optimizer (Adam) and criterion (MSELoss)
        7. Train with early stopping (patience-based)
        8. Save best model weights
        9. Save training history and plots
        10. Cleanup GPU memory

    Save variant_val_indices to JSON for pruning stage
```

### 6.3 Model Initialization

```python
# Create model with frozen backbone
model = config["class"](freeze_backbone=TRAINING_CONFIG["freeze_backbone"])

# Reset regression head for variant diversity
reset_regression_head(model)
```

The `reset_regression_head` function reinitializes the final linear layers:

```python
def reset_regression_head(model: nn.Module):
    """Reinitialize regression head weights for a distinct starting state."""
    for layer in model.regression_head.modules():
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()
```

### 6.4 Training Configuration

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=TRAINING_CONFIG["learning_rate"]
)

best_model, history = train_model(
    model=model,
    dataloaders={"train": train_loader, "val": val_loader},
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=TRAINING_CONFIG["max_epochs"],
    device=TRAINING_CONFIG["device"],
    patience=TRAINING_CONFIG["patience"],
)
```

### 6.5 Output Files

| File Type   | Pattern                                 | Example                             |
| ----------- | --------------------------------------- | ----------------------------------- |
| Weights     | `{model}_exp3a_variant{i}_best.pth`     | `vgg16_exp3a_variant1_best.pth`     |
| History     | `{model}_exp3a_variant{i}_history.npy`  | `vgg16_exp3a_variant1_history.npy`  |
| Plot        | `{model}_exp3a_variant{i}_training.png` | `vgg16_exp3a_variant1_training.png` |
| Val Indices | `variant_val_indices.json`              | (single file for all models)        |

### 6.6 Returns

```python
return (
    results,           # Training metrics per model/variant
    variant_val_indices  # Dict: model_name -> {variant_idx -> val_indices}
)
```

---

## 7. Experiment 3B: Pruning

### 7.1 Function Signature

```python
def experiment_3b_prune_all_variants(
    models_to_prune: List[str] = None,
    variant_val_indices: Dict[str, Dict[int, List[int]]] = None,
    global_test_indices: Dict[str, List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
```

### 7.2 Key Difference: Pruning on Validation Set

**Experiment 1:**

```python
pruner = FeatureMapsPruner(..., dataloader=test_loader, ...)
```

**Experiment 3:**

```python
pruner = FeatureMapsPruner(..., dataloader=val_loader, ...)  # Validation!
```

This ensures the test set remains completely unseen until final evaluation.

### 7.3 Pruning Flow

```
Load variant_val_indices.json (if not passed in memory)

For each trained weight file (*_exp3a_*.pth):
    1. Extract model_name and variant_idx from filename
    2. Load corresponding val_indices from saved JSON
    3. Create val_loader with SAME indices used during training
    4. Load trained model weights
    5. Create FeatureMapsPruner with val_loader
    6. Compute importance scores per channel
    7. Perform greedy pruning
    8. Save pruned weights and importance scores
    9. Cleanup GPU memory

Save pruning results to JSON
```

### 7.4 Greedy Pruning Algorithm

```python
# Importance score = baseline_mse - pruned_mse
# Positive = channel is important (removing hurts performance)
# Negative = channel is noisy (removing helps performance)

importance_scores = pruner.compute_importance_scores()

# Greedy removal
for channel in sorted_by_importance(least_important_first):
    zero_out(channel)
    new_mse = evaluate(model)
    if new_mse < current_best_mse:
        keep channel zeroed  # Improves performance
    else:
        restore channel       # Would hurt performance

save pruned_model
```

### 7.5 Output Files

| File Type      | Pattern                                      | Example                                  |
| -------------- | -------------------------------------------- | ---------------------------------------- |
| Pruned Weights | `{model}_exp3b_variant{i}_greedy_pruned.pth` | `vgg16_exp3b_variant1_greedy_pruned.pth` |
| Importance     | `{model}_exp3b_variant{i}_importance.npy`    | `vgg16_exp3b_variant1_importance.npy`    |
| Plot           | `{model}_exp3b_variant{i}_importance.png`    | `vgg16_exp3b_variant1_importance.png`    |
| Results        | `experiment_3b_pruning_results.json`         | (single file)                            |

### 7.6 Pruning Results Structure

```json
{
  "vgg16": {
    "variant1": {
      "baseline_mse": 0.0234,
      "baseline_rmse": 0.153,
      "final_mse": 0.0198,
      "final_rmse": 0.141,
      "improvement_mse": 0.0036,
      "improvement_rmse": 0.012,
      "removed_features": [45, 67, 89],
      "num_removed": 12,
      "reduction_percentage": 2.3,
      "pruned_weights_path": "...",
      "original_weights_path": "..."
    }
  }
}
```

---

## 8. Experiment 3C: Ensemble Evaluation

### 8.1 Function Signature

```python
def experiment_3c_evaluate_ensemble(
    models_filter: List[str] = None,
    global_test_indices: Dict[str, List[int]] = None,
    device: str = "cuda",
    ensemble_mode: List[str] = ["bagging"],
    stacking_cv_folds: int = 5,
    stacking_cv_repeats: int = 1,
) -> Dict[str, Any]:
```

### 8.2 Evaluation Flow

```
1. Recreate global_test_indices with seed=42 (if not provided)
2. Discover baseline files (`*_exp3a_variant*_best.pth`) and pruned files (`*_exp3b_variant*_greedy_pruned.pth`)
3. Group files by model architecture
4. Build test loaders for ImageNet and DenseNet transforms
5. Evaluate ALL baseline variants on test set (per-variant metrics + per-architecture mean/std)
6. Compute baseline ensemble by averaging baseline predictions
7. Evaluate ALL pruned variants on test set (per-variant metrics + per-architecture mean/std)
8. Compute one or more pruned ensembles depending on `ensemble_mode`:
   - `bagging`: simple average across all pruned predictions
   - `stacking`: linear meta-learner, 50/50 shuffled split (meta-train/meta-test)
   - `stacking_cv`: repeated k-fold out-of-fold linear stacking
9. Compute channel-reduction statistics directly from baseline/pruned weights
10. Build `comparison_summary` (baseline vs pruned vs ensemble variants)
11. Print full comparison and improvement deltas
12. Save full and comparison-only JSON outputs
```

### 8.3 Ensemble Prediction

Simple bagging averages model predictions:

$$\hat{y}_{ensemble} = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_i$$

Where $N$ is the number of models included in the selected ensemble.

For stacking modes, predictions from base models are used as features of a linear meta-learner:

$$\hat{y} = W x + b$$

- `stacking`: trains on half of the evaluation set and reports on the held-out half.
- `stacking_cv`: trains/infers out-of-fold across repeated k-fold splits.

### 8.4 Evaluation Metrics

| Metric | Description                            |
| ------ | -------------------------------------- |
| MSE    | Mean Squared Error                     |
| RMSE   | Root Mean Squared Error                |
| MAE    | Mean Absolute Error                    |
| PLCC   | Pearson Linear Correlation Coefficient |
| SRCC   | Spearman Rank Correlation Coefficient  |
| KRCC   | Kendall Rank Correlation Coefficient   |

Each ensemble variant also stores p-values for PLCC/SRCC/KRCC.

### 8.5 Results Structure

```json
{
  "baseline_ensemble": { "mse": 0.0, "plcc": 0.0, "...": "..." },
  "pruned_ensemble_bagging": { "mse": 0.0, "plcc": 0.0, "...": "..." },
  "pruned_ensemble_stacking": { "mse": 0.0, "plcc": 0.0, "...": "..." },
  "pruned_ensemble_stacking_cv": { "mse": 0.0, "plcc": 0.0, "...": "..." },
  "baseline_models": {
    "vgg16": {
      "variants": [{ "weights_path": "...", "mse": 0.0, "plcc": 0.0 }],
      "mean": { "mse": 0.0, "plcc": 0.0 },
      "std": { "mse": 0.0, "plcc": 0.0 }
    }
  },
  "pruned_models": {
    "vgg16": {
      "variants": [{ "weights_path": "...", "mse": 0.0, "plcc": 0.0 }],
      "mean": { "mse": 0.0, "plcc": 0.0 },
      "std": { "mse": 0.0, "plcc": 0.0 }
    }
  },
  "comparison_summary": {
    "baseline_architectures": {},
    "pruned_architectures": {},
    "channel_reduction": {},
    "overall_summary": {}
  }
}
```

---

## 9. Complete Pipeline

### 9.1 Function Signature

```python
def run_experiment_3(
    models: List[str] = None,
    run_training: bool = True,
    run_pruning: bool = True,
    run_evaluation: bool = True,
    save_results: bool = True,
    ensemble_mode: List[str] = ["bagging", "stacking"],
    stacking_cv_folds: int = 5,
    stacking_cv_repeats: int = 1,
) -> Dict[str, Any]:
```

### 9.2 Pipeline Orchestration

```python
# 1. Setup
setup_directories()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Create global test indices (shared across ALL stages)
global_test_indices = {
    "imagenet": create_global_test_indices(len(imageNet_dataset)),
    "densenet": create_global_test_indices(len(denseNet_dataset)),
}

# 3. Stage 3A: Training
if run_training:
    training_results, variant_val_indices = experiment_3a_train_all_variants(...)

# 4. Stage 3B: Pruning
if run_pruning:
    pruning_results = experiment_3b_prune_all_variants(...)

# 5. Stage 3C: Ensemble Evaluation
if run_evaluation:
    eval_results = experiment_3c_evaluate_ensemble(
        ...,
        ensemble_mode=ensemble_mode,
        stacking_cv_folds=stacking_cv_folds,
        stacking_cv_repeats=stacking_cv_repeats,
    )

# 6. Save combined results
save_json(results, "experiment_3_complete_results.json")
```

### 9.3 Main Block Default

Current `__main__` configuration runs evaluation only:

```python
run_experiment_3(
    models=["vgg16", "vgg19", "resnet152", "densenet161", "efficientnetb3", "barlowtwins"],
    run_training=False,
    run_pruning=False,
    run_evaluation=True,
    save_results=True,
    ensemble_mode=["bagging", "stacking", "stacking_cv"],
    stacking_cv_folds=5,
    stacking_cv_repeats=1,
)
```

---

## 10. Output Files & Directory Structure

### 10.1 Directory Tree

```
tmp_Outputs/Experiment_3_ensemble/
├── Weights/
│   ├── vgg16_exp3a_variant1_best.pth
│   ├── vgg16_exp3b_variant1_greedy_pruned.pth
│   └── (60 trained + 60 pruned = 120 files)
│
├── Ranking_arrays/
│   └── vgg16_exp3b_variant1_importance.npy (60 files)
│
├── Ranking_Plots/
│   └── vgg16_exp3b_variant1_importance.png (60 files)
│
├── Training_Plots/
│   └── vgg16_exp3a_variant1_training.png (60 files)
│
├── Training_History/
│   └── vgg16_exp3a_variant1_history.npy (60 files)
│
└── Results/
    ├── variant_val_indices.json
    ├── experiment_3b_pruning_results.json
    ├── experiment_3c_ensemble_results.json
    ├── experiment_3c_comparison.json
    └── experiment_3_complete_results.json
```

---

## 11. Usage Examples

### 11.1 Command Line

```bash
cd Image_Authenticity_prediction/main/Experiments/
conda activate <your_env>
python experiment_three.py
```

### 11.2 Programmatic Usage

```python
from experiment_three import run_experiment_3

# Full pipeline
results = run_experiment_3()

# Only training
results = run_experiment_3(run_pruning=False, run_evaluation=False)

# Only pruning (requires trained models)
results = run_experiment_3(run_training=False, run_evaluation=False)

# Only evaluation (requires pruned models)
results = run_experiment_3(run_training=False, run_pruning=False)

# Specific models only
results = run_experiment_3(models=['vgg16', 'resnet152'])

# Evaluation with all ensemble modes
results = run_experiment_3(
    run_training=False,
    run_pruning=False,
    run_evaluation=True,
    ensemble_mode=['bagging', 'stacking', 'stacking_cv'],
    stacking_cv_folds=5,
    stacking_cv_repeats=1,
)
```

---

## 12. Key Guarantees & Reproducibility

### 12.1 Data Leakage Protection

- Test split is created once via `seed=42` and reused globally.
- Variant-specific train/val splits are created only from non-test samples.
- Pruning uses the exact validation indices used during training.
- Final evaluation uses the reserved test split only.

### 12.2 Determinism Anchors

- Global test split: `create_global_test_indices(..., seed=42)`.
- Variant split shuffling: `seed = 42 + variant_idx`.
- Stacking split/CV shuffles: seeded with `SEED` (and `SEED + rep` for repeats).

### 12.3 Output Reproducibility Artifacts

- `variant_val_indices.json` stores validation indices per model/variant.
- Weights filenames encode stage (`exp3a` vs `exp3b`) and variant id.
- `experiment_3c_ensemble_results.json` stores full baseline/pruned/ensemble metrics.
- `experiment_3c_comparison.json` stores architecture-level and overall comparison summaries.

### 12.4 Methodological Notes

- `stacking` uses a 50/50 split inside the available evaluation set and is not a strictly untouched-test protocol for meta-learning.
- `stacking_cv` is marked exploratory in code and reports an explicit protocol note in outputs.
- Bagging on pruned models remains the primary leakage-safe ensemble path when test predictions are only averaged.

---
