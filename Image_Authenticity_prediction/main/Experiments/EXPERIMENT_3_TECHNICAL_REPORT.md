# Experiment 3: Ensemble Learning Strategies

**Status:** Work in Progress (WIP)

A comprehensive investigation of ensemble learning methods for image authenticity prediction, exploring how combining multiple models can improve prediction accuracy and robustness.

## 📋 Table of Contents

- [Overview](#overview)
- [Ensemble Strategies](#ensemble-strategies)
- [Methodology](#methodology)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Expected Outputs](#expected-outputs)
- [Research Questions](#research-questions)

## 🔍 Overview

Experiment 3 investigates whether ensemble methods can leverage the diversity of different CNN architectures and training variants to achieve better performance than individual models. Two main ensemble strategies are explored:

1. **Bagging (Bootstrap Aggregating)**: Averaging predictions from multiple variants of the same model trained with different random seeds
2. **Stacking (Meta-Learning)**: Training a meta-learner to optimally combine predictions from diverse base models

### Key Objectives

- Evaluate whether model diversity improves prediction accuracy
- Compare simple averaging (bagging) vs. learned combinations (stacking)
- Analyze the contribution of different base models to ensemble performance
- Investigate whether ensembles are more robust across different image types

## 🎯 Ensemble Strategies

### 1. Bagging Ensemble

**Concept**: Reduce variance by averaging predictions from multiple independently trained variants.

**Implementation**:

- Uses greedy-pruned model variants from Experiment 1B
- Each variant was trained with different random seeds and data shuffling
- Final prediction is the simple average of all variant predictions

**Advantages**:

- No additional training required
- Reduces overfitting through variance reduction
- Simple and interpretable

**Disadvantages**:

- Limited to variants of the same architecture
- Assumes equal weight for all models
- Cannot leverage architecture diversity

### 2. Stacking Ensemble

**Concept**: Train a meta-learner to optimally combine predictions from diverse base models.

**Implementation**:

- **Base Models**: Various CNN architectures (VGG, ResNet, DenseNet, etc.)
- **Meta-Learner**: Linear regression model trained on base model predictions
- **Training Strategy**: K-Fold cross-validation to generate out-of-fold (OOF) predictions
- **Architecture**: Simple linear layer (num_base_models → 1)

**Advantages**:

- Learns optimal weights for different models
- Can leverage architecture diversity
- Potentially better performance than simple averaging

**Disadvantages**:

- Requires additional meta-learner training
- Risk of overfitting to training data
- More complex to implement and debug

## 🔬 Methodology

### Phase 1: Base Model Preparation

1. **Load Pre-trained Models**: Use models trained in Experiment 1
2. **Validation**: Ensure all required model weights exist
3. **Dataset Preparation**: Prepare train/validation/test splits

### Phase 2: Bagging Ensemble

```python
# For each model architecture (e.g., VGG16):
#   1. Load all 10 greedy-pruned variants
#   2. Get predictions from each variant on test set
#   3. Average predictions across variants
#   4. Evaluate averaged predictions
```

**Aggregation Strategy**:

```python
final_prediction = mean([variant_1_pred, variant_2_pred, ..., variant_N_pred])
```

### Phase 3: Stacking Ensemble

**Step 1: Train Base Models**

- Train each architecture on full training data
- Save trained models for prediction generation

**Step 2: Generate Out-of-Fold Predictions**

- Use K-Fold cross-validation (K=7)
- For each fold:
  - Train base models on K-1 folds
  - Predict on held-out fold
- Combine OOF predictions as meta-features

**Step 3: Train Meta-Learner**

```python
# Meta-learner input: [pred_vgg16, pred_resnet, pred_densenet, ...]
# Meta-learner output: final_prediction
meta_learner = Linear(num_base_models, 1)
```

**Step 4: Evaluate on Test Set**

- Get predictions from all base models
- Feed into trained meta-learner
- Compare with individual model performance

## 💻 Implementation Details

### Configuration

```python
ENSEMBLE_CONFIG = {
    "batch_size": 32,
    "num_epochs_base": 500,          # Base model training
    "num_epochs_meta": 40,           # Meta-learner training
    "learning_rate": 0.001,
    "learning_rate_meta": 0.001,
    "n_splits": 7,                   # K-Fold splits
    "patience": 15,                  # Early stopping
}
```

### Directory Structure

```
Outputs/Experiment_3_ensemble/
├── Weights/
│   ├── Stacking/
│   │   ├── vgg16_stacking_base.pth
│   │   ├── resnet152_stacking_base.pth
│   │   ├── ...
│   │   └── meta_learner.pth
│   └── Bagging/
│       └── (uses weights from Experiment 1B)
└── Results/
    ├── bagging_results.json
    ├── stacking_results.json
    └── comparison_metrics.json
```

### Key Functions

#### `check_bagging_variants()`

Verifies availability of greedy-pruned model variants from Experiment 1B.

#### `train_ensemble()`

Trains stacking ensemble using K-Fold cross-validation:

1. Trains base models on full training data
2. Generates OOF predictions using K-Fold
3. Trains meta-learner on OOF predictions
4. Evaluates on test set

#### `evaluate_ensemble()`

Evaluates trained ensemble on test set and computes metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation Coefficient (PLCC)
- Spearman Rank Correlation (SRCC)
- Kendall Tau Correlation (KRCC)

## 🚀 Usage

### Command Line Interface

```bash
# Run bagging ensemble (uses pre-trained variants)
python -m Image_Authenticity_prediction experiment-three --strategy bagging

# Run stacking ensemble (requires training meta-learner)
python -m Image_Authenticity_prediction experiment-three --strategy stacking

# Run both strategies
python -m Image_Authenticity_prediction experiment-three --strategy both

# Specify models for stacking
python -m Image_Authenticity_prediction experiment-three \
    --strategy stacking \
    --models vgg16 resnet152 densenet161
```

### Python API

```python
from Image_Authenticity_prediction.main.Experiments.experiment_three import (
    run_experiment_3,
    check_bagging_variants,
    train_ensemble,
    evaluate_ensemble
)

# Check available bagging variants
availability = check_bagging_variants(
    models_filter=['vgg16', 'resnet152'],
    num_variants=10
)

# Train stacking ensemble
results = train_ensemble(
    models_filter=['vgg16', 'resnet152', 'densenet161'],
    device='cuda'
)

# Run complete experiment
experiment_results = run_experiment_3(
    strategy='both',  # 'bagging', 'stacking', or 'both'
    models_filter=None,  # None = all models
    device='cuda'
)
```

## 📊 Expected Outputs

### Bagging Results

```json
{
  "vgg16": {
    "num_variants": 10,
    "individual_performances": [0.023, 0.024, ...],
    "ensemble_performance": 0.021,
    "improvement": 0.002,
    "metrics": {
      "mse": 0.021,
      "rmse": 0.145,
      "plcc": 0.892,
      "srcc": 0.875,
      "krcc": 0.701
    }
  },
  "resnet152": {...},
  ...
}
```

### Stacking Results

```json
{
  "base_models": ["vgg16", "resnet152", "densenet161", ...],
  "meta_learner": {
    "architecture": "Linear(6 → 1)",
    "learned_weights": [0.15, 0.20, 0.18, 0.15, 0.17, 0.15]
  },
  "performance": {
    "mse": 0.019,
    "rmse": 0.138,
    "plcc": 0.908,
    "srcc": 0.893,
    "krcc": 0.728
  },
  "comparison_with_best_individual": {
    "best_individual_model": "resnet152",
    "best_individual_mse": 0.022,
    "ensemble_mse": 0.019,
    "improvement": 0.003
  }
}
```

## 🔍 Research Questions

### Primary Questions

1. **Does ensemble learning improve performance?**

   - Compare ensemble MSE vs. best individual model
   - Analyze performance across different metrics (PLCC, SRCC, KRCC)

2. **Which ensemble strategy works better?**

   - Compare bagging (simple averaging) vs. stacking (learned weights)
   - Evaluate computational cost vs. performance gain

3. **How does model diversity affect ensemble performance?**
   - Analyze correlation between base model predictions
   - Investigate if diverse architectures (VGG vs ResNet vs DenseNet) contribute more

### Secondary Questions

4. **Are ensembles more robust?**

   - Evaluate performance across different image types
   - Compare variance of predictions across test samples

5. **What are the optimal learned weights?**

   - Analyze meta-learner weights to understand model contributions
   - Compare learned weights with simple averaging

6. **Is there benefit in ensemble of pruned models?**
   - Compare bagging of pruned variants vs. original models
   - Evaluate if pruning affects ensemble diversity

## ⚠️ Current Status & Limitations

**Status: Work in Progress**

### Implemented

- ✅ Bagging ensemble framework
- ✅ Stacking ensemble with K-Fold cross-validation
- ✅ Meta-learner training and evaluation
- ✅ Comprehensive metrics computation

### In Progress

- 🔄 Extensive experimental evaluation
- 🔄 Analysis of model diversity and correlation
- 🔄 Comparison with state-of-the-art ensemble methods

### Future Work

- ⏳ Weighted voting strategies
- ⏳ Dynamic ensemble selection
- ⏳ Ensemble pruning (selecting optimal subset of base models)
- ⏳ Analysis of failure cases

## 📚 References

- Dietterich, T. G. (2000). "Ensemble Methods in Machine Learning"
- Wolpert, D. H. (1992). "Stacked Generalization"
- Breiman, L. (1996). "Bagging Predictors"

## 🔗 Related Experiments

- **[Experiment 1](EXPERIMENT_1_TECHNICAL_REPORT.md)**: Provides pre-trained base models and pruned variants
- **[Experiment 2](EXPERIMENT_2_TECHNICAL_REPORT.md)**: Analyzes explainability of individual models (can be extended to ensembles)

---

**Last Updated**: January 2026  
**Status**: Work in Progress  
**Contact**: Icaro Re Depaolini
