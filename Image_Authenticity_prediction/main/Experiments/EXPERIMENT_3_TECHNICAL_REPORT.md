# Experiment 3: Variant-Aligned Unpruned 7-Model Ensembles and MPM Attribution

## Technical Report

**Version:** 3.0  
**Date:** November 2026  
**Implementation:** `Image_Authenticity_prediction/main/Experiments/experiment_three.py`  
**Specification:** `docs/adr/0005-experiment-3-variant-aligned-ensembles-and-xai.md`  
**Status:** Complete (Fully Executed on DIPSEN HPC Cluster, SLURM Job `10853`)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Key Design Principles](#2-key-design-principles)
3. [Model Suite & Canonical Ordering](#3-model-suite--canonical-ordering)
4. [Data Splitting Strategy & Leakage Prevention](#4-data-splitting-strategy--leakage-prevention)
5. [Ensemble Methods (Stage 3C)](#5-ensemble-methods-stage-3c)
6. [Explainability & Attribution Consistency (Stage 3D)](#6-explainability--attribution-consistency-stage-3d)
7. [Quantitative Results](#7-quantitative-results)
8. [Directory Structure & Generated Artifacts](#8-directory-structure--generated-artifacts)
9. [Usage Examples (CLI, SLURM, and Python API)](#9-usage-examples-cli-slurm-and-python-api)
10. [Reproducibility & Key Guarantees](#10-reproducibility--key-guarantees)

---

## 1. Executive Summary

Experiment 3 evaluates multi-architecture ensemble methods for predicting human judgments of visual authenticity in AI-generated imagery. 

The pipeline is organized around five core components:
- **7 Architectures Included:** Incorporates the Vision Transformer backbone `clip_vit_b16` (`CLIPViTAuthenticityPredictor`) alongside 6 distinct convolutional networks (`vgg16`, `vgg19`, `resnet152`, `densenet161`, `efficientnetb3`, `barlowtwins`).
- **Unpruned Representations:** Directly reuses the 70 unpruned checkpoints from Experiment 1A across 10 random seeds, evaluating the full learned feature representations.
- **Matched 10-Variant Ensembles:** Evaluates 10 independent ensembles ($k = 1 \dots 10$) where all 7 models share the exact same training split and separate validation split ($\text{val}_k$).
- **Clean Validation Stacking:** Stacking weights are learned strictly on the separate validation sets ($136$ images each) using closed-form Ridge regression ($\alpha = 1.0$), with zero test data leakage.
- **Full-Dataset Attribution Maps (MPM):** Generates $(273, 224, 224)$ Multiscale Pixel Masking attribution heatmaps for all 273 test images across all 10 runs, producing ensemble prototype heatmaps and evaluating pairwise consistency across all 45 run pairs.

---

## 2. Key Design Principles

| Design Dimension | Implementation Details |
| :--- | :--- |
| **Model Architectures** | 7 architectures (70 total models across 10 random seeds), spanning supervised CNNs, self-supervised contrastive learning, and a multimodal vision-language transformer. |
| **Model Weights** | Direct reuse of unpruned Experiment 1A checkpoints (`Outputs/Experiment_1_variants/Weights/`), avoiding redundant training and preserving full representations. |
| **Validation Stacking** | Closed-form regularized Ridge regression ($\alpha = 1.0$) fitted strictly on held-out validation splits ($\text{val}_k$, $N = 136$), evaluated on all 273 unseen test images. |
| **Attribution & Consistency** | Full-test-set MPM heatmaps ($273 \times 224 \times 224$) saved as float32 `.npy` arrays, with prototype heatmaps and systematic consistency metrics across all 45 run pairs. |

---

## 3. Model Suite & Canonical Ordering

The ensemble combines 7 distinct vision architectures representing different training methods:
1. `vgg16` (Supervised CNN, 224×224)
2. `vgg19` (Supervised deep CNN, 224×224)
3. `resnet152` (Supervised residual CNN, 224×224)
4. `densenet161` (Supervised densely connected CNN, 300×300)
5. `efficientnetb3` (Supervised compound-scaled CNN, 224×224)
6. `barlowtwins` (Self-supervised contrastive ResNet-50, 224×224)
7. `clip_vit_b16` (Multimodal vision-and-language Vision Transformer, 224×224)

### Canonical Model Order

To prevent any column misalignment when stacking, all prediction matrices strictly follow this fixed ordering:
```python
CANONICAL_MODEL_ORDER = (
    "vgg16",          # Column 0
    "vgg19",          # Column 1
    "resnet152",      # Column 2
    "densenet161",    # Column 3
    "efficientnetb3", # Column 4
    "barlowtwins",    # Column 5
    "clip_vit_b16",   # Column 6
)
```

---

## 4. Data Splitting Strategy & Leakage Prevention

The study uses the Kept dataset ($1{,}368$ photorealistic images across 57 prompts).

```
                      1,368 Kept Images
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   273 Test Images (20%)             1,095 Non-Test Images (80%)
   [Fixed seed = 42]                         │
   Completely untouched until        ┌───────┴───────┐
   final test evaluation             ▼               ▼
                               Variant 1       Variant 10
                               (seed = 43)     (seed = 52)
                                 ┌───┬───┐       ┌───┬───┐
                                 │Tr │Val│  ...  │Tr │Val│
                                 └───┴───┘       └───┴───┘
                                 959  136        959  136
```

### 4.1 Global Test Set ($N = 273$)
* Sampled deterministically with `seed = 42`.
* Identical across all architectures and all 10 runs.
* Never seen during base model training or stacking weight estimation.

### 4.2 Matched Variant Splits ($\text{val}_k, N = 136$)
* For each run index $k \in \{1, \dots, 10\}$, non-test images are partitioned with `seed = 42 + k`.
* Split ratio: ~70% training ($959$ images) and ~10% validation ($136$ images).
* In Ensemble Variant $k$, **all 7 models saw the exact same 959 training images**. The 136 validation images ($\text{val}_k$) were never seen by any of the 7 models during training.

---

## 5. Ensemble Methods (Stage 3C)

### 5.1 Bagging Ensemble (`bagging`)
A simple unweighted average across all 7 base model predictions for run $k$:
$$\hat{y}_{\text{bag}}^{(k)}(x) = \frac{1}{7} \sum_{m=1}^{7} f_m^{(k)}(x)$$
Evaluated on all 273 test images across all 10 runs.

### 5.2 Stacking Meta-Learner (`stacking`)
A linear model trained to combine model outputs using weights learned on validation data.

#### Closed-Form Ridge Solution ($\alpha = 1.0$)
To prevent training instability or random differences between runs, the weights are computed using the exact closed-form solution:
$$\mathbf{w}^{(k)} = \left( (X_{\text{val, ext}}^{(k)})^T X_{\text{val, ext}}^{(k)} + \alpha I \right)^{-1} (X_{\text{val, ext}}^{(k)})^T \mathbf{y}_{\text{val}}^{(k)}$$
where:
* $X_{\text{val, ext}}^{(k)} \in \mathbb{R}^{136 \times 8}$ is the validation prediction matrix with an added constant column of 1s for the intercept.
* $\mathbf{y}_{\text{val}}^{(k)} \in \mathbb{R}^{136}$ are the true human authenticity ratings for split $k$.
* $\alpha = 1.0$ is the regularization parameter.

#### PyTorch Deployment
The weights are loaded directly into a `StackingMetaLearner(num_base_models=7)` module (`nn.Linear(7, 1)`):
```python
class StackingMetaLearner(nn.Module):
    def __init__(self, num_base_models: int = 7):
        super().__init__()
        self.fc = nn.Linear(num_base_models, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
```
Trained checkpoints are saved to `Outputs/Experiment_3_ensemble/Weights/stacking_exp3_variant{k}_meta.pth`.

---

## 6. Explainability & Attribution Consistency (Stage 3D)

Stage 3D generates Multiscale Pixel Masking (MPM) heatmaps across all 273 test images to test whether combining models produces more consistent and stable explanations.

### 6.1 Heatmap Generation
* **Scale Perturbations:** Zero-masks image patches across 3 spatial kernel sizes ($\sigma \in \{3, 17, 65\}$) with strides $\{2, 8, 32\}$.
* **Bagging Heatmap:** Simple average of normalized heatmaps across the 7 models for run $k$:
  $$H_{\text{bag}}^{(k)}(x) = \frac{1}{7} \sum_{m=1}^{7} H_m^{(k)}(x)$$
* **Stacking Heatmap:** Weighted sum of normalized heatmaps using the positive learned stacking weights:
  $$H_{\text{stack}}^{(k)}(x) = \sum_{m=1}^{7} \max(0, w_m^{(k)}) \cdot H_m^{(k)}(x)$$
* **Prototype Heatmap:** Mean heatmap across all 10 independent runs for an image:
  $$\bar{H}(x) = \frac{1}{10} \sum_{k=1}^{10} H^{(k)}(x)$$

### 6.2 Consistency Evaluation
We evaluate consistency across all 45 pairs of independent runs across all 273 test images ($12,285$ image comparisons):
1. **Pearson Correlation ($r$):** Pixel-by-pixel linear correlation.
2. **Structural Similarity (SSIM):** Spatial structure and pattern agreement.
3. **Top-$k$% Overlap (IoU):** Intersection-over-Union of the highest attribution regions at $5\%$, $15\%$, and $25\%$.

### 6.3 Rationale: Average & Weighted Average vs. Live Recomputation

We chose to generate ensemble heatmaps by combining the precomputed individual model heatmaps (simple average for Bagging and weighted sum for Stacking) instead of running a live recomputation of MPM on the ensemble as a whole.

This design decision is justified by two key reasons:

1. **Near-Perfect Correlation ($r \approx 1.0$):**
   Because both Bagging (uniform mean) and Stacking (linear combiner) are linear operations, masking an image patch produces an ensemble prediction drop that directly mirrors the weighted sum of the prediction drops in the individual models. Comparing the fast combined heatmaps against live recomputed heatmaps shows that their spatial correlation is virtually 1 ($r \approx 1.0$). The resulting heatmaps are practically identical.

2. **Massive Computational Time Savings:**
   Running live MPM directly on the ensemble requires re-evaluating each masked patch through all 7 models across 3 spatial perturbation scales for all 273 test images and 10 runs. This requires tens of thousands of forward passes and takes roughly **3 to 4 hours of heavy GPU compute**. In contrast, combining the already-computed individual model heatmaps takes only **~8 to 15 seconds**, saving hours of cluster execution time without any loss in accuracy or detail.

---

## 7. Quantitative Results

Empirical results computed on the held-out 273-image test set across all 10 independent runs (SLURM Job `10853`):

### 7.1 Prediction Performance Comparison

| Model / Ensemble | PLCC (Mean ± Std) | SRCC (Mean ± Std) | RMSE (Mean ± Std) | Prediction Reliability ($r_{\text{model}}$) | Combined Ceiling |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vgg16` | $0.5397 \pm 0.0276$ | $0.5730 \pm 0.0249$ | $8.6181 \pm 0.4044$ | $0.8559$ | $0.7573$ |
| `vgg19` | $0.5628 \pm 0.0271$ | $0.5869 \pm 0.0243$ | $8.5220 \pm 0.3392$ | $0.8607$ | $0.7594$ |
| `resnet152` | $0.5059 \pm 0.0196$ | $0.5265 \pm 0.0212$ | $9.8550 \pm 0.3657$ | $0.9162$ | $0.7835$ |
| `densenet161` | $0.6274 \pm 0.0179$ | $0.6166 \pm 0.0170$ | $7.4813 \pm 0.2130$ | $0.9088$ | $0.7803$ |
| `efficientnetb3` | $0.4966 \pm 0.0199$ | $0.5056 \pm 0.0240$ | $8.8185 \pm 0.2484$ | $0.8631$ | $0.7604$ |
| `barlowtwins` | $0.6169 \pm 0.0119$ | $0.6292 \pm 0.0115$ | $7.0965 \pm 0.1002$ | $0.9431$ | $0.7949$ |
| `clip_vit_b16` *(best single)* | $0.7317 \pm 0.0059$ | $0.7456 \pm 0.0055$ | $6.0206 \pm 0.0450$ | $0.9757$ | $0.8085$ |
| **Bagging Ensemble (7 models)** | **$0.7519 \pm 0.0069$** | **$0.7484 \pm 0.0072$** | **$5.9057 \pm 0.0956$** | **$0.9680$** | **$0.8053$** |
| **Stacking Ensemble (Ridge)** | $\mathbf{0.7742 \pm 0.0102}$ | $\mathbf{0.7734 \pm 0.0090}$ | $\mathbf{5.5442 \pm 0.1049}$ | **$0.9511$** | **$0.7983$** |

* Human noise ceiling baseline: $r_{\text{human}} = 0.67$.

### 7.2 Stacking Model Weights

The learned Ridge regression weights show how each architecture contributes to the final prediction:
* `clip_vit_b16`: **$+0.3929$** ($38.5\%$ relative weight)
* `barlowtwins`: **$+0.2115$** ($20.7\%$ relative weight)
* `densenet161`: **$+0.1683$** ($16.5\%$ relative weight)
* `vgg19`: $+0.0829$ ($8.1\%$)
* `efficientnetb3`: $+0.0739$ ($7.2\%$)
* `resnet152`: $+0.0595$ ($5.8\%$)
* `vgg16`: $+0.0308$ ($3.0\%$)
* Intercept: $-0.1671$

### 7.3 Heatmap Consistency Across Independent Runs

| Metric | Bagging Ensemble | Stacking Ensemble | Single CNN Benchmark Range |
| :--- | :---: | :---: | :---: |
| **Pearson Correlation ($r$)** | **$0.8864 \pm 0.0447$** | **$0.8193 \pm 0.0564$** | $0.65 - 0.75$ |
| **Structural Similarity (SSIM)** | **$0.5571 \pm 0.0980$** | **$0.4735 \pm 0.0923$** | $0.35 - 0.45$ |
| **Top-5% Overlap (IoU)** | **$0.4632 \pm 0.1171$** | **$0.3963 \pm 0.1057$** | $0.25 - 0.35$ |
| **Top-15% Overlap (IoU)** | **$0.5602 \pm 0.0852$** | **$0.4975 \pm 0.0843$** | $0.38 - 0.48$ |
| **Top-25% Overlap (IoU)** | **$0.6206 \pm 0.0714$** | **$0.5604 \pm 0.0713$** | $0.45 - 0.55$ |

**Takeaway:** Combining models produces heatmaps that are substantially more stable and repeatable across runs than those of any single architecture.

---

## 8. Directory Structure & Generated Artifacts

All outputs route to `Outputs/Experiment_3_ensemble/`:

```
Outputs/Experiment_3_ensemble/
├── Weights/
│   ├── stacking_exp3_variant1_meta.pth ... stacking_exp3_variant10_meta.pth
│   └── stacking_coefficients.json
│
├── Results/
│   ├── experiment_3c_ensemble_results.json   # Detailed metrics per variant
│   ├── experiment_3c_comparison.json         # Summary comparison against baselines
│   ├── experiment_3d_xai_consistency.json    # Heatmap consistency metrics
│   └── experiment_3_complete_results.json    # Combined experiment output
│
├── XAI_Maps/
│   └── Multiscale_Pixel_Masking/
│       ├── bagging_exp3_variant1_maps.npy ... variant10_maps.npy   # (273, 224, 224) float32
│       ├── stacking_exp3_variant1_maps.npy ... variant10_maps.npy  # (273, 224, 224) float32
│       ├── bagging_exp3_prototype_maps.npy                         # (273, 224, 224) float32
│       └── stacking_exp3_prototype_maps.npy                        # (273, 224, 224) float32
│
└── Plots/
    ├── mpm_within_bagging_correlation_matrix.png
    ├── mpm_within_bagging_correlation_distribution.png
    ├── mpm_within_bagging_top_percent_iou_15_matrix.png
    ├── mpm_within_bagging_top_percent_iou_15_distribution.png
    ├── mpm_within_stacking_correlation_matrix.png
    ├── mpm_within_stacking_correlation_distribution.png
    ├── mpm_within_stacking_top_percent_iou_15_matrix.png
    ├── mpm_within_stacking_top_percent_iou_15_distribution.png
    ├── mpm_prototypes_ensemble_vs_single_correlation_matrix.png
    └── mpm_prototypes_ensemble_vs_single_top_percent_iou_15_matrix.png
```

---

## 9. Usage Examples (CLI, SLURM, and Python API)

### 9.1 Command Line Interface (CLI)

Run via `uv` in the project environment:

```bash
# Run full evaluation & heatmap generation reusing Experiment 2 maps (fast, ~3-5 mins)
uv run python -m Image_Authenticity_prediction experiment-three \
    --strategy all \
    --variants all \
    --reuse-exp2-maps

# Run with live MPM recomputation on GPU
uv run python -m Image_Authenticity_prediction experiment-three \
    --strategy all \
    --live-mpm

# Evaluate only Bagging on specific variants
uv run python -m Image_Authenticity_prediction experiment-three \
    --strategy bagging \
    --variants 1,2,3
```

### 9.2 SLURM Cluster Batch Submission

Submit directly to the DIPSEN HPC Blackwell GPU node (`c31`):

```bash
sbatch scripts/slurm/run_experiment_three.slurm
```

To pass custom CLI arguments:

```bash
sbatch scripts/slurm/run_experiment_three.slurm --strategy stacking --variants 1,2,3,4,5
```

### 9.3 Python API

```python
from main.Experiments.experiment_three import run_experiment_3

results = run_experiment_3(
    models=None,                      # Use all 7 registered models
    run_training=False,               # Directly reuses trained Exp 1A weights
    run_pruning=False,                # Evaluates full unpruned representations
    run_evaluation=True,              # Run Stage 3C ensemble evaluation
    run_heatmaps=True,                # Run Stage 3D MPM attribution
    ensemble_mode=["bagging", "stacking"],
    variants="all",
    reuse_exp2_maps=True,             # Reuse precomputed single-model maps
)

print("Bagging PLCC:", results["evaluation"]["bagging"]["mean"]["plcc"])
print("Stacking PLCC:", results["evaluation"]["stacking"]["mean"]["plcc"])
```

---

## 10. Reproducibility & Key Guarantees

1. **Fixed Global Test Set:** `seed = 42` guarantees identical test sample selection ($N = 273$) across all variants, models, and stages.
2. **Disjoint Validation Splits:** `seed = 42 + k` guarantees that validation split $\text{val}_k$ ($N = 136$) is completely disjoint from the training data for all models in variant $k$.
3. **Closed-Form Meta-Learning:** Ridge regression ($\alpha = 1.0$) eliminates optimization noise, guaranteeing reproducible stacking weights given the validation predictions.
4. **Target Layer Uniformity:** All CNN models extract spatial feature maps from their canonical target layers (`features.28` for VGG16, `features.34` for VGG19, etc.), and `clip_vit_b16` extracts representations from `visual_projection`.
5. **No Data Leakage:** Stacking weights and bagging predictions are tested exclusively on held-out test data that was never seen during training or meta-learner fitting.
