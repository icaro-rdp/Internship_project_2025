# Experiment 2: Explainability Analysis - Technical Report

**Date:** November 2026  
**File:** `main/Experiments/experiment_two.py`  
**Utilities:** `main/Utils/comparisons.py`, `main/Utils/explainability.py`, `main/Utils/visualization.py`

---

## Executive Summary

Experiment 2 implements a comprehensive explainability analysis pipeline for image authenticity prediction models. The experiment consists of two main phases:

1. **Experiment 2A**: Generation of explainability maps and attribution representations using three distinct XAI (Explainable AI) paradigms:
   - **GradCAM** (gradient-based spatial activation mapping)
   - **Multiscale Pixel Masking (MPM)** (multi-scale Gaussian pixel-occlusion attribution)
   - **LIME** (GPU-vectorized superpixel perturbation and local Ridge surrogate modeling)
2. **Experiment 2B**: Comparative analysis of explainability across models, variants, and methods:
   - Inter-model prototype analysis across diverse neural architectures (CNNs and Vision Transformers)
   - Intra-model variant stability across random seeds and pruning strategies
   - LIME-specific surrogate fidelity ($R^2$) diagnostics and 1D superpixel coefficient ($\beta$) consistency
   - Cross-method alignment evaluating attribution consensus across different explainability mechanisms

The pipeline incorporates a novel **prototype-based comparison** framework, high-throughput GPU batching with Automatic Mixed Precision (AMP), shared canonical superpixel tessellations, and advanced **distribution visualization** capabilities.

---

## System Architecture

### Core Components

#### 1. **Explainability Attribution Generation (Experiment 2A)**

**Purpose:** Generate quantitative visual explanations and feature attributions of model predictions using three complementary XAI methods.

**Supported XAI Methods:**

- **GradCAM (Gradient-weighted Class Activation Mapping)**
  - Generates spatial heatmaps identifying critical latent feature regions influencing predictions
  - Backpropagates gradients from the scalar continuous authenticity prediction $\hat{y}$ to target feature layers
  - Output: Spatial attention heatmaps ($N \times H \times W$), normalized to $[0, 1]$

- **Multiscale Pixel Masking (MPM)**
  - Model-agnostic perturbation method evaluating prediction sensitivity under Gaussian pixel occlusion
  - Masks input regions at multiple spatial scales ($\sigma \in [3, 17, 65]$) with configurable stride and batching
  - Measures delta prediction changes: $|\hat{y}_{\text{orig}} - \hat{y}_{\text{masked}}|$
  - Output: Multi-scale importance heatmaps ($N \times H \times W$), normalized to $[0, 1]$

- **LIME (Local Interpretable Model-agnostic Explanations)** ✨ NEW
  - GPU-vectorized superpixel perturbation explainer tailored for continuous regression tasks
  - **Canonical Superpixel Tessellation**: Precomputes and caches a canonical $224 \times 224$ SLIC superpixel segmentation (`slic_segments.npy`, $N = 273$ test images, $K \approx 150$ segments, compactness = 10.0) with execution metadata in `tessellation_manifest.json`
  - **Vectorized Perturbation Sampling**: Generates $M = 1200$ binary superpixel retention masks via Bernoulli sampling ($p_{\text{keep}} = 0.7$, seed = 0, first sample unperturbed)
  - **Occlusion & Imputation**: Perturbed superpixels are imputed using per-image spatial-mean RGB background values
  - **Locality-Weighted Ridge Surrogate**: Fits a weighted Ridge linear regression surrogate ($\alpha = 1.0$) with exponential distance kernel weighting ($\sigma = 0.25$) on perturbed model outputs
  - **Multi-Faceted Outputs**:
    1. *2D Attribution Heatmaps* ($N \times H \times W$): Projected superpixel weights symmetrically normalized by $\max(|\text{heat}|) + 10^{-8}$ into $[-1.0, 1.0]$
    2. *Surrogate Fidelity ($R^2$)* ($N \times 1$): Per-image weighted $R^2$ scores tracking local surrogate goodness-of-fit (`.npy`)
    3. *1D Superpixel Beta Vectors* ($\beta \in \mathbb{R}^K$): Compressed per-image Ridge coefficient vectors stored in compressed archives (`.npz`)

**Model Support & Target Layers (Grad-CAM):**
- **VGG16**: Target layer `features.30` (final ReLU after conv5_3)
- **VGG19**: Target layer `features.36` (final ReLU after conv5_4)
- **ResNet152**: Target layer `features.7.2.conv3` (final residual block conv3)
- **DenseNet161**: Target layer `features.denseblock4.denselayer24.conv2` (final dense block conv); automatically handles $300 \times 300$ input resolution adaptation during LIME forward passes
- **EfficientNetB3**: Target layer `features.8.0` (final MBConv conv2d)
- **BarlowTwins**: Target layer `features.7.2.conv3` (final residual block conv3 before average pooling)
- **CLIP ViT-B/16 (Vision Transformer)**:
  - Backbone: `openai/clip-vit-base-patch16` with frozen weights and trained regression head
  - Target Layer for Grad-CAM: `vision_model.encoder.layers.11.layer_norm1`
  - Spatial Reshaping: Drops `[CLS]` token (index 0) from 197 tokens $\to$ reshapes 196 patch tokens to $(14, 14)$ grid $\to$ permutes to $(B, 768, 14, 14)$ $\to$ interpolates to target resolution $(224, 224)$
  - Attribution Flow: Backpropagated directly from scalar continuous prediction $\hat{y}$
  - LIME & MPM Support: Model-agnostic perturbation evaluated with custom CLIP normalization `mean=(0.48145466, 0.4578275, 0.40821073)`, `std=(0.26862954, 0.26130258, 0.27577711)` on canonical test images

**Variant Support:**
Each model can have multiple variants evaluated:
- `orig`: Original unpruned model
- `exp1a_variantN`: Unpruned baseline variants from Experiment 1A (variants 1–10)
- `exp1b_variantN_greedy_pruned`: Greedy-pruned variants from Experiment 1B
- `exp1b_variantN_negative_pruned`: Negative-impact pruned variants from Experiment 1B

---

#### 2. **Explainability Map & Coefficient Comparison (Experiment 2B)**

**Purpose:** Quantitatively compare explainability attributions across architectures, training runs, pruning conditions, and attribution methodologies.

**Comparison Paradigms:**

##### A. **Inter-Model Comparison (Between-Model Architectures / Prototype-Based)**
Creates a **prototype representation** for each architecture by averaging attributions across all its variants, then performs pairwise inter-architecture comparisons.

**1. 2D Spatial Heatmap Prototype Comparison (GradCAM, MPM, LIME):**
```
For each architecture M:
  1. Load variant heatmaps: V₁, V₂, ..., Vₙ (each shape: N_images × H × W)
  2. Resample / uniform to target resolution (224 × 224)
  3. Average across variants: Prototype_M = mean(V₁, V₂, ..., Vₙ)
  4. Result: One canonical prototype map per architecture (N_images × 224 × 224)

Compute pairwise similarity metrics (Pearson correlation, SSIM, IoU) between prototypes.
```

**2. 1D Superpixel Beta Prototype Comparison (`lime_superpixel_between_model_architectures`):**
```
For each architecture M:
  1. Load 1D superpixel beta vectors across variants: {img_key: β_v}
  2. Average beta vectors across variants: β_proto = mean(β₁, ..., βₙ)
  3. Result: One canonical superpixel vector per test image per architecture

Compute Pearson correlation, Spearman rank correlation, and top/bottom-K IoU between model prototypes.
```

**Key Advantages:**
- Filters out idiosyncratic initialization noise of individual runs
- Isolates fundamental inductive bias differences between architectures (e.g. CNN local receptive fields vs. ViT global self-attention)
- Directly connects 2D spatial alignment with 1D superpixel coefficient consistency on shared tessellations

##### B. **Intra-Model Comparison (Within-Model Variants)**
Evaluates attribution stability within the same architecture:
1. **Random-Seed Stability:** Pairwise comparison across 10 unpruned baseline variants (`exp1a_variant1` ... `exp1a_variant10`) to quantify attribution sensitivity to initialization and training stochasticity.
2. **Pruning Saliency Shift:** Comparing unpruned vs. greedy-pruned vs. negative-pruned models to assess feature reorganization under structured unit removal.
3. **1D Superpixel Beta Consistency (`lime_superpixel_within_model_variants`):** Direct vector similarity on LIME $\beta$ coefficients across variants, yielding Pearson, Spearman, and top/bottom-$K$ IoU metrics.
4. **Violin Distribution Visualizations:** Aggregates intra-model pairwise similarities into distribution plots per architecture to visualize attribution robustness.

##### C. **LIME Surrogate Fidelity Diagnostics ($R^2$)**
Evaluates whether the local linear surrogate accurately captures model behavior:
- Compiles weighted $R^2$ across all test images and variants per architecture
- Produces statistical summaries: mean, standard deviation, median, minimum, maximum, and total successful fits
- Generates publication-ready diagnostic plots:
  - Pooled $R^2$ histogram across all fits (`lime_r2_pooled_hist.png`)
  - Per-architecture small-multiples histograms (`lime_r2_by_arch_hist.png`)
  - Cross-architecture boxplots (`lime_r2_by_arch_boxplot.png`)

##### D. **Cross-Method Comparison (`cross_methods`)**
Compares attributions generated by different XAI paradigms on the same model variant:
- Pairwise comparisons: `gradcam_vs_mpm`, `gradcam_vs_lime`, `mpm_vs_lime`
- Highlights agreement and divergence between gradient-based attention, pixel-level occlusion, and superpixel surrogate attributions

---

## Key Functions

### 1. `prepare_or_load_slic_segments(n_segments, compactness) -> np.ndarray`

**Purpose:** Precomputes or loads the shared canonical $224 \times 224$ SLIC superpixel tessellation across the test dataset.

**Process:**
1. Checks for existing `slic_segments.npy` and `tessellation_manifest.json` in `Outputs/Experiment_2_variants/XAI_Maps/LIME/`
2. If absent, loads ImageNet test images, applies canonical `Resize(256) + CenterCrop(224)`, and executes `skimage.segmentation.slic`
3. Saves canonical segmentation store ($N_{\text{test}} \times 224 \times 224$, `int16`) and metadata manifest
4. Returns 3D segmentation tensor ensuring all models and variants operate on identical superpixel boundaries

### 2. `_create_prototype_heatmap(variant_paths: Dict[str, Path]) -> np.ndarray`

**Purpose:** Create an averaged prototype heatmap from multiple model variants.

**Process:**
1. Loads all variant heatmaps
2. Trims arrays to the common minimum image count
3. Stacks arrays: $(n_{\text{variants}}, n_{\text{images}}, H, W)$
4. Computes mean across axis 0: $\text{prototype} = \frac{1}{n} \sum_{i=1}^n V_i$
5. Returns shape: $(n_{\text{images}}, H, W)$

### 3. `compare_heatmaps(heatmap_arrays, metrics) -> ComparisonResults`

**Purpose:** Compute pairwise similarity metrics between multiple 2D heatmap collections.

**Available Metrics:**
- **Correlation (Pearson)**: Linear spatial correlation $[-1, 1]$
- **Cosine Similarity**: Angular similarity $[-1, 1]$
- **SSIM**: Structural similarity index $[-1, 1]$
- **Top-$K$ IoU**: Intersection-over-Union of top $K\%$ salient pixels (`top_percent_iou_5`, `top_percent_iou_15`, `top_percent_iou_25`)
- **MSE / EMD**: Mean squared error and Earth Mover's Distance

### 4. `compare_lime_betas(beta1, beta2, k_list) -> dict[str, float]` & `compare_lime_beta_collections(...)`

**Purpose:** Compare 1D superpixel Ridge coefficient vectors $\beta \in \mathbb{R}^K$ on shared tessellations.

**Metrics Computed:**
- **Pearson Correlation**: Linear coefficient alignment
- **Spearman Rank Correlation**: Monotonic ranking agreement of superpixel importance
- **Top-$K$ & Bottom-$K$ IoU**: Jaccard similarity of top-$K$ positive (authenticity-promoting) and bottom-$K$ negative (authenticity-demoting) superpixels for $K \in \{5, 15, 25\}$

### 5. `plot_lime_r2_distributions(r2_by_model, save_dir, dpi) -> None`

**Purpose:** Generate diagnostic visualizations assessing local surrogate fidelity across architectures.

**Output Figures:**
- `lime_r2_pooled_hist.png`: Overall distribution of weighted $R^2$ across all evaluated instances
- `lime_r2_by_arch_hist.png`: Faceted subplots comparing fidelity across model backbones
- `lime_r2_by_arch_boxplot.png`: Distribution percentiles and outliers by architecture

### 6. `visualize_similarity_matrix(results, model_names, metric, stat) -> Figure`

**Purpose:** Render an $N \times N$ matrix displaying pairwise similarity values across models or variants.

### 7. `visualize_similarity_distribution(results, metric) -> Figure`

**Purpose:** Analyze consensus versus controversy across images via a two-panel figure:
- **Panel 1 (Histogram of Agreement)**: Mean correlation across all model pairs per image
- **Panel 2 (Consensus vs. Controversy Scatter)**: Mean agreement versus standard deviation across images

### 8. `visualize_violin_distribution(model_data, metric, ...) -> Figure`

**Purpose:** Render violin plots displaying intra-model variant agreement distributions for each architecture.

---

## Workflow Execution

### Complete Pipeline (Python API)

```python
from main.Experiments.experiment_two import run_experiment_2

run_experiment_2(
    models=[
        "vgg16", "vgg19", "resnet152", "densenet161",
        "efficientnetb3", "barlowtwins", "clip_vit_b16"
    ],
    variants="unpruned",       # Target 10 unpruned baseline variants (exp1a_variant1...10)
    xai_methods="all",         # Supported: 'gradcam', 'mpm', 'lime', 'both' (GC+MPM), 'all' (GC+MPM+LIME)
    save_maps=True,            # Persist generated attribution arrays to disk
    comparison_only=False,     # False: generate maps + compare, True: skip generation
    run_comparison=True,       # Run comparative analysis
    comparison_kinds=[
        "within_model_variants",
        "between_model_architectures",
        "cross_methods",
    ],
    comparison_metrics=[
        "correlation",
        "ssim",
        "top_percent_iou_5",
        "top_percent_iou_15",
        "top_percent_iou_25",
    ],
    show_comparison_plots=False,
    save_plots=True,
    save_comparison_json=True,
    cluster_computing=False,   # Set True when executing on HPC / large GPU nodes
)
```

### Command Line Interface (CLI)

The experiment can be launched directly through the package CLI runner:

```bash
# Run full pipeline with all three XAI methods and comprehensive comparisons
uv run python -m Image_Authenticity_prediction experiment-two \
    --models clip_vit_b16 vgg16 resnet152 \
    --variants unpruned \
    --xai-methods all \
    --run-comparison \
    --comparison-kinds within_model_variants between_model_architectures cross_methods \
    --comparison-metrics correlation ssim top_percent_iou_5 top_percent_iou_15 top_percent_iou_25 \
    --save-plots

# Generate only LIME maps and superpixel statistics
uv run python -m Image_Authenticity_prediction experiment-two \
    --models clip_vit_b16 \
    --variants unpruned \
    --xai-methods lime \
    --save-maps

# Run comparison only on pre-existing maps
uv run python -m Image_Authenticity_prediction experiment-two \
    --xai-methods all \
    --comparison-only \
    --comparison-kinds within_model_variants between_model_architectures cross_methods \
    --save-plots
```

### HPC Cluster / SLURM Batch Execution

For large-scale map generation across 10 variants $\times$ 7 architectures $\times$ 3 methods, use the SLURM submission script (`scripts/slurm/run_experiment_two.slurm`):

```bash
# Submit automated two-step job (LIME map generation on NVMe /scratch -> full comparison)
sbatch scripts/slurm/run_experiment_two.slurm
```

The batch runner automatically stages inputs to fast node-local `/scratch`, activates CUDA Automatic Mixed Precision (AMP), scales batch sizes to 512, copies outputs back to persistent shared storage upon completion, and cleans temporary scratch directories via traps.

### SSH-Resilient Background Execution (`tmux` / `nohup`)

#### Recommended: Detached `tmux` Session
```bash
tmux new -s exp2

uv run python -m Image_Authenticity_prediction experiment-two \
    --variants unpruned \
    --xai-methods all \
    --run-comparison \
    --comparison-kinds within_model_variants between_model_architectures cross_methods \
    --save-plots

# Detach: Ctrl + b, then d
# Reattach: tmux attach -t exp2
```

#### Alternative: Background with `nohup`
```bash
nohup uv run python -m Image_Authenticity_prediction experiment-two \
    --variants unpruned \
    --xai-methods all \
    --run-comparison > experiment_2.log 2>&1 &

tail -f experiment_2.log
```

---

## Output Files

### Generated Attribution Files
```
Outputs/Experiment_2_variants/XAI_Maps/
├── GradCAM/
│   ├── vgg16_orig_maps.npy                            # (273, 224, 224) float32
│   ├── vgg16_exp1a_variant1_maps.npy
│   ├── clip_vit_b16_orig_maps.npy
│   └── ...
├── Multiscale_Pixel_Masking/
│   ├── vgg16_orig_maps.npy                            # (273, 224, 224) float32
│   ├── clip_vit_b16_orig_maps.npy
│   └── ...
└── LIME/
    ├── slic_segments.npy                              # Canonical test SLIC tessellation (273, 224, 224) int16
    ├── tessellation_manifest.json                     # Metadata on superpixel segmentation & image indices
    ├── vgg16_orig_maps.npy                            # 2D superpixel saliency heatmaps (273, 224, 224) float32
    ├── vgg16_orig_r2.npy                              # Weighted R² surrogate fidelity scores (273,) float32
    ├── vgg16_orig_betas.npz                           # 1D Ridge regression beta vectors (img_0000: K-dim vector)
    ├── clip_vit_b16_exp1a_variant1_maps.npy
    ├── clip_vit_b16_exp1a_variant1_r2.npy
    ├── clip_vit_b16_exp1a_variant1_betas.npz
    └── ...
```

### Visualizations and Diagnostic Plots
```
Outputs/Experiment_2_variants/Plots/
├── lime_r2_pooled_hist.png                            # Diagnostic: pooled R² histogram across all fits
├── lime_r2_by_arch_hist.png                          # Diagnostic: per-architecture R² subplots
├── lime_r2_by_arch_boxplot.png                       # Diagnostic: R² boxplots by architecture
├── lime_within_model_correlation_violin.png           # Intra-model variant stability violin plot
├── lime_within_model_correlation_violin.svg
├── inter_model_gradcam_correlation_matrix.png         # Pairwise prototype similarity matrices
├── inter_model_mpm_correlation_matrix.png
├── inter_model_lime_correlation_matrix.png
├── inter_model_lime_correlation_distribution.png      # Consensus vs controversy scatter/histogram
├── lime_superpixel_between_model_architectures_correlation_matrix.png
├── lime_superpixel_between_model_architectures_spearman_matrix.png
└── ...
```

### JSON Results Structure (`experiment_2b_comparison.json`)
```json
{
  "lime_r2_summary": {
    "vgg16": {
      "mean": 0.597,
      "std": 0.080,
      "median": 0.600,
      "min": 0.269,
      "max": 0.878,
      "n_variants": 10,
      "n_fits": 2730
    },
    "clip_vit_b16": {
      "mean": 0.607,
      "std": 0.097,
      "median": 0.608,
      "min": 0.297,
      "max": 0.871,
      "n_variants": 10,
      "n_fits": 2730
    }
  },
  "lime_within_model_variants": {
    "vgg16": {
      "per_image": { "correlation": { ... } },
      "summary": { "correlation": { ... } },
      "variants": ["exp1a_variant1", "exp1a_variant2", "..."]
    }
  },
  "lime_between_model_architectures": {
    "per_image": { "correlation": { ... } },
    "summary": { "correlation": { ... } },
    "models": ["vgg16", "vgg19", "resnet152", "densenet161", "efficientnetb3", "barlowtwins", "clip_vit_b16"]
  },
  "lime_superpixel_within_model_variants": {
    "vgg16": {
      "per_image": { "correlation": { ... }, "spearman": { ... }, "top_percent_iou_5": { ... } },
      "summary": { ... },
      "variants": ["exp1a_variant1", "..."]
    }
  },
  "lime_superpixel_between_model_architectures": {
    "summary": {
      "correlation": { ... },
      "spearman": { ... },
      "top_percent_iou_5": { ... }
    },
    "models": ["vgg16", "vgg19", "resnet152", "densenet161", "efficientnetb3", "barlowtwins", "clip_vit_b16"]
  },
  "cross_methods": {
    "gradcam_vs_mpm": { ... },
    "gradcam_vs_lime": { ... },
    "mpm_vs_lime": { ... }
  }
}
```

---

## Empirical Results (Experiment 2B Comparison Summary)

This section summarizes the empirical findings across all 7 visual architectures evaluated over 10 independently trained unpruned variants ($70$ models total) on the held-out $N = 273$ test images from AIGCIQA2023.

### 1. Within-Architecture Attribution Consistency ($N = 273$ Test Images, 10 Variants)

Evaluated across all $\binom{10}{2} = 45$ variant pairs on each of the $N = 273$ test images ($45 \times 273 = 12{,}285$ comparisons per architecture). Values report $\text{Mean} \pm \text{SD}$ across all evaluations (`std_all`), with image-level SD (`std_img`) and pair-level SD (`std_pair`) noted.

#### Table 1: Grad-CAM Within-Architecture Consistency

| Architecture | Pearson Correlation ($r$) | SSIM | Top-5% IoU | Top-15% IoU | Top-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`barlowtwins`** | **$0.9946 \pm 0.0072$** *(img: 0.0051, pair: 0.0014)* | **$0.9262 \pm 0.0600$** | **$0.7921 \pm 0.1313$** | **$0.8621 \pm 0.0844$** | **$0.8980 \pm 0.0662$** |
| **`resnet152`** | **$0.9897 \pm 0.0141$** *(img: 0.0096, pair: 0.0021)* | **$0.8568 \pm 0.1157$** | **$0.7742 \pm 0.1420$** | **$0.8585 \pm 0.0969$** | **$0.8920 \pm 0.0764$** |
| **`vgg16`** | **$0.9644 \pm 0.0327$** *(img: 0.0202, pair: 0.0049)* | **$0.7692 \pm 0.1287$** | **$0.6582 \pm 0.1845$** | **$0.7381 \pm 0.1256$** | **$0.7801 \pm 0.0992$** |
| **`vgg19`** | **$0.9626 \pm 0.0330$** *(img: 0.0189, pair: 0.0085)* | **$0.7646 \pm 0.1337$** | **$0.6699 \pm 0.1930$** | **$0.7331 \pm 0.1276$** | **$0.7736 \pm 0.1004$** |
| **`clip_vit_b16`** | **$0.9165 \pm 0.0707$** *(img: 0.0406, pair: 0.0359)* | **$0.6102 \pm 0.1548$** | **$0.5175 \pm 0.1504$** | **$0.6162 \pm 0.1276$** | **$0.6782 \pm 0.1156$** |
| **`densenet161`** | **$0.8804 \pm 0.1206$** *(img: 0.0782, pair: 0.0270)* | **$0.6077 \pm 0.2237$** | **$0.5195 \pm 0.2482$** | **$0.6037 \pm 0.1865$** | **$0.6589 \pm 0.1548$** |
| **`efficientnetb3`** | **$0.8668 \pm 0.1320$** *(img: 0.0907, pair: 0.0117)* | **$0.5779 \pm 0.2941$** | **$0.4670 \pm 0.2524$** | **$0.5577 \pm 0.1960$** | **$0.6205 \pm 0.1685$** |

#### Table 2: Multiscale Pixel Masking (MPM) Within-Architecture Consistency

| Architecture | Pearson Correlation ($r$) | SSIM | Top-5% IoU | Top-15% IoU | Top-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clip_vit_b16`** | **$0.9687 \pm 0.0303$** *(img: 0.0211, pair: 0.0113)* | **$0.7683 \pm 0.1398$** | **$0.6782 \pm 0.1385$** | **$0.7548 \pm 0.1073$** | **$0.7986 \pm 0.0899$** |
| **`barlowtwins`** | **$0.9010 \pm 0.0707$** *(img: 0.0477, pair: 0.0273)* | **$0.5848 \pm 0.1671$** | **$0.4758 \pm 0.1688$** | **$0.5871 \pm 0.1317$** | **$0.6504 \pm 0.1101$** |
| **`resnet152`** | **$0.8788 \pm 0.0792$** *(img: 0.0535, pair: 0.0159)* | **$0.5365 \pm 0.2126$** | **$0.4920 \pm 0.1741$** | **$0.5706 \pm 0.1276$** | **$0.6259 \pm 0.1053$** |
| **`densenet161`** | **$0.8784 \pm 0.0784$** *(img: 0.0520, pair: 0.0201)* | **$0.5392 \pm 0.1784$** | **$0.4460 \pm 0.1680$** | **$0.5476 \pm 0.1279$** | **$0.6112 \pm 0.1083$** |
| **`vgg19`** | **$0.8364 \pm 0.1027$** *(img: 0.0694, pair: 0.0201)* | **$0.4937 \pm 0.1736$** | **$0.4116 \pm 0.1976$** | **$0.5123 \pm 0.1459$** | **$0.5765 \pm 0.1206$** |
| **`vgg16`** | **$0.8355 \pm 0.0964$** *(img: 0.0655, pair: 0.0183)* | **$0.4908 \pm 0.1686$** | **$0.3989 \pm 0.1929$** | **$0.4933 \pm 0.1393$** | **$0.5646 \pm 0.1182$** |
| **`efficientnetb3`** | **$0.7905 \pm 0.1224$** *(img: 0.0805, pair: 0.0193)* | **$0.4052 \pm 0.1973$** | **$0.3728 \pm 0.1751$** | **$0.4695 \pm 0.1395$** | **$0.5286 \pm 0.1172$** |

#### Table 3: LIME Within-Architecture Consistency — 2D Pixel Space ($224 \times 224$)

| Architecture | Pearson Correlation ($r$) | SSIM | Top-5% IoU | Top-15% IoU | Top-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clip_vit_b16`** | **$0.9588 \pm 0.0346$** *(img: 0.0230, pair: 0.0154)* | **$0.8016 \pm 0.0935$** | **$0.7112 \pm 0.2459$** | **$0.7562 \pm 0.1331$** | **$0.7833 \pm 0.1026$** |
| **`barlowtwins`** | **$0.9189 \pm 0.0578$** *(img: 0.0366, pair: 0.0159)* | **$0.6588 \pm 0.0948$** | **$0.5871 \pm 0.2513$** | **$0.6404 \pm 0.1418$** | **$0.6843 \pm 0.1030$** |
| **`resnet152`** | **$0.8865 \pm 0.0675$** *(img: 0.0479, pair: 0.0146)* | **$0.5832 \pm 0.1001$** | **$0.5674 \pm 0.2539$** | **$0.5946 \pm 0.1380$** | **$0.6310 \pm 0.1036$** |
| **`densenet161`** | **$0.8669 \pm 0.0769$** *(img: 0.0471, pair: 0.0156)* | **$0.5489 \pm 0.0870$** | **$0.4880 \pm 0.2353$** | **$0.5509 \pm 0.1321$** | **$0.6076 \pm 0.0990$** |
| **`vgg16`** | **$0.8465 \pm 0.0961$** *(img: 0.0621, pair: 0.0152)* | **$0.5695 \pm 0.0858$** | **$0.4428 \pm 0.2274$** | **$0.5265 \pm 0.1280$** | **$0.5857 \pm 0.0968$** |
| **`vgg19`** | **$0.8350 \pm 0.0869$** *(img: 0.0552, pair: 0.0142)* | **$0.5297 \pm 0.0843$** | **$0.4563 \pm 0.2331$** | **$0.5261 \pm 0.1382$** | **$0.5806 \pm 0.1001$** |
| **`efficientnetb3`** | **$0.8037 \pm 0.1065$** *(img: 0.0726, pair: 0.0181)* | **$0.4641 \pm 0.1022$** | **$0.4249 \pm 0.2365$** | **$0.4797 \pm 0.1398$** | **$0.5370 \pm 0.1112$** |

#### Table 4: LIME 1D Superpixel-Space Coefficient Vector Consistency

| Architecture | Pearson ($r$) | Spearman ($\rho$) | Top-5% IoU | Top-15% IoU | Top-25% IoU | Bottom-5% IoU | Bottom-15% IoU | Bottom-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clip_vit_b16`** | $0.9586 \pm 0.0317$ | $0.9458 \pm 0.0363$ | $0.6886 \pm 0.2145$ | $0.7399 \pm 0.1232$ | $0.7695 \pm 0.0993$ | $0.6053 \pm 0.2136$ | $0.6798 \pm 0.1296$ | $0.7256 \pm 0.1056$ |
| **`barlowtwins`** | $0.9214 \pm 0.0394$ | $0.9081 \pm 0.0399$ | $0.5605 \pm 0.2102$ | $0.6309 \pm 0.1266$ | $0.6688 \pm 0.0974$ | $0.5453 \pm 0.2123$ | $0.6117 \pm 0.1240$ | $0.6637 \pm 0.0996$ |
| **`resnet152`** | $0.8856 \pm 0.0600$ | $0.8584 \pm 0.0587$ | $0.5329 \pm 0.2209$ | $0.5713 \pm 0.1285$ | $0.6106 \pm 0.1039$ | $0.4910 \pm 0.2111$ | $0.5452 \pm 0.1337$ | $0.5955 \pm 0.1048$ |
| **`densenet161`** | $0.8702 \pm 0.0561$ | $0.8518 \pm 0.0516$ | $0.4726 \pm 0.2074$ | $0.5407 \pm 0.1208$ | $0.5943 \pm 0.0971$ | $0.4762 \pm 0.2071$ | $0.5461 \pm 0.1194$ | $0.5937 \pm 0.0961$ |
| **`vgg16`** | $0.8509 \pm 0.0735$ | $0.8380 \pm 0.0528$ | $0.4292 \pm 0.1999$ | $0.5126 \pm 0.1167$ | $0.5726 \pm 0.0939$ | $0.4684 \pm 0.2003$ | $0.5300 \pm 0.1162$ | $0.5865 \pm 0.0939$ |
| **`vgg19`** | $0.8409 \pm 0.0674$ | $0.8255 \pm 0.0583$ | $0.4432 \pm 0.2029$ | $0.5144 \pm 0.1239$ | $0.5675 \pm 0.0978$ | $0.4398 \pm 0.1995$ | $0.5170 \pm 0.1193$ | $0.5709 \pm 0.0961$ |
| **`efficientnetb3`** | $0.8030 \pm 0.0926$ | $0.7744 \pm 0.0869$ | $0.4065 \pm 0.2061$ | $0.4611 \pm 0.1277$ | $0.5164 \pm 0.1062$ | $0.4099 \pm 0.2030$ | $0.4653 \pm 0.1313$ | $0.5186 \pm 0.1071$ |

#### Table 5: LIME Local Surrogate Fidelity ($R^2$) Summary

Evaluated across $10\text{ variants} \times 273\text{ test images} = 2{,}730$ local Ridge surrogate fits per architecture ($19{,}110$ fits total):

| Architecture | Mean $R^2$ | SD ($R^2$) | Median $R^2$ | Min $R^2$ | Max $R^2$ | Total Fits ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clip_vit_b16`** | **$0.6068$** | $0.0968$ | $0.6082$ | $0.2970$ | $0.8710$ | $2{,}730$ |
| **`vgg16`** | **$0.5972$** | $0.0803$ | $0.6002$ | $0.2689$ | $0.8782$ | $2{,}730$ |
| **`vgg19`** | **$0.5634$** | $0.0821$ | $0.5648$ | $0.3045$ | $0.8363$ | $2{,}730$ |
| **`barlowtwins`** | **$0.5313$** | $0.0887$ | $0.5298$ | $0.2357$ | $0.8184$ | $2{,}730$ |
| **`densenet161`** | **$0.4786$** | $0.0792$ | $0.4760$ | $0.2724$ | $0.7683$ | $2{,}730$ |
| **`resnet152`** | **$0.4469$** | $0.0853$ | $0.4416$ | $0.2173$ | $0.7879$ | $2{,}730$ |
| **`efficientnetb3`** | **$0.4342$** | $0.0759$ | $0.4318$ | $0.1978$ | $0.7565$ | $2{,}730$ |

---

### 2. Across-Architecture Prototype Consistency (`between_model_architectures`)

Prototypes are computed by averaging the 10 variant attribution maps per test image. Pairwise similarity is evaluated across all $\binom{7}{2} = 21$ architecture pairs ($\binom{6}{2} = 15$ pairs among CNNs).

#### Table 6: Summary of Across-Architecture Prototype Agreement ($N = 273$ Test Images)

| Method | Scope | Pearson Correlation ($r$) | SSIM | Top-5% IoU | Top-15% IoU | Top-25% IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Grad-CAM** | All 7 Architectures (21 pairs) | **$0.1319 \pm 0.3538$** | $0.0439 \pm 0.2054$ | $0.0547 \pm 0.1051$ | $0.1341 \pm 0.1356$ | $0.2069 \pm 0.1482$ |
| **Grad-CAM** | 6 CNNs Only (15 pairs) | **$0.2304 \pm 0.3414$** | $0.0727 \pm 0.2273$ | $0.0680 \pm 0.1189$ | $0.1616 \pm 0.1471$ | $0.2419 \pm 0.1553$ |
| **MPM** | All 7 Architectures (21 pairs) | **$0.0462 \pm 0.2718$** | $0.0236 \pm 0.1247$ | $0.0451 \pm 0.0744$ | $0.1077 \pm 0.0901$ | $0.1697 \pm 0.0966$ |
| **MPM** | 6 CNNs Only (15 pairs) | **$0.0688 \pm 0.2670$** | $0.0307 \pm 0.1295$ | $0.0453 \pm 0.0766$ | $0.1083 \pm 0.0892$ | $0.1719 \pm 0.0952$ |
| **LIME (2D)** | All 7 Architectures (21 pairs) | **$0.0533 \pm 0.2288$** | $0.0576 \pm 0.0785$ | $0.0923 \pm 0.1507$ | $0.1308 \pm 0.0940$ | $0.1786 \pm 0.0756$ |
| **LIME (2D)** | 6 CNNs Only (15 pairs) | **$0.0690 \pm 0.2266$** | $0.0600 \pm 0.0752$ | $0.0885 \pm 0.1483$ | $0.1259 \pm 0.0902$ | $0.1764 \pm 0.0745$ |
| **LIME (1D Superpixel)** | All 7 Architectures (21 pairs) | **$0.0516 \pm 0.1880$** | — *(SRCC: $0.0378$)* | $0.0741 \pm 0.1007$ | $0.1145 \pm 0.0748$ | $0.1577 \pm 0.0682$ |
| **LIME (1D Superpixel)** | 6 CNNs Only (15 pairs) | **$0.0658 \pm 0.1878$** | — *(SRCC: $0.0513$)* | $0.0724 \pm 0.1017$ | $0.1109 \pm 0.0732$ | $0.1556 \pm 0.0689$ |

#### Table 7: Grad-CAM Prototype Correlation Matrix ($7 \times 7$)

| Architecture | `barlowtwins` | `resnet152` | `densenet161` | `efficientnetb3` | `vgg16` | `vgg19` | `clip_vit_b16` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`barlowtwins`** | $1.0000$ | **$0.5566$** | $0.2648$ | $0.3204$ | $0.1095$ | $0.0930$ | $-0.1707$ |
| **`resnet152`** | **$0.5566$** | $1.0000$ | $0.3167$ | $0.3472$ | $0.1122$ | $0.0823$ | $-0.1698$ |
| **`densenet161`** | $0.2648$ | $0.3167$ | $1.0000$ | $0.1751$ | $0.1179$ | $0.1298$ | $-0.1312$ |
| **`efficientnetb3`** | $0.3204$ | $0.3472$ | $0.1751$ | $1.0000$ | $0.1232$ | $0.1025$ | $-0.1099$ |
| **`vgg16`** | $0.1095$ | $0.1122$ | $0.1179$ | $0.1232$ | $1.0000$ | **$0.6050$** | $-0.0624$ |
| **`vgg19`** | $0.0930$ | $0.0823$ | $0.1298$ | $0.1025$ | **$0.6050$** | $1.0000$ | $-0.0426$ |
| **`clip_vit_b16`** | $-0.1707$ | $-0.1698$ | $-0.1312$ | $-0.1099$ | $-0.0624$ | $-0.0426$ | $1.0000$ |

#### Table 8: MPM Prototype Correlation Matrix ($7 \times 7$)

| Architecture | `barlowtwins` | `resnet152` | `densenet161` | `efficientnetb3` | `vgg16` | `vgg19` | `clip_vit_b16` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`barlowtwins`** | $1.0000$ | $0.0282$ | $0.0995$ | $0.0128$ | $0.0212$ | $0.0929$ | $-0.0435$ |
| **`resnet152`** | $0.0282$ | $1.0000$ | $0.0178$ | $0.0496$ | $0.0006$ | $0.0141$ | $0.0402$ |
| **`densenet161`** | $0.0995$ | $0.0178$ | $1.0000$ | $0.0092$ | $0.1137$ | $0.1254$ | $-0.0871$ |
| **`efficientnetb3`** | $0.0128$ | $0.0496$ | $0.0092$ | $1.0000$ | $0.0085$ | $0.0127$ | $0.0514$ |
| **`vgg16`** | $0.0212$ | $0.0006$ | $0.1137$ | $0.0085$ | $1.0000$ | **$0.4264$** | $0.0067$ |
| **`vgg19`** | $0.0929$ | $0.0141$ | $0.1254$ | $0.0127$ | **$0.4264$** | $1.0000$ | $-0.0296$ |
| **`clip_vit_b16`** | $-0.0435$ | $0.0402$ | $-0.0871$ | $0.0514$ | $0.0067$ | $-0.0296$ | $1.0000$ |

#### Table 9: LIME 2D Prototype Correlation Matrix ($7 \times 7$)

| Architecture | `barlowtwins` | `resnet152` | `densenet161` | `efficientnetb3` | `vgg16` | `vgg19` | `clip_vit_b16` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`barlowtwins`** | $1.0000$ | $0.0712$ | $0.0841$ | $0.0488$ | $0.0422$ | $0.0609$ | $0.0782$ |
| **`resnet152`** | $0.0712$ | $1.0000$ | $0.0386$ | $0.0572$ | $-0.0087$ | $0.0194$ | $0.0579$ |
| **`densenet161`** | $0.0841$ | $0.0386$ | $1.0000$ | $0.0379$ | $0.0557$ | $0.0678$ | $-0.0164$ |
| **`efficientnetb3`** | $0.0488$ | $0.0572$ | $0.0379$ | $1.0000$ | $0.0328$ | $0.0220$ | $0.0427$ |
| **`vgg16`** | $0.0422$ | $-0.0087$ | $0.0557$ | $0.0328$ | $1.0000$ | **$0.4044$** | $-0.0531$ |
| **`vgg19`** | $0.0609$ | $0.0194$ | $0.0678$ | $0.0220$ | **$0.4044$** | $1.0000$ | $-0.0237$ |
| **`clip_vit_b16`** | $0.0782$ | $0.0579$ | $-0.0164$ | $0.0427$ | $-0.0531$ | $-0.0237$ | $1.0000$ |

---

### 3. Cross-Method Attribution Agreement Within Identical Variants (`cross_methods_pairs`)

Evaluated within the exact same model checkpoint across all 10 variants and 273 test images ($2{,}730$ comparisons per architecture per method pair):

#### Table 10: Within-Variant Cross-Method Attribution Alignment

| Architecture | Grad-CAM vs. MPM ($r$) | Grad-CAM vs. MPM (IoU@15%) | Grad-CAM vs. LIME ($r$) | Grad-CAM vs. LIME (IoU@15%) | MPM vs. LIME ($r$) | MPM vs. LIME (IoU@15%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vgg19`** | **$+0.3433 \pm 0.2438$** | $0.2218 \pm 0.1296$ | **$+0.1477 \pm 0.1676$** | $0.1527 \pm 0.0825$ | **$+0.2759 \pm 0.1571$** | $0.1900 \pm 0.0861$ |
| **`vgg16`** | **$+0.3201 \pm 0.2589$** | $0.2020 \pm 0.1243$ | **$+0.1433 \pm 0.1722$** | $0.1455 \pm 0.0725$ | **$+0.2610 \pm 0.1696$** | $0.1877 \pm 0.0854$ |
| **`efficientnetb3`** | $+0.1607 \pm 0.2448$ | $0.1564 \pm 0.1071$ | $+0.0813 \pm 0.1544$ | $0.1245 \pm 0.0758$ | $+0.1967 \pm 0.1354$ | $0.1457 \pm 0.0654$ |
| **`densenet161`** | $+0.0433 \pm 0.3047$ | $0.1297 \pm 0.1116$ | $+0.0277 \pm 0.1825$ | $0.1146 \pm 0.0709$ | $+0.1773 \pm 0.1677$ | $0.1410 \pm 0.0699$ |
| **`resnet152`** | $+0.0182 \pm 0.2811$ | $0.1358 \pm 0.1181$ | $+0.0282 \pm 0.1797$ | $0.1238 \pm 0.0806$ | $+0.2530 \pm 0.1366$ | $0.1689 \pm 0.0720$ |
| **`barlowtwins`** | $-0.0232 \pm 0.3420$ | $0.1193 \pm 0.1230$ | $-0.0144 \pm 0.2127$ | $0.1096 \pm 0.0813$ | $+0.2422 \pm 0.1838$ | $0.1583 \pm 0.0785$ |
| **`clip_vit_b16`** | $-0.0866 \pm 0.2421$ | $0.0670 \pm 0.0592$ | $-0.0116 \pm 0.1893$ | $0.0848 \pm 0.0587$ | $+0.1342 \pm 0.1884$ | $0.1393 \pm 0.0881$ |

---

### 4. Attribution Consistency vs. Human Authenticity Ratings & Model Prediction Error

#### Table 11: Correlation Between Per-Image Consistency and Ground-Truth Human Authenticity (MOS)

Pearson correlation $r$ ($p$-value) across the $N = 273$ held-out test stimuli:

| Architecture | Grad-CAM ($r$) | Grad-CAM (IoU@15%) | MPM ($r$) | MPM (IoU@15%) | LIME ($r$) | LIME (IoU@15%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vgg16`** | $-0.0341$ ($p=0.57$) | $-0.0452$ ($p=0.46$) | $+0.0456$ ($p=0.45$) | $+0.0390$ ($p=0.52$) | $+0.1161$ ($p=0.06$) | $+0.0005$ ($p=0.99$) |
| **`vgg19`** | $+0.0301$ ($p=0.62$) | $+0.0078$ ($p=0.90$) | $+0.0728$ ($p=0.23$) | $+0.0899$ ($p=0.14$) | $+0.0720$ ($p=0.24$) | $-0.0568$ ($p=0.35$) |
| **`resnet152`** | $-0.0469$ ($p=0.44$) | $-0.0080$ ($p=0.90$) | $-0.0041$ ($p=0.95$) | $-0.0268$ ($p=0.66$) | $+0.1156$ ($p=0.06$) | $+0.0464$ ($p=0.45$) |
| **`densenet161`** | $-0.0102$ ($p=0.87$) | $+0.0299$ ($p=0.62$) | $+0.0495$ ($p=0.42$) | $+0.0402$ ($p=0.51$) | $+0.0135$ ($p=0.82$) | $-0.0461$ ($p=0.45$) |
| **`efficientnetb3`** | $-0.0650$ ($p=0.28$) | $-0.0492$ ($p=0.42$) | $+0.1382$ ($p=0.02$) | $+0.1021$ ($p=0.09$) | $+0.0587$ ($p=0.33$) | $+0.0270$ ($p=0.66$) |
| **`barlowtwins`** | $-0.1044$ ($p=0.09$) | $-0.0279$ ($p=0.65$) | $+0.0228$ ($p=0.71$) | $+0.0028$ ($p=0.96$) | $+0.0065$ ($p=0.92$) | $+0.0763$ ($p=0.21$) |
| **`clip_vit_b16`** | $-0.0935$ ($p=0.12$) | $-0.0926$ ($p=0.13$) | $+0.0312$ ($p=0.61$) | $-0.0284$ ($p=0.64$) | $-0.0249$ ($p=0.68$) | $-0.1056$ ($p=0.08$) |

#### Table 12: Correlation Between Per-Image Consistency and Mean Absolute Prediction Error (MAE)

Pearson correlation $r$ ($p$-value) across $N = 273$ stimuli against $\text{MAE}_i = \frac{1}{10}\sum_{k=1}^{10} |\hat{y}_i^{(k)} - y_i|$:

| Architecture | Grad-CAM ($r$) | Grad-CAM (IoU@15%) | MPM ($r$) | MPM (IoU@15%) | LIME ($r$) | LIME (IoU@15%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vgg16`** | $-0.0199$ ($p=0.74$) | $+0.0813$ ($p=0.18$) | $+0.0200$ ($p=0.74$) | $+0.0003$ ($p=1.00$) | $-0.0053$ ($p=0.93$) | $-0.0679$ ($p=0.26$) |
| **`vgg19`** | $+0.0327$ ($p=0.59$) | $-0.0058$ ($p=0.92$) | $-0.0285$ ($p=0.64$) | $-0.0261$ ($p=0.67$) | $+0.0557$ ($p=0.36$) | $-0.0035$ ($p=0.95$) |
| **`resnet152`** | $-0.0993$ ($p=0.10$) | $-0.1017$ ($p=0.09$) | $-0.0298$ ($p=0.62$) | $-0.0376$ ($p=0.54$) | $-0.0752$ ($p=0.22$) | $-0.0465$ ($p=0.44$) |
| **`densenet161`** | $-0.0631$ ($p=0.30$) | $-0.0708$ ($p=0.24$) | $+0.0895$ ($p=0.14$) | $+0.0460$ ($p=0.45$) | $+0.0052$ ($p=0.93$) | $+0.0797$ ($p=0.19$) |
| **`efficientnetb3`** | $+0.0302$ ($p=0.62$) | $+0.0128$ ($p=0.83$) | $-0.0515$ ($p=0.40$) | $-0.0108$ ($p=0.86$) | $-0.0453$ ($p=0.46$) | $-0.1210$ ($p=0.05$) |
| **`barlowtwins`** | $-0.0628$ ($p=0.30$) | $-0.0400$ ($p=0.51$) | $-0.0032$ ($p=0.96$) | $+0.0097$ ($p=0.87$) | $-0.0600$ ($p=0.32$) | $-0.0002$ ($p=1.00$) |
| **`clip_vit_b16`** | $-0.1502$ ($p=0.01$) | $-0.0700$ ($p=0.25$) | $+0.0222$ ($p=0.72$) | $+0.0168$ ($p=0.78$) | $+0.0905$ ($p=0.14$) | $+0.0663$ ($p=0.28$) |

---

### 5. Across-Architecture Stimulus Agreement Rankings

Evaluated across all 7 architectural prototypes on the 273 test images:

#### Table 13: Top and Bottom Consensus Stimuli Across Architectural Prototypes

| Method | Mean Stimulus Agreement $\pm$ SD | Highest-Agreement Stimulus (Rank 1) | Lowest-Agreement Stimulus (Rank 273) | Ranking CSV Path |
| :--- | :---: | :--- | :--- | :--- |
| **Grad-CAM** | $0.1319 \pm 0.1055$ *(range: $[-0.0908, +0.5450]$)* | `42.png` (`test_idx=103`, *"A giraffe walking through a green grass covered field"*, $r = 0.5450$) | `2016.png` (`test_idx=203`, *"a flower"*, $r = -0.0908$) | `Outputs/Experiment_2_variants/Plots/agreement_figures/GradCAM/gradcam_across_architecture_agreement_ranking.csv` |
| **MPM** | $0.0462 \pm 0.0740$ *(range: $[-0.0870, +0.3523]$)* | `42.png` (`test_idx=103`, *"A giraffe walking through a green grass covered field"*, $r = 0.3523$) | `499.png` (`test_idx=56`, *"a tree reflected in the sunroof of a blue car"*, $r = -0.0870$) | `Outputs/Experiment_2_variants/Plots/agreement_figures/MPM/mpm_across_architecture_agreement_ranking.csv` |
| **LIME** | $0.0533 \pm 0.0848$ *(range: $[-0.0936, +0.7853]$)* | `41.png` (`test_idx=213`, *"A giraffe walking through a green grass covered field"*, $r = 0.7853$) | `524.png` (`test_idx=17`, *"a library filled with kids reading books"*, $r = -0.0936$) | `Outputs/Experiment_2_variants/Plots/agreement_figures/LIME/lime_across_architecture_agreement_ranking.csv` |

---

## Performance Considerations

### Computational Complexity

1. **GradCAM**: $\mathcal{O}(N \times C)$ forward and backward passes. Fast ($\sim 2$–$4$ seconds per 273 images on modern GPUs).
2. **Multiscale Pixel Masking (MPM)**: $\mathcal{O}(N \times P \times S)$ forward passes where $P$ is pixel count and $S = 3$ scales. Handled with batched spatial masking ($512$ masks per forward batch).
3. **LIME**: $\mathcal{O}(N \times M)$ forward passes where $M = 1200$ perturbations per image.
   - **Optimization**: Vectorized GPU forward evaluation with batch size $64$ (local) or $512$ (cluster) using PyTorch CUDA AMP (`bfloat16`).
   - **Shared Tessellation**: Precomputing canonical SLIC superpixels once across all models eliminates repetitive $\mathcal{O}(N \times \text{pixels})$ segmentation overhead.
   - **Ridge Fitting**: Vectorized normal equations in NumPy/SciPy execute in milliseconds per image.

### Storage & Memory Overhead

- **2D Heatmaps**: $273 \times 224 \times 224 \times 4\text{ bytes} \approx 54.8\text{ MB}$ per (model, variant, method) run.
- **LIME 1D Betas**: Compressed `.npz` archive containing 273 arrays of dimension $K \approx 150 \approx 240\text{ KB}$ per variant.
- **LIME Surrogate $R^2$**: $273 \times 4\text{ bytes} \approx 1{,}092\text{ bytes}$ per variant.
- **GPU VRAM**: Explicit `cleanup_model_and_data()` and `explainer.cleanup()` calls ensure intermediate tensors and activation caches are freed after each variant, preventing out-of-memory errors.

---

## Conclusion

Experiment 2 provides a unified framework for explainability analysis across modern deep authenticity architectures. By integrating **GradCAM**, **Multiscale Pixel Masking**, and **GPU-vectorized LIME**, the pipeline spans:
- White-box gradient spatial localization (GradCAM)
- Fine-grained multi-scale pixel sensitivity (MPM)
- Interpretable, model-agnostic superpixel surrogate modeling with statistical fidelity verification (LIME)

The inclusion of prototype-based inter-model comparisons, 1D superpixel beta vector metrics, and cross-method alignment allows rigorous, reproducible evaluation of explainability stability across model scales and architectures.
