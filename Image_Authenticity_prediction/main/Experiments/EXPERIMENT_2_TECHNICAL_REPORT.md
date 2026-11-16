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
  - **Canonical Superpixel Tessellation**: Precomputes and caches a canonical $224 \times 224$ SLIC superpixel segmentation (`slic_segments.npy`, $N = 138$ test images, $K \approx 150$ segments, compactness = 10.0) with execution metadata in `tessellation_manifest.json`
  - **Vectorized Perturbation Sampling**: Generates $M = 1200$ binary superpixel retention masks via Bernoulli sampling ($p_{\text{keep}} = 0.7$, seed = 0, first sample unperturbed)
  - **Occlusion & Imputation**: Perturbed superpixels are imputed using per-image spatial-mean RGB background values
  - **Locality-Weighted Ridge Surrogate**: Fits a weighted Ridge linear regression surrogate ($\alpha = 1.0$) with exponential distance kernel weighting ($\sigma = 0.25$) on perturbed model outputs
  - **Multi-Faceted Outputs**:
    1. *2D Attribution Heatmaps* ($N \times H \times W$): Projected superpixel weights symmetrically normalized by $\max(|\text{heat}|) + 10^{-8}$ into $[-1.0, 1.0]$
    2. *Surrogate Fidelity ($R^2$)* ($N \times 1$): Per-image weighted $R^2$ scores tracking local surrogate goodness-of-fit (`.npy`)
    3. *1D Superpixel Beta Vectors* ($\beta \in \mathbb{R}^K$): Compressed per-image Ridge coefficient vectors stored in compressed archives (`.npz`)

**Model Support:**
- **VGG16, VGG19**: Target layer `features.28` (or final convolutional block)
- **ResNet152**: Target layer `layer4.2`
- **DenseNet161**: Target layer `features.denseblock4`; automatically handles $300 \times 300$ input resolution adaptation during LIME forward passes
- **EfficientNetB3**: Target layer `features.7`
- **BarlowTwins**: Target layer `network.conv5`
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
│   ├── vgg16_orig_maps.npy                            # (138, 224, 224) float32
│   ├── vgg16_exp1a_variant1_maps.npy
│   ├── clip_vit_b16_orig_maps.npy
│   └── ...
├── Multiscale_Pixel_Masking/
│   ├── vgg16_orig_maps.npy                            # (138, 224, 224) float32
│   ├── clip_vit_b16_orig_maps.npy
│   └── ...
└── LIME/
    ├── slic_segments.npy                              # Canonical test SLIC tessellation (138, 224, 224) int16
    ├── tessellation_manifest.json                     # Metadata on superpixel segmentation & image indices
    ├── vgg16_orig_maps.npy                            # 2D superpixel saliency heatmaps (138, 224, 224) float32
    ├── vgg16_orig_r2.npy                              # Weighted R² surrogate fidelity scores (138,) float32
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
      "mean": 0.812,
      "std": 0.054,
      "median": 0.825,
      "min": 0.583,
      "max": 0.941,
      "n_variants": 10,
      "n_fits": 1380
    },
    "clip_vit_b16": {
      "mean": 0.795,
      "std": 0.061,
      "median": 0.804,
      "min": 0.521,
      "max": 0.932,
      "n_variants": 10,
      "n_fits": 1380
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

## Performance Considerations

### Computational Complexity

1. **GradCAM**: $\mathcal{O}(N \times C)$ forward and backward passes. Fast ($\sim 1$–$2$ seconds per 138 images on modern GPUs).
2. **Multiscale Pixel Masking (MPM)**: $\mathcal{O}(N \times P \times S)$ forward passes where $P$ is pixel count and $S = 3$ scales. Handled with batched spatial masking ($512$ masks per forward batch).
3. **LIME**: $\mathcal{O}(N \times M)$ forward passes where $M = 1200$ perturbations per image.
   - **Optimization**: Vectorized GPU forward evaluation with batch size $64$ (local) or $512$ (cluster) using PyTorch CUDA AMP (`bfloat16`).
   - **Shared Tessellation**: Precomputing canonical SLIC superpixels once across all models eliminates repetitive $\mathcal{O}(N \times \text{pixels})$ segmentation overhead.
   - **Ridge Fitting**: Vectorized normal equations in NumPy/SciPy execute in milliseconds per image.

### Storage & Memory Overhead

- **2D Heatmaps**: $138 \times 224 \times 224 \times 4\text{ bytes} \approx 27.7\text{ MB}$ per (model, variant, method) run.
- **LIME 1D Betas**: Compressed `.npz` archive containing 138 arrays of dimension $K \approx 150 \approx 120\text{ KB}$ per variant.
- **LIME Surrogate $R^2$**: $138 \times 4\text{ bytes} \approx 552\text{ bytes}$ per variant.
- **GPU VRAM**: Explicit `cleanup_model_and_data()` and `explainer.cleanup()` calls ensure intermediate tensors and activation caches are freed after each variant, preventing out-of-memory errors.

---

## Conclusion

Experiment 2 provides a unified framework for explainability analysis across modern deep authenticity architectures. By integrating **GradCAM**, **Multiscale Pixel Masking**, and **GPU-vectorized LIME**, the pipeline spans:
- White-box gradient spatial localization (GradCAM)
- Fine-grained multi-scale pixel sensitivity (MPM)
- Interpretable, model-agnostic superpixel surrogate modeling with statistical fidelity verification (LIME)

The inclusion of prototype-based inter-model comparisons, 1D superpixel beta vector metrics, and cross-method alignment allows rigorous, reproducible evaluation of explainability stability across model scales and architectures.
