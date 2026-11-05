# Experiment 3: Variant-Aligned 7-Model Ensembles and MPM Attribution Report

**Date:** November 5, 2026  
**Status:** Completed (Implements [ADR 0005](../adr/0005-experiment-3-variant-aligned-ensembles-and-xai.md))  
**Resolves:** GitHub Issues #13, #9, #17  
**Execution Environment:** DIPSEN HPC (`c31`, NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB VRAM, SLURM Job `10853`)  

---

## 1. Executive Summary

This report documents the final results of **Experiment 3**, redesigning the multi-architecture ensemble and explainability pipeline to evaluate 10 variant-aligned ensembles across 7 distinct visual architectures:
1. `vgg16` (Supervised CNN)
2. `vgg19` (Supervised CNN)
3. `resnet152` (Supervised Residual CNN)
4. `densenet161` (Supervised Densely Connected CNN)
5. `efficientnetb3` (Supervised Compound Scaled CNN)
6. `barlowtwins` (Self-Supervised Contrastive ResNet-50)
7. `clip_vit_b16` (Multimodal Vision-Language Transformer)

### 1.1 Key Achievements & Empirical Findings

- **Direct Reuse of Unpruned Checkpoints:** Reused all 70 unpruned Experiment 1A checkpoints (`Outputs/Experiment_1_variants/Weights/*_exp1a_variant{1..10}_best.pth`), completely avoiding lossy channel pruning and preserving peak feature representations.
- **Variant Alignment ($k = 1..10$):** Each ensemble variant $k$ shares an identical training partition (959 images) and a strictly disjoint validation partition $val_k$ (136 images, `seed = 42 + k`), eliminating data leakage.
- **Predictive Superiority on Held-Out Test Set (273 images):**
  - **Stacking Meta-Learner (Closed-Form Ridge $\alpha = 1.0$):** Achieves the highest performance in the entire research project:
    - **$\text{PLCC} = 0.7742 \pm 0.0102$** ($\mathbf{+0.0425}$ over best single model `clip_vit_b16` at $0.7317$, $\mathbf{+0.0223}$ over Bagging)
    - **$\text{SRCC} = 0.7734 \pm 0.0090$** ($\mathbf{+0.0278}$ over `clip_vit_b16` at $0.7456$, $\mathbf{+0.0250}$ over Bagging)
    - **$\text{RMSE} = 5.5442 \pm 0.1049$** (lowest error across all experiments)
    - **$\text{MSE} = 30.7487 \pm 1.1755$**
    - **Prediction Reliability ($r_{\text{model}}$):** $0.9511$
    - **Combined Correlation Ceiling:** $0.7983$ (approaching the human split-half ceiling of $0.8185$)
  - **Bagging (Uniform Mean):**
    - **$\text{PLCC} = 0.7519 \pm 0.0069$** ($\mathbf{+0.0202}$ over `clip_vit_b16`)
    - **$\text{SRCC} = 0.7484 \pm 0.0072$**
    - **$\text{RMSE} = 5.9057 \pm 0.0956$**
    - **Prediction Reliability ($r_{\text{model}}$):** $0.9680$
    - **Combined Correlation Ceiling:** $0.8053$
- **Resolution of Reviewer 1 Major Comment 3 (Issue #9):** Extended the full attribution consistency evaluation to ensemble heatmaps across all 10 variants. Bagging achieves mean within-ensemble attribution correlation of $\mathbf{0.8864} \pm 0.0447$ and $\text{IoU@15\%} = \mathbf{0.5602}$.

---

## 2. Quantitative Performance Table

Evaluation on the 273 global held-out test images across all 10 independent variants:

| Architecture / Model | PLCC (Mean ± Std) | SRCC (Mean ± Std) | RMSE (Mean ± Std) | MSE (Mean ± Std) | Prediction Reliability ($r_{\text{model}}$) | Combined Ceiling ($\sqrt{r_{\text{human}} \cdot r_{\text{model}}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `vgg16` | $0.5397 \pm 0.0276$ | $0.5730 \pm 0.0249$ | $8.6181 \pm 0.4044$ | $74.4358 \pm 7.1637$ | $0.8559$ | $0.7573$ |
| `vgg19` | $0.5628 \pm 0.0271$ | $0.5869 \pm 0.0243$ | $8.5220 \pm 0.3392$ | $72.7398 \pm 5.7700$ | $0.8607$ | $0.7594$ |
| `resnet152` | $0.5059 \pm 0.0196$ | $0.5265 \pm 0.0212$ | $9.8550 \pm 0.3657$ | $97.2546 \pm 7.2254$ | $0.9162$ | $0.7835$ |
| `densenet161` | $0.6274 \pm 0.0179$ | $0.6166 \pm 0.0170$ | $7.4813 \pm 0.2130$ | $56.0151 \pm 3.2102$ | $0.9088$ | $0.7803$ |
| `efficientnetb3` | $0.4966 \pm 0.0199$ | $0.5056 \pm 0.0240$ | $8.8185 \pm 0.2484$ | $77.8284 \pm 4.3614$ | $0.8631$ | $0.7604$ |
| `barlowtwins` | $0.6169 \pm 0.0119$ | $0.6292 \pm 0.0115$ | $7.0965 \pm 0.1002$ | $50.3708 \pm 1.4199$ | $0.9431$ | $0.7949$ |
| `clip_vit_b16` *(best single)* | $0.7317 \pm 0.0059$ | $0.7456 \pm 0.0055$ | $6.0206 \pm 0.0450$ | $36.2501 \pm 0.5428$ | $0.9757$ | $0.8085$ |
| **Bagging Ensemble (7 models)** | **$0.7519 \pm 0.0069$** | **$0.7484 \pm 0.0072$** | **$5.9057 \pm 0.0956$** | **$34.8864 \pm 1.1242$** | **$0.9680$** | **$0.8053$** |
| **Stacking Ensemble (Ridge)** | $\mathbf{0.7742 \pm 0.0102}$ | $\mathbf{0.7734 \pm 0.0090}$ | $\mathbf{5.5442 \pm 0.1049}$ | $\mathbf{30.7487 \pm 1.1755}$ | **$0.9511$** | **$0.7983$** |

*Note: Human split-half reliability is $r_{\text{human}} = 0.67$.*

---

## 3. Stacking Meta-Learner Weights & Architecture Synergy

Level-1 Ridge regression meta-learners ($\alpha = 1.0$) were fitted strictly on the disjoint validation partitions $val_k$ ($N = 136$ images) across all 10 variants. The resulting weights demonstrate that every architecture contributes positive, complementary information:

| Architecture | Paradigm | Mean Weight | Std Weight | Relative Importance |
| :--- | :--- | :---: | :---: | :---: |
| `clip_vit_b16` | Multimodal Vision-Language Transformer | **$+0.3929$** | $0.0526$ | $38.5\%$ |
| `barlowtwins` | Self-Supervised Contrastive ResNet-50 | **$+0.2115$** | $0.0384$ | $20.7\%$ |
| `densenet161` | Supervised Dense CNN ($300\times 300$) | **$+0.1683$** | $0.0360$ | $16.5\%$ |
| `vgg19` | Supervised Deep CNN | **$+0.0829$** | $0.0275$ | $8.1\%$ |
| `efficientnetb3` | Compound Scaled CNN | **$+0.0739$** | $0.0312$ | $7.2\%$ |
| `resnet152` | Supervised Residual CNN | **$+0.0595$** | $0.0304$ | $5.8\%$ |
| `vgg16` | Supervised CNN | **$+0.0308$** | $0.0251$ | $3.0\%$ |
| **Intercept** | — | **$-0.1671$** | $0.3421$ | — |

---

## 4. Explainability & Attribution Consistency (Stage 3D)

Within-ensemble attribution consistency evaluated across all 45 variant pairs on the 273 global test images ($12,285$ image pairs per metric):

| Metric | Bagging Ensemble (Mean ± Std) | Stacking Ensemble (Mean ± Std) |
| :--- | :---: | :---: |
| **Pearson Correlation ($r$)** | $\mathbf{0.8864 \pm 0.0447}$ | $\mathbf{0.8193 \pm 0.0564}$ |
| **Structural Similarity (SSIM)** | $\mathbf{0.5571 \pm 0.0980}$ | $\mathbf{0.4735 \pm 0.0923}$ |
| **Top-5% IoU** | $\mathbf{0.4632 \pm 0.1171}$ | $\mathbf{0.3963 \pm 0.1057}$ |
| **Top-15% IoU** | $\mathbf{0.5602 \pm 0.0852}$ | $\mathbf{0.4975 \pm 0.0843}$ |
| **Top-25% IoU** | $\mathbf{0.6206 \pm 0.0714}$ | $\mathbf{0.5604 \pm 0.0713}$ |

### 4.1 Resolution of Reviewer 1 Critique
These quantitative metrics establish that:
1. Ensembles produce attributions that are **more spatially consistent and robust to initialization** than individual CNN architectures (which averaged $r \approx 0.65 - 0.75$).
2. The prototype agreement analysis confirms that ensemble prototypes capture a unified consensus map across visual representations, directly answering Reviewer 1 Major Comment 3.

---

## 5. Artifacts and Output Files

All experiment files are organized in `Outputs/Experiment_3_ensemble/`:
- **Results JSON:**
  - `Results/experiment_3c_comparison.json`: Complete comparison summary across all models and ensembles.
  - `Results/experiment_3c_ensemble_results.json`: Detailed per-variant breakdown of regression metrics.
  - `Results/experiment_3d_xai_consistency.json`: Within-ensemble consistency and prototype similarity matrices.
- **Weights & Models:**
  - `Weights/stacking_coefficients.json`: Ridge weights, intercepts, and per-variant coefficients.
  - `Weights/stacking_exp3_variant{1..10}_meta.pth`: PyTorch `state_dict` for all 10 `StackingMetaLearner` models.
- **Attribution Maps:**
  - `XAI_Maps/Multiscale_Pixel_Masking/bagging_exp3_variant{1..10}_maps.npy` ($(273, 224, 224)$ float32)
  - `XAI_Maps/Multiscale_Pixel_Masking/stacking_exp3_variant{1..10}_maps.npy` ($(273, 224, 224)$ float32)
  - `XAI_Maps/Multiscale_Pixel_Masking/bagging_exp3_prototype_maps.npy`
  - `XAI_Maps/Multiscale_Pixel_Masking/stacking_exp3_prototype_maps.npy`
- **Plots:**
  - `Plots/mpm_within_{bagging,stacking}_correlation_matrix.png`
  - `Plots/mpm_within_{bagging,stacking}_top_percent_iou_15_matrix.png`
  - `Plots/mpm_prototypes_ensemble_vs_single_correlation_matrix.png`
