# Experiment 2: Post-Hoc Attribution Consistency, Prototype Agreement & Cross-Method Analysis Report

**Date:** November 10, 2026  
**Status:** Completed (Unpruned 7-Architecture Pipeline on $N = 273$ Global Test Images)  
**Resolves / Updates:** GitHub Issues #8, #14, #15, #16, #17, #18  
**Core Modules:** `Image_Authenticity_prediction/main/Experiments/experiment_two.py`, `Image_Authenticity_prediction/main/Utils/explainability.py`, `Image_Authenticity_prediction/main/Utils/comparisons.py`, `scripts/python/reproduce_agreement_figures.py`

---

## 1. Executive Summary

This report documents the complete empirical results of **Experiment 2** across all **7 visual architectures** (`barlowtwins`, `resnet152`, `densenet161`, `efficientnetb3`, `vgg16`, `vgg19`, and `clip_vit_b16`), evaluated over **10 independently trained unpruned variants per architecture** ($70$ checkpoints total from Experiment 1A) on the held-out **$N = 273$ global test split** ($20\%$ of the $1{,}368$ Kept AIGCIQA2023 stimuli) using three post-hoc explainability paradigms:
1. **Grad-CAM** (gradient-weighted activation mapping on target spatial representations)
2. **Multiscale Pixel Masking (MPM)** (model-agnostic Gaussian patch perturbation across $\sigma \in \{3, 17, 65\}$)
3. **LIME** (local linear surrogate fitted over SLIC superpixels, evaluated in both 2D pixel space and 1D native superpixel space)

### 1.1 Key Findings

- **High Within-Architecture Attribution Consistency Across All 3 Methods:**
  - Because backbones are frozen, variant-to-variant attribution consistency reflects stability across random regression-head initializations and 70%/10% train/validation split assignments ($45$ variant pairs $\times 273$ test images $= 12{,}285$ comparisons per architecture).
  - **Grad-CAM:** Mean pairwise Pearson correlation ranges from **$r = 0.8668$** (`efficientnetb3`) to **$r = 0.9946$** (`barlowtwins`), with `vgg16` at **$0.9644 \pm 0.0327$**, `vgg19` at **$0.9626 \pm 0.0330$**, and `clip_vit_b16` at **$0.9165 \pm 0.0707$**.
  - **MPM:** Mean pairwise Pearson correlation ranges from **$r = 0.7905$** (`efficientnetb3`) to **$r = 0.9687$** (`clip_vit_b16`).
  - **LIME:** Mean pairwise Pearson correlation ranges from **$r = 0.8037$** (`efficientnetb3`) to **$r = 0.9588$** (`clip_vit_b16`) in 2D pixel space ($0.8030$ to $0.9586$ in 1D superpixel space), with moderate-to-strong local surrogate fidelity ($R^2 = 0.4342$ to $0.6068$).
- **Pronounced Across-Architecture Non-Identifiability (Rashomon Effect):**
  - Despite high internal stability within each architecture, **architectural prototype heatmaps** diverge strongly across distinct backbones: mean pairwise prototype correlation across all 7 architectures is **$r = 0.1319 \pm 0.3538$** for Grad-CAM ($r = 0.2304 \pm 0.3414$ across the 6 CNNs), **$r = 0.0462 \pm 0.2718$** for MPM ($0.0688$ across 6 CNNs), and **$r = 0.0533 \pm 0.2288$** for LIME ($0.0690$ across 6 CNNs).
  - Only architecturally homologous pairs exhibit moderate prototype alignment: **`vgg16` vs. `vgg19`** ($r = 0.6050$ in Grad-CAM, $0.4264$ in MPM, $0.4044$ in LIME) and **`barlowtwins` vs. `resnet152`** ($r = 0.5566$ in Grad-CAM, both sharing a ResNet bottleneck topology).
- **Method-Dependence of Post-Hoc Explanations (Reviewer 1 Major Comment 5 / Issue #8):**
  - Comparing attribution methods *within the same model variant* (`cross_methods_pairs`) yields low-to-moderate spatial correlations (`mpm_vs_lime`: $r = 0.1342$ to $0.2759$; `gradcam_vs_mpm`: $r = -0.0866$ to $+0.3433$; `gradcam_vs_lime`: $r = -0.0144$ to $+0.1477$).
- **Independence from Stimulus Score and Prediction Error:**
  - Per-image within-architecture attribution consistency is near-orthogonal to both ground-truth human authenticity ratings ($|r| \le 0.14$) and model prediction error MAE ($|r| \le 0.15$), confirming that explanation stability is an intrinsic property of the backbone-readout mapping rather than an artifact of stimulus difficulty.

---

## 2. Experimental Setup & Target Attribution Layers

For gradient-based attribution (Grad-CAM), the explainer hooks the terminal spatial representation preceding global pooling and regression readout:

| Architecture | Paradigm | Input Resolution | Grad-CAM Target Layer | Spatial Grid ($H_L \times W_L$) | Channels ($C$) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **`vgg16`** | Supervised CNN | $224 \times 224$ | `features.30` *(post-ReLU + MaxPool2d)* | $7 \times 7$ | $512$ |
| **`vgg19`** | Supervised CNN | $224 \times 224$ | `features.36` *(post-ReLU + MaxPool2d)* | $7 \times 7$ | $512$ |
| **`resnet152`** | Supervised Residual CNN | $224 \times 224$ | `layer4.2` *(final Bottleneck block)* | $7 \times 7$ | $2048$ |
| **`densenet161`** | Supervised Dense CNN | $300 \times 300$ | `features.denseblock4` *(final DenseBlock)* | $9 \times 9$ | $2208$ |
| **`efficientnetb3`** | Supervised Compound-Scaled CNN | $224 \times 224$ | `features.8` *(final Conv2dNormActivation)* | $7 \times 7$ | $1536$ |
| **`barlowtwins`** | Self-Supervised ResNet-50 | $224 \times 224$ | `layer4.2` *(final Bottleneck block)* | $7 \times 7$ | $2048$ |
| **`clip_vit_b16`** | Multimodal Vision Transformer | $224 \times 224$ | `vision_model.encoder.layers.11.layer_norm1` | $14 \times 14$ *(196 patch tokens)* | $768$ |

---

## 3. Within-Architecture Attribution Consistency ($N = 273$ Test Images, 10 Variants)

For each architecture and XAI method, pairwise similarity is computed across all $\binom{10}{2} = 45$ variant pairs on each of the $N = 273$ held-out test images ($45 \times 273 = 12{,}285$ comparisons per metric). Values below report $\text{Mean} \pm \text{SD}$ across all $12{,}285$ image-pair evaluations (`std_all`), followed in parentheses by the image-level standard deviation (`std_img`, SD across the 273 per-image means) and variant-pair standard deviation (`std_pair`, SD across the 45 variant-pair means).

### 3.1 Grad-CAM Within-Architecture Consistency (Updates Table 4 `tab:consistency_summary`)

| Architecture | Pearson Correlation ($r$) | SSIM | Top-5% IoU | Top-15% IoU | Top-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`barlowtwins`** | **$0.9946 \pm 0.0072$** *(img: 0.0051, pair: 0.0014)* | **$0.9262 \pm 0.0600$** | **$0.7921 \pm 0.1313$** | **$0.8621 \pm 0.0844$** | **$0.8980 \pm 0.0662$** |
| **`resnet152`** | **$0.9897 \pm 0.0141$** *(img: 0.0096, pair: 0.0021)* | **$0.8568 \pm 0.1157$** | **$0.7742 \pm 0.1420$** | **$0.8585 \pm 0.0969$** | **$0.8920 \pm 0.0764$** |
| **`vgg16`** | **$0.9644 \pm 0.0327$** *(img: 0.0202, pair: 0.0049)* | **$0.7692 \pm 0.1287$** | **$0.6582 \pm 0.1845$** | **$0.7381 \pm 0.1256$** | **$0.7801 \pm 0.0992$** |
| **`vgg19`** | **$0.9626 \pm 0.0330$** *(img: 0.0189, pair: 0.0085)* | **$0.7646 \pm 0.1337$** | **$0.6699 \pm 0.1930$** | **$0.7331 \pm 0.1276$** | **$0.7736 \pm 0.1004$** |
| **`clip_vit_b16`** | **$0.9165 \pm 0.0707$** *(img: 0.0406, pair: 0.0359)* | **$0.6102 \pm 0.1548$** | **$0.5175 \pm 0.1504$** | **$0.6162 \pm 0.1276$** | **$0.6782 \pm 0.1156$** |
| **`densenet161`** | **$0.8804 \pm 0.1206$** *(img: 0.0782, pair: 0.0270)* | **$0.6077 \pm 0.2237$** | **$0.5195 \pm 0.2482$** | **$0.6037 \pm 0.1865$** | **$0.6589 \pm 0.1548$** |
| **`efficientnetb3`** | **$0.8668 \pm 0.1320$** *(img: 0.0907, pair: 0.0117)* | **$0.5779 \pm 0.2941$** | **$0.4670 \pm 0.2524$** | **$0.5577 \pm 0.1960$** | **$0.6205 \pm 0.1685$** |

### 3.2 Multiscale Pixel Masking (MPM) Within-Architecture Consistency (Updates Section 5.3 & Appendix A.1.2)

| Architecture | Pearson Correlation ($r$) | SSIM | Top-5% IoU | Top-15% IoU | Top-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clip_vit_b16`** | **$0.9687 \pm 0.0303$** *(img: 0.0211, pair: 0.0113)* | **$0.7683 \pm 0.1398$** | **$0.6782 \pm 0.1385$** | **$0.7548 \pm 0.1073$** | **$0.7986 \pm 0.0899$** |
| **`barlowtwins`** | **$0.9010 \pm 0.0707$** *(img: 0.0477, pair: 0.0273)* | **$0.5848 \pm 0.1671$** | **$0.4758 \pm 0.1688$** | **$0.5871 \pm 0.1317$** | **$0.6504 \pm 0.1101$** |
| **`resnet152`** | **$0.8788 \pm 0.0792$** *(img: 0.0535, pair: 0.0159)* | **$0.5365 \pm 0.2126$** | **$0.4920 \pm 0.1741$** | **$0.5706 \pm 0.1276$** | **$0.6259 \pm 0.1053$** |
| **`densenet161`** | **$0.8784 \pm 0.0784$** *(img: 0.0520, pair: 0.0201)* | **$0.5392 \pm 0.1784$** | **$0.4460 \pm 0.1680$** | **$0.5476 \pm 0.1279$** | **$0.6112 \pm 0.1083$** |
| **`vgg19`** | **$0.8364 \pm 0.1027$** *(img: 0.0694, pair: 0.0201)* | **$0.4937 \pm 0.1736$** | **$0.4116 \pm 0.1976$** | **$0.5123 \pm 0.1459$** | **$0.5765 \pm 0.1206$** |
| **`vgg16`** | **$0.8355 \pm 0.0964$** *(img: 0.0655, pair: 0.0183)* | **$0.4908 \pm 0.1686$** | **$0.3989 \pm 0.1929$** | **$0.4933 \pm 0.1393$** | **$0.5646 \pm 0.1182$** |
| **`efficientnetb3`** | **$0.7905 \pm 0.1224$** *(img: 0.0805, pair: 0.0193)* | **$0.4052 \pm 0.1973$** | **$0.3728 \pm 0.1751$** | **$0.4695 \pm 0.1395$** | **$0.5286 \pm 0.1172$** |

### 3.3 LIME Within-Architecture Consistency — 2D Pixel Space & 1D Superpixel Space (Updates Table 5 `tab:lime_within`)

#### 3.3.1 LIME 2D Pixel-Space Heatmaps ($224 \times 224$)

| Architecture | Pearson Correlation ($r$) | SSIM | Top-5% IoU | Top-15% IoU | Top-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clip_vit_b16`** | **$0.9588 \pm 0.0346$** *(img: 0.0230, pair: 0.0154)* | **$0.8016 \pm 0.0935$** | **$0.7112 \pm 0.2459$** | **$0.7562 \pm 0.1331$** | **$0.7833 \pm 0.1026$** |
| **`barlowtwins`** | **$0.9189 \pm 0.0578$** *(img: 0.0366, pair: 0.0159)* | **$0.6588 \pm 0.0948$** | **$0.5871 \pm 0.2513$** | **$0.6404 \pm 0.1418$** | **$0.6843 \pm 0.1030$** |
| **`resnet152`** | **$0.8865 \pm 0.0675$** *(img: 0.0479, pair: 0.0146)* | **$0.5832 \pm 0.1001$** | **$0.5674 \pm 0.2539$** | **$0.5946 \pm 0.1380$** | **$0.6310 \pm 0.1036$** |
| **`densenet161`** | **$0.8669 \pm 0.0769$** *(img: 0.0471, pair: 0.0156)* | **$0.5489 \pm 0.0870$** | **$0.4880 \pm 0.2353$** | **$0.5509 \pm 0.1321$** | **$0.6076 \pm 0.0990$** |
| **`vgg16`** | **$0.8465 \pm 0.0961$** *(img: 0.0621, pair: 0.0152)* | **$0.5695 \pm 0.0858$** | **$0.4428 \pm 0.2274$** | **$0.5265 \pm 0.1280$** | **$0.5857 \pm 0.0968$** |
| **`vgg19`** | **$0.8350 \pm 0.0869$** *(img: 0.0552, pair: 0.0142)* | **$0.5297 \pm 0.0843$** | **$0.4563 \pm 0.2331$** | **$0.5261 \pm 0.1382$** | **$0.5806 \pm 0.1001$** |
| **`efficientnetb3`** | **$0.8037 \pm 0.1065$** *(img: 0.0726, pair: 0.0181)* | **$0.4641 \pm 0.1022$** | **$0.4249 \pm 0.2365$** | **$0.4797 \pm 0.1398$** | **$0.5370 \pm 0.1112$** |

#### 3.3.2 LIME 1D Superpixel-Space Coefficient Vectors (`lime_superpixel_within_model_variants`)

| Architecture | Pearson ($r$) | Spearman ($\rho$) | Top-5% IoU | Top-15% IoU | Top-25% IoU | Bottom-5% IoU | Bottom-15% IoU | Bottom-25% IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`clip_vit_b16`** | $0.9586 \pm 0.0317$ | $0.9458 \pm 0.0363$ | $0.6886 \pm 0.2145$ | $0.7399 \pm 0.1232$ | $0.7695 \pm 0.0993$ | $0.6053 \pm 0.2136$ | $0.6798 \pm 0.1296$ | $0.7256 \pm 0.1056$ |
| **`barlowtwins`** | $0.9214 \pm 0.0394$ | $0.9081 \pm 0.0399$ | $0.5605 \pm 0.2102$ | $0.6309 \pm 0.1266$ | $0.6688 \pm 0.0974$ | $0.5453 \pm 0.2123$ | $0.6117 \pm 0.1240$ | $0.6637 \pm 0.0996$ |
| **`resnet152`** | $0.8856 \pm 0.0600$ | $0.8584 \pm 0.0587$ | $0.5329 \pm 0.2209$ | $0.5713 \pm 0.1285$ | $0.6106 \pm 0.1039$ | $0.4910 \pm 0.2111$ | $0.5452 \pm 0.1337$ | $0.5955 \pm 0.1048$ |
| **`densenet161`** | $0.8702 \pm 0.0561$ | $0.8518 \pm 0.0516$ | $0.4726 \pm 0.2074$ | $0.5407 \pm 0.1208$ | $0.5943 \pm 0.0971$ | $0.4762 \pm 0.2071$ | $0.5461 \pm 0.1194$ | $0.5937 \pm 0.0961$ |
| **`vgg16`** | $0.8509 \pm 0.0735$ | $0.8380 \pm 0.0528$ | $0.4292 \pm 0.1999$ | $0.5126 \pm 0.1167$ | $0.5726 \pm 0.0939$ | $0.4684 \pm 0.2003$ | $0.5300 \pm 0.1162$ | $0.5865 \pm 0.0939$ |
| **`vgg19`** | $0.8409 \pm 0.0674$ | $0.8255 \pm 0.0583$ | $0.4432 \pm 0.2029$ | $0.5144 \pm 0.1239$ | $0.5675 \pm 0.0978$ | $0.4398 \pm 0.1995$ | $0.5170 \pm 0.1193$ | $0.5709 \pm 0.0961$ |
| **`efficientnetb3`** | $0.8030 \pm 0.0926$ | $0.7744 \pm 0.0869$ | $0.4065 \pm 0.2061$ | $0.4611 \pm 0.1277$ | $0.5164 \pm 0.1062$ | $0.4099 \pm 0.2030$ | $0.4653 \pm 0.1313$ | $0.5186 \pm 0.1071$ |

### 3.4 LIME Local Surrogate Fidelity ($R^2$) (Updates Appendix Table 9 `tab:supmat_lime_r2_by_arch`)

Evaluated across $10\text{ variants} \times 273\text{ test images} = 2{,}730$ local Ridge surrogate fits per architecture:

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

## 4. Across-Architecture Prototype Consistency (`between_model_architectures`)

For each architecture, the **architectural prototype heatmap** is formed by averaging the 10 variant maps per test image (`axis=0`). Pairwise similarity is then evaluated across all $\binom{7}{2} = 21$ architecture pairs ($\binom{6}{2} = 15$ pairs when restricted to the 6 CNNs).

### 4.1 Summary of Across-Architecture Prototype Agreement ($N = 273$ Test Images)

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

### 4.2 Pairwise Prototype Correlation Matrices ($7 \times 7$)

#### 4.2.1 Grad-CAM Prototype Correlation Matrix

| Architecture | `barlowtwins` | `resnet152` | `densenet161` | `efficientnetb3` | `vgg16` | `vgg19` | `clip_vit_b16` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`barlowtwins`** | $1.0000$ | **$0.5566$** | $0.2648$ | $0.3204$ | $0.1095$ | $0.0930$ | $-0.1707$ |
| **`resnet152`** | **$0.5566$** | $1.0000$ | $0.3167$ | $0.3472$ | $0.1122$ | $0.0823$ | $-0.1698$ |
| **`densenet161`** | $0.2648$ | $0.3167$ | $1.0000$ | $0.1751$ | $0.1179$ | $0.1298$ | $-0.1312$ |
| **`efficientnetb3`** | $0.3204$ | $0.3472$ | $0.1751$ | $1.0000$ | $0.1232$ | $0.1025$ | $-0.1099$ |
| **`vgg16`** | $0.1095$ | $0.1122$ | $0.1179$ | $0.1232$ | $1.0000$ | **$0.6050$** | $-0.0624$ |
| **`vgg19`** | $0.0930$ | $0.0823$ | $0.1298$ | $0.1025$ | **$0.6050$** | $1.0000$ | $-0.0426$ |
| **`clip_vit_b16`** | $-0.1707$ | $-0.1698$ | $-0.1312$ | $-0.1099$ | $-0.0624$ | $-0.0426$ | $1.0000$ |

#### 4.2.2 MPM Prototype Correlation Matrix

| Architecture | `barlowtwins` | `resnet152` | `densenet161` | `efficientnetb3` | `vgg16` | `vgg19` | `clip_vit_b16` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`barlowtwins`** | $1.0000$ | $0.0282$ | $0.0995$ | $0.0128$ | $0.0212$ | $0.0929$ | $-0.0435$ |
| **`resnet152`** | $0.0282$ | $1.0000$ | $0.0178$ | $0.0496$ | $0.0006$ | $0.0141$ | $0.0402$ |
| **`densenet161`** | $0.0995$ | $0.0178$ | $1.0000$ | $0.0092$ | $0.1137$ | $0.1254$ | $-0.0871$ |
| **`efficientnetb3`** | $0.0128$ | $0.0496$ | $0.0092$ | $1.0000$ | $0.0085$ | $0.0127$ | $0.0514$ |
| **`vgg16`** | $0.0212$ | $0.0006$ | $0.1137$ | $0.0085$ | $1.0000$ | **$0.4264$** | $0.0067$ |
| **`vgg19`** | $0.0929$ | $0.0141$ | $0.1254$ | $0.0127$ | **$0.4264$** | $1.0000$ | $-0.0296$ |
| **`clip_vit_b16`** | $-0.0435$ | $0.0402$ | $-0.0871$ | $0.0514$ | $0.0067$ | $-0.0296$ | $1.0000$ |

#### 4.2.3 LIME 2D Prototype Correlation Matrix

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

## 5. Cross-Method Attribution Agreement Within Identical Variants (`cross_methods_pairs`)

To separate **architectural divergence** from **explainability method sensitivity** (Reviewer 1 Major Comment 5 / Issue #8), heatmaps generated by different XAI methods were compared on the *exact same model checkpoint* across all 10 variants and 273 test images ($2{,}730$ comparisons per architecture per method pair):

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

## 6. Attribution Consistency vs. Human Authenticity Ratings & Prediction Error

### 6.1 Correlation Between Per-Image Within-Architecture Consistency and Human Authenticity MOS (Updates Table 6 `tab:consistency_human`)

Pearson correlation $r$ ($p$-value) across the $N = 273$ held-out test stimuli between each image's mean within-architecture attribution consistency and its ground-truth human Authenticity rating:

| Architecture | Grad-CAM ($r$) | Grad-CAM (IoU@15%) | MPM ($r$) | MPM (IoU@15%) | LIME ($r$) | LIME (IoU@15%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vgg16`** | $-0.0341$ ($p=0.57$) | $-0.0452$ ($p=0.46$) | $+0.0456$ ($p=0.45$) | $+0.0390$ ($p=0.52$) | $+0.1161$ ($p=0.06$) | $+0.0005$ ($p=0.99$) |
| **`vgg19`** | $+0.0301$ ($p=0.62$) | $+0.0078$ ($p=0.90$) | $+0.0728$ ($p=0.23$) | $+0.0899$ ($p=0.14$) | $+0.0720$ ($p=0.24$) | $-0.0568$ ($p=0.35$) |
| **`resnet152`** | $-0.0469$ ($p=0.44$) | $-0.0080$ ($p=0.90$) | $-0.0041$ ($p=0.95$) | $-0.0268$ ($p=0.66$) | $+0.1156$ ($p=0.06$) | $+0.0464$ ($p=0.45$) |
| **`densenet161`** | $-0.0102$ ($p=0.87$) | $+0.0299$ ($p=0.62$) | $+0.0495$ ($p=0.42$) | $+0.0402$ ($p=0.51$) | $+0.0135$ ($p=0.82$) | $-0.0461$ ($p=0.45$) |
| **`efficientnetb3`** | $-0.0650$ ($p=0.28$) | $-0.0492$ ($p=0.42$) | $+0.1382$ ($p=0.02$) | $+0.1021$ ($p=0.09$) | $+0.0587$ ($p=0.33$) | $+0.0270$ ($p=0.66$) |
| **`barlowtwins`** | $-0.1044$ ($p=0.09$) | $-0.0279$ ($p=0.65$) | $+0.0228$ ($p=0.71$) | $+0.0028$ ($p=0.96$) | $+0.0065$ ($p=0.92$) | $+0.0763$ ($p=0.21$) |
| **`clip_vit_b16`** | $-0.0935$ ($p=0.12$) | $-0.0926$ ($p=0.13$) | $+0.0312$ ($p=0.61$) | $-0.0284$ ($p=0.64$) | $-0.0249$ ($p=0.68$) | $-0.1056$ ($p=0.08$) |

### 6.2 Correlation Between Per-Image Within-Architecture Consistency and Model MAE (Updates Appendix Table 11 `tab:consistency_vs_error`)

Pearson correlation $r$ ($p$-value) across the $N = 273$ held-out test stimuli between per-image attribution consistency and mean absolute prediction error $\text{MAE}_i = \frac{1}{10}\sum_{k=1}^{10} |\hat{y}_i^{(k)} - y_i|$:

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

## 7. Across-Architecture Stimulus Agreement Rankings (Issue #18)

Generated via `uv run scripts/python/reproduce_agreement_figures.py` across all 7 architectural prototypes on the 273 test images:

| Method | Mean Stimulus Agreement $\pm$ SD | Highest-Agreement Stimulus (Rank 1) | Lowest-Agreement Stimulus (Rank 273) | Ranking CSV Path |
| :--- | :---: | :--- | :--- | :--- |
| **Grad-CAM** | $0.1319 \pm 0.1055$ *(range: $[-0.0908, +0.5450]$)* | `42.png` (`test_idx=103`, *"A giraffe walking through a green grass covered field"*, $r = 0.5450$) | `2016.png` (`test_idx=203`, *"a flower"*, $r = -0.0908$) | `Outputs/Experiment_2_variants/Plots/agreement_figures/GradCAM/gradcam_across_architecture_agreement_ranking.csv` |
| **MPM** | $0.0462 \pm 0.0740$ *(range: $[-0.0870, +0.3523]$)* | `42.png` (`test_idx=103`, *"A giraffe walking through a green grass covered field"*, $r = 0.3523$) | `499.png` (`test_idx=56`, *"a tree reflected in the sunroof of a blue car"*, $r = -0.0870$) | `Outputs/Experiment_2_variants/Plots/agreement_figures/MPM/mpm_across_architecture_agreement_ranking.csv` |
| **LIME** | $0.0533 \pm 0.0848$ *(range: $[-0.0936, +0.7853]$)* | `41.png` (`test_idx=213`, *"A giraffe walking through a green grass covered field"*, $r = 0.7853$) | `524.png` (`test_idx=17`, *"a library filled with kids reading books"*, $r = -0.0936$) | `Outputs/Experiment_2_variants/Plots/agreement_figures/LIME/lime_across_architecture_agreement_ranking.csv` |

---

## 8. Artifacts & Output Locations

- **Comparison Summary JSON:**
  - `Outputs/Experiment_2_variants/experiment_2b_comparison.json`
- **Attribution Heatmaps ($N = 273$ Test Images, 10 Variants per Architecture):**
  - `Outputs/Experiment_2_variants/XAI_Maps/GradCAM/{model}_exp1a_variant{1..10}_maps.npy`
  - `Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking/{model}_exp1a_variant{1..10}_maps.npy`
  - `Outputs/Experiment_2_variants/XAI_Maps/LIME/{model}_exp1a_variant{1..10}_maps.npy` (plus `_coeffs.npy` and `_meta.npz`)
- **Comparison & Agreement Plots:**
  - `Outputs/Experiment_2_variants/Plots/{gradcam,mpm,lime}_within_model_{metric}_violin.png`
  - `Outputs/Experiment_2_variants/Plots/{gradcam,mpm,lime}_prototypes_{metric}_matrix.png`
  - `Outputs/Experiment_2_variants/Plots/agreement_figures/{GradCAM,MPM,LIME}/pair_01.{pdf,png}` ... `pair_05.{pdf,png}`
