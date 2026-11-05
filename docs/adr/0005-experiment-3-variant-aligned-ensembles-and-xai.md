---
status: accepted
---

# Experiment 3 Architecture: Variant-Aligned Unpruned 7-Model Ensembles, Leakage-Free Validation Stacking, and 273-Image MPM Attribution

To resolve GitHub Issue #13 (integrating `clip_vit_b16` into Experiment 3), Issue #9 (Reviewer 1 Major Comment 3 on ensemble explanatory rigor), and Issue #17 (removing channel pruning across the pipeline), we redesign Experiment 3 (`Image_Authenticity_prediction/main/Experiments/experiment_three.py`) around **10 variant-aligned, 7-architecture unpruned ensembles** trained and selected strictly on validation data, evaluated on the held-out 273-image global test set, and explained via per-variant `(273, 224, 224)` Multiscale Pixel Masking (MPM) `.npy` attribution maps and architectural prototype heatmaps that align 1:1 with Experiment 2.

---

## Context & Problem

A technical audit of the legacy `experiment_three.py` implementation against Experiment 1 (`experiment_one.py`) and Experiment 2 (`experiment_two.py`) identified four architectural and methodological flaws:

1. **Redundant Base Training & Pruning Lock-in (Stages 3A & 3B):**
   - `experiment_3a_train_all_variants` uses the exact same deterministic partitioning as `experiment_1a_train_all_models`: a fixed 20% global test set ($N = 273$ of $1{,}368$ Kept images, `seed = 42`) and per-variant 70%/10% train/validation splits ($959$ train and $136$ validation images, `seed = 42 + k` for $k \in \{1, \dots, 10\}$). Retraining base models in Experiment 3A is therefore 100% redundant with the 70 checkpoints already produced in Experiment 1A (`Outputs/Experiment_1_variants/Weights/{model}_exp1a_variant{1..10}_best.pth`).
   - Furthermore, `experiment_3c_evaluate_ensemble` and `experiment_3d_plot_ensemble_heatmaps` previously aborted if pruned `*_exp3b_*_greedy_pruned.pth` files were missing, incompatible with the project-wide removal of Sequential Backward Selection (SBS) pruning (#17).

2. **Test-Set Data Leakage in Stacking Meta-Learner Fitting (Stage 3C):**
   - In the legacy `experiment_3c_evaluate_ensemble`, base-model predictions were computed exclusively on `test_loader` ($N = 273$):
     - **`stacking` mode** split the 273 global test images 50/50, fitting `StackingMetaLearner` on 136 test images and evaluating on the remaining 137 test images—both leaking test labels into meta-learner training and evaluating on half the test partition ($N = 137$ vs. $N = 273$ for Bagging and individual models).
     - **`stacking_cv` mode** performed 5-fold out-of-fold (OOF) cross-validation directly on the 273 global test images, using held-out test labels to fit the level-1 linear combiner weights and producing 5 fold checkpoints rather than a single out-of-sample meta-learner per variant.

3. **Cross-Variant Split-Overlap Hazard:**
   - Because each variant index $k \in \{1, \dots, 10\}$ partitions the $1{,}095$ non-test images using a distinct random permutation (`seed = 42 + k`), validation split `val_k` ($136$ images) is disjoint from `train_k` **only for models trained with the same variant index $k$**.
   - If an ensemble combines mismatched variant indices across architectures (e.g., `variant3` of Architecture A validated on `val_3` and `variant5` of Architecture B validated on `val_5`), **87.5% of `val_3` belongs to Architecture B's training set `train_5`**. Training a stacking meta-learner on `val_3` (or `val_5`) across mismatched variants leaks in-sample training predictions into the meta-learner and distorts architecture weights.

4. **Incomplete Ensemble XAI Coverage & Format Mismatch (Stage 3D):**
   - `experiment_3d_plot_ensemble_heatmaps` previously generated MPM overlays for only 3 quantile test images (`num_sample_images=3`), saved `.png` figures rather than `(273, 224, 224)` `.npy` arrays, and lacked prototype aggregation across variants.
   - `MODEL_REGISTRY` lacked `clip_vit_b16` routing, and filename regexes (`^([a-z0-9]+)_...`) rejected identifiers containing underscores.

---

## Detailed Architectural Decisions

### 1. Direct Reuse of Unpruned Experiment 1A Weights & Standardized Paths
- **Base Weights Source:** Pull all 70 unpruned base-model checkpoints exclusively from:
  `Outputs/Experiment_1_variants/Weights/{model}_exp1a_variant{1..10}_best.pth`
  across all 7 architectures:
  - 6 Convolutional backbones: `vgg16`, `vgg19`, `resnet152`, `densenet161`, `efficientnetb3`, `barlowtwins`.
  - 1 Vision Transformer backbone: `clip_vit_b16` (`CLIPViTAuthenticityPredictor`, `CLIP_DATASET`, `clip_dataset`, `target_layer="visual_projection"`).
- **Optional Legacy Training & Pruning:** Disable `run_training` and `run_pruning` by default (`False`), keeping functions intact for backward compatibility. Update `FeatureMapsPruner` and filename regexes (`^([a-z0-9_]+)_(?:exp3a|exp1a)_variant(\d+)_best\.pth$`) to parse underscores and stage prefixes cleanly.
- **Standardized Directory Structure:** All outputs route to `Outputs/Experiment_3_ensemble/`:
  ```
  Outputs/Experiment_3_ensemble/
  ├── Weights/
  │   ├── stacking_exp3_variant1_meta.pth ... stacking_exp3_variant10_meta.pth
  │   └── stacking_coefficients.json
  ├── Results/
  │   ├── experiment_3c_ensemble_results.json
  │   └── experiment_3c_comparison.json
  ├── XAI_Maps/
  │   └── Multiscale_Pixel_Masking/
  │       ├── bagging_exp3_variant1_maps.npy ... bagging_exp3_variant10_maps.npy
  │       ├── stacking_exp3_variant1_maps.npy ... stacking_exp3_variant10_maps.npy
  │       ├── bagging_exp3_prototype_maps.npy
  │       └── stacking_exp3_prototype_maps.npy
  └── Plots/
      └── (Overlay visualizations and comparison distributions)
  ```

---

### 2. 10 Variant-Aligned 7-Model Ensembles ($k \in \{1, \dots, 10\}$)
- **Variant Grouping:** Construct **10 independent variants** for each ensemble paradigm ($k = 1, \dots, 10$), where **Ensemble Variant $k$** aggregates strictly the 7 base models trained on data split $k$ (`seed = 42 + k`):
  $$\mathcal{M}_k = \big\{f_{\text{vgg16}}^{(k)}, \, f_{\text{vgg19}}^{(k)}, \, f_{\text{resnet152}}^{(k)}, \, f_{\text{densenet161}}^{(k)}, \, f_{\text{efficientnetb3}}^{(k)}, \, f_{\text{barlowtwins}}^{(k)}, \, f_{\text{clip\_vit\_b16}}^{(k)}\big\}$$
- **Strict Architecture Column Ordering:**
  Feature matrices $X_{\text{val}}^{(k)} \in \mathbb{R}^{136 \times 7}$ and $X_{\text{test}}^{(k)} \in \mathbb{R}^{273 \times 7}$ must strictly order columns $m \in \{0, \dots, 6\}$ according to `get_model_order()` from `main.Utils.config`:
  1. `vgg16` (index 0)
  2. `vgg19` (index 1)
  3. `resnet152` (index 2)
  4. `densenet161` (index 3)
  5. `efficientnetb3` (index 4)
  6. `barlowtwins` (index 5)
  7. `clip_vit_b16` (index 6)
- **Split Isolation:** Every base model in $\mathcal{M}_k$ saw the exact same 959-image training set. Crucially, the 136-image validation set `val_k` was **never seen** by any model in $\mathcal{M}_k$, and the 273-image global test set remains completely held out.

---

### 3. Leakage-Free Validation Selection & Stacking Protocol (Stage 3C)

- **Validation Split Reconstruction:** Reconstruct `val_k` ($136$ images, `seed = 42 + k`) for each $k \in \{1, \dots, 10\}$ across datasets (`imageNet_dataset`, `denseNet_dataset`, `clip_dataset`). Compute base predictions on `val_k` ($X_{\text{val}}^{(k)} \in \mathbb{R}^{136 \times 7}$) and on `test` ($X_{\text{test}}^{(k)} \in \mathbb{R}^{273 \times 7}$).
- **Bagging Ensemble (`bagging`):**
  - No training phase. For each split $k$, compute the unweighted arithmetic mean:
    $$\hat{y}_{\text{bag}}^{(k)}(x) = \frac{1}{7} \sum_{m=1}^{7} f_m^{(k)}(x)$$
  - Evaluate directly on the 273 global test images across all 10 variants to compute test metrics (MSE, RMSE, MAE, PLCC, SRCC, KRCC) and report `mean ± std`.
- **Stacking Ensemble (`stacking`):**
  - **Closed-Form Regularized Solver:** Fit an unconstrained linear regression with intercept on $(X_{\text{val}}^{(k)}, \mathbf{y}_{\text{val}}^{(k)})$ using closed-form Ridge Regression ($\alpha = 1.0$):
    $$\mathbf{w}^{(k)} = \left( (X_{\text{val, ext}}^{(k)})^T X_{\text{val, ext}}^{(k)} + \alpha I \right)^{-1} (X_{\text{val, ext}}^{(k)})^T \mathbf{y}_{\text{val}}^{(k)}$$
    where $X_{\text{val, ext}}^{(k)} = [X_{\text{val}}^{(k)}, \mathbf{1}]$ includes the bias term. This guarantees a unique, deterministic, instant global optimum without training stochasticity.
  - **PyTorch Checkpoint Loading:** Instantiate `StackingMetaLearner(num_base_models=7)` (`nn.Linear(7, 1)`), set `fc.weight.data` to $\mathbf{w}^{(k)}$ and `fc.bias.data` to $b^{(k)}$, and save state dict to `Outputs/Experiment_3_ensemble/Weights/stacking_exp3_variant{k}_meta.pth`.
  - **Metadata Summary:** Export `Outputs/Experiment_3_ensemble/Weights/stacking_coefficients.json` recording the 10 weight vectors, intercepts, validation MSE, and validation $R^2$.
  - **Test Inference:** For each variant $k$, evaluate on all 273 unseen test samples:
    $$\hat{y}_{\text{stack}}^{(k)}(x) = \sum_{m=1}^{7} w_m^{(k)} f_m^{(k)}(x) + b^{(k)}$$
- **Results JSON Schema:**
  Stage 3C outputs `Outputs/Experiment_3_ensemble/Results/experiment_3c_ensemble_results.json` adhering to:
  ```json
  {
    "bagging": {
      "variants": [
        {"variant_idx": 1, "mse": 0.0, "rmse": 0.0, "mae": 0.0, "plcc": 0.0, "srcc": 0.0, "krcc": 0.0},
        ...
      ],
      "mean": {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "plcc": 0.0, "srcc": 0.0, "krcc": 0.0},
      "std": {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "plcc": 0.0, "srcc": 0.0, "krcc": 0.0},
      "prediction_reliability": 0.0,
      "combined_correlation_ceiling": 0.0
    },
    "stacking": {
      "variants": [
        {"variant_idx": 1, "weights": [...], "intercept": 0.0, "val_mse": 0.0, "mse": 0.0, "rmse": 0.0, "mae": 0.0, "plcc": 0.0, "srcc": 0.0, "krcc": 0.0},
        ...
      ],
      "mean": { ... },
      "std": { ... },
      "prediction_reliability": 0.0,
      "combined_correlation_ceiling": 0.0
    },
    "baseline_models": {
      "vgg16": {"mean": { ... }, "std": { ... }},
      ...
      "clip_vit_b16": {"mean": { ... }, "std": { ... }}
    }
  }
  ```
  A summary comparison table is exported to `Outputs/Experiment_3_ensemble/Results/experiment_3c_comparison.json`.

---

### 4. Ensemble Explainability via MPM & Prototype Heatmaps (Stage 3D)

- **Explainability Method:** `MultiscalePixelMasking` (MPM) with scales $\sigma \in \{3, 17, 65\}$.
- **Masking Impact Derivation & Intercept Cancellation:**
  MPM evaluates pixel importance by quantifying the prediction drop when an image patch is masked:
  $$\Delta \hat{y} = \hat{y}_{\text{clean}} - \hat{y}_{\text{masked}}$$
  For `StackingMetaLearner` ($\hat{y} = \sum w_m f_m + b$):
  $$\Delta \hat{y}_{\text{stack}}^{(k)}(r, c, \sigma) = \left( \sum_{m=1}^{7} w_m^{(k)} f_m^{(k)}(x) + b^{(k)} \right) - \left( \sum_{m=1}^{7} w_m^{(k)} f_m^{(k)}(x_{\text{masked}}) + b^{(k)} \right)$$
  The constant intercept $b^{(k)}$ cancels out identically ($(b^{(k)} - b^{(k)}) = 0$). The ensemble drop is strictly the weighted sum of the 7 base models' drops:
  $$\Delta \hat{y}_{\text{stack}}^{(k)}(r, c, \sigma) = \sum_{m=1}^{7} w_m^{(k)} \Delta f_m^{(k)}(r, c, \sigma)$$
  For Bagging, weights are equal ($w_m^{(k)} = \frac{1}{7}$).
- **Dual Execution Modes:**
  1. **Map Reuse Mode (`--reuse-exp2-maps`, Default):**
     - Loads the 7 precomputed Experiment 2 MPM arrays from `Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking/{model}_exp1a_variant{k}_maps.npy`.
     - Bilinearly downsamples `densenet161` maps from $300 \times 300 \to 224 \times 224$ via `uniform_heatmaps` from `main.Utils.comparisons`.
     - Computes the linear combination $\sum_{m=1}^{7} w_m^{(k)} \text{map}_m^{(k)}$ across all 273 images.
     - Normalizes each combined 2D map to $[-1.0, 1.0]$ using `normalize_data(..., min_range=-1.0, max_range=1.0)`.
     - Executes in seconds without GPU compute.
  2. **Live MPM Mode (`--live-mpm`):**
     - Runs `MultiscalePixelMasking(..., normalize=False)` on the 7 base models across the 273 test images.
     - Accumulates raw drops $\Delta f_m^{(k)}$ in a single forward pass for both Bagging ($w_m = \frac{1}{7}$) and Stacking ($w_m = w_m^{(k)}$).
     - Normalizes each final 2D map to $[-1.0, 1.0]$.
- **Per-Variant Heatmaps:**
  Save one 3D array of shape `(273, 224, 224)` (`float32`) for each variant:
  - `Outputs/Experiment_3_ensemble/XAI_Maps/Multiscale_Pixel_Masking/bagging_exp3_variant{1..10}_maps.npy`
  - `Outputs/Experiment_3_ensemble/XAI_Maps/Multiscale_Pixel_Masking/stacking_exp3_variant{1..10}_maps.npy`
- **Architectural Prototype Heatmaps:**
  For each ensemble paradigm, compute the arithmetic mean across the 10 variants (`axis=0`) for every test image:
  $$\text{Prototype}_{\text{bag}}(x) = \frac{1}{10} \sum_{k=1}^{10} \text{Map}_{\text{bag}}^{(k)}(x)$$
  $$\text{Prototype}_{\text{stack}}(x) = \frac{1}{10} \sum_{k=1}^{10} \text{Map}_{\text{stack}}^{(k)}(x)$$
  Save as 3D arrays of shape `(273, 224, 224)`:
  - `Outputs/Experiment_3_ensemble/XAI_Maps/Multiscale_Pixel_Masking/bagging_exp3_prototype_maps.npy`
  - `Outputs/Experiment_3_ensemble/XAI_Maps/Multiscale_Pixel_Masking/stacking_exp3_prototype_maps.npy`
  This enables direct evaluation of:
  - **Within-Ensemble Attribution Consistency:** Pairwise correlation across the 10 variants on the 273 test images (using `compare_heatmaps`).
  - **Ensemble vs. Single-Architecture Agreement:** Comparing `Prototype_stack` or `Prototype_bag` against `Prototype_clip`, `Prototype_barlowtwins`, etc.

---

### 5. CLI Interface & Execution Defaults

Invocation via `python -m Image_Authenticity_prediction experiment-three` supports:
* `--strategy`: `all` (default), `bagging`, `stacking`.
* `--variants`: `all` (default 1..10), or comma-separated list e.g. `1,2,3`.
* `--reuse-exp2-maps`: default `True` (auto-detects Exp 2 maps; falls back to live if missing).
* `--live-mpm`: flag to force end-to-end forward passes.
* `--skip-heatmaps`: flag to run only Stage 3C evaluation without generating XAI maps.
* `--device`: `cuda` or `cpu`.

---

### 6. Cluster Execution & SLURM Sizing (DIPSEN HPC)

To execute Experiment 3 seamlessly on the DIPSEN HPC cluster (`hpc.dipsen.unitn.it`), a dedicated SLURM production script (`scripts/slurm/run_experiment_three.slurm`) is specified adhering strictly to cluster golden rules:
- **Partition & Sizing:**
  - `#SBATCH --partition=all-compute-gpu` (Node `c31`, 2× RTX PRO 6000 Blackwell 96 GiB)
  - `#SBATCH --gres=gpu:1`
  - `#SBATCH --cpus-per-task=8` (even CPU count matching SMT allocation)
  - `#SBATCH --mem=32G`
  - `#SBATCH --time=04:00:00` (wall time generously sized for live MPM; fast reuse finishes in $<5$ minutes)
- **Local NVMe `/scratch` Staging & Cleanup:**
  - Uses validated scratch trap: stages dataset to `/scratch/$USER/$SLURM_JOB_ID/Dataset` to avoid NFS file-access contention during evaluation.
  - Automatically purges NVMe scratch on job exit/error.
- **Environment & Thread Pinning:**
  - Clears Lmod modules, pins OpenMP/MKL/OpenBLAS threads to `$SLURM_CPUS_PER_TASK`, and activates project `.venv`.
- **Command Playbook:**
  - **Fast Reuse Run ($<5$ mins):**
    ```bash
    sbatch scripts/slurm/run_experiment_three.slurm
    ```
    (Runs Stage 3C evaluation across all 10 variants and Stage 3D map reuse to instantly produce all `.npy` arrays and prototypes).
  - **Live MPM Run (~3-4 hours):**
    ```bash
    sbatch scripts/slurm/run_experiment_three.slurm --live-mpm
    ```
  - **Interactive Execution (Debugging on Login / Node):**
    ```bash
    ssh hpc
    cd ~/icaro_projects/Internship_project_2025
    srun --partition=all-compute-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --qos=interactive --pty bash
    python -m Image_Authenticity_prediction experiment-three --strategy all --variants all --reuse-exp2-maps
    ```

---

## Consequences & Reviewer 1 Issue Mapping

### Immediate Benefits
- **Zero Retraining Required:** Experiment 3 executes directly against the 70 existing Experiment 1A unpruned checkpoints in `Outputs/Experiment_1_variants/Weights/`.
- **Strict Data Hygiene:** All 273 global test images remain completely untouched and unseen until final evaluation and MPM attribution.
- **1:1 Alignment with Experiments 1 & 2:** Produces 10 variants per ensemble strategy with identical metric structures (MSE, RMSE, PLCC, SRCC, KRCC) and 3D `.npy` heatmaps of shape `(273, 224, 224)`.

### Resolution of Reviewer 1 Comments (GitHub Issue #1)
1. **Major Comment 1 (Backbones & Pretraining Rationale / Issue #5 & #12):**
   - Incorporates `clip_vit_b16` (vision-language transformer representation) directly into both Bagging and Stacking alongside the 6 CNN architectures, evaluating multimodal representations at the ensemble level.
2. **Major Comment 3 (Explanatory Rigor of Ensemble Methods / Issue #9):**
   - Directly addresses the criticism that ensemble explainability lacked the statistical rigor applied to individual architectures.
   - Replaces the 2 anecdotal qualitative sample maps (Figure 5) with full-dataset 273-image MPM heatmaps across all 10 ensemble variants.
   - Enables computing **Within-Ensemble Attribution Consistency** (pairwise spatial correlation across the 10 variants) and **Attribution Consistency vs. Authenticity Correlations**, directly mirroring the analyses in Tables 4, 5, and 6.
   - Disciplines manuscript claims: ensembling demonstrates an empirical ceiling for behavioral prediction ($r \approx 0.73\text{--}0.75$), while quantitatively examining whether ensembling resolves or merely blends divergent model explanations.
3. **Major Comment 6 (Channel Pruning Justification & Test Leakage / Issue #4 & #17):**
   - Completely removes Sequential Backward Selection (SBS) channel pruning from the ensemble pipeline, evaluating ensembles exclusively on raw, unpruned baseline checkpoints.
4. **Major Comment 8 (Data Partitions & Random Seeds / Issue #3):**
   - Variant alignment ensures that all 7 base models in Variant $k$ share the exact same 959-image training set and 136-image validation set. This prevents cross-variant data leakage where validation data from one model would overlap with training data from another.
5. **Minor Comments 4, 6, 7 & 8 (Framing, Labels & Metrics / Issue #10 & #11):**
   - Consistently retains the "Experiment 3 (Bagging & Stacking)" designation across logs, reports, and code outputs.
   - Reports both PLCC and SRCC with standard error across the 10 variants.
