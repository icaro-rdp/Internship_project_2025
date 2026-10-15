---
status: proposed
---

# Staged Integration of CLIP ViT-B/16: Backbone & Regression Head, Explainability (Grad-CAM), and Pruning

To address Reviewer 1 Major Comment 1 (#5, #12), we add a CLIP Vision Transformer (`openai/clip-vit-base-patch16`) as a seventh backbone alongside our six CNN architectures. Because Vision Transformers handle spatial tokens, attention mechanisms, and projection readouts differently than convolutional networks, we decompose the integration into three sequential steps: (1) integrating the frozen CLIP backbone with a regression head in `models.py` and training baseline variants (Exp 1A), (2) adapting Grad-CAM and relevancy methods to the ViT token grid and scalar regression output (Exp 2), and (3) applying validation-based unit pruning to the visual projection layer (Exp 1B).

## Context & Problem

Reviewer 1 requested evaluating a vision-language / transformer architecture to determine whether representation-level findings generalize beyond convolutional networks. However, integrating a Vision Transformer into our existing empirical pipeline introduces specific architectural challenges:

1. **Backbone & Output Interface**: ViTs output token sequences rather than 2D spatial feature maps. In CLIP, patch tokens are pooled into a `[CLS]` token and projected via `visual_projection` to a 512-dimensional embedding space.
2. **Attribution on Continuous Regression**: In standard classification, Grad-CAM targets a specific class logit. In our authenticity and quality regression task, the output is a single continuous scalar $\hat{y}$. Furthermore, patch tokens are discarded after the final self-attention block, so hooking the wrong layer yields zero spatial gradients.
3. **Pruning Granularity**: Unlike CNN convolutional channels, ViT representation units can be defined at the level of projection dimensions, attention heads, or spatial tokens.

To minimize complexity and validate behavioral performance early, we establish a three-step phased implementation roadmap.

## Phased Decisions

### Step 1: Model Integration & Regression Head (`models.py`)

- **Architecture**: Implement `CLIPViTAuthenticityPredictor` in `Image_Authenticity_prediction/main/Models/models.py` using Hugging Face `CLIPVisionModelWithProjection` (`openai/clip-vit-base-patch16`).
- **Frozen Backbone**: The entire vision transformer backbone remains frozen (`freeze_backbone=True`). Only the downstream regression head is trained with learning rate $0.001$, matching the protocol of all other backbones.
- **Head Placement**: The regression head attaches to `image_embeds` (512-d, after `visual_projection`), not `pooler_output` (768-d). Architecture:
  `nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))`.
- **Forward Contract**: Returns `(predictions, features)` where `predictions` has shape `(B, 1)` and `features` has shape `(B, 512)`, fulfilling the contract expected by training loops, evaluation functions, and explainers.
- **Preprocessing & Crop Geometry**: Maintain the ImageNet crop geometry (`Resize((256, 256))` $\to$ `CenterCrop(224)`) and apply CLIP normalization (`mean=[0.48145466, 0.4578275, 0.40821073]`, `std=[0.26862954, 0.26130258, 0.27577711]`). This preserves identical pixel spatial coordinates across CNNs and ViT for cross-architecture map comparisons.
- **Evaluation Gate**: Train 10 Monte Carlo variants (Exp 1A) to measure Pearson $r$, Spearman $\rho$, and RMSE against human MOS ratings. Verify that the CLIP visual features achieve competitive behavioral prediction before proceeding to attribution or pruning.

### Step 2: Explainability Adaptation (Grad-CAM & Attributions on ViT Regression)

- **Execution Gate**: Undertaken only after Step 1 confirms successful training convergence and competitive behavioral prediction.
- **Scalar Regression Attribution Mechanics**:
  - Gradients are backpropagated directly from the scalar output $\hat{y}$ (`score.backward()`).
  - **Positive Attribution ($\text{ReLU}(\text{Grad-CAM on } \hat{y})$)**: Identifies image regions driving predicted authenticity/quality _higher_.
  - **Negative Attribution ($\text{ReLU}(\text{Grad-CAM on } -\hat{y})$)**: Identifies image regions pulling predicted authenticity/quality _lower_.
  - **Signed / Magnitude Attribution**: Absolute values or unthresholded maps reveal overall influential regions regardless of valence.
- **ViT Gradient Flow & Hook Target**:
  - The regression head reads only the `[CLS]` token. In the final self-attention block, patch tokens (`1...196`) are discarded after self-attention; hooking the output of the final layer yields zero gradient on patch tokens.
  - To capture non-zero spatial gradients, the hook must target the input to the final attention block: `vision_model.encoder.layers.11.layer_norm1`. At this point, patch tokens still feed into the `[CLS]` token via final self-attention.
- **Spatial Reshaping (`reshape_transform`)**:
  - Activations and gradients at `layer_norm1` have shape `(B, 197, 768)`.
  - The explainer drops the `[CLS]` token at index 0 $\to$ `(B, 196, 768)`.
  - Reshapes the 196 tokens into a 2D spatial grid $\to$ `(B, 14, 14, 768)`.
  - Permutes dimensions to `(B, 768, 14, 14)`, allowing standard global average pooling over spatial dimensions and bilinear interpolation to $224 \times 224$.
- **Relevancy Propagation Alternative (Chefer et al., 2021 / arXiv:2012.09838)**:
  - If vanilla Grad-CAM on `layer_norm1` produces noisy token attributions, implement Transformer Relevancy Propagation (gradient-weighted self-attention rollout backpropagated from the regression scalar $\hat{y}$) as a specialized ViT explainer.
- **Model-Agnostic MPM**: Multi-scale Pixel Masking (MPM) operates unchanged on ViT because it evaluates input-perturbation output differences via the standard `(predictions, features)` forward contract.

### Step 3: Unit-Ablation Pruning Strategy

- **Execution Gate**: Executed as Experiment 1B after baseline training (Step 1) and attribution stability (Step 2) are verified.
- **Pruning Unit = Output Row of `visual_projection`**:
  - `visual_projection` is a `Linear(768, 512, bias=False)`.
  - Zeroing row $i$ in `weight[i, :]` sets embedding dimension $i$ to exactly 0, serving as the transformer analogue to zeroing an output channel of a CNN conv layer.
  - `FeatureMapsPruner` in `pruning.py` already implements row zeroing (`layer.weight.data[channel_idx, ...] = 0`). It only needs the unit count from `weight.shape[0]` rather than `out_channels`.
- **Strict Validation-Based Pruning (ADR 0002 Compliance)**:
  - Importance score ranking and greedy pruning iterations are executed strictly on `val_loader` to prevent test-set leakage, reserving `test_loader` exclusively for out-of-sample evaluation.
- **Considered Pruning Granularities**:
  - _Attention Heads (12 heads $\times$ 12 layers = 144 heads)_: Pruning attention heads modifies intermediate representations non-linearly and evaluates transformer head redundancy rather than feature readout importance. Deferred as a potential follow-up.
  - _Patch / Token Pruning_: Pruning spatial patch tokens is relevant for inference speedup but does not map onto the unit-importance analysis of Exp 1B.

## Consequences

- **Staged Risk Management**: Decoupling backbone training from explainability and pruning ensures that modeling bugs or poor regression performance are detected before engineering custom ViT backward hooks or pruners.
- **Comparability with CNNs**: ViT-B/16 provides a $14 \times 14$ token grid matching VGG's last conv layer grid, enabling direct cross-architecture heatmap comparisons in Exp 2.
- **Registry & Routing**: `MODEL_REGISTRY` entries must explicitly identify CLIP's base dataset and normalization so that data loaders route correctly without fallback issues.
- **Storage & Compute**: 10 variants $\times$ 3 conditions (baseline, greedy pruned, negative pruned) require caching the ~86M parameter ViT (~350 MB fp32 per checkpoint), totaling ~10 GB.
- **Reviewer Alignment**: Systematically resolves Reviewer 1 Major Comments 1 and 6.

