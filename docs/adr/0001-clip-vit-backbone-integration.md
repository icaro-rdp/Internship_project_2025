---
status: proposed
---

# Integrate CLIP ViT-B/16 as a frozen backbone, pruning the visual projection and using token-grid Grad-CAM

To answer Reviewer 1 Major Comment 1 (#5, #12) we add a CLIP image encoder as a seventh backbone. It has to go through the same pipeline as the CNNs: 10 Monte Carlo variants (Exp 1A), unit-ablation pruning (Exp 1B), Grad-CAM and MPM maps (Exp 2), and ensembles (Exp 3). Otherwise its results are not comparable. We use Hugging Face `CLIPVisionModelWithProjection` (`openai/clip-vit-base-patch16`), frozen, with the regression head on the 512-d `image_embeds`. The **pruning target is `visual_projection`**: each of its 512 output rows is one unit. The **Grad-CAM target is `vision_model.encoder.layers.11.layer_norm1`**: its patch tokens are reshaped into a 14×14 grid that stands in for the CNN feature-map tensor.

## Decisions

- **ViT-B/16 rather than B/32.** B/16 gives a 14×14 token grid, the same as VGG's last conv layer. B/32 gives 7×7, which is coarser than VGG's grid. Grad-CAM resolution is bounded by this grid. B/16 costs about 4× the FLOPs of B/32, which is acceptable on the RTX 4090 at 1368 images.
- **Head on `image_embeds` (512-d, after projection), not `pooler_output` (768-d).** This matches the issue and the CLIP linear-probe setting. It also gives a layer whose output rows are exactly the units the head sees, so the existing pruning scheme carries over.
- **Pruning unit = output row of `visual_projection` (Linear 768→512, `bias=False`).** Zeroing row *i* sets embedding dimension *i* to exactly 0. This is the transformer analogue of zeroing an output channel of the last conv layer. `FeatureMapsPruner` already zeroes `weight[i, ...]`. It only needs to read the unit count from `weight.shape[0]` instead of `out_channels`.
- **Grad-CAM on the input of the last encoder block.** The prediction reads only the CLS token of the final block. Patch tokens *output* by that block therefore get zero gradient. `layer_norm1` of the last block is the last place where patch tokens still feed CLS, through the final self-attention. This matches pytorch-grad-cam's guidance for ViTs (`blocks[-1].norm1`). `GradCAM` gets an optional `reshape_transform` that drops CLS and maps `(B, 197, 768)` → `(B, 768, 14, 14)`. The rest of the algorithm is unchanged.
- **MPM is used unchanged.** It only needs the `(predictions, features)` forward contract.
- **Preprocessing keeps the ImageNet crop geometry (`Resize((256, 256))` → `CenterCrop(224)`) and uses CLIP mean/std.** This departs from CLIP's official bicubic shortest-side-224 crop. The reason: CLIP maps then share the pixel grid of the 224-px CNN maps for the Exp 2 cross-architecture comparisons. The CLIP dataset is built from the same CSV, so the global test indices still line up.
- **Frozen backbone only.** This is consistent with `freeze_backbone: true` and `learning_rate: 0.001`. Fine-tuning CLIP would need per-group learning rates (about 1e-5 for the backbone) and is deferred.

## Considered options

- **open_clip / timm instead of `transformers`.** open_clip uses fused `nn.MultiheadAttention`, which makes hooks and per-head work awkward. timm's CLIP ViTs are clean, but the issue and the literature use the HF checkpoint IDs.
- **Attention heads as pruning units** (Michel et al., 2019). The last layer has only 12 heads, too few to compare with 48–2048 CNN channels. All 144 heads is a compression study, not the unit-importance analysis of Exp 1B. Kept as a possible follow-up.
- **Last-block MLP neurons (3072).** In the final block only the CLS row affects the output, so these units are non-spatial anyway. They are also 6× costlier than the projection.
- **Attention rollout** (Abnar & Zuidema, 2020). It is class-agnostic and ignores the regression head. With a frozen backbone it would give identical maps for every head and variant, so it cannot be compared with Grad-CAM.
- **Gradient-weighted attention relevance** (Chefer et al., 2021). It is target-specific and has been shown on CLIP. It is a reasonable secondary method, but it is not Grad-CAM, so it would not answer "same method across architectures". It also needs eager attention to expose attention weights.

## Consequences

- Experiments route datasets by `config["dataset"] is IMAGENET_DATASET`. A CLIP entry would silently fall back to fixed splits in Exp 1A (`experiment_one.py`, dataset branch). In Exp 3 it would get DenseNet test indices and loaders. Registry entries must carry their base dataset explicitly, and all three duplicated `MODEL_REGISTRY` dicts need the new entry.
- Ablating a CLIP unit removes the dimension *exactly*. Zeroing a ResNet `conv3` or a DenseNet `conv2` channel only removes one contribution before BN or the residual/concat. Unit-importance distributions are therefore not strictly like-for-like, and this should be stated in the paper.
- Each checkpoint stores the full ViT (about 86M parameters, ~350 MB fp32). Across 10 variants × {base, greedy, negative} that is about 10 GB.
- New runtime dependency (`uv add transformers`). The GPU servers need the HF weights cached, or network access on first run.
