---
status: proposed
---

# Speed up Multiscale Pixel Masking without giving up model agnosticism

Multiscale Pixel Masking (MPM) is our only model-agnostic saliency method, but it is the bottleneck of Experiment 2A and 3D: with `sigma: [3, 17, 65]` on 224×224 inputs it runs **150,528 forward passes per image per model** (one per pixel per scale). We first remove all host-side overhead with an exactly equivalent vectorised implementation (done). Any further order-of-magnitude gain has to come from **running fewer forward passes**. That would change the maps, so it needs explicit validation before it replaces the stride-1 definition used for the published results.

## Context

`MultiscalePixelMasking.generate_map` had three costs that were not the model itself:

1. **Mask construction in Python.** Each of the 150k masks was a separate `torch.ones(1, 1, H, W)` allocation plus a slice fill, followed by `torch.cat`. That means hundreds of small kernel launches per batch.
2. **One host sync per pixel.** Every masked score was read with `.item()` and written back one at a time, which forced 150k GPU→CPU synchronisations per image.
3. **Rebuilding the 50,176-element coordinate list in Python** for every scale.

The model cost does not shrink: about 2.3 PFLOP per image for VGG16 (15.5 GFLOP × 150,528 passes). That is still minutes per image per model on an RTX 4090 at FP32 (estimate, not measured), multiplied by 138 images × architectures × variants.

## Decisions

- **Vectorise mask generation and scoring (implemented, exact).** Occluders are built for a whole batch at once by broadcasting row and column range tests: `(B,H) & (B,W) → (B,1,H,W)`. They are applied with `torch.where`, and scores are written into a device-side flat map with one slice assignment per batch. The device is synchronised once per image. The loop runs under `torch.inference_mode()`. The public API (`sigma_list`, `pixel_batch_size`, `mask_value`, `generate_map(..., normalize)`) and both call sites are unchanged.
- **Treat `pixel_batch_size` as a pure throughput knob.** It does not change the map (tested). Raise it from 256 towards device-memory limits on the GPU servers once measured.
- **Shard across both GPU servers by model (no code).** `(model, variant)` units are independent, and `experiment-two --models/--variants` already allows running disjoint subsets on Server 1 (RTX 3090) and Server 2 (RTX 4090) at the same time. Exp 2 skips maps that already exist, so the two shards do not collide.
- **Proposed, needs a decision: per-scale stride (coarse grid + bilinear upsampling).** Evaluate occluders only on a stride-`s_σ` grid of centres and upsample each per-scale map to H×W. Adjacent σ=65 occluders overlap by about 98%, so stride 1 is wasteful there. The cost is dominated by the smallest scale:

  | Strides for σ = 3 / 17 / 65       | Forward passes per image | Speed-up |
  | --------------------------------- | ------------------------ | -------- |
  | 1 / 1 / 1 (current)               | 150,528                  | 1×       |
  | 1 / 4 / 16 (`s = max(1, σ // 4)`) | 53,508                   | 2.8×     |
  | 2 / 4 / 16                        | 15,876                   | 9.5×     |
  | 2 / 8 / 32                        | 13,377                   | 11.3×    |

  Acceptance must be decided before adoption. Proposed criteria: (a) per-map Spearman/Pearson against stride-1 maps on a sample of images × architectures, and (b) the downstream Exp 2B inter/intra-model correlation matrices must lead to the same conclusions. Default stays stride 1 until then.

## Considered Options

- **Mixed precision (`autocast` bf16/fp16), rejected as a default.** MPM saliency is a _difference_ of two nearly equal scores. For σ=3 occluders the drop is tiny, and bf16 (8-bit mantissa) or fp16 (10-bit) output quantisation can be as large as the signal. It could only be used opt-in, after the same validation as strides. Precision-preserving runtime options (`channels_last`, `torch.compile`, larger batches) are fine to try but must be measured on the GPU servers first.
- **Incremental inference (reuse unchanged activations, as in Krypton: Nakandala, Kumar & Papakonstantinou, SIGMOD 2019), rejected.** An occluder only changes activations inside its receptive field, so the large speed-ups are real. But it needs per-layer knowledge of the network graph, which breaks the model-agnostic property that is the reason MPM exists.
- **CPU multiprocessing, rejected.** The workload is GPU forward passes. Python-level parallelism does not help, and the cluster's PyTorch incompatibility rules it out.

## Consequences

- Existing MPM maps stay valid: the vectorised implementation is bit-identical to the previous one (max |diff| = 0 in the benchmark below), so no regeneration is needed.
- Local benchmark (M1 Pro, legacy vs vectorised, same inputs):

  | Case                                                  | Device | Legacy  | Vectorised | Speed-up           |
  | ----------------------------------------------------- | ------ | ------- | ---------- | ------------------ |
  | Trivial model, 224 px, σ=17 (explainer overhead only) | CPU    | 7.93 s  | 1.77 s     | 4.5×               |
  | Trivial model, 224 px, σ=17 (explainer overhead only) | MPS    | 16.74 s | 3.28 s     | 5.1×               |
  | ResNet-18, 64 px, σ=[3, 17]                           | MPS    | 4.50 s  | 1.56 s     | 2.9×               |
  | ResNet-18, 64 px, σ=[3, 17]                           | CPU    | 68.89 s | 69.49 s    | 1.0× (model-bound) |

  The end-to-end gain on CUDA with the project's backbones at 224 px is **unverified**. It shrinks as the forward pass gets heavier (VGG16, ResNet-152), which is why the stride decision is what matters for the long runs.

- Adopting strides would make new maps incomparable with existing stride-1 maps. The whole Exp 2 / 3D map set would need regenerating together, and the manuscript's method description would change.

