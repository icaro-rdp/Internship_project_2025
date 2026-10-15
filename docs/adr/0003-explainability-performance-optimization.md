---
status: proposed
---

# Explainability Performance Optimization: Device-Native Vectorization and Truncated Autograd for Attribution Heatmaps

Explainability methods in Experiment 2 and Experiment 3 generate attribution heatmaps using two complementary paradigms:
1. **Multi-scale Pixel Masking (MPM)**: Model-agnostic perturbation evaluating prediction drops across spatial scales ($\sigma \in \{3, 17, 65\}$). With default settings at 224×224 resolution, MPM executes **150,528 forward passes per image per model**.
2. **Grad-CAM**: Model-intrusive gradient-weighted class activation mapping targeting internal feature representations via backpropagation.

Historically, both explainers suffered from severe host-side bottlenecks, CPU–GPU memory synchronization thrashing, and unvectorized execution. This decision formalizes the performance architecture for both explainability pipelines.

## Context

Prior implementations introduced substantial non-model latency:

### Multiscale Pixel Masking (MPM)
- **Host-side mask generation**: 150k separate `torch.ones` allocations and slice assignments in Python.
- **Host-synchronization thrashing**: Every masked output score was read synchronously via `.item()`, forcing 150k GPU→CPU synchronizations per image.
- **Python coordinate reconstruction**: Rebuilding 50,176 coordinate tuples in Python for every scale.

### Grad-CAM
- **Premature CPU host transfers**: Immediate conversion of GPU activation and gradient tensors to NumPy arrays (`.cpu().numpy()[0]`), causing explicit GPU synchronization.
- **Python channel loop**: Linearly combining feature maps via a slow Python `for` loop over hundreds or thousands of channels ($C \in \{512, 2048\}$):
  ```python
  cam = np.zeros(activations.shape[1:], dtype=np.float32)
  for i, w in enumerate(weights):
      cam += w * activations[i, :, :]
  ```
- **CPU tensor ping-pong**: Converting the NumPy CAM back into a PyTorch tensor (`torch.tensor(cam)`) to run `F.interpolate` on the CPU, and then converting back to NumPy for normalization.
- **Full-network backpropagation**: Calling `score.backward()` traversed backwards through the entire architecture back to the input image, computing parameter gradients (`.grad`) for millions of model parameters despite the network operating in evaluation mode.
- **In-place hook hazards**: PyTorch's `register_full_backward_hook` fails when subsequent layers use in-place operations (`ReLU(inplace=True)`), requiring manual model modification workarounds.
- **Batch size 1 constraint**: Restricting `generate_map` to `B=1` prevented utilizing GPU tensor parallelism over evaluation datasets.
- **Hook churn**: Re-instantiating `GradCAM` inside DataLoader loops created and removed hooks on every image.

## Decisions

- **Device-Native Tensor Vectorization (Implemented, Bit-Identical).**
  Keep all intermediate activations, masks, pooling, interpolation, and normalization on the model's compute device (CUDA / MPS):
  - *MPM*: Build boolean occluder masks for a whole batch via broadcasted row and column range comparisons: `(B, H) & (B, W) → (B, 1, H, W)`. Apply with `torch.where` and write score drops directly into a flat device tensor. Synchronize to host CPU exactly once per image.
  - *Grad-CAM*: Compute channel importance weights via global average pooling `grads.mean(dim=(-2, -1), keepdim=True)`. Compute weighted channel combinations as a single tensor reduction `(weights * activations).sum(dim=1, keepdim=True)`. Apply `F.relu`, interpolate with `F.interpolate(mode="bilinear")`, and normalize ranges entirely on device tensors. Synchronize to host CPU NumPy once per batch.
- **Truncated Autograd for Grad-CAM (Implemented, Exact).**
  Replace `register_full_backward_hook` and `score.backward()` with a forward leaf hook:
  ```python
  def forward_hook(module, input, output):
      self.activations = output.clone().requires_grad_(True)
      return self.activations
  ```
  and compute target gradients directly via `torch.autograd.grad`:
  ```python
  grads = torch.autograd.grad(outputs=score, inputs=self.activations, retain_graph=False)[0]
  ```
  Backpropagation halts strictly at `target_layer`, skipping all earlier convolutional layers and computing zero model parameter gradients. In-place downstream activations (such as `ReLU(inplace=True)`) operate without error because `output.clone()` provides an independent leaf tensor.
- **Batch Processing Support ($B \ge 1$) as a Throughput Knob.**
  - *MPM*: `pixel_batch_size` (number of occluded patches evaluated per forward pass).
  - *Grad-CAM*: Accept batched image inputs `(B, C, H, W)`. Since evaluation forward passes are sample-independent, evaluating `score = model_output[:, target_index].sum()` yields mathematically identical per-sample gradients in a single batched pass. Returns 2D arrays `(H, W)` for single images ($B=1$) to maintain backwards compatibility, and 3D arrays `(B, H, W)` for mini-batches.
- **Lifecycle and Hook Reuse via Context Manager.**
  Equip `ModelsExplainer` with `__enter__` and `__exit__` context management to allow clean hook cleanup and explainer reuse across evaluation loops without per-image hook churn.
- **Shard across GPU Servers by Model.**
  `(model, variant)` units are independent, and `experiment-two --models/--variants` allows running disjoint subsets across available compute nodes (e.g., RTX 3090 and RTX 4090) simultaneously.
- **Proposed, Needs Decision: Per-Scale Stride for MPM.**
  Evaluate MPM occluders on a stride-$s_\sigma$ grid of centres and bilinearly upsample each per-scale map to $H \times W$. Adjacent $\sigma=65$ occluders overlap by ~98%, making stride-1 redundant at large scales:

  | Strides for $\sigma \in \{3, 17, 65\}$ | Forward passes per image | Speed-up |
  | -------------------------------------- | ------------------------ | -------- |
  | 1 / 1 / 1 (current default)            | 150,528                  | 1×       |
  | 1 / 4 / 16 ($s = \max(1, \sigma // 4)$)| 53,508                   | 2.8×     |
  | 2 / 4 / 16                             | 15,876                   | 9.5×     |
  | 2 / 8 / 32                             | 13,377                   | 11.3×    |

  Acceptance requires validating that (a) per-map Spearman/Pearson correlations against stride-1 maps remain high, and (b) downstream Experiment 2B inter/intra-model correlation matrices yield identical conclusions. Default remains stride 1 until formally adopted.

## Considered Options

- **Mixed precision (`autocast` bf16/fp16), rejected as a default.** MPM saliency measures minute differences between nearly identical scores; bf16 (8-bit mantissa) or fp16 (10-bit) quantization noise can exceed the signal for $\sigma=3$ occluders. Precision-preserving runtime options (`channels_last`, `torch.compile`, larger mini-batches) are preferred.
- **Incremental inference (Krypton), rejected.** Reusing intermediate activations outside an occluder's receptive field requires graph-level layer partitioning that compromises model agnosticism.
- **CPU multiprocessing, rejected.** Both workloads are GPU-compute-bound; CPU multiprocessing adds inter-process memory overhead without alleviating the bottleneck.

## Consequences

- Existing attribution heatmaps remain fully valid: both the vectorized MPM and vectorized Grad-CAM implementations are numerically identical to previous baselines within floating-point tolerance (max $|diff| < 10^{-6}$, bit-identical on identical hardware).
- Local benchmarks (Apple Silicon M-series, same inputs):

  | Component | Case / Architecture | Device | Baseline | Vectorized | Speed-up | Max \|Diff\| |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **MPM** | Trivial model, 224 px, $\sigma=17$ | CPU | 7.93 s | 1.77 s | **4.5×** | 0.0 |
  | **MPM** | Trivial model, 224 px, $\sigma=17$ | MPS | 16.74 s | 3.28 s | **5.1×** | 0.0 |
  | **MPM** | ResNet-18, 64 px, $\sigma \in \{3, 17\}$ | MPS | 4.50 s | 1.56 s | **2.9×** | 0.0 |
  | **Grad-CAM** | ResNet-18, 224 px, $B=1$ | MPS | 19.90 ms | 5.98 ms | **3.3×** | $7.15 \times 10^{-7}$ |
  | **Grad-CAM** | VGG-16, 224 px, $B=1$ | MPS | 101.28 ms | 22.59 ms | **4.48×** | 0.0 |
  | **Grad-CAM** | VGG-16, 224 px, $B=8$ (batched) | MPS | 101.28 ms | 16.06 ms/img | **6.31×** | 0.0 |

- The Grad-CAM implementation is robust against in-place activation layers and eliminates all parameter `.grad` allocation overhead during attribution generation.
- Adopting MPM strides remains an open proposal requiring experimental validation; if adopted, maps across Experiment 2 and 3D would be regenerated together.
