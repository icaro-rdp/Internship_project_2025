"""Explainability methods (Grad-CAM and Multiscale Pixel Masking) for saliency maps."""

import importlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .logger import debug, info, warn
from .normalization import normalize_data

T = TypeVar("T")

try:
    _tqdm_module = importlib.import_module("tqdm.auto")
    _tqdm = getattr(_tqdm_module, "tqdm", None)
except (ImportError, AttributeError):
    _tqdm = None


def _gpu_normalize(
    tensor: torch.Tensor, min_range: float = -1.0, max_range: float = 1.0
) -> torch.Tensor:
    """Min-max normalize a 2D ``(H, W)`` or 3D ``(B, H, W)`` tensor per spatial map."""
    is_2d = tensor.dim() == 2
    if is_2d:
        tensor = tensor.unsqueeze(0)

    flat = tensor.flatten(1)
    t_min, t_max = torch.aminmax(flat, dim=1, keepdim=True)
    t_min = t_min.unsqueeze(-1)
    t_max = t_max.unsqueeze(-1)

    denom = t_max - t_min
    safe_denom = torch.where(denom > 1e-12, denom, torch.ones_like(denom))
    scaled = (tensor - t_min) / safe_denom
    res = scaled * (max_range - min_range) + min_range
    res = torch.where(denom > 1e-12, res, torch.full_like(res, min_range))

    return res.squeeze(0) if is_2d else res


class ModelsExplainer(ABC):
    """Abstract base class for saliency map generation methods.

    Runs strictly in standard fp32 without mutating global CUDA/cuDNN backend
    flags (no TF32, AMP autocast, ``channels_last``, or ``torch.compile``),
    preventing sustained tensor-core power spikes on GPU servers and preserving
    floating-point precision for small occlusion score differences.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the explainer and place the model in evaluation mode.

        Args:
            model: PyTorch model returning ``(predictions, features)``.
        """
        self.model = model

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            warn("Could not determine model device. Assuming 'cpu'.")
            self.device = torch.device("cpu")

        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False

        self.model.eval()
        self.model.to(self.device)

    def _prepare_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move input tensor to the model's device in standard fp32."""
        return tensor.to(self.device)

    @abstractmethod
    def generate_map(
        self, input_tensor: torch.Tensor, target_index: int | Sequence[int] = 0
    ) -> np.ndarray:
        """Generate a saliency map for ``input_tensor``."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release any resources or hooks held by the explainer."""

    def __call__(
        self, input_tensor: torch.Tensor, target_index: int | Sequence[int] = 0
    ) -> np.ndarray:
        """Generate a saliency map by calling the explainer instance directly."""
        return self.generate_map(input_tensor, target_index)


class GradCAM(ModelsExplainer):
    """Gradient-weighted Class Activation Mapping (Grad-CAM) in fp32.

    Supports both single images ``(1, C, H, W)`` -> ``(H, W)`` and batches
    ``(B, C, H, W)`` -> ``(B, H, W)``, as well as 3D Vision Transformer token
    activations ``(B, 1 + H'*W', C)``. Uses ``torch.autograd.grad`` with respect
    to the target layer activation so backpropagation stops at ``target_layer``
    without mutating model parameter ``requires_grad`` flags.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str,
        relu: bool = True,
        gpu_normalize: bool = True,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize Grad-CAM and register the forward activation hook.

        Args:
            model: Network returning ``(predictions, features)``.
            target_layer: Target ``nn.Module`` or dotted attribute path.
            relu: If True, apply ReLU to the raw CAM and normalize to ``[0, 1]``;
                otherwise normalize to ``[-1, 1]``.
            gpu_normalize: Whether to run min-max normalization on the device.
            reshape_transform: Optional transform mapping 3D token tensors to
                4D spatial tensors ``(B, C, H', W')``.
        """
        super().__init__(model)

        self.target_layer = self._resolve_target_layer(target_layer)
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.relu = bool(relu)
        self.gpu_normalize = bool(gpu_normalize)
        self.reshape_transform = reshape_transform

        self.register_hooks()

    def _resolve_target_layer(self, target_layer: nn.Module | str) -> nn.Module:
        if isinstance(target_layer, nn.Module):
            return target_layer

        if not isinstance(target_layer, str):
            raise TypeError(
                "target_layer must be either an nn.Module or a dotted string path "
                f"(got type {type(target_layer).__name__})."
            )

        module: nn.Module = self.model
        for attr in target_layer.split("."):
            if attr.isdigit():
                index = int(attr)
                if not hasattr(module, "__getitem__"):
                    raise AttributeError(
                        f"Module '{module.__class__.__name__}' does not support "
                        f"indexing but received '{attr}'."
                    )
                module = module[index]  # type: ignore[index]
            else:
                if not hasattr(module, attr):
                    raise AttributeError(
                        f"Module '{module.__class__.__name__}' has no attribute "
                        f"'{attr}' while resolving target layer '{target_layer}'."
                    )
                module = getattr(module, attr)

            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"Resolved component '{attr}' within '{target_layer}' is not "
                    f"an nn.Module (obtained type {type(module).__name__})."
                )

        return module

    def register_hooks(self) -> None:
        """Attach a forward hook that clones target activations for autograd."""

        def forward_hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            if not torch.is_grad_enabled():
                return output
            self.activations = output.clone().requires_grad_(True)
            return self.activations

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))

    def generate_map(
        self,
        input_image: torch.Tensor,
        target_index: int | Sequence[int] | torch.Tensor = 0,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap(s) for a 4D input tensor ``(B, C, H, W)``."""
        if input_image.dim() != 4:
            raise ValueError(
                f"input_image must be a 4D tensor (B, C, H, W), got {input_image.shape}"
            )

        batch_size = input_image.shape[0]
        input_image = self._prepare_input(input_image)
        self.gradients = None
        self.activations = None

        with torch.enable_grad():
            model_output, _ = self.model(input_image)
            if isinstance(target_index, int):
                scores = model_output[:, target_index]
            else:
                idx = torch.as_tensor(target_index, device=self.device)
                scores = model_output[torch.arange(batch_size, device=self.device), idx]

            loss_scalar = scores.sum()

            if self.activations is None:
                msg = "Activations not captured. Check hook registration and target layer."
                warn(msg)
                raise RuntimeError(msg)

            self.gradients = torch.autograd.grad(
                outputs=loss_scalar,
                inputs=self.activations,
                retain_graph=False,
            )[0].detach()
            self.activations = self.activations.detach()

        with torch.no_grad():
            gradients = self.gradients.float()
            activations = self.activations.float()

            if self.reshape_transform is not None:
                gradients = self.reshape_transform(gradients)
                activations = self.reshape_transform(activations)
            elif gradients.dim() == 3:
                # ViT token reshape: (B, 1 + grid*grid, C) -> (B, C, grid, grid)
                b, n, c = gradients.shape
                grid_size = int((n - 1) ** 0.5)
                if grid_size * grid_size == n - 1:
                    gradients = (
                        gradients[:, 1:, :]
                        .reshape(b, grid_size, grid_size, c)
                        .permute(0, 3, 1, 2)
                    )
                    activations = (
                        activations[:, 1:, :]
                        .reshape(b, grid_size, grid_size, c)
                        .permute(0, 3, 1, 2)
                    )

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam_tensor = (weights * activations).sum(dim=1, keepdim=True)

            if self.relu:
                cam_tensor = F.relu(cam_tensor)

            target_size = (input_image.shape[2], input_image.shape[3])
            cam_resized = F.interpolate(
                cam_tensor,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            min_r = 0.0 if self.relu else -1.0
            if self.gpu_normalize:
                cam_resized = _gpu_normalize(
                    cam_resized, min_range=min_r, max_range=1.0
                )
                cam = (
                    cam_resized.squeeze(0).cpu().numpy()
                    if batch_size == 1
                    else cam_resized.cpu().numpy()
                )
            else:
                cam = (
                    cam_resized.squeeze(0).cpu().numpy()
                    if batch_size == 1
                    else cam_resized.cpu().numpy()
                )
                if batch_size == 1:
                    cam = normalize_data(cam, min_range=min_r, max_range=1.0)
                else:
                    cam = np.stack(
                        [
                            normalize_data(c, min_range=min_r, max_range=1.0)
                            for c in cam
                        ],
                        axis=0,
                    )

        return cam

    def cleanup(self) -> None:
        """Remove registered hooks and clear stored tensors."""
        info("Removing GradCAM hooks.")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = None
        self.activations = None


class MultiscalePixelMasking(ModelsExplainer):
    """Exact per-pixel Multiscale Pixel Masking (MPM) with GPU stability guardrails.

    Runs in pure fp32 under ``torch.no_grad()``, transfers batch score drops to
    host memory after each batch (naturally synchronizing the CUDA stream like
    the original implementation), and applies a small configurable cooldown pause
    on CUDA so long runs cannot saturate the GPU power limit continuously.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_list: Sequence[int],
        pixel_batch_size: int = 32,
        mask_value: float = 0.0,
        use_tqdm: bool = False,
        gpu_normalize: bool = False,
        batch_cooldown_sec: float = 0.002,
        cooldown_ratio: float = 0.20,
        stride: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize the MPM explainer.

        Args:
            model: Network returning ``(predictions, features)``.
            sigma_list: Side lengths, in pixels, of the square occluders.
            pixel_batch_size: Number of occluded copies evaluated per forward pass.
            mask_value: Fill value written into occluded regions.
            use_tqdm: Whether to display progress bars when ``tqdm`` is available.
            gpu_normalize: If True, normalize on device; defaults to False (CPU
                ``normalize_data`` for exact parity with the reference pipeline).
            batch_cooldown_sec: Minimum pause (in seconds) after each CUDA batch.
            cooldown_ratio: Fraction of each CUDA batch's compute time to sleep
                between batches (~0.45 caps GPU duty cycle at ~69% and keeps RTX
                4090 power draw below ~300W).
            stride: Optional spatial step size(s) for occluder centers. May be
                ``None`` (defaults to 1 for all scales), a positive ``int``, or
                a sequence of positive ints matching ``sigma_list`` (e.g.
                ``[4, 8, 16]``). Scales with ``stride > 1`` are bilinearly
                upsampled to ``(H, W)`` before summing.
        """
        super().__init__(model)
        self.sigma_list = list(sigma_list)
        if stride is None:
            self.strides = [1] * len(self.sigma_list)
        elif isinstance(stride, int):
            self.strides = [max(1, int(stride))] * len(self.sigma_list)
        else:
            self.strides = [max(1, int(s)) for s in stride]
            if len(self.strides) != len(self.sigma_list):
                raise ValueError(
                    f"stride length ({len(self.strides)}) must match sigma_list "
                    f"length ({len(self.sigma_list)})"
                )
        self.pixel_batch_size = int(pixel_batch_size)
        self.mask_value = float(mask_value)
        self.gpu_normalize = bool(gpu_normalize)
        self.batch_cooldown_sec = float(batch_cooldown_sec)
        self.cooldown_ratio = float(cooldown_ratio)
        self.use_tqdm = bool(use_tqdm and _tqdm is not None)
        if use_tqdm and _tqdm is None:
            warn(
                "tqdm is not available; disabling progress bars for MultiscalePixelMasking."
            )

    @staticmethod
    def _occlusion_masks(
        img_size: tuple[int, int], pixel_indices: torch.Tensor, sigma: int
    ) -> torch.Tensor:
        """Build boolean occlusion masks ``(B, 1, H, W)`` for a batch of centre pixels."""
        height, width = img_size
        rows = torch.arange(height, device=pixel_indices.device)
        cols = torch.arange(width, device=pixel_indices.device)
        center_rows = (pixel_indices // width).unsqueeze(1)
        center_cols = (pixel_indices % width).unsqueeze(1)
        return MultiscalePixelMasking._occlusion_masks_from_coords(
            rows, cols, center_rows, center_cols, sigma
        )

    @staticmethod
    def _occlusion_masks_from_coords(
        rows: torch.Tensor,
        cols: torch.Tensor,
        center_rows: torch.Tensor,
        center_cols: torch.Tensor,
        sigma: int,
    ) -> torch.Tensor:
        """Build boolean occlusion masks from precomputed row and column coordinates."""
        before, after = sigma // 2, (sigma + 1) // 2
        row_hit = (rows >= center_rows - before) & (rows < center_rows + after)
        col_hit = (cols >= center_cols - before) & (cols < center_cols + after)
        return (row_hit.unsqueeze(2) & col_hit.unsqueeze(1)).unsqueeze(1)

    def cleanup(self) -> None:
        """Release cached CUDA memory blocks if running on GPU."""
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_map(
        self,
        image_tensor: torch.Tensor,
        target_index: int = 0,
        normalize: bool = True,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
        """Compute the multiscale occlusion saliency map ``(H, W)`` for one image."""
        start_time = time.perf_counter()

        if image_tensor.dim() == 3:
            image = self._prepare_input(image_tensor.unsqueeze(0))
        elif image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image = self._prepare_input(image_tensor)
        else:
            raise ValueError(
                f"Input image tensor has unexpected dimensions: {image_tensor.shape}"
            )

        _, channels, height, width = image.shape
        total_pixels = height * width
        saliency_map_cpu = np.zeros(total_pixels, dtype=np.float32)

        with torch.no_grad():
            rows = torch.arange(height, device=self.device)
            cols = torch.arange(width, device=self.device)
            all_center_rows: torch.Tensor | None = None
            all_center_cols: torch.Tensor | None = None
            if 1 in self.strides:
                all_pixel_indices = torch.arange(total_pixels, device=self.device)
                all_center_rows = (all_pixel_indices // width).unsqueeze(1)
                all_center_cols = (all_pixel_indices % width).unsqueeze(1)
                del all_pixel_indices

            buffer_batch_size = min(self.pixel_batch_size, total_pixels)
            batch_buffer = torch.empty(
                (buffer_batch_size, channels, height, width),
                dtype=image.dtype,
                device=self.device,
            )
            mask_val_tensor = image.new_tensor(self.mask_value)

            original_output, _ = self.model(image)
            original_score = float(original_output[0, target_index].item())

            scale_pairs = list(zip(self.sigma_list, self.strides, strict=True))
            for sigma, step in self._progress(scale_pairs, desc="MPM sigma levels"):
                if step == 1:
                    assert all_center_rows is not None and all_center_cols is not None
                    center_rows = all_center_rows
                    center_cols = all_center_cols
                    num_centers = total_pixels
                    scale_scores = saliency_map_cpu
                else:
                    grid_r = torch.arange(0, height, step, device=self.device)
                    grid_c = torch.arange(0, width, step, device=self.device)
                    grid_h, grid_w = int(grid_r.numel()), int(grid_c.numel())
                    num_centers = grid_h * grid_w
                    mesh_r, mesh_c = torch.meshgrid(grid_r, grid_c, indexing="ij")
                    center_rows = mesh_r.reshape(-1, 1)
                    center_cols = mesh_c.reshape(-1, 1)
                    scale_scores = np.zeros(num_centers, dtype=np.float32)

                batch_starts = range(0, num_centers, self.pixel_batch_size)
                for batch_start in self._progress(batch_starts, f"MPM sigma {sigma}"):
                    t_batch_start = time.perf_counter()
                    batch_end = min(num_centers, batch_start + self.pixel_batch_size)
                    actual_batch_size = batch_end - batch_start

                    occluded = self._occlusion_masks_from_coords(
                        rows,
                        cols,
                        center_rows[batch_start:batch_end],
                        center_cols[batch_start:batch_end],
                        sigma,
                    )
                    masked_images = batch_buffer[:actual_batch_size]
                    torch.where(occluded, mask_val_tensor, image, out=masked_images)
                    output_batch, _ = self.model(masked_images)

                    batch_scores = (
                        output_batch[:, target_index]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                    )
                    scale_scores[batch_start:batch_end] += original_score - batch_scores

                    del occluded, output_batch
                    if self.is_cuda:
                        t_batch = time.perf_counter() - t_batch_start
                        sleep_dur = max(
                            self.batch_cooldown_sec, t_batch * self.cooldown_ratio
                        )
                        if sleep_dur > 0.0:
                            time.sleep(sleep_dur)

                if step > 1:
                    upsampled = F.interpolate(
                        torch.from_numpy(scale_scores).view(1, 1, grid_h, grid_w),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    saliency_map_cpu += upsampled.view(-1).numpy()

            del batch_buffer, mask_val_tensor

        saliency_map_numpy = saliency_map_cpu.reshape(height, width)

        if normalize:
            if self.gpu_normalize:
                saliency_map_numpy = (
                    _gpu_normalize(
                        torch.from_numpy(saliency_map_numpy),
                        min_range=-1.0,
                        max_range=1.0,
                    )
                    .cpu()
                    .numpy()
                )
            else:
                saliency_map_numpy = normalize_data(
                    saliency_map_numpy, min_range=-1.0, max_range=1.0
                )

        if log_debug_stats:
            debug(
                "MPM: returned %s map -> range [%.4f, %.4f] in %.2f seconds",
                "normalized" if normalize else "raw",
                float(saliency_map_numpy.min()),
                float(saliency_map_numpy.max()),
                time.perf_counter() - start_time,
            )
        return saliency_map_numpy

    def _progress(self, iterable: Iterable[T], desc: str) -> Iterable[T]:
        """Wrap an iterable in a transient tqdm bar when progress bars are enabled."""
        if self.use_tqdm and _tqdm is not None:
            return _tqdm(iterable, desc=desc, leave=False)
        return iterable
