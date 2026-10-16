# --- Imports ---
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
    """Fast GPU-resident min-max normalization per spatial map (B, H, W) or (H, W)."""
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
    """Abstract Base Class (Superclass) for saliency map generation methods."""

    def __init__(
        self,
        model: nn.Module,
        use_amp: bool = True,
        channels_last: bool = False,
        compile_model: bool = False,
        compile_mode: str = "default",
    ):
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

        self.use_amp = bool(use_amp and self.is_cuda)
        self.amp_dtype = (
            torch.bfloat16
            if (self.is_cuda and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.channels_last = bool(channels_last and self.is_cuda)

        self.model.eval()
        self.model.to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if compile_model and self.is_cuda and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=compile_mode)

    def _prepare_input(self, tensor: torch.Tensor) -> torch.Tensor:
        memory_format = (
            torch.channels_last
            if (self.channels_last and tensor.dim() == 4)
            else torch.preserve_format
        )
        return tensor.to(self.device, memory_format=memory_format, non_blocking=True)

    @abstractmethod
    def generate_map(
        self, input_tensor: torch.Tensor, target_index: int | Sequence[int] = 0
    ):
        raise NotImplementedError

    def cleanup(self):
        pass

    def __call__(
        self, input_tensor: torch.Tensor, target_index: int | Sequence[int] = 0
    ):
        return self.generate_map(input_tensor, target_index)


class GradCAM(ModelsExplainer):
    """GPU-accelerated Grad-CAM supporting both single images (B=1) and full batches (B >= 1),

    partial backpropagation (target_layer to head only), and fused GPU normalization.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str,
        relu: bool = True,
        use_amp: bool = True,
        channels_last: bool = False,
        freeze_parameters: bool = True,
        gpu_normalize: bool = True,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__(
            model, use_amp=use_amp, channels_last=channels_last, compile_model=False
        )

        self.target_layer = self._resolve_target_layer(target_layer)
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hooks = []
        self.relu = bool(relu)
        self.gpu_normalize = bool(gpu_normalize)
        self.reshape_transform = reshape_transform

        if freeze_parameters:
            self.model.requires_grad_(False)

        self.register_hooks()

    def _resolve_target_layer(self, target_layer: nn.Module | str) -> nn.Module:
        if isinstance(target_layer, nn.Module):
            return target_layer

        if not isinstance(target_layer, str):
            raise TypeError(
                "target_layer must be either an nn.Module or a dotted string path "
                f"(got type {type(target_layer).__name__})."
            )

        module = self.model
        for attr in target_layer.split("."):
            if attr.isdigit():
                index = int(attr)
                if not hasattr(module, "__getitem__"):
                    raise AttributeError(
                        f"Module '{module.__class__.__name__}' does not support indexing but received '{attr}'."
                    )
                module = module[index]
            else:
                if not hasattr(module, attr):
                    raise AttributeError(
                        f"Module '{module.__class__.__name__}' has no attribute '{attr}' while resolving target layer '{target_layer}'."
                    )
                module = getattr(module, attr)

            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"Resolved component '{attr}' within '{target_layer}' is not an nn.Module (obtained type {type(module).__name__})."
                )

        return module

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def register_hooks(self):
        def forward_hook(module, input, output):
            if not output.requires_grad:
                output = output.requires_grad_(True)
            self.activations = output.detach()
            output.register_hook(self._save_gradient)
            return output

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))

    def generate_map(
        self,
        input_image: torch.Tensor,
        target_index: int | Sequence[int] | torch.Tensor = 0,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
        """Generates Grad-CAM for a single image (1, C, H, W) -> returns (H, W)

        or a batch of images (B, C, H, W) -> returns (B, H, W).
        """
        if input_image.dim() != 4:
            raise ValueError(
                f"input_image must be a 4D tensor (B, C, H, W), got {input_image.shape}"
            )

        batch_size = input_image.shape[0]
        input_image = self._prepare_input(input_image)
        self.gradients = None
        self.activations = None

        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                model_output, _ = self.model(input_image)
                if isinstance(target_index, int):
                    scores = model_output[:, target_index]
                else:
                    idx = torch.as_tensor(target_index, device=self.device)
                    scores = model_output[
                        torch.arange(batch_size, device=self.device), idx
                    ]

                # Summing independent batch scores computes exact per-sample gradients in 1 backward pass
                loss_scalar = scores.sum()

            loss_scalar.backward()

        if self.gradients is None or self.activations is None:
            msg = "Gradients or activations not captured. Check hook registration and target layer."
            warn(msg)
            raise RuntimeError(msg)

        with torch.no_grad():
            gradients = self.gradients.float()
            activations = self.activations.float()

            if self.reshape_transform is not None:
                gradients = self.reshape_transform(gradients)
                activations = self.reshape_transform(activations)
            elif gradients.dim() == 3:
                # Default ViT token reshape if (B, N, C) where N-1 is a perfect square
                # e.g. 197 tokens -> 1 CLS + 196 patch tokens = 14x14 grid
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

            # (B, C, 1, 1)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            # (B, 1, H', W')
            cam_tensor = (weights * activations).sum(dim=1, keepdim=True)

            if self.relu:
                cam_tensor = F.relu(cam_tensor)

            target_size = (input_image.shape[2], input_image.shape[3])
            cam_resized = F.interpolate(
                cam_tensor,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (B, H, W)

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

    def cleanup(self):
        info("Removing GradCAM hooks.")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = None
        self.activations = None


class MultiscalePixelMasking(ModelsExplainer):
    """Exact per-pixel Multiscale Pixel Masking (MPM) with GPU batching and periodic queue synchronization."""

    def __init__(
        self,
        model: nn.Module,
        sigma_list: Sequence[int],
        pixel_batch_size: int = 128,
        mask_value: float = 0.0,
        use_tqdm: bool = False,
        use_amp: bool = True,
        channels_last: bool = False,
        compile_model: bool = False,
        gpu_normalize: bool = True,
        sync_interval: int = 32,
    ) -> None:
        super().__init__(
            model,
            use_amp=use_amp,
            channels_last=channels_last,
            compile_model=compile_model,
            compile_mode="default",
        )
        self.sigma_list = list(sigma_list)
        self.pixel_batch_size = int(pixel_batch_size)
        self.mask_value = float(mask_value)
        self.gpu_normalize = bool(gpu_normalize)
        self.sync_interval = int(sync_interval)
        self.use_tqdm = bool(use_tqdm and _tqdm is not None)
        if use_tqdm and _tqdm is None:
            warn(
                "tqdm is not available; disabling progress bars for MultiscalePixelMasking."
            )

        # Persistent batch buffer cached across calls of the same image shape
        self._cached_buffer: torch.Tensor | None = None

    @staticmethod
    def _occlusion_masks(
        img_size: tuple[int, int], pixel_indices: torch.Tensor, sigma: int
    ) -> torch.Tensor:
        """Build boolean occlusion masks for a batch of centre pixels."""
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
        """Build boolean occlusion masks from precomputed coordinate vectors."""
        before, after = sigma // 2, (sigma + 1) // 2
        row_hit = (rows >= center_rows - before) & (rows < center_rows + after)
        col_hit = (cols >= center_cols - before) & (cols < center_cols + after)
        return (row_hit.unsqueeze(2) & col_hit.unsqueeze(1)).unsqueeze(1)

    def _get_batch_buffer(
        self, channels: int, height: int, width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        buffer_batch_size = min(self.pixel_batch_size, height * width)
        target_shape = (buffer_batch_size, channels, height, width)
        if (
            self._cached_buffer is None
            or self._cached_buffer.shape != target_shape
            or self._cached_buffer.dtype != dtype
            or self._cached_buffer.device != self.device
        ):
            memory_format = (
                torch.channels_last if self.channels_last else torch.contiguous_format
            )
            self._cached_buffer = torch.empty(
                target_shape,
                dtype=dtype,
                device=self.device,
                memory_format=memory_format,
            )
        return self._cached_buffer

    def cleanup(self) -> None:
        """Release cached GPU batch buffer."""
        self._cached_buffer = None

    def generate_map(
        self,
        image_tensor: torch.Tensor,
        target_index: int = 0,
        normalize: bool = True,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
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

        with torch.inference_mode():
            rows = torch.arange(height, device=self.device)
            cols = torch.arange(width, device=self.device)
            all_pixel_indices = torch.arange(total_pixels, device=self.device)
            all_center_rows = (all_pixel_indices // width).unsqueeze(1)
            all_center_cols = (all_pixel_indices % width).unsqueeze(1)
            del all_pixel_indices

            saliency_map_gpu = torch.zeros(
                total_pixels, dtype=torch.float32, device=self.device
            )
            batch_buffer = self._get_batch_buffer(channels, height, width, image.dtype)
            mask_val_tensor = image.new_tensor(self.mask_value)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                original_output, _ = self.model(image)
                original_score = original_output[0, target_index].float()

                for sigma in self._progress(self.sigma_list, desc="MPM sigma levels"):
                    batch_starts = range(0, total_pixels, self.pixel_batch_size)
                    for step_idx, batch_start in enumerate(
                        self._progress(batch_starts, f"MPM sigma {sigma}"), start=1
                    ):
                        batch_end = min(
                            total_pixels, batch_start + self.pixel_batch_size
                        )
                        actual_batch_size = batch_end - batch_start

                        occluded = self._occlusion_masks_from_coords(
                            rows,
                            cols,
                            all_center_rows[batch_start:batch_end],
                            all_center_cols[batch_start:batch_end],
                            sigma,
                        )

                        masked_view = batch_buffer[:actual_batch_size]
                        torch.where(occluded, mask_val_tensor, image, out=masked_view)

                        output_batch, _ = self.model(masked_view)
                        saliency_map_gpu[batch_start:batch_end] += (
                            original_score - output_batch[:, target_index].float()
                        )

                        if (
                            self.is_cuda
                            and self.sync_interval > 0
                            and step_idx % self.sync_interval == 0
                        ):
                            torch.cuda.synchronize(self.device)

                    if self.is_cuda:
                        torch.cuda.synchronize(self.device)

            saliency_map_gpu = saliency_map_gpu.view(height, width)

            if normalize:
                if self.gpu_normalize:
                    saliency_map_gpu = _gpu_normalize(
                        saliency_map_gpu, min_range=-1.0, max_range=1.0
                    )
                    saliency_map_numpy = saliency_map_gpu.cpu().numpy()
                else:
                    saliency_map_numpy = normalize_data(
                        saliency_map_gpu.cpu().numpy(), min_range=-1.0, max_range=1.0
                    )
            else:
                saliency_map_numpy = saliency_map_gpu.cpu().numpy()

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
        if self.use_tqdm:
            return _tqdm(iterable, desc=desc, leave=False)
        return iterable
