# --- Imports ---
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import importlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .logger import debug, error, info, warn
from .normalization import normalize_data

try:
    _tqdm_module = importlib.import_module("tqdm.auto")
    _tqdm = getattr(_tqdm_module, "tqdm", None)
except Exception:
    _tqdm = None

# Enable TF32 on Ampere/Ada GPUs (RTX 4090) for immediate FP32 matmul/conv speedups
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class ModelsExplainer(ABC):
    """
    Abstract Base Class (Superclass) for saliency map generation methods.
    """

    def __init__(
        self,
        model: nn.Module,
        use_amp: bool = True,
        channels_last: bool = True,
        compile_model: bool = False,
    ):
        self.model = model

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            warn("Could not determine model device. Assuming 'cpu'.")
            self.device = torch.device("cpu")

        self.is_cuda = self.device.type == "cuda"
        self.use_amp = bool(use_amp and self.is_cuda)
        # RTX 4090 natively supports bfloat16 with full FP32 dynamic range
        self.amp_dtype = (
            torch.bfloat16
            if (self.is_cuda and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.channels_last = bool(channels_last and self.is_cuda)

        self.model.eval()
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if compile_model and self.is_cuda and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def _prepare_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves input tensor to device and applies channels_last if enabled."""
        memory_format = (
            torch.channels_last
            if (self.channels_last and tensor.dim() == 4)
            else torch.preserve_format
        )
        return tensor.to(self.device, memory_format=memory_format, non_blocking=True)

    @abstractmethod
    def generate_map(self, input_tensor: torch.Tensor, target_index: int = 0):
        raise NotImplementedError

    def cleanup(self):
        pass

    def __call__(self, input_tensor: torch.Tensor, target_index: int = 0):
        return self.generate_map(input_tensor, target_index)


class GradCAM(ModelsExplainer):
    """
    Implements GPU-accelerated Grad-CAM with partial backpropagation.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str,
        relu: bool = True,
        use_amp: bool = True,
        channels_last: bool = True,
        freeze_parameters: bool = True,
    ):
        super().__init__(
            model, use_amp=use_amp, channels_last=channels_last, compile_model=False
        )

        self.target_layer = self._resolve_target_layer(target_layer)
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hooks = []
        self.relu = bool(relu)

        # Freezing parameters stops autograd from computing weight gradients (dW)
        # and prevents backpropagating earlier than target_layer
        if freeze_parameters:
            self.model.requires_grad_(False)

        self.register_hooks()

    def _resolve_target_layer(self, target_layer: nn.Module | str) -> nn.Module:
        """Converts a string path into a module reference if needed."""
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
        """Attaches a forward hook that captures activations and hooks their gradients directly."""

        def forward_hook(module, input, output):
            # Ensure autograd tracks from target_layer to the model output
            # even when backbone parameters and input_image do not require grad
            if not output.requires_grad:
                output = output.requires_grad_(True)
            self.activations = output.detach()
            output.register_hook(self._save_gradient)
            return output

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))

    def generate_map(
        self,
        input_image: torch.Tensor,
        target_index: int = 0,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
        """
        Generates the Class Activation Map (CAM) entirely on GPU before final NumPy export.
        """
        if input_image.dim() != 4 or input_image.shape[0] != 1:
            raise ValueError(
                "input_image must be a 4D tensor with batch size 1 (B, C, H, W)"
            )

        input_image = self._prepare_input(input_image)
        self.gradients = None
        self.activations = None

        # 1. Forward & Backward pass (only from target_layer -> output head if params are frozen)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                model_output, _ = self.model(input_image)
                score = model_output[0, target_index]

            score.backward()

        # 2. Check hooks
        if self.gradients is None:
            msg = "Gradients not captured. Check hook registration and target layer."
            warn(msg)
            raise RuntimeError(msg)
        if self.activations is None:
            msg = "Activations not captured. Check hook registration."
            warn(msg)
            raise RuntimeError(msg)

        # 3. Compute weights & CAM entirely on GPU in FP32
        with torch.no_grad():
            gradients = self.gradients.float()
            activations = self.activations.float()

            # Global average pooling over spatial dimensions (H, W) -> (1, C, 1, 1)
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted channel combination -> (1, 1, H', W')
            cam_tensor = (weights * activations).sum(dim=1, keepdim=True)

            if log_debug_stats:
                raw_min, raw_max = torch._aminmax(cam_tensor)
                grad_min, grad_max = torch._aminmax(gradients)
                act_min, act_max = torch._aminmax(activations)

            # 4. Apply ReLU on GPU
            if self.relu:
                cam_tensor = F.relu(cam_tensor)
                if log_debug_stats:
                    post_min, post_max = torch._aminmax(cam_tensor)

            # 5. Resize CAM on GPU
            target_size = (input_image.shape[2], input_image.shape[3])
            cam_resized = F.interpolate(
                cam_tensor,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            # Single GPU -> CPU transfer at the very end
            cam = cam_resized.squeeze(0).squeeze(0).cpu().numpy()

        # 6. Normalize CAM
        if self.relu:
            cam = normalize_data(cam, min_range=0, max_range=1)
        else:
            cam = normalize_data(cam, min_range=-1, max_range=1)

        if log_debug_stats:
            final_min = float(cam.min())
            final_max = float(cam.max())
            if self.relu:
                debug(
                    "GradCAM stats -> raw[%.4f, %.4f] post_relu[%.4f, %.4f] final[%.4f, %.4f] gradients[%.4f, %.4f] activations[%.4f, %.4f]"
                    % (
                        raw_min.item(),
                        raw_max.item(),
                        post_min.item(),
                        post_max.item(),
                        final_min,
                        final_max,
                        grad_min.item(),
                        grad_max.item(),
                        act_min.item(),
                        act_max.item(),
                    )
                )
            else:
                debug(
                    "GradCAM stats -> raw[%.4f, %.4f] final[%.4f, %.4f] gradients[%.4f, %.4f] activations[%.4f, %.4f]"
                    % (
                        raw_min.item(),
                        raw_max.item(),
                        final_min,
                        final_max,
                        grad_min.item(),
                        grad_max.item(),
                        act_min.item(),
                        act_max.item(),
                    )
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
    """Multiscale occlusion saliency (Multiscale Pixel Masking, MPM) optimized for GPU."""

    def __init__(
        self,
        model: nn.Module,
        sigma_list: Sequence[int],
        pixel_batch_size: int,
        mask_value: float = 0.0,
        use_tqdm: bool = True,
        use_amp: bool = True,
        channels_last: bool = True,
        compile_model: bool = False,
    ) -> None:
        super().__init__(
            model,
            use_amp=use_amp,
            channels_last=channels_last,
            compile_model=compile_model,
        )
        self.sigma_list = sigma_list
        self.pixel_batch_size = pixel_batch_size
        self.mask_value = mask_value
        self.use_tqdm = bool(use_tqdm and _tqdm is not None)
        if use_tqdm and _tqdm is None:
            warn(
                "tqdm is not available; disabling progress bars for MultiscalePixelMasking."
            )

    @staticmethod
    def _occlusion_masks(
        rows: torch.Tensor,
        cols: torch.Tensor,
        center_rows: torch.Tensor,
        center_cols: torch.Tensor,
        sigma: int,
    ) -> torch.Tensor:
        """Build boolean occlusion masks using precomputed coordinate vectors."""
        before, after = sigma // 2, (sigma + 1) // 2
        row_hit = (rows >= center_rows - before) & (rows < center_rows + after)
        col_hit = (cols >= center_cols - before) & (cols < center_cols + after)
        return (row_hit.unsqueeze(2) & col_hit.unsqueeze(1)).unsqueeze(1)

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

        debug(
            "MPM: prepared run (target index %d, sigmas=%s, batch=%d)",
            target_index,
            list(self.sigma_list),
            self.pixel_batch_size,
        )

        _, channels, height, width = image.shape
        total_pixels = height * width

        with torch.inference_mode():
            # Precompute row/col coordinate tables once on the GPU
            rows = torch.arange(height, device=self.device)
            cols = torch.arange(width, device=self.device)
            all_pixel_indices = torch.arange(total_pixels, device=self.device)
            all_center_rows = (all_pixel_indices // width).unsqueeze(1)
            all_center_cols = (all_pixel_indices % width).unsqueeze(1)

            saliency_map_final = torch.zeros(
                total_pixels, dtype=torch.float32, device=self.device
            )

            # Pre-allocate batch buffer once to avoid repeated CUDA memory allocations
            memory_format = (
                torch.channels_last if self.channels_last else torch.contiguous_format
            )
            batch_buffer = torch.empty(
                (self.pixel_batch_size, channels, height, width),
                dtype=image.dtype,
                device=self.device,
                memory_format=memory_format,
            )

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                original_output, _ = self.model(image)
                original_score = original_output[0, target_index].float()

                for sigma in self._progress(self.sigma_list, desc="MPM sigma levels"):
                    batch_starts = range(0, total_pixels, self.pixel_batch_size)
                    for batch_start in self._progress(
                        batch_starts, f"MPM sigma {sigma}"
                    ):
                        batch_end = min(
                            total_pixels, batch_start + self.pixel_batch_size
                        )
                        actual_batch_size = batch_end - batch_start

                        occluded = self._occlusion_masks(
                            rows,
                            cols,
                            all_center_rows[batch_start:batch_end],
                            all_center_cols[batch_start:batch_end],
                            sigma,
                        )

                        # Write directly into the pre-allocated slice (zero allocation)
                        masked_view = batch_buffer[:actual_batch_size]
                        torch.where(occluded, self.mask_value, image, out=masked_view)

                        output_batch, _ = self.model(masked_view)
                        saliency_map_final[batch_start:batch_end] += (
                            original_score - output_batch[:, target_index].float()
                        )

                    if log_debug_stats:
                        s_min, s_max = torch._aminmax(saliency_map_final)
                        debug(
                            "MPM: cumulative after sigma=%s -> min %.4f | max %.4f | mean %.4f",
                            sigma,
                            s_min.item(),
                            s_max.item(),
                            saliency_map_final.mean().item(),
                        )

        saliency_map_numpy = saliency_map_final.reshape(height, width).cpu().numpy()

        if normalize:
            saliency_map_numpy = normalize_data(
                saliency_map_numpy, min_range=-1, max_range=1
            )

        if log_debug_stats:
            debug(
                "MPM: returned %s map -> range [%.4f, %.4f] in %.2f seconds",
                "normalized" if normalize else "raw",
                saliency_map_numpy.min(),
                saliency_map_numpy.max(),
                time.perf_counter() - start_time,
            )
        return saliency_map_numpy

    def _progress[T](self, iterable: Iterable[T], desc: str) -> Iterable[T]:
        if self.use_tqdm:
            return _tqdm(iterable, desc=desc, leave=False)
        return iterable
