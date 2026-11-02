"""Explainability methods (Grad-CAM, Multiscale Pixel Masking, and LIME) for saliency maps."""

import importlib
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from torch import nn

from Image_Authenticity_prediction.main.Utils.logger import debug, info, warn
from Image_Authenticity_prediction.main.Utils.normalization import normalize_data

T = TypeVar("T")

try:
    _tqdm_module = importlib.import_module("tqdm.auto")
    _tqdm = getattr(_tqdm_module, "tqdm", None)
except (ImportError, AttributeError):
    _tqdm = None


class ModelsExplainer(ABC):
    """Abstract base class for saliency map generation methods.

    Keeps execution simple and deterministic in standard fp32 without AMP
    autocast, ``channels_last`` conversions, ``torch.compile`` background
    workers, or cuDNN autotuning benchmarks.
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
        self.model.eval()
        self.model.to(device=self.device, dtype=torch.float32)

    def _prepare_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move input tensor to the model's device in standard fp32."""
        return tensor.to(device=self.device, dtype=torch.float32)

    @abstractmethod
    def generate_map(
        self, input_tensor: torch.Tensor, target_index: int | Sequence[int] = 0
    ) -> np.ndarray:
        """Generate a saliency map for ``input_tensor``."""
        raise NotImplementedError

    def cleanup(self, failed: bool = False) -> None:
        """Release any resources or hooks held by the explainer.

        Args:
            failed: Whether cleanup is being invoked following a failure. Subclasses
                can bypass hardware calls to avoid cascading exceptions on an unstable
                or poisoned GPU context.
        """

    def __call__(
        self, input_tensor: torch.Tensor, target_index: int | Sequence[int] = 0
    ) -> np.ndarray:
        """Generate a saliency map by calling the explainer instance directly."""
        return self.generate_map(input_tensor, target_index)


class GradCAM(ModelsExplainer):
    """Gradient-weighted Class Activation Mapping (Grad-CAM) with truncated autograd.

    Supports single images ``(1, C, H, W)`` -> ``(H, W)``, batches
    ``(B, C, H, W)`` -> ``(B, H, W)``, and 3D Vision Transformer token
    activations ``(B, 1 + H'*W', C)``. Computes gradients only with respect to
    ``target_layer`` activations via ``torch.autograd.grad`` without mutating
    model parameter ``requires_grad`` flags.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str,
        relu: bool = True,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the Grad-CAM explainer.

        Args:
            model: PyTorch model returning ``(predictions, features)``.
            target_layer: Layer module or dotted path to hook.
            relu: Whether to apply ReLU to the raw weighted activation map.
            reshape_transform: Optional transform converting 3D token tensors
                into 4D spatial feature maps ``(B, C, H', W')``.
        """
        super().__init__(model)

        self.target_layer = self._resolve_target_layer(target_layer)
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.relu = bool(relu)
        self.reshape_transform = reshape_transform
        self._is_capturing = False

        self.register_hooks()

    def _resolve_target_layer(self, target_layer: nn.Module | str) -> nn.Module:
        """Resolve a dotted string path into its corresponding ``nn.Module``."""
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
        """Attach a gated forward hook that isolates target layer activations."""

        def forward_hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            if not self._is_capturing or not torch.is_grad_enabled():
                return output
            self.activations = output.detach().requires_grad_(True)
            return self.activations.clone()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))

    def generate_map(
        self,
        input_image: torch.Tensor,
        target_index: int | Sequence[int] | torch.Tensor = 0,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap(s) for a 4D input tensor ``(B, C, H, W)``.

        Args:
            input_image: 4D image batch tensor of shape ``(B, C, H, W)``.
            target_index: Target class index or sequence of indices for the batch.
            log_debug_stats: Whether to log execution duration and output map range.

        Returns:
            Normalized Grad-CAM heatmap array of shape ``(H, W)`` for single images
            or ``(B, H, W)`` for multi-image batches.

        Raises:
            ValueError: If ``input_image`` is not a 4D tensor.
            RuntimeError: If target layer activations were not captured.
        """
        start_time = time.perf_counter()
        if input_image.dim() != 4:
            raise ValueError(
                f"input_image must be a 4D tensor (B, C, H, W), got {input_image.shape}"
            )

        batch_size = input_image.shape[0]
        input_image = self._prepare_input(input_image)
        self.gradients = None
        self.activations = None

        self._is_capturing = True
        try:
            with torch.enable_grad():
                model_output, _ = self.model(input_image)
                if isinstance(target_index, int):
                    scores = model_output[:, target_index]
                else:
                    idx = torch.as_tensor(
                        target_index, dtype=torch.long, device=self.device
                    )
                    scores = model_output[
                        torch.arange(batch_size, device=self.device), idx
                    ]

                loss_scalar = scores.sum()

                if self.activations is None:
                    msg = (
                        "Activations not captured. Check hook registration and "
                        "target layer."
                    )
                    warn(msg)
                    raise RuntimeError(msg)

                self.gradients = torch.autograd.grad(
                    outputs=loss_scalar,
                    inputs=self.activations,
                    retain_graph=False,
                )[0].detach()
                self.activations = self.activations.detach()
        finally:
            self._is_capturing = False

        with torch.no_grad():
            gradients = self.gradients.float()
            activations = self.activations.float()

            if self.reshape_transform is not None:
                gradients = self.reshape_transform(gradients)
                activations = self.reshape_transform(activations)
            elif gradients.dim() == 3:
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

            cam_np = (
                cam_resized.squeeze(0).cpu().numpy()
                if batch_size == 1
                else cam_resized.cpu().numpy()
            )

        self.gradients = None
        self.activations = None

        min_r = 0.0 if self.relu else -1.0
        if batch_size == 1:
            cam_out = normalize_data(cam_np, min_range=min_r, max_range=1.0)
        else:
            cam_out = np.stack(
                [normalize_data(c, min_range=min_r, max_range=1.0) for c in cam_np],
                axis=0,
            )

        if log_debug_stats:
            debug(
                "GradCAM: returned map -> range [%.4f, %.4f] in %.2f seconds",
                float(cam_out.min()),
                float(cam_out.max()),
                time.perf_counter() - start_time,
            )
        return cam_out

    def cleanup(self, failed: bool = False) -> None:
        """Remove registered hooks and clear stored tensors.

        Args:
            failed: Whether cleanup is invoked after an error.
        """
        info("Removing GradCAM hooks.")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = None
        self.activations = None


class MultiscalePixelMasking(ModelsExplainer):
    """Safe, vectorized Multiscale Pixel Masking (MPM) with per-scale striding.

    Key algorithmic features:
    1. Vectorized 2D boolean mask construction on device (``row_hit & col_hit``)
       combined with pre-allocated buffer writing via ``torch.where(..., out=masked_images)``.
    2. Optional per-scale spatial ``stride`` (e.g. ``[2, 4, 16]``) with bilinear
       upsampling back to ``(H, W)``.
    3. Batched GPU score accumulation across each scale level, transferring to host
       CPU RAM per scale level.

    Avoids ``torch.compile`` worker processes, AMP autocast, and ``channels_last``
    memory conversions to ensure deterministic and predictable execution.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_list: Sequence[int],
        pixel_batch_size: int = 64,
        mask_value: float = 0.0,
        use_tqdm: bool = False,
        stride: int | Sequence[int] | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the MPM explainer.

        Args:
            model: Network returning ``(predictions, features)``.
            sigma_list: Side lengths, in pixels, of the square occluders. Must be positive.
            pixel_batch_size: Number of occluded copies evaluated per forward pass.
                Must be positive.
            mask_value: Fill value written into occluded regions.
            use_tqdm: Whether to display progress bars when ``tqdm`` is available.
            stride: Optional spatial step size(s) for occluder centers. Must be positive.

        Raises:
            ValueError: If ``pixel_batch_size`` or any occluder/stride size is non-positive,
                or if ``sigma_list`` is empty.
        """
        super().__init__(model)
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        if pixel_batch_size <= 0:
            raise ValueError(
                f"pixel_batch_size must be positive, got {pixel_batch_size}"
            )
        self.pixel_batch_size = int(pixel_batch_size)

        if not sigma_list:
            raise ValueError("sigma_list cannot be empty")
        self.sigma_list = [int(s) for s in sigma_list]
        if any(s <= 0 for s in self.sigma_list):
            raise ValueError(
                f"All occluder sizes in sigma_list must be positive, got {sigma_list}"
            )

        if stride is None:
            self.strides = [1] * len(self.sigma_list)
        elif isinstance(stride, int):
            if stride <= 0:
                raise ValueError(f"stride must be positive, got {stride}")
            self.strides = [int(stride)] * len(self.sigma_list)
        else:
            stride_list = [int(s) for s in stride]
            if len(stride_list) != len(self.sigma_list):
                raise ValueError(
                    f"stride length ({len(stride_list)}) must match sigma_list "
                    f"length ({len(self.sigma_list)})"
                )
            if any(s <= 0 for s in stride_list):
                raise ValueError(f"All stride values must be positive, got {stride}")
            self.strides = stride_list

        self.mask_value = float(mask_value)
        self.use_tqdm = bool(use_tqdm and _tqdm is not None)
        if use_tqdm and _tqdm is None:
            warn(
                "tqdm is not available; disabling progress bars for "
                "MultiscalePixelMasking."
            )

    @staticmethod
    def _occlusion_masks(
        img_size: tuple[int, int], pixel_indices: torch.Tensor, sigma: int
    ) -> torch.Tensor:
        """Build boolean occlusion masks ``(B, 1, H, W)`` for flat centre indices."""
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
        """Build boolean occlusion masks from row and column coordinate vectors."""
        before, after = sigma // 2, (sigma + 1) // 2
        row_hit = (rows >= center_rows - before) & (rows < center_rows + after)
        col_hit = (cols >= center_cols - before) & (cols < center_cols + after)
        return (row_hit.unsqueeze(2) & col_hit.unsqueeze(1)).unsqueeze(1)

    def cleanup(self, failed: bool = False) -> None:
        """Release cached CUDA memory blocks if running on GPU.

        Args:
            failed: If True or an exception is in flight, avoids calling CUDA
                APIs to prevent cascading secondary failures on an unstable context.
        """
        if failed or sys.exc_info()[0] is not None:
            return
        if self.is_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (RuntimeError, torch.cuda.CudaError) as e:
                warn("CUDA empty_cache failed during cleanup: %s", e)

    def generate_map(
        self,
        image_tensor: torch.Tensor,
        target_index: int = 0,
        normalize: bool = True,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
        """Compute the multiscale occlusion saliency map ``(H, W)`` for one image.

        Args:
            image_tensor: Single image tensor of shape ``(C, H, W)`` or ``(1, C, H, W)``.
            target_index: Target regression or classification output index.
            normalize: Whether to min-max normalize the saliency map to ``[-1.0, 1.0]``.
            log_debug_stats: Whether to log execution duration and output map range.

        Returns:
            Occlusion saliency map array of shape ``(H, W)``.

        Raises:
            ValueError: If ``image_tensor`` has unexpected dimensions.
        """
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
                dtype=torch.float32,
                device=self.device,
            )
            mask_val_tensor = torch.tensor(
                self.mask_value, dtype=torch.float32, device=self.device
            )

            if self.use_amp and self.is_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    original_output, _ = self.model(image)
            else:
                original_output, _ = self.model(image)
            original_score_t = original_output[0, target_index].detach()

            scale_pairs = list(zip(self.sigma_list, self.strides, strict=True))
            for sigma, step in self._progress(scale_pairs, desc="MPM sigma levels"):
                if step == 1:
                    assert all_center_rows is not None and all_center_cols is not None
                    center_rows = all_center_rows
                    center_cols = all_center_cols
                    num_centers = total_pixels
                    grid_h, grid_w = height, width
                else:
                    grid_r = torch.arange(0, height, step, device=self.device)
                    grid_c = torch.arange(0, width, step, device=self.device)
                    grid_h, grid_w = int(grid_r.numel()), int(grid_c.numel())
                    num_centers = grid_h * grid_w
                    mesh_r, mesh_c = torch.meshgrid(grid_r, grid_c, indexing="ij")
                    center_rows = mesh_r.reshape(-1, 1)
                    center_cols = mesh_c.reshape(-1, 1)

                scale_scores_gpu = torch.zeros(
                    num_centers, dtype=torch.float32, device=self.device
                )

                batch_starts = range(0, num_centers, self.pixel_batch_size)
                for batch_idx, batch_start in enumerate(
                    self._progress(batch_starts, f"MPM sigma {sigma}")
                ):
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
                    if self.use_amp and self.is_cuda:
                        with torch.amp.autocast(
                            device_type="cuda", dtype=self.amp_dtype
                        ):
                            output_batch, _ = self.model(masked_images)
                    else:
                        output_batch, _ = self.model(masked_images)

                    scale_scores_gpu[batch_start:batch_end] = (
                        original_score_t - output_batch[:, target_index].detach()
                    )

                    del occluded, output_batch
                    if self.is_cuda and (batch_idx + 1) % 64 == 0:
                        torch.cuda.synchronize(self.device)

                if step == 1:
                    saliency_map_cpu += scale_scores_gpu.cpu().numpy()
                else:
                    upsampled = F.interpolate(
                        scale_scores_gpu.view(1, 1, grid_h, grid_w),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    saliency_map_cpu += upsampled.view(-1).cpu().numpy()

                del scale_scores_gpu

            del batch_buffer, mask_val_tensor

        saliency_map_numpy = saliency_map_cpu.reshape(height, width)

        if normalize:
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


def compute_slic_segments(
    img_rgb: np.ndarray,
    n_segments: int = 150,
    compactness: float = 10.0,
) -> np.ndarray:
    """Compute canonical SLIC superpixel labels for an RGB image.

    Matches the reference LIME protocol by converting ``uint8`` images to
    ``float32`` in ``[0, 1]`` and invoking ``skimage.segmentation.slic`` with
    ``start_label=0``.

    Args:
        img_rgb: 3D RGB image array of shape ``(H, W, 3)``, either ``uint8`` in
            ``[0, 255]`` or floating-point in ``[0, 1]``.
        n_segments: Target number of SLIC superpixels. Must be positive.
        compactness: Balance between color similarity and spatial proximity.

    Returns:
        2D array of shape ``(H, W)`` with ``int16`` superpixel labels in
        ``0 .. K-1``.

    Raises:
        ValueError: If ``img_rgb`` is not a 3D RGB array or parameters are invalid.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must have shape (H, W, 3), got {img_rgb.shape}")
    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")
    if compactness <= 0.0:
        raise ValueError(f"compactness must be positive, got {compactness}")

    if img_rgb.dtype == np.uint8 or float(np.nanmax(img_rgb)) > 1.0:
        img_float = img_rgb.astype(np.float32) / 255.0
    else:
        img_float = img_rgb.astype(np.float32)

    segments = slic(
        img_float,
        n_segments=int(n_segments),
        compactness=float(compactness),
        start_label=0,
    )
    return np.asarray(segments, dtype=np.int16)


def weighted_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the locality-weighted coefficient of determination ($R^2$).

    Args:
        y_true: 1D array of ground-truth perturbation score shifts.
        y_pred: 1D array of linear surrogate predictions.
        weights: 1D array of non-negative locality kernel weights.

    Returns:
        Weighted $R^2$ scalar as a standard Python float.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    wsum = float(w.sum()) + 1e-12
    mu = float((w * y_t).sum()) / wsum
    sse = float((w * (y_t - y_p) ** 2).sum())
    sst = float((w * (y_t - mu) ** 2).sum())
    return float(1.0 - sse / (sst + 1e-12))


class LIME(ModelsExplainer):
    """GPU-vectorized Local Interpretable Model-agnostic Explanations (LIME).

    Implements superpixel-based LIME for continuous regression using canonical
    ``224 x 224`` SLIC superpixels, binary Bernoulli perturbation sampling,
    per-image spatial-mean RGB occlusion, optional post-masking spatial resize
    (e.g. ``300 x 300`` for DenseNet-161), and a locality-weighted Ridge
    regression surrogate.
    """

    def __init__(
        self,
        model: nn.Module,
        n_segments: int = 150,
        compactness: float = 10.0,
        num_perturbations: int = 1200,
        keep_prob: float = 0.7,
        kernel_width: float = 0.25,
        ridge_alpha: float = 1.0,
        seed: int = 0,
        batch_size: int = 64,
        model_input_size: int | tuple[int, int] | None = None,
        norm_mean: Sequence[float] = (0.485, 0.456, 0.406),
        norm_std: Sequence[float] = (0.229, 0.224, 0.225),
        use_tqdm: bool = False,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the LIME explainer.

        Args:
            model: PyTorch model returning ``(predictions, features)``.
            n_segments: Target number of SLIC superpixels. Must be positive.
            compactness: SLIC compactness parameter. Must be positive.
            num_perturbations: Number of binary perturbation masks ($M > 1$).
            keep_prob: Bernoulli retention probability per superpixel in ``(0, 1)``.
            kernel_width: Exponential locality kernel bandwidth ($\\sigma > 0$).
            ridge_alpha: L2 regularization strength for the Ridge surrogate.
            seed: Random seed for deterministic perturbation mask generation.
            batch_size: Number of perturbed images evaluated per GPU forward pass.
            model_input_size: Optional forward-pass resolution ``(H_m, W_m)`` or
                integer side length when the model expects a different input size
                than the canonical segmentation grid (e.g. ``300`` for DenseNet-161).
            norm_mean: Per-channel normalization mean of the input tensor.
            norm_std: Per-channel normalization standard deviation.
            use_tqdm: Whether to display progress bars when ``tqdm`` is available.
            use_amp: Whether to enable CUDA automatic mixed precision.
            amp_dtype: Floating-point dtype used when ``use_amp=True``.

        Raises:
            ValueError: If any hyperparameter is outside its valid domain.
        """
        super().__init__(model)
        if n_segments <= 0:
            raise ValueError(f"n_segments must be positive, got {n_segments}")
        if compactness <= 0.0:
            raise ValueError(f"compactness must be positive, got {compactness}")
        if num_perturbations <= 1:
            raise ValueError(f"num_perturbations must be > 1, got {num_perturbations}")
        if not (0.0 < keep_prob < 1.0):
            raise ValueError(f"keep_prob must be in (0, 1), got {keep_prob}")
        if kernel_width <= 0.0:
            raise ValueError(f"kernel_width must be positive, got {kernel_width}")
        if ridge_alpha <= 0.0:
            raise ValueError(f"ridge_alpha must be positive, got {ridge_alpha}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.n_segments = int(n_segments)
        self.compactness = float(compactness)
        self.num_perturbations = int(num_perturbations)
        self.keep_prob = float(keep_prob)
        self.kernel_width = float(kernel_width)
        self.ridge_alpha = float(ridge_alpha)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

        if model_input_size is None:
            self.model_input_size: tuple[int, int] | None = None
        elif isinstance(model_input_size, int):
            if model_input_size <= 0:
                raise ValueError(
                    f"model_input_size must be positive, got {model_input_size}"
                )
            self.model_input_size = (int(model_input_size), int(model_input_size))
        else:
            if (
                len(model_input_size) != 2
                or model_input_size[0] <= 0
                or model_input_size[1] <= 0
            ):
                raise ValueError(
                    f"model_input_size must be positive (H, W), got {model_input_size}"
                )
            self.model_input_size = (
                int(model_input_size[0]),
                int(model_input_size[1]),
            )

        self.norm_mean = tuple(float(x) for x in norm_mean)
        self.norm_std = tuple(float(x) for x in norm_std)
        self.use_tqdm = bool(use_tqdm and _tqdm is not None)
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype

        self.last_beta: np.ndarray | None = None
        self.last_r2: float | None = None
        self.last_segments: np.ndarray | None = None
        self.last_yhat0: float | None = None

    @staticmethod
    def _sample_z(
        num_perturbations: int,
        num_superpixels: int,
        keep_prob: float = 0.7,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample binary superpixel perturbation matrix ``(M, K)`` with ``Z[0] = 1``."""
        rng = np.random.default_rng(seed)
        z_mat = (rng.random((num_perturbations, num_superpixels)) < keep_prob).astype(
            np.int8
        )
        z_mat[0, :] = 1
        return z_mat

    @staticmethod
    def _locality_weights(z_mat: np.ndarray, kernel_width: float = 0.25) -> np.ndarray:
        """Compute exponential cosine/Hamming locality weights ``(M,)``."""
        num_superpixels = z_mat.shape[1]
        dist = (num_superpixels - z_mat.sum(axis=1)) / num_superpixels
        return np.exp(-(dist**2) / (kernel_width**2))

    def cleanup(self, failed: bool = False) -> None:
        """Release cached CUDA memory blocks if running on GPU.

        Args:
            failed: If True or an exception is in flight, avoids calling CUDA
                APIs to prevent cascading secondary failures.
        """
        if failed or sys.exc_info()[0] is not None:
            return
        if self.is_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (RuntimeError, torch.cuda.CudaError) as e:
                warn("CUDA empty_cache failed during LIME cleanup: %s", e)

    def generate_map(
        self,
        image_tensor: torch.Tensor,
        target_index: int = 0,
        segments: np.ndarray | None = None,
        normalize: bool = True,
        log_debug_stats: bool = False,
    ) -> np.ndarray:
        """Compute the LIME superpixel attribution map ``(H, W)`` for one image.

        Args:
            image_tensor: Single normalized image tensor of shape ``(C, H, W)``
                or ``(1, C, H, W)``.
            target_index: Target regression or classification output index.
            segments: Optional precomputed 2D ``int16`` SLIC superpixel array of
                shape ``(H_s, W_s)``. When omitted, SLIC is computed on the fly.
            normalize: Whether to symmetrically normalize the output map by
                ``max(|heat|) + 1e-8`` so zero remains neutral ``0.0`` and
                extremes reach ``[-1.0, 1.0]``.
            log_debug_stats: Whether to log execution duration, $R^2$, and map range.

        Returns:
            2D ``float32`` attribution heatmap of shape ``(H_s, W_s)``. Also
            populates ``self.last_beta``, ``self.last_r2``, ``self.last_segments``,
            and ``self.last_yhat0``.

        Raises:
            ValueError: If ``image_tensor`` or ``segments`` has invalid dimensions.
        """
        start_time = time.perf_counter()

        if image_tensor.dim() == 3:
            image = self._prepare_input(image_tensor.unsqueeze(0))
        elif image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image = self._prepare_input(image_tensor)
        else:
            raise ValueError(
                f"Input image tensor has unexpected dimensions: {image_tensor.shape}"
            )

        if segments is not None:
            segments_np = np.asarray(segments, dtype=np.int16)
            if segments_np.ndim != 2:
                raise ValueError(
                    f"segments must be a 2D array (H, W), got {segments_np.shape}"
                )
            seg_h, seg_w = segments_np.shape
            if (image.shape[2], image.shape[3]) != (seg_h, seg_w):
                image = F.interpolate(
                    image,
                    size=(seg_h, seg_w),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            segments_np = None

        _, channels, height, width = image.shape
        mean_t = torch.tensor(
            self.norm_mean[:channels], dtype=torch.float32, device=self.device
        ).view(1, channels, 1, 1)
        std_t = torch.tensor(
            self.norm_std[:channels], dtype=torch.float32, device=self.device
        ).view(1, channels, 1, 1)

        img_u8_t = (
            torch.clamp((image * std_t + mean_t) * 255.0, 0.0, 255.0)
            .round()
            .to(torch.uint8)
        )
        fill_u8_t = (
            img_u8_t.to(torch.float32).mean(dim=(2, 3), keepdim=True).to(torch.uint8)
        )

        if segments_np is None:
            img_u8_np = img_u8_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            segments_np = compute_slic_segments(
                img_u8_np,
                n_segments=self.n_segments,
                compactness=self.compactness,
            )

        num_superpixels = int(segments_np.max()) + 1
        z_mat = self._sample_z(
            self.num_perturbations,
            num_superpixels,
            keep_prob=self.keep_prob,
            seed=self.seed,
        )
        pi_weights = self._locality_weights(z_mat, kernel_width=self.kernel_width)

        target_hw = (
            self.model_input_size
            if self.model_input_size is not None
            else (height, width)
        )
        needs_resize = target_hw != (height, width)

        y_preds = np.zeros(self.num_perturbations, dtype=np.float32)

        with torch.no_grad():
            segments_t = torch.from_numpy(segments_np).to(
                device=self.device, dtype=torch.long
            )
            z_tensor = torch.from_numpy(z_mat).to(device=self.device, dtype=torch.bool)

            img_norm_t = ((img_u8_t.to(torch.float32) / 255.0) - mean_t) / std_t
            fill_norm_t = ((fill_u8_t.to(torch.float32) / 255.0) - mean_t) / std_t

            batch_starts = range(0, self.num_perturbations, self.batch_size)
            if self.use_tqdm and _tqdm is not None:
                batch_starts = _tqdm(batch_starts, desc="LIME batches", leave=False)

            for batch_start in batch_starts:
                batch_end = min(self.num_perturbations, batch_start + self.batch_size)
                z_batch = z_tensor[batch_start:batch_end]
                keep_mask = z_batch[:, segments_t].unsqueeze(1)

                if needs_resize:
                    pert_u8 = torch.where(keep_mask, img_u8_t, fill_u8_t)
                    resized_u8 = TF.resize(pert_u8, list(target_hw), antialias=True)
                    x_batch = ((resized_u8.to(torch.float32) / 255.0) - mean_t) / std_t
                else:
                    x_batch = torch.where(
                        keep_mask, img_norm_t, fill_norm_t
                    ).contiguous()

                if self.use_amp and self.is_cuda:
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        out_batch, _ = self.model(x_batch)
                else:
                    out_batch, _ = self.model(x_batch)

                y_preds[batch_start:batch_end] = (
                    out_batch[:, target_index].detach().float().cpu().numpy()
                )

        yhat0 = float(y_preds[0])
        y_target = y_preds - yhat0

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        reg.fit(z_mat, y_target, sample_weight=pi_weights)
        beta = reg.coef_.astype(np.float32)
        y_sur = reg.predict(z_mat).astype(np.float32)
        r2_val = weighted_r2(y_target, y_sur, pi_weights)

        self.last_beta = beta
        self.last_r2 = float(r2_val)
        self.last_segments = segments_np
        self.last_yhat0 = yhat0

        heat = beta[segments_np].astype(np.float32)
        if normalize:
            max_abs = float(np.max(np.abs(heat))) + 1e-8
            heat = (heat / max_abs).astype(np.float32)

        if log_debug_stats:
            debug(
                "LIME: returned %s map -> range [%.4f, %.4f], R2=%.4f in %.2f seconds",
                "normalized" if normalize else "raw",
                float(heat.min()),
                float(heat.max()),
                self.last_r2,
                time.perf_counter() - start_time,
            )
        return heat
