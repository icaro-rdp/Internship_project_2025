# --- Imports ---
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import time
import importlib
from collections.abc import Iterable, Sequence
from .logger import info, warn, error, debug

try:
    _tqdm_module = importlib.import_module("tqdm.auto")
    _tqdm = getattr(_tqdm_module, "tqdm", None)
except Exception:
    _tqdm = None
from .normalization import normalize_data
from .logger import info, warn, error, debug
from abc import ABC, abstractmethod


class ModelsExplainer(ABC):
    """
    Abstract Base Class (Superclass) for saliency map generation methods.

    This class defines the common interface and shared setup logic
    for all saliency map "explainer" classes.
    """

    def __init__(self, model):
        self.model = model

        try:
            # Automatically determine and store the model's device
            self.device = next(model.parameters()).device
        except StopIteration:
            warn("Could not determine model device. Assuming 'cpu'.")
            self.device = torch.device("cpu")

        # Set model to eval mode (common setup)
        self.model.eval()
        self.model.to(self.device)

    @abstractmethod
    def generate_map(self, input_tensor, target_index=0):
        """
        Generates the saliency map. This is the "contract" that
        all subclasses must fulfill by implementing this method.

        Args:
            input_tensor (torch.Tensor): A 4D tensor (B, C, H, W).
                                         Usually B=1.
            target_index (int): The index of the output score to explain.

        Returns:
            np.array: A 2D numpy array (H, W) representing the saliency map.
        """
        # This code will never run, but it forces subclasses
        # to implement this method.
        raise NotImplementedError

    def cleanup(self):
        """
        Optional cleanup method. Subclasses can override this
        if they need to perform cleanup (e.g., remove hooks).
        """
        # By default, do nothing.
        pass

    def __call__(self, input_tensor, target_index=0):
        """
        A convenience method to make the instance callable.
        e.g., you can do `explainer(image)` instead of `explainer.generate_map(image)`.
        """
        return self.generate_map(input_tensor, target_index)


class GradCAM(ModelsExplainer):
    """
    Implements Grad-CAM, inheriting shared logic from ModelsExplainer.
    """

    def __init__(self, model, target_layer, relu=True):
        # 1. Run the parent's __init__ (handles model, device, eval())
        super().__init__(model)

        # 2. Add Grad-CAM specific attributes
        self.target_layer = self._resolve_target_layer(target_layer)
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.relu = bool(relu)

        # 3. Run Grad-CAM specific setup
        self.register_hooks()

    def _resolve_target_layer(self, target_layer):
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
            # Allow integer indexing for Sequential-style containers
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

    def register_hooks(self):
        """Attaches forward and backward hooks to the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_map(self, input_image, target_index=0):
        """
        Generates the Class Activation Map (CAM).
        This method fulfills the ModelsExplainer "contract".

        Args:
            input_image (torch.Tensor): A 4D tensor (B, C, H, W).
            target_index (int): The index of the output score.
        """
        if input_image.dim() != 4 or input_image.shape[0] != 1:
            raise ValueError(
                "input_image must be a 4D tensor with batch size 1 (B, C, H, W)"
            )

        # Ensure input image is on the same device as the model
        input_image = input_image.to(self.device)

        self.gradients = None
        self.activations = None

        input_image.requires_grad_(True)

        # 1. Forward pass
        model_output, _ = self.model(input_image)

        # 2. Backward pass
        self.model.zero_grad()
        score = model_output[0, target_index]  # Use target_index
        score.backward()

        # 2.5. Check hooks
        if self.gradients is None:
            msg = "Gradients not captured. Check hook registration and target layer."
            warn(msg)
            raise RuntimeError(msg)
        if self.activations is None:
            msg = "Activations not captured. Check hook registration."
            warn(msg)
            raise RuntimeError(msg)

        # 3. Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]

        # 4. Calculate weights
        weights = np.mean(gradients, axis=(1, 2))

        # 5. Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        raw_min = float(cam.min())
        raw_max = float(cam.max())
        grad_min = float(gradients.min())
        grad_max = float(gradients.max())
        act_min = float(activations.min())
        act_max = float(activations.max())

        # 6. Apply ReLU
        if self.relu:
            cam = np.maximum(cam, 0)
            post_min = float(cam.min())
            post_max = float(cam.max())
        else:
            post_min = None
            post_max = None

        # 7. Resize CAM
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        target_size = (input_image.shape[2], input_image.shape[3])
        cam_resized = F.interpolate(
            cam_tensor, size=target_size, mode="bilinear", align_corners=False
        )
        cam = cam_resized.squeeze().cpu().numpy()

        # 8. Normalize CAM
        if self.relu:
            cam = normalize_data(cam, min_range=0, max_range=1)
        else:
            cam = normalize_data(cam, min_range=-1, max_range=1)

        final_min = float(cam.min())
        final_max = float(cam.max())

        if self.relu:
            debug(
                "GradCAM stats -> raw[%.4f, %.4f] post_relu[%.4f, %.4f] final[%.4f, %.4f] gradients[%.4f, %.4f] activations[%.4f, %.4f]"
                % (
                    raw_min,
                    raw_max,
                    post_min,
                    post_max,
                    final_min,
                    final_max,
                    grad_min,
                    grad_max,
                    act_min,
                    act_max,
                )
            )
        else:
            debug(
                "GradCAM stats -> raw[%.4f, %.4f] final[%.4f, %.4f] gradients[%.4f, %.4f] activations[%.4f, %.4f]"
                % (
                    raw_min,
                    raw_max,
                    final_min,
                    final_max,
                    grad_min,
                    grad_max,
                    act_min,
                    act_max,
                )
            )

        return cam

    def cleanup(self):
        """
        Overrides the parent's empty cleanup method to
        remove the hooks.
        """
        info("Removing GradCAM hooks.")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = None
        self.activations = None


class MultiscalePixelMasking(ModelsExplainer):
    """Multiscale occlusion saliency (Multiscale Pixel Masking, MPM).

    For every scale ``sigma`` and every pixel, a ``sigma x sigma`` square centred on
    that pixel is replaced with ``mask_value``; the resulting drop in the target
    score is written to the pixel. Per-scale maps are summed.

    Occlusion masks are built and applied on the model's device in one vectorised
    operation per batch, and scores stay on the device until the map is complete,
    so each batch costs one forward pass plus a few tensor kernels and no host
    synchronisation. See ``docs/adr/0003-explainability-performance-optimization.md``.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_list: Sequence[int],
        pixel_batch_size: int,
        mask_value: float = 0.0,
        use_tqdm: bool = True,
    ) -> None:
        """Initialise the explainer.

        Args:
            model: Network returning ``(predictions, features)``.
            sigma_list: Side lengths, in pixels, of the square occluders.
            pixel_batch_size: Occluded copies per forward pass. Bounded by device
                memory; it does not change the resulting map.
            mask_value: Value written into occluded pixels.
            use_tqdm: Whether to show progress bars when tqdm is installed.
        """
        super().__init__(model)
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
        img_size: tuple[int, int], pixel_indices: torch.Tensor, sigma: int
    ) -> torch.Tensor:
        """Build boolean occlusion masks for a batch of centre pixels.

        Args:
            img_size: Image height and width.
            pixel_indices: Row-major flat indices of the centre pixels, shape (B,).
            sigma: Side length of the square occluder.

        Returns:
            Tensor of shape (B, 1, H, W) that is True inside each occluder. Along
            each axis the square spans ``[c - sigma // 2, c + (sigma + 1) // 2)``,
            clipped to the image.
        """
        height, width = img_size
        rows = torch.arange(height, device=pixel_indices.device)
        cols = torch.arange(width, device=pixel_indices.device)
        center_rows = (pixel_indices // width).unsqueeze(1)
        center_cols = (pixel_indices % width).unsqueeze(1)
        before, after = sigma // 2, (sigma + 1) // 2
        row_hit = (rows >= center_rows - before) & (rows < center_rows + after)
        col_hit = (cols >= center_cols - before) & (cols < center_cols + after)
        return (row_hit.unsqueeze(2) & col_hit.unsqueeze(1)).unsqueeze(1)

    def generate_map(
        self,
        image_tensor: torch.Tensor,
        target_index: int = 0,
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute the summed multiscale saliency map for a single image.

        Args:
            image_tensor: Image of shape (C, H, W) or (1, C, H, W).
            target_index: Index of the model output to explain.
            normalize: Whether to rescale the map to [-1, 1].

        Returns:
            Saliency map of shape (H, W). Positive values mark pixels whose
            occlusion lowers the target score.

        Raises:
            ValueError: If the input does not hold exactly one image.
        """
        start_time = time.perf_counter()

        if image_tensor.dim() == 3:
            image = image_tensor.unsqueeze(0).to(self.device)
        elif image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image = image_tensor.to(self.device)
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

        height, width = image.shape[2:]
        total_pixels = height * width
        all_pixel_indices = torch.arange(total_pixels, device=self.device)
        saliency_map_final = torch.zeros(
            total_pixels, dtype=torch.float32, device=self.device
        )

        with torch.inference_mode():
            original_output, _ = self.model(image)
            original_score = original_output[0, target_index]

            for sigma in self._progress(self.sigma_list, desc="MPM sigma levels"):
                saliency_map_sigma = torch.empty_like(saliency_map_final)
                batch_starts = range(0, total_pixels, self.pixel_batch_size)
                for batch_start in self._progress(batch_starts, f"MPM sigma {sigma}"):
                    batch_end = min(total_pixels, batch_start + self.pixel_batch_size)
                    occluded = self._occlusion_masks(
                        (height, width),
                        all_pixel_indices[batch_start:batch_end],
                        sigma,
                    )
                    masked_images = torch.where(occluded, self.mask_value, image)
                    output_batch, _ = self.model(masked_images)
                    saliency_map_sigma[batch_start:batch_end] = (
                        original_score - output_batch[:, target_index]
                    )

                saliency_map_final += saliency_map_sigma
                debug(
                    "MPM: sigma=%s stats -> min %.4f | max %.4f | mean %.4f",
                    sigma,
                    saliency_map_sigma.min(),
                    saliency_map_sigma.max(),
                    saliency_map_sigma.mean(),
                )

        saliency_map_numpy = saliency_map_final.reshape(height, width).cpu().numpy()

        if normalize:
            # Use the same normalization function as GradCAM for consistency
            saliency_map_numpy = normalize_data(
                saliency_map_numpy, min_range=-1, max_range=1
            )

        debug(
            "MPM: returned %s map -> range [%.4f, %.4f] in %.2f seconds",
            "normalized" if normalize else "raw",
            saliency_map_numpy.min(),
            saliency_map_numpy.max(),
            time.perf_counter() - start_time,
        )
        return saliency_map_numpy

    def _progress[T](self, iterable: Iterable[T], desc: str) -> Iterable[T]:
        """Wrap an iterable in a transient tqdm bar when progress bars are enabled."""
        if self.use_tqdm:
            return _tqdm(iterable, desc=desc, leave=False)
        return iterable
