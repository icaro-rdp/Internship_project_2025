# --- Imports ---
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import itertools
import math
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
    def __init__(self, model, target_layer):
        # 1. Run the parent's __init__ (handles model, device, eval())
        super().__init__(model)
        
        # 2. Add Grad-CAM specific attributes
        self.target_layer = self._resolve_target_layer(target_layer)
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.relu = True
        
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
        for attr in target_layer.split('.'):
            # Allow integer indexing for Sequential-style containers
            if attr.isdigit():
                index = int(attr)
                if not hasattr(module, '__getitem__'):
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
        """ Attaches forward and backward hooks to the target layer. """
        
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
            raise ValueError("input_image must be a 4D tensor with batch size 1 (B, C, H, W)")

        # Ensure input image is on the same device as the model
        input_image = input_image.to(self.device)

        self.gradients = None
        self.activations = None
        
        input_image.requires_grad_(True)
        
        # 1. Forward pass
        model_output, _ = self.model(input_image)
        
        # 2. Backward pass
        self.model.zero_grad()
        score = model_output[0, target_index] # Use target_index
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
        
        # 6. Apply ReLU
        if self.relu:
            cam = np.maximum(cam, 0)
        
        # 7. Resize CAM
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        target_size = (input_image.shape[2], input_image.shape[3])
        cam_resized = F.interpolate(
            cam_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        cam = cam_resized.squeeze().cpu().numpy()
        
        # 8. Normalize CAM
        if self.relu:
            cam = normalize_data(cam, min_range=0, max_range=1)
        else:
            cam = normalize_data(cam, min_range=-1, max_range=1)
                 
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
    """
    Implements Multiscale Occlusion Saliency, inheriting from ModelsExplainer.
    """
    def __init__(self, model, sigma_list, pixel_batch_size, mask_value=0.0):
        # 1. Run the parent's __init__ (handles model, device, eval())
        super().__init__(model)
        
        # 2. Add Occlusion-specific attributes
        self.sigma_list = sigma_list
        self.pixel_batch_size = pixel_batch_size
        self.mask_value = mask_value

    @staticmethod
    def _generate_mask(img_size, center, sigma, device):
        """ Generates a binary mask with a square of zeros. """
        mask = torch.ones(1, 1, img_size[0], img_size[1], device=device)
        start_x = max(0, int(center[0] - sigma // 2))
        end_x = min(img_size[1], int(center[0] + (sigma + 1) // 2))
        start_y = max(0, int(center[1] - sigma // 2))
        end_y = min(img_size[0], int(center[1] + (sigma + 1) // 2))
        
        if start_y < end_y and start_x < end_x:
            mask[:, :, start_y:end_y, start_x:end_x] = 0
        return mask

    def generate_map(self, image_tensor, target_index=0, normalize=True):
        """
        Calculates the full multiscale saliency map for a single image.
        This method fulfills the ModelsExplainer "contract".
        
        Returns (saliency_map_numpy, original_score).
        """
        # 1. Prepare image and get original score
        if image_tensor.dim() == 3:
            img_tensor_base = image_tensor.unsqueeze(0).to(self.device)
        elif image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            img_tensor_base = image_tensor.to(self.device)
        else:
            raise ValueError(f"Input image tensor has unexpected dimensions: {image_tensor.shape}")

        with torch.no_grad():
            original_output, _ = self.model(img_tensor_base)
            # Use target_index to get the correct score
            original_score = original_output[0, target_index].item()

        img_size = img_tensor_base.shape[2:]
        saliency_map_final = torch.zeros(img_size, dtype=torch.float32, device=self.device)

        # 2. Main Occlusion Loop
        for sigma in self.sigma_list:
            saliency_map_sigma = torch.zeros(img_size, dtype=torch.float32, device=self.device)
            all_pixel_coords = list(itertools.product(range(img_size[0]), range(img_size[1])))
            total_pixels = len(all_pixel_coords)
            num_batches = math.ceil(total_pixels / self.pixel_batch_size)
            
            for batch_idx in range(num_batches):
                batch_start_idx = batch_idx * self.pixel_batch_size
                batch_end_idx = min(total_pixels, (batch_idx + 1) * self.pixel_batch_size)
                current_coords_batch = all_pixel_coords[batch_start_idx:batch_end_idx]
                actual_batch_size = len(current_coords_batch)

                if actual_batch_size == 0: continue

                masked_images_list = []
                for y_coord, x_coord in current_coords_batch:
                    mask = self._generate_mask(img_size, (x_coord, y_coord), sigma, self.device)
                    masked_image = img_tensor_base * mask + self.mask_value * (1 - mask)
                    masked_images_list.append(masked_image)
                
                batch_of_masked_images = torch.cat(masked_images_list, dim=0)

                with torch.no_grad():
                    output_batch, _ = self.model(batch_of_masked_images)
                    # Get the scores for the correct target_index
                    masked_scores_tensor_batch = output_batch[:, target_index].squeeze()
                    if masked_scores_tensor_batch.dim() == 0:
                        masked_scores_tensor_batch = masked_scores_tensor_batch.unsqueeze(0)

                for k in range(actual_batch_size):
                    y, x = current_coords_batch[k]
                    masked_score_item = masked_scores_tensor_batch[k].item()
                    saliency_value = original_score - masked_score_item
                    saliency_map_sigma[y, x] = saliency_value

            saliency_map_final += saliency_map_sigma

        # 3. Normalize and Return Results
        saliency_map_numpy = saliency_map_final.cpu().numpy()
        
        if normalize:
            # Use the same normalization function as GradCAM for consistency
            saliency_map_numpy = normalize_data(saliency_map_numpy, min_range=-1, max_range=1)
        
        return saliency_map_numpy
    
