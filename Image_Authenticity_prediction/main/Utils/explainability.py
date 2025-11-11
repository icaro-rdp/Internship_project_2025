import torch.nn.functional as F
import torch
import numpy as np
import itertools
import math
from .normalization import normalize_data
from .logger import info, warn, error, debug

class GradCAM:
    """
    Implements the Grad-CAM visualization technique for a model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.model.eval()
        self.register_hooks()
        self.relu = True
    
    def register_hooks(self):
        """ Attaches forward and backward hooks to the target layer. """
        
        def forward_hook(module, input, output):
            # Store the activations from the forward pass
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            # Store the gradients from the backward pass
            # grad_output[0] is the gradient w.r.t. the module's output
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # Register the hooks and store their handles for later removal
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_image, score_idx=0):
        """
        Generates the Class Activation Map (CAM).
        
        Args:
            input_image (torch.Tensor): A 4D tensor (B, C, H, W). 
                                        Usually B=1.
            score_idx (int): The index of the output score to backpropagate from.
        """
        if input_image.dim() != 4 or input_image.shape[0] != 1:
            raise ValueError("input_image must be a 4D tensor with batch size 1 (B, C, H, W)")

        self.gradients = None
        self.activations = None
        
        input_image.requires_grad_(True)
        
        # 1. Forward pass
        model_output, _ = self.model(input_image)
        
        # 2. Backward pass
        self.model.zero_grad()
        score = model_output[0, score_idx]
        score.backward() 
        
        # 2.5. Check if gradients and activations were captured
        if self.gradients is None:
            msg = "Gradients not captured. Check hook registration and target layer."
            warn(msg)
            raise RuntimeError(msg)
        if self.activations is None:
            msg = "Activations not captured. Check hook registration."
            warn(msg)
            raise RuntimeError(msg)
        
        # 3. Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # (C, H_feat, W_feat)
        activations = self.activations.cpu().numpy()[0] # (C, H_feat, W_feat)
        
        # 4. Calculate importance weights
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # 5. Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32) # (H_feat, W_feat)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # 6. Apply ReLU
        if self.relu:
            cam = np.maximum(cam, 0)
        
        # 7. Resize CAM to original input image size 
        
        #Convert numpy array to tensor
        cam_tensor = torch.tensor(cam)
        
        # Add batch (B=1) and channel (C=1) dimensions for interpolate
        # Shape becomes [1, 1, H_feat, W_feat]
        cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)

        # Get target size (H, W) from the original input image
        target_size = (input_image.shape[2], input_image.shape[3])

        # Interpolate
        cam_resized = F.interpolate(
            cam_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )

        # Remove batch and channel dims, convert back to numpy
        # Shape is now (H, W)
        cam = cam_resized.squeeze().cpu().numpy()
        
        # 8. Normalize CAM
        if self.relu:
            cam = normalize_data(cam, min_range=0, max_range=1)
        else:
            cam = normalize_data(cam, min_range=-1, max_range=1)

                 
        return cam

    def cleanup(self):
        """ Removes the hooks to free up resources. """
        info("Removing GradCAM hooks.")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = None
        self.activations = None

class MultiscalePixelMasking:
    """
    Implements a minimal, core version of Multiscale Occlusion Saliency.
    All progress bars and print logs have been removed.
    
    This method is also known as Occlusion Saliency or Occlusion Sensitivity.
    """
    def __init__(self, model, sigma_list, pixel_batch_size, mask_value=0.0):
        self.model = model
        self.sigma_list = sigma_list
        self.pixel_batch_size = pixel_batch_size
        self.mask_value = mask_value
        self.model.eval()
        
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            print("Warning: Could not determine model device. Assuming 'cpu'.")
            self.device = torch.device("cpu")
            self.model.to(self.device)

    @staticmethod
    def _generate_mask(img_size, center, sigma, device):
        """
        Generates a binary mask with a square of zeros.
        """
        mask = torch.ones(1, 1, img_size[0], img_size[1], device=device)
        start_x = max(0, int(center[0] - sigma // 2))
        end_x = min(img_size[1], int(center[0] + (sigma + 1) // 2))
        start_y = max(0, int(center[1] - sigma // 2))
        end_y = min(img_size[0], int(center[1] + (sigma + 1) // 2))
        
        if start_y < end_y and start_x < end_x:
            mask[:, :, start_y:end_y, start_x:end_x] = 0
        return mask

    def generate_saliency_map(self, image_tensor, normalize=True):
        """
        Calculates the full multiscale saliency map for a single image.
        Returns (saliency_map_numpy, original_score).
        """
        # 1. Prepare image and get original score
        if image_tensor.dim() == 3: # (C, H, W)
            img_tensor_base = image_tensor.unsqueeze(0).to(self.device) # (1, C, H, W)
        elif image_tensor.dim() == 4 and image_tensor.shape[0] == 1: # (1, C, H, W)
            img_tensor_base = image_tensor.to(self.device)
        else:
            raise ValueError(f"Input image tensor has unexpected dimensions: {image_tensor.shape}")

        with torch.no_grad():
            original_output, _ = self.model(img_tensor_base)
            original_score = original_output.item()

        img_size = img_tensor_base.shape[2:] # H, W
        saliency_map_final = torch.zeros(img_size, dtype=torch.float32, device=self.device)

        # 2. Main Occlusion Loop
        for sigma in self.sigma_list:
            saliency_map_sigma = torch.zeros(img_size, dtype=torch.float32, device=self.device)
            
            # Use itertools to generate all pixel coordinates (y, x) instead of nested loops 
            all_pixel_coords = list(itertools.product(range(img_size[0]), range(img_size[1]))) # (y, x)
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
                    masked_scores_tensor_batch = output_batch.squeeze()
                    if masked_scores_tensor_batch.dim() == 0:
                        masked_scores_tensor_batch = masked_scores_tensor_batch.unsqueeze(0)

                for k in range(actual_batch_size):
                    y, x = current_coords_batch[k]
                    masked_score_item = masked_scores_tensor_batch[k].item()
                    saliency_value = original_score - masked_score_item # Score drop
                    saliency_map_sigma[y, x] = saliency_value

            saliency_map_final += saliency_map_sigma


        # 3. Check if Normalize and Return Results
        if not normalize:
            print("Returning unnormalized saliency map.")
            return saliency_map_final.cpu().numpy(), original_score
        
        min_val = torch.min(saliency_map_final)
        max_val = torch.max(saliency_map_final)

        # Normalize to [0, 1]
        if max_val > min_val:
            saliency_map_normalized = (saliency_map_final - min_val) / (max_val - min_val)
        else:
            saliency_map_normalized = torch.zeros_like(saliency_map_final)
            print("Warning: Saliency map has no variation; returning zero map.")
        
        return saliency_map_normalized.cpu().numpy(), original_score