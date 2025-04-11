import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

# Define the model architecture (same as in your original code)
class AuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained VGG16
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in vgg.features.parameters():
                param.requires_grad = False
                
        # Extract features up to fc2
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc1 = vgg.classifier[:-1]  # Up to fc2 (4096 -> 128)
        
        # New regression head
        self.regression_head = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Predict authenticity
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.fc1(x)
        predictions = self.regression_head(features)
        return predictions, features

# GradCAM implementation for model interpretation
class GradCAM:
    """
    Implements Gradient-weighted Class Activation Mapping for model interpretation.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture activations and gradients
        self.register_hooks()
        
    def register_hooks(self):
        def forward_hook(module, input, output):
            # Store the activations of the target layer
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            # Store the gradients coming into the target layer
            self.gradients = grad_output[0].detach()
        
        # Register the hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate_cam(self, input_image):
        # Forward pass through the model
        model_output, _ = self.model(input_image)
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Backward pass - for regression we use the output directly
        model_output.backward(retain_graph=True)
        
        # Get the gradients and activations
        gradients = self.gradients.data.cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.data.cpu().numpy()[0]  # [C, H, W]
        
        # Weight the channels by the average gradient
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Create weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU to focus on features that have a positive influence
        cam = np.maximum(cam, 0)
        
        # Resize CAM to input image size
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        
        # Normalize the CAM
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)  # Adding small constant to avoid division by zero
        
        return cam

# AIS implementation (formerly CAM)
class CAM:
    """
    Implements Activation Importance Score (AIS) for model interpretation.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.activations = None
        
        # Register hooks to capture activations
        self.register_hooks()
        
    def register_hooks(self):
        def forward_hook(module, input, output):
            # Store the activations of the target layer
            self.activations = output.detach()
            
        # Register the hooks
        self.target_layer.register_forward_hook(forward_hook)
        
    def generate_cam(self, input_image, importance_scores):
        # Forward pass through the model
        model_output, _ = self.model(input_image)
        
        # Get the activations
        activations = self.activations.data.cpu().numpy()[0]  # [C, H, W]
        
        # Weight the channels by the importance scores
        weights = importance_scores
            
        # Create weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        
        # Handle the case where weights might be multi-dimensional
        if len(weights.shape) == 1 and len(weights) == activations.shape[0]:
            # Standard case: one weight per channel
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        else:
            # Check the shapes before operation to provide helpful error message
            raise ValueError(f"Incompatible shapes: weights {weights.shape}, activations {activations.shape}")
        
        # Apply ReLU to focus on features that have a positive influence
        cam = np.maximum(cam, 0)
        
        # Resize CAM to input image size
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        
        # Normalize the CAM
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)  # Adding small constant to avoid division by zero
        
        return cam

def load_single_image(image_path):
    """
    Load and preprocess a single image for model inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image tensor
    """
    # Data transformations (same as in your original code)
    data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    image_tensor = data_transforms(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def process_single_image(image_path, baseline_model_path, noisy_pruned_model_path, importance_scores_path=None, output_dir='single_image_results'):
    """
    Process a single image with both models and generate visualizations.
    
    Args:
        image_path: Path to the image file
        baseline_model_path: Path to the baseline model weights
        noisy_pruned_model_path: Path to the noisy pruned model weights
        importance_scores_path: Path to the importance scores (optional)
        output_dir: Directory to save the results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    baseline_model = AuthenticityPredictor()
    baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=device))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    
    noisy_pruned_model = AuthenticityPredictor()
    noisy_pruned_model.load_state_dict(torch.load(noisy_pruned_model_path, map_location=device))
    noisy_pruned_model = noisy_pruned_model.to(device)
    noisy_pruned_model.eval()
    
    # Load importance scores if provided
    if importance_scores_path:
        importance_scores = np.load(importance_scores_path, allow_pickle=True)
        
        # Apply transformations to the importance scores (same as in your original code)
        def zero_negative_values(scores):
            scores[scores < 0] = 0
            return scores

        def min_max_scale(scores):
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val == min_val:
                return np.zeros_like(scores)
            return (scores - min_val) / (max_val - min_val)

        def transform_scores(scores):
            scores = zero_negative_values(scores)
            scores = min_max_scale(scores)
            return scores
        
        # Use the first set of importance scores for the single image
        if isinstance(importance_scores[0], np.ndarray):
            importance_scores = transform_scores(importance_scores[0])
        else:
            importance_scores = transform_scores(np.array(importance_scores[0]))
    else:
        # Use uniform importance scores if not provided
        importance_scores = np.ones(512)  # 512 is the number of channels in the last conv layer of VGG16
    
    # Load and preprocess the image
    image_tensor = load_single_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Initialize GradCAM and AIS visualizers
    target_layer_baseline = baseline_model.features[28]  # Last conv layer of VGG16
    target_layer_noisy_pruned = noisy_pruned_model.features[28]
    
    gradcam_baseline = GradCAM(baseline_model, target_layer_baseline)
    gradcam_noisy_pruned = GradCAM(noisy_pruned_model, target_layer_noisy_pruned)
    
    # For AIS, we're using the same CAM class but with different terminology
    ais_uniform = CAM(baseline_model, target_layer_baseline)
    ais_weighted = CAM(noisy_pruned_model, target_layer_noisy_pruned)
    
    # Make predictions
    with torch.no_grad():
        baseline_prediction, _ = baseline_model(image_tensor)
        noisy_pruned_prediction, _ = noisy_pruned_model(image_tensor)
    
    print(f"Baseline model prediction: {baseline_prediction.item():.4f}")
    print(f"Noisy pruned model prediction: {noisy_pruned_prediction.item():.4f}")
    
    # Generate visualizations
    # For GradCAM we need to re-enable gradients
    image_tensor.requires_grad = True
    
    # Generate GradCAM visualizations
    gradcam_baseline_heatmap = gradcam_baseline.generate_cam(image_tensor)
    gradcam_noisy_pruned_heatmap = gradcam_noisy_pruned.generate_cam(image_tensor)
    
    # Generate visualization using importance scores
    # For AIS with uniform weights
    ais_uniform_heatmap = ais_uniform.generate_cam(image_tensor, np.ones(512))  # Uniform importance scores
    # For AIS with learned importance weights
    ais_weighted_heatmap = ais_weighted.generate_cam(image_tensor, importance_scores)  # Apply importance scores
    
    # Convert to heatmap using jet colormap
    gradcam_baseline_colored = cv2.applyColorMap(np.uint8(255 * gradcam_baseline_heatmap), cv2.COLORMAP_JET)
    gradcam_noisy_pruned_colored = cv2.applyColorMap(np.uint8(255 * gradcam_noisy_pruned_heatmap), cv2.COLORMAP_JET)
    
    ais_uniform_colored = cv2.applyColorMap(np.uint8(255 * ais_uniform_heatmap), cv2.COLORMAP_JET)
    ais_weighted_colored = cv2.applyColorMap(np.uint8(255 * ais_weighted_heatmap), cv2.COLORMAP_JET)
    
    # Prepare original image for visualization
    # Detach the tensor to avoid the "requires_grad" error
    img_tensor = image_tensor[0].detach().cpu().numpy()
    img_tensor = np.transpose(img_tensor, (1, 2, 0))  # [H, W, C]
    
    # Denormalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_tensor = std * img_tensor + mean
    img_tensor = np.clip(img_tensor, 0, 1)
    
    # Convert to uint8 for OpenCV
    rgb_img = (img_tensor * 255).astype(np.uint8)
    bgr_img = rgb_img[:, :, ::-1]  # RGB to BGR for OpenCV
    
    # Resize heatmaps to match the image size
    gradcam_baseline_colored = cv2.resize(gradcam_baseline_colored, (bgr_img.shape[1], bgr_img.shape[0]))
    gradcam_noisy_pruned_colored = cv2.resize(gradcam_noisy_pruned_colored, (bgr_img.shape[1], bgr_img.shape[0]))
    
    ais_uniform_colored = cv2.resize(ais_uniform_colored, (bgr_img.shape[1], bgr_img.shape[0]))
    ais_weighted_colored = cv2.resize(ais_weighted_colored, (bgr_img.shape[1], bgr_img.shape[0]))
    
    # Create overlays
    gradcam_baseline_overlay = cv2.addWeighted(bgr_img, 0.6, gradcam_baseline_colored, 0.4, 0)
    gradcam_noisy_pruned_overlay = cv2.addWeighted(bgr_img, 0.6, gradcam_noisy_pruned_colored, 0.4, 0)
    
    ais_uniform_overlay = cv2.addWeighted(bgr_img, 0.6, ais_uniform_colored, 0.4, 0)
    ais_weighted_overlay = cv2.addWeighted(bgr_img, 0.6, ais_weighted_colored, 0.4, 0)
    
    # Create a figure for displaying the results - 1 row of 5 images
    fig, axes = plt.subplots(1, 5, figsize=(25, 8))
    
    # Plot original image
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot GradCAM visualizations with prediction scores
    axes[1].imshow(cv2.cvtColor(gradcam_baseline_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'GradCAM Baseline\nPrediction: {baseline_prediction.item():.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(gradcam_noisy_pruned_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'GradCAM Noisy Pruned\nPrediction: {noisy_pruned_prediction.item():.4f}')
    axes[2].axis('off')
    
    # Plot AIS visualizations with prediction scores
    axes[3].imshow(cv2.cvtColor(ais_uniform_overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f'AIS Uniform Weights\nPrediction: {baseline_prediction.item():.4f}')
    axes[3].axis('off')
    
    axes[4].imshow(cv2.cvtColor(ais_weighted_overlay, cv2.COLOR_BGR2RGB))
    axes[4].set_title(f'AIS Importance Weights\nPrediction: {noisy_pruned_prediction.item():.4f}')
    axes[4].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, 'single_image_results.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Results saved to {os.path.join(output_dir, 'single_image_results.png')}")
    
    # Save individual visualizations
    plt.figure()
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'original.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.imshow(cv2.cvtColor(gradcam_baseline_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'gradcam_baseline_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.imshow(cv2.cvtColor(gradcam_noisy_pruned_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'gradcam_noisy_pruned_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.imshow(cv2.cvtColor(ais_uniform_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'ais_uniform_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.imshow(cv2.cvtColor(ais_weighted_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'ais_weighted_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'baseline_prediction': baseline_prediction.item(),
        'noisy_pruned_prediction': noisy_pruned_prediction.item(),
        'overlays': {
            'gradcam_baseline': cv2.cvtColor(gradcam_baseline_overlay, cv2.COLOR_BGR2RGB),
            'gradcam_noisy_pruned': cv2.cvtColor(gradcam_noisy_pruned_overlay, cv2.COLOR_BGR2RGB),
            'ais_uniform': cv2.cvtColor(ais_uniform_overlay, cv2.COLOR_BGR2RGB),
            'ais_weighted': cv2.cvtColor(ais_weighted_overlay, cv2.COLOR_BGR2RGB)
        }
    }

# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual paths
    image_path = "test_image.jpeg"
    baseline_model_path = "Models/VGG-16_real_authenticity_finetuned.pth"
    noisy_pruned_model_path = "Models/real_authenticity_noise_out_pruned_model.pth"
    importance_scores_path = "Ranking_arrays/obj_x_obj_authenticity_importance_scores.npy"
    
    # Process the image
    results = process_single_image(
        image_path=image_path,
        baseline_model_path=baseline_model_path,
        noisy_pruned_model_path=noisy_pruned_model_path,
        importance_scores_path=importance_scores_path,
        output_dir="single_image_results"
    )
    
    print("Processing complete!")