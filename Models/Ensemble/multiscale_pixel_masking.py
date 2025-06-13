import os
import cv2
import itertools
import math
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, random_split

from torchvision import models, transforms
import torchvision.transforms.functional as F

# Assuming these custom modules are in the correct path
from Models.Ensemble.models import (
    BarlowTwinsAuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    InceptionV3AuthenticityPredictor,
)
from Models.Ensemble.dataset import IMAGENET_DATASET, DENSENET_DATASET
from Models.Ensemble.utils import get_predictions

# %% [markdown]
# ## 2. Configuration

# %%
class Config:
    """Consolidated configuration for the entire pipeline."""
    # --- Saliency Analysis Config ---
    
    SIGMA_LIST = [3, 5, 9, 17, 33, 65]
    MASK_VALUE = 0.0
    VIS_CMAP = 'bwr'
    VIS_ALPHA = 0.6
    PIXEL_BATCH_SIZE = 128  # Adjust based on GPU memory
    MAIN_OUTPUT_DIR = Path('multiscale_masking_outputs_ensemble')

    # --- Ensemble Model Config ---
    BATCH_SIZE = 1  # IMPORTANT: Saliency analysis requires a batch size of 1
    NUM_WORKERS = 20 # Adjust based on your system
    RANDOM_STATE = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    BASE_MODEL_WEIGHTS_DIR = Path('Models')
    META_LEARNER_SAVE_PATH = BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/stacking_ensemble_weights.pth'

    # Model configurations: (ModelClass, weights_filename, dataset_dict_to_use)
    MODEL_CONFIGS = [
        (BarlowTwinsAuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'BarlowTwins/Weights/BarlowTwins_real_authenticity_finetuned.pth', IMAGENET_DATASET),
        (EfficientNetB3AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/EfficientNetB3_weights.pth', IMAGENET_DATASET),
        (DenseNet161AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/DenseNet161_weights.pth', DENSENET_DATASET),
        (ResNet152AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/ResNet152_weights.pth', IMAGENET_DATASET),
        (VGG16AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/VGG16_weights.pth', IMAGENET_DATASET),
        (VGG19AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/VGG19_weights.pth', IMAGENET_DATASET),
        (InceptionV3AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/InceptionV3_weights.pth', DENSENET_DATASET),
    ]

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
Config.MAIN_OUTPUT_DIR.mkdir(exist_ok=True)

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(Config.RANDOM_STATE)


# %% [markdown]
# ## 3. Saliency Analysis Utility Functions

# %%
def generate_mask(img_size, center, sigma, device=Config.DEVICE):
    """Generates a binary mask with a square of zeros centered at 'center' with size 'sigma x sigma'."""
    mask = torch.ones(1, 1, img_size[0], img_size[1], device=device)
    start_x = max(0, int(center[0] - sigma // 2))
    end_x = min(img_size[1], int(center[0] + (sigma + 1) // 2))
    start_y = max(0, int(center[1] - sigma // 2))
    end_y = min(img_size[0], int(center[1] + (sigma + 1) // 2))
    if start_y < end_y and start_x < end_x:
        mask[:, :, start_y:end_y, start_x:end_x] = 0
    return mask

def calculate_saliency_map(model, image, original_score, sigma_list, mask_value=0.0, pixel_batch_size=32, device=Config.DEVICE):
    """
    Calculates the multiscale saliency map using the occlusion method with batched processing.
    """
    model.eval()
    if image.dim() == 3:
        img_tensor_base = image.unsqueeze(0).to(device)
    else:
        img_tensor_base = image.to(device)

    img_size = img_tensor_base.shape[2:]
    saliency_map_final = torch.zeros(img_size, dtype=torch.float32, device=device)
    logging.info(f"Calculating saliency for image size {img_size} using sigmas: {sigma_list}")

    outer_progress = tqdm(enumerate(sigma_list), total=len(sigma_list), desc="Overall Sigmas", unit="sigma")

    for i, sigma in outer_progress:
        saliency_map_sigma = torch.zeros(img_size, dtype=torch.float32, device=device)
        all_pixel_coords = list(itertools.product(range(img_size[0]), range(img_size[1])))
        num_batches = math.ceil(len(all_pixel_coords) / pixel_batch_size)

        inner_progress_bar = tqdm(range(num_batches), desc=f"  Sigma {sigma: >3} Batches", leave=False, unit="batch")
        
        for batch_idx in inner_progress_bar:
            batch_coords = all_pixel_coords[batch_idx * pixel_batch_size:(batch_idx + 1) * pixel_batch_size]
            if not batch_coords: 
                continue

            masked_images_list = [img_tensor_base * generate_mask(img_size, (x, y), sigma, device) + 
                                mask_value * (1 - generate_mask(img_size, (x, y), sigma, device)) 
                                for y, x in batch_coords]
            batch_of_masked_images = torch.cat(masked_images_list, dim=0)

            with torch.no_grad():
                # The model wrapper ensures output is a tuple (prediction, features)
                output_batch, _ = model(batch_of_masked_images)
                masked_scores = output_batch.squeeze()
                if masked_scores.dim() == 0:
                    masked_scores = masked_scores.unsqueeze(0)

            for k, (y, x) in enumerate(batch_coords):
                saliency_map_sigma[y, x] = original_score - masked_scores[k].item()
        
            # Add memory cleanup
            del batch_of_masked_images, output_batch, masked_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # FIX: Add the sigma map to the final map
        saliency_map_final += saliency_map_sigma
    
    # Normalize final map
    min_val, max_val = torch.min(saliency_map_final), torch.max(saliency_map_final)
    if max_val > min_val:
        saliency_map_normalized = (saliency_map_final - min_val) / (max_val - min_val)
    else:
        saliency_map_normalized = torch.zeros_like(saliency_map_final)
        logging.warning("Final saliency map was constant; result is a zero map.")
    
    return saliency_map_normalized.cpu().numpy()

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes an image tensor."""
    device = tensor.device
    mean_t = torch.tensor(mean, device=device, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=tensor.dtype).view(3, 1, 1)
    return torch.clamp(tensor * std_t + mean_t, 0., 1.)

def visualize_and_save_saliency(image_tensor, saliency_map, output_dir, filename_prefix, **kwargs):
    """Visualizes saliency map, creates an overlay, and saves images."""
    overlay_alpha = kwargs.get('overlay_alpha', 0.5)
    cmap_name = kwargs.get('cmap_name', 'bwr')
    
    image_specific_output_dir = Path(output_dir) / filename_prefix
    image_specific_output_dir.mkdir(exist_ok=True, parents=True)

    # FIX: Move tensor to CPU before converting to numpy
    img_denorm_np = denormalize_image(image_tensor).cpu().numpy().transpose(1, 2, 0)
    img_bgr = cv2.cvtColor((img_denorm_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save original
    cv2.imwrite(str(image_specific_output_dir / f"{filename_prefix}_original.png"), img_bgr)

    # Save heatmap
    heatmap = (cm.get_cmap(cmap_name)(saliency_map)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_specific_output_dir / f"{filename_prefix}_heatmap_{cmap_name}.png"), heatmap_bgr)

    # Save overlay
    overlay = cv2.addWeighted(img_bgr, 1.0 - overlay_alpha, heatmap_bgr, overlay_alpha, 0)
    cv2.imwrite(str(image_specific_output_dir / f"{filename_prefix}_overlay_{cmap_name}.png"), overlay)

def run_saliency_analysis(model, dataloader, output_dir, model_name, **kwargs):
    """Facade function to run and orchestrate the saliency analysis."""
    logging.info(f"\n--- Starting Saliency Analysis for {model_name} ---")
    model_output_dir = Path(output_dir) / model_name.replace(" ", "_")
    model_output_dir.mkdir(exist_ok=True)
    logging.info(f"Outputs will be saved in: {model_output_dir}")

    num_images_to_process = kwargs.get('num_images_to_process', len(dataloader.dataset))
    processed_count = 0
    model.eval()
    
    all_saliency_maps = []

    for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"{model_name} Image Progress")):
        if processed_count >= num_images_to_process:
            logging.info(f"Reached limit of {num_images_to_process} images. Stopping.")
            break

        # Saliency analysis is on one image at a time
        image_tensor = images[0].to(Config.DEVICE)
        label = labels[0]

        logging.info(f"\nProcessing image {processed_count + 1}/{num_images_to_process}")
        
        with torch.no_grad():
            original_output, _ = model(image_tensor.unsqueeze(0))
            original_score = original_output.item()
        
        logging.info(f"  True Authenticity: {label.item():.4f}, Predicted (Original): {original_score:.4f}")

        # Filter kwargs for calculate_saliency_map
        saliency_kwargs = {
            'sigma_list': kwargs.get('sigma_list', Config.SIGMA_LIST),
            'mask_value': kwargs.get('mask_value', Config.MASK_VALUE),
            'pixel_batch_size': kwargs.get('pixel_batch_size', Config.PIXEL_BATCH_SIZE)
        }

        saliency_map_np = calculate_saliency_map(
            model=model,
            image=image_tensor,
            original_score=original_score,
            device=Config.DEVICE,
            **saliency_kwargs
        )
        all_saliency_maps.append(saliency_map_np)

        filename_prefix = f"img_{processed_count:04d}_label_{label.item():.2f}"
        logging.info(f"  Visualizing and saving results with prefix: {filename_prefix}")
        visualize_and_save_saliency(image_tensor, saliency_map_np, model_output_dir, filename_prefix, **kwargs)
        
        processed_count += 1
    
    np.save(model_output_dir / "all_saliency_maps.npy", np.array(all_saliency_maps))
    logging.info(f"--- Saliency Analysis for {model_name} Finished ---")

def load_model_with_weights(model_class: type, weights_path: Path, device: str) -> nn.Module:
    """Loads a model and its weights onto the specified device."""
    model = model_class().to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logging.info(f"Loaded weights for {model_class.__name__} from {weights_path}")
    except Exception as e:
        logging.error(f"Error loading weights for {model_class.__name__}: {e}. Using initialized model.")
    return model

class StackingEnsemble(nn.Module):
    """The meta-learner model with one linear layer."""
    def __init__(self, input_size: int, num_classes: int = 1):
        super(StackingEnsemble, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class FullStackedEnsemble(nn.Module):
    """
    A wrapper to make the entire stacking ensemble behave like a single model.
    This is the key component for integration with the saliency analysis functions.
    """
    def __init__(self, base_models: list, meta_learner: nn.Module, device: str):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_learner = meta_learner
        self.device = device
        
        # Move all models to device once during initialization
        for model in self.base_models:
            model.to(device)
        self.meta_learner.to(device)
        self.eval()

    def forward(self, x: torch.Tensor):
        # Only move input to device, models are already there
        x = x.to(self.device)
        
        base_model_preds = []
        for model in self.base_models:
            # Remove redundant .to(device) calls
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            base_model_preds.append(pred)

        stacked_preds = torch.cat(base_model_preds, dim=1)
        # Remove redundant .to(device) call
        final_output = self.meta_learner(stacked_preds)
        
        return final_output, None

def main():
    """Main function to load models, create the ensemble, and run analysis."""
    
    # --- 1. Load Base Models and Meta-Learner ---
    logging.info("Loading all base models and the meta-learner...")
    
    base_models_loaded = []
    for model_class, weights_path, _ in Config.MODEL_CONFIGS:
        model = load_model_with_weights(model_class, weights_path, Config.DEVICE)
        base_models_loaded.append(model)

    num_base_models = len(base_models_loaded)
    meta_learner = StackingEnsemble(input_size=num_base_models).to(Config.DEVICE)
    try:
        meta_learner.load_state_dict(torch.load(Config.META_LEARNER_SAVE_PATH, map_location=Config.DEVICE))
        logging.info(f"Loaded trained meta-learner from {Config.META_LEARNER_SAVE_PATH}")
    except FileNotFoundError:
        logging.error(f"FATAL: Meta-learner weights not found at {Config.META_LEARNER_SAVE_PATH}. Cannot proceed.")
        return

    # --- 2. Create the Full Ensemble Wrapper ---
    logging.info("Creating the full stacked ensemble wrapper model...")
    full_ensemble_model = FullStackedEnsemble(
        base_models=base_models_loaded,
        meta_learner=meta_learner,
        device=Config.DEVICE
    )

    # --- 3. Prepare the DataLoader ---
    # The saliency analysis runs on test data.
    if 'test' not in IMAGENET_DATASET:
        logging.error("FATAL: Test dataset not found in IMAGENET_DATASET dictionary. Cannot create DataLoader.")
        return

    test_dataset = IMAGENET_DATASET['test']
    
    # If you want to run on specific images, create a subset
    indices_to_extract = random.sample(range(len(test_dataset)), 5)  # Adjust the number of images as needed
    print(f"Extracting {indices_to_extract} images from the test dataset for analysis.")
    test_dataset = Subset(test_dataset, indices_to_extract)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE, # Must be 1 for this analysis
        shuffle=False, # No need to shuffle for analysis
        num_workers=Config.NUM_WORKERS
    )
    logging.info(f"Test DataLoader created with {len(test_dataset)} images.")

    print(test_dataset)  # Print the first item to verify
    #print 

    # --- 4. Run Saliency Analysis on the Ensemble Model ---
    run_saliency_analysis(
        model=full_ensemble_model,
        dataloader=test_dataloader,
        output_dir=Config.MAIN_OUTPUT_DIR,
        num_images_to_process=len(test_dataset),  # Process all images in the subset
        sigma_list=Config.SIGMA_LIST,
        pixel_batch_size=Config.PIXEL_BATCH_SIZE,
        mask_value=Config.MASK_VALUE,
        vis_cmap=Config.VIS_CMAP,
        vis_alpha=Config.VIS_ALPHA,
        model_name="Stacking_Ensemble"
    )

    logging.info(f"\n--- All Saliency Analyses Completed. Outputs in '{Config.MAIN_OUTPUT_DIR}' ---")


if __name__ == '__main__':
    main()
