import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import numpy as np
import gc
import time
import re
import traceback
from pathlib import Path
import json


# Add main package to path - go up from  Experiments/ -> main/ -> Image_Authenticity_prediction/
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import models
from main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor
)

# Import utilities
from main.Utils.explainability import GradCAM, MultiscalePixelMasking
from main.Utils.cleanup import clear_gpu_memory, cleanup_model_and_data
from main.Utils.logger import info, warn, error, debug, set_level
from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    SINGLE_BATCH_SIZE,
    NUM_WORKERS,
)


# ============================================================================
# Configuration
# ============================================================================

# Model registry with their configurations
MODEL_REGISTRY = {
    'vgg16': {
        'class': VGG16AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.28',  # Last conv layer
        'input_size': 224
    },
    'vgg19': {
        'class': VGG19AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.34',  # Last conv layer
        'input_size': 224
    },
    'resnet152': {
        'class': ResNet152AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.7.2.conv3',  # Last residual block
        'input_size': 224
    },
    'densenet161': {
        'class': DenseNet161AuthenticityPredictor,
        'dataset': DENSENET_DATASET,
        'target_layer': 'features.denseblock4.denselayer24.conv2',  # Last dense block conv layer
        'input_size': 300
    },
    'efficientnetb3': {
        'class': EfficientNetB3AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.8.0',  # Last conv2d of the last block
        'input_size': 224
    },
    'barlowtwins': {
        'class': BarlowTwinsAuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.7.2.conv3',  # Last layer before avgpool
        'input_size': 224
    }
}

# Training hyperparameters

# Output directories
OUTPUT_DIR = Path('Outputs/Outouts/Experiment_2')
GRADCAM_DIR = OUTPUT_DIR / 'GradCAM_Maps'



# ============================================================================
# Experiment 2A: GradCAM-based explainability
# ============================================================================

def experiment_2a_gradcam(model_name, config, save_plots=False, show_plots=False, save_maps=True):
    """
    Conducts Experiment 2A using GradCAM for explainability.
        
    Args:
        model_name (str): Name of the model to use.
        config (dict): Configuration dictionary for the model.
        save_plots (bool): Whether to save plots.
        show_plots (bool): Whether to display plots.
        save_maps (bool): Whether to save explainability maps.
    
    Returns:
        None
    """
    print("=" * 80)
    info("EXPERIMENT 2A: GRADCAM-BASED EXPLAINABILITY")
    print("=" * 80)

    #create output directories
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    # Resolve weights dir relative to this script so behavior is robust to CWD
    OUTPUT_DIR = Path(__file__).resolve().parent / 'Outputs' / 'Experiment_1_variants'
    WEIGHTS_DIR = OUTPUT_DIR / 'Weights'

# ============================================================================
# Experiment 2B: Multiscale-pixel-masking based explainability
# ============================================================================

def experiment_2b_multiscale_pixel_masking(model_name, config, save_plots=False, show_plots=False, save_maps=True):
    pass  # Implementation of Experiment 2B goes here

# ============================================================================
# Full pipeline execution for Experiment 2
# ============================================================================

def run_experiment_2(save_plots=False, show_plots=False, save_maps=True, explainability_method='both'):
    """
    Runs the full pipeline for Experiment 2, handling different explainability methods.
    """
    # Create output directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAM_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, config in MODEL_REGISTRY.items():
        info(f"--- Processing model: {model_name.upper()} ---")

        run_gradcam = explainability_method in ['gradcam', 'both']
        run_masking = explainability_method in ['multiscale_pixel_masking', 'both']

        if not run_gradcam and not run_masking:
            warn(f"Unknown or no explainability method specified: '{explainability_method}'. Skipping model.")
            continue

        if run_gradcam:
            info(f"Starting GradCAM explainability for {model_name}")
            experiment_2a_gradcam(
                model_name=model_name,
                config=config,
                save_plots=save_plots,
                show_plots=show_plots,
                save_maps=save_maps
            )
        
        if run_masking:
            info(f"Starting Multiscale Pixel Masking explainability for {model_name}")
            experiment_2b_multiscale_pixel_masking(
                model_name=model_name,
                config=config,
                save_plots=save_plots,
                show_plots=show_plots,
                save_maps=save_maps
            )

    info("--- Experiment 2 Complete ---")
    if save_maps:
        info(f"All explainability maps saved in {CAM_DIR}")
    if show_plots:
        info("Plots were displayed during the run.")
    if save_plots:
        info(f"All plots were saved in {OUTPUT_DIR}")
        



if __name__ == '__main__':
    # Example run: adjust parameters as needed
    
    # Start timer
    start_time = time.time()
    
    run_experiment_2(
        save_plots=False,
        show_plots=False,
        save_maps=True,
        explainability_method='gradcam'  # Options: 'gradcam', 'multiscale_pixel_masking', 'both'
    )
     
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format elapsed time as H:M:S
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    info(f"\nTotal execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")