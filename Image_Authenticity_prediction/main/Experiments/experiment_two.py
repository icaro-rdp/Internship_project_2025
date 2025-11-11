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
OUTPUT_DIR = Path('Outputs/Experiment_2_variants')
XAI_MAPS_OUTPUT = OUTPUT_DIR / 'GradCAM_Maps'
WEIGHTS_DIR = OUTPUT_DIR = Path('Outputs/Experiment_1_variants/Weights')


# ============================================================================
# Experiment 2: Explainability Methods / GradCAM - Multiscale Pixel Masking
# ============================================================================

def generate_explainability_maps(xai_method="both",save_plots=False, show_plots=False, save_maps=True, verbose=True, models_to_test=None):
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
    XAI_MAPS_OUTPUT.mkdir(parents=True, exist_ok=True)
    # Resolve weights dir relative to this script so behavior is robust to CWD

    if not WEIGHTS_DIR.exists():
        error(f"Weights directory not found: {WEIGHTS_DIR}")
        return {}
    
    # Collect all .pth files 
    all_pth_files = sorted(WEIGHTS_DIR.glob('*.pth'))
    print(f"Found {len(all_pth_files)} weight files in {WEIGHTS_DIR}")

    if not all_pth_files:
        error(f"No .pth files found in {WEIGHTS_DIR}. Please run Experiment 1A and 1B first.")
        return {}
    
    # Group by model name extracted from filename prefix like 'vgg16_exp1a...'
    modelname_re = re.compile(r'^([A-Za-z0-9_]+)_exp1')
    weights_files_by_model = {}
    skipped_files = []
    
    for p in all_pth_files:
        m = modelname_re.match(p.name)
        if not m:
            # if filename does not follow naming convention, skip but record for inspection
            skipped_files.append(p)
            continue
            
        mn = m.group(1)
        
        # If user asked to test only specific models, skip others
        if models_to_test is not None and mn not in models_to_test:
            continue
            
        # Only include files for known models (we need config to instantiate the model)
        if mn not in MODEL_REGISTRY:
            warn(f"Found weights for unknown model '{mn}' -> skipping file {p.name}")
            skipped_files.append(p)
            continue
            
        weights_files_by_model.setdefault(mn, []).append(p)
        
    if not weights_files_by_model:
        error(f"No valid model weight files found in {WEIGHTS_DIR} (checked {len(all_pth_files)} files).")
        if skipped_files:
            info("Skipped files:")
            for s in skipped_files:
                info(f" - {s.name}")
        return {}
        
    total_files_found = sum(len(v) for v in weights_files_by_model.values())
    info(f"Found {total_files_found} weight file(s) across {len(weights_files_by_model)} model type(s) in {WEIGHTS_DIR}.")
    
    if verbose:
        info("Per-model file counts:")
        for mn, fls in sorted(weights_files_by_model.items()):
            info(f" - {mn}: {len(fls)} file(s)")
    
    
# ============================================================================
# Full pipeline execution for Experiment 2
# ============================================================================

def run_experiment_2(models=None,save_plots=False, show_plots=False, save_maps=True, xai_methods='both'):
    """
    Runs the full pipeline for Experiment 2, handling different explainability methods.
    """
    # Create output directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    XAI_MAPS_OUTPUT.mkdir(parents=True, exist_ok=True)


    run_gradcam = xai_methods in ['gradcam', 'both']
    run_masking = xai_methods in ['multiscale_pixel_masking', 'both']

    if not run_gradcam and not run_masking:
        warn(f"Unknown or no explainability method specified: '{xai_methods}'. Skipping model.")
        return

    if run_gradcam:
        info(f"Starting GradCAM explainability maps generation...")
        generate_explainability_maps(
            xai_method='gradcam',
            save_plots=save_plots,
            show_plots=show_plots,
            save_maps=save_maps
        )
    
    if run_masking:
        info(f"Starting Multiscale Pixel Masking explainability maps generation...")
        generate_explainability_maps(
            xai_method='multiscale_pixel_masking',
            save_plots=save_plots,
            show_plots=show_plots,
            save_maps=save_maps
        )

    info("--- Experiment 2 Complete ---")
    if save_maps:
        info(f"All explainability maps saved in {XAI_MAPS_OUTPUT}")
    if show_plots:
        info("Plots were displayed during the run.")
    if save_plots:
        info(f"All plots were saved in {OUTPUT_DIR}")
        



if __name__ == '__main__':
    # Example run: adjust parameters as needed
    
    # Start timer
    start_time = time.time()
    
    run_experiment_2(
        models=None,  # Use all models in MODEL_REGISTRY
        save_plots=False,
        show_plots=False,
        save_maps=True,
        xai_methods='gradcam'  # Options: 'gradcam', 'multiscale_pixel_masking', 'both'
    )
     
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format elapsed time as H:M:S
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    info(f"\nTotal execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")