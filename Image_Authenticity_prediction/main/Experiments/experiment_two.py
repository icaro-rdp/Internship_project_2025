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

# XAI method specific configurations
CONFIG = {
    'sigma_values' : [3, 5, 9, 17, 33, 65],
    'mask_value':0,
    'pixel_batch_size':128
}
# Training hyperparameters

# Output directories
OUTPUT_DIR = Path('Outputs/Experiment_2_variants')
XAI_MAPS_OUTPUT = OUTPUT_DIR / 'XAI_Maps'
GRADCAM_OUTPUT = XAI_MAPS_OUTPUT / 'GradCAM'
MPM_OUTPUT = XAI_MAPS_OUTPUT / 'Multiscale_Pixel_Masking'
WEIGHTS_DIR = OUTPUT_DIR = Path('Outputs/Experiment_1_variants/Weights')


# ============================================================================
# Experiment 2: Explainability Methods / GradCAM - Multiscale Pixel Masking
# ============================================================================

def generate_explainability_maps(
    variants="all",
    xai_method="both",
    save_plots=False,
    show_plots=False,
    save_maps=True,
    verbose=True,
    models_to_test=None,
):
    """
    Conducts Experiment 2A using GradCAM for explainability.
        
    Args:
        models_to_test (list or None): List of model names to test. If None, tests all models in MODEL_REGISTRY.
        model_name (str): Name of the model to use.
        config (dict): Configuration dictionary for the model.
        save_plots (bool): Whether to save plots.
        show_plots (bool): Whether to display plots.
        save_maps (bool): Whether to save explainability maps.
    variants (str|Sequence[str]): Which variants to test ('all' | 'base' | 'greedy' | 'negative' | 'orig').
    
    Returns:
        None
    """
    print("=" * 80)
    info("EXPERIMENT 2A: GRADCAM-BASED EXPLAINABILITY")
    print("=" * 80)

    #create output directories
    XAI_MAPS_OUTPUT.mkdir(parents=True, exist_ok=True)
    GRADCAM_OUTPUT.mkdir(parents=True, exist_ok=True)
    MPM_OUTPUT.mkdir(parents=True, exist_ok=True)

    if isinstance(variants, str):
        requested_variants = {variants.lower()}
    else:
        requested_variants = {str(v).lower() for v in variants}

    if not requested_variants:
        requested_variants = {"all"}

    valid_variant_keys = {"all", "base", "greedy", "negative", "orig"}
    invalid_variant_keys = requested_variants - valid_variant_keys
    if invalid_variant_keys:
        warn(
            "Unknown variant selection %s. Supported options are %s."
            % (sorted(invalid_variant_keys), sorted(valid_variant_keys))
        )
        requested_variants -= invalid_variant_keys

    if not requested_variants:
        warn("No valid variants requested. Aborting explainability run.")
        return {}

    include_all_variants = "all" in requested_variants
    selected_variants = requested_variants - {"all"}

    valid_methods = {"gradcam", "multiscale_pixel_masking", "both"}
    if xai_method not in valid_methods:
        warn(
            "Unknown explainability method '%s'. Choose from %s."
            % (xai_method, sorted(valid_methods))
        )
        return {}

    run_gradcam = xai_method in {"gradcam", "both"}
    run_masking = xai_method in {"multiscale_pixel_masking", "both"}

    def variant_matches(tag: str) -> bool:
        tag_lower = tag.lower()
        if include_all_variants:
            return True
        if "base" in selected_variants and (
            tag_lower == "orig" or tag_lower.startswith("exp1a_variant")
        ):
            return True
        if "orig" in selected_variants and tag_lower == "orig":
            return True
        if "greedy" in selected_variants and "greedy_pruned" in tag_lower:
            return True
        if "negative" in selected_variants and "negative_pruned" in tag_lower:
            return True
        return False

    # Verify weights directory exists
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
    
    # Results storage for numpy maps 
    results = {}
    
    # run explainability on each model (and its variant files)
    model_items = list(weights_files_by_model.items())
    
    for idx, (model_name, weight_file_list) in enumerate(model_items, 1):
        info(f"[{idx}/{len(model_items)}] XAI on {model_name.upper()}")
        print("-" * 80)

        try:
            config = MODEL_REGISTRY[model_name]
            
            dataset = config['dataset']
            test_loader = DataLoader(
                dataset['test'],
                batch_size=SINGLE_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS
            )
            
            # Prepare results container for this model
            results[model_name] = {}
            
            # Iterate over all weight files (variants) for this model
            for weights_path in weight_file_list:
                # prepare the naming convention for outputs
                m = re.search(r"exp1a_variant\d+|exp1b_variant\d+_greedy_pruned|exp1b_variant\d+_negative_pruned|orig", str(weights_path))
                variant_tag = m.group(0) if m else 'orig'
                model_name_output = f"{model_name}_{variant_tag}"

                if not variant_matches(variant_tag):
                    debug(
                        "Skipping variant '%s' for model '%s' based on selection %s."
                        % (variant_tag, model_name, sorted(selected_variants) if selected_variants else ['all'])
                    )
                    continue

                if verbose:
                    info(f"Loading model from {weights_path}...")

                # Instantiate model and load weights
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = config['class'](freeze_backbone=False)
                model.load_state_dict(torch.load(weights_path, weights_only=True))
                model.to(device)

                variant_result = results[model_name].setdefault(variant_tag, {})

                if run_gradcam:
                    print("Running GradCAM...")
                    # for each image in test set, generate GradCAM map
                    maps = []
                    for img_idx, (img, label) in enumerate(test_loader):
                        img = img.to(device)
                        label = label.to(device)

                        gradcam = GradCAM(
                            model=model,
                            target_layer=config['target_layer'],
                        )
                        gradcam_map = gradcam.generate_map(img, target_index=0)
                        gradcam.cleanup()
                        print (f"GradCAM map shape: {gradcam_map.shape}")
                        maps.append(gradcam_map)

                    if save_maps:
                        # Save the numpy array to disk
                        print(f"Saving GradCAM map for {model_name} with shape {np.array(maps).shape}...")
                        maps_array = np.array(maps)
                        gradcam_map_path = GRADCAM_OUTPUT / f"{model_name_output}_maps.npy"
                        np.save(gradcam_map_path, maps_array)
                        variant_result['gradcam_map_path'] = str(gradcam_map_path)
                    variant_result['gradcam_sample_count'] = len(maps)

                    

                if run_masking:
                    print("Running Multiscale Pixel Masking...")
                    # for each image in test set, generate MPM map
                    maps = []
                    for img_idx, (img, label) in enumerate(test_loader):
                        img = img.to(device)
                        label = label.to(device)
                        
                        mpm = MultiscalePixelMasking(
                            model=model,
                            sigma_list=CONFIG['sigma_values'],
                            pixel_batch_size=CONFIG['pixel_batch_size'],
                            mask_value=CONFIG['mask_value'])
                        
                        mpm_map = mpm.generate_map(img, target_index=0)
                        maps.append(mpm_map)

                    if save_maps:
                        print(f"Saving Multiscale Pixel Masking map for {model_name})...")
                        maps_array = np.array(maps)
                        mpm_map_path = MPM_OUTPUT / f"{model_name_output}_maps.npy"
                        np.save(mpm_map_path, maps_array)
                        variant_result['mpm_map_path'] = str(mpm_map_path)
                    variant_result['mpm_sample_count'] = len(maps)
                         
        except Exception as e:
            error(f"Error testing {model_name}: {e}")
            error(traceback.format_exc())
                 
        finally:
            # Clean up memory after each model (success or failure)
            info(f"Cleaning up {model_name} from memory...")
            cleanup_model_and_data(
                model=locals().get('model'),
                dataloaders=locals().get('test_loader'),
                optimizer=None
            )
            clear_gpu_memory()
            info(f"✓ {model_name} memory cleaned")

    return results
            
    
        
# ============================================================================
# Full pipeline execution for Experiment 2
# ============================================================================

def run_experiment_2(
    models=None,
    save_plots=False,
    show_plots=False,
    save_maps=True,
    variants='all',
    xai_methods='both',
):
    """
    Runs the full pipeline for Experiment 2, handling different explainability methods.
    """
    # Create output directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    XAI_MAPS_OUTPUT.mkdir(parents=True, exist_ok=True)


    run_gradcam = xai_methods in ['gradcam', 'both']
    run_masking = xai_methods in ['mpm', 'both']

    if not run_gradcam and not run_masking:
        warn(f"Unknown or no explainability method specified: '{xai_methods}'. Skipping model.")
        return

    if isinstance(models, str):
        models_to_test = [models]
    elif models is None:
        models_to_test = None
    else:
        models_to_test = list(models)

    if models_to_test is not None:
        unknown_models = [m for m in models_to_test if m not in MODEL_REGISTRY]
        if unknown_models:
            warn(
                "Unknown model(s) requested %s. Available models are %s."
                % (unknown_models, sorted(MODEL_REGISTRY))
            )
            models_to_test = [m for m in models_to_test if m in MODEL_REGISTRY]
        if not models_to_test:
            warn("No valid models requested. Aborting Experiment 2 run.")
            return

    if run_gradcam:
        info(f"Starting GradCAM explainability maps generation...")
        generate_explainability_maps(
            xai_method='gradcam',
            save_plots=save_plots,
            show_plots=show_plots,
            save_maps=save_maps,
            variants=variants,
            models_to_test=models_to_test,
        )
    
    if run_masking:
        info(f"Starting Multiscale Pixel Masking explainability maps generation...")
        generate_explainability_maps(
            xai_method='multiscale_pixel_masking',
            save_plots=save_plots,
            show_plots=show_plots,
            save_maps=save_maps,
            variants=variants,
            models_to_test=models_to_test,
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
        xai_methods='mpm',  # Options: 'gradcam', 'mpm', 'both'
        variants='greedy'  # Options: 'all', 'base', 'greedy', 'negative', 'orig'
    )
     
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format elapsed time as H:M:S
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    info(f"\nTotal execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")