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
from main.Utils import FeatureMapsPruner
from main.Utils.pruning import check_pruning_statistics
from main.Utils.cleanup import clear_gpu_memory, cleanup_model_and_data
from main.Utils.logger import info, warn, error, debug
from main.train import train_model, test_model, plot_loss_history
from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    BATCH_SIZE,
    NUM_WORKERS,
    imageNet_dataset,
    denseNet_dataset
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
    # 'inceptionv3': {
    #     'class': InceptionV3AuthenticityPredictor,
    #     'dataset': INCEPTIONV3_DATASET,
    #     'target_layer': None,  # InceptionV3 cannot be pruned with current method
    #     'input_size': 299
    # },
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
TRAINING_CONFIG = {
    'max_epochs': 500,
    'patience': 15,
    'learning_rate': 0.001,
    'freeze_backbone': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Pruning configuration
PRUNING_CONFIG = {
    'force_recompute': False,  # Set to True to recompute importance scores
}

# Output directories
OUTPUT_DIR = Path('Outputs/Experiment_1_variants')
WEIGHTS_DIR = OUTPUT_DIR / 'Weights'
RANKINGS_DIR = OUTPUT_DIR / 'Ranking_arrays'
RANKING_PLOTS_DIR = OUTPUT_DIR / 'Ranking_Plots'
TRAINING_PLOTS_DIR = OUTPUT_DIR / 'Training_Plots'
TRAINING_HISTORY_DIR = OUTPUT_DIR / 'Training_History'
TEST_RESULTS_DIR = OUTPUT_DIR / 'Test_Results'



# ============================================================================
# Experiment 1A: Train All Models
# ============================================================================

def experiment_1a_train_all_models(
    models_to_train=None,
    save_plots=True,
    verbose=True,
    global_test_indices=None
):
    """
    Experiment 1A: Train all model architectures with early stopping.
    
    Args:
        models_to_train (list, optional): List of model names to train.
                                         If None, trains all models.
        save_plots (bool): Whether to save training history plots.
        verbose (bool): Whether to print detailed progress.
        global_test_indices (list, optional): Pre-defined global test indices.
                                              If None, creates them internally.
    
    Returns:
        dict: Dictionary containing training results for each model.
    """
    info("=" * 80)
    info("EXPERIMENT 1A: TRAINING ALL MODELS")
    info("=" * 80)
    
    # Create output directories
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Number of variants per model (default 10 variants)
    variants_per_model = 10

    # Prepare a global test index set so the test split is identical across
    # all models and variants. We derive it from the imageNet_dataset length
    # (both imageNet_dataset and denseNet_dataset are built from the same CSV
    # so indices align).
    if global_test_indices is None:
        try:
            total_size = len(imageNet_dataset)
            test_size = int(0.2 * total_size)
            # Use seed 42 for global test selection as requested
            gen_global = torch.Generator().manual_seed(42)
            perm = torch.randperm(total_size, generator=gen_global).tolist()
            GLOBAL_TEST_INDICES = perm[:test_size]
        except Exception:
            GLOBAL_TEST_INDICES = None
    else:
        GLOBAL_TEST_INDICES = global_test_indices

    # Select models to train
    if models_to_train is None:
        models_to_train = list(MODEL_REGISTRY.keys())
    
    # Results storage
    results = {}
    
    # Train each model
    for idx, model_name in enumerate(models_to_train, 1):
        info(f"[{idx}/{len(models_to_train)}] Training {model_name.upper()}")
        info("-" * 80)
        
        try:
            # Get model configuration
            config = MODEL_REGISTRY[model_name]
            # We'll create multiple variants by reinitializing the regression head
            # and by using different random splits for train/test.
            if verbose:
                info(f"Preparing to train {variants_per_model} variants for {model_name}...")

            # Determine the base full dataset for this model so we can re-split
            # Use imageNet_dataset / denseNet_dataset depending on config
            backbone_dataset = None
            if config['dataset'] is IMAGENET_DATASET:
                backbone_dataset = imageNet_dataset
            elif config['dataset'] is DENSENET_DATASET:
                backbone_dataset = denseNet_dataset
            else:
                # Fall back to provided split if we can't find base dataset
                backbone_dataset = None

            # Train variants
            for variant_idx in range(1, variants_per_model + 1):
                info(f"Variant {variant_idx}/{variants_per_model} for {model_name}")

                # Initialize model for this variant
                if verbose:
                    debug(f"Initializing {model_name} model (variant {variant_idx})...")
                model = config['class'](freeze_backbone=TRAINING_CONFIG['freeze_backbone'])

                # Reinitialize regression head weights to get a distinct starting state
                try:
                    def reset_regression_head(m):
                        import torch.nn as _nn
                        for layer in m.regression_head.modules():
                            if isinstance(layer, _nn.Linear):
                                layer.reset_parameters()

                    reset_regression_head(model)
                except Exception:
                    # If model does not have regression_head attribute, ignore
                    pass

                # Create train/val/test split for this variant by using a different seed
                from torch.utils.data import random_split
                import torch as _torch

                if backbone_dataset is not None:
                    total_size = len(backbone_dataset)
                    # Use the global test indices if available to ensure same test set
                    if GLOBAL_TEST_INDICES is not None:
                        test_indices = GLOBAL_TEST_INDICES
                        # Remaining indices for training and validation
                        all_indices = list(range(total_size))
                        remaining_indices = [i for i in all_indices if i not in test_indices]

                        # Shuffle remaining indices per-variant to alter train/val split
                        gen = _torch.Generator().manual_seed(42 + variant_idx)
                        perm_remaining = _torch.randperm(len(remaining_indices), generator=gen).tolist()
                        shuffled_remaining = [remaining_indices[i] for i in perm_remaining]

                        # Split remaining into train and val (80% train, 10% val of total, since test 10%)
                        val_size = int(0.1 * total_size)
                        train_size = len(shuffled_remaining) - val_size

                        train_indices = shuffled_remaining[:train_size]
                        val_indices = shuffled_remaining[train_size:]

                        train_ds = Subset(backbone_dataset, train_indices)
                        val_ds = Subset(backbone_dataset, val_indices)
                        test_ds = Subset(backbone_dataset, test_indices)
                    else:
                        # Fallback to random_split if global indices not available
                        train_size = int(0.8 * total_size)
                        val_size = int(0.1 * total_size)
                        gen = _torch.Generator().manual_seed(42 + variant_idx)
                        train_ds, val_ds, test_ds = random_split(backbone_dataset, [train_size, val_size, total_size - train_size - val_size], generator=gen)
                else:
                    # Fall back to using the already-split dataset provided in config
                    train_ds = config['dataset']['train']
                    val_ds = config['dataset']['val']
                    test_ds = config['dataset']['test']

                train_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS
                )

                dataloaders = {'train': train_loader, 'val': val_loader}

                # Setup training for this variant
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=TRAINING_CONFIG['learning_rate']
                )

                if verbose:
                    info(f"Training on device: {TRAINING_CONFIG['device']}")
                    info(f"Max epochs: {TRAINING_CONFIG['max_epochs']}, Patience: {TRAINING_CONFIG['patience']}")

                # Train the model for this variant
                best_model, history = train_model(
                    model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=TRAINING_CONFIG['max_epochs'],
                    device=TRAINING_CONFIG['device'],
                    patience=TRAINING_CONFIG['patience']
                )

                # Evaluate on test set
                info("Evaluating on test set...")
                test_loss = test_model(
                    best_model,
                    test_loader,
                    criterion,
                    TRAINING_CONFIG['device']
                )
                test_rmse = np.sqrt(test_loss)  # Compute RMSE from MSE

                # Save model weights for this variant
                weights_path = WEIGHTS_DIR / f"{model_name}_exp1a_variant{variant_idx}_best.pth"
                torch.save(best_model.state_dict(), weights_path)
                info(f"✓ Variant weights saved to: {weights_path}")

                # Save training history for this variant
                history_path = TRAINING_HISTORY_DIR / f"{model_name}_exp1a_variant{variant_idx}_history.npy"
                np.save(history_path, history)

                # Plot and save training history (single image per model; variants share same filename)
                if save_plots:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')  # Use non-interactive backend
                        plot_loss_history(history)
                        import matplotlib.pyplot as plt
                        plot_path = TRAINING_PLOTS_DIR / f"{model_name}_exp1a_training_curve_variant{variant_idx}.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        info(f"✓ Training curve saved to: {plot_path}")
                    except Exception as e:
                        warn(f"Could not save plot: {e}")

                # Store results per-variant inside model dict
                results.setdefault(model_name, {})[f'variant{variant_idx}'] = {
                    'final_test_mse': test_loss,
                    'final_test_rmse': test_rmse,
                    'final_val_loss': history['val_loss'][-1],
                    'best_val_loss': min(history['val_loss']),
                    'epochs_trained': len(history['train_loss']),
                    'weights_path': str(weights_path),
                    'history': history
                }

                info(f"✓ {model_name.upper()} variant {variant_idx} training complete!")
                info(f"  - Test MSE: {test_loss:.4f}, RMSE: {test_rmse:.4f}")
                info(f"  - Best Val Loss: {results[model_name][f'variant{variant_idx}']['best_val_loss']:.4f}")
                info(f"  - Epochs trained: {results[model_name][f'variant{variant_idx}']['epochs_trained']}")
            
            # After all variants for this model have been processed, build a
            # compact per-model summary so downstream code/tests that expect
            # top-level keys like 'final_test_mse' will work.
            try:
                if model_name in results and isinstance(results[model_name], dict):
                    # Collect only variant entries that have test metrics
                    variant_items = [ (k, v) for k, v in results[model_name].items()
                                      if isinstance(v, dict) and 'final_test_mse' in v ]
                    if variant_items:
                        # Best variant by lowest test MSE
                        best_variant, best_data = min(variant_items, key=lambda iv: iv[1]['final_test_mse'])
                        # Averages across variants
                        avg_mse = float(np.mean([v['final_test_mse'] for _, v in variant_items]))
                        avg_rmse = float(np.mean([v['final_test_rmse'] for _, v in variant_items]))

                        # Populate model-level summary fields
                        results[model_name]['best_variant'] = best_variant
                        results[model_name]['final_test_mse'] = best_data['final_test_mse']
                        results[model_name]['final_test_rmse'] = best_data['final_test_rmse']
                        results[model_name]['avg_test_mse'] = avg_mse
                        results[model_name]['avg_test_rmse'] = avg_rmse
                        # Keep reference to the best weights file if available
                        results[model_name]['weights_path'] = best_data.get('weights_path')
                        # Add some aggregated training info
                        try:
                            results[model_name]['best_val_loss'] = float(min([v.get('best_val_loss', np.inf) for _, v in variant_items]))
                        except Exception:
                            pass
                        try:
                            results[model_name]['epochs_trained'] = int(np.round(np.mean([v.get('epochs_trained', 0) for _, v in variant_items])))
                        except Exception:
                            pass
            except Exception:
                # Protect summary aggregation from breaking the main training loop
                pass

        except Exception as e:
            error(f"Error training {model_name}: {e}")
            error(traceback.format_exc())
            results[model_name] = {'error': str(e)}
        
        finally:
            # Clean up memory after each model (success or failure)
            info(f"Cleaning up {model_name} from memory...")
            cleanup_model_and_data(
                model=locals().get('best_model') or locals().get('model'),
                dataloaders=locals().get('dataloaders'),
                optimizer=locals().get('optimizer')
            )
            info(f"✓ {model_name} memory cleaned")
    
    # Print summary
    info("=" * 80)
    info("EXPERIMENT 1A: TRAINING SUMMARY")
    info("=" * 80)
    
    # Collect all successful variants
    all_variants = []
    for model_name, model_results in results.items():
        if 'error' in model_results:
            continue
        for variant_key, variant_data in model_results.items():
            if isinstance(variant_data, dict) and 'final_test_mse' in variant_data:
                all_variants.append((f"{model_name}_{variant_key}", variant_data['final_test_mse'], variant_data['final_test_rmse']))
    
    successful_variants = len(all_variants)
    total_expected = len(models_to_train) * variants_per_model
    failed_models = [k for k, v in results.items() if 'error' in v]
    
    info(f"Successfully trained: {successful_variants}/{total_expected} variants")
    
    if all_variants:
        info("Variant Performance (Test Loss):")
        info(f"{'Variant':<25} {'MSE':<10} {'RMSE':<10}")
        info("-" * 47)
        sorted_variants = sorted(all_variants, key=lambda x: x[1])  # Sort by MSE
        for rank, (variant_name, test_mse, test_rmse) in enumerate(sorted_variants, 1):
            info(f"{rank}. {variant_name:<22} {test_mse:<10.4f} {test_rmse:<10.4f}")
    
    if failed_models:
        warn(f"Failed models: {', '.join(failed_models)}")
    
    info("=" * 80)
    
    # Final memory cleanup
    info("Performing final memory cleanup...")
    clear_gpu_memory()
    
    return results

# ============================================================================
# Experiment 1B: Prune All Trained Models
# ============================================================================

def experiment_1b_prune_all_models(
    models_to_prune=None,
    pruning_method='greedy',
    threshold=0.0,
    verbose=True,
    global_test_indices=None
):
    """
    Experiment 1B: Prune trained models using feature importance analysis.
    
    Args:
        models_to_prune (list, optional): List of model names to prune.
                                         If None, prunes all trained models.
        pruning_method (str): Pruning method to use ('greedy', 'negative_impact', or 'both').
        threshold (float): Threshold for negative_impact pruning (ignored for greedy).
        verbose (bool): Whether to print detailed progress.
        global_test_indices (list, optional): Pre-defined global test indices.
                                              If None, falls back to config dataset.
    
    Returns:
        dict: Dictionary containing pruning results for each model.
              If pruning_method='both', each model will have 'greedy' and 'negative_impact' sub-dicts.
    """
    info("=" * 80)
    info("EXPERIMENT 1B: PRUNING ALL TRAINED MODELS")
    info("=" * 80)
    
    # Create output directories
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
    RANKING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple behavior: use every .pth file in WEIGHTS_DIR (process each file)
    # Resolve weights dir relative to this script so behavior is robust to CWD
    OUTPUT_DIR = Path(__file__).resolve().parent / 'Outputs' / 'Experiment_1_variants'
    WEIGHTS_DIR = OUTPUT_DIR / 'Weights'

    if not WEIGHTS_DIR.exists():
        error(f"Weights directory not found: {WEIGHTS_DIR}")
        return {}

    # Collect all .pth files (you asked to simply take all files in the Weights folder)
    all_pth_files = sorted(WEIGHTS_DIR.glob('*.pth'))
    if not all_pth_files:
        error(f"No .pth files found in {WEIGHTS_DIR}. Please run Experiment 1A first.")
        return {}

    # Group by model name extracted from filename prefix like 'vgg16_exp1a...'
    import re
    modelname_re = re.compile(r'^([A-Za-z0-9_]+)_exp1a')
    weights_files_by_model = {}
    skipped_files = []
    for p in all_pth_files:
        m = modelname_re.match(p.name)
        if not m:
            # if filename does not follow naming convention, skip but record for inspection
            skipped_files.append(p)
            continue
        mn = m.group(1)
        # If user asked to prune only specific models, skip others
        if models_to_prune is not None and mn not in models_to_prune:
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
                info(f"  - {s.name}")
        return {}

    total_files_found = sum(len(v) for v in weights_files_by_model.values())
    info(f"Found {total_files_found} weight file(s) across {len(weights_files_by_model)} model type(s) in {WEIGHTS_DIR}.")
    if verbose:
        info("Per-model file counts:")
        for mn, fls in sorted(weights_files_by_model.items()):
            info(f"  - {mn}: {len(fls)} file(s)")
    # Determine which methods to run
    methods_to_run = []
    if pruning_method == 'both':
        methods_to_run = ['greedy', 'negative_impact']
    elif pruning_method in ['greedy', 'negative_impact']:
        methods_to_run = [pruning_method]
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}. Use 'greedy', 'negative_impact', or 'both'.")
    
    # Results storage
    results = {}
    
    # Prune each model (and its variant files)
    model_items = list(weights_files_by_model.items())
    for idx, (model_name, weight_file_list) in enumerate(model_items, 1):
        info(f"[{idx}/{len(model_items)}] Pruning {model_name.upper()}")
        info("-" * 80)

        try:
            config = MODEL_REGISTRY[model_name]

            # Prepare test loader (same for all variants unless dataset differs)
            # Use global test indices if provided, else fallback to config dataset
            if global_test_indices is not None:
                # Determine the base full dataset for this model
                backbone_dataset = None
                if config['dataset'] is IMAGENET_DATASET:
                    backbone_dataset = imageNet_dataset
                elif config['dataset'] is DENSENET_DATASET:
                    backbone_dataset = denseNet_dataset
                else:
                    backbone_dataset = None
                
                if backbone_dataset is not None:
                    test_ds = Subset(backbone_dataset, global_test_indices)
                    test_loader = DataLoader(
                        test_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS
                    )
                else:
                    # Fallback
                    dataset = config['dataset']
                    test_loader = DataLoader(
                        dataset['test'],
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS
                    )
            else:
                # Fallback to original
                dataset = config['dataset']
                test_loader = DataLoader(
                    dataset['test'],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS
                )

            # Setup criterion and device
            criterion = nn.MSELoss()
            device = torch.device(TRAINING_CONFIG['device'])

            # Prepare results container for this model
            if len(methods_to_run) > 1:
                results[model_name] = {}
            else:
                results[model_name] = {}

            # Iterate over all weight files (variants) for this model
            import re
            for weights_path in weight_file_list:
                if verbose:
                    info(f"Loading trained model from {weights_path}...")
                model = config['class'](freeze_backbone=False)
                # Load only tensor weights (safer against pickle attacks). The checkpoints
                # saved by this project are plain state_dicts, so weights_only=True is appropriate.
                model.load_state_dict(torch.load(weights_path, weights_only=True))

                target_layer = config['target_layer']
                if verbose:
                    info(f"Target layer for pruning: {target_layer}")

                pruner = FeatureMapsPruner(
                    model=model,
                    dataloader=test_loader,
                    layer_name=target_layer,
                    criterion=criterion,
                    eval_function=test_model,
                    device=device
                )

                # Compute importance scores per-variant
                info("Computing feature importance scores...")
                # derive a variant tag from filename
                m = re.search(r"variant\d+", str(weights_path))
                variant_tag = m.group(0) if m else 'orig'
                importance_path = RANKINGS_DIR / f"{model_name}_exp1b_{variant_tag}_importance_scores.npy"
                importance_scores = pruner.compute_importance_scores(
                    save_path=str(importance_path),
                    force_recompute=PRUNING_CONFIG['force_recompute']
                )

                # Plot and save importance scores per-variant
                info("Generating importance score plot...")
                plot_path = RANKING_PLOTS_DIR / f"{model_name}_exp1b_{variant_tag}_importance_scores.png"
                try:
                    pruner.plot_importance_scores(save_path=str(plot_path))
                    info(f"✓ Importance score plot saved to: {plot_path}")
                except Exception as e:
                    warn(f"Could not save importance plot: {e}")

                info(f"✓ Importance scores computed and saved to: {importance_path}")
                info(f"  - Baseline MSE: {pruner.baseline_mse:.4f}, RMSE: {pruner.baseline_rmse:.4f}")
                info(f"  - Number of feature maps: {len(importance_scores)}")

                # Perform pruning methods for this variant
                for method in methods_to_run:
                    if method == 'greedy':
                        info("Performing greedy pruning...")
                        pruned_weights_path = WEIGHTS_DIR / f"{model_name}_exp1b_{variant_tag}_greedy_pruned.pth"
                        pruning_results = pruner.greedy_pruning(
                            model_save_path=str(pruned_weights_path)
                        )

                    elif method == 'negative_impact':
                        info(f"Performing negative impact pruning (threshold={threshold})...")
                        pruned_weights_path = WEIGHTS_DIR / f"{model_name}_exp1b_{variant_tag}_negative_pruned.pth"
                        pruning_results = pruner.negative_impact_pruning(
                            model_save_path=str(pruned_weights_path),
                            threshold=threshold
                        )

                    # Prepare result dict for this variant+method
                    method_results = {
                        'baseline_mse': pruning_results['baseline_mse'],
                        'baseline_rmse': pruning_results['baseline_rmse'],
                        'final_mse': pruning_results['final_mse'],
                        'final_rmse': pruning_results['final_rmse'],
                        'improvement_mse': pruning_results['improvement_mse'],
                        'improvement_rmse': pruning_results['improvement_rmse'],
                        'removed_features': pruning_results['removed_features'],
                        'num_removed': len(pruning_results['removed_features']),
                        'reduction_percentage': pruning_results['reduction_percentage'],
                        'pruned_weights_path': str(pruned_weights_path),
                        'importance_scores_path': str(importance_path),
                        'variant_tag': variant_tag,
                        'original_weights_path': str(weights_path)
                    }

                    if method == 'greedy' and 'mse_history' in pruning_results:
                        method_results['mse_history'] = pruning_results['mse_history']

                    # Store results
                    results.setdefault(model_name, {}).setdefault(variant_tag, {})[method] = method_results

                    # Print method-specific results
                    info(f"✓ {model_name.upper()} {variant_tag} {method} pruning complete!")
                    info(f"  - Baseline MSE: {pruning_results['baseline_mse']:.4f}, RMSE: {pruning_results['baseline_rmse']:.4f}")
                    info(f"  - Final MSE: {pruning_results['final_mse']:.4f}, RMSE: {pruning_results['final_rmse']:.4f}")
                    info(f"  - Improvement MSE: {pruning_results['improvement_mse']:.4f}, RMSE: {pruning_results['improvement_rmse']:.4f}")
                    info(f"  - Features removed: {len(pruning_results['removed_features'])}")
                    info(f"  - Reduction: {pruning_results['reduction_percentage']:.1f}%")
            
            # After processing all variants for this model, aggregate results across variants for each method
            for method in methods_to_run:
                variant_results = [results[model_name][vt][method] for vt in results[model_name] if isinstance(results[model_name][vt], dict) and method in results[model_name][vt]]
                if variant_results:
                    # Compute averages across variants
                    avg_baseline_mse = float(np.mean([r['baseline_mse'] for r in variant_results]))
                    avg_baseline_rmse = float(np.mean([r['baseline_rmse'] for r in variant_results]))
                    avg_final_mse = float(np.mean([r['final_mse'] for r in variant_results]))
                    avg_final_rmse = float(np.mean([r['final_rmse'] for r in variant_results]))
                    avg_improvement_mse = float(np.mean([r['improvement_mse'] for r in variant_results]))
                    avg_improvement_rmse = float(np.mean([r['improvement_rmse'] for r in variant_results]))
                    avg_num_removed = float(np.mean([r['num_removed'] for r in variant_results]))
                    avg_reduction_percentage = float(np.mean([r['reduction_percentage'] for r in variant_results]))

                    # Store aggregated results at model level
                    results[model_name][method] = {
                        'baseline_mse': avg_baseline_mse,
                        'baseline_rmse': avg_baseline_rmse,
                        'final_mse': avg_final_mse,
                        'final_rmse': avg_final_rmse,
                        'improvement_mse': avg_improvement_mse,
                        'improvement_rmse': avg_improvement_rmse,
                        'num_removed': avg_num_removed,
                        'reduction_percentage': avg_reduction_percentage,
                        'variants': [r['variant_tag'] for r in variant_results]
                    }
            
            
        except Exception as e:
            error(f"Error pruning {model_name}: {e}")
            error(traceback.format_exc())
            results[model_name] = {'error': str(e)}
        
        finally:
            # Clean up memory after each model (success or failure)
            info(f"Cleaning up {model_name} from memory...")
            cleanup_model_and_data(
                model=locals().get('model'),
                dataloaders=locals().get('test_loader'),
                optimizer=None
            )
            # Also cleanup the pruner object which holds references
            if 'pruner' in locals():
                del pruner
            clear_gpu_memory()
            info(f"✓ {model_name} memory cleaned")
    
    # Print summary
    info("=" * 80)
    info("EXPERIMENT 1B: PRUNING SUMMARY")
    info("=" * 80)
    
    successful_models = [k for k, v in results.items() if 'error' not in v]
    failed_models = [k for k, v in results.items() if 'error' in v]
    
    # Use the number of discovered models as the denominator to avoid errors
    info(f"Successfully pruned: {len(successful_models)}/{len(weights_files_by_model)} models")
    
    if successful_models:
        if len(methods_to_run) > 1:
            # Print results for both methods
            for method in methods_to_run:
                info(f"{method.upper()} Pruning Results:")
                info(f"{'Model':<20} {'Base MSE':<10} {'Base RMSE':<11} {'Final MSE':<10} {'Final RMSE':<11} {'Δ MSE':<10} {'Removed':<10}")
                info("-" * 92)
                for model_name in successful_models:
                    r = results[model_name][method]
                    info(f"{model_name:<20} {r['baseline_mse']:<10.4f} {r['baseline_rmse']:<11.4f} "
                          f"{r['final_mse']:<10.4f} {r['final_rmse']:<11.4f} {r['improvement_mse']:<10.4f} "
                          f"{r['num_removed']:<10} ({r['reduction_percentage']:.1f}%)")
        else:
            # Print results for single method
            method = methods_to_run[0]
            info(f"{method.upper()} Pruning Results:")
            info(f"{'Model':<20} {'Base MSE':<10} {'Base RMSE':<11} {'Final MSE':<10} {'Final RMSE':<11} {'Δ MSE':<10} {'Removed':<10}")
            info("-" * 92)
            for model_name in successful_models:
                r = results[model_name][method]
                info(f"{model_name:<20} {r['baseline_mse']:<10.4f} {r['baseline_rmse']:<11.4f} "
                      f"{r['final_mse']:<10.4f} {r['final_rmse']:<11.4f} {r['improvement_mse']:<10.4f} "
                      f"{r['num_removed']:<10.0f} ({r['reduction_percentage']:.1f}%)")
    
    if failed_models:
        warn(f"Failed models: {', '.join(failed_models)}")
    
    info("=" * 80)
    
    # Final memory cleanup
    info("Performing final memory cleanup...")
    clear_gpu_memory()
    
    return results

# ============================================================================
# Test All Trained and Pruned Models
# ============================================================================

def experiment_one_test_models(models_to_test=None, verbose=True,
                             global_test_indices=None):
    """
    Test trained models from Experiment 1A and pruned on 1B on the test set.

    Args:
        models_to_test (list, optional): List of model names to test.
            If None, tests all trained models.
        verbose (bool): Whether to print detailed progress.
        global_test_indices (list, optional): Pre-defined global test indices.
            If None, falls back to config dataset.

    Returns:
        dict: Dictionary containing test results for each model.
    """
    print("=" * 80)
    info("EXPERIMENT 1: TESTING TRAINED AND PRUNED MODELS")
    print("=" * 80)
    
    # Create output directories
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Resolve weights dir relative to this script so behavior is robust to CWD
    OUTPUT_DIR = Path(__file__).resolve().parent / 'Outputs' / 'Experiment_1_variants'
    WEIGHTS_DIR = OUTPUT_DIR / 'Weights'
    
    if not WEIGHTS_DIR.exists():
        error(f"Weights directory not found: {WEIGHTS_DIR}")
        return {}
        
    # Collect all .pth files 
    all_pth_files = sorted(WEIGHTS_DIR.glob('*.pth'))
    
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
            
    # Results storage
    results = {}
    
    # Test each model (and its variant files)
    model_items = list(weights_files_by_model.items())
    
    for idx, (model_name, weight_file_list) in enumerate(model_items, 1):
        info(f"[{idx}/{len(model_items)}] Testing {model_name.upper()}")
        print("-" * 80)
        
        try:
            config = MODEL_REGISTRY[model_name]
            
            # Prepare test loader (same for all variants unless dataset differs)
            # Use global test indices if provided, else fallback to config dataset
            if global_test_indices is not None:
                # Determine the base full dataset for this model
                backbone_dataset = None
                if config['dataset'] is IMAGENET_DATASET:
                    backbone_dataset = imageNet_dataset
                elif config['dataset'] is DENSENET_DATASET:
                    backbone_dataset = denseNet_dataset
                else:
                    backbone_dataset = None
                    
                if backbone_dataset is not None:
                    test_ds = Subset(backbone_dataset, global_test_indices)
                    test_loader = DataLoader(
                        test_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS
                    )
                else:
                    # Fallback
                    dataset = config['dataset']
                    test_loader = DataLoader(
                        dataset['test'],
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS
                    )
            else:
                # Fallback to original
                dataset = config['dataset']
                test_loader = DataLoader(
                    dataset['test'],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS
                )
                
            # Setup criterion and device
            criterion = nn.MSELoss()
            device = torch.device(TRAINING_CONFIG['device'])
            
            # Prepare results container for this model
            results[model_name] = {}
            
            # Iterate over all weight files (variants) for this model
            for weights_path in weight_file_list:
                if verbose:
                    info(f"Loading model from {weights_path}...")
                    
                model = config['class'](freeze_backbone=False)
                model.load_state_dict(torch.load(weights_path, weights_only=True))
                
                # Evaluate on test set
                info("Evaluating on test set...")
                test_results = test_model(
                    model,
                    test_loader,
                    criterion,
                    device,
                    return_additional_metrics=True
                )
                
                m = re.search(r"exp1a_variant\d+|exp1b_variant\d+_greedy_pruned|exp1b_variant\d+_negative_pruned|orig", str(weights_path))
                variant_tag = m.group(0) if m else 'orig'
                
                results[model_name][variant_tag] = {
                    'test_mse': test_results['mse'],
                    'test_rmse': test_results['rmse'],
                    'plcc': test_results['plcc'],
                    'srcc': test_results['srcc'],
                    'krcc': test_results['krcc'],
                    'weights_path': str(weights_path)
                }

                pruning_stats = None
                if "exp1b_" in variant_tag and ("greedy_pruned" in variant_tag or "negative_pruned" in variant_tag):
                    variant_match = re.search(r'variant(\d+)', variant_tag)
                    if variant_match:
                        variant_index = variant_match.group(1)
                        baseline_weights_path = WEIGHTS_DIR / f"{model_name}_exp1a_variant{variant_index}_best.pth"
                        if baseline_weights_path.exists():
                            try:
                                original_model = config['class'](freeze_backbone=False)
                                original_model.load_state_dict(torch.load(baseline_weights_path, map_location='cpu', weights_only=True))

                                pruned_model_for_stats = config['class'](freeze_backbone=False)
                                pruned_model_for_stats.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))

                                pruning_stats = check_pruning_statistics(
                                    original_model=original_model,
                                    pruned_model=pruned_model_for_stats,
                                    layer_name=config['target_layer'],
                                    save_json=False
                                )

                                reduction_pct = pruning_stats.get('reduction_percentage')
                                if reduction_pct is not None:
                                    info(f" - Pruning stats computed for {variant_tag}: reduction {reduction_pct:.2f}%")
                            except Exception as stat_err:
                                warn(f"Failed to compute pruning statistics for {weights_path.name}: {stat_err}")
                                pruning_stats = {'error': str(stat_err)}
                            finally:
                                try:
                                    del original_model
                                    del pruned_model_for_stats
                                except NameError:
                                    pass
                        else:
                            warn(f"Original weights not found for {weights_path.name}: expected {baseline_weights_path.name}")
                            pruning_stats = {'error': f'Original weights missing: {str(baseline_weights_path)}'}
                    else:
                        warn(f"Could not infer baseline variant for pruned weights {weights_path.name}")
                        pruning_stats = {'error': 'Failed to parse variant index from pruned weight filename.'}

                if pruning_stats is not None:
                    results[model_name][variant_tag]['pruning_statistics'] = pruning_stats
                
                info(f"✓ {model_name.upper()} {variant_tag} testing complete!")
                plcc_str = f"{test_results['plcc']:.4f}" if test_results['plcc'] is not None else "N/A"
                srcc_str = f"{test_results['srcc']:.4f}" if test_results['srcc'] is not None else "N/A"
                krcc_str = f"{test_results['krcc']:.4f}" if test_results['krcc'] is not None else "N/A"
                info(f" - Test MSE: {test_results['mse']:.4f}, RMSE: {test_results['rmse']:.4f}, PLCC: {plcc_str}, SRCC: {srcc_str}, KRCC: {krcc_str}")
                
        except Exception as e:
            error(f"Error testing {model_name}: {e}")
            error(traceback.format_exc())
            results[model_name] = {'error': str(e)}
            
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
            
    # Print summary
    info("=" * 80)
    info("EXPERIMENT 1: TESTING SUMMARY")
    info("=" * 80)
    
    successful_models = [k for k, v in results.items() if 'error' not in v]
    failed_models = [k for k, v in results.items() if 'error' in v]
    
    # Use the number of discovered models as the denominator to avoid errors
    info(f"Successfully tested: {len(successful_models)}/{len(weights_files_by_model)} models")
    
    if successful_models:
        info("Test Results:")
        info(f"{'Model':<20} {'Variant':<18} {'Pruning':<10} {'Test MSE':<10} {'Test RMSE':<11} {'PLCC':<8} {'SRCC':<8} {'KRCC':<8}")
        print("-" * 95)
        
        for model_name in successful_models:
            # Sort variants for consistent output order
            sorted_variants = sorted(results[model_name].items())
            
            for variant_tag, variant_data in sorted_variants:
                if isinstance(variant_data, dict) and 'test_mse' in variant_data:
                    
                    # Parse variant_tag to separate variant from pruning type
                    pruning_type = "Baseline"
                    display_variant = variant_tag
                    
                    if "greedy_pruned" in variant_tag:
                        pruning_type = "Greedy"
                        display_variant = variant_tag.replace("_greedy_pruned", "")
                    elif "negative_pruned" in variant_tag:
                        pruning_type = "Negative"
                        display_variant = variant_tag.replace("_negative_pruned", "")

                    plcc_str = f"{variant_data['plcc']:<8.4f}" if variant_data['plcc'] is not None else f"{'N/A':<8}"
                    srcc_str = f"{variant_data['srcc']:<8.4f}" if variant_data['srcc'] is not None else f"{'N/A':<8}"
                    krcc_str = f"{variant_data['krcc']:<8.4f}" if variant_data['krcc'] is not None else f"{'N/A':<8}"
                    
                    info(f"{model_name:<20} {display_variant:<18} {pruning_type:<10} {variant_data['test_mse']:<10.4f} {variant_data['test_rmse']:<11.4f} {plcc_str} {srcc_str} {krcc_str}")
                    
    if failed_models:
        warn(f"Failed models: {', '.join(failed_models)}")
        
    print("=" * 80)
    
    # Final memory cleanup
    info("Performing final memory cleanup...")
    clear_gpu_memory()
    
    return results

# ============================================================================
# Complete Experiment Pipeline
# ============================================================================

def run_experiment_one_complete(
    models_to_process=None,
    run_training=True,
    run_pruning=True,
    run_testing=False,
    pruning_method='both',
    threshold=0.0
):
    """
    Run the complete Experiment 1 pipeline (1A + 1B).
    
    Args:
        models_to_process (list, optional): List of model names to process.
                                          If None, processes all models.
        run_training (bool): Whether to run Experiment 1A (training).
        run_pruning (bool): Whether to run Experiment 1B (pruning).
        pruning_method (str): Pruning method ('greedy', 'negative_impact', or 'both').
                             Default is 'both' to run both methods.
        run_testing (bool): Whether to run testing of trained and pruned models.
        threshold (float): Threshold for negative_impact pruning.
    
    Returns:
        dict: Dictionary containing results from both experiments.
              Structure: {'training': {...}, 'pruning': {...}, 'testing': {...}}
    """
    # Create global test indices to ensure consistency between training and pruning
    try:
        total_size = len(imageNet_dataset)
        test_size = int(0.2 * total_size)
        gen_global = torch.Generator().manual_seed(42)
        perm = torch.randperm(total_size, generator=gen_global).tolist()
        global_test_indices = perm[:test_size]
    except Exception:
        global_test_indices = None
    
    results = {
        'training': None,
        'pruning': None,
        'testing': None
    }
    
    # Run Experiment 1A: Training
    if run_training:
        info("=" * 80)
        info("STARTING EXPERIMENT 1A: TRAINING")
        info("=" * 80)
        
        training_results = experiment_1a_train_all_models(
            models_to_train=models_to_process,
            save_plots=True,
            verbose=True,
            global_test_indices=global_test_indices
        )
        results['training'] = training_results
    
    # Run Experiment 1B: Pruning
    if run_pruning:
        print("=" * 80)
        info("STARTING EXPERIMENT 1B: PRUNING")
        print("=" * 80)
        
        pruning_results = experiment_1b_prune_all_models(
            models_to_prune=models_to_process,
            pruning_method=pruning_method,
            threshold=threshold,
            verbose=True,
            global_test_indices=global_test_indices
        )
        results['pruning'] = pruning_results
    
    # Run Testing of trained and pruned models
    if run_testing:
        print("=" * 80)
        info("STARTING EXPERIMENT 1: TESTING TRAINED AND PRUNED MODELS")
        print("=" * 80)
        
        testing_results = experiment_one_test_models(
            models_to_test=models_to_process,
            verbose=True,
            global_test_indices=global_test_indices
        )
        results['testing'] = testing_results
        # save testing results to a JSON file
        test_results_path = TEST_RESULTS_DIR / "experiment_1_test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(testing_results, f, indent=4)
        info(f"✓ Testing results saved to: {test_results_path}")

    return results


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    
    """
    Instruction to run Experiment 1:
    ===========================================================================
    - cd to the Image_Authenticity_prediction/main/Experiments/ directory
    - activate your Python environment : conda activate <your_env>
    - python experiment_one.py
    ===========================================================================
    
    # Run complete pipeline for all models
    results = run_experiment_one_complete()
    
    # Run only training
    results = run_experiment_one_complete(run_pruning=False)
    
    # Run only pruning (requires trained models)
    results = run_experiment_one_complete(run_training=False)
    
    # Run for specific models only
    results = run_experiment_one_complete(
        models_to_process=['vgg16', 'resnet152'],
        pruning_method='both'
    )
    
    # Run only greedy pruning
    results = run_experiment_one_complete(
        models_to_process=None,
        run_training=False,
        pruning_method='greedy'
    )
    
    # Run only negative impact pruning
    results = run_experiment_one_complete(
        models_to_process=None,
        run_training=False,
        pruning_method='negative_impact',
        threshold=0.0
    )
    """
    
    # Start timer
    start_time = time.time()
    
    # Change the following line to configure which parts to run your experiment
    results = run_experiment_one_complete(
        run_training=False,
        run_pruning=False,
        run_testing=True,
    )
    
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format elapsed time as H:M:S
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    info(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")