import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import numpy as np
import gc

# Add main package to path - go up from  Experiments/ -> main/ -> Image_Authenticity_prediction/
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import models
from main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    InceptionV3AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor
)

# Import utilities
from main.Utils import FeatureMapsPruner
from main.train import train_model, test_model, plot_loss_history
from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    INCEPTIONV3_DATASET,
    BATCH_SIZE,
    NUM_WORKERS
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
        'target_layer': 'features.7',  # Last residual block
        'input_size': 224
    },
    'densenet161': {
        'class': DenseNet161AuthenticityPredictor,
        'dataset': DENSENET_DATASET,
        'target_layer': 'features.denseblock4',  # Last dense block
        'input_size': 300
    },
    'inceptionv3': {
        'class': InceptionV3AuthenticityPredictor,
        'dataset': INCEPTIONV3_DATASET,
        'target_layer': 'features.16',  # Last mixed layer
        'input_size': 299
    },
    'efficientnetb3': {
        'class': EfficientNetB3AuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.8',  # Last MBConv block
        'input_size': 224
    },
    'barlowtwins': {
        'class': BarlowTwinsAuthenticityPredictor,
        'dataset': IMAGENET_DATASET,
        'target_layer': 'features.7',  # Last layer before avgpool
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
OUTPUT_DIR = Path('Output')
WEIGHTS_DIR = OUTPUT_DIR / 'Weights'
RANKINGS_DIR = OUTPUT_DIR / 'Ranking_arrays'
RANKING_PLOTS_DIR = OUTPUT_DIR / 'Ranking_Plots'
TRAINING_PLOTS_DIR = OUTPUT_DIR / 'Training_Plots'
TRAINING_HISTORY_DIR = OUTPUT_DIR / 'Training_History'



# ============================================================================
# Memory Management Utilities
# ============================================================================

def clear_gpu_memory():
    """
    Clear GPU memory by collecting garbage and emptying CUDA cache.
    Call this after each model training/pruning to prevent memory accumulation.
    """
    # Collect Python garbage
    gc.collect()
    
    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            # Print memory stats for monitoring
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            print(f"  [GPU Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def cleanup_model_and_data(model, dataloaders=None, optimizer=None):
    """
    Properly cleanup model, dataloaders, and optimizer to free memory.
    
    Args:
        model: PyTorch model to cleanup
        dataloaders: Dict or list of dataloaders to cleanup
        optimizer: Optimizer to cleanup
    """
    # Move model to CPU and delete
    if model is not None:
        model.cpu()
        del model
    
    # Cleanup optimizer
    if optimizer is not None:
        del optimizer
    
    # Cleanup dataloaders
    if dataloaders is not None:
        if isinstance(dataloaders, dict):
            for loader in dataloaders.values():
                del loader
        elif isinstance(dataloaders, (list, tuple)):
            for loader in dataloaders:
                del loader
        else:
            del dataloaders
    
    # Force garbage collection and clear CUDA cache
    clear_gpu_memory()


# ============================================================================
# Experiment 1A: Train All Models
# ============================================================================

def experiment_1a_train_all_models(
    models_to_train=None,
    save_plots=True,
    verbose=True
):
    """
    Experiment 1A: Train all model architectures with early stopping.
    
    Args:
        models_to_train (list, optional): List of model names to train.
                                         If None, trains all models.
        save_plots (bool): Whether to save training history plots.
        verbose (bool): Whether to print detailed progress.
    
    Returns:
        dict: Dictionary containing training results for each model.
    """
    print("=" * 80)
    print("EXPERIMENT 1A: TRAINING ALL MODELS")
    print("=" * 80)
    
    # Create output directories
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select models to train
    if models_to_train is None:
        models_to_train = list(MODEL_REGISTRY.keys())
    
    # Results storage
    results = {}
    
    # Train each model
    for idx, model_name in enumerate(models_to_train, 1):
        print(f"\n[{idx}/{len(models_to_train)}] Training {model_name.upper()}")
        print("-" * 80)
        
        try:
            # Get model configuration
            config = MODEL_REGISTRY[model_name]
            
            # Initialize model
            if verbose:
                print(f"Initializing {model_name} model...")
            model = config['class'](freeze_backbone=TRAINING_CONFIG['freeze_backbone'])
            
            # Get dataset and create dataloaders
            dataset = config['dataset']
            train_loader = DataLoader(
                dataset['train'],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS
            )
            test_loader = DataLoader(
                dataset['test'],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS
            )
            
            dataloaders = {
                'train': train_loader,
                'val': test_loader
            }
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=TRAINING_CONFIG['learning_rate']
            )
            
            # Train the model
            if verbose:
                print(f"Training on device: {TRAINING_CONFIG['device']}")
                print(f"Max epochs: {TRAINING_CONFIG['max_epochs']}, Patience: {TRAINING_CONFIG['patience']}")
            
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
            print("\nEvaluating on test set...")
            test_loss = test_model(
                best_model,
                test_loader,
                criterion,
                TRAINING_CONFIG['device']
            )
            test_rmse = np.sqrt(test_loss)  # Compute RMSE from MSE
            
            # Save model weights
            weights_path = WEIGHTS_DIR / f"{model_name}_exp1a_best.pth"
            torch.save(best_model.state_dict(), weights_path)
            print(f"✓ Model saved to: {weights_path}")
            
            # Save training history
            history_path = TRAINING_HISTORY_DIR / f"{model_name}_exp1a_history.npy"
            np.save(history_path, history)
            
            # Plot and save training history
            if save_plots:
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Use non-interactive backend
                    plot_loss_history(history)
                    import matplotlib.pyplot as plt
                    plot_path = TRAINING_PLOTS_DIR / f"{model_name}_exp1a_training_curve.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Training curve saved to: {plot_path}")
                except Exception as e:
                    print(f"Warning: Could not save plot: {e}")
            
            # Store results
            results[model_name] = {
                'final_test_mse': test_loss,
                'final_test_rmse': test_rmse,
                'final_val_loss': history['val_loss'][-1],
                'best_val_loss': min(history['val_loss']),
                'epochs_trained': len(history['train_loss']),
                'weights_path': str(weights_path),
                'history': history
            }
            
            print(f"\n✓ {model_name.upper()} training complete!")
            print(f"  - Test MSE: {test_loss:.4f}, RMSE: {test_rmse:.4f}")
            print(f"  - Best Val Loss: {results[model_name]['best_val_loss']:.4f}")
            print(f"  - Epochs trained: {results[model_name]['epochs_trained']}")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
        
        finally:
            # Clean up memory after each model (success or failure)
            print(f"\nCleaning up {model_name} from memory...")
            cleanup_model_and_data(
                model=locals().get('best_model') or locals().get('model'),
                dataloaders=locals().get('dataloaders'),
                optimizer=locals().get('optimizer')
            )
            print(f"✓ {model_name} memory cleaned")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 1A: TRAINING SUMMARY")
    print("=" * 80)
    
    successful_models = [k for k, v in results.items() if 'error' not in v]
    failed_models = [k for k, v in results.items() if 'error' in v]
    
    print(f"\nSuccessfully trained: {len(successful_models)}/{len(models_to_train)} models")
    
    if successful_models:
        print("\nModel Performance (Test Loss):")
        print(f"{'Model':<20} {'MSE':<10} {'RMSE':<10}")
        print("-" * 42)
        sorted_results = sorted(
            [(k, v['final_test_mse'], v['final_test_rmse']) for k, v in results.items() if 'error' not in v],
            key=lambda x: x[1]  # Sort by MSE
        )
        for rank, (model_name, test_mse, test_rmse) in enumerate(sorted_results, 1):
            print(f"{rank}. {model_name:<17} {test_mse:<10.4f} {test_rmse:<10.4f}")
    
    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")
    
    print("\n" + "=" * 80)
    
    # Final memory cleanup
    print("\nPerforming final memory cleanup...")
    clear_gpu_memory()
    
    return results


# ============================================================================
# Experiment 1B: Prune All Trained Models
# ============================================================================

def experiment_1b_prune_all_models(
    models_to_prune=None,
    pruning_method='both',
    threshold=0.0,
    verbose=True
):
    """
    Experiment 1B: Prune trained models using feature importance analysis.
    
    Args:
        models_to_prune (list, optional): List of model names to prune.
                                         If None, prunes all trained models.
        pruning_method (str): Pruning method to use ('greedy', 'negative_impact', or 'both').
        threshold (float): Threshold for negative_impact pruning (ignored for greedy).
        verbose (bool): Whether to print detailed progress.
    
    Returns:
        dict: Dictionary containing pruning results for each model.
              If pruning_method='both', each model will have 'greedy' and 'negative_impact' sub-dicts.
    """
    print("=" * 80)
    print("EXPERIMENT 1B: PRUNING ALL TRAINED MODELS")
    print("=" * 80)
    
    # Create output directories
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
    RANKING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select models to prune
    if models_to_prune is None:
        # Find all trained models
        models_to_prune = []
        for model_name in MODEL_REGISTRY.keys():
            weights_path = WEIGHTS_DIR / f"{model_name}_exp1a_best.pth"
            if weights_path.exists():
                models_to_prune.append(model_name)
        
        if not models_to_prune:
            print("Error: No trained models found. Please run Experiment 1A first.")
            return {}
    
    print(f"Found {len(models_to_prune)} trained models to prune.")
    print(f"Pruning method: {pruning_method}")
    if pruning_method in ['negative_impact', 'both']:
        print(f"Threshold for negative_impact: {threshold}")
    
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
    
    # Prune each model
    for idx, model_name in enumerate(models_to_prune, 1):
        print(f"\n[{idx}/{len(models_to_prune)}] Pruning {model_name.upper()}")
        print("-" * 80)
        
        try:
            # Get model configuration
            config = MODEL_REGISTRY[model_name]
            
            # Load trained model
            weights_path = WEIGHTS_DIR / f"{model_name}_exp1a_best.pth"
            if not weights_path.exists():
                print(f"Warning: Weights not found at {weights_path}. Skipping.")
                continue
            
            if verbose:
                print(f"Loading trained model from {weights_path}...")
            model = config['class'](freeze_backbone=False)  # Unfreeze for pruning
            model.load_state_dict(torch.load(weights_path))
            
            # Get dataset and create dataloader
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
            
            # Initialize pruner
            target_layer = config['target_layer']
            if verbose:
                print(f"Target layer for pruning: {target_layer}")
            
            pruner = FeatureMapsPruner(
                model=model,
                dataloader=test_loader,
                layer_name=target_layer,
                criterion=criterion,
                eval_function=test_model,
                device=device
            )
            
            # Compute importance scores
            print("\nComputing feature importance scores...")
            importance_path = RANKINGS_DIR / f"{model_name}_exp1b_importance_scores.npy"
            importance_scores = pruner.compute_importance_scores(
                save_path=str(importance_path),
                force_recompute=PRUNING_CONFIG['force_recompute']
            )
            
            # Plot and save importance scores
            print("\nGenerating importance score plot...")
            plot_path = RANKING_PLOTS_DIR / f"{model_name}_exp1b_importance_scores.png"
            try:
                pruner.plot_importance_scores(save_path=str(plot_path))
                print(f"✓ Importance score plot saved to: {plot_path}")
            except Exception as e:
                print(f"Warning: Could not save importance plot: {e}")
            
            print(f"✓ Importance scores computed and saved to: {importance_path}")
            print(f"  - Baseline MSE: {pruner.baseline_mse:.4f}, RMSE: {pruner.baseline_rmse:.4f}")
            print(f"  - Number of feature maps: {len(importance_scores)}")
            
            # Store results structure
            if len(methods_to_run) > 1:
                # Multiple methods: store results in nested dict
                results[model_name] = {}
            
            # Perform pruning for each method
            for method in methods_to_run:
                if method == 'greedy':
                    print("\nPerforming greedy pruning...")
                    pruned_weights_path = WEIGHTS_DIR / f"{model_name}_exp1b_greedy_pruned.pth"
                    pruning_results = pruner.greedy_pruning(
                        model_save_path=str(pruned_weights_path)
                    )
                    
                elif method == 'negative_impact':
                    print(f"\nPerforming negative impact pruning (threshold={threshold})...")
                    pruned_weights_path = WEIGHTS_DIR / f"{model_name}_exp1b_negative_pruned.pth"
                    pruning_results = pruner.negative_impact_pruning(
                        model_save_path=str(pruned_weights_path),
                        threshold=threshold
                    )
                
                # Prepare result dict for this method
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
                    'importance_scores_path': str(importance_path)
                }
                
                if method == 'greedy' and 'mse_history' in pruning_results:
                    method_results['mse_history'] = pruning_results['mse_history']
                
                # Store results
                if len(methods_to_run) > 1:
                    results[model_name][method] = method_results
                else:
                    results[model_name] = method_results
                
                # Print method-specific results
                print(f"\n✓ {model_name.upper()} {method} pruning complete!")
                print(f"  - Baseline MSE: {pruning_results['baseline_mse']:.4f}, RMSE: {pruning_results['baseline_rmse']:.4f}")
                print(f"  - Final MSE: {pruning_results['final_mse']:.4f}, RMSE: {pruning_results['final_rmse']:.4f}")
                print(f"  - Improvement MSE: {pruning_results['improvement_mse']:.4f}, RMSE: {pruning_results['improvement_rmse']:.4f}")
                print(f"  - Features removed: {len(pruning_results['removed_features'])}")
                print(f"  - Reduction: {pruning_results['reduction_percentage']:.1f}%")
            
            
        except Exception as e:
            print(f"\n✗ Error pruning {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
        
        finally:
            # Clean up memory after each model (success or failure)
            print(f"\nCleaning up {model_name} from memory...")
            cleanup_model_and_data(
                model=locals().get('model'),
                dataloaders=locals().get('test_loader'),
                optimizer=None
            )
            # Also cleanup the pruner object which holds references
            if 'pruner' in locals():
                del pruner
            clear_gpu_memory()
            print(f"✓ {model_name} memory cleaned")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 1B: PRUNING SUMMARY")
    print("=" * 80)
    
    successful_models = [k for k, v in results.items() if 'error' not in v]
    failed_models = [k for k, v in results.items() if 'error' in v]
    
    print(f"\nSuccessfully pruned: {len(successful_models)}/{len(models_to_prune)} models")
    
    if successful_models:
        if len(methods_to_run) > 1:
            # Print results for both methods
            for method in methods_to_run:
                print(f"\n{method.upper()} Pruning Results:")
                print(f"{'Model':<20} {'Base MSE':<10} {'Base RMSE':<11} {'Final MSE':<10} {'Final RMSE':<11} {'Δ MSE':<10} {'Removed':<10}")
                print("-" * 92)
                for model_name in successful_models:
                    r = results[model_name][method]
                    print(f"{model_name:<20} {r['baseline_mse']:<10.4f} {r['baseline_rmse']:<11.4f} "
                          f"{r['final_mse']:<10.4f} {r['final_rmse']:<11.4f} {r['improvement_mse']:<10.4f} "
                          f"{r['num_removed']:<10} ({r['reduction_percentage']:.1f}%)")
        else:
            # Print results for single method
            print(f"\n{methods_to_run[0].upper()} Pruning Results:")
            print(f"{'Model':<20} {'Base MSE':<10} {'Base RMSE':<11} {'Final MSE':<10} {'Final RMSE':<11} {'Δ MSE':<10} {'Removed':<10}")
            print("-" * 92)
            for model_name in successful_models:
                r = results[model_name]
                print(f"{model_name:<20} {r['baseline_mse']:<10.4f} {r['baseline_rmse']:<11.4f} "
                      f"{r['final_mse']:<10.4f} {r['final_rmse']:<11.4f} {r['improvement_mse']:<10.4f} "
                      f"{r['num_removed']:<10} ({r['reduction_percentage']:.1f}%)")
    
    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")
    
    print("\n" + "=" * 80)
    
    # Final memory cleanup
    print("\nPerforming final memory cleanup...")
    clear_gpu_memory()
    
    return results


# ============================================================================
# Complete Experiment Pipeline
# ============================================================================

def run_experiment_one_complete(
    models_to_process=None,
    run_training=True,
    run_pruning=True,
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
        threshold (float): Threshold for negative_impact pruning.
    
    Returns:
        dict: Dictionary containing results from both experiments.
              Structure: {'training': {...}, 'pruning': {...}}
    """
    results = {
        'training': None,
        'pruning': None
    }
    
    # Run Experiment 1A: Training
    if run_training:
        print("\n" + "=" * 80)
        print("STARTING EXPERIMENT 1A: TRAINING")
        print("=" * 80 + "\n")
        
        training_results = experiment_1a_train_all_models(
            models_to_train=models_to_process,
            save_plots=True,
            verbose=True
        )
        results['training'] = training_results
    
    # Run Experiment 1B: Pruning
    if run_pruning:
        print("\n" + "=" * 80)
        print("STARTING EXPERIMENT 1B: PRUNING")
        print("=" * 80 + "\n")
        
        pruning_results = experiment_1b_prune_all_models(
            models_to_prune=models_to_process,
            pruning_method=pruning_method,
            threshold=threshold,
            verbose=True
        )
        results['pruning'] = pruning_results
    
    return results


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    """
    Isntruction to run Experiment 1:
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
    
    # Default: Run complete pipeline for all models with both pruning methods
    print("Starting Experiment 1: Complete Training and Pruning Pipeline")
    print("This will train and prune all 7 models using both pruning methods.")
    print("To run only specific parts or models, edit this section or import the functions.\n")
    
    results = run_experiment_one_complete(
        models_to_process=None,  # None = all models
        run_training=True,
        run_pruning=True,
        pruning_method='both'  # Run both greedy and negative_impact
    )
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Model weights: {WEIGHTS_DIR}")
    print(f"  - Importance scores: {RANKINGS_DIR}")
