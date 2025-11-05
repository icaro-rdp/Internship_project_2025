"""
Minimal Testing Script for Experiment 1

This script runs a quick test with reduced epochs and a single model
to verify that everything is working correctly before running the full experiment.

Usage:
    - activate your virtual environment: 
        conda activate <your_env> 
    - Go to Image_Authenticity_prediction directory:
        cd Image_Authenticity_prediction
    - Run the test script:
        python main/Experiments/test/test_experiment_one_minimal.py
"""

import sys
from pathlib import Path

# Add main package to path - go up from test/ -> Experiments/ -> main/ -> Image_Authenticity_prediction/
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import using the full package path
from main.Experiments import experiment_one

def test_train_only():
    """Test training only with minimal settings."""
    print("\n" + "=" * 80)
    print("TEST 3: Training Only (VGG16)")
    print("=" * 80)
    
    # Temporarily modify config
    original_epochs = experiment_one.TRAINING_CONFIG['max_epochs']
    original_patience = experiment_one.TRAINING_CONFIG['patience']
    
    try:
        experiment_one.TRAINING_CONFIG['max_epochs'] = 100
        experiment_one.TRAINING_CONFIG['patience'] = 10
        
        # Run training only
        results = experiment_one.run_experiment_one_complete(
            models_to_process=['vgg16'],
            run_training=True,
            run_pruning=False
        )
        
        # Check results
        training_ok = ('training' in results and 
                      results['training'] and 
                      'vgg16' in results['training'] and 
                      'error' not in results['training']['vgg16'])
        
        if training_ok:
            print("\n✓ Training only test PASSED")
            print("\n  Training:")
            print(f"    - Test MSE: {results['training']['vgg16']['final_test_mse']:.4f}")
            print(f"    - Test RMSE: {results['training']['vgg16']['final_test_rmse']:.4f}")
            return True, results
        else:
            print("\n✗ Training only test FAILED")
            return False, results
            
    finally:
        experiment_one.TRAINING_CONFIG['max_epochs'] = original_epochs
        experiment_one.TRAINING_CONFIG['patience'] = original_patience


def test_complete_pipeline():
    """Test the complete pipeline with minimal settings."""
    print("\n" + "=" * 80)
    print("TEST 4: Complete Pipeline (Train + Prune, VGG16)")
    print("=" * 80)
    
    # Temporarily modify config
    original_epochs = experiment_one.TRAINING_CONFIG['max_epochs']
    original_patience = experiment_one.TRAINING_CONFIG['patience']
    
    try:
        experiment_one.TRAINING_CONFIG['max_epochs'] = 500
        experiment_one.TRAINING_CONFIG['patience'] = 15
        
        # Run complete pipeline
        results = experiment_one.run_experiment_one_complete(
            models_to_process=['vgg16'],
            run_training=True,
            run_pruning=True,
            pruning_method='greedy'
        )
        
        # Check results
        training_ok = ('training' in results and 
                      results['training'] and 
                      'vgg16' in results['training'] and 
                      'error' not in results['training']['vgg16'])
        
        pruning_ok = ('pruning' in results and 
                     results['pruning'] and 
                     'vgg16' in results['pruning'] and 
                     'error' not in results['pruning']['vgg16'])
        
        if training_ok and pruning_ok:
            print("\n✓ Complete pipeline test PASSED")
            print("\n  Training:")
            print(f"    - Test MSE: {results['training']['vgg16']['final_test_mse']:.4f}")
            print(f"    - Test RMSE: {results['training']['vgg16']['final_test_rmse']:.4f}")
            print("\n  Pruning:")
            # Check if results have both methods or just one
            r = results['pruning']['vgg16']
            if 'greedy' in r:
                # Both methods were run, show both
                print(f"    Greedy:")
                print(f"      - Improvement MSE: {r['greedy']['improvement_mse']:.4f}")
                print(f"      - Features removed: {r['greedy']['num_removed']}")
                print(f"    Negative Impact:")
                print(f"      - Improvement MSE: {r['negative_impact']['improvement_mse']:.4f}")
                print(f"      - Features removed: {r['negative_impact']['num_removed']}")
            else:
                # Only one method was run
                print(f"    - Improvement MSE: {r['improvement_mse']:.4f}")
                print(f"    - Features removed: {r['num_removed']}")
            return True, results
        else:
            print("\n✗ Complete pipeline test FAILED")
            if not training_ok:
                print("  - Training failed")
            if not pruning_ok:
                print("  - Pruning failed")
            return False, results
            
    finally:
        experiment_one.TRAINING_CONFIG['max_epochs'] = original_epochs
        experiment_one.TRAINING_CONFIG['patience'] = original_patience


if __name__ == '__main__':
    success, results = test_complete_pipeline()
    