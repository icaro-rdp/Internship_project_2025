"""
Minimal Testing Script for Experiment 1

This script runs a quick test with reduced epochs and a single model
to verify that everything is working correctly before running the full experiment.

Usage:
   Go to Image_Authenticity_prediction directory:
        python main/Experiments/test/test_experiment_one_minimal.py
"""

import sys
from pathlib import Path

# Add main package to path - go up from test/ -> Experiments/ -> main/ -> Image_Authenticity_prediction/
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import using the full package path
from main.Experiments import experiment_one


def test_complete_pipeline():
    """Test the complete pipeline with minimal settings."""
    print("\n" + "=" * 80)
    print("TEST 4: Complete Pipeline (Train + Prune, VGG16)")
    print("=" * 80)
    
    # Temporarily modify config
    original_epochs = experiment_one.TRAINING_CONFIG['max_epochs']
    original_patience = experiment_one.TRAINING_CONFIG['patience']
    
    try:
        experiment_one.TRAINING_CONFIG['max_epochs'] = 50
        experiment_one.TRAINING_CONFIG['patience'] = 7
        
        # Run complete pipeline
        results = experiment_one.run_experiment_one_complete(
            models_to_process=['vgg16'],
            run_training=True,
            run_pruning=True,
            pruning_method='both'
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
            print(f"    - Test Loss: {results['training']['vgg16']['final_test_loss']:.4f}")
            print("\n  Pruning:")
            r = results['pruning']['vgg16']
            print(f"    - Improvement: {r['improvement']:.4f}")
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
    