#!/usr/bin/env python3
"""
Test script to verify all imports and module structure are correct.

Usage:
    python test_imports.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_model_imports():
    """Test that all models can be imported."""
    print("Testing model imports...")
    try:
        from Image_Authenticity_prediction.main.Models import (
            VGG16AuthenticityPredictor,
            VGG19AuthenticityPredictor,
            ResNet152AuthenticityPredictor,
            DenseNet161AuthenticityPredictor,
            InceptionV3AuthenticityPredictor,
            EfficientNetB3AuthenticityPredictor,
            BarlowTwinsAuthenticityPredictor
        )
        print("✓ All models imported successfully")
        return True
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False

def test_utils_imports():
    """Test that all utilities can be imported."""
    print("\nTesting utils imports...")
    try:
        from Image_Authenticity_prediction.main.Utils import (
            GradCAM,
            MultiscalePixelMasking,
            FeatureMapsPruner
        )
        print("✓ All utils imported successfully")
        return True
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
        return False

def test_data_imports():
    """Test that data module can be imported."""
    print("\nTesting data imports...")
    try:
        from Image_Authenticity_prediction.main.data import (
            ImageAuthenticityDataset,
            IMAGENET_DATASET,
            DENSENET_DATASET,
            INCEPTIONV3_DATASET,
            IMAGENET_TRANSFORM,
            DENSENET_TRANSFORM,
            BATCH_SIZE,
            NUM_WORKERS
        )
        print("✓ Data module imported successfully")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Num workers: {NUM_WORKERS}")
        return True
    except Exception as e:
        print(f"✗ Data import failed: {e}")
        return False

def test_train_imports():
    """Test that training module can be imported."""
    print("\nTesting train imports...")
    try:
        from Image_Authenticity_prediction.main.train import (
            train_model,
            test_model,
            plot_loss_history
        )
        print("✓ Train module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Train import failed: {e}")
        return False

def test_experiments_imports():
    """Test that experiments module can be imported."""
    print("\nTesting experiments imports...")
    try:
        from Image_Authenticity_prediction.main import Experiments
        print("✓ Experiments module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Experiments import failed: {e}")
        return False

def test_package_version():
    """Test that package can be imported and has version."""
    print("\nTesting package initialization...")
    try:
        import Image_Authenticity_prediction.main as iap
        if hasattr(iap, '__version__'):
            print(f"✓ Package version: {iap.__version__}")
        else:
            print("✓ Package imported (no version attribute)")
        return True
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Image Authenticity Prediction - Import Test")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(test_package_version())
    results.append(test_model_imports())
    results.append(test_utils_imports())
    results.append(test_data_imports())
    results.append(test_train_imports())
    results.append(test_experiments_imports())
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All imports working correctly!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python -m Image_Authenticity_prediction train --model vgg16")
        return 0
    else:
        print("✗ Some imports failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
