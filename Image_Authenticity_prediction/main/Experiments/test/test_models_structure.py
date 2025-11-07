#!/usr/bin/env python3
"""
Test script to verify all imports and module structure are correct.

Usage:
    python test_imports.py
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def print_model_structure(models_to_print=None):
    """Print the structure of all models."""
    
    try:
        from main.Models import (
            VGG16AuthenticityPredictor,
            VGG19AuthenticityPredictor,
            ResNet152AuthenticityPredictor,
            DenseNet161AuthenticityPredictor,
            InceptionV3AuthenticityPredictor,
            EfficientNetB3AuthenticityPredictor,
            BarlowTwinsAuthenticityPredictor
        )
        print("✓ All models imported successfully")
        vgg16 = VGG16AuthenticityPredictor()
        vgg19 = VGG19AuthenticityPredictor()
        resnet152 = ResNet152AuthenticityPredictor()
        densenet161 = DenseNet161AuthenticityPredictor()
        inceptionv3 = InceptionV3AuthenticityPredictor()
        efficientnetb3 = EfficientNetB3AuthenticityPredictor()
        barlowtwins = BarlowTwinsAuthenticityPredictor()
        
        models = {
            'VGG16': vgg16,
            'VGG19': vgg19,
            'ResNet152': resnet152,
            'DenseNet161': densenet161,
            'InceptionV3': inceptionv3,
            'EfficientNetB3': efficientnetb3,
            'BarlowTwins': barlowtwins
        }

        for model_name, model in models.items():
            if models_to_print and model_name not in models_to_print:
                continue
            print(f"\nModel Structure: {model_name}")
            print(model)

        return True
    except Exception as e:
        print(f"✗ Error importing models: {e}")    
        return False


if __name__ == '__main__':
    # Specify models to print or None for all
    success = print_model_structure(models_to_print=None)
    if success:
        print("\nAll model structures printed successfully.")
    else:
        print("\nFailed to print some model structures.")