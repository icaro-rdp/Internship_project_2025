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

def print_model_structure():
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
        print("\nModel Structures:")
        # print("\nVGG16AuthenticityPredictor:\n", vgg16)
        # print("\nVGG19AuthenticityPredictor:\n", vgg19)
        # print("\nResNet152AuthenticityPredictor:\n", resnet152)
        # print("\nDenseNet161AuthenticityPredictor:\n", densenet161)
        # print("\nInceptionV3AuthenticityPredictor:\n", inceptionv3)
        print("\nEfficientNetB3AuthenticityPredictor:\n", efficientnetb3)
        # print("\nBarlowTwinsAuthenticityPredictor:\n", barlowtwins)
        
        return True
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False


if __name__ == '__main__':
    print("Testing model imports and structures...")
    success = print_model_structure()
    if success:
        print("\n✓ Model structure printed successfully")
    else:
        print("\n✗ Failed to print model structure")