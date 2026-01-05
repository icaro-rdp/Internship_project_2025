"""
Image Authenticity Prediction Package

A deep learning framework for predicting image authenticity using multiple CNN architectures
and advanced techniques including feature map pruning and explainability analysis.

Main Components:
- main.Models: 7 pre-trained CNN architectures (VGG16, VGG19, ResNet152, DenseNet161, InceptionV3, EfficientNetB3, BarlowTwins)
- main.data: Dataset and data loader definitions
- main.train: Training and evaluation utilities
- main.Utils: Explainability (GradCAM, MPM), pruning, and visualization tools
- main.Experiments: Three research experiments (training, XAI, ensemble methods)

Usage:
    python -m Image_Authenticity_prediction train --model vgg16
    python -m Image_Authenticity_prediction experiment-one --train --test
    python -m Image_Authenticity_prediction experiment-two --xai-methods both
    python -m Image_Authenticity_prediction experiment-three --strategy both

For more information, see README.md or SETUP_GUIDE.md
"""

__version__ = "1.0.0"
__author__ = "Icaro Redepaolini"
