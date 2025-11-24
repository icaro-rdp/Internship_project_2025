# Quick Reference - Import Cheat Sheet

## CLI Quick Start

### Most Common Use Cases

**Run complete Experiment One (train 10 variants per model, prune, test):**
```bash
python -m Image_Authenticity_prediction experiment-one --train --prune --test
```

**Generate and compare XAI heatmaps:**
```bash
# Generate heatmaps
python -m Image_Authenticity_prediction experiment-two --xai-methods both

# Compare heatmaps
python -m Image_Authenticity_prediction experiment-two --comparison-only
```

**Train a single model:**
```bash
python -m Image_Authenticity_prediction train --model vgg16 --freeze-backbone --epochs 50
```

**Get help on any command:**
```bash
python -m Image_Authenticity_prediction --help
python -m Image_Authenticity_prediction experiment-one --help
python -m Image_Authenticity_prediction experiment-two --help
```

## Quick Imports

### All Models at Once
```python
from Image_Authenticity_prediction.main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    InceptionV3AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor
)
```

### All Utils at Once
```python
from Image_Authenticity_prediction.main.Utils import (
    GradCAM,
    MultiscalePixelMasking,
    FeatureMapsPruner
)
```

### Data and Training
```python
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, DENSENET_DATASET
from Image_Authenticity_prediction.main.train import train_model, test_model, plot_loss_history
```

## Quick Commands

### Train Models
```bash
# VGG16
python -m Image_Authenticity_prediction train --model vgg16 --freeze-backbone --epochs 50 --plot

# ResNet152
python -m Image_Authenticity_prediction train --model resnet152 --epochs 50 --patience 7

# DenseNet161
python -m Image_Authenticity_prediction train --model densenet161 --learning-rate 0.0001
```

### Evaluate Models
```bash
python -m Image_Authenticity_prediction evaluate --model vgg16 --weights Weights/vgg16_best.pth
```

### Run Experiment One (Training, Pruning, Testing)
```bash
# Complete pipeline for all models
python -m Image_Authenticity_prediction experiment-one --train --prune --test

# Training only for specific models
python -m Image_Authenticity_prediction experiment-one --train --models vgg16 resnet152

# Pruning with greedy method only
python -m Image_Authenticity_prediction experiment-one --prune --pruning-method greedy

# Testing all trained and pruned models
python -m Image_Authenticity_prediction experiment-one --test

# Training + greedy pruning for specific models
python -m Image_Authenticity_prediction experiment-one --train --prune --models vgg16 vgg19 --pruning-method greedy

# All phases with both pruning methods
python -m Image_Authenticity_prediction experiment-one --train --prune --test --pruning-method both
```

### Run Experiment Two (XAI Heatmaps & Comparison)
```bash
# Generate GradCAM and MPM heatmaps for all models
python -m Image_Authenticity_prediction experiment-two --xai-methods both

# Generate GradCAM only for specific models
python -m Image_Authenticity_prediction experiment-two --xai-methods gradcam --models vgg16 resnet152

# Generate heatmaps for baseline variants only
python -m Image_Authenticity_prediction experiment-two --variants base

# Run comparison analysis only (requires pre-generated heatmaps)
python -m Image_Authenticity_prediction experiment-two --comparison-only

# Compare between model architectures
python -m Image_Authenticity_prediction experiment-two --comparison-only \
    --comparison-kinds between_model_architectures \
    --comparison-metrics correlation ssim

# Compare within model variants
python -m Image_Authenticity_prediction experiment-two --comparison-only \
    --comparison-kinds within_model_variants \
    --comparison-metrics correlation

# Multiple comparison types
python -m Image_Authenticity_prediction experiment-two --comparison-only \
    --comparison-kinds between_model_architectures within_model_variants cross_methods \
    --comparison-metrics correlation ssim rmse
```

## Model-Dataset Mapping

| Model | Use Dataset | Input Size |
|-------|-------------|------------|
| VGG16, VGG19, ResNet152, EfficientNetB3, BarlowTwins | `IMAGENET_DATASET` | 224×224 |
| DenseNet161 | `DENSENET_DATASET` | 300×300 |
| InceptionV3 | `INCEPTIONV3_DATASET` | 299×299 |

## GradCAM Target Layers

```python
# VGG16
GradCAM(model, target_layer=model.features[28])

# VGG19
GradCAM(model, target_layer=model.features[34])

# ResNet152
GradCAM(model, target_layer=model.features[-1])

# DenseNet161
GradCAM(model, target_layer=model.features.denseblock4)

# InceptionV3
GradCAM(model, target_layer=model.features[-1])

# EfficientNetB3
GradCAM(model, target_layer=model.features[-1])

# BarlowTwins
GradCAM(model, target_layer=model.features[-1])
```

## File Locations

- **Configs**: `Configs/config.yaml`
- **Saved Models**: `Output/Weights/*.pth`
- **Importance Scores**: `Output/Ranking_arrays/*.npy`
- **Dataset**: `Dataset/AIGCIQA2023/`
