# Quick Reference - Import Cheat Sheet

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
- **Saved Models**: `Weights/*.pth`
- **Importance Scores**: `Ranking_arrays/*.npy`
- **Dataset**: `Dataset/AIGCIQA2023/`
