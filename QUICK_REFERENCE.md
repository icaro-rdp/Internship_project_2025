# Quick Reference Guide

Quick cheat sheet for common commands, imports, and configurations.

## 📌 Table of Contents

- [CLI Commands](#-cli-commands)
- [Quick Imports](#-quick-imports)
- [Model-Dataset Mapping](#-model-dataset-mapping)
- [Target Layers for GradCAM](#-target-layers-for-gradcam)
- [File Locations](#-file-locations)
- [Common Code Snippets](#-common-code-snippets)

## 🚀 CLI Commands

### Training Models

```bash
# Basic training
python -m Image_Authenticity_prediction train --model vgg16

# With custom parameters
python -m Image_Authenticity_prediction train \
    --model resnet152 \
    --epochs 500 \
    --patience 15 \
    --learning-rate 0.001 \
    --freeze-backbone

# Save and plot training history
python -m Image_Authenticity_prediction train --model vgg16 --plot

# Train without freezing backbone (fine-tune entire model)
python -m Image_Authenticity_prediction train --model densenet161 --no-freeze-backbone
```

### Evaluating Models

```bash
# Basic evaluation
python -m Image_Authenticity_prediction evaluate \
    --model vgg16 \
    --weights Outputs/Experiment_1_variants/Weights/vgg16_best.pth

# Evaluate with different model
python -m Image_Authenticity_prediction evaluate \
    --model resnet152 \
    --weights Outputs/Experiment_1_variants/Weights/resnet152_exp1_orig.pth
```

### Experiment One (Training, Pruning, Testing)

```bash
# Complete pipeline for all models
python -m Image_Authenticity_prediction experiment-one --train --prune --test

# Train only
python -m Image_Authenticity_prediction experiment-one --train

# Train specific models only
python -m Image_Authenticity_prediction experiment-one --train --models vgg16 resnet152

# Pruning only (requires pre-trained weights)
python -m Image_Authenticity_prediction experiment-one --prune

# Pruning with specific method
python -m Image_Authenticity_prediction experiment-one --prune --pruning-method greedy

# Pruning with custom threshold
python -m Image_Authenticity_prediction experiment-one --prune --threshold 0.5

# Testing only (requires trained models)
python -m Image_Authenticity_prediction experiment-one --test

# Full pipeline for specific models with both pruning methods
python -m Image_Authenticity_prediction experiment-one \
    --train --prune --test \
    --models vgg16 vgg19 resnet152 \
    --pruning-method both
```

### Experiment Two (XAI Heatmaps & Comparisons)

```bash
# Generate both GradCAM and MPM heatmaps for all models
python -m Image_Authenticity_prediction experiment-two --xai-methods both

# Generate GradCAM only
python -m Image_Authenticity_prediction experiment-two --xai-methods gradcam

# Generate MPM only
python -m Image_Authenticity_prediction experiment-two --xai-methods mpm

# Generate for specific models
python -m Image_Authenticity_prediction experiment-two \
    --xai-methods both \
    --models vgg16 resnet152 densenet161

# Generate for baseline variants only
python -m Image_Authenticity_prediction experiment-two \
    --xai-methods both \
    --variants base

# Run comparison analysis only (requires pre-generated heatmaps)
python -m Image_Authenticity_prediction experiment-two --comparison-only

# Compare between model architectures
python -m Image_Authenticity_prediction experiment-two --comparison-only \
    --comparison-kinds between_model_architectures \
    --comparison-metrics correlation ssim

# Compare within model variants (original vs pruned)
python -m Image_Authenticity_prediction experiment-two --comparison-only \
    --comparison-kinds within_model_variants \
    --comparison-metrics correlation rmse

# Compare across XAI methods (GradCAM vs MPM)
python -m Image_Authenticity_prediction experiment-two --comparison-only \
    --comparison-kinds cross_methods \
    --comparison-metrics correlation

# Multiple comparison types with multiple metrics
python -m Image_Authenticity_prediction experiment-two --comparison-only \
    --comparison-kinds between_model_architectures within_model_variants cross_methods \
    --comparison-metrics correlation ssim rmse scc
```

### Experiment Three (Ensemble Strategies)

```bash
# Run both bagging and stacking ensemble strategies
python -m Image_Authenticity_prediction experiment-three --strategy both

# Run bagging only
python -m Image_Authenticity_prediction experiment-three --strategy bagging

# Run stacking only
python -m Image_Authenticity_prediction experiment-three --strategy stacking

# Ensemble with specific models only
python -m Image_Authenticity_prediction experiment-three \
    --models vgg16 resnet152 densenet161 \
    --strategy both

# Train ensemble without evaluation
python -m Image_Authenticity_prediction experiment-three \
    --strategy both \
    --no-evaluate

# Evaluate without training (requires pre-trained ensemble models)
python -m Image_Authenticity_prediction experiment-three \
    --strategy both \
    --no-train

# Train, evaluate, and save results
python -m Image_Authenticity_prediction experiment-three \
    --strategy both \
    --train \
    --evaluate \
    --save-results
```

### Get Help

```bash
# General help
python -m Image_Authenticity_prediction --help

# Help for specific command
python -m Image_Authenticity_prediction train --help
python -m Image_Authenticity_prediction evaluate --help
python -m Image_Authenticity_prediction experiment-one --help
python -m Image_Authenticity_prediction experiment-two --help
python -m Image_Authenticity_prediction experiment-three --help
```

## 📦 Quick Imports

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

### All Utilities at Once

```python
from Image_Authenticity_prediction.main.Utils import (
    GradCAM,
    MultiscalePixelMasking,
    FeatureMapsPruner
)
```

### Data and Training

```python
from Image_Authenticity_prediction.main.data import (
    ImageAuthenticityDataset,
    IMAGENET_DATASET,
    DENSENET_DATASET,
    INCEPTIONV3_DATASET,
    IMAGENET_TRANSFORM,
    DENSENET_TRANSFORM,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED
)

from Image_Authenticity_prediction.main.train import (
    train_model,
    test_model,
    plot_loss_history
)
```

### Cleanup and Logging

```python
from Image_Authenticity_prediction.main.Utils.cleanup import (
    clear_gpu_memory,
    cleanup_model_and_data
)

from Image_Authenticity_prediction.main.Utils.logger import (
    info,
    warn,
    error,
    debug,
    set_level
)
```

### Within Package (Relative Imports)

When writing code inside the `main/` package:

```python
# From experiment files
from ..Models import VGG16AuthenticityPredictor, ResNet152AuthenticityPredictor
from ..Utils import GradCAM, FeatureMapsPruner
from ..Utils.cleanup import clear_gpu_memory
from ..Utils.logger import info, warn, error
from ..data import IMAGENET_DATASET, BATCH_SIZE
from ..train import train_model, test_model
```

## 🗺️ Model-Dataset Mapping

**Actively Used Models**:

| Model          | Dataset            | Input Size | Target Layer for GradCAM                  |
| -------------- | ------------------ | ---------- | ----------------------------------------- |
| VGG16          | `IMAGENET_DATASET` | 224×224    | `features.28`                             |
| VGG19          | `IMAGENET_DATASET` | 224×224    | `features.34`                             |
| ResNet152      | `IMAGENET_DATASET` | 224×224    | `features.7.2.conv3`                      |
| EfficientNetB3 | `IMAGENET_DATASET` | 224×224    | `features.8.0`                            |
| BarlowTwins    | `IMAGENET_DATASET` | 224×224    | `features.7.2.conv3`                      |
| DenseNet161    | `DENSENET_DATASET` | 300×300    | `features.denseblock4.denselayer24.conv2` |

**Implemented but Not Used in Experiments**:

| Model       | Dataset               | Input Size | Note                               |
| ----------- | --------------------- | ---------- | ---------------------------------- |
| InceptionV3 | `INCEPTIONV3_DATASET` | 299×299    | Not compatible with pruning method |

## 🎯 Target Layers for GradCAM

### Usage in Code

```python
from Image_Authenticity_prediction.main.Utils import GradCAM

# VGG16 - Last convolutional layer
grad_cam = GradCAM(model, target_layer='features.28')

# VGG19 - Last convolutional layer
grad_cam = GradCAM(model, target_layer='features.34')

# ResNet152 - Last layer of final residual block
grad_cam = GradCAM(model, target_layer='features.7.2.conv3')

# DenseNet161 - Last conv in final dense block
grad_cam = GradCAM(model, target_layer='features.denseblock4.denselayer24.conv2')

# EfficientNetB3 - First layer of last block
grad_cam = GradCAM(model, target_layer='features.8.0')

# BarlowTwins - Last layer before avgpool
grad_cam = GradCAM(model, target_layer='features.7.2.conv3')
```

### Finding Target Layers

```python
# Print model structure to find layer names
print(model)

# Or use this helper to see all named modules
for name, module in model.named_modules():
    print(name, type(module))
```

## 📂 File Locations

### Input Files

| File Type     | Location                                          | Description                                          |
| ------------- | ------------------------------------------------- | ---------------------------------------------------- |
| Dataset CSV   | `Dataset/AIGCIQA2023/real_images_annotations.csv` | Aggregated image annotations and authenticity scores |
| Images        | `Dataset/AIGCIQA2023/Image/`                      | Actual image files                                   |
| Single Scores | `Dataset/Single_score/`                           | 25 CSV files with individual participant scores      |

### Output Files

| File Type         | Location                                                                | Created By         |
| ----------------- | ----------------------------------------------------------------------- | ------------------ |
| Model Weights     | `Outputs/Experiment_1_variants/Weights/*.pth`                           | Training / Pruning |
| Importance Scores | `Outputs/Experiment_1_variants/Ranking_arrays/*.npy`                    | Pruning            |
| Training History  | `Outputs/Experiment_1_variants/Training_History/*.json`                 | Training           |
| Training Plots    | `Outputs/Experiment_1_variants/Training_Plots/*.png`                    | Training           |
| Test Results      | `Outputs/Experiment_1_variants/Test_Results/*.json`                     | Testing            |
| GradCAM Maps      | `Outputs/Experiment_2_variants/XAI_Maps/GradCAM/*.png`                  | Experiment Two     |
| MPM Maps          | `Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking/*.png` | Experiment Two     |
| Comparison Plots  | `Outputs/Experiment_2_variants/Plots/*.png`                             | Experiment Two     |

### Naming Conventions

**Model Weights**:

- Original trained: `{model_name}_exp1_orig.pth`
- After training: `{model_name}_best.pth`
- Pruned: `{model_name}_exp1_greedy_{threshold}.pth`

**Examples**:

- `vgg16_exp1_orig.pth`
- `resnet152_best.pth`
- `densenet161_exp1_greedy_0.5.pth`

## 💻 Common Code Snippets

### Setup Path and Import

```python
import sys
from pathlib import Path

# Add project to path
project_parent = Path('/path/to/parent/of/Image_Authenticity_prediction')
sys.path.insert(0, str(project_parent))

# Now import
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
```

### Train a Model

```python
import torch
from torch.utils.data import DataLoader
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE, NUM_WORKERS
from Image_Authenticity_prediction.main.train import train_model

# Create model
model = VGG16AuthenticityPredictor(freeze_backbone=True)

# Setup
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Data loaders
train_loader = DataLoader(IMAGENET_DATASET['train'], batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(IMAGENET_DATASET['val'], batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS)

dataloaders = {'train': train_loader, 'val': val_loader}

# Train
best_model, history = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=500,
    device='cuda',
    patience=15
)

# Save
torch.save(best_model.state_dict(), 'Outputs/Experiment_1_variants/Weights/vgg16_best.pth')
```

### Load and Evaluate

```python
import torch
from torch.utils.data import DataLoader
from Image_Authenticity_prediction.main.Models import ResNet152AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE
from Image_Authenticity_prediction.main.train import test_model

# Load model
model = ResNet152AuthenticityPredictor(freeze_backbone=False)
model.load_state_dict(
    torch.load('Outputs/Experiment_1_variants/Weights/resnet152_best.pth',
               weights_only=True)
)

# Test loader
test_loader = DataLoader(IMAGENET_DATASET['test'], batch_size=BATCH_SIZE, shuffle=False)

# Evaluate
criterion = torch.nn.MSELoss()
results = test_model(model, test_loader, criterion, device='cuda',
                    return_additional_metrics=True)

print(f"RMSE: {results['rmse']:.4f}")
print(f"PLCC: {results['plcc']:.4f}")
print(f"SRCC: {results['srcc']:.4f}")
```

### Generate GradCAM

```python
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.Utils import GradCAM
from Image_Authenticity_prediction.main.data import IMAGENET_TRANSFORM
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = VGG16AuthenticityPredictor()
model.load_state_dict(
    torch.load('Outputs/Experiment_1_variants/Weights/vgg16_best.pth',
               weights_only=True)
)
model.eval()

# Setup GradCAM
grad_cam = GradCAM(model, target_layer='features.28')

# Load image
image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = IMAGENET_TRANSFORM(image).unsqueeze(0)

# Generate CAM
cam = grad_cam.generate_cam(image_tensor, device='cuda')

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(cam, cmap='jet')
plt.colorbar()
plt.title('GradCAM Heatmap')
plt.savefig('gradcam_output.png', dpi=300, bbox_inches='tight')
plt.show()

# Cleanup
grad_cam.cleanup()
```

### Prune a Model

```python
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.Utils import FeatureMapsPruner
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE
from Image_Authenticity_prediction.main.train import test_model
from torch.utils.data import DataLoader
import torch.nn as nn

# Load model
model = VGG16AuthenticityPredictor()
model.load_state_dict(
    torch.load('Outputs/Experiment_1_variants/Weights/vgg16_best.pth',
               weights_only=True)
)

# Test loader
test_loader = DataLoader(IMAGENET_DATASET['test'], batch_size=BATCH_SIZE)

# Initialize pruner
pruner = FeatureMapsPruner(
    model=model,
    dataloader=test_loader,
    layer_name='features.0',  # First conv layer
    criterion=nn.MSELoss(),
    eval_function=test_model,
    device='cuda'
)

# Compute importance scores (cached to .npy file)
scores = pruner.compute_importance_scores()

# Prune using greedy strategy
results = pruner.greedy_pruning(
    save_path='Outputs/Experiment_1_variants/Weights/vgg16_pruned_50.pth',
    threshold=0.5  # Remove 50% least important filters
)

print(f"Original MSE: {results['original_performance']:.4f}")
print(f"Pruned MSE: {results['pruned_performance']:.4f}")
```

### Memory Management

```python
from Image_Authenticity_prediction.main.Utils.cleanup import (
    clear_gpu_memory,
    cleanup_model_and_data
)

# After training or evaluation
cleanup_model_and_data(model, dataloaders, optimizer)
clear_gpu_memory()

# Check GPU memory
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    print(f"GPU Memory Allocated: {allocated:.2f} GB")
```

### Colored Logging

```python
from Image_Authenticity_prediction.main.Utils.logger import info, warn, error, debug

info("Training started")           # Blue
warn("High memory usage detected") # Yellow
error("Training failed!")          # Red
debug("Batch 10/100 processed")    # Gray
```

## 🔍 Quick Checks

### Verify Dataset

```bash
# Check if dataset exists
ls Dataset/AIGCIQA2023/
ls Dataset/Single_score/

# AIGCIQA2023 should show:
# - real_images_annotations.csv
# - Image/ folder

# Single_score should show:
# - 25 participant CSV files
```

### Check GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Test Imports

```python
# Quick import test
try:
    from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
    from Image_Authenticity_prediction.main.Utils import GradCAM
    from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

## 📚 Related Documentation

- **SETUP_GUIDE.md**: Detailed setup, imports, outputs, troubleshooting
- **PROJECT_STRUCTURE.md**: Complete project structure documentation
- **README.md**: Project overview and installation
- **TODO.md**: Current tasks and known issues

---

**Last Updated**: January 5, 2026  
**Maintainer**: Icaro Redepaolini ([github.com/icaro-rdp](https://github.com/icaro-rdp))
