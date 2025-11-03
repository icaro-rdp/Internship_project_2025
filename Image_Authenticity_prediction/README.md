# Image Authenticity Prediction

A deep learning framework for predicting image authenticity using various CNN architectures including VGG, ResNet, DenseNet, EfficientNet, InceptionV3, and BarlowTwins.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
  - [Running Experiments](#running-experiments)
- [Available Models](#available-models)
- [Dataset](#dataset)
- [Features](#features)
- [Contributing](#contributing)

## 🔍 Overview

This project implements multiple deep learning models for image authenticity prediction. It supports:

- **Multiple CNN Architectures**: VGG16, VGG19, ResNet-152, DenseNet-161, InceptionV3, EfficientNet-B3, and BarlowTwins
- **Transfer Learning**: Pretrained models with custom regression heads
- **Training with Early Stopping**: Automatic model checkpointing based on validation loss
- **Model Explainability**: GradCAM and Multiscale Pixel Masking for visualization
- **Feature Pruning**: Tools for network compression and analysis

## 📁 Project Structure

```
Image_Authenticity_prediction/
├── __main__.py                 # Main entry point for CLI
├── TODO.md                     # Development tasks
├── README.md                   # This file
├── Configs/
│   └── config.yaml            # Configuration file
├── Dataset/
│   └── AIGCIQA2023/           # Dataset directory (contact the authors for access)
└── main/
    ├── __init__.py
    ├── data.py                # Dataset and data loaders
    ├── train.py               # Training and evaluation functions
    ├── Models/
    │   ├── __init__.py
    │   └── models.py          # Model architectures
    ├──Output/            # Output directory for results (weights, rankings, etc.)
        ├── Weights/               # Saved model weights
        ├── Ranking_arrays/        # Feature importance rankings
        └── ...                     # Other output files
    ├── Utils/
    │   ├── __init__.py
    │   ├── explainability.py  # GradCAM and visualization tools
    │   └── pruning.py         # Feature map pruning utilities
    └── Experiments/
        ├── __init__.py
        ├── experiment_one.py
        ├── experiment_two.py
        └── experiment_three.py
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.10+

### Setup

1. **Clone the repository**:
   ```bash
   cd /path/to/icaro_rdp_projects
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install pandas pillow numpy matplotlib pyyaml tqdm
   ```

3. **Prepare the dataset**:
   Ensure your dataset is located at:
   ```
   Dataset/AIGCIQA2023/
   ├── real_images_annotations.csv
   └── [image files]
   ```

## ⚙️ Configuration

Edit `Configs/config.yaml` to customize your settings:

```yaml
run_settings:
  device: 'cuda'  # 'cuda' or 'cpu'

pruning:
  layer_name: 'features.2'
  threshold: 0.0

paths:
  weights_dir: 'Weights'
  rankings_dir: 'Ranking_arrays'
  greedy_pruned_model: 'pruned_model.pth'
  negative_pruned_model: 'negative_impact_pruned_model.pth'
  importance_scores: 'real_authenticity_batch_importance_scores.npy'

data:
  batch_size: 16
```

## 🚀 Usage

### Training Models

Train a model using the command-line interface:

```bash
# Train VGG16 with frozen backbone
python -m Image_Authenticity_prediction train --model vgg16 --freeze-backbone

# Train ResNet-152 with custom settings
python -m Image_Authenticity_prediction train \
    --model resnet152 \
    --epochs 50 \
    --patience 7 \
    --learning-rate 0.001 \
    --plot

# Available options:
#   --model: vgg16, vgg19, resnet152, densenet161, inceptionv3, efficientnetb3, barlowtwins
#   --epochs: Maximum number of training epochs (default: 50)
#   --patience: Early stopping patience (default: 7)
#   --learning-rate: Learning rate (default: 0.001)
#   --freeze-backbone: Freeze pretrained backbone weights
#   --plot: Show training history plot after training
```

### Evaluating Models

Evaluate a trained model on the test set:

```bash
# Evaluate a trained model
python -m Image_Authenticity_prediction evaluate \
    --model vgg16 \
    --weights Weights/vgg16_best.pth
```

### Running Experiments

Use the Python API for custom experiments:

```python
import sys
sys.path.insert(0, '/path/to/icaro_rdp_projects')

from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
from Image_Authenticity_prediction.main.train import train_model
from torch.utils.data import DataLoader
import torch

# Initialize model
model = VGG16AuthenticityPredictor(freeze_backbone=True)

# Prepare data
train_loader = DataLoader(IMAGENET_DATASET['train'], batch_size=64, shuffle=True)
val_loader = DataLoader(IMAGENET_DATASET['test'], batch_size=64, shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader}

# Setup training
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
best_model, history = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=50,
    device='cuda',
    patience=7
)

# Save model
torch.save(best_model.state_dict(), 'Weights/my_model.pth')
```

### Using Explainability Tools

#### GradCAM Visualization

```python
from Image_Authenticity_prediction.main.Utils import GradCAM
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
import torch

# Load model
model = VGG16AuthenticityPredictor()
model.load_state_dict(torch.load('Weights/vgg16_best.pth'))

# Initialize GradCAM
# For VGG16, use the last conv layer
grad_cam = GradCAM(model, target_layer=model.features[28])

# Generate CAM for an image
image_tensor = torch.randn(1, 3, 224, 224)  # Replace with actual image
cam = grad_cam.generate_cam(image_tensor)

# Cleanup
grad_cam.cleanup()
```

#### Multiscale Pixel Masking

```python
from Image_Authenticity_prediction.main.Utils import MultiscalePixelMasking

# Initialize
mpm = MultiscalePixelMasking(
    model=model,
    sigma_list=[8, 16, 32],
    pixel_batch_size=100,
    mask_value=0.0
)

# Generate saliency map
saliency_map, original_score = mpm.generate_saliency_map(image_tensor)
```

### Feature Map Pruning

```python
from Image_Authenticity_prediction.main.Utils import FeatureMapsPruner
from Image_Authenticity_prediction.main.train import test_model
import torch.nn as nn

# Initialize pruner
pruner = FeatureMapsPruner(
    model=model,
    dataloader=test_loader,
    layer_name='features.0',  # Layer to prune, check Target Layers
    criterion=nn.MSELoss(),
    eval_function=test_model,
    device=torch.device('cuda')
)

# Compute importance scores
scores = pruner.compute_importance_scores(
    save_path='Ranking_arrays/importance_scores.npy'
)

# Run greedy pruning
results = pruner.greedy_pruning(
    model_save_path='Weights/pruned_model.pth'
)

print(f"Removed {len(results['removed_features'])} features")
print(f"Improvement: {results['improvement']:.4f}")
```

## 🤖 Available Models

All models are based on pretrained backbones with custom regression heads for authenticity prediction:

| Model | Input Size | Parameters | Backbone |
|-------|-----------|-----------|----------|
| **VGG16** | 224×224 | ~138M | ImageNet pretrained |
| **VGG19** | 224×224 | ~144M | ImageNet pretrained |
| **ResNet-152** | 224×224 | ~60M | ImageNet pretrained |
| **DenseNet-161** | 300×300 | ~28M | ImageNet pretrained |
| **InceptionV3** | 299×299 | ~27M | ImageNet pretrained |
| **EfficientNet-B3** | 300×300 | ~12M | ImageNet pretrained |
| **BarlowTwins** | 224×224 | ~25M | Self-supervised pretrained |

### Target Layers for GradCAM

| Model | Recommended Target Layer |
|-------|-------------------------|
| VGG16 | `model.features[28]` |
| VGG19 | `model.features[34]` |
| ResNet-152 | `model.features[-1]` or `model.features[7][-1]` |
| DenseNet-161 | `model.features.denseblock4` |
| InceptionV3 | `model.features[-1]` |
| EfficientNet-B3 | `model.features[-1]` |
| BarlowTwins | `model.features[-1]` |

## 📊 Dataset

The project uses the **AIGCIQA2023** dataset:

- **CSV Annotations**: `real_images_annotations.csv`
- **Split**: 80% training, 20% testing (random seed: 42)

### Data Transforms

- **ImageNet Models**: Resize to 256×256, center crop to 224×224
- **DenseNet/InceptionV3**: Resize to 320×320, center crop to 300×300
- **Normalization**: ImageNet mean and std

## ✨ Features

### Training
- ✅ Early stopping with patience
- ✅ Model checkpointing (saves best model)
- ✅ Training history tracking
- ✅ Loss curve visualization

### Explainability
- ✅ GradCAM for visual explanations (gradient over regression activation maps)
- ✅ Multiscale Pixel Masking (Occlusion Saliency)
- ✅ Automated saliency map generation

### Model Optimization
- ✅ Feature map importance ranking (Importance Scores (IS) based on performance drop (MSE))
- ✅ Greedy pruning strategy (removes least important features iteratively)
- ✅ Negative impact pruning (removes features that negatively impact performance (IS > 0))
- ✅ Automated performance evaluation after pruning

## 📝 TODO

See [TODO.md](TODO.md) for current development tasks and roadmap.

## 🤝 Contributing

This is a research project done during an internship period at CiMEC. The project is part of Re Depaolini Icaro's Thesis. For questions or contributions, please contact the project maintainer.

## 🔗 Reference
- Icaro Re Depaolini, University of Trento, "Predicting Image Authenticity: Human alignment and Explainability methods", Oct 14 2025.


## 📄 License

This project is part of academic research. Please contact the authors for usage permissions.

## 🙏 Acknowledgments

- Pretrained models from PyTorch and torchvision
- BarlowTwins implementation from Facebook Research
- AIGCIQA2023 dataset
- CiMEC, University of Trento

---

**Author**: Icaro Re Depaolini  
**Institution**: CiMec, University of Trento  
**Last Updated**: November 2025
