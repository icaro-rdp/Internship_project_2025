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

#### Experiment One: Model Training, Pruning, and Testing

Experiment One provides a complete pipeline for training multiple model variants, pruning them using feature importance analysis, and testing their performance.

**Run the complete pipeline:**
```bash
# Run all phases for all models (training, pruning with both methods, and testing)
python -m Image_Authenticity_prediction experiment-one --train --prune --test

# Run only training for specific models
python -m Image_Authenticity_prediction experiment-one --train --models vgg16 resnet152

# Run only pruning with greedy method
python -m Image_Authenticity_prediction experiment-one --prune --pruning-method greedy

# Run only testing
python -m Image_Authenticity_prediction experiment-one --test

# Run training and greedy pruning for specific models
python -m Image_Authenticity_prediction experiment-one \
    --train --prune \
    --models vgg16 vgg19 resnet152 \
    --pruning-method greedy

# Run all phases with negative impact pruning and custom threshold
python -m Image_Authenticity_prediction experiment-one \
    --train --prune --test \
    --pruning-method negative_impact \
    --threshold 0.1
```

**Experiment One Options:**
- `--models`: Specific models to process (vgg16, vgg19, resnet152, densenet161, efficientnetb3, barlowtwins)
- `--train`: Run training phase (trains 10 variants per model)
- `--prune`: Run pruning phase on trained models
- `--test`: Run testing phase to evaluate trained and pruned models
- `--pruning-method`: Pruning strategy (greedy, negative_impact, both)
- `--threshold`: Threshold for negative_impact pruning (default: 0.0)

**What Experiment One Does:**
1. **Training (1A)**: Trains 10 variants of each model with different random seeds and data splits
2. **Pruning (1B)**: Analyzes feature importance and removes redundant/harmful features
3. **Testing**: Evaluates all trained and pruned models on the test set with comprehensive metrics

**Output Structure:**
```
Image_Authenticity_prediction/main/Experiments/Outputs/Experiment_1_variants/
├── Weights/                    # Model weights for all variants
├── Ranking_arrays/             # Feature importance scores
├── Ranking_Plots/             # Importance score visualizations
├── Training_Plots/            # Training curves
├── Training_History/          # Training history data
└── Test_Results/              # Test evaluation results
```

#### Experiment Two: XAI Heatmap Generation and Comparison

Experiment Two generates explainability heatmaps using GradCAM and Multiscale Pixel Masking, then compares them across models and methods.

**Generate heatmaps:**
```bash
# Generate both GradCAM and MPM heatmaps for all models and variants
python -m Image_Authenticity_prediction experiment-two \
    --xai-methods both \
    --variants all

# Generate only GradCAM heatmaps for specific models
python -m Image_Authenticity_prediction experiment-two \
    --xai-methods gradcam \
    --models vgg16 resnet152 \
    --variants base

# Generate only MPM heatmaps for greedy pruned variants
python -m Image_Authenticity_prediction experiment-two \
    --xai-methods mpm \
    --variants greedy
```

**Run comparison analysis:**
```bash
# Run comparison only (requires pre-generated heatmaps)
python -m Image_Authenticity_prediction experiment-two --comparison-only

# Compare between model architectures
python -m Image_Authenticity_prediction experiment-two \
    --comparison-only \
    --comparison-kinds between_model_architectures \
    --comparison-metrics correlation ssim

# Compare within model variants
python -m Image_Authenticity_prediction experiment-two \
    --comparison-only \
    --comparison-kinds within_model_variants \
    --comparison-metrics correlation

# Compare across XAI methods
python -m Image_Authenticity_prediction experiment-two \
    --comparison-only \
    --comparison-kinds cross_methods \
    --comparison-metrics correlation ssim rmse

# Run multiple comparison types with custom resolution
python -m Image_Authenticity_prediction experiment-two \
    --comparison-only \
    --comparison-kinds between_model_architectures within_model_variants \
    --comparison-metrics correlation ssim \
    --target-resolution 256,256
```

**Experiment Two Options:**
- `--models`: Specific models to process (default: all)
- `--xai-methods`: XAI methods to use (gradcam, mpm, both)
- `--variants`: Variants to process (all, orig, base, greedy, negative)
- `--save-maps` / `--no-save-maps`: Control heatmap saving
- `--comparison-only`: Skip generation, only run comparisons
- `--run-comparison`: Run comparison after generation
- `--comparison-kinds`: Types of comparisons (between_model_architectures, within_model_variants, cross_methods)
- `--comparison-metrics`: Metrics to use (correlation, ssim, rmse, scc)
- `--target-resolution`: Target resolution for comparison (e.g., "224,224")

**What Experiment Two Does:**
1. **Generation**: Creates GradCAM and/or MPM heatmaps for specified models and variants
2. **Comparison**: Analyzes similarity between heatmaps using various metrics
3. **Visualization**: Generates similarity matrices, distributions, and violin plots

**Output Structure:**
```
Image_Authenticity_prediction/main/Experiments/Outputs/Experiment_2_variants/
├── XAI_Maps/
│   ├── GradCAM/               # GradCAM heatmaps (.npy files)
│   └── Multiscale_Pixel_Masking/  # MPM heatmaps (.npy files)
├── Plots/                     # Comparison visualizations
│   ├── *_matrix.png          # Similarity matrices
│   ├── *_distribution.png    # Distribution plots
│   └── *_violin.png          # Violin plots
└── experiment_2b_comparison.json  # Comparison results
```

### Using the Python API

For custom experiments, you can use the Python API directly:

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

**Run experiments programmatically:**

```python
# Experiment One
from Image_Authenticity_prediction.main.Experiments.experiment_one import run_experiment_one_complete

results = run_experiment_one_complete(
    models_to_process=['vgg16', 'resnet152'],
    run_training=True,
    run_pruning=True,
    run_testing=True,
    pruning_method='both',
    threshold=0.0
)

# Experiment Two
from Image_Authenticity_prediction.main.Experiments.experiment_two import run_experiment_2

run_experiment_2(
    models=['vgg16', 'resnet152'],
    xai_methods='both',
    save_maps=True,
    comparison_only=False,
    comparison_kinds=('between_model_architectures', 'within_model_variants'),
    comparison_metrics=('correlation', 'ssim'),
    comparison_target_resolution=(224, 224)
)
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
