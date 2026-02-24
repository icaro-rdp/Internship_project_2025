# Setup and Import Structure Guide

A comprehensive guide to setting up and using the Image Authenticity Prediction project.

## 📋 Table of Contents

- [Getting Started (5 Steps)](#-getting-started-5-steps) ⭐ **START HERE**
- [Project Structure](#-project-structure)
- [Directory Purposes](#-directory-purposes)
- [Git-Ignored Directories](#-git-ignored-directories)
- [Output Generation](#-output-generation)
- [Module Organization](#-module-organization)
- [Import Patterns](#-import-patterns)
- [Module Exports](#-module-exports)
- [Common Usage Patterns](#-common-usage-patterns)
- [Troubleshooting](#-troubleshooting)

## 🚀 Getting Started (5 Steps)

**New to the project?** Follow these steps to get everything set up and running.

### Step 1: Install PyTorch and Dependencies

```bash
# Install PyTorch with CUDA support (recommended for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only (slower):
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

**What to install:**

- pandas, pillow, numpy, matplotlib, tqdm
- scipy, scikit-learn, scikit-image, seaborn, opencv-python

### Step 2: Understand the Project Structure

This project has a main package `Image_Authenticity_prediction/` with:

- **Models**: 7 CNN architectures for image authenticity prediction
- **Utils**: XAI (explainability) tools, pruning, logging, visualization
- **Data**: Dataset classes and pre-configured data loaders
- **Experiments**: 3 research experiments (training, explainability, ensemble)

**Key files to know:**

- `Image_Authenticity_prediction/__main__.py` - CLI entry point
- `Image_Authenticity_prediction/main/data.py` - Dataset loading
- `Image_Authenticity_prediction/main/train.py` - Training/testing
- `Image_Authenticity_prediction/main/Models/models.py` - Model definitions

### Step 3: Set Up Your Dataset

Create the dataset directory structure:

```bash
# From project root
mkdir -p Dataset/AIGCIQA2023/Image Dataset/Single_score

# Download dataset files (ask authors or check terabox link in docs)
# Place in:
# - Dataset/AIGCIQA2023/real_images_annotations.csv
# - Dataset/AIGCIQA2023/Image/ (image files)
# - Dataset/Single_score/ (25 participant CSV files)
```

**Dataset structure needed:**

```
Dataset/
├── AIGCIQA2023/
│   ├── real_images_annotations.csv    # Required: image paths and scores
│   └── Image/                         # Required: image files (0.png, 1.png, etc.)
└── Single_score/                      # Optional: for noise ceiling calculation
    ├── participant_01.csv
    ├── participant_02.csv
    └── ... (25 files)
```

### Step 4: Try the CLI (Easiest Way to Start)

The CLI provides simple commands without writing code:

```bash
# From project root directory
cd /path/to/Internship_project_2025

# Train a single model
python -m Image_Authenticity_prediction train --model vgg16 --epochs 50

# Evaluate a trained model
python -m Image_Authenticity_prediction evaluate \
    --model vgg16 \
    --weights Outputs/Experiment_1_variants/Weights/vgg16_best.pth

# Run complete experiment pipeline
python -m Image_Authenticity_prediction experiment-one --train --prune --test
```

**See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for all CLI commands.**

### Step 5: Use the Python API (For Advanced Use)

Import and use models directly in your code:

```python
import torch
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE
from Image_Authenticity_prediction.main.train import train_model, test_model
from torch.utils.data import DataLoader

# Initialize model
model = VGG16AuthenticityPredictor(freeze_backbone=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Create data loaders
train_loader = DataLoader(IMAGENET_DATASET['train'], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(IMAGENET_DATASET['val'], batch_size=BATCH_SIZE)
dataloaders = {'train': train_loader, 'val': val_loader}

# Train with early stopping
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_model, history = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=500,
    device=device,
    patience=15
)

# Test on test set
test_loader = DataLoader(IMAGENET_DATASET['test'], batch_size=BATCH_SIZE)
metrics = test_model(best_model, test_loader, criterion, device=device, return_additional_metrics=True)
print(f"Test RMSE: {metrics['rmse']:.4f}")

# Save model
torch.save(best_model.state_dict(), 'Outputs/my_model.pth')
```

---

## 📁 Project Structure

The Image Authenticity Prediction project follows a hierarchical module structure:

```text
Image_Authenticity_prediction/
├── __main__.py                    # CLI entry point
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── SETUP_GUIDE.md                 # This file
├── TODO.md                        # Development tasks
├── PROJECT_STRUCTURE.md           # Detailed structure documentation
├── QUICK_REFERENCE.md             # Quick reference guide
│
├── Dataset/                       # ⚠️ GITIGNORED - User must create
│   ├── AIGCIQA2023/              # Main dataset location
│   │   ├── real_images_annotations.csv
│   │   └── Image/                # Images folder
│   │       ├── 0.png
│   │       ├── 1.png
│   │       └── ...
│   └── Single_score/             # Single participant scores (25 CSV files)
│       ├── participant_01.csv
│       ├── participant_02.csv
│       └── ...
│
├── main/                          # Core package
│   ├── __init__.py               # Package initialization
│   ├── data.py                   # Dataset classes and data loaders
│   ├── train.py                  # Training/testing utilities
│   │
│   ├── Models/                   # Model architectures
│   │   ├── __init__.py          # Exports all model classes
│   │   └── models.py            # 7 CNN architectures
│   │
│   ├── Utils/                    # Utility modules
│   │   ├── __init__.py          # Exports utility classes
│   │   ├── cleanup.py           # Memory management
│   │   ├── comparisons.py       # Model comparison metrics
│   │   ├── explainability.py    # GradCAM & Multiscale Pixel Masking
│   │   ├── logger.py            # Colored logging utilities
│   │   ├── normalization.py     # Data normalization functions
│   │   ├── pruning.py           # Network pruning tools
│   │   └── visualization.py     # Plotting and visualization
│   │
│   └── Experiments/              # Research experiments
│       ├── __init__.py
│       ├── experiment_one.py     # Training, pruning, and testing models
│       ├── experiment_two.py     # XAI methods and comparisons
│       ├── experiment_three.py   # Ensemble strategies (WIP)
│       ├── analysis_consistency_vs_accuracy.py
│       ├── noise_ceiling_analysis.ipynb
│       ├── EXPERIMENT_1_TECHNICAL_REPORT.md
│       ├── EXPERIMENT_2_TECHNICAL_REPORT.md
│       └── test/                 # Unit tests for experiments
│           ├── test_experiment_one_minimal.py
│           ├── test_imports.py
│           ├── test_models_performances.py
│           └── test_models_structure.py
```

## 🎯 Directory Purposes

### Core Directories

- **`__main__.py`**: Command-line interface (CLI) entry point. Allows running the package as a module with commands like `train`, `evaluate`, and `experiment`.

- **`main/`**: The core Python package containing all source code.

  - **`data.py`**: Defines `ImageAuthenticityDataset` class, creates train/val/test splits, and provides pre-configured datasets with transformations for different model architectures (IMAGENET_DATASET, DENSENET_DATASET, INCEPTIONV3_DATASET).

  - **`train.py`**: Contains `train_model()` with early stopping, `test_model()` for evaluation with metrics (MSE, RMSE, PLCC, SRCC, KRCC), and `plot_loss_history()` for visualization.

- **`main/Models/`**: Model architecture definitions.

  - **`models.py`**: Implements 7 CNN-based authenticity predictors: VGG16, VGG19, ResNet152, DenseNet161, InceptionV3, EfficientNetB3, and BarlowTwins. Each model uses transfer learning with frozen backbones and custom regression heads.

- **`main/Utils/`**: Utility modules for various tasks.

  - **`cleanup.py`**: Memory management functions (`clear_gpu_memory()`, `cleanup_model_and_data()`) to prevent GPU memory leaks.

  - **`comparisons.py`**: Functions for comparing model outputs and computing similarity metrics between heatmaps.

  - **`explainability.py`**: Implements explainability methods: `GradCAM` and `MultiscalePixelMasking` for generating saliency maps.

  - **`logger.py`**: Colored console logging with functions: `info()`, `warn()`, `error()`, `debug()`.

  - **`normalization.py`**: Data normalization utilities.

  - **`pruning.py`**: `FeatureMapsPruner` class for network compression by pruning less important convolutional filters.

  - **`visualization.py`**: Plotting functions for similarity matrices, distributions, and visualizations.

- **`main/Experiments/`**: Research experiment scripts.

  - **`experiment_one.py`**: Main experiment for training all models, performing feature map pruning, and testing performance. Creates outputs in `Outputs/Experiment_1_variants/`.

  - **`experiment_two.py`**: Generates and compares explainability maps (GradCAM and Multiscale Pixel Masking) across models. Saves outputs to `Outputs/Experiment_2_variants/`.

  - **`experiment_three.py`**: Implements ensemble learning strategies (bagging and stacking) - Work in Progress. Outputs to `Outputs/Experiment_3_ensemble/`.

### Output Directories

⚠️ **All output directories are automatically created by the code when needed and are GITIGNORED.**

**From Experiments:**

- **`Outputs/Experiment_1_variants/`**: Generated by experiment_one.py

  - `Weights/`: Trained model weights (.pth files)
  - `Ranking_arrays/`: Feature importance rankings (.npy files)
  - `Ranking_Plots/`: Visualizations of feature importance
  - `Training_Plots/`: Loss curves and training history plots
  - `Training_History/`: JSON files with epoch-by-epoch metrics
  - `Test_Results/`: Performance metrics on test set

- **`Outputs/Experiment_2_variants/`**: Generated by experiment_two.py

  - `XAI_Maps/GradCAM/`: GradCAM saliency maps for each model
  - `XAI_Maps/Multiscale_Pixel_Masking/`: Multiscale pixel masking maps
  - `Plots/`: Comparison plots and similarity matrices

- **`Outputs/Experiment_3_ensemble/`**: Generated by experiment_three.py (WIP)
  - `Weights/Stacking/`: Ensemble model weights
  - `Results/`: Ensemble performance metrics

**Legacy Output Patterns (also gitignored):**

The `.gitignore` also excludes these patterns from older project structure:

- `Models/*/Weights`: Model weights in various subdirectories
- `Models/*/Ranking_arrays`: Ranking arrays in model-specific folders
- `Models/*/saliency_experiment_outputs`: Saliency map outputs
- `Models/*/multiscale_masking_outputs_ensemble`: Ensemble masking outputs

## 🚫 Git-Ignored Directories

The following directories are excluded from version control and must be set up locally:

### 1. Dataset Directory (`Dataset/`)

**Why gitignored**: Large image datasets (potentially GBs) should not be in version control.

**How to set up**:

```bash
# Create the directory structure
mkdir -p Image_Authenticity_prediction/Dataset/AIGCIQA2023
mkdir -p Image_Authenticity_prediction/Dataset/Single_score

# Place your dataset files:
# - real_images_annotations.csv (required, ask the authors for access)
# - Image files referenced in the CSV (ask the authors for access or download from Terabox: https://www.terabox.com/sharing/link?surl=DtV-A9XiuQQDvVPXn6rYvg)
# - Single participant score CSV files for noise ceiling calculation
```

**Expected structure**:

```text
Dataset/
├── AIGCIQA2023/
│   ├── real_images_annotations.csv    # Required CSV file with aggregated annotations
│   └── Image/                         # Images folder
│       ├── 0.png
│       ├── 1.png
│       └── ...
│
└── Single_score/                      # Single participant scores for noise ceiling
    ├── participant_01.csv
    ├── participant_02.csv
    └── ... (25 CSV files total)
```

**CSV format**: The CSV should contain columns including:

- Image path (column index 3)
- Authenticity score (column index 1)

The code in [data.py](Image_Authenticity_prediction/main/data.py) automatically resolves paths relative to the project root, so CSV paths like `./Dataset/AIGCIQA2023/image.jpg` will work correctly.

### 2. Output Directories

**Why gitignored**: Generated files (weights, plots, results) are large, experiment-specific, and reproducible.

**How they're created**: All output directories are **automatically created** by the code when running experiments or training. You don't need to create them manually.

**When generated**:

- Training: Creates `Outputs/Experiment_1_variants/Weights/` and saves `.pth` files
- Pruning: Creates `Ranking_arrays/` and saves feature importance scores
- Experiments: Create their respective output folders as needed

### 3. Other Ignored Items

- **Python artifacts**: `__pycache__/`, `*.pyc`, `*.egg-info/`
- **Virtual environments**: `venv/`, `env/`, `ENV/`
- **IDE files**: `.vscode/`, `.idea/`, `.DS_Store`
- **Jupyter checkpoints**: `.ipynb_checkpoints/`
- **Logs**: `*.log`, temporary files
- **Pickled data**: `*.pkl`, `*.h5`

## 🔄 Output Generation

Understanding how the application generates and stores outputs:

### Training Outputs

**Generated by**: [train.py](Image_Authenticity_prediction/main/train.py) `train_model()` function

**Location**: Specified when saving model (typically `Outputs/Experiment_1_variants/Weights/`)

**What's created**:

- **Model weights**: `.pth` files containing `state_dict`
- **Training history**: Dictionary with `'train_loss'` and `'val_loss'` lists
- **Plots**: Training/validation loss curves (if `plot_loss_history()` is called)

**How it works**:

```python
# Training creates a history dictionary
best_model, history = train_model(model, dataloaders, criterion, optimizer, ...)

# Saving weights
torch.save(best_model.state_dict(), 'Outputs/Experiment_1_variants/Weights/model_name.pth')

# History contains epoch-by-epoch losses
# {'train_loss': [loss1, loss2, ...], 'val_loss': [loss1, loss2, ...]}
```

### Pruning Outputs

**Generated by**: [pruning.py](Image_Authenticity_prediction/main/Utils/pruning.py) `FeatureMapsPruner` class

**Location**: `Outputs/Experiment_1_variants/Ranking_arrays/`

**What's created**:

- **Importance scores**: `.npy` files containing per-filter importance scores
- **Pruned models**: `.pth` files with reduced number of filters
- **Statistics**: JSON files with pruning metrics

**How it works**:

```python
pruner = FeatureMapsPruner(model, dataloader, layer_name, criterion, eval_function, device)

# Compute importance scores (cached to .npy file)
scores = pruner.compute_importance_scores()

# Prune and save model
results = pruner.greedy_pruning(save_path='Outputs/.../Weights/pruned_model.pth')
```

### Explainability Outputs

**Generated by**: [explainability.py](Image_Authenticity_prediction/main/Utils/explainability.py) - `GradCAM` and `MultiscalePixelMasking`

**Location**: `Outputs/Experiment_2_variants/XAI_Maps/`

**What's created**:

- **GradCAM maps**: Heatmap images showing important regions (.png or .jpg)
- **Multiscale masking maps**: Pixel-level saliency maps
- **Comparison data**: JSON files with similarity metrics

**How it works**:

```python
# GradCAM generates activation maps
grad_cam = GradCAM(model, target_layer='features.28')
cam = grad_cam.generate_cam(image_tensor)

# Multiscale Pixel Masking generates occlusion-based saliency
mpm = MultiscalePixelMasking(model, sigma=[3, 17, 65])
saliency_map = mpm.generate_saliency_map(image_tensor)
```

### Experiment Outputs

**Experiment 1** ([experiment_one.py](Image_Authenticity_prediction/main/Experiments/experiment_one.py)):

- Trains all 6 models (VGG16, VGG19, ResNet152, DenseNet161, EfficientNetB3, BarlowTwins)
- Saves weights to `Outputs/Experiment_1_variants/Weights/`
- Computes feature importance and saves to `Ranking_arrays/`
- Performs greedy pruning and saves pruned models
- Generates test results with MSE, RMSE, PLCC, SRCC metrics

**Experiment 2** ([experiment_two.py](Image_Authenticity_prediction/main/Experiments/experiment_two.py)):

- Loads trained models from Experiment 1
- Generates GradCAM and Multiscale Pixel Masking maps
- Compares saliency maps across models
- Creates similarity matrices and distribution plots
- Saves all outputs to `Outputs/Experiment_2_variants/`

**Experiment 3** ([experiment_three.py](Image_Authenticity_prediction/main/Experiments/experiment_three.py)) - WIP:

- Implements ensemble strategies (bagging and stacking)
- Loads base models and trains meta-learners
- Saves ensemble results to `Outputs/Experiment_3_ensemble/`

## 📦 Module Organization

The project uses Python's package system with proper `__init__.py` files that define `__all__` exports for clean imports.

### Package Hierarchy

```
main/                          # Main package
├── __init__.py               # Package initialization
├── data.py                   # Dataset and data loading
├── train.py                  # Training and testing functions
│
├── Models/                   # Subpackage for model architectures
│   ├── __init__.py          # Exports 7 model classes (6 actively used + 1 not used)
│   └── models.py            # Model definitions
│
├── Utils/                    # Subpackage for utilities
│   ├── __init__.py          # Exports GradCAM, MultiscalePixelMasking, FeatureMapsPruner
│   ├── cleanup.py           # Memory management
│   ├── comparisons.py       # Comparison metrics
│   ├── explainability.py    # XAI methods
│   ├── logger.py            # Colored logging
│   ├── normalization.py     # Data normalization
│   ├── pruning.py           # Network pruning
│   └── visualization.py     # Plotting functions
│
└── Experiments/              # Research experiments
    ├── __init__.py          # Empty
    ├── experiment_one.py
    ├── experiment_two.py
    └── experiment_three.py
```

## 🔗 Import Patterns

### From External Scripts

When importing from outside the package directory (e.g., from a Jupyter notebook or external script):

```python
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/path/to/parent/of/Image_Authenticity_prediction')

# Import models
from Image_Authenticity_prediction.main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    InceptionV3AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor
)

# Import utilities
from Image_Authenticity_prediction.main.Utils import (
    GradCAM,
    MultiscalePixelMasking,
    FeatureMapsPruner
)

# Import data and training
from Image_Authenticity_prediction.main.data import (
    ImageAuthenticityDataset,
    IMAGENET_DATASET,
    DENSENET_DATASET,
    INCEPTIONV3_DATASET,
    BATCH_SIZE,
    NUM_WORKERS
)
from Image_Authenticity_prediction.main.train import (
    train_model,
    test_model,
    plot_loss_history
)
```

### From Within the Package

When working inside the package (e.g., in experiment files) use relative imports:

```python
# From experiment_one.py or experiment_two.py
from ..Models import (
    VGG16AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor
)
from ..Utils import GradCAM, FeatureMapsPruner
from ..Utils.cleanup import clear_gpu_memory, cleanup_model_and_data
from ..Utils.logger import info, warn, error, debug
from ..data import IMAGENET_DATASET, DENSENET_DATASET, BATCH_SIZE
from ..train import train_model, test_model
```

### Using the CLI

The CLI provides a convenient way to run common tasks without writing scripts:

```bash
# From the parent directory of Image_Authenticity_prediction
cd /path/to/parent/directory
python -m Image_Authenticity_prediction train --model vgg16 --epochs 50

# Evaluate a trained model
python -m Image_Authenticity_prediction evaluate \
    --model vgg16 \
    --weights Outputs/Experiment_1_variants/Weights/vgg16_best.pth

# Run experiments
python -m Image_Authenticity_prediction experiment-one --train --test
python -m Image_Authenticity_prediction experiment-two --models vgg16 resnet152
python -m Image_Authenticity_prediction experiment-three --strategy both
```

## 📤 Module Exports

### main.Models

All 7 model classes are exported from `main/Models/__init__.py` (6 actively used in experiments, 1 implemented but not used):

```python
from main.Models import (
    VGG16AuthenticityPredictor,        # VGG16 backbone
    VGG19AuthenticityPredictor,        # VGG19 backbone
    ResNet152AuthenticityPredictor,    # ResNet152 backbone
    DenseNet161AuthenticityPredictor,  # DenseNet161 backbone (requires 300x300 input)
    InceptionV3AuthenticityPredictor,  # InceptionV3 backbone (requires 299x299 input) - not used in experiments
    EfficientNetB3AuthenticityPredictor,  # EfficientNetB3 backbone
    BarlowTwinsAuthenticityPredictor   # BarlowTwins self-supervised ResNet50
)

# Note: InceptionV3 is implemented but excluded from Experiment 1 due to pruning incompatibility
```

**Model Features**:

- All models use transfer learning with pre-trained weights
- `freeze_backbone=True` (default): Only train the regression head
- `freeze_backbone=False`: Fine-tune the entire model
- All models return `(predictions, features)` tuple from `forward()`

### main.Utils

Exports explainability and pruning tools from `main/Utils/__init__.py`:

```python
from main.Utils import (
    GradCAM,                  # Gradient-weighted Class Activation Mapping
    MultiscalePixelMasking,   # Occlusion-based saliency maps
    FeatureMapsPruner         # Network compression via filter pruning
)
```

**Utility Features**:

- **GradCAM**: Generates visual explanations by highlighting important regions
- **MultiscalePixelMasking**: Creates saliency maps using multiscale occlusion
- **FeatureMapsPruner**: Computes filter importance and prunes networks

### main.data

Provides dataset classes and pre-configured data loaders:

```python
from main.data import (
    # Dataset class
    ImageAuthenticityDataset,

    # Pre-split datasets (train/val/test)
    IMAGENET_DATASET,              # 224x224 images (VGG, ResNet, EfficientNet, BarlowTwins)
    DENSENET_DATASET,              # 300x300 images (DenseNet161)
    INCEPTIONV3_DATASET,           # 300x300 images (same as DENSENET - InceptionV3 not used in experiments)

    # Transforms
    IMAGENET_TRANSFORM,            # Resize→CenterCrop(224)→Normalize
    DENSENET_TRANSFORM,            # Resize→CenterCrop(300)→Normalize

    # Configuration
    BATCH_SIZE,                    # Default: 64
    NUM_WORKERS,                   # Default: 20
    SEED                           # Random seed: 42
)
```

**Data Splitting**:

- Train: 70%, Val: 10%, Test: 20%
- Splits are deterministic (seeded) and consistent across all dataset variants
- Test set is immutable across experiments to ensure fair comparison

### main.train

Training and evaluation functions:

```python
from main.train import (
    train_model,           # Training with early stopping
    test_model,            # Evaluation with metrics
    plot_loss_history      # Plot training/validation curves
)
```

**Function Signatures**:

```python
# Training
train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cuda', patience=5)
# Returns: (best_model, history_dict)

# Testing
test_model(model, dataloader, criterion, device='cuda', return_rmse=False, return_additional_metrics=False)
# Returns: float (MSE or RMSE) or dict with {'mse', 'rmse', 'plcc', 'srcc', 'krcc', 'preds', 'labels'}

# Plotting
plot_loss_history(history_dict)
# Shows matplotlib plot with train/val loss curves
```

## 💡 Common Usage Patterns

### 1. Training a Model from Scratch

```python
import torch
from torch.utils.data import DataLoader
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE, NUM_WORKERS
from Image_Authenticity_prediction.main.train import train_model

# Initialize model
model = VGG16AuthenticityPredictor(freeze_backbone=True)

# Setup training
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create data loaders
train_loader = DataLoader(
    IMAGENET_DATASET['train'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    IMAGENET_DATASET['val'],
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

dataloaders = {'train': train_loader, 'val': val_loader}

# Train with early stopping
best_model, history = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=500,
    device='cuda',
    patience=15  # Stop if no improvement for 15 epochs
)

# Save best model
torch.save(best_model.state_dict(), 'Outputs/Experiment_1_variants/Weights/vgg16_best.pth')
```

### 2. Evaluating a Trained Model

```python
import torch
from torch.utils.data import DataLoader
from Image_Authenticity_prediction.main.Models import ResNet152AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE
from Image_Authenticity_prediction.main.train import test_model

# Load model
model = ResNet152AuthenticityPredictor(freeze_backbone=False)
model.load_state_dict(torch.load('Outputs/Experiment_1_variants/Weights/resnet152_exp1_orig.pth', weights_only=True))

# Create test loader
test_loader = DataLoader(IMAGENET_DATASET['test'], batch_size=BATCH_SIZE, shuffle=False)

# Evaluate with detailed metrics
criterion = torch.nn.MSELoss()
results = test_model(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    device='cuda',
    return_additional_metrics=True
)

# results contains: {'mse', 'rmse', 'plcc', 'srcc', 'krcc', 'preds', 'labels'}
print(f"RMSE: {results['rmse']:.4f}")
print(f"PLCC: {results['plcc']:.4f}")
print(f"SRCC: {results['srcc']:.4f}")
```

### 3. Generating GradCAM Visualizations

```python
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.Utils import GradCAM
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = VGG16AuthenticityPredictor()
model.load_state_dict(torch.load('Outputs/Experiment_1_variants/Weights/vgg16_best.pth', weights_only=True))
model.eval()

# Setup GradCAM
grad_cam = GradCAM(model, target_layer='features.28')  # Last conv layer of VGG16

# Load and preprocess image
image = Image.open('path/to/image.jpg').convert('RGB')
from Image_Authenticity_prediction.main.data import IMAGENET_TRANSFORM
image_tensor = IMAGENET_TRANSFORM(image).unsqueeze(0)

# Generate CAM
cam = grad_cam.generate_cam(image_tensor, device='cuda')

# Visualize
plt.imshow(cam, cmap='jet')
plt.colorbar()
plt.show()

# Clean up
grad_cam.cleanup()
```

### 4. Pruning a Model

```python
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.Utils import FeatureMapsPruner
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE
from Image_Authenticity_prediction.main.train import test_model
from torch.utils.data import DataLoader
import torch.nn as nn

# Load trained model
model = VGG16AuthenticityPredictor()
model.load_state_dict(torch.load('Outputs/Experiment_1_variants/Weights/vgg16_best.pth', weights_only=True))

# Create test loader
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
print(f"Filter importance scores: {scores}")

# Prune using greedy strategy
results = pruner.greedy_pruning(
    save_path='Outputs/Experiment_1_variants/Weights/vgg16_pruned.pth',
    threshold=0.5  # Remove 50% of filters
)

print(f"Original performance: {results['original_performance']}")
print(f"Pruned performance: {results['pruned_performance']}")
```

### 5. Using the CLI

```bash
# Train a model
python -m Image_Authenticity_prediction train \
    --model resnet152 \
    --epochs 500 \
    --patience 15 \
    --learning-rate 0.001 \
    --freeze-backbone

# Evaluate a model
python -m Image_Authenticity_prediction evaluate \
    --model resnet152 \
    --weights Outputs/Experiment_1_variants/Weights/resnet152_best.pth

# Run complete experiments
python -m Image_Authenticity_prediction experiment-one --train --prune --test
python -m Image_Authenticity_prediction experiment-two
python -m Image_Authenticity_prediction experiment-three --strategy stacking
```

## 🔧 Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError` when trying to import from the package

**Solutions**:

1. **Check your working directory**: Make sure you're in the correct location
2. **Add parent directory to path**:

   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```

3. **Verify `__init__.py` files exist** in all package directories
4. **Use absolute imports** from external scripts, **relative imports** from within the package

### Dataset Not Found

**Problem**: `FileNotFoundError: Annotations CSV not found` or image loading errors

**Solutions**:

1. **Verify dataset structure**:
   ```bash
   ls Image_Authenticity_prediction/Dataset/AIGCIQA2023/
   # Should show: real_images_annotations.csv and image files
   ```
2. **Check CSV paths**: The `ImageAuthenticityDataset` in [data.py](Image_Authenticity_prediction/main/data.py) expects paths relative to project root
3. **Verify CSV format**: Column 3 should contain image paths, column 1 should contain authenticity scores
4. **Check file permissions**: Ensure you have read access to the dataset directory

### CUDA/GPU Errors

**Problem**: CUDA out of memory or GPU not available

**Solutions**:

1. **Check GPU availability**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
   ```
2. **Reduce batch size**: Lower `BATCH_SIZE` in [data.py](Image_Authenticity_prediction/main/data.py) from 64 to 32 or 16
3. **Use CPU**: Set `device='cpu'` in training/evaluation calls (much slower)
4. **Clear GPU memory**:
   ```python
   from Image_Authenticity_prediction.main.Utils.cleanup import clear_gpu_memory
   clear_gpu_memory()
   ```
5. **Close other GPU-using processes**: Check with `nvidia-smi`

### Windows Performance Issues

**Problem**: Training is extremely slow on Windows, even with a powerful GPU

**Cause**: Linux/macOS use `fork()` to spawn workers (fast, copy-on-write memory sharing). Windows lacks `fork()` and uses `spawn`, which creates a new Python interpreter per worker, reimports all modules, and serializes the dataset for each — causing significant overhead.

See: [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/stable/data.html#platform-specific-behaviors)
**Solutions**:

1. **Reduce `num_workers` in config.yaml**:

   ```yaml
   # In Image_Authenticity_prediction/Configs/config.yaml
   data:
     num_workers: 2 # Use 2-4 on Windows instead of 20
   ```

2. **Recommended values**:

   - **Windows**: `num_workers: 2` to `num_workers: 4`
   - **Linux/macOS**: `num_workers: 20` (default) works well
   - **General rule**: Set `num_workers` ≤ number of CPU cores. Check with:
     ```python
     import os
     print(os.cpu_count())  # Your available CPU cores
     ```

### Model Loading Errors

**Problem**: `RuntimeError: Error(s) in loading state_dict` or shape mismatches

**Solutions**:

1. **Use `weights_only=True`** when loading (security best practice):
   ```python
   model.load_state_dict(torch.load('path/to/weights.pth', weights_only=True))
   ```
2. **Match model architecture**: Ensure the model class matches the saved weights
3. **Check `freeze_backbone` setting**: Load model with same setting used during training
4. **Verify file integrity**: Re-download or re-train if weights file is corrupted

### Experiment Output Not Found

**Problem**: Cannot find weights or results from experiments

**Solutions**:

1. **Check output directories** (created automatically):
   - Experiment 1: `Outputs/Experiment_1_variants/`
   - Experiment 2: `Outputs/Experiment_2_variants/`
   - Experiment 3: `Outputs/Experiment_3_ensemble/`
2. **Run the experiment first**: Outputs are only created after running experiments
3. **Check file naming**: Weights follow pattern `{model_name}_exp1*_*.pth` or `{model_name}_best.pth`

### Visualization/Plotting Issues

**Problem**: Plots not showing or saving incorrectly

**Solutions**:

1. **For Jupyter notebooks**: Use `%matplotlib inline` at the start
2. **For scripts**: Add `plt.show()` to display plots or `plt.savefig()` to save
3. **Check backend**: If headless environment, use `matplotlib.use('Agg')` before importing pyplot
4. **Install display dependencies**: Ensure `matplotlib` is properly installed

## ✅ Best Practices

### 1. Always Use Relative Imports Within the Package

```python
# ✅ Good (within package)
from ..Models import VGG16AuthenticityPredictor
from ..Utils import GradCAM
from ..data import IMAGENET_DATASET

# ❌ Avoid (within package)
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
```

### 2. Memory Management

```python
from Image_Authenticity_prediction.main.Utils.cleanup import clear_gpu_memory, cleanup_model_and_data

# After training/evaluation
cleanup_model_and_data(model, dataloaders, optimizer)
clear_gpu_memory()
```

### 3. Use the CLI for Standard Operations

The CLI handles paths automatically and is more convenient than writing scripts:

```bash
# Much easier than writing a training script
python -m Image_Authenticity_prediction train --model vgg16 --epochs 50
```

### 4. Leverage Pre-configured Datasets

The project provides pre-split, transformed datasets:

```python
# ✅ Good - use pre-configured datasets
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
train_loader = DataLoader(IMAGENET_DATASET['train'], batch_size=64)

# ❌ Avoid - manually creating splits
# This breaks reproducibility across experiments
```

### 5. Save Weights and Outputs Properly

```python
# Save model state_dict (preferred - smaller files)
torch.save(model.state_dict(), 'path/to/model.pth')

# Load with weights_only for security
model.load_state_dict(torch.load('path/to/model.pth', weights_only=True))

# Use descriptive names
torch.save(model.state_dict(), f'Outputs/Experiment_1_variants/Weights/{model_name}_exp1_orig.pth')
```

### 6. Structured Experiment Organization

Keep experiments in the `Experiments/` folder with clear naming:

- `experiment_one.py`: Training and pruning
- `experiment_two.py`: Explainability analysis
- `experiment_three.py`: Ensemble methods

### 7. Use the Logger

```python
from Image_Authenticity_prediction.main.Utils.logger import info, warn, error, debug

info("Starting training...")  # Blue, for general info
warn("Memory usage is high")  # Yellow, for warnings
error("Training failed!")     # Red, for errors
debug("Batch 10/100")         # Gray, for debug info
```

## 📚 Additional Resources

- **[README.md](README.md)**: Project overview and quick start
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Quick reference guide
- **[TODO.md](TODO.md)**: Current development tasks and known issues
- **[EXPERIMENT_1_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_1_TECHNICAL_REPORT.md)**: Detailed report on Experiment 1
- **[EXPERIMENT_2_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_2_TECHNICAL_REPORT.md)**: Detailed report on Experiment 2
- **[EXPERIMENT_3_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_3_TECHNICAL_REPORT.md)**: Detailed report on Experiment 3 (ensemble methods)

## 🆘 Getting Help

1. **Check this guide** for setup and import issues
2. **Review the README.md** for usage examples and available models
3. **Check TODO.md** for known limitations and planned features
4. **Examine experiment code** in `main/Experiments/` for real-world usage patterns
5. **Contact project maintainer**: [github.com/icaro-rdp](https://github.com/icaro-rdp)

## 🎯 Next Steps

After setting up the project:

1. **Verify dataset location**:

   ```bash
   ls Image_Authenticity_prediction/Dataset/AIGCIQA2023/
   ls Image_Authenticity_prediction/Dataset/Single_score/
   ```

2. **Test imports**:

   ```python
   from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
   from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
   print("✅ Imports successful!")
   ```

3. **Run a quick training test**:

   ```bash
   python -m Image_Authenticity_prediction train --model vgg16 --epochs 2
   ```

4. **Explore experiments**:
   ```bash
   python -m Image_Authenticity_prediction experiment-one --help
   ```

---

**Last Updated**: January 5, 2026  
**Maintainer**: Icaro Redepauolini ([github.com/icaro-rdp](https://github.com/icaro-rdp))
