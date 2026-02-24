# Project Structure Documentation

A comprehensive overview of the Image Authenticity Prediction project structure, showing all directories, files, and their purposes.

## 📁 Complete Directory Structure

```
Image_Authenticity_prediction/
├── __main__.py                    # CLI entry point with commands
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and quick start
├── SETUP_GUIDE.md                 # Detailed setup and import guide
├── QUICK_REFERENCE.md             # Command and import cheat sheet
├── PROJECT_STRUCTURE.md           # This file
├── TODO.md                        # Development tasks and known issues
│
├── Dataset/                       # 🚫 GITIGNORED - User must create
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
├── Outputs/                       # 🚫 GITIGNORED - Auto-generated
│   ├── Experiment_1_variants/    # From experiment_one.py
│   │   ├── Weights/              # Trained model weights (.pth)
│   │   ├── Ranking_arrays/       # Feature importance (.npy)
│   │   ├── Ranking_Plots/        # Importance visualizations
│   │   ├── Training_Plots/       # Loss curves
│   │   ├── Training_History/     # Epoch metrics (JSON)
│   │   └── Test_Results/         # Performance metrics
│   ├── Experiment_2_variants/    # From experiment_two.py
│   │   ├── XAI_Maps/
│   │   │   ├── GradCAM/          # GradCAM saliency maps
│   │   │   └── Multiscale_Pixel_Masking/  # MPM saliency maps
│   │   └── Plots/                # Comparison visualizations
│   └── Experiment_3_ensemble/    # From experiment_three.py (WIP)
│       ├── Weights/Stacking/     # Ensemble model weights
│       └── Results/              # Ensemble metrics
│
└── main/                          # ✅ Core Python package
    ├── __init__.py               # Package initialization
    ├── data.py                   # Dataset class and data loaders
    ├── train.py                  # Training/testing functions
    │
    ├── Models/                   # ✅ Model architectures
    │   ├── __init__.py          # Exports all 7 model classes
    │   └── models.py            # Model definitions
    │
    ├── Utils/                    # ✅ Utility modules
    │   ├── __init__.py          # Exports GradCAM, MPM, FeatureMapsPruner
    │   ├── cleanup.py           # Memory management utilities
    │   ├── comparisons.py       # Model comparison metrics
    │   ├── explainability.py    # GradCAM & Multiscale Pixel Masking
    │   ├── logger.py            # Colored console logging
    │   ├── normalization.py     # Data normalization functions
    │   ├── pruning.py           # Network compression tools
    │   └── visualization.py     # Plotting and visualization
    │
    └── Experiments/              # ✅ Research experiments
        ├── __init__.py          # Package initialization
        ├── experiment_one.py    # ✅ Training, pruning, testing
        ├── experiment_two.py    # ✅ XAI methods and comparisons
        ├── experiment_three.py  # ✅ Ensemble strategies
        ├── analysis_consistency_vs_accuracy.py
        ├── noise_ceiling_analysis.ipynb
        ├── EXPERIMENT_1_TECHNICAL_REPORT.md
        ├── EXPERIMENT_2_TECHNICAL_REPORT.md
        ├── EXPERIMENT_3_TECHNICAL_REPORT.md
        └── test/                # ✅ Test files
            ├── ...
```

**Legend**:

- ✅ = Implemented and working
- 🚧 = Work in Progress
- 🚫 = Git-ignored (not in version control)

## 📋 File Purposes

### Root Level Files

- **`__main__.py`**: CLI entry point enabling `python -m Image_Authenticity_prediction` commands. Handles training, evaluation, and experiment execution.

- **`requirements.txt`**: Lists all Python dependencies. Install with `pip install -r requirements.txt`.

- **`README.md`**: Project overview with installation instructions, usage examples, and available features.

- **`SETUP_GUIDE.md`**: Comprehensive guide covering project structure, git-ignored directories, output generation, module organization, import patterns, and troubleshooting.

- **`QUICK_REFERENCE.md`**: Quick reference for common CLI commands, imports, and model configurations.

- **`PROJECT_STRUCTURE.md`**: This file - detailed project structure documentation.

- **`TODO.md`**: Development tasks, known issues, and planned features.

### Core Package (`main/`)

#### Data Module (`data.py`)

**Purpose**: Dataset definition and data loading

**Key Components**:

- `ImageAuthenticityDataset`: Custom PyTorch Dataset class for loading images and authenticity scores from CSV
- Pre-configured datasets: `IMAGENET_DATASET`, `DENSENET_DATASET`, `INCEPTIONV3_DATASET`
- Transforms: `IMAGENET_TRANSFORM` (224×224), `DENSENET_TRANSFORM` (300×300)
- Data splits: 70% train, 10% val, 20% test (deterministic, seeded at 42)
- Constants: `BATCH_SIZE=64`, `NUM_WORKERS=20`

**Dataset Structure**:

- Expects CSV at `Dataset/AIGCIQA2023/real_images_annotations.csv`
- CSV column 1: Authenticity score (float)
- CSV column 3: Image path (relative to project root)

#### Training Module (`train.py`)

**Purpose**: Training and evaluation utilities

**Key Functions**:

- `train_model()`: Trains with early stopping, returns best model and history
  - Parameters: model, dataloaders, criterion, optimizer, num_epochs, device, patience
  - Returns: (best_model, history_dict)
- `test_model()`: Evaluates model on test set
  - Can return MSE, RMSE, or detailed metrics (PLCC, SRCC, KRCC)
  - Parameters: model, dataloader, criterion, device, return_rmse, return_additional_metrics
- `plot_loss_history()`: Visualizes training/validation loss curves

### Models Package (`main/Models/`)

**Purpose**: CNN architectures for authenticity prediction

**Files**:

- `__init__.py`: Exports all 7 model classes (6 actively used + 1 not used in experiments)
- `models.py`: Contains model definitions

**Available Models** (all inherit from `nn.Module`):

**Actively Used in Experiments**:

1. **VGG16AuthenticityPredictor**: VGG16 backbone + regression head
2. **VGG19AuthenticityPredictor**: VGG19 backbone + regression head
3. **ResNet152AuthenticityPredictor**: ResNet152 backbone + regression head
4. **DenseNet161AuthenticityPredictor**: DenseNet161 backbone + regression head (requires 300×300 input)
5. **EfficientNetB3AuthenticityPredictor**: EfficientNetB3 backbone + regression head
6. **BarlowTwinsAuthenticityPredictor**: BarlowTwins self-supervised ResNet50 + regression head

**Implemented but Not Used**: 7. **InceptionV3AuthenticityPredictor**: InceptionV3 backbone + regression head (requires 299×299 input) - _Excluded from Experiment 1 due to pruning incompatibility_

**Common Features**:

- Transfer learning with pre-trained weights
- `freeze_backbone` parameter (default=True): Only train regression head
- All models return `(predictions, features)` tuple
- Regression heads: Multi-layer with ReLU, Dropout, final Linear(→1)

### Utilities Package (`main/Utils/`)

**Purpose**: Helper functions and tools

#### `__init__.py`

Exports: `GradCAM`, `MultiscalePixelMasking`, `FeatureMapsPruner`

#### `cleanup.py`

**Purpose**: Memory management

- `clear_gpu_memory()`: Clears CUDA cache and runs garbage collection
- `cleanup_model_and_data()`: Properly deletes models, dataloaders, optimizers

#### `comparisons.py`

**Purpose**: Model comparison metrics

- Functions for comparing model outputs
- Computes similarity metrics between heatmaps
- Type definitions: `MetricSummary`, `ComparisonResults`

#### `explainability.py`

**Purpose**: Explainability methods (XAI)

- `ModelsExplainer`: Abstract base class for saliency methods
- `GradCAM`: Gradient-weighted Class Activation Mapping
  - Generates visual explanations highlighting important regions
  - Usage: `GradCAM(model, target_layer='features.28')`
- `MultiscalePixelMasking`: Occlusion-based saliency maps
  - Creates saliency maps using multiscale occlusion
  - Usage: `MultiscalePixelMasking(model, sigma=[3, 17, 65])`

#### `logger.py`

**Purpose**: Colored console logging

- `ColoredFormatter`: Custom formatter for colored output
- Functions: `info()`, `warn()`, `error()`, `debug()`, `set_level()`
- Color-coded: Blue (info), Yellow (warn), Red (error), Gray (debug)

#### `normalization.py`

**Purpose**: Data normalization utilities

#### `pruning.py`

**Purpose**: Network compression via filter pruning

- `FeatureMapsPruner`: Main pruning class
  - Computes filter importance scores
  - Performs greedy pruning to remove less important filters
  - Saves importance rankings to `.npy` files
  - Caches scores for efficiency

#### `visualization.py`

**Purpose**: Plotting and visualization

- `visualize_similarity_matrix()`: Creates heatmap of similarity between models
- `visualize_similarity_distribution()`: Distribution plots for similarity metrics
- `visualize_violin_distribution()`: Violin plots for metric distributions
- Functions for plotting heatmaps, overlays, and comparisons

### Experiments Package (`main/Experiments/`)

**Purpose**: Research experiment scripts

#### `experiment_one.py` ✅

**Purpose**: Complete training, pruning, and testing pipeline

**What it does**:

1. Trains all 6 models (VGG16, VGG19, ResNet152, DenseNet161, EfficientNetB3, BarlowTwins)
2. Computes feature importance scores for each model
3. Performs greedy pruning with multiple thresholds
4. Tests all original and pruned models on test set
5. Saves results to `Outputs/Experiment_1_variants/`

**Outputs**:

- `Weights/`: Model `.pth` files (original and pruned)
- `Ranking_arrays/`: Feature importance `.npy` files
- `Ranking_Plots/`: Importance visualizations
- `Training_Plots/`: Loss curves
- `Training_History/`: JSON files with metrics
- `Test_Results/`: Performance on test set

**Configuration**:

- Max epochs: 500, Patience: 15
- Learning rate: 0.001
- Freeze backbone: True
- Pruning methods: Greedy

#### `experiment_two.py` ✅

**Purpose**: XAI method generation and comparison

**What it does**:

1. Loads trained models from Experiment 1
2. Generates GradCAM and Multiscale Pixel Masking saliency maps
3. Compares maps across models and variants
4. Creates similarity matrices and distribution plots
5. Saves to `Outputs/Experiment_2_variants/`

**Outputs**:

- `XAI_Maps/GradCAM/`: GradCAM heatmaps for each model
- `XAI_Maps/Multiscale_Pixel_Masking/`: MPM heatmaps
- `Plots/`: Similarity matrices, distributions, violin plots

**Configuration**:

- Sigma scales: [3, 17, 65]
- Mask value: 0
- Pixel batch: 256

#### `experiment_three.py` ⚠️

**Purpose**: Ensemble learning strategies (Work in Progress)

**What it does**:

1. Implements bagging and stacking ensemble methods
2. Loads base models from Experiment 1
3. Trains meta-learners on base model predictions
4. Evaluates ensemble performance

**Outputs**:

- `Outputs/Experiment_3_ensemble/Weights/Stacking/`
- `Outputs/Experiment_3_ensemble/Results/`

**Status**: Work in Progress

#### Other Files

- **`analysis_consistency_vs_accuracy.py`**: Analysis script comparing consistency and accuracy
- **`noise_ceiling_analysis.ipynb`**: Jupyter notebook for noise ceiling analysis
- **`EXPERIMENT_1_TECHNICAL_REPORT.md`**: Detailed report on Experiment 1 results
- **`EXPERIMENT_2_TECHNICAL_REPORT.md`**: Detailed report on Experiment 2 results
- **`EXPERIMENT_3_TECHNICAL_REPORT.md`**: Detailed report on Experiment 3 results (ensemble methods)

#### Test Suite (`test/`)

- **`test_experiment_one_minimal.py`**: Minimal test for experiment one pipeline
- **`test_imports.py`**: Tests import system
- **`test_models_performances.py`**: Tests model performance
- **`test_models_structure.py`**: Tests model architecture definitions

## 🚫 Git-Ignored Items

The `.gitignore` file excludes the following:

### 1. Dataset Directory

**Pattern**: `Dataset/`, `Image_Authenticity_prediction/Dataset`, `*.pkl`, `*.h5`

**Why**: Large image files (GBs) should not be in version control

**User Action Required**: Create `Dataset/AIGCIQA2023/` and `Dataset/Single_score/` and add dataset files locally

### 2. Output Directories

**Patterns**:

- `*/main/Experiments/Outputs/*`
- `Image_Authenticity_prediction/main/Experiments/Outputs`
- `Models/*/Weights`
- `Models/*/Ranking_arrays`
- `Models/*/saliency_experiment_outputs/*`

**Why**: Generated files are large, experiment-specific, and reproducible

**User Action Required**: None - automatically created by code

### 3. Python Artifacts

**Patterns**: `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `build/`, `dist/`

**Why**: Standard Python build artifacts

### 4. Virtual Environments

**Patterns**: `venv/`, `env/`, `ENV/`

**Why**: Environment-specific files

### 5. IDE Files

**Patterns**: `.idea/`, `.vscode/`, `.DS_Store`, `*.swp`, `*.swo`

**Why**: Editor-specific settings

### 6. Jupyter Checkpoints

**Pattern**: `.ipynb_checkpoints`, `*/.ipynb_checkpoints/*`

**Why**: Jupyter notebook metadata

### 7. Logs and Temporary Files

**Patterns**: `*.log`, `/tmp`

**Why**: Runtime artifacts

## 📦 Module System

### Import Patterns

**External (from notebooks or scripts)**:

```python
import sys
sys.path.insert(0, '/path/to/parent/of/Image_Authenticity_prediction')

from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.Utils import GradCAM
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
```

**Internal (from within package)**:

```python
from ..Models import VGG16AuthenticityPredictor
from ..Utils import GradCAM
from ..data import IMAGENET_DATASET
```

### Exported Classes and Functions

**From `main.Models`**:

```python
VGG16AuthenticityPredictor, VGG19AuthenticityPredictor,
ResNet152AuthenticityPredictor, DenseNet161AuthenticityPredictor,
InceptionV3AuthenticityPredictor, EfficientNetB3AuthenticityPredictor,
BarlowTwinsAuthenticityPredictor
```

**From `main.Utils`**:

```python
GradCAM, MultiscalePixelMasking, FeatureMapsPruner
```

**From `main.data`**:

```python
ImageAuthenticityDataset, IMAGENET_DATASET, DENSENET_DATASET,
INCEPTIONV3_DATASET, IMAGENET_TRANSFORM, DENSENET_TRANSFORM,
BATCH_SIZE, NUM_WORKERS, SEED
```

**From `main.train`**:

```python
train_model, test_model, plot_loss_history
```

## 🎯 Usage Workflows

### 1. Train a Model

```bash
python -m Image_Authenticity_prediction train --model vgg16 --epochs 50 --patience 15
```

### 2. Evaluate a Model

```bash
python -m Image_Authenticity_prediction evaluate --model vgg16 --weights Outputs/Experiment_1_variants/Weights/vgg16_best.pth
```

### 3. Run Complete Experiment 1

```bash
python -m Image_Authenticity_prediction experiment-one --train --prune --test
```

### 4. Generate XAI Maps

```bash
python -m Image_Authenticity_prediction experiment-two --xai-methods both
```

### 5. Compare XAI Maps

```bash
python -m Image_Authenticity_prediction experiment-two --comparison-only
```

### 6. Run Ensemble Strategies

```bash
python -m Image_Authenticity_prediction experiment-three --strategy both
```

## 📚 Documentation Structure

- **README.md**: Start here - project overview and quick start
- **SETUP_GUIDE.md**: Detailed setup, imports, outputs, and troubleshooting
- **QUICK_REFERENCE.md**: Quick command and import cheat sheet
- **PROJECT_STRUCTURE.md**: This file - complete structure documentation
- **TODO.md**: Current tasks and known issues
- **EXPERIMENT\_\*\_TECHNICAL_REPORT.md**: Detailed experiment reports

## 🔄 Data Flow

1. **Dataset** (`Dataset/AIGCIQA2023/`) → `ImageAuthenticityDataset` → DataLoader
2. **DataLoader** → `train_model()` → Best model weights
3. **Weights** → Model → `test_model()` → Performance metrics
4. **Model + Weights** → `FeatureMapsPruner` → Pruned model + Importance scores
5. **Model + Weights** → `GradCAM`/`MultiscalePixelMasking` → Saliency maps
6. **Saliency maps** → Comparison functions → Similarity metrics and plots

## ✨ Key Design Principles

1. **Separation of Concerns**: Models, utilities, data, and experiments are separate
2. **Reproducibility**: Seeded random splits, deterministic test sets
3. **Modularity**: Each component can be used independently
4. **Git-friendly**: Large files (data, outputs) are gitignored
5. **CLI + Python API**: Use via command line or import as library
6. **Type Safety**: Type hints used throughout (where applicable)
7. **Documentation**: Comprehensive docs at multiple levels

## 📞 Related Documentation

- Setup instructions: `SETUP_GUIDE.md`
- Quick commands: `QUICK_REFERENCE.md`
- Project overview: `README.md`
- Development tasks: `TODO.md`
- Experiment results: `main/Experiments/EXPERIMENT_*_TECHNICAL_REPORT.md`

---

**Last Updated**: January 5, 2026  
**Maintainer**: Icaro Redepaolini ([github.com/icaro-rdp](https://github.com/icaro-rdp))
