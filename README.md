# Image Authenticity Prediction

A comprehensive deep learning framework for predicting image authenticity and understanding model behavior through explainability methods. This project explores how different CNN architectures perceive AI-generated versus real images, with a focus on human-alignment and interpretable AI.

## 📋 Table of Contents

- [Image Authenticity Prediction](#image-authenticity-prediction)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
  - [📚 Documentation](#-documentation)
  - [🚀 Quick Start](#-quick-start)
    - [Installation](#installation)
    - [Dataset Setup](#dataset-setup)
  - [💡 Usage](#-usage)
    - [Basic Commands](#basic-commands)
    - [Python API](#python-api)
  - [🤖 Available Models](#-available-models)
  - [📊 Dataset](#-dataset)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## 🔍 Overview

This research project investigates image authenticity prediction using multiple deep learning architectures, analyzing how different models perceive and evaluate AI-generated content. The work focuses on three key areas:

**1. Model Performance & Architecture Comparison**

- Evaluates 7 state-of-the-art CNN architectures on authenticity prediction
- Compares traditional supervised (ImageNet) vs. self-supervised (BarlowTwins) pretraining
- Analyzes performance across different network depths and architectural designs

**2. Model Explainability & Human Alignment**

- Implements GradCAM and Multiscale Pixel Masking for visual explanations
- Compares what different models "look at" when judging authenticity
- Studies alignment between model attention and human perception

**3. Network Optimization & Feature Analysis**

- Identifies and removes redundant or harmful features through pruning
- Analyzes feature importance across different layers
- Investigates model efficiency and compression possibilities

## 📚 Documentation

Complete documentation for this project:

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Setup instructions, dataset structure, import patterns, and troubleshooting
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project architecture and module organization
- **[DATASET_EXCLUSIONS.md](Dataset/AIGCIQA2023/DATASET_EXCLUSIONS.md)** - Dataset filtering criteria, prompt taxonomy, and reproduction manifest
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference and common operations
- **[TODO.md](TODO.md)** - Development roadmap and known issues
- **Experiment Reports**:
  - [EXPERIMENT_1_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_1_TECHNICAL_REPORT.md) - Training, pruning, and evaluation
  - [EXPERIMENT_2_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_2_TECHNICAL_REPORT.md) - Explainability methods comparison
  - [EXPERIMENT_3_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_3_TECHNICAL_REPORT.md) - Ensemble learning strategies

**New to the project?** Start with [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete setup instructions.

## 🚀 Quick Start

### Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install pandas pillow numpy matplotlib tqdm scipy scikit-learn scikit-image seaborn opencv-python
```

### Dataset Setup & Exclusions

The dataset annotations and exclusion manifests are tracked directly in git under `Image_Authenticity_prediction/Dataset/AIGCIQA2023/`. Large image assets (`Image/`) are git-ignored.

- **Tracked Annotations**: `Image_Authenticity_prediction/Dataset/AIGCIQA2023/real_images_annotations.csv` (1,368 retained images across 57 prompts and 6 models).
- **Full Exclusion Rationale & Manifest**: See [`DATASET_EXCLUSIONS.md`](Image_Authenticity_prediction/Dataset/AIGCIQA2023/DATASET_EXCLUSIONS.md) and [`prompt_exclusion_manifest.csv`](Image_Authenticity_prediction/Dataset/AIGCIQA2023/prompt_exclusion_manifest.csv) for the exact prompt-level criteria behind the 43 excluded prompts (1,032 images).
- **Reproduction Test**: Verify dataset partitioning and consistency with `uv run pytest tests/test_dataset_exclusions.py`.

To run visual evaluations, place the raw AIGCIQA2023 images into:
`Image_Authenticity_prediction/Dataset/AIGCIQA2023/Image/allimg/`

See [SETUP_GUIDE.md#git-ignored-directories](SETUP_GUIDE.md#-git-ignored-directories) for additional setup details.

## 💡 Usage

### Basic Commands

```bash
# Train a model
python -m Image_Authenticity_prediction train --model vgg16 --epochs 50

# Evaluate a model
python -m Image_Authenticity_prediction evaluate --model vgg16 --weights path/to/weights.pth

# Run complete experiments
python -m Image_Authenticity_prediction experiment-one --train --prune --test
python -m Image_Authenticity_prediction experiment-two --xai-methods both
python -m Image_Authenticity_prediction experiment-three --strategy both

# generate ensemble heatmaps for bagging strategy (use --heatmaps flag)
python -m Image_Authenticity_prediction experiment-three --strategy bagging --heatmaps
```

### Python API

```python
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
from Image_Authenticity_prediction.main.train import train_model

# Initialize and train
model = VGG16AuthenticityPredictor(freeze_backbone=True)
# ... training code
```

**For detailed usage, see:**

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - All commands and options
- [EXPERIMENT_1_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_1_TECHNICAL_REPORT.md) - Experiment 1 details
- [EXPERIMENT_2_TECHNICAL_REPORT.md](Image_Authenticity_prediction/main/Experiments/EXPERIMENT_2_TECHNICAL_REPORT.md) - Experiment 2 details

## 🤖 Available Models

All models use transfer learning with pretrained weights and custom regression heads for authenticity score prediction.

| Model               | Input Size | Status                       |
| ------------------- | ---------- | ---------------------------- |
| **VGG16**           | 224×224    | ✅ Active                    |
| **VGG19**           | 224×224    | ✅ Active                    |
| **ResNet-152**      | 224×224    | ✅ Active                    |
| **DenseNet-161**    | 300×300    | ✅ Active                    |
| **InceptionV3**     | 299×299    | ⚠️ Not used in experiments\* |
| **EfficientNet-B3** | 300×300    | ✅ Active                    |
| **BarlowTwins**     | 224×224    | ✅ Active                    |

\*InceptionV3 is implemented but excluded from Experiment 1 due to incompatibility with the current pruning method.

## 📊 Dataset

Uses the **AIGCIQA2023** dataset with Mean Opinion Scores (MOS) for image authenticity, quality, and correspondence.

To focus strictly on natural image realism rather than stylized or impossible content, 43 prompts (1,032 images) were excluded at the prompt level, retaining 57 prompts (1,368 images) of natural scenes and real-world entities.

See **[DATASET_EXCLUSIONS.md](Dataset/AIGCIQA2023/DATASET_EXCLUSIONS.md)** for complete exclusion criteria, taxonomy, and reproduction manifests.

**Directory Structure:**

```
Dataset/
├── AIGCIQA2023/
│   ├── DATASET_EXCLUSIONS.md          # Exclusion rationale and reproduction guide
│   ├── prompt_exclusion_manifest.csv  # Complete 100-prompt status and criteria
│   ├── real_images_annotations.csv    # 1,368 retained image annotations (tracked in git)
│   ├── removed_images_details.csv     # Full metadata of all 1,032 excluded images
│   ├── removed_images_summary.csv     # Aggregated category/challenge counts
│   ├── removed_images.txt             # Paths of all 1,032 excluded images
│   └── Image/                         # Image files (gitignored to avoid syncing large media)
│       └── allimg/ (0.png, ...)
└── Single_score/                      # Individual participant scores (25 CSVs)
```

## 🤝 Contributing

Research project by Icaro Re Depaolini as part of thesis work at CiMEC, University of Trento.

For questions or contributions, please contact the project maintainer via [GitHub](https://github.com/icaro-rdp/Internship_project_2025).

## 📄 License

Academic research project. Contact authors for usage permissions.

## 🙏 Acknowledgments

- Pretrained models from PyTorch and torchvision
- BarlowTwins implementation from Facebook Research
- AIGCIQA2023 dataset authors
- CiMEC, University of Trento

---

**Author**: Icaro Re Depaolini  
**Institution**: CiMEC, University of Trento  
**Last Updated**: Feb 2026
