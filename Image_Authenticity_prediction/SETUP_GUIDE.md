# Setup and Import Structure Guide

### ✅ Module Organization
- Created package hierarchy
- Added proper `__all__` exports
- Documented all import patterns

### ✅ CLI Integration
- Full command-line interface
- Argument parsing
- Model registry system
- Config file integration

## 📞 Support

1. Check `SETUP_GUIDE.md` for import problems
2. Review `README.md` for usage examples
3. Check `TODO.md` for known limitations
4. Contact project maintainer : github.com/icaro-rdp

## Module Organization

The Image Authenticity Prediction project follows a hierarchical module structure:

```
Image_Authenticity_prediction/
├── __main__.py              # CLI entry point
├── main/                    # Core package
│   ├── __init__.py         # Package initialization
│   ├── data.py             # Dataset definitions
│   ├── train.py            # Training utilities
│   ├── Models/             # Model architectures
│   │   ├── __init__.py
│   │   └── models.py
│   ├── Utils/              # Utility functions
│   │   ├── __init__.py
│   │   ├── explainability.py
│   │   └── pruning.py
│   └── Experiments/        # Experiment scripts
│       ├── __init__.py
│       ├── experiment_one.py
│       ├── experiment_two.py
│       └── experiment_three.py
```

## Import Patterns

### From External Scripts

When importing from outside the package directory:

```python
import sys
sys.path.insert(0, '/path/to/icaro_rdp_projects')

# Import models
from Image_Authenticity_prediction.main.Models import (
    VGG16AuthenticityPredictor,
    ResNet152AuthenticityPredictor
)

# Import utilities
from Image_Authenticity_prediction.main.Utils import GradCAM, FeatureMapsPruner

# Import data and training
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET
from Image_Authenticity_prediction.main.train import train_model
```

### From Within the Package

When working inside the package (e.g., in experiment files):

```python
# From experiment_one.py
from ..Models import VGG16AuthenticityPredictor
from ..Utils import GradCAM
from ..data import IMAGENET_DATASET
from ..train import train_model
```

### Using the CLI

```bash
# From the parent directory of Image_Authenticity_prediction
cd /path/to/icaro_rdp_projects
python -m Image_Authenticity_prediction train --model vgg16

# Or from anywhere with full path
python -m /path/to/icaro_rdp_projects/Image_Authenticity_prediction train --model vgg16
```

## Module Exports

### main.Models

```python
from main.Models import (
    BarlowTwinsAuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    InceptionV3AuthenticityPredictor
)
```

### main.Utils

```python
from main.Utils import (
    GradCAM,                  # For GradCAM visualization
    MultiscalePixelMasking,   # For occlusion saliency
    FeatureMapsPruner         # For network pruning
)
```

### main.data

```python
from main.data import (
    ImageAuthenticityDataset,      # Dataset class
    IMAGENET_DATASET,              # Dict with 'train' and 'test'
    DENSENET_DATASET,              # Dict with 'train' and 'test'
    INCEPTIONV3_DATASET,           # Dict with 'train' and 'test'
    IMAGENET_VISUALIZATION_DATASET,
    DENSENET_VISUALIZATION_DATASET,
    IMAGENET_TRANSFORM,            # Transform for standard models
    DENSENET_TRANSFORM,            # Transform for DenseNet/InceptionV3
    BATCH_SIZE,                    # Default: 64
    NUM_WORKERS                    # Default: 10
)
```

### main.train

```python
from main.train import (
    train_model,           # Main training function with early stopping
    test_model,            # Evaluation function
    plot_loss_history      # Plot training curves
)
```

## Common Usage Patterns

### 1. Training a Model

```python
import torch
from torch.utils.data import DataLoader
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.data import IMAGENET_DATASET, BATCH_SIZE
from Image_Authenticity_prediction.main.train import train_model

# Setup
model = VGG16AuthenticityPredictor(freeze_backbone=True)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Data loaders
train_loader = DataLoader(IMAGENET_DATASET['train'], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(IMAGENET_DATASET['test'], batch_size=BATCH_SIZE)

dataloaders = {'train': train_loader, 'val': val_loader}

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

# Save
torch.save(best_model.state_dict(), 'Weights/vgg16_best.pth')
```

### 2. Generating GradCAM

```python
from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
from Image_Authenticity_prediction.main.Utils import GradCAM
import torch

# Load model
model = VGG16AuthenticityPredictor()
model.load_state_dict(torch.load('Weights/vgg16_best.pth'))
model.eval()

# Setup GradCAM
grad_cam = GradCAM(model, target_layer=model.features[28])

# Generate CAM
image_tensor = torch.randn(1, 3, 224, 224)  # Your image
cam = grad_cam.generate_cam(image_tensor)

# Cleanup
grad_cam.cleanup()
```

### 3. Feature Map Pruning

```python
from Image_Authenticity_prediction.main.Utils import FeatureMapsPruner
from Image_Authenticity_prediction.main.train import test_model
import torch.nn as nn

# Initialize
pruner = FeatureMapsPruner(
    model=model,
    dataloader=test_loader,
    layer_name='features.0',
    criterion=nn.MSELoss(),
    eval_function=test_model,
    device=torch.device('cuda')
)

# Compute and prune
scores = pruner.compute_importance_scores()
results = pruner.greedy_pruning('Weights/pruned_model.pth')
```

## Directory Structure Requirements

Ensure these directories exist (will be created automatically if needed):

```
Image_Authenticity_prediction/
├── Weights/              # For saving model checkpoints
├── Ranking_arrays/       # For saving importance scores
└── Dataset/              # Dataset location
    └── AIGCIQA2023/
        ├── real_images_annotations.csv
        ├── 2_images_to_plot.csv
        └── [images]/
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

1. Check your working directory
2. Ensure you've added the parent directory to sys.path
3. Verify all `__init__.py` files exist

### CUDA Errors

If you get CUDA errors:

1. Check GPU availability: `torch.cuda.is_available()`
2. Set device to 'cpu' in config.yaml if needed
3. Reduce batch size if running out of memory

### Dataset Not Found

Ensure the dataset path is correct:
- The code expects: `Dataset/AIGCIQA2023/` relative to the workspace root
- Update paths in `main/data.py` if your structure differs

## Best Practices

1. **Always use relative imports within the package**
   ```python
   # Good
   from ..Models import VGG16AuthenticityPredictor
   
   # Avoid
   from Image_Authenticity_prediction.main.Models import VGG16AuthenticityPredictor
   ```

2. **Export only what's needed in `__init__.py`**
   - Keeps the API clean
   - Prevents circular imports

3. **Use the CLI for standard operations**
   - More convenient than writing scripts
   - Handles paths automatically

4. **Keep experiments in the Experiments folder**
   - Organized structure
   - Easy to track different approaches

## Next Steps

1. Review the TODO.md file for current tasks
2. Check the README.md for usage examples
3. Run a test training to verify setup:
   ```bash
   python -m Image_Authenticity_prediction train --model vgg16 --epochs 2
   ```
