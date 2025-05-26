import torch
import torch.nn as nn
from torch.utils.data import  DataLoader

# Import your existing models and utilities
from Models.Ensemble.models import (
    BarlowTwinsAuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    InceptionV3AuthenticityPredictor,
)
from Models.Ensemble.dataset import IMAGENET_DATASET, DENSENET_DATASET
from Models.Ensemble.utils import test_model


# Helper function to load a model and its weights
def load_model_with_weights(model_class, weights_path, device='cuda'):
    model = model_class()
    # Load weights with proper device mapping
    if device == 'cuda' and torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model = model.to('cpu')
    return model

# List of (model class, weights path) tuples
model_configs = [
    (BarlowTwinsAuthenticityPredictor, 'Models/BarlowTwins/Weights/BarlowTwins_real_authenticity_finetuned.pth'),
    (EfficientNetB3AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/EfficientNetB3_weights.pth'),
    (DenseNet161AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/DenseNet161_weights.pth'),
    (ResNet152AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/ResNet152_weights.pth'),
    (VGG16AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/VGG16_weights.pth'),
    (VGG19AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/VGG19_weights.pth'),
    (InceptionV3AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/InceptionV3_weights.pth'),
]

# Load all models with their weights
models = []
for model_class, weights_path in model_configs:
    model = load_model_with_weights(model_class, weights_path)
    models.append(model)
# Create a dictionary to hold DataLoaders for each model
test_dataloaders = {
    'BarlowTwinsAuthenticityPredictor': DataLoader(IMAGENET_DATASET['test'], batch_size=64, shuffle=False, num_workers=20),
    'EfficientNetB3AuthenticityPredictor': DataLoader(IMAGENET_DATASET['test'], batch_size=64, shuffle=False, num_workers=20),
    'DenseNet161AuthenticityPredictor': DataLoader(DENSENET_DATASET['test'], batch_size=64, shuffle=False, num_workers=20),
    'ResNet152AuthenticityPredictor': DataLoader(IMAGENET_DATASET['test'], batch_size=64, shuffle=False, num_workers=20),
    'InceptionV3AuthenticityPredictor': DataLoader(DENSENET_DATASET['test'], batch_size=64, shuffle=False, num_workers=20),
    'VGG16AuthenticityPredictor': DataLoader(IMAGENET_DATASET['test'], batch_size=64, shuffle=False, num_workers=20),
    'VGG19AuthenticityPredictor': DataLoader(IMAGENET_DATASET['test'], batch_size=64, shuffle=False, num_workers=20)
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test each model and collect predictions
predictions = {}
criterion = nn.MSELoss()
for model_idx, model in enumerate(models):
    model_name = model_configs[model_idx][0].__name__
    print(f"\nTesting {model_name} on {device}...")
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Test the model and collect predictions
    predictions[model_name] = test_model(
        model=model,
        test_dataloader=test_dataloaders[model_name],
        criterion=criterion,
        device=device
    )
    print(f"{model_name} predictions collected.")

