import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

# These are your direct imports. The code below assumes they work as expected.
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

# Initialize models
barlow_model = BarlowTwinsAuthenticityPredictor()
efficient_model = EfficientNetB3AuthenticityPredictor()
densenet_model = DenseNet161AuthenticityPredictor()
resnet_model = ResNet152AuthenticityPredictor()
inception_model = InceptionV3AuthenticityPredictor()
vgg16_model = VGG16AuthenticityPredictor()
vgg19_model = VGG19AuthenticityPredictor()

ensemble_models = [barlow_model, efficient_model, densenet_model, resnet_model, inception_model, vgg16_model, vgg19_model]

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model in ensemble_models:
    model.to(device)

## 1. Bootstrap the Datasets
def create_bootstrap_datasets(original_dataset: Dataset, num_bootstrap_samples: int):
    """
    Creates bootstrap samples (as PyTorch Subsets) from the original dataset.
    Each bootstrap sample is created by sampling with replacement and has the same size
    as the original dataset.

    Args:
        original_dataset (Dataset): The original PyTorch Dataset.
        num_bootstrap_samples (int): The number of bootstrap samples to create.

    Returns:
        list: A list of tuples, where each tuple contains:
              (Subset object for the bootstrap sample, numpy array of bootstrap indices)
    """
    dataset_size = len(original_dataset)
    bootstrapped_datasets_with_indices = []
    
    # Attempt to get a dataset name for print statements, default if not available
    dataset_name = getattr(original_dataset, 'name', original_dataset.__class__.__name__)

    print(f"\nCreating {num_bootstrap_samples} bootstrap samples from '{dataset_name}' (size: {dataset_size}):")
    for i in range(num_bootstrap_samples):
        bootstrap_indices = np.random.choice(dataset_size, size=dataset_size, replace=True)
        bootstrap_subset = Subset(original_dataset, bootstrap_indices)
        bootstrapped_datasets_with_indices.append((bootstrap_subset, bootstrap_indices))
        
        unique_indices_count = len(np.unique(bootstrap_indices))
        print(f"  Bootstrap sample {i+1}/{num_bootstrap_samples} created: {len(bootstrap_subset)} data points ({unique_indices_count} unique).")
        
    return bootstrapped_datasets_with_indices

# The number of bootstrap samples will match the number of models in the ensemble.
num_models_in_ensemble = len(ensemble_models)


imagenet_bootstraps_info = create_bootstrap_datasets(IMAGENET_DATASET['train'], num_models_in_ensemble)




