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
from Models.Ensemble.utils import  train_model

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
    
    for bootstrap_idx in range(num_bootstrap_samples):
        bootstrap_indices = np.random.choice(dataset_size, size=dataset_size, replace=True)
        bootstrap_subset = Subset(original_dataset, bootstrap_indices)
        bootstrapped_datasets_with_indices.append((bootstrap_subset, bootstrap_indices))
        
        unique_indices_count = len(np.unique(bootstrap_indices))
        print(f"  Bootstrap sample {bootstrap_idx+1}/{num_bootstrap_samples} created: "
              f"{len(bootstrap_subset)} data points ({unique_indices_count} unique).")
        
    return bootstrapped_datasets_with_indices

def prepare_bagging_dataloaders(
    ensemble_models,
    model_names,
    imagenet_dataset,
    densenet_dataset,
    batch_size=64,
    num_workers=20
):
    """
    Prepares a dictionary of DataLoaders for bagging ensemble training.

    Args:
        ensemble_models (list): List of model instances.
        model_names (list): List of model names (str), must match ensemble_models order.
        imagenet_dataset (Dataset): The ImageNet training dataset.
        densenet_dataset (Dataset): The DenseNet-specific training dataset.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoaders.

    Returns:
        dict: Mapping from model name to DataLoader.

    Raises:
        ValueError: If ensemble_models and model_names have different lengths.
    """
    # Validate input consistency
    if len(ensemble_models) != len(model_names):
        raise ValueError(f"Mismatch: {len(ensemble_models)} models but {len(model_names)} names provided")
    
    num_models_in_ensemble = len(ensemble_models)
    
    # Count how many models need each dataset type
    # DenseNet161 and InceptionV3 both need the DenseNet dataset (different transformations)
    densenet_model_count = sum(1 for name in model_names if ("DenseNet" in name or "InceptionV3" in name))
    imagenet_model_count = num_models_in_ensemble - densenet_model_count
    
    print(f"\nPreparing dataloaders for {num_models_in_ensemble} models:")
    print(f"  - {imagenet_model_count} models using ImageNet dataset")
    print(f"  - {densenet_model_count} models using DenseNet dataset")
    
    # Create appropriate number of bootstrap samples for each dataset
    imagenet_bootstraps_info = []
    densenet_bootstraps_info = []
    
    if imagenet_model_count > 0:
        imagenet_bootstraps_info = create_bootstrap_datasets(imagenet_dataset, imagenet_model_count)
    
    if densenet_model_count > 0:
        densenet_bootstraps_info = create_bootstrap_datasets(densenet_dataset, densenet_model_count)

    # Assign bootstrap samples to models
    train_dataloaders = {}
    imagenet_bootstrap_idx = 0
    densenet_bootstrap_idx = 0
    
    for model_idx, model_name in enumerate(model_names):
        if "DenseNet" in model_name or "InceptionV3" in model_name:
            if densenet_bootstrap_idx >= len(densenet_bootstraps_info):
                raise RuntimeError(f"Not enough DenseNet bootstrap samples for model {model_name}")
            
            bootstrap_subset, bootstrap_indices = densenet_bootstraps_info[densenet_bootstrap_idx]
            densenet_bootstrap_idx += 1
            dataset_type = "DenseNet"
        else:
            if imagenet_bootstrap_idx >= len(imagenet_bootstraps_info):
                raise RuntimeError(f"Not enough ImageNet bootstrap samples for model {model_name}")
                
            bootstrap_subset, bootstrap_indices = imagenet_bootstraps_info[imagenet_bootstrap_idx]
            imagenet_bootstrap_idx += 1
            dataset_type = "ImageNet"
        
        model_dataloader = DataLoader(
            bootstrap_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        train_dataloaders[model_name] = model_dataloader
        
        # Provide detailed information about each dataloader
        unique_samples = len(np.unique(bootstrap_indices))
        total_samples = len(bootstrap_subset)
        print(f"  DataLoader for {model_name} ({dataset_type}): "
              f"{total_samples} samples ({unique_samples} unique), "
              f"batch_size={batch_size}, num_workers={num_workers}")

    return train_dataloaders


# Example usage
if __name__ == "__main__":
    
    # Initialize models with more descriptive variable names
    barlow_twins_model = BarlowTwinsAuthenticityPredictor()
    efficientnet_b3_model = EfficientNetB3AuthenticityPredictor()
    densenet_161_model = DenseNet161AuthenticityPredictor()
    resnet_152_model = ResNet152AuthenticityPredictor()
    inception_v3_model = InceptionV3AuthenticityPredictor()
    vgg_16_model = VGG16AuthenticityPredictor()
    vgg_19_model = VGG19AuthenticityPredictor()

    ensemble_models = [
        barlow_twins_model, 
        efficientnet_b3_model, 
        densenet_161_model, 
        resnet_152_model, 
        inception_v3_model, 
        vgg_16_model, 
        vgg_19_model
    ]

    model_names = [
        'BarlowTwins', 
        'EfficientNetB3', 
        'DenseNet161', 
        'ResNet152', 
        'InceptionV3', 
        'VGG16', 
        'VGG19'
    ]

    # Determine device and move models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    for model_idx, model in enumerate(ensemble_models):
        model.to(device)
        print(f"  {model_names[model_idx]} moved to {device}")

    # Prepare bagging dataloaders with error handling
    try:
        train_dataloaders = prepare_bagging_dataloaders(
            ensemble_models,
            model_names,
            IMAGENET_DATASET['train'],
            DENSENET_DATASET['train'],
            batch_size=64,
            num_workers=20
        )
        
        print(f"\nSuccessfully prepared {len(train_dataloaders)} dataloaders for ensemble training.")
        
    except Exception as error:
        print(f"Error preparing dataloaders: {error}")
        raise


    ### TRAINING INDIVIDUAL MODELS ###
    criterion = torch.nn.MSELoss()  # Or your preferred loss function
    NUMBER_OF_EPOCHS = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print("\nStarting training for individual ensemble members...")

    for model_name_key, current_train_dataloader in train_dataloaders.items():
        
        current_model_instance = next((m for m, name in zip(ensemble_models, model_names) if name == model_name_key), None)
        
        if current_model_instance is None:
            print(f"  Model '{model_name_key}' not found in ensemble_models list. Skipping.")
            continue
            
        optimizer = torch.optim.Adam(current_model_instance.parameters(), lr=0.001)
        
        print(f"\nTraining model: '{model_name_key}' for {NUMBER_OF_EPOCHS} epochs...")
        
        try:
            trained_model, training_stats = train_model(
                model=current_model_instance,
                train_dataloader=current_train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=NUMBER_OF_EPOCHS,
                device=device,
                save_model=True,
                path_to_save="Models/Ensemble/Weights/",
                model_name=model_name_key 
            )
            print(f"  Model '{model_name_key}' training completed.")
    

        except Exception as error:
            print(f"  Error training model '{model_name_key}': {error}")
            import traceback
            traceback.print_exc() 