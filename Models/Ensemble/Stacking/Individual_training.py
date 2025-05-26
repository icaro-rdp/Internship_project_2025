import torch
from torch.utils.data import DataLoader

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

    train_dataloaders = {
        'BarlowTwins': DataLoader(IMAGENET_DATASET['train'], batch_size=64, shuffle=True, num_workers=20),
        'EfficientNetB3': DataLoader(IMAGENET_DATASET['train'], batch_size=64, shuffle=True, num_workers=20),
        'DenseNet161': DataLoader(DENSENET_DATASET['train'], batch_size=64, shuffle=True, num_workers=20),
        'ResNet152': DataLoader(IMAGENET_DATASET['train'], batch_size=64, shuffle=True, num_workers=20),
        'InceptionV3': DataLoader(IMAGENET_DATASET['train'], batch_size=64, shuffle=True, num_workers=20),
        'VGG16': DataLoader(IMAGENET_DATASET['train'], batch_size=64, shuffle=True, num_workers=20),
        'VGG19': DataLoader(IMAGENET_DATASET['train'], batch_size=64, shuffle=True, num_workers=20)
    }


    # Determine device and move models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    for model_idx, model in enumerate(ensemble_models):
        model.to(device)
        print(f"  {model_names[model_idx]} moved to {device}")

    ### TRAINING INDIVIDUAL MODELS ###
    criterion = torch.nn.MSELoss()  # Or your preferred loss function
    NUMBER_OF_EPOCHS = 40
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
                path_to_save="Models/Ensemble/Weights/Stacking/",
                model_name=model_name_key 
            )
            print(f"  Model '{model_name_key}' training completed.")
    

        except Exception as error:
            print(f"  Error training model '{model_name_key}': {error}")
            import traceback
            traceback.print_exc() 