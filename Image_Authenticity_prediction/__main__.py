"""
Main entry point for the Image Authenticity Prediction package.

This script allows you to run training, evaluation, and experiments
from the command line.

Usage:
    python -m Image_Authenticity_prediction --help
    python -m Image_Authenticity_prediction train --model vgg16
    python -m Image_Authenticity_prediction evaluate --model vgg16 --weights path/to/weights.pth
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    InceptionV3AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor
)
from main.data import IMAGENET_DATASET, DENSENET_DATASET, INCEPTIONV3_DATASET, BATCH_SIZE, NUM_WORKERS
from main.train import train_model, test_model, plot_loss_history
from torch.utils.data import DataLoader

# Model registry
MODEL_REGISTRY = {
    'vgg16': VGG16AuthenticityPredictor,
    'vgg19': VGG19AuthenticityPredictor,
    'resnet152': ResNet152AuthenticityPredictor,
    'densenet161': DenseNet161AuthenticityPredictor,
    'inceptionv3': InceptionV3AuthenticityPredictor,
    'efficientnetb3': EfficientNetB3AuthenticityPredictor,
    'barlowtwins': BarlowTwinsAuthenticityPredictor
}

# Dataset mapping based on model input requirements
DATASET_MAPPING = {
    'vgg16': IMAGENET_DATASET,
    'vgg19': IMAGENET_DATASET,
    'resnet152': IMAGENET_DATASET,
    'efficientnetb3': IMAGENET_DATASET,
    'barlowtwins': IMAGENET_DATASET,
    'densenet161': DENSENET_DATASET,
    'inceptionv3': INCEPTIONV3_DATASET
}


def load_config(config_path='Configs/config.yaml'):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_command(args):
    """Execute training command."""
    print(f"Starting training for model: {args.model}")
    
    # Load config
    config = load_config()
    device = config['run_settings']['device']
    
    # Get model
    if args.model not in MODEL_REGISTRY:
        print(f"Error: Model '{args.model}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
        return
    
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(freeze_backbone=args.freeze_backbone)
    
    # Get dataset
    dataset = DATASET_MAPPING[args.model]
    train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    
    # Setup training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train
    print(f"Training on device: {device}")
    best_model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        patience=args.patience
    )
    
    # Save model
    save_path = Path(config['paths']['weights_dir']) / f"{args.model}_best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    
    # Plot training history
    if args.plot:
        plot_loss_history(history)


def evaluate_command(args):
    """Execute evaluation command."""
    print(f"Evaluating model: {args.model}")
    
    # Load config
    config = load_config()
    device = config['run_settings']['device']
    
    # Get model
    if args.model not in MODEL_REGISTRY:
        print(f"Error: Model '{args.model}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
        return
    
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(freeze_backbone=False)
    
    # Load weights
    if args.weights:
        model.load_state_dict(torch.load(args.weights))
        print(f"Loaded weights from: {args.weights}")
    else:
        print("Warning: No weights specified, using randomly initialized model")
    
    # Get dataset
    dataset = DATASET_MAPPING[args.model]
    test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Evaluate
    criterion = torch.nn.MSELoss()
    test_loss = test_model(model, test_loader, criterion, device)
    print(f"Test Loss (RMSE): {test_loss:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Image Authenticity Prediction - Training and Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, required=True, 
                             choices=list(MODEL_REGISTRY.keys()),
                             help='Model architecture to train')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Maximum number of epochs (default: 50)')
    train_parser.add_argument('--patience', type=int, default=7,
                             help='Early stopping patience (default: 7)')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate (default: 0.001)')
    train_parser.add_argument('--freeze-backbone', action='store_true',
                             help='Freeze backbone weights during training')
    train_parser.add_argument('--plot', action='store_true',
                             help='Plot training history after training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', type=str, required=True,
                            choices=list(MODEL_REGISTRY.keys()),
                            help='Model architecture to evaluate')
    eval_parser.add_argument('--weights', type=str, required=True,
                            help='Path to model weights file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
