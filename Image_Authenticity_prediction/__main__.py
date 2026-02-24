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
    BarlowTwinsAuthenticityPredictor,
)
from main.data import (
    IMAGENET_DATASET,
    DENSENET_DATASET,
    INCEPTIONV3_DATASET,
)
from main.Utils.config import (
    load_config,
    get_device,
    get_data_config,
    get_paths_config,
)
from main.train import train_model, test_model, plot_loss_history
from torch.utils.data import DataLoader

# Model registry
MODEL_REGISTRY = {
    "vgg16": VGG16AuthenticityPredictor,
    "vgg19": VGG19AuthenticityPredictor,
    "resnet152": ResNet152AuthenticityPredictor,
    "densenet161": DenseNet161AuthenticityPredictor,
    "inceptionv3": InceptionV3AuthenticityPredictor,
    "efficientnetb3": EfficientNetB3AuthenticityPredictor,
    "barlowtwins": BarlowTwinsAuthenticityPredictor,
}

# Dataset mapping based on model input requirements
DATASET_MAPPING = {
    "vgg16": IMAGENET_DATASET,
    "vgg19": IMAGENET_DATASET,
    "resnet152": IMAGENET_DATASET,
    "efficientnetb3": IMAGENET_DATASET,
    "barlowtwins": IMAGENET_DATASET,
    "densenet161": DENSENET_DATASET,
    "inceptionv3": INCEPTIONV3_DATASET,
}


def train_command(args):
    """Execute training command."""
    print(f"Starting training for model: {args.model}")

    # Load config
    data_cfg = get_data_config()
    paths_cfg = get_paths_config()
    device = get_device()
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]

    # Get model
    if args.model not in MODEL_REGISTRY:
        print(
            f"Error: Model '{args.model}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
        return

    model_class = MODEL_REGISTRY[args.model]
    model = model_class(freeze_backbone=args.freeze_backbone)

    # Get dataset
    dataset = DATASET_MAPPING[args.model]
    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    dataloaders = {"train": train_loader, "val": val_loader}

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
        patience=args.patience,
    )

    # Save model
    save_path = paths_cfg["weights_dir"] / f"{args.model}_best.pth"
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
    data_cfg = get_data_config()
    device = get_device()
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]

    # Get model
    if args.model not in MODEL_REGISTRY:
        print(
            f"Error: Model '{args.model}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
        return

    model_class = MODEL_REGISTRY[args.model]
    model = model_class(freeze_backbone=False)

    # Load weights
    if args.weights:
        # Load only tensor weights (safer). The CLI expects a state_dict saved via torch.save(model.state_dict()).
        model.load_state_dict(torch.load(args.weights, weights_only=True))
        print(f"Loaded weights from: {args.weights}")
    else:
        print("Warning: No weights specified, using randomly initialized model")

    # Get dataset
    dataset = DATASET_MAPPING[args.model]
    test_loader = DataLoader(
        dataset["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Evaluate
    criterion = torch.nn.MSELoss()
    test_loss = test_model(model, test_loader, criterion, device)
    test_rmse = test_loss**0.5  # Compute RMSE from MSE
    print(f"Test MSE: {test_loss:.4f}, RMSE: {test_rmse:.4f}")


def experiment_one_command(args):
    """Execute experiment one command."""
    from main.Experiments.experiment_one import run_experiment_one_complete

    print("=" * 80)
    print("STARTING EXPERIMENT ONE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Models: {args.models if args.models else 'all'}")
    print(f"  - Run training: {args.train}")
    print(f"  - Run pruning: {args.prune}")
    print(f"  - Run testing: {args.test}")
    if args.prune:
        print(f"  - Pruning method: {args.pruning_method}")
        print(f"  - Pruning threshold: {args.threshold}")
    print("=" * 80)

    # Run experiment
    results = run_experiment_one_complete(
        models_to_process=args.models,
        run_training=args.train,
        run_pruning=args.prune,
        run_testing=args.test,
        pruning_method=args.pruning_method,
        threshold=args.threshold,
    )

    print("=" * 80)
    print("EXPERIMENT ONE COMPLETE")
    print("=" * 80)


def experiment_two_command(args):
    """Execute experiment two command."""
    from main.Experiments.experiment_two import run_experiment_2

    print("=" * 80)
    print("STARTING EXPERIMENT TWO")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Models: {args.models if args.models else 'all'}")
    print(f"  - XAI Methods: {args.xai_methods}")
    print(f"  - Variants: {args.variants}")
    print(f"  - Save maps: {args.save_maps}")
    print(f"  - Comparison only: {args.comparison_only}")
    if args.comparison_only or args.run_comparison:
        print(f"  - Comparison kinds: {args.comparison_kinds}")
        print(f"  - Comparison metrics: {args.comparison_metrics}")
        print(f"  - Target resolution: {args.target_resolution}")
    print("=" * 80)

    # Parse comparison kinds
    kinds = (
        tuple(args.comparison_kinds)
        if args.comparison_kinds
        else ("between_model_architectures",)
    )

    # Parse comparison metrics
    metrics = (
        tuple(args.comparison_metrics) if args.comparison_metrics else ("correlation",)
    )

    # Parse target resolution
    target_res = (
        tuple(map(int, args.target_resolution.split(",")))
        if args.target_resolution
        else (224, 224)
    )

    # Run experiment - handle generation, comparison, or both
    # comparison_only: skip generation, only run comparison
    # run_comparison: run generation AND comparison in one call
    run_experiment_2(
        models=args.models,
        save_maps=args.save_maps if not args.comparison_only else False,
        variants=args.variants,
        xai_methods=args.xai_methods,
        comparison_only=args.comparison_only,
        run_comparison=args.run_comparison,
        comparison_kinds=kinds,
        comparison_metrics=metrics,
        comparison_target_resolution=target_res,
        save_comparison_json=True,
    )

    print("=" * 80)
    print("EXPERIMENT TWO COMPLETE")
    print("=" * 80)


def experiment_three_command(args):
    """Execute experiment three command."""
    from main.Experiments.experiment_three import run_experiment_3

    print("=" * 80)
    print("STARTING EXPERIMENT THREE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Models: {args.models if args.models else 'all'}")
    print(f"  - Strategy: {args.strategy}")
    print(f"  - Train: {args.train}")
    print(f"  - Evaluate: {args.evaluate}")
    print(f"  - Save results: {args.save_results}")
    print("=" * 80)

    # Map CLI strategy to experiment_3 ensemble_mode
    ensemble_mode = ["bagging", "stacking"] if args.strategy == "both" else [args.strategy]

    # Run experiment
    run_experiment_3(
        models=args.models,
        run_training=args.train,
        run_pruning=args.train,
        run_evaluation=args.evaluate,
        save_results=args.save_results,
        ensemble_mode=ensemble_mode,
    )

    print("=" * 80)
    print("EXPERIMENT THREE COMPLETE")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Image Authenticity Prediction - Training, Evaluation, and Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs (default: 50)"
    )
    train_parser.add_argument(
        "--patience", type=int, default=7, help="Early stopping patience (default: 7)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone weights during training",
    )
    train_parser.add_argument(
        "--plot", action="store_true", help="Plot training history after training"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to evaluate",
    )
    eval_parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights file"
    )

    # Experiment One command
    exp1_parser = subparsers.add_parser(
        "experiment-one", help="Run Experiment 1 (Training, Pruning, Testing)"
    )
    exp1_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        help="Specific models to process (default: all models)",
    )
    exp1_parser.add_argument(
        "--train", action="store_true", help="Run training phase (Experiment 1A)"
    )
    exp1_parser.add_argument(
        "--prune", action="store_true", help="Run pruning phase (Experiment 1B)"
    )
    exp1_parser.add_argument(
        "--test",
        action="store_true",
        help="Run testing phase on trained and pruned models",
    )
    exp1_parser.add_argument(
        "--pruning-method",
        type=str,
        default="both",
        choices=["greedy", "negative_impact", "both"],
        help="Pruning method to use (default: both)",
    )
    exp1_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for negative_impact pruning (default: 0.0)",
    )

    # Experiment Two command
    exp2_parser = subparsers.add_parser(
        "experiment-two",
        help="Run Experiment 2 (XAI Heatmap Generation and Comparison)",
    )
    exp2_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        help="Specific models to process (default: all models)",
    )
    exp2_parser.add_argument(
        "--xai-methods",
        type=str,
        default="both",
        choices=["gradcam", "mpm", "both"],
        help="XAI methods to use (default: both)",
    )
    exp2_parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Variants to process: all, orig, base, greedy, negative (default: all)",
    )
    exp2_parser.add_argument(
        "--save-maps",
        action="store_true",
        default=True,
        help="Save generated heatmaps (default: True)",
    )
    exp2_parser.add_argument(
        "--no-save-maps",
        dest="save_maps",
        action="store_false",
        help="Do not save generated heatmaps",
    )
    exp2_parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only run comparison analysis (skip generation)",
    )
    exp2_parser.add_argument(
        "--run-comparison", action="store_true", help="Run comparison after generation"
    )
    exp2_parser.add_argument(
        "--comparison-kinds",
        type=str,
        nargs="+",
        choices=[
            "between_model_architectures",
            "within_model_variants",
            "cross_methods",
        ],
        default=["between_model_architectures"],
        help="Types of comparisons to run (default: between_model_architectures)",
    )
    exp2_parser.add_argument(
        "--comparison-metrics",
        type=str,
        nargs="+",
        choices=[
            "correlation",
            "ssim",
            "rmse",
            "scc",
            "top_percent_iou_5",
            "top_percent_iou_15",
            "top_percent_iou_25",
        ],
        default=["correlation"],
        help="Metrics to use for comparison (default: correlation)",
    )
    exp2_parser.add_argument(
        "--target-resolution",
        type=str,
        default="224,224",
        help="Target resolution for comparison as width,height (default: 224,224)",
    )

    # Experiment Three command
    exp3_parser = subparsers.add_parser(
        "experiment-three",
        help="Run Experiment 3 (Ensemble Strategies: Bagging and Stacking)",
    )
    exp3_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        help="Specific models to process (default: all models)",
    )
    exp3_parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["bagging", "stacking", "both"],
        help="Ensemble strategy to use (default: both)",
    )
    exp3_parser.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Train stacking models (default: True, bagging uses pre-trained)",
    )
    exp3_parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Skip training stacking models",
    )
    exp3_parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Evaluate ensemble models (default: True)",
    )
    exp3_parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="Skip evaluation",
    )
    exp3_parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save results to JSON (default: True)",
    )
    exp3_parser.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        help="Do not save results to JSON",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "experiment-one":
        experiment_one_command(args)
    elif args.command == "experiment-two":
        experiment_two_command(args)
    elif args.command == "experiment-three":
        experiment_three_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
