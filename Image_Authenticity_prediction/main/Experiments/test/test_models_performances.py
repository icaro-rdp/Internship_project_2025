#!/usr/bin/env python3
"""
Test script to verify all imports and module structure are correct.

Usage:
    python test_imports.py
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def load_models_class():
    """Print the structure of all models."""
    
    try:
        from main.Models import (
            VGG16AuthenticityPredictor,
            VGG19AuthenticityPredictor,
            ResNet152AuthenticityPredictor,
            DenseNet161AuthenticityPredictor,
            InceptionV3AuthenticityPredictor,
            EfficientNetB3AuthenticityPredictor,
            BarlowTwinsAuthenticityPredictor
        )
        print("✓ All models imported successfully")
        vgg16 = VGG16AuthenticityPredictor()
        vgg19 = VGG19AuthenticityPredictor()
        resnet152 = ResNet152AuthenticityPredictor()
        densenet161 = DenseNet161AuthenticityPredictor()
        efficientnetb3 = EfficientNetB3AuthenticityPredictor()
        barlowtwins = BarlowTwinsAuthenticityPredictor()
        
        
        return {
            "vgg16": vgg16,
            "vgg19": vgg19,
            "resnet152": resnet152,
            "densenet161": densenet161,
            "efficientnetB3": efficientnetb3,
            "barlowtwins": barlowtwins
        }
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False
import re
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _instantiate_model_safe(model_cls):
    """Try common constructor signatures and fall back if needed."""
    try:
        return model_cls()
    except TypeError:
        try:
            return model_cls(freeze_backbone=False)
        except Exception:
            # Last resort: try without args again to raise original error
            return model_cls()


def evaluate_weights_on_testset(weights_dir: Path, device=None):
    """Evaluate all .pth weight files in weights_dir and return grouped summary by model.

    The function will:
    - find all .pth files in weights_dir
    - group them by model prefix (e.g. 'vgg16' from 'vgg16_exp1a_variant1_best.pth')
    - for each weight file instantiate the corresponding model class, load weights,
      evaluate on the test split using main.train.test_model(return_additional_metrics=True)
    - aggregate results per model and print a concise summary
    """
    # Ensure device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Import datasets and training utilities (lazy import so this file remains importable)
    from main.data import IMAGENET_DATASET, DENSENET_DATASET, BATCH_SIZE, NUM_WORKERS
    from main.train import test_model
    # Import model classes
    from main.Models import (
        VGG16AuthenticityPredictor,
        VGG19AuthenticityPredictor,
        ResNet152AuthenticityPredictor,
        DenseNet161AuthenticityPredictor,
        InceptionV3AuthenticityPredictor,
        EfficientNetB3AuthenticityPredictor,
        BarlowTwinsAuthenticityPredictor
    )

    # Mapping from file prefix -> class. Use lowercase keys matching filenames.
    MODEL_MAP = {
        'vgg16': VGG16AuthenticityPredictor,
        'vgg19': VGG19AuthenticityPredictor,
        'resnet152': ResNet152AuthenticityPredictor,
        'densenet161': DenseNet161AuthenticityPredictor,
        'inceptionv3': InceptionV3AuthenticityPredictor,
        'efficientnetb3': EfficientNetB3AuthenticityPredictor,
        'barlowtwins': BarlowTwinsAuthenticityPredictor
    }

    # Resolve weights files
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    all_pths = sorted(weights_dir.glob('*.pth'))
    if not all_pths:
        raise FileNotFoundError(f"No .pth files found in {weights_dir}")

    # Group by model prefix using pattern like 'vgg16_exp' or 'vgg16_exp1a'
    grouping = defaultdict(list)
    prefix_re = re.compile(r'^(?P<model>[A-Za-z0-9]+)_exp', re.IGNORECASE)
    for p in all_pths:
        m = prefix_re.match(p.name)
        if not m:
            # fallback: use everything up to first underscore
            prefix = p.name.split('_')[0].lower()
        else:
            prefix = m.group('model').lower()
        grouping[prefix].append(p)

    # Prepare result container
    summary = {}

    criterion = nn.MSELoss()

    for model_name, files in sorted(grouping.items()):
        print(f"\nProcessing model: {model_name} ({len(files)} weight file(s))")
        if model_name not in MODEL_MAP:
            print(f"  - No known model class for '{model_name}' — skipping")
            continue

        model_cls = MODEL_MAP[model_name]

        # Select test dataset: most models use IMAGENET_DATASET; DenseNet uses DENSENET_DATASET
        if model_name == 'densenet161':
            dataset = DENSENET_DATASET
        else:
            dataset = IMAGENET_DATASET

        test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        per_variant = []

        for wpath in files:
            print(f"  - Evaluating variant: {wpath.name} ...", end=' ')
            try:
                model = _instantiate_model_safe(model_cls)
                state = torch.load(str(wpath), map_location=device)
                # If the file contains a full checkpoint dict, try to extract state_dict
                if isinstance(state, dict) and 'state_dict' in state and len(state) > 1:
                    state_dict = state['state_dict']
                else:
                    state_dict = state
                model.load_state_dict(state_dict)

                metrics = test_model(model, test_loader, criterion, device=device, return_additional_metrics=True)

                entry = {
                    'weights': str(wpath),
                    'mse': float(metrics.get('mse', float('nan'))),
                    'rmse': float(metrics.get('rmse', float('nan'))),
                    'plcc': metrics.get('plcc'),
                    'srcc': metrics.get('srcc'),
                    'krcc': metrics.get('krcc')
                }
                per_variant.append(entry)
                print('✓')
            except Exception as e:
                print(f'✗ (error: {e})')
                per_variant.append({'weights': str(wpath), 'error': str(e)})

        # Aggregate
        valid = [p for p in per_variant if 'mse' in p and not (p.get('mse') is None)]
        if valid:
            avg_mse = float(sum(v['mse'] for v in valid) / len(valid))
            avg_rmse = float(sum(v['rmse'] for v in valid) / len(valid))
            best = min(valid, key=lambda x: x['mse'])
            # Average PLCC/SRCC ignoring None
            plcc_vals = [v['plcc'] for v in valid if v.get('plcc') is not None]
            srcc_vals = [v['srcc'] for v in valid if v.get('srcc') is not None]
            avg_plcc = float(sum(plcc_vals) / len(plcc_vals)) if plcc_vals else None
            avg_srcc = float(sum(srcc_vals) / len(srcc_vals)) if srcc_vals else None
        else:
            avg_mse = avg_rmse = best = avg_plcc = avg_srcc = None

        summary[model_name] = {
            'num_variants': len(files),
            'variants': per_variant,
            'avg_mse': avg_mse,
            'avg_rmse': avg_rmse,
            'avg_plcc': avg_plcc,
            'avg_srcc': avg_srcc,
            'best_variant': best
        }

        # Print a compact table row for this model
        if valid:
            print(f"  -> Best: {Path(best['weights']).name} | MSE: {best['mse']:.4f} | RMSE: {best['rmse']:.4f}")
            print(f"     Avg MSE: {avg_mse:.4f} | Avg RMSE: {avg_rmse:.4f} | Avg PLCC: {avg_plcc} | Avg SRCC: {avg_srcc}")
        else:
            print(f"  -> No valid evaluation results for model {model_name}")

    return summary


def test_models_on_testset():
    """Pytest-compatible function: evaluates all found models and prints summary.

    This function does not assert; it is intended as a convenience test-runner that
    reports model performance from saved weights. Use it with pytest or run directly.
    """
    WEIGHTS_PATH = project_root / 'Outputs' / 'Experiment_1_variants' / 'Weights'
    summary = evaluate_weights_on_testset(WEIGHTS_PATH)

    # Dump a compact JSON summary to stdout for CI/debugging
    print('\n==== AGGREGATED SUMMARY JSON ====>')
    print(json.dumps({k: {'num_variants': v['num_variants'], 'avg_mse': v['avg_mse'], 'avg_rmse': v['avg_rmse'], 'best_variant': Path(v['best_variant']['weights']).name if v['best_variant'] else None} for k, v in summary.items()}, indent=2))


if __name__ == '__main__':
    # Allow running as a standalone script
    WEIGHTS_PATH = project_root / 'main' / 'Experiments'/'Outputs'/'Experiment_1_variants' / 'Weights'
    s = evaluate_weights_on_testset(WEIGHTS_PATH)
    # Save detailed summary next to weights for later inspection
    out_path = WEIGHTS_PATH.parent / 'models_test_summary.json'
    with open(out_path, 'w') as fh:
        json.dump(s, fh, indent=2)
    print(f"\nSaved detailed summary to: {out_path}")
    