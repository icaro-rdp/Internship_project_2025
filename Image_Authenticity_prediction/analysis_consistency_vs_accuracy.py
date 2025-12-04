import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import project modules
import sys
sys.path.append('.')
from main.Models import VGG16AuthenticityPredictor, ResNet152AuthenticityPredictor # Add other models as needed
from main.train import test_model
from main.data import IMAGENET_DATASET, NUM_WORKERS

# Configuration
MODEL_NAME = 'vgg16'
ModelClass = VGG16AuthenticityPredictor
JSON_PATH = Path('Image_Authenticity_prediction/main/Experiments/Outputs/Experiment_2_variants/experiment_2b_comparison.json')
WEIGHTS_DIR = Path('Image_Authenticity_prediction/main/Experiments/Outputs/Experiment_1_variants/Weights')

def get_consistency_scores(json_path, model_name, method='gradcam'):
    """
    Extracts the average consistency score (correlation) for each image 
    across all variant pairs from the JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Access structure: method_within_model_variants -> model -> per_image -> correlation
    key = f"{method}_within_model_variants"
    if key not in data or model_name not in data[key]:
        raise ValueError(f"Data for {model_name} not found in JSON")
        
    correlations_map = data[key][model_name]['per_image']['correlation']
    
    # Initialize arrays
    first_pair = next(iter(correlations_map.values()))
    n_images = len(first_pair)
    sum_scores = np.zeros(n_images)
    count_pairs = 0
    
    # Sum up all pairwise correlations per image
    for pair_key, scores in correlations_map.items():
        sum_scores += np.array(scores)
        count_pairs += 1
        
    avg_consistency = sum_scores / count_pairs
    return avg_consistency

def get_prediction_errors(model_cls, weights_dir, model_name, device='cuda'):
    """
    Computes the average Squared Error for each image across all model variants.
    """
    # Find all variant weights for this model
    weight_files = sorted(list(weights_dir.glob(f"{model_name}_exp1b_variant*_greedy_pruned.pth")))
    if not weight_files:
        # Fallback to base variants if pruned not found
        weight_files = sorted(list(weights_dir.glob(f"{model_name}_exp1a_variant*_best.pth")))
    
    print(f"Found {len(weight_files)} variants for {model_name}")
    
    # Setup Data
    test_loader = DataLoader(
        IMAGENET_DATASET['test'], 
        batch_size=32, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    all_squared_errors = []
    
    for w_path in weight_files:
        print(f"Evaluating {w_path.name}...")
        model = model_cls(freeze_backbone=False)
        model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
        model.to(device)
        
        # Get predictions using existing train.py utility
        metrics = test_model(
            model, 
            test_loader, 
            torch.nn.MSELoss(), 
            device=device, 
            return_additional_metrics=True
        )
        
        preds = metrics['preds']
        labels = metrics['labels']
        
        # Calculate Squared Error per image: (y_pred - y_true)^2
        se = (preds - labels) ** 2
        all_squared_errors.append(se)
        
    # Average error across variants (N_variants x N_images -> N_images)
    avg_squared_errors = np.mean(np.stack(all_squared_errors), axis=0)
    return avg_squared_errors

def main():
    # 1. Get Consistency Scores (X-axis)
    print("Extracting consistency scores...")
    consistency = get_consistency_scores(JSON_PATH, MODEL_NAME)
    
    # 2. Get Prediction Errors (Y-axis)
    print("Calculating prediction errors...")
    errors = get_prediction_errors(ModelClass, WEIGHTS_DIR, MODEL_NAME)
    
    # 3. Analysis
    # We expect negative correlation: Higher Consistency -> Lower Error
    corr, p_val = pearsonr(consistency, errors)
    
    print(f"\nResults for {MODEL_NAME}:")
    print(f"Correlation (Consistency vs MSE): {corr:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=consistency, y=errors, alpha=0.6)
    sns.regplot(x=consistency, y=errors, scatter=False, color='red')
    
    plt.title(f'Explanation Consistency vs. Prediction Error ({MODEL_NAME})')
    plt.xlabel('Avg. Pairwise Heatmap Correlation (Consistency)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{MODEL_NAME}_consistency_vs_accuracy.png')
    plt.show()

if __name__ == "__main__":
    main()