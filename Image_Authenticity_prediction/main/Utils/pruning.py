import os
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Dict, Any, List



class FeatureMapsPruner:
    """
    Encapsulates the logic for computing feature map importance and performing
    greedy pruning on a specified convolutional layer of a PyTorch model.
    """

    def __init__(self,
                 model: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 layer_name: str,
                 criterion: nn.Module,
                 eval_function: Callable,
                 device: torch.device):
        """
        Initializes the Pruner.

        Args:
            model: The neural network model to prune.
            dataloader: DataLoader for evaluation.
            layer_name: The name of the layer to prune (e.g., 'features.0').
            criterion: The loss function to use for evaluation (e.g., nn.MSELoss()).
            eval_function: A callable function (like test_model) that takes
                           (model, dataloader, criterion, device) and returns
                           a performance metric (MSE loss).
            device: The device (e.g., torch.device('cuda')) to run on.
        """
        self.model = model
        self.dataloader = dataloader
        self.layer_name = layer_name
        self.criterion = criterion
        self.eval_function = eval_function
        self.device = device

        self.model.to(self.device)
        self.layer = self._get_layer()
        if not isinstance(self.layer, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
             print(f"Warning: Layer {layer_name} is not a standard Conv layer. "
                   "Pruning targets .weight and .bias attributes.")

        # Internal state
        self.importance_scores: Optional[np.ndarray] = None
        self.baseline_mse: Optional[float] = None
        self.baseline_rmse: Optional[float] = None
        self._original_weights: torch.Tensor = self.layer.weight.clone().detach()
        self._original_bias: Optional[torch.Tensor] = None
        if self.layer.bias is not None:
            self._original_bias = self.layer.bias.clone().detach()

    def _get_layer(self) -> nn.Module:
        """Retrieves the layer module from the model using its name."""
        try:
            dict_modules = dict(self.model.named_modules())
            return dict_modules[self.layer_name]
        except KeyError:
            raise ValueError(f"Layer '{self.layer_name}' not found in model. "
                             f"Available layers: {list(dict_modules.keys())}")

    def _evaluate_model(self) -> Tuple[float, float]:
        """
        Helper function to evaluate the model's current state.
        
        Returns:
            Tuple of (MSE, RMSE)
        """
        self.model.eval()
        with torch.no_grad():
            mse = self.eval_function(self.model, self.dataloader, self.criterion, self.device)
            rmse = math.sqrt(mse)
        return mse, rmse

    def _restore_weights(self):
        """Restores the layer's weights and bias to their original state."""
        self.layer.weight.data.copy_(self._original_weights)
        if self._original_bias is not None:
            self.layer.bias.data.copy_(self._original_bias)
        print("Model weights restored to original state.")

    def _zero_out_channel(self, channel_idx: int):
        """Zeros out a specific channel's weights and bias."""
        self.layer.weight.data[channel_idx, ...] = 0
        if self.layer.bias is not None:
            self.layer.bias.data[channel_idx] = 0

    def _restore_channel(self, channel_idx: int):
        """Restores a specific channel's weights and bias from backup."""
        self.layer.weight.data[channel_idx, ...] = self._original_weights[channel_idx, ...]
        if self.layer.bias is not None:
            self.layer.bias.data[channel_idx] = self._original_bias[channel_idx]

    def compute_importance_scores(self, 
                                  save_path: Optional[str] = None,
                                  force_recompute: bool = False) -> np.ndarray:
        """
        Computes the importance of each feature map in the layer.

        Importance is measured as: baseline_mse - pruned_mse
        A positive score means removing the channel HURTS performance (it's important).
        A negative score means removing the channel HELPS performance (it's noisy).

        Args:
            save_path: Optional path to save the computed scores as a .npy file.
            force_recompute: If True, recomputes scores even if they exist in memory
                             or on disk.

        Returns:
            A numpy array of [channel_index, importance_score], sorted by
            importance (most negative/harmful first).
        """
        if self.importance_scores is not None and not force_recompute:
            print("Using cached importance scores.")
            return self.importance_scores
        
        if save_path and os.path.exists(save_path) and not force_recompute:
            print(f"Loading importance scores from {save_path}")
            self.importance_scores = np.load(save_path)
            return self.importance_scores

        print("Computing importance scores...")
        self._restore_weights()  # Ensure we start from a clean state
        self.model.eval()

        self.baseline_mse, self.baseline_rmse = self._evaluate_model()
        print(f'Baseline - MSE: {self.baseline_mse:.4f}, RMSE: {self.baseline_rmse:.4f}')

        scores = []
        num_channels = self.layer.out_channels

        for i in tqdm(range(num_channels), desc=f"Computing importance for {self.layer_name}"):
            self._zero_out_channel(i)
            
            pruned_mse, pruned_rmse = self._evaluate_model()
            
            # Importance = baseline - pruned (using MSE)
            # positive = bad to remove (important)
            # negative = good to remove (noisy)
            importance_score = self.baseline_mse - pruned_mse
            scores.append([i, importance_score])

            self._restore_channel(i) # Restore just this channel

        # Sort by importance score (descending, larger values first).
        # Higher scores = more harmful to remove 
        # Lower/negative scores = less important 
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        self.importance_scores = np.array(sorted_scores)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, self.importance_scores)
            print(f"Importance scores saved to {save_path}")

        return self.importance_scores

    def greedy_pruning(self, model_save_path: str) -> Dict[str, Any]:
        """
        Iteratively removes feature maps, keeping a removal only if it
        improves (lowers) the MSE loss.

        This method modifies the model, saves the pruned version, and then
        restores the original model state.

        Args:
            model_save_path: Path to save the final pruned model state_dict.

        Returns:
            A dictionary with pruning results and performance metrics.
        """
        if self.importance_scores is None:
            print("Importance scores not computed. Running compute_importance_scores first.")
            self.compute_importance_scores()

        self._restore_weights()  # Start from a clean slate
        
        if self.baseline_mse is None:
             self.baseline_mse, self.baseline_rmse = self._evaluate_model()

        print(f"Starting greedy pruning. Baseline - MSE: {self.baseline_mse:.4f}, RMSE: {self.baseline_rmse:.4f}")
        print("------------------")

        removed_features = []
        mse_history = [(removed_features.copy(), self.baseline_mse, self.baseline_rmse)]
        current_best_mse = self.baseline_mse
        current_best_rmse = self.baseline_rmse

        # Iterate from most noisy (lowest score) to most important (highest score)
        for (channel_idx, importance_score) in tqdm(self.importance_scores, desc=f"Greedy pruning {self.layer_name}"):
            channel_idx = int(channel_idx)
            
            self._zero_out_channel(channel_idx)
            
            pruned_mse, pruned_rmse = self._evaluate_model()

            print(f"Testing removal of channel {channel_idx}, "
                  f"Importance: {importance_score:.4f}, "
                  f"New MSE: {pruned_mse:.4f}, RMSE: {pruned_rmse:.4f}")

            # If MSE improved (got smaller), keep the channel zeroed out
            if pruned_mse < current_best_mse:
                current_best_mse = pruned_mse
                current_best_rmse = pruned_rmse
                removed_features.append(channel_idx)
                mse_history.append((removed_features.copy(), current_best_mse, current_best_rmse))
                print(f"  ✓ IMPROVING: Keeping channel {channel_idx} zeroed. New best MSE: {current_best_mse:.4f}, RMSE: {current_best_rmse:.4f}")
            else:
                # Restore this channel
                self._restore_channel(channel_idx)
                print(f"  ✗ NOT IMPROVING: Restoring channel {channel_idx}.")
            
            print("------------------")

        # Save the pruned model (which is the current state of self.model)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"\nPruned model saved to {model_save_path}")

        # Restore the model to its original state for any future operations
        self._restore_weights()

        # Final statistics
        final_mse = current_best_mse
        final_rmse = current_best_rmse
        improvement_mse = self.baseline_mse - final_mse
        improvement_rmse = self.baseline_rmse - final_rmse
        reduction_pct = (len(removed_features) / self.layer.out_channels) * 100

        print("------------------")
        print(f"Final MSE: {final_mse:.4f}, RMSE: {final_rmse:.4f} after removing {len(removed_features)} feature maps")
        print(f"Improvement - MSE: {improvement_mse:.4f}, RMSE: {improvement_rmse:.4f}")
        print(f"Feature reduction: {reduction_pct:.1f}%")

        return {
            'removed_features': removed_features,
            'baseline_mse': self.baseline_mse,
            'baseline_rmse': self.baseline_rmse,
            'final_mse': final_mse,
            'final_rmse': final_rmse,
            'improvement_mse': improvement_mse,
            'improvement_rmse': improvement_rmse,
            'reduction_percentage': reduction_pct,
            'mse_history': mse_history
        }

    def negative_impact_pruning(self, model_save_path: str, threshold: float = 0.0) -> Dict[str, Any]:
        """
        Removes all feature maps with an importance score above a threshold (> threshold).
        

        Args:
            model_save_path: Path to save the final pruned model state_dict.
            threshold: Importance score threshold. Channels with a score
                       *above* this will be removed.

        Returns:
            A dictionary with pruning results.
        """
        if self.importance_scores is None:
            print("Importance scores not computed. Running compute_importance_scores first.")
            self.compute_importance_scores()

        self._restore_weights()  # Start from a clean slate
        
        if self.baseline_mse is None:
             self.baseline_mse, self.baseline_rmse = self._evaluate_model()

        print(f"Starting negative impact pruning. Baseline - MSE: {self.baseline_mse:.4f}, RMSE: {self.baseline_rmse:.4f}")
        
        removed_features = []
        for (channel_idx, importance_score) in tqdm(self.importance_scores, desc=f"Pruning negative impact features"):
            if importance_score > threshold:
                channel_idx = int(channel_idx)
                self._zero_out_channel(channel_idx)
                removed_features.append(channel_idx)
        
        print(f"Removed {len(removed_features)} features with score > {threshold}.")

        # Evaluate the final pruned model
        final_mse, final_rmse = self._evaluate_model()
        print(f"Final MSE: {final_mse:.4f}, RMSE: {final_rmse:.4f} after pruning")

        # Save the pruned model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Pruned model saved to {model_save_path}")

        # Restore original weights
        self._restore_weights()

        # Final stats
        improvement_mse = self.baseline_mse - final_mse
        improvement_rmse = self.baseline_rmse - final_rmse
        reduction_pct = (len(removed_features) / self.layer.out_channels) * 100

        return {
            'removed_features': removed_features,
            'baseline_mse': self.baseline_mse,
            'baseline_rmse': self.baseline_rmse,
            'final_mse': final_mse,
            'final_rmse': final_rmse,
            'improvement_mse': improvement_mse,
            'improvement_rmse': improvement_rmse,
            'reduction_percentage': reduction_pct
        }
    
#! Example usage:
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

#! --- 1. Define external dependencies (as in your project) ---

# Placeholder for your test_model function
# def test_model(model, dataloader, criterion, device):
#     model.eval()
#     total_se = 0
#     total_count = 0
#     with torch.no_grad():
#         for x_batch, y_batch in dataloader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             total_se += loss.item() * x_batch.size(0)
#             total_count += x_batch.size(0)
#     rmse = (total_se / total_count) ** 0.5
#     return rmse

#! --- 2. Create a dummy model and data for demonstration ---

# A simple CNN
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, padding=1), # Layer 'features.0'
#             nn.ReLU(),
#             nn.Conv2d(8, 16, kernel_size=3, padding=1), # Layer 'features.2'
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.classifier = nn.Linear(16, 1)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# Dummy data
# X = torch.randn(100, 1, 28, 28)
# Y = torch.randn(100, 1)
# dataset = TensorDataset(X, Y)
# dataloader = DataLoader(dataset, batch_size=16)

#! --- 3. Setup and run the pruner ---

# Define components
# my_model = SimpleCNN()
# my_criterion = nn.MSELoss()
# my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# target_layer_name = 'features.2' # Prune the second conv layer

# Create directories for outputs
# os.makedirs('Weights', exist_ok=True)
# os.makedirs('Ranking_arrays', exist_ok=True)

# Instantiate the pruner
# pruner = FeaturePruner(
#     model=my_model,
#     dataloader=dataloader,
#     layer_name=target_layer_name,
#     criterion=my_criterion,
#     eval_function=test_model,
#     device=my_device
# )

#! --- 4. Run the pruning methods ---

# Compute importance scores
# Note: The original file names are used for compatibility
# scores = pruner.compute_importance_scores(
#     save_path='Ranking_arrays/real_authenticity_batch_importance_scores.npy'
# )
# print("\nComputed Importance Scores (Channel, Score):")
# print(scores)

# Run greedy pruning
# print("\n--- STARTING GREEDY PRUNING ---")
# greedy_results = pruner.greedy_pruning(
#     model_save_path='Weights/pruned_model.pth'
# )
# print("\nGreedy Pruning Results:")
# print(greedy_results)

# Run negative impact pruning
# This will use the same computed scores
# print("\n--- STARTING NEGATIVE IMPACT PRUNING ---")
# negative_results = pruner.negative_impact_pruning(
#     model_save_path='Weights/negative_impact_pruned_model.pth',
#     threshold=0.0 # Remove channels that *improve* RMSE (score < 0)
# )
# print("\nNegative Impact Pruning Results:")
# print(negative_results)
