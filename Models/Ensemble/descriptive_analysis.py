import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os


@dataclass
class VisualizationConfig:
    """Configuration class for visualization settings."""
    figure_size_correlation: Tuple[int, int] = (16, 18)
    figure_size_residuals: Tuple[int, int] = (18, 6)
    scatter_alpha: float = 0.7
    scatter_color: str = 'blue'
    residual_color: str = 'orange'
    histogram_color: str = 'green'
    grid_alpha: float = 0.3
    dpi: int = 300
    histogram_bins: int = 20
    font_size_title: int = 16
    font_size_labels: int = 12

class ModelEvaluationVisualizer:
    """
    A comprehensive utility class for model evaluation visualization including
    scatter plots, residual analysis, and statistical metrics computation.
    
    This class provides methods to:
    - Compute evaluation metrics (Spearman correlation, R², RMSE)
    - Generate correlation scatter plots
    - Create residual analysis plots
    - Visualize error distributions
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer with configuration settings.
        
        Args:
            config: VisualizationConfig object with plotting parameters.
                   If None, uses default configuration.
        """
        self.config = config or VisualizationConfig()
        self.evaluation_results: Optional[Dict[str, Any]] = None
    
    def compute_evaluation_metrics(self, 
                                 model: torch.nn.Module, 
                                 dataloader: torch.utils.data.DataLoader, 
                                 device: str) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics for a model.
        
        Args:
            model: The trained PyTorch model to evaluate
            dataloader: Test dataloader containing images and true scores
            device: Device to run the model on ('cuda' or 'cpu')
            
        Returns:
            Dictionary containing predictions, ground truth, residuals, and metrics
            
        Raises:
            ValueError: If model outputs don't have the expected shape
            RuntimeError: If evaluation fails during forward pass
        """
        try:
            # Set model to evaluation mode and move to device
            model.eval()
            model.to(device)
            
            # Initialize storage for predictions and ground truth
            prediction_batches = []
            ground_truth_batches = []
            
            # Collect predictions without gradient computation
            with torch.no_grad():
                for batch_inputs, batch_labels in dataloader:
                    batch_inputs = batch_inputs.to(device)
                    model_outputs, _ = model(batch_inputs)
                    
                    prediction_batches.append(model_outputs.cpu())
                    ground_truth_batches.append(batch_labels)
            
            # Concatenate all batches
            all_predictions = torch.cat(prediction_batches, dim=0).numpy()
            all_ground_truth = torch.cat(ground_truth_batches, dim=0).numpy()
            
            # Check if predictions have one or two score outputs
            # if it has one output, we assume it's a single authenticity score, if it has two outputs, we assume it's quality and authenticity scores, and we handle both cases
            if all_predictions.shape[1] not in [1, 2]:
                raise ValueError("Model outputs must have either 1 (authenticity) or 2 (quality and authenticity) scores per sample.")
            if all_ground_truth.shape[1] not in [1, 2]:
                raise ValueError("Ground truth labels must have either 1 (authenticity) or 2 (quality and authenticity) scores per sample.")
            # Ensure predictions and ground truth have the same number of samples
            if all_predictions.shape[0] != all_ground_truth.shape[0]:
                raise ValueError("Mismatch between number of predictions and ground truth samples.")
            # If two score outputs are present, we assume they are quality and authenticity scores
            evaluation_results = {}
            # If only one score output is present, we assume it's a single authenticity score
            if all_predictions.shape[1] == 2:
                # Extract individual score components
                predicted_quality_scores = all_predictions[:, 0]
                true_quality_scores = all_ground_truth[:, 0]
                predicted_authenticity_scores = all_predictions[:, 1]
                true_authenticity_scores = all_ground_truth[:, 1]
                
                # Calculate averaged scores
                true_average_scores = (true_quality_scores + true_authenticity_scores) / 2
                predicted_average_scores = (predicted_quality_scores + predicted_authenticity_scores) / 2
                
                # Compute residuals (prediction errors)
                quality_residuals = predicted_quality_scores - true_quality_scores
                authenticity_residuals = predicted_authenticity_scores - true_authenticity_scores
                average_residuals = predicted_average_scores - true_average_scores
                
                # Calculate statistical metrics for each score type
                quality_metrics = self._calculate_score_metrics(predicted_quality_scores, true_quality_scores)
                authenticity_metrics = self._calculate_score_metrics(predicted_authenticity_scores, true_authenticity_scores)
                average_metrics = self._calculate_score_metrics(predicted_average_scores, true_average_scores)
                
                # Structure results in a comprehensive dictionary
                evaluation_results = {
                    'predictions': {
                        'quality': predicted_quality_scores,
                        'authenticity': predicted_authenticity_scores,
                        'average': predicted_average_scores
                    },
                    'ground_truth': {
                        'quality': true_quality_scores,
                        'authenticity': true_authenticity_scores,
                        'average': true_average_scores
                    },
                    'residuals': {
                        'quality': quality_residuals,
                        'authenticity': authenticity_residuals,
                        'average': average_residuals
                    },
                    'metrics': {
                        'quality': quality_metrics,
                        'authenticity': authenticity_metrics,
                        'average': average_metrics
                    }
                }
            if all_predictions.shape[1] == 1:
                # Extract single score output
                predicted_authenticity_scores = all_predictions[:, 0]
                true_authenticity_scores = all_ground_truth[:, 0]
                
                # Compute residuals (prediction errors)
                authenticity_residuals = predicted_authenticity_scores - true_authenticity_scores
                
                # Calculate statistical metrics for authenticity score
                authenticity_metrics = self._calculate_score_metrics(predicted_authenticity_scores, true_authenticity_scores)
                
                # Structure results in a comprehensive dictionary
                evaluation_results = {
                    'predictions': {
                        'authenticity': predicted_authenticity_scores
                    },
                    'ground_truth': {
                        'authenticity': true_authenticity_scores
                    },
                    'residuals': {
                        'authenticity': authenticity_residuals
                    },
                    'metrics': {
                        'authenticity': authenticity_metrics
                    }
                }
            self.evaluation_results = evaluation_results
            return evaluation_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute evaluation metrics: {str(e)}")
    
    def compute_evaluation_metrics_from_arrays(self, 
                                         predictions: np.ndarray, 
                                         ground_truth: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics from prediction and ground truth arrays.
        
        Args:
            predictions: Array of model predictions with shape (n_samples, n_outputs)
            ground_truth: Array of ground truth labels with shape (n_samples, n_outputs)
            
        Returns:
            Dictionary containing predictions, ground truth, residuals, and metrics
            
        Raises:
            ValueError: If arrays don't have the expected shape or don't match
        """
        try:
            # Convert to numpy arrays if they aren't already
            all_predictions = np.array(predictions)
            all_ground_truth = np.array(ground_truth)
            
            # Ensure 2D arrays
            if all_predictions.ndim == 1:
                all_predictions = all_predictions.reshape(-1, 1)
            if all_ground_truth.ndim == 1:
                all_ground_truth = all_ground_truth.reshape(-1, 1)
            
            # Check if predictions have one or two score outputs
            if all_predictions.shape[1] not in [1, 2]:
                raise ValueError("Predictions must have either 1 (authenticity) or 2 (quality and authenticity) scores per sample.")
            if all_ground_truth.shape[1] not in [1, 2]:
                raise ValueError("Ground truth labels must have either 1 (authenticity) or 2 (quality and authenticity) scores per sample.")
            
            # Ensure predictions and ground truth have the same shape
            if all_predictions.shape != all_ground_truth.shape:
                raise ValueError(f"Shape mismatch: predictions {all_predictions.shape} vs ground truth {all_ground_truth.shape}")
            
            # Rest of the logic is the same as the original method
            evaluation_results = {}
            
            if all_predictions.shape[1] == 2:
                # Extract individual score components
                predicted_quality_scores = all_predictions[:, 0]
                true_quality_scores = all_ground_truth[:, 0]
                predicted_authenticity_scores = all_predictions[:, 1]
                true_authenticity_scores = all_ground_truth[:, 1]
                
                # Calculate averaged scores
                true_average_scores = (true_quality_scores + true_authenticity_scores) / 2
                predicted_average_scores = (predicted_quality_scores + predicted_authenticity_scores) / 2
                
                # Compute residuals (prediction errors)
                quality_residuals = predicted_quality_scores - true_quality_scores
                authenticity_residuals = predicted_authenticity_scores - true_authenticity_scores
                average_residuals = predicted_average_scores - true_average_scores
                
                # Calculate statistical metrics for each score type
                quality_metrics = self._calculate_score_metrics(predicted_quality_scores, true_quality_scores)
                authenticity_metrics = self._calculate_score_metrics(predicted_authenticity_scores, true_authenticity_scores)
                average_metrics = self._calculate_score_metrics(predicted_average_scores, true_average_scores)
                
                # Structure results in a comprehensive dictionary
                evaluation_results = {
                    'predictions': {
                        'quality': predicted_quality_scores,
                        'authenticity': predicted_authenticity_scores,
                        'average': predicted_average_scores
                    },
                    'ground_truth': {
                        'quality': true_quality_scores,
                        'authenticity': true_authenticity_scores,
                        'average': true_average_scores
                    },
                    'residuals': {
                        'quality': quality_residuals,
                        'authenticity': authenticity_residuals,
                        'average': average_residuals
                    },
                    'metrics': {
                        'quality': quality_metrics,
                        'authenticity': authenticity_metrics,
                        'average': average_metrics
                    }
                }
            
            if all_predictions.shape[1] == 1:
                # Extract single score output
                predicted_authenticity_scores = all_predictions[:, 0]
                true_authenticity_scores = all_ground_truth[:, 0]
                
                # Compute residuals (prediction errors)
                authenticity_residuals = predicted_authenticity_scores - true_authenticity_scores
                
                # Calculate statistical metrics for authenticity score
                authenticity_metrics = self._calculate_score_metrics(predicted_authenticity_scores, true_authenticity_scores)
                
                # Structure results in a comprehensive dictionary
                evaluation_results = {
                    'predictions': {
                        'authenticity': predicted_authenticity_scores
                    },
                    'ground_truth': {
                        'authenticity': true_authenticity_scores
                    },
                    'residuals': {
                        'authenticity': authenticity_residuals
                    },
                    'metrics': {
                        'authenticity': authenticity_metrics
                    }
                }
            
            self.evaluation_results = evaluation_results
            return evaluation_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute evaluation metrics from arrays: {str(e)}")

    def analyze_from_arrays(self, 
                        predictions: np.ndarray, 
                        ground_truth: np.ndarray,
                        save_path: Optional[str] = None, 
                        plot_title: str = "") -> Dict[str, Any]:
        """
        Convenience method that computes metrics from arrays and creates comprehensive visualization.
        
        Args:
            predictions: Array of model predictions
            ground_truth: Array of ground truth labels
            save_path: Path to save the plot. If None, displays the plot.
            plot_title: Title for the plot
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = self.compute_evaluation_metrics_from_arrays(predictions, ground_truth)
        self.create_comprehensive_visualization(results, save_path, plot_title)
        return results['metrics']

    def _calculate_score_metrics(self, predicted_scores: np.ndarray, true_scores: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical metrics for a pair of predicted and true scores.
        
        Args:
            predicted_scores: Array of predicted values
            true_scores: Array of true values
            
        Returns:
            Dictionary containing Spearman correlation, p-value, R², and RMSE
        """
        spearman_correlation, p_value = spearmanr(predicted_scores, true_scores)
        r_squared = r2_score(true_scores, predicted_scores)
        root_mean_squared_error = np.sqrt(mean_squared_error(true_scores, predicted_scores))
        
        return {
            'spearman': spearman_correlation,
            'p_value': p_value,
            'r2': r_squared,
            'rmse': root_mean_squared_error
        }
    
    def create_comprehensive_visualization(self, 
                                         results: Optional[Dict[str, Any]] = None,
                                         save_path: Optional[str] = None, 
                                         plot_title: str = "") -> None:
        """
        Create a comprehensive visualization with scatter plots, residuals, and histograms.
        
        Args:
            results: Evaluation results dictionary. If None, uses stored results.
            save_path: Path to save the plot. If None, displays the plot.
            plot_title: Title for the overall plot
            
        Raises:
            ValueError: If no results are available for plotting
        """
        # Use provided results or stored results
        plot_results = results or self.evaluation_results
        if plot_results is None:
            raise ValueError("No evaluation results available. Run compute_evaluation_metrics first.")
        
        # Extract data for plotting
        plot_data = self._extract_plot_data(plot_results)
        available_score_types = self._get_available_score_types(plot_results)
        num_scores = len(available_score_types)
        
        # Adjust figure size based on number of scores
        if num_scores == 1:
            figure_size = (8, 18)  # Narrower for single column
        else:
            figure_size = self.config.figure_size_correlation
        
        # Create figure with appropriate grid layout
        figure = plt.figure(figsize=figure_size)
        figure.suptitle(plot_title, fontsize=self.config.font_size_title)
        
        subplot_rows, subplot_cols = self._get_subplot_configuration(num_scores)
        grid_spec = gridspec.GridSpec(subplot_rows, subplot_cols, height_ratios=[1, 1, 1])
        
        # Create subplots for different visualization types
        scatter_axes = []
        residual_axes = []
        histogram_axes = []
        
        for i in range(num_scores):
            col_idx = i if num_scores > 1 else 0
            scatter_axes.append(plt.subplot(grid_spec[0, col_idx]))
            residual_axes.append(plt.subplot(grid_spec[1, col_idx]))
            histogram_axes.append(plt.subplot(grid_spec[2, col_idx]))
        
        # Generate plots for each available score type
        for index, score_type in enumerate(available_score_types):
            self._create_scatter_plot(
                scatter_axes[index], 
                plot_data['true_values'][score_type],
                plot_data['predicted_values'][score_type],
                plot_data['metrics'][score_type],
                score_type.title()
            )
            
            self._create_residual_plot(
                residual_axes[index],
                plot_data['predicted_values'][score_type],
                plot_data['residuals'][score_type],
                score_type.title()
            )
            
            self._create_residual_histogram(
                histogram_axes[index],
                plot_data['residuals'][score_type],
                score_type.title()
            )
        
        plt.tight_layout()
        self._save_or_show_plot(save_path)
        self._print_evaluation_summary(plot_results, plot_title)
    
    def create_residual_vs_true_plot(self, 
                                   results: Optional[Dict[str, Any]] = None,
                                   save_path: Optional[str] = None, 
                                   plot_title: str = "") -> None:
        """
        Create residual plots against true values.
        
        Args:
            results: Evaluation results dictionary. If None, uses stored results.
            save_path: Path to save the plot. If None, displays the plot.
            plot_title: Title for the plot
        """
        plot_results = results or self.evaluation_results
        if plot_results is None:
            raise ValueError("No evaluation results available. Run compute_evaluation_metrics first.")
        
        plot_data = self._extract_plot_data(plot_results)
        available_score_types = self._get_available_score_types(plot_results)
        num_scores = len(available_score_types)
        
        # Adjust figure size based on number of scores
        if num_scores == 1:
            figure_size = (6, 6)  # Square for single plot
        else:
            figure_size = self.config.figure_size_residuals
        
        figure, axes = plt.subplots(1, num_scores, figsize=figure_size)
        figure.suptitle(plot_title, fontsize=self.config.font_size_title)
        
        # Handle single subplot case (axes is not a list when num_scores=1)
        if num_scores == 1:
            axes = [axes]
        
        for index, score_type in enumerate(available_score_types):
            self._create_residual_vs_true_plot(
                axes[index],
                plot_data['true_values'][score_type],
                plot_data['residuals'][score_type],
                score_type.title()
            )
        
        plt.tight_layout()
        self._save_or_show_plot(save_path)
    
    def create_residual_vs_predicted_plot(self, 
                                        results: Optional[Dict[str, Any]] = None,
                                        save_path: Optional[str] = None, 
                                        plot_title: str = "") -> None:
        """
        Create residual plots against predicted values.
        
        Args:
            results: Evaluation results dictionary. If None, uses stored results.
            save_path: Path to save the plot. If None, displays the plot.
            plot_title: Title for the plot
        """
        plot_results = results or self.evaluation_results
        if plot_results is None:
            raise ValueError("No evaluation results available. Run compute_evaluation_metrics first.")
        
        plot_data = self._extract_plot_data(plot_results)
        available_score_types = self._get_available_score_types(plot_results)
        num_scores = len(available_score_types)
        
        # Adjust figure size based on number of scores
        if num_scores == 1:
            figure_size = (6, 6)  # Square for single plot
        else:
            figure_size = self.config.figure_size_residuals
        
        figure, axes = plt.subplots(1, num_scores, figsize=figure_size)
        figure.suptitle(plot_title, fontsize=self.config.font_size_title)
        
        # Handle single subplot case (axes is not a list when num_scores=1)
        if num_scores == 1:
            axes = [axes]
        
        for index, score_type in enumerate(available_score_types):
            self._create_residual_plot(
                axes[index],
                plot_data['predicted_values'][score_type],
                plot_data['residuals'][score_type],
                score_type.title()
            )
        
        plt.tight_layout()
        self._save_or_show_plot(save_path)
    
    def _get_available_score_types(self, results: Dict[str, Any]) -> List[str]:
        """
        Determine which score types are available in the results.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            List of available score types (e.g., ['authenticity'] or ['quality', 'authenticity', 'average'])
        """
        return list(results['predictions'].keys())
    
    def _get_subplot_configuration(self, num_scores: int) -> Tuple[int, int]:
        """
        Get subplot configuration based on number of score types.
        
        Args:
            num_scores: Number of score types to plot
            
        Returns:
            Tuple of (rows, columns) for subplot grid
        """
        if num_scores == 1:
            return (3, 1)  # 3 rows, 1 column for single score
        else:
            return (3, 3)  # 3 rows, 3 columns for multiple scores
    
    def _extract_plot_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize data for plotting from results dictionary."""
        return {
            'predicted_values': results['predictions'],
            'true_values': results['ground_truth'],
            'residuals': results['residuals'],
            'metrics': results['metrics']
        }
    
    def _create_scatter_plot(self, 
                           axis: plt.Axes, 
                           true_values: np.ndarray, 
                           predicted_values: np.ndarray, 
                           metrics: Dict[str, float], 
                           score_label: str) -> None:
        """Create a scatter plot with correlation analysis."""
        # Plot data points
        axis.scatter(true_values, predicted_values, 
                    alpha=self.config.scatter_alpha, 
                    color=self.config.scatter_color)
        
        # Add perfect prediction line
        min_value = min(min(true_values), min(predicted_values))
        max_value = max(max(true_values), max(predicted_values))
        axis.plot([min_value, max_value], [min_value, max_value], 
                 'r--', label='Perfect prediction')
        
        # Add regression line
        regression_coefficients = np.polyfit(true_values, predicted_values, 1)
        regression_polynomial = np.poly1d(regression_coefficients)
        sorted_true_values = np.sort(true_values)
        axis.plot(sorted_true_values, regression_polynomial(sorted_true_values), 
                 'g-', label=f'Best fit (y = {regression_coefficients[0]:.3f}x + {regression_coefficients[1]:.3f})')
        
        # Configure axis properties
        axis.set_xlabel('True Value', fontsize=self.config.font_size_labels)
        axis.set_ylabel('Predicted Value', fontsize=self.config.font_size_labels)
        axis.set_title(f'{score_label}\nSpearman ρ = {metrics["spearman"]:.4f}, '
                      f'R² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
        axis.grid(alpha=self.config.grid_alpha)
        axis.legend(loc='upper left')
    
    def _create_residual_plot(self, 
                            axis: plt.Axes, 
                            predicted_values: np.ndarray, 
                            residuals: np.ndarray, 
                            score_label: str) -> None:
        """Create a residual plot against predicted values."""
        axis.scatter(predicted_values, residuals, 
                    alpha=self.config.scatter_alpha, 
                    color=self.config.residual_color)
        axis.axhline(y=0, color='r', linestyle='--')
        axis.set_xlabel('Predicted Value', fontsize=self.config.font_size_labels)
        axis.set_ylabel('Residual (Pred - True)', fontsize=self.config.font_size_labels)
        axis.set_title(f'{score_label} Residuals vs Predicted')
        axis.grid(alpha=self.config.grid_alpha)
    
    def _create_residual_vs_true_plot(self, 
                                    axis: plt.Axes, 
                                    true_values: np.ndarray, 
                                    residuals: np.ndarray, 
                                    score_label: str) -> None:
        """Create a residual plot against true values."""
        axis.scatter(true_values, residuals, 
                    alpha=self.config.scatter_alpha, 
                    color=self.config.scatter_color)
        axis.axhline(y=0, color='r', linestyle='--')
        axis.set_xlabel(f'{score_label} True Value', fontsize=self.config.font_size_labels)
        axis.set_ylabel(f'{score_label} Residual (Pred - True)', fontsize=self.config.font_size_labels)
        axis.set_title(f'{score_label} Residuals vs True')
        axis.grid(alpha=self.config.grid_alpha)
    
    def _create_residual_histogram(self, 
                                 axis: plt.Axes, 
                                 residuals: np.ndarray, 
                                 score_label: str) -> None:
        """Create a histogram of residuals with statistical annotations."""
        axis.hist(residuals, bins=self.config.histogram_bins, 
                 alpha=self.config.scatter_alpha, 
                 color=self.config.histogram_color)
        axis.axvline(x=0, color='r', linestyle='--')
        axis.set_xlabel('Residual Value', fontsize=self.config.font_size_labels)
        axis.set_ylabel('Frequency', fontsize=self.config.font_size_labels)
        axis.set_title(f'{score_label} Residual Distribution')
        
        # Add statistical annotations
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        axis.text(0.05, 0.95, f'Mean: {residual_mean:.4f}\nStd: {residual_std:.4f}', 
                 transform=axis.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def _save_or_show_plot(self, save_path: Optional[str]) -> None:
        """Save plot to file or display it."""
        if save_path:
            # Create directory if it doesn't exist and path contains a directory
            dir_path = os.path.dirname(save_path)
            if dir_path:  # Only create directory if there is one
                os.makedirs(dir_path, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        else:
            plt.show()
    
    def _print_evaluation_summary(self, results: Dict[str, Any], title: str) -> None:
        """Print a summary of evaluation metrics."""
        available_score_types = self._get_available_score_types(results)
        
        print("-" * 50)
        print(f"MODEL EVALUATION: {title}")
        for score_type in available_score_types:
            metrics = results['metrics'][score_type]
            score_label = score_type.title()
            print(f"{score_label} Score - Spearman ρ: {metrics['spearman']:.4f} "
                  f"(p-value: {metrics['p_value']:.4g}), "
                  f"R²: {metrics['r2']:.4f}, "
                  f"RMSE: {metrics['rmse']:.4f}")
    
    def evaluate_and_visualize(self, 
                             model: torch.nn.Module, 
                             dataloader: torch.utils.data.DataLoader, 
                             device: str,
                             save_path: Optional[str] = None, 
                             plot_title: str = "") -> Dict[str, Any]:
        """
        Convenience method that computes metrics and creates comprehensive visualization.
        
        Args:
            model: The trained PyTorch model to evaluate
            dataloader: Test dataloader containing images and true scores
            device: Device to run the model on ('cuda' or 'cpu')
            save_path: Path to save the plot. If None, displays the plot.
            plot_title: Title for the plot
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = self.compute_evaluation_metrics(model, dataloader, device)
        self.create_comprehensive_visualization(results, save_path, plot_title)
        return results['metrics']


