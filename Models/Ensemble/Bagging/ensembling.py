import torch
import torch.nn as nn
from torch.utils.data import  DataLoader


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
from Models.Ensemble.utils import test_model, test_ensemble
from Models.Ensemble.descriptive_analysis import ModelEvaluationVisualizer
import math

# Define the Ensemble class
class EnsembleAuthenticityPredictor:
    """
    Ensemble model for authenticity prediction using bagging technique.
    This class combines multiple models to improve prediction accuracy.
    Supports two different datasets, mapping each model to its dataset.
    """
    def __init__(self, models, datasets, batch_size=32, num_workers=4):
        """
        Args:
            models (list): List of model instances.
            datasets (list): List of datasets, one for each model.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        assert len(models) == len(datasets), "Each model must have a corresponding dataset."
        self.models = models
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict(self, data):
        """
        Args:
            data: Input data (e.g., image tensor or batch).
        Returns:
            Tensor: Averaged predictions from all models.
        """
        predictions = []
        for model, dataset in zip(self.models, self.datasets):
            model.eval()
            with torch.no_grad():
                # Optionally, preprocess data according to dataset if needed
                output, _ = model(data)
                predictions.append(output)
        return torch.mean(torch.stack(predictions), dim=0)

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
    (BarlowTwinsAuthenticityPredictor, 'Models/Ensemble/Weights/BarlowTwins_weights.pth'),
    (EfficientNetB3AuthenticityPredictor, 'Models/Ensemble/Weights/EfficientNetB3_weights.pth'),
    (DenseNet161AuthenticityPredictor, 'Models/Ensemble/Weights/DenseNet161_weights.pth'),
    (ResNet152AuthenticityPredictor, 'Models/Ensemble/Weights/ResNet152_weights.pth'),
    (VGG16AuthenticityPredictor, 'Models/Ensemble/Weights/VGG16_weights.pth'),
    (VGG19AuthenticityPredictor, 'Models/Ensemble/Weights/VGG19_weights.pth'),
    (InceptionV3AuthenticityPredictor, 'Models/Ensemble/Weights/InceptionV3_weights.pth'),
]

# Define criterion for testing
criterion = nn.MSELoss()  # Changed back to MSELoss as requested
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load all models with their weights
ensemble_models = [load_model_with_weights(cls, path, device) for cls, path in model_configs]

# Create the ensemble predictor
ensemble_predictor = EnsembleAuthenticityPredictor(
    models=ensemble_models,
    datasets=[IMAGENET_DATASET['test'], DENSENET_DATASET['test'], IMAGENET_DATASET['test'], IMAGENET_DATASET['test'], IMAGENET_DATASET['test'], IMAGENET_DATASET['test'], DENSENET_DATASET['test']],
    batch_size=64,
    num_workers=20
)

# Test the models individually
print("Testing individual models:")
model_names = ['BarlowTwins', 'EfficientNetB3', 'DenseNet161', 'ResNet152', 'VGG16', 'VGG19', 'InceptionV3']

individual_results = []
for i, (model, dataset) in enumerate(zip(ensemble_models, ensemble_predictor.datasets)):
    print(f"\nTesting {model_names[i]}:")
    test_dataloader = DataLoader(dataset, batch_size=ensemble_predictor.batch_size, 
                                num_workers=ensemble_predictor.num_workers, shuffle=False)
    loss = test_model(model, test_dataloader, criterion, device)
    individual_results.append((model_names[i], loss))

# Test the ensemble model
print("\nTesting ensemble model:")
# Use the first dataset for ensemble testing (you can modify this based on your needs)
ensemble_test_dataloader = DataLoader(IMAGENET_DATASET['test'], 
                                     batch_size=ensemble_predictor.batch_size, 
                                     num_workers=ensemble_predictor.num_workers, 
                                     shuffle=False)

ensemble_loss, ensemble_predictions, labels = test_ensemble(ensemble_predictor, ensemble_test_dataloader, criterion, device)

# Print comparison results
print(f"\n{'='*60}")
print("RESULTS COMPARISON:")
print(f"{'='*60}")
print(f"{'Model':<15} {'MSE Loss':<12} {'RMSE':<10}")
print(f"{'-'*40}")
for name, loss in individual_results:
    rmse = math.sqrt(loss)
    print(f"{name:<15} {loss:<12.4f} {rmse:<10.4f}")
print(f"{'-'*40}")
ensemble_rmse = math.sqrt(ensemble_loss)
print(f"{'ENSEMBLE':<15} {ensemble_loss:<12.4f} {ensemble_rmse:<10.4f}")
print(f"{'='*60}")

# Calculate improvement
best_individual_loss = min([loss for _, loss in individual_results])
best_individual_name = [name for name, loss in individual_results if loss == best_individual_loss][0]
improvement = best_individual_loss - ensemble_loss

print(f"\nBest individual model: {best_individual_name} with {best_individual_loss:.4f} loss")
print(f"Ensemble loss: {ensemble_loss:.4f}")
print(f"Improvement: {improvement:.4f} ({improvement/best_individual_loss*100:.2f}% better)" if improvement > 0 else f"Degradation: {abs(improvement):.4f} ({abs(improvement)/best_individual_loss*100:.2f}% worse)")

# Visualize descriptive results using scatter plot

visualizer = ModelEvaluationVisualizer()
visualizer.analyze_from_arrays(
    predictions=ensemble_predictions.numpy(),
    ground_truth= labels.numpy(),
    save_path='bagging_model_scatterplot_ypred_ytrue.png',
    plot_title="Bagging Model pred vs True Labels",

    )


