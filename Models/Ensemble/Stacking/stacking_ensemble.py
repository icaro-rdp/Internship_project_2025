import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
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
from Models.Ensemble.utils import test_model, train_model, get_predictions


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
    (BarlowTwinsAuthenticityPredictor, 'Models/BarlowTwins/Weights/BarlowTwins_real_authenticity_finetuned.pth'),
    (EfficientNetB3AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/EfficientNetB3_weights.pth'),
    (DenseNet161AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/DenseNet161_weights.pth'),
    (ResNet152AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/ResNet152_weights.pth'),
    (VGG16AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/VGG16_weights.pth'),
    (VGG19AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/VGG19_weights.pth'),
    (InceptionV3AuthenticityPredictor, 'Models/Ensemble/Weights/Stacking/InceptionV3_weights.pth'),
]

# Load all models with their weights
models = []
for model_class, weights_path in model_configs:
    model = load_model_with_weights(model_class, weights_path)
    models.append(model)

# Define constants for K-Fold and DataLoaders
N_SPLITS = 5
BATCH_SIZE = 64
NUM_WORKERS = 20
RANDOM_STATE = 42 # For reproducibility

# --- Generate Out-of-Fold (OOF) predictions for meta-learner training ---

main_train_subset = IMAGENET_DATASET['train']
main_original_dataset = main_train_subset.dataset 
main_subset_indices = main_train_subset.indices

num_total_samples = len(main_subset_indices)

y_train_full_numpy = main_original_dataset.data.iloc[main_subset_indices, 1].values
y_train_full_tensor = torch.tensor(y_train_full_numpy, dtype=torch.float)

# 3. Initialize placeholders for OOF predictions and labels
# oof_predictions_all_models will store predictions: rows are samples, columns are models
oof_predictions_all_models = torch.zeros(num_total_samples, len(models), device='cpu')
oof_labels = torch.zeros(num_total_samples, device='cpu')

# 4. Initialize KFold
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print(f"Starting K-Fold cross-validation with {N_SPLITS} splits...")
# 5. Iterate through K-Folds
# train_fold_local_idx, val_fold_local_idx are indices *within* our main_train_subset (0 to num_total_samples-1)
for fold_num, (train_fold_local_idx, val_fold_local_idx) in enumerate(kf.split(np.arange(num_total_samples))):
    print(f"Processing Fold {fold_num + 1}/{N_SPLITS}...")
    
    # Get the actual global indices for the validation fold samples
    # These indices refer to the positions in main_original_dataset.data
    current_val_global_indices = [main_subset_indices[i] for i in val_fold_local_idx]
    
    # Store the true labels for this validation fold
    oof_labels[val_fold_local_idx] = y_train_full_tensor[val_fold_local_idx]
    
    # For each base model, get predictions on this validation fold
    for model_idx, model_instance in enumerate(models):
        model_name = model_instance.__class__.__name__
        # print(f"  Getting predictions from {model_name} for fold {fold_num + 1}...")
        
        # Determine which original dataset this model uses
        if model_name in ['DenseNet161AuthenticityPredictor', 'InceptionV3AuthenticityPredictor']:
            # These models use DENSENET_DATASET transformations
            original_full_dataset_for_model = DENSENET_DATASET['train'].dataset
        else:
            # Other models use IMAGENET_DATASET transformations
            original_full_dataset_for_model = IMAGENET_DATASET['train'].dataset
            
        # Create a Subset for the current validation fold using global indices
        # This ensures the correct items are fetched with the model-specific transformations
        val_fold_subset = Subset(original_full_dataset_for_model, current_val_global_indices)
        val_fold_dataloader = DataLoader(val_fold_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # Get predictions (ensure get_predictions returns a CPU tensor)
        fold_model_preds = get_predictions(model_instance, val_fold_dataloader) # Expected shape: (len(val_fold_local_idx), 1) or (len(val_fold_local_idx),)
        
        # Store predictions
        oof_predictions_all_models[val_fold_local_idx, model_idx] = fold_model_preds.squeeze().cpu()

print("K-Fold cross-validation finished.")

combined_predictions = oof_predictions_all_models
labels_tensor = oof_labels

# Define the training and validation datasets for the meta-learner
train_dataset = torch.utils.data.TensorDataset(combined_predictions, labels_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Define the stacking model
class StackingEnsemble(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StackingEnsemble, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)
# Initialize the stacking model
stacking_model = StackingEnsemble(input_size=combined_predictions.shape[1], num_classes=1)
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(stacking_model.parameters(), lr=0.001)

# Define the training and validation functions
def train_stacking_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
# Train the stacking model
train_stacking_model(stacking_model, train_dataloader, criterion, optimizer, num_epochs=20)

# ---- TESTING THE STACKED MODEL ----
print("\n--- Testing Stacking Model ---")

# Ensure to import metrics if not already done at the top
from sklearn.metrics import mean_squared_error # Removed other classification metrics

# 0. Define device (if not already globally defined and used for stacking_model training)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Ensure the trained stacking_model is on the correct device for testing
stacking_model = stacking_model.to(device)
stacking_model.eval() # Set to evaluation mode


y_test_true_tensor = None
num_test_samples = 0
main_test_subset_indices_for_all = None 

if 'test' not in IMAGENET_DATASET or IMAGENET_DATASET['test'] is None:
    print("Test dataset IMAGENET_DATASET['test'] not found. Skipping testing.")
else:
    if isinstance(IMAGENET_DATASET['test'], Subset):
        main_test_subset_ref = IMAGENET_DATASET['test']
        main_original_test_dataset_ref = main_test_subset_ref.dataset
        main_test_subset_indices_for_all = main_test_subset_ref.indices
        num_test_samples = len(main_test_subset_indices_for_all)
        
        # Example: Extracting true labels (modify if your dataset stores labels differently)
        try:
            # Attempting pandas-style access first, as in your training data
            y_test_true_numpy = main_original_test_dataset_ref.data.iloc[main_test_subset_indices_for_all, 1].values
            y_test_true_tensor = torch.tensor(y_test_true_numpy, dtype=torch.float)
        except AttributeError:
            # Fallback if .data or .iloc is not available (e.g., standard torchvision dataset)
            print("Pandas-style label extraction failed for test set, trying DataLoader method.")
            y_test_true_list = []
            temp_label_loader = DataLoader(main_test_subset_ref, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            for _, labels_batch in temp_label_loader:
                y_test_true_list.extend(labels_batch.tolist())
            y_test_true_tensor = torch.tensor(y_test_true_list, dtype=torch.float)

    else: # Assuming IMAGENET_DATASET['test'] is a full Dataset object
        main_test_dataset_ref = IMAGENET_DATASET['test']
        num_test_samples = len(main_test_dataset_ref)
        # No main_test_subset_indices_for_all here, assumes full dataset is used by all or DENSENET_DATASET['test'] is also full and aligned
        
        y_test_true_list = []
        temp_label_loader = DataLoader(main_test_dataset_ref, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        for _, labels_batch in temp_label_loader:
            y_test_true_list.extend(labels_batch.tolist())
        y_test_true_tensor = torch.tensor(y_test_true_list, dtype=torch.float)

    if num_test_samples > 0:
        print(f"Number of test samples: {num_test_samples}")

        # 2. Get Predictions from Base Models on the Test Set
        test_predictions_all_models = torch.zeros(num_test_samples, len(models), device='cpu') # Store on CPU

        print("Getting predictions from base models on the test set...")
        for model_idx, model_instance in enumerate(models):
            model_instance.to(device) # Ensure model is on the correct device
            model_instance.eval()
            model_name = model_instance.__class__.__name__
            print(f"  Getting predictions from {model_name} for test set...")

            current_test_dataloader_for_model = None
            if model_name in ['DenseNet161AuthenticityPredictor', 'InceptionV3AuthenticityPredictor']:
                if 'test' not in DENSENET_DATASET or DENSENET_DATASET['test'] is None:
                    print(f"Test dataset for {model_name} not found. Skipping model.")
                    test_predictions_all_models[:, model_idx] = torch.nan # Mark as NaN if dataset missing
                    continue
                
                dataset_to_use_for_model = DENSENET_DATASET['test'].dataset if isinstance(DENSENET_DATASET['test'], Subset) else DENSENET_DATASET['test']
                if main_test_subset_indices_for_all is not None and isinstance(DENSENET_DATASET['test'], Subset): # Use consistent indices if primary test set was a subset
                    final_subset_for_model = Subset(dataset_to_use_for_model, main_test_subset_indices_for_all)
                else: # Use the full DENSENET_DATASET['test'] or its original dataset if it was a Subset (without re-subsetting)
                    final_subset_for_model = DENSENET_DATASET['test']

                current_test_dataloader_for_model = DataLoader(final_subset_for_model, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            else:
                
                dataset_to_use_for_model = main_test_subset_ref if isinstance(IMAGENET_DATASET['test'], Subset) else main_test_dataset_ref
                current_test_dataloader_for_model = DataLoader(dataset_to_use_for_model, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
            # Using get_predictions as it's used in your OOF generation
            # Ensure your get_predictions function handles device internally or accepts `device`
            # model_test_preds = get_predictions(model_instance, current_test_dataloader_for_model, device) # If get_predictions accepts device
            model_test_preds = get_predictions(model_instance, current_test_dataloader_for_model) # As per your current usage in OOF

            test_predictions_all_models[:, model_idx] = model_test_preds.squeeze().cpu()

        print("Base model predictions for test set collected.")

        # 3. Prepare Meta-Learner Input and Get Final Predictions
        X_meta_test = test_predictions_all_models.to(device)
        final_predictions_on_test = None

        with torch.no_grad():
            meta_test_outputs = stacking_model(X_meta_test)
            final_predictions_on_test = meta_test_outputs.squeeze().cpu()

        print("Final predictions from stacking model obtained for test set.")

        # 4. Evaluate Performance
        y_test_true_numpy = y_test_true_tensor.cpu().numpy() 
        final_predictions_numpy = final_predictions_on_test.numpy() # Continuous scores

        print(f"\nStacking Model - Test Performance:")

        # Calculate and print MSE on the test scores
        mse_test = mean_squared_error(y_test_true_numpy, final_predictions_numpy)
        r_squared_test = 1 - (mse_test / np.var(y_test_true_numpy)) if np.var(y_test_true_numpy) > 0 else float('nan')
        print(f"  MSE on test scores: {mse_test:.4f}")
        print(f" RMSE on test scores: {np.sqrt(mse_test):.4f}")
        print(f"R^2 on test scores: {r_squared_test:.4f}")
        # Optionally, you can also calculate R^2 or other metrics if needed
    else:
        print("No test samples found or processed. Skipping evaluation.")

# Optional: Save the trained stacking model
torch.save(stacking_model.state_dict(), 'stacking_ensemble_final.pth')
print("\nTrained stacking model state_dict saved to stacking_ensemble_final.pth")
