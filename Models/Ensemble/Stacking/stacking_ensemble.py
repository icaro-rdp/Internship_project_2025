import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import random
from pathlib import Path
import logging

# --- Model and Dataset Imports (Assuming these are in the specified paths) ---
# These would need to be resolvable in your environment
# For this example, I'll define dummy classes/dicts if they aren't found
try:
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
except ImportError:
    logging.warning("Could not import all custom models/datasets/utils. Using dummy placeholders.")
    # Dummy placeholders for demonstration if imports fail
    class DummyModel(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Linear(10,1)
        def forward(self, x): return torch.rand(x.size(0) if isinstance(x, torch.Tensor) else x[0].size(0), 1) # Simplified
    BarlowTwinsAuthenticityPredictor = EfficientNetB3AuthenticityPredictor = DenseNet161AuthenticityPredictor = \
    ResNet152AuthenticityPredictor = VGG16AuthenticityPredictor = VGG19AuthenticityPredictor = \
    InceptionV3AuthenticityPredictor = DummyModel

    # Dummy dataset structure
    def _create_dummy_dataset(size=100, num_classes=2):
        # Ensuring data has .iloc for consistency with original code's access pattern
        import pandas as pd
        data = pd.DataFrame({
            'features': [torch.randn(3, 224, 224) for _ in range(size)],
            'labels': np.random.randint(0, num_classes, size)
        })
        class DummyTorchDataset(torch.utils.data.Dataset):
            def __init__(self, data_df): self.data = data_df
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data.iloc[idx, 0], self.data.iloc[idx, 1]
        
        dummy_torch_ds = DummyTorchDataset(data)
        return {
            'train': Subset(dummy_torch_ds, list(range(size // 2))),
            'test': Subset(dummy_torch_ds, list(range(size // 2, size))),
            'dataset_object': dummy_torch_ds # Reference to the full original dataset
        }
    IMAGENET_DATASET = _create_dummy_dataset(200) # Larger for KFold
    DENSENET_DATASET = _create_dummy_dataset(200)

    def get_predictions(model: nn.Module, dataloader: DataLoader, device: str = 'cpu') -> torch.Tensor:
        model.eval()
        model.to(device)
        all_preds = []
        with torch.no_grad():
            for inputs, _ in dataloader: # Assuming labels are not needed for preds for now
                if isinstance(inputs, list): inputs = inputs[0] # If dataloader yields list
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_preds.append(outputs.cpu())
        return torch.cat(all_preds).squeeze()


# --- Configuration ---
class Config:
    N_SPLITS = 7
    BATCH_SIZE = 32 
    NUM_WORKERS = 20  
    RANDOM_STATE = 42
    LEARNING_RATE_META = 0.001
    EPOCHS_META = 40 # Original value
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    BASE_MODEL_WEIGHTS_DIR = Path('Models')
    OUTPUT_DIR = Path('.') # Current directory for output
    META_LEARNER_SAVE_PATH = OUTPUT_DIR / 'Models/Ensemble/Weights/Stacking/stacking_ensemble_weights.pth'

    # Model configurations: (ModelClass, weights_filename, dataset_dict_to_use)
    MODEL_CONFIGS = [
        (BarlowTwinsAuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'BarlowTwins/Weights/BarlowTwins_real_authenticity_finetuned.pth', IMAGENET_DATASET),
        (EfficientNetB3AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/EfficientNetB3_weights.pth', IMAGENET_DATASET),
        (DenseNet161AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/DenseNet161_weights.pth', DENSENET_DATASET),
        (ResNet152AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/ResNet152_weights.pth', IMAGENET_DATASET),
        (VGG16AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/VGG16_weights.pth', IMAGENET_DATASET),
        (VGG19AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/VGG19_weights.pth', IMAGENET_DATASET),
        (InceptionV3AuthenticityPredictor, BASE_MODEL_WEIGHTS_DIR / 'Ensemble/Weights/Stacking/InceptionV3_weights.pth', DENSENET_DATASET),
    ]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Reproducibility ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Helper Functions ---
def load_model_with_weights(model_class: type, weights_path: Path, device: str) -> nn.Module:
    """Loads a model and its weights onto the specified device."""
    model = model_class()
    try:
        if device == 'cuda' and torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
            model = model.to(device)
        else:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            model = model.to('cpu')
        logging.info(f"Loaded weights for {model_class.__name__} from {weights_path}")
    except FileNotFoundError:
        logging.warning(f"Weights file not found for {model_class.__name__} at {weights_path}. Using initialized model.")
    except Exception as e:
        logging.error(f"Error loading weights for {model_class.__name__} from {weights_path}: {e}. Using initialized model.")
    return model

# --- Stacking Ensemble Model ---
class StackingEnsemble(nn.Module):
    """A simple meta-learner with one linear layer."""
    def __init__(self, input_size: int, num_classes: int):
        super(StackingEnsemble, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def train_stacking_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader, # Added for meta-learner validation
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str
) -> None:
    """Trains the stacking model (meta-learner) with validation."""
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataloader.dataset)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_dataloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs.squeeze(), val_labels)
                val_running_loss += val_loss.item() * val_inputs.size(0)
        
        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

# --- Main Execution ---
def main():
    set_seed(Config.RANDOM_STATE) #

    # Load all base models with their weights
    loaded_models = []
    for model_class, weights_path, _ in Config.MODEL_CONFIGS: #
        model = load_model_with_weights(model_class, weights_path, Config.DEVICE) #
        loaded_models.append(model)

    # --- Generate Out-of-Fold (OOF) predictions for meta-learner training ---
    # Using IMAGENET_DATASET as the primary reference for subsetting, as in original code
    main_train_subset = IMAGENET_DATASET['train'] #
    # Handle if 'dataset' attribute doesn't exist, use the Subset's dataset directly
    # Or, if 'dataset_object' is the intended full dataset as per dummy.
    main_original_dataset = getattr(main_train_subset, 'dataset', None)
    if main_original_dataset is None and 'dataset_object' in IMAGENET_DATASET: # Fallback for dummy
        main_original_dataset = IMAGENET_DATASET['dataset_object']
    elif main_original_dataset is None:
        raise ValueError("Cannot determine the original dataset from IMAGENET_DATASET['train']")
        
    main_subset_indices = main_train_subset.indices #
    num_total_samples_oof = len(main_subset_indices) #

    # Extracting labels - ensure this matches your true dataset structure
    # Original code: main_original_dataset.data.iloc[main_subset_indices, 1].values
    # This requires main_original_dataset to have a .data attribute that is a pandas DataFrame
    try:
        y_train_full_numpy = main_original_dataset.data.iloc[main_subset_indices, 1].values #
    except (AttributeError, TypeError) as e:
        logging.warning(f"Failed to get labels via .data.iloc: {e}. Trying to extract from DataLoader (slower).")
        # Fallback: extract labels via DataLoader (slower, but more general)
        temp_label_loader = DataLoader(main_train_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        y_train_list = []
        for _, labels_batch in temp_label_loader:
            y_train_list.extend(labels_batch.tolist())
        y_train_full_numpy = np.array(y_train_list)

    y_train_full_tensor = torch.tensor(y_train_full_numpy, dtype=torch.float) #

    oof_predictions_all_models = torch.zeros(num_total_samples_oof, len(loaded_models), device='cpu') #
    oof_labels = torch.zeros(num_total_samples_oof, device='cpu') #

    kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE) #
    logging.info(f"Starting K-Fold cross-validation with {Config.N_SPLITS} splits...")

    for fold_num, (train_fold_local_idx, val_fold_local_idx) in enumerate(kf.split(np.arange(num_total_samples_oof))): #
        logging.info(f"Processing Fold {fold_num + 1}/{Config.N_SPLITS}...")
        
        current_val_global_indices = [main_subset_indices[i] for i in val_fold_local_idx] #
        oof_labels[val_fold_local_idx] = y_train_full_tensor[val_fold_local_idx] #
        
        for model_idx, model_instance in enumerate(loaded_models):
            _, _, model_dataset_dict = Config.MODEL_CONFIGS[model_idx] # Get the dataset dict for this model
            
            # Use 'dataset_object' if available (full dataset), else use .dataset from subset
            original_full_dataset_for_model = model_dataset_dict.get('dataset_object', model_dataset_dict['train'].dataset)
            
            val_fold_subset = Subset(original_full_dataset_for_model, current_val_global_indices) #
            val_fold_dataloader = DataLoader(val_fold_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS) #
            
            fold_model_preds = get_predictions(model_instance, val_fold_dataloader, Config.DEVICE) #
            oof_predictions_all_models[val_fold_local_idx, model_idx] = fold_model_preds.squeeze().cpu() #

    logging.info("K-Fold cross-validation finished.")

    # Split OOF predictions for meta-learner training and validation
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        oof_predictions_all_models, y_train_full_tensor,
        random_state=Config.RANDOM_STATE
    )

    train_meta_dataset = TensorDataset(X_meta_train, y_meta_train)
    val_meta_dataset = TensorDataset(X_meta_val, y_meta_val)

    train_meta_dataloader = DataLoader(train_meta_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_meta_dataloader = DataLoader(val_meta_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Initialize and train the stacking model (meta-learner)
    stacking_model = StackingEnsemble(input_size=oof_predictions_all_models.shape[1], num_classes=1) #
    criterion_meta = nn.MSELoss() #
    optimizer_meta = torch.optim.Adam(stacking_model.parameters(), lr=Config.LEARNING_RATE_META) #

    logging.info("Training the stacking model (meta-learner)...")
    train_stacking_model(stacking_model, train_meta_dataloader, val_meta_dataloader, criterion_meta, optimizer_meta, Config.EPOCHS_META, Config.DEVICE)

    # ---- TESTING THE STACKED MODEL ----
    logging.info("\n--- Testing Stacking Model ---")
    stacking_model.eval() #

    y_test_true_tensor = None
    num_test_samples = 0
    main_test_subset_indices_for_all = None

    primary_test_set_key = 'test' # Assuming 'test' is the key for test data

    if primary_test_set_key not in IMAGENET_DATASET or IMAGENET_DATASET[primary_test_set_key] is None: #
        logging.warning("Test dataset IMAGENET_DATASET['test'] not found. Skipping testing.")
    else:
        current_test_set_ref = IMAGENET_DATASET[primary_test_set_key]
        # Determine original dataset and indices for the primary test set (IMAGENET)
        if isinstance(current_test_set_ref, Subset): #
            main_test_subset_ref = current_test_set_ref
            main_original_test_dataset_ref = getattr(main_test_subset_ref, 'dataset', IMAGENET_DATASET.get('dataset_object'))
            main_test_subset_indices_for_all = main_test_subset_ref.indices #
            num_test_samples = len(main_test_subset_indices_for_all) #
        else: # Assuming it's a full Dataset object
            main_test_dataset_ref = current_test_set_ref
            main_original_test_dataset_ref = main_test_dataset_ref # It is the original itself
            num_test_samples = len(main_test_dataset_ref)
            main_test_subset_indices_for_all = list(range(num_test_samples))


        if num_test_samples > 0:
            logging.info(f"Number of test samples: {num_test_samples}")
            # Extract true labels for the test set
            try: #
                y_test_true_numpy = main_original_test_dataset_ref.data.iloc[main_test_subset_indices_for_all, 1].values #
                y_test_true_tensor = torch.tensor(y_test_true_numpy, dtype=torch.float) #
            except (AttributeError, TypeError) as e: #
                logging.warning(f"Pandas-style label extraction failed for test set: {e}. Trying DataLoader method.") #
                y_test_true_list = []
                # Use the appropriate subset/dataset for label extraction
                label_extraction_dataset = Subset(main_original_test_dataset_ref, main_test_subset_indices_for_all) if isinstance(current_test_set_ref, Subset) else main_test_dataset_ref
                temp_label_loader = DataLoader(label_extraction_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS) #
                for _, labels_batch in temp_label_loader: #
                    y_test_true_list.extend(labels_batch.tolist()) #
                y_test_true_tensor = torch.tensor(y_test_true_list, dtype=torch.float) #


            test_predictions_all_models = torch.zeros(num_test_samples, len(loaded_models), device='cpu') #
            logging.info("Getting predictions from base models on the test set...")

            for model_idx, model_instance in enumerate(loaded_models):
                model_instance.to(Config.DEVICE) #
                model_instance.eval() #
                model_class, _, model_dataset_dict = Config.MODEL_CONFIGS[model_idx] #
                model_name = model_class.__name__ #
                logging.info(f"  Getting predictions from {model_name} for test set...") #

                current_test_dataloader_for_model = None
                if primary_test_set_key not in model_dataset_dict or model_dataset_dict[primary_test_set_key] is None: #
                    logging.warning(f"Test dataset for {model_name} not found. Predictions will be NaN.") #
                    test_predictions_all_models[:, model_idx] = torch.nan #
                    continue
                
                # Determine the dataset to use for THIS model (could be IMAGENET or DENSENET based)
                specific_model_test_set_ref = model_dataset_dict[primary_test_set_key]
                original_full_test_dataset_for_model = getattr(specific_model_test_set_ref, 'dataset', model_dataset_dict.get('dataset_object'))

                
                final_subset_for_model_test = Subset(original_full_test_dataset_for_model, main_test_subset_indices_for_all)
                current_test_dataloader_for_model = DataLoader(final_subset_for_model_test, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS) #
                            
                model_test_preds = get_predictions(model_instance, current_test_dataloader_for_model, Config.DEVICE) #
                test_predictions_all_models[:, model_idx] = model_test_preds.squeeze().cpu() #

            logging.info("Base model predictions for test set collected.")

            X_meta_test = test_predictions_all_models.to(Config.DEVICE) #
            final_predictions_on_test = None
            with torch.no_grad(): #
                meta_test_outputs = stacking_model(X_meta_test) #
                final_predictions_on_test = meta_test_outputs.squeeze().cpu() #
            logging.info("Final predictions from stacking model obtained for test set.")

            y_test_true_numpy = y_test_true_tensor.cpu().numpy() #
            final_predictions_numpy = final_predictions_on_test.numpy() #

            logging.info(f"\nStacking Model - Test Performance:")
            
            # Handle potential NaNs in predictions before calculating metrics
            valid_indices = ~np.isnan(final_predictions_numpy) & ~np.isnan(y_test_true_numpy)
            if np.sum(valid_indices) == 0:
                logging.warning("No valid (non-NaN) predictions available for metric calculation.")
            else:
                y_test_true_numpy_clean = y_test_true_numpy[valid_indices]
                final_predictions_numpy_clean = final_predictions_numpy[valid_indices]

                if len(y_test_true_numpy_clean) > 0:
                    mse_test = mean_squared_error(y_test_true_numpy_clean, final_predictions_numpy_clean) # uses original sklearn
                    rmse_test = np.sqrt(mse_test)
                    
                    var_true = np.var(y_test_true_numpy_clean) # Using cleaned data for variance
                    r_squared_test = 1 - (mse_test / var_true) if var_true > 0 else float('nan') #
                    
                    logging.info(f"  MSE on test scores (NaNs excluded): {mse_test:.4f}") #
                    logging.info(f" RMSE on test scores (NaNs excluded): {rmse_test:.4f}") #
                    logging.info(f"R^2 on test scores (NaNs excluded): {r_squared_test:.4f}") #
                else:
                    logging.warning("No valid samples left after NaN cleaning for metric calculation.")
        else:
            logging.warning("No test samples found or processed. Skipping evaluation.") #

    # Save the trained stacking model
    torch.save(stacking_model.state_dict(), Config.META_LEARNER_SAVE_PATH) #
    logging.info(f"\nTrained stacking model state_dict saved to {Config.META_LEARNER_SAVE_PATH}")


if __name__ == "__main__":
    main()