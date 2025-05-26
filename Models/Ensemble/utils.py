import torch
import torch.nn as nn
import time
from typing import Dict, Optional, Tuple, Any
from tqdm.auto import tqdm
import os


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a readable string.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 0: # Handle potential negative ETA briefly during startup
        return "calculating..."
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"

def train_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 20,
    device: str = 'cuda',
    test_dataloader: Optional[torch.utils.data.DataLoader] = None,
    save_best_test_model: Optional[bool] = False,
    save_model: Optional[bool] = True,
    path_to_save: Optional[str] = None,
    model_name: str = "model"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains the model with comprehensive progress tracking and timing using tqdm.

    Args:
        model (nn.Module): The model to train
        train_dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of epochs to train. Defaults to 20
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'
        test_dataloader (DataLoader, optional): Test data loader for validation
        save_best_test_model (bool): Whether to save the best model based on test loss
        save_model (bool): Whether to save the final model. Defaults to True
        model_name (str): Name for model identification in logs and tqdm

    Returns:
        Tuple[nn.Module, Dict]: Trained model and training statistics
    """
    print(f"\n{'='*60}")
    print(f"Starting training for {model_name}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Training batches: {len(train_dataloader)}")
    if test_dataloader:
        print(f"Test samples: {len(test_dataloader.dataset)}")
        print(f"Test batches: {len(test_dataloader)}")
    print(f"{'='*60}\n")

    model.to(device)

    training_stats = {
        'train_losses': [],
        'test_losses': [],
        'epoch_times': [],
        'total_training_time': 0,
        'best_test_loss': float('inf'),
        'best_epoch': 0
    }

    total_training_start_time = time.time()
    best_model_state = None

    epoch_iterator = tqdm(range(num_epochs), desc=f"Training '{model_name}'", unit="epoch")

    for epoch in epoch_iterator:
        epoch_start_time = time.time()

        # ================== TRAINING PHASE ==================
        model.train()
        running_train_loss = 0.0
        total_train_samples_epoch = 0

        train_batch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
            unit="batch",
            leave=False
        )

        for batch_idx, (inputs, labels) in enumerate(train_batch_iterator):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            batch_loss = loss.item() * batch_size

            running_train_loss += batch_loss
            total_train_samples_epoch += batch_size

            if total_train_samples_epoch > 0:
                current_avg_loss = running_train_loss / total_train_samples_epoch
                train_batch_iterator.set_postfix(loss=f"{current_avg_loss:.4f}")

        epoch_train_loss = running_train_loss / total_train_samples_epoch if total_train_samples_epoch > 0 else 0
        training_stats['train_losses'].append(epoch_train_loss)

        # ================== TESTING PHASE ==================
        epoch_test_loss = None

        if test_dataloader is not None:
            model.eval()
            running_test_loss = 0.0
            total_test_samples_epoch = 0

            test_batch_iterator = tqdm(
                test_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                unit="batch",
                leave=False
            )

            with torch.no_grad():
                for inputs_val, labels_val in test_batch_iterator:
                    inputs_val = inputs_val.to(device, non_blocking=True)
                    labels_val = labels_val.to(device, non_blocking=True)

                    outputs_val, _ = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)

                    batch_size_val = inputs_val.size(0)
                    running_test_loss += loss_val.item() * batch_size_val
                    total_test_samples_epoch += batch_size_val

                    if total_test_samples_epoch > 0:
                        current_avg_val_loss = running_test_loss / total_test_samples_epoch
                        test_batch_iterator.set_postfix(loss=f"{current_avg_val_loss:.4f}")

            epoch_test_loss = running_test_loss / total_test_samples_epoch if total_test_samples_epoch > 0 else 0
            training_stats['test_losses'].append(epoch_test_loss)

            if save_best_test_model and epoch_test_loss is not None and epoch_test_loss < training_stats['best_test_loss']:
                training_stats['best_test_loss'] = epoch_test_loss
                training_stats['best_epoch'] = epoch + 1
                best_model_state = model.state_dict().copy()

        epoch_time = time.time() - epoch_start_time
        training_stats['epoch_times'].append(epoch_time)

        # Update outer epoch iterator's postfix
        postfix_data = {
            'TrainLoss': f"{epoch_train_loss:.4f}",
        }
        if epoch_test_loss is not None:
            postfix_data['TestLoss'] = f"{epoch_test_loss:.4f}"
        postfix_data['Time'] = format_time(epoch_time)
        epoch_iterator.set_postfix(postfix_data)

        # Log epoch summary
        tqdm.write(f"\nEpoch {epoch + 1} Summary:")
        tqdm.write(f"  Train Loss: {epoch_train_loss:.4f}")
        if epoch_test_loss is not None:
            tqdm.write(f"  Test Loss:  {epoch_test_loss:.4f}")
        tqdm.write(f"  Epoch Time: {format_time(epoch_time)}")
        if epoch < num_epochs - 1:
            avg_epoch_time = sum(training_stats['epoch_times']) / len(training_stats['epoch_times']) if training_stats['epoch_times'] else 0
            remaining_epochs = num_epochs - (epoch + 1)
            eta_total = remaining_epochs * avg_epoch_time
            tqdm.write(f"  Overall Training ETA: {format_time(eta_total)}")
        tqdm.write("-" * 50)

    total_training_time = time.time() - total_training_start_time
    training_stats['total_training_time'] = total_training_time

    print(f"\n{'='*60}")
    print(f"Training completed for {model_name}")
    print(f"Total Training Time: {format_time(total_training_time)}")
    if num_epochs > 0:
        print(f"Average Epoch Time: {format_time(total_training_time / num_epochs)}")
    if training_stats['train_losses']:
        print(f"Final Train Loss: {training_stats['train_losses'][-1]:.4f}")

    if test_dataloader is not None and training_stats['test_losses']:
        print(f"Final Test Loss: {training_stats['test_losses'][-1]:.4f}")
        
        if save_best_test_model and training_stats['best_epoch'] > 0:
            print(f"Best Test Loss: {training_stats['best_test_loss']:.4f} (at Epoch {training_stats['best_epoch']})")
    print(f"{'='*60}\n")

    if save_best_test_model and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state from epoch {training_stats['best_epoch']}")

    # Save model at the end of training
    if save_model:
        path_to_save = path_to_save if path_to_save else "."
        model_name = model_name.replace(" ", "_")
        # check if the path exists
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        model_name = os.path.join(path_to_save, model_name)
        # Save the final model
        model_filename = f"{model_name}_weights.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Final model saved as: {model_filename}")
        
        if save_best_test_model and best_model_state is not None:
            best_model_filename = f"{model_name}_best.pth"
            torch.save(best_model_state, best_model_filename)
            print(f"Best model saved as: {best_model_filename}")

    return model, training_stats

def test_model(
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> float:
    """
    Tests the model on the provided test dataloader.

    Args:
        model (nn.Module): The model to test
        test_dataloader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'

    Returns:
        float: Average loss on the test set
    """
    model.to(device)
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing", unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)

    average_loss = running_loss / total_samples if total_samples > 0 else 0.0

    print(f"Test Loss: {average_loss:.4f}")
    return average_loss

def test_ensemble(
    ensemble_predictor,
    test_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> float:
    """
    Tests the ensemble model on the provided test dataloader.

    Args:
        ensemble_predictor: The ensemble predictor instance
        test_dataloader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'

    Returns:
        float: Average loss on the test set
    """
    # Move all models in ensemble to device
    for model in ensemble_predictor.models:
        model.to(device)
        model.eval()

    running_loss = 0.0
    total_samples = 0

    print(f"\n{'='*60}")
    print(f"Testing Ensemble Model")
    print(f"Number of models in ensemble: {len(ensemble_predictor.models)}")
    print(f"Test samples: {len(test_dataloader.dataset)}")
    print(f"Test batches: {len(test_dataloader)}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing Ensemble", unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Get ensemble predictions
            ensemble_outputs = ensemble_predictor.predict(inputs)
            loss = criterion(ensemble_outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)

    average_loss = running_loss / total_samples if total_samples > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Ensemble Test Results:")
    print(f"Test Loss: {average_loss:.4f}")
    print(f"{'='*60}\n")

    return average_loss

def get_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Get predictions from the model for the given dataloader.

    Args:
        model (nn.Module): The model to use for predictions
        dataloader (DataLoader): DataLoader containing the data
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'

    Returns:
        torch.Tensor: Predictions from the model
    """
    model.to(device)
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Getting Predictions", unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            outputs, _ = model(inputs)
            all_predictions.append(outputs.cpu())

    return torch.cat(all_predictions, dim=0)
