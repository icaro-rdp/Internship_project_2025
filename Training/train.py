import copy
import torch
import matplotlib.pyplot as plt

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cuda', patience=5):
    """
    Trains the model with early stopping based on validation loss.

    Args:
        model (nn.Module): The model to train.
        dataloaders (dict): A dictionary containing 'train' and 'val' data loaders.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int): The maximum number of epochs to train for.
        device (str): Device to use for training ('cuda' or 'cpu').
        patience (int): How many epochs to wait for validation loss to improve before stopping.

    Returns:
        nn.Module: The model with the best performing weights on the validation set.
        dict: A dictionary containing the 'train_loss' and 'val_loss' history.
    """
    model.to(device)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')  # Set best validation loss to infinity
    patience_counter = 0          # Counter for epochs with no improvement
    
    # Store the best model weights
    best_model_wts = copy.deepcopy(model.state_dict()) 

    # Keep track of loss history for plotting
    history = {'train_loss': [], 'val_loss': []}

    print("Starting training...")

    # Loop over the specified maximum number of epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(f'{phase} phase')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0

            # Iterate over data in the current phase
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Enable gradient computation only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate the loss
                running_loss += loss.item() * inputs.size(0)

            # Calculate average loss for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Record loss history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)
                
                # --- Early Stopping Check ---
                # Check if the validation loss improved
                if epoch_loss < best_val_loss:
                    print(f'Validation loss decreased ({best_val_loss:.6f} --> {epoch_loss:.6f}). Saving model...')
                    best_val_loss = epoch_loss
                    # Save a deep copy of the model's weights
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0  # Reset patience counter
                else:
                    # If validation loss did not improve, increment counter
                    patience_counter += 1
                    print(f'Validation loss did not improve. Patience: {patience_counter}/{patience}')

        # Check if patience has been exceeded after each epoch
        if patience_counter >= patience:
            print('Early stopping triggered! Stopping training.')
            break  # Exit the main epoch loop

    # --- End of Training ---
    print("Finished Training.")
    
    # Load the best model weights found during training
    print(f'Loading best model weights with validation loss: {best_val_loss:.4f}')
    model.load_state_dict(best_model_wts)
    
    # Return the best model and the loss history
    return model, history

def test_model(model, dataloader, criterion, device='cuda'):
    """
    Tests the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): The test data loader.
        criterion (nn.Module): The loss function.
        device (str): Device to use for testing ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        float: The average loss on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    running_loss = 0.0

    # Disable gradient calculation during evaluation
    with torch.no_grad():  
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate the loss
            running_loss += loss.item() * inputs.size(0)

    # Calculate the average loss over the entire test dataset
    test_loss = running_loss / len(dataloader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss

def plot_loss_history(history):

    """
    Plots the training and validation loss from the history dictionary.

    Args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss' lists.
    """
    
    # Get the training and validation loss lists
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    # Create a list of epoch numbers
    epochs = range(1, len(train_loss) + 1)
    
    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')    # 'bo-' gives blue dots connected by a line
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')  # 'ro-' gives red dots connected by a line
    
    # Add title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Add a legend
    plt.legend()
    
    # Add a grid
    plt.grid(True)
    
    # Find the epoch with the minimum validation loss
    min_val_loss = min(val_loss)
    min_val_epoch = val_loss.index(min_val_loss) + 1
    
    # Add an annotation for the best epoch
    plt.annotate(
        f'Best Epoch: {min_val_epoch}\nMin Val Loss: {min_val_loss:.4f}',
        xy=(min_val_epoch, min_val_loss),
        xytext=(min_val_epoch + 1, min_val_loss + 0.05), # Offset text slightly
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.7)
    )
    
    # Show the plot
    plt.show()


# ---- Example usage ----
# --- Assuming model, dataloaders, criterion, optimizer are defined ---

# MAX_EPOCHS = 50
# PATIENCE = 7

# # 1. Run your training
# best_model, history = train_model(
#     model, 
#     dataloaders, 
#     criterion, 
#     optimizer, 
#     num_epochs=MAX_EPOCHS,
#     device='cuda',
#     patience=PATIENCE
# )

# # 2. Plot the results
# # This will open a new window showing your loss curves
# plot_loss_history(history)