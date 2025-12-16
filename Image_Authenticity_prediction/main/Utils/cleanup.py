import gc
import torch
from .logger import debug

# ============================================================================
# Memory Management Utilities
# ============================================================================


def clear_gpu_memory():
    """
    Clear GPU memory by collecting garbage and emptying CUDA cache.
    Call this after each model training/pruning to prevent memory accumulation.
    """
    # Collect Python garbage
    gc.collect()

    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if torch.cuda.is_available():
            # Print memory stats for monitoring
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            debug(
                f"  [GPU Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )


def cleanup_model_and_data(model, dataloaders=None, optimizer=None):
    """
    Properly cleanup model, dataloaders, and optimizer to free memory.

    Args:
        model: PyTorch model to cleanup
        dataloaders: Dict or list of dataloaders to cleanup
        optimizer: Optimizer to cleanup
    """
    debug("Starting model and data cleanup...")

    # Move model to CPU and delete
    if model is not None:
        model.cpu()
        del model
        debug("Model moved to CPU and deleted")

    # Cleanup optimizer
    if optimizer is not None:
        del optimizer
        debug("Optimizer deleted")

    # Cleanup dataloaders
    if dataloaders is not None:
        if isinstance(dataloaders, dict):
            for loader in dataloaders.values():
                del loader
        elif isinstance(dataloaders, (list, tuple)):
            for loader in dataloaders:
                del loader
        else:
            del dataloaders
        debug("Dataloaders deleted")

    # Force garbage collection and clear CUDA cache
    clear_gpu_memory()
    debug("Cleanup completed successfully")
