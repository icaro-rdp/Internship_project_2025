"""Utilities exposed by :mod:`main.Utils`.

This module performs lazy attribute loading so that importing
``Image_Authenticity_prediction.main.Utils`` does not immediately import heavy
PyTorch dependencies. Each utility is pulled in only when accessed.
"""

from importlib import import_module
from typing import Any, Dict, List, Tuple

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "GradCAM": ("explainability", "GradCAM"),
    "MultiscalePixelMasking": ("explainability", "MultiscalePixelMasking"),
    "FeatureMapsPruner": ("pruning", "FeatureMapsPruner"),
    "clear_gpu_memory": ("cleanup", "clear_gpu_memory"),
    "cleanup_model_and_data": ("cleanup", "cleanup_model_and_data"),
    "visualize_and_save_saliency": ("visualization", "visualize_and_save_saliency"),
    "load_config": ("config", "load_config"),
    "get_device": ("config", "get_device"),
    "get_training_config": ("config", "get_training_config"),
    "get_pruning_config": ("config", "get_pruning_config"),
    "get_xai_config": ("config", "get_xai_config"),
    "get_ensemble_config": ("config", "get_ensemble_config"),
    "get_data_config": ("config", "get_data_config"),
    "get_paths_config": ("config", "get_paths_config"),
}

__all__ = sorted(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    """Import the requested utility on demand and cache it."""

    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f"{__name__}.{module_name}")
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> List[str]:
    return sorted(set(__all__) | set(globals().keys()))
