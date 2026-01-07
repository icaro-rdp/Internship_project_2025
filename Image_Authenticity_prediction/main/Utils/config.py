"""
Centralized configuration loader for the Image Authenticity Prediction project.

This module provides a single source of truth for all configuration values,
loading them from the YAML config file.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import torch

# Cache the config to avoid repeated file I/O
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is at: project_root/main/Utils/config.py
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        Dictionary containing all configuration values.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        config_file = get_project_root() / "Configs" / "config.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Please ensure 'Configs/config.yaml' exists in the project root."
        )

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Handle 'auto' device setting
    if config.get("run_settings", {}).get("device") == "auto":
        config["run_settings"]["device"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Cache the config if using default path
    if config_path is None:
        _CONFIG_CACHE = config

    return config


def get_device() -> str:
    """Get the configured device (cuda/cpu)."""
    config = load_config()
    return config["run_settings"]["device"]


def get_seed() -> int:
    """Get the random seed."""
    config = load_config()
    return config["run_settings"]["seed"]


def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    config = load_config()
    return {
        "max_epochs": config["training"]["max_epochs"],
        "patience": config["training"]["patience"],
        "learning_rate": config["training"]["learning_rate"],
        "freeze_backbone": config["training"]["freeze_backbone"],
        "device": get_device(),
    }


def get_pruning_config() -> Dict[str, Any]:
    """Get pruning configuration."""
    config = load_config()
    return {
        "force_recompute": config["pruning"]["force_recompute"],
        "methods": config["pruning"]["methods"],
        "threshold": config["pruning"]["threshold"],
    }


def get_xai_config() -> Dict[str, Any]:
    """Get XAI (explainability) configuration."""
    config = load_config()
    return {
        "sigma": config["xai"]["sigma"],
        "mask_val": config["xai"]["mask_val"],
        "px_batch": config["xai"]["pixel_batch"],
        "gc_interval": config["xai"]["gradcam_interval"],
        "mpm_interval": config["xai"]["mpm_interval"],
    }


def get_ensemble_config() -> Dict[str, Any]:
    """Get ensemble configuration."""
    config = load_config()
    return {
        "batch_size": config["ensemble"]["batch_size"],
        "num_epochs_base": config["ensemble"]["num_epochs_base"],
        "num_epochs_meta": config["ensemble"]["num_epochs_meta"],
        "learning_rate": config["ensemble"]["learning_rate"],
        "learning_rate_meta": config["ensemble"]["learning_rate_meta"],
        "n_splits": config["ensemble"]["n_splits"],
        "patience": config["ensemble"]["patience"],
    }


def get_data_config() -> Dict[str, Any]:
    """Get data configuration."""
    config = load_config()
    return {
        "annotation_file": config["data"]["annotation_file"],
        "batch_size": config["data"]["batch_size"],
        "single_batch_size": config["data"]["single_batch_size"],
        "num_workers": config["data"]["num_workers"],
        "train_fraction": config["data"]["train_fraction"],
        "val_fraction": config["data"]["val_fraction"],
        "test_fraction": config["data"]["test_fraction"],
    }


def get_paths_config() -> Dict[str, Path]:
    """Get paths configuration."""
    config = load_config()
    root = get_project_root()
    return {
        "weights_dir": root / config["paths"]["weights_dir"],
        "results_dir": root / config["paths"]["results_dir"],
        "plots_dir": root / config["paths"]["plots_dir"],
    }


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    config = load_config()
    if model_name not in config["models"]:
        raise KeyError(
            f"Model '{model_name}' not found in config. Available: {list(config['models'].keys())}"
        )
    return config["models"][model_name]


def get_model_order() -> list:
    """Get the preferred model ordering for visualization."""
    config = load_config()
    return config["visualization"]["model_order"]


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
