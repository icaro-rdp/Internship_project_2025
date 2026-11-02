"""
Centralized configuration loader for the Image Authenticity Prediction project.

This module provides a single source of truth for all configuration values,
loading them from the YAML config file.
"""

from pathlib import Path
from typing import Any

import torch
import yaml

# Cache the config to avoid repeated file I/O
_CONFIG_CACHE: dict[str, Any] | None = None


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is at: project_root/main/Utils/config.py
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict[str, Any]:
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


def get_training_config() -> dict[str, Any]:
    """Get training configuration."""
    config = load_config()
    return {
        "max_epochs": config["training"]["max_epochs"],
        "patience": config["training"]["patience"],
        "learning_rate": config["training"]["learning_rate"],
        "freeze_backbone": config["training"]["freeze_backbone"],
        "num_variants": config["training"].get("num_variants", 10),
        "device": get_device(),
    }


def get_pruning_config() -> dict[str, Any]:
    """Get pruning configuration."""
    config = load_config()
    return {
        "force_recompute": config["pruning"]["force_recompute"],
        "methods": config["pruning"]["methods"],
        "threshold": config["pruning"]["threshold"],
    }


def get_xai_config() -> dict[str, Any]:
    """Get XAI (explainability) configuration."""
    config = load_config()
    xai_cfg = config["xai"]
    return {
        "sigma": xai_cfg["sigma"],
        "stride": xai_cfg.get("stride", 1),
        "mask_val": xai_cfg["mask_val"],
        "px_batch": xai_cfg["pixel_batch"],
        "gc_interval": xai_cfg["gradcam_interval"],
        "mpm_interval": xai_cfg["mpm_interval"],
        "lime_interval": xai_cfg.get("lime_interval", 25),
        "lime_n_segments": xai_cfg.get("lime_n_segments", 150),
        "lime_compactness": xai_cfg.get("lime_compactness", 10.0),
        "lime_num_perturbations": xai_cfg.get("lime_num_perturbations", 1200),
        "lime_keep_prob": xai_cfg.get("lime_keep_prob", 0.7),
        "lime_kernel_width": xai_cfg.get("lime_kernel_width", 0.25),
        "lime_ridge_alpha": xai_cfg.get("lime_ridge_alpha", 1.0),
        "lime_seed": xai_cfg.get("lime_seed", 0),
        "lime_batch_size": xai_cfg.get("lime_batch_size", 64),
    }


def get_ensemble_config() -> dict[str, Any]:
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


def get_data_config() -> dict[str, Any]:
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


def get_paths_config() -> dict[str, Path]:
    """Get paths configuration."""
    config = load_config()
    root = get_project_root()
    return {
        "weights_dir": root / config["paths"]["weights_dir"],
        "results_dir": root / config["paths"]["results_dir"],
        "plots_dir": root / config["paths"]["plots_dir"],
    }


def get_model_config(model_name: str) -> dict[str, Any]:
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
