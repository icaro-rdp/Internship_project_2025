"""Convenience exports for the available model architectures.

The concrete model classes live in :mod:`.models`. Importing this subpackage
now defers loading the heavy ``torch`` dependencies until one of the classes is
requested, improving import times for lightweight tooling (for example CLI
usage that only needs configuration data).
"""

from importlib import import_module
from types import ModuleType
from typing import Any, List, Optional

__all__ = [
    "BarlowTwinsAuthenticityPredictor",
    "EfficientNetB3AuthenticityPredictor",
    "DenseNet161AuthenticityPredictor",
    "ResNet152AuthenticityPredictor",
    "VGG16AuthenticityPredictor",
    "VGG19AuthenticityPredictor",
    "InceptionV3AuthenticityPredictor",
]

_MODELS_MODULE: Optional[ModuleType] = None


def _load_models_module() -> ModuleType:
    """Load and cache the underlying ``models`` module on demand."""

    global _MODELS_MODULE
    if _MODELS_MODULE is None:
        _MODELS_MODULE = import_module(f"{__name__}.models")
    return _MODELS_MODULE


def __getattr__(name: str) -> Any:
    """Expose model classes lazily to avoid needless imports."""

    if name in __all__:
        module = _load_models_module()
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(__all__) | set(globals().keys()))
