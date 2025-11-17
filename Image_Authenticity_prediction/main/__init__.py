"""Top-level package for the Image Authenticity Prediction project.

The package exposes commonly used submodules (``data``, ``train``, ``Models``,
``Utils`` and ``Experiments``) via lazy imports so that importing
``Image_Authenticity_prediction.main`` does not immediately pull the heavier
dependencies (for example PyTorch) into memory. Each submodule is imported on
first access and then cached in the module globals.
"""

from importlib import import_module
from typing import Any, List

__version__ = "0.1.0"

_SUBMODULES = {
	"data": "data",
	"train": "train",
	"Models": "Models",
	"Utils": "Utils",
	"Experiments": "Experiments",
}

__all__ = ["__version__", *sorted(_SUBMODULES.keys())]


def __getattr__(name: str) -> Any:
	"""Dynamically import known submodules on first access.

	Parameters
	----------
	name:
		Attribute being requested from the package.
	"""

	try:
		target = _SUBMODULES[name]
	except KeyError as exc:  # pragma: no cover - mirrors standard AttributeError
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

	module = import_module(f"{__name__}.{target}")
	globals()[name] = module
	return module


def __dir__() -> List[str]:
	"""Ensure interactive environments show lazily imported names."""

	return sorted(set(__all__) | set(globals().keys()))
