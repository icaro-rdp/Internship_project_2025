"""Utilities for discovering the available experiment modules."""

from importlib import import_module
from typing import Dict, List

_EXPERIMENT_MODULES: Dict[str, str] = {
	"experiment_one": "experiment_one",
	"experiment_two": "experiment_two",
	"experiment_three": "experiment_three",
}

__all__ = [*sorted(_EXPERIMENT_MODULES.keys()), "get_experiment_module", "list_experiments"]


def _load_experiment(name: str):
	module_name = _EXPERIMENT_MODULES[name]
	module = import_module(f"{__name__}.{module_name}")
	globals()[name] = module
	return module


def __getattr__(name: str):
	if name in _EXPERIMENT_MODULES:
		return _load_experiment(name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_experiment_module(name: str):
	"""Return an experiment submodule by name.

	Parameters
	----------
	name:
		Key present in :func:`list_experiments`.
	"""

	if name not in _EXPERIMENT_MODULES:
		raise KeyError(f"Unknown experiment module: {name}")
	return _load_experiment(name)


def list_experiments() -> List[str]:
	"""Return the names of all available experiment submodules."""

	return sorted(_EXPERIMENT_MODULES.keys())


def __dir__() -> List[str]:
	return sorted(set(__all__) | set(globals().keys()))
