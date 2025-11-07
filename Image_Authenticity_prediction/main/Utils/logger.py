"""Simple logging utility wrapper for the project.

Provides thin wrappers around Python's logging so the codebase can
call `info`, `warn`, `error`, `debug` without scattering print() calls.

This module configures a StreamHandler to stdout and a reasonable
default format. Call `set_level('DEBUG')` to enable debug output.
"""
import logging
import sys
from typing import Optional

_LOGGER_NAME = 'image_authenticity'
_logger = logging.getLogger(_LOGGER_NAME)


def _ensure_configured():
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(fmt)
        _logger.addHandler(handler)
    # Default level is INFO unless changed
    if _logger.level == 0:
        _logger.setLevel(logging.INFO)


def set_level(level_name: str):
    """Set logging level by name (DEBUG, INFO, WARNING, ERROR).

    Args:
        level_name: case-insensitive level name
    """
    _ensure_configured()
    level = getattr(logging, level_name.upper(), None)
    if level is None:
        raise ValueError(f"Unknown log level: {level_name}")
    _logger.setLevel(level)


def info(msg: str, *args, **kwargs):
    _ensure_configured()
    _logger.info(msg, *args, **kwargs)


def warn(msg: str, *args, **kwargs):
    _ensure_configured()
    _logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    _ensure_configured()
    _logger.error(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    _ensure_configured()
    _logger.debug(msg, *args, **kwargs)


def get_logger(name: Optional[str] = None):
    """Return the underlying logger (for advanced usage)."""
    _ensure_configured()
    return logging.getLogger(name or _LOGGER_NAME)
