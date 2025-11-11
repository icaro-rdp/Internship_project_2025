"""Simple logging utility wrapper for the project with optional colored output.

Provides thin wrappers around Python's logging so the codebase can
call `info`, `warn`, `error`, `debug` without scattering print() calls.

Color is only enabled when stdout is a TTY. If you want better Windows
support, install `colorama` (it will be used automatically).
"""
import logging
import sys
from typing import Optional

_LOGGER_NAME = 'image_authenticity'
_logger = logging.getLogger(_LOGGER_NAME)


# ANSI fallbacks; if colorama is available it will be used and initialized
try:
    import colorama as _colorama  # optional
    _colorama.init()
    RESET_SEQ = _colorama.Style.RESET_ALL
    RED = _colorama.Fore.RED
    YELLOW = _colorama.Fore.YELLOW
    GREEN = _colorama.Fore.GREEN
    CYAN = _colorama.Fore.CYAN
    BOLD = _colorama.Style.BRIGHT
except Exception:
    RESET_SEQ = '\033[0m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to the level name."""
    LEVEL_COLORS = {
        'DEBUG': CYAN,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': BOLD + RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Only colorize levelname if output is a tty (avoid color codes in logs/files)
        use_color = sys.stdout.isatty()
        orig_levelname = record.levelname
        if use_color:
            color = self.LEVEL_COLORS.get(orig_levelname, '')
            record.levelname = f"{color}{orig_levelname}{RESET_SEQ}"
        try:
            return super().format(record)
        finally:
            record.levelname = orig_levelname


def _ensure_configured():
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        datefmt = '%H:%M:%S'
        if sys.stdout.isatty():
            handler.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))
        else:
            handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
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
