"""Logging utilities for Sofia.

Provides a consistent logger hierarchy and formatting without modifying the
process root logger. All Sofia code should obtain loggers via get_logger().
"""
from __future__ import annotations

import logging
import sys
from typing import Optional, Union

_FORMAT = logging.Formatter('%(levelname)s %(name)s: %(message)s')


def _ensure_sofia_root() -> logging.Logger:
    """Ensure the 'sofia' logger has a single stream handler and is isolated
    from the process root logger. Returns the 'sofia' logger.
    """
    sofia_root = logging.getLogger('sofia')
    # If only NullHandlers are present (added by package __init__), replace with a StreamHandler
    has_non_null = any(not isinstance(h, logging.NullHandler) for h in sofia_root.handlers)
    if not sofia_root.handlers or not has_non_null:
        # Remove existing NullHandlers to avoid swallowing logs
        for h in list(sofia_root.handlers):
            if isinstance(h, logging.NullHandler):
                sofia_root.removeHandler(h)
        # Attach a stdout StreamHandler if none present
        has_non_null_after = any(not isinstance(h, logging.NullHandler) for h in sofia_root.handlers)
        if not has_non_null_after:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(_FORMAT)
            sofia_root.addHandler(handler)
    # Do not propagate to the process root
    sofia_root.propagate = False
    return sofia_root


def _to_level(level: Union[str, int, None], default: int = logging.INFO) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, str(level).upper())
    except Exception:
        return default


def configure_logging(level: Union[str, int] = 'INFO', mute_external: bool = True) -> None:
    """Configure the 'sofia' logger family level and optional external noise suppression.

    This does NOT modify the process root logger.
    """
    sofia_root = _ensure_sofia_root()
    lvl = _to_level(level)
    sofia_root.setLevel(lvl)
    # Optionally reduce very noisy third-party DEBUG logs
    if mute_external and lvl <= logging.DEBUG:
        for noisy in ('matplotlib', 'matplotlib.font_manager'):
            try:
                logging.getLogger(noisy).setLevel(logging.INFO)
            except Exception:
                pass


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Return a configured logger under the 'sofia' namespace.

    If a level is provided, it sets the logger's level; otherwise the logger
    is set to NOTSET so it inherits from the 'sofia' parent configured via
    configure_logging(). This ensures INFO-level logs on children are visible
    when the root 'sofia' logger is set to INFO.
    """
    _ensure_sofia_root()
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(_to_level(level))
    else:
        # Inherit from parent 'sofia' logger by default
        log.setLevel(logging.NOTSET)
    return log


__all__ = ['get_logger', 'configure_logging']
