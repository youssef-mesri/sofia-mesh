"""Per-run context for Sofia drivers using contextvars.

This isolates mutable flags or hooks so multiple runs can execute in parallel
without interfering with each other (e.g., in threads or async tasks).
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import contextvars

_CTX: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar('sofia_run_ctx', default={})


def set_context(values: Dict[str, Any]) -> None:
    current = dict(_CTX.get())
    current.update(values)
    _CTX.set(current)


def get_context() -> Dict[str, Any]:
    return _CTX.get()


def get(key: str, default: Optional[Any] = None) -> Any:
    return _CTX.get().get(key, default)


__all__ = ['set_context', 'get_context', 'get']
