"""Compatibility wrapper for pocket-fill strategies.

Historically the pocket-fill strategies lived in this module. The canonical
implementations have been moved to ``sofia.sofia.triangulation`` to centralize
triangulation logic. To preserve backward compatibility we re-export the
functions here and emit a gentle deprecation warning for callers that import
this module directly.
"""
from __future__ import annotations
import warnings

# Re-export implementations from the triangulation module (single source of truth)
from .triangulation import (
    fill_pocket_quad,
    fill_pocket_steiner,
    fill_pocket_earclip,
)

__all__ = [
    'fill_pocket_quad',
    'fill_pocket_steiner',
    'fill_pocket_earclip',
]

# Warn once when this module is imported to encourage using the triangulation
# module directly in new code.
warnings.warn(
    'sofia.sofia.pocket_fill is deprecated; import pocket-fill strategies from '
    'sofia.sofia.triangulation instead',
    DeprecationWarning,
)
