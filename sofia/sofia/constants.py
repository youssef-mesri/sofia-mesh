"""Central numerical tolerances and small geometry constants.

This module centralizes tiny numeric thresholds used across the codebase so
they can be tuned consistently and referenced without scattering literals.
"""
from __future__ import annotations

# Geometry tolerances
EPS_AREA: float = 1e-12           # minimum positive (absolute) triangle area
EPS_MIN_ANGLE_DEG: float = 1e-9   # tolerance for min-angle comparisons (degrees)
EPS_IMPROVEMENT: float = 1e-12    # significance threshold for local improvement

# Auxiliary small epsilons
EPS_COLINEAR: float = 1e-15       # near-colinearity threshold for polygon/tri tests

__all__ = [
    'EPS_AREA',
    'EPS_MIN_ANGLE_DEG',
    'EPS_IMPROVEMENT',
    'EPS_COLINEAR',
]
