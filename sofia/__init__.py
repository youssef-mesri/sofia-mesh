"""Public package API for the Sofia mesh toolkit.

This facade provides a stable, flatter import surface on top of the
internal implementation package ``sofia.core`` while deferring heavy
imports (drivers, pocket fill) until first use to avoid circular import
issues and keep ``import sofia`` fast.

Example
-------
    from sofia import greedy_remesh, PatchDriverConfig, triangle_area

The deeper modules (``sofia.core.*``) are considered internal and may
change; rely on this layer for public symbols. Root-level legacy shim
modules emit ``DeprecationWarning`` and will be removed in a future
release.
"""
from importlib import import_module as _imp
import warnings as _w
import logging as _logging

try:  # Python 3.8+ runtime version export
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("sofia-mesh")  # populated when installed
except Exception:  # pragma: no cover - editable / unknown state
    __version__ = "0.0.0+dev"

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Eager light-weight submodules
_geom = _imp('sofia.core.geometry')
_const = _imp('sofia.core.constants')
_conf = _imp('sofia.core.conformity')
_tri  = _imp('sofia.core.triangulation')
_ops  = _imp('sofia.core.operations')
_quality = _imp('sofia.core.quality')
_stats = _imp('sofia.core.stats')
_io = _imp('sofia.core.io')

def _lazy_module(mod_name):
    class _ModuleProxy:
        __slots__ = ('_m',)
        def _load(self):  # type: ignore
            if hasattr(self, '_m'):
                return self._m  # type: ignore
            self._m = _imp(mod_name)  # type: ignore
            return self._m  # type: ignore
        def __getattr__(self, item):  # type: ignore
            return getattr(self._load(), item)
        def __dir__(self):  # type: ignore
            return dir(self._load())
    return _ModuleProxy()

# Lazily loaded heavy / dependency-rich modules
pocket_fill = _lazy_module('sofia.core.pocket_fill')

def _lazy_driver_attr(name):
    def _wrapper(*args, **kwargs):
        drv = _imp('sofia.core.remesh_driver')
        return getattr(drv, name)(*args, **kwargs)
    return _wrapper

def _lazy_driver_type(name):
    class _DriverProxy:
        def __call__(self, *args, **kwargs):
            drv = _imp('sofia.core.remesh_driver')
            cls = getattr(drv, name)
            return cls(*args, **kwargs)
        def __getattr__(self, item):
            drv = _imp('sofia.core.remesh_driver')
            return getattr(getattr(drv, name), item)
    return _DriverProxy()

# Public callable entry points
greedy_remesh = _lazy_driver_attr('greedy_remesh')
run_patch_batch_driver = _lazy_driver_attr('run_patch_batch_driver')
tri_min_angle = _lazy_driver_attr('tri_min_angle')
PatchDriverConfig = _lazy_driver_type('PatchDriverConfig')

# Fine-grained geometry exports
triangle_area = _geom.triangle_area
triangle_angles = getattr(_geom, 'triangle_angles', None)
# Export tolerances from constants (not via geometry)
EPS_AREA = getattr(_const, 'EPS_AREA', 0.0)
EPS_MIN_ANGLE_DEG = getattr(_const, 'EPS_MIN_ANGLE_DEG', 0.0)
EPS_IMPROVEMENT = getattr(_const, 'EPS_IMPROVEMENT', 0.0)

# I/O functions
read_msh = _io.read_msh
write_vtk = _io.write_vtk

# Namespace submodules for exploratory users
geometry = _geom
conformity = _conf
triangulation = _tri
operations = _ops
stats = _stats
quality = _quality
constants = _const
io = _io

# Editor main class (imported lazily via internal path; prefers internal over root shim)
try:  # pragma: no cover - defensive
    from .core.mesh_modifier2 import PatchBasedMeshEditor  # type: ignore
except Exception:  # pragma: no cover
    try:
        PatchBasedMeshEditor = _imp('mesh_modifier2').PatchBasedMeshEditor  # fallback to shim
    except Exception:
        PatchBasedMeshEditor = None  # type: ignore

_w.filterwarnings('default', category=DeprecationWarning)

__all__ = [
    '__version__',
    # geometry primitives
    'triangle_area','triangle_angles',
    # tolerances
    'EPS_AREA','EPS_MIN_ANGLE_DEG','EPS_IMPROVEMENT',
    # drivers
    'greedy_remesh','run_patch_batch_driver','PatchDriverConfig','tri_min_angle',
    # high-level editor
    'PatchBasedMeshEditor',
    # submodules / namespaces
    'geometry','conformity','triangulation','operations','stats','pocket_fill','quality','constants'
]
