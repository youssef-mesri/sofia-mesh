Module Relocation Plan
======================

Goal: Move core source files under `sofia/sofia/` package while keeping backward compatibility.

Phase 1 (current):
- Package scaffold present (`sofia/`).
- Core modules still at repository root.

Phase 2:
- Move: `remesh_driver.py`, `patch_driver.py`, `mesh_modifier2.py`, `operations.py`, `diagnostics.py`, `visualization.py`, `conformity.py`, `gif_capture.py`, `geometry.py`, `helpers.py`, `triangulation.py`, `stats.py`, `pocket_fill.py` into `sofia/sofia/`.
- Add thin shim files at old root paths importing from new package and issuing a DeprecationWarning.

Phase 3:
- Update console script in `pyproject.toml` to reference `sofia.remesh_driver:main`.
- Update tests to import from `sofia` where possible.

Phase 4:
- Remove shims in a major version bump (1.0.0).

Shim Template:
```python
import warnings
from sofia.sofia import remesh_driver as _impl  # adjust actual path
warnings.warn("Importing remesh_driver from repository root is deprecated; use 'from sofia import remesh_driver'", DeprecationWarning)
from sofia.sofia.remesh_driver import *  # noqa
```

Tracking:
- [ ] Move geometry helpers
- [ ] Move drivers
- [ ] Add shims
- [ ] Adjust tests
- [ ] Update entry point
- [ ] Remove shims (future major release)
