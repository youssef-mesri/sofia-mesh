# Mesh Editor Demos & Diagnostics

This directory contains visualization and diagnostic helpers for the patch-based mesh editor.

## Quick Start

Run demos directly as Python scripts from the project root:

```bash
python demos/adapt_before_after.py
python demos/adapt_scenario.py
python demos/boundary_split_demo.py
python demos/coarsening_scenario.py
python demos/edge_collapse_demo.py
python demos/refinement_scenario.py
```

Or invoke functions from Python:

```python
from demos import test_random_mesh, run_patch_batches

test_random_mesh(npts=60)
run_patch_batches(npts=80, seed=5)
```

## Available Demos

| Script | Purpose | Notes |
|--------|---------|-------|
| mesh_editor_demos.py | Basic mesh operation showcases | Contains test_random_mesh, test_non_convex_cavity_star, etc. |
| adapt_before_after.py | Visualize adaptation before/after | Generates adapt_pair.png |
| adapt_scenario.py | Scenario-based adaptation demo | |
| anisotropic_adaptation_naca0012.py | NACA0012 anisotropic adaptation | Requires naca0012.msh file |
| boundary_split_demo.py | Boundary split operations | |
| coarsening_scenario.py | Mesh coarsening scenario | |
| coarsening_scenario2.py | Alternative coarsening scenario | |
| edge_collapse_demo.py | Edge collapse operation | |
| generate_scenario.py | Scenario mesh generation | |
| parallel_patch.py | Parallel patch processing | |
| partition_parallel.py | Mesh partitioning | |
| patch_batches.py | Color node-centered patches | run_patch_batches() function |
| refinement_scenario.py | Mesh refinement scenario | |

## Output Artifacts

Most demos write PNG files to the current directory:
* adapt_pair.png (from adapt_before_after.py)
* Various scenario output images

## Notes

These demos are visualization-focused and excluded from test coverage. Their APIs are stable at the function level (`run_*` functions exported in `__init__.py`).

To use these demos as a library, import functions from the demos package:
```python
from demos import test_random_mesh, run_patch_batches
```
 