# Mesh Editor Demos & Diagnostics

This directory contains visualization and diagnostic helpers for the patch-based mesh editor.

## Quick Start

Run any demo with the module flag:

```bash
python -m demos.patch_diagnose
python -m demos.patch_batches
python -m demos.patch_visual_labels
```

Or invoke a function from Python:

```python
from demos import run_patch_diagnose
run_patch_diagnose(npts=60, seed=3)
```

## Modules

| Module | Function(s) | Purpose |
|--------|-------------|---------|
| mesh_editor_demos.py | test_random_mesh, test_non_convex_cavity_star, ... | Basic mesh operation showcases |
| adapt_before_after.py | visualize adaptation before/after |
| adapt_scenario.py | scenario-based adaptation demo |
| anisotropic_adaptation_naca0012.py | advanced NACA0012 anisotropic adaptation |
| boundary_split_demo.py | boundary split operations demo |
| coarsening_scenario.py | mesh coarsening scenario demo |
| coarsening_scenario2.py | alternative coarsening scenario demo |
| edge_collapse_demo.py | edge collapse operation demo |
| generate_scenario.py | scenario mesh generation demo |
| mesh_editor_demos.py | test_random_mesh, test_non_convex_cavity_star, ... | Basic mesh operation showcases |
| parallel_patch.py | parallel patch demo |
| partition_parallel.py | partitioning demo |
| patch_batches.py | run_patch_batches | Color node-centered patches |
| refinement_scenario.py | mesh refinement scenario demo |

## Output Artifacts

Most demos write PNG (and sometimes CSV) files to the project root so they are easy to locate:
* patch_diagnose.png
* patch_batches.png
* patch_boundaries_*.png
* patch_boundary_report.csv
* patch_zoom_<id>.png

## Notes

These demos are excluded from test coverage (they are visualization-only) and should not be imported by production code paths. Their APIs are stable at the function level (`run_*`).
 