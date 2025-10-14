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
| patch_diagnose.py | run_patch_diagnose | Label patch boundary loops & seeds |
| patch_batches.py | run_patch_batches | Color node-centered patches |
| patch_centering_check.py | run_patch_centering_check | Validate seed centering invariants |
| patch_stats_debug.py | run_patch_stats_debug | Print patch metric summaries |
| patch_boundary_report.py | run_patch_boundary_report | CSV + explicit boundary visualization |
| patch_visual_labels.py | run_patch_visual_labels | Patch IDs and per-patch zoom PNGs |
| inspect_patch0_nodes.py | run_inspect_patch0_nodes | Inspect interior vs boundary nodes for patch 0 |

## Output Artifacts

Most demos write PNG (and sometimes CSV) files to the project root so they are easy to locate:
* patch_diagnose.png
* patch_batches.png
* patch_boundaries_*.png
* patch_boundary_report.csv
* patch_zoom_<id>.png

## Notes

These demos are excluded from test coverage (they are visualization-only) and should not be imported by production code paths. Their APIs are stable at the function level (`run_*`).
