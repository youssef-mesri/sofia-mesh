"""Demos and diagnostic visualization helpers for the mesh editor.

Each module exposes a run_* function callable from Python and a module-level
__main__ guard so it can be executed via:

    python -m demos.patch_diagnose

Primary categories:
 - Basic operation demos (mesh_editor_demos.py)
 - Patch visualization (patch_diagnose, patch_batches, patch_visual_labels)
 - Diagnostics / reports (patch_centering_check, patch_boundary_report, patch_stats_debug)
 - Targeted inspection (inspect_patch0_nodes)
"""
from .mesh_editor_demos import (
    test_random_mesh,
    test_non_convex_cavity_star,
    test_flip_and_add_node,
    test_locate_point_optimized,
    test_maps_consistency,
    test_improve_min_angle_loop,
)
from .patch_diagnose import run_patch_diagnose
from .patch_batches import run_patch_batches
from .patch_centering_check import run_patch_centering_check
from .patch_stats_debug import run_patch_stats_debug
from .patch_boundary_report import run_patch_boundary_report
from .patch_visual_labels import run_patch_visual_labels
from .inspect_patch0_nodes import run_inspect_patch0_nodes

__all__ = [
    # basic demos
    'test_random_mesh','test_non_convex_cavity_star','test_flip_and_add_node','test_locate_point_optimized',
    'test_maps_consistency','test_improve_min_angle_loop',
    # patch visualizations / diagnostics
    'run_patch_diagnose','run_patch_batches','run_patch_centering_check','run_patch_stats_debug',
    'run_patch_boundary_report','run_patch_visual_labels','run_inspect_patch0_nodes'
]