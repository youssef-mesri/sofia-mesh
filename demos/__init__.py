"""Demos and diagnostic visualization helpers for the mesh editor.

Primary categories:
 - Basic operation demos (mesh_editor_demos.py)
 - Patch visualization (patch_batches.py)
 - Mesh adaptation scenarios (adapt_scenario.py, coarsening_scenario.py, etc.)
"""
from .mesh_editor_demos import (
    test_random_mesh,
    test_non_convex_cavity_star,
    test_flip_and_add_node,
    test_locate_point_optimized,
    test_maps_consistency,
    test_improve_min_angle_loop,
)
from .patch_batches import run_patch_batches

__all__ = [ 
    # basic demos
    'test_random_mesh',
    'test_non_convex_cavity_star',
    'test_flip_and_add_node',
    'test_locate_point_optimized',
    'test_maps_consistency',
    'test_improve_min_angle_loop',
    # patch visualization
    'run_patch_batches',
]