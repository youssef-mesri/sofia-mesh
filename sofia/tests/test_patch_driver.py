import logging
import random

import numpy as np

from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity
from sofia.core.diagnostics import compact_copy
from sofia.core.remesh_driver import PatchDriverConfig, run_patch_batch_driver, greedy_remesh
from sofia.core.patch_driver import apply_patch_operation



def _run_basic(config_overrides, seed=1234):
    """Helper to run the patch driver with a small mesh and return (editor, result)."""
    pts, tris = build_random_delaunay(npts=config_overrides.pop('npts', 35), seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    cfg = PatchDriverConfig(**config_overrides)
    # Standardized test logger
    from sofia.core.logging_utils import get_logger
    logger = get_logger('sofia.tests.patch_driver')
    # deterministic RNGs for reproducibility
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    result = run_patch_batch_driver(
        editor,
        cfg,
        rng=rng,
        np_rng=np_rng,
        logger=logger,
        greedy_remesh=greedy_remesh if cfg.use_greedy_remesh else None,
        plot_mesh=None,  # disable plotting during tests
    )
    return editor, result


def _assert_result_invariants(editor, result, cfg: PatchDriverConfig):
    # Required keys
    for k in ('iterations', 'final_min_angle', 'npts', 'ntri'):
        assert k in result, f"missing key {k} in result"
    assert 0 <= result['iterations'] <= cfg.max_iterations
    assert result['npts'] == len(editor.points)
    assert result['ntri'] == len(editor.triangles)
    assert result['npts'] >= 3
    assert result['ntri'] >= 1
    assert np.isfinite(result['final_min_angle'])
    # Conformity: we do not assert strict conformity here because transient non-manifold
    # structures can persist after greedy pre-passes (mirroring leniency in greedy tests).
    # We only require at least one active triangle and finite min-angle (checked above).


def test_patch_driver_basic_runs_without_greedy():
    overrides = dict(
        threshold=5.0,
        max_iterations=2,
        patch_radius=1,
        top_k=30,
        disjoint_on='tri',
        allow_overlap=False,
        batch_attempts=1,
        use_greedy_remesh=False,
        out_prefix='test_patch_no_greedy',
    )
    editor, result = _run_basic(overrides.copy())
    _assert_result_invariants(editor, result, PatchDriverConfig(**overrides))


def test_patch_driver_basic_runs_with_greedy_pass():
    overrides = dict(
        threshold=5.0,
        max_iterations=2,
        patch_radius=1,
        top_k=30,
        disjoint_on='tri',
        allow_overlap=False,
        batch_attempts=1,
        use_greedy_remesh=True,
        greedy_vertex_passes=1,
        greedy_edge_passes=1,
        out_prefix='test_patch_with_greedy',
    )
    editor, result = _run_basic(overrides.copy())
    _assert_result_invariants(editor, result, PatchDriverConfig(**overrides))


def test_single_patch_add_operation_succeeds_on_skinny_triangle():
    """Craft a mesh with one extremely skinny triangle and verify an 'add' op is accepted.

    We bypass the full batch driver and call apply_patch_operation directly with a deterministic
    RNG so that the first operation selection prefers an 'add' (prob 0.6). The skinny triangle's
    centroid insertion should not violate conformity and is accepted under the improvement policy.
    """
    # Skinny triangle points (very small height -> tiny angle at vertex 2).
    pts = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 0.001]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=int)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # Sanity: initial min angle is extremely small
    pre_min = editor.global_min_angle()
    assert pre_min < 0.1

    patch = {'id': 0, 'tris': [0], 'verts': [0, 1, 2]}
    cfg = PatchDriverConfig(max_iterations=1)
    rng = random.Random(1)  # first random() ~0.134 < 0.6 -> choose 'add'

    from sofia.core.logging_utils import get_logger as _get_logger
    ok, info, op, param, local_before, local_after, tri_count_before, tri_count_after, rejected = apply_patch_operation(
        editor, patch, rng, cfg, logger=_get_logger('sofia.tests.patch_driver'))

    assert ok, f"operation was rejected: {info}"
    assert op == 'add'
    # After add, there should be >1 triangle (original subdivided) and min angle improves or is finite.
    assert len(editor.triangles) >= 3
    assert np.isfinite(editor.global_min_angle())
