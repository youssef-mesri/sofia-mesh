import numpy as np
import pytest

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity


def make_editor_square():
    pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    return PatchBasedMeshEditor(pts, tris)


def assert_invariants(editor):
    summary = editor.stats_summary()
    for op, stats in summary.items():
        attempts = stats['attempts']
        success = stats['success']
        fail = stats['fail']
        # Core counting invariant
        assert attempts == success + fail, f"{op}: attempts({attempts}) != success({success}) + fail({fail})"
        # Timing invariants (only meaningful if attempts > 0)
        if attempts > 0:
            t_tot = stats['time_total']
            t_min = stats['time_min']
            t_max = stats['time_max']
            t_avg = stats['time_avg']
            # Non-negativity
            assert t_tot >= 0 and t_min >= 0 and t_max >= 0 and t_avg >= 0, f"{op}: negative timing value"
            # Ordering
            assert t_min <= t_max + 1e-12, f"{op}: time_min > time_max (min={t_min}, max={t_max})"
            assert t_max <= t_tot + 1e-12, f"{op}: time_max > time_total (max={t_max}, total={t_tot})"
            # Average consistency (allow small floating error)
            if attempts > 0:
                assert abs(t_avg * attempts - t_tot) <= max(1e-9, 1e-6 * t_tot), f"{op}: average inconsistency"


def test_stats_invariants_basic_operations():
    editor = make_editor_square()

    # Perform a series of valid operations
    ok, msg, _ = editor.flip_edge((0,2))
    assert ok, f"flip failed: {msg}"

    # Pick an interior edge (shared by 2 triangles) for split; mesh may have changed after flip
    interior_edges = [e for e,s in editor.edge_map.items() if len(s)==2]
    assert interior_edges, "Expected at least one interior edge to split"
    e_split = interior_edges[0]
    ok, msg, _ = editor.split_edge(e_split)
    assert ok, f"split failed on edge {e_split}: {msg}"

    # Add node inside first active triangle
    active = [i for i,t in enumerate(editor.triangles) if not np.all(np.array(t)==-1)]
    tri0 = editor.triangles[active[0]]
    centroid = np.mean(editor.points[list(tri0)], axis=0)
    ok, msg, _ = editor.add_node(centroid * 0.999 + 1e-8, tri_idx=active[0])  # slight perturbation
    assert ok, f"add_node failed: {msg}"

    # Force a failure attempt: try flipping a boundary edge (should fail and count fail)
    boundary_edges = [e for e,s in editor.edge_map.items() if len(s)==1]
    assert boundary_edges, "Expected boundary edges to attempt a failing flip"
    _ok_fail, _msg_fail, _ = editor.flip_edge(boundary_edges[0])
    # We don't assert _ok_fail is False strictly (if mesh evolved), but invariant must hold

    # Conformity sanity (with tombstones allowed) before invariants
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_conf, f"Mesh not conforming before stats invariant check: {msgs}"

    assert_invariants(editor)


def test_stats_invariants_randomized(seed=7):
    # Random sequence to touch multiple operations & failure paths
    editor = make_editor_square()
    rng = np.random.RandomState(seed)
    ops = ['flip','split','add']
    for _ in range(40):
        choice = rng.choice(ops)
        if choice == 'flip':
            edges = list(editor.edge_map.keys())
            if not edges: continue
            e = edges[rng.randint(len(edges))]
            editor.flip_edge(e)
        elif choice == 'split':
            interior = [e for e,s in editor.edge_map.items() if len(s)==2]
            if not interior: continue
            e = interior[rng.randint(len(interior))]
            editor.split_edge(e)
        elif choice == 'add':
            active = [i for i,t in enumerate(editor.triangles) if not np.all(np.array(t)==-1)]
            if not active: continue
            tri = editor.triangles[active[rng.randint(len(active))]]
            centroid = np.mean(editor.points[list(tri)], axis=0)
            jitter = rng.randn(2) * 1e-8
            editor.add_node(centroid + jitter)
    # Invariants after random ops
    assert_invariants(editor)
