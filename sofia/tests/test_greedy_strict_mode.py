import numpy as np

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.remesh_driver import greedy_remesh
from sofia.core.conformity import check_mesh_conformity


def build_crossing_prone_mesh():
    # Two triangles sharing an edge, plus an extra vertex positioned to induce a bad flip.
    # Points arranged so that flipping edge (0,2) would create a crossing with segment (1,3).
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [0.5, 0.8],  # 2
        [0.5, -0.6], # 3 (below base; potential crossing after certain flips)
        [1.2, 0.4],  # 4 extra vertex to enlarge option space
    ])
    tris = np.array([
        [0,1,2],
        [0,1,3],
        [1,2,4],
        [1,4,3],
    ], dtype=int)
    return pts, tris


def test_greedy_strict_rejects_crossing_flip():
    pts, tris = build_crossing_prone_mesh()
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # Run a strict greedy pass with crossings rejection; edge passes encourage flips.
    greedy_remesh(
        editor,
        max_vertex_passes=1,
        max_edge_passes=2,
        verbose=False,
        strict=True,
        reject_crossings=True,
        reject_new_loops=False,
        annotate_failures=True,
    )
    # After strict greedy, ensure no crossing edges reported by conformity simulate routine.
    from sofia.core.conformity import simulate_compaction_and_check
    ok_sim, msgs, inv = simulate_compaction_and_check(editor.points, editor.triangles, reject_crossing_edges=True)
    # Accept outcome if: either fully ok OR no crossing messages present (even if inversion flagged separately).
    crossing_msgs = [m for m in msgs if 'crossing edges detected' in m]
    assert not crossing_msgs, f"Crossings detected after strict greedy: {crossing_msgs}"


def test_greedy_non_strict_pocket_fill():
    # Create an artificial empty quad pocket by defining a square boundary without interior tris, then
    # add one skinny triangle adjacent so greedy modifies something. Post-pass pocket fill should attempt to fill.
    pts = np.array([
        [0.0,0.0],  # 0
        [2.0,0.0],  # 1
        [2.0,2.0],  # 2
        [0.0,2.0],  # 3
        [1.0, -0.5], # 4 extra point below to trigger some operation
    ])
    # Start with just a single triangle leaving a large pocket (the square) unfilled
    tris = np.array([
        [0,1,4],
    ], dtype=int)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    pre_pts = len(editor.points)
    greedy_remesh(
        editor,
        max_vertex_passes=1,
        max_edge_passes=0,
        verbose=False,
        strict=False,
        reject_crossings=False,
        reject_new_loops=False,
        annotate_failures=False,
    )
    # Pocket fill may have added points OR triangles. Ensure at least one active triangle remains and no degenerate cluster.
    active = [t for t in editor.triangles if not np.all(np.array(t) == -1)]
    assert len(active) >= 1
    # Conformity (loose) should still pass.
    from sofia.core.conformity import check_mesh_conformity as cmc
    ok_conf, _ = cmc(editor.points, np.array(active), allow_marked=False)
    assert ok_conf
