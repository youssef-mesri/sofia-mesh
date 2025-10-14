import numpy as np
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor


def test_reset_stats_zeroes_counters():
    pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    editor = PatchBasedMeshEditor(pts, tris)

    # perform some ops
    editor.flip_edge((0,2))
    interior_edges = [e for e,s in editor.edge_map.items() if len(s)==2]
    if interior_edges:
        editor.split_edge(interior_edges[0])
    # ensure stats populated
    summary_before = editor.stats_summary()
    assert any(v['attempts'] > 0 for v in summary_before.values()), "Expected attempts > 0 before reset"

    editor.reset_stats()
    summary_after = editor.stats_summary()
    for op, stats in summary_after.items():
        assert stats['attempts'] == 0
        assert stats['success'] == 0
        assert stats['fail'] == 0
        assert stats['time_total'] == 0.0
        assert stats['time_max'] == 0.0
        assert stats['time_min'] == 0.0


def test_reset_stats_drop_ops():
    pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    editor = PatchBasedMeshEditor(pts, tris)
    editor.flip_edge((0,2))
    assert 'flip' in editor.stats_summary()
    editor.reset_stats(drop_ops=True)
    # After dropping ops registry should be empty except potential legacy alias recreation
    summary = editor.stats_summary()
    assert set(summary.keys()) in (set(), {'remove_node'}), f"Unexpected keys after drop: {list(summary.keys())}"
    # trigger an op again
    editor.flip_edge((0,2))
    assert 'flip' in editor.stats_summary()
