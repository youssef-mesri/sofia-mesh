import json
import os
import numpy as np

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.config import GreedyConfig
from sofia.core.remesh_driver import greedy_remesh


def build_quad_pocket():
    # Four boundary vertices forming a square with no interior triangles -> pocket fill should add at least 2 triangles
    pts = np.array([
        [0.0,0.0],  #0
        [1.0,0.0],  #1
        [1.0,1.0],  #2
        [0.0,1.0],  #3
    ])
    tris = np.empty((0,3), dtype=int)  # start empty
    return pts, tris


def _run_greedy(editor, *, rejected_log_path=None, **cfg_kwargs):
    cfg = GreedyConfig(**cfg_kwargs)
    return greedy_remesh(editor, config=cfg, rejected_log_path=rejected_log_path)


def test_force_pocket_fill_adds_triangles(tmp_path):
    pts, tris = build_quad_pocket()
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # Run greedy with force pocket fill so it attempts to fill the square.
    _run_greedy(editor, max_vertex_passes=0, max_edge_passes=0, strict=True, force_pocket_fill=True)
    active = [t for t in editor.triangles if not np.all(np.array(t)==-1)]
    assert len(active) >= 2, f"Expected pocket fill to add triangles, found {len(active)}"


def test_rejected_ops_json_logging(tmp_path):
    # Construct a mesh that will cause at least one crossing rejection when strict+crossing enabled
    pts = np.array([
        [0.0,0.0], [1.0,0.0], [0.5,0.8], [0.5,-0.6], [1.2,0.4]
    ])
    tris = np.array([[0,1,2],[0,1,3],[1,2,4],[1,4,3]])
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    log_file = tmp_path / 'rejected.json'
    _run_greedy(
        editor,
        max_vertex_passes=0,
        max_edge_passes=2,
        strict=True,
        reject_crossings=True,
        rejected_log_path=str(log_file),
    )
    assert log_file.exists()
    data = json.loads(log_file.read_text())
    assert 'n_rejected' in data and data['n_rejected'] >= 0
    if data['n_rejected']:
        # Check schema of first entry
        first = data['rejections'][0]
        assert {'phase','op','reason'} <= set(first.keys())
