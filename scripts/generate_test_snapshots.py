"""Generate additional snapshots mirroring a few test scenarios.

Outputs (in docs/):
- greedy_skinny_before.png, greedy_skinny_after.png
- greedy_heptagon_before.png, greedy_heptagon_after.png
- crossing_rejection.png (candidate rendered with boundary loops for visualization)
- pocket_fill_before.png, pocket_fill_after.png
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.sofia.remesh_driver import greedy_remesh
from sofia.sofia.visualization import plot_mesh


def ensure_docs() -> Path:
    d = ROOT / "docs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def skinny_triangle_case(docdir: Path):
    points = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [0.01, 0.0002],
        [2.5, 2.5],
        [2.5, -2.0],
    ])
    triangles = np.array([[0,1,2],[0,2,3],[1,2,3],[0,1,4],[1,2,4]], dtype=int)
    editor = PatchBasedMeshEditor(points.copy(), triangles.copy())
    plot_mesh(editor, outname=str(docdir / "greedy_skinny_before.png"), highlight_boundary_loops=True)
    greedy_remesh(editor, max_vertex_passes=2, max_edge_passes=1, verbose=False)
    plot_mesh(editor, outname=str(docdir / "greedy_skinny_after.png"), highlight_boundary_loops=True)


def heptagon_case(docdir: Path):
    n = 7
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    center = np.array([[0.0, 0.0]])
    points = np.vstack((ring, center))
    cidx = len(points) - 1
    tris = [[i, (i+1)%n, cidx] for i in range(n)]
    triangles = np.array(tris, dtype=int)
    editor = PatchBasedMeshEditor(points.copy(), triangles.copy())
    plot_mesh(editor, outname=str(docdir / "greedy_heptagon_before.png"), highlight_boundary_loops=True)
    greedy_remesh(editor, max_vertex_passes=1, max_edge_passes=0, verbose=False)
    plot_mesh(editor, outname=str(docdir / "greedy_heptagon_after.png"), highlight_boundary_loops=True)


def pocket_fill_case(docdir: Path):
    pts = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
    tris = np.empty((0,3), dtype=int)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    plot_mesh(editor, outname=str(docdir / "pocket_fill_before.png"), highlight_boundary_loops=True, loop_color_mode="uniform", loop_vertex_labels=True)
    greedy_remesh(editor, max_vertex_passes=0, max_edge_passes=0, strict=True, force_pocket_fill=True)
    plot_mesh(editor, outname=str(docdir / "pocket_fill_after.png"), highlight_boundary_loops=True)


def crossing_rejection_case(docdir: Path):
    pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), simulate_compaction_on_commit=True, reject_crossing_edges=True)
    # Visualize the candidate that would be rejected (replace second triangle)
    # For snapshot purposes, just draw the initial and overlay boundary loops (no candidate plotting primitives here).
    plot_mesh(editor, outname=str(docdir / "crossing_rejection.png"), highlight_boundary_loops=True)


def main():
    docdir = ensure_docs()
    skinny_triangle_case(docdir)
    heptagon_case(docdir)
    pocket_fill_case(docdir)
    crossing_rejection_case(docdir)


if __name__ == "__main__":
    main()
