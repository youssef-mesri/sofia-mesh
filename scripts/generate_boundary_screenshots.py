"""Generate fresh screenshots demonstrating boundary loop rendering modes.

Outputs:
- boundary_loops_per_loop.png
- boundary_loops_uniform_labeled.png
"""
from __future__ import annotations

import numpy as np

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")

from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.visualization import plot_mesh


def carve_circular_holes(editor: PatchBasedMeshEditor, centers_radii):
    pts = np.asarray(editor.points)
    tris = np.asarray(editor.triangles)
    # compute centroids
    centroids = np.mean(pts[tris], axis=1)
    remove = np.zeros(len(tris), dtype=bool)
    for (cx, cy, r) in centers_radii:
        d2 = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
        remove |= d2 <= (r ** 2)
    # mark triangles as inactive
    editor.triangles[remove] = [-1, -1, -1]


def main():
    pts, tris = build_random_delaunay(npts=140, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # ensure docs folder exists at repo root
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    # place two circular holes relative to bbox
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    w, h = (maxx - minx), (maxy - miny)
    r = 0.18 * min(w, h)
    centers = [
        (minx + 0.35 * w, miny + 0.4 * h),
        (minx + 0.65 * w, miny + 0.6 * h),
    ]
    carve_circular_holes(editor, [(cx, cy, r) for (cx, cy) in centers])

    # Render per-loop colored boundaries
    plot_mesh(
        editor,
        outname=str(docs_dir / "boundary_loops_per_loop.png"),
        highlight_boundary_loops=True,
        loop_color_mode="per-loop",
        loop_vertex_labels=False,
        highlight_crossings=False,
    )

    # Render uniform boundaries with vertex indices
    plot_mesh(
        editor,
        outname=str(docs_dir / "boundary_loops_uniform_labeled.png"),
        highlight_boundary_loops=True,
        loop_color_mode="uniform",
        loop_vertex_labels=True,
        highlight_crossings=False,
    )


if __name__ == "__main__":
    main()
