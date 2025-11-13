#!/usr/bin/env python3
"""
Small demo: square with 4 boundary points and 1 interior point.
Split a boundary edge and save before/after plots.
"""
from __future__ import annotations

import argparse
import numpy as np

from sofia.core.logging_utils import configure_logging
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.visualization import plot_mesh


def square_with_center():
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 interior
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 0],
    ], dtype=int)
    return pts, tris


def main():
    ap = argparse.ArgumentParser(description='Boundary split demo: before/after plots')
    ap.add_argument('--edge', type=str, default='0,1', help='Boundary edge to split as "u,v" (default: 0,1)')
    ap.add_argument('--out-before', type=str, default='boundary_demo_before.png')
    ap.add_argument('--out-after', type=str, default='boundary_demo_after.png')
    ap.add_argument('--log-level', type=str, default='INFO')
    ap.add_argument('--virtual-boundary', action='store_true', help='Enable virtual_boundary_mode on editor')
    ap.add_argument('--enforce-split-quality', action='store_true', help='Enforce non-worsening split quality policy')
    args = ap.parse_args()

    configure_logging()

    pts, tris = square_with_center()
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(),
                                  virtual_boundary_mode=bool(args.virtual_boundary),
                                  enforce_split_quality=bool(args.enforce_split_quality))

    plot_mesh(editor, outname=args.out_before)

    # Parse edge and split
    try:
        u_str, v_str = args.edge.split(',')
        e = (int(u_str), int(v_str))
    except Exception:
        e = (0, 1)
    ok, msg, _ = editor.split_edge(e)
    if not ok:
        print(f"split_edge{e} failed: {msg}")

    # Optionally compact for a clean after view
    editor.compact_triangle_indices()
    plot_mesh(editor, outname=args.out_after)


if __name__ == '__main__':  # pragma: no cover
    main()
