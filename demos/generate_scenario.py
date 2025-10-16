#!/usr/bin/env python3
"""
Generate a scenario: build a polygon boundary (no interior vertices), create a base mesh,
apply pocket_fill to the first boundary domain, then run a simple refinement pass.

This script reuses utilities from `demos/refinement_scenario.py` where appropriate.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Tuple, List

import numpy as np
import json
import csv

from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.sofia.pocket_fill import fill_pocket_earclip, fill_pocket_steiner, fill_pocket_quad
from sofia.sofia.visualization import plot_mesh
from sofia.sofia.quality import compute_h
from sofia.sofia.constants import EPS_COLINEAR
from sofia.sofia.quality import _triangle_qualities_norm


def regular_ngon(n: int = 8, radius: float = 1.0, center=(0.0, 0.0)) -> np.ndarray:
    cx, cy = float(center[0]), float(center[1])
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    return pts


def build_mesh_from_polygon(poly_pts: np.ndarray, extra_seed_pts: int = 0) -> PatchBasedMeshEditor:
    """Construct an editor whose points are the polygon vertices and no initial triangles.

    The pocket-fill routines will then triangulate the polygon by appending triangles
    to the editor. We intentionally avoid calling Delaunay here.
    """
    pts = np.asarray(poly_pts, dtype=float)
    # Start with no triangles (empty Mx3 array)
    tris = np.empty((0, 3), dtype=int)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    return editor


def try_fill_first_boundary(editor: PatchBasedMeshEditor, boundary_cycle: List[int]):
    """Try pocket fill strategies on the supplied `boundary_cycle` (list of vertex indices).

    Strategy order: quad -> steiner -> earclip (fallback).
    """
    min_tri_area = 1e-8
    reject_min_angle_deg = None
    if len(boundary_cycle) == 4:
        ok, det = fill_pocket_quad(editor, boundary_cycle, min_tri_area, reject_min_angle_deg)
        if ok:
            return True, f'quad: {det}'
    if len(boundary_cycle) >= 5:
        ok, det = fill_pocket_steiner(editor, boundary_cycle, min_tri_area, reject_min_angle_deg)
        if ok:
            return True, f'steiner: {det}'
    ok, det = fill_pocket_earclip(editor, boundary_cycle, min_tri_area, reject_min_angle_deg)
    return ok, det


def map_coords_to_indices(editor: PatchBasedMeshEditor, coords: np.ndarray, tol: float = 1e-8) -> List[int]:
    """Map polygon coordinates to the nearest editor vertex indices.

    Returns a list of indices (same length as coords). Raises ValueError if any
    coordinate does not match an editor vertex within `tol`.
    """
    coords = np.asarray(coords, dtype=float)
    out = []
    pts = np.asarray(editor.points, dtype=float)
    for p in coords:
        d2 = np.sum((pts - p)**2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] > (tol * tol):
            raise ValueError(f'No editor vertex within tol for coord {p} (min d2={d2[idx]:.3g})')
        out.append(idx)
    return out


def validate_polygon_cycle(editor: PatchBasedMeshEditor, cycle_indices: List[int]) -> Tuple[bool, str]:
    """Validate winding and colinearity of a polygon cycle (vertex indices).

    Returns (ok, message). Uses `EPS_COLINEAR` to detect degenerate polygons.
    """
    if len(cycle_indices) < 3:
        return False, 'polygon must have at least 3 vertices'
    pts = editor.points
    coords = np.asarray([pts[int(i)] for i in cycle_indices], dtype=float)
    # compute signed area
    x = coords[:, 0]; y = coords[:, 1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(signed_area) <= EPS_COLINEAR:
        return False, 'polygon is (near-)colinear or degenerate'
    # ensure consistent winding (positive area); if negative, reverse
    if signed_area < 0:
        return True, 'ok (reversed winding)'
    return True, 'ok'


def list_internal_edges(editor: PatchBasedMeshEditor, include_boundary: bool = False):
    edges = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts if not np.all(editor.triangles[int(t)] == -1)]
        deg = len(ts_list)
        if deg == 2 or (include_boundary and deg == 1):
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
    edges = list({tuple(x) for x in edges})
    return edges


def avg_internal_edge_length(editor: PatchBasedMeshEditor, include_boundary: bool = False) -> float:
    edges = list_internal_edges(editor, include_boundary=include_boundary)
    if not edges:
        return 0.0
    lengths = []
    for u, v in edges:
        d = editor.points[int(u)] - editor.points[int(v)]
        lengths.append(float(np.hypot(d[0], d[1])))
    return float(np.mean(lengths)) if lengths else 0.0


def refine_to_target_h_local(editor: PatchBasedMeshEditor, target_h: float, include_boundary: bool = False,
                             max_splits: int = 1000, tol: float = 1e-8):
    """Refine by splitting longest internal edges until avg internal edge length <= target_h.

    Returns (ok: bool, details: dict)
    """
    details = {'initial_h': None, 'target_h': float(target_h), 'splits': 0, 'quality_before': None, 'quality_after': None}
    curr_h = avg_internal_edge_length(editor, include_boundary=include_boundary)
    details['initial_h'] = float(curr_h)
    if curr_h <= target_h + tol:
        # nothing to do
        # compute quality and return
        active_mask = ~np.all(editor.triangles == -1, axis=1)
        active_tris = np.asarray(editor.triangles[active_mask], dtype=int) if np.any(active_mask) else np.empty((0,3), dtype=int)
        q = _triangle_qualities_norm(editor.points, active_tris) if active_tris.size else np.array([])
        details['quality_before'] = float(np.mean(q)) if q.size else 1.0
        details['quality_after'] = details['quality_before']
        return True, details
    # compute quality before
    active_mask = ~np.all(editor.triangles == -1, axis=1)
    active_tris = np.asarray(editor.triangles[active_mask], dtype=int) if np.any(active_mask) else np.empty((0,3), dtype=int)
    q_before = _triangle_qualities_norm(editor.points, active_tris) if active_tris.size else np.array([])
    details['quality_before'] = float(np.mean(q_before)) if q_before.size else 1.0

    splits = 0
    while curr_h > target_h + tol and splits < int(max_splits):
        edges = list_internal_edges(editor, include_boundary=include_boundary)
        if not edges:
            break
        # pick longest edge
        def edge_len(e):
            u, v = int(e[0]), int(e[1]); d = editor.points[u] - editor.points[v]; return float(np.hypot(d[0], d[1]))
        edges.sort(key=edge_len, reverse=True)
        longest = edges[0]
        ok, msg, _ = editor.split_edge((int(longest[0]), int(longest[1])))
        if not ok:
            # if split failed, remove this edge from consideration
            # try next longest
            edges.pop(0)
            if not edges:
                break
            continue
        splits += 1
        curr_h = avg_internal_edge_length(editor, include_boundary=include_boundary)
    details['splits'] = splits
    # compute quality after
    active_mask = ~np.all(editor.triangles == -1, axis=1)
    active_tris = np.asarray(editor.triangles[active_mask], dtype=int) if np.any(active_mask) else np.empty((0,3), dtype=int)
    q_after = _triangle_qualities_norm(editor.points, active_tris) if active_tris.size else np.array([])
    details['quality_after'] = float(np.mean(q_after)) if q_after.size else 1.0
    return (curr_h <= target_h + tol), details


def simple_refine(editor: PatchBasedMeshEditor, target_factor: float = 0.5):
    # Try to import a refinement helper from the package; fallback to the demo script if unavailable.
    cfg = {'target_h_factor': target_factor, 'h_metric': 'avg_internal_edge_length', 'max_h_iters': 3}
    try:
        from sofia.sofia.refinement import refine_to_target_h as _refine
    except Exception:
        try:
            from demos.refinement_scenario import refine_to_target_h as _refine
        except Exception:
            return False, 'no refine implementation available'
    try:
        _refine(editor, cfg)
        return True, 'refined'
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=8, help='polygon vertex count')
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--out', type=str, default='generate_scenario.png')
    parser.add_argument('--out-before', type=str, dest='out_before', default=None,
                        help='path to write mesh plot before pocket fill')
    parser.add_argument('--out-after', type=str, dest='out_after', default=None,
                        help='path to write mesh plot after pocket fill')
    parser.add_argument('--poly-file', type=str, dest='poly_file', default=None,
                        help='path to polygon file (.json, .csv, .npy) with sequence of [x,y] coords')
    args = parser.parse_args()

    # Load polygon from file if provided, otherwise create a regular n-gon
    if args.poly_file:
        def load_polygon(path: str) -> np.ndarray:
            path = str(path)
            if path.lower().endswith('.json'):
                with open(path, 'r') as fh:
                    data = json.load(fh)
                arr = np.asarray(data, dtype=float)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError('JSON polygon must be a list of [x,y] pairs')
                return arr
            if path.lower().endswith('.csv'):
                pts = []
                with open(path, 'r') as fh:
                    rdr = csv.reader(fh)
                    for row in rdr:
                        if not row: continue
                        if len(row) < 2: continue
                        pts.append([float(row[0]), float(row[1])])
                return np.asarray(pts, dtype=float)
            if path.lower().endswith('.npy'):
                return np.load(path)
            raise ValueError('Unsupported polygon file type; use .json, .csv, or .npy')

        poly = load_polygon(args.poly_file)
    else:
        poly = regular_ngon(args.n, radius=args.radius)
    editor = build_mesh_from_polygon(poly, extra_seed_pts=0)

    # Validate mapping and polygon winding/degeneracy
    try:
        # since we constructed editor from polygon points, indices are 0..len(poly)-1
        cycle_idx = list(range(len(poly)))
        ok_val, msg_val = validate_polygon_cycle(editor, cycle_idx)
        if not ok_val:
            print('polygon validation failed:', msg_val)
            return
        # If winding is reversed, reverse the cycle and the polygon coordinates
        if 'reversed' in msg_val:
            poly = poly[::-1]
            editor = build_mesh_from_polygon(poly, extra_seed_pts=0)
            cycle_idx = list(range(len(poly)))
    except Exception as e:
        print('polygon validation error:', e)
        return

    # Optional pre-fill plot
    if args.out_before:
        try:
            plot_mesh(editor, args.out_before)
            print('wrote before-plot', args.out_before)
        except Exception as e:
            print('before-plot failed:', e)

    ok, det = try_fill_first_boundary(editor, list(range(len(poly))))
    print('pocket_fill result:', ok, det)

    # Optional post-fill plot
    if args.out_after:
        try:
            plot_mesh(editor, args.out_after)
            print('wrote after-plot', args.out_after)
        except Exception as e:
            print('after-plot failed:', e)

    # Attempt a tiny refine pass if available
    try:
        from sofia.sofia.refinement import refine_to_target_h as refine_func
        okr, d = True, 'skipped'
        try:
            refine_func(editor, {'target_h_factor': 0.5, 'h_metric': 'avg_internal_edge_length', 'max_h_iters': 2})
            okr, d = True, 'refined'
        except Exception as e:
            okr, d = False, str(e)
        print('refine result:', okr, d)
    except Exception:
        print('refine: not available')

    try:
        plot_mesh(editor, args.out)
        print('wrote', args.out)
    except Exception as e:
        print('plot failed:', e)


if __name__ == '__main__':
    main()
