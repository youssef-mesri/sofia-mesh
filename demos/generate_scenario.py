#!/usr/bin/env python3
"""
Generate a scenario: build a polygon boundary (no interior vertices), create a base mesh,
apply pocket_fill to the first boundary domain, then run a simple refinement pass.

This script reuses utilities from `demos/refinement_scenario.py` where appropriate.
"""
from __future__ import annotations

import argparse
import cProfile as _cprof
import io as _io
import math
import os
import logging
import pstats as _pstats
from typing import Tuple, List

import numpy as np
import json
import csv

from sofia.core.logging_utils import configure_logging, get_logger


from sofia.core.geometry import triangle_area, triangle_angles, EPS_AREA


from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.pocket_fill import fill_pocket_earclip, fill_pocket_steiner, fill_pocket_quad
from sofia.core.visualization import plot_mesh
from sofia.core.quality import compute_h
from sofia.core.constants import EPS_COLINEAR
from sofia.core.quality import _triangle_qualities_norm
from sofia.core.refinement import refine_to_target_h
from sofia.core.triangulation import triangulate_polygon_with_holes

import os as _os
import matplotlib as _mpl
if not _os.environ.get('MPLBACKEND'):
    try:
        _mpl.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt

log = get_logger('sofia.demo.generate')

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


def _is_convex(a, b, c, pts):
    """Return True if angle abc is convex (polygon winding assumed CCW)."""
    pa, pb, pc = pts[a], pts[b], pts[c]
    return (pb[0]-pa[0])*(pc[1]-pa[1]) - (pb[1]-pa[1])*(pc[0]-pa[0]) > 0

def _ear_clip_decompose(boundary_cycle, pts):
    """Robust greedy ear clipping: returns list of triangles (as index lists).

    Heuristics:
    - Use EPS_COLINEAR to test convexity (robust against near-colinear triples).
    - Use a robust point-in-triangle test with a small tolerance.
    - Select the ear with the largest triangle area at each step to avoid leaving
      awkward skinny leftovers.
    - Safeguard against infinite loops by limiting iterations and handling
      colinear triples.
    """
    n = len(boundary_cycle)
    if n < 3:
        return []
    verts = list(boundary_cycle)
    pieces = []

    def orient(a, b, c):
        pa, pb, pc = pts[a], pts[b], pts[c]
        return (pb[0]-pa[0])*(pc[1]-pa[1]) - (pb[1]-pa[1])*(pc[0]-pa[0])

    def point_in_tri(p_coord, a, b, c, tol=1e-12):
        # p_coord is a coordinate array [x,y]. Use coordinate cross products.
        pa, pb, pc = pts[a], pts[b], pts[c]
        def cross(pa, pb, pc):
            return (pb[0]-pa[0])*(pc[1]-pa[1]) - (pb[1]-pa[1])*(pc[0]-pa[0])
        w0 = cross(pb, pc, p_coord)
        w1 = cross(pc, pa, p_coord)
        w2 = cross(pa, pb, p_coord)
        # If all have the same sign (or zero within tol), point is inside or on edge
        if (w0 >= -tol and w1 >= -tol and w2 >= -tol) or (w0 <= tol and w1 <= tol and w2 <= tol):
            return True
        return False

    max_iters = max(3 * n, 2000)
    iters = 0
    # Main loop: clip ears until 3 vertices remain
    while len(verts) > 3 and iters < max_iters:
        iters += 1
        best_idx = None
        best_area = -1.0
        L = len(verts)
        for i in range(L):
            a, b, c = verts[i-1], verts[i], verts[(i+1) % L]
            cross = orient(a, b, c)
            # require strictly positive orientation greater than colinearity tolerance
            if cross <= float(EPS_COLINEAR):
                continue
            # check no other vertex lies inside triangle abc
            any_inside = False
            for v in verts:
                if v in (a, b, c):
                    continue
                if point_in_tri(pts[v], a, b, c, tol=1e-12):
                    any_inside = True
                    break
            if any_inside:
                continue
            # compute triangle area as heuristic (choose largest)
            area = abs(0.5 * cross)
            if area > best_area:
                best_area = area
                best_idx = i

        # No valid ear found; try to clean colinear vertices: remove any vertex whose
        # adjacent triple is nearly colinear (this reduces degenerate loops)
        if best_idx is None:
            cleaned = False
            for j in range(len(verts)):
                a, b, c = verts[j-1], verts[j], verts[(j+1) % len(verts)]
                if abs(orient(a, b, c)) <= float(EPS_COLINEAR):
                    # drop b from polygon (it's redundant)
                    log.debug("[ear_clip_decompose] Removing nearly-colinear vertex: %s", b)
                    verts.pop(j)
                    cleaned = True
                    break
            if cleaned:
                continue
            # Still nothing: return remaining polygon as one piece (caller may retry other strategies)
            log.debug("[ear_clip_decompose] Fallback: remaining verts as one piece: %s", verts)
            pieces.append(list(verts))
            return pieces

        # Clip the chosen ear
        a, b, c = verts[best_idx-1], verts[best_idx], verts[(best_idx+1) % len(verts)]
        pieces.append([a, b, c])
        log.debug("[ear_clip_decompose] Added triangle: %s (area=%.6g)", [a, b, c], best_area)
        # remove the ear vertex b
        verts.pop(best_idx)

    # Append final triangle if present
    if len(verts) == 3:
        pieces.append(list(verts))
        log.debug("[ear_clip_decompose] Final triangle: %s", verts)
    log.debug("[ear_clip_decompose] All triangles/pieces: %s", pieces)
    return pieces

def try_fill_first_boundary(editor: PatchBasedMeshEditor, boundary_cycle: List[int], boundary_holes: List[List[int]] = None):
    """Try pocket fill strategies on the supplied `boundary_cycle` (list of vertex indices).

    Strategy order: quad -> steiner -> earclip; fallback: convex decomposition and fill each piece.
    """
    min_tri_area = 1e-8
    reject_min_angle_deg = None
    pts = np.asarray(editor.points)
    # Compute area of the polygon
    poly_coords = np.asarray([pts[i] for i in boundary_cycle])
    x = poly_coords[:, 0]; y = poly_coords[:, 1]
    poly_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # Helper to check area preservation for a set of triangles
    def area_preserving(tris_to_add, area_ref):
        curr_pts = np.asarray(editor.points)
        tri_areas = [abs(triangle_area(curr_pts[t[0]], curr_pts[t[1]], curr_pts[t[2]])) for t in tris_to_add]
        sum_tri_area = sum(tri_areas)
        # Use package EPS_AREA tolerance for area checks
        return abs(sum_tri_area - abs(area_ref)) <= float(EPS_AREA), tri_areas, sum_tri_area

    # If holes are provided as lists of editor vertex indices, try the
    # polygon-with-holes triangulation helper which returns triangles as
    # coordinate triplets. We then map coordinates back to vertex indices and
    # append to the editor after area checks.
    if boundary_holes:
        try:
            shell_coords = np.asarray([pts[i] for i in boundary_cycle], dtype=float)
            hole_coords_list = [np.asarray([pts[int(j)] for j in h], dtype=float) for h in boundary_holes]
            tris_coords = triangulate_polygon_with_holes(shell_coords, hole_coords_list, points=pts, prefer_earcut=True)
            tris_to_add = []
            for tri in tris_coords:
                coords = np.asarray([tri[0], tri[1], tri[2]], dtype=float)
                idxs = map_coords_to_indices(editor, coords)
                tris_to_add.append([int(idxs[0]), int(idxs[1]), int(idxs[2])])
            # compute reference area = outer - sum(hole areas)
            hole_area_sum = 0.0
            for hc in hole_coords_list:
                hx = hc[:, 0]; hy = hc[:, 1]
                hole_area_sum += 0.5 * (np.dot(hx, np.roll(hy, -1)) - np.dot(hy, np.roll(hx, -1)))
            total_area_ref = abs(poly_area) - abs(hole_area_sum)
            area_ok, tri_areas, sum_tri_area = area_preserving(tris_to_add, total_area_ref)
            if not area_ok:
                return False, {'holes': 'area not preserved', 'area': total_area_ref, 'tri_areas': tri_areas}
            # append triangles (orient if necessary)
            oriented = []
            eps_area = EPS_AREA
            for t in tris_to_add:
                try:
                    p0 = np.asarray(editor.points[int(t[0])]); p1 = np.asarray(editor.points[int(t[1])]); p2 = np.asarray(editor.points[int(t[2])])
                    a = triangle_area(p0, p1, p2)
                    if a <= eps_area:
                        oriented.append((t[0], t[2], t[1]))
                    else:
                        oriented.append(tuple(t))
                except Exception:
                    oriented.append(tuple(t))
            start = len(editor.triangles)
            if oriented:
                arr = np.array(oriented, dtype=np.int32)
                editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, arr]).astype(np.int32))
                for idx in range(start, len(editor.triangles)):
                    try:
                        editor._add_triangle_to_maps(idx)
                    except Exception:
                        pass
            if hasattr(editor, '_update_maps'):
                editor._update_maps()
            return True, {'holes': 'triangulated', 'area': total_area_ref, 'tri_areas': tri_areas}
        except Exception as e:
            return False, {'holes': f'triangulation failed: {e}'}

    # Detect non-convexity: if any internal angle > 180 deg, it's non-convex
    def is_convex_polygon(indices, pts):
        n = len(indices)
        for i in range(n):
            a, b, c = indices[i-1], indices[i], indices[(i+1)%n]
            pa, pb, pc = pts[a], pts[b], pts[c]
            cross = (pb[0]-pa[0])*(pc[1]-pa[1]) - (pb[1]-pa[1])*(pc[0]-pa[0])
            if cross < 0:
                return False
        return True

    import pprint
    # Constrained Delaunay fallback moved to helper module
    try:
        from sofia.core.triangulation_utils import constrained_delaunay_triangulate
    except Exception:
        constrained_delaunay_triangulate = None
    if not is_convex_polygon(boundary_cycle, pts):
        # Use ear clip decomposition directly for non-convex polygons
        pieces = _ear_clip_decompose(boundary_cycle, pts)
        # If decomposition made no progress (returned the same cycle as a piece),
        # avoid recursing indefinitely and report failure for this cycle.
        for piece in pieces:
            if len(piece) > 3 and set(piece) == set(boundary_cycle):
                log.debug("[try_fill_first_boundary] Ear-clip decomposition failed for cycle; aborting to avoid recursion: %s", boundary_cycle)
                return False, {'convex_decomp': [{'piece': boundary_cycle, 'result': 'earclip decomposition failed'}]}
        all_ok = True
        details = []
        for piece in pieces:
            poly_coords = np.asarray([pts[i] for i in piece])
            x = poly_coords[:, 0]; y = poly_coords[:, 1]
            poly_area_piece = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            tris_to_add = []
            if len(piece) == 3:
                tris_to_add = [piece]
                log.debug("[try_fill_first_boundary] Adding triangle: %s", piece)
            else:
                ok_piece, det_piece = try_fill_first_boundary(editor, piece)
                if ok_piece and isinstance(det_piece, dict) and 'convex_decomp' in det_piece:
                    for sub in det_piece['convex_decomp']:
                        if 'piece' in sub and len(sub['piece']) == 3:
                            tris_to_add.append(sub['piece'])
                if not tris_to_add:
                    # Try constrained Delaunay fallback
                    cd = constrained_delaunay_triangulate(piece)
                    if cd:
                        tris_to_add = cd
                    else:
                        all_ok = False
                        details.append({'piece': piece, 'result': 'triangulation failed'})
                        continue
            area_ok, tri_areas, sum_tri_area = area_preserving(tris_to_add, poly_area_piece)
            if not area_ok:
                all_ok = False
                details.append({'piece': piece, 'result': f'area mismatch: poly={poly_area_piece}, tris={sum_tri_area}'})
                continue
            for t, a in zip(tris_to_add, tri_areas):
                if a < 1e-10:
                    details.append({'piece': t, 'result': f'skipped: near-zero area ({a})'})
                    continue
                tris = np.asarray(editor.triangles)
                if not any(np.all(tris == t, axis=1)):
                    editor.triangles = np.vstack([editor.triangles, t])
                    log.debug("[try_fill_first_boundary] Triangle inserted: %s", t)
                    # Print current edge_map after each triangle insertion (will be rebuilt after all insertions)
            # record success for this piece
            details.append({'piece': piece, 'method': 'direct', 'area': poly_area_piece, 'tri_areas': tri_areas})
        # Rebuild edge and vertex maps once after all triangles are added
        if hasattr(editor, '_update_maps'):
            editor._update_maps()
        return all_ok, {'convex_decomp': details}

    # If convex, use classical methods
    # Try quad
    if len(boundary_cycle) == 4:
        ok, det = fill_pocket_quad(editor, boundary_cycle, min_tri_area, reject_min_angle_deg)
        if ok:
            tris_to_add = [editor.triangles[-2], editor.triangles[-1]] if len(editor.triangles) >= 2 else []
            area_ok, tri_areas, sum_tri_area = area_preserving(tris_to_add, poly_area)
            if area_ok:
                return True, {'quad': det, 'area': poly_area, 'tri_areas': tri_areas}
            else:
                editor.triangles = editor.triangles[:-2]
                return False, {'quad': 'area not preserved', 'area': poly_area, 'tri_areas': tri_areas}

    # Try steiner
    if len(boundary_cycle) >= 5:
        n_before = len(editor.triangles)
        ok, det = fill_pocket_steiner(editor, boundary_cycle, min_tri_area, reject_min_angle_deg)
        n_after = len(editor.triangles)
        tris_to_add = [editor.triangles[i] for i in range(n_before, n_after)]
        area_ok, tri_areas, sum_tri_area = area_preserving(tris_to_add, poly_area)
        if ok and area_ok:
            return True, {'steiner': det, 'area': poly_area, 'tri_areas': tri_areas}
        elif ok:
            editor.triangles = editor.triangles[:n_before]
            return False, {'steiner': 'area not preserved', 'area': poly_area, 'tri_areas': tri_areas}

    # Try earclip
    n_before = len(editor.triangles)
    ok, det = fill_pocket_earclip(editor, boundary_cycle, min_tri_area, reject_min_angle_deg)
    n_after = len(editor.triangles)
    tris_to_add = [editor.triangles[i] for i in range(n_before, n_after)]
    area_ok, tri_areas, sum_tri_area = area_preserving(tris_to_add, poly_area)
    if ok and area_ok:
        return True, {'earclip': det, 'area': poly_area, 'tri_areas': tri_areas}
    elif ok:
        editor.triangles = editor.triangles[:n_before]
        return False, {'earclip': 'area not preserved', 'area': poly_area, 'tri_areas': tri_areas}

    # Fallback: convex decomposition and fill each piece, but only if area is preserved
    pieces = _ear_clip_decompose(boundary_cycle, pts)
    # As above: if any piece is identical (as a vertex set) to the input cycle,
    # ear-clip couldn't decompose this polygon — bail out instead of recursing.
    for piece in pieces:
        if len(piece) > 3 and set(piece) == set(boundary_cycle):
            log.debug("[try_fill_first_boundary] Fallback ear-clip decomposition made no progress for cycle: %s", boundary_cycle)
            return False, {'convex_decomp': [{'piece': boundary_cycle, 'result': 'earclip decomposition failed (no progress)'}]}
    all_ok = True
    details = []
    for piece in pieces:
        poly_coords = np.asarray([pts[i] for i in piece])
        x = poly_coords[:, 0]; y = poly_coords[:, 1]
        poly_area_piece = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        tris_to_add = []
        if len(piece) == 3:
            tris_to_add = [piece]
        else:
            ok_piece, det_piece = try_fill_first_boundary(editor, piece)
            if ok_piece and isinstance(det_piece, dict) and 'convex_decomp' in det_piece:
                for sub in det_piece['convex_decomp']:
                    if 'piece' in sub and len(sub['piece']) == 3:
                        tris_to_add.append(sub['piece'])
            if not tris_to_add:
                # Try constrained Delaunay fallback
                cd = constrained_delaunay_triangulate(piece)
                if cd:
                    tris_to_add = cd
                else:
                    all_ok = False
                    details.append({'piece': piece, 'result': 'triangulation failed'})
                    continue
        area_ok, tri_areas, sum_tri_area = area_preserving(tris_to_add, poly_area_piece)
        if not area_ok:
            all_ok = False
            details.append({'piece': piece, 'result': f'area mismatch: poly={poly_area_piece}, tris={sum_tri_area}'})
            continue
        for t, a in zip(tris_to_add, tri_areas):
            if a < 1e-10:
                details.append({'piece': t, 'result': f'skipped: near-zero area ({a})'})
                continue
            tris = np.asarray(editor.triangles)
            if not any(np.all(tris == t, axis=1)):
                editor.triangles = np.vstack([editor.triangles, t])
        details.append({'piece': piece, 'method': 'direct', 'area': poly_area_piece, 'tri_areas': tri_areas})
    return all_ok, {'convex_decomp': details}


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


def _remove_bad_quality_vertices(editor: PatchBasedMeshEditor,
                                 min_deg: float = 20.0,
                                 max_rem: int = 2,
                                 allow_boundary: bool = True,
                                 prefix: str = 'sanity') -> int:
    """Remove up to max_rem vertices whose local star min-angle is below min_deg.
    Returns number of successful removals.
    """
    def local_star_min_angle(v: int) -> float:
        tris_idx = [int(t) for t in editor.v_map.get(int(v), []) if not np.all(editor.triangles[int(t)] == -1)]
        if not tris_idx:
            return 180.0
        mins = []
        for ti in tris_idx:
            a, b, c = [int(x) for x in editor.triangles[int(ti)]]
            A, B, C = triangle_angles(editor.points[a], editor.points[b], editor.points[c])
            mins.append(float(min(A, B, C)))
        return float(min(mins)) if mins else 180.0

    def is_boundary_vertex(v: int) -> bool:
        # deg==1 on any incident edge implies boundary
        for ti in editor.v_map.get(int(v), []):
            t = editor.triangles[int(ti)]
            for i in range(3):
                u, w = int(t[i]), int(t[(i+1)%3])
                key = tuple(sorted((u, w)))
                if v in (u, w):
                    inc = [tt for tt in editor.edge_map.get(key, []) if not np.all(editor.triangles[int(tt)] == -1)]
                    if len(inc) == 1:
                        return True
        return False

    all_vertices = list(editor.v_map.keys())
    scored = [(int(v), local_star_min_angle(int(v))) for v in all_vertices]
    scored = [(v, ang) for v, ang in scored if ang < float(min_deg)]
    if not allow_boundary:
        scored = [(v, ang) for v, ang in scored if not is_boundary_vertex(int(v))]
    scored.sort(key=lambda x: x[1])
    removed = 0
    for v, ang in scored:
        ok_rm, msg_rm, _ = editor.remove_node_with_patch(int(v))
        log.debug('%s remove_node v=%d (min_angle=%.3g) -> %s (%s)', prefix, v, ang, ok_rm, msg_rm)
        if ok_rm:
            removed += 1
            if removed >= int(max_rem):
                break
    if removed:
        log.info('%s: removed=%d (min_angle<%.3g)', prefix, removed, min_deg)
    return removed


def _compact_for_plot(editor: PatchBasedMeshEditor):
    """Return (new_points, new_tris, mapping, active_idx).
    mapping: orig_vertex_index -> compacted vertex index used in new_points/new_tris
    active_idx: list of original triangle indices kept (non-tombstoned), in order of new_tris rows.
    """
    tris = np.array(editor.triangles)
    pts = np.array(editor.points)
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    active_idx = np.nonzero(active_mask)[0].tolist()
    used_verts = sorted({int(v) for t in active_tris for v in t if int(v) >= 0})
    mapping = {old: new for new, old in enumerate(used_verts)}
    new_points = pts[used_verts]
    new_tris = []
    for t in active_tris:
        if np.any(np.array(t) < 0):
            continue
        try:
            new_tris.append([mapping[int(t[0])], mapping[int(t[1])], mapping[int(t[2])]])
        except KeyError:
            continue
    new_tris = np.array(new_tris, dtype=int)
    return new_points, new_tris, mapping, active_idx


def _boundary_edges_from_compacted(new_tris: np.ndarray):
    """Compute boundary edges (deg==1) from compacted triangles array."""
    from collections import defaultdict
    edge_count = defaultdict(int)
    for t in new_tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for e in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(e))
            edge_count[key] += 1
    return [e for e, c in edge_count.items() if c == 1]


def plot_mesh_with_annotations(editor: PatchBasedMeshEditor, outname: str,
                               new_vertices: list[int] | None = None,
                               show_boundary_edges: bool = True,
                               boundary_color=(0.85, 0.2, 0.2)):
    """Plot the mesh like plot_mesh, and overlay boundary edges and newly added vertices.

    new_vertices: list of original vertex indices to highlight (e.g., those added in the last step).
    """
    new_vertices = new_vertices or []
    pts, tris, mapping, _ = _compact_for_plot(editor)
    if tris.size == 0:
        plt.figure(figsize=(6, 6))
        plt.title('empty mesh')
        plt.savefig(outname, dpi=150)
        plt.close()
        return
    plt.figure(figsize=(6, 6))
    plt.triplot(pts[:, 0], pts[:, 1], tris, lw=0.6)
    plt.scatter(pts[:, 0], pts[:, 1], s=6)
    # Boundary edges overlay
    if show_boundary_edges:
        b_edges = _boundary_edges_from_compacted(tris)
        for u, v in b_edges:
            p, q = pts[int(u)], pts[int(v)]
            plt.plot([p[0], q[0]], [p[1], q[1]], color=boundary_color, linewidth=1.4)
    # Newly added vertices overlay (map original indices to compacted)
    for ov in new_vertices:
        cv = mapping.get(int(ov))
        if cv is None:
            continue
        p = pts[int(cv)]
        plt.plot([p[0]], [p[1]], marker='o', markersize=5.0, color=(0.1, 0.7, 0.2))
    plt.gca().set_aspect('equal')
    plt.title(outname)
    plt.savefig(outname, dpi=150)
    plt.close()


def _verify_boundary_split(editor: PatchBasedMeshEditor, edge, new_idx: int):
    """Sanity-check connectivity for a boundary split.
    Returns (ok, messages). Checks:
    - old edge removed
    - new edges (u,new) and (new,v) exist
    - both new edges are boundary (deg==1)
    - new vertex referenced by exactly 2 active triangles
    """
    msgs = []
    ok = True
    u, v = int(edge[0]), int(edge[1])
    old = tuple(sorted((u, v)))
    if old in editor.edge_map:
        ok = False; msgs.append(f"old edge {old} still present")
    e1 = (min(u, new_idx), max(u, new_idx))
    e2 = (min(v, new_idx), max(v, new_idx))
    if e1 not in editor.edge_map:
        ok = False; msgs.append(f"edge {e1} missing")
    if e2 not in editor.edge_map:
        ok = False; msgs.append(f"edge {e2} missing")
    # boundary degree check
    if e1 in editor.edge_map:
        deg1 = len([t for t in editor.edge_map[e1] if not np.all(editor.triangles[int(t)] == -1)])
        if deg1 != 1:
            ok = False; msgs.append(f"edge {e1} deg={deg1} (expected 1)")
    if e2 in editor.edge_map:
        deg2 = len([t for t in editor.edge_map[e2] if not np.all(editor.triangles[int(t)] == -1)])
        if deg2 != 1:
            ok = False; msgs.append(f"edge {e2} deg={deg2} (expected 1)")
    # new vertex star size
    if new_idx not in editor.v_map:
        ok = False; msgs.append(f"new vertex {new_idx} not in v_map")
    else:
        tri_refs = [ti for ti in editor.v_map[new_idx] if not np.all(editor.triangles[int(ti)] == -1)]
        if len(tri_refs) != 2:
            ok = False; msgs.append(f"new vertex {new_idx} referenced by {len(tri_refs)} active tris (expected 2)")
    return ok, msgs, (e1, e2)

def list_boundary_edges_only(editor: PatchBasedMeshEditor):
    edges = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts if not np.all(editor.triangles[int(t)] == -1)]
        if len(ts_list) == 1:
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
    return list({tuple(x) for x in edges})

def list_refinable_edges(editor: PatchBasedMeshEditor, include_boundary: bool = False):
    """Return edges considered for splitting.

    - When include_boundary=False: only edges with exactly 2 adjacent active triangles (interior)
    - When include_boundary=True: also include edges with exactly 1 adjacent active triangle (boundary)
    """
    edges = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts]
        # filter out tombstoned triangles
        ts_list = [t for t in ts_list if not np.all(editor.triangles[int(t)] == -1)]
        deg = len(ts_list)
        if deg == 2 or (include_boundary and deg == 1):
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
    # Deduplicate
    edges = list({tuple(x) for x in edges})
    return edges

# Use package-level `refine_to_target_h` imported from sofia.core.refinement

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
        refine_to_target_h(editor, cfg)
        return True, 'refined'
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Apply generate scenario (add_node, split_edge) from JSON')
    parser.add_argument('--scenario', type=str, required=True, help='Path to scenario JSON file')
    parser.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    parser.add_argument('--no-plot', action='store_true', help='Skip before/after plotting (useful for headless runs)')
    parser.add_argument('--n', type=int, default=8, help='polygon vertex count')
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--out', type=str, default='generate_scenario.png')
    parser.add_argument('--out-before', type=str, dest='out_before', default=None,
                        help='path to write mesh plot before pocket fill')
    parser.add_argument('--out-after', type=str, dest='out_after', default=None,
                        help='path to write mesh plot after pocket fill')
    parser.add_argument('--poly-file', type=str, dest='poly_file', default=None,
                        help='path to polygon file (.json, .csv, .npy) with sequence of [x,y] coords')
    parser.add_argument('--use-holes', action='store_true', dest='use_holes',
                        help='If set, read "holes" from the scenario JSON and pass them to try_fill_first_boundary')
    parser.add_argument('--target-h', type=float, dest='target_h', default=None,
                        help='absolute target average internal edge length to refine to')
    parser.add_argument('--target-factor', type=float, dest='target_factor', default=None,
                        help='relative factor of current avg internal edge length to refine to (e.g. 0.5)')
    parser.add_argument('--max-splits', type=int, dest='max_splits', default=200,
                        help='maximum edge splits when refining')
    parser.add_argument('--include-boundary-refine', action='store_true', dest='include_boundary',
                        help='include boundary edges in refinement edge list')
    parser.add_argument('--profile', action='store_true', help='Enable cProfile and print top hotspots')
    parser.add_argument('--profile-out', type=str, default=None, help='Write raw cProfile stats to this .pstats file when --profile is set')
    parser.add_argument('--profile-top', type=int, default=25, help='How many entries to show in hotspots (default: 25)')
    args = parser.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    
    if args.profile:
        pr = _cprof.Profile()
        pr.enable()
        run_generate_scenario(args)
        pr.disable()
        
        if args.profile_out:
            pr.dump_stats(args.profile_out)
            log.info('Wrote profiling stats to %s', args.profile_out)
        
        s = _io.StringIO()
        ps = _pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(args.profile_top)
        log.info('Profile (top %d by cumulative time):\n%s', args.profile_top, s.getvalue())
    else:
        run_generate_scenario(args)




def run_generate_scenario(args: argparse.Namespace):
    """Run the generate_scenario workflow using a parsed argparse Namespace.

    This function contains the previous `main` body so callers (and tests)
    can invoke the demo logic programmatically.
    """
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
    # default fill cycle: indices of polygon vertices (may be overridden by scenario)
    fill_cycle = list(range(len(poly)))

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


    # Always plot the empty mesh (polygon, no triangles) before pocket_fill
    # Use out_empty from config if present, else fallback to 'empty_mesh.png'
    out_empty = None
    try:
        with open(args.scenario, 'r') as f:
            scenario = json.load(f)
        plot_cfg = scenario.get('plot', {})
        out_empty = plot_cfg.get('out_empty', None)
    except Exception:
        pass
    if not out_empty:
        out_empty = 'empty_mesh.png'

    # Optional pre-fill plot (legacy: before pocket fill, but after empty mesh)
    if args.out_before:
        try:
            plot_mesh(editor, args.out_before)
            print('wrote before-plot', args.out_before)
        except Exception as e:
            print('before-plot failed:', e)

    # Helper to draw polygon shell and holes for empty-mesh visualization
    def _plot_empty_with_holes(shell_coords, holes_coords_list, outpath):
        try:
            plt.figure(figsize=(6, 6))
            # outer shell
            xs = list(shell_coords[:, 0]) + [shell_coords[0, 0]]
            ys = list(shell_coords[:, 1]) + [shell_coords[0, 1]]
            plt.plot(xs, ys, color=(0.85, 0.2, 0.2), linewidth=1.8)
            # holes
            for hc in holes_coords_list or []:
                if len(hc) >= 2:
                    hxs = list(hc[:, 0]) + [hc[0, 0]]
                    hys = list(hc[:, 1]) + [hc[0, 1]]
                    plt.plot(hxs, hys, color=(0.2, 0.6, 0.8), linewidth=1.6)
            # points
            all_pts = np.vstack([shell_coords] + (holes_coords_list or [])) if holes_coords_list else shell_coords
            npts = max(1, all_pts.shape[0])
            s = max(0.6, min(12.0, 200.0 / float(npts)))
            plt.scatter(all_pts[:, 0], all_pts[:, 1], s=s, color='black')
            plt.title(outpath)
            plt.gca().set_aspect('equal')
            plt.savefig(outpath, dpi=150)
            plt.close()
            print('wrote empty-mesh plot', outpath)
        except Exception as e:
            print('empty-mesh plot failed:', e)


    # If requested, attempt to read holes from scenario JSON and pass them to the fill helper.
    holes_to_pass = None
    try:
        with open(args.scenario, 'r') as f:
            scenario = json.load(f)
    except Exception:
        scenario = {}
    if args.use_holes:
        # If the scenario provides a 'poly' key (combined outer + hole verts), rebuild editor
        scen_poly = scenario.get('poly', None)
        if scen_poly is not None:
            try:
                poly = np.asarray(scen_poly, dtype=float)
                editor = build_mesh_from_polygon(poly, extra_seed_pts=0)
                # boundary indices for the outer ring may be supplied explicitly
                if 'boundary' in scenario:
                    fill_cycle = [int(x) for x in scenario.get('boundary')]
                else:
                    fill_cycle = list(range(len(poly)))
            except Exception as e:
                log.debug('Failed to load scenario poly: %s', e)
        raw_holes = scenario.get('holes', None)
        if raw_holes:
            holes_to_pass = []
            # raw_holes may be a list of index-lists or coordinate-lists.
            for h in raw_holes:
                # detect if the hole entries are coordinates (list of [x,y])
                if h and isinstance(h[0], (list, tuple, np.ndarray)):
                    # map coordinates to nearest indices in the editor
                    try:
                        mapped = map_coords_to_indices(editor, np.asarray(h, dtype=float))
                        holes_to_pass.append([int(x) for x in mapped])
                    except Exception as e:
                        log.debug('Failed to map hole coordinates to indices: %s', e)
                        holes_to_pass = None
                        break
                else:
                    # assume it's already an index list
                    try:
                        holes_to_pass.append([int(x) for x in h])
                    except Exception:
                        holes_to_pass = None
                        break
    # Plot empty mesh showing holes if available
    try:
        if holes_to_pass and 'poly' in scenario:
            shell_coords = np.asarray([editor.points[int(i)] for i in fill_cycle], dtype=float)
            holes_coords_list = [np.asarray([editor.points[int(j)] for j in h], dtype=float) for h in holes_to_pass]
            _plot_empty_with_holes(shell_coords, holes_coords_list, out_empty)
        else:
            plot_mesh(editor, out_empty)
            print('wrote empty-mesh plot', out_empty)
    except Exception as e:
        print('empty-mesh plot failed:', e)

    ok, det = try_fill_first_boundary(editor, fill_cycle, boundary_holes=holes_to_pass)
    print('pocket_fill result:', ok, det)

    # Diagnostics: print triangle areas and warn about near-zero area triangles
    tris = np.asarray(editor.triangles)
    pts = np.asarray(editor.points)
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    if active_tris.size > 0:
        areas = [abs(triangle_area(pts[t[0]], pts[t[1]], pts[t[2]])) for t in active_tris]
        min_area = min(areas)
        print(f"After pocket fill: {len(active_tris)} triangles, min area = {min_area:.3e}")
        for i, (t, a) in enumerate(zip(active_tris, areas)):
            if a < 1e-10:
                print(f"WARNING: Triangle {i} {t} has near-zero area ({a:.3e})")
        # Additional diagnostics: internal edges
        internal_edges = list_internal_edges(editor, include_boundary=False)
        print(f"After pocket fill: {len(internal_edges)} internal edges (not on boundary)")
        if internal_edges:
            print(f"Sample internal edges: {internal_edges[:5]}")
        else:
            print("No internal edges found; refinement will not proceed.")
    else:
        print("After pocket fill: mesh has no active triangles.")

    with open(args.scenario, 'r') as f:
        scenario = json.load(f)

    plot_cfg = scenario.get('plot', {})
    out_before = plot_cfg.get('out_before', 'gen_before.png')
    out_after = plot_cfg.get('out_after', 'gen_after.png')
    if not args.no_plot:
        plot_mesh(editor, outname=out_before)
        log.info('Wrote %s', out_before)

    # Either apply explicit ops list, or an auto refinement block
    auto_cfg = scenario.get('auto', None)
    if auto_cfg:
        # Enable virtual boundary topological mode when requested or when refining boundary edges
        try:
            if 'virtual_boundary_mode' in auto_cfg:
                editor.virtual_boundary_mode = bool(auto_cfg.get('virtual_boundary_mode', False))
                log.info('virtual_boundary_mode from scenario: %s', editor.virtual_boundary_mode)
            elif bool(auto_cfg.get('include_boundary_edges', False)):
                editor.virtual_boundary_mode = True
                log.info('virtual_boundary_mode enabled by include_boundary_edges')
        except Exception:
            pass
        # Optional toggle: enforce split quality (improvement) vs relax (refinement)
        if 'enforce_split_quality' in auto_cfg:
            val = bool(auto_cfg.get('enforce_split_quality', True))
            try:
                editor.enforce_split_quality = val
            except Exception:
                pass
        # Optional toggle: enable quad fast-path in remove_node
        if 'enable_remove_quad_fastpath' in auto_cfg:
            try:
                editor.enable_remove_quad_fastpath = bool(auto_cfg.get('enable_remove_quad_fastpath', False))
                log.info('enable_remove_quad_fastpath from scenario: %s', editor.enable_remove_quad_fastpath)
            except Exception:
                pass
        refine_to_target_h(editor, auto_cfg)
        editor.move_vertices_to_barycenter()
        refine_to_target_h(editor, auto_cfg)
        editor.move_vertices_to_barycenter()
        refine_to_target_h(editor, auto_cfg)
        editor.move_vertices_to_barycenter()
        refine_to_target_h(editor, auto_cfg)
    editor.compact_triangle_indices()
    # Always apply a post-pass smoothing to improve quality
    bary_passes = max(1, int(auto_cfg.get('barycenter_passes', 1)))
    total_moves_post = 0
    for _ in range(bary_passes):
        total_moves_post += editor.move_vertices_to_barycenter()
    log.info('auto: post barycenter moves applied=%d (passes=%d)', total_moves_post, bary_passes)
    if not args.no_plot:
        plot_mesh(editor, outname=out_after)
        log.info('Wrote %s', out_after)
    # Optionally refine to a target average edge length (absolute or relative)
    if args.target_h is not None or args.target_factor is not None:
        curr_h = avg_internal_edge_length(editor, include_boundary=args.include_boundary)
        if args.target_h is not None:
            target_h = float(args.target_h)
        else:
            target_h = float(curr_h * float(args.target_factor))
        ok_ref, det_ref = refine_to_target_h_local(editor, target_h, include_boundary=args.include_boundary,
                                                   max_splits=args.max_splits)
        print('refine_to_target_h_local result:', ok_ref, det_ref)


if __name__ == '__main__': 
    main()
