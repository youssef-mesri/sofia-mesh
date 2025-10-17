"""Pocket fill strategies extracted from op_try_fill_pocket.

Each strategy mutates the editor on success and returns (ok: bool, details: dict).
The orchestrator in operations.op_try_fill_pocket is responsible for stats bookkeeping.
"""
from __future__ import annotations
import numpy as np
from .geometry import triangle_area, triangle_angles
from .conformity import check_mesh_conformity
from .constants import EPS_AREA, EPS_COLINEAR

__all__ = [
    'fill_pocket_quad',
    'fill_pocket_steiner',
    'fill_pocket_earclip'
]

def _point_in_tri(pt, a, b, c):
    v0 = c - a; v1 = b - a; v2 = pt - a
    den = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(den) < EPS_COLINEAR:
        return False
    invDen = 1.0 / den
    u = (v2[0]*v1[1] - v1[0]*v2[1]) * invDen
    v = (v0[0]*v2[1] - v2[0]*v0[1]) * invDen
    return (u >= 0) and (v >= 0) and (u + v <= 1)

# ---------------- Quad strategy -----------------

def fill_pocket_quad(editor, verts, min_tri_area, reject_min_angle_deg):
    details = {'method': None, 'triangles': [], 'new_point_idx': None, 'failure_reasons': []}
    if len(verts) != 4:
        details['failure_reasons'].append('not a quad')
        return False, details
    diag_pairs = [ ((verts[0], verts[2]), [(verts[0], verts[1], verts[2]), (verts[0], verts[2], verts[3])]),
                   ((verts[1], verts[3]), [(verts[0], verts[1], verts[3]), (verts[1], verts[2], verts[3])]) ]
    active_tris = [tuple(t) for t in editor.triangles if not np.all(np.array(t) == -1)]
    for diag, tri_list in diag_pairs:
        good = True
        for tri in tri_list:
            try:
                p0 = np.asarray(editor.points[int(tri[0])]); p1 = np.asarray(editor.points[int(tri[1])]); p2 = np.asarray(editor.points[int(tri[2])])
                area = abs(triangle_area(p0, p1, p2))
                if area <= min_tri_area:
                    details['failure_reasons'].append(f'quad_diag {diag} rejected: area {area} <= min_tri_area')
                    good = False; break
                angs = triangle_angles(p0, p1, p2)
                mn = min(angs)
                if reject_min_angle_deg is not None and mn < float(reject_min_angle_deg):
                    details['failure_reasons'].append(f'quad_diag {diag} rejected: min_angle {mn} < {reject_min_angle_deg}')
                    good = False; break
            except Exception:
                details['failure_reasons'].append(f'quad_diag {diag} exception during checks')
                good = False; break
        if not good:
            continue
        try:
            tmp_tri_full = active_tris.copy(); tmp_tri_full.extend([tuple(t) for t in tri_list])
            tmp_tri_full = np.array(tmp_tri_full, dtype=int) if tmp_tri_full else np.empty((0,3), dtype=int)
            ok_sub, msgs = check_mesh_conformity(editor.points, tmp_tri_full, allow_marked=False)
        except Exception:
            details['failure_reasons'].append(f'quad_diag {diag} conformity check exception')
            ok_sub = False
        if not ok_sub:
            details['failure_reasons'].append(f'quad_diag {diag} failed conformity: {msgs}')
            continue
        oriented = []
        for tri in tri_list:
            try:
                p0 = np.asarray(editor.points[int(tri[0])]); p1 = np.asarray(editor.points[int(tri[1])]); p2 = np.asarray(editor.points[int(tri[2])])
                a = triangle_area(p0, p1, p2)
                if a <= EPS_AREA:
                    oriented.append((tri[0], tri[2], tri[1]))
                else:
                    oriented.append(tuple(tri))
            except Exception:
                oriented.append(tuple(tri))
        start = len(editor.triangles)
        arr = np.array(oriented, dtype=np.int32)
        editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, arr]).astype(np.int32))
        for idx in range(start, len(editor.triangles)):
            editor._add_triangle_to_maps(idx)
        details['method'] = 'quad'
        details['triangles'] = [tuple(t) for t in oriented]
        return True, details
    if not details['failure_reasons']:
        details['failure_reasons'].append('quad: no valid diagonal')
    return False, details

# --------------- Steiner (fan) strategy ---------------

def fill_pocket_steiner(editor, verts, min_tri_area, reject_min_angle_deg):
    details = {'method': None, 'triangles': [], 'new_point_idx': None, 'failure_reasons': []}
    if len(verts) < 5:  # we only consider steiner for polygons > 4 in orchestrator
        details['failure_reasons'].append('polygon too small for steiner')
        return False, details
    poly = list(verts)
    polygon_coords = np.asarray([editor.points[int(v)] for v in poly])
    x = polygon_coords[:,0]; y = polygon_coords[:,1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    steiner_pt = None
    if abs(signed_area) > EPS_COLINEAR:
        cx = (1.0/(6.0*signed_area)) * np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))
        cy = (1.0/(6.0*signed_area)) * np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))
        cand = np.array([cx, cy])
        from .triangulation import point_in_polygon
        if point_in_polygon(cand[0], cand[1], [tuple(p) for p in polygon_coords]):
            steiner_pt = cand
    if steiner_pt is None:
        for i in range(1, len(poly)-1):
            a = polygon_coords[0]; b = polygon_coords[i]; c = polygon_coords[i+1]
            cand = (a + b + c) / 3.0
            from .triangulation import point_in_polygon
            if point_in_polygon(cand[0], cand[1], [tuple(p) for p in polygon_coords]):
                steiner_pt = cand; break
    if steiner_pt is None:
        details['failure_reasons'].append('no interior steiner point')
        return False, details
    try:
        new_points_candidate = np.vstack([editor.points, steiner_pt])
        new_idx = len(editor.points)
        fan_tris = []
        n = len(poly)
        for i in range(n):
            a = int(poly[i]); b = int(poly[(i+1) % n])
            fan_tris.append((a, b, new_idx))
        valid = True
        for tri in fan_tris:
            p0 = np.asarray(new_points_candidate[int(tri[0])]); p1 = np.asarray(new_points_candidate[int(tri[1])]); p2 = np.asarray(new_points_candidate[int(tri[2])])
            area = abs(triangle_area(p0, p1, p2))
            if area < min_tri_area: valid = False; break
            angs = triangle_angles(p0, p1, p2)
            if reject_min_angle_deg is not None and min(angs) < float(reject_min_angle_deg): valid = False; break
        if not valid:
            details['failure_reasons'].append('steiner fan failed local checks')
            return False, details
        active_tris = [tuple(t) for t in editor.triangles if not np.all(np.array(t) == -1)]
        tmp_tri_full = active_tris.copy(); tmp_tri_full.extend(fan_tris)
        tmp_tri_full = np.array(tmp_tri_full, dtype=int) if tmp_tri_full else np.empty((0,3), dtype=int)
        ok_sub, msgs = check_mesh_conformity(new_points_candidate, tmp_tri_full, allow_marked=False)
        if not ok_sub:
            details['failure_reasons'].append(f'steiner fan failed conformity: {msgs}')
            return False, details
        # commit
        editor.points = new_points_candidate
        start = len(editor.triangles)
        arr = np.array(fan_tris, dtype=np.int32)
        editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, arr]).astype(np.int32))
        for idx in range(start, len(editor.triangles)):
            editor._add_triangle_to_maps(idx)
        details['method'] = 'steiner'
        details['triangles'] = [tuple(t) for t in fan_tris]
        details['new_point_idx'] = new_idx
        return True, details
    except Exception as e:
        details['failure_reasons'].append(f'steiner exception: {e}')
        return False, details

# --------------- Ear clipping fallback ---------------

def fill_pocket_earclip(editor, verts, min_tri_area, reject_min_angle_deg):
    details = {'method': None, 'triangles': [], 'new_point_idx': None, 'failure_reasons': []}
    poly = list(verts)
    n = len(poly)
    if n < 3:
        details['failure_reasons'].append('invalid_input: less than 3 vertices')
        return False, details
    coords = np.asarray([editor.points[int(v)] for v in poly])
    x = coords[:,0]; y = coords[:,1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed_area < 0:
        poly = poly[::-1]
    verts_work = list(poly)
    tris_out = []
    max_iter = max(1000, n * 10)
    iter_count = 0
    while len(verts_work) > 3 and iter_count < max_iter:
        m = len(verts_work); ear_found = False
        for i in range(m):
            prev = verts_work[(i-1) % m]; curr = verts_work[i]; nxt = verts_work[(i+1) % m]
            pa = np.asarray(editor.points[int(prev)])
            pb = np.asarray(editor.points[int(curr)])
            pc = np.asarray(editor.points[int(nxt)])
            cross = np.cross(pb - pa, pc - pb)
            if cross <= EPS_AREA: continue
            contains = False
            for v in verts_work:
                if v in (prev, curr, nxt): continue
                pv = np.asarray(editor.points[int(v)])
                if _point_in_tri(pv, pa, pb, pc): contains = True; break
            if contains: continue
            area = abs(triangle_area(pa, pb, pc))
            if area < min_tri_area: continue
            angs = triangle_angles(pa, pb, pc)
            if reject_min_angle_deg is not None and min(angs) < float(reject_min_angle_deg): continue
            tris_out.append((prev, curr, nxt))
            del verts_work[i]; ear_found = True; break
        if not ear_found: break
        iter_count += 1
    if len(verts_work) == 3:
        a,b,c = verts_work
        area = abs(triangle_area(editor.points[int(a)], editor.points[int(b)], editor.points[int(c)]))
        if area >= min_tri_area and len({a,b,c})==3:
            tris_out.append((a,b,c))
    if not tris_out:
        details['failure_reasons'].append('earclip produced no triangles')
        return False, details
    # Area-preserving check: sum of triangle areas must match polygon area within tolerance
    poly_coords = np.asarray([editor.points[int(v)] for v in poly])
    x = poly_coords[:, 0]; y = poly_coords[:, 1]
    poly_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    tri_areas = [abs(triangle_area(editor.points[int(t[0])], editor.points[int(t[1])], editor.points[int(t[2])])) for t in tris_out]
    sum_tri_area = sum(tri_areas)
    if abs(sum_tri_area - abs(poly_area)) > 1e-8:
        details['failure_reasons'].append(f'earclip area mismatch: poly={poly_area}, tris={sum_tri_area}')
        return False, details
    active_tris = [tuple(t) for t in editor.triangles if not np.all(np.array(t) == -1)]
    try:
        tmp_tri_full = active_tris.copy(); tmp_tri_full.extend([tuple(t) for t in tris_out])
        tmp_tri_full = np.array(tmp_tri_full, dtype=int) if tmp_tri_full else np.empty((0,3), dtype=int)
        ok_sub, msgs = check_mesh_conformity(editor.points, tmp_tri_full, allow_marked=False)
    except Exception:
        ok_sub = False; msgs = []
    if not ok_sub:
        details['failure_reasons'].append(f'earclip failed conformity: {msgs}')
        return False, details
    oriented = []
    for tri in tris_out:
        try:
            p0 = np.asarray(editor.points[int(tri[0])]); p1 = np.asarray(editor.points[int(tri[1])]); p2 = np.asarray(editor.points[int(tri[2])])
            a = triangle_area(p0, p1, p2)
            if a <= EPS_AREA:
                oriented.append((tri[0], tri[2], tri[1]))
            else:
                oriented.append(tuple(tri))
        except Exception:
            oriented.append(tuple(tri))
    start = len(editor.triangles)
    arr = np.array(oriented, dtype=np.int32)
    editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, arr]).astype(np.int32))
    for idx in range(start, len(editor.triangles)):
        editor._add_triangle_to_maps(idx)
    details['method'] = 'earclip'
    details['triangles'] = [tuple(t) for t in oriented]
    return True, details
