"""Triangulation utilities and strategies for polygon retriangulation.
Provides star triangulation selection methods, ear-clipping, and polygon
with-holes triangulation.
"""
from __future__ import annotations
import numpy as np
import logging
from scipy.spatial import Delaunay, ConvexHull
from .geometry import triangle_area, triangle_angles
from .constants import EPS_AREA, EPS_IMPROVEMENT, EPS_TINY
from .conformity import check_mesh_conformity
from .helpers import boundary_cycle_from_incident_tris

__all__ = [
    'optimal_star_triangulation',            # min total area star
    'best_star_triangulation_by_min_angle',  # maximize worst (minimum) angle
    'best_star_triangulation_area_preserving',  # choose star whose area matches polygon area
    'retriangulate_star',
    'retriangulate_patch_strict',
    'ear_clip_triangulation',
    'point_in_polygon',
    'polygon_signed_area',
    'polygon_has_self_intersections',
    'simplify_polygon_cycle',
    # pocket fill strategies
    'fill_pocket_quad', 'fill_pocket_steiner', 'fill_pocket_earclip',
    # removal triangulation strategies (new)
    'RemovalTriangulationStrategy',
    'OptimalStarStrategy',
    'QualityStarStrategy',
    'AreaPreservingStarStrategy',
    'EarClipStrategy',
    'SimplifyAndRetryStrategy',
    'ChainedStrategy',
]

def triangulate_polygon_with_holes(shell, holes, points=None, prefer_earcut=True):
    """Triangulate a polygon with holes without adding new dependencies.

    Uses mapbox_earcut if present and preferred; otherwise falls back to:
      - triangulate outer shell with ear-clipping
      - remove triangles whose centroids lie inside any hole

    Returns list of triangles as (x,y) coordinate triplets.
    """
    shell_coords = [tuple(p) for p in shell]
    hole_list = [[tuple(p) for p in h] for h in holes] if holes else []
    # try earcut if available
    if prefer_earcut:
        try:
            import mapbox_earcut as earcut  # type: ignore
        except Exception:
            earcut = None
        if earcut is not None:
            verts = []
            dims = 2
            hole_indices = []
            for v in shell_coords:
                verts.extend([float(v[0]), float(v[1])])
            for h in hole_list:
                hole_indices.append(len(verts)//dims)
                for v in h:
                    verts.extend([float(v[0]), float(v[1])])
            tri_func = getattr(earcut, 'triangulate_float32', None) or getattr(earcut, 'triangulate')
            idxs = tri_func(verts, hole_indices, dims)
            coord_list = list(shell_coords)
            for h in hole_list:
                coord_list.extend(h)
            tris = []
            for i in range(0, len(idxs), 3):
                a,b,c = int(idxs[i]), int(idxs[i+1]), int(idxs[i+2])
                tris.append((coord_list[a], coord_list[b], coord_list[c]))
            return tris

    # Fallback: perform Delaunay on all boundary vertices (shell + holes) then keep triangles
    # whose centroid lies inside the shell and outside all holes. This avoids complex bridge logic.
    try:
        pts_list = list(shell_coords)
        for h in hole_list:
            pts_list.extend(h)
        coords = np.asarray(pts_list, dtype=float)
        if coords.shape[0] < 3:
            raise RuntimeError("not enough points for triangulation")
        tri = Delaunay(coords)
        tris_coords = []
        for s in tri.simplices:
            a = coords[int(s[0])]; b = coords[int(s[1])]; c = coords[int(s[2])]
            area = abs(triangle_area(a, b, c))
            if area < EPS_AREA:
                continue
            centroid = np.mean(np.vstack((a, b, c)), axis=0)
            if not _point_in_polygon(centroid[0], centroid[1], shell_coords):
                continue
            inside_hole = False
            for h in hole_list:
                if _point_in_polygon(centroid[0], centroid[1], h):
                    inside_hole = True; break
            if inside_hole:
                continue
            tris_coords.append((tuple(a), tuple(b), tuple(c)))
        if not tris_coords:
            raise RuntimeError("Delaunay produced no usable triangles for polygon with holes")
        return tris_coords
    except Exception as e:
        raise RuntimeError(f"Failed to triangulate polygon with holes fallback: {e}")


def optimal_star_triangulation(points, boundary_indices, debug: bool=False):
    """Try all star triangulations of a simple polygon boundary selecting minimal total area.
    Returns list of triangle index triplets or None if no valid star.
    """
    if len(set(boundary_indices)) != len(boundary_indices):
        raise ValueError("Duplicate vertices detected in boundary_indices for optimal_star_triangulation")
    if _indices_polygon_self_intersects(points, boundary_indices):
        raise ValueError("Self-intersecting polygon (boundary_indices) in optimal_star_triangulation")
    def total_area(tris):
        return sum(abs(triangle_area(points[t[0]], points[t[1]], points[t[2]])) for t in tris)
    best_tris = None
    min_area = float('inf')
    n = len(boundary_indices)
    eps_area = EPS_AREA
    for i in range(n):
        v0 = boundary_indices[i]
        cycle = boundary_indices[i:] + boundary_indices[:i]
        tris = []
        degenerate = False
        for j in range(1, n-1):
            tri = [v0, cycle[j], cycle[j+1]]
            a = abs(triangle_area(points[tri[0]], points[tri[1]], points[tri[2]]))
            if a < eps_area:
                degenerate = True
                if debug:
                    logging.getLogger('sofia.triangulation').debug("Skipping star from %s due to near-zero area %s %.3e", v0, tri, a)
                break
            tris.append(tri)
        if degenerate:
            continue
        area = total_area(tris)
        if debug:
            logging.getLogger('sofia.triangulation').debug("Star from %s: area=%.6f tris=%s", v0, area, tris)
        if area < min_area:
            min_area = area
            best_tris = tris
    if debug:
        logging.getLogger('sofia.triangulation').debug("Chosen star: %s area=%.6f", best_tris, min_area)
    return best_tris

def best_star_triangulation_by_min_angle(points, boundary_indices, debug: bool=False):
    """Enumerate all star triangulations anchored at each boundary vertex and choose the one
    with the maximal worst (minimum) triangle angle (quality oriented). Falls back to area-based
    if angle computation encounters errors. Returns list of triangle triplets or None.
    """
    if len(set(boundary_indices)) != len(boundary_indices):
        raise ValueError("Duplicate vertices detected in boundary_indices for best_star_triangulation_by_min_angle")
    if _indices_polygon_self_intersects(points, boundary_indices):
        raise ValueError("Self-intersecting polygon (boundary_indices) in best_star_triangulation_by_min_angle")
    n = len(boundary_indices)
    if n < 3:
        return None
    best = None
    best_quality = -1.0
    eps_area = EPS_AREA
    for i in range(n):
        v0 = boundary_indices[i]
        cycle = boundary_indices[i:] + boundary_indices[:i]
        tris = []
        degenerate = False
        local_worst = 180.0
        for j in range(1, n-1):
            tri = [v0, cycle[j], cycle[j+1]]
            a = abs(triangle_area(points[tri[0]], points[tri[1]], points[tri[2]]))
            if a < eps_area:
                degenerate = True
                break
            try:
                angs = triangle_angles(points[tri[0]], points[tri[1]], points[tri[2]])
                local_worst = min(local_worst, min(angs))
            except Exception:
                pass
            tris.append(tri)
        if degenerate:
            continue
        if not tris:
            continue
        quality = local_worst
        if debug:
            logging.getLogger('sofia.triangulation').debug("Star(anchor=%s) worst_angle=%.4f tris=%s", v0, quality, tris)
        if quality > best_quality + EPS_IMPROVEMENT:
            best_quality = quality
            best = tris
    if debug and best is not None:
        logging.getLogger('sofia.triangulation').debug("Chosen quality star worst_angle=%.4f tris=%s", best_quality, best)
    return best

def best_star_triangulation_area_preserving(points, boundary_indices, area_tol_rel: float = EPS_TINY, area_tol_abs: float = EPS_AREA*4, debug: bool=False):
    """Enumerate star triangulations anchored at each boundary vertex and select one whose
    summed triangle area matches the polygon area within tolerance. Among valid candidates,
    prefer the one maximizing the worst (minimum) triangle angle for robustness.

    Returns list of triangle index triplets or None.
    """
    if len(set(boundary_indices)) != len(boundary_indices):
        raise ValueError("Duplicate vertices detected in boundary_indices for best_star_triangulation_area_preserving")
    if _indices_polygon_self_intersects(points, boundary_indices):
        raise ValueError("Self-intersecting polygon (boundary_indices) in best_star_triangulation_area_preserving")
    n = len(boundary_indices)
    if n < 3:
        return None
    # polygon area (absolute)
    poly = [points[int(v)] for v in boundary_indices]
    poly_area = abs(polygon_signed_area(poly))
    best = None
    best_quality = -1.0
    for i in range(n):
        v0 = boundary_indices[i]
        cycle = boundary_indices[i:] + boundary_indices[:i]
        tris = []
        degenerate = False
        worst_angle = 180.0
        area_sum = 0.0
        for j in range(1, n-1):
            tri = [v0, cycle[j], cycle[j+1]]
            p0 = points[int(tri[0])]; p1 = points[int(tri[1])]; p2 = points[int(tri[2])]
            a = abs(triangle_area(p0, p1, p2))
            if a < EPS_AREA:
                degenerate = True
                break
            area_sum += a
            try:
                angs = triangle_angles(p0, p1, p2)
                worst_angle = min(worst_angle, min(angs))
            except Exception:
                pass
            tris.append(tri)
        if degenerate or not tris:
            continue
        # area tolerance test
        if abs(area_sum - poly_area) <= max(area_tol_abs, area_tol_rel * max(1.0, poly_area)):
            if worst_angle > best_quality + EPS_IMPROVEMENT:
                best_quality = worst_angle
                best = tris
                if debug:
                    logging.getLogger('sofia.triangulation').debug("Area-preserving star (anchor=%s) worst_angle=%.4f area_sum=%.6e poly_area=%.6e", v0, worst_angle, area_sum, poly_area)
    if debug and best is not None:
        logging.getLogger('sofia.triangulation').debug("Chosen area-preserving star worst_angle=%.4f tris=%s", best_quality, best)
    return best

def retriangulate_star(boundary_indices):
    """Simple fan triangulation from first boundary vertex."""
    tris = []
    if not boundary_indices:
        return tris
    v0 = boundary_indices[0]
    for i in range(1, len(boundary_indices)-1):
        tris.append([v0, boundary_indices[i], boundary_indices[i+1]])
    return tris

def retriangulate_patch_strict(points, triangles, patch_tri_indices, new_point_coords=None, strict_mode='centroid'):
    """Local patch retriangulation using Delaunay (with optional added points) plus fallback ear-clipping.
    Returns (new_points, new_triangles_array, success_bool, new_tri_list).
    """
    patch_tri_indices = sorted(set(patch_tri_indices))
    if not patch_tri_indices:
        return points, triangles, False, []
    # collect patch nodes
    patch_nodes = sorted({int(v) for i in patch_tri_indices for v in triangles[i]})
    if len(patch_nodes) < 3:
        return points, triangles, False, []
    # Build outer polygon by boundary edge walk of patch
    # Reuse boundary cycle helper if patch is a star around a vertex; otherwise fall back to hull
    # For robustness we attempt hull if helper fails.
    try:
        # build candidate outer polygon by collecting boundary edges of patch
        from collections import defaultdict
        edge_count = defaultdict(int)
        for i in patch_tri_indices:
            tri = triangles[i]
            a,b,c = int(tri[0]), int(tri[1]), int(tri[2])
            for e in [(a,b),(b,c),(c,a)]:
                e = (min(e[0],e[1]), max(e[0],e[1]))
                edge_count[e] += 1
        boundary_edges = [e for e,cnt in edge_count.items() if cnt == 1]
        # order boundary edges into polygon
        if boundary_edges:
            adj = {}
            for a,b in boundary_edges:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            start = boundary_edges[0][0]
            poly = [start]
            prev = None; curr = start
            while True:
                nxts = [x for x in adj[curr] if x != prev]
                if not nxts: break
                nxt = nxts[0]
                if nxt == poly[0]: break
                poly.append(nxt)
                prev, curr = curr, nxt
            outer_poly = poly if len(poly) >= 3 else None
        else:
            outer_poly = None
    except Exception:
        outer_poly = None
    if outer_poly is None:
        coords = points[patch_nodes]
        if len(coords) < 3:
            return points, triangles, False, []
        try:
            hull = ConvexHull(coords)
            outer_poly = [patch_nodes[i] for i in hull.vertices]
        except Exception:
            return points, triangles, False, []
    polygon_coords = [tuple(points[int(v)]) for v in outer_poly]
    if _indices_polygon_self_intersects(points, outer_poly):
        return points, triangles, False, []
    # prepare Delaunay coords
    local_coords = points[patch_nodes].copy()
    added = np.asarray(new_point_coords) if new_point_coords is not None else np.empty((0,2))
    coords_for_delaunay = np.vstack((local_coords, added)) if added.size else local_coords
    if coords_for_delaunay.shape[0] < 3:
        return points, triangles, False, []
    try:
        tri_local = Delaunay(coords_for_delaunay).simplices.copy()
    except Exception:
        tri_local = None
    new_points_list = points.copy().tolist()
    new_point_global_indices = []
    if added.size:
        for coord in added:
            new_point_global_indices.append(len(new_points_list))
            new_points_list.append(np.asarray(coord))
    new_triangles_list = []
    if tri_local is not None:
        for t in tri_local:
            global_tri = []
            tri_pts = []
            for li in t:
                if li < len(patch_nodes):
                    gidx = int(patch_nodes[li])
                else:
                    gidx = new_point_global_indices[int(li - len(patch_nodes))]
                global_tri.append(gidx)
                tri_pts.append(np.asarray(new_points_list[gidx]))
            centroid = np.mean(np.vstack(tri_pts), axis=0)
            in_poly_centroid = _point_in_polygon(centroid[0], centroid[1], polygon_coords)
            if strict_mode == 'centroid':
                include = in_poly_centroid
            else:
                include = all(_point_in_polygon(p[0], p[1], polygon_coords) for p in tri_pts)
            if include:
                area = abs(triangle_area(tri_pts[0], tri_pts[1], tri_pts[2]))
                if area < EPS_AREA: continue
                if len(set(global_tri)) == 3:
                    new_triangles_list.append(tuple(global_tri))
    # fallback ear clipping
    if not new_triangles_list:
        ear_tris = _ear_clip_triangulation(outer_poly, points)
        if ear_tris:
            cleaned = []
            for t in ear_tris:
                a = abs(triangle_area(points[int(t[0])], points[int(t[1])], points[int(t[2])]))
                if a >= EPS_AREA and len(set(t))==3:
                    cleaned.append(tuple(t))
            new_triangles_list = cleaned
    triangles_new = np.array(new_triangles_list, dtype=int) if new_triangles_list else np.empty((0,3), dtype=int)
    new_points = np.asarray(new_points_list)
    tmp_tri_full = [tuple(t) for i,t in enumerate(triangles) if i not in patch_tri_indices]
    tmp_tri_full.extend(new_triangles_list)
    tmp_tri_full = np.array(tmp_tri_full, dtype=int) if tmp_tri_full else np.empty((0,3), dtype=int)
    ok_sub, _ = check_mesh_conformity(new_points, tmp_tri_full, allow_marked=False)
    if not ok_sub:
        return points, triangles, False, []
    return new_points, triangles_new, True, new_triangles_list

def polygon_signed_area(polygon):
    """Return signed area of polygon (list of (x,y)); positive if CCW."""
    arr = np.asarray(polygon)
    if arr.shape[0] < 3:
        return 0.0
    x = arr[:,0]; y = arr[:,1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def point_in_polygon(x, y, polygon):
    """Ray casting even-odd rule; polygon: list of (x,y)."""
    return _point_in_polygon(x, y, polygon)

def _point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    if n < 3: return False
    for i in range(n):
        x1,y1 = polygon[i]; x2,y2 = polygon[(i+1)%n]
        intersect = ((y1>y) != (y2>y)) and (x < (x2 - x1)*(y - y1)/(y2 - y1 + 1e-30) + x1)
        if intersect:
            inside = not inside
    return inside

def ear_clip_triangulation(points, poly_indices):
    """Public wrapper returning ear clipping triangulation for a polygon (list of global vertex indices)."""
    if len(set(poly_indices)) != len(poly_indices):
        raise ValueError("Duplicate vertices detected in poly_indices for ear_clip_triangulation")
    if _indices_polygon_self_intersects(points, poly_indices):
        raise ValueError("Self-intersecting polygon in ear_clip_triangulation")
    return _ear_clip_triangulation(poly_indices, points)

def _ear_clip_triangulation(poly_indices, points):
    if len(poly_indices) < 3:
        return []
    coords = np.asarray([points[int(v)] for v in poly_indices])
    x = coords[:,0]; y = coords[:,1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    verts = list(poly_indices)
    if signed_area < 0:
        verts = verts[::-1]
    def point_in_tri(pt, a, b, c):
        v0 = c - a; v1 = b - a; v2 = pt - a
        den = v0[0]*v1[1] - v1[0]*v0[1]
        if abs(den) < 1e-15: return False
        invDen = 1.0 / den
        u = (v2[0]*v1[1] - v1[0]*v2[1]) * invDen
        v = (v0[0]*v2[1] - v2[0]*v0[1]) * invDen
        return (u>=0) and (v>=0) and (u+v<=1)
    tris_out = []
    max_iter = max(1000, len(verts)*10)
    iter_count = 0
    while len(verts) > 3 and iter_count < max_iter:
        m = len(verts); ear_found = False
        for i in range(m):
            prev = verts[(i-1)%m]; curr = verts[i]; nxt = verts[(i+1)%m]
            pa = np.asarray(points[int(prev)]); pb = np.asarray(points[int(curr)]); pc = np.asarray(points[int(nxt)])
            cross = np.cross(pb - pa, pc - pb)
            if cross <= EPS_AREA: continue
            contains = False
            for v in verts:
                if v in (prev,curr,nxt): continue
                pv = np.asarray(points[int(v)])
                if point_in_tri(pv, pa,pb,pc): contains = True; break
            if contains: continue
            area = abs(triangle_area(pa,pb,pc))
            if area < EPS_AREA: continue
            tris_out.append((prev,curr,nxt))
            del verts[i]; ear_found = True; break
        if not ear_found: break
        iter_count += 1
    if len(verts) == 3:
        a,b,c = verts
        area = abs(triangle_area(points[int(a)], points[int(b)], points[int(c)]))
    if area >= EPS_AREA and len({a,b,c})==3:
            tris_out.append((a,b,c))
    return tris_out

# -------------------------
# Self-intersection detection utilities
# -------------------------
def polygon_has_self_intersections(polygon):
    """Return True if polygon (sequence of (x,y)) contains any pair of crossing non-adjacent edges."""
    if len(polygon) < 4:
        return False
    pts = [np.asarray(p, dtype=float) for p in polygon]
    n = len(pts)
    def seg_inter(a,b,c,d):
        if (np.all(a==c) or np.all(a==d) or np.all(b==c) or np.all(b==d)):
            return False
        def orient(p,q,r):
            return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
        o1 = orient(a,b,c); o2 = orient(a,b,d); o3 = orient(c,d,a); o4 = orient(c,d,b)
        if o1==0 and o2==0 and o3==0 and o4==0:
            return False
        return (o1*o2 < 0) and (o3*o4 < 0)
    for i in range(n):
        a = pts[i]; b = pts[(i+1)%n]
        for j in range(i+1, n):
            if j in (i, (i+1)%n, (i-1)%n):
                continue
            c = pts[j]; d = pts[(j+1)%n]
            if (i==0 and (j+1)%n==0):
                continue
            if seg_inter(a,b,c,d):
                return True
    return False

def _indices_polygon_self_intersects(points, indices):
    if len(indices) < 4:
        return False
    coords = [tuple(points[int(i)]) for i in indices]
    return polygon_has_self_intersections(coords)

# -------------------------
# Polygon cycle simplification (internal heuristic)
# -------------------------
def simplify_polygon_cycle(points, cycle_indices, cavity_tri_indices=None, triangles=None,
                           collinear_eps=1e-14, max_passes=2):
    """Attempt to produce a simple polygon from a possibly messy cycle.

    Steps:
      1. Remove consecutive duplicate indices.
      2. Prune strictly collinear middle vertices (area below collinear_eps scaled by bbox).
      3. If still self-intersecting and patch context provided (cavity_tri_indices & triangles),
         rebuild boundary by walking edges with single incidence.
    Returns new cycle list or None if unable to simplify safely.
    """
    import math
    if not cycle_indices or len(cycle_indices) < 3:
        return None
    # 1. Remove consecutive duplicates (including closing duplicate)
    cleaned = []
    for v in cycle_indices:
        if not cleaned or cleaned[-1] != v:
            cleaned.append(int(v))
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        return None
    pts = points
    # scale collinear_eps by bounding box diagonal
    bb_min = np.min(pts[cleaned], axis=0)
    bb_max = np.max(pts[cleaned], axis=0)
    diag = np.linalg.norm(bb_max - bb_min)
    scaled_eps = collinear_eps * max(1.0, diag*diag)

    def is_collinear(a,b,c):
        pa, pb, pc = pts[a], pts[b], pts[c]
        area2 = abs((pb[0]-pa[0])*(pc[1]-pa[1]) - (pb[1]-pa[1])*(pc[0]-pa[0]))
        return area2 <= scaled_eps

    passes = 0
    changed = True
    while changed and passes < max_passes and len(cleaned) > 3:
        changed = False
        out = []
        n = len(cleaned)
        for i in range(n):
            a = cleaned[(i-1) % n]
            b = cleaned[i]
            c = cleaned[(i+1) % n]
            if n > 3 and is_collinear(a,b,c):
                changed = True
                continue
            out.append(b)
        cleaned = out
        passes += 1
        if len(cleaned) < 3:
            return None

    # Early exit if already simple
    if len(set(cleaned)) == len(cleaned) and not _indices_polygon_self_intersects(pts, cleaned):
        return cleaned

    # Attempt boundary reconstruction from patch context
    if cavity_tri_indices is not None and triangles is not None:
        try:
            from collections import defaultdict
            edge_count = defaultdict(int)
            for ti in cavity_tri_indices:
                tri = triangles[int(ti)]
                a,b,c = int(tri[0]), int(tri[1]), int(tri[2])
                for e in [(a,b),(b,c),(c,a)]:
                    e2 = (min(e[0],e[1]), max(e[0],e[1]))
                    edge_count[e2] += 1
            boundary_edges = [e for e,cnt in edge_count.items() if cnt == 1]
            if not boundary_edges:
                return None
            adj = {}
            for a,b in boundary_edges:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            # pick a start with degree 2; if none, degree 1
            start = None
            for k,v in adj.items():
                if len(v) == 2:
                    start = k; break
            if start is None:
                start = boundary_edges[0][0]
            ordered = [start]
            prev = None; curr = start
            for _ in range(len(adj)*2):
                nxts = [x for x in adj[curr] if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt == ordered[0]:
                    break
                ordered.append(nxt)
                prev, curr = curr, nxt
            if len(ordered) >= 3 and len(set(ordered)) == len(ordered) and not _indices_polygon_self_intersects(pts, ordered):
                return ordered
        except Exception:
            return None
    return None

# -------------------------
# Pocket fill strategies (moved from pocket_fill.py)
# -------------------------

def _pf_point_in_tri(pt, a, b, c):
    v0 = c - a; v1 = b - a; v2 = pt - a
    den = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(den) < 1e-15:
        return False
    invDen = 1.0 / den
    u = (v2[0]*v1[1] - v1[0]*v2[1]) * invDen
    v = (v0[0]*v2[1] - v2[0]*v0[1]) * invDen
    return (u >= 0) and (v >= 0) and (u + v <= 1)

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
        eps_area = EPS_AREA
        for tri in tri_list:
            try:
                p0 = np.asarray(editor.points[int(tri[0])]); p1 = np.asarray(editor.points[int(tri[1])]); p2 = np.asarray(editor.points[int(tri[2])])
                a = triangle_area(p0, p1, p2)
                if a <= eps_area:
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

def fill_pocket_steiner(editor, verts, min_tri_area, reject_min_angle_deg):
    details = {'method': None, 'triangles': [], 'new_point_idx': None, 'failure_reasons': []}
    if len(verts) < 5:
        details['failure_reasons'].append('polygon too small for steiner')
        return False, details
    poly = list(verts)
    polygon_coords = np.asarray([editor.points[int(v)] for v in poly])
    x = polygon_coords[:,0]; y = polygon_coords[:,1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    steiner_pt = None
    if abs(signed_area) > 1e-15:
        cx = (1.0/(6.0*signed_area)) * np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))
        cy = (1.0/(6.0*signed_area)) * np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))
        cand = np.array([cx, cy])
        if point_in_polygon(cand[0], cand[1], [tuple(p) for p in polygon_coords]):
            steiner_pt = cand
    if steiner_pt is None:
        for i in range(1, len(poly)-1):
            a = polygon_coords[0]; b = polygon_coords[i]; c = polygon_coords[i+1]
            cand = (a + b + c) / 3.0
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
                if _pf_point_in_tri(pv, pa, pb, pc): contains = True; break
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
    eps_area = EPS_AREA
    for tri in tris_out:
        try:
            p0 = np.asarray(editor.points[int(tri[0])]); p1 = np.asarray(editor.points[int(tri[1])]); p2 = np.asarray(editor.points[int(tri[2])])
            a = triangle_area(p0, p1, p2)
            if a <= eps_area:
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


# ============================================================================
# Strategy Pattern for Removal Triangulation
# ============================================================================

class RemovalTriangulationStrategy:
    """Base class for triangulation strategies when removing a vertex.
    
    This provides a unified interface for different triangulation approaches
    used in vertex removal operations. Each strategy wraps existing triangulation
    functions and provides consistent error handling.
    """
    
    def try_triangulate(self, points, boundary_cycle, config=None):
        """Try to triangulate a boundary cycle.
        
        Args:
            points: (N, 2) array of all mesh points
            boundary_cycle: List of vertex indices forming the boundary
            config: Optional configuration (BoundaryRemoveConfig)
            
        Returns:
            tuple: (success: bool, triangles: list or None, error: str)
                - success: True if triangulation succeeded
                - triangles: List of triangle index triplets, or None on failure
                - error: Error message string (empty on success)
        """
        raise NotImplementedError("Subclasses must implement try_triangulate")


class OptimalStarStrategy(RemovalTriangulationStrategy):
    """Triangulation strategy using optimal star (minimum total area).
    
    This strategy creates a star triangulation by choosing the boundary vertex
    that minimizes the total area of the resulting triangulation.
    """
    
    def try_triangulate(self, points, boundary_cycle, config=None):
        """Try optimal star triangulation."""
        try:
            triangles = optimal_star_triangulation(points, boundary_cycle, debug=False)
            if triangles is None:
                return (False, None, "optimal_star_triangulation returned None")
            return (True, triangles, "")
        except ValueError as e:
            return (False, None, f"optimal_star_triangulation raised ValueError: {e}")
        except Exception as e:
            return (False, None, f"optimal_star_triangulation raised exception: {e}")


class QualityStarStrategy(RemovalTriangulationStrategy):
    """Triangulation strategy maximizing the minimum angle (quality-oriented).
    
    This strategy enumerates star triangulations anchored at each boundary vertex
    and selects the one that maximizes the worst (minimum) triangle angle.
    """
    
    def try_triangulate(self, points, boundary_cycle, config=None):
        """Try quality star triangulation."""
        try:
            triangles = best_star_triangulation_by_min_angle(points, boundary_cycle, debug=False)
            if triangles is None:
                return (False, None, "best_star_triangulation_by_min_angle returned None")
            return (True, triangles, "")
        except ValueError as e:
            return (False, None, f"best_star_triangulation_by_min_angle raised ValueError: {e}")
        except Exception as e:
            return (False, None, f"best_star_triangulation_by_min_angle raised exception: {e}")


class AreaPreservingStarStrategy(RemovalTriangulationStrategy):
    """Triangulation strategy that preserves area within tolerance.
    
    This strategy finds a star triangulation whose total area matches the
    original polygon area within specified tolerances. Among valid candidates,
    it prefers the one with the best minimum angle.
    """
    
    def try_triangulate(self, points, boundary_cycle, config=None):
        """Try area-preserving star triangulation."""
        try:
            # Extract tolerances from config if provided
            if config and hasattr(config, 'area_tol_rel'):
                from .constants import EPS_AREA
                area_tol_rel = config.area_tol_rel
                area_tol_abs = float(getattr(config, 'area_tol_abs_factor', 4.0)) * EPS_AREA
                triangles = best_star_triangulation_area_preserving(
                    points, boundary_cycle,
                    area_tol_rel=area_tol_rel,
                    area_tol_abs=area_tol_abs,
                    debug=False
                )
            else:
                triangles = best_star_triangulation_area_preserving(
                    points, boundary_cycle, debug=False
                )
            
            if triangles is None:
                return (False, None, "best_star_triangulation_area_preserving returned None")
            return (True, triangles, "")
        except ValueError as e:
            return (False, None, f"best_star_triangulation_area_preserving raised ValueError: {e}")
        except Exception as e:
            return (False, None, f"best_star_triangulation_area_preserving raised exception: {e}")


class EarClipStrategy(RemovalTriangulationStrategy):
    """Triangulation strategy using ear clipping algorithm.
    
    This is a robust fallback strategy that uses ear clipping to triangulate
    arbitrary simple polygons.
    """
    
    def try_triangulate(self, points, boundary_cycle, config=None):
        """Try ear clipping triangulation."""
        try:
            triangles = ear_clip_triangulation(points, boundary_cycle)
            if not triangles:
                return (False, None, "ear_clip_triangulation returned empty list")
            return (True, triangles, "")
        except ValueError as e:
            return (False, None, f"ear_clip_triangulation raised ValueError: {e}")
        except Exception as e:
            return (False, None, f"ear_clip_triangulation raised exception: {e}")


class SimplifyAndRetryStrategy(RemovalTriangulationStrategy):
    """Meta-strategy that simplifies the boundary cycle and retries with another strategy.
    
    This strategy first simplifies the boundary cycle by removing nearly-collinear
    vertices, then attempts triangulation with a wrapped strategy.
    """
    
    def __init__(self, wrapped_strategy):
        """Initialize with a wrapped strategy to apply after simplification.
        
        Args:
            wrapped_strategy: RemovalTriangulationStrategy to use after simplification
        """
        self.wrapped_strategy = wrapped_strategy
    
    def try_triangulate(self, points, boundary_cycle, config=None):
        """Try triangulation after simplifying the boundary cycle."""
        try:
            simplified = simplify_polygon_cycle(points, boundary_cycle)
            if not simplified or len(simplified) < 3:
                return (False, None, "simplify_polygon_cycle produced invalid result")
            
            # Try wrapped strategy on simplified cycle
            return self.wrapped_strategy.try_triangulate(points, simplified, config)
        except ValueError as e:
            return (False, None, f"simplify_polygon_cycle raised ValueError: {e}")
        except Exception as e:
            return (False, None, f"simplification raised exception: {e}")


class ChainedStrategy(RemovalTriangulationStrategy):
    """Meta-strategy that tries multiple strategies in sequence until one succeeds.
    
    This allows building a fallback chain of strategies, where each is tried
    in order until one produces a valid triangulation.
    """
    
    def __init__(self, strategies):
        """Initialize with a list of strategies to try in order.
        
        Args:
            strategies: List of RemovalTriangulationStrategy instances
        """
        self.strategies = strategies
    
    def try_triangulate(self, points, boundary_cycle, config=None):
        """Try each strategy in order until one succeeds."""
        errors = []
        for i, strategy in enumerate(self.strategies):
            success, triangles, error = strategy.try_triangulate(points, boundary_cycle, config)
            if success:
                return (True, triangles, "")
            errors.append(f"Strategy {i} ({strategy.__class__.__name__}): {error}")
        
        # All strategies failed
        combined_error = "; ".join(errors)
        return (False, None, f"All strategies failed: {combined_error}")

