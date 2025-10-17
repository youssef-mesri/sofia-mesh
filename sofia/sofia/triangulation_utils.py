"""Triangulation helper utilities.

Provides a constrained Delaunay fallback used by demo scripts.
"""
from typing import List, Optional
import numpy as np


def constrained_delaunay_triangulate(piece_indices: List[int], pts: np.ndarray,
                                     area_tol: float = 1e-14) -> Optional[List[List[int]]]:
    """Triangulate the polygon defined by `piece_indices` on coordinates `pts`.

    Uses scipy.spatial.Delaunay on the polygon vertex coordinates and keeps
    triangles whose centroids fall inside the polygon (matplotlib.path.Path).
    Returns a list of triangles as lists of original vertex indices, or None on
    failure.
    """
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return None
    try:
        from matplotlib.path import Path
    except Exception:
        return None

    if len(piece_indices) < 3:
        return None
    coords = np.asarray([pts[int(i)] for i in piece_indices], dtype=float)
    try:
        dela = Delaunay(coords)
    except Exception:
        return None

    path = Path(coords)
    out_tris = []
    for s in dela.simplices:
        tri_coords = coords[s]
        centroid = np.mean(tri_coords, axis=0)
        if path.contains_point(centroid):
            # map back to original indices
            tri = [int(piece_indices[int(i)]) for i in s]
            # sanity area check
            a = abs(0.5 * ((pts[tri[1]][0]-pts[tri[0]][0])*(pts[tri[2]][1]-pts[tri[0]][1]) -
                           (pts[tri[1]][1]-pts[tri[0]][1])*(pts[tri[2]][0]-pts[tri[0]][0])))
            if a > float(area_tol):
                out_tris.append(tri)
    if not out_tris:
        return None
    return out_tris
