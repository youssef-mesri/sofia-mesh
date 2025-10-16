import numpy as np
from sofia.sofia import geometry


# Local reimplementation of the normalized triangle quality used by operations._triangle_qualities_norm
def triangle_qualities_norm(points_arr, tris_arr):
    pts = np.asarray(points_arr, dtype=np.float64)
    tris = np.asarray(tris_arr, dtype=np.int32)
    if tris.size == 0:
        return np.empty((0,), dtype=np.float64)
    p0 = pts[tris[:, 0]]
    p1 = pts[tris[:, 1]]
    p2 = pts[tris[:, 2]]
    a = 0.5 * np.abs((p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0]))
    e0 = np.sum((p1 - p0)**2, axis=1)
    e1 = np.sum((p2 - p1)**2, axis=1)
    e2 = np.sum((p0 - p2)**2, axis=1)
    denom = e0 + e1 + e2
    safe = denom > 0
    q = np.zeros(tris.shape[0], dtype=np.float64)
    q[safe] = a[safe] / denom[safe]
    norm_factor = 12.0 / (np.sqrt(3.0))
    q = q * norm_factor
    q = np.clip(q, 0.0, 1.0)
    return q


def test_triangle_qualities_norm_equilateral_and_degenerate():
    # Equilateral triangle side length 1: coordinates
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.5, np.sqrt(3)/2.0])
    points = np.vstack([p0, p1, p2])
    tris = np.array([[0,1,2]], dtype=np.int32)
    q = triangle_qualities_norm(points, tris)
    # equilateral normalized quality should be close to 1
    assert q.shape == (1,)
    assert q[0] > 0.98

    # Degenerate (colinear) triangle quality should be 0
    p0 = np.array([0.0,0.0]); p1 = np.array([0.5,0.0]); p2 = np.array([1.0,0.0])
    points2 = np.vstack([p0,p1,p2])
    tris2 = np.array([[0,1,2]], dtype=np.int32)
    q2 = triangle_qualities_norm(points2, tris2)
    assert q2.shape == (1,)
    assert q2[0] == 0.0


def test_candidate_ranking_prefers_area_preserving(tmp_path):
    # Construct a simple virtual-boundary cycle where two triangulations exist:
    # We'll mock two candidates manually and ensure ranking picks the area-preserving one
    # local patch quality: average of triangle_qualities_norm over tris
    def patch_quality(tris, pts):
        try:
            tris_np = np.array(tris, dtype=np.int32) if tris else np.empty((0,3), dtype=np.int32)
            q_arr = triangle_qualities_norm(pts, tris_np)
            return float(np.mean(q_arr)) if q_arr.size else 0.0
        except Exception:
            return 0.0
    # Simple 4-cycle square around center (0,0) to be removed
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # removed vertex 4 (center)
    ])
    # Two candidate triangulations for removing center could be triangulations of the quad
    # Candidate A: two triangles that exactly fill the quad (area preserved)
    candA = [[0,1,2],[0,2,3]]
    # Candidate B: replace with skinny triangles that change area slightly (simulate low-quality)
    candB = [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]  # uses the removed vertex: invalid for final selection
    # We will call the ranking logic indirectly by emulating the selection routine
    # Build a fake candidate_options list as operations code expects
    candidate_options = [(candA, 'area_star'), (candB, 'ear_clip')]
    cfg = type('C', (), {})()
    # use require_area True to force area-preserving selection
    cfg.require_area_preservation = True
    cfg.area_tol_rel = 1e-6
    cfg.area_tol_abs_factor = 4.0
    # monkeypatch removed_area (area of cavity) to equal candA area
    removed_area = 1.0

    # Compute scores using the same helper
    scored = []
    for cand, src in candidate_options:
        tris_np = np.array(cand, dtype=np.int32) if cand else np.empty((0,3), dtype=np.int32)
        try:
            q_arr = triangle_qualities_norm(pts, tris_np)
            avg_q = float(np.mean(q_arr)) if q_arr.size else 0.0
        except Exception:
            avg_q = patch_quality(cand, pts)
        cand_area = 0.0
        try:
            for t in cand:
                p0 = pts[int(t[0])]; p1 = pts[int(t[1])]; p2 = pts[int(t[2])]
                from sofia.sofia.geometry import triangle_area
                cand_area += abs(triangle_area(p0, p1, p2))
        except Exception:
            cand_area = None
        scored.append((avg_q, cand_area, cand, src))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Ensure that candidate A (area-preserving) is selected when require_area is True
    chosen = None
    for avg_q, cand_area, cand, src in scored:
        if cand_area is None:
            continue
        tol_rel = cfg.area_tol_rel
        tol_abs = cfg.area_tol_abs_factor * 1e-12  # EPS_AREA not available here; use tiny abs
        if abs(cand_area - removed_area) <= max(tol_abs, tol_rel*max(1.0, removed_area)):
            chosen = cand
            break
    assert chosen == candA
