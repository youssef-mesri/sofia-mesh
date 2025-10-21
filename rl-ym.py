# mesh_patch_env_realflip.py
# Patch-based mesh editing environment with real edge-flip operation.
# Requirements: numpy, scipy, matplotlib
# Usage: python mesh_patch_env_realflip.py

import numpy as np
from scipy.spatial import Delaunay
import math
import matplotlib.pyplot as plt
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.rl_ym')

# ---------- Utilities ----------
def triangle_angles(p0, p1, p2):
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p0 - p2)
    c = np.linalg.norm(p0 - p1)
    def ang(A,B,C):
        cosang = (B*B + C*C - A*A) / (2*B*C)
        return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))
    return [ang(a,b,c), ang(b,c,a), ang(c,a,b)]

def mesh_min_angle(points, triangles):
    min_a = 180.0
    for tri in triangles:
        angs = triangle_angles(points[tri[0]], points[tri[1]], points[tri[2]])
        min_a = min(min_a, min(angs))
    return min_a

def build_edge_to_tri_map(triangles):
    edge_map = {}
    for t_idx, tri in enumerate(triangles):
        for i in range(3):
            a = int(tri[i])
            b = int(tri[(i+1)%3])
            key = tuple(sorted((a,b)))
            edge_map.setdefault(key, []).append(t_idx)
    return edge_map

def edge_flip_if_possible(points, triangles, e):
    """
    e: tuple (i,j) sorted
    triangles: numpy array (ntri,3)
    Return (new_triangles, success)
    """
    edge_map = build_edge_to_tri_map(triangles)
    if e not in edge_map:
        return triangles, False
    adj = edge_map[e]
    if len(adj) != 2:
        # boundary edge or non-flippable
        return triangles, False
    t1_idx, t2_idx = adj
    t1 = triangles[t1_idx].tolist()
    t2 = triangles[t2_idx].tolist()
    a,b = e
    opp1 = [v for v in t1 if v not in e][0]
    opp2 = [v for v in t2 if v not in e][0]
    # Build candidate new triangles
    new_tri1 = [opp1, opp2, a]
    new_tri2 = [opp1, opp2, b]
    # Check positive area
    def area(tri):
        p0,p1,p2 = points[int(tri[0])], points[int(tri[1])], points[int(tri[2])]
        return 0.5 * np.cross(p1-p0, p2-p0)
    if abs(area(new_tri1)) < 1e-12 or abs(area(new_tri2)) < 1e-12:
        return triangles, False
    # Avoid duplicate triangles
    existing_sets = [set(tri) for tri in triangles]
    if set(new_tri1) in existing_sets or set(new_tri2) in existing_sets:
        return triangles, False
    # Perform flip
    new_tris = triangles.copy()
    new_tris[t1_idx] = new_tri1
    new_tris[t2_idx] = new_tri2
    return new_tris, True

# ---------- Patch extraction ----------
def extract_patch_nodes(points, triangles, tri_idx, ring=1):
    """
    Return list of node indices belonging to the patch: tri_idx triangle + ring neighbors.
    """
    tri = triangles[tri_idx]
    patch_tris = set([tri_idx])
    # iterative growth by triangle neighbors sharing >=2 vertices
    for _ in range(ring):
        new = set()
        for j in range(len(triangles)):
            if j in patch_tris: continue
            # if triangle j shares an edge with any triangle in patch_tris -> include
            for t in patch_tris:
                if len(set(triangles[j]).intersection(triangles[t])) >= 2:
                    new.add(j); break
        if not new:
            break
        patch_tris.update(new)
    patch_nodes = set()
    for t in patch_tris:
        patch_nodes.update(triangles[t])
    return sorted(patch_nodes), sorted(patch_tris)

def opposite_edge_of_smallest_angle(points, triangle):
    angs = triangle_angles(points[triangle[0]], points[triangle[1]], points[triangle[2]])
    i_min = int(np.argmin(angs))
    indices = list(triangle)
    opp_edge = (indices[(i_min+1)%3], indices[(i_min+2)%3])
    return tuple(sorted(opp_edge))

# ---------- Editor class ----------
class PatchBasedMeshEditor:  # pragma: no cover (deprecated shim)
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Deprecated experimental module: use PatchBasedMeshEditor from 'mesh_modifier2'."
        )
    def sample_patch(self):
        return 0, [], []
    def apply_flip_on_triangle_smallest_angle(self, tri_idx):
        return False, None
    def global_min_angle(self):
        return 0.0

# ---------- Helper to build random Delaunay ----------
def build_random_delaunay(npts=60, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.rand(npts, 2)
    corners = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    pts = np.vstack((pts, corners))
    tri = Delaunay(pts)
    return pts, tri.simplices.copy()

# ---------- Demo ----------
if __name__ == "__main__":
    pts, tris = build_random_delaunay(50, seed=123)
    editor = PatchBasedMeshEditor(pts, tris)
    logger.info('Initial min angle: %s', editor.global_min_angle())
    # show initial mesh
    plt.figure(figsize=(5,5))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, linewidth=0.6)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=8)
    plt.gca().set_aspect('equal')
    plt.title("Initial mesh")
    plt.savefig("mesh_before.png", dpi=150)
    plt.close()

    # random policy: sample patch and flip with probability p
    n_steps = 200
    succ = 0
    for i in range(n_steps):
        tri_idx, nodes, neigh = editor.sample_patch()
        if np.random.rand() < 0.5:
            ok, e = editor.apply_flip_on_triangle_smallest_angle(tri_idx)
            if ok: succ += 1

    logger.info('After %d random attempts, successful flips = %d', n_steps, succ)
    logger.info('Final min angle: %s', editor.global_min_angle())

    plt.figure(figsize=(5,5))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, linewidth=0.6)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=8)
    plt.gca().set_aspect('equal')
    plt.title("After random attempts")
    plt.savefig("mesh_after.png", dpi=150)
    logger.info('Saved images mesh_before.png and mesh_after.png in current directory.')
