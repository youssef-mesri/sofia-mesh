"""Patch-building and greedy MIS batching utilities for mesh patches.

Internal canonical location (migrated from repository root).
"""
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import math
import numpy as np

from .mesh_modifier2 import is_boundary_vertex_from_maps
from .helpers import (boundary_polygons_from_patch, patch_nodes_for_triangles,
                      boundary_cycle_from_incident_tris)
from .geometry import EPS_AREA, point_in_polygon

# Re-export public functions expected by callers / tests
__all__ = [
    'triangle_metric','triangle_to_vertex_metric','vertex_to_edge_metric',
    'build_patches_from_metrics_strict','build_patches_from_triangles',
    'partition_batches','patch_boundary_loops','vertex_patch','edge_patch'
]

def vertex_patch(editor, v_idx: int, radius: int = 1):
    tri_set = set(editor.v_map.get(int(v_idx), set()))
    if radius <= 1:
        return set(tri_set)
    for _ in range(radius - 1):
        new_verts = set()
        for t in list(tri_set):
            tri = editor.triangles[int(t)]
            for v in tri:
                if int(v) >= 0:
                    new_verts.add(int(v))
        for v in new_verts:
            tri_set.update(editor.v_map.get(v, set()))
    return tri_set

def edge_patch(editor, edge, radius: int = 1):
    edge = tuple(sorted((int(edge[0]), int(edge[1]))))
    tri_set = set(editor.edge_map.get(edge, set()))
    if radius <= 1:
        return set(tri_set)
    for _ in range(radius - 1):
        new_verts = set()
        for t in list(tri_set):
            tri = editor.triangles[int(t)]
            for v in tri:
                if int(v) >= 0:
                    new_verts.add(int(v))
        for v in new_verts:
            tri_set.update(editor.v_map.get(v, set()))
    return tri_set

def triangle_metric(editor, t_idx: int, weights: Dict[str, float] = None) -> float:
    if weights is None:
        weights = {'angle': 1.0, 'circum_ratio': 1.0, 'aspect': 1.0}
    tri = editor.triangles[int(t_idx)]
    try:
        if tri is None or len(tri) < 3 or all(int(v) < 0 for v in tri):
            return 0.0
    except Exception:
        return 0.0
    p0 = editor.points[int(tri[0])]; p1 = editor.points[int(tri[1])]; p2 = editor.points[int(tri[2])]
    def dist(a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    a = dist(p1,p2); b = dist(p2,p0); c = dist(p0,p1)
    shortest = min(a,b,c); longest = max(a,b,c)
    # area (robust)
    area = max(1e-20, abs(0.5*((p0[0]*(p1[1]-p2[1]) + p1[0]*(p2[1]-p0[1]) + p2[0]*(p0[1]-p1[1])))))
    R = (a*b*c)/(4.0*area + 1e-30)
    def ang(opp, s1, s2):
        num = s1*s1 + s2*s2 - opp*opp
        den = 2*s1*s2 + 1e-30
        v = max(-1.0, min(1.0, num/den))
        return math.degrees(math.acos(v))
    min_ang = min(ang(a,b,c), ang(b,c,a), ang(c,a,b))
    angle_comp = (90.0 - min_ang)/90.0
    circum_comp = math.log1p(R/(shortest+EPS_AREA))
    aspect_comp = math.log1p(longest/(shortest+EPS_AREA))
    return (weights['angle']*angle_comp + weights['circum_ratio']*circum_comp + weights['aspect']*aspect_comp)

def triangle_to_vertex_metric(editor, weights: Dict[str,float]=None) -> Dict[int,float]:
    v_metric = defaultdict(float); counts = defaultdict(int)
    for t_idx, tri in enumerate(editor.triangles):
        try:
            if np.all(np.array(tri) == -1):
                continue
        except Exception:
            continue
        score = triangle_metric(editor, t_idx, weights=weights)
        for v in tri:
            iv = int(v)
            if iv < 0: continue
            v_metric[iv] += score; counts[iv] += 1
    for v in list(v_metric.keys()):
        if counts[v] > 0:
            v_metric[v] /= counts[v]
    return dict(v_metric)

def vertex_to_edge_metric(editor, v_metric: Dict[int,float]) -> Dict[Tuple[int,int], float]:
    e_metric = {}
    for e in editor.edge_map.keys():
        a,b = int(e[0]), int(e[1])
        e_metric[tuple(sorted((a,b)))] = 0.5*(v_metric.get(a,0.0)+v_metric.get(b,0.0))
    return e_metric

def patch_boundary_loops(tris: Set[int], editor) -> List[List[int]]:
    # extract boundary edges (those used exactly once) and walk loops
    edge_count = {}
    for t in tris:
        tri = editor.triangles[int(t)]
        if np.all(np.array(tri) == -1):
            continue
        for i in range(3):
            a = int(tri[i]); b = int(tri[(i+1)%3])
            key = tuple(sorted((a,b)))
            edge_count[key] = edge_count.get(key,0)+1
    boundary_edges = [e for e,c in edge_count.items() if c==1]
    if not boundary_edges:
        return []
    adj = {}
    for a,b in boundary_edges:
        adj.setdefault(a,[]).append(b)
        adj.setdefault(b,[]).append(a)
    loops = []
    used = set()
    for a,b in boundary_edges:
        if (a,b) in used or (b,a) in used: continue
        loop = [a,b]
        used.add((a,b)); used.add((b,a))
        prev=a; cur=b
        while True:
            nbrs = adj.get(cur, [])
            nxt = None
            for nb in nbrs:
                if nb!=prev:
                    nxt=nb; break
            if nxt is None: break
            if nxt==loop[0]:
                loop.append(nxt); break
            if (cur,nxt) in used: break
            loop.append(nxt)
            used.add((cur,nxt)); used.add((nxt,cur))
            prev,cur = cur,nxt
        if len(loop)>=3 and loop[0]==loop[-1]:
            loop = loop[:-1]
        if len(loop)>=3:
            loops.append(loop)
    return loops

def build_patches_from_metrics_strict(editor, node_top_k: int=None, edge_top_k: int=None, radius: int=1, disjoint_on: str='tri', allow_overlap: bool=False) -> List[Dict]:
    v_metric = triangle_to_vertex_metric(editor)
    e_metric = vertex_to_edge_metric(editor, v_metric)
    nodes = sorted(v_metric.keys(), key=lambda v: v_metric[v], reverse=True)
    if node_top_k is not None: nodes = nodes[:node_top_k]
    edges = sorted(e_metric.keys(), key=lambda e: e_metric[e], reverse=True)
    if edge_top_k is not None: edges = list(edges)[:edge_top_k]
    patches = []; assigned_tris=set(); next_id=0
    for v in nodes:
        tris = set(vertex_patch(editor, v, radius=radius))
        if not tris: continue
        tris_used = tris if allow_overlap else (tris - assigned_tris)
        if not tris_used: continue
        polys = patch_boundary_loops(tris_used, editor)
        if not polys: continue
        seed_coord = tuple(editor.points[int(v)])
        chosen=None
        for poly in polys:
            coords=[tuple(editor.points[int(x)]) for x in poly]
            if point_in_polygon(seed_coord[0], seed_coord[1], coords):
                chosen=poly; break
        if chosen is None: continue
        verts=set()
        for t in tris_used:
            verts.update([int(x) for x in editor.triangles[int(t)]])
        patches.append({'id':next_id,'type':'node','seed':v,'tris':set(tris_used),'verts':verts,'badness':v_metric.get(v,0.0),'boundary':[chosen]})
        assigned_tris.update(tris_used); next_id+=1
    for e in edges:
        tris = set(edge_patch(editor, e, radius=radius))
        if not tris: continue
        tris_used = tris if allow_overlap else (tris - assigned_tris)
        if not tris_used: continue
        polys = patch_boundary_loops(tris_used, editor)
        if not polys: continue
        verts=set()
        for t in tris_used:
            verts.update([int(x) for x in editor.triangles[int(t)]])
        patches.append({'id':next_id,'type':'edge','seed':e,'tris':set(tris_used),'verts':verts,'badness':e_metric.get(tuple(sorted(e)),0.0),'boundary':polys})
        assigned_tris.update(tris_used); next_id+=1
    return patches

def build_patches_from_triangles(editor, tri_indices: List[int], mode: str='vertex', radius: int=1, disjoint_on: str='tri', top_k: int=None, allow_overlap: bool=False) -> List[Dict]:
    chosen = tri_indices[:]
    if top_k is not None:
        chosen = chosen[:top_k]
    patches=[]; assigned=set(); next_id=0
    for tidx in chosen:
        center_tris = {tidx}
        if mode=='vertex':
            # gather triangles incident to vertices of tidx within radius
            tri = editor.triangles[int(tidx)]
            verts = [int(v) for v in tri]
            local_tris=set()
            for v in verts:
                local_tris.update(vertex_patch(editor, v, radius=radius))
            center_tris = local_tris
        tris_used = center_tris if allow_overlap else (center_tris - assigned)
        if not tris_used: continue
        polys = patch_boundary_loops(tris_used, editor)
        if not polys: continue
        verts=set()
        for t in tris_used:
            verts.update([int(x) for x in editor.triangles[int(t)]])
        if disjoint_on=='tri' and not allow_overlap:
            assigned.update(tris_used)
        patches.append({'id':next_id,'type':mode,'seed':tidx,'tris':set(tris_used),'verts':verts,'badness':0.0,'boundary':polys})
        next_id+=1
    return patches

def partition_batches(patches: List[Dict]) -> List[List[int]]:
    # Greedy MIS over conflict graph (tri-based conflicts)
    tris_by_patch={p['id']: set(int(t) for t in p['tris']) for p in patches}
    remaining=set(tris_by_patch.keys())
    batches=[]
    while remaining:
        batch=[]; used_tris=set()
        for pid in sorted(list(remaining)):
            if tris_by_patch[pid] & used_tris:
                continue
            batch.append(pid)
            used_tris.update(tris_by_patch[pid])
        batches.append(batch)
        for pid in batch:
            remaining.remove(pid)
    return batches
