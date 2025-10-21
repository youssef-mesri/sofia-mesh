#!/usr/bin/env python3
import numpy as np
import math
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.patch_batching import vertex_patch
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.utilities.inspect_boundary_debug')

def main():  # pragma: no cover
    pts, tris = build_random_delaunay(npts=40, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    nodes = sorted([v for v in editor.v_map.keys()], key=lambda v: len(editor.v_map.get(v,[])), reverse=True)
    seed = nodes[0]
    logger.info('seed %s', seed)
    tri_set = set(vertex_patch(editor, seed, radius=1))
    logger.info('tri_set %s', tri_set)
    points = np.asarray(editor.points)
    edge_count = {}
    for t_idx in tri_set:
        tri = editor.triangles[int(t_idx)]
        for i in range(3):
            a = int(tri[i]); b = int(tri[(i+1)%3])
            key = tuple(sorted((a,b)))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [e for e,c in edge_count.items() if c==1]
    logger.info('boundary_edges %s', boundary_edges)
    from collections import defaultdict
    adj = defaultdict(list)
    for a,b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    logger.info('adjacency:')
    for k in sorted(adj.keys()):
        logger.info('%s -> %s', k, adj[k])
    remaining = set(adj.keys())
    comps = []
    while remaining:
        start = next(iter(remaining))
        stack=[start]; comp=set()
        while stack:
            x=stack.pop()
            if x in comp: continue
            comp.add(x)
            for nb in adj[x]:
                if nb not in comp: stack.append(nb)
        comps.append(comp)
        for v in comp: remaining.discard(v)
    logger.info('components: %s', comps)
    comp = comps[0]
    comp_edges = set()
    for v in comp:
        for nb in adj[v]: comp_edges.add(tuple(sorted((v,nb))))
    comp_edges = list(comp_edges)
    logger.info('comp_edges %s', comp_edges)
    start_edge = min(comp_edges)
    logger.info('start_edge %s', start_edge)
    def angle_from(u,v):
        vu = points[int(v)] - points[int(u)]
        return math.atan2(vu[1], vu[0])
    visited_edges = set()
    def walk_oriented(a0,b0):
        logger.info('walking %s -> %s', a0, b0)
        loop = [int(a0), int(b0)]
        prev = int(a0); curr = int(b0)
        safety=0
        while True:
            neighs = adj.get(curr, [])
            candidates = [int(n) for n in neighs if int(n) != int(prev)]
            logger.debug('\tcurr %s prev %s neighs %s candidates %s', curr, prev, neighs, candidates)
            if not candidates:
                if loop[0] in neighs:
                    loop.append(loop[0]); logger.debug('\tclosing to start')
                break
            ang_in = angle_from(prev, curr)
            logger.debug('\tang_in %s', ang_in)
            best = None; best_ang=None
            for c in candidates:
                ang_out = angle_from(curr, c)
                d = (ang_out - ang_in) % (2*math.pi)
                if d <= 1e-8: d = 2*math.pi
                logger.debug('\tcandidate %s ang_out %s d %s', c, ang_out, d)
                if best is None or d < best_ang:
                    best = int(c); best_ang = d
            nxt = best
            ekey = tuple(sorted((int(curr), int(nxt))))
            logger.debug('\tnext %s edgekey %s visited? %s', nxt, ekey, ekey in visited_edges)
            if ekey in visited_edges:
                if nxt == loop[0]: loop.append(loop[0]); logger.debug('\tclosing by visited edge')
                break
            loop.append(int(nxt)); visited_edges.add(ekey)
            prev, curr = curr, nxt
            safety += 1
            if safety > 1000:
                logger.warning('safety break'); break
        logger.info('loop result %s', loop)
        if len(loop) >=4 and loop[0] == loop[-1]:
            return loop[:-1]
        return None
    p = walk_oriented(start_edge[0], start_edge[1])
    if p is None:
        p = walk_oriented(start_edge[1], start_edge[0])
    logger.info('poly %s', p)
    logger.info('visited_edges after walk %s', visited_edges)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
