#!/usr/bin/env python3
"""
Demo: Partition one mesh and run per-part smoothing in parallel (shared master mesh).

Relocated from repo root to `demos/`.
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import List, Dict, Set, Tuple, Any

import numpy as np

from sofia.core.logging_utils import configure_logging, get_logger
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.remesh_driver import build_random_delaunay, greedy_remesh
from sofia.core.patch_driver import run_patch_batch_driver
from sofia.core.config import PatchDriverConfig, RemeshConfig, GreedyConfig
from sofia.core.conformity import build_edge_to_tri_map
from sofia.core.visualization import plot_mesh, plot_mesh_by_tri_groups

log = get_logger('sofia.demos.partition')


def tri_adjacency(triangles: np.ndarray) -> List[List[int]]:
    """Adjacency list of triangles sharing an edge."""
    edge_map = build_edge_to_tri_map(triangles)
    adj: List[Set[int]] = [set() for _ in range(len(triangles))]
    for _, tri_set in edge_map.items():
        ts = list(set(int(t) for t in tri_set))
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                adj[a].add(b)
                adj[b].add(a)
    return [list(s) for s in adj]


def try_metis_partition(adj: List[List[int]], nparts: int) -> List[int]:
    import pymetis  # type: ignore
    n_cuts, parts = pymetis.part_graph(nparts, adjacency=adj)
    log.info('PyMetis partition: cuts=%d', n_cuts)
    return list(parts)


def bfs_balance_partition(adj: List[List[int]], nparts: int) -> List[int]:
    n = len(adj)
    parts = [-1] * n
    target = max(1, n // nparts)
    cur_part = 0
    remaining = set(range(n))
    from collections import deque
    while remaining and cur_part < nparts:
        start = remaining.pop()
        q = deque([start])
        parts[start] = cur_part
        count = 1
        while q and count < target:
            u = q.popleft()
            for v in adj[u]:
                if v in remaining:
                    remaining.remove(v)
                    parts[v] = cur_part
                    q.append(v)
                    count += 1
                    if count >= target:
                        break
        cur_part += 1
    # Assign any leftovers round-robin
    for v in list(remaining):
        parts[v] = (parts[v-1] + 1) % nparts if v > 0 else 0
    return parts


essential_keys = ('points', 'triangles')


def submesh_from_triangles(points: np.ndarray, triangles: np.ndarray, tri_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """Extract a compact submesh defined by tri_indices.

    Returns (sub_points, sub_tris, global_to_local_vert, local_to_global_vert).
    """
    tris = triangles[np.array(tri_indices, dtype=int)]
    used_verts = sorted({int(v) for t in tris for v in t if int(v) >= 0})
    g2l = {g: i for i, g in enumerate(used_verts)}
    l2g = {i: g for g, i in g2l.items()}
    sub_pts = points[used_verts].copy()
    sub_tris = np.array([[g2l[int(t[0])], g2l[int(t[1])], g2l[int(t[2])]] for t in tris], dtype=int)
    return sub_pts, sub_tris, g2l, l2g


def compute_interior_vertex_sets(triangles: np.ndarray, parts: List[int]) -> List[Set[int]]:
    """Compute per-part interior vertex sets (exclude cut-border vertices)."""
    ntri = len(triangles)
    part_tris: Dict[int, List[int]] = {}
    for ti in range(ntri):
        p = int(parts[ti])
        part_tris.setdefault(p, []).append(ti)
    # Vertex -> parts touching
    vert_parts: Dict[int, Set[int]] = {}
    for ti in range(ntri):
        p = int(parts[ti])
        for v in triangles[ti]:
            vv = int(v)
            if vv < 0:
                continue
            vert_parts.setdefault(vv, set()).add(p)
    # Interior vertices: those used by only one part
    interior_vs_by_part: Dict[int, Set[int]] = {p: set() for p in part_tris.keys()}
    for v, ps in vert_parts.items():
        if len(ps) == 1:
            p = next(iter(ps))
            interior_vs_by_part[p].add(v)
    return [interior_vs_by_part.get(i, set()) for i in range(max(parts)+1)]


def worker_apply_smoothing(run_id: int, points: np.ndarray, triangles: np.ndarray, tri_indices: List[int], logger_name: str, greedy_cfg: GreedyConfig = None) -> Tuple[int, Dict[int, np.ndarray]]:
    """Run a local greedy pass on a submesh and return updated coords for original vertices.

    We ignore topology changes on commit; only return coordinates for pre-existing vertices.
    """
    sub_pts, sub_tris, g2l, l2g = submesh_from_triangles(points, triangles, tri_indices)
    editor = PatchBasedMeshEditor(sub_pts.copy(), sub_tris.copy())
    lg = get_logger(f'{logger_name}.run{run_id}')
    lg.info('start submesh: npts=%d ntri=%d', len(editor.points), len(editor.triangles))
    # Lightweight greedy pass (prefer unified config to avoid deprecation)
    g_cfg = greedy_cfg if greedy_cfg is not None else GreedyConfig(max_vertex_passes=1, max_edge_passes=1, verbose=False)
    greedy_remesh(editor, config=g_cfg)
    # Collect updated coordinates for original vertices
    updated: Dict[int, np.ndarray] = {}
    for g, l in g2l.items():
        if l < len(editor.points):
            updated[g] = np.array(editor.points[l], dtype=float)
    return run_id, updated


def worker_remesh_topology(
    run_id: int,
    points: np.ndarray,
    triangles: np.ndarray,
    tri_indices: List[int],
    logger_name: str,
    max_iterations: int = 20,
    patch_cfg: PatchDriverConfig = None,
) -> Tuple[int, Dict[str, Any]]:
    """Run a small patch-driver remesh on an interior-only submesh and return a merge payload.

    Returns (run_id, payload) where payload contains:
      - 'remove_tris': List[int]   # global tri indices to remove from master
      - 'orig_vert_updates': Dict[int, np.ndarray]  # global vid -> coords
      - 'new_vertices': np.ndarray  # (M,2)
      - 'new_tris_local': np.ndarray  # (K,3) in local indexing (includes refs to new vertices)
      - 'l2g': Dict[int, int]  # local original vertex id -> global id
    """
    sub_pts, sub_tris, g2l, l2g = submesh_from_triangles(points, triangles, tri_indices)
    editor = PatchBasedMeshEditor(sub_pts.copy(), sub_tris.copy())
    lg = get_logger(f'{logger_name}.run{run_id}')
    lg.info('start topo submesh: npts=%d ntri=%d', len(editor.points), len(editor.triangles))
    # Configure a conservative patch driver, allow override via provided cfg
    cfg = patch_cfg if patch_cfg is not None else PatchDriverConfig(
        threshold=20.0,
        max_iterations=max_iterations,
        patch_radius=1,
        top_k=40,
        disjoint_on='tri',
        allow_overlap=False,
        batch_attempts=2,
        min_triangle_area=1e-12,
        reject_new_loops=True,
        reject_crossings=True,
        auto_fill_pockets=False,
        angle_unit='deg',
        plot_every=10**9,
        use_greedy_remesh=False,
        gif_capture=False,
    )
    remesh_cfg = RemeshConfig.from_patch_config(cfg)
    run_patch_batch_driver(editor, cfg, rng=random.Random(123 + run_id), np_rng=np.random.RandomState(123 + run_id), logger=lg, greedy_remesh=None, plot_mesh=None, remesh_config=remesh_cfg)
    # Identify original vs new local vertices
    local_orig_ids = set(l2g.keys())
    n_local = len(editor.points)
    # Build updates for original vertices
    orig_updates: Dict[int, np.ndarray] = {l2g[l]: np.array(editor.points[l], dtype=float) for l in local_orig_ids if l < n_local}
    # New vertices are those local ids not in l2g
    new_local_ids = [i for i in range(n_local) if i not in local_orig_ids]
    new_vertices = np.array([editor.points[i] for i in new_local_ids], dtype=float) if new_local_ids else np.empty((0,2))
    # Triangles in local indexing
    new_tris_local = np.array(editor.triangles, dtype=int)
    payload: Dict[str, Any] = {
        'remove_tris': list(tri_indices),
        'orig_vert_updates': orig_updates,
        'new_vertices': new_vertices,
        'new_tris_local': new_tris_local,
        'l2g': l2g,
        'new_local_ids': new_local_ids,
    }
    return run_id, payload


def main():
    ap = argparse.ArgumentParser(description='Partition mesh and run per-part smoothing in parallel on shared mesh')
    ap.add_argument('--npts', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--parts', type=int, default=4)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--allow-topology', action='store_true', help='Allow per-part topology changes (patch driver) and merge back interior region')
    ap.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    ap.add_argument('--config-json', type=str, default=None, help='Path to JSON config (type=greedy/patch) like remesh_driver')
    ap.add_argument('--out-before', type=str, default='partition_before.png')
    ap.add_argument('--out-after', type=str, default='partition_after.png')
    args = ap.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    pts, tris = build_random_delaunay(npts=args.npts, seed=args.seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # Optional: load config JSON for greedy or patch like remesh_driver
    greedy_cfg_from_json: GreedyConfig = None
    patch_cfg_from_json: PatchDriverConfig = None
    if args.config_json:
        try:
            with open(args.config_json, 'r') as f:
                payload = json.load(f)
            cfg_dict = payload.get('config', payload)
            payload_type = payload.get('type', None)
            if payload_type == 'greedy' or ('max_vertex_passes' in cfg_dict and 'max_edge_passes' in cfg_dict):
                allowed_g = set(GreedyConfig().__dict__.keys())
                filtered = {k: v for k, v in cfg_dict.items() if k in allowed_g}
                greedy_cfg_from_json = GreedyConfig(**filtered)
                log.info('Loaded GreedyConfig from %s', args.config_json)
            if payload_type == 'patch' or any(k in cfg_dict for k in PatchDriverConfig().__dict__.keys()):
                allowed_p = set(PatchDriverConfig().__dict__.keys())
                filtered_p = {k: v for k, v in cfg_dict.items() if k in allowed_p}
                # Ensure conservative defaults for missing fields
                patch_cfg_from_json = PatchDriverConfig(**filtered_p)
                log.info('Loaded PatchDriverConfig from %s', args.config_json)
        except Exception as e:
            log.error('Failed to load config JSON (%s), proceeding with built-in defaults', e)

    # Partition triangles
    adj = tri_adjacency(editor.triangles)
    try:
        parts = try_metis_partition(adj, args.parts)
    except Exception as e:
        log.error('PyMetis is required for this demo. Install with: pip install pymetis (%s)', e)
        return
    # Build per-part triangle lists
    part_to_tris: Dict[int, List[int]] = {}
    for ti, p in enumerate(parts):
        part_to_tris.setdefault(int(p), []).append(int(ti))
    # Compute interior vertex sets; keep only triangles whose all vertices are interior to their part
    interior_vs_by_part = compute_interior_vertex_sets(editor.triangles, parts)
    work_tris_by_part: Dict[int, List[int]] = {}
    for p, tri_list in part_to_tris.items():
        interior_vs = interior_vs_by_part[p]
        safe_tris = []
        for ti in tri_list:
            v0, v1, v2 = [int(v) for v in editor.triangles[int(ti)]]
            if v0 in interior_vs and v1 in interior_vs and v2 in interior_vs:
                safe_tris.append(int(ti))
        if safe_tris:
            work_tris_by_part[p] = safe_tris
    log.info('parts=%d usable=%d (triangles per part: %s)', args.parts, len(work_tris_by_part), {p: len(ts) for p, ts in work_tris_by_part.items()})

    # Plot before with partitions colored
    part_to_tris: Dict[int, List[int]] = {}
    for ti, p in enumerate(parts):
        part_to_tris.setdefault(int(p), []).append(int(ti))
    plot_mesh_by_tri_groups(editor, part_to_tris, outname=args.out_before)

    # Run workers
    workers = args.workers or args.parts
    updates: Dict[int, np.ndarray] = {}
    if not args.allow_topology:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for p, tri_list in work_tris_by_part.items():
                futs.append(ex.submit(worker_apply_smoothing, p, editor.points, editor.triangles, tri_list, 'sofia.demos.partition', greedy_cfg_from_json))
            for fut in as_completed(futs):
                rid, upd = fut.result()
                updates.update(upd)
        # Commit updates back to master editor (vertex-disjoint by construction)
        for gvid, new_xy in updates.items():
            editor.points[int(gvid)] = new_xy
        editor.compact_triangle_indices()
        # After: color all partitions so the domain is fully covered
        plot_mesh_by_tri_groups(editor, part_to_tris, outname=args.out_after)
        log.info('Wrote %s and %s', args.out_before, args.out_after)
        return

    # Topology-changing per-part remesh and merge
    merge_payloads: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for p, tri_list in work_tris_by_part.items():
            futs.append(ex.submit(worker_remesh_topology, p, editor.points, editor.triangles, tri_list, 'sofia.demos.partition', 20, patch_cfg_from_json))
        for fut in as_completed(futs):
            rid, payload = fut.result()
            merge_payloads[rid] = payload
    # Remove interior triangles and collect new vertices incrementally
    to_remove = sorted({ti for payload in merge_payloads.values() for ti in payload['remove_tris']}, reverse=True)
    tris_list = editor.triangles.tolist()
    for ti in to_remove:
        if 0 <= ti < len(tris_list):
            tris_list[ti] = [-1, -1, -1]
    # Assign global ids to new vertices sequentially in master
    for rid in sorted(merge_payloads.keys()):
        payload = merge_payloads[rid]
        # Update original vertices
        for gvid, xy in payload['orig_vert_updates'].items():
            editor.points[int(gvid)] = xy
        # Assign new global ids
        base_gid = len(editor.points)
        new_vs = payload['new_vertices']
        n_new = int(new_vs.shape[0]) if new_vs is not None else 0
        if n_new > 0:
            editor.points = np.vstack([editor.points, new_vs])
        # Reindex triangles and append
        l2g = payload['l2g']
        new_local_ids = payload['new_local_ids']
        new_local_map = {loc: base_gid + i for i, loc in enumerate(new_local_ids)}
        for t in payload['new_tris_local']:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            ga = l2g.get(a, new_local_map.get(a))
            gb = l2g.get(b, new_local_map.get(b))
            gc = l2g.get(c, new_local_map.get(c))
            if ga is None or gb is None or gc is None:
                continue
            tris_list.append([int(ga), int(gb), int(gc)])
    editor.triangles = np.array(tris_list, dtype=int)
    editor.compact_triangle_indices()
    plot_mesh(editor, outname=args.out_after)

    log.info('Wrote %s and %s', args.out_before, args.out_after)


if __name__ == '__main__':
    main()
