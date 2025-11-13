#!/usr/bin/env python3
"""
Demo: Coarsening scenario using only remove_node_with_patch2 operations.

This script focuses on evaluating the performance of op_remove_node_with_patch2
by iteratively removing low-degree vertices from the mesh.

Example JSON (see configs/coarsening_scenario2_h2.json):
{
  "mesh": { "type": "random_delaunay", "npts": 120, "seed": 7 },
  "auto": {
    "iterations": 5,
    "remove_degree_max": 6,
    "remove_max_per_iter": 30,
    "allow_boundary_remove": false,
    "barycenter_passes": 1
  },
  "plot": { "out_before": "coarsen2_before.png", "out_after": "coarsen2_after.png" }
}
"""
from __future__ import annotations

import argparse
import cProfile as _cprof
import io as _io
import json
import logging
import pstats as _pstats
from typing import Any, Dict

import numpy as np

from sofia.core.logging_utils import configure_logging, get_logger
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.visualization import plot_mesh
from sofia.core.operations import op_remove_node_with_patch2

log = get_logger('sofia.demo.coarsen2')


def load_mesh_from_cfg(mesh_cfg: Dict[str, Any]) -> PatchBasedMeshEditor:
    mtype = (mesh_cfg or {}).get('type', 'random_delaunay')
    if mtype == 'random_delaunay':
        npts = int(mesh_cfg.get('npts', 120))
        seed = int(mesh_cfg.get('seed', 7))
        pts, tris = build_random_delaunay(npts=npts, seed=seed)
        return PatchBasedMeshEditor(pts.copy(), tris.copy())
    raise ValueError(f"Unsupported mesh type: {mtype}")


def is_boundary_vertex(editor: PatchBasedMeshEditor, v: int) -> bool:
    """Check if vertex v is on the boundary of the mesh."""
    for ti in editor.v_map.get(int(v), []):
        t = editor.triangles[int(ti)]
        for i in range(3):
            u, w = int(t[i]), int(t[(i+1)%3])
            key = tuple(sorted((u, w)))
            if v in (u, w):
                # Optimized: count non-tombstoned triangles without list comprehension
                incident_tris = editor.edge_map.get(key, [])
                count = sum(1 for tt in incident_tris if editor.triangles[int(tt)][0] != -1)
                if count == 1:
                    return True
    return False


def degree(editor: PatchBasedMeshEditor, v: int) -> int:
    """Get the degree (number of unique neighbors) of vertex v."""
    neighbors = set()
    for ti in editor.v_map.get(int(v), []):
        t = editor.triangles[int(ti)]
        for x in t:
            if int(x) != int(v) and int(x) >= 0:
                neighbors.add(int(x))
    return len(neighbors)


def auto_coarsen_remove_only(editor: PatchBasedMeshEditor, auto_cfg: Dict[str, Any]):
    """Coarsen the mesh by iteratively removing low-degree vertices using op_remove_node_with_patch2.
    
    This focuses purely on the remove_node operation for performance evaluation.
    """
    iters = int(auto_cfg.get('iterations', 5))
    remove_degree_max = int(auto_cfg.get('remove_degree_max', 6))
    remove_max = int(auto_cfg.get('remove_max_per_iter', 30))
    allow_boundary_remove = bool(auto_cfg.get('allow_boundary_remove', False))
    barycenter_passes = int(auto_cfg.get('barycenter_passes', 1))
    
    log.info('Starting coarsening with remove_node_with_patch2 only')
    log.info('iterations=%d, remove_degree_max=%d, remove_max_per_iter=%d, allow_boundary=%s',
             iters, remove_degree_max, remove_max, allow_boundary_remove)
    
    initial_verts = len([v for v in editor.v_map.keys()])
    initial_tris = len([t for t in editor.triangles if t[0] != -1])
    
    log.info('Initial mesh: vertices=%d, triangles=%d', initial_verts, initial_tris)
    
    total_removed = 0
    total_attempts = 0
    
    for it in range(iters):
        # Get all vertices and sort by degree (lowest first)
        verts = list(editor.v_map.keys())
        scored = []
        
        for v in verts:
            v = int(v)
            # Skip boundary vertices if not allowed
            if not allow_boundary_remove and is_boundary_vertex(editor, v):
                continue
            
            deg = degree(editor, v)
            if deg <= remove_degree_max:
                scored.append((v, deg))
        
        # Sort by degree (ascending), then by vertex index for consistency
        scored.sort(key=lambda x: (x[1], x[0]))
        
        removed = 0
        attempts = 0
        
        for v, degv in scored:
            attempts += 1
            total_attempts += 1
            ok, msg, _ = op_remove_node_with_patch2(editor, int(v))
            
            if ok:
                removed += 1
                total_removed += 1
                log.debug('iter=%d: removed vertex v=%d (degree=%d)', it, v, degv)
                
                if removed >= remove_max:
                    break
            else:
                log.debug('iter=%d: failed to remove vertex v=%d (degree=%d): %s', it, v, degv, msg)
        
        # Optional smoothing after removals
        moves = 0
        if barycenter_passes > 0:
            for _ in range(barycenter_passes):
                moves += editor.move_vertices_to_barycenter()
        
        current_verts = len([v for v in editor.v_map.keys()])
        current_tris = len([t for t in editor.triangles if t[0] != -1])
        
        log.info('iter=%d: removed=%d/%d attempts, total_removed=%d, vertices=%d, triangles=%d, smooth_moves=%d',
                 it, removed, attempts, total_removed, current_verts, current_tris, moves)
        
        # Stop if no vertices were removed in this iteration
        if removed == 0:
            log.info('No vertices removed in iteration %d, stopping', it)
            break
    
    final_verts = len([v for v in editor.v_map.keys()])
    final_tris = len([t for t in editor.triangles if t[0] != -1])
    
    log.info('Coarsening complete: %d/%d attempts successful', total_removed, total_attempts)
    log.info('Final mesh: vertices=%d (-%d), triangles=%d (-%d)',
             final_verts, initial_verts - final_verts,
             final_tris, initial_tris - final_tris)


def _run(args):
    """Main logic extracted for profiling."""
    with open(args.scenario, 'r') as f:
        scenario = json.load(f)

    mesh_cfg = scenario.get('mesh', {})
    plot_cfg = scenario.get('plot', {})
    out_before = plot_cfg.get('out_before', 'coarsen2_before.png')
    out_after = plot_cfg.get('out_after', 'coarsen2_after.png')

    editor = load_mesh_from_cfg(mesh_cfg)
    plot_mesh(editor, outname=out_before)
    log.info('Wrote %s', out_before)

    auto_cfg = scenario.get('auto', None)
    if auto_cfg:
        # Handle virtual boundary mode if specified
        if 'virtual_boundary_mode' in auto_cfg:
            try:
                editor.virtual_boundary_mode = bool(auto_cfg.get('virtual_boundary_mode', False))
            except Exception:
                pass
        auto_coarsen_remove_only(editor, auto_cfg)
    else:
        log.info('No auto block provided; nothing to do')

    editor.compact_triangle_indices()
    plot_mesh(editor, outname=out_after)
    log.info('Wrote %s', out_after)


def main():
    ap = argparse.ArgumentParser(description='Coarsening scenario using only op_remove_node_with_patch2')
    ap.add_argument('--scenario', type=str, required=True, help='Path to scenario JSON file')
    ap.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    ap.add_argument('--profile', action='store_true', help='Enable cProfile and print top hotspots')
    ap.add_argument('--profile-out', type=str, default=None, help='Write raw cProfile stats to this .pstats file')
    ap.add_argument('--profile-top', type=int, default=25, help='How many entries to show in hotspots (default: 25)')
    args = ap.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.profile:
        pr = _cprof.Profile()
        pr.enable()
        _run(args)
        pr.disable()
        
        if args.profile_out:
            pr.dump_stats(args.profile_out)
            log.info('Wrote profiling stats to %s', args.profile_out)
        
        s = _io.StringIO()
        ps = _pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(args.profile_top)
        log.info('Profile (top %d by cumulative time):\n%s', args.profile_top, s.getvalue())
    else:
        _run(args)


if __name__ == '__main__': 
    main()
