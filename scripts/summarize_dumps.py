#!/usr/bin/env python3
import os
import glob
import numpy as np
from sofia.sofia.logging_utils import get_logger

logger = get_logger('sofia.scripts.summarize_dumps')
from collections import Counter, defaultdict

root = os.path.abspath(os.path.dirname(__file__) + '/..')
# search patterns
pocket_patterns = [os.path.join(root, 'pocket_dump_*.npz')]
reject_patterns = [os.path.join(root, 'reject_min_angle_dump_*.npz'), os.path.join(root, 'reject_min_angle_dump_add_op_*.npz')]

pocket_files = []
for p in pocket_patterns:
    pocket_files.extend(glob.glob(p))
reject_files = []
for p in reject_patterns:
    reject_files.extend(glob.glob(p))

summary = {
    'n_pocket_files': len(pocket_files),
    'n_reject_files': len(reject_files),
}

# analyze pocket dumps
pocket_details = []
empty_pocket_count = 0
pocket_vertices_counter = Counter()
for fn in pocket_files:
    try:
        d = dict(np.load(fn, allow_pickle=True))
    except Exception as e:
        pocket_details.append((fn, 'load-error', str(e)))
        continue
    # detect expected keys
    # detect saved 'pockets' or 'verts' depending on driver dump
    # We try to interpret keys
    if 'pockets' in d:
        pockets = d['pockets']
    else:
        # detect keys that may contain 'verts' or 'coords'
        # driver saved pocket arrays via detect_and_dump_pockets which likely saved pockets as arrays
        # fallback: look for arrays with 'verts' substring in filename or data
        pockets = None
        # sometimes pocket dump saves arrays named 'verts' etc
        # try to reconstruct from arrays present
        for k in d:
            if 'verts' in k:
                pockets = d[k]
                break
    # fallback: if file contains arrays for 'coords' and 'inside_pts', assume single pocket
    if pockets is None:
        # attempt to read 'coords' and 'inside_pts'
        coords = d.get('coords', None)
        inside_pts = d.get('inside_pts', None)
        verts = d.get('verts', None)
        if verts is not None:
            verts_list = list(verts)
            pocket_details.append((fn, 'single', {'verts': verts_list, 'inside_pts': len(inside_pts) if inside_pts is not None else None}))
            pocket_vertices_counter.update([tuple(verts_list)])
            if inside_pts is None or (hasattr(inside_pts, '__len__') and len(inside_pts) == 0):
                empty_pocket_count += 1
            continue
        else:
            pocket_details.append((fn, 'unknown-contents', list(d.keys())))
            continue
    # if pockets is an array-like saved structure
    try:
        # attempt to treat pockets as sequence of dict-like objects
        pseq = list(pockets)
        for p in pseq:
            try:
                # if p is a numpy void or object array with fields
                if isinstance(p, np.ndarray) and p.dtype.names:
                    verts = p['verts'].tolist() if 'verts' in p.dtype.names else None
                elif isinstance(p, (tuple, list, np.ndarray)):
                    # may be array of arrays
                    verts = list(p)
                else:
                    verts = None
                inside_pts = None
                # count emptiness using inside_pts key if available in file-level
                if 'inside_pts' in d:
                    inside_pts = d['inside_pts']
                pocket_details.append((fn, 'pocket', {'verts': verts, 'inside_pts_len': len(inside_pts) if inside_pts is not None else None}))
                if verts is not None:
                    pocket_vertices_counter.update([tuple(verts)])
                if inside_pts is None or (hasattr(inside_pts, '__len__') and len(inside_pts) == 0):
                    empty_pocket_count += 1
            except Exception:
                continue
    except Exception:
        pocket_details.append((fn, 'pocket-parse-failed', list(d.keys())))

summary['empty_pocket_files'] = empty_pocket_count
summary['unique_pockets_seen'] = len(pocket_vertices_counter)
summary['most_common_pockets'] = pocket_vertices_counter.most_common(10)

# analyze reject dumps: just count and try to read pre/post loops if present
reject_details = []
for fn in reject_files:
    try:
        d = dict(np.load(fn, allow_pickle=True))
    except Exception as e:
        reject_details.append((fn, 'load-error', str(e)))
        continue
    pre_pts = d.get('pre_pts', None)
    pre_tris = d.get('pre_tris', None)
    post_pts = d.get('post_pts', None)
    post_tris = d.get('post_tris', None)
    # compute loop counts if possible
    def count_loops(pts, tris):
        if pts is None or tris is None:
            return None
        try:
            # simple boundary edge count via edge_map
            edges = {}
            for t in tris:
                t = list(t)
                if any(x < 0 for x in t):
                    continue
                for i in range(3):
                    a = int(t[i]); b = int(t[(i+1)%3])
                    key = tuple(sorted((a,b)))
                    edges[key] = edges.get(key, 0) + 1
            boundary = [e for e,c in edges.items() if c == 1]
            return len(boundary)
        except Exception:
            return None
    pre_b = count_loops(pre_pts, pre_tris)
    post_b = count_loops(post_pts, post_tris)
    reject_details.append((fn, {'pre_boundary_edges': pre_b, 'post_boundary_edges': post_b}))

summary['reject_files_details'] = reject_details

# Print concise summary
logger.info('SUMMARY')
logger.info('Root: %s', root)
logger.info('Pocket dump files: %d', summary['n_pocket_files'])
logger.info('Reject-min-angle dump files: %d', summary['n_reject_files'])
logger.info('Empty pocket files (inside_pts empty or not present): %d', summary['empty_pocket_files'])
logger.info('Unique pocket vertex-tuples seen: %d', summary['unique_pockets_seen'])
logger.info('Most common pockets (up to 10):')
for k,v in summary['most_common_pockets']:
    logger.info('  %s count=%d', k, v)
logger.info('\nReject dump samples (file -> pre/post boundary-edge counts):')
for fn, info in reject_details[:10]:
    logger.info('  %s -> %s', os.path.basename(fn), info)

# Exit code 0
