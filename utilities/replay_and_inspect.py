"""
Replay applied operations from a per-patch CSV up to a target iteration, compact the mesh,
check for crossing (non-manifold/intersecting) edges, and save an inspection PNG.

Usage: python replay_and_inspect.py /path/to/patch_log.csv --npts 40 --seed 7 --iter 1 --out png
"""
import sys, csv, ast
import numpy as np
from sofia.core.geometry import EPS_AREA
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor


def parse_param(s):
    if not s:
        return None
    # convert 'array(' to 'np.array(' for eval
    s2 = s.replace('array(', 'np.array(')
    try:
        val = eval(s2, {'np': np})
        return val
    except Exception:
        # fallback: try literal_eval after removing np.array
        try:
            s3 = s2.replace('np.array', 'array')
            return ast.literal_eval(s3)
        except Exception:
            return None


def segs_from_tris(points, tris):
    edges = set()
    for t in tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        edges.add(tuple(sorted((a,b))))
        edges.add(tuple(sorted((b,c))))
        edges.add(tuple(sorted((c,a))))
    segs = []
    for e in edges:
        p = points[e[0]]
        q = points[e[1]]
        segs.append((e, (p, q)))
    return segs


def orient(a,b,c):
    return np.cross(b-a, c-a)


def on_seg(a,b,c):
    # check c on segment ab
    return min(a[0], b[0]) - EPS_AREA <= c[0] <= max(a[0], b[0]) + EPS_AREA and min(a[1], b[1]) -EPS_AREA <= c[1] <= max(a[1], b[1]) +EPS_AREA


def seg_intersect(p1,q1,p2,q2):
    # exclude touching at endpoints considered non-crossing
    # general segment intersection test
    d1 = orient(p1, q1, p2)
    d2 = orient(p1, q1, q2)
    d3 = orient(p2, q2, p1)
    d4 = orient(p2, q2, q1)
    if abs(d1) < EPS_AREA and on_seg(p1,q1,p2):
        return False
    if abs(d2) < EPS_AREA and on_seg(p1,q1,q2):
        return False
    if abs(d3) < EPS_AREA and on_seg(p2,q2,p1):
        return False
    if abs(d4) < EPS_AREA and on_seg(p2,q2,q1):
        return False
    return (d1*d2 < 0) and (d3*d4 < 0)


def compact_copy(editor):
    tris = np.array(editor.triangles)
    pts = np.array(editor.points)
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    used_verts = sorted(set(active_tris.flatten().tolist()))
    used_verts = [v for v in used_verts if v >= 0]
    mapping = {old: new for new, old in enumerate(used_verts)}
    new_points = pts[used_verts]
    new_tris = []
    for t in active_tris:
        if np.any(t < 0):
            continue
        try:
            new_tris.append([mapping[int(t[0])], mapping[int(t[1])], mapping[int(t[2])]])
        except KeyError:
            continue
    new_tris = np.array(new_tris, dtype=int)
    return new_points, new_tris


def main():
    if len(sys.argv) < 2:
        print('Usage: python replay_and_inspect.py path/to/log.csv [--npts N --seed S --iter I --out file.png]')
        sys.exit(1)
    path = sys.argv[1]
    # defaults
    npts = 40
    seed = 7
    target_iter = 1
    out = 'inspect.png'
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--npts':
            npts = int(args[i+1]); i+=2
        elif a == '--seed':
            seed = int(args[i+1]); i+=2
        elif a == '--iter':
            target_iter = int(args[i+1]); i+=2
        elif a == '--out':
            out = args[i+1]; i+=2
        else:
            i+=1

    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # read CSV and apply rows with iter <= target_iter and result == ok
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            it = int(row.get('iter', 0))
            if it > target_iter:
                break
            res = row.get('result','')
            if not res.startswith('ok') and res != 'ok':
                continue
            op = row.get('op_attempted') or row.get('op')
            param_s = row.get('op_param') or row.get('op_param')
            param = parse_param(param_s)
            try:
                if op == 'flip':
                    editor.flip_edge(tuple(param))
                elif op == 'split':
                    editor.split_edge(tuple(param))
                elif op == 'remove':
                    # param likely int
                    editor.remove_node_with_patch(int(param))
                elif op == 'add':
                    # param is (tri_idx, np.array([x,y]))
                    tri_idx = int(param[0])
                    pt = np.array(param[1], dtype=float)
                    editor.add_node(pt, tri_idx=tri_idx)
            except Exception as e:
                print('Error applying', op, param, e)

    new_points, new_tris = compact_copy(editor)
    segs = segs_from_tris(new_points, new_tris)

    # detect intersections
    crossings = []
    for i in range(len(segs)):
        ei, (p1,q1) = segs[i]
        for j in range(i+1, len(segs)):
            ej, (p2,q2) = segs[j]
            # ignore if share vertex
            if set(ei) & set(ej):
                continue
            if seg_intersect(np.array(p1), np.array(q1), np.array(p2), np.array(q2)):
                crossings.append((ei, ej))

    print('Found crossings:', len(crossings))
    # plot and highlight crossings
    plt.figure(figsize=(6,6))
    plt.triplot(new_points[:,0], new_points[:,1], new_tris, lw=0.6, color='gray')
    plt.scatter(new_points[:,0], new_points[:,1], s=6)
    for ei, (p,q) in segs:
        p = np.array(p); q = np.array(q)
        plt.plot([p[0], q[0]], [p[1], q[1]], color='black', lw=0.5)
    for (e1,e2) in crossings:
        p1 = new_points[e1[0]]; q1 = new_points[e1[1]]
        p2 = new_points[e2[0]]; q2 = new_points[e2[1]]
        plt.plot([p1[0], q1[0]], [p1[1], q1[1]], color='red', lw=2)
        plt.plot([p2[0], q2[0]], [p2[1], q2[1]], color='red', lw=2)
    plt.gca().set_aspect('equal')
    plt.title(f'inspect iter {target_iter} crossings {len(crossings)}')
    plt.savefig(out, dpi=150)
    plt.close()
    print('Wrote', out)

if __name__ == '__main__':
    main()
