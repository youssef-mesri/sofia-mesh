"""
Replay operations sequentially and stop at the first operation that introduces an edge crossing.
Saves before/after zoomed PNGs and prints the offending CSV row.

Usage:
  python replay_until_first_crossing.py path/to/patch_log.csv --npts 40 --seed 7 --iter 1
"""
import sys, csv, ast
import numpy as np
from geometry import EPS_AREA
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor


def parse_param(s):
    if not s:
        return None
    s2 = s.replace('array(', 'np.array(')
    try:
        val = eval(s2, {'np': np, 'array': lambda *a: None})
        return val
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def orient(a,b,c):
    return np.cross(b-a, c-a)


def on_seg(a,b,c):
    return min(a[0], b[0]) - EPS_AREA <= c[0] <= max(a[0], b[0]) + EPS_AREA and min(a[1], b[1]) -EPS_AREA <= c[1] <= max(a[1], b[1]) +EPS_AREA


def seg_intersect(p1,q1,p2,q2):
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
    active_idx = np.nonzero(active_mask)[0].tolist()
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
    return new_points, new_tris, mapping, active_idx


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


def save_colored(points, tris, highlight_tris=None, outname='mesh_colored.png'):
    plt.figure(figsize=(6,6))
    plt.triplot(points[:,0], points[:,1], tris, lw=0.6, color='gray')
    plt.scatter(points[:,0], points[:,1], s=6)
    if highlight_tris:
        for t_idx in highlight_tris:
            if t_idx < 0 or t_idx >= len(tris):
                continue
            tri = tris[t_idx]
            coords = points[np.array(tri, dtype=int)]
            plt.fill(coords[:,0], coords[:,1], facecolor='orange', alpha=0.6)
    plt.gca().set_aspect('equal')
    plt.savefig(outname, dpi=150)
    plt.close()
    return outname


def find_crossings(points, tris):
    segs = segs_from_tris(points, tris)
    crossings = []
    for i in range(len(segs)):
        ei, (p1,q1) = segs[i]
        for j in range(i+1, len(segs)):
            ej, (p2,q2) = segs[j]
            if set(ei) & set(ej):
                continue
            if seg_intersect(np.array(p1), np.array(q1), np.array(p2), np.array(q2)):
                crossings.append((ei, ej, p1, q1, p2, q2))
    return crossings


def save_zoom(points, tris, crossings, out_prefix):
    # If no crossings, save whole mesh
    if not crossings:
        plt.figure(figsize=(6,6))
        plt.triplot(points[:,0], points[:,1], tris, lw=0.6)
        plt.scatter(points[:,0], points[:,1], s=6)
        plt.gca().set_aspect('equal')
        plt.savefig(out_prefix + '_full.png', dpi=150)
        plt.close()
        return out_prefix + '_full.png'
    # focus on first crossing midpoint
    _, _, p1, q1, p2, q2 = crossings[0]
    p1 = np.array(p1); q1 = np.array(q1); p2 = np.array(p2); q2 = np.array(q2)
    mid = (p1 + q1 + p2 + q2) / 4.0
    xs = np.concatenate([points[:,0], [mid[0]]])
    ys = np.concatenate([points[:,1], [mid[1]]])
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    # compute local box
    box_size = max(0.05, 0.15 * span)
    xmin = mid[0] - box_size; xmax = mid[0] + box_size
    ymin = mid[1] - box_size; ymax = mid[1] + box_size
    plt.figure(figsize=(6,6))
    plt.triplot(points[:,0], points[:,1], tris, lw=0.6, color='gray')
    plt.scatter(points[:,0], points[:,1], s=6)
    for (e,pq) in segs_from_tris(points, tris):
        p,q = pq
        plt.plot([p[0], q[0]], [p[1], q[1]], color='black', lw=0.5)
    # highlight crossings
    for (ei, ej, p1, q1, p2, q2) in crossings:
        p1 = np.array(p1); q1 = np.array(q1); p2 = np.array(p2); q2 = np.array(q2)
        plt.plot([p1[0], q1[0]], [p1[1], q1[1]], color='red', lw=2)
        plt.plot([p2[0], q2[0]], [p2[1], q2[1]], color='red', lw=2)
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal')
    out = out_prefix + '_zoom.png'
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    if len(sys.argv) < 2:
        print('Usage: python replay_until_first_crossing.py path/to/log.csv --npts N --seed S --iter I')
        sys.exit(1)
    path = sys.argv[1]
    npts = 40; seed = 7; target_iter = 1
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--npts': npts = int(args[i+1]); i+=2
        elif a == '--seed': seed = int(args[i+1]); i+=2
        elif a == '--iter': target_iter = int(args[i+1]); i+=2
        else: i+=1

    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            it = int(row.get('iter', 0))
            if it > target_iter:
                break
            res = row.get('result','')
            # only consider applied ops
            if not (res.startswith('ok') or res == 'ok' or res.startswith('applied-')):
                continue
            op = row.get('op_attempted') or row.get('op')
            op_param_s = row.get('op_param') or row.get('op_param')
            param = parse_param(op_param_s)

            # compute compact before
            before_pts, before_tris, before_map, before_active_idx = compact_copy(editor)
            crossings_before = find_crossings(before_pts, before_tris)
            if crossings_before:
                print('Crossings already present before operation:', len(crossings_before))
                # save image and exit
                outb = save_zoom(before_pts, before_tris, crossings_before, '/home/ymesri/Sofia/before_already')
                print('Saved before image', outb)
                print('Row:', row)
                return

            # apply operation
            try:
                if op == 'flip':
                    # capture local triangle indices around the edge before flip
                    edge = tuple(sorted(tuple(param)))
                    local_tris = list(editor.edge_map.get(edge, []))
                    editor.flip_edge(tuple(param))
                elif op == 'split':
                    editor.split_edge(tuple(param))
                elif op == 'remove':
                    editor.remove_node_with_patch(int(param))
                elif op == 'add':
                    tri_idx = int(param[0]); pt = np.array(param[1], dtype=float)
                    editor.add_node(pt, tri_idx=tri_idx)
            except Exception as e:
                print('Error applying', op, param, e)
                continue

            # check after
            after_pts, after_tris, after_map, after_active_idx = compact_copy(editor)
            crossings = find_crossings(after_pts, after_tris)
            if crossings:
                print('First crossing introduced by row:')
                print(row)
                # save before/after images focused on crossing and color the affected triangles
                # Map editor global triangle indices to compacted local indices for before/after highlights
                before_local_idxs = []
                for gidx in local_tris:
                    if gidx in before_active_idx:
                        before_local_idxs.append(before_active_idx.index(gidx))
                # For after, identify new triangles that replaced the old ones: try to find triangles that contain opp verts
                # We'll just save the zoomed images and a colored full image of the before set
                out_before = save_colored(before_pts, before_tris, highlight_tris=before_local_idxs, outname='/home/ymesri/Sofia/first_cross_before_colored.png')
                # map post-flip global triangle indices to compacted local indices for coloring
                after_local_idxs = []
                for gidx in local_tris:
                    if gidx in after_active_idx:
                        after_local_idxs.append(after_active_idx.index(gidx))
                out_after = save_colored(after_pts, after_tris, highlight_tris=after_local_idxs, outname='/home/ymesri/Sofia/first_cross_after_colored.png')
                print('Saved before colored image:', out_before)
                print('Saved after colored image:', out_after)
                return

    print('No crossing introduced up to iter', target_iter)

if __name__ == '__main__':
    main()
