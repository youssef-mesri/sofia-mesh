import sys, csv, ast
import numpy as np
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


def triangle_area(p0,p1,p2):
    return 0.5 * np.cross(p1 - p0, p2 - p0)


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


def save_inverted(points, tris, inverted_idxs, outname):
    plt.figure(figsize=(6,6))
    plt.triplot(points[:,0], points[:,1], tris, lw=0.6, color='gray')
    plt.scatter(points[:,0], points[:,1], s=6)
    for ti in inverted_idxs:
        if ti < 0 or ti >= len(tris):
            continue
        coords = points[np.array(tris[ti], dtype=int)]
        plt.fill(coords[:,0], coords[:,1], facecolor='red', alpha=0.6)
    plt.gca().set_aspect('equal')
    plt.savefig(outname, dpi=150)
    plt.close()
    return outname


def main():
    if len(sys.argv) < 2:
        print('Usage: inspect_inverted.py path/to/log.csv --npts N --seed S --iter I')
        return
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
            if not (res.startswith('ok') or res == 'ok' or res.startswith('applied-')):
                continue
            op = row.get('op_attempted') or row.get('op')
            op_param_s = row.get('op_param') or row.get('op_param')
            param = parse_param(op_param_s)
            try:
                if op == 'flip':
                    edge = tuple(sorted(tuple(param)))
                    local_tris = list(editor.edge_map.get(edge, []))
                    ok, msg, _ = editor.flip_edge(tuple(param)) if hasattr(editor, 'flip_edge') else (False, 'no flip')
                elif op == 'split':
                    ok, msg, _ = editor.split_edge(tuple(param))
                elif op == 'remove':
                    ok, msg, _ = editor.remove_node_with_patch(int(param))
                elif op == 'add':
                    tri_idx = int(param[0]); pt = np.array(param[1], dtype=float)
                    ok, msg, _ = editor.add_node(pt, tri_idx=tri_idx)
                else:
                    ok, msg = True, 'skip'
            except Exception as e:
                print('Error applying', op, param, e)
                ok, msg = False, str(e)

            # after applying, compact and inspect for inverted triangles
            pts_c, tris_c, mapping, active_idx = compact_copy(editor)
            inverted = []
            for ti, t in enumerate(tris_c):
                p0 = pts_c[int(t[0])]; p1 = pts_c[int(t[1])]; p2 = pts_c[int(t[2])]
                a = triangle_area(p0,p1,p2)
                if a <= 0.0:
                    inverted.append((ti, float(a)))
            if inverted:
                print('Found inverted triangles after operation row:')
                print(row)
                print('Inverted triangles (local_index, signed_area):')
                for itx, av in inverted:
                    print(itx, av)
                out = save_inverted(pts_c, tris_c, [itx for itx,_ in inverted], '/home/ymesri/Sofia/inverted_after.png')
                print('Saved inverted highlight image:', out)
                return
    print('No inverted triangles found up to iter', target_iter)

if __name__ == '__main__':
    main()
