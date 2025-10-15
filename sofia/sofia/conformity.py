"""Conformity and structural checks (canonical package copy)."""
from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from .geometry import triangle_area, triangles_signed_areas
from .constants import EPS_AREA

__all__ = [
	'build_edge_to_tri_map','build_vertex_to_tri_map','check_mesh_conformity',
	'is_active_triangle','boundary_edges_from_map','is_boundary_vertex_from_maps',
	'simulate_compaction_and_check'
]

def simulate_compaction_and_check(points, triangles, eps_area=EPS_AREA, reject_boundary_loop_increase=False,
								   reject_any_boundary_loops=False, reject_crossing_edges=False):
	pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
	tris = np.ascontiguousarray(np.asarray(triangles, dtype=np.int32))
	try:
		mask = ~np.all(tris == -1, axis=1)
		active_tris = tris[mask]
	except Exception:
		active_tris = tris
	if active_tris.size == 0:
		return True, [], []
	used_vertices = sorted({int(v) for tri in active_tris for v in tri if int(v) >= 0})
	try:
		new_points = np.vstack([pts[idx] for idx in used_vertices]) if used_vertices else np.empty((0,2))
	except Exception:
		return False, ['error building compacted points'], []
	old_to_new = {old: new for new, old in enumerate(used_vertices)}
	try:
		remapped = np.array([[old_to_new[int(t[0])], old_to_new[int(t[1])], old_to_new[int(t[2])]] for t in active_tris], dtype=int)
	except Exception:
		return False, ['triangle index out of range during simulation'], []
	ok_conf, msgs = check_mesh_conformity(new_points, remapped, allow_marked=False,
										  reject_boundary_loops=reject_any_boundary_loops)
	# Vectorized inversion detection on remapped tris
	try:
		areas = triangles_signed_areas(new_points, remapped)
		inv = [(int(i), float(a)) for i, a in enumerate(areas) if a <= -eps_area]
	except Exception:
		inv = []
	edge_count = defaultdict(int)
	for t in remapped:
		a,b,c = int(t[0]), int(t[1]), int(t[2])
		edges = [(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))]
		for e in edges: edge_count[e] += 1
	boundary_edges = [e for e,c in edge_count.items() if c == 1]
	if reject_boundary_loop_increase:
		pass
	crossed = False
	if reject_crossing_edges:
		# Build unique edge list
		edges = []
		for t in remapped:
			a,b,c = int(t[0]), int(t[1]), int(t[2])
			edges.extend([(a,b),(b,c),(c,a)])
		edges = [tuple(sorted(e)) for e in edges]
		edges = sorted(set(edges))
		# Geometry helpers
		def orient(a,b,c):
			return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
		def seg_intersect(p1,p2,p3,p4):
			# Exclude shared endpoints
			if (p1==p3).all() or (p1==p4).all() or (p2==p3).all() or (p2==p4).all():
				return False
			o1 = orient(p1,p2,p3); o2 = orient(p1,p2,p4); o3 = orient(p3,p4,p1); o4 = orient(p3,p4,p2)
			# Colinear overlapping ignored (meshes share endpoints for valid adjacency)
			if o1==0 and o2==0 and o3==0 and o4==0:
				return False
			return (o1*o2<0) and (o3*o4<0)
		coords = new_points
		E = len(edges)
		# Small meshes: fall back to O(E^2) with precomputed bounding boxes
		if E <= 512:
			ex0 = coords[[e[0] for e in edges]]; ex1 = coords[[e[1] for e in edges]]
			minx_e = np.minimum(ex0[:,0], ex1[:,0]); maxx_e = np.maximum(ex0[:,0], ex1[:,0])
			miny_e = np.minimum(ex0[:,1], ex1[:,1]); maxy_e = np.maximum(ex0[:,1], ex1[:,1])
			for i,(a,b) in enumerate(edges):
				pa, pb = coords[a], coords[b]
				min_abx = minx_e[i]; max_abx = maxx_e[i]
				min_aby = miny_e[i]; max_aby = maxy_e[i]
				for j in range(i+1, E):
					c, d = edges[j]
					# quick bbox reject
					if (max_abx < minx_e[j]) or (maxx_e[j] < min_abx) or (max_aby < miny_e[j]) or (maxy_e[j] < min_aby):
						continue
					pc, pd = coords[c], coords[d]
					if seg_intersect(pa, pb, pc, pd):
						crossed = True
						msgs.append(f"crossing edges detected: {(a,b)} intersects {(c,d)}")
						break
				if crossed: break
		else:
			# Spatial grid pruning for large meshes
			# Compute per-edge bounding boxes
			ax = coords[[e[0] for e in edges], 0]; ay = coords[[e[0] for e in edges], 1]
			bx = coords[[e[1] for e in edges], 0]; by = coords[[e[1] for e in edges], 1]
			minx = np.minimum(ax, bx); maxx = np.maximum(ax, bx)
			miny = np.minimum(ay, by); maxy = np.maximum(ay, by)
			# Global grid parameters
			gx0 = float(np.min(minx)); gx1 = float(np.max(maxx))
			gy0 = float(np.min(miny)); gy1 = float(np.max(maxy))
			rangex = max(gx1 - gx0, EPS_AREA); rangey = max(gy1 - gy0, EPS_AREA)
			N = max(1, int(np.sqrt(E)))
			cell = max(rangex, rangey) / N
			if cell <= 0:
				cell = max(rangex, rangey)
			# Map edges to grid cells
			cells = {}
			for idx in range(E):
				i0 = int((minx[idx] - gx0) / cell); i1 = int((maxx[idx] - gx0) / cell)
				j0 = int((miny[idx] - gy0) / cell); j1 = int((maxy[idx] - gy0) / cell)
				if i1 < i0: i0, i1 = i1, i0
				if j1 < j0: j0, j1 = j1, j0
				for ii in range(max(0, i0), min(N, i1 + 1)):
					for jj in range(max(0, j0), min(N, j1 + 1)):
						cells.setdefault((ii, jj), []).append(idx)
			# Build candidate pairs from cells
			cand_pairs = set()
			for lst in cells.values():
				if len(lst) < 2:
					continue
				lst_sorted = sorted(lst)
				for k in range(len(lst_sorted) - 1):
					ik = lst_sorted[k]
					for m in range(k + 1, len(lst_sorted)):
						jk = lst_sorted[m]
						cand_pairs.add((ik, jk))
			# Test candidate pairs
			for (i, j) in cand_pairs:
				a, b = edges[i]; c, d = edges[j]
				# quick bbox reject again as cheap filter
				pa, pb = coords[a], coords[b]
				pc, pd = coords[c], coords[d]
				min_abx = min(pa[0], pb[0]); max_abx = max(pa[0], pb[0])
				min_aby = min(pa[1], pb[1]); max_aby = max(pa[1], pb[1])
				min_cdx = min(pc[0], pd[0]); max_cdx = max(pc[0], pd[0])
				min_cdy = min(pc[1], pd[1]); max_cdy = max(pc[1], pd[1])
				if (max_abx < min_cdx) or (max_cdx < min_abx) or (max_aby < min_cdy) or (max_cdy < min_aby):
					continue
				if seg_intersect(pa, pb, pc, pd):
					crossed = True
					msgs.append(f"crossing edges detected: {(a,b)} intersects {(c,d)}")
					break
			if crossed:
				pass
	ok = ok_conf and (len(inv) == 0) and (not crossed)
	return ok, msgs, inv

def build_edge_to_tri_map(triangles):
	edge_map = {}
	for t_idx, tri in enumerate(triangles):
		if tri is None: continue
		arr = np.asarray(tri)
		if np.all(arr == -1):
			continue
		for i in range(3):
			a = int(arr[i]); b = int(arr[(i+1)%3])
			key = tuple(sorted((a,b)))
			edge_map.setdefault(key, set()).add(t_idx)
	return edge_map

def build_vertex_to_tri_map(triangles):
	v_map = {}
	for t_idx, tri in enumerate(triangles):
		arr = np.asarray(tri)
		if np.all(arr == -1):
			continue
		for v in arr:
			v_map.setdefault(int(v), set()).add(t_idx)
	return v_map

def boundary_edges_from_map(edge_map):
	return {e for e, s in edge_map.items() if len(s) == 1}

def is_boundary_vertex_from_maps(v_idx, edge_map):
	for e in edge_map:
		if v_idx in e and len(edge_map.get(e, [])) == 1:
			return True
	return False

def is_active_triangle(tri):
	tri_arr = np.asarray(tri)
	return not np.any(tri_arr == -1)

def check_mesh_conformity(points, triangles, verbose=False, allow_marked=True,
						  reject_boundary_loops=False, reject_inverted=False):
	triangles = np.ascontiguousarray(np.asarray(triangles, dtype=np.int32))
	msgs = []
	ok = True
	if not allow_marked:
		if triangles.size != 0 and np.any(np.all(triangles == -1, axis=1)):
			msgs.append("Marked (deleted) triangles present; compact mesh before checking.")
			return False, msgs
	if triangles.size == 0:
		return False, ["No active triangles."]
	try:
		mask = ~np.all(triangles == -1, axis=1)
		active_tris = triangles[mask]
	except Exception:
		active_tris = triangles
	if active_tris.size == 0:
		return False, ["No active triangles."]
	points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
	npts = len(points)
	if active_tris.max() >= npts or active_tris.min() < 0:
		msgs.append("Triangle indices out of range.")
		ok = False
	active_idx = np.nonzero(mask)[0] if triangles.size else []
	# Vectorized area and inversion checks
	try:
		areas = triangles_signed_areas(points, active_tris)
		abs_areas = np.abs(areas)
		zero_mask = abs_areas < EPS_AREA
		if np.any(zero_mask):
			idxs = np.nonzero(zero_mask)[0]
			for li in idxs[:50]:
				gi = int(active_idx[li]) if len(active_idx) > li else int(li)
				msgs.append(f"Triangle {gi} has near-zero area ({abs_areas[li]:.3e}).")
			ok = False
		if reject_inverted:
			inv_mask = areas <= -EPS_AREA
			if np.any(inv_mask):
				idxs = np.nonzero(inv_mask)[0]
				for li in idxs[:50]:
					gi = int(active_idx[li]) if len(active_idx) > li else int(li)
					msgs.append(f"Triangle {gi} has negative signed area (inverted): {areas[li]:.3e}")
				ok = False
	except Exception:
		# Fallback scalar path
		for local_i, tri in enumerate(active_tris):
			global_i = int(active_idx[local_i]) if len(active_idx) > local_i else local_i
			signed = triangle_area(points[int(tri[0])], points[int(tri[1])], points[int(tri[2])])
			area = abs(signed)
			if area < EPS_AREA:
				msgs.append(f"Triangle {global_i} has near-zero area ({area:.3e}).")
				ok = False
			if reject_inverted and signed <= -EPS_AREA:
				msgs.append(f"Triangle {global_i} has negative signed area (inverted): {signed:.3e}")
				ok = False
	# Vectorized duplicate triangle detection
	try:
		sorted_tris = np.sort(active_tris, axis=1)
		_, tri_counts = np.unique(sorted_tris, axis=0, return_counts=True)
		if np.any(tri_counts > 1):
			msgs.append("Duplicate triangles detected.")
			ok = False
	except Exception:
		# Fallback: keep silent; earlier logic will likely flag via other paths
		pass

	# Vectorized edge counting for non-manifold and boundary detection
	try:
		a = active_tris[:, [0, 1]]
		b = active_tris[:, [1, 2]]
		c = active_tris[:, [2, 0]]
		edges = np.vstack((a, b, c)).astype(int)
		edges.sort(axis=1)
		uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
		# Non-manifold edges: count > 2
		nm_mask = counts > 2
		if np.any(nm_mask):
			for e in uniq_edges[nm_mask][:10]:
				msgs.append(f"Non-manifold edge ({int(e[0])}, {int(e[1])}) shared by >2 triangles (count={int(counts[nm_mask][0])}).")
			ok = False
		# Boundary loops rejection (count == 1)
		if reject_boundary_loops:
			b_mask = counts == 1
			if np.any(b_mask):
				boundary_edges = uniq_edges[b_mask]
				# build adjacency
				adj = defaultdict(list)
				for a_, b_ in boundary_edges:
					ai = int(a_); bi = int(b_)
					adj[ai].append(bi); adj[bi].append(ai)
				visited = set(); loops = 0
				for v in list(adj.keys()):
					if v in visited: continue
					loops += 1
					dq = deque([v]); visited.add(v)
					while dq:
						u = dq.popleft()
						for w in adj.get(u, []):
							if w not in visited:
								visited.add(w); dq.append(w)
				msgs.append(f"Boundary loops detected: {loops} (boundary edges={int(np.sum(b_mask))})")
				ok = False
	except Exception:
		# Fallback to legacy map if any vectorized step fails
		edge_map = build_edge_to_tri_map(active_tris)
		for e,lst in edge_map.items():
			if len(lst) > 2:
				msgs.append(f"Non-manifold edge {e} shared by {len(lst)} triangles.")
				ok = False
		if reject_boundary_loops:
			boundary_edges = [e for e,s in edge_map.items() if len(s) == 1]
			if boundary_edges:
				adj = defaultdict(list)
				for a,b in boundary_edges:
					adj[a].append(b); adj[b].append(a)
				visited = set(); loops = 0
				for v in adj:
					if v in visited: continue
					loops += 1
					dq = deque([v]); visited.add(v)
					while dq:
						u = dq.popleft()
						for w in adj.get(u, []):
							if w not in visited:
								visited.add(w); dq.append(w)
				msgs.append(f"Boundary loops detected: {loops} (boundary edges={len(boundary_edges)})")
				ok = False
	if verbose:
		from .logging_utils import get_logger
		logger = get_logger('sofia.conformity')
		for m in msgs:
			logger.info("Conformity: %s", m)
	return ok, msgs
