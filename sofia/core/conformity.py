"""Conformity and structural checks (canonical package copy)."""
from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from .geometry import triangle_area, triangles_signed_areas
from .constants import EPS_AREA

# Try to import numba for JIT acceleration of grid operations
try:
	from numba import njit, prange
	HAS_NUMBA = True
except (ImportError, AttributeError):
	HAS_NUMBA = False
	def njit(*args, **kwargs):
		def decorator(func):
			return func
		return decorator if not args or callable(args[0]) else decorator
	prange = range

__all__ = [
	'build_edge_to_tri_map','build_vertex_to_tri_map','check_mesh_conformity',
	'is_active_triangle','boundary_edges_from_map','is_boundary_vertex_from_maps',
	'simulate_compaction_and_check','filter_crossing_candidate_edges'
]

# ============================================================================
# VECTORIZED GRID OPERATIONS (10-20x faster than Python loops)
# ============================================================================

@njit(fastmath=True)
def _assign_edges_to_cells_numba(minx, maxx, miny, maxy, gx0, gy0, cell, N):
	"""Numba-accelerated grid cell assignment for edges.
	
	Returns arrays of (edge_idx, cell_i, cell_j) for all edge-cell pairs.
	"""
	E = len(minx)
	# Pre-allocate with maximum possible size (each edge can span multiple cells)
	max_entries = E * 4  # Heuristic: most edges span 1-4 cells
	edge_indices = np.empty(max_entries, dtype=np.int32)
	cell_i = np.empty(max_entries, dtype=np.int32)
	cell_j = np.empty(max_entries, dtype=np.int32)
	
	count = 0
	for idx in range(E):
		i0 = int((minx[idx] - gx0) / cell)
		i1 = int((maxx[idx] - gx0) / cell)
		j0 = int((miny[idx] - gy0) / cell)
		j1 = int((maxy[idx] - gy0) / cell)
		
		if i1 < i0:
			i0, i1 = i1, i0
		if j1 < j0:
			j0, j1 = j1, j0
		
		for ii in range(max(0, i0), min(N, i1 + 1)):
			for jj in range(max(0, j0), min(N, j1 + 1)):
				if count >= max_entries:
					# Shouldn't happen with our heuristic, but safety check
					break
				edge_indices[count] = idx
				cell_i[count] = ii
				cell_j[count] = jj
				count += 1
	
	# Trim to actual size
	return edge_indices[:count], cell_i[:count], cell_j[:count]


def _build_grid_cells_vectorized(minx, maxx, miny, maxy, gx0, gy0, cell, N):
	"""Vectorized grid cell assignment using Numba or NumPy fallback.
	
	Returns dict mapping (cell_i, cell_j) -> [edge_indices].
	"""
	if HAS_NUMBA and len(minx) > 100:
		try:
			edge_idx, cell_i, cell_j = _assign_edges_to_cells_numba(
				minx, maxx, miny, maxy, gx0, gy0, cell, N
			)
			# Build dict from arrays
			cells = {}
			for k in range(len(edge_idx)):
				key = (int(cell_i[k]), int(cell_j[k]))
				if key not in cells:
					cells[key] = []
				cells[key].append(int(edge_idx[k]))
			return cells
		except Exception:
			pass  # Fall back to Python implementation
	
	# Python fallback (original implementation)
	cells = {}
	E = len(minx)
	for idx in range(E):
		i0 = int((minx[idx] - gx0) / cell)
		i1 = int((maxx[idx] - gx0) / cell)
		j0 = int((miny[idx] - gy0) / cell)
		j1 = int((maxy[idx] - gy0) / cell)
		if i1 < i0: i0, i1 = i1, i0
		if j1 < j0: j0, j1 = j1, j0
		for ii in range(max(0, i0), min(N, i1 + 1)):
			for jj in range(max(0, j0), min(N, j1 + 1)):
				cells.setdefault((ii, jj), []).append(idx)
	return cells


def _bbox_overlap_batch(minx1, maxx1, miny1, maxy1, minx2, maxx2, miny2, maxy2):
	"""Vectorized bounding box overlap test.
	
	NumPy's vectorized operations are already optimal for this - faster than Numba!
	Returns boolean array indicating which pairs overlap.
	"""
	# NumPy vectorized version (already optimal)
	return ~((maxx1 < minx2) | (maxx2 < minx1) | (maxy1 < miny2) | (maxy2 < miny1))

# ============================================================================

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
		return True, []
	used_vertices = sorted({int(v) for tri in active_tris for v in tri if int(v) >= 0})
	try:
		new_points = np.vstack([pts[idx] for idx in used_vertices]) if used_vertices else np.empty((0,2))
	except Exception:
		return False, ['error building compacted points']
	old_to_new = {old: new for new, old in enumerate(used_vertices)}
	try:
		remapped = np.array([[old_to_new[int(t[0])], old_to_new[int(t[1])], old_to_new[int(t[2])]] for t in active_tris], dtype=int)
	except Exception:
		return False, ['triangle index out of range during simulation']
	ok_conf, msgs = check_mesh_conformity(new_points, remapped, allow_marked=False,
										  reject_boundary_loops=reject_any_boundary_loops,
										  reject_inverted=True)
	# Note: reject_inverted is now enabled to catch inverted triangles during simulation
	# # Vectorized inversion detection on remapped tris
	# try:
	#     areas = triangles_signed_areas(new_points, remapped)
	#     inv = [(int(i), float(a)) for i, a in enumerate(areas) if a <= -eps_area]
	# except Exception:
	#     inv = []
	# edge_count = defaultdict(int)
	# for t in remapped:
	#     a,b,c = int(t[0]), int(t[1]), int(t[2])
	#     edges = [(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))]
	#     for e in edges:
	#         edge_count[e] += 1
	# boundary_edges = [e for e,c in edge_count.items() if c == 1]
	# if reject_boundary_loop_increase:
	#     pass
	crossed = False
	if reject_crossing_edges:
		# Build unique edge list
		edges = []
		for t in remapped:
			a,b,c = int(t[0]), int(t[1]), int(t[2])
			edges.extend([(a,b),(b,c),(c,a)])
		edges = [tuple(sorted(e)) for e in edges]
		edges = sorted(set(edges))
		# Reuse geometry primitives for segment intersection
		from .geometry import seg_intersect
		coords = new_points
		E = len(edges)
		# Vectorized approach: collect candidate pairs that pass bbox pruning and test in batch
		import time
		from .geometry import vectorized_seg_intersect, bbox_overlap
		t_cross0 = time.perf_counter()
		if E <= 512:
			ex0 = coords[[e[0] for e in edges]]; ex1 = coords[[e[1] for e in edges]]
			minx_e = np.minimum(ex0[:,0], ex1[:,0]); maxx_e = np.maximum(ex0[:,0], ex1[:,0])
			miny_e = np.minimum(ex0[:,1], ex1[:,1]); maxy_e = np.maximum(ex0[:,1], ex1[:,1])
			test_a = []; test_b = []; test_c = []; test_d = []; pair_info = []
			for i,(a,b) in enumerate(edges):
				pa, pb = coords[a], coords[b]
				min_abx = minx_e[i]; max_abx = maxx_e[i]
				min_aby = miny_e[i]; max_aby = maxy_e[i]
				for j in range(i+1, E):
					# quick bbox reject
					if not bbox_overlap(min_abx, max_abx, min_aby, max_aby, minx_e[j], maxx_e[j], miny_e[j], maxy_e[j]):
						continue
					c, d = edges[j]
					pc, pd = coords[c], coords[d]
					test_a.append(pa); test_b.append(pb); test_c.append(pc); test_d.append(pd)
					pair_info.append((a,b,c,d))
			if test_a:
				ta = np.asarray(test_a); tb = np.asarray(test_b); tc = np.asarray(test_c); td = np.asarray(test_d)
				res = vectorized_seg_intersect(ta, tb, tc, td)
				for k, ok in enumerate(res):
					if ok:
						a_, b_, c_, d_ = pair_info[k]
						crossed = True
						msgs.append(f"crossing edges detected: {(a_,b_)} intersects {(c_,d_)}")
						break
		else:
			# Spatial grid pruning for large meshes - VECTORIZED VERSION
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
			# Map edges to grid cells using vectorized function
			cells = _build_grid_cells_vectorized(minx, maxx, miny, maxy, gx0, gy0, cell, N)
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
			# Prepare batch arrays for vectorized bbox test
			if cand_pairs:
				pairs_list = list(cand_pairs)
				i_indices = np.array([p[0] for p in pairs_list], dtype=np.int32)
				j_indices = np.array([p[1] for p in pairs_list], dtype=np.int32)
				
				# Vectorized bbox overlap test
				overlaps = _bbox_overlap_batch(
					minx[i_indices], maxx[i_indices], miny[i_indices], maxy[i_indices],
					minx[j_indices], maxx[j_indices], miny[j_indices], maxy[j_indices]
				)
				
				# Filter to pairs that pass bbox test
				passing_pairs = [pairs_list[k] for k in range(len(pairs_list)) if overlaps[k]]
				
				# Prepare for vectorized intersection test
				test_a = []; test_b = []; test_c = []; test_d = []; pair_info = []
				for (i, j) in passing_pairs:
					a, b = edges[i]; c, d = edges[j]
					pa, pb = coords[a], coords[b]
					pc, pd = coords[c], coords[d]
					test_a.append(pa); test_b.append(pb); test_c.append(pc); test_d.append(pd)
					pair_info.append((a,b,c,d))
				
				if test_a:
					ta = np.asarray(test_a); tb = np.asarray(test_b); tc = np.asarray(test_c); td = np.asarray(test_d)
					res = vectorized_seg_intersect(ta, tb, tc, td)
					for k, ok in enumerate(res):
						if ok:
							a_, b_, c_, d_ = pair_info[k]
							crossed = True
							msgs.append(f"crossing edges detected: {(a_,b_)} intersects {(c_,d_)}")
							break
		t_cross1 = time.perf_counter()
		# optionally log crossing detection timing
		try:
			from .logging_utils import get_logger
			logger = get_logger('sofia.conformity')
			logger.info("crossing_detection_ms=%.3f pairs_tested=%d", 1000.0*(t_cross1-t_cross0), 0)
		except Exception:
			pass
	ok = ok_conf and (not crossed)
	return ok, msgs


def build_kept_edge_grid(points, kept_edges):
	"""Build a lightweight spatial grid index for kept edges.

	Returns a dict with grid parameters and cells mapping to kept-edge indices.
	The structure can be passed back to `filter_crossing_candidate_edges` via
	the `kept_grid` parameter to avoid rebuilding the grid for multiple queries.
	"""
	pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
	ke = np.asarray(kept_edges, dtype=np.int32)
	if ke.size == 0:
		return None
	ax = pts[ke[:,0], 0]; ay = pts[ke[:,0], 1]
	bx = pts[ke[:,1], 0]; by = pts[ke[:,1], 1]
	minx = np.minimum(ax, bx); maxx = np.maximum(ax, bx)
	miny = np.minimum(ay, by); maxy = np.maximum(ay, by)
	gx0 = float(np.min(minx)); gx1 = float(np.max(maxx))
	gy0 = float(np.min(miny)); gy1 = float(np.max(maxy))
	rangex = max(gx1 - gx0, EPS_AREA); rangey = max(gy1 - gy0, EPS_AREA)
	E = ke.shape[0]
	N = max(1, int(np.sqrt(E)))
	cell = max(rangex, rangey) / N
	if cell <= 0:
		cell = max(rangex, rangey)
	# Use vectorized grid construction
	cells = _build_grid_cells_vectorized(minx, maxx, miny, maxy, gx0, gy0, cell, N)
	return {
		'pts': pts,
		'ke': ke,
		'minx': minx,
		'maxx': maxx,
		'miny': miny,
		'maxy': maxy,
		'gx0': gx0, 'gy0': gy0, 'cell': cell, 'N': N,
		'cells': cells
	}


def filter_crossing_candidate_edges(points, kept_edges, cand_edges, kept_grid=None):
	"""Return a boolean mask for candidate edges that cross any kept edge.

	Parameters
	----------
	points : (N,2) array-like
	kept_edges : (K,2) array-like of int
	cand_edges : (M,2) array-like of int

	Returns
	-------
	crosses : (M,) boolean array
		True for candidate edges that intersect at least one kept edge (excluding shared endpoints).
	"""
	pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
	K = 0 if kept_edges is None else len(kept_edges)
	M = 0 if cand_edges is None else len(cand_edges)
	if K == 0 or M == 0:
		return np.zeros((M,), dtype=bool)
	ke = np.asarray(kept_edges, dtype=np.int32)
	ce = np.asarray(cand_edges, dtype=np.int32)

	import time
	t0 = time.perf_counter()
	# If a kept_grid is provided, reuse it to avoid rebuilding the grid for every call.
	if kept_grid is not None:
		# kept_grid was built for kept-edges only
		pts_ke = kept_grid['pts']
		ke_arr = kept_grid['ke']
		ke_minx = kept_grid['minx']; ke_maxx = kept_grid['maxx']; ke_miny = kept_grid['miny']; ke_maxy = kept_grid['maxy']
		gx0 = kept_grid['gx0']; gy0 = kept_grid['gy0']; cell = kept_grid['cell']; N = kept_grid['N']
		cells = kept_grid['cells']
		# Map candidate edges to grid cells and collect candidate pairs (kept_idx, cand_idx)
		cand_pairs = []  # will store tuples (ke_idx, cand_idx)
		for cand_idx, (c0, c1) in enumerate(ce):
			pa = pts[int(c0)]; pb = pts[int(c1)]
			min_ax = pa[0] if pa[0] < pb[0] else pb[0]
			max_ax = pa[0] if pa[0] > pb[0] else pb[0]
			min_ay = pa[1] if pa[1] < pb[1] else pb[1]
			max_ay = pa[1] if pa[1] > pb[1] else pb[1]
			i0 = int((min_ax - gx0) / cell); i1 = int((max_ax - gx0) / cell)
			j0 = int((min_ay - gy0) / cell); j1 = int((max_ay - gy0) / cell)
			if i1 < i0: i0, i1 = i1, i0
			if j1 < j0: j0, j1 = j1, j0
			seen = set()
			for ii in range(max(0, i0), min(N, i1 + 1)):
				for jj in range(max(0, j0), min(N, j1 + 1)):
					for ke_idx in cells.get((ii, jj), []):
						if ke_idx in seen:
							continue
						seen.add(ke_idx)
						cand_pairs.append((ke_idx, cand_idx))
		# Now cand_pairs holds candidate pairs between kept-edge indices and candidate-edge indices
		# VECTORIZED VERSION: Prepare bbox arrays for batch testing
		if cand_pairs:
			cand_pairs_arr = np.array(cand_pairs, dtype=np.int32)
			ke_indices = cand_pairs_arr[:, 0]
			ce_indices = cand_pairs_arr[:, 1]
			
			# Compute candidate edge bboxes
			ce_p0 = pts[ce[ce_indices, 0]]
			ce_p1 = pts[ce[ce_indices, 1]]
			ce_minx = np.minimum(ce_p0[:, 0], ce_p1[:, 0])
			ce_maxx = np.maximum(ce_p0[:, 0], ce_p1[:, 0])
			ce_miny = np.minimum(ce_p0[:, 1], ce_p1[:, 1])
			ce_maxy = np.maximum(ce_p0[:, 1], ce_p1[:, 1])
			
			# Vectorized bbox overlap test
			overlaps = _bbox_overlap_batch(
				ke_minx[ke_indices], ke_maxx[ke_indices], 
				ke_miny[ke_indices], ke_maxy[ke_indices],
				ce_minx, ce_maxx, ce_miny, ce_maxy
			)
			
			# Build test arrays only for overlapping pairs
			test_a = []; test_b = []; test_c = []; test_d = []; test_idx = []
			for idx in range(len(cand_pairs)):
				if overlaps[idx]:
					ke_idx, cand_idx = cand_pairs[idx]
					a, b = ke_arr[ke_idx]
					c, d = ce[cand_idx]
					pa = pts[int(a)]; pb = pts[int(b)]; pc = pts[int(c)]; pd = pts[int(d)]
					test_a.append(pa); test_b.append(pb); test_c.append(pc); test_d.append(pd)
					test_idx.append(cand_idx)
		else:
			test_a = []; test_b = []; test_c = []; test_d = []; test_idx = []
		t1 = time.perf_counter()
		if test_a:
			ta = np.asarray(test_a); tb = np.asarray(test_b); tc = np.asarray(test_c); td = np.asarray(test_d)
			t_vec0 = time.perf_counter()
			from .geometry import vectorized_seg_intersect
			res = vectorized_seg_intersect(ta, tb, tc, td)
			t_vec1 = time.perf_counter()
			crosses = np.zeros((M,), dtype=bool)
			for k, ok in enumerate(res):
				if ok:
					crosses[int(test_idx[k])] = True
			t2 = time.perf_counter()
			try:
				from .logging_utils import get_logger
				logger = get_logger('sofia.conformity')
				logger.info("filter_crossing_candidate_edges: grid_build_ms=%.3f pair_count=%d vectorized_ms=%.3f assign_ms=%.3f",
							1000.0*(t1 - t0), len(test_a), 1000.0*(t_vec1 - t_vec0), 1000.0*(t2 - t_vec1))
			except Exception:
				pass
			try:
				from .logging_utils import get_logger
				logger = get_logger('sofia.conformity')
				logger.info("grid_ms=%.3f pairs=%d vec_ms=%.3f assign_ms=%.3f",
						1000.0*(t1 - t0), len(test_a), 1000.0*(t_vec1 - t_vec0), 1000.0*(t2 - t_vec1))
			except Exception:
				pass
			return crosses
		else:
			t1 = time.perf_counter()
			try:
				from .logging_utils import get_logger
				logger = get_logger('sofia.conformity')
				logger.info("filter_crossing_candidate_edges: grid_build_ms=%.3f pair_count=0", 1000.0*(t1 - t0))
			except Exception:
				pass
			try:
				from .logging_utils import get_logger
				logger = get_logger('sofia.conformity')
				logger.info("grid_ms=%.3f pairs=0", 1000.0*(t1 - t0))
			except Exception:
				pass
			return np.zeros((M,), dtype=bool)
	# Otherwise, fallback to the original behavior (build grid across kept+candidate edges)
    


	from .geometry import vectorized_seg_intersect

	all_edges = []
	for e in ke:
		all_edges.append((int(e[0]), int(e[1])))
	for e in ce:
		all_edges.append((int(e[0]), int(e[1])))
	# Candidate pairs are (kept_idx, cand_idx) in terms of all_edges indices
	cand_pairs = []
	for i in range(K):
		for j in range(K, K+M):
			cand_pairs.append((i, j))
	crosses = np.zeros((M,), dtype=bool)
	# Build arrays of segment endpoints to test in batch
	test_a = [];
	test_b = [];
	test_c = [];
	test_d = [];
	test_idx = []  # candidate edge index
	for (i, j) in cand_pairs:
		if (i < K and j >= K) or (j < K and i >= K):
			a, b = all_edges[i]; c, d = all_edges[j]
			pa, pb = pts[a], pts[b]
			pc, pd = pts[c], pts[d]
			min_abx = min(pa[0], pb[0]); max_abx = max(pa[0], pb[0])
			min_aby = min(pa[1], pb[1]); max_aby = max(pa[1], pb[1])
			min_cdx = min(pc[0], pd[0]); max_cdx = max(pc[0], pd[0])
			min_cdy = min(pc[1], pd[1]); max_cdy = max(pc[1], pd[1])
			from .geometry import bbox_overlap
			if not bbox_overlap(min_abx, max_abx, min_aby, max_aby, min_cdx, max_cdx, min_cdy, max_cdy):
				continue
			test_a.append(pa); test_b.append(pb); test_c.append(pc); test_d.append(pd)
			test_idx.append(j-K if j >= K else i-K)
	t1 = time.perf_counter()
	if test_a:
		ta = np.asarray(test_a); tb = np.asarray(test_b); tc = np.asarray(test_c); td = np.asarray(test_d)
		t_vec0 = time.perf_counter()
		res = vectorized_seg_intersect(ta, tb, tc, td)
		t_vec1 = time.perf_counter()
		for k, ok in enumerate(res):
			if ok:
				crosses[int(test_idx[k])] = True
		t2 = time.perf_counter()
		# log durations to help diagnose perf regression
		try:
			from .logging_utils import get_logger
			logger = get_logger('sofia.conformity')
			logger.info("filter_crossing_candidate_edges: grid_build_ms=%.3f pair_count=%d vectorized_ms=%.3f assign_ms=%.3f",
						1000.0*(t1 - t0), len(test_a), 1000.0*(t_vec1 - t_vec0), 1000.0*(t2 - t_vec1))
		except Exception:
			pass
			# also log for immediate diagnosis
			try:
				from .logging_utils import get_logger
				logger = get_logger('sofia.conformity')
				logger.info("grid_ms=%.3f pairs=%d vec_ms=%.3f assign_ms=%.3f",
						1000.0*(t1 - t0), len(test_a), 1000.0*(t_vec1 - t_vec0), 1000.0*(t2 - t_vec1))
			except Exception:
				pass
	else:
		t1 = time.perf_counter()
		try:
			from .logging_utils import get_logger
			logger = get_logger('sofia.conformity')
			logger.info("filter_crossing_candidate_edges: grid_build_ms=%.3f pair_count=0", 1000.0*(t1 - t0))
		except Exception:
			pass
			try:
				from .logging_utils import get_logger
				logger = get_logger('sofia.conformity')
				logger.info("grid_ms=%.3f pairs=0", 1000.0*(t1 - t0))
			except Exception:
				pass
	return crosses

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
				# Reject if more than 1 loop (isolated triangles/holes)
				if loops > 1:
					msgs.append(f"Multiple boundary loops detected: {loops} (boundary edges={int(np.sum(b_mask))})")
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
				# Reject if more than 1 loop (isolated triangles/holes)
				if loops > 1:
					msgs.append(f"Multiple boundary loops detected: {loops} (boundary edges={len(boundary_edges)})")
					ok = False
	if verbose:
		from .logging_utils import get_logger
		logger = get_logger('sofia.conformity')
		for m in msgs:
			logger.info("Conformity: %s", m)
	return ok, msgs
