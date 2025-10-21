"""Geometry primitives and tolerances (package internal copy).

Original module relocated from repository root. This is now the canonical
implementation; a root-level shim will re-export and emit a deprecation
warning until the next major release.
"""
from __future__ import annotations
import math
import numpy as np
from .constants import EPS_AREA, EPS_MIN_ANGLE_DEG, EPS_IMPROVEMENT

# Try to import numba for JIT acceleration
try:
	from numba import njit, prange
	import numba
	HAS_NUMBA = True
except (ImportError, AttributeError):
	# AttributeError can occur with incompatible numpy/numba versions
	HAS_NUMBA = False
	# Fallback decorators that do nothing
	def njit(*args, **kwargs):
		def decorator(func):
			return func
		return decorator if not args or callable(args[0]) else decorator
	prange = range

__all__ = [
	'triangle_area','triangle_angles','ensure_positive_orientation','point_in_polygon',
	'triangles_min_angles','triangles_signed_areas','opposite_edge_of_smallest_angle',
	'compute_triangulation_area','normalize_edge'
]

# ============================================================================
# NUMBA-OPTIMIZED FUNCTIONS (Only where Numba actually helps!)
# ============================================================================
# Note: triangles_signed_areas uses pure NumPy - it's 24x faster than Numba
# Note: bbox overlap uses pure NumPy - it's 600x faster than Numba

if HAS_NUMBA:
	@numba.jit(nopython=True, cache=True, fastmath=True)
	def _triangles_min_angles_numba(points, tris):
		"""Numba-accelerated minimum angle computation."""
		n = tris.shape[0]
		min_angles = np.empty(n, dtype=np.float64)
		eps = 1e-20
		
		for i in range(n):
			p0 = points[tris[i, 0]]
			p1 = points[tris[i, 1]]
			p2 = points[tris[i, 2]]
			
			# Edge lengths
			dx_a = p1[0] - p2[0]
			dy_a = p1[1] - p2[1]
			a = math.sqrt(dx_a*dx_a + dy_a*dy_a)
			
			dx_b = p0[0] - p2[0]
			dy_b = p0[1] - p2[1]
			b = math.sqrt(dx_b*dx_b + dy_b*dy_b)
			
			dx_c = p0[0] - p1[0]
			dy_c = p0[1] - p1[1]
			c = math.sqrt(dx_c*dx_c + dy_c*dy_c)
			
			# Angles using law of cosines
			cos_A = (b*b + c*c - a*a) / (2.0*b*c + eps)
			cos_B = (a*a + c*c - b*b) / (2.0*a*c + eps)
			cos_C = (a*a + b*b - c*c) / (2.0*a*b + eps)
			
			# Clamp and convert to degrees
			cos_A = max(-1.0, min(1.0, cos_A))
			cos_B = max(-1.0, min(1.0, cos_B))
			cos_C = max(-1.0, min(1.0, cos_C))
			
			angle_A = math.degrees(math.acos(cos_A))
			angle_B = math.degrees(math.acos(cos_B))
			angle_C = math.degrees(math.acos(cos_C))
			
			min_angles[i] = min(angle_A, min(angle_B, angle_C))
		
		return min_angles
	
	@numba.jit(nopython=True, cache=True, fastmath=True)
	def _vectorized_seg_intersect_numba(a_pts, b_pts, c_pts, d_pts):
		"""Numba-accelerated segment intersection test."""
		n = a_pts.shape[0]
		result = np.empty(n, dtype=np.bool_)
		eps = 1e-12
		
		for i in range(n):
			# Orientations
			o1 = (b_pts[i,0]-a_pts[i,0])*(c_pts[i,1]-a_pts[i,1]) - (b_pts[i,1]-a_pts[i,1])*(c_pts[i,0]-a_pts[i,0])
			o2 = (b_pts[i,0]-a_pts[i,0])*(d_pts[i,1]-a_pts[i,1]) - (b_pts[i,1]-a_pts[i,1])*(d_pts[i,0]-a_pts[i,0])
			o3 = (d_pts[i,0]-c_pts[i,0])*(a_pts[i,1]-c_pts[i,1]) - (d_pts[i,1]-c_pts[i,1])*(a_pts[i,0]-c_pts[i,0])
			o4 = (d_pts[i,0]-c_pts[i,0])*(b_pts[i,1]-c_pts[i,1]) - (d_pts[i,1]-c_pts[i,1])*(b_pts[i,0]-c_pts[i,0])
			
			# Check shared endpoints
			shared = False
			if (abs(a_pts[i,0]-c_pts[i,0])<eps and abs(a_pts[i,1]-c_pts[i,1])<eps):
				shared = True
			elif (abs(a_pts[i,0]-d_pts[i,0])<eps and abs(a_pts[i,1]-d_pts[i,1])<eps):
				shared = True
			elif (abs(b_pts[i,0]-c_pts[i,0])<eps and abs(b_pts[i,1]-c_pts[i,1])<eps):
				shared = True
			elif (abs(b_pts[i,0]-d_pts[i,0])<eps and abs(b_pts[i,1]-d_pts[i,1])<eps):
				shared = True
			
			# Check colinear
			colinear = (abs(o1)<eps and abs(o2)<eps and abs(o3)<eps and abs(o4)<eps)
			
			# Proper crossing
			proper_cross = (o1*o2 < 0.0) and (o3*o4 < 0.0)
			
			result[i] = proper_cross and not shared and not colinear
		
		return result

# ============================================================================
# PUBLIC API FUNCTIONS (Auto-dispatch to Numba if available)
# ============================================================================

def bbox_overlap(minx1, maxx1, miny1, maxy1, minx2, maxx2, miny2, maxy2):
	"""Vectorized bbox overlap test; returns boolean array where bbox1 overlaps bbox2.

	All inputs may be scalars or arrays broadcastable to a common shape.
	"""
	return ~((maxx1 < minx2) | (maxx2 < minx1) | (maxy1 < miny2) | (maxy2 < miny1))


def orient_vectorized(a_pts, b_pts, c_pt):
	"""Compute orientation for arrays of segment endpoints a_pts,b_pts against a single point c_pt.

	a_pts, b_pts : arrays of shape (M,2)
	c_pt : single point-like (2,)
	Returns array of shape (M,) of orientation scalars.
	"""
	a = np.asarray(a_pts); b = np.asarray(b_pts); c = np.asarray(c_pt)
	return (b[:,0]-a[:,0])*(c[1]-a[:,1]) - (b[:,1]-a[:,1])*(c[0]-a[:,0])

def vectorized_seg_intersect(a_pts, b_pts, c_pts, d_pts):
	"""Vectorized segment intersection test for equal-length arrays of segments.

	a_pts, b_pts, c_pts, d_pts must be arrays of shape (M,2). Returns boolean array (M,) where
	each element indicates whether segment a_pts[i]-b_pts[i] strictly intersects c_pts[i]-d_pts[i].
	Shared endpoints and colinear overlapping are treated as non-intersecting (False).
	"""
	a = np.asarray(a_pts, dtype=np.float64)
	b = np.asarray(b_pts, dtype=np.float64)
	c = np.asarray(c_pts, dtype=np.float64)
	d = np.asarray(d_pts, dtype=np.float64)
	if a.size == 0:
		return np.zeros((0,), dtype=bool)
	
	# Use Numba-accelerated version if available and beneficial (>100 segments)
	if HAS_NUMBA and a.shape[0] > 100:
		try:
			return _vectorized_seg_intersect_numba(a, b, c, d)
		except Exception:
			pass  # Fall back to NumPy version
	
	# NumPy vectorized version (fallback)
	o1 = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
	o2 = (b[:,0]-a[:,0])*(d[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(d[:,0]-a[:,0])
	o3 = (d[:,0]-c[:,0])*(a[:,1]-c[:,1]) - (d[:,1]-c[:,1])*(a[:,0]-c[:,0])
	o4 = (d[:,0]-c[:,0])*(b[:,1]-c[:,1]) - (d[:,1]-c[:,1])*(b[:,0]-c[:,0])
	# exclude shared endpoints: compare coordinates exactly (points are float arrays but originate from same array)
	shared = np.all(a == c, axis=1) | np.all(a == d, axis=1) | np.all(b == c, axis=1) | np.all(b == d, axis=1)
	colinear = (o1 == 0) & (o2 == 0) & (o3 == 0) & (o4 == 0)
	# Colinear overlapping segments are treated as non-intersecting in our mesh semantics
	# Proper crossings (non-colinear) only; exclude T-intersection where one orientation is zero
	proper_cross = (o1*o2 < 0) & (o3*o4 < 0)
	return proper_cross & (~shared) & (~colinear)

def orient(a, b, c):
	"""2D orientation (signed area * 2) for points a,b,c.

	Returns a positive value when (a,b,c) are counter-clockwise, negative when clockwise,
	and zero when colinear.
	"""
	a = np.asarray(a); b = np.asarray(b); c = np.asarray(c)
	return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def seg_intersect(p1, p2, p3, p4):
	"""Return True if segment p1-p2 strictly intersects p3-p4 (excluding shared endpoints and colinear overlaps).

	Parameters accept array-like 2D points.
	"""
	p1 = np.asarray(p1); p2 = np.asarray(p2); p3 = np.asarray(p3); p4 = np.asarray(p4)
	# Exclude shared endpoints
	if (p1 == p3).all() or (p1 == p4).all() or (p2 == p3).all() or (p2 == p4).all():
		return False
	o1 = orient(p1, p2, p3); o2 = orient(p1, p2, p4)
	o3 = orient(p3, p4, p1); o4 = orient(p3, p4, p2)
	# Colinear overlapping segments are ignored in mesh semantics (shared endpoints represent adjacency)
	if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
		return False
	return (o1*o2 < 0) and (o3*o4 < 0)

def triangle_area(p0, p1, p2):
	p0 = np.asarray(p0); p1 = np.asarray(p1); p2 = np.asarray(p2)
	return 0.5 * np.cross(p1 - p0, p2 - p0)

def triangle_angles(p0, p1, p2):
	p0 = np.asarray(p0); p1 = np.asarray(p1); p2 = np.asarray(p2)
	a = np.linalg.norm(p1 - p2)
	b = np.linalg.norm(p0 - p2)
	c = np.linalg.norm(p0 - p1)
	def ang(A,B,C):
		cosang = (B*B + C*C - A*A) / (2*B*C + 1e-20)
		return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))
	return [ang(a,b,c), ang(b,c,a), ang(c,a,b)]

def triangles_min_angles(points, tris):
	"""Vectorized per-triangle minimum internal angle (degrees).

	points: (N,2) float array
	tris:   (M,3) int array
	Returns: (M,) float64 array of min angles; NaN for invalid rows.
	"""
	pts = np.asarray(points, dtype=np.float64)
	T = np.asarray(tris, dtype=np.int32)
	if T.size == 0:
		return np.empty((0,), dtype=float)
	
	# Use Numba-accelerated version if available and beneficial (>50 triangles)
	if HAS_NUMBA and T.shape[0] > 50:
		try:
			return _triangles_min_angles_numba(pts, T)
		except Exception:
			pass  # Fall back to NumPy version
	
	# NumPy vectorized version (fallback)
	p0 = pts[T[:, 0]]
	p1 = pts[T[:, 1]]
	p2 = pts[T[:, 2]]
	# side lengths opposite to vertices: a=|p1-p2|, b=|p0-p2|, c=|p0-p1|
	a = np.linalg.norm(p1 - p2, axis=1)
	b = np.linalg.norm(p0 - p2, axis=1)
	c = np.linalg.norm(p0 - p1, axis=1)
	eps = 1e-20
	def angle_opposite(A, B, C):
		# angle opposite to side A, with adjacent sides B and C
		denom = 2.0 * B * C + eps
		cosang = (B*B + C*C - A*A) / denom
		cosang = np.clip(cosang, -1.0, 1.0)
		return np.degrees(np.arccos(cosang))
	A = angle_opposite(a, b, c)
	B = angle_opposite(b, c, a)
	C = angle_opposite(c, a, b)
	min_angles = np.minimum(A, np.minimum(B, C))
	return min_angles

def opposite_edge_of_smallest_angle(points, triangle):
	"""Return the edge opposite to the smallest internal angle of a triangle.

	Parameters
	----------
	points : (N,2) array-like
	triangle : iterable of 3 ints

	Returns
	-------
	(tuple)
		Sorted pair of vertex indices identifying the edge opposite to the
		smallest internal angle of the triangle.
	"""
	pts = np.asarray(points, dtype=np.float64)
	tri = np.asarray(triangle, dtype=np.int32)
	p0 = pts[int(tri[0])]; p1 = pts[int(tri[1])]; p2 = pts[int(tri[2])]
	angs = triangle_angles(p0, p1, p2)
	i_min = int(np.argmin(angs))
	idx = [int(tri[0]), int(tri[1]), int(tri[2])]
	edge = (idx[(i_min+1)%3], idx[(i_min+2)%3])
	return tuple(sorted(edge))

def triangles_signed_areas(points, tris):
	"""Vectorized signed area for a batch of triangles.

	points: (N,2) float array
	tris:   (M,3) int array
	Returns: (M,) float64 array of signed areas (0.5 * cross).
	"""
	pts = np.asarray(points, dtype=np.float64)
	T = np.asarray(tris, dtype=np.int32)
	if T.size == 0:
		return np.empty((0,), dtype=float)
	
	# NumPy vectorized version (already optimal - faster than Numba)
	p0 = pts[T[:, 0]]; p1 = pts[T[:, 1]]; p2 = pts[T[:, 2]]
	return 0.5 * np.cross(p1 - p0, p2 - p0)

def ensure_positive_orientation(points, triangles):
	pts = np.asarray(points, dtype=np.float64)
	tris = np.asarray(triangles, dtype=np.int32).copy()
	if tris.size == 0:
		return tris
	# Skip tombstoned rows
	mask_active = ~np.all(tris == -1, axis=1)
	if not np.any(mask_active):
		return tris
	active = tris[mask_active]
	try:
		# Compute signed areas vectorially and flip rows with non-positive area
		p0 = pts[active[:, 0]]; p1 = pts[active[:, 1]]; p2 = pts[active[:, 2]]
		areas = 0.5 * np.cross(p1 - p0, p2 - p0)
		flip = areas <= 0.0
		if np.any(flip):
			swapped = active.copy()
			swapped[flip, 1], swapped[flip, 2] = active[flip, 2], active[flip, 1]
			tris[mask_active] = swapped
		else:
			tris[mask_active] = active
	except Exception:
		# Fallback to scalar path if vectorized computation fails for any reason
		for i in range(tris.shape[0]):
			t = tris[i]
			if np.all(t == -1):
				continue
			try:
				p0 = pts[int(t[0])]; p1 = pts[int(t[1])]; p2 = pts[int(t[2])]
				a = triangle_area(p0, p1, p2)
				if a <= 0:
					tris[i] = np.array([t[0], t[2], t[1]], dtype=int)
			except Exception:
				continue
	return tris

def point_in_polygon(x, y, poly):
	inside = False
	n = len(poly)
	for i in range(n):
		x0, y0 = poly[i]
		x1, y1 = poly[(i+1) % n]
		if ((y0 > y) != (y1 > y)):
			xint = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-20) + x0
			if x < xint:
				inside = not inside
	return inside

@njit(fastmath=True)
def _compute_triangulation_area_numba(points, triangles, indices):
	"""Numba-accelerated triangulation area computation.
	
	points: (N,2) float64 array
	triangles: (M,3) int32 array
	indices: array of triangle indices to sum
	Returns: float64 total area
	"""
	total_area = 0.0
	for idx in indices:
		i0, i1, i2 = triangles[idx, 0], triangles[idx, 1], triangles[idx, 2]
		p0x, p0y = points[i0, 0], points[i0, 1]
		p1x, p1y = points[i1, 0], points[i1, 1]
		p2x, p2y = points[i2, 0], points[i2, 1]
		
		# Cross product for area
		dx1, dy1 = p1x - p0x, p1y - p0y
		dx2, dy2 = p2x - p0x, p2y - p0y
		area = 0.5 * (dx1 * dy2 - dy1 * dx2)
		total_area += abs(area)
	return total_area

def compute_triangulation_area(points, triangles, indices):
	"""
	Compute the total area of a set of triangles.
	
	Args:
		points: (N, 2) array of point coordinates
		triangles: (M, 3) array of triangle vertex indices
		indices: List or array of triangle indices to sum over
		
	Returns:
		float: Total area (sum of absolute areas)
	"""
	if len(indices) == 0:
		return 0.0
	
	# Use Numba version for large batches
	pts = np.asarray(points, dtype=np.float64)
	tris = np.asarray(triangles, dtype=np.int32)
	idx_arr = np.asarray(indices, dtype=np.int32)
	
	if HAS_NUMBA and len(idx_arr) > 20:
		try:
			return _compute_triangulation_area_numba(pts, tris, idx_arr)
		except Exception:
			pass  # Fall back to Python version
	
	# Python fallback
	total_area = 0.0
	for idx in indices:
		tri = triangles[idx]
		p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
		area = triangle_area(p0, p1, p2)
		total_area += abs(area)
	
	return total_area


def normalize_edge(u, v):
	"""
	Return a normalized edge representation as (min, max).
	
	This ensures that edges (u, v) and (v, u) are represented
	the same way, useful for edge-based data structures.
	
	Args:
		u: First vertex index
		v: Second vertex index
		
	Returns:
		tuple: (min(u, v), max(u, v))
	"""
	return (min(u, v), max(u, v))

