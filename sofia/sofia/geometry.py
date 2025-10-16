"""Geometry primitives and tolerances (package internal copy).

Original module relocated from repository root. This is now the canonical
implementation; a root-level shim will re-export and emit a deprecation
warning until the next major release.
"""
from __future__ import annotations
import math
import numpy as np
from .constants import EPS_AREA, EPS_MIN_ANGLE_DEG, EPS_IMPROVEMENT

__all__ = [
	'triangle_area','triangle_angles','ensure_positive_orientation','point_in_polygon',
	'triangles_min_angles','triangles_signed_areas','opposite_edge_of_smallest_angle'
]

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
	o1 = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
	o2 = (b[:,0]-a[:,0])*(d[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(d[:,0]-a[:,0])
	o3 = (d[:,0]-c[:,0])*(a[:,1]-c[:,1]) - (d[:,1]-c[:,1])*(a[:,0]-c[:,0])
	o4 = (d[:,0]-c[:,0])*(b[:,1]-c[:,1]) - (d[:,1]-c[:,1])*(b[:,0]-c[:,0])
	# exclude shared endpoints: compare coordinates exactly (points are float arrays but originate from same array)
	shared = np.all(a == c, axis=1) | np.all(a == d, axis=1) | np.all(b == c, axis=1) | np.all(b == d, axis=1)
	colinear = (o1 == 0) & (o2 == 0) & (o3 == 0) & (o4 == 0)
	crosses = (o1*o2 < 0) & (o3*o4 < 0) & (~colinear) & (~shared)
	return crosses

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

