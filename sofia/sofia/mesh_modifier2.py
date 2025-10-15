"""Core patch-based mesh editor (demos moved to demos/mesh_editor_demos)."""

import logging
import math
import os
import shutil
from collections import defaultdict

import numpy as np
import os as _os
import matplotlib as _mpl
# Ensure a non-interactive backend in headless environments before importing pyplot
if not _os.environ.get('MPLBACKEND'):
    try:
        _mpl.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull

from .geometry import (triangle_area, triangle_angles, triangles_min_angles, ensure_positive_orientation,
                      point_in_polygon, opposite_edge_of_smallest_angle)
from .quality import mesh_min_angle
from .constants import EPS_AREA, EPS_MIN_ANGLE_DEG, EPS_IMPROVEMENT
from .conformity import (build_edge_to_tri_map, build_vertex_to_tri_map, check_mesh_conformity,
                        boundary_edges_from_map, is_boundary_vertex_from_maps, is_active_triangle)
from .stats import OpStats, print_stats as _print_stats
from .triangulation import retriangulate_patch_strict, retriangulate_star, optimal_star_triangulation
from .helpers import (boundary_cycle_from_incident_tris, patch_nodes_for_triangles,
                      boundary_polygons_from_patch, select_outer_polygon, detect_empty_pockets,
                      extract_patch_nodes)
from .logging_utils import get_logger

try:  # optional dependency
    import imageio  # noqa: F401
except Exception:  # pragma: no cover
    imageio = None

logger = get_logger('sofia.editor')


## mesh_min_angle moved to quality.py

# point_in_polygon now imported from geometry


# -----------------------
# Strict local retriangulation
# -----------------------
from .triangulation import retriangulate_patch_strict, retriangulate_star, optimal_star_triangulation

# -----------------------
# Patch helpers
# -----------------------
## moved to helpers: patch_nodes_for_triangles, boundary_polygons_from_patch, select_outer_polygon,
## detect_empty_pockets, extract_patch_nodes

## moved to geometry.opposite_edge_of_smallest_angle


from collections import defaultdict
# -----------------------
# Editor class (with conformity check before changes)
# -----------------------
class PatchBasedMeshEditor:
    def __init__(self, points, triangles, debug: bool = False,
                 simulate_compaction_on_commit: bool = False,
                 reject_boundary_loop_increase: bool = False,
                 reject_any_boundary_loops: bool = False,
                 reject_crossing_edges: bool = False,
                 virtual_boundary_mode: bool = False,
                 enable_polygon_simplification: bool = True,
                 simplify_log_every: int = 50,
                 enforce_split_quality: bool = True,
                 enforce_remove_quality: bool = True,
                 boundary_remove_config=None):
        """Primary mesh editor.

        Parameters
        ----------
        points : (N,2) float array-like
        triangles : (M,3) int array-like
            May contain tombstoned rows ([-1,-1,-1]) although initial input normally shouldn't.
        debug : bool
            Enable verbose debug logging of map mutations.
        simulate_compaction_on_commit : bool
            If True, each structural op simulates compaction before commit and rejects on failure.
        reject_boundary_loop_increase : bool
            If True, simulation rejects ops that increase number of boundary loops.
        reject_any_boundary_loops : bool
            If True, simulation rejects ops that produce any boundary loops.
        reject_crossing_edges : bool
            If True, simulation rejects ops that introduce crossing edges post-compaction.
        """
        # Standardized editor logger hierarchy
        self.logger = get_logger(f'sofia.editor.{self.__class__.__name__}')
        self.debug = debug
        # Assign via property setters to enforce canonical storage once at load
        self._points = None  # backing fields
        self._triangles = None
        self._maps_maybe_dirty = True  # adjacency maps need a build at start
        self.points = points
        self.triangles = triangles
        # Amortization knobs (reduce frequency of heavy routines)
        # - Simulated compaction check cooldown (ops): run at most every N ops unless change magnitude is large
        self.check_cooldown_ops = 8
        self._check_cooldown = 0
        self.check_risk_threshold = 12  # force check if tombstones+appends >= threshold for a single op
        # - Compaction thresholds
        self.op_counter = 0
        self.last_compact_op_index = -10**9
        self.compaction_min_interval_ops = 50
        self.compaction_tombstone_threshold = 256
        self.compaction_ratio_threshold = 0.15
        # Only apply ratio-based compaction when mesh is at least this large
        self.compaction_ratio_min_total = 100
        self._pending_tombstones = 0
        self._pending_appends = 0
        # Safety: disable automatic compaction during iterative algorithms unless explicitly enabled
        # Compaction can reindex triangles and invalidate cached indices in drivers.
        self.enable_auto_compaction = False
        # Policy flags
        self.simulate_compaction_on_commit = simulate_compaction_on_commit
        self.reject_boundary_loop_increase = reject_boundary_loop_increase
        self.reject_any_boundary_loops = reject_any_boundary_loops
        self.reject_crossing_edges = reject_crossing_edges
        # Feature flag: attempt lightweight polygon cycle simplification before falling back during node removal.
        self.enable_polygon_simplification = enable_polygon_simplification
        # Sampling frequency for simplification logs (log first few + every Nth thereafter)
        self.simplify_log_every = max(1, int(simplify_log_every))
        # Policy: enforce non-worsening quality for split_edge (improvement) vs relax for refinement
        self.enforce_split_quality = bool(enforce_split_quality)
        # Policy: enforce non-worsening quality for remove_node_with_patch; can be relaxed to allow removal even if quality worsens
        self.enforce_remove_quality = bool(enforce_remove_quality)
        # Boundary removal strategy preferences
        try:
            from .config import BoundaryRemoveConfig
            self.boundary_remove_config = boundary_remove_config or BoundaryRemoveConfig()
        except Exception:
            self.boundary_remove_config = None
        # Treat boundary as interior topologically (no flips over boundary edges).
        # This does not modify triangles; it only changes certain operation guards/fallbacks.
        self.virtual_boundary_mode = bool(virtual_boundary_mode)
        # Unified operation stats registry
        self._op_stats = defaultdict(OpStats)
        # Backwards compat alias (remove_node_stats was previously used in tests)
        self.remove_node_stats = self._op_stats['remove_node']
        # Sanity guard on canonical storage to avoid silent re-copies later
        self._assert_canonical()
        # Derived adjacency maps
        self._update_maps()

    # Canonical storage properties
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must have shape (N,2)")
        self._points = np.ascontiguousarray(arr)

    @property
    def triangles(self):
        return self._triangles

    @triangles.setter
    def triangles(self, value):
        # Coerce to int32 C and ensure positive orientation against current points when available
        arr = np.asarray(value, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("triangles must have shape (M,3)")
        try:
            pts = self._points if getattr(self, '_points', None) is not None else None
            if pts is not None:
                arr = ensure_positive_orientation(pts, arr)
        except Exception:
            # Orientation enforcement is best-effort; keep arr as-is on failure
            pass
        self._triangles = np.ascontiguousarray(arr.astype(np.int32, copy=False))
        # Mark maps dirty: topology may have changed; callers maintaining maps incrementally can mark clean later
        self._maps_maybe_dirty = True

    # --- Canonical storage guards (dtype + contiguity) ---
    def _assert_canonical(self):
        """Validate that internal arrays conform to canonical storage:
        - points: float64, shape (N,2), C-contiguous
        - triangles: int32, shape (M,3), C-contiguous
        Raises ValueError if not satisfied. This avoids repeated hidden copies.
        """
        pts = self.points
        tris = self.triangles
        if pts.dtype != np.float64 or pts.ndim != 2 or pts.shape[1] != 2 or not pts.flags.c_contiguous:
            raise ValueError("points must be float64 (N,2) and C-contiguous; standardize at load time")
        if tris.dtype != np.int32 or tris.ndim != 2 or tris.shape[1] != 3 or not tris.flags.c_contiguous:
            raise ValueError("triangles must be int32 (M,3) and C-contiguous; standardize at load time")

    # --- Stats helpers ---
    def _get_op_stats(self, name: str) -> OpStats:
        return self._op_stats[name]

    def stats_summary(self):
        return {k: v.to_dict() for k, v in self._op_stats.items()}

    def print_stats(self, pretty: bool = True, file=None):
        """Delegate to `stats.print_stats` for presentation."""
        _print_stats(self.stats_summary(), file=file, pretty=pretty)

    # --- Timing helpers ---
    def _record_time(self, op_name: str, duration: float):
        stats = self._op_stats[op_name]
        stats.time_total += duration
        if duration > stats.time_max:
            stats.time_max = duration
        if stats.time_min == 0.0 or duration < stats.time_min:
            stats.time_min = duration

    # --- Stats maintenance ---
    def reset_stats(self, drop_ops: bool = False):
        """Reset operation statistics counters and timings.

        Parameters
        ----------
        drop_ops : bool, default False
            If True, completely clears the internal stats registry so only
            future operations recreate entries. If False, keeps existing op
            keys but zeroes all counters and timing fields.

        Notes
        -----
        The legacy alias `remove_node_stats` is refreshed so external code
        holding a reference sees the updated (zeroed) object.
        """
        if drop_ops:
            self._op_stats.clear()
        else:
            for s in self._op_stats.values():
                s.attempts = 0
                s.success = 0
                s.fail = 0
                s.quality_rejects = 0
                s.simulation_rejects = 0
                s.fallback_used = 0
                s.star_success = 0
                s.simplify_attempted = 0
                s.simplify_helped = 0
                s.pocket_quad_attempts = 0
                s.pocket_quad_success = 0
                s.pocket_steiner_attempts = 0
                s.pocket_steiner_success = 0
                s.pocket_earclip_attempts = 0
                s.pocket_earclip_success = 0
                s.time_total = 0.0
                s.time_max = 0.0
                s.time_min = 0.0
        # Refresh alias to maintain backwards compatibility
        self.remove_node_stats = self._op_stats['remove_node']

    # ...existing methods follow...
    def compact_triangle_indices(self):
        """
        Reconstruit la numérotation des triangles (0...N-1) et les maps d'adjacence.
        À appeler après toutes les modifications locales pour garantir la cohérence.

        Side-effects:
        - Stores fast, vectorizable mappings for vertices:
          * `last_compaction_old_to_new` -> np.ndarray (N_old,), int32, -1 for removed, else new index.
          * `last_compaction_new_to_old` -> np.ndarray (N_new,), int32, mapping new index -> old index.

        Returns
        -------
        np.ndarray
            The old-to-new index mapping array (shape (N_old,), int32), with -1 for removed vertices.
        """
        # Ensure arrays
        tris = np.ascontiguousarray(np.asarray(self.triangles, dtype=np.int32))
        pts = np.ascontiguousarray(np.asarray(self.points, dtype=np.float64))

        # 1) keep only active triangles
        mask = ~np.all(tris == -1, axis=1)
        active_tris = tris[mask]
        old_ntri = tris.shape[0]
        kept_ntri = active_tris.shape[0]

        # 2) determine used vertex indices and build vectorized mappings
        n_old = pts.shape[0]
        if kept_ntri == 0:
            used_vertices = []
            old_to_new_arr = np.full(n_old, -1, dtype=np.int32)
            new_to_old_arr = np.empty((0,), dtype=np.int32)
            new_points = np.empty((0, 2), dtype=np.float64)
            remapped_tris = np.empty((0, 3), dtype=np.int32)
        else:
            used_vertices = sorted({int(v) for tri in active_tris for v in tri})
            old_to_new_arr = np.full(n_old, -1, dtype=np.int32)
            for new_idx, old in enumerate(used_vertices):
                old_to_new_arr[int(old)] = int(new_idx)
            new_to_old_arr = np.asarray(used_vertices, dtype=np.int32)
            # 3) build new points array containing only used vertices (order corresponds to new indices)
            new_points = pts[new_to_old_arr]
            # 4) remap triangles to new vertex indices (vectorized)
            remapped_tris = old_to_new_arr[active_tris]

        # Commit
        self.points = np.ascontiguousarray(new_points)
        self.triangles = np.ascontiguousarray(remapped_tris)
        # Expose mapping info for downstream validation/tests (robust vs FP coordinate checks)
        self.last_compaction_old_to_new = old_to_new_arr
        self.last_compaction_new_to_old = new_to_old_arr if kept_ntri != 0 else np.empty((0,), dtype=np.int32)

        # 5) rebuild maps
        self._update_maps(force=True)
        # Log summary with valid counts
        self.logger.info(
            "[INFO] Triangles and vertices compacted: triangles %d->%d, vertices %d->%d",
            old_ntri, kept_ntri, n_old, self.points.shape[0])
        return self.last_compaction_old_to_new

    def _simulate_compaction_and_check(self, cand_points, cand_tris, eps_area=EPS_AREA):
        from .conformity import simulate_compaction_and_check
        return simulate_compaction_and_check(
            cand_points, cand_tris, eps_area=eps_area,
            reject_boundary_loop_increase=getattr(self,'reject_boundary_loop_increase',False),
            reject_any_boundary_loops=getattr(self,'reject_any_boundary_loops',False),
            reject_crossing_edges=getattr(self,'reject_crossing_edges',False)
        )
    # Removed legacy duplicate __init__ (unified above)

    # Helpers for dealing with tombstoned triangles
    def active_tri_indices(self):
        return [i for i, t in enumerate(self.triangles) if not np.all(np.array(t) == -1)]

    def active_triangles(self):
        idx = self.active_tri_indices()
        return self.triangles[idx]

    def has_tombstones(self) -> bool:
        """Return True if any triangle row is tombstoned (all -1).

        Centralized helper to gate compaction and expensive rebuilds.
        Safe against empty arrays.
        """
        try:
            tris = np.asarray(self.triangles)
            return bool(tris.size and np.any(np.all(tris == -1, axis=1)))
        except Exception:
            return False

    def _update_maps(self, force: bool = False):
        """Met à jour edge_map et v_map après modification du maillage.
        Rebuild only when marked dirty, unless force=True.
        """
        if not force and not getattr(self, '_maps_maybe_dirty', True):
            return
        # Ensure arrays remain canonical
        self._assert_canonical()
        self.edge_map = build_edge_to_tri_map(self.triangles)
        self.v_map = build_vertex_to_tri_map(self.triangles)
        self._maps_maybe_dirty = False

    # -----------------------
    # Amortization helpers
    # -----------------------
    def _should_run_simulation_check(self, change_magnitude: int) -> bool:
        """Decide whether to run a heavy simulated compaction check this op.

        Heuristic:
        - If a single op changes many triangles (tombstones + appends >= risk threshold), force check.
        - Otherwise, run check only when cooldown reaches zero, then reset cooldown.
        """
        try:
            # Force check for large local edits
            if int(change_magnitude) >= int(getattr(self, 'check_risk_threshold', 12)):
                self._check_cooldown = int(getattr(self, 'check_cooldown_ops', 8))
                return True
            # Periodic sampling check
            cd = int(getattr(self, '_check_cooldown', 0))
            if cd <= 0:
                self._check_cooldown = int(getattr(self, 'check_cooldown_ops', 8))
                return True
            else:
                self._check_cooldown = cd - 1
                return False
        except Exception:
            # Be conservative if anything goes wrong
            self._check_cooldown = int(getattr(self, 'check_cooldown_ops', 8))
            return True

    def _on_op_committed(self, tombstoned: int = 0, appended: int = 0):
        """Record a topological mutation commit and maybe compact based on thresholds.

        This should be called by operations after they finish tombstoning/appending triangles.
        """
        try:
            self.op_counter += 1
            self._pending_tombstones += int(tombstoned)
            self._pending_appends += int(appended)
            self._maybe_compact()
        except Exception:
            # Non-fatal; amortization is best-effort
            pass

    def _maybe_compact(self, reason: str = None, force: bool = False) -> bool:
        """Compact triangles/vertices if thresholds suggest it's worth it.

        Returns True if compaction was performed.
        """
        if force:
            self.compact_triangle_indices()
            self._pending_tombstones = 0
            self._pending_appends = 0
            self.last_compact_op_index = int(self.op_counter)
            return True
        # Guard: only compact automatically if explicitly enabled (default is False)
        if not getattr(self, 'enable_auto_compaction', False):
            return False
        try:
            min_interval = int(getattr(self, 'compaction_min_interval_ops', 50))
            since = int(self.op_counter) - int(getattr(self, 'last_compact_op_index', 0))
            if since < min_interval:
                return False
            # Thresholds based on absolute tombstones or ratio of tombstones to total triangles
            tomb = int(getattr(self, '_pending_tombstones', 0))
            total = int(len(self.triangles)) if hasattr(self, 'triangles') else 0
            abs_thr = int(getattr(self, 'compaction_tombstone_threshold', 256))
            ratio_thr = float(getattr(self, 'compaction_ratio_threshold', 0.15))
            min_total = int(getattr(self, 'compaction_ratio_min_total', 100))
            ratio_ok = (total >= min_total and (tomb / max(1, total)) >= ratio_thr)
            if tomb >= abs_thr or ratio_ok:
                # Perform compaction
                self.compact_triangle_indices()
                self._pending_tombstones = 0
                self._pending_appends = 0
                self.last_compact_op_index = int(self.op_counter)
                if self.debug:
                    self.logger.debug("[amortize] compaction triggered tomb=%d total=%d since=%d reason=%s", tomb, total, since, reason or '')
                return True
        except Exception as e:
            if self.debug:
                self.logger.debug("[amortize] maybe_compact error: %s", e)
        return False

    def _extract_ordered_boundary_loops(self):
        """
        Return a list of ordered boundary loops (each a list of vertex indices) for the current compacted mesh.
        Assumes `self.edge_map` and `self.triangles` are up-to-date and that the mesh is compacted
        (no tombstoned triangles and vertex indices contiguous). Works even if there are multiple loops.
        """
        # build list of boundary edges
        edge_map = self.edge_map
        boundary_edges = [e for e, s in edge_map.items() if len(s) == 1]
        if not boundary_edges:
            return []
        # adjacency for walk
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        loops = []
        visited_edges = set()
        for e in boundary_edges:
            if e in visited_edges:
                continue
            a, b = e
            # orient edge as (a->b) and walk
            loop = [a, b]
            visited_edges.add(e)
            prev, curr = a, b
            while True:
                neighs = adj.get(curr, [])
                next_v = None
                for nb in neighs:
                    if nb == prev:
                        continue
                    next_v = nb
                    break
                if next_v is None:
                    break
                edge_key = tuple(sorted((curr, next_v)))
                if edge_key in visited_edges:
                    # closed loop if next_v equals start
                    if next_v == loop[0]:
                        loop.append(next_v)
                    break
                loop.append(next_v)
                visited_edges.add(edge_key)
                prev, curr = curr, next_v
                if len(loop) > 10000:
                    break
            if len(loop) >= 3 and loop[0] == loop[-1]:
                loops.append(loop[:-1])
        return loops

    def fill_boundary_loops(self, max_loop_len=50, reject_if_fail=False, verbose=False):
        """
        Compact the mesh, detect boundary loops, and attempt to fill each loop using `try_fill_pocket`.
        - max_loop_len: maximum number of vertices in a loop to attempt filling (avoid huge non-triangular holes)
        - reject_if_fail: if True, return False if any fill attempt fails; otherwise continue and report failures.
        Returns (ok: bool, summary: dict) where summary contains 'found_loops', 'filled', 'failed'
        """
        # Compact first to operate on a clean mesh
        # Only compact if tombstoned triangles are present; otherwise skip rebuild
        has_tombstones = bool(len(self.triangles) and np.any(np.all(self.triangles == -1, axis=1)))
        if has_tombstones:
            try:
                self.compact_triangle_indices()
            except Exception:
                # fallback: attempt to update maps
                self._update_maps()
        # Check canonical form post-compaction/update
        self._assert_canonical()
        loops = self._extract_ordered_boundary_loops()
        found = len(loops)
        filled = []
        failed = []
        for loop in loops:
            if len(loop) > max_loop_len:
                failed.append({'loop': loop, 'reason': 'too_large'})
                if reject_if_fail:
                    return False, {'found_loops': found, 'filled': filled, 'failed': failed}
                continue
            ok, details = self.try_fill_pocket(loop)
            if ok:
                filled.append({'loop': loop, 'details': details})
            else:
                failed.append({'loop': loop, 'details': details})
                if reject_if_fail:
                    return False, {'found_loops': found, 'filled': filled, 'failed': failed}
            # update maps after each successful commit
            self._update_maps()
        return True, {'found_loops': found, 'filled': filled, 'failed': failed}

    def _add_triangle_to_maps(self, tri_idx):
        tri = self.triangles[tri_idx]
        if self.debug:
            self.logger.debug(f"Adding triangle {tri_idx}: {tri}")
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3])
            key = tuple(sorted((a, b)))
            self.edge_map.setdefault(key, set()).add(tri_idx)
        for v in tri:
            self.v_map.setdefault(int(v), set()).add(tri_idx)
        if self.debug:
            self.logger.debug(f"edge_map after add: {self.edge_map}")
            self.logger.debug(f"v_map after add: {self.v_map}")
        # Maps updated incrementally -> keep them marked clean
        self._maps_maybe_dirty = False

    def _remove_triangle_from_maps(self, tri_idx):
        if tri_idx >= len(self.triangles):
            if self.debug:
                self.logger.debug(f"Attempt to remove invalid triangle index {tri_idx}")
            return
        tri = self.triangles[tri_idx]
        if self.debug:
            self.logger.debug(f"Removing triangle {tri_idx}: {tri}")
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i+1)%3])
            key = tuple(sorted((a, b)))
            if key in self.edge_map:
                self.edge_map[key].discard(tri_idx)
                if not self.edge_map[key]:
                    del self.edge_map[key]
        for v in tri:
            if v in self.v_map:
                self.v_map[v].discard(tri_idx)
                if not self.v_map[v]:
                    del self.v_map[v]
        if self.debug:
            self.logger.debug(f"edge_map after remove: {self.edge_map}")
            self.logger.debug(f"v_map after remove: {self.v_map}")
        # Maps updated incrementally -> keep them marked clean
        self._maps_maybe_dirty = False

    def global_min_angle(self):
        return mesh_min_angle(self.points, self.triangles)

    def split_edge_delaunay(self, edge, strict_mode='centroid'):
        from .operations import op_split_edge_delaunay
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            return op_split_edge_delaunay(self, edge, strict_mode=strict_mode)
        finally:
            self._record_time('split_delaunay', time.perf_counter() - t0)
    
    def split_edge(self, edge=None):
        from .operations import op_split_edge
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            return op_split_edge(self, edge=edge)
        finally:
            self._record_time('split_midpoint', time.perf_counter() - t0)

    def edge_collapse(self, edge, position: str = 'midpoint'):
        from .operations import op_edge_collapse
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            return op_edge_collapse(self, edge=edge, position=position)
        finally:
            self._record_time('edge_collapse', time.perf_counter() - t0)



    def remove_node_with_patch(self, v_idx, force_strict=False):
        from .operations import op_remove_node_with_patch
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            # Early default area-preservation guard for boundary removals
            try:
                if bool(getattr(self, 'virtual_boundary_mode', False)):
                    cfg = getattr(self, 'boundary_remove_config', None)
                    require_area = bool(getattr(cfg, 'require_area_preservation', False)) if cfg is not None else False
                    if require_area:
                        # Compute cavity (incident triangles) area
                        tri_idx = sorted(set(self.v_map.get(int(v_idx), [])))
                        if tri_idx:
                            from .geometry import triangle_area as _tarea
                            removed_area = 0.0
                            for ti in tri_idx:
                                t = self.triangles[int(ti)]
                                a,b,c = int(t[0]), int(t[1]), int(t[2])
                                p0,p1,p2 = self.points[a], self.points[b], self.points[c]
                                removed_area += abs(_tarea(p0,p1,p2))
                            # Build sanitized boundary polygon and compare areas
                            from .helpers import boundary_polygons_from_patch, select_outer_polygon
                            from .triangulation import polygon_signed_area as _poly_area
                            polys = boundary_polygons_from_patch(self.triangles, tri_idx)
                            cyc = select_outer_polygon(self.points, polys)
                            if cyc:
                                filtered = [int(v) for v in cyc if int(v) != int(v_idx)]
                                if len(filtered) >= 2 and filtered[0] == filtered[-1]:
                                    filtered = filtered[:-1]
                                if len(filtered) >= 3:
                                    poly_area = abs(_poly_area([self.points[int(v)] for v in filtered]))
                                    from .constants import EPS_TINY, EPS_AREA
                                    tol_rel = float(getattr(cfg, 'area_tol_rel', EPS_TINY)) if cfg is not None else EPS_TINY
                                    tol_abs = float(getattr(cfg, 'area_tol_abs_factor', 4.0)) * float(EPS_AREA)
                                    if abs(poly_area - removed_area) > max(tol_abs, tol_rel*max(1.0, removed_area)):
                                        return False, (
                                            f"cavity area changed: poly={poly_area:.6e} cavity={removed_area:.6e}"
                                        ), None
            except Exception:
                pass
            return op_remove_node_with_patch(self, v_idx, force_strict=force_strict)
        finally:
            self._record_time('remove_node', time.perf_counter() - t0)
    
    def remove_node(self, v_idx, *args, **kwargs):
        """
        Backwards-compatible wrapper for remove_node_with_patch.
        """
        return self.remove_node_with_patch(v_idx, *args, **kwargs)
    

    def try_fill_pocket(self, verts, min_tri_area=EPS_AREA, reject_min_angle_deg=None):
        from .operations import op_try_fill_pocket
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            return op_try_fill_pocket(self, verts, min_tri_area=min_tri_area, reject_min_angle_deg=reject_min_angle_deg)
        finally:
            self._record_time('fill_pocket', time.perf_counter() - t0)



    def flip_edge(self, edge):
        from .operations import op_flip_edge
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            return op_flip_edge(self, edge)
        finally:
            self._record_time('flip', time.perf_counter() - t0)

    def add_node(self, point, tri_idx=None):
        from .operations import op_add_node
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            return op_add_node(self, point, tri_idx=tri_idx)
        finally:
            self._record_time('add_node', time.perf_counter() - t0)

    def move_vertices_to_barycenter(self, only_interior: bool = True) -> int:
        """Move vertices to the barycenter of their neighbors.

        This is a thin wrapper around `operations.op_move_vertices_to_barycenter`.

        Parameters
        ----------
        only_interior : bool, default True
            If True, boundary vertices are not moved.

        Returns
        -------
        int
            Number of vertices moved.
        """
        from .operations import op_move_vertices_to_barycenter
        import time
        t0 = time.perf_counter()
        try:
            self._assert_canonical()
            return op_move_vertices_to_barycenter(self, only_interior=only_interior)
        finally:
            self._record_time('smooth_barycenter', time.perf_counter() - t0)

    def locate_point(self, point):
        """
        Retourne l'indice du triangle contenant strictement le point (pas sur le bord).
        Si aucun triangle ne contient le point, retourne None.
        """
        for i in self.active_tri_indices():
            tri = self.triangles[i]
            pts_tri = [self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]]
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            d1 = sign(point, pts_tri[0], pts_tri[1])
            d2 = sign(point, pts_tri[1], pts_tri[2])
            d3 = sign(point, pts_tri[2], pts_tri[0])
            if ((d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)) and (d1 != 0 and d2 != 0 and d3 != 0):
                return i
        return None

    def locate_point_optimized(self, point, start_tri_idx=None):
        """
        Version optimisée : navigation triangle à triangle guidée par le signe.
        Si start_tri_idx n'est pas donné, commence au triangle 0.
        Retourne l'indice du triangle contenant strictement le point, ou None.
        """
        # Reuse existing edge_map if available; otherwise build once
        edge_map = getattr(self, 'edge_map', None)
        if not isinstance(edge_map, dict) or not edge_map:
            edge_map = build_edge_to_tri_map(self.triangles)
        active = self.active_tri_indices()
        if start_tri_idx is None:
            curr = active[0] if active else None
        else:
            curr = start_tri_idx
        visited = set()
        while curr is not None and curr not in visited:
            visited.add(curr)
            tri = self.triangles[curr]
            pts_tri = [self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]]
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            d1 = sign(point, pts_tri[0], pts_tri[1])
            d2 = sign(point, pts_tri[1], pts_tri[2])
            d3 = sign(point, pts_tri[2], pts_tri[0])
            # Strictement à l'intérieur
            if ((d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)) and (d1 != 0 and d2 != 0 and d3 != 0):
                return curr
            # Sinon, déterminer l'arête franchie
            # Si d1 <= 0, point est du mauvais côté de l'arête (v0,v1)
            if d1 <= 0:
                edge = tuple(sorted((tri[0], tri[1])))
            elif d2 <= 0:
                edge = tuple(sorted((tri[1], tri[2])))
            elif d3 <= 0:
                edge = tuple(sorted((tri[2], tri[0])))
            else:
                return None
            # Chercher le triangle voisin
            neighbors = edge_map.get(edge, [])
            next_tri = [t for t in neighbors if t != curr]
            curr = next_tri[0] if next_tri else None
        return None


# -----------------------
# Demo / small unit tests
# -----------------------
def build_random_delaunay(npts=60, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.rand(npts, 2).astype(np.float64)
    corners = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=np.float64)
    pts = np.vstack((pts, corners)).astype(np.float64)
    tri = Delaunay(pts)
    return np.ascontiguousarray(pts, dtype=np.float64), np.ascontiguousarray(tri.simplices.copy().astype(np.int32))

def patch_replace_cavity(points, triangles, cavity_tri_indices, new_triangles, removed_vertex=None, added_points=None):
    """
    Remplace une cavité locale dans le maillage par une nouvelle triangulation.
    - points: tableau (N,2) des sommets
    - triangles: tableau (M,3) des triangles
    - cavity_tri_indices: indices des triangles à supprimer
    - new_triangles: liste de triangles (indices globaux ou locaux à remapper)
    - removed_vertex: indice du sommet à supprimer (optionnel)
    - added_points: liste de nouveaux sommets à ajouter (optionnel)
    Retourne (points_new, triangles_new)
    """
    import numpy as np
    # 1. Retirer les triangles de la cavité
    triangles_new = [t for i, t in enumerate(triangles) if i not in cavity_tri_indices]
    # 2. Gérer la suppression d'un sommet
    points_new = points.copy()
    remap_func = None
    if removed_vertex is not None:
        points_new = np.delete(points_new, removed_vertex, axis=0)
        # Remapper les indices des triangles restants
        def remap(tri):
            return [v-1 if v > removed_vertex else v for v in tri]
        triangles_new = [remap(t) for t in triangles_new]
        remap_func = remap
    # 3. Ajouter de nouveaux sommets si besoin
    if added_points is not None:
        for p in added_points:
            points_new = np.vstack([points_new, p])
    # 4. Ajouter les nouveaux triangles (avec remapping si suppression)
    if remap_func is not None:
        new_triangles = [remap_func(t) for t in new_triangles]
    triangles_new.extend(new_triangles)
    triangles_new = np.array(triangles_new, dtype=int)
    return points_new, triangles_new


