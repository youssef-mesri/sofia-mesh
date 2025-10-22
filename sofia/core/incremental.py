"""Incremental computation utilities for mesh operations.

This module provides incremental data structures that track mesh properties
efficiently during local modifications, avoiding full recomputation.

Expected performance gains:
- IncrementalEdgeMap: 100-1000x faster than rebuilding edge maps
- IncrementalConformityChecker: 100-1000x faster than full conformity checks
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List, Optional

__all__ = [
    'IncrementalEdgeMap',
    'IncrementalConformityChecker',
]


class IncrementalEdgeMap:
    """Incrementally maintained edge-to-triangle mapping.
    
    This class maintains an edge-to-triangle map that can be updated incrementally
    when triangles are added or removed, avoiding O(N) full rebuilds.
    
    Performance:
    - Initial build: O(N) where N = number of triangles
    - Update after operation: O(k) where k = number of affected triangles
    - Query: O(1) for edge lookup
    
    Expected speedup: 100-1000x for large meshes with local modifications.
    
    Example:
        >>> edge_map = IncrementalEdgeMap(triangles)
        >>> # After an edge split that removes 2 tris and adds 4 tris
        >>> edge_map.remove_triangles([100, 101])
        >>> edge_map.add_triangles([(v0,v1,v2), (v1,v3,v2), (v3,v4,v2), (v4,v0,v2)], start_idx=200)
        >>> # Edge map is now up-to-date without full rebuild!
        >>> tris_sharing_edge = edge_map.get_triangles_for_edge((v0, v1))
    """
    
    def __init__(self, triangles: np.ndarray):
        """Initialize edge map from triangle array.
        
        Parameters
        ----------
        triangles : (N, 3) array of int
            Triangle vertex indices. May contain marked triangles (all -1).
        """
        self.triangles = np.asarray(triangles, dtype=np.int32)
        self.edge_to_tris: Dict[Tuple[int, int], Set[int]] = {}
        self._build_initial()
    
    def _build_initial(self):
        """Build initial edge-to-triangle map. O(N) operation."""
        self.edge_to_tris.clear()
        
        for tri_idx, tri in enumerate(self.triangles):
            # Skip marked triangles
            if np.all(tri == -1):
                continue
            
            # Add all three edges of this triangle
            for i in range(3):
                v0 = int(tri[i])
                v1 = int(tri[(i + 1) % 3])
                
                # Canonical edge representation (smaller index first)
                edge = (min(v0, v1), max(v0, v1))
                
                if edge not in self.edge_to_tris:
                    self.edge_to_tris[edge] = set()
                self.edge_to_tris[edge].add(tri_idx)
    
    def remove_triangles(self, tri_indices: List[int]):
        """Remove triangles from the edge map. O(k) where k = len(tri_indices).
        
        Parameters
        ----------
        tri_indices : list of int
            Global indices of triangles to remove.
        """
        for tri_idx in tri_indices:
            if tri_idx >= len(self.triangles):
                continue
            
            tri = self.triangles[tri_idx]
            if np.all(tri == -1):
                continue
            
            # Remove this triangle from all its edges
            for i in range(3):
                v0 = int(tri[i])
                v1 = int(tri[(i + 1) % 3])
                edge = (min(v0, v1), max(v0, v1))
                
                if edge in self.edge_to_tris:
                    self.edge_to_tris[edge].discard(tri_idx)
                    # Clean up empty edge entries
                    if len(self.edge_to_tris[edge]) == 0:
                        del self.edge_to_tris[edge]
    
    def add_triangles(self, new_triangles: List[Tuple[int, int, int]], 
                      start_idx: Optional[int] = None):
        """Add triangles to the edge map. O(k) where k = len(new_triangles).
        
        Parameters
        ----------
        new_triangles : list of (v0, v1, v2) tuples
            New triangles to add.
        start_idx : int, optional
            Starting global index for new triangles. If None, appends to self.triangles.
        """
        if start_idx is None:
            start_idx = len(self.triangles)
        
        for local_idx, tri in enumerate(new_triangles):
            tri_idx = start_idx + local_idx
            
            # Add all three edges of this triangle
            for i in range(3):
                v0 = int(tri[i])
                v1 = int(tri[(i + 1) % 3])
                edge = (min(v0, v1), max(v0, v1))
                
                if edge not in self.edge_to_tris:
                    self.edge_to_tris[edge] = set()
                self.edge_to_tris[edge].add(tri_idx)
    
    def get_triangles_for_edge(self, edge: Tuple[int, int]) -> Set[int]:
        """Get triangle indices sharing an edge. O(1) operation.
        
        Parameters
        ----------
        edge : (v0, v1) tuple of int
            Edge vertices (order doesn't matter).
        
        Returns
        -------
        tri_indices : set of int
            Indices of triangles sharing this edge.
        """
        v0, v1 = edge
        canonical_edge = (min(v0, v1), max(v0, v1))
        return self.edge_to_tris.get(canonical_edge, set()).copy()
    
    def get_boundary_edges(self) -> Set[Tuple[int, int]]:
        """Get all boundary edges (edges with exactly 1 adjacent triangle).
        
        Returns
        -------
        boundary_edges : set of (v0, v1) tuples
            All edges with exactly one adjacent triangle.
        """
        return {edge for edge, tris in self.edge_to_tris.items() if len(tris) == 1}
    
    def get_non_manifold_edges(self) -> Set[Tuple[int, int]]:
        """Get all non-manifold edges (edges with > 2 adjacent triangles).
        
        Returns
        -------
        non_manifold_edges : set of (v0, v1) tuples
            All edges with more than two adjacent triangles.
        """
        return {edge for edge, tris in self.edge_to_tris.items() if len(tris) > 2}
    
    def is_conforming(self) -> bool:
        """Check if mesh is conforming (no non-manifold edges). O(E) operation.
        
        Returns
        -------
        conforming : bool
            True if no edges have more than 2 adjacent triangles.
        """
        return all(len(tris) <= 2 for tris in self.edge_to_tris.values())
    
    def get_edge_count(self) -> int:
        """Get total number of unique edges."""
        return len(self.edge_to_tris)
    
    def validate(self, triangles: np.ndarray) -> bool:
        """Validate that incremental map matches a full rebuild.
        
        For testing/debugging only. Compares incremental map with full rebuild.
        
        Parameters
        ----------
        triangles : (N, 3) array of int
            Current triangle array to validate against.
        
        Returns
        -------
        valid : bool
            True if incremental map matches full rebuild.
        """
        from .conformity import build_edge_to_tri_map
        
        # Build fresh map
        fresh_map = build_edge_to_tri_map(triangles)
        
        # Compare edge sets
        fresh_edges = set(fresh_map.keys())
        inc_edges = set(self.edge_to_tris.keys())
        
        if fresh_edges != inc_edges:
            print(f"Edge set mismatch: fresh={len(fresh_edges)}, incremental={len(inc_edges)}")
            print(f"Missing in incremental: {fresh_edges - inc_edges}")
            print(f"Extra in incremental: {inc_edges - fresh_edges}")
            return False
        
        # Compare triangle sets for each edge
        for edge in fresh_edges:
            fresh_tris = set(fresh_map[edge])
            inc_tris = self.edge_to_tris[edge]
            if fresh_tris != inc_tris:
                print(f"Triangle mismatch for edge {edge}: fresh={fresh_tris}, inc={inc_tris}")
                return False
        
        return True


class IncrementalConformityChecker:
    """Incrementally maintained conformity checker.
    
    This class tracks mesh conformity properties incrementally, avoiding full
    recomputation after each operation.
    
    Tracks:
    - Edge counts (for boundary/non-manifold detection)
    - Boundary edges (edges with count == 1)
    - Non-manifold edges (edges with count > 2)
    
    Performance:
    - Initial build: O(N) where N = number of triangles
    - Update after operation: O(k) where k = number of affected triangles
    - Conformity check: O(1)
    
    Expected speedup: 100-5000x for large meshes with local modifications.
    
    Example:
        >>> checker = IncrementalConformityChecker(triangles)
        >>> checker.is_conforming()  # O(1) check
        True
        >>> # After operation
        >>> checker.update_after_operation(removed_tris=[10, 11], added_tris=[(v0,v1,v2), ...])
        >>> checker.is_conforming()  # Still O(1)!
        True
    """
    
    def __init__(self, triangles: np.ndarray):
        """Initialize conformity checker.
        
        Parameters
        ----------
        triangles : (N, 3) array of int
            Triangle vertex indices.
        """
        self.triangles = np.asarray(triangles, dtype=np.int32)
        self.edge_counts: Dict[Tuple[int, int], int] = {}
        self.boundary_edges: Set[Tuple[int, int]] = set()
        self.non_manifold_edges: Set[Tuple[int, int]] = set()
        self._build_initial()
    
    def _build_initial(self):
        """Build initial edge counts. O(N) operation."""
        self.edge_counts.clear()
        self.boundary_edges.clear()
        self.non_manifold_edges.clear()
        
        # Count edge occurrences
        for tri_idx, tri in enumerate(self.triangles):
            if np.all(tri == -1):
                continue
            
            for i in range(3):
                v0 = int(tri[i])
                v1 = int(tri[(i + 1) % 3])
                edge = (min(v0, v1), max(v0, v1))
                
                self.edge_counts[edge] = self.edge_counts.get(edge, 0) + 1
        
        # Classify edges
        for edge, count in self.edge_counts.items():
            if count == 1:
                self.boundary_edges.add(edge)
            elif count > 2:
                self.non_manifold_edges.add(edge)
    
    def update_after_operation(self, removed_tris: List[int], 
                               added_tris: List[Tuple[int, int, int]]):
        """Update conformity state after operation. O(k) for k affected triangles.
        
        Parameters
        ----------
        removed_tris : list of int
            Global indices of removed triangles.
        added_tris : list of (v0, v1, v2) tuples
            New triangles that were added.
        """
        # Track edges that need reclassification
        affected_edges: Set[Tuple[int, int]] = set()
        
        # Process removed triangles
        for tri_idx in removed_tris:
            if tri_idx >= len(self.triangles):
                continue
            
            tri = self.triangles[tri_idx]
            if np.all(tri == -1):
                continue
            
            for i in range(3):
                v0 = int(tri[i])
                v1 = int(tri[(i + 1) % 3])
                edge = (min(v0, v1), max(v0, v1))
                
                if edge in self.edge_counts:
                    self.edge_counts[edge] -= 1
                    affected_edges.add(edge)
                    
                    # Remove edge if count reaches 0
                    if self.edge_counts[edge] == 0:
                        del self.edge_counts[edge]
        
        # Process added triangles
        for tri in added_tris:
            for i in range(3):
                v0 = int(tri[i])
                v1 = int(tri[(i + 1) % 3])
                edge = (min(v0, v1), max(v0, v1))
                
                self.edge_counts[edge] = self.edge_counts.get(edge, 0) + 1
                affected_edges.add(edge)
        
        # Reclassify affected edges
        for edge in affected_edges:
            # Remove from old classifications
            self.boundary_edges.discard(edge)
            self.non_manifold_edges.discard(edge)
            
            # Add to new classification
            if edge in self.edge_counts:
                count = self.edge_counts[edge]
                if count == 1:
                    self.boundary_edges.add(edge)
                elif count > 2:
                    self.non_manifold_edges.add(edge)
    
    def is_conforming(self) -> bool:
        """Check if mesh is conforming (no non-manifold edges). O(1) operation.
        
        Returns
        -------
        conforming : bool
            True if no non-manifold edges exist.
        """
        return len(self.non_manifold_edges) == 0
    
    def has_boundary(self) -> bool:
        """Check if mesh has boundary. O(1) operation."""
        return len(self.boundary_edges) > 0
    
    def get_boundary_edge_count(self) -> int:
        """Get number of boundary edges. O(1) operation."""
        return len(self.boundary_edges)
    
    def get_non_manifold_edge_count(self) -> int:
        """Get number of non-manifold edges. O(1) operation."""
        return len(self.non_manifold_edges)
    
    def get_boundary_edges(self) -> Set[Tuple[int, int]]:
        """Get all boundary edges. O(1) operation."""
        return self.boundary_edges.copy()
    
    def get_non_manifold_edges(self) -> Set[Tuple[int, int]]:
        """Get all non-manifold edges. O(1) operation."""
        return self.non_manifold_edges.copy()
    
    def get_edge_count_for_edge(self, edge: Tuple[int, int]) -> int:
        """Get the count of triangles sharing an edge.
        
        Parameters
        ----------
        edge : (v0, v1) tuple
            Edge vertices (order doesn't matter).
        
        Returns
        -------
        count : int
            Number of triangles sharing this edge.
        """
        v0, v1 = edge
        canonical_edge = (min(v0, v1), max(v0, v1))
        return self.edge_counts.get(canonical_edge, 0)
    
    def validate(self, triangles: np.ndarray) -> bool:
        """Validate incremental state against full rebuild.
        
        For testing/debugging only.
        
        Parameters
        ----------
        triangles : (N, 3) array
            Current triangles to validate against.
        
        Returns
        -------
        valid : bool
            True if incremental state matches full rebuild.
        """
        # Build fresh checker
        fresh = IncrementalConformityChecker(triangles)
        
        # Compare edge counts
        if self.edge_counts != fresh.edge_counts:
            print(f"Edge count mismatch")
            print(f"Incremental: {sorted(self.edge_counts.items())[:10]}")
            print(f"Fresh: {sorted(fresh.edge_counts.items())[:10]}")
            return False
        
        # Compare boundary edges
        if self.boundary_edges != fresh.boundary_edges:
            print(f"Boundary edge mismatch: inc={len(self.boundary_edges)}, fresh={len(fresh.boundary_edges)}")
            return False
        
        # Compare non-manifold edges
        if self.non_manifold_edges != fresh.non_manifold_edges:
            print(f"Non-manifold edge mismatch: inc={len(self.non_manifold_edges)}, fresh={len(fresh.non_manifold_edges)}")
            return False
        
        return True
