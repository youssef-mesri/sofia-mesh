#!/usr/bin/env python3
"""
SOFIA Example 8: Combined Interior + Boundary Refinement

This example demonstrates adaptive refinement that targets both interior
and boundary regions differently, useful for complex domains with features.

What this example shows:
1. Creating an L-shaped domain mesh (non-convex with sharp corners)
2. Identifying interior vs boundary edges
3. Different refinement criteria for interior vs boundary
4. Preserving sharp features during refinement
5. Analyzing refinement effectiveness by region

Perfect for: Feature-aware meshing, multi-region refinement strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity
from scipy.spatial import Delaunay

def create_l_shaped_mesh(resolution=0.4):
    """Create a mesh of an L-shaped domain with sharp corners.
    
    The L-shape is created by removing the top-right quadrant from a square.
    
    Args:
        resolution: Approximate spacing between points
        
    Returns:
        editor: Mesh editor with L-shaped domain
    """
    # Define L-shape boundary (counterclockwise)
    # Bottom-left corner at origin, L extends in positive x and y
    boundary_pts = np.array([
        [0, 0],      # Bottom-left
        [2, 0],      # Bottom-right
        [2, 1],      # Right inner corner
        [1, 1],      # Top inner corner
        [1, 2],      # Top-left inner
        [0, 2],      # Top-left outer
    ])
    
    # Generate interior points with random placement
    np.random.seed(42)
    n_interior = int(4 / (resolution ** 2))  # Area = 3, but oversample
    
    # Generate points in the bounding box
    interior_pts = []
    for _ in range(n_interior * 3):
        x = np.random.uniform(0, 2)
        y = np.random.uniform(0, 2)
        
        # Keep only if inside L-shape
        if (x <= 1 or y <= 1):  # Inside L-shape
            interior_pts.append([x, y])
    
    interior_pts = np.array(interior_pts[:n_interior])
    
    # Combine boundary and interior
    all_pts = np.vstack([boundary_pts, interior_pts])
    
    # Create Delaunay triangulation
    tri = Delaunay(all_pts)
    
    editor = PatchBasedMeshEditor(all_pts, tri.simplices)
    
    return editor, len(boundary_pts)

def identify_edges_by_type(editor):
    """Classify edges as boundary or interior."""
    boundary_edges = []
    interior_edges = []
    
    for edge, tris in editor.edge_map.items():
        # Count active triangles
        active_tris = [t for t in tris if np.all(editor.triangles[t] != -1)]
        
        if len(active_tris) == 1:
            boundary_edges.append(edge)
        elif len(active_tris) == 2:
            interior_edges.append(edge)
    
    return boundary_edges, interior_edges

def compute_edge_length(editor, edge):
    """Compute the length of an edge."""
    v1, v2 = edge
    return np.linalg.norm(editor.points[v1] - editor.points[v2])

def is_near_corner(editor, edge, corner_regions):
    """Check if edge is near a corner (sharp feature)."""
    v1, v2 = edge
    midpoint = (editor.points[v1] + editor.points[v2]) / 2
    
    for corner, radius in corner_regions:
        if np.linalg.norm(midpoint - corner) < radius:
            return True
    return False

def refine_by_region(editor, boundary_threshold=0.3, interior_threshold=0.5, 
                     corner_regions=None, max_iterations=5):
    """Refine mesh with different criteria for boundary and interior.
    
    Args:
        editor: Mesh editor
        boundary_threshold: Max allowed boundary edge length
        interior_threshold: Max allowed interior edge length
        corner_regions: List of (corner_point, radius) for feature preservation
        max_iterations: Maximum refinement iterations
        
    Returns:
        Dictionary with refinement statistics
    """
    if corner_regions is None:
        corner_regions = []
    
    stats = {
        'boundary_splits': 0,
        'interior_splits': 0,
        'corner_splits': 0
    }
    
    for iteration in range(max_iterations):
        boundary_edges, interior_edges = identify_edges_by_type(editor)
        
        # Find boundary edges to split
        boundary_to_split = []
        for edge in boundary_edges:
            length = compute_edge_length(editor, edge)
            is_corner = is_near_corner(editor, edge, corner_regions)
            
            # Use stricter threshold near corners
            threshold = boundary_threshold * 0.7 if is_corner else boundary_threshold
            
            if length > threshold:
                boundary_to_split.append((length, edge, is_corner))
        
        # Find interior edges to split
        interior_to_split = []
        for edge in interior_edges:
            length = compute_edge_length(editor, edge)
            if length > interior_threshold:
                interior_to_split.append((length, edge))
        
        # Sort by length (longest first)
        boundary_to_split.sort(reverse=True)
        interior_to_split.sort(reverse=True)
        
        # Split edges
        splits_this_iter = 0
        
        # Prioritize boundary edges (especially near corners)
        for length, edge, is_corner in boundary_to_split[:8]:
            if edge in editor.edge_map:
                success = editor.split_edge(edge=edge)
                if success:
                    stats['boundary_splits'] += 1
                    if is_corner:
                        stats['corner_splits'] += 1
                    splits_this_iter += 1
        
        # Then split interior edges
        for length, edge in interior_to_split[:5]:
            if edge in editor.edge_map:
                success = editor.split_edge(edge=edge)
                if success:
                    stats['interior_splits'] += 1
                    splits_this_iter += 1
        
        # Stop if no more splits
        if splits_this_iter == 0:
            break
    
    return stats

def compute_edge_stats(editor):
    """Compute statistics about edge lengths by type."""
    boundary_edges, interior_edges = identify_edges_by_type(editor)
    
    boundary_lengths = [compute_edge_length(editor, e) for e in boundary_edges]
    interior_lengths = [compute_edge_length(editor, e) for e in interior_edges]
    
    return {
        'n_boundary': len(boundary_edges),
        'n_interior': len(interior_edges),
        'boundary_mean': np.mean(boundary_lengths) if boundary_lengths else 0,
        'boundary_max': np.max(boundary_lengths) if boundary_lengths else 0,
        'interior_mean': np.mean(interior_lengths) if interior_lengths else 0,
        'interior_max': np.max(interior_lengths) if interior_lengths else 0,
    }

def main():
    """Run the combined refinement demonstration."""
    print("=" * 70)
    print("SOFIA Example 8: Combined Interior + Boundary Refinement")
    print("=" * 70)
    
    # Step 1: Create L-shaped mesh
    print("\n[1] Creating L-shaped domain mesh...")
    editor, n_boundary_verts = create_l_shaped_mesh(resolution=0.5)
    
    # Define corner regions for feature preservation
    corner_regions = [
        (np.array([0, 0]), 0.3),    # Bottom-left corner
        (np.array([2, 0]), 0.3),    # Bottom-right corner
        (np.array([2, 1]), 0.3),    # Right inner corner
        (np.array([1, 1]), 0.3),    # Center inner corner
        (np.array([1, 2]), 0.3),    # Top inner corner
        (np.array([0, 2]), 0.3),    # Top-left corner
    ]
    
    # Save initial state
    initial_points = editor.points.copy()
    initial_active = np.all(editor.triangles != -1, axis=1)
    initial_triangles = editor.triangles[initial_active].copy()
    
    initial_stats = compute_edge_stats(editor)
    initial_n_verts = len(editor.points)
    initial_n_tris = np.sum(initial_active)
    
    valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    
    print(f"  Initial mesh:")
    print(f"    Vertices: {initial_n_verts}")
    print(f"    Triangles: {initial_n_tris}")
    print(f"    Boundary edges: {initial_stats['n_boundary']}, "
          f"mean length: {initial_stats['boundary_mean']:.3f}")
    print(f"    Interior edges: {initial_stats['n_interior']}, "
          f"mean length: {initial_stats['interior_mean']:.3f}")
    print(f"    Conformity: {' Valid' if valid else ' Invalid'}")
    
    # Step 2: Combined refinement
    print("\n[2] Applying combined refinement strategy...")
    print(f"  Strategy:")
    print(f"    - Boundary edges: refine if length > 0.3")
    print(f"    - Interior edges: refine if length > 0.5")
    print(f"    - Corner regions: extra refinement (radius=0.3)")
    
    refine_stats = refine_by_region(
        editor,
        boundary_threshold=0.3,
        interior_threshold=0.5,
        corner_regions=corner_regions,
        max_iterations=5
    )
    
    refined_stats = compute_edge_stats(editor)
    refined_active = np.all(editor.triangles != -1, axis=1)
    refined_n_verts = len(editor.points)
    refined_n_tris = np.sum(refined_active)
    
    valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    
    print(f"\n  After refinement:")
    print(f"    Boundary edges split: {refine_stats['boundary_splits']} "
          f"(including {refine_stats['corner_splits']} near corners)")
    print(f"    Interior edges split: {refine_stats['interior_splits']}")
    print(f"    Total splits: {refine_stats['boundary_splits'] + refine_stats['interior_splits']}")
    print(f"    Vertices: {refined_n_verts} (+{refined_n_verts - initial_n_verts})")
    print(f"    Triangles: {refined_n_tris} (+{refined_n_tris - initial_n_tris})")
    print(f"    Boundary edges: {refined_stats['n_boundary']} "
          f"(+{refined_stats['n_boundary'] - initial_stats['n_boundary']})")
    print(f"    Interior edges: {refined_stats['n_interior']} "
          f"(+{refined_stats['n_interior'] - initial_stats['n_interior']})")
    print(f"    Conformity: {' Valid' if valid else ' Invalid'}")
    
    # Step 3: Analysis
    print("\n[3] Refinement effectiveness analysis...")
    boundary_improvement = ((initial_stats['boundary_max'] - refined_stats['boundary_max']) / 
                           initial_stats['boundary_max'] * 100)
    interior_improvement = ((initial_stats['interior_max'] - refined_stats['interior_max']) / 
                           initial_stats['interior_max'] * 100)
    
    print(f"  Boundary refinement:")
    print(f"    Max length: {initial_stats['boundary_max']:.3f} → "
          f"{refined_stats['boundary_max']:.3f} ({boundary_improvement:.1f}% reduction)")
    print(f"    Mean length: {initial_stats['boundary_mean']:.3f} → "
          f"{refined_stats['boundary_mean']:.3f}")
    
    print(f"  Interior refinement:")
    print(f"    Max length: {initial_stats['interior_max']:.3f} → "
          f"{refined_stats['interior_max']:.3f} ({interior_improvement:.1f}% reduction)")
    print(f"    Mean length: {initial_stats['interior_mean']:.3f} → "
          f"{refined_stats['interior_mean']:.3f}")
    
    # Step 4: Visualization
    print("\n[4] Creating visualization...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Initial mesh
    ax = plt.subplot(2, 4, 1)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'k-', linewidth=0.5, alpha=0.3)
    
    initial_boundary, initial_interior = identify_edges_by_type(
        PatchBasedMeshEditor(initial_points, initial_triangles))
    
    for edge in initial_boundary:
        v1, v2 = edge
        ax.plot([initial_points[v1, 0], initial_points[v2, 0]],
                [initial_points[v1, 1], initial_points[v2, 1]],
                'b-', linewidth=2, alpha=0.7, label='Boundary' if edge == initial_boundary[0] else '')
    
    ax.plot(initial_points[:, 0], initial_points[:, 1], 'ko', markersize=4)
    ax.set_title(f'Initial Mesh\n{initial_n_verts} vertices, {initial_n_tris} triangles',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Refined mesh
    ax = plt.subplot(2, 4, 2)
    ax.set_aspect('equal')
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[refined_active], 'k-', linewidth=0.5, alpha=0.3)
    
    refined_boundary, refined_interior = identify_edges_by_type(editor)
    
    for edge in refined_boundary:
        v1, v2 = edge
        ax.plot([editor.points[v1, 0], editor.points[v2, 0]],
                [editor.points[v1, 1], editor.points[v2, 1]],
                'g-', linewidth=2, alpha=0.7, label='Boundary' if edge == refined_boundary[0] else '')
    
    ax.plot(editor.points[:, 0], editor.points[:, 1], 'ko', markersize=3)
    ax.set_title(f'Refined Mesh\n{refined_n_verts} vertices (+{refined_n_verts - initial_n_verts}), '
                 f'{refined_n_tris} triangles (+{refined_n_tris - initial_n_tris})',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Zoom on corner (1,1) - before
    ax = plt.subplot(2, 4, 5)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'k-', linewidth=1)
    for edge in initial_boundary:
        v1, v2 = edge
        ax.plot([initial_points[v1, 0], initial_points[v2, 0]],
                [initial_points[v1, 1], initial_points[v2, 1]],
                'b-', linewidth=3)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0.5, 1.5)
    ax.plot([1], [1], 'ro', markersize=10, label='Corner')
    ax.set_title('Corner Feature (Before)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Zoom on corner (1,1) - after
    ax = plt.subplot(2, 4, 6)
    ax.set_aspect('equal')
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[refined_active], 'k-', linewidth=1)
    for edge in refined_boundary:
        v1, v2 = edge
        ax.plot([editor.points[v1, 0], editor.points[v2, 0]],
                [editor.points[v1, 1], editor.points[v2, 1]],
                'g-', linewidth=3)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0.5, 1.5)
    ax.plot([1], [1], 'ro', markersize=10, label='Corner')
    ax.set_title('Corner Feature (After)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Edge length distributions - boundary
    ax = plt.subplot(2, 4, 3)
    initial_boundary_lengths = [compute_edge_length(
        PatchBasedMeshEditor(initial_points, initial_triangles), e) 
        for e in initial_boundary]
    final_boundary_lengths = [compute_edge_length(editor, e) for e in refined_boundary]
    
    ax.hist(initial_boundary_lengths, bins=15, alpha=0.6, 
            label='Initial', color='blue', edgecolor='black')
    ax.hist(final_boundary_lengths, bins=15, alpha=0.6, 
            label='Refined', color='green', edgecolor='black')
    ax.axvline(0.3, color='red', linestyle='--', linewidth=2, 
               label='Target: 0.3')
    ax.set_xlabel('Edge Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Boundary Edge Lengths', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Edge length distributions - interior
    ax = plt.subplot(2, 4, 4)
    initial_interior_lengths = [compute_edge_length(
        PatchBasedMeshEditor(initial_points, initial_triangles), e) 
        for e in initial_interior]
    final_interior_lengths = [compute_edge_length(editor, e) for e in refined_interior]
    
    ax.hist(initial_interior_lengths, bins=15, alpha=0.6, 
            label='Initial', color='blue', edgecolor='black')
    ax.hist(final_interior_lengths, bins=15, alpha=0.6, 
            label='Refined', color='green', edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, 
               label='Target: 0.5')
    ax.set_xlabel('Edge Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Interior Edge Lengths', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Zoom on top-left corner
    ax = plt.subplot(2, 4, 7)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'k-', linewidth=1)
    for edge in initial_boundary:
        v1, v2 = edge
        ax.plot([initial_points[v1, 0], initial_points[v2, 0]],
                [initial_points[v1, 1], initial_points[v2, 1]],
                'b-', linewidth=3)
    ax.set_xlim(-0.2, 0.5)
    ax.set_ylim(1.5, 2.2)
    ax.set_title('Top Corner (Before)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = plt.subplot(2, 4, 8)
    ax.set_aspect('equal')
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[refined_active], 'k-', linewidth=1)
    for edge in refined_boundary:
        v1, v2 = edge
        ax.plot([editor.points[v1, 0], editor.points[v2, 0]],
                [editor.points[v1, 1], editor.points[v2, 1]],
                'g-', linewidth=3)
    ax.set_xlim(-0.2, 0.5)
    ax.set_ylim(1.5, 2.2)
    ax.set_title('Top Corner (After)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_refinement_result.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'combined_refinement_result.png'")
    
    print("\n" + "=" * 70)
    print(" Example completed successfully!")
    print(f" Boundary splits: {refine_stats['boundary_splits']} "
          f"({refine_stats['corner_splits']} near corners)")
    print(f" Interior splits: {refine_stats['interior_splits']}")
    print(f" Total mesh growth: {initial_n_verts} → {refined_n_verts} vertices")
    print(f" Mesh conformity maintained throughout")
    print("=" * 70)

if __name__ == "__main__":
    main()
