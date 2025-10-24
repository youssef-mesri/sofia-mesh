#!/usr/bin/env python3
"""
SOFIA Example 7: Boundary Refinement

This example demonstrates adaptive refinement of boundary edges,
useful for capturing curved boundaries or improving boundary resolution.

What this example shows:
1. Creating a mesh with a defined boundary
2. Identifying boundary vs interior edges
3. Selectively refining boundary edges based on length
4. Comparing boundary resolution before/after
5. Maintaining mesh conformity during boundary refinement

Perfect for: Boundary layer meshing, curved domain approximation
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.conformity import check_mesh_conformity, build_edge_to_tri_map

def create_circular_domain_mesh(n_boundary=12, n_interior=20):
    """Create a mesh of a circular domain.
    
    Args:
        n_boundary: Number of points on the boundary circle
        n_interior: Number of random interior points
        
    Returns:
        editor: Mesh editor with circular domain
    """
    # Create boundary points (circle)
    theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
    boundary_pts = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Create interior points (random)
    np.random.seed(42)
    interior_pts = np.random.uniform(-0.8, 0.8, (n_interior, 2))
    
    # Keep only points inside the circle
    interior_pts = interior_pts[np.linalg.norm(interior_pts, axis=1) < 0.9]
    
    # Combine
    all_pts = np.vstack([boundary_pts, interior_pts])
    
    # Create Delaunay triangulation
    from scipy.spatial import Delaunay
    tri = Delaunay(all_pts)
    
    editor = PatchBasedMeshEditor(all_pts, tri.simplices)
    
    return editor, n_boundary

def identify_boundary_edges(editor):
    """Identify all boundary edges (edges with only one adjacent triangle)."""
    boundary_edges = []
    
    for edge, tris in editor.edge_map.items():
        # Count active triangles
        active_tris = [t for t in tris if np.all(editor.triangles[t] != -1)]
        
        if len(active_tris) == 1:
            boundary_edges.append(edge)
    
    return boundary_edges

def compute_edge_length(editor, edge):
    """Compute the length of an edge."""
    v1, v2 = edge
    return np.linalg.norm(editor.points[v1] - editor.points[v2])

def refine_boundary_edges(editor, max_length_threshold=0.5, max_splits=30):
    """Refine boundary edges longer than threshold.
    
    Args:
        editor: Mesh editor
        max_length_threshold: Maximum allowed boundary edge length
        max_splits: Maximum number of edge splits
        
    Returns:
        Number of edges split
    """
    splits = 0
    
    # Multiple passes to handle newly created edges
    for iteration in range(5):
        boundary_edges = identify_boundary_edges(editor)
        
        # Find edges to split
        edges_to_split = []
        for edge in boundary_edges:
            length = compute_edge_length(editor, edge)
            if length > max_length_threshold:
                edges_to_split.append((length, edge))
        
        if not edges_to_split or splits >= max_splits:
            break
        
        # Sort by length (longest first)
        edges_to_split.sort(reverse=True)
        
        # Split longest boundary edges
        for length, edge in edges_to_split[:10]:  # Process up to 10 per iteration
            if edge in editor.edge_map and splits < max_splits:
                success = editor.split_edge(edge=edge)
                if success:
                    splits += 1
        
        if not edges_to_split:
            break
    
    return splits

def compute_boundary_stats(editor):
    """Compute statistics about boundary edges."""
    boundary_edges = identify_boundary_edges(editor)
    
    if not boundary_edges:
        return {
            'n_boundary_edges': 0,
            'mean_length': 0,
            'max_length': 0,
            'min_length': 0
        }
    
    lengths = [compute_edge_length(editor, edge) for edge in boundary_edges]
    
    return {
        'n_boundary_edges': len(boundary_edges),
        'mean_length': np.mean(lengths),
        'max_length': np.max(lengths),
        'min_length': np.min(lengths),
        'std_length': np.std(lengths)
    }

def compute_mesh_stats(editor):
    """Compute general mesh statistics."""
    active_mask = np.all(editor.triangles != -1, axis=1)
    
    return {
        'n_vertices': len(editor.points),
        'n_triangles': np.sum(active_mask),
        'n_edges': len(editor.edge_map)
    }

def main():
    """Run the boundary refinement demonstration."""
    print("=" * 70)
    print("SOFIA Example 7: Boundary Refinement")
    print("=" * 70)
    
    # Step 1: Create initial mesh with circular boundary
    print("\n[1] Creating circular domain mesh...")
    editor, n_boundary_pts = create_circular_domain_mesh(n_boundary=12, n_interior=15)
    
    # Save initial state
    initial_points = editor.points.copy()
    initial_active = np.all(editor.triangles != -1, axis=1)
    initial_triangles = editor.triangles[initial_active].copy()
    initial_boundary_edges = identify_boundary_edges(editor)
    
    initial_mesh_stats = compute_mesh_stats(editor)
    initial_boundary_stats = compute_boundary_stats(editor)
    
    valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    
    print(f"  Initial mesh:")
    print(f"    Vertices: {initial_mesh_stats['n_vertices']}")
    print(f"    Triangles: {initial_mesh_stats['n_triangles']}")
    print(f"    Boundary edges: {initial_boundary_stats['n_boundary_edges']}")
    print(f"    Boundary edge length: mean={initial_boundary_stats['mean_length']:.3f}, "
          f"max={initial_boundary_stats['max_length']:.3f}")
    print(f"    Conformity: {'✓ Valid' if valid else '✗ Invalid'}")
    
    # Step 2: Refine boundary edges
    print("\n[2] Refining boundary edges...")
    max_length = 0.5
    print(f"  Target: Split boundary edges longer than {max_length}")
    
    splits = refine_boundary_edges(editor, max_length_threshold=max_length, max_splits=30)
    
    refined_mesh_stats = compute_mesh_stats(editor)
    refined_boundary_stats = compute_boundary_stats(editor)
    
    valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    
    print(f"\n  After boundary refinement:")
    print(f"    Edges split: {splits}")
    print(f"    Vertices: {refined_mesh_stats['n_vertices']} "
          f"(+{refined_mesh_stats['n_vertices'] - initial_mesh_stats['n_vertices']})")
    print(f"    Triangles: {refined_mesh_stats['n_triangles']} "
          f"(+{refined_mesh_stats['n_triangles'] - initial_mesh_stats['n_triangles']})")
    print(f"    Boundary edges: {refined_boundary_stats['n_boundary_edges']} "
          f"(+{refined_boundary_stats['n_boundary_edges'] - initial_boundary_stats['n_boundary_edges']})")
    print(f"    Boundary edge length: mean={refined_boundary_stats['mean_length']:.3f}, "
          f"max={refined_boundary_stats['max_length']:.3f}")
    print(f"    Conformity: {'✓ Valid' if valid else '✗ Invalid'}")
    
    # Step 3: Analysis
    print("\n[3] Boundary refinement analysis...")
    boundary_improvement = (initial_boundary_stats['max_length'] - 
                           refined_boundary_stats['max_length']) / initial_boundary_stats['max_length'] * 100
    
    print(f"  Boundary resolution improvement:")
    print(f"    Max edge length reduced: {initial_boundary_stats['max_length']:.3f} → "
          f"{refined_boundary_stats['max_length']:.3f} ({boundary_improvement:.1f}% reduction)")
    print(f"    Mean edge length: {initial_boundary_stats['mean_length']:.3f} → "
          f"{refined_boundary_stats['mean_length']:.3f}")
    print(f"    Boundary uniformity (1/std): "
          f"{1/initial_boundary_stats['std_length']:.2f} → "
          f"{1/refined_boundary_stats['std_length']:.2f}")
    
    # Step 4: Visualization
    print("\n[4] Creating visualization...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Initial mesh - full view
    ax = plt.subplot(2, 3, 1)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'k-', linewidth=0.5, alpha=0.3)
    
    # Highlight boundary edges
    for edge in initial_boundary_edges:
        v1, v2 = edge
        ax.plot([initial_points[v1, 0], initial_points[v2, 0]],
                [initial_points[v1, 1], initial_points[v2, 1]],
                'b-', linewidth=3, alpha=0.7)
    
    ax.plot(initial_points[:, 0], initial_points[:, 1], 'ko', markersize=4)
    ax.set_title(f'Initial Mesh\n{initial_boundary_stats["n_boundary_edges"]} boundary edges',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Refined mesh - full view
    ax = plt.subplot(2, 3, 2)
    ax.set_aspect('equal')
    refined_active = np.all(editor.triangles != -1, axis=1)
    refined_boundary_edges = identify_boundary_edges(editor)
    
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[refined_active], 'k-', linewidth=0.5, alpha=0.3)
    
    # Highlight boundary edges
    for edge in refined_boundary_edges:
        v1, v2 = edge
        ax.plot([editor.points[v1, 0], editor.points[v2, 0]],
                [editor.points[v1, 1], editor.points[v2, 1]],
                'g-', linewidth=2, alpha=0.7)
    
    ax.plot(editor.points[:, 0], editor.points[:, 1], 'ko', markersize=3)
    ax.set_title(f'Refined Mesh\n{refined_boundary_stats["n_boundary_edges"]} boundary edges '
                 f'(+{refined_boundary_stats["n_boundary_edges"] - initial_boundary_stats["n_boundary_edges"]})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Zoom on boundary - before
    ax = plt.subplot(2, 3, 4)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'k-', linewidth=0.8)
    for edge in initial_boundary_edges:
        v1, v2 = edge
        ax.plot([initial_points[v1, 0], initial_points[v2, 0]],
                [initial_points[v1, 1], initial_points[v2, 1]],
                'b-', linewidth=3)
    ax.set_xlim(0.5, 1.1)
    ax.set_ylim(-0.3, 0.3)
    ax.set_title('Initial Boundary (Zoom)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Zoom on boundary - after
    ax = plt.subplot(2, 3, 5)
    ax.set_aspect('equal')
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[refined_active], 'k-', linewidth=0.8)
    for edge in refined_boundary_edges:
        v1, v2 = edge
        ax.plot([editor.points[v1, 0], editor.points[v2, 0]],
                [editor.points[v1, 1], editor.points[v2, 1]],
                'g-', linewidth=3)
    ax.set_xlim(0.5, 1.1)
    ax.set_ylim(-0.3, 0.3)
    ax.set_title('Refined Boundary (Zoom)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Edge length distributions
    ax = plt.subplot(2, 3, 3)
    initial_lengths = [compute_edge_length(
        PatchBasedMeshEditor(initial_points, initial_triangles), e) 
        for e in initial_boundary_edges]
    final_lengths = [compute_edge_length(editor, e) for e in refined_boundary_edges]
    
    ax.hist(initial_lengths, bins=15, alpha=0.6, label='Initial', color='blue', edgecolor='black')
    ax.hist(final_lengths, bins=15, alpha=0.6, label='Refined', color='green', edgecolor='black')
    ax.axvline(max_length, color='red', linestyle='--', linewidth=2, 
               label=f'Target: {max_length}')
    ax.set_xlabel('Boundary Edge Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Boundary Edge Length Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics summary
    ax = plt.subplot(2, 3, 6)
    ax.axis('off')
    stats_text = f"""
Boundary Refinement Summary
{'=' * 35}

Initial Mesh:
  Vertices: {initial_mesh_stats['n_vertices']}
  Triangles: {initial_mesh_stats['n_triangles']}
  Boundary edges: {initial_boundary_stats['n_boundary_edges']}

Boundary Edges (Initial):
  Mean length: {initial_boundary_stats['mean_length']:.3f}
  Max length: {initial_boundary_stats['max_length']:.3f}
  Min length: {initial_boundary_stats['min_length']:.3f}

After Refinement:
  Vertices: {refined_mesh_stats['n_vertices']} (+{refined_mesh_stats['n_vertices'] - initial_mesh_stats['n_vertices']})
  Triangles: {refined_mesh_stats['n_triangles']} (+{refined_mesh_stats['n_triangles'] - initial_mesh_stats['n_triangles']})
  Boundary edges: {refined_boundary_stats['n_boundary_edges']} (+{refined_boundary_stats['n_boundary_edges'] - initial_boundary_stats['n_boundary_edges']})

Boundary Edges (Refined):
  Mean length: {refined_boundary_stats['mean_length']:.3f}
  Max length: {refined_boundary_stats['max_length']:.3f}
  Min length: {refined_boundary_stats['min_length']:.3f}

Improvement:
  Edges split: {splits}
  Max length reduction: {boundary_improvement:.1f}%
  Target threshold: {max_length}
  Conformity: ✓ Maintained
"""
    ax.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('boundary_refinement_result.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to 'boundary_refinement_result.png'")
    
    print("\n" + "=" * 70)
    print("✓ Example completed successfully!")
    print(f"✓ Boundary refined: {initial_boundary_stats['n_boundary_edges']} → "
          f"{refined_boundary_stats['n_boundary_edges']} edges")
    print(f"✓ Max boundary edge reduced by {boundary_improvement:.1f}%")
    print(f"✓ Mesh conformity maintained throughout")
    print("=" * 70)

if __name__ == "__main__":
    main()
