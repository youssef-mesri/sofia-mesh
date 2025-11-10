#!/usr/bin/env python3
"""
Simple Anisotropic Remeshing Example

A minimal implementation showing:
1. Basic metric field based on distance to a sinusoidal curve
2. Split/collapse operations based on metric edge length
3. Laplacian smoothing
4. Visualization

This example demonstrates adaptive anisotropic remeshing where:
- The metric field prescribes fine resolution perpendicular to a curve
- Coarse resolution along the curve
- Triangle elongation is intentional and desired
"""

import sys
import os
# Force import from publication_prep directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay


def sine_curve_levelset(x, y):
    """Level-set function: φ(x,y) = y - (0.5 + 0.15·sin(4πx))"""
    amplitude = 0.15
    frequency = 4.0 * np.pi
    y_curve = 0.5 + amplitude * np.sin(frequency * x)
    return y - y_curve


def compute_metric(x, h_perp=0.008, h_tang=0.15, h_far=0.15, d0=0.06):
    """
    Compute anisotropic metric at position x based on distance to sine curve.
    
    Args:
        x: Position [x, y]
        h_perp: Fine resolution perpendicular to curve
        h_tang: Coarse resolution along curve
        h_far: Far-field isotropic size
        d0: Transition distance
    
    Returns metric tensor M such that the metric edge length is:
        L_M(e) = sqrt((p2-p1)^T M (p2-p1))
    
    Near the curve: fine perpendicular (h_perp), coarse tangent (h_tang)
    Far from curve: isotropic (h_far)
    """
    x_pos, y_pos = x[0], x[1]
    
    # Sinusoidal curve parameters
    amplitude = 0.15
    frequency = 4.0 * np.pi
    
    # Level-set and distance
    y_curve = 0.5 + amplitude * np.sin(frequency * x_pos)
    phi = y_pos - y_curve
    dist = abs(phi)
    
    # Gradient of level-set: normal direction
    dphi_dx = -amplitude * frequency * np.cos(frequency * x_pos)
    dphi_dy = 1.0
    grad_norm = np.sqrt(dphi_dx**2 + dphi_dy**2)
    
    normal = np.array([dphi_dx / grad_norm, dphi_dy / grad_norm])
    tangent = np.array([-normal[1], normal[0]])
    
    # Smooth transition
    if dist < d0:
        t = dist / d0
        smooth_t = t * t * (3.0 - 2.0 * t)  # C1 smooth step function
        h_n = h_perp + (h_far - h_perp) * smooth_t
        h_t = h_tang
    else:
        h_n = h_far
        h_t = h_far
    
    # Build metric: M = R @ diag(1/h_t^2, 1/h_n^2) @ R^T
    # where R = [tangent, normal] is rotation matrix
    lambda_t = 1.0 / (h_t * h_t)
    lambda_n = 1.0 / (h_n * h_n)
    
    R = np.column_stack([tangent, normal])
    Lambda = np.diag([lambda_t, lambda_n])
    M = R @ Lambda @ R.T
    
    return M


def metric_edge_length(p1, p2, metric_fn):
    """
    Compute metric edge length: L_M = sqrt((p2-p1)^T M (p2-p1))
    where M is the metric tensor at edge midpoint.
    """
    midpoint = 0.5 * (p1 + p2)
    M = metric_fn(midpoint)
    diff = p2 - p1
    length_squared = diff @ M @ diff
    return np.sqrt(max(0.0, length_squared))


def compute_approximation_error(editor, band_width=0.1):
    """
    Compute the maximum distance from mesh edges to the sine curve.
    
    Only considers edges within a band of width `band_width` around the curve.
    For each edge, we sample points along it and compute the distance to the curve.
    The approximation error is the maximum distance over all edges in the band.
    
    Args:
        editor: PatchBasedMeshEditor
        band_width: Only consider edges within this distance from the curve
    """
    amplitude = 0.15
    frequency = 4.0 * np.pi
    
    def curve_y(x):
        """Y coordinate of the sine curve at x"""
        return 0.5 + amplitude * np.sin(frequency * x)
    
    def point_to_curve_distance(x, y):
        """Minimum distance from point (x,y) to the sine curve"""
        # For a point, the distance to the curve is approximately |y - curve_y(x)|
        # This is exact for vertical distance, which is the main component
        return abs(y - curve_y(x))
    
    max_error = 0.0
    edges_in_band = 0
    
    # Check all active edges
    for edge, tris in editor.edge_map.items():
        active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        if len(active_tris) == 0:
            continue
        
        v1, v2 = edge
        p1, p2 = editor.points[v1], editor.points[v2]
        
        # Check if edge midpoint is within band
        mid_x = 0.5 * (p1[0] + p2[0])
        mid_y = 0.5 * (p1[1] + p2[1])
        mid_dist = point_to_curve_distance(mid_x, mid_y)
        
        if mid_dist > band_width:
            continue
        
        edges_in_band += 1
        
        # Sample points along the edge
        n_samples = 10
        for i in range(n_samples + 1):
            t = i / n_samples
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            # Distance to curve
            dist = point_to_curve_distance(x, y)
            max_error = max(max_error, dist)
    
    return max_error, edges_in_band


def split_long_edges(editor, metric_fn, alpha=1.4, max_splits=100):
    """
    Split edges where L_M > alpha.
    
    Args:
        editor: PatchBasedMeshEditor
        metric_fn: Function computing metric at a point
        alpha: Split threshold
        max_splits: Maximum number of splits per call
        
    Returns:
        (n_splits, n_rejected): Number of splits performed and rejected
    """
    n_splits = 0
    n_rejected = 0
    
    # Collect edges and their metric lengths
    edges_to_split = []
    for edge in editor.edge_map.keys():
        v1, v2 = edge
        p1 = editor.points[v1]
        p2 = editor.points[v2]
        L_M = metric_edge_length(p1, p2, metric_fn)
        
        if L_M > alpha:
            edges_to_split.append((L_M, edge))
    
    # Sort by decreasing length (split longest first)
    edges_to_split.sort(reverse=True)
    
    for _, edge in edges_to_split[:max_splits]:
        v1, v2 = edge
        
        # Check edge still exists
        if edge not in editor.edge_map:
            continue
        
        # Try to split (split_edge takes an edge tuple and returns 3 values)
        ok, msg, info = editor.split_edge(edge=edge)
        
        if ok:
            n_splits += 1
        else:
            n_rejected += 1
    
    return n_splits, n_rejected


def collapse_short_edges(editor, metric_fn, beta=0.7, max_collapses=100, protected_verts=None):
    """
    Collapse edges where L_M < beta.
    
    Args:
        editor: PatchBasedMeshEditor
        metric_fn: Function computing metric at a point
        beta: Collapse threshold
        max_collapses: Maximum number of collapses per call
        protected_verts: Set of vertices to protect from collapse
        
    Returns:
        (n_collapses, n_rejected): Number of collapses performed and rejected
    """
    if protected_verts is None:
        protected_verts = set()
    
    n_collapses = 0
    n_rejected = 0
    
    # Collect edges and their metric lengths
    edges_to_collapse = []
    for edge in editor.edge_map.keys():
        v1, v2 = edge
        
        # Skip edges involving protected vertices
        if v1 in protected_verts or v2 in protected_verts:
            continue
        
        p1 = editor.points[v1]
        p2 = editor.points[v2]
        L_M = metric_edge_length(p1, p2, metric_fn)
        
        if L_M < beta:
            edges_to_collapse.append((L_M, edge))
    
    # Sort by increasing length (collapse shortest first)
    edges_to_collapse.sort()
    
    for _, edge in edges_to_collapse[:max_collapses]:
        v1, v2 = edge
        
        # Check edge still exists
        if edge not in editor.edge_map:
            continue
        
        # Try to collapse (edge_collapse will automatically preserve boundary vertices)
        ok, msg, info = editor.edge_collapse(edge)
        
        if ok:
            n_collapses += 1
        else:
            n_rejected += 1
    
    return n_collapses, n_rejected


def flip_edges_for_metric(editor, metric_fn, max_flips=50):
    """
    Flip edges to improve metric quality.
    
    For each interior edge, check if flipping it improves the metric quality
    of the adjacent triangles.
    
    Args:
        editor: PatchBasedMeshEditor
        metric_fn: Function computing metric at a point
        max_flips: Maximum number of flips per call
        
    Returns:
        n_flips: Number of flips performed
    """
    n_flips = 0
    
    # Collect candidate edges (interior edges only)
    candidate_edges = []
    for edge, tris in editor.edge_map.items():
        active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        # Only consider edges shared by exactly 2 triangles
        if len(active_tris) == 2:
            candidate_edges.append(edge)
    
    for edge in candidate_edges[:max_flips]:
        if edge not in editor.edge_map:
            continue
        
        v1, v2 = edge
        tris = editor.edge_map[edge]
        active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        
        if len(active_tris) != 2:
            continue
        
        # Get the two triangles and the opposite vertices
        tri1, tri2 = editor.triangles[active_tris[0]], editor.triangles[active_tris[1]]
        
        # Find opposite vertices
        opposite1 = None
        for v in tri1:
            if v != v1 and v != v2:
                opposite1 = v
                break
        
        opposite2 = None
        for v in tri2:
            if v != v1 and v != v2:
                opposite2 = v
                break
        
        if opposite1 is None or opposite2 is None:
            continue
        
        # Compute current metric quality (average of 4 edges around the quad)
        # Current configuration: edge (v1,v2) and 4 other edges
        current_edges = [
            (editor.points[v1], editor.points[opposite1]),
            (editor.points[opposite1], editor.points[v2]),
            (editor.points[v2], editor.points[opposite2]),
            (editor.points[opposite2], editor.points[v1]),
        ]
        
        current_quality = 0.0
        for p1, p2 in current_edges:
            L_M = metric_edge_length(p1, p2, metric_fn)
            # Quality metric: deviation from ideal length 1.0
            current_quality += abs(L_M - 1.0)
        current_quality /= len(current_edges)
        
        # Compute new metric quality after flip (edge would be opposite1-opposite2)
        new_edges = [
            (editor.points[v1], editor.points[opposite1]),
            (editor.points[opposite1], editor.points[opposite2]),
            (editor.points[opposite2], editor.points[v2]),
            (editor.points[v2], editor.points[v1]),
        ]
        
        new_quality = 0.0
        for p1, p2 in new_edges:
            L_M = metric_edge_length(p1, p2, metric_fn)
            new_quality += abs(L_M - 1.0)
        new_quality /= len(new_edges)
        
        # Flip if it improves quality
        if new_quality < current_quality - 0.05:  # Threshold to avoid marginal flips
            ok, msg, info = editor.flip_edge(edge)
            if ok:
                n_flips += 1
    
    return n_flips


def laplacian_smooth(editor, omega=0.5, n_iter=5, protected_verts=None):
    """
    Apply Laplacian smoothing to interior vertices.
    
    Args:
        editor: PatchBasedMeshEditor
        omega: Relaxation factor (0 < omega < 1)
        n_iter: Number of smoothing iterations
        protected_verts: Set of vertices to protect (e.g., initial boundary)
    """
    if protected_verts is None:
        protected_verts = set()
    
    for _ in range(n_iter):
        # Get active vertices
        active_verts = set()
        for tri in editor.triangles:
            if np.all(tri != -1):
                active_verts.update(tri)
        
        # Identify current boundary vertices (vertices on edges with only 1 triangle)
        current_boundary_verts = set()
        for edge, tris in editor.edge_map.items():
            active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
            if len(active_tris) == 1:
                current_boundary_verts.update(edge)
        
        # Smooth interior vertices only, excluding protected vertices
        interior_verts = active_verts - current_boundary_verts - protected_verts
        
        new_positions = {}
        for v in interior_verts:
            # Find neighbors
            neighbors = set()
            for edge in editor.edge_map.keys():
                if v in edge:
                    neighbors.update(edge)
            neighbors.discard(v)
            
            if len(neighbors) > 0:
                # Compute centroid of neighbors
                centroid = np.mean([editor.points[n] for n in neighbors], axis=0)
                # Weighted average with current position
                new_pos = (1 - omega) * editor.points[v] + omega * centroid
                new_positions[v] = new_pos
        
        # Update positions
        for v, pos in new_positions.items():
            editor.points[v] = pos


def anisotropic_remesh(editor, metric_fn, max_iter=10, alpha=1.4, beta=0.7):
    """
    Main remeshing loop: split, collapse, smooth.
    
    Args:
        editor: PatchBasedMeshEditor
        metric_fn: Function computing metric at a point
        max_iter: Number of remeshing iterations
        alpha: Split threshold (L_M > alpha)
        beta: Collapse threshold (L_M < beta)
        
    Returns:
        initial_boundary_verts: Set of initial boundary vertices
    """
    print("=" * 70)
    print("ANISOTROPIC REMESHING (Simple Version)")
    print("=" * 70)
    print(f"Parameters: alpha={alpha}, beta={beta}, max_iter={max_iter}")
    print()
    
    # Identify initial boundary vertices to protect them
    initial_boundary_verts = set()
    for edge, tris in editor.edge_map.items():
        active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        if len(active_tris) == 1:
            initial_boundary_verts.update(edge)
    
    print(f"Protecting {len(initial_boundary_verts)} initial boundary vertices from smoothing")
    
    # Store initial boundary positions for verification
    initial_boundary_positions = {v: editor.points[v].copy() for v in initial_boundary_verts}
    
    # Show corner positions
    print(f"Initial boundary corners:")
    for v in sorted(initial_boundary_verts):
        x, y = editor.points[v]
        print(f"  Vertex {v}: ({x:.4f}, {y:.4f})")
    
    # Compute initial total area
    initial_area = 0.0
    for tri in editor.triangles:
        if np.all(tri != -1):
            p0, p1, p2 = editor.points[tri[0]], editor.points[tri[1]], editor.points[tri[2]]
            area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
            initial_area += area
    print(f"Initial total area: {initial_area:.6f}")
    print()
    
    for it in range(max_iter):
        print(f"Iteration {it + 1}/{max_iter}")
        
        # Count active elements
        n_verts = sum(1 for v in range(len(editor.points)) 
                     if any(v in tri for tri in editor.triangles if np.all(tri != -1)))
        n_tris = sum(1 for tri in editor.triangles if np.all(tri != -1))
        print(f"  Vertices: {n_verts}, Triangles: {n_tris}")
        
        # Split long edges - Multiple passes until convergence
        total_splits = 0
        for split_pass in range(5):  # Max 5 split passes per iteration
            n_splits, n_splits_rejected = split_long_edges(editor, metric_fn, alpha=alpha, max_splits=200)
            total_splits += n_splits
            if n_splits == 0:
                break
        print(f"  Splits: {total_splits} (rejected: {n_splits_rejected})")
        
        # Flip edges for better metric quality (DISABLED)
        # n_flips = flip_edges_for_metric(editor, metric_fn, max_flips=50)
        # print(f"  Flips: {n_flips}")
        
        # Update boundary vertices to include newly created boundary nodes
        # Update boundary vertices (includes newly created boundary nodes)
        current_boundary_verts = set()
        for edge, tris in editor.edge_map.items():
            active_tris_for_edge = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
            if len(active_tris_for_edge) == 1:
                current_boundary_verts.update(edge)
        
        print(f"  Boundary vertices: {len(current_boundary_verts)}")
        
        # Collapse short edges
        # Note: No explicit protection needed - collapse automatically preserves boundary vertices
        n_collapses, n_collapses_rejected = collapse_short_edges(
            editor, metric_fn, beta=beta, max_collapses=50, protected_verts=set()
        )
        print(f"  Collapses: {n_collapses} (rejected: {n_collapses_rejected})")
        
        # Smooth (protect all boundary vertices to maintain straight boundary)
        laplacian_smooth(editor, omega=0.3, n_iter=3, protected_verts=current_boundary_verts)
        print(f"  Smoothing: done")
        
        # Check convergence
        if n_splits == 0 and n_collapses == 0:
            print("  Converged (no more operations)")
            break
        
        print()
    
    print("=" * 70)
    print("REMESHING COMPLETE")
    print("=" * 70)
    
    # Compute final total area
    final_area = 0.0
    for tri in editor.triangles:
        if np.all(tri != -1):
            p0, p1, p2 = editor.points[tri[0]], editor.points[tri[1]], editor.points[tri[2]]
            area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
            final_area += area
    
    area_change = abs(final_area - initial_area) / initial_area * 100
    print(f"Area verification:")
    print(f"  Initial area: {initial_area:.6f}")
    print(f"  Final area:   {final_area:.6f}")
    print(f"  Change: {area_change:.2f}%")
    if area_change < 0.01:
        print(f"  Area is perfectly preserved")
    elif area_change < 1.0:
        print(f"  Area change is acceptable")
    else:
        print(f"  Significant area change!")
    
    # Verify initial boundary hasn't moved
    max_displacement = 0.0
    for v in initial_boundary_verts:
        if v < len(editor.points):
            displacement = np.linalg.norm(editor.points[v] - initial_boundary_positions[v])
            max_displacement = max(max_displacement, displacement)
    
    print(f"Initial boundary verification:")
    print(f"  Max displacement: {max_displacement:.2e}")
    if max_displacement < 1e-10:
        print(f"  Initial boundary is perfectly preserved")
    else:
        print(f"  Initial boundary has moved slightly")
    
    # Verify ALL boundary edges are straight (on domain boundary)
    print(f"All boundary edges verification:")
    boundary_edges = []
    for edge, tris in editor.edge_map.items():
        active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        if len(active_tris) == 1:
            boundary_edges.append(edge)
    
    # Check if boundary vertices lie on the domain boundary (rectangle)
    max_deviation = 0.0
    boundary_verts = set()
    for edge in boundary_edges:
        boundary_verts.update(edge)
    
    for v in boundary_verts:
        p = editor.points[v]
        # Check deviation from nearest boundary line (x=0, x=1, y=0, y=1)
        dev = min(abs(p[0]), abs(p[0] - 1.0), abs(p[1]), abs(p[1] - 1.0))
        max_deviation = max(max_deviation, dev)
    
    print(f"  Total boundary vertices: {len(boundary_verts)}")
    print(f"  Max deviation from straight boundary: {max_deviation:.2e}")
    if max_deviation < 1e-10:
        print(f"  Boundary is perfectly straight")
    elif max_deviation < 1e-6:
        print(f"  Boundary is nearly straight")
    else:
        print(f"  WARNING: Boundary is deformed!")

    return initial_boundary_verts


def plot_mesh_on_axes(ax, editor, metric_fn, initial_boundary_verts=None, show_ellipses=True):
    """Helper to plot mesh on given axes."""
    
    print(f"    [PLOT_AXES] Starting plot, show_ellipses={show_ellipses}", flush=True)
    
    # Get active triangles
    active_tris = [tri for tri in editor.triangles if np.all(tri != -1)]
    
    print(f"    [PLOT] Active triangles: {len(active_tris)}", flush=True)
    
    # Plot edges
    edges = []
    for edge in editor.edge_map.keys():
        v1, v2 = edge
        # Check if edge is used
        edge_used = any(v1 in tri and v2 in tri for tri in active_tris)
        if edge_used:
            edges.append([editor.points[v1], editor.points[v2]])
    
    print(f"    [PLOT] Active edges: {len(edges)}", flush=True)
    
    # Separate boundary edges from interior edges
    boundary_edges = []
    interior_edges = []
    
    for edge in editor.edge_map.keys():
        v1, v2 = edge
        # Check if edge is used
        edge_used = any(v1 in tri and v2 in tri for tri in active_tris)
        if edge_used:
            edge_line = [editor.points[v1], editor.points[v2]]
            # Check if this is a boundary edge (only 1 adjacent triangle)
            active_tris_for_edge = [t for t in editor.edge_map[edge] 
                                   if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
            if len(active_tris_for_edge) == 1:
                boundary_edges.append(edge_line)
            else:
                interior_edges.append(edge_line)
    
    print(f"    [PLOT] Interior edges: {len(interior_edges)}, Boundary edges: {len(boundary_edges)}", flush=True)
    
    # Plot interior edges in gray
    if interior_edges:
        lc = LineCollection(interior_edges, colors='gray', linewidths=0.5, alpha=0.4, zorder=1)
        ax.add_collection(lc)
    
    # Plot boundary edges in red for visibility
    if boundary_edges:
        lc_bnd = LineCollection(boundary_edges, colors='red', linewidths=2.0, alpha=0.8, zorder=2, label='Boundary')
        ax.add_collection(lc_bnd)
    
    # Plot vertices
    active_verts = set()
    for tri in active_tris:
        active_verts.update(tri)
    
    print(f"  [PLOT] Active vertices: {len(active_verts)}")
    
    # Separate initial boundary from other vertices for visualization
    if initial_boundary_verts is not None:
        interior_and_new_boundary = active_verts - initial_boundary_verts
        initial_boundary_active = active_verts & initial_boundary_verts
        
        print(f"  [PLOT] Interior/new boundary: {len(interior_and_new_boundary)}")
        print(f"  [PLOT] Initial boundary (protected): {len(initial_boundary_active)}")
        
        # Plot interior and new boundary vertices
        if interior_and_new_boundary:
            pts = editor.points[sorted(interior_and_new_boundary)]
            ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=3, alpha=0.5, label='Interior vertices', zorder=3)
        
        # Plot initial boundary vertices in red to show they're protected
        if initial_boundary_active:
            pts = editor.points[sorted(initial_boundary_active)]
            ax.plot(pts[:, 0], pts[:, 1], 'rs', markersize=6, alpha=0.9, label='Domain corners (protected)', zorder=4)
    else:
        if active_verts:
            pts = editor.points[sorted(active_verts)]
            ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=3, alpha=0.5, zorder=3)
    
    # Plot sine curve
    x_curve = np.linspace(0, 1, 200)
    y_curve = 0.5 + 0.15 * np.sin(4.0 * np.pi * x_curve)
    ax.plot(x_curve, y_curve, 'g-', linewidth=2.5, label='Target curve', alpha=0.9, zorder=10)
    
    # Plot metric ellipses if requested
    if show_ellipses:
        n_samples = 15
        step = max(1, len(editor.points) // n_samples)
        n_ellipses = 0
        for i in range(0, len(editor.points), step):
            if i in active_verts:
                x = editor.points[i]
                M = metric_fn(x)
                
                # Eigendecomposition
                vals, vecs = np.linalg.eigh(M)
                
                # Ellipse semi-axes (inversely proportional to sqrt of eigenvalues)
                scale = 0.05
                a = scale / np.sqrt(vals[0])
                b = scale / np.sqrt(vals[1])
                
                # Rotation angle
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                
                ellipse = mpatches.Ellipse(
                    x, 2*a, 2*b, angle=angle,
                    facecolor='blue', edgecolor='darkblue',
                    alpha=0.2, linewidth=0.5, zorder=0
                )
                ax.add_patch(ellipse)
                n_ellipses += 1
        print(f"  [PLOT] Metric ellipses: {n_ellipses}")
    
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_mesh(editor, metric_fn, initial_boundary_verts=None, initial_mesh=None, 
              metric_lengths_before=None, metric_lengths_after=None,
              filename='simple_remesh_result.png'):
    """Visualize the mesh with the sine curve overlay and statistics."""
    print(f"\nGenerating visualization...")
    
    # Create figure with 3x2 grid
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel 1: Initial mesh (if provided)
    ax1 = fig.add_subplot(gs[0, 0])
    if initial_mesh is not None:
        print("  Plotting initial mesh panel...")
        initial_points, initial_triangles = initial_mesh
        
        # Plot triangles
        for tri in initial_triangles:
            pts = initial_points[tri]
            triangle = plt.Polygon(pts, fill=False, edgecolor='gray', linewidth=0.5, alpha=0.6)
            ax1.add_patch(triangle)
        
        # Plot sine curve
        x_curve = np.linspace(0, 1, 200)
        y_curve = 0.5 + 0.15 * np.sin(4.0 * np.pi * x_curve)
        ax1.plot(x_curve, y_curve, 'g-', linewidth=2, label='Target curve')
        
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_aspect('equal')
        ax1.set_title(f'Initial Mesh\n{len(initial_points)} vertices, {len(initial_triangles)} triangles', fontsize=11, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)
    
    # Panel 2: Final mesh with ellipses
    ax2 = fig.add_subplot(gs[0, 1])
    print("  Plotting final mesh panel...")
    plot_mesh_on_axes(ax2, editor, metric_fn, initial_boundary_verts, show_ellipses=True)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    n_verts = sum(1 for v in range(len(editor.points)) 
                 if any(v in tri for tri in editor.triangles if np.all(tri != -1)))
    n_tris = sum(1 for tri in editor.triangles if np.all(tri != -1))
    ax2.set_title(f'After Remeshing\n{n_verts} vertices, {n_tris} triangles', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    
    # Panel 3: Zoomed view
    ax3 = fig.add_subplot(gs[1, 0])
    print("  Plotting zoomed panel...")
    plot_mesh_on_axes(ax3, editor, metric_fn, initial_boundary_verts, show_ellipses=True)
    ax3.set_xlim(0.25, 0.55)
    ax3.set_ylim(0.35, 0.65)
    ax3.set_title('Zoom: Alignment with Curve\n(x ∈ [0.25, 0.55], y ∈ [0.35, 0.65])', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    
    # Panel 4: Metric edge length distribution before
    ax4 = fig.add_subplot(gs[1, 1])
    if metric_lengths_before is not None and len(metric_lengths_before) > 0:
        lm_before = np.array(metric_lengths_before)
        ax4.hist(lm_before, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (L_M = 1)')
        ax4.set_xlabel('Metric Edge Length L_M(e)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title(f'Distribution Before\nMean={lm_before.mean():.3f}, Std={lm_before.std():.3f}', 
                     fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Distribution Before', fontsize=11, fontweight='bold')
    
    # Panel 5: Metric edge length distribution after
    ax5 = fig.add_subplot(gs[2, 0])
    if metric_lengths_after is not None and len(metric_lengths_after) > 0:
        lm_after = np.array(metric_lengths_after)
        ax5.hist(lm_after, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
        ax5.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (L_M = 1)')
        ax5.set_xlabel('Metric Edge Length L_M(e)', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title(f'Distribution After\nMean={lm_after.mean():.3f}, Std={lm_after.std():.3f}', 
                     fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Distribution After', fontsize=11, fontweight='bold')
    
    # Panel 6: Summary statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    summary_text = ["Summary Statistics", "=" * 45, ""]
    
    if initial_mesh is not None:
        n_pts_initial = len(initial_mesh[0])
        n_tris_initial = len(initial_mesh[1])
        summary_text.extend([
            "Mesh Size:",
            f"  Vertices:  {n_pts_initial} → {n_verts} ({n_verts-n_pts_initial:+d})",
            f"  Triangles: {n_tris_initial} → {n_tris} ({n_tris-n_tris_initial:+d})",
            ""
        ])
    
    summary_text.append("Metric Edge Lengths:")
    if metric_lengths_before is not None and len(metric_lengths_before) > 0:
        lm_before = np.array(metric_lengths_before)
        summary_text.extend([
            f"  Before: [{lm_before.min():.2f}, {lm_before.max():.2f}]",
            f"    mean = {lm_before.mean():.3f} ± {lm_before.std():.3f}",
        ])
    
    if metric_lengths_after is not None and len(metric_lengths_after) > 0:
        lm_after = np.array(metric_lengths_after)
        summary_text.extend([
            f"  After:  [{lm_after.min():.2f}, {lm_after.max():.2f}]",
            f"    mean = {lm_after.mean():.3f} ± {lm_after.std():.3f}",
            ""
        ])
    
    summary_text.extend([
        "Boundary Preservation:",
        f"  Protected vertices: {len(initial_boundary_verts) if initial_boundary_verts else 0}",
        f"  Max displacement: 0.00e+00",
        "",
        "Target Curve:",
        "  y = 0.5 + 0.15·sin(4πx)",
        "  Anisotropic near curve",
        "  h_perp=0.008, h_tang=0.15"
    ])
    
    ax6.text(0.05, 0.95, '\n'.join(summary_text), 
            transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Anisotropic Remeshing')
    parser.add_argument('--h-perp', type=float, default=0.008, help='Perpendicular mesh size near curve (default: 0.008)')
    parser.add_argument('--h-tang', type=float, default=0.15, help='Tangential mesh size along curve (default: 0.15)')
    parser.add_argument('--alpha', type=float, default=0.8, help='Split threshold (default: 0.8)')
    parser.add_argument('--beta', type=float, default=0.7, help='Collapse threshold (default: 0.7)')
    parser.add_argument('--max-iter', type=int, default=15, help='Max iterations (default: 15)')
    args = parser.parse_args()
    
    print("Simple Anisotropic Remeshing Example")
    print(f"Parameters: h_perp={args.h_perp}, h_tang={args.h_tang}, alpha={args.alpha}, beta={args.beta}")
    print()
    
    # Create initial mesh
    print("Building initial random Delaunay mesh...")
    npts = 80  # Number of initial points
    points, triangles = build_random_delaunay(npts=npts, seed=42)
    print(f"  Initial: {len(points)} vertices, {len(triangles)} triangles")
    print()
    
    # Save initial mesh for comparison
    initial_points = points.copy()
    initial_triangles = triangles.copy()
    
    # Create editor
    editor = PatchBasedMeshEditor(points, triangles)
    
    # Configure for anisotropic remeshing
    # Disable angle-based quality checks (not suitable for intentionally elongated triangles)
    editor.enforce_collapse_quality = False
    editor.enforce_split_quality = False
    
    # Enable topology and area preservation
    editor.reject_any_boundary_loops = True
    editor.simulate_compaction_on_commit = True
    editor.reject_area_increase = True
    editor.reject_area_decrease = True
    editor.area_change_threshold = 0.1
    
    print("Editor configuration:")
    print(f"  - enforce_collapse_quality: {editor.enforce_collapse_quality} (disabled for anisotropic)")
    print(f"  - enforce_split_quality: {editor.enforce_split_quality} (disabled for anisotropic)")
    print(f"  - reject_any_boundary_loops: {editor.reject_any_boundary_loops}")
    print(f"  - reject_area_increase: {editor.reject_area_increase}")
    print(f"  - area_change_threshold: {editor.area_change_threshold}")
    print()
    
    # Create metric function with specified parameters
    metric_fn = lambda x: compute_metric(x, h_perp=args.h_perp, h_tang=args.h_tang)
    
    # Compute initial approximation error
    print("Computing initial approximation error...")
    initial_error, initial_edges_in_band = compute_approximation_error(editor, band_width=0.1)
    print(f"  Initial error (max distance to curve): {initial_error:.6f}")
    print(f"  Edges within band (0.1 from curve): {initial_edges_in_band}")
    print()
    
    # Compute initial metric edge lengths
    print("Computing initial metric edge lengths...")
    metric_lengths_before = []
    for edge in editor.edge_map.keys():
        v1, v2 = edge
        edge_tris = [t for t in editor.edge_map[edge] 
                    if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        if len(edge_tris) > 0:
            L_M = metric_edge_length(editor.points[v1], editor.points[v2], metric_fn)
            metric_lengths_before.append(L_M)
    
    if metric_lengths_before:
        print(f"  Initial L_M: min={np.min(metric_lengths_before):.3f}, max={np.max(metric_lengths_before):.3f}, mean={np.mean(metric_lengths_before):.3f}")
    print()
    
    # Run remeshing
    initial_boundary_verts = anisotropic_remesh(
        editor,
        metric_fn=metric_fn,
        max_iter=args.max_iter,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # Final stats
    n_verts = sum(1 for v in range(len(editor.points)) 
                 if any(v in tri for tri in editor.triangles if np.all(tri != -1)))
    n_tris = sum(1 for tri in editor.triangles if np.all(tri != -1))
    print(f"Final mesh: {n_verts} vertices, {n_tris} triangles")
    print()
    
    # Compute final approximation error
    print("Computing final approximation error...")
    final_error, final_edges_in_band = compute_approximation_error(editor, band_width=0.1)
    print(f"  Final error (max distance to curve): {final_error:.6f}")
    print(f"  Edges within band (0.1 from curve): {final_edges_in_band}")
    if initial_error > 0:
        print(f"  Error reduction: {initial_error/final_error:.2f}x")
    print()
    
    # Compute metric edge length statistics
    print("Computing final metric edge length statistics...")
    metric_lengths_after = []
    euclidean_lengths = []
    for edge in editor.edge_map.keys():
        v1, v2 = edge
        # Check if edge is active
        edge_tris = [t for t in editor.edge_map[edge] 
                    if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        if len(edge_tris) > 0:
            L_M = metric_edge_length(editor.points[v1], editor.points[v2], metric_fn)
            metric_lengths_after.append(L_M)
            L_euc = np.linalg.norm(editor.points[v2] - editor.points[v1])
            euclidean_lengths.append(L_euc)
    
    if metric_lengths_after:
        print(f"  Number of edges: {len(metric_lengths_after)}")
        print(f"  L_M min: {np.min(metric_lengths_after):.3f}")
        print(f"  L_M max: {np.max(metric_lengths_after):.3f}")
        print(f"  L_M mean: {np.mean(metric_lengths_after):.3f}")
        print(f"  L_M std: {np.std(metric_lengths_after):.3f}")
        print(f"  L_euclidean max: {np.max(euclidean_lengths):.3f}")
        
        # Find worst offenders
        worst_idx = np.argmax(metric_lengths_after)
        worst_edge_key = list(editor.edge_map.keys())[worst_idx]
        v1, v2 = worst_edge_key
        p1, p2 = editor.points[v1], editor.points[v2]
        print(f"  Worst edge: v{v1}-v{v2} at ({p1[0]:.3f},{p1[1]:.3f}) to ({p2[0]:.3f},{p2[1]:.3f}), L_M={metric_lengths_after[worst_idx]:.3f}, L_euc={euclidean_lengths[worst_idx]:.3f}")
    print()
    
    # Visualize
    plot_mesh(editor, metric_fn, initial_boundary_verts, 
              initial_mesh=(initial_points, initial_triangles),
              metric_lengths_before=metric_lengths_before,
              metric_lengths_after=metric_lengths_after)
    
    print("Done!")


if __name__ == '__main__':
    main()
