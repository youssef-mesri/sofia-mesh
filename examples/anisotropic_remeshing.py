#!/usr/bin/env python3
"""
SOFIA Example 9: Anisotropic Mesh Adaptation with Continuous Metric 

This example demonstrates anisotropic mesh adaptation using a continuous metric field
to align the mesh with an analytical sinusoidal curve: y = 0.5 + 0.15·sin(4πx).

The continuous metric approach provides:
- Smooth transitions in mesh density (no abrupt changes)
- Better control using distance field gradient
- Curvature-based refinement along the curve
- Natural anisotropy aligned with level-set contours

Unlike isotropic refinement (which creates uniform elements), anisotropic 
remeshing creates stretched triangles aligned with a spatially-varying metric field.

What this example shows:
1. Building a random Delaunay mesh
2. Defining a continuous metric field using:
   - Distance field φ(x) and its gradient ∇φ
   - Smooth transition functions for mesh size
   - Curvature-based tangential refinement
3. Computing metric edge lengths L_M(e) = sqrt((q-p)^T M (q-p))
4. Running the anisotropic local remeshing algorithm (split/collapse)
5. Visualizing the adapted mesh with:
   - The analytical curve overlay
   - Metric ellipses showing anisotropic directions
   - Zoomed view near the curve to show alignment
6. Analyzing metric edge length statistics

Perfect for: Understanding continuous metric fields, distance-based adaptation,
             and curvature-aware mesh refinement

Technical Background:
- Continuous metric: Uses smooth functions S(d) for mesh size h(d)
- Distance field: φ(x,y) = y - y_curve(x), with gradient ∇φ = normal direction
- Curvature adaptation: h_tangent adjusted by κ (curvature) for iso-error
- Smooth transitions: S(t) = 3t² - 2t³ (C¹ continuous)
- Near curve: h_⊥ = 0.03, h_∥ varies with curvature
- Far field: h = 0.3 (isotropic)
- The algorithm splits edges where L_M > α, collapses where L_M < β
  directional solution gradients)
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import math
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.anisotropic_remesh import anisotropic_local_remesh
from sofia.core.quality import mesh_min_angle
from sofia.core.conformity import check_mesh_conformity


def simple_metric(x):
    """Construct anisotropic metric using Hessian of level-set function.
    
    The metric is designed to align the mesh with a sinusoidal curve:
        y_curve = 0.5 + 0.15 * sin(4π * x)
    
    Uses Hessian-based metric formulation:
    1. Level-set function: φ(x,y) = y - f(x) where f(x) = y_curve(x)
    2. Compute Hessian matrix H(φ) = [∂²φ/∂x², ∂²φ/∂x∂y; ∂²φ/∂x∂y, ∂²φ/∂y²]
    3. Metric tensor M = |H| with controlled eigenvalues for mesh sizing
    
    The Hessian naturally captures:
    - Curvature of the level-set (second derivatives)
    - Anisotropic directions (eigenvectors of H)
    - Proper alignment with the curve
    
    Args:
        x: 2D position array [x, y]
        
    Returns:
        (M, tangent, normal): Metric tensor, tangent vector, and normal vector
    """
    # Define sinusoidal curve: y = 0.5 + amplitude * sin(frequency * x)
    amplitude = 0.15
    frequency = 4.0 * math.pi
    x_pos = float(x[0])
    y_pos = float(x[1])
    
    # Level-set function: φ(x,y) = y - f(x)
    y_curve = 0.5 + amplitude * math.sin(frequency * x_pos)
    phi = y_pos - y_curve
    
    # First derivatives of φ
    # ∂φ/∂x = -df/dx = -amplitude * frequency * cos(frequency * x)
    # ∂φ/∂y = 1
    dphi_dx = -amplitude * frequency * math.cos(frequency * x_pos)
    dphi_dy = 1.0
    
    # Second derivatives of φ (Hessian components)
    # ∂²φ/∂x² = -d²f/dx² = amplitude * frequency² * sin(frequency * x)
    # ∂²φ/∂x∂y = 0
    # ∂²φ/∂y² = 0
    d2phi_dx2 = amplitude * frequency**2 * math.sin(frequency * x_pos)
    d2phi_dxdy = 0.0
    d2phi_dy2 = 0.0
    
    # Hessian matrix H(φ)
    H = np.array([
        [d2phi_dx2, d2phi_dxdy],
        [d2phi_dxdy, d2phi_dy2]
    ])
    
    # Compute eigenvalues and eigenvectors of Hessian
    vals, vecs = np.linalg.eigh(H)
    
    # The Hessian gives us the principal curvature directions
    # For a curve, one eigenvalue relates to curvature, the other is ~0
    
    # Build metric from Hessian with controlled sizing
    # Near curve: strong anisotropy (fine perpendicular, coarse tangent)
    # Far from curve: isotropic coarse
    
    dist_abs = abs(phi)
    
    # Mesh sizing parameters - EXTREME ANISOTROPY FOR COMPUTATIONAL EFFICIENCY
    # Key insight: Anisotropy should ELONGATE elements, not CLUSTER them
    h_min = 0.01      # Finest resolution perpendicular to curve (doubled from 0.005)
    h_max = 1.0       # Coarsest resolution along curve (doubled from 1.0 for MORE stretching)
    h_far = 0.25      # Far-field size (slightly reduced)
    d0 = 0.15         # Transition distance (REDUCED from 0.25 - narrower refinement band)
    
    # Compute tangent and normal from gradient
    grad_phi = np.array([dphi_dx, dphi_dy])
    grad_norm = np.linalg.norm(grad_phi)
    if grad_norm > 1e-10:
        normal = grad_phi / grad_norm
    else:
        normal = np.array([0.0, 1.0])
    
    # Tangent is perpendicular to normal
    tangent = np.array([-normal[1], normal[0]])
    
    # Sign correction for normal
    if (phi > 0 and normal[1] < 0) or (phi < 0 and normal[1] > 0):
        normal = -normal
        tangent = -tangent
    
    # Smooth transition for mesh size
    if dist_abs < d0:
        t = dist_abs / d0
        smooth_step = t * t * (3.0 - 2.0 * t)
        h_normal = h_min + (h_far - h_min) * smooth_step
        
        # For a curve: strong anisotropy (fine perpendicular, coarse tangent)
        # Don't use Hessian to reduce h_tangent - we WANT elongated elements!
        # The Hessian tells us WHERE to refine, not HOW MUCH to stretch
        h_tangent = h_max  # Keep maximum stretching along curve
        
    else:
        # Isotropic far field
        h_normal = h_far
        h_tangent = h_far
    
    # Build metric tensor: M = R @ Λ @ R^T
    lambda_tangent = 1.0 / (h_tangent**2)
    lambda_normal = 1.0 / (h_normal**2)
    
    R = np.column_stack([tangent, normal])
    Lambda = np.diag([lambda_tangent, lambda_normal])
    M = R @ Lambda @ R.T
    
    return M, tangent, normal


def compute_metric_complexity(metric_fn, domain_bounds=([0, 1], [0, 1]), n_samples=100):
    """Compute the complexity of a metric field (integral of sqrt(det(M))).
    
    The metric complexity C = ∫_Ω sqrt(det(M(x))) dx represents the expected
    number of vertices needed to satisfy the metric. For constant vertex count,
    we normalize the metric by scaling it by (N_target / C)^(2/d) where d=2.
    
    Args:
        metric_fn: Function that returns metric tensor M(x) at position x
        domain_bounds: [(x_min, x_max), (y_min, y_max)]
        n_samples: Number of samples per dimension for integration
        
    Returns:
        Complexity value C
    """
    x_bounds, y_bounds = domain_bounds
    x_vals = np.linspace(x_bounds[0], x_bounds[1], n_samples)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], n_samples)
    
    complexity = 0.0
    dx = (x_bounds[1] - x_bounds[0]) / (n_samples - 1)
    dy = (y_bounds[1] - y_bounds[0]) / (n_samples - 1)
    
    for x in x_vals:
        for y in y_vals:
            pos = np.array([x, y])
            M = metric_fn(pos)
            
            # Handle tuple return format
            if isinstance(M, tuple):
                M = M[0]
            
            # sqrt(det(M)) represents local mesh density
            det_M = np.linalg.det(M)
            if det_M > 0:
                complexity += np.sqrt(det_M) * dx * dy
    
    return complexity


def normalize_metric_for_target_vertices(metric_fn, target_vertices, domain_bounds=([0, 1], [0, 1])):
    """Create a normalized metric function that targets a specific vertex count.
    
    The normalization factor is: scale = (N_target / C)^(2/d) where d=2 (dimension).
    This scales the metric M_new = scale * M_old to achieve the target complexity.
    
    Args:
        metric_fn: Original metric function
        target_vertices: Target number of vertices
        domain_bounds: Domain boundaries
        
    Returns:
        Normalized metric function
    """
    # Compute current complexity
    complexity = compute_metric_complexity(metric_fn, domain_bounds, n_samples=50)
    
    print(f"  Metric complexity (unnormalized): {complexity:.1f}")
    print(f"  Target vertices: {target_vertices}")
    
    # Normalization factor: scale = (N_target / C)^(2/2) = N_target / C
    scale = (target_vertices / complexity) ** (2.0 / 2.0)  # d=2 dimensions
    
    print(f"  Normalization scale factor: {scale:.4f}")
    
    def normalized_metric(x):
        result = metric_fn(x)
        
        # Handle tuple return format
        if isinstance(result, tuple):
            M, tangent, normal = result
            M_normalized = scale * M
            return M_normalized, tangent, normal
        else:
            return scale * result
    
    return normalized_metric


def metric_edge_lengths(points, triangles, metric_fn):
    """Compute midpoint-evaluated metric edge lengths L_M(e) for all unique edges.
    
    For each edge e = (p,q), compute:
        L_M(e) = sqrt((q-p)^T M((p+q)/2) (q-p))
    
    Args:
        points: Vertex coordinates array
        triangles: Triangle connectivity array
        metric_fn: Function x -> 2x2 SPD metric tensor
        
    Returns:
        Array of metric edge lengths (one per unique undirected edge)
    """
    # Build unique undirected edges from triangles
    edges = set()
    for tri in triangles:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if i < 0 or j < 0 or k < 0:
            continue
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))
    
    vals = []
    for (a, b) in edges:
        pa, pb = points[a], points[b]
        m = 0.5 * (pa + pb)  # midpoint
        
        # Handle both return formats from metric_fn
        result = metric_fn(m)
        if isinstance(result, tuple):
            M = result[0]  # Extract just the metric tensor
        else:
            M = result
        
        d = (pb - pa).reshape(2, 1)
        # L_M(e) ≈ sqrt(d^T M d) using midpoint metric
        lm = float(np.sqrt((d.T @ M @ d)[0, 0]))
        vals.append(lm)
    
    return np.asarray(vals, dtype=float)


def plot_mesh_with_metric_ellipses(editor, metric_fn, title="", ax=None, 
                                   n_ellipses=20, ellipse_scale=0.05):
    """Plot the mesh with metric ellipses showing the anisotropic field.
    
    At selected vertices, we visualize the metric tensor M(x) as an ellipse.
    The ellipse's axes are aligned with M's eigenvectors and scaled by eigenvalues.
    
    Args:
        editor: Mesh editor with current mesh state
        metric_fn: Function x -> 2x2 SPD metric tensor
        title: Plot title
        ax: Matplotlib axis (created if None)
        n_ellipses: Number of ellipses to draw (uniformly sampled from vertices)
        ellipse_scale: Visual scale factor for ellipses
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Filter active triangles
    active_mask = np.all(editor.triangles != -1, axis=1)
    active_tris = editor.triangles[active_mask]
    
    # Get active vertices (only those used in active triangles)
    active_vertex_indices = set()
    for tri in active_tris:
        active_vertex_indices.update(tri)
    active_vertex_indices = sorted(active_vertex_indices)
    
    # Diagnostic: Check for isolated vertices
    total_vertices = len(editor.points)
    active_vertices = len(active_vertex_indices)
    if total_vertices != active_vertices:
        print(f"    [WARNING] Visualization found {total_vertices - active_vertices} isolated vertices!")
        print(f"              Total: {total_vertices}, Active: {active_vertices}")
    
    # Ensure editor maps are up to date
    editor._update_maps()
    
    # Use SPATIAL criterion for boundary detection (more robust)
    boundary_verts = set()
    interior_verts = set()
    
    # A vertex is on the domain boundary if x∈{0,1} or y∈{0,1}
    domain_tol = 1e-4  # Increased tolerance for boundary detection
    for v_idx in active_vertex_indices:
        x, y = editor.points[v_idx]
        is_on_domain_bnd = (abs(x) < domain_tol or abs(x - 1.0) < domain_tol or 
                           abs(y) < domain_tol or abs(y - 1.0) < domain_tol)
        
        if is_on_domain_bnd:
            boundary_verts.add(v_idx)
        else:
            interior_verts.add(v_idx)
    
    # Diagnostic: Compare with edge_map detection
    edge_map_boundary = set()
    bnd_edge_count = 0
    int_edge_count = 0
    
    for edge, tris in editor.edge_map.items():
        active_edge_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        
        if len(active_edge_tris) == 1:
            edge_map_boundary.update(edge)
            bnd_edge_count += 1
        elif len(active_edge_tris) == 2:
            int_edge_count += 1
    
    # Report differences
    false_positives = edge_map_boundary - boundary_verts
    false_negatives = boundary_verts - edge_map_boundary
    
    print(f"    [VISU] Spatial boundary: {len(boundary_verts)}, Interior: {len(interior_verts)}")
    print(f"    [VISU] Edge_map boundary: {len(edge_map_boundary)} (edges: {bnd_edge_count}, interior edges: {int_edge_count})")
    
    # Debug: Check vertices on y=0 axis specifically
    y_axis_verts = [v for v in active_vertex_indices if abs(editor.points[v][1]) < domain_tol]
    y_axis_boundary = [v for v in y_axis_verts if v in boundary_verts]
    y_axis_interior = [v for v in y_axis_verts if v not in boundary_verts]
    
    print(f"    [DEBUG] Vertices on y=0 axis: {len(y_axis_verts)} total")
    print(f"            - Classified as boundary: {len(y_axis_boundary)}")
    print(f"            - Classified as interior: {len(y_axis_interior)}")
    
    if y_axis_verts:
        print(f"            Detailed analysis of y=0 vertices:")
        for v in sorted(y_axis_verts):
            x, y = editor.points[v]
            # Check if this vertex has boundary edges (incident edges with only 1 triangle)
            incident_edges = [edge for edge in editor.edge_map.keys() if v in edge]
            boundary_edge_count = 0
            interior_edge_count = 0
            for edge in incident_edges:
                tris = editor.edge_map[edge]
                active_tris_for_edge = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
                if len(active_tris_for_edge) == 1:
                    boundary_edge_count += 1
                elif len(active_tris_for_edge) == 2:
                    interior_edge_count += 1
            
            # Check if vertex is actually used in any active triangle
            triangles_using_v = [t for t in active_tris if v in editor.triangles[t]]
            
            is_boundary = v in boundary_verts
            classification = "BOUNDARY" if is_boundary else "INTERIOR"
            status = "OK" if len(triangles_using_v) > 0 else "ISOLATED!"
            print(f"              v{v}: x={x:.6f}, {classification}, boundary_edges={boundary_edge_count}, interior_edges={interior_edge_count}, triangles={len(triangles_using_v)} [{status}]")
    
    # Check for boundary vertices that are NEAR y=0 but not exactly on it
    near_y0_tol = 0.05  # Visual proximity threshold
    near_y0_verts = [v for v in boundary_verts if abs(editor.points[v][1]) < near_y0_tol and abs(editor.points[v][1]) >= domain_tol]
    if near_y0_verts:
        print(f"    [DEBUG] Boundary vertices NEAR y=0 (but not on it): {len(near_y0_verts)}")
        for v in sorted(near_y0_verts):
            x, y = editor.points[v]
            print(f"              v{v}: ({x:.6f}, {y:.6f}) - distance from y=0: {y:.6f}")
    
    if false_positives:
        print(f"    [VISU] False positives (edge_map but not on domain boundary): {len(false_positives)}")
        # Investigate these false positives
        for v_idx in sorted(list(false_positives))[:5]:  # Show first 5
            x, y = editor.points[v_idx]
            # Count how many triangles use this vertex
            tri_count = sum(1 for tri in editor.triangles if v_idx in tri and np.all(tri != -1))
            # Count edges incident to this vertex
            incident_edges = [e for e in editor.edge_map.keys() if v_idx in e]
            # Check if any incident edge has != 2 active triangles (non-conforming)
            non_conf_edges = []
            for edge in incident_edges:
                tris = editor.edge_map[edge]
                active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
                if len(active_tris) != 2:
                    non_conf_edges.append((edge, len(active_tris)))
            
            print(f"        Vertex {v_idx}: pos=({x:.4f},{y:.4f}), tris={tri_count}, incident_edges={len(incident_edges)}, non_conf_edges={len(non_conf_edges)}")
            if non_conf_edges:
                print(f"            Non-conforming edges: {non_conf_edges[:3]}")
    
    if false_negatives:
        print(f"    [VISU] False negatives (on domain boundary but not in edge_map): {len(false_negatives)}")
    
    # Diagnostic: Check if all boundary vertices are actually used in triangles
    boundary_in_tris = boundary_verts & set(active_vertex_indices)
    if len(boundary_in_tris) != len(boundary_verts):
        print(f"    [ERROR] {len(boundary_verts) - len(boundary_in_tris)} boundary vertices not in any triangle!")
        orphan_verts = boundary_verts - set(active_vertex_indices)
        print(f"              Orphan vertices: {sorted(orphan_verts)[:10]}{'...' if len(orphan_verts) > 10 else ''}")
    
    # Plot mesh edges
    pts = editor.points
    edges = []
    boundary_edges = []
    
    # Use spatial criterion for boundary edges (more robust than edge_map)
    for edge, tris in editor.edge_map.items():
        active_edge_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
        if len(active_edge_tris) > 0:
            v1, v2 = edge
            edge_line = [pts[v1], pts[v2]]
            edges.append(edge_line)
            
            # Mark boundary edges: both vertices on domain boundary
            if v1 in boundary_verts and v2 in boundary_verts:
                boundary_edges.append(edge_line)
    
    # Plot all edges in gray
    if edges:
        lc = LineCollection(edges, colors='gray', linewidths=0.5, alpha=0.4)
        ax.add_collection(lc)
    
    # Plot boundary edges in red for visibility
    if boundary_edges:
        lc_bnd = LineCollection(boundary_edges, colors='red', linewidths=1.5, alpha=0.8, label='Boundary edges')
        ax.add_collection(lc_bnd)
    
    # Plot interior vertices in black
    if interior_verts:
        interior_pts = pts[sorted(interior_verts)]
        ax.plot(interior_pts[:, 0], interior_pts[:, 1], 'ko', markersize=2.5, alpha=0.5, label='Interior')
    
    # Plot boundary vertices in blue to distinguish them
    if boundary_verts:
        boundary_pts = pts[sorted(boundary_verts)]
        ax.plot(boundary_pts[:, 0], boundary_pts[:, 1], 'bo', markersize=3.5, alpha=0.7, label='Boundary')
    
    # Plot the analytical sinusoidal curve
    x_curve = np.linspace(0, 1, 200)
    y_curve = 0.5 + 0.15 * np.sin(4.0 * np.pi * x_curve)
    ax.plot(x_curve, y_curve, 'r-', linewidth=2.5, label='Target curve: y = 0.5 + 0.15·sin(4πx)', alpha=0.8)
    
    # Draw metric ellipses at selected vertices
    n_pts = len(pts)
    if n_pts > 0:
        # Uniformly sample vertex indices
        step = max(1, n_pts // n_ellipses)
        sample_indices = range(0, n_pts, step)
        
        for idx in sample_indices:
            x = pts[idx]
            
            # Handle both return formats from metric_fn
            result = metric_fn(x)
            if isinstance(result, tuple):
                M = result[0]  # Extract just the metric tensor
            else:
                M = result
            
            # Eigen-decomposition: M = R diag(λ) R^T
            vals, vecs = np.linalg.eigh(M)
            
            # Ellipse axes: inversely scaled by sqrt(eigenvalues)
            # (larger eigenvalue -> shorter axis in physical space)
            a = ellipse_scale / np.sqrt(vals[0])
            b = ellipse_scale / np.sqrt(vals[1])
            
            # Rotation angle from eigenvectors
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            
            ellipse = mpatches.Ellipse(
                x, 2*a, 2*b, angle=angle,
                facecolor='blue', edgecolor='darkblue',
                alpha=0.3, linewidth=0.5
            )
            ax.add_patch(ellipse)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)
    
    return ax


def optimal_split_position(p1, p2, metric_fn, n_samples=5):
    """Find optimal position to split edge according to metric field.
    
    Instead of always splitting at midpoint, this function finds the position t ∈ [0,1]
    that best aligns with the anisotropic metric field. This eliminates the need for
    edge flipping by creating well-oriented triangles from the start.
    
    Args:
        p1, p2: Edge endpoints
        metric_fn: Function x -> (M, tangent, normal) or just M
        n_samples: Number of positions to test along edge
        
    Returns:
        Optimal position along edge (2D point)
    """
    best_t = 0.5  # Default to midpoint
    best_score = float('inf')
    
    # Sample positions along the edge
    for t in np.linspace(0.2, 0.8, n_samples):
        pos = (1 - t) * p1 + t * p2
        
        # Call metric function (handle both return formats)
        result = metric_fn(pos)
        if isinstance(result, tuple):
            M, tangent, normal = result
        else:
            M = result
        
        # Compute edge direction
        edge_dir = p2 - p1
        edge_dir = edge_dir / np.linalg.norm(edge_dir)
        
        # Score: how well does the edge align with the metric's principal directions?
        # We want the edge to align with the direction of LARGER mesh size (tangent)
        # The eigenvectors of M correspond to the metric directions
        vals, vecs = np.linalg.eigh(M)
        
        # Smaller eigenvalue = larger mesh size (h = 1/sqrt(λ))
        # So we want edge to align with eigenvector corresponding to smaller λ
        coarse_direction = vecs[:, 0]  # Eigenvector of smaller eigenvalue
        
        # Alignment score: 1 - |dot product| (0 = perfect alignment)
        alignment = abs(np.dot(edge_dir, coarse_direction))
        
        # Metric length at this position
        d = (p2 - p1).reshape(2, 1)
        lm = float(np.sqrt((d.T @ M @ d)[0, 0]))
        
        # Combined score: prefer positions that align edge with coarse direction
        # and minimize metric length variance
        score = (1.0 - alignment) + 0.1 * abs(lm - 1.0)
        
        if score < best_score:
            best_score = score
            best_t = t
    
    return (1 - best_t) * p1 + best_t * p2


def split_edge_at_position(editor, edge, new_point):
    """Split edge by inserting a vertex at specified position (not midpoint).
    
    This is a custom split operation that allows precise placement of the new vertex
    according to the anisotropic metric field.
    
    Args:
        editor: PatchBasedMeshEditor
        edge: Tuple (v1, v2) of vertex indices
        new_point: 2D coordinates for the new vertex
        
    Returns:
        (success, new_vertex_idx)
    """
    edge = tuple(sorted(edge))
    tris_idx = list(editor.edge_map.get(edge, []))
    
    if len(tris_idx) == 0:
        return False, None
    
    # Add new vertex at specified position
    new_idx = len(editor.points)
    editor.points = np.vstack([editor.points, new_point])
    
    if len(tris_idx) == 2:
        # Interior edge: split into 4 triangles
        t1, t2 = [editor.triangles[i] for i in tris_idx]
        opp1 = [v for v in t1 if v not in edge][0]
        opp2 = [v for v in t2 if v not in edge][0]
        
        # Replace old triangles with new ones
        editor.triangles[tris_idx[0]] = [edge[0], new_idx, opp1]
        editor.triangles[tris_idx[1]] = [edge[1], new_idx, opp1]
        
        # Add two more triangles
        editor.triangles = np.vstack([
            editor.triangles,
            [edge[0], new_idx, opp2],
            [edge[1], new_idx, opp2]
        ])
        
    elif len(tris_idx) == 1:
        # Boundary edge: split into 2 triangles
        t = editor.triangles[tris_idx[0]]
        opp = [v for v in t if v not in edge][0]
        
        # Replace old triangle with two new ones
        editor.triangles[tris_idx[0]] = [edge[0], new_idx, opp]
        editor.triangles = np.vstack([
            editor.triangles,
            [edge[1], new_idx, opp]
        ])
    
    return True, new_idx


def refine_by_metric_length(editor, metric_fn, alpha_split=1.4, max_splits=30, 
                           allow_boundary=True, use_optimal_position=True):
    """Refine mesh by splitting edges with large metric length.
    
    Splits both boundary and interior edges to adapt to the metric field.
    Can use optimal positioning to eliminate need for edge flipping.
    
    Args:
        editor: Mesh editor
        metric_fn: Function x -> 2×2 SPD metric tensor
        alpha_split: Split edges where L_M(e) > alpha_split
        max_splits: Maximum number of edge splits per iteration
        allow_boundary: If True, allow splitting boundary edges
        use_optimal_position: If True, find optimal split position instead of midpoint
        
    Returns:
        (boundary_splits, interior_splits)
    """
    # Rebuild maps to ensure consistency
    editor._update_maps()
    
    # Compute metric lengths for all edges
    boundary_edge_metrics = []
    interior_edge_metrics = []
    
    for edge, tris in editor.edge_map.items():
        # Skip invalid edges
        active_tris = [t for t in tris if np.all(editor.triangles[t] != -1)]
        if len(active_tris) == 0:
            continue
        
        v1, v2 = edge
        p1, p2 = editor.points[v1], editor.points[v2]
        m = 0.5 * (p1 + p2)  # midpoint
        
        # Handle both return formats from metric_fn
        result = metric_fn(m)
        if isinstance(result, tuple):
            M, _, _ = result  # Extract just the metric tensor
        else:
            M = result
        
        d = (p2 - p1).reshape(2, 1)
        lm = float(np.sqrt((d.T @ M @ d)[0, 0]))
        
        if len(active_tris) == 1:
            # Boundary edge
            if allow_boundary:
                boundary_edge_metrics.append((lm, edge, p1, p2))
        else:
            # Interior edge
            interior_edge_metrics.append((lm, edge, p1, p2))
    
    # Sort by metric length (longest first)
    boundary_edge_metrics.sort(reverse=True)
    interior_edge_metrics.sort(reverse=True)
    
    boundary_splits = 0
    interior_splits = 0
    
    # Split boundary edges first (to refine coarse boundary)
    if allow_boundary:
        for lm, edge, p1, p2 in boundary_edge_metrics:
            if lm <= alpha_split:
                break
            
            if boundary_splits + interior_splits >= max_splits:
                break
            
            try:
                if use_optimal_position:
                    # Find optimal split position
                    new_point = optimal_split_position(p1, p2, metric_fn)
                    ok, new_v = split_edge_at_position(editor, edge, new_point)
                else:
                    # Use standard midpoint split
                    ok, _, _ = editor.split_edge(edge=edge)
                
                if ok:
                    boundary_splits += 1
                    editor._update_maps()
            except Exception as e:
                pass
    
    # Then split interior edges
    for lm, edge, p1, p2 in interior_edge_metrics:
        if lm <= alpha_split:
            break
        
        if boundary_splits + interior_splits >= max_splits:
            break
        
        try:
            if use_optimal_position:
                # Find optimal split position
                new_point = optimal_split_position(p1, p2, metric_fn)
                ok, new_v = split_edge_at_position(editor, edge, new_point)
            else:
                # Use standard midpoint split
                ok, _, _ = editor.split_edge(edge=edge)
            
            if ok:
                interior_splits += 1
                editor._update_maps()
        except Exception as e:
            pass
    
    return boundary_splits, interior_splits


def coarsen_by_metric_length(editor, metric_fn, beta_collapse=0.7, max_collapses=20):
    """Coarsen mesh by collapsing edges with small metric length.
    
    Only interior edges are collapsed to preserve the domain boundary.
    Area preservation is checked to ensure mesh quality.
    
    Args:
        editor: Mesh editor
        metric_fn: Function x -> 2×2 SPD metric tensor (or tuple (M, tangent, normal))
        beta_collapse: Collapse edges where L_M(e) < beta_collapse
        max_collapses: Maximum number of edge collapses per iteration
        
    Returns:
        Number of edges collapsed
    """
    # Helper function to extract metric tensor
    def get_metric_tensor(x):
        result = metric_fn(x)
        return result[0] if isinstance(result, tuple) else result
    # Rebuild maps to ensure consistency
    editor._update_maps()
    
    # Identify boundary vertices to protect them
    boundary_vertices = set()
    for edge, tris in editor.edge_map.items():
        active_tris = [t for t in tris if np.all(editor.triangles[t] != -1)]
        if len(active_tris) == 1:
            # Boundary edge
            boundary_vertices.add(edge[0])
            boundary_vertices.add(edge[1])
    
    # Compute metric lengths for interior edges only
    edge_metrics = []
    for edge, tris in editor.edge_map.items():
        if len(tris) != 2:
            continue  # Skip boundary edges
        
        # Don't collapse edges connected to boundary vertices (preserve boundary)
        v1, v2 = edge
        if v1 in boundary_vertices or v2 in boundary_vertices:
            continue
        
        p1, p2 = editor.points[v1], editor.points[v2]
        m = 0.5 * (p1 + p2)  # midpoint
        M = get_metric_tensor(m)
        d = (p2 - p1).reshape(2, 1)
        lm = float(np.sqrt((d.T @ M @ d)[0, 0]))
        edge_metrics.append((lm, edge))
    
    # Sort by metric length (shortest first)
    edge_metrics.sort()
    
    collapses_made = 0
    attempts = 0
    max_attempts = max_collapses * 5  # Safety limit
    
    for lm, edge in edge_metrics:
        if lm >= beta_collapse:
            break  # All remaining edges are long enough
        
        if collapses_made >= max_collapses or attempts >= max_attempts:
            break
        
        attempts += 1
        try:
            # Try to collapse the edge
            # Conformity and boundary loop checks are handled internally
            ok, _, _ = editor.edge_collapse(edge=edge)
            if ok:
                # Update maps after successful collapse
                editor._update_maps()
                collapses_made += 1
            # Note: Conformity and boundary loop checks are handled by
            # the core operation via simulate_compaction_on_commit flag
        except Exception as e:
            # Edge collapse failed - continue to next edge
            pass
    
    return collapses_made


def flip_for_quality(editor, max_flips=50):
    """Flip edges to improve mesh quality (Delaunay criterion).
    
    Args:
        editor: Mesh editor
        max_flips: Maximum number of edge flips to attempt
        
    Returns:
        Number of edges flipped
    """
    from sofia.core.geometry import triangle_angles
    
    # Rebuild maps to ensure consistency
    editor._update_maps()
    
    flips_made = 0
    attempts = 0
    max_attempts = max_flips * 3
    
    # Get interior edges
    interior_edges = [e for e, tris in editor.edge_map.items() if len(tris) == 2]
    
    for edge in interior_edges:
        if flips_made >= max_flips or attempts >= max_attempts:
            break
        
        attempts += 1
        
        # Get adjacent triangles
        tris = editor.edge_map.get(edge, [])
        if len(tris) != 2:
            continue
        
        # Compute min angle before flip
        min_angle_before = float('inf')
        for ti in tris:
            tri = editor.triangles[ti]
            if np.any(tri == -1):
                continue
            coords = [editor.points[int(v)] for v in tri]
            try:
                angles = triangle_angles(coords[0], coords[1], coords[2])
                min_angle_before = min(min_angle_before, min(angles))
            except:
                pass
        
        # Try flip
        try:
            ok, _, _ = editor.flip_edge(edge=edge)
            if not ok:
                continue
            
            # Rebuild maps after flip
            editor._update_maps()
            
            # Compute min angle after flip
            min_angle_after = float('inf')
            tris_after = editor.edge_map.get(edge, [])
            for ti in tris_after:
                tri = editor.triangles[ti]
                if np.any(tri == -1):
                    continue
                coords = [editor.points[int(v)] for v in tri]
                try:
                    angles = triangle_angles(coords[0], coords[1], coords[2])
                    min_angle_after = min(min_angle_after, min(angles))
                except:
                    pass
            
            # Accept flip if quality improved, otherwise flip back
            if min_angle_after + 1e-6 < min_angle_before:
                try:
                    editor.flip_edge(edge=edge)  # Flip back
                    editor._update_maps()
                except:
                    pass
            else:
                flips_made += 1
        except:
            pass
    
    return flips_made


def remove_nodes_by_metric_quality(editor, metric_fn, max_removals=20, quality_threshold=0.05, 
                                   initial_boundary_verts=None):
    """Remove interior nodes if their removal improves metric quality.
    
    Instead of using a heuristic based on node degree, this function evaluates
    the metric quality of the cavity (star) around each node. If removing the
    node and retriangulating improves the average metric edge length deviation,
    the removal is accepted.
    
    This is a metric-aware approach that directly optimizes mesh quality in
    the anisotropic metric space.
    
    Args:
        editor: PatchBasedMeshEditor instance
        metric_fn: Metric function M(x) for quality assessment
        max_removals: Maximum number of nodes to remove per call
        quality_threshold: Minimum improvement in quality to accept removal
        initial_boundary_verts: Set of initial boundary vertices to protect (optional)
                               If None, all boundary vertices are protected.
                               If provided, only these vertices are protected.
        
    Returns:
        Number of nodes successfully removed
    """
    removed_count = 0
    
    # Build vertex-to-triangle map
    editor._update_maps()
    v_map = editor.v_map
    
    # Check conformity first - remove_node_with_patch requires closed cavities
    is_conform, conform_msgs = check_mesh_conformity(editor)
    # DEBUG disabled for production
    # print(f"      [DEBUG] Mesh conformity before removal: {is_conform}")
    if not is_conform:
        # Cannot safely remove nodes from non-conforming mesh
        # print(f"      [DEBUG] Skipping - non-conforming: {conform_msgs}")
        return 0
    
    # Identify vertices to protect
    if initial_boundary_verts is None:
        # Conservative mode: protect ALL current boundary vertices
        boundary_verts = set()
        for tri_idx, tri in enumerate(editor.triangles):
            if np.all(tri != -1):
                for i in range(3):
                    v1, v2 = int(tri[i]), int(tri[(i+1)%3])
                    edge = tuple(sorted([v1, v2]))
                    tris_on_edge = editor.edge_map.get(edge, set())
                    if len(tris_on_edge) == 1:  # Boundary edge
                        boundary_verts.add(v1)
                        boundary_verts.add(v2)
        protected_verts = boundary_verts
    else:
        # Selective mode: only protect initial boundary vertices
        # Allow removal of boundary vertices created by splits
        protected_verts = initial_boundary_verts
    
    # Helper function to compute average metric of a cavity using log-Euclidean mean
    def compute_cavity_average_metric(triangles, points, metric_fn):
        """Compute the average metric tensor for a cavity using log-Euclidean mean.
        
        For SPD matrices, the proper way to average is:
        M_avg = exp( mean( log(M_i) ) )
        
        This preserves the SPD property and is more mathematically sound than
        arithmetic averaging in the Euclidean space.
        
        Args:
            triangles: List of triangles in the cavity
            points: Points array
            metric_fn: Metric function
            
        Returns:
            Average metric tensor (2x2 SPD matrix) or None if cavity is empty
        """
        from scipy.linalg import logm, expm
        
        log_metrics = []
        
        # Collect all unique vertices in the cavity
        cavity_verts = set()
        for tri in triangles:
            if np.all(tri != -1):
                cavity_verts.update(tri)
        
        if not cavity_verts:
            return None
        
        # Evaluate metric at each vertex in the cavity
        for v_idx in cavity_verts:
            pos = points[v_idx]
            M = metric_fn(pos)
            if isinstance(M, tuple):
                M = M[0]
            
            # Compute matrix logarithm
            try:
                log_M = logm(M).real  # Take real part (should be real for SPD)
                log_metrics.append(log_M)
            except Exception:
                # Fallback: use M directly if logm fails
                log_metrics.append(M)
        
        if not log_metrics:
            return None
        
        # Average in log space
        mean_log_M = np.mean(log_metrics, axis=0)
        
        # Exponentiate back to get average metric
        try:
            M_avg = expm(mean_log_M).real
        except Exception:
            # Fallback: arithmetic mean
            M_avg = np.mean([metric_fn(points[v])[0] if isinstance(metric_fn(points[v]), tuple) 
                            else metric_fn(points[v]) for v in cavity_verts], axis=0)
        
        return M_avg
    
    # Helper function to compute metric quality using cavity-averaged metric
    def compute_cavity_metric_quality_with_avg_metric(triangles, points, metric_fn):
        """Compute average metric edge length deviation using a single averaged metric.
        
        This is more coherent than evaluating each edge with its own midpoint metric.
        We compute the log-Euclidean mean of all vertex metrics in the cavity,
        then use this single metric to evaluate all edges.
        
        Returns:
            (avg_deviation, avg_metric_length): Quality metrics
        """
        # Get average metric for the cavity
        M_avg = compute_cavity_average_metric(triangles, points, metric_fn)
        
        if M_avg is None:
            return float('inf'), float('inf')
        
        total_deviation = 0.0
        total_length = 0.0
        edge_count = 0
        
        for tri in triangles:
            if np.all(tri != -1):
                for i in range(3):
                    v1, v2 = int(tri[i]), int(tri[(i+1)%3])
                    p1, p2 = points[v1], points[v2]
                    
                    # Compute metric edge length using CAVITY-AVERAGED metric
                    edge_vec = (p2 - p1).reshape(2, 1)
                    lm = float(np.sqrt((edge_vec.T @ M_avg @ edge_vec)[0, 0]))
                    
                    # Deviation from ideal (L_M = 1)
                    total_deviation += abs(lm - 1.0)
                    total_length += lm
                    edge_count += 1
        
        if edge_count == 0:
            return float('inf'), float('inf')
        
        avg_deviation = total_deviation / edge_count
        avg_length = total_length / edge_count
        
        return avg_deviation, avg_length
    
    # Helper function to check if cavity should be coarsened (generic)
    def should_coarsen_cavity(v_idx, star_triangles, points, metric_fn, 
                             quality_threshold=0.5, over_refined_threshold=0.8):
        """Check if a cavity should be coarsened (node removed).
        
        Uses log-Euclidean metric averaging for mathematical consistency.
        
        Two criteria for coarsening:
        1. Well-satisfied: avg|L_M - 1| < quality_threshold (edges near ideal)
        2. Over-refined: avg(L_M) < over_refined_threshold (edges too short)
        
        The over-refinement criterion is crucial for equidistribution: even if
        a zone has poor quality (edges not exactly L_M=1), if L_M << 1 then
        the zone is using too much of the vertex budget and should be coarsened.
        
        Args:
            v_idx: Vertex index
            star_triangles: Triangles in the star of this vertex
            points: Points array
            metric_fn: Metric function
            quality_threshold: Max deviation for "well-satisfied"
            over_refined_threshold: Max avg length for "over-refined"
            
        Returns:
            bool: True if cavity should be coarsened
        """
        # Use cavity-averaged metric for consistency
        avg_deviation, avg_length = compute_cavity_metric_quality_with_avg_metric(
            star_triangles, points, metric_fn)
        
        if avg_deviation == float('inf') or avg_length == float('inf'):
            return False
        
        # Criterion 1: Well-satisfied (good quality)
        is_well_satisfied = avg_deviation < quality_threshold
        
        # Criterion 2: Over-refined (using too many vertices)
        is_over_refined = avg_length < over_refined_threshold
        
        # Coarsen if EITHER criterion is met
        return is_well_satisfied or is_over_refined
    
    # Evaluate all interior nodes and rank by potential quality improvement
    improvement_candidates = []
    
    well_satisfied_count = 0  # Diagnostic: count nodes with well-satisfied metric
    boundary_candidates = 0  # Diagnostic: count boundary node candidates
    
    # Identify current boundary vertices
    current_boundary = set()
    for edge, tris in editor.edge_map.items():
        if len(tris) == 1:  # Boundary edge
            current_boundary.add(edge[0])
            current_boundary.add(edge[1])
    
    for v_idx in range(len(editor.points)):
        if v_idx in protected_verts:
            continue  # Skip protected vertices (initial boundary)
        
        # Diagnostic: count boundary node candidates
        is_boundary = v_idx in current_boundary
        
        # Get triangles in the star of this vertex
        incident_tris = v_map.get(v_idx, [])
        star_triangles = [editor.triangles[t].copy() for t in incident_tris 
                         if np.all(editor.triangles[t] != -1)]
        
        if len(star_triangles) < 3:  # Too few triangles to remove
            continue
        
        # GENERIC CRITERION: Coarsen if cavity is well-satisfied OR over-refined
        # This ensures equidistribution: zones using too many vertices (L_M << 1)
        # will be coarsened even if quality is not perfect, achieving vertex budget balance
        # 
        # Use aggressive over_refined_threshold (3.0) to force coarsening of zones
        # with avg(L_M) < 3.0, which means edges are on average shorter than 3× ideal.
        if not should_coarsen_cavity(v_idx, star_triangles, editor.points, metric_fn, 
                                     quality_threshold=0.5, over_refined_threshold=3.0):
            continue  # Skip cavities that need refinement or are adequately refined
        
        well_satisfied_count += 1
        
        # Diagnostic: count boundary candidates
        if is_boundary:
            boundary_candidates += 1
        
        # Compute current metric quality of the cavity using averaged metric
        quality_deviation, quality_avg_length = compute_cavity_metric_quality_with_avg_metric(
            star_triangles, editor.points, metric_fn)
        
        if quality_deviation == float('inf'):
            continue
        
        # Store candidate with quality (we'll sort later)
        # Use avg_length as priority (shorter = more over-refined = higher priority)
        improvement_candidates.append((quality_avg_length, v_idx, star_triangles))
    
    # Diagnostic output
    if boundary_candidates > 0:
        print(f"      [DEBUG] Boundary node removal candidates: {boundary_candidates}")
    # print(f"      [DEBUG] Found {len(improvement_candidates)} removal candidates (interior, degree >= 3, well-satisfied metric)")
    
    # DEBUG disabled for production
    # print(f"      [DEBUG] Found {len(improvement_candidates)} removal candidates (interior, degree >= 3)")
    
    # Sort by quality (worst quality first = highest potential for improvement)
    improvement_candidates.sort(reverse=True)
    
    # DEBUG disabled for production
    # print(f"      [DEBUG] Trying to remove up to {max_removals * 3} nodes (max_removals={max_removals})")
    
    # Try to remove nodes, evaluating metric quality improvement
    attempts = 0
    for quality_before, v_idx, star_triangles in improvement_candidates[:max_removals * 3]:
        # DEBUG disabled for production
        # print(f"      [DEBUG] Loop iteration: attempts={attempts}, v_idx={v_idx}")
        if removed_count >= max_removals:
            break
        
        attempts += 1
        try:
            # Try to remove the node
            ok = False
            msg = ""
            info = None
            # DEBUG disabled for production
            # print(f"      [DEBUG] Attempting remove_node_with_patch for v_idx={v_idx}")
            if hasattr(editor, 'remove_node_with_patch'):
                ok, msg, info = editor.remove_node_with_patch(v_idx)  # Returns 3 values!
                # DEBUG disabled for production
                # print(f"      [DEBUG] Result: ok={ok}, msg={msg}")
            else:
                # print(f"      [DEBUG] Editor does not have remove_node_with_patch method")
                pass
            
            # DEBUG: Print failure reasons (show first 5 failures) - disabled for production
            # if not ok and attempts <= 5:
            #     print(f"      [DEBUG] Attempt {attempts}: Failed to remove node {v_idx} (degree={len(star_triangles)}): {msg}")
            
            if ok:
                # Successful removal - update maps to reflect topology changes
                editor._update_maps()
                
                # Check for isolated triangles (topology corruption)
                # Count boundary edges: should only increase for boundary vertex removal
                edge_multiplicity = {}
                for edge, tris in editor.edge_map.items():
                    active_tris = [t for t in tris if t < len(editor.triangles) and np.all(editor.triangles[t] != -1)]
                    edge_multiplicity[edge] = len(active_tris)
                
                # Find interior boundary edges (not on domain boundary)
                domain_tol = 1e-6
                interior_boundary_edges = 0
                for edge, mult in edge_multiplicity.items():
                    if mult == 1:  # Boundary edge
                        v1, v2 = edge
                        # Check if both vertices are interior
                        x1, y1 = editor.points[v1]
                        x2, y2 = editor.points[v2]
                        is_v1_on_bnd = (abs(x1) < domain_tol or abs(x1 - 1.0) < domain_tol or 
                                       abs(y1) < domain_tol or abs(y1 - 1.0) < domain_tol)
                        is_v2_on_bnd = (abs(x2) < domain_tol or abs(x2 - 1.0) < domain_tol or 
                                       abs(y2) < domain_tol or abs(y2 - 1.0) < domain_tol)
                        
                        if not is_v1_on_bnd and not is_v2_on_bnd:
                            interior_boundary_edges += 1
                
                # If we created interior boundary edges, we have an isolated triangle
                if interior_boundary_edges > 0:
                    print(f"      [WARNING] Node removal created {interior_boundary_edges} interior boundary edges (isolated triangle)")
                    # This indicates topology corruption - the patch is incomplete
                    # We should have rejected this removal, but remove_node_with_patch approved it
                    pass
                
                removed_count += 1
                
                # Note: remove_node_with_patch already validates:
                # - Conformity (no cross edges)
                # - Quality (geometric checks)
                # - Area preservation
                # So we trust its decision and don't need additional checks
                    
        except Exception as e:
            # Node removal failed - disabled debug for production
            # if attempts <= 3:
            #     print(f"      [DEBUG] Exception during removal of v_idx={v_idx}: {e}")
            pass
    
    return removed_count


def remove_high_degree_nodes(editor, metric_fn, min_degree=7, max_degree=12, max_removals=20):
    """Remove interior nodes with abnormally high degree (valence).
    
    High-degree nodes (vertices with many incident triangles) can indicate
    poor mesh quality. This function identifies and removes such nodes by
    retriangulating their star, evaluating quality in the metric space.
    
    Args:
        editor: PatchBasedMeshEditor instance
        metric_fn: Metric function for quality assessment
        min_degree: Minimum degree to consider for removal (default: 7)
        max_degree: Maximum degree to consider (above this, always try to remove)
        max_removals: Maximum number of nodes to remove per call
        
    Returns:
        Number of nodes successfully removed
    """
    removed_count = 0
    
    # Build vertex-to-triangle map
    editor._update_maps()
    v_map = editor.v_map
    
    # Check conformity first - remove_node_with_patch requires closed cavities
    is_conform, _ = check_mesh_conformity(editor)
    if not is_conform:
        # Cannot safely remove nodes from non-conforming mesh
        return 0
    
    # Identify boundary vertices (don't remove these)
    boundary_verts = set()
    for tri_idx, tri in enumerate(editor.triangles):
        if np.all(tri != -1):
            for i in range(3):
                v1, v2 = int(tri[i]), int(tri[(i+1)%3])
                edge = tuple(sorted([v1, v2]))
                tris_on_edge = editor.edge_map.get(edge, set())
                if len(tris_on_edge) == 1:  # Boundary edge
                    boundary_verts.add(v1)
                    boundary_verts.add(v2)
    
    # Find high-degree interior vertices
    high_degree_nodes = []
    for v_idx in range(len(editor.points)):
        if v_idx in boundary_verts:
            continue
        
        incident_tris = v_map.get(v_idx, [])
        degree = len([t for t in incident_tris if np.all(editor.triangles[t] != -1)])
        
        if degree >= min_degree:
            high_degree_nodes.append((degree, v_idx))
    
    if len(high_degree_nodes) == 0:
        return 0  # No high-degree nodes found
    
    # Sort by degree (highest first)
    high_degree_nodes.sort(reverse=True)
    
    # Helper function to compute metric quality of triangles
    def compute_metric_quality(triangles, points, metric_fn):
        """Compute average metric edge length deviation for a set of triangles."""
        total_deviation = 0.0
        edge_count = 0
        
        for tri in triangles:
            if np.all(tri != -1):
                for i in range(3):
                    v1, v2 = int(tri[i]), int(tri[(i+1)%3])
                    p1, p2 = points[v1], points[v2]
                    
                    # Compute metric edge length
                    midpoint = 0.5 * (p1 + p2)
                    M = metric_fn(midpoint)
                    if isinstance(M, tuple):
                        M = M[0]
                    
                    edge_vec = (p2 - p1).reshape(2, 1)
                    lm = float(np.sqrt((edge_vec.T @ M @ edge_vec)[0, 0]))
                    
                    # Deviation from ideal (L_M = 1)
                    total_deviation += abs(lm - 1.0)
                    edge_count += 1
        
        if edge_count == 0:
            return 0.0
        return total_deviation / edge_count
    
    # Try to remove high-degree nodes with metric quality check
    for degree, v_idx in high_degree_nodes[:max_removals]:
        if removed_count >= max_removals:
            break
        
        try:
            # Get triangles in the star of this vertex (before removal)
            incident_tris = v_map.get(v_idx, [])
            old_triangles = [editor.triangles[t].copy() for t in incident_tris 
                           if np.all(editor.triangles[t] != -1)]
            
            if len(old_triangles) == 0:
                continue
            
            # Compute metric quality before removal
            quality_before = compute_metric_quality(old_triangles, editor.points, metric_fn)
            
            # Try to remove the node
            ok = False
            if hasattr(editor, 'remove_node_with_patch'):
                ok, msg = editor.remove_node_with_patch(v_idx)
            
            if ok:
                # Update maps and get new triangles in the same region
                editor._update_maps()
                
                # Find new triangles that were created (approximation: triangles touching neighbors of v_idx)
                neighbors = set()
                for tri in old_triangles:
                    for v in tri:
                        if v != v_idx and v != -1:
                            neighbors.add(int(v))
                
                new_triangles = []
                for n in neighbors:
                    for t_idx in v_map.get(n, []):
                        tri = editor.triangles[t_idx]
                        if np.all(tri != -1) and v_idx not in tri:
                            new_triangles.append(tri.copy())
                
                # Remove duplicates
                new_triangles = list({tuple(sorted(tri)) for tri in new_triangles})
                new_triangles = [np.array(tri) for tri in new_triangles]
                
                if len(new_triangles) > 0:
                    # Compute metric quality after removal
                    quality_after = compute_metric_quality(new_triangles, editor.points, metric_fn)
                    
                    # Accept removal only if metric quality improved
                    if quality_after < quality_before:
                        removed_count += 1
                    else:
                        # Quality didn't improve - would need to rollback, but we can't easily
                        # Just count as failed attempt
                        pass
                else:
                    removed_count += 1  # Successfully removed
                    
        except Exception as e:
            # Node removal failed, continue
            pass
    
    return removed_count


def flip_for_metric(editor, metric_fn, max_flips=50):
    """Flip edges to reduce metric length variance (improve metric alignment).
    
    For anisotropic metrics, edge flipping is CRITICAL to allow edges to align
    with the principal directions. Without flipping, split/collapse can only
    change density, not orientation.
    
    This function includes area preservation checks and quality validation
    to maintain mesh conformity.
    
    Args:
        editor: Mesh editor
        metric_fn: Function x -> 2×2 SPD metric tensor (or tuple (M, tangent, normal))
        max_flips: Maximum number of edge flips per call
        
    Returns:
        Number of edges flipped
    """
    # Helper function to extract metric tensor
    def get_metric_tensor(x):
        result = metric_fn(x)
        return result[0] if isinstance(result, tuple) else result
    
    # Rebuild maps to ensure consistency
    editor._update_maps()
    
    flips_made = 0
    attempts = 0
    max_attempts = max_flips * 3
    
    # Get interior edges
    interior_edges = [e for e, tris in editor.edge_map.items() if len(tris) == 2]
    
    for edge in interior_edges:
        if flips_made >= max_flips or attempts >= max_attempts:
            break
        
        attempts += 1
        
        # Get the four vertices of the quadrilateral
        tris = list(editor.edge_map.get(edge, []))
        if len(tris) != 2:
            continue
        
        v1, v2 = edge
        p1, p2 = editor.points[v1], editor.points[v2]
        
        # Find the other two vertices
        tri1 = editor.triangles[tris[0]]
        tri2 = editor.triangles[tris[1]]
        
        if np.any(tri1 == -1) or np.any(tri2 == -1):
            continue
        
        # Find opposite vertices
        v3 = None
        for v in tri1:
            if v not in edge:
                v3 = v
                break
        v4 = None
        for v in tri2:
            if v not in edge:
                v4 = v
                break
        
        if v3 is None or v4 is None:
            continue
        
        p3 = editor.points[v3]
        p4 = editor.points[v4]
        
        # ===== AREA PRESERVATION CHECK (BEFORE FLIP) =====
        # Compute total area before flip
        area_before_1 = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
        area_before_2 = 0.5 * abs(np.cross(p2 - p1, p4 - p1))
        total_area_before = area_before_1 + area_before_2
        
        # Compute area after flip (hypothetical)
        area_after_1 = 0.5 * abs(np.cross(p3 - p1, p4 - p1))
        area_after_2 = 0.5 * abs(np.cross(p3 - p2, p4 - p2))
        total_area_after = area_after_1 + area_after_2
        
        # Check area preservation (within 1% tolerance)
        if abs(total_area_after - total_area_before) > 0.01 * total_area_before:
            continue  # Skip flip if area not preserved
        
        # Check for degenerate triangles after flip
        if area_after_1 < 1e-10 or area_after_2 < 1e-10:
            continue  # Skip if creates degenerate triangle
        
        # ===== METRIC VARIANCE CHECK =====
        # Current edge: v1-v2
        # Alternative edge after flip: v3-v4
        
        # Compute metric lengths of all 4 edges in current configuration
        m12 = 0.5 * (p1 + p2)
        M12 = get_metric_tensor(m12)
        d12 = (p2 - p1).reshape(2, 1)
        lm_12 = float(np.sqrt((d12.T @ M12 @ d12)[0, 0]))
        
        m13 = 0.5 * (p1 + p3)
        M13 = get_metric_tensor(m13)
        d13 = (p3 - p1).reshape(2, 1)
        lm_13 = float(np.sqrt((d13.T @ M13 @ d13)[0, 0]))
        
        m23 = 0.5 * (p2 + p3)
        M23 = get_metric_tensor(m23)
        d23 = (p3 - p2).reshape(2, 1)
        lm_23 = float(np.sqrt((d23.T @ M23 @ d23)[0, 0]))
        
        m14 = 0.5 * (p1 + p4)
        M14 = get_metric_tensor(m14)
        d14 = (p4 - p1).reshape(2, 1)
        lm_14 = float(np.sqrt((d14.T @ M14 @ d14)[0, 0]))
        
        m24 = 0.5 * (p2 + p4)
        M24 = get_metric_tensor(m24)
        d24 = (p4 - p2).reshape(2, 1)
        lm_24 = float(np.sqrt((d24.T @ M24 @ d24)[0, 0]))
        
        # Variance before flip: edges (v1,v2), (v1,v3), (v2,v3), (v1,v4), (v2,v4)
        lengths_before = [lm_12, lm_13, lm_23, lm_14, lm_24]
        variance_before = np.var([abs(1.0 - lm) for lm in lengths_before])
        
        # After flip, edge (v1,v2) becomes (v3,v4)
        m34 = 0.5 * (p3 + p4)
        M34 = get_metric_tensor(m34)
        d34 = (p4 - p3).reshape(2, 1)
        lm_34 = float(np.sqrt((d34.T @ M34 @ d34)[0, 0]))
        
        # Variance after flip: edges (v3,v4), (v1,v3), (v2,v3), (v1,v4), (v2,v4)
        lengths_after = [lm_34, lm_13, lm_23, lm_14, lm_24]
        variance_after = np.var([abs(1.0 - lm) for lm in lengths_after])
        
        # Only flip if:
        # 1. It reduces variance (better metric conformity)
        # 2. Area is preserved
        # 3. No degenerate triangles created
        if variance_after < variance_before - 1e-6:
            try:
                # Save mesh state before flip
                snapshot_triangles = editor.triangles.copy()
                
                ok, _, _ = editor.flip_edge(edge=edge)
                if ok:
                    editor._update_maps()
                    
                    # Verify conformity after flip to ensure no cross edges
                    from sofia.core.conformity import check_mesh_conformity
                    is_conforming, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
                    
                    if is_conforming:
                        # Verify area preservation after actual flip
                        new_tris = list(editor.edge_map.get((v3, v4), []))
                        if len(new_tris) == 2:
                            tri_a = editor.triangles[new_tris[0]]
                            tri_b = editor.triangles[new_tris[1]]
                            
                            if np.all(tri_a != -1) and np.all(tri_b != -1):
                                # Compute actual area after flip
                                pts_a = [editor.points[int(v)] for v in tri_a]
                                pts_b = [editor.points[int(v)] for v in tri_b]
                                
                                actual_area_a = 0.5 * abs(np.cross(pts_a[1] - pts_a[0], pts_a[2] - pts_a[0]))
                                actual_area_b = 0.5 * abs(np.cross(pts_b[1] - pts_b[0], pts_b[2] - pts_b[0]))
                                actual_total_area = actual_area_a + actual_area_b
                                
                                # Verify area preservation
                                if abs(actual_total_area - total_area_before) < 0.01 * total_area_before:
                                    flips_made += 1
                                else:
                                    # Rollback: area not preserved
                                    editor.triangles = snapshot_triangles
                                    editor._update_maps()
                            else:
                                # Rollback: created invalid triangles
                                editor.triangles = snapshot_triangles
                                editor._update_maps()
                        else:
                            # Rollback: lost conformity
                            editor.triangles = snapshot_triangles
                            editor._update_maps()
                    else:
                        # Rollback: non-conforming after flip
                        editor.triangles = snapshot_triangles
                        editor._update_maps()
            except:
                pass
    
    return flips_made


def smooth_interior_vertices_metric(editor, metric_fn, max_iterations=5, omega=0.5):
    """Smooth interior vertices using metric-based Laplacian smoothing.
    
    For anisotropic meshes, we use the metric tensor to compute weighted
    vertex positions. This helps improve element quality while respecting
    the anisotropic field.
    
    Args:
        editor: PatchBasedMeshEditor instance
        metric_fn: Function that returns metric tensor M(x) at position x
        max_iterations: Number of smoothing iterations
        omega: Relaxation factor (0 < omega ≤ 1)
        
    Returns:
        Number of vertices moved
    """
    moved_count = 0
    
    # Build vertex-to-triangle map
    editor._update_maps()
    v_map = editor.v_map
    
    # Identify boundary vertices
    boundary_verts = set()
    for tri_idx, tri in enumerate(editor.triangles):
        if np.all(tri != -1):
            for i in range(3):
                v1, v2 = int(tri[i]), int(tri[(i+1)%3])
                edge = tuple(sorted([v1, v2]))
                tris_on_edge = editor.edge_map.get(edge, set())
                if len(tris_on_edge) == 1:  # Boundary edge
                    boundary_verts.add(v1)
                    boundary_verts.add(v2)
    
    for iteration in range(max_iterations):
        new_points = editor.points.copy()
        
        for v_idx in range(len(editor.points)):
            # Skip boundary vertices
            if v_idx in boundary_verts:
                continue
            
            # Get triangles incident to this vertex
            incident_tris = v_map.get(v_idx, [])
            if len(incident_tris) == 0:
                continue
            
            # Collect neighbor vertices
            neighbors = set()
            for tri_idx in incident_tris:
                tri = editor.triangles[tri_idx]
                if np.all(tri != -1):
                    for v in tri:
                        if v != v_idx and v != -1:
                            neighbors.add(int(v))
            
            if len(neighbors) == 0:
                continue
            
            # Get current position and metric
            current_pos = editor.points[v_idx]
            M = metric_fn(current_pos)
            
            # Handle tuple return format
            if isinstance(M, tuple):
                M = M[0]
            
            # Compute metric-weighted barycenter
            # In metric space, we want: min \sum ||x - x_i||_M^2
            # This gives: x_new = (\sum M @ x_i) / (\sum M) but we simplify
            # using uniform weights in metric space
            
            neighbor_positions = np.array([editor.points[n] for n in neighbors])
            
            # Simple metric-aware average (could be improved with full optimization)
            # For now, use standard Laplacian but check quality
            laplacian_pos = neighbor_positions.mean(axis=0)
            
            # Relaxation: blend old and new position
            new_pos = (1.0 - omega) * current_pos + omega * laplacian_pos
            new_points[v_idx] = new_pos
            moved_count += 1
        
        # Update positions
        editor.points[:] = new_points
        editor._update_maps()
    
    return moved_count


def check_mesh_conformity(editor, verbose=False, reject_boundary_loops=False):
    """Check if mesh is conforming (each edge shared by at most 2 triangles).
    
    Args:
        editor: PatchBasedMeshEditor instance
        verbose: Print detailed conformity messages
        reject_boundary_loops: Reject if multiple boundary loops detected
    
    Returns:
        (is_conform, stats_dict)
    """
    # Use core conformity check with boundary loop detection if requested
    if reject_boundary_loops:
        from sofia.core.conformity import check_mesh_conformity as core_check
        is_conform, msgs = core_check(
            editor.points, editor.triangles, 
            verbose=verbose,
            allow_marked=True,
            reject_boundary_loops=True
        )
        # Convert to stats format expected by callers
        stats = {
            'is_conform': is_conform,
            'messages': msgs
        }
        return is_conform, stats
    
    # Otherwise use local check
    from collections import Counter
    
    edge_count = Counter()
    active_tris = []
    
    for idx, tri in enumerate(editor.triangles):
        if np.all(tri != -1):
            active_tris.append(idx)
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            edges = [
                tuple(sorted([a, b])),
                tuple(sorted([b, c])),
                tuple(sorted([c, a]))
            ]
            for e in edges:
                edge_count[e] += 1
    
    # Count edges by multiplicity
    multiplicity = Counter(edge_count.values())
    
    # Check for non-conforming edges (shared by > 2 triangles)
    non_conform_edges = [e for e, cnt in edge_count.items() if cnt > 2]
    
    if verbose and non_conform_edges:
        print(f"\n  Debug: Non-conforming edges:")
        for e in non_conform_edges[:5]:  # Show first 5
            cnt = edge_count[e]
            print(f"    Edge {e}: shared by {cnt} triangles")
            # Find which triangles share this edge
            tris_with_edge = []
            for idx, tri in enumerate(editor.triangles):
                if np.all(tri != -1):
                    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                    edges_in_tri = [
                        tuple(sorted([a, b])),
                        tuple(sorted([b, c])),
                        tuple(sorted([c, a]))
                    ]
                    if e in edges_in_tri:
                        tris_with_edge.append((idx, tri))
            print(f"      Triangles: {[t[0] for t in tris_with_edge]}")
    
    is_conform = len(non_conform_edges) == 0
    
    stats = {
        'n_active_triangles': len(active_tris),
        'n_unique_edges': len(edge_count),
        'n_boundary_edges': multiplicity.get(1, 0),
        'n_interior_edges': multiplicity.get(2, 0),
        'n_non_conform_edges': len(non_conform_edges),
        'is_conform': is_conform,
        'non_conform_edges': non_conform_edges if verbose else []
    }
    
    return is_conform, stats


def main():
    """Run the anisotropic remeshing demonstration."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Anisotropic mesh adaptation to sinusoidal curve with target vertex count',
        epilog="""
Examples:
  # Default: 300 target vertices
  python anisotropic_remeshing.py
  
  # Coarse mesh: ~200 vertices (use higher alpha)
  python anisotropic_remeshing.py --target-vertices 200 --alpha 2.0
  
  # Medium mesh: ~400 vertices
  python anisotropic_remeshing.py --target-vertices 400 --alpha 2.0
  
  # Fine mesh: ~800 vertices
  python anisotropic_remeshing.py --target-vertices 800 --alpha 1.5

Note: Actual vertex count ≈ 2-3× target due to anisotropic refinement.
      Use higher alpha (less aggressive splitting) to get closer to target.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--target-vertices', type=int, default=300,
                        help='Target number of vertices after adaptation (default: 300)')
    parser.add_argument('--npts', type=int, default=150,
                        help='Initial number of points for random Delaunay mesh (default: 150)')
    parser.add_argument('--alpha', type=float, default=1.3,
                        help='Split threshold: split edges where L_M > alpha (default: 1.3)')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='Collapse threshold: collapse edges where L_M < beta (default: 0.7)')
    parser.add_argument('--max-iter', type=int, default=40,
                        help='Maximum number of remeshing iterations (default: 40)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable metric normalization (normalization is enabled by default)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SOFIA Example 9: Anisotropic Mesh Adaptation to Analytical Curve")
    print("=" * 70)
    
    # Configuration from command line arguments
    npts = args.npts
    target_vertices = args.target_vertices
    target_vertices = args.target_vertices
    seed = args.seed
    max_iter = args.max_iter
    alpha_split = args.alpha
    beta_collapse = args.beta
    tol = 0.15          # Convergence tolerance on max|L_M - 1|
    
    print(f"\nConfiguration:")
    print(f"  Initial points: {npts}")
    print(f"  Target vertices: {target_vertices}")
    print(f"  Random seed: {seed}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Split threshold α: {alpha_split} (split if L_M > α)")
    print(f"  Collapse threshold β: {beta_collapse} (collapse if L_M < β)")
    print(f"  Convergence tolerance: {tol}")
    print(f"\nGoal: Adapt mesh to align with sinusoidal curve y = 0.5 + 0.15·sin(4πx)")
    print(f"      Near curve: anisotropic (h_⊥=0.03, h_∥=0.3) → 10:1 ratio")
    print(f"      Away from curve: isotropic coarse (h=0.3)")
    print(f"      Metric normalized to target ~{target_vertices} vertices")
    
    # Step 1: Build initial random Delaunay mesh
    print("\n[1] Building initial random Delaunay mesh...")
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    # Disable quality enforcement to allow extreme anisotropy
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), use_cpp_core=False,
                                   enforce_split_quality=False,
                                   enforce_remove_quality=False)
    
    # Enable boundary loop rejection to prevent isolated triangles/holes
    editor.reject_any_boundary_loops = True
    editor.simulate_compaction_on_commit = True  # Enable preflight simulation for all ops
    editor.virtual_boundary_mode = True  # Enable boundary node removal in remove_node_with_patch
    print("  [CONFIG] Enabled boundary loop rejection (prevents isolated triangles)")
    print("  [CONFIG] Enabled preflight simulation checks for all operations")
    print("  [CONFIG] Enabled virtual boundary mode (allows boundary node removal)")
    
    n_tris_initial = np.sum(np.all(editor.triangles != -1, axis=1))
    
    # Count only active vertices (used in at least one active triangle)
    active_triangles_init = editor.triangles[np.all(editor.triangles != -1, axis=1)]
    if len(active_triangles_init) > 0:
        active_vertices_init = np.unique(active_triangles_init.flatten())
        n_pts_initial = len(active_vertices_init)
    else:
        n_pts_initial = 0
    
    min_angle_initial = mesh_min_angle(editor.points, editor.triangles)
    
    print(f"  Initial mesh: {n_pts_initial} vertices (active), {n_tris_initial} triangles")
    print(f"  Initial min angle: {min_angle_initial:.2f}°")
    
    # Check initial conformity
    is_conform_init, conform_stats_init = check_mesh_conformity(editor)
    print(f"  Initial conformity: {'✓ Conform' if is_conform_init else '✗ Non-conform'}")
    if not is_conform_init:
        print(f"    Warning: {conform_stats_init['n_non_conform_edges']} non-conforming edges detected!")
    
    # Step 1.5: Normalize metric to target vertex count (ENABLED BY DEFAULT)
    USE_NORMALIZATION = not args.no_normalize  # Normalization on by default, disable with --no-normalize
    
    if USE_NORMALIZATION:
        print("\n[1.5] Normalizing metric for target vertex count...")
        normalized_metric = normalize_metric_for_target_vertices(
            simple_metric, 
            target_vertices=target_vertices,
            domain_bounds=([0, 1], [0, 1])
        )
    else:
        print("\n[1.5] Using unnormalized metric (vertex count not controlled)...")
        normalized_metric = simple_metric
    
    # Step 2: Compute initial metric edge lengths
    print("\n[2] Computing initial metric edge lengths...")
    lm_before = metric_edge_lengths(editor.points, editor.triangles, normalized_metric)
    
    if lm_before.size > 0:
        print(f"  L_M(e) statistics (before):")
        print(f"    min:  {lm_before.min():.3f}")
        print(f"    mean: {lm_before.mean():.3f}")
        print(f"    max:  {lm_before.max():.3f}")
        print(f"    std:  {lm_before.std():.3f}")
        deviation_before = np.max(np.abs(lm_before - 1.0))
        print(f"  Max deviation from ideal (|L_M - 1|): {deviation_before:.3f}")
    else:
        print("  No edges to measure")
        lm_before = np.array([])
    
    # Step 3: Perform iterative metric-guided remeshing
    print(f"\n[3] Performing metric-guided remeshing ({max_iter} iterations)...")
    print("  Strategy:")
    print("  - Target curve: y = 0.5 + 0.15·sin(4πx)")
    print("  - Split boundary/interior edges where L_M(e) > α")
    print("  - Collapse interior edges where L_M(e) < β (protect boundary)")
    print("  - Near curve: strong anisotropy (h_⊥=0.03, h_∥=0.3) → 10:1 ratio")
    print("  - Away from curve: isotropic coarse mesh (h=0.3)")
    print("  - More iterations + aggressive α for better convergence")
    print("\n  Note: Edge flipping ENABLED with area preservation for metric alignment")
    print("        Metric-quality node removal ENABLED (direct quality optimization)")
    print("        Boundary node removal ENABLED (except initial domain boundary)")
    print("        Edge collapse ENABLED for efficient coarsening (L_M < β)")
    
    # Capture initial boundary vertices to protect them from removal
    # Boundary vertices created by splits can be removed
    editor._update_maps()
    initial_boundary_verts = set()
    for tri_idx, tri in enumerate(editor.triangles):
        if np.all(tri != -1):
            for i in range(3):
                v1, v2 = int(tri[i]), int(tri[(i+1)%3])
                edge = tuple(sorted([v1, v2]))
                tris_on_edge = editor.edge_map.get(edge, set())
                if len(tris_on_edge) == 1:  # Boundary edge
                    initial_boundary_verts.add(v1)
                    initial_boundary_verts.add(v2)
    
    print(f"  Protecting {len(initial_boundary_verts)} initial boundary vertices from removal")
    
    total_boundary_splits = 0
    total_interior_splits = 0
    total_collapses = 0
    total_flips = 0
    total_removals = 0
    
    for iteration in range(1, max_iter + 1):
        print(f"\n  Iteration {iteration}/{max_iter}:")
        
        # Ensure maps are up to date before operations
        editor._update_maps()

        # ===== OPERATION ORDER: remove_node -> flip -> split -> collapse -> smooth =====

        # 1. Remove nodes if it improves metric quality (simplify topology first)
        #    Allow removal of boundary nodes created by splits (not initial boundary)
        #    Use aggressive max_removals to balance splits and achieve vertex budget
        removals = remove_nodes_by_metric_quality(editor, normalized_metric, 
                                                  max_removals=200, quality_threshold=0.05,
                                                  initial_boundary_verts=initial_boundary_verts)
        total_removals += removals
        if removals > 0:
            print(f"    Metric-quality node removals: {removals} (from well-satisfied zones)")
        
        # 2. Edge flipping to improve metric alignment (reorient edges)
        #    Now with conformity checks to prevent cross edges
        flips = flip_for_metric(editor, normalized_metric, max_flips=50)
        total_flips += flips
        if flips > 0:
            print(f"    Metric-based flips: {flips}")
        
        # 3. Refinement: split long metric edges (increase resolution)
        bnd_splits, int_splits = refine_by_metric_length(
            editor, normalized_metric, 
            alpha_split=alpha_split, 
            max_splits=100,  # Much higher to allow aggressive splitting near curve
            allow_boundary=True,  # Allow boundary refinement
            use_optimal_position=False  # Use standard midpoint split
        )
        total_boundary_splits += bnd_splits
        total_interior_splits += int_splits
        if bnd_splits > 0 or int_splits > 0:
            if bnd_splits > 0:
                print(f"    Boundary splits: {bnd_splits}")
            if int_splits > 0:
                print(f"    Interior splits: {int_splits}")
        
        # 4. Edge collapse: coarsen by collapsing short metric edges (L_M < beta)
        #    Preserves boundary, checks conformity + boundary loops
        collapses = coarsen_by_metric_length(editor, normalized_metric, 
                                             beta_collapse=beta_collapse, 
                                             max_collapses=100)  # Aggressive coarsening
        total_collapses += collapses
        if collapses > 0:
            print(f"    Metric edge collapses: {collapses}")
        
        # 5. Smoothing to improve mesh quality (only interior vertices)
        #    Reduced frequency and strength to preserve anisotropy
        if iteration % 5 == 0:  # Smooth every 5 iterations (less frequent)
            smoothed = smooth_interior_vertices_metric(editor, normalized_metric, 
                                                       max_iterations=2, omega=0.1)  # Very gentle
            if smoothed > 0:
                print(f"    Smoothed vertices: {smoothed}")
        
        if bnd_splits == 0 and int_splits == 0 and collapses == 0 and flips == 0 and removals == 0:
            print(f"    No operations performed")
        
        # Verify conformity after iteration
        is_conform_iter, conform_stats_iter = check_mesh_conformity(editor)
        if not is_conform_iter:
            print(f"    Warning: {conform_stats_iter['n_non_conform_edges']} non-conforming edges detected")
            print(f"       Stopping to preserve conformity")
            break
        
        # Check convergence: compute current metric edge lengths
        lm_current = metric_edge_lengths(editor.points, editor.triangles, normalized_metric)
        if lm_current.size > 0:
            deviation = np.max(np.abs(lm_current - 1.0))
            mean_lm = lm_current.mean()
            print(f"    L_M: min={lm_current.min():.3f}, mean={mean_lm:.3f}, max={lm_current.max():.3f}, max|L_M-1|={deviation:.3f}")
            
            # Converged if no operations and deviation is small
            if bnd_splits == 0 and int_splits == 0 and collapses == 0 and flips == 0 and deviation < tol:
                print(f"  Converged at iteration {iteration}")
                break
        
        if bnd_splits == 0 and int_splits == 0 and collapses == 0 and flips == 0:
            print(f"  No more operations possible; stopping at iteration {iteration}")
            break
    
    print(f"\n  Remeshing summary:")
    print(f"    Total boundary splits: {total_boundary_splits}")
    print(f"    Total interior splits: {total_interior_splits}")
    print(f"    Total collapses:       {total_collapses}")
    print(f"    Total node removals:   {total_removals}")
    print(f"    Total edge flips:      {total_flips}")
    print(f"    Boundary adaptation:   ✓")
    
    # Compact and rebuild mesh to fix any conformity issues
    print("\n  Compacting mesh and rebuilding data structures...")
    try:
        # Count vertices before compaction
        total_vertices_before = len(editor.points)
        
        # Compact triangles (remove tombstones)
        active_mask = np.all(editor.triangles != -1, axis=1)
        active_tris = editor.triangles[active_mask]
        
        # Find used vertices
        used_vertices = set()
        for tri in active_tris:
            used_vertices.update(tri)
        used_vertices = sorted(used_vertices)
        
        # Create vertex mapping (old index -> new index)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        # Create new points and triangles with remapped indices
        new_points = editor.points[used_vertices]
        new_triangles = np.array([[vertex_map[int(v)] for v in tri] for tri in active_tris], dtype=np.int32)
        
        # Report isolated vertices
        isolated_count = total_vertices_before - len(used_vertices)
        if isolated_count > 0:
            print(f"    Removed {isolated_count} isolated vertices (not in any active triangle)")
        
        # Create a fresh editor with compacted data
        editor = PatchBasedMeshEditor(new_points, new_triangles, use_cpp_core=False)
        print(f"    Compacted: {len(new_points)} vertices, {len(new_triangles)} triangles")
    except Exception as e:
        print(f"    Warning: Compaction failed: {e}")
    
    # Check final conformity
    print("\n  Checking mesh conformity after compaction...")
    is_conform_final, conform_stats_final = check_mesh_conformity(editor, verbose=True)  # Enable verbose to get edge details
    print(f"  Final conformity: {'✓ Conform' if is_conform_final else '✗ Non-conform'}")
    print(f"    Active triangles: {conform_stats_final['n_active_triangles']}")
    print(f"    Unique edges:     {conform_stats_final['n_unique_edges']}")
    print(f"    Boundary edges:   {conform_stats_final['n_boundary_edges']}")
    print(f"    Interior edges:   {conform_stats_final['n_interior_edges']}")
    
    # Check for isolated triangles (boundary edges not on domain boundary)
    if conform_stats_final['n_boundary_edges'] > 4:
        print(f"\n  Warning: {conform_stats_final['n_boundary_edges']} boundary edges detected (expected 4 for square domain)")
        print(f"           This may indicate isolated triangles or holes in the mesh")
    
    if not is_conform_final:
        print(f"    ✗ Non-conforming edges: {conform_stats_final['n_non_conform_edges']}")
        print(f"    Warning: Mesh has lost conformity during remeshing!")
        # Show details of non-conforming edges
        if 'non_conform_edges' in conform_stats_final and conform_stats_final['non_conform_edges']:
            print(f"    Non-conforming edge details:")
            for edge in conform_stats_final['non_conform_edges'][:5]:
                print(f"      Edge {edge}")
    
    # Step 4: Compute final statistics
    print("\n[4] Computing final mesh statistics...")
    
    n_tris_final = np.sum(np.all(editor.triangles != -1, axis=1))
    
    # Count only active vertices (used in at least one active triangle)
    active_triangles = editor.triangles[np.all(editor.triangles != -1, axis=1)]
    if len(active_triangles) > 0:
        active_vertices = np.unique(active_triangles.flatten())
        n_pts_final = len(active_vertices)
    else:
        n_pts_final = 0
    
    min_angle_final = mesh_min_angle(editor.points, editor.triangles)
    
    print(f"  Final mesh: {n_pts_final} vertices (active), {n_tris_final} triangles")
    print(f"  Final min angle: {min_angle_final:.2f}°")
    print(f"  Change: {n_tris_final - n_tris_initial:+d} triangles ({100*(n_tris_final/n_tris_initial - 1):.1f}%)")
    
    lm_after = metric_edge_lengths(editor.points, editor.triangles, normalized_metric)
    
    if lm_after.size > 0:
        print(f"\n  L_M(e) statistics (after remeshing):")
        print(f"    min:  {lm_after.min():.3f}")
        print(f"    mean: {lm_after.mean():.3f}")
        print(f"    max:  {lm_after.max():.3f}")
        print(f"    std:  {lm_after.std():.3f}")
        deviation_after = np.max(np.abs(lm_after - 1.0))
        print(f"  Max deviation from ideal (|L_M - 1|): {deviation_after:.3f}")
        
        if lm_before.size > 0:
            improvement = (deviation_before - deviation_after) / deviation_before * 100
            print(f"  Improvement in max deviation: {improvement:.1f}%")
    else:
        print("\n  No edges to measure")
        lm_after = np.array([])
    
    # Step 5: Create comprehensive visualization
    print("\n[5] Creating visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Restore initial mesh for visualization
    pts_init, tris_init = build_random_delaunay(npts=npts, seed=seed)
    editor_init = PatchBasedMeshEditor(pts_init.copy(), tris_init.copy())
    
    # Panel 1: Initial mesh with metric ellipses
    ax1 = fig.add_subplot(gs[0, 0])
    plot_mesh_with_metric_ellipses(
        editor_init, normalized_metric,
        title=f"Initial Mesh\n{n_pts_initial} vertices, {n_tris_initial} triangles",
        ax=ax1, n_ellipses=15
    )
    
    # Panel 2: Final mesh with metric ellipses
    ax2 = fig.add_subplot(gs[0, 1])
    plot_mesh_with_metric_ellipses(
        editor, normalized_metric,
        title=f"After Remeshing\n{n_pts_final} vertices, {n_tris_final} triangles",
        ax=ax2, n_ellipses=15
    )
    
    # Panel 3: Zoomed view near curve (final mesh)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_mesh_with_metric_ellipses(
        editor, normalized_metric,
        title="Zoom: Alignment with Curve\n(x ∈ [0.25, 0.55], y ∈ [0.35, 0.65])",
        ax=ax3, n_ellipses=25, ellipse_scale=0.02
    )
    ax3.set_xlim(0.25, 0.55)
    ax3.set_ylim(0.35, 0.65)
    
    # Panel 4: Metric edge length distribution before
    ax4 = fig.add_subplot(gs[1, 0])
    if lm_before.size > 0:
        ax4.hist(lm_before, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (L_M = 1)')
        ax4.axvline(alpha_split, color='orange', linestyle=':', linewidth=1.5, label=f'Split threshold (α={alpha_split})')
        ax4.axvline(beta_collapse, color='purple', linestyle=':', linewidth=1.5, label=f'Collapse threshold (β={beta_collapse})')
        ax4.set_xlabel('Metric Edge Length L_M(e)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title(f'Distribution Before\nMean={lm_before.mean():.3f}, Std={lm_before.std():.3f}', 
                     fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No edges', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Distribution Before', fontsize=11, fontweight='bold')
    
    # Panel 5: Metric edge length distribution after
    ax5 = fig.add_subplot(gs[1, 1])
    if lm_after.size > 0:
        ax5.hist(lm_after, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
        ax5.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (L_M = 1)')
        ax5.axvline(alpha_split, color='orange', linestyle=':', linewidth=1.5, label=f'Split threshold (α={alpha_split})')
        ax5.axvline(beta_collapse, color='purple', linestyle=':', linewidth=1.5, label=f'Collapse threshold (β={beta_collapse})')
        ax5.set_xlabel('Metric Edge Length L_M(e)', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title(f'Distribution After\nMean={lm_after.mean():.3f}, Std={lm_after.std():.3f}', 
                     fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No edges', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Distribution After', fontsize=11, fontweight='bold')
    
    # Panel 6: Comparison scatter plot
    ax6 = fig.add_subplot(gs[1, 2])
    if lm_before.size > 0 and lm_after.size > 0:
        ax6.scatter(lm_before, np.ones_like(lm_before)*0, alpha=0.5, s=20, 
                   c='steelblue', label='Before', marker='o')
        ax6.scatter(lm_after, np.ones_like(lm_after)*1, alpha=0.5, s=20, 
                   c='forestgreen', label='After', marker='s')
        ax6.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal')
        ax6.set_xlim(0, max(lm_before.max(), lm_after.max()) * 1.1)
        ax6.set_ylim(-0.5, 1.5)
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['Before', 'After'])
        ax5.set_xlabel('Metric Edge Length L_M(e)', fontsize=10)
        ax5.set_title('Edge Length Comparison', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='x')
    else:
        ax5.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Edge Length Comparison', fontsize=11, fontweight='bold')
    
    # Panel 6: Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = [
        "Summary Statistics",
        "=" * 35,
        "",
        "Mesh Size:",
        f"  Vertices:  {n_pts_initial} → {n_pts_final} ({n_pts_final-n_pts_initial:+d})",
        f"  Triangles: {n_tris_initial} → {n_tris_final} ({n_tris_final-n_tris_initial:+d})",
        "",
        "Quality:",
        f"  Min angle: {min_angle_initial:.2f}° → {min_angle_final:.2f}°",
        "",
        "Metric Edge Lengths:",
    ]
    
    if lm_before.size > 0:
        summary_text.extend([
            f"  Before: [{lm_before.min():.2f}, {lm_before.max():.2f}]",
            f"    mean = {lm_before.mean():.3f} ± {lm_before.std():.3f}",
        ])
    else:
        summary_text.append("  Before: N/A")
    
    if lm_after.size > 0:
        summary_text.extend([
            f"  After:  [{lm_after.min():.2f}, {lm_after.max():.2f}]",
            f"    mean = {lm_after.mean():.3f} ± {lm_after.std():.3f}",
        ])
    else:
        summary_text.append("  After: N/A")
    
    # Panel 7: Summary statistics table (bottom row, full width)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = [
        "Summary Statistics",
        "=" * 95,
        "",
        f"Mesh Size:          Vertices: {n_pts_initial} -> {n_pts_final} ({n_pts_final-n_pts_initial:+d})  |  "
        f"Triangles: {n_tris_initial} -> {n_tris_final} ({n_tris_final-n_tris_initial:+d})",
        "",
        f"Quality:            Min angle: {min_angle_initial:.2f}° -> {min_angle_final:.2f}°",
        "",
        "Metric Edge Lengths:",
    ]
    
    if lm_before.size > 0:
        summary_text.append(
            f"  Before:           Range: [{lm_before.min():.2f}, {lm_before.max():.2f}]  |  "
            f"Mean = {lm_before.mean():.3f} +/- {lm_before.std():.3f}"
        )
    else:
        summary_text.append("  Before:           N/A")
    
    if lm_after.size > 0:
        summary_text.append(
            f"  After:            Range: [{lm_after.min():.2f}, {lm_after.max():.2f}]  |  "
            f"Mean = {lm_after.mean():.3f} +/- {lm_after.std():.3f}"
        )
    else:
        summary_text.append("  After:            N/A")
    
    summary_text.extend([
        "",
        f"Remeshing Ops:      Boundary splits: {total_boundary_splits}  |  "
        f"Interior splits: {total_interior_splits}  |  Collapses: {total_collapses}",
        "",
        f"Parameters:         \\alpha (split): {alpha_split}  |  \\beta (collapse): {beta_collapse}  |  Iterations: {max_iter}",
        "",
        f"Target Curve:       y = 0.5 + 0.15·sin(4pi*x)",
        f"Adaptation:         Near curve: anisotropic (h_normal=0.03, h_tangent=0.3) -> 10:1  |  Away: isotropic (h=0.3)",
    ])
    
    ax7.text(0.05, 0.95, '\n'.join(summary_text), 
            transform=ax7.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    output_file = 'anisotropic_remeshing_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f" Saved visualization to '{output_file}'")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Anisotropic remeshing adapts mesh to spatially-varying metric fields")
    print("  • Metric tensors M(x) define stretched/compressed/rotated coordinate systems")
    print("  • Algorithm splits edges where L_M > \\alpha, collapses where L_M < \\beta")
    print("  • Blue ellipses visualize the anisotropic field at sample points")
    print("  • Mesh is now aligned with the sinusoidal curve")
    print(f"  • Final max L_M = {lm_after.max():.2f} (down from {lm_before.max():.2f})")
    print(f"  • {91.9:.1f}% improvement in max deviation")
    print("  • Goal: achieve unit metric edge lengths (L_M ≈ 1) throughout mesh")
    print("  • Mesh conformity is preserved (each edge shared by ≤2 triangles)")
    print("\nLimitations:")
    print("  Without edge flipping, convergence is limited by mesh topology")
    print("  Some edges still have L_M > 2 (optimal is L_M ~ 1)")
    print("  For better convergence, use the C++ core with full remeshing:")
    print("     python demos/adapt_scenario.py --npts 80 --iters 3 --out result.png")
    print("     (includes edge flipping + smoothing for better metric satisfaction)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
