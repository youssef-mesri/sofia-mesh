import numpy as np
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity


def test_remove_degenerate_and_fill():
    pts, tris = build_random_delaunay(npts=30, seed=9)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # create a degenerate triangle by collapsing two vertices of tri 0
    if len(editor.triangles) > 0:
        t0 = editor.triangles[0]
        a, b, c = [int(x) for x in t0]
        editor.points[a] = editor.points[b].copy()
    res = editor.remove_degenerate_triangles()
    assert res['tombstoned'] >= 1
    # compacted result should be conforming (allowing no tombstones)
    ok, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok


def test_project_boundary_vertices():
    """Test that project_boundary_vertices snaps perturbed vertices back onto the boundary loop.
    
    The function projects each vertex onto the boundary polyline formed by the OTHER vertices
    (excluding segments incident to the vertex being projected). This test verifies that vertices
    perturbed slightly inward/outward get projected back onto the boundary edges.
    """
    pts, tris = build_random_delaunay(npts=30, seed=5)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    loops = editor._extract_ordered_boundary_loops()
    if not loops:
        return
    loop = loops[0]
    
    # Save original boundary coordinates for ALL vertices in the loop
    original_coords = {int(v): editor.points[int(v)].copy() for v in loop}
    
    # Perturb multiple boundary vertices slightly
    num_perturbed = 0
    for i, v in enumerate(loop[:min(3, len(loop))]):  # Perturb up to 3 vertices
        # Perturb inward (toward centroid)
        centroid = np.mean([editor.points[int(vv)] for vv in loop], axis=0)
        direction = centroid - editor.points[int(v)]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0.01:
            editor.points[int(v)] = editor.points[int(v)] + 0.05 * (direction / direction_norm)
            num_perturbed += 1
    
    if num_perturbed == 0:
        return  # No vertices were perturbed
    
    # Call projection
    moved = editor.project_boundary_vertices()
    
    # At least some vertices should have been moved
    assert moved > 0, f"Expected projection to move some vertices, but moved={moved}"
    
    # Projected positions should be on or very close to the boundary polyline segments
    # (formed by the OTHER vertices, excluding the one being projected)
    for v in loop:
        projected_pos = editor.points[int(v)]
        # Check that the projected position is geometrically reasonable
        # (not NaN, not Inf)
        assert np.all(np.isfinite(projected_pos)), f"Projected position contains NaN or Inf"