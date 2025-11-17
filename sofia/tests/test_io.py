"""Tests for mesh file I/O (read_msh, write_vtk)."""
import numpy as np
import pytest
from pathlib import Path

from sofia.core.io import read_msh, write_vtk
from sofia.core.constants import EPS_AREA


def test_write_vtk_basic(tmp_path):
    """Test basic VTK export with 2D triangle mesh."""
    # Create simple triangle mesh
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8]
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    
    output_file = tmp_path / "test_mesh.vtk"
    write_vtk(str(output_file), points, triangles, title="Test Mesh")
    
    assert output_file.exists()
    
    # Verify file contents
    content = output_file.read_text()
    assert "# vtk DataFile Version 2.0" in content
    assert "Test Mesh" in content
    assert "ASCII" in content
    assert "DATASET UNSTRUCTURED_GRID" in content
    assert "POINTS 3 double" in content
    assert "CELLS 1 4" in content
    assert "CELL_TYPES 1" in content
    assert "3 0 1 2" in content  # Triangle connectivity


def test_write_vtk_with_3d_points(tmp_path):
    """Test VTK export with 3D points."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.8, 0.1]
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    
    output_file = tmp_path / "test_mesh_3d.vtk"
    write_vtk(str(output_file), points, triangles)
    
    assert output_file.exists()
    content = output_file.read_text()
    # Check z-coordinate is preserved (scientific notation: 1.0e-01)
    assert "1.0000000000000001e-01" in content or "1e-01" in content


def test_write_vtk_with_scalar_cell_data(tmp_path):
    """Test VTK export with scalar field on triangles."""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
        [1.0, 1.0]
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=np.int32)
    
    # Add quality metric
    quality = np.array([0.95, 0.87])
    
    output_file = tmp_path / "test_quality.vtk"
    write_vtk(str(output_file), points, triangles, 
              cell_data={'quality': quality})
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "CELL_DATA 2" in content
    assert "SCALARS quality double 1" in content
    assert "LOOKUP_TABLE default" in content
    # Check values (scientific notation: 9.4999999999999996e-01)
    assert "9.4999999999999996e-01" in content or "9.5e-01" in content


def test_write_vtk_with_scalar_point_data(tmp_path):
    """Test VTK export with scalar field on vertices."""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8]
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    
    # Add temperature field
    temperature = np.array([100.0, 200.0, 150.0])
    
    output_file = tmp_path / "test_temperature.vtk"
    write_vtk(str(output_file), points, triangles,
              point_data={'temperature': temperature})
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "POINT_DATA 3" in content
    assert "SCALARS temperature double 1" in content
    # Check values (scientific notation: 1.0000000000000000e+02)
    assert "1.0000000000000000e+02" in content or "100.0" in content


def test_write_vtk_with_vector_point_data(tmp_path):
    """Test VTK export with vector field on vertices."""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8]
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    
    # Add displacement vectors (2D)
    displacement = np.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [-0.05, 0.05]
    ])
    
    output_file = tmp_path / "test_displacement.vtk"
    write_vtk(str(output_file), points, triangles,
              point_data={'displacement': displacement})
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "POINT_DATA 3" in content
    assert "VECTORS displacement double" in content
    # Check z-component added (should be 0)
    lines = content.split('\n')
    vector_section = False
    for line in lines:
        if "VECTORS displacement" in line:
            vector_section = True
        elif vector_section and "0.1" in line:
            # Should have format: x y z
            parts = line.split()
            assert len(parts) == 3
            break


def test_write_vtk_with_multiple_data_fields(tmp_path):
    """Test VTK export with both point and cell data."""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
        [1.0, 1.0]
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=np.int32)
    
    # Point data
    vertex_id = np.arange(4, dtype=np.float64)
    
    # Cell data
    triangle_area = np.array([0.4, 0.3])
    
    output_file = tmp_path / "test_multi_data.vtk"
    write_vtk(str(output_file), points, triangles,
              point_data={'vertex_id': vertex_id},
              cell_data={'area': triangle_area})
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "POINT_DATA 4" in content
    assert "SCALARS vertex_id" in content
    assert "CELL_DATA 2" in content
    assert "SCALARS area" in content


def test_write_vtk_invalid_points_shape(tmp_path):
    """Test VTK export rejects invalid points shape."""
    points = np.array([0.0, 1.0, 2.0])  # 1D array
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    
    output_file = tmp_path / "invalid.vtk"
    
    with pytest.raises(ValueError, match="points must be"):
        write_vtk(str(output_file), points, triangles)


def test_write_vtk_invalid_triangles_shape(tmp_path):
    """Test VTK export rejects invalid triangles shape."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]], dtype=np.float64)
    triangles = np.array([0, 1, 2])  # 1D array
    
    output_file = tmp_path / "invalid.vtk"
    
    with pytest.raises(ValueError, match="triangles must be"):
        write_vtk(str(output_file), points, triangles)


def test_write_vtk_larger_mesh(tmp_path):
    """Test VTK export with realistic mesh size."""
    # Create a small grid mesh
    n = 5
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Create triangles (2 per grid cell)
    triangles_list = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = i * n + (j + 1)
            v2 = (i + 1) * n + j
            v3 = (i + 1) * n + (j + 1)
            triangles_list.append([v0, v1, v2])
            triangles_list.append([v1, v3, v2])
    
    triangles = np.array(triangles_list, dtype=np.int32)
    
    output_file = tmp_path / "grid_mesh.vtk"
    write_vtk(str(output_file), points, triangles)
    
    assert output_file.exists()
    content = output_file.read_text()
    assert f"POINTS {n*n} double" in content
    assert f"CELLS {len(triangles)} {len(triangles) * 4}" in content


def create_msh_v2_file(filepath: Path, points: np.ndarray, triangles: np.ndarray):
    """Helper to create a Gmsh v2.2 format file for testing."""
    with open(filepath, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Nodes (1-indexed)
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for i, pt in enumerate(points, start=1):
            f.write(f"{i} {pt[0]:.16e} {pt[1]:.16e} 0.0\n")
        f.write("$EndNodes\n")
        
        # Elements (1-indexed, type 2 = triangle)
        f.write("$Elements\n")
        f.write(f"{len(triangles)}\n")
        for i, tri in enumerate(triangles, start=1):
            # Format: elem-id type num-tags tag1 tag2 v0 v1 v2
            # num-tags=2: physical-tag, elementary-tag (both set to 1)
            f.write(f"{i} 2 2 1 1 {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        f.write("$EndElements\n")


def create_msh_v4_file(filepath: Path, points: np.ndarray, triangles: np.ndarray):
    """Helper to create a Gmsh v4.1 format file for testing."""
    with open(filepath, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("4.1 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Nodes (1-indexed)
        # Format: numEntityBlocks numNodes minNodeTag maxNodeTag
        f.write("$Nodes\n")
        f.write(f"1 {len(points)} 1 {len(points)}\n")
        # Entity block: entityDim entityTag parametric numNodesInBlock
        f.write(f"2 1 0 {len(points)}\n")
        # Node tags
        for i in range(1, len(points) + 1):
            f.write(f"{i}\n")
        # Node coordinates
        for pt in points:
            f.write(f"{pt[0]:.16e} {pt[1]:.16e} 0.0\n")
        f.write("$EndNodes\n")
        
        # Elements (1-indexed, type 2 = triangle)
        # Format: numEntityBlocks numElements minElementTag maxElementTag
        f.write("$Elements\n")
        f.write(f"1 {len(triangles)} 1 {len(triangles)}\n")
        # Entity block: entityDim entityTag elementType numElementsInBlock
        f.write(f"2 1 2 {len(triangles)}\n")
        for i, tri in enumerate(triangles, start=1):
            # Format: elemTag v0 v1 v2
            f.write(f"{i} {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        f.write("$EndElements\n")


def test_read_msh_v2_basic(tmp_path):
    """Test reading Gmsh v2.2 format."""
    # Create test mesh
    points_orig = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8]
    ], dtype=np.float64)
    triangles_orig = np.array([[0, 1, 2]], dtype=np.int32)
    
    msh_file = tmp_path / "test_v2.msh"
    create_msh_v2_file(msh_file, points_orig, triangles_orig)
    
    # Read back
    points, triangles = read_msh(str(msh_file))
    
    assert points.shape == (3, 2)
    assert triangles.shape == (1, 3)
    assert points.dtype == np.float64
    assert triangles.dtype == np.int32
    
    # Check values (should match original)
    np.testing.assert_allclose(points, points_orig, rtol=1e-14)
    np.testing.assert_array_equal(triangles, triangles_orig)


def test_read_msh_v4_basic(tmp_path):
    """Test reading Gmsh v4.1 format."""
    # Create test mesh
    points_orig = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8]
    ], dtype=np.float64)
    triangles_orig = np.array([[0, 1, 2]], dtype=np.int32)
    
    msh_file = tmp_path / "test_v4.msh"
    create_msh_v4_file(msh_file, points_orig, triangles_orig)
    
    # Read back
    points, triangles = read_msh(str(msh_file))
    
    assert points.shape == (3, 2)
    assert triangles.shape == (1, 3)
    assert points.dtype == np.float64
    assert triangles.dtype == np.int32
    
    # Check values
    np.testing.assert_allclose(points, points_orig, rtol=1e-14)
    np.testing.assert_array_equal(triangles, triangles_orig)


def test_read_msh_larger_mesh(tmp_path):
    """Test reading mesh with multiple triangles."""
    points_orig = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
        [1.0, 1.0]
    ], dtype=np.float64)
    triangles_orig = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=np.int32)
    
    msh_file = tmp_path / "test_multi.msh"
    create_msh_v2_file(msh_file, points_orig, triangles_orig)
    
    points, triangles = read_msh(str(msh_file))
    
    assert points.shape == (4, 2)
    assert triangles.shape == (2, 3)
    np.testing.assert_allclose(points, points_orig, rtol=1e-14)
    np.testing.assert_array_equal(triangles, triangles_orig)


def test_read_msh_missing_file():
    """Test reading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        read_msh("nonexistent_file.msh")


def test_read_msh_empty_file(tmp_path):
    """Test reading empty file raises error."""
    empty_file = tmp_path / "empty.msh"
    empty_file.touch()
    
    with pytest.raises(ValueError, match="Empty file"):
        read_msh(str(empty_file))


def test_read_msh_no_mesh_format(tmp_path):
    """Test reading file without MeshFormat section."""
    bad_file = tmp_path / "bad.msh"
    with open(bad_file, 'w') as f:
        f.write("Some random content\n")
    
    with pytest.raises(ValueError, match="Could not detect Gmsh format version"):
        read_msh(str(bad_file))


def test_read_msh_unsupported_version(tmp_path):
    """Test reading unsupported Gmsh version."""
    bad_file = tmp_path / "bad_version.msh"
    with open(bad_file, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("1.0 0 8\n")
        f.write("$EndMeshFormat\n")
    
    with pytest.raises(ValueError, match="Unsupported Gmsh format version"):
        read_msh(str(bad_file))


def test_read_msh_no_triangles(tmp_path):
    """Test reading file without triangular elements."""
    # Create file with points but no triangle elements
    msh_file = tmp_path / "no_tris.msh"
    with open(msh_file, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write("3\n")
        f.write("1 0.0 0.0 0.0\n")
        f.write("2 1.0 0.0 0.0\n")
        f.write("3 0.5 0.8 0.0\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write("1\n")
        # Type 1 = line element (not triangle)
        f.write("1 1 2 1 1 1 2\n")
        f.write("$EndElements\n")
    
    with pytest.raises(ValueError, match="No triangular elements found"):
        read_msh(str(msh_file))


def test_read_write_roundtrip(tmp_path):
    """Test reading and writing produces consistent results."""
    # Create original mesh
    points_orig = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    triangles_orig = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [0, 2, 4]
    ], dtype=np.int32)
    
    # Write MSH
    msh_file = tmp_path / "roundtrip.msh"
    create_msh_v2_file(msh_file, points_orig, triangles_orig)
    
    # Read MSH
    points_read, triangles_read = read_msh(str(msh_file))
    
    # Write VTK
    vtk_file = tmp_path / "roundtrip.vtk"
    write_vtk(str(vtk_file), points_read, triangles_read)
    
    # Verify VTK file was created and contains expected data
    assert vtk_file.exists()
    content = vtk_file.read_text()
    assert f"POINTS {len(points_read)} double" in content
    assert f"CELLS {len(triangles_read)} {len(triangles_read) * 4}" in content


def test_read_msh_verbose_mode(tmp_path, capsys):
    """Test verbose mode prints parsing information."""
    points_orig = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8]
    ], dtype=np.float64)
    triangles_orig = np.array([[0, 1, 2]], dtype=np.int32)
    
    msh_file = tmp_path / "test_verbose.msh"
    create_msh_v2_file(msh_file, points_orig, triangles_orig)
    
    # Read with verbose=True
    read_msh(str(msh_file), verbose=True)
    
    captured = capsys.readouterr()
    assert "Detected Gmsh format version" in captured.out
    assert "Read 3 nodes" in captured.out
    assert "Read 1 triangles" in captured.out


def test_integration_with_editor(tmp_path):
    """Test I/O integration with PatchBasedMeshEditor."""
    from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
    from sofia.core.geometry import triangles_min_angles
    
    # Create and export mesh
    points_orig = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.8],
        [1.0, 1.0]
    ], dtype=np.float64)
    triangles_orig = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=np.int32)
    
    # Write MSH file
    msh_file = tmp_path / "editor_test.msh"
    create_msh_v2_file(msh_file, points_orig, triangles_orig)
    
    # Read and create editor
    points, triangles = read_msh(str(msh_file))
    editor = PatchBasedMeshEditor(points, triangles)
    
    # Verify mesh was loaded correctly
    assert len(editor.points) == 4
    assert len(editor.triangles) == 2
    
    # Compute and export quality metrics without modifying mesh
    min_angles = triangles_min_angles(editor.points, editor.triangles)
    vtk_file = tmp_path / "editor_output.vtk"
    write_vtk(str(vtk_file), editor.points, editor.triangles,
              cell_data={'min_angle': min_angles})
    
    assert vtk_file.exists()
    content = vtk_file.read_text()
    assert "min_angle" in content
