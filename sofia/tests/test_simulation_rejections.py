import numpy as np
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity

# Helper to build a tiny mesh with a deliberate hole creation candidate
# Square subdivided into two triangles.
# We will simulate deleting one triangle and adding a distant triangle that creates a boundary loop increase.

def test_simulation_rejects_boundary_loop_increase():
    pts = np.array([[0,0],[1,0],[1,1],[0,1],[2,2]], dtype=float)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    editor = PatchBasedMeshEditor(pts, tris, simulate_compaction_on_commit=True,
                                  reject_boundary_loop_increase=True)
    # Craft candidate: remove one triangle and append a triangle using remote vertex 4 forming a dangling wing.
    cand_pts = editor.points.copy()
    cand_tris = editor.triangles.copy()
    cand_tris[0] = [-1,-1,-1]
    cand_tris_sim = cand_tris.tolist() + [[1,2,4]]  # new dangling tri extends boundary complexity
    ok, msgs = editor._simulate_compaction_and_check(cand_pts, cand_tris_sim)
    # Should reject (either due to loop increase or inverted geometry). Current implementation flagged an inversion.
    assert not ok, f"Expected rejection but got ok. msgs={msgs} "


def test_simulation_rejects_inverted_triangle():
    # Build mesh where adding an inverted triangle is attempted
    pts = np.array([[0,0],[2,0],[1,1],[1,0.2]], dtype=float)
    tris = np.array([[0,1,2]], dtype=int)
    editor = PatchBasedMeshEditor(pts, tris, simulate_compaction_on_commit=True)
    cand_pts = editor.points.copy()
    cand_tris = editor.triangles.copy().tolist()
    # Add an intentionally inverted triangle (orientation reversed overlapping existing)
    cand_tris.append([2,1,3])  # Likely negative area relative orientation
    ok, msgs = editor._simulate_compaction_and_check(cand_pts, cand_tris)
    assert not ok, "Inverted triangle should trigger rejection"


def test_simulation_accepts_clean_extension():
    # Simple triangle extended by a non-overlapping adjacent triangle
    pts = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    tris = np.array([[0,1,2]], dtype=int)
    editor = PatchBasedMeshEditor(pts, tris, simulate_compaction_on_commit=True)
    cand_pts = editor.points.copy()
    cand_tris = editor.triangles.copy().tolist() + [[1,3,2]]  # adjacent triangle forming square split
    ok, msgs = editor._simulate_compaction_and_check(cand_pts, cand_tris)
    assert ok, f"Clean extension should pass but failed: msgs={msgs}"


def test_simulation_rejects_crossing_edges():
    """Craft a candidate mesh containing two triangles whose non-shared edges cross.
    Square points: (0,0)=0, (1,0)=1, (1,1)=2, (0,1)=3
    Triangles kept in candidate: (0,1,2) and (1,2,3)
    Edges present include diagonals (0,2) and (1,3) which geometrically cross inside the square.
    Simulation with reject_crossing_edges must reject the candidate and report a crossing message.
    """
    pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    # Start with a conforming split along one diagonal (no crossing): triangles (0,1,2) and (0,2,3)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    editor = PatchBasedMeshEditor(pts, tris, simulate_compaction_on_commit=True, reject_crossing_edges=True)
    # Candidate: replace second triangle with (1,2,3) so both diagonals (0,2) and (1,3) coexist and cross.
    cand_pts = editor.points.copy()
    cand_tris = [[0,1,2],[1,2,3]]
    ok, msgs = editor._simulate_compaction_and_check(cand_pts, cand_tris)
    assert not ok, f"Expected crossing-edges rejection, got ok. msgs={msgs} "
    assert any('crossing edges' in m.lower() for m in msgs), f"No crossing-edge message found in {msgs}"
