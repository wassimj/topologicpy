"""Regression checks for native incidence during grid slicing."""
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").lower() != "pythonocc",
    reason="Exercises PythonOCC's sewn-shell assembly",
)


@pytest.mark.parametrize("size", [6, 30])
def test_grid_slice_uses_native_incidence(monkeypatch, size):
    from topologicpy.Face import Face
    from topologicpy.Grid import Grid
    from topologicpy.Topology import Topology
    from topologicpy.pythonocc_backend.shell import Shell as NativeShell

    face = Face.Rectangle(width=size, length=size)
    grid = Grid.OnFace(face, spacing=1)

    def reject_pairwise_matching(*args, **kwargs):
        pytest.fail("Sewn grid faces must use native identity, not curve sampling")

    monkeypatch.setattr(NativeShell, "_EdgesSame", reject_pairwise_matching)
    result = Topology.Slice(face, grid)
    assert Topology.IsInstance(result, "Shell")
    faces = Topology.Faces(result)
    assert len(faces) > 1
    assert sum(Face.Area(f) for f in faces) == pytest.approx(Face.Area(face))
    edges = Topology.Edges(result)
    # Sample both ends: boundary-first extraction puts internal Edges last.
    counts = [len(edge.Faces(result)) for edge in edges[:3] + edges[-3:]]
    assert set(counts) == {1, 2}


def test_sewn_membership_preserves_shared_curved_edges(monkeypatch):
    from topologicpy.Face import Face
    from topologicpy.Shell import Shell
    from topologicpy.Topology import Topology
    from topologicpy.pythonocc_backend.shell import Shell as NativeShell

    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.TopAbs import TopAbs_FACE
    from topologicpy.pythonocc_backend.topology import _iter_occ_subshapes

    native = BRepPrimAPI_MakeCylinder(2, 3).Shape()
    faces = [Topology.ByOCCTShape(shape) for shape in _iter_occ_subshapes(native, TopAbs_FACE)]

    def reject_pairwise_matching(*args, **kwargs):
        pytest.fail("Sewn curved faces must use native edge identity")

    monkeypatch.setattr(NativeShell, "_EdgesSame", reject_pairwise_matching)
    shell = Shell.ByFaces(faces)
    assert Topology.IsInstance(shell, "Shell")
    assert len(Topology.Faces(shell)) == 3
    assert sum(Face.Area(f) for f in Topology.Faces(shell)) == pytest.approx(20 * 3.141592653589793)
    assert any(len(edge.Faces(shell)) == 2 for edge in Topology.Edges(shell))
