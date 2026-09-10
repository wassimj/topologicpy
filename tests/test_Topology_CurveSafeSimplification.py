import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "topologic_core").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _rect_face(x0, y0, x1, y1, z=0.0):
    vertices = [
        Vertex.ByCoordinates(x0, y0, z),
        Vertex.ByCoordinates(x1, y0, z),
        Vertex.ByCoordinates(x1, y1, z),
        Vertex.ByCoordinates(x0, y1, z),
    ]
    wire = Wire.ByVertices(vertices, close=True, silent=True)
    assert Topology.IsInstance(wire, "Wire")
    face = Face.ByWire(wire, silent=True)
    assert Topology.IsInstance(face, "Face")
    return face


def _split_rectangle_face():
    vertices = [
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 0.0, 0.0),
        Vertex.ByCoordinates(2.0, 0.0, 0.0),
        Vertex.ByCoordinates(2.0, 1.0, 0.0),
        Vertex.ByCoordinates(0.0, 1.0, 0.0),
    ]
    wire = Wire.ByVertices(vertices, close=True, silent=True)
    face = Face.ByWire(wire, silent=True)
    assert Topology.IsInstance(face, "Face")
    return face


def _semicircle_face_with_split_diameter():
    arc = Edge.Arc(radius=1.0, fromAngle=0.0, toAngle=180.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    a = Edge.EndVertex(arc)
    b = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    c = Edge.StartVertex(arc)
    e1 = Edge.ByVertices([a, b], silent=True)
    e2 = Edge.ByVertices([b, c], silent=True)
    wire = Wire.ByEdges([arc, e1, e2], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    face = Face.ByWire(wire, silent=True)
    assert Topology.IsInstance(face, "Face")
    return face


def _circle_wire(z, radius=1.0):
    edge = Edge.Circle(
        origin=Vertex.ByCoordinates(0.0, 0.0, z),
        radius=radius,
        silent=True,
    )
    assert Topology.IsInstance(edge, "Edge")
    wire = Wire.ByEdges([edge], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


def test_remove_collinear_edges_still_simplifies_polyhedral_face():
    face = _split_rectangle_face()
    before = Topology.Edges(face) or []
    assert len(before) == 5

    result = Topology.RemoveCollinearEdges(
        face,
        polyhedron=True,
        silent=True,
    )
    assert Topology.IsInstance(result, "Face")
    after = Topology.Edges(result) or []
    assert len(after) == 4
    assert math.isclose(Face.Area(result, mantissa=None, silent=True), 2.0, rel_tol=1e-6, abs_tol=1e-6)


def test_curve_safe_remove_collinear_edges_never_flattens_arc():
    face = _semicircle_face_with_split_diameter()
    before = Topology.Edges(face) or []
    assert len(before) == 3
    assert sum(Edge.IsLinear(edge, silent=True) is False for edge in before) == 1

    result = Topology.RemoveCollinearEdges(
        face,
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(result, "Face")

    after = Topology.Edges(result) or []
    curved = [edge for edge in after if Edge.IsLinear(edge, silent=True) is False]
    assert len(curved) == 1

    if IS_PYTHONOCC:
        # Native KeepShape protection allows only the split straight diameter
        # to simplify while retaining the exact arc.
        assert len(after) == 2
    else:
        # TopologicCore deliberately preserves the mixed topology unchanged
        # rather than rebuilding and flattening the curve.
        assert len(after) == 3

    assert math.isclose(
        Edge.Length(curved[0], mantissa=None, silent=True),
        math.pi,
        rel_tol=2e-6,
        abs_tol=2e-6,
    )


def test_remove_coplanar_faces_still_merges_adjacent_planar_faces():
    f1 = _rect_face(0.0, 0.0, 1.0, 1.0)
    f2 = _rect_face(1.0, 0.0, 2.0, 1.0)
    shell = Shell.ByFaces([f1, f2], silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert len(Topology.Faces(shell) or []) == 2

    result = Topology.RemoveCoplanarFaces(
        shell,
        silent=True,
    )
    assert Topology.IsInstance(result, "Topology")
    faces = Topology.Faces(result) or []
    if Topology.IsInstance(result, "Face"):
        faces = [result]
    assert len(faces) == 1
    assert math.isclose(Face.Area(faces[0], mantissa=None, silent=True), 2.0, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.skipif(not IS_PYTHONOCC, reason="Exact curved Shell construction is PythonOCC-specific.")
def test_remove_coplanar_faces_preserves_and_unifies_cylindrical_surfaces():
    shell_a = Shell.ByWires(
        [_circle_wire(0.0), _circle_wire(1.0)],
        polyhedron=False,
        silent=True,
    )
    shell_b = Shell.ByWires(
        [_circle_wire(1.0), _circle_wire(2.0)],
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(shell_a, "Shell")
    assert Topology.IsInstance(shell_b, "Shell")

    faces = (Topology.Faces(shell_a) or []) + (Topology.Faces(shell_b) or [])
    shell = Shell.ByFaces(faces, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    before = Topology.Faces(shell) or []
    assert len(before) >= 2
    assert all(Face.IsPlanar(face, silent=True) is False for face in before)

    area_before = sum(Face.Area(face, mantissa=None, silent=True) for face in before)
    result = Topology.RemoveCoplanarFaces(
        shell,
        silent=True,
    )
    assert Topology.IsInstance(result, "Topology")
    after = Topology.Faces(result) or []
    if Topology.IsInstance(result, "Face"):
        after = [result]

    # Adjacent faces on the same cylindrical support surface may be
    # legitimately unified into a single curved face.
    assert len(after) >= 1
    assert len(after) <= len(before)

    # The operation must not flatten the cylindrical geometry.
    assert all(
        Face.IsPlanar(face, silent=True) is False
        for face in after
    )

    # The total curved surface area must be preserved.
    area_after = sum(
        Face.Area(
            face,
            mantissa=None,
            silent=True
        )
        for face in after
    )

    assert math.isclose(
        area_after,
        area_before,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )


def test_remove_collinear_edges_validates_polyhedron_flag():
    face = _split_rectangle_face()

    assert Topology.RemoveCollinearEdges(
        face,
        polyhedron="False",
        silent=True
    ) is None
