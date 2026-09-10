import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "topologic_core").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _xyz(vertex):
    return [
        Vertex.X(vertex, mantissa=9),
        Vertex.Y(vertex, mantissa=9),
        Vertex.Z(vertex, mantissa=9),
    ]


def _quarter_cylinder_face(radius=1.0, height=2.0):
    s2 = math.sqrt(2.0) / 2.0
    cps = [
        [Vertex.ByCoordinates(radius, 0.0, 0.0), Vertex.ByCoordinates(radius, 0.0, height)],
        [Vertex.ByCoordinates(radius, radius, 0.0), Vertex.ByCoordinates(radius, radius, height)],
        [Vertex.ByCoordinates(0.0, radius, 0.0), Vertex.ByCoordinates(0.0, radius, height)],
    ]
    weights = [
        [1.0, 1.0],
        [s2, s2],
        [1.0, 1.0],
    ]
    return Face.ByNurbsParameters(
        cps,
        weights=weights,
        uKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 1.0, 1.0],
        isRational=True,
        uDegree=2,
        vDegree=1,
        silent=True,
    )


def test_transform_preserves_arc_geometry_and_start_end_direction():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=90.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    original_length = Edge.Length(arc, mantissa=None, silent=True)

    matrix = [
        [0.0, -1.0, 0.0, 10.0],
        [1.0,  0.0, 0.0, 20.0],
        [0.0,  0.0, 1.0,  5.0],
        [0.0,  0.0, 0.0,  1.0],
    ]
    transformed = Topology.Transform(arc, matrix, silent=True)
    assert Topology.IsInstance(transformed, "Edge")
    assert Edge.IsLinear(transformed, silent=True) is False
    assert math.isclose(
        Edge.Length(transformed, mantissa=None, silent=True),
        original_length,
        rel_tol=2e-6,
        abs_tol=2e-6,
    )

    # Original start is (2,0,0). After +90deg about Z and translation it is
    # (10,22,5). This explicitly guards topological start->end orientation.
    start = _xyz(Edge.StartVertex(transformed))
    assert all(abs(a-b) <= 2e-6 for a,b in zip(start, [10.0, 22.0, 5.0]))


@pytest.mark.pythonocc_only
def test_transform_preserves_nurbs_surface_and_area():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False
    area = Face.Area(face, mantissa=None, silent=True)

    matrix = [
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 0.0,-1.0, 3.0],
        [0.0, 1.0, 0.0,-2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    transformed = Topology.Transform(face, matrix, silent=True)
    assert Topology.IsInstance(transformed, "Face")
    assert Face.IsPlanar(transformed, silent=True) is False
    assert math.isclose(
        Face.Area(transformed, mantissa=None, silent=True),
        area,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


@pytest.mark.pythonocc_only
def test_geometry_and_meshdata_tessellate_curved_face_without_failure():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")

    geometry = Topology.Geometry(face, triangulate=True, silent=True)
    assert isinstance(geometry, dict)
    assert isinstance(geometry.get("vertices"), list) and len(geometry["vertices"]) >= 3
    assert isinstance(geometry.get("faces"), list) and len(geometry["faces"]) >= 1

    mesh = Topology.MeshData(face, mode=0, silent=True)
    assert isinstance(mesh, dict)
    assert isinstance(mesh.get("vertices"), list) and len(mesh["vertices"]) >= 3
    assert isinstance(mesh.get("faces"), list) and len(mesh["faces"]) >= 1


@pytest.mark.pythonocc_only
def test_triangulate_curved_face_intentionally_returns_planar_triangles():
    face = _quarter_cylinder_face()
    result = Topology.Triangulate(face, silent=True)
    assert Topology.IsInstance(result, "Topology")

    faces = Topology.Faces(result) or []
    if Topology.IsInstance(result, "Face"):
        faces = [result]
    assert len(faces) >= 1
    for triangle in faces:
        vertices = Topology.Vertices(triangle) or []
        assert len(vertices) == 3
        assert Face.IsPlanar(triangle, silent=True) is True
