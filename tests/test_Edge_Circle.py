import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "auto").strip().lower()
IS_PYTHONOCC = BACKEND in {"pythonocc", "occ", "python_occ"}


def _coords(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _distance(a, b):
    pa = _coords(a)
    pb = _coords(b)
    return math.sqrt(sum((float(pa[i]) - float(pb[i])) ** 2 for i in range(3)))


def test_edge_circle_creates_closed_non_linear_exact_length():
    radius = 2.0
    circle = Edge.Circle(radius=radius, silent=True)

    assert Topology.IsInstance(circle, "Edge")
    assert Edge.IsClosed(circle, silent=True) is True
    assert Edge.IsLinear(circle, silent=True) is False
    assert math.isclose(Edge.Length(circle, mantissa=9), 2.0 * math.pi * radius, rel_tol=1e-7, abs_tol=1e-7)


def test_edge_circle_center_placement_and_parameter_samples():
    origin = Vertex.ByCoordinates(10.0, -3.0, 7.0)
    radius = 3.0
    circle = Edge.Circle(origin=origin, radius=radius, direction=[0, 0, 1], placement="center", silent=True)

    assert Topology.IsInstance(circle, "Edge")
    for u in (0.0, 0.25, 0.5, 0.75):
        vertex = Edge.VertexByParameter(circle, u=u, silent=True)
        assert Topology.IsInstance(vertex, "Vertex")
        assert math.isclose(_distance(vertex, origin), radius, rel_tol=1e-7, abs_tol=1e-7)
        assert math.isclose(Vertex.Z(vertex, mantissa=9), 7.0, rel_tol=0.0, abs_tol=1e-7)


def test_edge_circle_corner_placement():
    origin = Vertex.ByCoordinates(5.0, 8.0, 0.0)
    radius = 2.0
    circle = Edge.Circle(origin=origin, radius=radius, placement="lowerleft", silent=True)

    assert Topology.IsInstance(circle, "Edge")
    samples = [Edge.VertexByParameter(circle, u=u, silent=True) for u in (0.0, 0.25, 0.5, 0.75)]
    xs = [Vertex.X(v, mantissa=9) for v in samples]
    ys = [Vertex.Y(v, mantissa=9) for v in samples]
    assert math.isclose(min(xs), 5.0, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(min(ys), 8.0, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(max(xs), 9.0, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(max(ys), 12.0, rel_tol=0.0, abs_tol=1e-7)


def test_edge_circle_oriented_plane():
    origin = Vertex.ByCoordinates(1.0, 2.0, 3.0)
    radius = 1.5
    circle = Edge.Circle(origin=origin, radius=radius, direction=[0, 1, 0], placement="center", silent=True)

    assert Topology.IsInstance(circle, "Edge")
    for u in (0.0, 0.25, 0.5, 0.75):
        vertex = Edge.VertexByParameter(circle, u=u, silent=True)
        assert Topology.IsInstance(vertex, "Vertex")
        # Plane normal is +Y, so every point lies at y=2.
        assert math.isclose(Vertex.Y(vertex, mantissa=9), 2.0, rel_tol=0.0, abs_tol=1e-7)
        assert math.isclose(_distance(vertex, origin), radius, rel_tol=1e-7, abs_tol=1e-7)


def test_edge_circle_validates_inputs():
    assert Edge.Circle(origin="not a vertex", silent=True) is None
    assert Edge.Circle(radius=0.0, silent=True) is None
    assert Edge.Circle(radius=1e-6, tolerance=1e-4, silent=True) is None
    assert Edge.Circle(direction=[0, 0, 0], silent=True) is None
    assert Edge.Circle(direction=[0, 1], silent=True) is None
    assert Edge.Circle(placement="banana", silent=True) is None
    assert Edge.Circle(tolerance=0.0, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_circle_is_native_occt_circle():
    circle = Edge.Circle(radius=2.5, silent=True)
    assert Topology.IsInstance(circle, "Edge")

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Circle

    adaptor = BRepAdaptor_Curve(circle.shape)
    assert adaptor.GetType() == GeomAbs_Circle
