import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "auto").strip().lower()
IS_PYTHONOCC = BACKEND in {"pythonocc", "occ", "python_occ"}


def _coords(vertex):
    return [float(v) for v in Vertex.Coordinates(vertex, mantissa=None)]


def _assert_xyz(vertex, expected, tol=1.0e-6):
    xyz = _coords(vertex)
    assert all(math.isclose(xyz[i], float(expected[i]), rel_tol=0.0, abs_tol=tol) for i in range(3))


def _distance(a, b):
    pa = _coords(a)
    pb = _coords(b)
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))


def _quarter_circle_nurbs(radius=2.0):
    control_points = [
        Vertex.ByCoordinates(radius, 0.0, 0.0),
        Vertex.ByCoordinates(radius, radius, 0.0),
        Vertex.ByCoordinates(0.0, radius, 0.0),
    ]
    return Edge.ByNurbsParameters(
        controlPoints=control_points,
        weights=[1.0, math.sqrt(0.5), 1.0],
        knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        isRational=True,
        isPeriodic=False,
        degree=2,
        silent=True,
    )


def test_edge_bynurbsparameters_exact_quarter_circle():
    radius = 2.0
    edge = _quarter_circle_nurbs(radius)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsClosed(edge, silent=True) is False
    assert Edge.IsLinear(edge, silent=True) is False
    assert math.isclose(
        Edge.Length(edge, mantissa=9),
        0.5 * math.pi * radius,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

    _assert_xyz(Edge.StartVertex(edge, silent=True), [radius, 0.0, 0.0])
    _assert_xyz(Edge.EndVertex(edge, silent=True), [0.0, radius, 0.0])
    midpoint = Edge.VertexByParameter(edge, u=0.5, silent=True)
    expected = radius / math.sqrt(2.0)
    _assert_xyz(midpoint, [expected, expected, 0.0], tol=2.0e-6)


def test_edge_bynurbsparameters_nonrational_quadratic():
    control_points = [
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 1.0, 0.0),
        Vertex.ByCoordinates(2.0, 0.0, 0.0),
    ]
    edge = Edge.ByNurbsParameters(
        controlPoints=control_points,
        knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        degree=2,
        isRational=False,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    _assert_xyz(Edge.VertexByParameter(edge, u=0.5, silent=True), [1.0, 0.5, 0.0], tol=2.0e-6)
    assert Edge.Length(edge, mantissa=9) > 2.0


def test_edge_bynurbsparameters_validates_inputs():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)
    p2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)

    assert Edge.ByNurbsParameters([], silent=True) is None
    assert Edge.ByNurbsParameters([p0, p1], degree=2, silent=True) is None
    assert Edge.ByNurbsParameters([p0, p1, p2], weights=[1.0, 1.0], degree=2, silent=True) is None
    assert Edge.ByNurbsParameters([p0, p1, p2], weights=[1.0, 0.0, 1.0], degree=2, silent=True) is None
    assert Edge.ByNurbsParameters(
        [p0, p1, p2],
        knots=[0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
        degree=2,
        silent=True,
    ) is None


def test_edge_arc_quarter_circle_exact_geometry():
    radius = 2.0
    arc = Edge.Arc(radius=radius, fromAngle=0.0, toAngle=90.0, silent=True)

    assert Topology.IsInstance(arc, "Edge")
    assert Edge.IsClosed(arc, silent=True) is False
    assert Edge.IsLinear(arc, silent=True) is False
    assert math.isclose(
        Edge.Length(arc, mantissa=9),
        0.5 * math.pi * radius,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )
    _assert_xyz(Edge.StartVertex(arc, silent=True), [radius, 0.0, 0.0])
    _assert_xyz(Edge.EndVertex(arc, silent=True), [0.0, radius, 0.0])
    expected = radius / math.sqrt(2.0)
    _assert_xyz(Edge.VertexByParameter(arc, u=0.5, silent=True), [expected, expected, 0.0], tol=2.0e-6)


def test_edge_arc_wraps_angles_counter_clockwise():
    radius = 1.5
    arc = Edge.Arc(radius=radius, fromAngle=300.0, toAngle=60.0, silent=True)

    assert Topology.IsInstance(arc, "Edge")
    expected_length = radius * math.radians(120.0)
    assert math.isclose(Edge.Length(arc, mantissa=9), expected_length, rel_tol=1.0e-6, abs_tol=1.0e-6)

    a0 = math.radians(300.0)
    a1 = math.radians(60.0)
    _assert_xyz(Edge.StartVertex(arc, silent=True), [radius * math.cos(a0), radius * math.sin(a0), 0.0], tol=2.0e-6)
    _assert_xyz(Edge.EndVertex(arc, silent=True), [radius * math.cos(a1), radius * math.sin(a1), 0.0], tol=2.0e-6)


def test_edge_arc_start_placement_and_oriented_plane():
    origin = Vertex.ByCoordinates(10.0, -3.0, 7.0)
    radius = 1.25
    arc = Edge.Arc(
        origin=origin,
        radius=radius,
        fromAngle=0.0,
        toAngle=90.0,
        direction=[0.0, 1.0, 0.0],
        placement="start",
        silent=True,
    )

    assert Topology.IsInstance(arc, "Edge")
    assert _distance(Edge.StartVertex(arc, silent=True), origin) <= 2.0e-6
    assert math.isclose(Edge.Length(arc, mantissa=9), 0.5 * math.pi * radius, rel_tol=1.0e-6, abs_tol=1.0e-6)

    # The arc plane normal is +Y, so every sampled point lies in y = origin.y.
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        vertex = Edge.VertexByParameter(arc, u=u, silent=True)
        assert Topology.IsInstance(vertex, "Vertex")
        assert math.isclose(Vertex.Y(vertex, mantissa=9), -3.0, rel_tol=0.0, abs_tol=2.0e-6)


def test_edge_arc_validates_inputs():
    assert Edge.Arc(origin="not a vertex", silent=True) is None
    assert Edge.Arc(radius=0.0, silent=True) is None
    assert Edge.Arc(radius=1.0e-6, tolerance=1.0e-4, silent=True) is None
    assert Edge.Arc(fromAngle=10.0, toAngle=10.0, silent=True) is None
    assert Edge.Arc(fromAngle=0.0, toAngle=360.0, silent=True) is None
    assert Edge.Arc(direction=[0.0, 0.0, 0.0], silent=True) is None
    assert Edge.Arc(direction=[0.0, 1.0], silent=True) is None
    assert Edge.Arc(placement="banana", silent=True) is None
    assert Edge.Arc(tolerance=0.0, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_nurbs_and_arc_use_native_curve_types():
    nurbs = _quarter_circle_nurbs(2.0)
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=90.0, silent=True)
    assert Topology.IsInstance(nurbs, "Edge")
    assert Topology.IsInstance(arc, "Edge")

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_Circle

    assert BRepAdaptor_Curve(nurbs.shape).GetType() == GeomAbs_BSplineCurve
    assert BRepAdaptor_Curve(arc.shape).GetType() == GeomAbs_Circle
