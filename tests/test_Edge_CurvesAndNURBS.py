import math

import pytest

from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


TOL = 1.0e-4


def _xyz(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _assert_xyz_close(vertex, expected, abs_tol=1.0e-6):
    actual = _xyz(vertex)
    assert actual is not None
    assert len(actual) == 3
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


def test_exact_arc_and_circle_are_true_curved_edges():
    arc = Edge.Arc(
        radius=2.0,
        fromAngle=0.0,
        toAngle=180.0,
        tolerance=TOL,
        silent=True,
    )
    assert Topology.IsInstance(arc, "Edge")
    assert Edge.IsLinear(arc, tolerance=TOL, silent=True) is False
    assert Edge.IsClosed(arc, tolerance=TOL, silent=True) is False
    assert Edge.Length(arc, mantissa=None, tolerance=TOL, silent=True) == pytest.approx(
        2.0 * math.pi,
        rel=1.0e-6,
        abs=1.0e-6,
    )

    circle = Edge.Circle(
        radius=2.0,
        tolerance=TOL,
        silent=True,
    )
    assert Topology.IsInstance(circle, "Edge")
    assert Edge.IsLinear(circle, tolerance=TOL, silent=True) is False
    assert Edge.IsClosed(circle, tolerance=TOL, silent=True) is True
    assert Edge.Length(circle, mantissa=None, tolerance=TOL, silent=True) == pytest.approx(
        4.0 * math.pi,
        rel=1.0e-6,
        abs=1.0e-6,
    )


def test_by_nurbs_parameters_creates_one_curved_edge_with_exact_endpoints():
    control_points = [
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 2.0, 0.0),
        Vertex.ByCoordinates(2.0, 2.0, 0.0),
        Vertex.ByCoordinates(3.0, 0.0, 0.0),
    ]

    edge = Edge.ByNurbsParameters(
        controlPoints=control_points,
        degree=3,
        isRational=False,
        isPeriodic=False,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert len(Topology.Edges(edge, silent=True) or []) == 1
    assert Edge.IsLinear(edge, tolerance=TOL, silent=True) is False

    _assert_xyz_close(
        Edge.VertexByParameter(edge, u=0.0, tolerance=TOL, silent=True),
        [0.0, 0.0, 0.0],
    )
    _assert_xyz_close(
        Edge.VertexByParameter(edge, u=1.0, tolerance=TOL, silent=True),
        [3.0, 0.0, 0.0],
    )


def test_quadratic_bezier_midpoint_is_exact():
    edge = Edge.Bezier(
        [
            Vertex.ByCoordinates(0.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 2.0, 0.0),
            Vertex.ByCoordinates(2.0, 0.0, 0.0),
        ],
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, tolerance=TOL, silent=True) is False

    midpoint = Edge.VertexByParameter(
        edge,
        u=0.5,
        tolerance=TOL,
        silent=True,
    )
    _assert_xyz_close(midpoint, [1.0, 1.0, 0.0])


def test_rational_bezier_is_supported():
    edge = Edge.Bezier(
        [
            Vertex.ByCoordinates(1.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 1.0, 0.0),
            Vertex.ByCoordinates(0.0, 1.0, 0.0),
        ],
        weights=[1.0, math.sqrt(0.5), 1.0],
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, tolerance=TOL, silent=True) is False

    _assert_xyz_close(
        Edge.VertexByParameter(edge, u=0.0, tolerance=TOL, silent=True),
        [1.0, 0.0, 0.0],
    )
    _assert_xyz_close(
        Edge.VertexByParameter(edge, u=1.0, tolerance=TOL, silent=True),
        [0.0, 1.0, 0.0],
    )


def test_parabola_samples_satisfy_implicit_equation():
    focal_length = 0.75

    edge = Edge.Parabola(
        focalLength=focal_length,
        fromParameter=-1.5,
        toParameter=1.5,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, tolerance=TOL, silent=True) is False

    for u in [0.0, 0.2, 0.5, 0.8, 1.0]:
        vertex = Edge.VertexByParameter(
            edge,
            u=u,
            tolerance=TOL,
            silent=True,
        )
        x, y, z = _xyz(vertex)

        assert z == pytest.approx(0.0, abs=1.0e-7)
        assert y == pytest.approx(
            x * x / (4.0 * focal_length),
            rel=1.0e-6,
            abs=1.0e-6,
        )


def test_hyperbola_samples_satisfy_implicit_equation():
    a = 1.5
    b = 0.8

    edge = Edge.Hyperbola(
        a=a,
        b=b,
        fromParameter=-1.25,
        toParameter=1.25,
        branch="right",
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, tolerance=TOL, silent=True) is False

    for u in [0.0, 0.2, 0.5, 0.8, 1.0]:
        vertex = Edge.VertexByParameter(
            edge,
            u=u,
            tolerance=TOL,
            silent=True,
        )
        x, y, z = _xyz(vertex)

        assert z == pytest.approx(0.0, abs=1.0e-7)
        value = (x * x) / (a * a) - (y * y) / (b * b)
        assert value == pytest.approx(1.0, rel=2.0e-6, abs=2.0e-6)


def test_helix_is_one_curved_edge_with_expected_endpoints():
    radius = 1.25
    height = 3.0
    turns = 1.5

    edge = Edge.Helix(
        radius=radius,
        height=height,
        turns=turns,
        sides=16,
        clockwise=False,
        placement="base",
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert len(Topology.Edges(edge, silent=True) or []) == 1
    assert Edge.IsLinear(edge, tolerance=TOL, silent=True) is False

    _assert_xyz_close(
        Edge.VertexByParameter(edge, u=0.0, tolerance=TOL, silent=True),
        [radius, 0.0, 0.0],
        abs_tol=1.0e-6,
    )
    _assert_xyz_close(
        Edge.VertexByParameter(edge, u=1.0, tolerance=TOL, silent=True),
        [-radius, 0.0, height],
        abs_tol=1.0e-5,
    )

    for u in [0.2, 0.4, 0.6, 0.8]:
        x, y, z = _xyz(
            Edge.VertexByParameter(
                edge,
                u=u,
                tolerance=TOL,
                silent=True,
            )
        )
        radial_distance = math.sqrt(x * x + y * y)
        assert radial_distance == pytest.approx(radius, rel=5.0e-3, abs=5.0e-3)
        assert -1.0e-6 <= z <= height + 1.0e-6
