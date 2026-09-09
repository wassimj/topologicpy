import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").strip().lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _coords(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _close_xyz(a, b, tol=1.0e-6):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def test_bezier_quadratic_geometry_and_parameterization():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 2.0, 0.0)
    p2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)

    edge = Edge.Bezier([p0, p1, p2], silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsClosed(edge, silent=True) is False
    assert Edge.IsLinear(edge, silent=True) is False
    assert _close_xyz(_coords(Edge.StartVertex(edge)), [0.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.EndVertex(edge)), [2.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.VertexByParameter(edge, 0.5, silent=True)), [1.0, 1.0, 0.0])


def test_bezier_collinear_control_points_remain_linear():
    points = [
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 0.0, 0.0),
        Vertex.ByCoordinates(3.0, 0.0, 0.0),
        Vertex.ByCoordinates(4.0, 0.0, 0.0),
    ]
    edge = Edge.Bezier(points, silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, tolerance=1.0e-6, silent=True) is True
    assert math.isclose(Edge.Length(edge, mantissa=9), 4.0, rel_tol=1.0e-8, abs_tol=1.0e-8)


def test_rational_bezier_exact_quarter_circle():
    s2 = math.sqrt(2.0) / 2.0
    points = [
        Vertex.ByCoordinates(1.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 1.0, 0.0),
        Vertex.ByCoordinates(0.0, 1.0, 0.0),
    ]
    edge = Edge.Bezier(points, weights=[1.0, s2, 1.0], silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    mid = _coords(Edge.VertexByParameter(edge, 0.5, silent=True))
    assert _close_xyz(mid, [s2, s2, 0.0], tol=1.0e-6)
    assert math.isclose(Edge.Length(edge, mantissa=9), math.pi / 2.0, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_bezier_reversed_control_points_preserve_direction():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 2.0, 0.0)
    p2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)

    edge = Edge.Bezier([p2, p1, p0], silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert _close_xyz(_coords(Edge.StartVertex(edge)), [2.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.EndVertex(edge)), [0.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.VertexByParameter(edge, 0.5, silent=True)), [1.0, 1.0, 0.0])
    tangent = Edge.TangentAtParameter(edge, 0.25, mantissa=None, silent=True)
    assert tangent is not None
    assert tangent[0] < 0.0


def test_bezier_validates_inputs():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)

    assert Edge.Bezier(None, silent=True) is None
    assert Edge.Bezier([p0], silent=True) is None
    assert Edge.Bezier([p0, None, p1], silent=True) is None
    assert Edge.Bezier([p0, p1], weights=[1.0], silent=True) is None
    assert Edge.Bezier([p0, p1], weights=[1.0, 0.0], silent=True) is None
    assert Edge.Bezier([p0, p1], weights=[1.0, float("nan")], silent=True) is None
    assert Edge.Bezier([p0, p1], tolerance=0.0, silent=True) is None


@pytest.mark.skipif(not IS_PYTHONOCC, reason="Native OCCT curve-type inspection is PythonOCC-specific.")
def test_pythonocc_bezier_is_native_bspline_curve():
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve

    edge = Edge.Bezier(
        [
            Vertex.ByCoordinates(0.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 2.0, 0.0),
            Vertex.ByCoordinates(2.0, 0.0, 0.0),
        ],
        silent=True,
    )
    assert Topology.IsInstance(edge, "Edge")
    adaptor = BRepAdaptor_Curve(edge.shape)
    assert adaptor.GetType() == GeomAbs_BSplineCurve
