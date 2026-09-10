"""Tranche 05B: remaining Edge curve constructors and curve-integrity operations."""

import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _xyz(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _close(a, b, tol=1.0e-5):
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


def _norm(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def test_bycurve_creates_single_non_linear_bspline_edge():
    points = [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(1, 2, 0),
        Vertex.ByCoordinates(3, 2, 0),
        Vertex.ByCoordinates(4, 0, 0),
    ]
    edge = Edge.ByCurve(points, degree=3, silent=True)
    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    assert _close(_xyz(Edge.StartVertex(edge, silent=True)), [0, 0, 0])
    assert _close(_xyz(Edge.EndVertex(edge, silent=True)), [4, 0, 0])


def test_parabola_is_exact_quadratic_conic():
    f = 0.75
    edge = Edge.Parabola(focalLength=f, fromParameter=-1.5, toParameter=1.5, silent=True)
    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    for u in [0.0, 0.2, 0.5, 0.8, 1.0]:
        x, y, z = _xyz(Edge.VertexByParameter(edge, u, silent=True))
        assert abs(z) <= 1.0e-7
        assert math.isclose(y, x*x/(4.0*f), rel_tol=2.0e-6, abs_tol=2.0e-6)


def test_hyperbola_is_exact_rational_conic_on_both_branches():
    a, b = 2.0, 0.8
    for branch, sign in [("right", 1.0), ("left", -1.0)]:
        edge = Edge.Hyperbola(a=a, b=b, fromParameter=-0.8, toParameter=0.8, branch=branch, silent=True)
        assert Topology.IsInstance(edge, "Edge")
        assert Edge.IsLinear(edge, silent=True) is False
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x, y, z = _xyz(Edge.VertexByParameter(edge, u, silent=True))
            assert sign*x > 0.0
            assert abs(z) <= 1.0e-7
            value = x*x/(a*a) - y*y/(b*b)
            assert math.isclose(value, 1.0, rel_tol=3.0e-6, abs_tol=3.0e-6)


def test_helix_is_one_curved_edge_with_expected_endpoints_and_length():
    radius, height, turns = 1.25, 3.0, 1.5
    edge = Edge.Helix(radius=radius, height=height, turns=turns, sides=20, placement="base", silent=True)
    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    start = _xyz(Edge.StartVertex(edge, silent=True))
    end = _xyz(Edge.EndVertex(edge, silent=True))
    expected_end = [radius * math.cos(2*math.pi*turns), radius * math.sin(2*math.pi*turns), height]
    assert _close(start, [radius, 0.0, 0.0], tol=2.0e-5)
    assert _close(end, expected_end, tol=2.0e-5)
    analytic = math.sqrt((2.0*math.pi*radius*turns)**2 + height**2)
    actual = Edge.Length(edge, mantissa=None, silent=True)
    assert actual is not None
    assert math.isclose(actual, analytic, rel_tol=3.0e-4, abs_tol=3.0e-4)


def test_curve_normal_is_unit_and_perpendicular_to_tangent():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=180.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    tangent = Edge.TangentAtParameter(arc, u=0.5, mantissa=None, silent=True)
    normal = Edge.NormalAtParameter(arc, u=0.5, mantissa=None, silent=True)
    assert tangent is not None and normal is not None
    assert math.isclose(_norm(tangent), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert math.isclose(_norm(normal), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert abs(_dot(tangent, normal)) <= 2.0e-5
    # Midpoint of the upper semicircle is (0, 2, 0); principal normal points inward.
    midpoint = _xyz(Edge.VertexByParameter(arc, 0.5, silent=True))
    radial_inward = [-midpoint[0], -midpoint[1], -midpoint[2]]
    rmag = _norm(radial_inward)
    radial_inward = [x/rmag for x in radial_inward]
    assert _dot(normal, radial_inward) > 0.999


def test_normal_and_normaledge_use_local_curve_frame():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=180.0, silent=True)
    normal = Edge.Normal(arc, silent=True)
    assert normal is not None
    assert math.isclose(_norm(normal), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    nedge = Edge.NormalEdge(arc, length=2.5, u=0.5, silent=True)
    assert Topology.IsInstance(nedge, "Edge")
    assert math.isclose(Edge.Length(nedge, mantissa=6, silent=True), 2.5, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert _close(_xyz(Edge.StartVertex(nedge, silent=True)), _xyz(Edge.VertexByParameter(arc, 0.5, silent=True)))


@pytest.mark.pythonocc_only
def test_reverse_preserves_curved_geometry_and_orientation():
    arc = Edge.Arc(radius=3.0, fromAngle=20.0, toAngle=140.0, silent=True)
    rev = Edge.Reverse(arc, silent=True)
    assert Topology.IsInstance(rev, "Edge")
    assert Edge.IsLinear(rev, silent=True) is False
    assert math.isclose(Edge.Length(rev, mantissa=None, silent=True), Edge.Length(arc, mantissa=None, silent=True), rel_tol=1.0e-8, abs_tol=1.0e-8)
    assert _close(_xyz(Edge.StartVertex(rev, silent=True)), _xyz(Edge.EndVertex(arc, silent=True)))
    assert _close(_xyz(Edge.EndVertex(rev, silent=True)), _xyz(Edge.StartVertex(arc, silent=True)))
    assert _close(_xyz(Edge.VertexByParameter(rev, 0.5, silent=True)), _xyz(Edge.VertexByParameter(arc, 0.5, silent=True)), tol=2.0e-6)
    ta = Edge.TangentAtParameter(arc, 0.5, mantissa=None, silent=True)
    tr = Edge.TangentAtParameter(rev, 0.5, mantissa=None, silent=True)
    assert _dot(ta, tr) < -0.999999


def test_vertex_by_distance_uses_curvilinear_distance_on_arc():
    radius = 2.0
    arc = Edge.Arc(radius=radius, fromAngle=0.0, toAngle=180.0, silent=True)
    half_length = 0.5 * Edge.Length(arc, mantissa=None, silent=True)
    point = Edge.VertexByDistance(arc, distance=half_length, mantissa=None, silent=True)
    expected = Edge.VertexByParameter(arc, 0.5, silent=True)
    assert Topology.IsInstance(point, "Vertex")
    assert _close(_xyz(point), _xyz(expected), tol=2.0e-5)


def test_vertex_by_distance_wraps_on_closed_circle():
    circle = Edge.Circle(radius=1.5, silent=True)
    circumference = Edge.Length(circle, mantissa=None, silent=True)
    quarter = Edge.VertexByDistance(circle, distance=1.25*circumference, mantissa=None, silent=True)
    expected = Edge.VertexByParameter(circle, 0.25, silent=True)
    assert Topology.IsInstance(quarter, "Vertex")
    assert _close(_xyz(quarter), _xyz(expected), tol=5.0e-5)


@pytest.mark.pythonocc_only
def test_distance_trim_preserves_arc_and_exact_remaining_length():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=180.0, silent=True)
    original = Edge.Length(arc, mantissa=None, silent=True)
    trimmed = Edge.Trim(arc, distance=1.0, bothSides=True, silent=True)
    assert Topology.IsInstance(trimmed, "Edge")
    assert Edge.IsLinear(trimmed, silent=True) is False
    assert math.isclose(Edge.Length(trimmed, mantissa=None, silent=True), original - 1.0, rel_tol=2.0e-5, abs_tol=2.0e-5)
    assert _close(_xyz(Edge.VertexByParameter(trimmed, 0.5, silent=True)), _xyz(Edge.VertexByParameter(arc, 0.5, silent=True)), tol=3.0e-5)


def test_topologiccore_unsupported_curve_operations_do_not_flatten():
    if IS_PYTHONOCC:
        pytest.skip("TopologicCore capability guard.")
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=120.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    assert Edge.IsLinear(arc, silent=True) is False
    # TopologicCore exposes NURBS construction/evaluation but not an exact
    # curve-reversal or curve-trim API. Returning None is safer than silently
    # rebuilding the operation as a straight chord or sampled approximation.
    assert Edge.Reverse(arc, silent=True) is None
    assert Edge.Trim(arc, distance=0.5, bothSides=True, silent=True) is None


def test_linear_only_operations_refuse_to_flatten_curves():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=120.0, silent=True)
    assert Edge.SetLength(arc, length=5.0, silent=True) is None
    assert Edge.Extend(arc, distance=1.0, silent=True) is None
    assert Edge.Normalize(arc, silent=True) is None


def test_length_none_returns_unrounded_float_and_direction_handles_closed_edge():
    arc = Edge.Arc(radius=1.0, fromAngle=0.0, toAngle=90.0, silent=True)
    length = Edge.Length(arc, mantissa=None, silent=True)
    assert isinstance(length, float)
    assert math.isclose(length, math.pi/2.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    circle = Edge.Circle(radius=1.0, silent=True)
    assert Edge.Direction(circle, mantissa=None, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_special_curves_are_native_bspline_edges():
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve

    curves = [
        Edge.ByCurve([
            Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(1, 2, 0),
            Vertex.ByCoordinates(3, 2, 0), Vertex.ByCoordinates(4, 0, 0),
        ], silent=True),
        Edge.Parabola(silent=True),
        Edge.Hyperbola(silent=True),
        Edge.Helix(sides=12, silent=True),
    ]
    assert all(Topology.IsInstance(e, "Edge") for e in curves)
    for edge in curves:
        adaptor = BRepAdaptor_Curve(edge.shape)
        assert adaptor.GetType() == GeomAbs_BSplineCurve
