import pytest

from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Wire import Wire


TOL = 1.0e-4


def _edges(wire):
    return Topology.Edges(wire, silent=True) or []


def _all_linear(wire):
    edges = _edges(wire)
    return bool(edges) and all(
        Edge.IsLinear(edge, tolerance=TOL, silent=True)
        for edge in edges
    )


def _has_curved_edge(wire):
    return any(
        not Edge.IsLinear(edge, tolerance=TOL, silent=True)
        for edge in _edges(wire)
    )


def test_circle_explicit_curved_and_polyline_modes():
    curved = Wire.Circle(
        radius=1.25,
        sides=1,
        polyline=False,
        tolerance=TOL,
        silent=True,
    )
    polyline = Wire.Circle(
        radius=1.25,
        sides=16,
        polyline=True,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(curved, "Wire")
    assert Topology.IsInstance(polyline, "Wire")

    curved_edges = _edges(curved)
    polyline_edges = _edges(polyline)

    assert len(curved_edges) == 1
    assert Edge.IsClosed(curved_edges[0], tolerance=TOL, silent=True) is True
    assert Edge.IsLinear(curved_edges[0], tolerance=TOL, silent=True) is False

    assert len(polyline_edges) == 16
    assert _all_linear(polyline)


def test_ellipse_all_retains_input_mode_and_exact_curves():
    result = Wire.EllipseAll(
        inputMode=1,
        width=4.0,
        length=2.0,
        sides=4,
        polyline=False,
        tolerance=TOL,
    )

    assert isinstance(result, dict)
    assert "ellipse" in result

    ellipse = result["ellipse"]

    assert Topology.IsInstance(ellipse, "Wire")
    assert len(_edges(ellipse)) == 4
    assert _has_curved_edge(ellipse)


def test_ellipse_polyline_mode_remains_available():
    ellipse = Wire.Ellipse(
        inputMode=1,
        width=4.0,
        length=2.0,
        sides=20,
        polyline=True,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(ellipse, "Wire")
    assert _all_linear(ellipse)


def test_golden_spiral_curved_mode_uses_true_arc_edges():
    spiral = Wire.GoldenSpiral(
        width=4.0,
        maxIterations=6,
        sides=48,
        polyline=False,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(spiral, "Wire")
    assert len(_edges(spiral)) == 6
    assert _has_curved_edge(spiral)


def test_golden_spiral_polyline_mode_remains_linear():
    spiral = Wire.GoldenSpiral(
        width=4.0,
        maxIterations=6,
        sides=48,
        polyline=True,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(spiral, "Wire")
    assert _all_linear(spiral)


def test_spatial_spiral_curved_mode_contains_nurbs_edges():
    spiral = Wire.Spiral(
        radiusA=0.5,
        radiusB=1.5,
        height=3.0,
        turns=2,
        sides=8,
        polyline=False,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(spiral, "Wire")
    assert len(_edges(spiral)) >= 2
    assert _has_curved_edge(spiral)


def test_spatial_spiral_polyline_mode_contains_only_linear_edges():
    spiral = Wire.Spiral(
        radiusA=0.5,
        radiusB=1.5,
        height=3.0,
        turns=2,
        sides=8,
        polyline=True,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(spiral, "Wire")
    assert _all_linear(spiral)


def test_squircle_curved_mode_contains_nurbs_edges():
    squircle = Wire.Squircle(
        radius=2.0,
        sides=24,
        a=2.0,
        b=2.0,
        polyline=False,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(squircle, "Wire")
    assert len(_edges(squircle)) >= 4
    assert _has_curved_edge(squircle)


def test_squircle_polyline_mode_remains_linear():
    squircle = Wire.Squircle(
        radius=2.0,
        sides=24,
        a=2.0,
        b=2.0,
        polyline=True,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(squircle, "Wire")
    assert _all_linear(squircle)
