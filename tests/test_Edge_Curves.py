# Copyright (C) 2026
# TopologicPy native curve tests.

import math

import pytest

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Topology = pytest.importorskip("topologicpy.Topology").Topology


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _coords(vertex):
    return Vertex.Coordinates(vertex, mantissa=10)


def test_exact_rational_quarter_circle_nurbs():
    control_points = [_v(1, 0), _v(1, 1), _v(0, 1)]
    edge = Edge.ByNurbsParameters(
        controlPoints=control_points,
        weights=[1.0, math.sqrt(2.0) / 2.0, 1.0],
        knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        isRational=True,
        isPeriodic=False,
        degree=2,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert len(Topology.Edges(edge)) == 1
    assert Edge.Length(edge, mantissa=8) == pytest.approx(math.pi / 2.0, abs=1e-6)

    mid = Edge.VertexByParameter(edge, 0.5, silent=True)
    expected = math.sqrt(0.5)
    xyz = _coords(mid)
    assert xyz[0] == pytest.approx(expected, abs=1e-6)
    assert xyz[1] == pytest.approx(expected, abs=1e-6)
    assert xyz[2] == pytest.approx(0.0, abs=1e-6)
    assert Edge.ParameterAtVertex(edge, mid, mantissa=8, silent=True) == pytest.approx(0.5, abs=1e-6)


def test_curve_constructor_creates_one_genuinely_curved_edge():
    edge = Edge.ByCurve(
        [_v(0, 0), _v(2, 4), _v(5, -2), _v(8, 3), _v(10, 0)],
        degree=3,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert len(Topology.Edges(edge)) == 1
    chord = math.dist(_coords(Edge.StartVertex(edge, silent=True)), _coords(Edge.EndVertex(edge, silent=True)))
    assert Edge.Length(edge, mantissa=8) > chord


def test_local_tangent_normal_and_parameter_round_trip_on_curve():
    edge = Edge.ByNurbsParameters(
        controlPoints=[_v(1, 0), _v(1, 1), _v(0, 1)],
        weights=[1.0, math.sqrt(2.0) / 2.0, 1.0],
        knots=[0, 0, 0, 1, 1, 1],
        isRational=True,
        degree=2,
        silent=True,
    )

    tangent = Edge.TangentAtParameter(edge, 0.5, mantissa=8, silent=True)
    normal = Edge.NormalAtParameter(edge, 0.5, mantissa=8, silent=True)
    assert tangent is not None
    assert normal is not None
    assert math.sqrt(sum(value * value for value in tangent)) == pytest.approx(1.0, abs=1e-6)
    assert math.sqrt(sum(value * value for value in normal)) == pytest.approx(1.0, abs=1e-6)
    assert sum(tangent[i] * normal[i] for i in range(3)) == pytest.approx(0.0, abs=1e-6)

    vertex = Edge.VertexByParameter(edge, 0.37, silent=True)
    assert Edge.ParameterAtVertex(edge, vertex, mantissa=6, silent=True) == pytest.approx(0.37, abs=1e-4)


def test_trim_by_parameters_uses_actual_curve_points():
    edge = Edge.ByCurve([_v(0, 0), _v(2, 4), _v(5, -2), _v(8, 3), _v(10, 0)], degree=3, silent=True)
    expected_start = Edge.VertexByParameter(edge, 0.2, silent=True)
    expected_end = Edge.VertexByParameter(edge, 0.8, silent=True)
    trimmed = Edge.TrimByParameters(edge, 0.2, 0.8, silent=True)

    assert Topology.IsInstance(trimmed, "Edge")
    assert _coords(Edge.StartVertex(trimmed, silent=True)) == pytest.approx(_coords(expected_start), abs=1e-6)
    assert _coords(Edge.EndVertex(trimmed, silent=True)) == pytest.approx(_coords(expected_end), abs=1e-6)


def test_adjacent_edges_returns_edges_sharing_a_topological_endpoint():
    shared = _v(1, 0)
    edge_a = Edge.ByVertices(_v(0, 0), shared, silent=True)
    edge_b = Edge.ByVertices(shared, _v(1, 1), silent=True)
    edge_c = Edge.ByVertices(_v(3, 0), _v(4, 0), silent=True)
    host = Cluster.ByTopologies([edge_a, edge_b, edge_c])

    adjacent = Edge.AdjacentEdges(edge_a, host, silent=True)
    assert isinstance(adjacent, list)
    assert len(adjacent) == 1
    assert Topology.IsSame(adjacent[0], edge_b) or Edge.Index(edge_b, adjacent, strict=True) == 0
