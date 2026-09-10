"""Tranche 06: Wire-level curve preservation, ordering and arc-length queries."""

import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _xyz(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _close(a, b, tol=2.0e-5):
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


def _mixed_wire():
    v0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    v1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)
    v2 = Vertex.ByCoordinates(2.0, 1.0, 0.0)
    v3 = Vertex.ByCoordinates(3.0, 1.0, 0.0)
    e0 = Edge.ByStartVertexEndVertex(v0, v1, silent=True)
    arc = Edge.Arc(
        origin=Vertex.ByCoordinates(1.0, 1.0, 0.0),
        radius=1.0,
        fromAngle=270.0,
        toAngle=360.0,
        placement="center",
        silent=True,
    )
    e2 = Edge.ByStartVertexEndVertex(v2, v3, silent=True)
    assert Topology.IsInstance(e0, "Edge")
    assert Topology.IsInstance(arc, "Edge")
    assert Topology.IsInstance(e2, "Edge")
    wire = Wire.ByEdges([e0, arc, e2], orient=True, silent=True)
    return wire, (v0, v1, v2, v3), (e0, arc, e2)


def test_mixed_wire_preserves_curve_length_and_direction():
    wire, vertices, _ = _mixed_wire()
    assert Topology.IsInstance(wire, "Wire")
    assert Wire.IsManifold(wire, silent=True) is True
    assert Wire.IsClosed(wire, silent=True) is False
    assert Wire.IsPolyline(wire, silent=True) is False
    assert _close(_xyz(Wire.StartVertex(wire, silent=True)), _xyz(vertices[0]))
    assert _close(_xyz(Wire.EndVertex(wire, silent=True)), _xyz(vertices[-1]))
    expected = 2.0 + math.pi / 2.0
    actual = Wire.Length(wire, mantissa=None, silent=True)
    assert isinstance(actual, float)
    assert math.isclose(actual, expected, rel_tol=2.0e-6, abs_tol=2.0e-6)
    edges = Wire.Edges(wire, silent=True)
    assert isinstance(edges, list) and len(edges) == 3
    assert any(Edge.IsLinear(e, silent=True) is False for e in edges)


def test_global_parameterization_uses_true_curvilinear_length():
    wire, _, _ = _mixed_wire()
    total = Wire.Length(wire, mantissa=None, silent=True)
    target_distance = 1.0 + math.pi / 4.0
    u = target_distance / total
    point = Wire.VertexByParameter(wire, u=u, silent=True)
    assert Topology.IsInstance(point, "Vertex")
    s2 = math.sqrt(2.0) / 2.0
    expected = [1.0 + s2, 1.0 - s2, 0.0]
    assert _close(_xyz(point), expected, tol=8.0e-5)
    recovered = Wire.ParameterAtVertex(wire, point, mantissa=None, silent=True)
    assert recovered is not None
    assert math.isclose(recovered, u, rel_tol=5.0e-5, abs_tol=5.0e-5)


def test_vertex_by_distance_and_vertex_distance_cross_curved_edge():
    wire, vertices, _ = _mixed_wire()
    target_distance = 1.0 + math.pi / 4.0
    point = Wire.VertexByDistance(wire, distance=target_distance, silent=True)
    assert Topology.IsInstance(point, "Vertex")
    measured = Wire.VertexDistance(wire, point, origin=vertices[0], mantissa=None, silent=True)
    assert measured is not None
    assert math.isclose(measured, target_distance, rel_tol=5.0e-5, abs_tol=5.0e-5)


def test_linear_wire_is_polyline_and_orientedges_respects_requested_start():
    v0 = Vertex.ByCoordinates(0, 0, 0)
    v1 = Vertex.ByCoordinates(1, 0, 0)
    v2 = Vertex.ByCoordinates(2, 0, 0)
    # Intentionally mixed orientations and order.
    e1 = Edge.ByStartVertexEndVertex(v2, v1, silent=True)
    e0 = Edge.ByStartVertexEndVertex(v0, v1, silent=True)
    wire = Wire.ByEdges([e1, e0], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    assert Wire.IsPolyline(wire, silent=True) is True
    oriented = Wire.OrientEdges(wire, v0, silent=True)
    assert Topology.IsInstance(oriented, "Wire")
    assert _close(_xyz(Wire.StartVertex(oriented, silent=True)), _xyz(v0))
    assert _close(_xyz(Wire.EndVertex(oriented, silent=True)), _xyz(v2))


def test_reverse_linear_wire_on_both_backends():
    v0 = Vertex.ByCoordinates(0, 0, 0)
    v1 = Vertex.ByCoordinates(1, 0, 0)
    v2 = Vertex.ByCoordinates(2, 0, 0)
    wire = Wire.ByEdges([
        Edge.ByStartVertexEndVertex(v0, v1, silent=True),
        Edge.ByStartVertexEndVertex(v1, v2, silent=True),
    ], orient=True, silent=True)
    rev = Wire.Reverse(wire, silent=True)
    assert Topology.IsInstance(rev, "Wire")
    assert _close(_xyz(Wire.StartVertex(rev, silent=True)), _xyz(v2))
    assert _close(_xyz(Wire.EndVertex(rev, silent=True)), _xyz(v0))
    assert math.isclose(Wire.Length(rev, mantissa=None, silent=True), 2.0, abs_tol=1.0e-9)


@pytest.mark.pythonocc_only
def test_reverse_mixed_curved_wire_preserves_geometry_pythonocc():
    wire, vertices, _ = _mixed_wire()
    rev = Wire.Reverse(wire, silent=True)
    assert Topology.IsInstance(rev, "Wire")
    assert _close(_xyz(Wire.StartVertex(rev, silent=True)), _xyz(vertices[-1]))
    assert _close(_xyz(Wire.EndVertex(rev, silent=True)), _xyz(vertices[0]))
    assert math.isclose(
        Wire.Length(rev, mantissa=None, silent=True),
        Wire.Length(wire, mantissa=None, silent=True),
        rel_tol=1.0e-8,
        abs_tol=1.0e-8,
    )
    assert Wire.IsPolyline(rev, silent=True) is False


def test_topologiccore_curved_reverse_refuses_to_flatten():
    if IS_PYTHONOCC:
        pytest.skip("TopologicCore capability guard.")
    wire, _, _ = _mixed_wire()
    assert Wire.Reverse(wire, silent=True) is None


def test_closed_circle_wire_classification_and_length():
    circle = Edge.Circle(radius=2.0, silent=True)
    wire = Wire.ByEdges([circle], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    assert Wire.IsClosed(wire, silent=True) is True
    assert Wire.IsManifold(wire, silent=True) is True
    assert Wire.IsPolyline(wire, silent=True) is False
    assert Wire.StartVertex(wire, silent=True) is None
    assert Wire.EndVertex(wire, silent=True) is None
    assert math.isclose(Wire.Length(wire, mantissa=None, silent=True), 4.0*math.pi, rel_tol=2.0e-6, abs_tol=2.0e-6)
