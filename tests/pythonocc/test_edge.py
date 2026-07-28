# Copyright (C) 2026
# PythonOCC backend Edge parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_edge.py -v

import math
import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _e(x1, y1, x2, y2, z1=0, z2=0):
    return Edge.ByStartVertexEndVertex(_v(x1, y1, z1), _v(x2, y2, z2))


# ===========================================================================
# Constructors
# ===========================================================================

class TestEdgeConstructors:
    def test_by_start_vertex_end_vertex(self):
        e = _e(0, 0, 1, 0)
        assert Topology.IsInstance(e, "Edge")

    def test_preserves_endpoints(self):
        v1 = _v(0, 0, 0)
        v2 = _v(1, 1, 1)
        e = Edge.ByStartVertexEndVertex(v1, v2)
        assert Topology.IsInstance(e, "Edge")
        sv = Edge.StartVertex(e)
        ev = Edge.EndVertex(e)
        assert Vertex.X(sv) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.Y(sv) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.Z(sv) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.X(ev) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Y(ev) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Z(ev) == pytest.approx(1, abs=TOLERANCE)

    def test_by_vertices_list(self):
        v1 = _v(0, 0, 0)
        v2 = _v(1, 0, 0)
        e = Edge.ByVertices([v1, v2])
        assert Topology.IsInstance(e, "Edge")

    def test_degenerate_edge_returns_none(self):
        v1 = _v(0, 0, 0)
        e = Edge.ByStartVertexEndVertex(v1, v1)
        # Degenerate edges may return None or a zero-length edge
        assert e is None or Edge.Length(e) == pytest.approx(0, abs=TOLERANCE)


# ===========================================================================
# Accessors
# ===========================================================================

class TestEdgeAccessors:
    def test_start_vertex(self):
        e = _e(1, 2, 3, 4)
        sv = Edge.StartVertex(e)
        assert Vertex.X(sv) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Y(sv) == pytest.approx(2, abs=TOLERANCE)

    def test_end_vertex(self):
        e = _e(1, 2, 3, 4)
        ev = Edge.EndVertex(e)
        assert Vertex.X(ev) == pytest.approx(3, abs=TOLERANCE)
        assert Vertex.Y(ev) == pytest.approx(4, abs=TOLERANCE)

    def test_vertices(self):
        e = _e(0, 0, 1, 0)
        verts = Edge.Vertices(e)
        assert len(verts) == 2

    def test_length_horizontal(self):
        e = _e(0, 0, 5, 0)
        assert Edge.Length(e) == pytest.approx(5.0, abs=TOLERANCE)

    def test_length_vertical(self):
        e = _e(0, 0, 0, 5)
        assert Edge.Length(e) == pytest.approx(5.0, abs=TOLERANCE)

    def test_length_diagonal(self):
        e = _e(0, 0, 3, 4)
        assert Edge.Length(e) == pytest.approx(5.0, abs=TOLERANCE)

    def test_length_3d(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0, 0), _v(1, 1, 1))
        assert Edge.Length(e) == pytest.approx(math.sqrt(3), abs=TOLERANCE)


# ===========================================================================
# Type checking
# ===========================================================================

class TestEdgeType:
    def test_is_instance_edge(self):
        e = _e(0, 0, 1, 0)
        assert Topology.IsInstance(e, "Edge") is True

    def test_is_not_vertex(self):
        e = _e(0, 0, 1, 0)
        assert Topology.IsInstance(e, "Vertex") is False

    def test_type_returns_edge(self):
        e = _e(0, 0, 1, 0)
        assert Topology.Type(e) == 2  # Edge type ID


# ===========================================================================
# Operations
# ===========================================================================

class TestEdgeOperations:
    def test_reverse(self):
        e = _e(0, 0, 1, 0)
        e2 = Edge.Reverse(e)
        sv = Edge.StartVertex(e2)
        ev = Edge.EndVertex(e2)
        assert Vertex.X(sv) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.X(ev) == pytest.approx(0, abs=TOLERANCE)

    def test_parameter_at_vertex(self):
        e = _e(0, 0, 1, 0)
        v = _v(0.5, 0, 0)
        param = Edge.ParameterAtVertex(e, v)
        assert param == pytest.approx(0.5, abs=TOLERANCE)

    def test_vertex_at_parameter(self):
        e = _e(0, 0, 1, 0)
        v = Edge.VertexAtParameter(e, 0.5)
        assert Vertex.X(v) == pytest.approx(0.5, abs=TOLERANCE)
        assert Vertex.Y(v) == pytest.approx(0, abs=TOLERANCE)

    def test_direction(self):
        e = _e(0, 0, 1, 0)
        d = Edge.DirectionVector(e)
        assert d is not None

    def test_normal(self):
        e = _e(0, 0, 1, 0)
        n = Edge.NormalVector(e)
        assert n is not None


# ===========================================================================
# Serialization
# ===========================================================================

class TestEdgeSerialization:
    def test_brep_roundtrip(self):
        e = _e(0, 0, 1, 0)
        brep = Topology.BREPString(e)
        assert brep is not None
        e2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(e2, "Edge")
        assert Edge.Length(e2) == pytest.approx(1.0, abs=TOLERANCE)
