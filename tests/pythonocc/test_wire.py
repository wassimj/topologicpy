# Copyright (C) 2026
# PythonOCC backend Wire parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_wire.py -v

import math
import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
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


def _square_wire(size=1.0):
    """Create a unit square wire."""
    v1 = _v(0, 0)
    v2 = _v(size, 0)
    v3 = _v(size, size)
    v4 = _v(0, size)
    return Wire.ByVertices([v1, v2, v3, v4], close=True)


# ===========================================================================
# Constructors
# ===========================================================================

class TestWireConstructors:
    def test_by_vertices_open(self):
        v1 = _v(0, 0)
        v2 = _v(1, 0)
        v3 = _v(1, 1)
        w = Wire.ByVertices([v1, v2, v3], close=False)
        assert Topology.IsInstance(w, "Wire")

    def test_by_vertices_closed(self):
        w = _square_wire()
        assert Topology.IsInstance(w, "Wire")

    def test_by_edges(self):
        v1 = _v(0, 0)
        v2 = _v(1, 0)
        v3 = _v(1, 1)
        e1 = Edge.ByStartVertexEndVertex(v1, v2)
        e2 = Edge.ByStartVertexEndVertex(v2, v3)
        w = Wire.ByEdges([e1, e2])
        assert Topology.IsInstance(w, "Wire")

    def test_rectangle(self):
        w = Wire.Rectangle(2.0, 3.0)
        assert Topology.IsInstance(w, "Wire")

    def test_circle(self):
        w = Wire.Circle(radius=1.0)
        assert Topology.IsInstance(w, "Wire")

    def test_square(self):
        w = Wire.Square(2.0)
        assert Topology.IsInstance(w, "Wire")


# ===========================================================================
# Accessors
# ===========================================================================

class TestWireAccessors:
    def test_vertices(self):
        w = _square_wire()
        verts = Wire.Vertices(w)
        assert len(verts) == 4

    def test_edges(self):
        w = _square_wire()
        edges = Wire.Edges(w)
        assert len(edges) == 4

    def test_length(self):
        w = _square_wire(1.0)
        assert Wire.Length(w) == pytest.approx(4.0, abs=TOLERANCE)

    def test_area_square(self):
        w = _square_wire(2.0)
        area = Wire.Area(w)
        assert area == pytest.approx(4.0, abs=TOLERANCE)


# ===========================================================================
# Type checking
# ===========================================================================

class TestWireType:
    def test_is_instance_wire(self):
        w = _square_wire()
        assert Topology.IsInstance(w, "Wire") is True

    def test_is_not_edge(self):
        w = _square_wire()
        assert Topology.IsInstance(w, "Edge") is False

    def test_type_returns_wire(self):
        w = _square_wire()
        assert Topology.Type(w) == 4  # Wire type ID


# ===========================================================================
# Operations
# ===========================================================================

class TestWireOperations:
    def test_close_open_wire(self):
        v1 = _v(0, 0)
        v2 = _v(1, 0)
        v3 = _v(1, 1)
        w = Wire.ByVertices([v1, v2, v3], close=False)
        w2 = Wire.Close(w)
        assert Topology.IsInstance(w2, "Wire")

    def test_reverse(self):
        w = _square_wire()
        w2 = Wire.Reverse(w)
        assert Topology.IsInstance(w2, "Wire")
        assert Wire.Length(w2) == pytest.approx(Wire.Length(w), abs=TOLERANCE)

    def test_is_closed(self):
        w = _square_wire()
        assert Wire.IsClosed(w) is True

    def test_is_not_closed(self):
        v1 = _v(0, 0)
        v2 = _v(1, 0)
        w = Wire.ByVertices([v1, v2], close=False)
        assert Wire.IsClosed(w) is False


# ===========================================================================
# Serialization
# ===========================================================================

class TestWireSerialization:
    def test_brep_roundtrip(self):
        w = _square_wire()
        brep = Topology.BREPString(w)
        assert brep is not None
        w2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(w2, "Wire")
        assert Wire.Length(w2) == pytest.approx(Wire.Length(w), abs=TOLERANCE)
