# Copyright (C) 2026
# PythonOCC backend Topology parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_topology.py -v

import math
import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Cell = pytest.importorskip("topologicpy.Cell").Cell
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
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


# ===========================================================================
# Type checking
# ===========================================================================

class TestTopologyType:
    def test_type_vertex(self):
        v = _v(0, 0, 0)
        assert Topology.Type(v) == 1

    def test_type_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        assert Topology.Type(e) == 2

    def test_type_wire(self):
        w = Wire.ByVertices([_v(0, 0), _v(1, 0), _v(1, 1)], close=True)
        assert Topology.Type(w) == 4

    def test_type_face(self):
        f = Face.ByWire(Wire.ByVertices([_v(0, 0), _v(1, 0), _v(1, 1), _v(0, 1)], close=True))
        assert Topology.Type(f) == 8

    def test_type_cell(self):
        c = Cell.Prism()
        assert Topology.Type(c) == 32

    def test_is_instance_vertex(self):
        v = _v(0, 0, 0)
        assert Topology.IsInstance(v, "Vertex") is True
        assert Topology.IsInstance(v, "Edge") is False

    def test_is_instance_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        assert Topology.IsInstance(e, "Edge") is True
        assert Topology.IsInstance(e, "Vertex") is False


# ===========================================================================
# Sub-topology accessors
# ===========================================================================

class TestTopologySubTopologies:
    def test_vertices_of_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        verts = Topology.Vertices(e)
        assert len(verts) == 2

    def test_edges_of_wire(self):
        w = Wire.ByVertices([_v(0, 0), _v(1, 0), _v(1, 1)], close=True)
        edges = Topology.Edges(w)
        assert len(edges) == 3

    def test_vertices_of_face(self):
        f = Face.ByWire(Wire.ByVertices([_v(0, 0), _v(1, 0), _v(1, 1), _v(0, 1)], close=True))
        verts = Topology.Vertices(f)
        assert len(verts) == 4

    def test_edges_of_face(self):
        f = Face.ByWire(Wire.ByVertices([_v(0, 0), _v(1, 0), _v(1, 1), _v(0, 1)], close=True))
        edges = Topology.Edges(f)
        assert len(edges) == 4


# ===========================================================================
# Dictionary
# ===========================================================================

class TestTopologyDictionary:
    def test_set_get_dictionary(self):
        v = _v(1, 2, 3)
        d = Dictionary.ByKeysValues(["x", "y", "z"], [1, 2, 3])
        v2 = Topology.SetDictionary(v, d)
        d2 = Topology.Dictionary(v2)
        assert Dictionary.ValueAtKey(d2, "x") == 1
        assert Dictionary.ValueAtKey(d2, "y") == 2
        assert Dictionary.ValueAtKey(d2, "z") == 3

    def test_dictionary_on_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        d = Dictionary.ByKeysValues(["name"], ["edge1"])
        e2 = Topology.SetDictionary(e, d)
        d2 = Topology.Dictionary(e2)
        assert Dictionary.ValueAtKey(d2, "name") == "edge1"


# ===========================================================================
# Geometry
# ===========================================================================

class TestTopologyGeometry:
    def test_volume_cell(self):
        c = Cell.Prism(5.0, 5.0, 5.0)
        vol = Topology.Volume(c)
        assert vol == pytest.approx(125.0, abs=TOLERANCE)

    def test_area_face(self):
        f = Face.Rectangle(2.0, 3.0)
        area = Topology.Area(f)
        assert area == pytest.approx(6.0, abs=TOLERANCE)

    def test_length_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(3, 4))
        length = Topology.Length(e)
        assert length == pytest.approx(5.0, abs=TOLERANCE)


# ===========================================================================
# Transformation
# ===========================================================================

class TestTopologyTransformation:
    def test_translate_vertex(self):
        v = _v(0, 0, 0)
        v2 = Topology.Translate(v, 1, 2, 3)
        assert Vertex.X(v2) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Y(v2) == pytest.approx(2, abs=TOLERANCE)
        assert Vertex.Z(v2) == pytest.approx(3, abs=TOLERANCE)

    def test_translate_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        e2 = Topology.Translate(e, 5, 0, 0)
        sv = Edge.StartVertex(e2)
        assert Vertex.X(sv) == pytest.approx(5, abs=TOLERANCE)

    def test_copy(self):
        v = _v(1, 2, 3)
        v2 = Topology.Copy(v)
        assert Vertex.X(v2) == pytest.approx(1, abs=TOLERANCE)
        # Modify original, copy should be unchanged
        v3 = Topology.Translate(v, 10, 0, 0)
        assert Vertex.X(v2) == pytest.approx(1, abs=TOLERANCE)


# ===========================================================================
# Serialization
# ===========================================================================

class TestTopologySerialization:
    def test_brep_roundtrip_vertex(self):
        v = _v(1, 2, 3)
        brep = Topology.BREPString(v)
        v2 = Topology.ByBREPString(brep)
        assert Topology.Type(v2) == Topology.Type(v)
        assert Vertex.X(v2) == pytest.approx(1, abs=TOLERANCE)

    def test_brep_roundtrip_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 1))
        brep = Topology.BREPString(e)
        e2 = Topology.ByBREPString(brep)
        assert Topology.Type(e2) == Topology.Type(e)

    def test_brep_roundtrip_face(self):
        f = Face.Rectangle(1.0, 1.0)
        brep = Topology.BREPString(f)
        f2 = Topology.ByBREPString(brep)
        assert Topology.Type(f2) == Topology.Type(f)
        assert Face.Area(f2) == pytest.approx(Face.Area(f), abs=TOLERANCE)

    def test_json_roundtrip(self):
        v = _v(1, 2, 3)
        json_str = Topology.JSON(v)
        assert json_str is not None
        v2 = Topology.ByJSON(json_str)
        assert Topology.IsInstance(v2, "Vertex")
