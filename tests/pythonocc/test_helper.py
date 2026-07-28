# Copyright (C) 2026
# PythonOCC backend Helper parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_helper.py -v

import math
import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Cell = pytest.importorskip("topologicpy.Cell").Cell
Topology = pytest.importorskip("topologicpy.Topology").Topology
Helper = pytest.importorskip("topologicpy.Helper").Helper

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


# ===========================================================================
# Vertex helpers
# ===========================================================================

class TestVertexHelpers:
    def test_vertex_coordinates(self):
        v = _v(1, 2, 3)
        coords = Helper.VertexCoordinates(v)
        assert len(coords) == 3
        assert coords[0] == pytest.approx(1, abs=TOLERANCE)
        assert coords[1] == pytest.approx(2, abs=TOLERANCE)
        assert coords[2] == pytest.approx(3, abs=TOLERANCE)

    def test_vertex_distance(self):
        v1 = _v(0, 0, 0)
        v2 = _v(3, 4, 0)
        dist = Helper.VertexDistance(v1, v2)
        assert dist == pytest.approx(5.0, abs=TOLERANCE)


# ===========================================================================
# Edge helpers
# ===========================================================================

class TestEdgeHelpers:
    def test_edge_start_vertex(self):
        e = Edge.ByStartVertexEndVertex(_v(1, 2), _v(3, 4))
        sv = Helper.EdgeStartVertex(e)
        assert Vertex.X(sv) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Y(sv) == pytest.approx(2, abs=TOLERANCE)

    def test_edge_end_vertex(self):
        e = Edge.ByStartVertexEndVertex(_v(1, 2), _v(3, 4))
        ev = Helper.EdgeEndVertex(e)
        assert Vertex.X(ev) == pytest.approx(3, abs=TOLERANCE)
        assert Vertex.Y(ev) == pytest.approx(4, abs=TOLERANCE)

    def test_edge_length(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(3, 4))
        length = Helper.EdgeLength(e)
        assert length == pytest.approx(5.0, abs=TOLERANCE)


# ===========================================================================
# Wire helpers
# ===========================================================================

class TestWireHelpers:
    def test_wire_vertices(self):
        w = Wire.Rectangle(1.0, 1.0)
        verts = Helper.WireVertices(w)
        assert len(verts) == 4

    def test_wire_edges(self):
        w = Wire.Rectangle(1.0, 1.0)
        edges = Helper.WireEdges(w)
        assert len(edges) == 4

    def test_wire_length(self):
        w = Wire.Rectangle(2.0, 3.0)
        length = Helper.WireLength(w)
        assert length == pytest.approx(10.0, abs=TOLERANCE)


# ===========================================================================
# Face helpers
# ===========================================================================

class TestFaceHelpers:
    def test_face_vertices(self):
        f = Face.Rectangle(1.0, 1.0)
        verts = Helper.FaceVertices(f)
        assert len(verts) == 4

    def test_face_edges(self):
        f = Face.Rectangle(1.0, 1.0)
        edges = Helper.FaceEdges(f)
        assert len(edges) == 4

    def test_face_area(self):
        f = Face.Rectangle(2.0, 3.0)
        area = Helper.FaceArea(f)
        assert area == pytest.approx(6.0, abs=TOLERANCE)

    def test_face_normal(self):
        f = Face.Rectangle(1.0, 1.0)
        normal = Helper.FaceNormal(f)
        assert normal is not None
        assert len(normal) == 3


# ===========================================================================
# Cell helpers
# ===========================================================================

class TestCellHelpers:
    def test_cell_vertices(self):
        c = Cell.Prism()
        verts = Helper.CellVertices(c)
        assert len(verts) == 8

    def test_cell_edges(self):
        c = Cell.Prism()
        edges = Helper.CellEdges(c)
        assert len(edges) == 12

    def test_cell_faces(self):
        c = Cell.Prism()
        faces = Helper.CellFaces(c)
        assert len(faces) == 6

    def test_cell_volume(self):
        c = Cell.Prism(2.0, 2.0, 2.0)
        volume = Helper.CellVolume(c)
        assert volume == pytest.approx(8.0, abs=TOLERANCE)


# ===========================================================================
# Topology helpers
# ===========================================================================

class TestTopologyHelpers:
    def test_topology_type(self):
        v = _v(0, 0, 0)
        t = Helper.TopologyType(v)
        assert t == 1  # Vertex type

    def test_topology_type_name(self):
        v = _v(0, 0, 0)
        name = Helper.TopologyTypeName(v)
        assert name == "Vertex"

    def test_topology_brep(self):
        v = _v(1, 2, 3)
        brep = Helper.TopologyBREPString(v)
        assert brep is not None
        assert len(brep) > 0
