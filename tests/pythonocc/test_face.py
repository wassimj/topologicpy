# Copyright (C) 2026
# PythonOCC backend Face parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_face.py -v

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


def _square_face(size=1.0):
    """Create a unit square face."""
    v1 = _v(0, 0)
    v2 = _v(size, 0)
    v3 = _v(size, size)
    v4 = _v(0, size)
    w = Wire.ByVertices([v1, v2, v3, v4], close=True)
    return Face.ByWire(w)


# ===========================================================================
# Constructors
# ===========================================================================

class TestFaceConstructors:
    def test_by_wire(self):
        f = _square_face()
        assert Topology.IsInstance(f, "Face")

    def test_by_vertices(self):
        v1 = _v(0, 0)
        v2 = _v(1, 0)
        v3 = _v(1, 1)
        v4 = _v(0, 1)
        f = Face.ByVertices([v1, v2, v3, v4])
        assert Topology.IsInstance(f, "Face")

    def test_rectangle(self):
        f = Face.Rectangle(2.0, 3.0)
        assert Topology.IsInstance(f, "Face")

    def test_square(self):
        f = Face.Square(2.0)
        assert Topology.IsInstance(f, "Face")

    def test_circle(self):
        f = Face.Circle(radius=1.0)
        assert Topology.IsInstance(f, "Face")


# ===========================================================================
# Accessors
# ===========================================================================

class TestFaceAccessors:
    def test_area_square(self):
        f = _square_face(2.0)
        assert Face.Area(f) == pytest.approx(4.0, abs=TOLERANCE)

    def test_area_rectangle(self):
        f = Face.Rectangle(2.0, 3.0)
        assert Face.Area(f) == pytest.approx(6.0, abs=TOLERANCE)

    def test_perimeter(self):
        f = _square_face(1.0)
        assert Face.Perimeter(f) == pytest.approx(4.0, abs=TOLERANCE)

    def test_vertices(self):
        f = _square_face()
        verts = Face.Vertices(f)
        assert len(verts) == 4

    def test_edges(self):
        f = _square_face()
        edges = Face.Edges(f)
        assert len(edges) == 4

    def test_external_boundary(self):
        f = _square_face()
        eb = Face.ExternalBoundary(f)
        assert Topology.IsInstance(eb, "Wire")


# ===========================================================================
# Type checking
# ===========================================================================

class TestFaceType:
    def test_is_instance_face(self):
        f = _square_face()
        assert Topology.IsInstance(f, "Face") is True

    def test_is_not_wire(self):
        f = _square_face()
        assert Topology.IsInstance(f, "Wire") is False

    def test_type_returns_face(self):
        f = _square_face()
        assert Topology.Type(f) == 8  # Face type ID


# ===========================================================================
# Geometry
# ===========================================================================

class TestFaceGeometry:
    def test_normal_vector(self):
        f = _square_face()
        n = Face.NormalVector(f)
        assert n is not None
        # For a face in XY plane, normal should be along Z
        assert abs(n[2]) > 0.9  # Should point mostly in Z direction

    def test_compactness(self):
        f = _square_face(1.0)
        c = Face.Compactness(f)
        # Circle has compactness 1.0, square is less
        assert 0 < c <= 1.0

    def test_internal_vertex(self):
        f = _square_face(2.0)
        iv = Face.InternalVertex(f)
        assert Topology.IsInstance(iv, "Vertex")
        # Internal vertex should be inside the face
        assert 0 < Vertex.X(iv) < 2.0
        assert 0 < Vertex.Y(iv) < 2.0


# ===========================================================================
# Operations
# ===========================================================================

class TestFaceOperations:
    def test_reverse(self):
        f = _square_face()
        f2 = Face.Reverse(f)
        assert Topology.IsInstance(f2, "Face")
        assert Face.Area(f2) == pytest.approx(Face.Area(f), abs=TOLERANCE)

    def test_by_offset(self):
        f = _square_face(2.0)
        f2 = Face.ByOffset(f, 0.5)
        assert Topology.IsInstance(f2, "Face")
        # Offset face should be smaller
        assert Face.Area(f2) < Face.Area(f)


# ===========================================================================
# Serialization
# ===========================================================================

class TestFaceSerialization:
    def test_brep_roundtrip(self):
        f = _square_face()
        brep = Topology.BREPString(f)
        assert brep is not None
        f2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(f2, "Face")
        assert Face.Area(f2) == pytest.approx(Face.Area(f), abs=TOLERANCE)
