# Copyright (C) 2026
# PythonOCC backend Shell parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_shell.py -v

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


def _box_shell(size=1.0):
    """Create a shell from a box (open box without one face)."""
    c = Cell.Prism(size, size, size)
    return Cell.Shells(c)[0]


def _pyramid_shell():
    """Create a pyramid shell."""
    v1 = _v(0, 0, 0)
    v2 = _v(1, 0, 0)
    v3 = _v(1, 1, 0)
    v4 = _v(0, 1, 0)
    v5 = _v(0.5, 0.5, 1)  # Apex
    
    # Base face
    w1 = Wire.ByVertices([v1, v2, v3, v4], close=True)
    f1 = Face.ByWire(w1)
    
    # Side faces
    w2 = Wire.ByVertices([v1, v2, v5], close=True)
    f2 = Face.ByWire(w2)
    
    w3 = Wire.ByVertices([v2, v3, v5], close=True)
    f3 = Face.ByWire(w3)
    
    w4 = Wire.ByVertices([v3, v4, v5], close=True)
    f4 = Face.ByWire(w4)
    
    w5 = Wire.ByVertices([v4, v1, v5], close=True)
    f5 = Face.ByWire(w5)
    
    return Shell.ByFaces([f1, f2, f3, f4, f5])


# ===========================================================================
# Constructors
# ===========================================================================

class TestShellConstructors:
    def test_by_faces(self):
        f1 = Face.Rectangle(1.0, 1.0)
        f2 = Face.Rectangle(1.0, 1.0)
        f2 = Topology.Translate(f2, 0, 0, 1.0)
        s = Shell.ByFaces([f1, f2])
        assert Topology.IsInstance(s, "Shell")

    def test_by_faces_single(self):
        f = Face.Rectangle(1.0, 1.0)
        s = Shell.ByFaces([f])
        assert Topology.IsInstance(s, "Shell")

    def test_by_thickened_face(self):
        f = Face.Rectangle(1.0, 1.0)
        s = Shell.ByThickenedFace(f, 0.1)
        assert Topology.IsInstance(s, "Shell")

    def test_by_window(self):
        s = Shell.ByWindow()
        assert Topology.IsInstance(s, "Shell")

    def test_by_grid(self):
        s = Shell.ByGrid()
        assert Topology.IsInstance(s, "Shell")


# ===========================================================================
# Accessors
# ===========================================================================

class TestShellAccessors:
    def test_faces(self):
        s = _box_shell()
        faces = Shell.Faces(s)
        assert len(faces) >= 5  # Open box has 5 faces

    def test_edges(self):
        s = _box_shell()
        edges = Shell.Edges(s)
        assert len(edges) >= 8

    def test_vertices(self):
        s = _box_shell()
        verts = Shell.Vertices(s)
        assert len(verts) >= 4

    def test_external_boundary(self):
        s = _box_shell()
        eb = Shell.ExternalBoundary(s)
        assert eb is not None


# ===========================================================================
# Type checking
# ===========================================================================

class TestShellType:
    def test_is_instance_shell(self):
        s = _box_shell()
        assert Topology.IsInstance(s, "Shell") is True

    def test_is_not_face(self):
        s = _box_shell()
        assert Topology.IsInstance(s, "Face") is False

    def test_type_returns_shell(self):
        s = _box_shell()
        assert Topology.Type(s) == 16  # Shell type ID


# ===========================================================================
# Geometry
# ===========================================================================

class TestShellGeometry:
    def test_area(self):
        s = _box_shell(1.0)
        area = Shell.Area(s)
        # Box shell (6 faces) area = 6 * 1 = 6
        assert area == pytest.approx(6.0, abs=TOLERANCE)

    def test_external_boundary_length(self):
        s = _box_shell()
        eb = Shell.ExternalBoundary(s)
        length = Topology.Length(eb)
        assert length > 0


# ===========================================================================
# Operations
# ===========================================================================

class TestShellOperations:
    def test_close(self):
        # Create an open shell
        f = Face.Rectangle(1.0, 1.0)
        s = Shell.ByFaces([f])
        # Close should work on open shells
        s2 = Shell.Close(s)
        assert Topology.IsInstance(s2, "Shell")

    def test_self_merge(self):
        s = _box_shell()
        s2 = Shell.SelfMerge(s)
        assert Topology.IsInstance(s2, "Shell")

    def test_remove_collinear_edges(self):
        s = _box_shell()
        s2 = Shell.RemoveCollinearEdges(s)
        assert Topology.IsInstance(s2, "Shell")


# ===========================================================================
# Serialization
# ===========================================================================

class TestShellSerialization:
    def test_brep_roundtrip(self):
        s = _box_shell()
        brep = Topology.BREPString(s)
        assert brep is not None
        s2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(s2, "Shell")
        assert Shell.Area(s2) == pytest.approx(Shell.Area(s), abs=TOLERANCE)
