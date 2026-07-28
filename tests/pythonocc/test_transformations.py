# Copyright (C) 2026
# PythonOCC backend Transformations parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_transformations.py -v

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
Matrix = pytest.importorskip("topologicpy.Matrix").Matrix
Vector = pytest.importorskip("topologicpy.Vector").Vector

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _cube():
    return Cell.Prism(1.0, 1.0, 1.0)


# ===========================================================================
# Translation
# ===========================================================================

class TestTranslation:
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

    def test_translate_face(self):
        f = Face.Rectangle(1.0, 1.0)
        f2 = Topology.Translate(f, 0, 0, 2)
        verts = Face.Vertices(f2)
        for v in verts:
            assert Vertex.Z(v) == pytest.approx(2, abs=TOLERANCE)

    def test_translate_cell(self):
        c = _cube()
        c2 = Topology.Translate(c, 10, 20, 30)
        iv = Cell.InternalVertex(c2)
        assert Vertex.X(iv) == pytest.approx(10.5, abs=TOLERANCE)
        assert Vertex.Y(iv) == pytest.approx(20.5, abs=TOLERANCE)
        assert Vertex.Z(iv) == pytest.approx(30.5, abs=TOLERANCE)


# ===========================================================================
# Rotation
# ===========================================================================

class TestRotation:
    def test_rotate_vertex_around_z(self):
        v = _v(1, 0, 0)
        v2 = Topology.Rotate(v, 0, 0, 1, 90)  # 90 degrees around Z
        assert Vertex.X(v2) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.Y(v2) == pytest.approx(1, abs=TOLERANCE)

    def test_rotate_vertex_around_x(self):
        v = _v(0, 1, 0)
        v2 = Topology.Rotate(v, 1, 0, 0, 90)  # 90 degrees around X
        assert Vertex.Y(v2) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.Z(v2) == pytest.approx(1, abs=TOLERANCE)

    def test_rotate_360_returns_original(self):
        v = _v(1, 2, 3)
        v2 = Topology.Rotate(v, 0, 0, 1, 360)
        assert Vertex.X(v2) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Y(v2) == pytest.approx(2, abs=TOLERANCE)
        assert Vertex.Z(v2) == pytest.approx(3, abs=TOLERANCE)

    def test_rotate_edge(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        e2 = Topology.Rotate(e, 0, 0, 1, 90)
        sv = Edge.StartVertex(e2)
        ev = Edge.EndVertex(e2)
        assert Vertex.X(sv) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.Y(sv) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.X(ev) == pytest.approx(0, abs=TOLERANCE)
        assert Vertex.Y(ev) == pytest.approx(1, abs=TOLERANCE)


# ===========================================================================
# Scaling
# ===========================================================================

class TestScaling:
    def test_scale_uniform(self):
        c = _cube()
        c2 = Topology.Scale(c, 0, 0, 0, 2)  # Scale 2x from origin
        vol = Cell.Volume(c2)
        # Volume should be 8x (2^3)
        assert vol == pytest.approx(8.0, abs=TOLERANCE)

    def test_scale_from_center(self):
        c = _cube()
        c2 = Topology.Scale(c, 0.5, 0.5, 0.5, 2)  # Scale from center
        iv = Cell.InternalVertex(c2)
        # Center should remain at 0.5, 0.5, 0.5
        assert Vertex.X(iv) == pytest.approx(0.5, abs=TOLERANCE)
        assert Vertex.Y(iv) == pytest.approx(0.5, abs=TOLERANCE)
        assert Vertex.Z(iv) == pytest.approx(0.5, abs=TOLERANCE)

    def test_scale_non_uniform(self):
        c = _cube()
        c2 = Topology.Scale(c, 0, 0, 0, 2, 3, 4)  # Non-uniform scale
        vol = Cell.Volume(c2)
        # Volume should be 2*3*4 = 24
        assert vol == pytest.approx(24.0, abs=TOLERANCE)


# ===========================================================================
# Mirror
# ===========================================================================

class TestMirror:
    def test_mirror_vertex(self):
        v = _v(1, 0, 0)
        v2 = Topology.Mirror(v, 0, 0, 0, 0, 1, 0)  # Mirror across YZ plane
        assert Vertex.X(v2) == pytest.approx(-1, abs=TOLERANCE)

    def test_mirror_preserves_distance(self):
        v1 = _v(1, 0, 0)
        v2 = _v(2, 0, 0)
        v1m = Topology.Mirror(v1, 0, 0, 0, 0, 1, 0)
        v2m = Topology.Mirror(v2, 0, 0, 0, 0, 1, 0)
        d1 = Topology.Distance(v1, v2)
        d2 = Topology.Distance(v1m, v2m)
        assert d1 == pytest.approx(d2, abs=TOLERANCE)


# ===========================================================================
# Copy
# ===========================================================================

class TestCopy:
    def test_copy_independent(self):
        c1 = _cube()
        c2 = Topology.Copy(c1)
        # Modify c1
        c1 = Topology.Translate(c1, 100, 0, 0)
        # c2 should be unchanged
        iv = Cell.InternalVertex(c2)
        assert Vertex.X(iv) == pytest.approx(0.5, abs=TOLERANCE)

    def test_copy_preserves_geometry(self):
        c1 = _cube()
        c2 = Topology.Copy(c1)
        assert Cell.Volume(c1) == pytest.approx(Cell.Volume(c2), abs=TOLERANCE)


# ===========================================================================
# Matrix operations
# ===========================================================================

class TestMatrix:
    def test_identity_matrix(self):
        m = Matrix.ByTranslation(0, 0, 0)
        assert m is not None

    def test_translation_matrix(self):
        m = Matrix.ByTranslation(1, 2, 3)
        v = _v(0, 0, 0)
        v2 = Topology.TransformByMatrix(v, m)
        assert Vertex.X(v2) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Y(v2) == pytest.approx(2, abs=TOLERANCE)
        assert Vertex.Z(v2) == pytest.approx(3, abs=TOLERANCE)
