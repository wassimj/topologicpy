# Copyright (C) 2026
# PythonOCC backend Boolean/CSG operation parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_boolean.py -v

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
CSG = pytest.importorskip("topologicpy.CSG").CSG

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _cube(size=1.0, x=0, y=0, z=0):
    """Create a cube at specified position."""
    c = Cell.Prism(size, size, size)
    return Topology.Translate(c, x, y, z)


def _sphere(radius=0.5, x=0, y=0, z=0):
    """Create a sphere at specified position."""
    s = Cell.Sphere(radius)
    return Topology.Translate(s, x, y, z)


# ===========================================================================
# Union operations
# ===========================================================================

class TestCSGUnion:
    def test_union_two_cubes(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 0.5, 0, 0)  # Overlapping
        result = CSG.Union(c1, c2)
        assert result is not None
        assert Topology.IsInstance(result, "Cell") or Topology.IsInstance(result, "CellComplex")

    def test_union_disjoint(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 5, 0, 0)  # Far apart
        result = CSG.Union(c1, c2)
        assert result is not None

    def test_union_preserves_volume(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 0, 0, 0)  # Same position
        result = CSG.Union(c1, c2)
        vol = Topology.Volume(result)
        assert vol == pytest.approx(1.0, abs=TOLERANCE)


# ===========================================================================
# Difference operations
# ===========================================================================

class TestCSGDifference:
    def test_difference_cube_sphere(self):
        c = _cube(2.0, 0, 0, 0)
        s = _sphere(0.5, 0, 0, 0)
        result = CSG.Difference(c, s)
        assert result is not None
        vol = Topology.Volume(result)
        # Volume should be less than original cube
        assert vol < Topology.Volume(c)

    def test_difference_no_overlap(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 5, 0, 0)
        result = CSG.Difference(c1, c2)
        assert result is not None
        vol = Topology.Volume(result)
        assert vol == pytest.approx(1.0, abs=TOLERANCE)

    def test_difference_complete(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(2.0, 0, 0, 0)  # Larger, contains c1
        result = CSG.Difference(c1, c2)
        # Result should be empty or None
        assert result is None or Topology.Volume(result) == 0


# ===========================================================================
# Intersection operations
# ===========================================================================

class TestCSGIntersection:
    def test_intersection_two_cubes(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 0.5, 0, 0)  # Overlapping
        result = CSG.Intersection(c1, c2)
        assert result is not None
        vol = Topology.Volume(result)
        # Intersection should be smaller than either cube
        assert vol < Topology.Volume(c1)
        assert vol < Topology.Volume(c2)

    def test_intersection_no_overlap(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 5, 0, 0)
        result = CSG.Intersection(c1, c2)
        # No intersection
        assert result is None or Topology.Volume(result) == 0

    def test_intersection_same(self):
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 0, 0, 0)
        result = CSG.Intersection(c1, c2)
        assert result is not None
        vol = Topology.Volume(result)
        assert vol == pytest.approx(1.0, abs=TOLERANCE)


# ===========================================================================
# Complex operations
# ===========================================================================

class TestCSGComplex:
    def test_pipe(self):
        """Test pipe operation (sweep along edge)."""
        f = Face.Rectangle(0.1, 0.1)
        e = Edge.ByStartVertexEndVertex(_v(0, 0, 0), _v(2, 0, 0))
        result = CSG.Pipe(f, e)
        assert result is not None

    def test_slice(self):
        """Test slice operation."""
        c = _cube(2.0, 0, 0, 0)
        plane_face = Face.Rectangle(5.0, 5.0)
        plane_face = Topology.Rotate(plane_face, 0, 0, 1, 45)
        result = CSG.Slice(c, plane_face)
        assert result is not None

    def test_imprint(self):
        """Test imprint operation."""
        c1 = _cube(1.0, 0, 0, 0)
        c2 = _cube(1.0, 0.5, 0, 0)
        result = CSG.Imprint(c1, c2)
        assert result is not None


# ===========================================================================
# Face operations
# ===========================================================================

class TestCSGFaceOperations:
    def test_face_by_offset(self):
        f = Face.Rectangle(2.0, 2.0)
        f2 = CSG.FaceByOffset(f, 0.5)
        assert f2 is not None
        # Offset face should be smaller
        assert Face.Area(f2) < Face.Area(f)

    def test_face_by_thickened(self):
        f = Face.Rectangle(1.0, 1.0)
        cell = CSG.FaceByThickened(f, 0.1)
        assert cell is not None
        assert Topology.IsInstance(cell, "Cell")
