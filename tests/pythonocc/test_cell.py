# Copyright (C) 2026
# PythonOCC backend Cell parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_cell.py -v

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


# ===========================================================================
# Constructors
# ===========================================================================

class TestCellConstructors:
    def test_prism_default(self):
        c = Cell.Prism()
        assert Topology.IsInstance(c, "Cell")

    def test_prism_custom_dimensions(self):
        c = Cell.Prism(2.0, 3.0, 4.0)
        assert Topology.IsInstance(c, "Cell")

    def test_cube(self):
        c = Cell.Cube(2.0)
        assert Topology.IsInstance(c, "Cell")

    def test_sphere(self):
        c = Cell.Sphere(radius=1.0)
        assert Topology.IsInstance(c, "Cell")

    def test_cylinder(self):
        c = Cell.Cylinder(radius=1.0, height=2.0)
        assert Topology.IsInstance(c, "Cell")

    def test_cone(self):
        c = Cell.Cone(radius=1.0, height=2.0)
        assert Topology.IsInstance(c, "Cell")


# ===========================================================================
# Accessors
# ===========================================================================

class TestCellAccessors:
    def test_volume_prism(self):
        c = Cell.Prism(2.0, 3.0, 4.0)
        vol = Cell.Volume(c)
        assert vol == pytest.approx(24.0, abs=TOLERANCE)

    def test_volume_cube(self):
        c = Cell.Cube(2.0)
        vol = Cell.Volume(c)
        assert vol == pytest.approx(8.0, abs=TOLERANCE)

    def test_faces(self):
        c = Cell.Prism()
        faces = Cell.Faces(c)
        assert len(faces) == 6  # Cube has 6 faces

    def test_edges(self):
        c = Cell.Prism()
        edges = Cell.Edges(c)
        assert len(edges) == 12  # Cube has 12 edges

    def test_vertices(self):
        c = Cell.Prism()
        verts = Cell.Vertices(c)
        assert len(verts) == 8  # Cube has 8 vertices

    def test_shells(self):
        c = Cell.Prism()
        shells = Cell.Shells(c)
        assert len(shells) == 1  # Single shell for simple cell


# ===========================================================================
# Type checking
# ===========================================================================

class TestCellType:
    def test_is_instance_cell(self):
        c = Cell.Prism()
        assert Topology.IsInstance(c, "Cell") is True

    def test_is_not_face(self):
        c = Cell.Prism()
        assert Topology.IsInstance(c, "Face") is False

    def test_type_returns_cell(self):
        c = Cell.Prism()
        assert Topology.Type(c) == 32  # Cell type ID


# ===========================================================================
# Geometry
# ===========================================================================

class TestCellGeometry:
    def test_internal_vertex(self):
        c = Cell.Prism(2.0, 2.0, 2.0)
        iv = Cell.InternalVertex(c)
        assert Topology.IsInstance(iv, "Vertex")
        # Internal vertex should be inside the cell
        x, y, z = Vertex.X(iv), Vertex.Y(iv), Vertex.Z(iv)
        assert 0 < x < 2.0
        assert 0 < y < 2.0
        assert 0 < z < 2.0

    def test_surface_area_cube(self):
        c = Cell.Cube(2.0)
        # Surface area of 2x2x2 cube = 6 * 4 = 24
        area = Cell.SurfaceArea(c)
        assert area == pytest.approx(24.0, abs=TOLERANCE)


# ===========================================================================
# Serialization
# ===========================================================================

class TestCellSerialization:
    def test_brep_roundtrip(self):
        c = Cell.Prism(2.0, 3.0, 4.0)
        brep = Topology.BREPString(c)
        assert brep is not None
        c2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(c2, "Cell")
        assert Cell.Volume(c2) == pytest.approx(Cell.Volume(c), abs=TOLERANCE)
