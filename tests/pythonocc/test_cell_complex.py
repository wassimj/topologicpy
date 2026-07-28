# Copyright (C) 2026
# PythonOCC backend CellComplex parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_cell_complex.py -v

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


def _simple_cell_complex():
    """Create a simple 2-cell complex (two adjacent cubes)."""
    c1 = Cell.Prism(1.0, 1.0, 1.0)
    c2 = Cell.Prism(1.0, 1.0, 1.0)
    c2 = Topology.Translate(c2, 1.0, 0, 0)
    return CellComplex.ByCells([c1, c2])


def _grid_cell_complex():
    """Create a 2x2x1 grid of cells."""
    cells = []
    for x in range(2):
        for y in range(2):
            c = Cell.Prism(1.0, 1.0, 1.0)
            c = Topology.Translate(c, float(x), float(y), 0)
            cells.append(c)
    return CellComplex.ByCells(cells)


# ===========================================================================
# Constructors
# ===========================================================================

class TestCellComplexConstructors:
    def test_by_cells(self):
        cc = _simple_cell_complex()
        assert Topology.IsInstance(cc, "CellComplex")

    def test_by_faces(self):
        f1 = Face.Rectangle(1.0, 1.0)
        f2 = Face.Rectangle(1.0, 1.0)
        f2 = Topology.Translate(f2, 0, 0, 1.0)
        cc = CellComplex.ByFaces([f1, f2])
        assert Topology.IsInstance(cc, "CellComplex")

    def test_prism(self):
        cc = CellComplex.Prism(2.0, 2.0, 2.0)
        assert Topology.IsInstance(cc, "CellComplex")

    def test_by_sweeps(self):
        f = Face.Rectangle(1.0, 1.0)
        cc = CellComplex.BySweeps([f], [Vertex.ByCoordinates(0, 0, 1)])
        assert Topology.IsInstance(cc, "CellComplex")


# ===========================================================================
# Accessors
# ===========================================================================

class TestCellComplexAccessors:
    def test_cells(self):
        cc = _simple_cell_complex()
        cells = CellComplex.Cells(cc)
        assert len(cells) == 2

    def test_faces(self):
        cc = _simple_cell_complex()
        faces = CellComplex.Faces(cc)
        assert len(faces) >= 10  # Two cubes share a face

    def test_edges(self):
        cc = _simple_cell_complex()
        edges = CellComplex.Edges(cc)
        assert len(edges) >= 16

    def test_vertices(self):
        cc = _simple_cell_complex()
        verts = CellComplex.Vertices(cc)
        assert len(verts) >= 8  # Shared vertices

    def test_shells(self):
        cc = _simple_cell_complex()
        shells = CellComplex.Shells(cc)
        assert len(shells) >= 2

    def test_external_boundary(self):
        cc = _simple_cell_complex()
        eb = CellComplex.ExternalBoundary(cc)
        assert eb is not None


# ===========================================================================
# Type checking
# ===========================================================================

class TestCellComplexType:
    def test_is_instance_cell_complex(self):
        cc = _simple_cell_complex()
        assert Topology.IsInstance(cc, "CellComplex") is True

    def test_is_not_cell(self):
        cc = _simple_cell_complex()
        assert Topology.IsInstance(cc, "Cell") is False

    def test_type_returns_cell_complex(self):
        cc = _simple_cell_complex()
        assert Topology.Type(cc) == 64  # CellComplex type ID


# ===========================================================================
# Geometry
# ===========================================================================

class TestCellComplexGeometry:
    def test_volume(self):
        cc = _simple_cell_complex()
        vol = CellComplex.Volume(cc)
        assert vol == pytest.approx(2.0, abs=TOLERANCE)  # 2 cubes of volume 1

    def test_area(self):
        cc = _simple_cell_complex()
        area = CellComplex.Area(cc)
        assert area > 0

    def test_external_boundary_volume(self):
        cc = _simple_cell_complex()
        eb = CellComplex.ExternalBoundary(cc)
        vol = Topology.Volume(eb)
        assert vol == pytest.approx(2.0, abs=TOLERANCE)


# ===========================================================================
# Adjacency
# ===========================================================================

class TestCellComplexAdjacency:
    def test_adjacent_cells(self):
        cc = _simple_cell_complex()
        cells = CellComplex.Cells(cc)
        c1 = cells[0]
        adj = CellComplex.AdjacentCells(cc, c1)
        assert len(adj) >= 1  # At least one adjacent cell

    def test_containing_cell(self):
        cc = _grid_cell_complex()
        v = _v(0.5, 0.5, 0.5)
        c = CellComplex.ContainingCell(cc, v)
        assert c is not None
        assert Topology.IsInstance(c, "Cell")


# ===========================================================================
# Operations
# ===========================================================================

class TestCellComplexOperations:
    def test_self_merge(self):
        cc = _simple_cell_complex()
        cc2 = CellComplex.SelfMerge(cc)
        assert Topology.IsInstance(cc2, "CellComplex")

    def test_decompose(self):
        cc = _simple_cell_complex()
        decomposed = CellComplex.Decompose(cc)
        assert decomposed is not None


# ===========================================================================
# Serialization
# ===========================================================================

class TestCellComplexSerialization:
    def test_brep_roundtrip(self):
        cc = _simple_cell_complex()
        brep = Topology.BREPString(cc)
        assert brep is not None
        cc2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(cc2, "CellComplex")
        assert CellComplex.Volume(cc2) == pytest.approx(CellComplex.Volume(cc), abs=TOLERANCE)
