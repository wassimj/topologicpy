import math

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Face = pytest.importorskip("topologicpy.Face").Face
Cell = pytest.importorskip("topologicpy.Cell").Cell
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
Topology = pytest.importorskip("topologicpy.Topology").Topology


def _assert_cc(value):
    assert Topology.IsInstance(value, "CellComplex")


def test_bycells_single_cell_returns_cellcomplex():
    cell = Cell.Prism(width=1, length=1, height=1, silent=True)
    cc = CellComplex.ByCells([cell], silent=True)
    _assert_cc(cc)
    assert len(CellComplex.Cells(cc, silent=True)) == 1
    assert math.isclose(CellComplex.Volume(cc, mantissa=12, silent=True), 1.0, rel_tol=1e-9)


def test_prism_builds_exact_regular_grid():
    cc = CellComplex.Prism(
        width=2,
        length=3,
        height=4,
        uSides=2,
        vSides=3,
        wSides=2,
        placement="lowerleft",
        silent=True,
    )
    _assert_cc(cc)
    assert len(CellComplex.Cells(cc, silent=True)) == 12
    assert math.isclose(CellComplex.Volume(cc, mantissa=12, silent=True), 24.0, rel_tol=1e-9)


def test_prism_rejects_invalid_divisions():
    assert CellComplex.Prism(uSides=0, silent=True) is None
    assert CellComplex.Prism(vSides=0, silent=True) is None
    assert CellComplex.Prism(wSides=0, silent=True) is None


def test_octahedron_lowerleft_places_bbox_minimum_at_origin():
    cc = CellComplex.Octahedron(radius=1.0, placement="lowerleft", silent=True)
    _assert_cc(cc)
    vertices = CellComplex.Vertices(cc, silent=True)
    assert min(Vertex.X(v, mantissa=9) for v in vertices) == pytest.approx(0.0, abs=1e-7)
    assert min(Vertex.Y(v, mantissa=9) for v in vertices) == pytest.approx(0.0, abs=1e-7)
    assert min(Vertex.Z(v, mantissa=9) for v in vertices) == pytest.approx(0.0, abs=1e-7)
    assert len(CellComplex.Cells(cc, silent=True)) == 2


def test_tetrahedron_depth_zero_and_one_preserve_volume():
    expected = 1.0 / (6.0 * math.sqrt(2.0))

    cc0 = CellComplex.Tetrahedron(length=1.0, depth=0, silent=True)
    _assert_cc(cc0)
    assert len(CellComplex.Cells(cc0, silent=True)) == 1
    assert math.isclose(CellComplex.Volume(cc0, mantissa=12, silent=True), expected, rel_tol=1e-8)

    cc1 = CellComplex.Tetrahedron(length=1.0, depth=1, silent=True)
    _assert_cc(cc1)
    assert len(CellComplex.Cells(cc1, silent=True)) == 8
    assert math.isclose(CellComplex.Volume(cc1, mantissa=12, silent=True), expected, rel_tol=1e-8)


def test_delaunay_builds_cells_directly():
    vertices = [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(1, 0, 0),
        Vertex.ByCoordinates(0, 1, 0),
        Vertex.ByCoordinates(0, 0, 1),
        Vertex.ByCoordinates(0.2, 0.2, 0.2),
    ]
    cc = CellComplex.Delaunay(vertices=vertices, silent=True)
    _assert_cc(cc)
    cells = CellComplex.Cells(cc, silent=True)
    assert isinstance(cells, list)
    assert len(cells) >= 1


def test_byfaces_default_rebuilds_without_shapely_path():
    source = CellComplex.Prism(width=2, length=2, height=2, uSides=2, vSides=1, wSides=1, silent=True)
    faces = CellComplex.Faces(source, silent=True)
    rebuilt = CellComplex.ByFaces(faces, silent=True)
    _assert_cc(rebuilt)
    assert len(CellComplex.Cells(rebuilt, silent=True)) == 2


def test_accessors_accept_silent_on_invalid_input():
    assert CellComplex.Cells(None, silent=True) is None
    assert CellComplex.Edges(None, silent=True) is None
    assert CellComplex.Faces(None, silent=True) is None
    assert CellComplex.InternalFaces(None, silent=True) is None
    assert CellComplex.NonManifoldFaces(None, silent=True) is None
    assert CellComplex.Shells(None, silent=True) is None
    assert CellComplex.Vertices(None, silent=True) is None
    assert CellComplex.Wires(None, silent=True) is None
