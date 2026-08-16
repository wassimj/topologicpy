# Copyright (C) 2026
# TopologicPy CellComplex unit tests.

import pytest


pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("shapely")

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Cell = pytest.importorskip("topologicpy.Cell").Cell
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Topology = pytest.importorskip("topologicpy.Topology").Topology


TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _assert_topology(topology):
    assert Topology.IsInstance(topology, "Topology")


def _assert_cell(cell):
    assert Topology.IsInstance(cell, "Cell")


def _assert_cellcomplex(cell_complex):
    assert Topology.IsInstance(cell_complex, "CellComplex")


def _assert_shell(shell):
    assert Topology.IsInstance(shell, "Shell")


def _rectangle_wire(z=0, width=2, length=2):
    return Wire.Rectangle(
        origin=_v(0, 0, z),
        width=width,
        length=length,
        direction=[0, 0, 1],
        placement="center",
        silent=True,
    )


@pytest.fixture
def simple_cellcomplex():
    cell_complex = CellComplex.Prism(
        width=2,
        length=2,
        height=2,
        uSides=2,
        vSides=1,
        wSides=1,
        tolerance=0.0001,
    )
    _assert_cellcomplex(cell_complex)
    return cell_complex


def test_prism_box_and_cube_create_cellcomplexes():
    prism = CellComplex.Prism(width=2, length=3, height=4, uSides=2, vSides=1, wSides=1)
    box = CellComplex.Box(width=2, length=3, height=4, uSides=2, vSides=1, wSides=1)
    cube = CellComplex.Cube(size=2, uSides=2, vSides=1, wSides=1)

    for cell_complex in [prism, box, cube]:
        _assert_cellcomplex(cell_complex)
        cells = CellComplex.Cells(cell_complex)
        assert isinstance(cells, list)
        assert len(cells) >= 2
        volume = CellComplex.Volume(cell_complex)
        assert volume is not None
        assert abs(volume) > 0


def test_accessors_return_collections_and_external_boundary(simple_cellcomplex):
    cells = CellComplex.Cells(simple_cellcomplex)
    faces = CellComplex.Faces(simple_cellcomplex)
    edges = CellComplex.Edges(simple_cellcomplex, silent=True)
    vertices = CellComplex.Vertices(simple_cellcomplex)
    wires = CellComplex.Wires(simple_cellcomplex)
    shells = CellComplex.Shells(simple_cellcomplex)
    external_boundary = CellComplex.ExternalBoundary(simple_cellcomplex, silent=True)
    external_faces = CellComplex.ExternalFaces(simple_cellcomplex)
    internal_faces = CellComplex.InternalFaces(simple_cellcomplex)
    non_manifold_faces = CellComplex.NonManifoldFaces(simple_cellcomplex)

    assert isinstance(cells, list)
    assert len(cells) >= 2
    assert isinstance(faces, list)
    assert len(faces) > 0
    assert isinstance(edges, list)
    assert len(edges) > 0
    assert isinstance(vertices, list)
    assert len(vertices) > 0
    assert isinstance(wires, list)
    assert len(wires) > 0
    assert isinstance(shells, list)
    assert len(shells) > 0
    _assert_shell(external_boundary)
    assert isinstance(external_faces, list)
    assert len(external_faces) > 0
    assert isinstance(internal_faces, list)
    assert isinstance(non_manifold_faces, list)


def test_decompose_returns_expected_component_dictionary(simple_cellcomplex):
    decomposition = CellComplex.Decompose(simple_cellcomplex, silent=True)

    assert isinstance(decomposition, dict)
    for key in [
        "cells",
        "externalVerticalFaces",
        "internalVerticalFaces",
        "topHorizontalFaces",
        "bottomHorizontalFaces",
        "internalHorizontalFaces",
        "externalInclinedFaces",
        "internalInclinedFaces",
    ]:
        assert key in decomposition
        assert isinstance(decomposition[key], list)


def test_bycells_and_bycellscluster_round_trip(simple_cellcomplex):
    cells = CellComplex.Cells(simple_cellcomplex)
    assert isinstance(cells, list)
    assert len(cells) >= 2

    rebuilt = CellComplex.ByCells(cells, silent=True)
    _assert_topology(rebuilt)

    cluster = Cluster.ByTopologies(cells, silent=True)
    rebuilt_from_cluster = CellComplex.ByCellsCluster(cluster, silent=True)
    _assert_topology(rebuilt_from_cluster)




def test_byfacescluster_rebuilds_from_face_cluster(simple_cellcomplex):
    faces = CellComplex.Faces(simple_cellcomplex)
    cluster = Cluster.ByTopologies(faces, silent=True)

    rebuilt = CellComplex.ByFacesCluster(cluster, silent=True)

    _assert_cellcomplex(rebuilt)


def test_bywires_and_bywirescluster_loft_between_matching_wires():
    lower = _rectangle_wire(z=-1, width=2, length=2)
    upper = _rectangle_wire(z=1, width=2, length=2)

    lofted = CellComplex.ByWires([lower, upper], triangulate=False)
    _assert_topology(lofted)

    cluster = Cluster.ByTopologies([lower, upper], silent=True)
    lofted_from_cluster = CellComplex.ByWiresCluster(cluster, triangulate=False)
    _assert_topology(lofted_from_cluster)


def test_remove_collinear_edges_preserves_valid_topology(simple_cellcomplex):
    cleaned = CellComplex.RemoveCollinearEdges(simple_cellcomplex, silent=True)

    _assert_topology(cleaned)


def test_octahedron_tetrahedron_and_torus_create_topologies():
    octahedron = CellComplex.Octahedron(radius=0.5)
    tetrahedron = CellComplex.Tetrahedron(length=1, depth=1, silent=True)
    torus = CellComplex.Torus(majorRadius=0.6, minorRadius=0.15, uSides=8, vSides=6)

    for topology in [octahedron, tetrahedron, torus]:
        _assert_topology(topology)


def test_delaunay_and_voronoi_reject_empty_or_invalid_input():
    assert CellComplex.Delaunay(vertices=[]) is None
    assert CellComplex.Voronoi(vertices=[]) is None


def test_by_disjointed_faces_rejects_invalid_inputs():
    face = Face.Rectangle(width=1, length=1, silent=True)

    assert CellComplex.ByDisjointedFaces([], silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face], silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face, face], minOffset=-1, silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face, face], minOffset=2, maxOffset=1, silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face, face], minCells=1, silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face, face], minCells=4, maxCells=3, silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face, face], maxAttempts=0, silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face, face], patience=-1, silent=True) is None
    assert CellComplex.ByDisjointedFaces([face, face, face], patience=10, maxAttempts=5, silent=True) is None





def test_byfaces_public_variants_rebuild_from_existing_faces(simple_cellcomplex):
    faces = CellComplex.Faces(simple_cellcomplex)
    assert isinstance(faces, list)
    assert len(faces) > 0

    rebuilt = CellComplex.ByFaces(faces, silent=True)
    rebuilt_topologic = CellComplex.ByFacesTopologic(faces, silent=True)

    _assert_cellcomplex(rebuilt)
    _assert_topology(rebuilt_topologic)


def test_invalid_inputs_return_none_for_public_builders_and_accessors():
    assert CellComplex.ByCells(None, silent=True) is None
    assert CellComplex.ByCells([], silent=True) is None
    assert CellComplex.ByCellsCluster(None, silent=True) is None
    assert CellComplex.ByFaces(None, silent=True) is None
    assert CellComplex.ByFaces([], silent=True) is None
    assert CellComplex.ByFacesTopologic(None, silent=True) is None
    assert CellComplex.ByFacesTopologic([], silent=True) is None
    assert CellComplex.ByFacesCluster(None, silent=True) is None
    assert CellComplex.ByWires(None) is None
    assert CellComplex.ByWires([]) is None
    assert CellComplex.ByWiresCluster(None) is None
    assert CellComplex.Cells(None) is None
    assert CellComplex.Decompose(None, silent=True) is None
    assert CellComplex.Edges(None, silent=True) is None
    assert CellComplex.ExternalBoundary(None, silent=True) is None
    assert CellComplex.Faces(None) is None
    assert CellComplex.RemoveCollinearEdges(None, silent=True) is None
    assert CellComplex.Shells(None) is None
    assert CellComplex.Vertices(None) is None
    assert CellComplex.Volume(None) is None
    assert CellComplex.Wires(None) is None
