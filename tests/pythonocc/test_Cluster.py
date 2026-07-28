# Auto-generated PythonOCC backend parity test
# Copied from test_Cluster.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# TopologicPy Cluster unit tests.

import pytest


pytest.importorskip("numpy")
pytest.importorskip("scipy")

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Cell = pytest.importorskip("topologicpy.Cell").Cell
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary


TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _assert_vertex(vertex):
    assert Topology.IsInstance(vertex, "Vertex")


def _assert_edge(edge):
    assert Topology.IsInstance(edge, "Edge")


def _assert_wire(wire):
    assert Topology.IsInstance(wire, "Wire")


def _assert_face(face):
    assert Topology.IsInstance(face, "Face")


def _assert_cell(cell):
    assert Topology.IsInstance(cell, "Cell")


def _assert_cluster(cluster):
    assert Topology.IsInstance(cluster, "Cluster")


def _assert_topology(topology):
    assert Topology.IsInstance(topology, "Topology")


@pytest.fixture
def basic_vertices():
    return [
        _v(0, 0, 0),
        _v(1, 0, 0),
        _v(2, 0, 0),
        _v(10, 0, 0),
    ]


@pytest.fixture
def simple_cluster(basic_vertices):
    edge = Edge.ByVertices([basic_vertices[0], basic_vertices[1]], silent=True)
    wire = Wire.Rectangle(width=1, length=1, silent=True)
    face = Face.Rectangle(width=1, length=1, silent=True)
    return Cluster.ByTopologies([basic_vertices[2], edge, wire, face], silent=True)


def test_by_topologies_accepts_list_and_variadic_inputs(basic_vertices):
    edge = Edge.ByVertices([basic_vertices[0], basic_vertices[1]], silent=True)

    list_cluster = Cluster.ByTopologies([basic_vertices[0], basic_vertices[1], edge], silent=True)
    variadic_cluster = Cluster.ByTopologies(basic_vertices[0], basic_vertices[1], edge, silent=True)

    for cluster in [list_cluster, variadic_cluster]:
        _assert_cluster(cluster)
        vertices = Cluster.Vertices(cluster, silent=True)
        edges = Cluster.Edges(cluster, silent=True)
        assert isinstance(vertices, list)
        assert isinstance(edges, list)
        assert len(vertices) >= 2
        assert len(edges) >= 1

    assert Cluster.ByTopologies([], silent=True) is None
    assert Cluster.ByTopologies([None, "bad"], silent=True) is None


def test_by_topologies_with_one_topology_returns_that_topology(basic_vertices):
    result = Cluster.ByTopologies(basic_vertices[0], silent=True)
    _assert_vertex(result)


def test_by_formula_creates_vertex_clusters_from_2d_and_3d_formulas():
    parabola = Cluster.ByFormula("X**2", xRange=(0, 2, 1))
    surface = Cluster.ByFormula("X + Y", xRange=(0, 1, 1), yRange=(0, 1, 1))

    for cluster in [parabola, surface]:
        _assert_cluster(cluster)
        vertices = Cluster.Vertices(cluster, silent=True)
        assert isinstance(vertices, list)
        assert len(vertices) > 0
        assert all(Topology.IsInstance(v, "Vertex") for v in vertices)

    assert len(Cluster.Vertices(parabola, silent=True)) == 3
    assert len(Cluster.Vertices(surface, silent=True)) == 4

    assert Cluster.ByFormula("X", xRange=None, yRange=None) is None
    assert Cluster.ByFormula("x", xRange=(0, 1, 1), xString="x") is None
    assert Cluster.ByFormula("y", yRange=(0, 1, 1), yString="y") is None


def test_by_function_groups_numeric_boolean_string_and_none_values(basic_vertices):
    numeric_groups = Cluster.ByFunction(
        basic_vertices,
        lambda topology, mantissa=6, tolerance=0.0001: Vertex.X(topology, mantissa=mantissa) < 5,
        silent=True,
    )
    string_groups = Cluster.ByFunction(
        basic_vertices,
        lambda topology, mantissa=6, tolerance=0.0001: "left" if Vertex.X(topology) < 5 else "right",
        silent=True,
    )
    none_groups = Cluster.ByFunction(
        basic_vertices,
        lambda topology, mantissa=6, tolerance=0.0001: None,
        silent=True,
    )

    assert sorted(len(group) for group in numeric_groups) == [1, 3]
    assert sorted(len(group) for group in string_groups) == [1, 3]
    assert len(none_groups) == 1
    assert len(none_groups[0]) == len(basic_vertices)

    assert Cluster.ByFunction(None, lambda topology, mantissa=6, tolerance=0.0001: 0, silent=True) is None
    assert Cluster.ByFunction(basic_vertices, None, silent=True) is None
    assert Cluster.ByFunction([], lambda topology, mantissa=6, tolerance=0.0001: 0, silent=True) == []


def test_cluster_accessors_return_expected_collection_types(simple_cluster):
    topologies = Cluster.Topologies(simple_cluster, silent=True)
    vertices = Cluster.Vertices(simple_cluster, silent=True)
    edges = Cluster.Edges(simple_cluster, silent=True)
    wires = Cluster.Wires(simple_cluster, silent=True)
    faces = Cluster.Faces(simple_cluster, silent=True)
    shells = Cluster.Shells(simple_cluster, silent=True)
    cells = Cluster.Cells(simple_cluster, silent=True)
    cell_complexes = Cluster.CellComplexes(simple_cluster, silent=True)

    assert isinstance(topologies, list)
    assert isinstance(vertices, list)
    assert isinstance(edges, list)
    assert isinstance(wires, list)
    assert isinstance(faces, list)
    assert isinstance(shells, list)
    assert isinstance(cells, list)
    assert isinstance(cell_complexes, list)

    assert len(topologies) > 0
    assert len(vertices) > 0
    assert len(edges) > 0
    assert len(wires) > 0
    assert len(faces) > 0
    assert len(shells) == 0
    assert len(cells) == 0
    assert len(cell_complexes) == 0


def test_invalid_accessor_inputs_return_none():
    assert Cluster.Topologies(None, silent=True) is None
    assert Cluster.CellComplexes(None, silent=True) is None
    assert Cluster.Cells(None, silent=True) is None
    assert Cluster.Edges(None, silent=True) is None
    assert Cluster.Faces(None, silent=True) is None
    assert Cluster.Shells(None, silent=True) is None
    assert Cluster.Vertices(None, silent=True) is None
    assert Cluster.Wires(None, silent=True) is None
    assert Cluster.HighestType(None) is None
    assert Cluster.ExternalBoundary(None, silent=True) is None
    assert Cluster.Simplify(None) is None


def test_free_topology_accessors_on_mixed_cluster_return_lists(simple_cluster):
    free_vertices = Cluster.FreeVertices(simple_cluster, silent=True)
    free_edges = Cluster.FreeEdges(simple_cluster, silent=True)
    free_wires = Cluster.FreeWires(simple_cluster, silent=True)
    free_faces = Cluster.FreeFaces(simple_cluster, silent=True)
    free_shells = Cluster.FreeShells(simple_cluster, silent=True)
    free_cells = Cluster.FreeCells(simple_cluster, silent=True)
    free_topologies = Cluster.FreeTopologies(simple_cluster)

    for result in [
        free_vertices,
        free_edges,
        free_wires,
        free_faces,
        free_shells,
        free_cells,
        free_topologies,
    ]:
        assert isinstance(result, list)

    assert len(free_topologies) > 0


def test_highest_type_and_simplify_for_single_topology_cluster():
    face = Face.Rectangle(width=1, length=1, silent=True)
    cluster = Cluster.ByTopologies([face], silent=True)

    _assert_cluster(cluster)
    assert Cluster.HighestType(cluster) == Topology.TypeID("Face")

    simplified = Cluster.Simplify(cluster)
    _assert_face(simplified)


def test_external_boundary_returns_topology_for_mixed_cluster(simple_cluster):
    external_boundary = Cluster.ExternalBoundary(simple_cluster, silent=True)

    _assert_topology(external_boundary)


def test_dbscan_clusters_nearby_vertices_and_reports_noise_when_present(basic_vertices):
    clusters, noise = Cluster.DBSCAN(
        basic_vertices,
        keys=["x", "y", "z"],
        epsilon=1.25,
        minSamples=2,
    )

    assert isinstance(clusters, list)
    assert len(clusters) >= 1
    assert all(Topology.IsInstance(cluster, "Cluster") for cluster in clusters)
    assert noise is None or Topology.IsInstance(noise, "Cluster")

    invalid_clusters, invalid_noise = Cluster.DBSCAN(None)
    assert invalid_clusters is None
    assert invalid_noise is None


def test_kmeans_returns_requested_number_of_clusters_with_centroid_dictionary(basic_vertices):
    clusters = Cluster.KMeans(
        basic_vertices,
        keys=["x", "y", "z"],
        k=2,
        maxIterations=20,
        nInit=2,
        randomSeed=7,
        silent=True,
    )

    assert isinstance(clusters, list)
    assert len(clusters) == 2

    for cluster in clusters:
        _assert_cluster(cluster)
        dictionary = Topology.Dictionary(cluster)
        centroid = Dictionary.ValueAtKey(dictionary, "k_centroid")
        assert isinstance(centroid, list)
        assert len(centroid) == 3

    assert Cluster.KMeans(None, silent=True) is None
    assert Cluster.KMeans(basic_vertices, k=0, silent=True) is None
    assert Cluster.KMeans(basic_vertices, k=len(basic_vertices) + 1, silent=True) is None


def test_mystic_rose_creates_edge_cluster_from_default_circle():
    rose = Cluster.MysticRose(radius=1, sides=6, perimeter=True, silent=True)

    _assert_cluster(rose)
    edges = Cluster.Edges(rose, silent=True)
    assert isinstance(edges, list)
    assert len(edges) >= 6

    open_wire = Wire.Line(length=1, sides=2, silent=True)
    assert Cluster.MysticRose(wire=open_wire, silent=True) is None


def test_merge_cells_groups_adjacent_or_single_cells():
    cell_a = Cell.Prism(width=1, length=1, height=1, placement="center")
    cell_b = Topology.Translate(cell_a, x=1, y=0, z=0)

    merged = Cluster.MergeCells([cell_a, cell_b])

    _assert_cluster(merged)
    assert len(Cluster.Topologies(merged, silent=True)) > 0

    assert Cluster.MergeCells(None) is None
    assert Cluster.MergeCells([]) is None


def test_tripod_creates_cluster_with_three_axes():
    tripod = Cluster.Tripod(size=1, radius=0.03, sides=4)

    _assert_cluster(tripod)
    topologies = Cluster.Topologies(tripod, silent=True)
    assert isinstance(topologies, list)
    assert len(topologies) >= 3


def test_single_dimension_clusters_have_expected_free_members():
    vertices = [_v(0, 0, 0), _v(1, 0, 0), _v(2, 0, 0)]
    cluster = Cluster.ByTopologies(vertices, silent=True)

    _assert_cluster(cluster)
    free_vertices = Cluster.FreeVertices(cluster, silent=True)
    assert isinstance(free_vertices, list)
    assert len(free_vertices) >= 3

    topologies = Cluster.Topologies(cluster, silent=True)
    assert isinstance(topologies, list)
    assert len(topologies) >= 3
