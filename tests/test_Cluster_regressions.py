import pytest

pytest.importorskip("numpy")

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Face = pytest.importorskip("topologicpy.Face").Face
Cell = pytest.importorskip("topologicpy.Cell").Cell
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary


def _v(x, y=0, z=0):
    return Vertex.ByCoordinates(x, y, z)


def test_single_topology_constructor_has_stable_cluster_return_type():
    vertex = _v(0, 0, 0)
    cluster = Cluster.ByTopologies(vertex, silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    members = Cluster.Topologies(cluster, silent=True)
    assert isinstance(members, list)
    assert len(members) == 1
    assert Topology.IsInstance(members[0], "Vertex")


def test_by_function_returns_topologic_clusters():
    vertices = [_v(0), _v(1), _v(2), _v(10)]
    groups = Cluster.ByFunction(
        vertices,
        lambda topology, mantissa=6, tolerance=0.0001: "left" if Vertex.X(topology) < 5 else "right",
        silent=True,
    )
    assert isinstance(groups, list)
    assert len(groups) == 2
    assert all(Topology.IsInstance(group, "Cluster") for group in groups)
    assert sorted(len(Cluster.Topologies(group, silent=True)) for group in groups) == [1, 3]


def test_by_function_numeric_tolerance_does_not_create_empty_groups():
    vertices = [_v(0.00), _v(0.05), _v(1.00)]
    groups = Cluster.ByFunction(
        vertices,
        lambda topology, mantissa=6, tolerance=0.1: Vertex.X(topology),
        tolerance=0.1,
        silent=True,
    )
    assert len(groups) == 2
    assert sorted(len(Cluster.Topologies(group, silent=True)) for group in groups) == [1, 2]


def test_by_formula_is_safe_and_supports_descending_ranges():
    cluster = Cluster.ByFormula("X**2", xRange=(2, 0, -1), silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    assert len(Cluster.Vertices(cluster, silent=True)) == 3
    assert Cluster.ByFormula("__import__('os').system('echo unsafe')", xRange=(0, 1, 1), silent=True) is None
    assert Cluster.ByFormula("X", xRange=(0, 1, 0), silent=True) is None


def test_dictionary_transfer_is_explicit():
    v1 = Topology.SetDictionary(_v(0), Dictionary.ByKeyValue("a", 1), silent=True)
    v2 = Topology.SetDictionary(_v(1), Dictionary.ByKeyValue("b", 2), silent=True)

    plain = Cluster.ByTopologies(v1, v2, transferDictionaries=False, silent=True)
    transferred = Cluster.ByTopologies(v1, v2, transferDictionaries=True, silent=True)

    d0 = Topology.Dictionary(plain)
    d1 = Topology.Dictionary(transferred)
    assert Dictionary.ValueAtKey(d0, "a") is None
    assert Dictionary.ValueAtKey(d0, "b") is None
    assert Dictionary.ValueAtKey(d1, "a") == 1
    assert Dictionary.ValueAtKey(d1, "b") == 2


def test_free_topologies_are_direct_constituents():
    face = Face.Rectangle(width=2, length=2, silent=True)
    edge = Edge.ByVertices([_v(-1, -1), _v(1, 1)], silent=True)
    cluster = Cluster.ByTopologies(face, edge, silent=True)

    free = Cluster.FreeTopologies(cluster, silent=True)
    assert isinstance(free, list)
    assert len(free) == 2
    assert sum(Topology.IsInstance(item, "Face") for item in free) == 1
    assert sum(Topology.IsInstance(item, "Edge") for item in free) == 1
    assert Topology.IsInstance(Cluster.Simplify(cluster, silent=True), "Cluster")


def test_simplify_only_collapses_one_member_cluster():
    face = Face.Rectangle(width=1, length=1, silent=True)
    cluster = Cluster.ByTopologies(face, silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    assert Topology.IsInstance(Cluster.Simplify(cluster, silent=True), "Face")


def test_dbscan_is_dependency_light_and_reports_noise():
    vertices = [_v(0), _v(1), _v(2), _v(10)]
    groups, noise = Cluster.DBSCAN(
        vertices,
        keys=["x"],
        epsilon=1.1,
        minSamples=2,
        silent=True,
    )
    assert isinstance(groups, list)
    assert len(groups) == 1
    assert len(Cluster.Topologies(groups[0], silent=True)) == 3
    assert Topology.IsInstance(noise, "Cluster")
    assert len(Cluster.Topologies(noise, silent=True)) == 1


def test_dbscan_rejects_missing_noncoordinate_feature():
    vertices = [_v(0), _v(1)]
    groups, noise = Cluster.DBSCAN(
        vertices,
        keys=["missing"],
        epsilon=1,
        minSamples=2,
        silent=True,
    )
    assert groups is None
    assert noise is None


def test_kmeans_rejects_unknown_metric_without_recursion():
    vertices = [_v(0), _v(1), _v(10), _v(11)]
    assert Cluster.KMeans(vertices, k=2, distanceMeasure="invalid", silent=True) is None


def test_kmeans_mahalanobis_returns_requested_clusters():
    vertices = [_v(0), _v(1), _v(10), _v(11)]
    groups = Cluster.KMeans(
        vertices,
        keys=["x", "y", "z"],
        k=2,
        distanceMeasure="mahalanobis",
        nInit=2,
        randomSeed=7,
        silent=True,
    )
    assert isinstance(groups, list)
    assert len(groups) == 2
    assert all(Topology.IsInstance(group, "Cluster") for group in groups)


def test_merge_cells_uses_transitive_connected_components():
    cell_a = Cell.Prism(width=1, length=1, height=1, placement="center")
    cell_b = Topology.Translate(cell_a, x=1, y=0, z=0)
    cell_c = Topology.Translate(cell_a, x=2, y=0, z=0)

    merged = Cluster.MergeCells([cell_a, cell_b, cell_c], silent=True)
    assert Topology.IsInstance(merged, "Cluster")
    members = Cluster.Topologies(merged, silent=True)
    assert isinstance(members, list)
    assert len(members) == 1
    assert Topology.IsInstance(members[0], "CellComplex")
    assert len(CellComplex.Cells(members[0], silent=True)) == 3
