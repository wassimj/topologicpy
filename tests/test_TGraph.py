"""Unit tests for topologicpy.TGraph.

These tests focus on the pure-Python TGraph data model and algorithms. They avoid
TopologicCore, IFC geometry, igraph, NetworkX, Plotly rendering, and other optional
runtime dependencies unless a test explicitly verifies graceful dependency handling.
"""

from __future__ import annotations

import builtins
import os

import pytest

from topologicpy.TGraph import TGraph


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _path_graph():
    g = TGraph(directed=False, dictionary={"label": "path"})
    g.AddVertex({"label": "A", "x": 0.0, "y": 0.0, "z": 0.0})
    g.AddVertex({"label": "B", "x": 3.0, "y": 4.0, "z": 0.0})
    g.AddVertex({"label": "C", "x": 6.0, "y": 4.0, "z": 0.0})
    g.AddEdge(0, 1, dictionary={"weight": 2.0, "label": "ab"})
    g.AddEdge(1, 2, dictionary={"weight": 3.0, "label": "bc"})
    return g


def _routing_graph():
    """
    Returns a graph with deliberately different geometric and weighted routes.

    Geometric / hop route 0-1-2:
        length = 2
        weight = 20

    Dictionary-weighted route 0-3-4-2:
        length = 6
        weight = 3
    """
    g = TGraph(directed=False)

    for label, (x, y, z) in [
        ("A", (0.0, 0.0, 0.0)),
        ("B", (1.0, 0.0, 0.0)),
        ("C", (2.0, 0.0, 0.0)),
        ("D", (0.0, 2.0, 0.0)),
        ("E", (2.0, 2.0, 0.0)),
    ]:
        g.AddVertex({
            "label": label,
            "x": x,
            "y": y,
            "z": z,
            "penalty": 0.0,
        })

    g.AddEdge(0, 1, dictionary={"weight": 10.0, "blocked": False})
    g.AddEdge(1, 2, dictionary={"weight": 10.0, "blocked": False})
    g.AddEdge(0, 3, dictionary={"weight": 1.0, "blocked": False})
    g.AddEdge(3, 4, dictionary={"weight": 1.0, "blocked": False})
    g.AddEdge(4, 2, dictionary={"weight": 1.0, "blocked": False})

    return g


def _topological_vs_geometric_graph():
    """
    Returns a graph where hop-shortest and geometric-shortest paths differ.

    Hop-shortest:
        0-1-2 (2 edges, very long geometrically)

    Geometric-shortest:
        0-3-4-2 (3 edges, length 10)
    """
    g = TGraph(directed=False)

    for label, (x, y, z) in [
        ("A", (0.0, 0.0, 0.0)),
        ("Far", (0.0, 100.0, 0.0)),
        ("C", (10.0, 0.0, 0.0)),
        ("B1", (3.0, 0.0, 0.0)),
        ("B2", (6.0, 0.0, 0.0)),
    ]:
        g.AddVertex({"label": label, "x": x, "y": y, "z": z})

    g.AddEdge(0, 1)
    g.AddEdge(1, 2)
    g.AddEdge(0, 3)
    g.AddEdge(3, 4)
    g.AddEdge(4, 2)

    return g


def test_constructor_add_vertex_add_edge_and_basic_accessors():
    g = _path_graph()

    assert "TGraph" in repr(g)
    assert TGraph.Order(g) == 3
    assert TGraph.Size(g) == 2
    assert TGraph.IsDirected(g) is False
    assert TGraph.Dictionary(g)["label"] == "path"

    assert TGraph.VertexIndex(g, 1) == 1
    assert TGraph.EdgeIndex(g, 0) == 0
    assert TGraph.Vertex(g, 0)["dictionary"]["label"] == "A"
    assert TGraph.Edge(g, 0)["dictionary"]["weight"] == 2.0
    assert TGraph.EdgeBetween(g, 0, 1)["index"] == 0
    assert TGraph.EdgesBetween(g, 1, 0)[0]["index"] == 0

    assert TGraph.ContainsVertex(g, 0) is True
    assert TGraph.ContainsEdge(g, 0) is True
    assert TGraph.ContainsVertex(g, 999) is False
    assert TGraph.ContainsEdge(g, 999) is False


def test_directed_adjacency_modes_and_duplicate_edge_rules():
    g = TGraph(directed=True, allowSelfLoops=False, allowParallelEdges=False)
    for i in range(3):
        g.AddVertex({"label": str(i)})

    assert g.AddEdge(0, 1, dictionary={"name": "first"}) == 0
    assert g.AddEdge(0, 1, dictionary={"name": "duplicate"}) is None
    assert g.AddEdge(1, 1) is None
    assert g.AddEdge(1, 0) == 1

    assert TGraph.AdjacentIndices(g, 0, mode="out") == [1]
    assert TGraph.AdjacentIndices(g, 0, mode="in") == [1]
    assert sorted(TGraph.AdjacentIndices(g, 0, mode="all")) == [1]
    assert TGraph.AdjacencyMatrix(g, bidirectional=False) == [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]


def test_allow_parallel_edges_and_self_loops_when_enabled():
    g = TGraph(directed=True, allowSelfLoops=True, allowParallelEdges=True)
    g.AddVertex({"label": "A"})
    g.AddVertex({"label": "B"})

    e0 = g.AddEdge(0, 1, dictionary={"weight": 1})
    e1 = g.AddEdge(0, 1, dictionary={"weight": 2})
    e2 = g.AddEdge(0, 0, dictionary={"weight": 3})

    assert [e0, e1, e2] == [0, 1, 2]
    assert TGraph.Size(g) == 3
    assert len(TGraph.EdgesBetween(g, 0, 1, directed=True)) == 2
    assert TGraph.EdgeBetween(g, 0, 0, directed=True)["index"] == 2


def test_set_dictionaries_and_coordinates_are_reflected_in_distance_helpers():
    g = TGraph()
    TGraph.AddVertexByData(g, dictionary={"id": "a"}, x=0, y=0, z=0)
    TGraph.AddVertexByData(g, dictionary={"id": "b"}, x=3, y=4, z=12)
    g.AddEdge(0, 1)

    assert TGraph.Coordinates(g, 0) == [0.0, 0.0, 0.0]
    assert TGraph.MetricDistance(g, 0, 1) == pytest.approx(13.0)
    assert TGraph.Distance(g, 0, 1, distanceType="metric") == pytest.approx(13.0)
    assert TGraph.PathLength(g, [0, 1]) == pytest.approx(13.0)
    assert TGraph.TopologicalDistance(g, 0, 1) == 1
    assert TGraph.Distance(g, 0, 1, distanceType="topological") == 1

    assert TGraph.SetVertexCoordinates(g, 1, coordinates=[0, 0, 5]) is True
    assert TGraph.PathLength(g, [0, 1]) == pytest.approx(5.0)

    g.SetDictionary({"label": "coords"})
    TGraph.SetVertexDictionary(g, 0, {"label": "origin", "x": 0, "y": 0, "z": 0})
    TGraph.SetEdgeDictionary(g, 0, {"relationship": "connects", "weight": 7})
    assert TGraph.Dictionary(g)["label"] == "coords"
    assert TGraph.VertexDictionary(g, 0)["label"] == "origin"
    assert TGraph.EdgeDictionary(g, 0)["relationship"] == "connects"


def test_topological_distance_is_independent_of_geometric_shortest_path():
    g = _topological_vs_geometric_graph()

    geometric_path, geometric_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        edgeKey="Length",
        returnCost=True,
    )
    hop_path, hop_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        edgeKey="hop",
        returnCost=True,
    )

    assert geometric_path == [0, 3, 4, 2]
    assert geometric_cost == pytest.approx(10.0)
    assert hop_path == [0, 1, 2]
    assert hop_cost == pytest.approx(2.0)

    assert TGraph.TopologicalDistance(g, 0, 2, mode="all") == 2
    assert TGraph.Distance(g, 0, 2, distanceType="topological", mode="all") == 2


def test_constructors_from_edge_pairs_adjacency_matrix_and_dictionary():
    g = TGraph.ByEdgeIndexPairs(3, [(0, 1), (1, 2)], directed=False)
    assert TGraph.Order(g) == 3
    assert TGraph.Size(g) == 2
    assert TGraph.AdjacencyList(g, mode="all") == [[1], [0, 2], [1]]

    gm = TGraph.ByAdjacencyMatrix([[0, 2], [0, 0]], directed=True)
    assert TGraph.Order(gm) == 2
    assert TGraph.Size(gm) == 1
    assert TGraph.Edge(gm, 0)["dictionary"]["weight"] == 2
    assert TGraph.AdjacencyMatrix(gm, bidirectional=False) == [[0, 1], [0, 0]]

    gd = TGraph.ByAdjacencyDictionary({"A": ["B", "C"], "B": ["C"]}, directed=True)
    assert TGraph.Order(gd) == 3
    assert TGraph.Size(gd) == 3
    assert TGraph.AdjacencyDictionary(gd) == {"A": ["B", "C"], "B": ["C"], "C": []}


def test_json_round_trip_copy_and_python_data_are_independent():
    g = _path_graph()
    g.SetDictionary({"label": "roundtrip"})

    data = TGraph.JSONData(g)
    assert data["type"] == "TGraph"
    assert data["dictionary"]["label"] == "roundtrip"

    text = TGraph.JSONString(g)
    restored = TGraph.ByJSONString(text)
    assert isinstance(restored, TGraph)
    assert TGraph.Order(restored) == TGraph.Order(g)
    assert TGraph.Size(restored) == TGraph.Size(g)
    assert TGraph.Dictionary(restored)["label"] == "roundtrip"
    assert TGraph.AdjacencyMatrix(restored) == TGraph.AdjacencyMatrix(g)

    copied = TGraph.Copy(g)
    TGraph.SetVertexDictionary(copied, 0, {"label": "changed"})
    assert TGraph.VertexDictionary(g, 0)["label"] == "A"
    assert TGraph.VertexDictionary(copied, 0)["label"] == "changed"


def test_csv_export_and_import_round_trip(tmp_path):
    g = TGraph(directed=True, allowParallelEdges=True, dictionary={"label": "csv_graph"})
    for i in range(3):
        g.AddVertex({
            "label": i,
            "x": float(i),
            "y": float(i + 1),
            "z": 0.0,
            "feat_a": float(i) + 0.5,
            "mask": "train" if i < 2 else "test",
        })
    g.AddEdge(0, 1, dictionary={"label": "a", "weight": 2.5, "feat_e": 7.0, "mask": "train"})
    g.AddEdge(1, 2, dictionary={"label": "b", "weight": 3.5, "feat_e": 8.0, "mask": "test"})

    ok = TGraph.ExportToCSV(
        g,
        str(tmp_path),
        overwrite=True,
        graphFeaturesKeys=[],
        nodeFeaturesKeys=["feat_a"],
        edgeFeaturesKeys=["feat_e"],
        bidirectional=False,
        silent=True,
    )
    assert ok is True
    assert {"graphs.csv", "nodes.csv", "edges.csv", "meta.yaml"}.issubset(
        {p.name for p in tmp_path.iterdir()}
    )

    graphs = TGraph.ByCSVPath(
        str(tmp_path),
        directed=True,
        allowParallelEdges=True,
        silent=True,
    )
    assert isinstance(graphs, list)
    assert len(graphs) == 1
    imported = graphs[0]
    assert TGraph.Order(imported) == 3
    assert TGraph.Size(imported) == 2
    assert TGraph.VertexDictionary(imported, 2)["feat_a"] == pytest.approx(2.5)
    assert TGraph.VertexDictionary(imported, 2)["feat"] == [pytest.approx(2.5)]
    assert TGraph.EdgeDictionary(imported, 1)["feat_e"] == pytest.approx(8.0)
    assert TGraph.EdgeDictionary(imported, 1)["feat"] == [pytest.approx(8.0)]


def test_csv_string_round_trip_preserves_active_records():
    g = _path_graph()
    g.RemoveEdge(1)
    g.RemoveVertex(2)

    vertices_csv = TGraph.VerticesCSVString(g, includeInactive=True)
    edges_csv = TGraph.EdgesCSVString(g, includeInactive=True)
    restored = TGraph.ByCSVStrings(
        vertices_csv,
        edges_csv,
        metadata={"directed": False},
    )

    assert isinstance(restored, TGraph)
    assert TGraph.Order(restored) == 2
    assert TGraph.Size(restored) == 1
    assert TGraph.ActiveVertexIndices(restored) == [0, 1]
    assert TGraph.ActiveEdgeIndices(restored) == [0]


def test_remove_vertex_remove_edge_and_active_indices():
    g = TGraph.ByEdgeIndexPairs(4, [(0, 1), (1, 2), (2, 3)], directed=False)
    assert TGraph.ActiveVertexIndices(g) == [0, 1, 2, 3]
    assert TGraph.ActiveEdgeIndices(g) == [0, 1, 2]

    g.RemoveVertex(1)
    assert TGraph.ActiveVertexIndices(g) == [0, 2, 3]
    assert TGraph.ActiveEdgeIndices(g) == [2]
    assert TGraph.Order(g) == 3
    assert TGraph.Size(g) == 1

    g.RemoveEdge(2)
    assert TGraph.ActiveEdgeIndices(g) == []
    assert TGraph.Size(g) == 0


def test_breadth_first_and_depth_first_traversals_are_explicit_and_deterministic():
    g = TGraph.ByEdgeIndexPairs(
        4,
        [(0, 1), (1, 3), (0, 2), (2, 3)],
        directed=False,
    )

    assert TGraph.BreadthFirstSearch(g, 0, mode="all") == [0, 1, 2, 3]
    assert TGraph.DepthFirstSearch(g, 0, mode="all") == [0, 1, 3, 2]


def test_shortest_path_defaults_to_geometric_length_and_supports_hop_and_weight_costs():
    g = _routing_graph()

    geometric_path, geometric_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        returnCost=True,
    )
    hop_path, hop_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        edgeKey="hop",
        returnCost=True,
    )
    weighted_path, weighted_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        edgeKey="weight",
        returnCost=True,
    )

    assert geometric_path == [0, 1, 2]
    assert geometric_cost == pytest.approx(2.0)

    assert hop_path == [0, 1, 2]
    assert hop_cost == pytest.approx(2.0)

    assert weighted_path == [0, 3, 4, 2]
    assert weighted_cost == pytest.approx(3.0)


def test_shortest_path_rich_returns_filters_vertex_costs_and_astar():
    g = _routing_graph()

    path, vertices, edges, cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        edgeKey="weight",
        returnVertices=True,
        returnEdges=True,
        returnCost=True,
    )

    assert path == [0, 3, 4, 2]
    assert [v["index"] for v in vertices] == path
    assert [e["index"] for e in edges] == [2, 3, 4]
    assert cost == pytest.approx(3.0)

    filtered_path, filtered_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        edgeKey="weight",
        edgeFilter=lambda edge: edge["index"] < 2,
        returnCost=True,
    )
    assert filtered_path == [0, 1, 2]
    assert filtered_cost == pytest.approx(20.0)

    g._vertices[1]["dictionary"]["penalty"] = 100.0
    penalized_path, penalized_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        vertexKey="penalty",
        returnCost=True,
    )
    assert penalized_path == [0, 3, 4, 2]
    assert penalized_cost == pytest.approx(6.0)

    astar_path, astar_cost = TGraph.ShortestPath(
        g,
        0,
        2,
        mode="all",
        useAStar=True,
        returnCost=True,
    )
    assert astar_path == [0, 1, 2]
    assert astar_cost == pytest.approx(2.0)


def test_shortest_path_respects_directed_out_in_and_all_modes():
    g = TGraph(directed=True)
    for i in range(3):
        g.AddVertex({"label": str(i), "x": float(i), "y": 0.0, "z": 0.0})
    g.AddEdge(0, 1, directed=True)
    g.AddEdge(1, 2, directed=True)

    assert TGraph.ShortestPath(g, 0, 2, mode="out", edgeKey="hop") == [0, 1, 2]
    assert TGraph.ShortestPath(g, 2, 0, mode="out", edgeKey="hop") is None
    assert TGraph.ShortestPath(g, 2, 0, mode="in", edgeKey="hop") == [2, 1, 0]
    assert TGraph.ShortestPath(g, 2, 0, mode="all", edgeKey="hop") == [2, 1, 0]


def test_shortest_path_tree_reports_costs_hops_paths_and_edges():
    g = _routing_graph()

    tree = TGraph.ShortestPathTree(
        g,
        0,
        mode="all",
        edgeKey="weight",
        includePaths=True,
        includeEdges=True,
    )

    assert tree["source"] == 0
    assert tree["mode"] == "all"
    assert tree["distance"][2] == pytest.approx(3.0)
    assert tree["hops"][2] == 3
    assert tree["parent"][2] == 4
    assert tree["parentEdge"][2] == 4
    assert tree["paths"][2] == [0, 3, 4, 2]
    assert tree["edgePaths"][2] == [2, 3, 4]
    assert tree["reachable"] == [0, 1, 2, 3, 4]


def test_shortest_paths_from_source_and_batch_shortest_paths_match_single_queries():
    g = _routing_graph()

    from_source = TGraph.ShortestPathsFromSource(
        g,
        0,
        targets=[2, 4],
        mode="all",
        edgeKey="weight",
        returnCost=True,
        returnTree=True,
    )

    assert from_source[2] == ([0, 3, 4, 2], pytest.approx(3.0))
    assert from_source[4] == ([0, 3, 4], pytest.approx(2.0))
    assert isinstance(from_source["_tree"], dict)

    batch = TGraph.ShortestPaths(
        g,
        [(0, 2), (0, 4), (3, 2)],
        mode="all",
        grouped=True,
        edgeKey="weight",
        returnCost=True,
    )

    assert batch[0] == ([0, 3, 4, 2], pytest.approx(3.0))
    assert batch[1] == ([0, 3, 4], pytest.approx(2.0))
    assert batch[2] == ([3, 4, 2], pytest.approx(2.0))


def test_shortest_path_via_vertices_supports_explicit_and_dictionary_waypoints():
    g = _routing_graph()

    path, edges, cost = TGraph.ShortestPathViaVertices(
        g,
        0,
        2,
        vertices=[3],
        mode="all",
        returnEdges=True,
        returnCost=True,
    )

    assert path == [0, 3, 4, 2]
    assert [e["index"] for e in edges] == [2, 3, 4]
    assert cost == pytest.approx(6.0)

    via_dictionary, via_cost = TGraph.ShortestPathViaVertices(
        g,
        0,
        2,
        viaKey="label",
        viaValues=["D"],
        mode="all",
        returnCost=True,
    )

    assert via_dictionary == [0, 3, 4, 2]
    assert via_cost == pytest.approx(6.0)


def test_path_all_paths_longest_path_tree_and_connectedness():
    g = TGraph.ByEdgeIndexPairs(
        4,
        [(0, 1), (1, 3), (0, 2), (2, 3)],
        directed=False,
    )
    for index, (x, y) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        TGraph.SetVertexCoordinates(g, index, coordinates=[x, y, 0])

    assert TGraph.Path(g, 0, 3) in ([0, 1, 3], [0, 2, 3])
    assert sorted(TGraph.AllPaths(g, 0, 3)) == [[0, 1, 3], [0, 2, 3]]
    assert TGraph.ConnectedComponents(g) == [[0, 1, 2, 3]]
    assert TGraph.IsConnected(g) is True
    assert TGraph.IsTree(g) is False

    g.RemoveEdge(1)
    g.RemoveEdge(3)
    comps = sorted(tuple(c) for c in TGraph.ConnectedComponents(g))
    assert comps == [(0, 1, 2), (3,)]
    assert TGraph.IsConnected(g) is False

    path_graph = TGraph.ByEdgeIndexPairs(
        4,
        [(0, 1), (1, 2), (2, 3)],
        directed=False,
    )
    for i in range(4):
        TGraph.SetVertexCoordinates(path_graph, i, coordinates=[float(i), 0.0, 0.0])

    assert TGraph.LongestPath(path_graph) == [0, 1, 2, 3]

    tree = TGraph.Tree(path_graph, vertex=0)
    assert isinstance(tree, TGraph)
    assert TGraph.Order(tree) == 4
    assert TGraph.Size(tree) == 3
    assert TGraph.IsDirected(tree) is True
    assert TGraph.Dictionary(tree)["root"] == 0


def test_degree_clustering_complete_complement_mst_and_line_graph():
    path = TGraph.ByEdgeIndexPairs(3, [(0, 1), (1, 2)], directed=False)
    assert TGraph.DegreeSequence(path) == [1, 2, 1]
    assert TGraph.Degree(path, 1) == 2
    assert TGraph.DegreeCentrality(
        path,
        key="dc",
        colorKey=None,
        nxCompatible=True,
    ) == [0.5, 1.0, 0.5]
    assert TGraph.LocalClusteringCoefficient(path, key="lcc") == [0.0, 0.0, 0.0]
    assert TGraph.AverageClusteringCoefficient(path) == 0.0

    complete = TGraph.Complete(path)
    assert TGraph.Size(complete) == 3
    assert TGraph.IsComplete(complete) is True

    complement = TGraph.Complement(path)
    assert TGraph.Size(complement) == 1
    assert {
        TGraph.Edge(complement, 0)["src"],
        TGraph.Edge(complement, 0)["dst"],
    } == {0, 2}

    mst = TGraph.MinimumSpanningTree(complete)
    assert TGraph.Order(mst) == 3
    assert TGraph.Size(mst) == 2

    line = TGraph.LineGraph(path)
    assert TGraph.Order(line) == 2
    assert TGraph.Size(line) == 1


def test_subgraph_induced_subgraph_and_neighborhood_alias():
    g = TGraph.ByEdgeIndexPairs(
        5,
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        directed=False,
    )

    sub = TGraph.Subgraph(g, [1, 2, 3], induced=True)
    assert TGraph.Order(sub) == 3
    assert TGraph.Size(sub) == 2
    assert TGraph.AdjacencyMatrix(sub) == [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]

    induced = TGraph.InducedSubgraph(g, [0, 1, 2])
    assert TGraph.Order(induced) == 3
    assert TGraph.Size(induced) == 2

    neighborhood = TGraph.Neighborhood(g, vertices=[2], k=1)
    assert isinstance(neighborhood, TGraph)
    assert TGraph.Order(neighborhood) == 3


def test_compile_cache_adjacency_helpers_compile_info_and_warmup():
    g = _path_graph()
    assert TGraph.IsCompiled(g) is False

    compiled = TGraph.Compile(
        g,
        weightKey="weight",
        useNumpy=False,
        useSciPy=False,
        useNumba=False,
    )
    assert isinstance(compiled, dict)
    assert TGraph.IsCompiled(g, weightKey="weight") is True
    assert TGraph.CompiledAdjacency(g, mode="all") == [[1], [0, 2], [1]]
    assert TGraph.ActiveVertexIndices(g) == [0, 1, 2]
    assert TGraph.ActiveEdgeIndices(g) == [0, 1]

    info = TGraph.CompileInfo(g)
    assert info["compiled"] is True
    assert info["weightKey"] == "weight"

    TGraph.ClearCompiled(g)
    assert TGraph.IsCompiled(g) is False

    report = TGraph.WarmUpAcceleration(g, mode="all", useNumba=False)
    assert report["compiled"] is True
    assert {"numpy", "scipy", "numba"}.issubset(report)


def test_annotation_ontology_helpers_and_semantic_summary_without_rdflib():
    g = _path_graph()
    TGraph.AnnotateOntology(
        g,
        ontologyClass="top:Graph",
        category="spatial",
        label="My Graph",
    )
    TGraph.AnnotateOntology(
        g,
        ontologyClass="top:Vertex",
        element="vertex",
        index=0,
        label="Start",
    )
    TGraph.AnnotateIFC(
        g,
        ifcClass="IfcWall",
        ifcGUID="abc",
        ifcName="Wall A",
        element="vertex",
        index=1,
    )

    gd = TGraph.Dictionary(g)
    assert gd["ontology_class"] == "top:Graph"
    assert gd["category"] == "spatial"
    assert gd["label"] == "My Graph"
    assert TGraph.VertexDictionary(g, 0)["ontology_class"] == "top:Vertex"
    assert TGraph.VertexDictionary(g, 1)["ifc_class"] == "IfcWall"
    assert TGraph.VertexDictionary(g, 1)["ifc_guid"] == "abc"

    summary = TGraph.SemanticSummary(g)
    assert summary["vertices"] == 3
    assert summary["edges"] == 2
    assert "top:Graph" in summary["ontology_class_counts"]


def test_cardinality_report_and_guid_are_stable_for_simple_semantic_graph():
    g = TGraph(directed=True, dictionary={"label": "kg"})
    a = g.AddVertex({"id": "A", "label": "A"})
    b = g.AddVertex({"id": "B", "label": "B"})
    c = g.AddVertex({"id": "C", "label": "C"})
    g.AddEdge(a, b, dictionary={"predicate": "top:connectsTo"})
    g.AddEdge(a, c, dictionary={"predicate": "top:connectsTo"})
    g.AddEdge(b, c, dictionary={"predicate": "top:adjacentTo"})

    guid1 = TGraph.Guid(g)
    guid2 = TGraph.Guid(g)
    assert isinstance(guid1, str)
    assert guid1 == guid2

    report = TGraph.CardinalityReport(
        g,
        vertexKey="id",
        edgeKey="predicate",
        predicates=["top:connectsTo"],
    )
    assert isinstance(report, list)
    row_a = next(row for row in report if row["vertex"] == "A")
    assert row_a["top:connectsTo"] == 2


def test_by_ifc_path_missing_dependency_does_not_attempt_runtime_install(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("ifcopenshell"):
            raise ImportError("forced missing ifcopenshell")
        return real_import(name, *args, **kwargs)

    def forbidden_system(*args, **kwargs):
        raise AssertionError("runtime install should not be attempted")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(os, "system", forbidden_system)

    assert TGraph.ByIFCPath("dummy.ifc", silent=True) is None


def test_louvain_missing_dependency_does_not_attempt_runtime_install(monkeypatch):
    g = TGraph.ByEdgeIndexPairs(3, [(0, 1), (1, 2)], directed=False)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "igraph" or name.startswith("igraph."):
            raise ImportError("forced missing igraph")
        return real_import(name, *args, **kwargs)

    def forbidden_system(*args, **kwargs):
        raise AssertionError("runtime install should not be attempted")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(os, "system", forbidden_system)

    assert TGraph.CommunityPartition(g, algorithm="louvain", silent=True) in (None, [])
