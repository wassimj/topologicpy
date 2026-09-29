"""Unified unit tests for :mod:`topologicpy.TGraph`.

The suite combines the core TGraph, connectivity/cut, flow/disjoint-path, and
WireByPath direction tests in one file. Most tests exercise the pure-Python
TGraph data model and algorithms. Geometry imports are kept local to the
WireByPath sentinel so the rest of the suite remains lightweight at collection
time.
"""

from __future__ import annotations

import builtins
from collections import deque
import os

import pytest

from topologicpy.TGraph import TGraph


# ============================================================================
# Shared pytest configuration
# ============================================================================

@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()

# ============================================================================
# Shared graph builders and assertion helpers
# ============================================================================

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

def _connectivity_grid_graph(rows=10, columns=10):
    """Return an undirected orthogonal unit grid."""
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for y in range(rows):
        for x in range(columns):
            graph.AddVertex(
                {
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,
                }
            )

    for y in range(rows):
        for x in range(columns):
            index = y * columns + x

            if x < columns - 1:
                graph.AddEdge(index, index + 1)

            if y < rows - 1:
                graph.AddEdge(index, index + columns)

    return graph

def _three_path_internal_bottleneck_graph():
    """Return a graph with endpoint degree four but local vertex connectivity three."""
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(10):
        graph.AddVertex(
            {
                "x": float(index),
                "y": 0.0,
                "z": 0.0,
            }
        )

    # Source degree = 4.
    graph.AddEdge(0, 1)
    graph.AddEdge(0, 2)
    graph.AddEdge(0, 3)
    graph.AddEdge(0, 4)

    # Three genuinely independent channels.
    graph.AddEdge(1, 5)
    graph.AddEdge(2, 6)
    graph.AddEdge(3, 7)

    # Fourth source branch merges into channel 1.
    graph.AddEdge(4, 1)

    # Target degree = 4.
    graph.AddEdge(5, 9)
    graph.AddEdge(6, 9)
    graph.AddEdge(7, 9)
    graph.AddEdge(8, 9)

    # Fourth target branch also merges into channel 1.
    graph.AddEdge(5, 8)

    return graph

def _connectivity_parallel_branch_graph():
    """Return three internally vertex-disjoint source-target branches."""
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(5):
        graph.AddVertex(
            {
                "x": float(index),
                "y": 0.0,
                "z": 0.0,
            }
        )

    graph.AddEdge(0, 1)
    graph.AddEdge(1, 4)

    graph.AddEdge(0, 2)
    graph.AddEdge(2, 4)

    graph.AddEdge(0, 3)
    graph.AddEdge(3, 4)

    return graph

def _is_reachable_without_vertices(graph, source, target, removed):
    """Return True if target remains reachable after excluding removed vertices."""
    removed = set(removed)

    if source in removed or target in removed:
        return False

    adjacency = TGraph._UndirectedAdjacency(graph)

    visited = {source}
    queue = deque([source])

    while queue:
        u = queue.popleft()

        if u == target:
            return True

        for v in adjacency.get(u, set()):
            if v in removed or v in visited:
                continue
            visited.add(v)
            queue.append(v)

    return False

def _is_reachable_without_edges(graph, source, target, removed_edges):
    """Return True if target remains reachable after excluding removed edge indices."""
    removed_edges = set(removed_edges)

    adjacency = {
        index: []
        for index in TGraph._ActiveVertexIndices(graph)
    }

    for edge in TGraph._ActiveEdges(graph):
        edge_index = edge.get("index")

        if edge_index in removed_edges:
            continue

        u = edge.get("src")
        v = edge.get("dst")

        if u not in adjacency or v not in adjacency:
            continue

        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = {source}
    queue = deque([source])

    while queue:
        u = queue.popleft()

        if u == target:
            return True

        for v in adjacency.get(u, []):
            if v in visited:
                continue
            visited.add(v)
            queue.append(v)

    return False

def _flow_grid_graph(rows=10, columns=10):
    """Return an orthogonal unit grid with row-major stable vertex indices."""
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for y in range(rows):
        for x in range(columns):
            graph.AddVertex(
                {
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,
                    "penalty": 0.0,
                }
            )

    for y in range(rows):
        for x in range(columns):
            index = y * columns + x

            if x < columns - 1:
                graph.AddEdge(
                    index,
                    index + 1,
                    directed=False,
                    dictionary={
                        "capacity": 1.0,
                        "weight": 1.0,
                    },
                )

            if y < rows - 1:
                graph.AddEdge(
                    index,
                    index + columns,
                    directed=False,
                    dictionary={
                        "capacity": 1.0,
                        "weight": 1.0,
                    },
                )

    return graph

def _weighted_parallel_branch_graph():
    """Return three internally vertex-disjoint two-edge branches.

    Branch costs using edgeKey="weight":

        0-1-4 : 2
        0-2-4 : 4
        0-3-4 : 6
    """
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    coordinates = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (2.0, 0.0, 0.0),
    ]

    for x, y, z in coordinates:
        graph.AddVertex(
            {
                "x": x,
                "y": y,
                "z": z,
                "penalty": 0.0,
            }
        )

    graph.AddEdge(0, 1, dictionary={"weight": 1.0, "capacity": 1.0})
    graph.AddEdge(1, 4, dictionary={"weight": 1.0, "capacity": 1.0})

    graph.AddEdge(0, 2, dictionary={"weight": 2.0, "capacity": 1.0})
    graph.AddEdge(2, 4, dictionary={"weight": 2.0, "capacity": 1.0})

    graph.AddEdge(0, 3, dictionary={"weight": 3.0, "capacity": 1.0})
    graph.AddEdge(3, 4, dictionary={"weight": 3.0, "capacity": 1.0})

    return graph

def _edge_vs_vertex_disjoint_graph():
    """Return two edge-disjoint routes that share one internal vertex."""
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(7):
        graph.AddVertex(
            {
                "x": float(index),
                "y": 0.0,
                "z": 0.0,
            }
        )

    for a, b in [
        (0, 1),
        (1, 3),
        (3, 4),
        (4, 6),
        (0, 2),
        (2, 3),
        (3, 5),
        (5, 6),
    ]:
        graph.AddEdge(a, b, dictionary={"capacity": 1.0})

    return graph

def _assert_internal_vertex_disjoint(paths):
    """Assert that no pair of paths shares an internal vertex."""
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            assert set(paths[i][1:-1]).isdisjoint(paths[j][1:-1])

def _undirected_path_edges(path):
    """Return canonical undirected edge pairs for one path."""
    return {
        (min(a, b), max(a, b))
        for a, b in zip(path, path[1:])
    }

def _assert_edge_disjoint(paths):
    """Assert that no pair of paths shares a physical undirected edge."""
    edge_sets = [_undirected_path_edges(path) for path in paths]

    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            assert edge_sets[i].isdisjoint(edge_sets[j])

def _wire_vertex_xyz(vertex):
    """Return vertex coordinates without mantissa rounding."""
    from topologicpy.Vertex import Vertex

    return Vertex.Coordinates(vertex, mantissa=None)

def _coordinates_close(a, b, tol=1.0e-7):
    """Return True when two coordinate triples are equal within tolerance."""
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))

# ============================================================================
# Core graph model, mutation, adjacency, and accessors
# ============================================================================

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

# ============================================================================
# Construction, serialization, copying, and CSV round-trips
# ============================================================================

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

# ============================================================================
# Traversal, distances, routing, and path algorithms
# ============================================================================

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

# ============================================================================
# Connectivity, minimum cuts, articulation structure, and biconnected components
# ============================================================================

def test_vertex_connectivity_grid_is_four():
    graph = _connectivity_grid_graph()

    assert TGraph.VertexConnectivity(
        graph,
        11,
        88,
    ) == 4

def test_edge_connectivity_grid_is_four():
    graph = _connectivity_grid_graph()

    assert TGraph.EdgeConnectivity(
        graph,
        11,
        88,
    ) == 4

def test_minimum_vertex_cut_grid_has_size_four_and_disconnects_endpoints():
    graph = _connectivity_grid_graph()

    result = TGraph.MinimumCut(
        graph,
        11,
        88,
        cut="vertex",
    )

    assert isinstance(result, dict)
    assert result["value"] == pytest.approx(4.0)
    assert result["cutType"] == "vertex"
    assert len(result["cut"]) == 4
    assert result["cutCapacity"] == pytest.approx(4.0)
    assert result["isPureCut"] is True
    assert result["source"] == 11
    assert result["target"] == 88

    assert 11 not in result["cut"]
    assert 88 not in result["cut"]

    assert _is_reachable_without_vertices(
        graph,
        11,
        88,
        result["cut"],
    ) is False

def test_minimum_cut_default_result_is_compact():
    graph = _connectivity_grid_graph()

    result = TGraph.MinimumCut(
        graph,
        11,
        88,
        cut="vertex",
    )

    assert "sourceSideNodes" not in result
    assert "targetSideNodes" not in result
    assert "cutArcs" not in result

def test_minimum_cut_include_details_exposes_residual_diagnostics():
    graph = _connectivity_grid_graph()

    result = TGraph.MinimumCut(
        graph,
        11,
        88,
        cut="vertex",
        includeDetails=True,
    )

    assert "sourceSideNodes" in result
    assert "targetSideNodes" in result
    assert "cutArcs" in result

    assert isinstance(result["sourceSideNodes"], list)
    assert isinstance(result["targetSideNodes"], list)
    assert isinstance(result["cutArcs"], list)
    assert result["cutArcs"]

def test_internal_vertex_bottleneck_limits_connectivity_to_three():
    graph = _three_path_internal_bottleneck_graph()

    assert TGraph.Degree(graph, 0, mode="all") == 4
    assert TGraph.Degree(graph, 9, mode="all") == 4

    assert TGraph.VertexConnectivity(
        graph,
        0,
        9,
    ) == 3

    result = TGraph.MinimumCut(
        graph,
        0,
        9,
        cut="vertex",
    )

    assert result["value"] == pytest.approx(3.0)
    assert len(result["cut"]) == 3

    assert _is_reachable_without_vertices(
        graph,
        0,
        9,
        result["cut"],
    ) is False

def test_minimum_edge_cut_parallel_branches_has_size_three():
    graph = _connectivity_parallel_branch_graph()

    assert TGraph.EdgeConnectivity(
        graph,
        0,
        4,
    ) == 3

    result = TGraph.MinimumCut(
        graph,
        0,
        4,
        cut="edge",
    )

    assert isinstance(result, dict)
    assert result["value"] == pytest.approx(3.0)
    assert result["cutType"] == "edge"
    assert len(result["cut"]) == 3
    assert result["cutCapacity"] == pytest.approx(3.0)
    assert result["isPureCut"] is True

    assert _is_reachable_without_edges(
        graph,
        0,
        4,
        result["cut"],
    ) is False

def test_biconnected_components_path_returns_one_block_per_bridge():
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(4):
        graph.AddVertex({"x": float(index), "y": 0.0, "z": 0.0})

    graph.AddEdge(0, 1)
    graph.AddEdge(1, 2)
    graph.AddEdge(2, 3)

    assert TGraph.BiconnectedComponents(graph) == [
        [0, 1],
        [1, 2],
        [2, 3],
    ]

def test_biconnected_components_cycle_returns_single_block():
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(4):
        graph.AddVertex({"x": float(index), "y": 0.0, "z": 0.0})

    graph.AddEdge(0, 1)
    graph.AddEdge(1, 2)
    graph.AddEdge(2, 3)
    graph.AddEdge(3, 0)

    assert TGraph.BiconnectedComponents(graph) == [
        [0, 1, 2, 3],
    ]

def test_biconnected_components_two_cycles_share_articulation_vertex():
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(5):
        graph.AddVertex({"x": float(index), "y": 0.0, "z": 0.0})

    # First triangle.
    graph.AddEdge(0, 1)
    graph.AddEdge(1, 2)
    graph.AddEdge(2, 0)

    # Second triangle sharing articulation vertex 2.
    graph.AddEdge(2, 3)
    graph.AddEdge(3, 4)
    graph.AddEdge(4, 2)

    assert TGraph.BiconnectedComponents(graph) == [
        [0, 1, 2],
        [2, 3, 4],
    ]

    cut_vertices = TGraph.CutVertices(graph)

    assert [
        vertex["index"]
        for vertex in cut_vertices
    ] == [2]

def test_biconnected_components_parallel_edges_form_single_two_vertex_block():
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=True,
    )

    graph.AddVertex({"x": 0.0, "y": 0.0, "z": 0.0})
    graph.AddVertex({"x": 1.0, "y": 0.0, "z": 0.0})

    graph.AddEdge(0, 1)
    graph.AddEdge(0, 1)

    assert TGraph.BiconnectedComponents(graph) == [
        [0, 1],
    ]

    # Neither parallel edge is a bridge.
    assert TGraph.Bridges(graph) == []

def test_biconnected_components_includes_isolated_vertices():
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(3):
        graph.AddVertex({"x": float(index), "y": 0.0, "z": 0.0})

    graph.AddEdge(0, 1)

    assert TGraph.BiconnectedComponents(graph) == [
        [0, 1],
        [2],
    ]

def test_biconnected_components_disconnected_cycles_remain_separate():
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(6):
        graph.AddVertex({"x": float(index), "y": 0.0, "z": 0.0})

    graph.AddEdge(0, 1)
    graph.AddEdge(1, 2)
    graph.AddEdge(2, 0)

    graph.AddEdge(3, 4)
    graph.AddEdge(4, 5)
    graph.AddEdge(5, 3)

    assert TGraph.BiconnectedComponents(graph) == [
        [0, 1, 2],
        [3, 4, 5],
    ]

# ============================================================================
# Flow engines and mutually disjoint paths
# ============================================================================

def test_maximum_flow_engine_and_flow_paths_find_four_grid_routes():
    graph = _flow_grid_graph()

    source = 11
    target = 88

    network = TGraph._FlowNetwork(
        graph,
        capacityKey="capacity",
        defaultCapacity=1.0,
    )

    result = TGraph._MaximumFlowEngine(
        nodes=network["nodes"],
        arcs=network["arcs"],
        source=source,
        sink=target,
    )

    paths, flows = TGraph._FlowPaths(
        result,
        returnFlows=True,
    )

    assert result["value"] == pytest.approx(4.0)
    assert result["augmentations"] == 4
    assert len(paths) == 4
    assert flows == pytest.approx([1.0, 1.0, 1.0, 1.0])

    for path in paths:
        assert path[0] == source
        assert path[-1] == target

def test_minimum_cost_flow_engine_reroutes_previous_flow_to_reach_maximum_cardinality():
    """Regression test for residual rerouting after the first cheap augmentation."""
    nodes = ["s", "u1", "u2", "v1", "v2", "t"]

    arcs = [
        {"src": "s", "dst": "u1", "capacity": 1.0, "cost": 0.0},
        {"src": "s", "dst": "u2", "capacity": 1.0, "cost": 0.0},
        {"src": "u1", "dst": "v1", "capacity": 1.0, "cost": 0.0},
        {"src": "u1", "dst": "v2", "capacity": 1.0, "cost": 1.0},
        {"src": "u2", "dst": "v1", "capacity": 1.0, "cost": 0.0},
        {"src": "v1", "dst": "t", "capacity": 1.0, "cost": 0.0},
        {"src": "v2", "dst": "t", "capacity": 1.0, "cost": 0.0},
    ]

    result = TGraph._MinimumCostFlowEngine(
        nodes=nodes,
        arcs=arcs,
        source="s",
        sink="t",
    )

    paths = TGraph._FlowPaths(result)

    assert result["value"] == pytest.approx(2.0)
    assert result["cost"] == pytest.approx(1.0)
    assert result["augmentations"] == 2
    assert len(paths) == 2

    assert {tuple(path) for path in paths} == {
        ("s", "u1", "v2", "t"),
        ("s", "u2", "v1", "t"),
    }

def test_vertex_disjoint_flow_network_uses_non_limiting_edge_capacity():
    graph = _flow_grid_graph()

    network = TGraph._VertexDisjointFlowNetwork(
        graph,
        11,
        88,
        vertexCapacity=1.0,
        edgeCapacity=1.0,
    )

    assert network is not None
    assert network["transit_capacity"] >= 100.0

    vertex_arcs = [
        arc
        for arc in network["arcs"]
        if arc.get("kind") == "vertex"
    ]

    traversal_arcs = [
        arc
        for arc in network["arcs"]
        if arc.get("kind") == "edge"
    ]

    assert vertex_arcs
    assert traversal_arcs
    assert all(arc["capacity"] == pytest.approx(1.0) for arc in vertex_arcs)
    assert all(
        arc["capacity"] == pytest.approx(network["transit_capacity"])
        for arc in traversal_arcs
    )

def test_vertex_disjoint_private_pipeline_finds_four_grid_paths():
    graph = _flow_grid_graph()

    source = 11
    target = 88

    network = TGraph._VertexDisjointFlowNetwork(
        graph,
        source,
        target,
    )

    result = TGraph._MaximumFlowEngine(
        nodes=network["nodes"],
        arcs=network["arcs"],
        source=network["source"],
        sink=network["sink"],
    )

    split_paths = TGraph._FlowPaths(result)

    paths = TGraph._CollapseFlowPaths(
        split_paths,
        network,
    )

    assert result["value"] == pytest.approx(4.0)
    assert len(paths) == 4
    assert sorted(len(path) - 1 for path in paths) == [14, 14, 18, 18]

    _assert_internal_vertex_disjoint(paths)

def test_collapse_flow_paths_ignores_auxiliary_edge_gadget_nodes():
    network = {
        "node_to_vertex": {
            0: 0,
            ("vertex_in", 1): 1,
            ("vertex_out", 1): 1,
            2: 2,
        }
    }

    transformed = [
        [
            0,
            ("edge_in", 99),
            ("edge_out", 99),
            ("vertex_in", 1),
            ("vertex_out", 1),
            2,
        ]
    ]

    assert TGraph._CollapseFlowPaths(
        transformed,
        network,
    ) == [[0, 1, 2]]

def test_edge_disjoint_network_uses_one_shared_capacity_arc_per_undirected_edge():
    graph = TGraph(
        directed=False,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    graph.AddVertex({"x": 0.0, "y": 0.0, "z": 0.0})
    graph.AddVertex({"x": 1.0, "y": 0.0, "z": 0.0})
    graph.AddEdge(0, 1, dictionary={"weight": 7.0})

    network = TGraph._EdgeDisjointFlowNetwork(
        graph,
        0,
        1,
        edgeCosts={0: 7.0},
    )

    capacity_arcs = [
        arc
        for arc in network["arcs"]
        if arc.get("kind") == "edge_capacity"
    ]

    assert len(capacity_arcs) == 1
    assert capacity_arcs[0]["edge_index"] == 0
    assert capacity_arcs[0]["capacity"] == pytest.approx(1.0)
    assert capacity_arcs[0]["cost"] == pytest.approx(7.0)

def test_disjoint_paths_vertex_grid_returns_four_paths():
    graph = _flow_grid_graph()

    paths = TGraph.DisjointPaths(
        graph,
        11,
        88,
        disjoint="vertex",
        optimize=False,
    )

    assert len(paths) == 4

    _assert_internal_vertex_disjoint(paths)

def test_disjoint_paths_vertex_optimized_grid_is_minimum_cost_four_path_family():
    graph = _flow_grid_graph()

    paths = TGraph.DisjointPaths(
        graph,
        11,
        88,
        disjoint="vertex",
        optimize=True,
        edgeKey="Length",
    )

    lengths = sorted(
        len(path) - 1
        for path in paths
    )

    assert len(paths) == 4
    assert lengths == [14, 14, 18, 18]
    assert sum(lengths) == 64

    _assert_internal_vertex_disjoint(paths)

def test_disjoint_paths_max_paths_caps_cardinality():
    graph = _flow_grid_graph()

    paths = TGraph.DisjointPaths(
        graph,
        11,
        88,
        disjoint="vertex",
        maxPaths=2,
        optimize=True,
        edgeKey="Length",
    )

    assert len(paths) == 2

    _assert_internal_vertex_disjoint(paths)

def test_edge_disjoint_can_exceed_vertex_disjoint_cardinality():
    graph = _edge_vs_vertex_disjoint_graph()

    vertex_paths = TGraph.DisjointPaths(
        graph,
        0,
        6,
        disjoint="vertex",
        optimize=False,
    )

    edge_paths = TGraph.DisjointPaths(
        graph,
        0,
        6,
        disjoint="edge",
        optimize=False,
    )

    assert len(vertex_paths) == 1
    assert len(edge_paths) == 2

    _assert_edge_disjoint(edge_paths)
    assert all(3 in path for path in edge_paths)

def test_vertex_disjoint_paths_detect_internal_bottleneck_despite_degree_four_endpoints():
    graph = _three_path_internal_bottleneck_graph()

    assert TGraph.Degree(graph, 0, mode="all") == 4
    assert TGraph.Degree(graph, 9, mode="all") == 4

    paths = TGraph.DisjointPaths(
        graph,
        0,
        9,
        disjoint="vertex",
        optimize=False,
    )

    assert len(paths) == 3

    _assert_internal_vertex_disjoint(paths)

def test_disjoint_paths_edge_key_optimizes_complete_path_family():
    graph = _weighted_parallel_branch_graph()

    paths = TGraph.DisjointPaths(
        graph,
        0,
        4,
        disjoint="vertex",
        maxPaths=2,
        optimize=True,
        edgeKey="weight",
    )

    assert len(paths) == 2
    assert {path[1] for path in paths} == {1, 2}

    _assert_internal_vertex_disjoint(paths)

def test_disjoint_paths_vertex_key_can_change_optimized_family_without_reducing_cardinality():
    graph = _weighted_parallel_branch_graph()

    unpenalized = TGraph.DisjointPaths(
        graph,
        0,
        4,
        disjoint="vertex",
        maxPaths=2,
        optimize=True,
        edgeKey="weight",
        vertexKey="penalty",
    )

    assert {path[1] for path in unpenalized} == {1, 2}

    graph._vertices[1]["dictionary"]["penalty"] = 100.0

    penalized = TGraph.DisjointPaths(
        graph,
        0,
        4,
        disjoint="vertex",
        maxPaths=2,
        optimize=True,
        edgeKey="weight",
        vertexKey="penalty",
    )

    assert len(penalized) == 2
    assert {path[1] for path in penalized} == {2, 3}

    _assert_internal_vertex_disjoint(penalized)

def test_flow_costs_match_shortest_path_edge_and_vertex_cost_conventions():
    graph = TGraph(directed=False)

    graph.AddVertex(
        {"x": 0.0, "y": 0.0, "z": 0.0, "penalty": 100.0}
    )
    graph.AddVertex(
        {"x": 3.0, "y": 4.0, "z": 0.0, "penalty": 9.0}
    )
    graph.AddVertex(
        {"x": 6.0, "y": 4.0, "z": 0.0, "penalty": 100.0}
    )

    graph.AddEdge(
        0,
        1,
        dictionary={"weight": 7.0},
    )
    graph.AddEdge(
        1,
        2,
        dictionary={"weight": 8.0},
    )

    geometric = TGraph._FlowCosts(
        graph,
        0,
        2,
        edgeKey="Length",
        vertexKey="penalty",
    )

    hops = TGraph._FlowCosts(
        graph,
        0,
        2,
        edgeKey="hop",
        vertexKey="penalty",
    )

    weighted = TGraph._FlowCosts(
        graph,
        0,
        2,
        edgeKey="weight",
        vertexKey="penalty",
    )

    assert geometric["edge_costs"][0] == pytest.approx(5.0)
    assert geometric["edge_costs"][1] == pytest.approx(3.0)

    assert hops["edge_costs"][0] == pytest.approx(1.0)
    assert hops["edge_costs"][1] == pytest.approx(1.0)

    assert weighted["edge_costs"][0] == pytest.approx(7.0)
    assert weighted["edge_costs"][1] == pytest.approx(8.0)

    assert weighted["vertex_costs"][0] == pytest.approx(0.0)
    assert weighted["vertex_costs"][1] == pytest.approx(9.0)
    assert weighted["vertex_costs"][2] == pytest.approx(0.0)

def test_disjoint_paths_rejects_negative_optimization_costs():
    graph = _weighted_parallel_branch_graph()

    graph._edges[0]["dictionary"]["weight"] = -1.0

    assert TGraph.DisjointPaths(
        graph,
        0,
        4,
        disjoint="vertex",
        optimize=True,
        edgeKey="weight",
        silent=True,
    ) == []

def test_disjoint_paths_respects_directed_edges():
    graph = TGraph(
        directed=True,
        allowSelfLoops=False,
        allowParallelEdges=False,
    )

    for index in range(4):
        graph.AddVertex(
            {
                "x": float(index),
                "y": 0.0,
                "z": 0.0,
            }
        )

    graph.AddEdge(0, 1, directed=True)
    graph.AddEdge(1, 3, directed=True)
    graph.AddEdge(0, 2, directed=True)
    graph.AddEdge(2, 3, directed=True)

    forward = TGraph.DisjointPaths(
        graph,
        0,
        3,
        disjoint="vertex",
    )

    reverse = TGraph.DisjointPaths(
        graph,
        3,
        0,
        disjoint="vertex",
    )

    assert len(forward) == 2
    assert reverse == []

    _assert_internal_vertex_disjoint(forward)

@pytest.mark.parametrize(
    "alias",
    [
        "vertex",
        "vertices",
        "node",
        "nodes",
        "vertex-disjoint",
        "vertex_disjoint",
    ],
)
def test_disjoint_paths_vertex_aliases(alias):
    graph = _weighted_parallel_branch_graph()

    paths = TGraph.DisjointPaths(
        graph,
        0,
        4,
        disjoint=alias,
        maxPaths=2,
    )

    assert len(paths) == 2
    _assert_internal_vertex_disjoint(paths)

@pytest.mark.parametrize(
    "alias",
    [
        "edge",
        "edges",
        "edge-disjoint",
        "edge_disjoint",
    ],
)
def test_disjoint_paths_edge_aliases(alias):
    graph = _edge_vs_vertex_disjoint_graph()

    paths = TGraph.DisjointPaths(
        graph,
        0,
        6,
        disjoint=alias,
    )

    assert len(paths) == 2
    _assert_edge_disjoint(paths)

def test_disjoint_paths_invalid_mode_and_non_positive_max_paths_return_empty():
    graph = _weighted_parallel_branch_graph()

    assert TGraph.DisjointPaths(
        graph,
        0,
        4,
        disjoint="invalid",
        silent=True,
    ) == []

    assert TGraph.DisjointPaths(
        graph,
        0,
        4,
        maxPaths=0,
        silent=True,
    ) == []

    assert TGraph.DisjointPaths(
        graph,
        0,
        4,
        maxPaths=-1,
        silent=True,
    ) == []

# ============================================================================
# Graph analytics, transforms, subgraphs, and compiled acceleration
# ============================================================================

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

# ============================================================================
# Ontology, semantic metadata, and optional-dependency guards
# ============================================================================

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

# ============================================================================
# Geometry bridge: ShortestPath -> WireByPath direction preservation
# ============================================================================

def test_shortest_path_wire_by_path_preserves_direction_in_both_traversals():
    from topologicpy.Edge import Edge
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    from topologicpy.Wire import Wire
    vertices = [Vertex.ByCoordinates(float(i), 0.0, 0.0) for i in range(4)]
    # Deliberately store every representation opposite to the 0 -> 3 traversal.
    edges = [
        Edge.ByStartVertexEndVertex(vertices[1], vertices[0], silent=True),
        Edge.ByStartVertexEndVertex(vertices[2], vertices[1], silent=True),
        Edge.ByStartVertexEndVertex(vertices[3], vertices[2], silent=True),
    ]
    graph = TGraph.ByVerticesEdges(vertices=vertices, edges=edges, directed=False, silent=True)
    assert isinstance(graph, TGraph)

    path_forward = TGraph.ShortestPath(graph, 0, 3, mode="all", silent=True)
    assert path_forward == [0, 1, 2, 3]
    wire_forward = TGraph.WireByPath(graph, path_forward, silent=True)
    assert Topology.IsInstance(wire_forward, "Wire")
    assert _coordinates_close(_wire_vertex_xyz(Wire.StartVertex(wire_forward, silent=True)), _wire_vertex_xyz(vertices[0]))
    assert _coordinates_close(_wire_vertex_xyz(Wire.EndVertex(wire_forward, silent=True)), _wire_vertex_xyz(vertices[3]))

    path_reverse = TGraph.ShortestPath(graph, 3, 0, mode="all", silent=True)
    assert path_reverse == [3, 2, 1, 0]
    wire_reverse = TGraph.WireByPath(graph, path_reverse, silent=True)
    assert Topology.IsInstance(wire_reverse, "Wire")
    assert _coordinates_close(_wire_vertex_xyz(Wire.StartVertex(wire_reverse, silent=True)), _wire_vertex_xyz(vertices[3]))
    assert _coordinates_close(_wire_vertex_xyz(Wire.EndVertex(wire_reverse, silent=True)), _wire_vertex_xyz(vertices[0]))
