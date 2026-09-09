"""Unit tests for TGraph connectivity, cuts, and biconnected components.

These tests exercise pure-Python TGraph behaviour only and require no optional
geometry, graph, plotting, or IFC dependencies.
"""

from __future__ import annotations

from collections import deque

import pytest

from topologicpy.TGraph import TGraph


def _grid_graph(rows=10, columns=10):
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


def _parallel_branch_graph():
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


def _reachable_without_vertices(graph, source, target, removed):
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


def _reachable_without_edges(graph, source, target, removed_edges):
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


def test_vertex_connectivity_grid_is_four():
    graph = _grid_graph()

    assert TGraph.VertexConnectivity(
        graph,
        11,
        88,
    ) == 4


def test_edge_connectivity_grid_is_four():
    graph = _grid_graph()

    assert TGraph.EdgeConnectivity(
        graph,
        11,
        88,
    ) == 4


def test_minimum_vertex_cut_grid_has_size_four_and_disconnects_endpoints():
    graph = _grid_graph()

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

    assert _reachable_without_vertices(
        graph,
        11,
        88,
        result["cut"],
    ) is False


def test_minimum_cut_default_result_is_compact():
    graph = _grid_graph()

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
    graph = _grid_graph()

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

    assert _reachable_without_vertices(
        graph,
        0,
        9,
        result["cut"],
    ) is False


def test_minimum_edge_cut_parallel_branches_has_size_three():
    graph = _parallel_branch_graph()

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

    assert _reachable_without_edges(
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
