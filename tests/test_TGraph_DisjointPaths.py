"""Unit tests for TGraph flow and mutually disjoint path algorithms.

These tests intentionally exercise pure-Python TGraph behaviour only. They avoid
TopologicCore, IFC, igraph, NetworkX, Plotly, and other optional dependencies.
"""

from __future__ import annotations

import pytest

from topologicpy.TGraph import TGraph


def _grid_graph(rows=10, columns=10):
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


def _parallel_branch_graph():
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


def _three_path_internal_bottleneck_graph():
    """Return a graph whose endpoints have degree four but vertex connectivity three."""
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

    graph.AddEdge(0, 1)
    graph.AddEdge(0, 2)
    graph.AddEdge(0, 3)
    graph.AddEdge(0, 4)

    graph.AddEdge(1, 5)
    graph.AddEdge(2, 6)
    graph.AddEdge(3, 7)

    graph.AddEdge(4, 1)

    graph.AddEdge(5, 9)
    graph.AddEdge(6, 9)
    graph.AddEdge(7, 9)
    graph.AddEdge(8, 9)

    graph.AddEdge(5, 8)

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


def test_maximum_flow_engine_and_flow_paths_find_four_grid_routes():
    graph = _grid_graph()

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
    graph = _grid_graph()

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
    graph = _grid_graph()

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
    graph = _grid_graph()

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
    graph = _grid_graph()

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
    graph = _grid_graph()

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
    graph = _parallel_branch_graph()

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
    graph = _parallel_branch_graph()

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
    graph = _parallel_branch_graph()

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
    graph = _parallel_branch_graph()

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
    graph = _parallel_branch_graph()

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
