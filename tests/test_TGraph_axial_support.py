"""Regression coverage for axial graph construction and centrality support."""
import networkx as nx
import pytest

from topologicpy.TGraph import TGraph


def make_graph(points, pairs, weights=None, weight_key="cost"):
    graph = TGraph(allowSelfLoops=False)
    for x, y, z in points:
        graph.AddVertex({"x": x, "y": y, "z": z})
    for i, (u, v) in enumerate(pairs):
        graph.AddEdge(u, v, dictionary={weight_key: weights[i]} if weights else {})
    return graph


@pytest.mark.parametrize("weight_key", [None, "cost", "distance_penalty"])
@pytest.mark.parametrize("radius", [None, 0, 1, 2, 3.5, 100])
def test_radius_centrality_matches_enumerated_shortest_paths(weight_key, radius):
    points = [(i, 0, 0) for i in range(6)]
    pairs = [(0, 1), (1, 2), (0, 3), (3, 2), (2, 4)]
    weights = [1, 2, 1, 2, 1]
    graph = make_graph(points, pairs, weights, weight_key or "cost")
    reference = nx.Graph()
    reference.add_nodes_from(range(6))
    for i, (u, v) in enumerate(pairs):
        reference.add_edge(u, v, cost=weights[i] if weight_key else 1)
    expected_bc = [0.0] * 6
    expected_edges = [0.0] * len(pairs)
    pair_lookup = {frozenset(pair): i for i, pair in enumerate(pairs)}
    for u in range(6):
        for v in range(u + 1, 6):
            if not nx.has_path(reference, u, v):
                continue
            distance = nx.shortest_path_length(reference, u, v, weight="cost")
            if radius is not None and distance > radius:
                continue
            paths = list(nx.all_shortest_paths(reference, u, v, weight="cost"))
            for path in paths:
                for interior in path[1:-1]:
                    expected_bc[interior] += 1 / len(paths)
                for a, b in zip(path, path[1:]):
                    expected_edges[pair_lookup[frozenset((a, b))]] += 1 / len(paths)
    expected_cc = []
    for u in range(6):
        distances = nx.single_source_dijkstra_path_length(reference, u, cutoff=radius, weight="cost")
        count = len(distances) - 1
        total = sum(distances.values())
        expected_cc.append(count / total if total else 0)
    assert TGraph.BetweennessCentrality(graph, weightKey=weight_key, radius=radius,
                                      nxCompatible=False, colorKey=None) == pytest.approx(expected_bc, abs=1e-6)
    assert TGraph.BetweennessCentrality(graph, weightKey=weight_key, radius=radius,
                                      nxCompatible=False, colorKey=None, useEdges=True) == pytest.approx(expected_edges, abs=1e-6)
    assert TGraph.ClosenessCentrality(graph, weightKey=weight_key, radius=radius,
                                    nxCompatible=False, colorKey=None) == pytest.approx(expected_cc, abs=1e-6)


@pytest.mark.parametrize("method", [TGraph.ClosenessCentrality, TGraph.BetweennessCentrality])
@pytest.mark.parametrize("radius", [-1, float("inf"), float("nan"), "3", True])
def test_invalid_radius_is_rejected(method, radius):
    graph = make_graph([(0, 0, 0)], [])
    assert method(graph, radius=radius, silent=True) is None


def test_choice_and_integration_wrappers_and_radius():
    graph = make_graph([(i, 0, 0) for i in range(4)], [(0, 1), (1, 2), (2, 3)])
    assert TGraph.Choice(graph, normalize=False) == [0, 2, 2, 0]
    assert TGraph.Choice(graph, radius=1) == [0, 0, 0, 0]
    assert TGraph.Integration(graph, normalize=False, radius=1) == pytest.approx([1/3, 2/3, 2/3, 1/3], abs=1e-6)


@pytest.mark.parametrize("points", [
    [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)],
    [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2)],
])
def test_angular_wrappers_compute_segment_values_and_preserve_metadata(points):
    graph = make_graph(points, [(0, 1), (1, 2), (2, 3)])
    graph._edges[0]["dictionary"]["u_edge_id"] = "user-value"
    values = TGraph.AngularIntegration(graph, normalize=False, radius=0.5)
    assert values[0] > 0 and values[1] > 0 and values[2] == 0
    assert graph._edges[0]["dictionary"]["u_edge_id"] == "user-value"
    assert TGraph.AngularChoice(graph, normalize=False) == [0, 1, 0]
    assert TGraph.AngularBetweenness(graph, normalize=False) == [0, 1, 0]
    assert TGraph.AngularChoice(graph, normalize=False, radius=0.5) == [0, 0, 0]
    assert TGraph.AngularConnectivity(graph) == [1, 2, 1]
    assert [e["dictionary"]["connectivity"] for e in graph._edges] == [1, 2, 1]
    assert all("integration" not in v["dictionary"] for v in graph._vertices)


def test_angular_radius_and_stable_indices_after_edge_removal():
    graph = make_graph([(i, 0, 0) for i in range(5)], [(0, 1), (1, 2), (2, 3), (3, 4)])
    graph._edges[0]["active"] = False
    assert TGraph.AngularChoice(graph, normalize=False) == [0, 1, 0]
    assert TGraph.AngularConnectivity(graph) == [1, 2, 1]
    assert len(TGraph.AngularIntegration(graph)) == 3


@pytest.mark.parametrize("use_bvh", [True, False])
def test_spatial_relationships_preserve_original_lines_and_default_points(use_bvh):
    from topologicpy.Edge import Edge
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    lines = [
        Edge.ByVertices(Vertex.ByCoordinates(-1, 0, 0), Vertex.ByCoordinates(1, 0, 0)),
        Edge.ByVertices(Vertex.ByCoordinates(0, -1, 0), Vertex.ByCoordinates(0, 1, 0)),
        Edge.ByVertices(Vertex.ByCoordinates(0, -1, 2), Vertex.ByCoordinates(0, 1, 2)),
    ]
    kept = TGraph.BySpatialRelationships(lines, include=["intersects"], preserveRepresentations=True, useBVH=use_bvh)
    default = TGraph.BySpatialRelationships(lines, include=["intersects"], useBVH=use_bvh)
    assert TGraph.Order(kept) == 3
    assert TGraph.Size(kept) == 1
    assert {(e["src"], e["dst"]) for e in kept._edges} == {(0, 1)}
    assert all(v["representation"] is line for v, line in zip(kept._vertices, lines))
    assert all(Topology.IsInstance(v["representation"], "Vertex") for v in default._vertices)
    assert TGraph.Coordinates(kept, 2) == [0.0, 0.0, 2.0]


@pytest.mark.parametrize("method", [TGraph.ClosenessCentrality, TGraph.BetweennessCentrality])
def test_angular_results_do_not_depend_on_dictionary_key(method):
    graph = make_graph([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)], [(0, 1), (1, 2), (2, 3)])
    expected = method(graph, useEdges=True, angular=True, normalize=False,
                      nxCompatible=False, colorKey=None)
    actual = method(graph, useEdges=True, angular=True, normalize=False,
                    nxCompatible=False, key=None, colorKey=None)
    assert actual == expected


@pytest.mark.parametrize("method", [TGraph.AngularChoice, TGraph.AngularIntegration, TGraph.AngularConnectivity])
def test_empty_angular_graph(method):
    assert method(TGraph()) == []


def test_global_normalization_matches_networkx():
    graph = make_graph([(i, 0, 0) for i in range(5)], [(0, 1), (1, 2), (2, 3)])
    reference = nx.Graph([(0, 1), (1, 2), (2, 3)])
    reference.add_node(4)
    assert TGraph.BetweennessCentrality(graph, colorKey=None) == pytest.approx(list(nx.betweenness_centrality(reference).values()), abs=1e-6)
    assert TGraph.ClosenessCentrality(graph, colorKey=None) == pytest.approx(list(nx.closeness_centrality(reference).values()), abs=1e-6)
