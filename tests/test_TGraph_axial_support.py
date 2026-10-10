"""Regression coverage for axial graph construction and centrality support."""
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
    adjacency = [[] for _ in points]
    for i, (u, v) in enumerate(pairs):
        cost = weights[i] if weight_key else 1
        adjacency[u].append((v, cost))
        adjacency[v].append((u, cost))

    # Exhaustively enumerate simple paths: an independent oracle for this tiny
    # positive-weight graph, including tied routes and an isolated vertex.
    def shortest_paths(source, target):
        candidates = []

        def visit(path, cost):
            if path[-1] == target:
                candidates.append((cost, path))
                return
            for neighbor, weight in adjacency[path[-1]]:
                if neighbor not in path:
                    visit(path + [neighbor], cost + weight)

        visit([source], 0)
        if not candidates:
            return None, []
        distance = min(cost for cost, _ in candidates)
        return distance, [path for cost, path in candidates if cost == distance]

    distances = [{} for _ in points]
    expected_bc = [0.0] * 6
    expected_edges = [0.0] * len(pairs)
    pair_lookup = {frozenset(pair): i for i, pair in enumerate(pairs)}
    for u in range(6):
        for v in range(u + 1, 6):
            distance, paths = shortest_paths(u, v)
            if distance is None or (radius is not None and distance > radius):
                continue
            distances[u][v] = distance
            distances[v][u] = distance
            for path in paths:
                for interior in path[1:-1]:
                    expected_bc[interior] += 1 / len(paths)
                for a, b in zip(path, path[1:]):
                    expected_edges[pair_lookup[frozenset((a, b))]] += 1 / len(paths)
    expected_cc = []
    for u in range(6):
        count = len(distances[u])
        total = sum(distances[u].values())
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
    assert values == [-1, -1, 0]
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


def test_global_normalization_on_disconnected_path():
    graph = make_graph([(i, 0, 0) for i in range(5)], [(0, 1), (1, 2), (2, 3)])
    # Two interior vertices each serve two unordered pairs; normalization is
    # 2 / ((5 - 1) * (5 - 2)). Closeness includes the reachable fraction 3/4.
    assert TGraph.BetweennessCentrality(graph, colorKey=None) == pytest.approx(
        [0, 1/3, 1/3, 0, 0], abs=1e-6)
    assert TGraph.ClosenessCentrality(graph, colorKey=None) == pytest.approx(
        [3/8, 9/16, 9/16, 3/8, 0], abs=1e-6)


def directed_path():
    g = TGraph(directed=True)
    for i in range(3):
        g.AddVertex({"x": i, "y": 0, "z": 0})
    g.AddEdge(0, 1, dictionary={"cost": 1})
    g.AddEdge(1, 2, dictionary={"cost": 1})
    return g


@pytest.mark.parametrize("weight_key", [None, "cost"])
def test_directed_centrality_respects_one_way_routes(weight_key):
    g = directed_path()
    assert TGraph.ClosenessCentrality(g, weightKey=weight_key) == pytest.approx([2/3, 1/2, 0], abs=1e-6)
    assert TGraph.ClosenessCentrality(g, weightKey=weight_key, mode="in") == pytest.approx([0, 1/2, 2/3], abs=1e-6)
    assert TGraph.ClosenessCentrality(g, weightKey=weight_key, mode="all") == pytest.approx([2/3, 1, 2/3], abs=1e-6)
    assert TGraph.BetweennessCentrality(g, weightKey=weight_key) == [0, 0.5, 0]
    assert TGraph.BetweennessCentrality(g, weightKey=weight_key, nxCompatible=False) == [0, 1, 0]
    assert TGraph.BetweennessCentrality(g, weightKey=weight_key, useEdges=True) == pytest.approx([1/3, 1/3], abs=1e-6)
    assert TGraph.BetweennessCentrality(g, weightKey=weight_key, nxCompatible=False, useEdges=True) == [2, 2]
    assert TGraph.BetweennessCentrality(g, weightKey=weight_key, radius=1) == [0, 0, 0]
    assert TGraph.ClosenessCentrality(g, weightKey=weight_key, radius=1) == [0.5, 0.5, 0]


def test_line_graph_direction_override_and_legal_transitions():
    g = directed_path()
    assert TGraph.Size(TGraph.LineGraph(g)) == 1
    assert TGraph.Size(TGraph.LineGraph(g, directed=False)) == 1
    # Both edges terminate at vertex 1: no legal directed transition.
    g._edges[1]["src"], g._edges[1]["dst"] = 2, 1
    assert TGraph.Size(TGraph.LineGraph(g)) == 0
    assert TGraph.AngularChoice(g, normalize=False) == [0, 0]
    assert TGraph.AngularIntegration(g, normalize=False) == [0, 0]
    assert TGraph.Size(TGraph.LineGraph(g, directed=False)) == 1
    u = make_graph([(i, 0, 0) for i in range(3)], [(0, 1), (1, 2)])
    assert TGraph.LineGraph(u, directed=True)._directed
    assert [(e["src"], e["dst"]) for e in TGraph.LineGraph(u, directed=True)._edges] == [(0, 1)]


def test_straight_segment_zero_cost_radius_and_undefined_integration():
    from topologicpy.Edge import Edge
    from topologicpy.Vertex import Vertex
    edges = [Edge.ByVertices(Vertex.ByCoordinates(i, 0, 0), Vertex.ByCoordinates(i+1, 0, 0)) for i in range(3)]
    g = TGraph.SegmentGraph(edges)
    assert [e["dictionary"]["angular_weight"] for e in g._edges] == [0, 0]
    assert TGraph.AngularIntegration(g, normalize=False, radius=0) == [-1, -1, -1]
    assert TGraph.AngularIntegration(g, normalize=True, radius=0) == [-1, -1, -1]
    assert all(v["dictionary"]["cc_color"] == "#7f7f7f" for v in g._vertices)
    assert TGraph.AngularChoice(g, normalize=False, radius=0) == [0, 1, 0]
    assert TGraph.AngularChoice(g, normalize=False) == [0, 1, 0]


def test_zero_cost_cycle_uses_finite_minimum_hop_routes():
    g = make_graph([(i, 0, 0) for i in range(4)], [(0, 1), (1, 2), (2, 0), (2, 3)], [0, 0, 0, 1])
    # Each of 0 and 1 reaches 3 through 2; zero-cost triangle routes prefer
    # the direct edge, so a cycle never contributes an extra route.
    assert TGraph.BetweennessCentrality(g, weightKey="cost", nxCompatible=False) == [0, 0, 2, 0]
    assert TGraph.ClosenessCentrality(g, weightKey="cost", nxCompatible=False) == [3, 3, 3, 1]
    assert TGraph.ClosenessCentrality(g, weightKey="cost", radius=0) == [-1, -1, -1, 0]


def test_directed_angular_turn_cost_and_edge_dictionary_transfer():
    g = TGraph(directed=True)
    for x, y in [(0, 0), (1, 0), (2, 0), (2, 1)]:
        g.AddVertex({"x": x, "y": y, "z": 0})
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        g.AddEdge(u, v)
    assert TGraph.AngularChoice(g, normalize=False) == [0, 1, 0]
    assert TGraph.AngularIntegration(g, normalize=False) == [2, 0.5, 0]
    assert [e["dictionary"]["integration"] for e in g._edges] == [2, 0.5, 0]
    assert TGraph.AngularIntegration(g, normalize=False, radius=0) == [-1, 0, 0]
    assert TGraph.ClosenessCentrality(g, useEdges=True, angular=True, nxCompatible=False, mode="in") == [0, -1, 1]


@pytest.mark.parametrize("method", [TGraph.ClosenessCentrality, TGraph.BetweennessCentrality])
@pytest.mark.parametrize("cost", [-1, float("inf"), float("nan")])
def test_invalid_edge_costs_are_not_silently_replaced(method, cost):
    g = make_graph([(0, 0, 0), (1, 0, 0)], [(0, 1)], [cost])
    assert method(g, weightKey="cost", silent=True) is None


def test_mixed_edge_directions_are_respected():
    g = TGraph(directed=True)
    for i in range(3):
        g.AddVertex({"x": i, "y": 0, "z": 0})
    g.AddEdge(0, 1)
    g.AddEdge(1, 2, directed=False)
    assert TGraph.ClosenessCentrality(g) == pytest.approx([2/3, 0.5, 0.5], abs=1e-6)
    assert TGraph.ClosenessCentrality(g, mode="in") == pytest.approx([0, 1, 2/3], abs=1e-6)
    assert TGraph.BetweennessCentrality(g, nxCompatible=False) == [0, 1, 0]
    assert [(e["src"], e["dst"]) for e in TGraph.LineGraph(g)._edges] == [(0, 1)]
    # A directed edge overrides an undirected graph's default too.
    h = TGraph()
    for i in range(3):
        h.AddVertex({"x": i, "y": 0, "z": 0})
    h.AddEdge(0, 1, directed=True)
    h.AddEdge(1, 2)
    assert TGraph.ClosenessCentrality(h) == TGraph.ClosenessCentrality(g)
    assert TGraph.BetweennessCentrality(h) == TGraph.BetweennessCentrality(g)
    assert TGraph.LineGraph(h)._directed


@pytest.mark.parametrize("zero_cost", [False, True])
@pytest.mark.parametrize("radius", [None, 0, 1, 2, 5])
def test_directed_weighted_routes_match_independent_enumeration(zero_cost, radius):
    g = TGraph(directed=True, allowParallelEdges=True)
    for i in range(5):
        g.AddVertex({"x": i, "y": 0, "z": 0})
    # Includes parallel alternatives, a cycle, equal-cost routes of different
    # hop counts, and an unreachable source. The oracle enumerates simple paths.
    records = [(0, 1, 0 if zero_cost else 1), (0, 1, 0 if zero_cost else 1),
               (1, 2, 1), (0, 2, 1 if zero_cost else 2), (2, 1, 0 if zero_cost else 1),
               (2, 3, 1)]
    for u, v, cost in records:
        g.AddEdge(u, v, dictionary={"cost": cost})
    vertex_values = [0.0] * 5
    edge_values = [0.0] * len(records)
    closeness = []
    for source in range(5):
        reached_costs = []
        for target in range(5):
            if source == target:
                continue
            candidates = []
            def visit(vertices, edges, cost):
                if vertices[-1] == target:
                    candidates.append((cost, vertices, edges))
                    return
                for ei, (u, v, weight) in enumerate(records):
                    if u == vertices[-1] and v not in vertices:
                        visit(vertices + [v], edges + [ei], cost + weight)
            visit([source], [], 0)
            if not candidates:
                continue
            distance = min(c for c, _, _ in candidates)
            if radius is not None and distance > radius:
                continue
            reached_costs.append(distance)
            routes = [r for r in candidates if r[0] == distance]
            if zero_cost:
                min_hops = min(len(edges) for _, _, edges in routes)
                routes = [r for r in routes if len(r[2]) == min_hops]
            for _, vertices, edges in routes:
                for vertex in vertices[1:-1]:
                    vertex_values[vertex] += 1/len(routes)
                for edge in edges:
                    edge_values[edge] += 1/len(routes)
        total = sum(reached_costs)
        closeness.append(len(reached_costs)/total if total else (-1 if reached_costs else 0))
    assert TGraph.BetweennessCentrality(g, weightKey="cost", nxCompatible=False, radius=radius) == pytest.approx(vertex_values, abs=1e-6)
    assert TGraph.BetweennessCentrality(g, weightKey="cost", nxCompatible=False, useEdges=True, radius=radius) == pytest.approx(edge_values, abs=1e-6)
    assert TGraph.ClosenessCentrality(g, weightKey="cost", nxCompatible=False, radius=radius) == pytest.approx(closeness, abs=1e-6)
