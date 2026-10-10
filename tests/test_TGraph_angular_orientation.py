"""Physical traversal checks and original Depthmap endpoint/turn references."""
import json
import math
from pathlib import Path

import pytest
from topologicpy.TGraph import TGraph


def graph_by_segments(endpoints):
    graph = TGraph(allowSelfLoops=False)
    vertices = []
    for p, q in endpoints:
        ends = []
        for point in (p, q):
            index = next((i for i, other in enumerate(vertices) if math.dist(point, other) <= 1e-5), None)
            if index is None:
                index = len(vertices)
                vertices.append(point)
                graph.AddVertex(dict(zip(("x", "y", "z"), point)))
            ends.append(index)
        graph.AddEdge(*ends)
    return graph


def test_barnsbury_original_depthmap_endpoint_directions_and_turn_costs():
    fixture = json.loads((Path(__file__).parent / "fixtures/depthmap/barnsbury_segment_connections.json").read_text())
    graph = graph_by_segments(fixture["endpoints"])
    records, states, starts, adjacency, directed = TGraph._AngularStateData(graph)
    actual = {(states[s][0], states[s][1], states[t][0], states[t][1]): cost
              for s, neighbors in enumerate(adjacency) for t, cost in neighbors}
    expected = {tuple(row[:4]): row[4] for row in fixture["transitions"]}
    assert len(records) == 178 and len(expected) == 746
    assert actual.keys() == expected.keys()
    for transition in expected:
        assert actual[transition] == pytest.approx(expected[transition], rel=2e-6, abs=2e-6)
    assert TGraph.AngularConnectivity(graph, method="depthmap", mantissa=None) == pytest.approx(
        fixture["angular_connectivity"], rel=2e-6, abs=2e-6)
    for i in range(len(records)):
        neighbors = [cost for state in starts[i] for _, cost in adjacency[state]]
        assert len(neighbors) == fixture["connectivity"][i]
        assert sum(neighbors) == pytest.approx(fixture["angular_connectivity"][i], rel=2e-6, abs=2e-6)


def test_shared_junction_does_not_allow_departure_through_entry_endpoint():
    endpoints = [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (-1, 0, 0)),
                 ((0, 0, 0), (1, 0.1, 0))]
    graph = graph_by_segments(endpoints)
    turn = math.atan(0.1)/(math.pi/2)
    assert TGraph.AngularChoice(graph, normalize=False) == [0, 0, 0]
    assert TGraph.ClosenessCentrality(graph, useEdges=True, angular=True, nxCompatible=False) == pytest.approx(
        [2/(2-turn), 2/turn, 1], abs=1e-6)
    # A small radius cannot reach segment 2 from segment 0 via segment 1.
    assert TGraph.AngularIntegration(graph, normalize=False, radius=0.1)[0] == -1
    from topologicpy.Edge import Edge
    from topologicpy.Vertex import Vertex
    edges = [Edge.ByVertices(Vertex.ByCoordinates(*p), Vertex.ByCoordinates(*q)) for p, q in endpoints]
    segment_graph = TGraph.SegmentGraph(edges)
    assert TGraph.AngularChoice(segment_graph, normalize=False) == [0, 0, 0]
    assert TGraph.AngularConnectivity(segment_graph, method="depthmap") == TGraph.AngularConnectivity(graph, method="depthmap")
    assert TGraph.AngularIntegration(segment_graph, normalize=False) == TGraph.AngularIntegration(graph, normalize=False)
    # Plain weighted graph traversal deliberately lacks physical orientation.
    assert TGraph.BetweennessCentrality(segment_graph, weightKey="angular_weight", nxCompatible=False) == [0, 1, 0]


@pytest.mark.parametrize("radius", [None, 0, 0.5, 1, 2, 4])
@pytest.mark.parametrize("endpoints", [
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0)), ((2, 0, 0), (2, 1, 0))],
    [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0)),
     ((1, 1, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 0))],
    [((0, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 0, 2)),
     ((0, 0, 1), (1, 0, 1)), ((1, 0, 1), (1, 1, 1))],
])
def test_angular_results_match_independent_physical_route_enumeration(endpoints, radius):
    # Enumerate traversals directly from endpoint coordinates. Each intermediate
    # segment must be crossed to its far endpoint; no line-graph adjacency is used.
    graph = graph_by_segments(endpoints)
    n = len(endpoints)
    choices, integrations = [0.0]*n, []
    for source in range(n):
        distances = []
        for target in range(n):
            if source == target:
                continue
            candidates = []
            def visit(path, entry, exit, cost):
                if path[-1] == target:
                    candidates.append((cost, path))
                    return
                incoming = [b-a for a, b in zip(entry, exit)]
                for segment, (p, q) in enumerate(endpoints):
                    if segment in path:
                        continue
                    for start, end in ((p, q), (q, p)):
                        if tuple(start) != tuple(exit):
                            continue
                        outgoing = [b-a for a, b in zip(start, end)]
                        dot = sum(a*b for a, b in zip(incoming, outgoing))
                        length = math.sqrt(sum(a*a for a in incoming)*sum(b*b for b in outgoing))
                        turn = math.acos(max(-1, min(1, dot/length)))/(math.pi/2)
                        visit(path+[segment], start, end, cost+turn)
            p, q = endpoints[source]
            visit([source], p, q, 0)
            visit([source], q, p, 0)
            if not candidates:
                continue
            best = min(cost for cost, _ in candidates)
            if radius is not None and best > radius:
                continue
            distances.append(best)
            paths = [path for cost, path in candidates if abs(cost-best) < 1e-10]
            # These networks' equal-cost routes also tie in hop count.
            for path in paths:
                for interior in path[1:-1]:
                    choices[interior] += 0.5/len(paths)
        total = sum(distances)
        integrations.append(len(distances)/total if total else (-1 if distances else 0))
    assert TGraph.AngularChoice(graph, normalize=False, radius=radius) == pytest.approx(choices, abs=1e-6)
    assert TGraph.ClosenessCentrality(graph, useEdges=True, angular=True, nxCompatible=False, radius=radius) == pytest.approx(integrations, abs=1e-6)


def test_angular_connectivity_filters_removed_relationships_and_inactive_segments():
    from topologicpy.Edge import Edge
    from topologicpy.Vertex import Vertex
    edges = [Edge.ByVertices(Vertex.ByCoordinates(i, 0, 0), Vertex.ByCoordinates(i+1, 0, 0)) for i in range(3)]
    graph = TGraph.SegmentGraph(edges)
    graph.RemoveEdge(0)
    assert TGraph.AngularChoice(graph, normalize=False) == [0, 0, 0]
    assert TGraph.AngularIntegration(graph, normalize=False) == [0, -1, -1]
    graph._vertices[1]["active"] = False
    assert TGraph.AngularIntegration(graph, normalize=False) == [0, 0]


def test_angular_results_survive_copy_and_preserve_originator_transfer():
    from topologicpy.Edge import Edge
    from topologicpy.Vertex import Vertex
    from topologicpy.Topology import Topology
    from topologicpy.Dictionary import Dictionary
    edges = [Edge.ByVertices(Vertex.ByCoordinates(i, 0, 0), Vertex.ByCoordinates(i+1, 0, 0)) for i in range(3)]
    graph = TGraph.Copy(TGraph.SegmentGraph(edges))
    values = TGraph.AngularChoice(graph, normalize=False, key="angular_choice")
    assert values == [0, 1, 0]
    TGraph.TransferDictionariesToOriginators(graph, keys=["angular_choice"])
    assert [Dictionary.ValueAtKey(Topology.Dictionary(edge), "angular_choice") for edge in TGraph.Originators(graph)] == values


def test_connectivity_mode_validation_and_result_storage():
    graph = graph_by_segments([((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0)),
                               ((2, 0, 0), (2, 1, 0))])
    assert TGraph.AngularConnectivity(graph, method="depthmap") == [0, 1, 1]
    assert TGraph.AngularConnectivity(graph) == [1, 2, 1]
    assert TGraph.AngularConnectivity(graph, method="depthmap", normalize=True) == [0, 1, 1]
    assert TGraph.AngularConnectivity(graph, method="depthmap", colorKey="color", colorScale="syntax") == [0, 1, 1]
    assert all("color" in e["dictionary"] for e in graph._edges)
    assert TGraph.AngularConnectivity(graph, method="unknown", silent=True) is None
    assert TGraph.AngularConnectivity(graph, method="depthmap", mode="unknown", silent=True) is None


def test_orientation_validation_and_non_linear_colours():
    graph = graph_by_segments([((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0)),
                               ((2, 0, 0), (2, 1, 0)), ((2, 1, 0), (3, 1, 0))])
    values = TGraph.BetweennessCentrality(graph, useEdges=True, angular=True, colorScaleMode="sqrt", colorScale="syntax")
    assert len(values) == 4 and all("bc_color" in e["dictionary"] for e in graph._edges)
    graph._vertices[0]["dictionary"]["x"] = float("nan")
    assert TGraph.AngularIntegration(graph, silent=True) is None
