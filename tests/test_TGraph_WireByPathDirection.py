"""Sentinel: ShortestPath -> WireByPath must preserve source-to-target direction."""

from topologicpy.Edge import Edge
from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


def _xyz(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _close(a, b, tol=1.0e-7):
    return all(abs(float(a[i])-float(b[i])) <= tol for i in range(3))


def test_shortestpath_wirebypath_preserves_direction_in_both_traversals():
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
    assert _close(_xyz(Wire.StartVertex(wire_forward, silent=True)), _xyz(vertices[0]))
    assert _close(_xyz(Wire.EndVertex(wire_forward, silent=True)), _xyz(vertices[3]))

    path_reverse = TGraph.ShortestPath(graph, 3, 0, mode="all", silent=True)
    assert path_reverse == [3, 2, 1, 0]
    wire_reverse = TGraph.WireByPath(graph, path_reverse, silent=True)
    assert Topology.IsInstance(wire_reverse, "Wire")
    assert _close(_xyz(Wire.StartVertex(wire_reverse, silent=True)), _xyz(vertices[3]))
    assert _close(_xyz(Wire.EndVertex(wire_reverse, silent=True)), _xyz(vertices[0]))
