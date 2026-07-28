# Auto-generated PythonOCC backend parity test
# Copied from test_Graph.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Graph.

These tests focus on small deterministic graph operations and on regressions
fixed in the corrected Graph.py: no import-time installation side effects,
zero-coordinate AddVertexByData, AddEdgeByIndex edge construction, AdjacentEdges
filtering, and Complete graph construction.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

from topologicpy.Dictionary import Dictionary
from topologicpy.Edge import Edge
from topologicpy.Graph import Graph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _vertex(x, y, z=0, **dictionary):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    if dictionary:
        d = Dictionary.ByPythonDictionary(dictionary, silent=True)
        v = Topology.SetDictionary(v, d, silent=True)
    return v


def _edge(start, end, **dictionary):
    try:
        e = Edge.ByStartVertexEndVertex(start, end, tolerance=0.0001, silent=True)
    except Exception:
        e = None
    if e is None:
        try:
            e = Edge.ByVertices([start, end], tolerance=0.0001, silent=True)
        except TypeError:
            e = Edge.ByVertices([start, end])
    assert Topology.IsInstance(e, "Edge")
    if dictionary:
        d = Dictionary.ByPythonDictionary(dictionary, silent=True)
        e = Topology.SetDictionary(e, d, silent=True)
    return e


def _chain_graph():
    vertices = [
        _vertex(0, 0, 0, name="A"),
        _vertex(1, 0, 0, name="B"),
        _vertex(2, 0, 0, name="C"),
    ]
    edges = [_edge(vertices[0], vertices[1]), _edge(vertices[1], vertices[2])]
    graph = Graph.ByVerticesEdges(vertices, edges, index=True, ontology=False, silent=True)
    assert Topology.IsInstance(graph, "Graph")
    return graph, vertices, edges


def _python_dict(topology):
    return Dictionary.PythonDictionary(Topology.Dictionary(topology), silent=True) or {}


def _coords(vertex):
    return [round(float(c), 6) for c in Vertex.Coordinates(vertex)]


def test_graph_module_contains_no_active_os_system_calls():
    source = inspect.getsource(Graph)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                assert not (node.func.value.id == "os" and node.func.attr == "system")


def test_invalid_inputs_return_none_or_original_without_exceptions():
    assert Graph.AABB(None, silent=True) is None
    assert Graph.AddEdgeByIndex(None, [0, 1], silent=True) is None
    assert Graph.AddVertexByData(None, x=0, y=0, z=0, silent=True) is None
    assert Graph.AdjacentEdges(None, None, silent=True) is None
    assert Graph.Complete(None, silent=True) is None
    assert Graph.Order(None) is None
    assert Graph.Size(None) is None


def test_by_vertices_edges_indexes_vertices_and_edges():
    graph, _, _ = _chain_graph()

    assert Graph.Order(graph) == 3
    assert Graph.Size(graph) == 2

    vertices = Graph.Vertices(graph)
    assert [Dictionary.ValueAtKey(Topology.Dictionary(v), "index", silent=True) for v in vertices] == [0, 1, 2]

    edge_pairs = []
    for edge in Graph.Edges(graph):
        d = _python_dict(edge)
        edge_pairs.append((d.get("src"), d.get("dst")))
    assert sorted(edge_pairs) == [(0, 1), (1, 2)]


def test_adjacency_matrix_default_bidirectional_chain():
    graph, _, _ = _chain_graph()
    assert Graph.AdjacencyMatrix(graph) == [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]


def test_adjacency_matrix_can_be_directional():
    graph, _, _ = _chain_graph()
    assert Graph.AdjacencyMatrix(graph, bidirectional=False) == [
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]


def test_adjacency_matrix_edge_index_and_length_modes():
    graph, _, _ = _chain_graph()

    by_length = Graph.AdjacencyMatrix(graph, useEdgeLength=True, mantissa=6)
    assert by_length[0][1] == pytest.approx(1.0)
    assert by_length[1][2] == pytest.approx(1.0)


def test_adjacency_list_and_dictionary_for_chain():
    graph, _, _ = _chain_graph()

    assert Graph.AdjacencyList(graph, reverse=False) == [[1], [0, 2], [1]]

    adj_dict = Graph.AdjacencyDictionary(graph, includeWeights=False)
    assert adj_dict == {
        "000": ["001"],
        "001": ["000", "002"],
        "002": ["001"],
    }


def test_adjacent_vertices_for_middle_vertex():
    graph, _, _ = _chain_graph()
    vertices = Graph.Vertices(graph)
    adjacent = Graph.AdjacentVertices(graph, vertices[1], silent=True)
    assert len(adjacent) == 2
    assert sorted(Vertex.Index(v, vertices) for v in adjacent) == [0, 2]


def test_edge_lookup_and_contains_helpers():
    graph, _, _ = _chain_graph()
    vertices = Graph.Vertices(graph)
    edge = Graph.Edge(graph, vertices[0], vertices[1], silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert Graph.ContainsVertex(graph, vertices[0]) is True
    assert Graph.ContainsEdge(graph, edge) is True


def test_add_edge_by_index_adds_edge_and_src_dst_dictionary():
    vertices = [_vertex(0, 0, 0), _vertex(1, 0, 0), _vertex(0, 1, 0)]
    graph = Graph.ByVerticesEdges(vertices, [], index=True, ontology=False, silent=True)

    graph = Graph.AddEdgeByIndex(graph, [0, 2], silent=True)
    assert Topology.IsInstance(graph, "Graph")
    assert Graph.Size(graph) == 1

    edge = Graph.Edges(graph)[0]
    d = _python_dict(edge)
    assert d.get("src") == 0
    assert d.get("dst") == 2


def test_add_edge_by_index_rejects_bad_indices_without_modifying_graph():
    vertices = [_vertex(0, 0, 0), _vertex(1, 0, 0)]
    graph = Graph.ByVerticesEdges(vertices, [], index=True, ontology=False, silent=True)

    assert Graph.AddEdgeByIndex(graph, [-1, 1], silent=True) is None
    assert Graph.AddEdgeByIndex(graph, [0, 5], silent=True) is None
    assert Graph.AddEdgeByIndex(graph, [0], silent=True) is None
    assert Graph.Size(graph) == 0


def test_add_vertex_by_data_preserves_zero_coordinates_and_dictionary():
    graph = Graph.ByVerticesEdges([_vertex(10, 0, 0)], [], index=True, ontology=False, silent=True)
    d = Dictionary.ByPythonDictionary({"name": "origin"}, silent=True)

    graph = Graph.AddVertexByData(graph, dictionary=d, x=0, y=0, z=0, silent=True)
    assert Topology.IsInstance(graph, "Graph")

    vertices = Graph.Vertices(graph, sortBy=None)
    origin_vertices = [v for v in vertices if _coords(v) == [0.0, 0.0, 0.0]]
    assert len(origin_vertices) == 1
    assert Dictionary.ValueAtKey(Topology.Dictionary(origin_vertices[0]), "name", silent=True) == "origin"


def test_adjacent_edges_excludes_input_edge_and_duplicates(capfd):
    graph, _, _ = _chain_graph()
    edges = Graph.Edges(graph)

    adjacent = Graph.AdjacentEdges(graph, edges[0], silent=True)
    captured = capfd.readouterr()

    assert captured.out == ""
    assert len(adjacent) == 1
    assert Topology.IsSame(adjacent[0], edges[1])


def test_complete_turns_three_isolated_vertices_into_complete_graph():
    vertices = [_vertex(0, 0, 0), _vertex(1, 0, 0), _vertex(0, 1, 0)]
    graph = Graph.ByVerticesEdges(vertices, [_edge(vertices[0], vertices[1])], index=True, ontology=False, silent=True)

    complete = Graph.Complete(graph, ontology=False, silent=True)
    assert Topology.IsInstance(complete, "Graph")
    assert Graph.Order(complete) == 3
    assert Graph.Size(complete) == 3
    assert Graph.AdjacencyMatrix(complete) == [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]


def test_graph_dictionary_roundtrip():
    graph, _, _ = _chain_graph()
    graph = Graph.SetDictionary(graph, {"name": "chain"})

    assert Dictionary.ValueAtKey(Graph.Dictionary(graph), "name", silent=True) == "chain"


def test_remove_edge_and_remove_vertex_mutate_graph():
    graph, _, _ = _chain_graph()
    edges = Graph.Edges(graph)
    graph = Graph.RemoveEdge(graph, edges[0], silent=True)
    assert Topology.IsInstance(graph, "Graph")
    assert Graph.Size(graph) == 1

    vertices = Graph.Vertices(graph)
    graph = Graph.RemoveVertex(graph, vertices[0], silent=True)
    assert Topology.IsInstance(graph, "Graph")
    assert Graph.Order(graph) == 2


def test_degree_sequence_for_chain():
    graph, _, _ = _chain_graph()
    assert sorted(Graph.DegreeSequence(graph)) == [1, 1, 2]


def test_adjacency_matrix_csv_string_has_expected_rows():
    graph, _, _ = _chain_graph()
    csv_text = Graph.AdjacencyMatrixCSVString(graph)

    rows = [row.strip() for row in csv_text.strip().splitlines()]
    assert rows == ["0,1,0", "1,0,1", "0,1,0"]


def test_access_graph_key_none_does_not_lower_none(monkeypatch):
    """Regression test for AccessGraph(key=None) before it calls ByTopology."""
    from topologicpy import Topology as topology_module

    sentinel_topology = object()
    sentinel_graph = object()

    monkeypatch.setattr(
        topology_module.Topology,
        "IsInstance",
        staticmethod(lambda obj, typ: obj is sentinel_topology or obj is sentinel_graph),
    )
    monkeypatch.setattr(Graph, "ByTopology", staticmethod(lambda **kwargs: sentinel_graph))

    assert Graph.AccessGraph(sentinel_topology, key=None, silent=True, ontology=False) is sentinel_graph


def test_access_graph_handles_non_string_key_without_lower_crash(monkeypatch, capfd):
    from topologicpy import Topology as topology_module

    sentinel_topology = object()
    sentinel_graph = object()
    sentinel_vertex = object()

    monkeypatch.setattr(
        topology_module.Topology,
        "IsInstance",
        staticmethod(lambda obj, typ: obj is sentinel_topology or obj is sentinel_graph),
    )
    monkeypatch.setattr(Graph, "ByTopology", staticmethod(lambda **kwargs: sentinel_graph))
    monkeypatch.setattr(Graph, "Edges", staticmethod(lambda graph, *args, **kwargs: []))
    monkeypatch.setattr(Graph, "Vertices", staticmethod(lambda graph, *args, **kwargs: [sentinel_vertex]))

    assert Graph.AccessGraph(sentinel_topology, key=123, silent=True, ontology=False) is sentinel_graph
    assert capfd.readouterr().out == ""
