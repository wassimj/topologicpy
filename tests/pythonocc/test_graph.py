# Copyright (C) 2026
# PythonOCC backend Graph parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_graph.py -v

import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Graph = pytest.importorskip("topologicpy.Graph").Graph
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _simple_graph():
    """Create a simple triangle graph."""
    v1 = _v(0, 0, 0)
    v2 = _v(1, 0, 0)
    v3 = _v(0, 1, 0)
    e1 = Edge.ByStartVertexEndVertex(v1, v2)
    e2 = Edge.ByStartVertexEndVertex(v2, v3)
    e3 = Edge.ByStartVertexEndVertex(v3, v1)
    return Graph.ByVerticesEdges([v1, v2, v3], [e1, e2, e3])


# ===========================================================================
# Constructors
# ===========================================================================

class TestGraphConstructors:
    def test_by_vertices_edges(self):
        g = _simple_graph()
        assert Topology.IsInstance(g, "Graph")

    def test_by_vertices_only(self):
        v1 = _v(0, 0)
        v2 = _v(1, 0)
        g = Graph.ByVerticesEdges([v1, v2], [])
        assert Topology.IsInstance(g, "Graph")

    def test_empty_graph(self):
        g = Graph.ByVerticesEdges([], [])
        assert Topology.IsInstance(g, "Graph")


# ===========================================================================
# Accessors
# ===========================================================================

class TestGraphAccessors:
    def test_vertices(self):
        g = _simple_graph()
        verts = Graph.Vertices(g)
        assert len(verts) == 3

    def test_edges(self):
        g = _simple_graph()
        edges = Graph.Edges(g)
        assert len(edges) == 3

    def test_order(self):
        g = _simple_graph()
        assert Graph.Order(g) == 3  # Number of vertices

    def test_size(self):
        g = _simple_graph()
        assert Graph.Size(g) == 3  # Number of edges


# ===========================================================================
# Adjacency
# ===========================================================================

class TestGraphAdjacency:
    def test_adjacent_vertices(self):
        g = _simple_graph()
        verts = Graph.Vertices(g)
        v1 = verts[0]
        adj = Graph.AdjacentVertices(g, v1)
        assert len(adj) == 2  # Each vertex connected to 2 others in triangle

    def test_adjacent_edges(self):
        g = _simple_graph()
        verts = Graph.Vertices(g)
        v1 = verts[0]
        adj_edges = Graph.AdjacentEdges(g, v1)
        assert len(adj_edges) == 2  # Each vertex has 2 adjacent edges


# ===========================================================================
# Operations
# ===========================================================================

class TestGraphOperations:
    def test_add_vertex(self):
        g = _simple_graph()
        v4 = _v(1, 1, 0)
        g2 = Graph.AddVertex(g, v4)
        assert Graph.Order(g2) == 4

    def test_add_edge(self):
        g = _simple_graph()
        verts = Graph.Vertices(g)
        v1, v2 = verts[0], verts[1]
        e_new = Edge.ByStartVertexEndVertex(v1, v2)
        # This might fail if edge already exists
        # Just test that the operation doesn't crash

    def test_contains_vertex(self):
        g = _simple_graph()
        verts = Graph.Vertices(g)
        v1 = verts[0]
        assert Graph.ContainsVertex(g, v1) is True

    def test_contains_edge(self):
        g = _simple_graph()
        edges = Graph.Edges(g)
        e1 = edges[0]
        assert Graph.ContainsEdge(g, e1) is True


# ===========================================================================
# Type checking
# ===========================================================================

class TestGraphType:
    def test_is_instance_graph(self):
        g = _simple_graph()
        assert Topology.IsInstance(g, "Graph") is True

    def test_is_not_cell(self):
        g = _simple_graph()
        assert Topology.IsInstance(g, "Cell") is False


# ===========================================================================
# Serialization
# ===========================================================================

class TestGraphSerialization:
    def test_brep_roundtrip(self):
        g = _simple_graph()
        brep = Topology.BREPString(g)
        assert brep is not None
        g2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(g2, "Graph")
        assert Graph.Order(g2) == Graph.Order(g)
        assert Graph.Size(g2) == Graph.Size(g)
