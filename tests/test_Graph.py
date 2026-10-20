"""Graph regression scenarios migrated to the TGraph record API."""
import pytest
from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge


@pytest.fixture(autouse=True)
def suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def chain():
    vertices = [Vertex.ByCoordinates(i, 0, 0) for i in range(3)]
    edges = [Edge.ByVertices(vertices[i:i+2], silent=True) for i in range(2)]
    return TGraph.ByVerticesEdges(vertices, edges, ontology=False, silent=True)


def test_invalid_membership_is_false():
    assert not TGraph.ContainsVertex(None, 0)
    assert not TGraph.ContainsEdge(None, 0)


def test_by_vertices_edges_indexes_vertices_and_edges():
    graph = chain()
    assert [v['index'] for v in TGraph.Vertices(graph)] == [0, 1, 2]
    assert [(e['src'], e['dst']) for e in TGraph.Edges(graph)] == [(0, 1), (1, 2)]


def test_adjacency_matrix_default_bidirectional_chain():
    assert TGraph.AdjacencyMatrix(chain()) == [[0,1,0],[1,0,1],[0,1,0]]


def test_adjacency_matrix_can_be_directional():
    graph = chain()
    for edge in graph._edges: edge['directed'] = True
    assert TGraph.AdjacencyMatrix(graph, bidirectional=False) == [[0,1,0],[0,0,1],[0,0,0]]


def test_adjacency_matrix_edge_index_and_length_modes():
    graph = chain()
    assert TGraph.AdjacencyMatrix(graph, useEdgeIndex=True) == [[0,1,0],[1,0,2],[0,2,0]]
    assert TGraph.AdjacencyMatrix(graph, useEdgeLength=True)[1][2] == pytest.approx(1)


def test_adjacency_list_and_dictionary_for_chain():
    graph = chain()
    assert TGraph.AdjacencyList(graph) == [[1],[0,2],[1]]
    assert TGraph.AdjacencyDictionary(graph) == {0:[1],1:[0,2],2:[1]}


def test_adjacent_vertices_for_middle_vertex():
    assert [v['index'] for v in TGraph.AdjacentVertices(chain(), 1)] == [0,2]


def test_edge_lookup_and_contains_helpers():
    graph = chain()
    edge = TGraph.Edge(graph, vertexA=0, vertexB=1)
    assert edge['src'] == 0 and edge['dst'] == 1
    assert TGraph.ContainsVertex(graph, 0)
    assert TGraph.ContainsEdge(graph, edge)


def test_add_edge_by_index_adds_edge_and_endpoints():
    graph = TGraph.ByVerticesEdges([{'id':i} for i in range(3)], [], ontology=False)
    index = graph.AddEdgeByIndex([0,2])
    assert index == 0
    assert TGraph.Edges(graph)[0]['src'] == 0
    assert TGraph.Edges(graph)[0]['dst'] == 2


@pytest.mark.parametrize('pair', [[-1,1],[0,5],[0]])
def test_add_edge_by_index_rejects_bad_indices_without_mutation(pair):
    graph = TGraph.ByVerticesEdges([{'id':0},{'id':1}], [], ontology=False)
    assert graph.AddEdgeByIndex(pair, silent=True) is None
    assert TGraph.Size(graph) == 0


def test_add_vertex_by_data_preserves_zero_coordinates_and_dictionary():
    graph = chain()
    assert TGraph.AddVertexByData(graph, dictionary={'name':'origin'}, x=0, y=0, z=0) is graph
    assert TGraph.Coordinates(graph, 3) == [0,0,0]
    assert TGraph.Vertices(graph)[3]['dictionary']['name'] == 'origin'


def test_adjacent_edges_excludes_input_edge_and_duplicates():
    graph = chain()
    assert [e['index'] for e in TGraph.AdjacentEdges(graph, 0)] == [1]


def test_complete_turns_chain_into_complete_graph():
    complete = TGraph.Complete(chain(), ontology=False)
    assert TGraph.Order(complete) == 3
    assert TGraph.Size(complete) == 3
    assert TGraph.AdjacencyMatrix(complete) == [[0,1,1],[1,0,1],[1,1,0]]


def test_graph_dictionary_roundtrip():
    graph = chain().SetDictionary({'name':'chain'})
    assert TGraph.Dictionary(graph) == {'name':'chain'}


def test_remove_edge_and_vertex_mutate_graph():
    graph = chain()
    assert graph.RemoveEdge(0) is graph
    assert TGraph.Size(graph) == 1
    assert graph.RemoveVertex(0) is graph
    assert TGraph.Order(graph) == 2


def test_degree_sequence_for_chain():
    assert sorted(TGraph.DegreeSequence(chain())) == [1,1,2]


def test_mesh_data_contains_index_pairs_and_coordinates():
    assert TGraph.MeshData(chain()) == {'vertices':[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],'edges':[[0,1],[1,2]]}


@pytest.mark.parametrize('key',[None,123])
def test_access_graph_handles_optional_keys(monkeypatch,key):
    graph = chain()
    monkeypatch.setattr(TGraph,'ByTopology',staticmethod(lambda *args,**kwargs: graph))
    assert TGraph.AccessGraph(object(), key=key, silent=True) is graph
