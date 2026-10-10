"""Source identity, dictionary transfer and axial/segment graph regressions."""
import json
import pytest

from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.Dictionary import Dictionary


def edge(a, b):
    return Edge.ByVertices(Vertex.ByCoordinates(*a), Vertex.ByCoordinates(*b))


def data(topology):
    return TGraph._TopologyDictionaryToPython(topology)


def test_generic_originators_are_independent_of_display_and_transfer_values():
    sources = [Vertex.ByCoordinates(0, 0, 0), edge((0, 0, 0), (1, 0, 0)),
               Face.Rectangle(), Cell.Prism()]
    graph = TGraph()
    for i, source in enumerate(sources):
        Topology.SetDictionary(source, Dictionary.ByKeysValues(["existing"], ["retained"]))
        graph.AddVertex({"score": i+0.5, "color": "#ff0000"},
                        representation=Vertex.ByCoordinates(i, 0, 0), originator=source)
    assert all(a is b for a, b in zip(TGraph.Originators(graph), sources))
    assert all(graph._vertices[i]["representation"] is not source for i, source in enumerate(sources))
    updated = TGraph.TransferDictionariesToOriginators(graph, keys=["score", "color"])
    assert len(updated) == 4
    for i, source in enumerate(sources):
        assert data(source)["score"] == i+0.5
        assert data(source)["color"] == "#ff0000"
        assert data(source)["existing"] == "retained"
        assert "index" not in data(source)
        assert data(source)["uuid"] == graph._vertices[i]["dictionary"]["originator_id"]


def test_automatic_links_copy_subgraph_and_json_resolution():
    source = edge((0, 0, 0), (1, 0, 0))
    graph = TGraph()
    graph.AddVertex({"score": 2}, representation=source)
    graph.AddVertex({"score": 3}, representation=Vertex.ByCoordinates(3, 0, 0))
    assert TGraph.VertexOriginator(graph, 0) is source
    assert TGraph.VertexOriginator(TGraph.Copy(graph), 0) is source
    assert TGraph.VertexOriginator(TGraph.Subgraph(graph, [0]), 0) is source
    restored = TGraph.FromPython(json.loads(json.dumps(TGraph.ToPython(TGraph.Subgraph(graph, [0])))))
    assert TGraph.VertexOriginator(restored, 0, silent=True) is None
    assert TGraph.VertexOriginator(restored, 0, originators=[source]) is source
    assert TGraph.TransferDictionariesToOriginators(restored, keys="score") == [source]
    assert data(source)["score"] == 2


def test_duplicate_uuid_and_missing_links_fail_before_modifying_topology():
    a = edge((0, 0, 0), (1, 0, 0))
    graph = TGraph()
    graph.AddVertex({"score": 2}, originator=a)
    b = Topology.Copy(a)
    report = TGraph.TransferDictionariesToOriginators(graph, keys="score", originators=[a, b], returnReport=True, silent=True)
    assert report["errors"] and report["transferred"] == 0
    assert "score" not in data(a)
    graph.AddVertex({"score": 4})
    assert TGraph.TransferDictionariesToOriginators(graph, keys="score", silent=True) is None
    assert "score" not in data(a)


def test_multiple_nodes_require_explicit_aggregation_and_identity_is_protected():
    source = Vertex.ByCoordinates(0, 0, 0)
    graph = TGraph()
    for value in (2, 4):
        graph.AddVertex({"score": value, "color": "red"}, originator=source)
    original_uuid = data(source)["uuid"]
    assert TGraph.TransferDictionariesToOriginators(graph, keys="score", silent=True) is None
    assert "score" not in data(source)
    assert TGraph.TransferDictionariesToOriginators(graph, keys="score", aggregation="mean") == [source]
    assert data(source)["score"] == 3
    assert TGraph.TransferDictionariesToOriginators(graph, keys="color") == [source]
    graph._vertices[0]["dictionary"]["uuid"] = "wrong"
    assert TGraph.TransferDictionariesToOriginators(graph, keys="uuid") == [source]
    assert data(source)["uuid"] == original_uuid
    TGraph.TransferDictionariesToOriginators(graph, keys="score", aggregation="sum", overwrite=False)
    assert data(source)["score"] == 3


def test_spatial_constructor_originators_even_with_point_representations():
    sources = [edge((-1, 0, 0), (1, 0, 0)), edge((0, -1, 0), (0, 1, 0))]
    graph = TGraph.BySpatialRelationships(sources, include=["intersects"])
    assert all(a is b for a, b in zip(TGraph.Originators(graph), sources))
    assert all(Topology.IsInstance(v["representation"], "Vertex") for v in graph._vertices)
    assert TGraph.VertexOriginator(TGraph.Copy(graph), 0) is sources[0]


def test_links_survive_dictionary_replacement_and_graph_derivatives():
    source = Vertex.ByCoordinates(0, 0, 0)
    graph = TGraph()
    graph.AddVertex({"id": "one"}, representation=Vertex.ByCoordinates(1, 1, 1), originator=source)
    graph._vertices[0]["dictionary"] = {"score": 5}
    assert TGraph.VertexOriginator(graph, 0) is source
    exported = TGraph.ToPython(graph)
    restored = TGraph.FromPython(json.loads(json.dumps(exported)))
    assert TGraph.VertexOriginator(restored, 0, originators=[source]) is source
    for derivative in (TGraph.Tree(graph), TGraph.MinimumSpanningTree(graph), TGraph._CopyGraph(graph)):
        assert TGraph.VertexOriginator(derivative, 0) is source


def test_by_topology_links_to_face_rather_than_representative_vertex():
    source = Face.Rectangle()
    graph = TGraph.ByTopology(source, silent=True)
    assert graph is not None
    assert any(originator is source for originator in TGraph.Originators(graph, silent=True))


def test_by_vertices_edges_retains_sources_without_display_representations():
    sources = [Vertex.ByCoordinates(0,0,0), Vertex.ByCoordinates(1,0,0)]
    graph = TGraph.ByVerticesEdges(sources, [], storeRepresentations=False)
    assert all(v["representation"] is None for v in graph._vertices)
    assert all(a is b for a,b in zip(TGraph.Originators(graph), sources))


def test_link_assignment_clearing_and_inactive_nodes():
    source = Face.Rectangle()
    graph = TGraph()
    graph.AddVertex({"score": 1})
    assert TGraph.SetVertexOriginator(graph, 0, source) is graph
    assert TGraph.VertexOriginator(graph, 0) is source
    assert TGraph.SetVertexOriginator(graph, 0, None) is graph
    assert TGraph.VertexOriginator(graph, 0) is None
    assert "originator_id" not in graph._vertices[0]["dictionary"]
    graph.AddVertex({"score": 2}, originator=source)
    graph.RemoveVertex(0)
    assert TGraph.Originators(graph) == [source]
    assert TGraph.TransferDictionariesToOriginators(graph, keys="score") == [source]
    assert data(source)["score"] == 2


def test_parent_references_survive_subgraph_reindexing():
    parent = edge((0,0,0), (2,0,0))
    graph = TGraph.SegmentGraph([parent, edge((1,0,0), (1,1,0))])
    subset = TGraph.Subgraph(graph, [1])
    assert subset._vertices[0]["parent_originator"] is parent
    assert TGraph.VertexOriginator(subset, 0) is TGraph.VertexOriginator(graph, 1)


def test_numeric_aggregation_refuses_colour_strings():
    source = Vertex.ByCoordinates(0,0,0)
    graph = TGraph()
    graph.AddVertex({"color":"red"}, originator=source)
    graph.AddVertex({"color":"blue"}, originator=source)
    assert TGraph.TransferDictionariesToOriginators(graph, keys="color", aggregation="mean", silent=True) is None
    assert "color" not in data(source)


def test_unsupported_values_fail_preflight_for_all_sources():
    first, second = Vertex.ByCoordinates(0,0,0), Vertex.ByCoordinates(1,0,0)
    graph = TGraph()
    graph.AddVertex({"score":2}, originator=first)
    graph.AddVertex({"score":object()}, originator=second)
    report = TGraph.TransferDictionariesToOriginators(graph, keys="score", returnReport=True, silent=True)
    assert report["errors"] and report["transferred"] == 0
    assert "score" not in data(first)
    assert "score" not in data(second)


def test_by_edge_index_pairs_retains_originators():
    source = Face.Rectangle()
    graph = TGraph.ByEdgeIndexPairs(1, [], representations={
        "vertices":[Vertex.ByCoordinates(0,0,0)], "originators":[source]})
    assert TGraph.VertexOriginator(graph, 0) is source


def test_axial_graph_intersections_ids_and_colour_transfer():
    sources = [edge((-1, 0, 0), (1, 0, 0)), edge((0, -1, 0), (0, 1, 0)),
               edge((0, -1, 2), (0, 1, 2))]
    graph = TGraph.AxialGraph(edges=sources)
    assert TGraph.Order(graph) == 3 and TGraph.Size(graph) == 1
    assert all(a is b for a, b in zip(TGraph.Originators(graph), sources))
    assert [v["dictionary"]["edge_id"] for v in graph._vertices] == [Topology.UUID(e) for e in sources]
    TGraph.Connectivity(graph, key="score")
    TGraph.TransferDictionariesToOriginators(graph, keys="score")
    assert [data(e)["score"] for e in sources] == [1, 1, 0]


def test_segment_graph_splits_crossing_and_retains_parent_identity():
    sources = [edge((-1, 0, 0), (1, 0, 0)), edge((0, -1, 0), (0, 1, 0))]
    graph = TGraph.SegmentGraph(edges=sources)
    assert TGraph.Order(graph) == 4 and TGraph.Size(graph) == 6
    assert all(Topology.IsInstance(t, "Edge") for t in TGraph.Originators(graph))
    assert set(v["dictionary"]["parent_edge_id"] for v in graph._vertices) == set(Topology.UUID(e) for e in sources)
    assert sorted(e["dictionary"]["angular_weight"] for e in graph._edges) == [0, 0, 1, 1, 1, 1]
    assert len(TGraph.AngularChoice(graph)) == 4
    assert len(TGraph.AngularIntegration(graph)) == 4
    assert TGraph.AngularConnectivity(graph) == [3, 3, 3, 3]
    TGraph.TransferDictionariesToOriginators(graph, keys="connectivity")
    assert [data(e)["connectivity"] for e in TGraph.Originators(graph)] == [3]*4
    again = TGraph.SegmentGraph(sources)
    assert [v["dictionary"]["edge_id"] for v in again._vertices] == [v["dictionary"]["edge_id"] for v in graph._vertices]
    copied = TGraph.Copy(graph)
    assert copied._vertices[0]["parent_originator"] is sources[0]


def test_segment_graph_t_junction_and_projected_crossing():
    sources = [edge((-1, 0, 0), (1, 0, 0)), edge((0, 0, 0), (0, 1, 0)),
               edge((-1, 0, 3), (1, 0, 3))]
    graph = TGraph.SegmentGraph(sources)
    assert TGraph.Order(graph) == 4 and TGraph.Size(graph) == 3
    assert sorted(TGraph.Connectivity(graph)) == [0, 2, 2, 2]


def test_segment_overlaps_split_deduplicate_and_preserve_all_parents():
    sources = [edge((0, 0, 0), (2, 0, 0)), edge((1, 0, 0), (3, 0, 0))]
    graph = TGraph.SegmentGraph(sources)
    assert TGraph.Order(graph) == 3
    assert TGraph.Size(graph) == 2
    shared = [v for v in graph._vertices if len(v["dictionary"]["parent_edge_ids"]) == 2]
    assert len(shared) == 1
    assert shared[0]["parent_originators"] == sources
    subset = TGraph.Subgraph(graph, [shared[0]["index"]])
    assert subset._vertices[0]["parent_originators"] == sources
    assert set(shared[0]["dictionary"]["parent_edge_ids"]) == {Topology.UUID(e) for e in sources}
    assert sorted(len(v["dictionary"]["parent_edge_ids"]) for v in graph._vertices) == [1,1,2]
    assert sorted(data(t)["parent_edge_ids"] for t in TGraph.Originators(graph)) == sorted(v["dictionary"]["parent_edge_ids"] for v in graph._vertices)


@pytest.mark.parametrize("a,b", [((1,0,0), (3,0,0)),
                                 ((0.5,0,0), (1.5,0,0)),
                                 ((2,0,0), (0,0,0)),
                                 ((0,0,0), (2,0,0)),
                                 ((2,0,0), (3,0,0))])
def test_axial_overlap_contact_and_colour_transfer(a, b):
    sources = [edge((0,0,0), (2,0,0)), edge(a,b)]
    identities = [Topology.UUID(e) for e in sources]
    graph = TGraph.AxialGraph(sources)
    assert TGraph.Order(graph) == 2
    assert TGraph.Size(graph) == 1
    assert all(a is b for a, b in zip(TGraph.Originators(graph), sources))
    assert [v["dictionary"]["edge_id"] for v in graph._vertices] == identities
    TGraph.Connectivity(graph, key="score")
    TGraph.TransferDictionariesToOriginators(graph, keys="score")
    assert [data(e)["score"] for e in sources] == [1,1]
    assert [Topology.UUID(e) for e in sources] == identities


def test_tilted_3d_overlap_and_crossing_connect_once_per_pair():
    sources = [edge((0,0,0), (3,3,3)), edge((1,1,1), (4,4,4)),
               edge((2,1,3), (2,3,1))]
    graph = TGraph.AxialGraph(sources)
    assert TGraph.Order(graph) == 3
    assert TGraph.Size(graph) == 3
    assert TGraph.Connectivity(graph) == [2,2,2]


def test_projected_collinear_overlap_on_another_floor_does_not_connect():
    sources = [edge((0,0,0), (2,0,0)), edge((1,0,2), (3,0,2))]
    graph = TGraph.AxialGraph(sources)
    assert TGraph.Order(graph) == 2
    assert TGraph.Size(graph) == 0


@pytest.mark.parametrize("method", [TGraph.AxialGraph, TGraph.SegmentGraph])
def test_empty_axial_and_segment_graphs(method):
    assert TGraph.Order(method([], silent=True)) == 0


@pytest.mark.parametrize("first, second, expected_count", [
    (((0,0,0),(3,0,0)), ((3,0,0),(0,0,0)), 1),
    (((0,0,0),(4,0,0)), ((1,0,0),(3,0,0)), 3),
    (((0,0,0),(3,3,3)), ((1,1,1),(4,4,4)), 3),
    (((0,0,0),(3,0,0)), ((1,0,2),(4,0,2)), 2),
])
def test_segment_overlap_variants(first, second, expected_count):
    sources = [edge(*first), edge(*second)]
    g = TGraph.SegmentGraph(sources)
    assert g is not None
    assert TGraph.Order(g) == expected_count
    assert all(v["dictionary"]["parent_edge_ids"] for v in g._vertices)
    copied = TGraph.Copy(g)
    assert [v["parent_originators"] for v in copied._vertices] == [v["parent_originators"] for v in g._vertices]


def test_segment_overlap_crossing_splits_all_parents_consistently():
    sources = [edge((0,0,0),(3,0,0)), edge((1,0,0),(4,0,0)), edge((2,-1,0),(2,1,0))]
    g = TGraph.SegmentGraph(sources)
    assert TGraph.Order(g) == 6
    shared = [v for v in g._vertices if len(v["dictionary"]["parent_edge_ids"]) == 2]
    assert len(shared) == 2
    assert TGraph.IsConnected(g)
