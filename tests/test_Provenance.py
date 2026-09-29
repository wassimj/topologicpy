# Copyright (C) 2026
# Wassim Jabi <wassimj@gmail.com>
#
# Tests for public TopologicPy provenance.
#
# Exact Boolean provenance depends on PythonOCC/OCCT BRepTools_History.
# TopologicCore does not expose the native history required by this API.

import pytest

from topologicpy.Cell import Cell
from topologicpy.Provenance import Provenance
from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


pytestmark = pytest.mark.pythonocc_only


def _role_counts(graph):
    counts = {}
    for vertex in TGraph.Vertices(graph):
        dictionary = vertex.get("dictionary") or {}
        role = dictionary.get("role")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _edge_relations(graph):
    relations = []
    for edge in TGraph.Edges(graph):
        dictionary = edge.get("dictionary") or {}
        relation = dictionary.get("relation")
        if relation is not None:
            relations.append(relation)
    return relations


def _make_overlapping_prisms():
    p1 = Cell.Prism(
        origin=Vertex.ByCoordinates(0, 0, 0)
    )
    p2 = Cell.Prism(
        origin=Vertex.ByCoordinates(0.6, 0.6, 0.6)
    )
    return p1, p2


def test_provenance_container_from_synthetic_records():
    source = Vertex.ByCoordinates(0, 0, 0)
    result = Vertex.ByCoordinates(1, 0, 0)

    provenance = Provenance.ByRecords(
        [
            {
                "operation": "Test",
                "relation": "modified",
                "source": source,
                "result": result,
                "sourceType": "Vertex",
                "resultType": "Vertex",
                "sourceRole": "self",
            }
        ],
        result=result,
    )

    history = provenance.History()
    assert len(history) == 1
    assert history[0]["relation"] == "modified"

    records = provenance.Records()
    assert len(records) == 1

    origins = provenance.Origins(result)
    assert len(origins) == 1
    assert Topology.IsSame(origins[0]["source"], source)

    descendants = provenance.Descendants(source)
    assert len(descendants) == 1
    assert Topology.IsSame(descendants[0]["result"], result)


def test_merge_returns_provenance_without_dictionary_transfer():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Merge(
        p1,
        p2,
        returnProvenance=True,
    )

    assert result is not None
    assert isinstance(provenance, Provenance)
    assert provenance.History()
    assert provenance.Records()


def test_merge_default_return_type_remains_backward_compatible():
    p1, p2 = _make_overlapping_prisms()

    result = Topology.Merge(
        p1,
        p2,
    )

    assert result is not None
    assert not isinstance(result, tuple)


def test_merge_face_graph_matches_authoritative_result():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Merge(
        p1,
        p2,
        returnProvenance=True,
    )

    result_faces = Topology.Faces(result)
    graph = provenance.Graph(topologyType="Face")
    counts = _role_counts(graph)

    assert counts.get("source", 0) == 12
    assert len(result_faces) == 18
    assert counts.get("result", 0) == len(result_faces)
    assert counts.get("intermediate", 0) == 0


def test_merge_unchanged_faces_are_explicit_source_to_result_states():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Merge(
        p1,
        p2,
        returnProvenance=True,
    )

    graph = provenance.Graph(topologyType="Face")
    relations = _edge_relations(graph)

    assert relations.count("unchanged") >= 6

    counts = _role_counts(graph)
    assert counts.get("source", 0) == 12
    assert counts.get("result", 0) == 18


def test_merge_every_semantic_result_face_belongs_to_returned_topology():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Merge(
        p1,
        p2,
        returnProvenance=True,
    )

    result_faces = Topology.Faces(result)
    graph = provenance.Graph(topologyType="Face")

    graph_result_faces = []
    for vertex in TGraph.Vertices(graph):
        dictionary = vertex.get("dictionary") or {}
        if dictionary.get("role") == "result":
            representation = vertex.get("representation")
            assert representation is not None
            graph_result_faces.append(representation)

    assert len(graph_result_faces) == len(result_faces)

    for graph_face in graph_result_faces:
        assert any(
            Topology.IsSame(graph_face, result_face)
            for result_face in result_faces
        )


def test_merge_origins_resolve_for_every_result_face():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Merge(
        p1,
        p2,
        returnProvenance=True,
    )

    for face in Topology.Faces(result):
        origins = provenance.Origins(face)
        assert origins
        assert all(record.get("source") is not None for record in origins)


def test_difference_face_graph_contains_only_contributing_sources():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Difference(
        p1,
        p2,
        returnProvenance=True,
    )

    assert result is not None

    graph = provenance.Graph(topologyType="Face")
    counts = _role_counts(graph)
    result_faces = Topology.Faces(result)

    assert counts.get("source", 0) == 9
    assert counts.get("result", 0) == len(result_faces)
    assert counts.get("intermediate", 0) == 0


def test_difference_does_not_require_dictionary_transfer_for_provenance():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Difference(
        p1,
        p2,
        tranDict=False,
        returnProvenance=True,
    )

    assert result is not None
    assert provenance.History()
    assert provenance.Records()

    graph = provenance.Graph(topologyType="Face")
    assert TGraph.Order(graph) > 0
    assert TGraph.Size(graph) > 0


def test_direct_boolean_graph_result_nodes_are_authoritative_for_difference():
    p1, p2 = _make_overlapping_prisms()

    result, provenance = Topology.Difference(
        p1,
        p2,
        returnProvenance=True,
    )

    result_faces = Topology.Faces(result)
    graph = provenance.Graph(topologyType="Face")

    graph_result_faces = []
    for vertex in TGraph.Vertices(graph):
        dictionary = vertex.get("dictionary") or {}
        if dictionary.get("role") == "result":
            graph_result_faces.append(vertex.get("representation"))

    assert len(graph_result_faces) == len(result_faces)

    for graph_face in graph_result_faces:
        assert graph_face is not None
        assert any(
            Topology.IsSame(graph_face, result_face)
            for result_face in result_faces
        )
