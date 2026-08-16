"""Unit tests for topologicpy.KnowledgeGraph.

The suite is intentionally dependency-light. Most tests construct KnowledgeGraph
with useRDFLib=False and use monkeypatched fake Ontology/TGraph/Reasoner classes
for bridge behaviour. This keeps the tests deterministic on CI runners with or
without RDFLib installed.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from topologicpy.KnowledgeGraph import KG, KnowledgeGraph


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _kg(*triples):
    return KnowledgeGraph.ByTriples(triples, useRDFLib=False, silent=True)


def test_namespace_qname_and_normalize_term_helpers():
    namespaces = KnowledgeGraph.Namespaces({"ex": "https://example.org/"})

    assert namespaces["ex"] == "https://example.org/"
    assert KnowledgeGraph.ExpandQName("ex:Room", namespaces) == "https://example.org/Room"
    assert KnowledgeGraph.ExpandQName("<https://example.org/Room>", namespaces) == "https://example.org/Room"
    assert KnowledgeGraph.QName("https://example.org/Room", namespaces) == "ex:Room"
    assert KnowledgeGraph.QName("<https://example.org/Room>", namespaces) == "ex:Room"

    assert KnowledgeGraph.NormalizeTerm("Room 101", role="subject", namespaces=namespaces) == "inst:Room_101"
    assert KnowledgeGraph.NormalizeTerm("Fire Rating", role="predicate", namespaces=namespaces) == "dict:Fire_Rating"
    assert KnowledgeGraph.NormalizeTerm("ex:Room", role="object", namespaces=namespaces) == "ex:Room"
    assert KnowledgeGraph.NormalizeTerm("A room", role="object", namespaces=namespaces) == '"A room"'
    assert KnowledgeGraph.NormalizeTerm(True, role="object") == '"true"^^xsd:boolean'
    assert KnowledgeGraph.NormalizeTerm(3, role="object") == '"3"^^xsd:integer'
    assert KnowledgeGraph.NormalizeTerm(2.5, role="object") == '"2.5"^^xsd:double'


def test_core_triple_store_add_filter_remove_clear_and_iterate():
    kg = KnowledgeGraph(useRDFLib=False)

    triple = kg.AddTriple("Room 101", "label", "Office", objectIsLiteral=True)
    assert triple[0] == "inst:Room_101"
    assert triple[1] in {"top:label", "dict:label"}
    assert triple[2] == '"Office"'
    assert len(kg) == 1
    assert triple in kg
    assert kg.HasTriple(("Room 101", "label", "Office"))

    kg.AddTriple("Room 101", "rdf:type", "top:TGraph")
    type_objects = set(kg.Objects(subject="Room 101", predicate="rdf:type"))
    assert type_objects & {"top:TGraph", "top:Graph"}

    kg.AddTriple("Room 101", "custom property", "Value", objectIsLiteral=True)
    assert kg.Subjects() == ["inst:Room_101"]
    assert triple[1] in kg.Predicates()
    assert '"Office"' in kg.Objects(predicate=triple[1])
    assert '"Office"' in kg.Literals()
    assert "inst:Room_101" in kg.Resources()
    assert triple[1] not in kg.Resources(includePredicates=False)
    assert triple[1] in kg.Resources(includePredicates=True)

    iterated = list(iter(kg))
    assert iterated == kg.Triples(sort=True)

    assert kg.RemoveTriple("Room 101", "label", "Office") is True
    assert kg.RemoveTriple("Room 101", "label", "Office", silent=True) is False
    assert not kg.HasTriple(("Room 101", triple[1], "Office"))

    assert kg.Clear() is kg
    assert kg.Triples() == []


def test_dictionary_json_and_native_json_file_round_trip(tmp_path):
    kg = _kg(("inst:a", "rdf:type", "top:Graph"), ("inst:a", "dict:label", "Alpha"))

    data = kg.Dictionary()
    assert data["type"] == "KnowledgeGraph"
    assert len(data["triples"]) == 2

    from_dict = KnowledgeGraph.ByDictionary(data, useRDFLib=False, silent=True)
    assert from_dict.Triples() == kg.Triples()

    from_json = KnowledgeGraph.ByJSON(kg.JSONString(), useRDFLib=False, silent=True)
    assert from_json.Triples() == kg.Triples()

    json_path = tmp_path / "kg.json"
    assert kg.Export(str(json_path), overwrite=True, silent=True) == str(json_path)
    assert json_path.exists()

    imported = KnowledgeGraph.ByFile(str(json_path), silent=True)
    assert imported is not None
    assert imported.Triples() == kg.Triples()

    assert kg.Export(str(json_path), overwrite=False, silent=True) is None




def test_turtle_fallback_export_and_static_export_wrappers(tmp_path, monkeypatch):
    monkeypatch.setattr(KnowledgeGraph, "_ontology_class", staticmethod(lambda: None))
    kg = _kg(("inst:a", "rdf:type", "top:Graph"), ("inst:a", "dict:label", "Alpha"))

    ttl = kg.TurtleString(includeHeader=True, useRDFLib=False, silent=True)
    assert "@prefix top:" in ttl
    assert "inst:a rdf:type top:Graph ." in ttl
    assert 'inst:a dict:label "Alpha" .' in ttl

    assert KnowledgeGraph.TurtleStringByTriples([("inst:b", "dict:label", "Beta")], includeHeader=False) == 'inst:b dict:label "Beta" .\n'

    ttl_path = tmp_path / "kg.ttl"
    assert KnowledgeGraph.ExportTTL(kg, str(ttl_path), overwrite=True, silent=True) == str(ttl_path)
    assert ttl_path.read_text(encoding="utf-8").strip()

    json_path = tmp_path / "wrapped.json"
    assert KnowledgeGraph.ExportRDF(kg.Dictionary(), str(json_path), overwrite=True, silent=True) == str(json_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["type"] == "KnowledgeGraph"


def test_merge_difference_diff_copy_statistics_and_wrappers():
    before = _kg(("inst:a", "dict:label", "A"), ("inst:b", "dict:label", "B"))
    after = _kg(("inst:b", "dict:label", "B"), ("inst:c", "dict:label", "C"))

    copied = before.Copy()
    assert copied is not before
    assert copied.Triples() == before.Triples()

    merged = before.Merge(after, inplace=False)
    assert len(merged) == 3
    assert len(before) == 2

    before.Merge(after, inplace=True)
    assert len(before) == 3

    assert _kg(("inst:a", "dict:label", "A")).Difference(after, silent=True) == [("inst:a", "dict:label", '"A"')]
    assert _kg(("inst:a", "dict:label", "A")).Difference(after, direction="other_minus_self", silent=True) == [
        ("inst:b", "dict:label", '"B"'),
        ("inst:c", "dict:label", '"C"'),
    ]

    diff = _kg(("inst:a", "dict:label", "A")).Diff(after, silent=True)
    assert diff["added_count"] == 2
    assert diff["removed_count"] == 1
    assert diff["unchanged_count"] == 0

    multi = KnowledgeGraph.MergeGraphs([[("inst:x", "dict:label", "X")], after], silent=True)
    assert len(multi) == 3

    static_diff = KnowledgeGraph.DiffGraphs(_kg(("inst:a", "dict:label", "A")), after, silent=True)
    assert static_diff["added_count"] == 2

    stats = KnowledgeGraph.Statistics(after)
    assert stats["triple_count"] == 2
    assert stats["subject_count"] == 2


def test_summary_and_validate_without_requiring_rdflib(monkeypatch):
    kg = _kg(("inst:a", "rdf:type", "top:Graph"), ("inst:a", "dict:label", "Alpha"))

    summary = kg.Summary()
    assert summary["triple_count"] == 2
    assert summary["subject_count"] == 1
    assert summary["predicate_count"] == 2
    assert summary["literal_count"] == 1
    assert "rdflib_available" in summary

    report = kg.Validate(parseWithRDFLib=False, silent=True)
    assert report["valid"] is True
    assert report["triple_count"] == 2
    assert report["subject_count"] == 1
    assert isinstance(report["warnings"], list)


def test_by_topology_uses_fake_ontology_and_filters_kwargs(monkeypatch):
    calls = []

    class FakeOntology:
        NAMESPACES = KnowledgeGraph.Namespaces({"fake": "https://example.org/fake#"})

        @staticmethod
        def _is_graph_like(topology):
            return topology == "graph-like"

        @staticmethod
        def GraphTriples(topology, includeDictionaries=True, includeBOT=True, namespacePrefix="inst", silent=False, accepted=None):
            calls.append(("graph", topology, accepted))
            return [("inst:g", "rdf:type", "top:Graph")]

        @staticmethod
        def Triples(topology, includeDictionaries=True, includeBOT=True, namespacePrefix="inst", silent=False, accepted=None):
            calls.append(("topology", topology, accepted))
            return [("inst:t", "rdf:type", "top:Face")]

        @staticmethod
        def OntologyTriples(includeBOT=True):
            return [("top:Graph", "rdf:type", "owl:Class")]

        @staticmethod
        def PropertyQName(value):
            return {"hasStartVertex": "top:startsAt"}.get(value, value if str(value).startswith(("top:", "rdf:", "rdfs:", "owl:")) else None)

        @staticmethod
        def CanonicalClass(value, defaultValue=None):
            return {"top:TGraph": "top:Graph"}.get(value, defaultValue if defaultValue is not None else value)

    monkeypatch.setattr(KnowledgeGraph, "_ontology_class", staticmethod(lambda: FakeOntology))

    kg_graph = KnowledgeGraph.ByTopology("graph-like", includeOntologyAxioms=True, useRDFLib=False, accepted="ok", ignored="drop", silent=True)
    assert kg_graph is not None
    assert kg_graph.HasTriple(("inst:g", "rdf:type", "top:Graph"))
    assert kg_graph.HasTriple(("top:Graph", "rdf:type", "owl:Class"))
    assert calls[-1] == ("graph", "graph-like", "ok")

    kg_topology = KnowledgeGraph.ByTopology("face-like", useRDFLib=False, accepted="ok", silent=True)
    assert kg_topology.HasTriple(("inst:t", "rdf:type", "top:Face"))
    assert calls[-1] == ("topology", "face-like", "ok")

    assert KnowledgeGraph.ByTGraph("graph-like", useRDFLib=False, silent=True).HasTriple(("inst:g", "rdf:type", "top:Graph"))
    assert KnowledgeGraph.TriplesByTopology("face-like", useRDFLib=False, silent=True) == [("inst:t", "rdf:type", "top:Face")]


def test_to_tgraph_and_static_tgraph_wrapper(monkeypatch):
    class FakeTGraph:
        def __init__(self, directed=True, allowSelfLoops=False, allowParallelEdges=False, dictionary=None):
            self.directed = directed
            self.allowSelfLoops = allowSelfLoops
            self.allowParallelEdges = allowParallelEdges
            self.dictionary = dictionary or {}
            self.vertices = []
            self.edges = []

        def AddVertex(self, dictionary=None):
            self.vertices.append(dict(dictionary or {}))
            return len(self.vertices) - 1

        def AddEdge(self, src, dst, directed=True, dictionary=None):
            self.edges.append({"src": src, "dst": dst, "directed": directed, "dictionary": dict(dictionary or {})})
            return len(self.edges) - 1

    monkeypatch.setattr(KnowledgeGraph, "_tgraph_class", staticmethod(lambda: FakeTGraph))

    kg = _kg(("inst:a", "dict:label", "Alpha"), ("inst:a", "top:connectsTo", "inst:b"))
    graph = kg.ToTGraph(includeLiterals=True, directed=True)

    assert isinstance(graph, FakeTGraph)
    assert graph.directed is True
    assert len(graph.vertices) == 3
    assert len(graph.edges) == 2
    assert any(v["category"] == "literal" and v["label"] == "Alpha" for v in graph.vertices)
    assert any(e["dictionary"]["predicate"] == "top:connectsTo" for e in graph.edges)

    no_literals = kg.ToTGraph(includeLiterals=False)
    assert len(no_literals.vertices) == 2
    assert len(no_literals.edges) == 1

    wrapped = KnowledgeGraph.TGraphByKnowledgeGraph(kg.Dictionary(), includeLiterals=False, silent=True)
    assert isinstance(wrapped, FakeTGraph)
    assert len(wrapped.edges) == 1


def test_tgraph_static_fallback_when_ontology_is_unavailable(monkeypatch):
    class FakeTGraph:
        @staticmethod
        def OntologyTriples(graph, includeDictionaries=True, includeBOT=True, namespacePrefix="inst"):
            return [("inst:fallback", "rdf:type", "top:Graph")]

    monkeypatch.setattr(KnowledgeGraph, "_ontology_class", staticmethod(lambda: None))
    monkeypatch.setattr(KnowledgeGraph, "_tgraph_class", staticmethod(lambda: FakeTGraph))

    kg = KnowledgeGraph.ByTopology(FakeTGraph(), useRDFLib=False, silent=True)
    assert kg is not None
    assert kg.HasTriple(("inst:fallback", "rdf:type", "top:Graph"))


def test_infer_bridge_with_fake_reasoner(monkeypatch):
    class FakeReasoner:
        @staticmethod
        def Infer(rdf, profile="rdfs", includeOntologyAxioms=True, includeBOT=True, silent=False, **kwargs):
            assert rdf == "fake-rdf"
            assert profile == "rdfs"
            assert includeOntologyAxioms is True
            assert includeBOT is False
            assert kwargs["extra"] == 7
            return "fake-inferred-rdf"

    monkeypatch.setattr(KnowledgeGraph, "_reasoner_class", staticmethod(lambda: FakeReasoner))
    monkeypatch.setattr(KnowledgeGraph, "RDFGraph", lambda self, silent=False: "fake-rdf")
    monkeypatch.setattr(
        KnowledgeGraph,
        "ByRDFGraph",
        staticmethod(lambda rdfGraph, namespaces=None, silent=False: _kg(("inst:inferred", "rdf:type", "top:Graph"))),
    )

    kg = _kg(("inst:a", "rdf:type", "top:Graph"))
    inferred = kg.Infer(includeBOT=False, extra=7, silent=True)
    assert inferred is not None
    assert inferred.HasTriple(("inst:inferred", "rdf:type", "top:Graph"))

    kg.Infer(includeBOT=False, inplace=True, extra=7, silent=True)
    assert kg.HasTriple(("inst:inferred", "rdf:type", "top:Graph"))


def test_query_and_update_error_paths_do_not_require_rdflib(monkeypatch):
    kg = _kg(("inst:a", "rdf:type", "top:Graph"))
    monkeypatch.setattr(KnowledgeGraph, "_rdflib", staticmethod(lambda silent=False: None))

    assert kg.RDFGraph(rebuild=True, silent=True) is None
    assert kg.RDFString(format="xml", silent=True) is None
    assert kg.Query("ASK WHERE { ?s ?p ?o }", silent=True) is None
    assert kg.Update("INSERT DATA { inst:a top:label \"A\" . }", silent=True) is None


