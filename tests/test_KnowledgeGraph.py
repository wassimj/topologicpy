import json
import types

import pytest

from topologicpy.KnowledgeGraph import KnowledgeGraph
from topologicpy.Ontology import Ontology
from topologicpy.TGraph import TGraph


def test_core_triple_store_add_filter_remove_clear_and_iterate():
    kg = KnowledgeGraph(useRDFLib=False)
    assert kg.AddTriple("inst:a", "rdf:type", "top:Graph") == ("inst:a", "rdf:type", "top:Graph")
    assert kg.AddTriple("inst:a", "label", "Alpha", objectIsLiteral=True) == (
        "inst:a", "rdfs:label", '"Alpha"'
    )
    assert len(kg) == 2
    assert kg.HasTriple(("inst:a", "rdf:type", "top:Graph"))
    assert kg.Triples(subject="inst:a") == sorted(list(kg))
    assert set(kg.Predicates()) == {"rdf:type", "rdfs:label"}
    assert set(kg.Objects(predicate="rdf:type")) == {"top:Graph"}
    assert kg.RemoveTriple("inst:a", "rdf:type", "top:Graph") is True
    assert not kg.HasTriple(("inst:a", "rdf:type", "top:Graph"))
    assert len(kg) == 1
    assert kg.Clear() is kg
    assert len(kg) == 0


def test_dictionary_json_and_native_json_file_round_trip(tmp_path):
    kg = KnowledgeGraph.ByTriples(
        [
            ("inst:a", "rdf:type", "top:Graph"),
            ("inst:a", "rdfs:label", '"Alpha"'),
        ],
        useRDFLib=False,
    )
    data = kg.Dictionary()
    assert len(data["triples"]) == 2

    copied = KnowledgeGraph.ByDictionary(data, useRDFLib=False)
    assert copied.Triples() == kg.Triples()

    restored = KnowledgeGraph.ByJSON(kg.JSONString(), useRDFLib=False)
    assert restored.Triples() == kg.Triples()

    path = tmp_path / "kg.json"
    assert kg.Export(str(path), format="json") == str(path)
    loaded = KnowledgeGraph.ByFile(str(path), useRDFLib=False)
    assert loaded.Triples() == kg.Triples()


def test_turtle_fallback_export_and_static_export_wrappers(tmp_path, monkeypatch):
    kg = KnowledgeGraph.ByTriples(
        [
            ("inst:a", "rdf:type", "top:Graph"),
            ("inst:a", "rdfs:label", '"Alpha"'),
        ],
        useRDFLib=False,
    )
    monkeypatch.setattr(KnowledgeGraph, "_rdflib", staticmethod(lambda silent=False: None))
    monkeypatch.setattr(Ontology, "_rdflib", staticmethod(lambda silent=False: None))
    ttl = kg.TurtleString()
    assert "top:Graph" in ttl
    assert "Alpha" in ttl

    p = tmp_path / "kg.ttl"
    assert KnowledgeGraph.ExportTTL(kg, str(p)) == str(p)
    assert "top:Graph" in p.read_text(encoding="utf-8")


def test_summary_and_validate_without_requiring_rdflib(monkeypatch):
    kg = KnowledgeGraph.ByTriples(
        [
            ("inst:a", "rdf:type", "top:Graph"),
            ("inst:a", "rdfs:label", '"Alpha"'),
        ],
        useRDFLib=False,
    )
    monkeypatch.setattr(KnowledgeGraph, "_rdflib", staticmethod(lambda silent=False: None))
    summary = kg.Summary()
    assert summary["triple_count"] == 2
    assert summary["subject_count"] == 1
    report = kg.Validate(parseWithRDFLib=False)
    assert report["valid"] is True
    assert report["triple_count"] == 2


def test_by_topology_uses_fake_ontology_and_filters_kwargs(monkeypatch):
    class FakeObject:
        pass

    class FakeOntology:
        @staticmethod
        def _is_graph_like(obj):
            return False

        @staticmethod
        def Triples(topology, includeDictionaries=True, includeBOT=True, namespacePrefix="inst", silent=False, accepted=None):
            assert isinstance(topology, FakeObject)
            assert accepted == 42
            return [
                ("inst:x", "rdf:type", "top:Graph"),
                ("inst:x", "rdfs:label", '"X"'),
            ]

        @staticmethod
        def Namespaces():
            return KnowledgeGraph.Namespaces()

    monkeypatch.setattr(KnowledgeGraph, "_ontology_class", staticmethod(lambda: FakeOntology))
    kg = KnowledgeGraph.ByTopology(FakeObject(), useRDFLib=False, accepted=42, ignored=99)
    assert kg is not None
    assert len(kg) == 2
    assert kg.HasTriple(("inst:x", "rdf:type", "top:Graph"))


def test_by_topology_requires_ontology_as_single_semantic_authority(monkeypatch):
    monkeypatch.setattr(KnowledgeGraph, "_ontology_class", staticmethod(lambda: None))
    assert KnowledgeGraph.ByTopology(object(), useRDFLib=False, silent=True) is None


def test_by_tgraph_uses_canonical_ontology_export():
    g = TGraph(
        directed=True,
        dictionary={"uri": "inst:g", "ontology_class": "top:Graph", "label": "G"},
    )
    a = g.AddVertex(dictionary={"uri": "inst:a", "ontology_class": "top:Node"})
    b = g.AddVertex(dictionary={"uri": "inst:b", "ontology_class": "top:Node"})
    g.AddEdge(
        a,
        b,
        directed=True,
        dictionary={
            "uri": "inst:r0",
            "ontology_class": "top:Relationship",
            "ontology_predicate": "brick:feeds",
        },
    )
    kg = KnowledgeGraph.ByTGraph(g, useRDFLib=False)
    assert kg is not None
    assert kg.HasTriple(("inst:g", "rdf:type", "top:Graph"))
    assert kg.HasTriple(("inst:r0", "top:hasPredicate", "brick:feeds"))
    assert kg.HasTriple(("inst:a", "brick:feeds", "inst:b"))


def test_to_tgraph_semantic_projection_preserves_exact_predicate():
    pytest.importorskip("rdflib")
    kg = KnowledgeGraph.ByTriples(
        [
            ("inst:a", "rdf:type", "top:Node"),
            ("inst:b", "rdf:type", "top:Node"),
            ("inst:a", "brick:feeds", "inst:b"),
        ]
    )
    graph = kg.ToTGraph(includeLiterals=False)
    assert isinstance(graph, TGraph)
    assert len([e for e in graph._edges if e.get("active", True)]) >= 1
    predicates = [
        e.get("dictionary", {}).get("ontology_predicate")
        for e in graph._edges
        if e.get("active", True)
    ]
    assert "brick:feeds" in predicates


def test_static_tgraph_wrapper():
    pytest.importorskip("rdflib")
    kg = KnowledgeGraph.ByTriples([
        ("inst:a", "brick:feeds", "inst:b"),
    ])
    graph = KnowledgeGraph.TGraphByKnowledgeGraph(kg, includeLiterals=False)
    assert isinstance(graph, TGraph)


def test_infer_bridge_with_fake_reasoner(monkeypatch):
    pytest.importorskip("rdflib")

    class FakeReasoner:
        @staticmethod
        def Infer(rdfGraph, **kwargs):
            out = KnowledgeGraph.ByRDFGraph(rdfGraph, silent=True)
            out.AddTriple("inst:inferred", "rdf:type", "top:Graph")
            return out

    monkeypatch.setattr(KnowledgeGraph, "_reasoner_class", staticmethod(lambda: FakeReasoner))
    kg = KnowledgeGraph.ByTriples([("inst:a", "rdf:type", "top:Graph")])
    inferred = kg.Infer()
    assert isinstance(inferred, KnowledgeGraph)
    assert inferred.HasTriple(("inst:inferred", "rdf:type", "top:Graph"))


def test_explicit_unknown_top_predicate_is_preserved_but_validation_fails():
    kg = KnowledgeGraph.ByTriples(
        [("inst:a", "top:notDeclared", "inst:b")],
        useRDFLib=False,
    )
    assert kg.HasTriple(("inst:a", "top:notDeclared", "inst:b"))
    report = kg.Validate(parseWithRDFLib=False)
    assert report["valid"] is False
    assert "top:notDeclared" in report["unknown_top_predicates"]


def test_explicit_unknown_top_class_is_preserved_but_validation_fails():
    kg = KnowledgeGraph.ByTriples(
        [("inst:a", "rdf:type", "top:NotDeclared")],
        useRDFLib=False,
    )
    assert kg.HasTriple(("inst:a", "rdf:type", "top:NotDeclared"))
    report = kg.Validate(parseWithRDFLib=False)
    assert report["valid"] is False
    assert "top:NotDeclared" in report["unknown_top_classes"]


def test_bare_python_predicate_uses_canonical_ontology_or_dict_namespace():
    kg = KnowledgeGraph(useRDFLib=False)
    kg.AddTriple("inst:a", "generated_by", "method", objectIsLiteral=True)
    kg.AddTriple("inst:a", "my_custom_value", 42)
    assert kg.HasTriple(("inst:a", "top:generatedByMethod", '"method"'))
    assert kg.HasTriple(("inst:a", "dict:my_custom_value", '"42"^^xsd:integer'))


def test_explicit_legacy_top_terms_are_not_silently_rewritten():
    kg = KnowledgeGraph.ByTriples(
        [
            ("inst:a", "top:hasArea", 10.0),
            ("inst:a", "rdf:type", "top:TGraph"),
        ],
        useRDFLib=False,
    )
    assert kg.HasTriple(("inst:a", "top:hasArea", '"10.0"^^xsd:double'))
    assert kg.HasTriple(("inst:a", "rdf:type", "top:TGraph"))
    assert not kg.HasTriple(("inst:a", "top:area", '"10.0"^^xsd:double'))
    report = kg.Validate(parseWithRDFLib=False)
    assert report["valid"] is False
    assert "top:hasArea" in report["unknown_top_predicates"]
    assert "top:TGraph" in report["unknown_top_classes"]


def test_rdf_round_trip_preserves_language_and_datatype():
    rdflib = pytest.importorskip("rdflib")
    from rdflib import Graph, Literal, URIRef
    from rdflib.namespace import RDFS, XSD

    g = Graph()
    a = URIRef("http://w3id.org/topologicpy/instance#a")
    g.add((a, RDFS.label, Literal("Kitchen", lang="en")))
    g.add((a, URIRef("http://w3id.org/topologicpy#area"), Literal("12.5", datatype=XSD.double)))

    kg = KnowledgeGraph.ByRDFGraph(g)
    rebuilt = kg.RDFGraph(rebuild=True)
    assert (a, RDFS.label, Literal("Kitchen", lang="en")) in rebuilt
    assert (a, URIRef("http://w3id.org/topologicpy#area"), Literal("12.5", datatype=XSD.double)) in rebuilt


def test_merge_difference_and_diff_are_deterministic():
    a = KnowledgeGraph.ByTriples([("inst:a", "rdf:type", "top:Graph")], useRDFLib=False)
    b = KnowledgeGraph.ByTriples([("inst:b", "rdf:type", "top:Graph")], useRDFLib=False)
    merged = a.Merge(b)
    assert len(merged) == 2
    assert a.Difference(b) == [("inst:a", "rdf:type", "top:Graph")]
    diff = a.Diff(merged)
    assert diff["added"] == [("inst:b", "rdf:type", "top:Graph")]
    assert diff["removed"] == []
