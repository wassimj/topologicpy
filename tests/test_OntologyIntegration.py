import pytest

from topologicpy.KnowledgeGraph import KnowledgeGraph
from topologicpy.Ontology import Ontology
from topologicpy.Reasoner import Reasoner
from topologicpy.TGraph import TGraph


def _active_vertices(graph):
    return [v for v in graph._vertices if v.get("active", True)]


def _active_edges(graph):
    return [e for e in graph._edges if e.get("active", True)]


def test_canonical_graph_relationship_protocol_round_trip():
    pytest.importorskip("rdflib")
    g = TGraph(
        directed=True,
        dictionary={"uri": "inst:g", "ontology_class": "top:Graph"},
    )
    a = g.AddVertex(dictionary={"uri": "inst:a", "ontology_class": "top:Node"})
    b = g.AddVertex(dictionary={"uri": "inst:b", "ontology_class": "top:Node"})
    g.AddEdge(a, b, directed=True, dictionary={
        "uri": "inst:r",
        "ontology_class": "top:Relationship",
        "ontology_predicate": "brick:feeds",
        "inverse_predicate": "brick:isFedBy",
    })

    triples = Ontology.GraphTriples(g)
    assert ("inst:r", "top:startsAt", "inst:a") in triples
    assert ("inst:r", "top:endsAt", "inst:b") in triples
    assert ("inst:r", "top:hasPredicate", "brick:feeds") in triples
    assert ("inst:r", "top:hasInversePredicate", "brick:isFedBy") in triples
    assert ("inst:a", "brick:feeds", "inst:b") in triples
    assert ("inst:b", "brick:isFedBy", "inst:a") in triples

    rdf = Ontology.RDFGraph(g)
    rebuilt = Ontology.GraphByRDFGraph(rdf)
    assert isinstance(rebuilt, TGraph)
    assert len(_active_vertices(rebuilt)) == 2
    assert len(_active_edges(rebuilt)) == 1
    ed = _active_edges(rebuilt)[0]["dictionary"]
    assert ed["ontology_predicate"] == "brick:feeds"
    assert ed["inverse_predicate"] == "brick:isFedBy"


def test_reasoner_uses_only_canonical_ontology_axioms():
    pytest.importorskip("rdflib")
    before = Reasoner.RDFGraphByTriples([
        ("inst:room", "rdf:type", "top:Room"),
    ])
    after = Reasoner.Infer(before, profile="rdfs", includeOntologyAxioms=True)
    types = Reasoner.Types(after, "inst:room")
    assert "top:Room" in types
    assert "top:Space" in types
    assert "top:TGraph" not in types


def test_arbitrary_rdf_literal_fidelity_through_knowledge_graph():
    pytest.importorskip("rdflib")
    from rdflib import Graph, Literal, URIRef
    from rdflib.namespace import RDFS, XSD

    s = URIRef("http://example.org/resource")
    area = URIRef("http://w3id.org/topologicpy#area")
    source = Graph()
    source.add((s, RDFS.label, Literal("Kitchen", lang="en")))
    source.add((s, area, Literal("12.5", datatype=XSD.double)))

    kg = KnowledgeGraph.ByRDFGraph(source)
    rebuilt = kg.RDFGraph(rebuild=True)
    assert set(source) == set(rebuilt)


def test_duplicate_labels_never_collapse_graph_resources():
    rdflib = pytest.importorskip("rdflib")
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import RDF, RDFS

    TOP = Namespace("http://w3id.org/topologicpy#")
    INST = Namespace("http://w3id.org/topologicpy/instance#")
    g = Graph()
    g.add((INST.g, RDF.type, TOP.Graph))
    g.add((INST.g, TOP.hasNode, INST.a))
    g.add((INST.g, TOP.hasNode, INST.b))
    g.add((INST.a, RDF.type, TOP.Node))
    g.add((INST.b, RDF.type, TOP.Node))
    g.add((INST.a, RDFS.label, Literal("Same")))
    g.add((INST.b, RDFS.label, Literal("Same")))

    rebuilt = Ontology.GraphByRDFGraph(g)
    assert isinstance(rebuilt, TGraph)
    vertices = _active_vertices(rebuilt)
    assert len(vertices) == 2
    uris = {v["dictionary"].get("_rdf_uri") or v["dictionary"].get("uri") for v in vertices}
    assert len(uris) == 2


def test_clean_slate_vocabulary_has_no_removed_terms():
    ttl = Ontology.OntologyTTLString()
    assert isinstance(ttl, str)
    for removed in (
        "top:TGraph",
        "top:hasStartVertex",
        "top:hasEndVertex",
        "top:hasX",
        "top:hasY",
        "top:hasZ",
        "top:hasArea",
        "top:hasVolume",
        "top:srcId",
        "top:dstId",
    ):
        assert removed not in ttl


def test_false_bot_inverse_pairs_are_not_in_canonical_graph_protocol():
    triples = Ontology.OntologyTriples()
    text = "\n".join(" ".join(map(str, t)) for t in triples)
    assert "bot:hasElement" not in text
    assert "bot:interfaceOf" not in text
