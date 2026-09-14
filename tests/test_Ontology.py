import sys
import types
from pathlib import Path

import pytest

from topologicpy.Ontology import Ontology, _RDFLiteral
from topologicpy.TGraph import TGraph


def _reset_vocab():
    Ontology._VOCAB_CACHE = None
    Ontology.TOP_SUPERCLASSES = {}
    Ontology.OBJECT_PROPERTIES = {}
    Ontology.DATA_PROPERTIES = {}
    Ontology.ANNOTATION_PROPERTIES = {}


def test_ontology_triples_are_unique_and_include_expected_schema_terms():
    rdflib = pytest.importorskip("rdflib")
    _reset_vocab()
    triples = Ontology.OntologyTriples()
    assert len(triples) == len({repr(t) for t in triples})
    assert any(s == "top:Graph" and p == "rdf:type" and o == "owl:Class" for s, p, o in triples)
    assert any(s == "top:startsAt" and p == "rdf:type" and o == "owl:ObjectProperty" for s, p, o in triples)
    assert any(s == "top:generatedByMethod" and p == "rdf:type" and o == "owl:DatatypeProperty" for s, p, o in triples)
    # Clean-slate vocabulary: no deprecated aliases.
    text = Ontology.OntologyTTLString()
    assert "top:TGraph" not in text
    assert "top:hasX" not in text
    assert "top:hasStartVertex" not in text
    assert "top:srcId" not in text
    assert "top:dstId" not in text


def test_invalid_inputs_return_documented_fallbacks():
    assert Ontology.SetClass({}, "", silent=True) is None
    assert Ontology.SetClass({}, "top:DefinitelyNotAClass", silent=True) is None
    assert Ontology.SetValue(None, "x", 1, silent=True) is None
    assert Ontology.SetURI({}, "", silent=True) is None
    assert Ontology.CanonicalClass("top:DefinitelyNotAClass", defaultValue=None) is None
    assert Ontology.PropertyQName("top:definitelyNotAProperty") is None


def test_namespace_qname_class_and_resource_helpers():
    assert Ontology.Namespace("top") == "http://w3id.org/topologicpy#"
    assert Ontology.ExpandQName("top:Room") == "http://w3id.org/topologicpy#Room"
    assert Ontology.QName("http://w3id.org/topologicpy#Room") == "top:Room"
    assert Ontology.IsQName("top:Room") is True
    assert Ontology.IsQName("Room") is False
    assert Ontology.IsClass("top:Room") is True
    assert Ontology.IsProperty("top:area") is True
    assert Ontology.IsResourceString("top:Room") is True
    assert Ontology.IsResourceString("https://example.org/a") is True
    assert Ontology.IsResourceString("Room") is False


def test_hierarchy_category_ifc_and_bot_mappings():
    assert Ontology.CanonicalClass("top:Room") == "top:Room"
    assert Ontology.CategoryByClass("top:Room") == "space"
    assert Ontology.ClassByIFCClass("IfcSpace") == "top:Space"
    assert Ontology.BOTClassByClass("top:Room") == "bot:Space"
    assert Ontology.IsA({"ontology_class": "top:Room"}, "top:Space") is True


def test_plain_dictionary_annotation_setters_and_validation_signature():
    d = {}
    assert Ontology.SetClass(d, "top:Room") is d
    assert d["ontology_class"] == "top:Room"
    assert d["category"] == "space"
    assert Ontology.SetLabel(d, "Room 101") is d
    assert Ontology.SetCategory(d, "space") is d
    assert Ontology.SetURI(d, "inst:room_101") is d
    assert d["label"] == "Room 101"
    assert d["uri"] == "inst:room_101"
    report = Ontology.Validate(d)
    assert isinstance(report, dict)
    assert report["valid"] is True
    assert report["ok"] is True


def test_validate_reports_missing_required_keys_and_unknown_top_terms_as_errors():
    d = {"ontology_class": "top:Room"}
    report = Ontology.Validate(d, requiredKeys=["label"])
    assert report["valid"] is False
    assert any("Missing required key: label" in e for e in report["errors"])

    bad = {"ontology_class": "top:DefinitelyNotAClass"}
    report = Ontology.Validate(bad)
    assert report["valid"] is False
    assert any("Unknown top: class" in e for e in report["errors"])


def test_normalize_dictionary_on_plain_python_dict_and_ifc_annotation():
    d = {
        "IFC_type": "IfcSpace",
        "IFC_global_id": "GUID-1",
        "IFC_name": "Room A",
        "ontology_uri": "top:Room",
    }
    assert Ontology.NormalizeDictionary(d) is d
    assert d["ifc_class"] == "IfcSpace"
    assert d["ifc_guid"] == "GUID-1"
    assert d["label"] == "Room A"
    assert d["ontology_class"] == "top:Space"
    assert d["category"] == "space"
    assert "ontology_uri" not in d

    e = {}
    assert Ontology.AnnotateIFC(e, ifcClass="IfcDoor", ifcGUID="D-1", ifcName="Door") is e
    assert e["ontology_class"] == "top:Door"
    assert e["ifc_guid"] == "D-1"
    assert e["label"] == "Door"


def test_set_value_preserves_returned_legacy_topology_object(monkeypatch):
    class FakeTopologyObject:
        pass

    original = FakeTopologyObject()
    replacement = FakeTopologyObject()
    store = {id(original): {"a": 1}}

    class FakeDictionary:
        @staticmethod
        def PythonDictionary(d):
            return dict(d)

        @staticmethod
        def ByPythonDictionary(d):
            return dict(d)

    class FakeTopology:
        @staticmethod
        def Dictionary(obj):
            return store.get(id(obj), {})

        @staticmethod
        def SetDictionary(obj, d):
            store[id(replacement)] = dict(d)
            return replacement

    mod_topology = types.ModuleType("topologicpy.Topology")
    mod_topology.Topology = FakeTopology
    mod_dict = types.ModuleType("topologicpy.Dictionary")
    mod_dict.Dictionary = FakeDictionary
    monkeypatch.setitem(sys.modules, "topologicpy.Topology", mod_topology)
    monkeypatch.setitem(sys.modules, "topologicpy.Dictionary", mod_dict)

    result = Ontology.SetValue(original, "b", 2)
    assert result is replacement
    assert store[id(replacement)]["a"] == 1
    assert store[id(replacement)]["b"] == 2


def test_class_by_topology_and_inferred_annotation(monkeypatch):
    class FakeVertex:
        pass

    class FakeTopology:
        @staticmethod
        def IsInstance(obj, name):
            return isinstance(obj, FakeVertex) and name == "Vertex"

    mod = types.ModuleType("topologicpy.Topology")
    mod.Topology = FakeTopology
    monkeypatch.setitem(sys.modules, "topologicpy.Topology", mod)

    assert Ontology.ClassByTopology(FakeVertex()) == "top:Vertex"

    d = {}
    result = Ontology.Annotate(d, ontologyClass="top:Room", inferClass=True)
    assert result is d
    assert d["ontology_class"] == "top:Room"


def test_property_literals_resource_objects_and_turtle_serialization():
    d = {
        "uri": "inst:room-1",
        "ontology_class": "top:Room",
        "label": "Room",
        "area": 12.5,
        "generated_by": "UnitTest",
        "derived_from": "inst:source-1",
    }
    triples = Ontology.Triples(d, includeBOT=False)
    assert ("inst:room-1", "rdf:type", "top:Room") in triples
    assert any(s == "inst:room-1" and p == "top:area" and isinstance(o, _RDFLiteral) for s, p, o in triples)
    assert any(s == "inst:room-1" and p == "top:generatedByMethod" for s, p, o in triples)
    assert ("inst:room-1", "prov:wasDerivedFrom", "inst:source-1") in triples
    ttl = Ontology.TurtleFromTriples(triples)
    assert isinstance(ttl, str)
    assert "top:generatedByMethod" in ttl
    assert "top:area" in ttl


def test_triples_use_dictionary_identity_bot_mapping_and_deduplication():
    d = {
        "uri": "inst:room-1",
        "ontology_class": "top:Room",
        "label": "Room",
        "category": "space",
        "ifc_guid": "GUID-1",
    }
    triples = Ontology.Triples(d, includeBOT=True)
    assert triples.count(("inst:room-1", "rdf:type", "top:Room")) == 1
    assert triples.count(("inst:room-1", "rdf:type", "bot:Space")) == 1
    assert not any(p in {"top:ontologyClass", "top:ontologyURI"} for _, p, _ in triples)


def test_tgraph_helpers_graph_triples_and_validate_graph():
    g = TGraph(directed=True, allowSelfLoops=False, allowParallelEdges=True,
               dictionary={"uri": "inst:g", "ontology_class": "top:Graph"})
    a = g.AddVertex(dictionary={"uri": "inst:a", "ontology_class": "top:Node", "x": 0.0, "y": 0.0, "z": 0.0})
    b = g.AddVertex(dictionary={"uri": "inst:b", "ontology_class": "top:Node", "x": 1.0, "y": 0.0, "z": 0.0})
    g.AddEdge(a, b, directed=True, dictionary={
        "uri": "inst:r0",
        "ontology_class": "top:Relationship",
        "ontology_predicate": "brick:feeds",
        "inverse_predicate": "brick:isFedBy",
    })
    triples = Ontology.GraphTriples(g)
    assert ("inst:g", "rdf:type", "top:Graph") in triples
    assert ("inst:r0", "top:startsAt", "inst:a") in triples
    assert ("inst:r0", "top:endsAt", "inst:b") in triples
    assert ("inst:r0", "top:hasPredicate", "brick:feeds") in triples
    assert ("inst:a", "brick:feeds", "inst:b") in triples
    assert ("inst:b", "brick:isFedBy", "inst:a") in triples
    report = Ontology.ValidateGraph(g)
    assert report["valid"] is True


def test_validate_graph_reports_unresolved_tgraph_edge_endpoints():
    g = TGraph(dictionary={"ontology_class": "top:Graph"})
    a = g.AddVertex(dictionary={"ontology_class": "top:Node"})
    b = g.AddVertex(dictionary={"ontology_class": "top:Node"})
    e = g.AddEdge(a, b, dictionary={"ontology_class": "top:Relationship"})
    g._edges[e]["dst"] = 9999
    report = Ontology.ValidateGraph(g)
    assert report["valid"] is False
    assert any("unresolved target endpoint" in e for e in report["errors"])


def test_annotate_subtopologies_uses_fake_topology_extractors(monkeypatch):
    vertex = {}
    edge = {}

    class FakeTopology:
        @staticmethod
        def Vertices(topology):
            return [vertex]

        @staticmethod
        def Edges(topology):
            return [edge]

    mod = types.ModuleType("topologicpy.Topology")
    mod.Topology = FakeTopology
    monkeypatch.setitem(sys.modules, "topologicpy.Topology", mod)

    owner = object()
    result = Ontology.AnnotateSubtopologies(owner, topologyTypes=["Vertex", "Edge"])
    assert result is owner
    assert vertex["ontology_class"] == "top:Vertex"
    assert edge["ontology_class"] == "top:Edge"


def test_ttl_string_export_ttl_and_ontology_ttl(tmp_path):
    d = {"uri": "inst:a", "ontology_class": "top:Graph", "label": "A"}
    ttl = Ontology.TTLString(d)
    assert isinstance(ttl, str)
    assert "top:Graph" in ttl

    p1 = tmp_path / "instance.ttl"
    assert Ontology.ExportTTL(d, str(p1)) == str(p1)
    assert p1.exists()

    p2 = tmp_path / "ontology.ttl"
    assert Ontology.ExportOntologyTTL(str(p2)) == str(p2)
    assert "owl:Ontology" in p2.read_text(encoding="utf-8")


def test_validate_ttl_and_rdf_methods_without_rdflib(monkeypatch):
    monkeypatch.setattr(Ontology, "_rdflib", staticmethod(lambda silent=False: None))
    report = Ontology.ValidateTTLString("@prefix top: <http://w3id.org/topologicpy#> .")
    assert report["available"] is False
    assert report["valid"] is None
    report2 = Ontology.ValidateRDFGraph(object())
    assert report2["available"] is False
    assert report2["valid"] is None


def test_labels_never_determine_identity():
    a = {"label": "Same Label", "index": 1}
    b = {"label": "Same Label", "index": 2}
    assert Ontology._identity(a, role="node", fallbackIndex=1) != Ontology._identity(b, role="node", fallbackIndex=2)


def test_unknown_dictionary_keys_go_to_dict_namespace():
    d = {"uri": "inst:a", "ontology_class": "top:Graph", "my_custom_value": 42}
    triples = Ontology.Triples(d, includeBOT=False)
    assert any(p == "dict:my_custom_value" for _, p, _ in triples)
