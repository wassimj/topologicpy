"""Unit tests for topologicpy.Ontology."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class FakeTopologyObject:
    def __init__(self, type_name="vertex", dictionary=None):
        self.type_name = type_name
        self.dictionary = dict(dictionary or {})
        self.vertices = []
        self.edges = []
        self.subtopologies = {}
        self.was_returned_from_set_dictionary = False
        self.return_replacement = True


@pytest.fixture(autouse=True)
def _fake_topologicpy_modules(monkeypatch):
    """Install minimal fake TopologicPy modules used by Ontology's lazy imports."""

    dictionary_module = types.ModuleType("topologicpy.Dictionary")

    class Dictionary:
        @staticmethod
        def ValueAtKey(dictionary, key, default=None):
            if dictionary is None:
                return default
            try:
                return dictionary.get(key, default)
            except AttributeError:
                return default

        @staticmethod
        def SetValueAtKey(dictionary, key, value):
            dictionary = dict(dictionary or {})
            dictionary[key] = value
            return dictionary

        @staticmethod
        def ByPythonDictionary(dictionary):
            return dict(dictionary or {})

        @staticmethod
        def PythonDictionary(dictionary):
            return dict(dictionary or {})

        @staticmethod
        def Keys(dictionary):
            return list((dictionary or {}).keys())

    dictionary_module.Dictionary = Dictionary

    topology_module = types.ModuleType("topologicpy.Topology")

    class Topology:
        @staticmethod
        def Dictionary(topology):
            return getattr(topology, "dictionary", None)

        @staticmethod
        def SetDictionary(topology, dictionary, silent=True):
            # Return a replacement object to verify Ontology preserves returned topology objects.
            if isinstance(topology, FakeTopologyObject):
                if getattr(topology, "return_replacement", True):
                    replacement = FakeTopologyObject(topology.type_name, dictionary)
                    replacement.vertices = topology.vertices
                    replacement.edges = topology.edges
                    replacement.subtopologies = topology.subtopologies
                    replacement.return_replacement = topology.return_replacement
                    replacement.was_returned_from_set_dictionary = True
                    return replacement
                topology.dictionary = dict(dictionary or {})
                return topology
            try:
                topology.dictionary = dict(dictionary or {})
            except Exception:
                pass
            return topology

        @staticmethod
        def TypeAsString(topology):
            return getattr(topology, "type_name", "topology")

        @staticmethod
        def IsInstance(topology, type_name):
            if topology is None:
                return False
            return str(getattr(topology, "type_name", "")).lower() == str(type_name).lower()

        @staticmethod
        def UUID(topology, silent=True):
            return getattr(topology, "uuid", "uuid-123")

        @staticmethod
        def Vertices(topology):
            return list(getattr(topology, "subtopologies", {}).get("vertices", []))

        @staticmethod
        def Edges(topology):
            return list(getattr(topology, "subtopologies", {}).get("edges", []))

        @staticmethod
        def Wires(topology):
            return list(getattr(topology, "subtopologies", {}).get("wires", []))

        @staticmethod
        def Faces(topology):
            return list(getattr(topology, "subtopologies", {}).get("faces", []))

        @staticmethod
        def Shells(topology):
            return list(getattr(topology, "subtopologies", {}).get("shells", []))

        @staticmethod
        def Cells(topology):
            return list(getattr(topology, "subtopologies", {}).get("cells", []))

        @staticmethod
        def CellComplexes(topology):
            return list(getattr(topology, "subtopologies", {}).get("cellComplexes", []))

        @staticmethod
        def SubTopologies(topology, subTopologyType=None):
            key_map = {
                "vertex": "vertices",
                "edge": "edges",
                "wire": "wires",
                "face": "faces",
                "shell": "shells",
                "cell": "cells",
                "cellcomplex": "cellComplexes",
            }
            key = key_map.get(str(subTopologyType).lower(), subTopologyType)
            return list(getattr(topology, "subtopologies", {}).get(key, []))

    topology_module.Topology = Topology

    graph_module = types.ModuleType("topologicpy.Graph")

    class Graph:
        @staticmethod
        def Vertices(graph):
            return list(getattr(graph, "vertices", []))

        @staticmethod
        def Edges(graph):
            return list(getattr(graph, "edges", []))

        @staticmethod
        def ByVerticesEdges(vertices, edges, *args, **kwargs):
            graph = FakeTopologyObject("graph", {"ontology_class": "top:Graph", "category": "graph"})
            graph.vertices = list(vertices or [])
            graph.edges = list(edges or [])
            return graph

        @staticmethod
        def StartVertex(graph, edge):
            return getattr(edge, "start", None)

        @staticmethod
        def EndVertex(graph, edge):
            return getattr(edge, "end", None)

    graph_module.Graph = Graph

    vertex_module = types.ModuleType("topologicpy.Vertex")

    class Vertex:
        @staticmethod
        def ByCoordinates(x, y, z):
            return FakeTopologyObject("vertex", {"x": x, "y": y, "z": z})

    vertex_module.Vertex = Vertex

    edge_module = types.ModuleType("topologicpy.Edge")

    class Edge:
        @staticmethod
        def ByStartVertexEndVertex(start, end):
            edge = FakeTopologyObject("edge", {})
            edge.start = start
            edge.end = end
            return edge

        @staticmethod
        def ByVertices(vertices, *args, **kwargs):
            edge = FakeTopologyObject("edge", {})
            edge.start = vertices[0]
            edge.end = vertices[1]
            return edge

        @staticmethod
        def StartVertex(edge):
            return getattr(edge, "start", None)

        @staticmethod
        def EndVertex(edge):
            return getattr(edge, "end", None)

    edge_module.Edge = Edge

    tgraph_module = types.ModuleType("topologicpy.TGraph")

    class TGraph:
        def __init__(self, dictionary=None, vertices=None, edges=None):
            self.dictionary = dict(dictionary or {})
            self.vertices = list(vertices or [])
            self.edges = list(edges or [])

        def SetDictionary(self, dictionary):
            self.dictionary = dict(dictionary or {})
            return self

        def SetVertexDictionary(self, index, dictionary):
            for vertex in self.vertices:
                if vertex.get("index") == index:
                    vertex["dictionary"] = dict(dictionary or {})
                    return True
            return False

        def SetEdgeDictionary(self, index, dictionary):
            for edge in self.edges:
                if edge.get("index") == index:
                    edge["dictionary"] = dict(dictionary or {})
                    return True
            return False

        @staticmethod
        def Dictionary(graph):
            return dict(getattr(graph, "dictionary", {}) or {})

        @staticmethod
        def Vertices(graph, asTopologic=False, active=True):
            return list(getattr(graph, "vertices", []) or [])

        @staticmethod
        def Edges(graph, asTopologic=False, active=True):
            return list(getattr(graph, "edges", []) or [])

        @staticmethod
        def ByVerticesEdges(vertices=None, edges=None, dictionary=None, **kwargs):
            return TGraph(dictionary=dictionary, vertices=vertices, edges=edges)

        @staticmethod
        def Coordinates(graph, index, default=None):
            for vertex in getattr(graph, "vertices", []) or []:
                if vertex.get("index") == index:
                    dictionary = vertex.get("dictionary", {})
                    if all(key in dictionary for key in ("x", "y", "z")):
                        return [dictionary["x"], dictionary["y"], dictionary["z"]]
            return default

    tgraph_module.TGraph = TGraph

    for name, module in {
        "topologicpy.Dictionary": dictionary_module,
        "topologicpy.Topology": topology_module,
        "topologicpy.Graph": graph_module,
        "topologicpy.Vertex": vertex_module,
        "topologicpy.Edge": edge_module,
        "topologicpy.TGraph": tgraph_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    yield


def _ontology():
    module = importlib.import_module("topologicpy.Ontology")
    return module.Ontology


def _fake_tgraph_class():
    return sys.modules["topologicpy.TGraph"].TGraph


def test_namespace_qname_class_and_resource_helpers():
    Ontology = _ontology()
    namespaces = Ontology.Namespaces()
    assert namespaces["top"].endswith("topologicpy#")
    assert namespaces["dict"].endswith("/dictionary#")
    assert Ontology.Namespace("missing", defaultValue="fallback") == "fallback"
    assert Ontology.ExpandQName("top:Room") == namespaces["top"] + "Room"
    assert Ontology.ExpandQName("bad:Room", defaultValue="fallback") == "fallback"
    assert Ontology.IsQName("top:Room") is True
    assert Ontology.IsQName("Room") is False
    assert Ontology.CanonicalClass("top:TGraph") == "top:Graph"
    assert Ontology.CanonicalClass("TGraph") == "top:Graph"
    assert Ontology.IsResourceString("https://example.org/a") is True
    assert Ontology.IsResourceString("top:Room") is True
    assert Ontology.IsResourceString("plain") is False


def test_hierarchy_category_ifc_and_bot_mappings():
    Ontology = _ontology()
    assert Ontology.CategoryByClass("top:Room") == "space"
    assert "top:Space" in Ontology.Superclasses("top:Room", transitive=True)
    assert "top:Zone" in Ontology.Superclasses("top:Room", transitive=True)
    assert Ontology.Superclasses("top:Room", transitive=False) == ["top:Space"]
    assert Ontology.ClassByIFCClass("IfcSpace") == "top:Space"
    assert Ontology.ClassByIFCClass("IfcUnknown") == "top:Element"
    assert Ontology.BOTClassByClass("top:Room") == "bot:Space"
    assert Ontology.BOTClassByClass("top:Column") == "bot:Element"


def test_plain_dictionary_annotation_setters_and_validation_signature():
    Ontology = _ontology()
    topology = {}
    returned = Ontology.SetClass(topology, "top:Room", silent=True)
    assert returned is topology
    assert topology["ontology_class"] == "top:Room"
    assert topology["category"] == "space"
    assert topology["ontology_uri"] == Ontology.ExpandQName("top:Room")

    Ontology.SetLabel(topology, "Room 101", silent=True)
    Ontology.SetURI(topology, "room-101", silent=True)
    assert Ontology.Class(topology) == "top:Room"
    assert Ontology.Category(topology) == "space"
    assert Ontology.Label(topology) == "Room 101"
    assert Ontology.URI(topology) == "room-101"

    report = Ontology.Validate(
        topology,
        requireClass=True,
        requireCategory=True,
        requireLabel=True,
        requireURI=True,
        silent=True,
    )
    assert report["ok"] is True
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["dictionary"]["label"] == "Room 101"


def test_validate_reports_missing_required_keys_and_warnings():
    Ontology = _ontology()
    report = Ontology.Validate(
        {"ontology_class": "top:NotBuiltIn", "category": "space"},
        requireClass=True,
        requireCategory=True,
        requireLabel=True,
        requireURI=True,
        silent=True,
    )
    assert report["ok"] is False
    assert report["valid"] is False
    assert any("Missing label" in error for error in report["errors"])
    assert any("Missing uri" in error for error in report["errors"])
    assert any("Unknown top" in warning for warning in report["warnings"])

    bad_prefix = Ontology.Validate({"ontology_class": "bad:Class"}, silent=True)
    assert bad_prefix["ok"] is True
    assert any("QName" in warning or "prefix" in warning for warning in bad_prefix["warnings"])


def test_normalize_dictionary_on_plain_python_dict_and_ifc_annotation():
    Ontology = _ontology()
    topology = {
        "Name": "Lobby",
        "ObjectType": "Public",
        "IfcClass": "IfcSpace",
        "GlobalId": "0ABC",
    }
    result = Ontology.NormalizeDictionary(topology, silent=True)
    assert result is topology
    assert topology["label"] == "Lobby"
    assert topology["category"] == "space"  # IFC class annotation overrides generic ObjectType category.
    assert topology["ifc_class"] == "IfcSpace"
    assert topology["ifc_guid"] == "0ABC"
    assert topology["ontology_class"] == "top:Space"

    door = Ontology.AnnotateIFC({}, ifcClass="IfcDoor", ifcGUID="D1", ifcName="Door A", source="model.ifc", silent=True)
    assert door["ontology_class"] == "top:Door"
    assert door["ifc_guid"] == "D1"
    assert door["label"] == "Door A"
    assert door["source"] == "model.ifc"


def test_set_value_preserves_returned_legacy_topology_object():
    Ontology = _ontology()
    original = FakeTopologyObject("vertex", {})
    returned = Ontology.SetLabel(original, "Returned Object", silent=True)
    assert returned is not original
    assert returned.was_returned_from_set_dictionary is True
    assert returned.dictionary["label"] == "Returned Object"
    assert original.dictionary == {}

    classified = Ontology.SetClass(returned, "top:Vertex", silent=True)
    assert classified.was_returned_from_set_dictionary is True
    assert classified.dictionary["ontology_class"] == "top:Vertex"
    assert classified.dictionary["category"] == "topology"


def test_class_by_topology_and_inferred_annotation():
    Ontology = _ontology()
    vertex = FakeTopologyObject("vertex", {})
    assert Ontology.ClassByTopology(vertex) == "top:Vertex"
    annotated = Ontology.Annotate(vertex, inferClass=True, label="V1", generatedBy="unit-test", silent=True)
    assert annotated.dictionary["ontology_class"] == "top:Vertex"
    assert annotated.dictionary["category"] == "topology"
    assert annotated.dictionary["label"] == "V1"
    assert annotated.dictionary["generated_by"] == "unit-test"

    assert Ontology.IsA(annotated, "top:Topology") is True
    assert Ontology.IsA(annotated, "top:Face") is False


def test_property_literals_resource_objects_and_turtle_serialization():
    Ontology = _ontology()
    assert Ontology.PropertyQName("generated_by") == "top:generatedByMethod"
    assert Ontology.PropertyQName("hasStartVertex") == "top:startsAt"
    assert Ontology.PropertyQName("ifc:globalId") == "ifc:globalId"

    ttl = Ontology.TurtleFromTriples(
        [
            ("inst:a", "rdf:type", "top:Room"),
            ("inst:a", "top:hasX", '"1"^^xsd:integer'),
            ("inst:a", "rdf:type", "top:Room"),
            ("https://example.org/s", "plain predicate", "plain object"),
            None,
        ]
    )
    assert ttl.count("inst:a rdf:type top:Room .") == 1
    assert "<https://example.org/s> dict:plain_predicate \"plain object\" ." in ttl
    assert "@prefix top:" in ttl


def test_triples_use_dictionary_identity_bot_mapping_and_deduplication():
    Ontology = _ontology()
    topology = {
        "id": "room-1",
        "ontology_class": "top:Room",
        "label": "Room",
        "category": "space",
        "ifc_guid": "GUID-1",
        "GlobalId": "GUID-1",
        "x": 1.0,
        "tags": ["a", "b"],
    }
    triples = Ontology.Triples(topology, includeBOT=True)
    assert triples.count(("inst:room-1", "rdf:type", "top:Room")) == 1
    assert ("inst:room-1", "rdf:type", "bot:Space") in triples
    assert ("inst:room-1", "rdfs:label", '"Room"') in triples
    assert ("inst:room-1", "top:category", '"space"') in triples
    assert triples.count(("inst:room-1", "top:ifcGUID", '"GUID-1"')) == 1
    assert ("inst:room-1", "top:x", '"1.0"^^xsd:double') in triples
    assert ("inst:room-1", "dict:tags", '"a"') in triples
    assert len(triples) == len(set(triples))


def test_tgraph_helpers_graph_triples_and_validate_graph():
    Ontology = _ontology()
    TGraph = _fake_tgraph_class()
    graph = TGraph(
        dictionary={"id": "g1", "ontology_class": "top:Graph", "label": "Graph 1"},
        vertices=[
            {"index": 10, "dictionary": {"id": "a", "ontology_class": "top:Node", "label": "A", "x": 1, "y": 2, "z": 3}},
            {"index": 20, "dictionary": {"id": "b", "ontology_class": "top:Node", "label": "B"}, "coordinates": [4, 5, 6]},
        ],
        edges=[
            {"index": 0, "src": 10, "dst": 20, "dictionary": {"id": "e1", "ontology_class": "top:Relationship", "label": "A-B", "ontology_predicate": "adjacentTo"}},
        ],
    )
    triples = Ontology.GraphTriples(graph)
    assert ("inst:g1", "top:hasNode", "inst:a") in triples
    assert ("inst:g1", "top:hasRelationship", "inst:e1") in triples
    assert ("inst:e1", "top:startsAt", "inst:a") in triples
    assert ("inst:e1", "top:endsAt", "inst:b") in triples
    assert ("inst:a", "top:connectsTo", "inst:b") in triples
    assert ("inst:a", "top:adjacentTo", "inst:b") in triples
    assert ("inst:b", "top:x", '"4.0"^^xsd:double') in triples
    assert len(triples) == len(set(triples))

    report = Ontology.ValidateGraph(graph, requireLabels=True, silent=True)
    assert report["ok"] is True
    assert report["errors"] == []
    assert len(report["vertices"]) == 2
    assert len(report["edges"]) == 1


def test_validate_graph_reports_unresolved_tgraph_edge_endpoints():
    Ontology = _ontology()
    TGraph = _fake_tgraph_class()
    graph = TGraph(
        dictionary={"id": "bad", "ontology_class": "top:Graph"},
        vertices=[{"index": 1, "dictionary": {"id": "a", "ontology_class": "top:Node"}}],
        edges=[{"index": 0, "src": 1, "dst": 99, "dictionary": {"id": "e", "ontology_class": "top:Relationship"}}],
    )
    report = Ontology.ValidateGraph(graph, checkConnectivity=True, silent=True)
    assert report["ok"] is False
    assert any("Could not resolve edge" in error for error in report["errors"])




def test_annotate_subtopologies_uses_fake_topology_extractors():
    Ontology = _ontology()
    host = FakeTopologyObject("cellcomplex", {})
    vertex = FakeTopologyObject("vertex", {})
    edge = FakeTopologyObject("edge", {})
    face = FakeTopologyObject("face", {})
    vertex.return_replacement = False
    edge.return_replacement = False
    face.return_replacement = False
    host.subtopologies = {"vertices": [vertex], "edges": [edge], "faces": [face]}

    result = Ontology.AnnotateSubtopologies(host, wires=False, shells=False, cells=False, cellComplexes=False, silent=True)
    assert result is host
    assert vertex.dictionary["ontology_class"] == "top:Vertex"
    assert edge.dictionary["ontology_class"] == "top:Edge"
    assert face.dictionary["ontology_class"] == "top:Face"


def test_ttl_string_export_ttl_and_ontology_ttl(tmp_path):
    Ontology = _ontology()
    topology = {"id": "wall-1", "ontology_class": "top:Wall", "label": "Wall", "category": "element"}
    ttl = Ontology.TTLString(topology, includeGraph=False, includeDictionaries=True, includeBOT=True, silent=True)
    assert isinstance(ttl, str)
    assert "inst:wall-1 rdf:type top:Wall ." in ttl
    assert "inst:wall-1 rdf:type bot:Element ." in ttl

    export_path = tmp_path / "wall.ttl"
    assert Ontology.ExportTTL(topology, str(export_path), includeGraph=False, silent=True) == str(export_path)
    assert "top:Wall" in export_path.read_text(encoding="utf-8")

    ontology_ttl = Ontology.OntologyTTLString(includeBOT=False)
    assert "owl:Ontology" in ontology_ttl
    assert "top:Graph" in ontology_ttl

    ontology_path = tmp_path / "ontology.ttl"
    assert Ontology.ExportOntologyTTL(str(ontology_path), includeBOT=False, silent=True) == str(ontology_path)
    assert ontology_path.read_text(encoding="utf-8").startswith("@prefix")


def test_validate_ttl_and_rdf_methods_without_rdflib(monkeypatch, tmp_path):
    Ontology = _ontology()
    monkeypatch.setitem(sys.modules, "rdflib", None)
    monkeypatch.setitem(sys.modules, "rdflib.namespace", None)

    report = Ontology.ValidateTTLString("@prefix top: <http://w3id.org/topologicpy#> .", silent=True)
    assert report["ok"] is True
    assert report["triple_count"] == 0
    assert report["warnings"]

    ttl_path = tmp_path / "x.ttl"
    ttl_path.write_text("@prefix top: <http://w3id.org/topologicpy#> .", encoding="utf-8")
    file_report = Ontology.ValidateTTLFile(str(ttl_path), silent=True)
    assert file_report["ok"] is True

    topology = {"ontology_class": "top:Graph", "id": "g"}
    assert Ontology.RDFGraph(topology, silent=True) is None
    assert Ontology.RDFString(topology, silent=True) is None
    assert Ontology.ExportRDF(topology, str(tmp_path / "g.ttl"), silent=True) is None
    assert Ontology.GraphByTTLString("@prefix top: <http://w3id.org/topologicpy#> .", silent=True) is None
    assert Ontology.GraphByRDFFile(str(ttl_path), silent=True) is None


def test_ontology_triples_are_unique_and_include_expected_schema_terms():
    Ontology = _ontology()
    triples = Ontology.OntologyTriples(includeBOT=True)
    assert len(triples) == len(set(triples))
    assert any(t[0] == "top:Graph" and t[1] == "rdf:type" and t[2] == "owl:Class" for t in triples)
    assert any(t[0] == "top:hasNode" and t[1] == "rdf:type" and t[2] == "owl:ObjectProperty" for t in triples)
    assert any(t[0] == "top:Room" and t[1] == "rdfs:subClassOf" and t[2] == "bot:Space" for t in triples)


def test_invalid_inputs_return_documented_fallbacks(tmp_path):
    Ontology = _ontology()
    assert Ontology.SetClass(None, "top:Room", silent=True) is None
    assert Ontology.SetClass({}, "", silent=True) is None
    assert Ontology.SetCategory({}, "", silent=True) is None
    assert Ontology.Annotate(None, silent=True) is None
    assert Ontology.NormalizeDictionary(None, silent=True) is None
    assert Ontology.Triples(None, silent=True) == []
    assert Ontology.GraphTriples(None, silent=True) == []
    assert Ontology.TTLString(None, silent=True) is None
    assert Ontology.ExportTTL(None, str(tmp_path / "x.ttl"), silent=True) is None
    assert Ontology.ExportTTL({}, None, silent=True) is None
    assert Ontology.ValidateGraph(None, silent=True)["ok"] is False
    assert Ontology.ValidateTTLString("", silent=True)["ok"] is False
    assert Ontology.ValidateTTLFile(None, silent=True)["ok"] is False
