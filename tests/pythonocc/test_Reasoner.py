# Auto-generated PythonOCC backend parity test
# Copied from test_Reasoner.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

import builtins
import importlib
import sys
import types
import zipfile

import pytest


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


@pytest.fixture()
def rs():
    import topologicpy.Reasoner as reasoner_module
    return reasoner_module.Reasoner


def _triple_strings(graph):
    return {(str(s), str(p), str(o)) for s, p, o in graph}


def _rdflib_available(rs):
    return rs._rdflib(silent=True) is not None


def _assert_optional_rdflib_absent_path(rs):
    assert rs._rdflib(silent=True) is None
    assert rs.RDFGraphByTriples([], silent=True) is None


def test_namespaces_qnames_uri_refs_and_dependency_report(rs):
    ns = rs.Namespaces()
    assert ns["rdf"].endswith("rdf-syntax-ns#")
    assert ns["rdfs"].endswith("rdf-schema#")
    assert ns["top"].startswith("http://w3id.org/topologicpy")

    assert rs.ExpandQName("top:Graph").endswith("topologicpy#Graph")
    assert rs.ExpandQName("<http://example.org/a>") == "http://example.org/a"
    assert rs.ExpandQName("not_a_qname", defaultValue="fallback") == "fallback"
    assert rs.QName(rs.ExpandQName("top:Graph")) == "top:Graph"
    assert rs.QName("<http://example.org/a>") == "http://example.org/a"

    report = rs.Dependencies()
    assert set(report.keys()) == {"rdflib", "owlrl", "pyshacl"}
    assert isinstance(report["rdflib"], dict)

    rd = rs._rdflib(silent=True)
    if rd is None:
        assert report["rdflib"]["available"] is False
        assert rs._uri_ref("top:Graph") is None
        return
    assert "BNode" in rd
    uri = rs._uri_ref("top:Graph", URIRef=rd["URIRef"])
    assert str(uri).endswith("topologicpy#Graph")


def test_rdflib_guard_returns_none_without_printing(monkeypatch, rs):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "rdflib" or name.startswith("rdflib."):
            raise ImportError("blocked rdflib for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert rs._rdflib(silent=True) is None


def test_rdf_graph_by_triples_canonicalises_tokens_and_literals(rs):
    if not _rdflib_available(rs):
        _assert_optional_rdflib_absent_path(rs)
        return
    graph = rs.RDFGraphByTriples([
        ("inst:room_1", "rdf:type", "top:TGraph"),
        ("inst:room_1", "top:hasStartVertex", "inst:vertex_a"),
        ("inst:room_1", "custom property", '"custom literal"'),
        ("inst:room_1", "hasWeight", 3.5),
        (None, "p", "o"),
        ("bad",),
    ], silent=True)
    assert graph is not None
    triples = _triple_strings(graph)
    assert any(s.endswith("/instance#room_1") and p.endswith("rdf-syntax-ns#type") and o.endswith("topologicpy#Graph") for s, p, o in triples)
    assert any(p.endswith("topologicpy#startsAt") for _, p, _ in triples)
    assert any(p.endswith("topologicpy/dictionary#custom_property") and o == "custom literal" for _, p, o in triples)
    assert any(o == "3.5" for _, _, o in triples)


def test_infer_rdfs_subclass_subproperty_domain_range_inverse_and_queries(rs):
    if not _rdflib_available(rs):
        _assert_optional_rdflib_absent_path(rs)
        return
    before = rs.RDFGraphByTriples([
        ("inst:room", "rdf:type", "top:Room"),
        ("top:Room", "rdfs:subClassOf", "top:Space"),
        ("top:Space", "rdfs:subClassOf", "top:Topology"),
        ("top:touches", "rdfs:subPropertyOf", "top:relatedTo"),
        ("inst:a", "top:touches", "inst:b"),
        ("top:hasPart", "rdfs:domain", "top:Assembly"),
        ("top:hasPart", "rdfs:range", "top:Element"),
        ("inst:assembly", "top:hasPart", "inst:part"),
        ("top:inside", "owl:inverseOf", "top:contains"),
        ("inst:box", "top:inside", "inst:room"),
        ("top:Wall", "owl:equivalentClass", "top:VerticalElement"),
        ("top:parentOf", "owl:equivalentProperty", "top:hasChild"),
    ], silent=True)

    after = rs.Infer(before, includeOntologyAxioms=False, inplace=False, silent=True)
    assert after is not before
    assert "top:Space" in rs.Types(after, "inst:room")
    assert "top:Topology" in rs.Types(after, "inst:room")
    assert "top:Assembly" in rs.Types(after, "inst:assembly")
    assert "top:Element" in rs.Types(after, "inst:part")
    assert "top:Topology" in rs.SuperClasses(after, "top:Room")

    diff = rs.Difference(before, after, compact=True)
    assert ("inst:a", "top:relatedTo", "inst:b") in diff
    assert ("inst:room", "top:contains", "inst:box") in diff
    assert ("top:Wall", "rdfs:subClassOf", "top:VerticalElement") in diff
    assert ("top:parentOf", "rdfs:subPropertyOf", "top:hasChild") in diff

    summary = rs.Summary(before, after)
    assert summary["output_triples"] >= summary["input_triples"]
    assert summary["inferred_triples"] == summary["output_triples"] - summary["input_triples"]


def test_result_proof_tree_explain_proof_graph_data_rule_statistics_and_diff(rs):
    if not _rdflib_available(rs):
        _assert_optional_rdflib_absent_path(rs)
        return
    before = rs.RDFGraphByTriples([
        ("inst:room", "rdf:type", "top:Room"),
        ("top:Room", "rdfs:subClassOf", "top:Space"),
        ("top:Space", "rdfs:subClassOf", "top:Topology"),
        ("top:touches", "rdfs:subPropertyOf", "top:relatedTo"),
        ("inst:a", "top:touches", "inst:b"),
    ], silent=True)
    after = rs.Infer(before, includeOntologyAxioms=False, silent=True)
    result = rs.Result(before, after, profile="rdfs")

    assert result.Summary()["asserted_facts"] == len(before)
    assert result.Summary()["inferred_facts"] >= 1
    assert ("inst:room", "rdf:type", "top:Topology") in result.InferredTriples()

    asserted_tree = rs.ProofTree(result, triple=("inst:room", "rdf:type", "top:Room"))
    assert asserted_tree["asserted"] is True
    assert asserted_tree["rule"] == "Asserted"

    inferred_tree = rs.ProofTree(result, subject="inst:room", predicate="rdf:type", object="top:Topology")
    assert inferred_tree["inferred"] is True
    assert "subClassOf" in inferred_tree["rule"]
    assert ("inst:room", "rdf:type", "top:Room") in inferred_tree["premises"]

    explanation = rs.Explain(result, triple=("inst:room", "rdf:type", "top:Topology"))
    assert "The following fact is inferred" in explanation
    assert "Premises" in explanation

    proof_data = rs.ProofGraphData(result, triple=("inst:room", "rdf:type", "top:Topology"))
    assert {"nodes", "edges", "target", "proof_tree"}.issubset(proof_data.keys())
    assert any(n["kind"] == "conclusion" for n in proof_data["nodes"])
    assert any(e["label"] == "premise" for e in proof_data["edges"])

    stats = rs.RuleStatistics(result)
    assert stats["Asserted"] == len(before)
    assert stats["RDFS/OWL-RL closure"] >= 1

    diff = rs.Diff(result, limit=2)
    assert diff["count_added"] <= 2
    assert diff["removed"] == []


def test_domain_range_and_subproperty_proof_tree_paths(rs):
    if not _rdflib_available(rs):
        _assert_optional_rdflib_absent_path(rs)
        return
    before = rs.RDFGraphByTriples([
        ("top:hasPart", "rdfs:domain", "top:Assembly"),
        ("top:hasPart", "rdfs:range", "top:Element"),
        ("inst:assembly", "top:hasPart", "inst:part"),
        ("top:touches", "rdfs:subPropertyOf", "top:relatedTo"),
        ("inst:a", "top:touches", "inst:b"),
    ], silent=True)
    after = rs.Infer(before, includeOntologyAxioms=False, silent=True)
    result = rs.Result(before, after)

    domain_tree = rs.ProofTree(result, triple=("inst:assembly", "rdf:type", "top:Assembly"))
    assert domain_tree["rule"] == "RDFS2: domain inference"

    range_tree = rs.ProofTree(result, triple=("inst:part", "rdf:type", "top:Element"))
    assert range_tree["rule"] == "RDFS3: range inference"

    subprop_tree = rs.ProofTree(result, triple=("inst:a", "top:relatedTo", "inst:b"))
    assert subprop_tree["rule"] == "RDFS7: property inheritance through rdfs:subPropertyOf"


def test_turtle_export_validate_and_no_axiom_inference_paths(tmp_path, rs):
    if not _rdflib_available(rs):
        _assert_optional_rdflib_absent_path(rs)
        return
    graph = rs.RDFGraphByTriples([("inst:a", "rdf:type", "top:Vertex")], silent=True)
    ttl = rs.TurtleString(graph)
    assert isinstance(ttl, str)
    assert "@prefix" in ttl or "inst:a" in ttl

    out = tmp_path / "graph.ttl"
    assert rs.ExportRDF(graph, str(out), overwrite=True, silent=True) == str(out)
    assert out.exists() and out.read_text(encoding="utf-8").strip()
    assert rs.ExportRDF(graph, str(out), overwrite=False, silent=True) is None
    assert rs.ExportRDF(None, str(tmp_path / "bad.ttl"), silent=True) is None

    none_profile = rs.Infer(graph, profile="none", includeOntologyAxioms=False, inplace=False, silent=True)
    assert len(none_profile) == len(graph)

    validation = rs.Validate(graph, silent=True)
    assert set(["available", "conforms", "results_graph", "results_text"]).issubset(validation.keys())


def test_add_ontology_axioms_fallback_without_ontology_module(rs):
    if not _rdflib_available(rs):
        _assert_optional_rdflib_absent_path(rs)
        return
    graph = rs.RDFGraphByTriples([], silent=True)
    out = rs.AddOntologyAxioms(graph, includeBOT=True, silent=True)
    assert out is graph
    triples = _triple_strings(graph)
    assert any(s.endswith("topologicpy#TGraph") and p.endswith("subClassOf") and o.endswith("topologicpy#Graph") for s, p, o in triples)
    assert any(p.endswith("equivalentProperty") for _, p, _ in triples)


def test_bnode_compaction_and_literal_tokens(rs):
    if not _rdflib_available(rs):
        _assert_optional_rdflib_absent_path(rs)
        return
    rd = rs._rdflib(silent=True)
    bnode = rd["BNode"]("abc")
    lit = rd["Literal"]("hello")
    import topologicpy.Reasoner as mod

    assert mod._reasoner_compact_node(bnode) == "_:abc"
    assert mod._reasoner_compact_node(lit).startswith('"hello"')
    assert mod._reasoner_is_literal_token('"hello"') is True
    assert mod._reasoner_is_literal_token("top:Graph") is False


def test_apply_inferences_to_fake_tgraph(monkeypatch, rs):
    fake_mod = types.ModuleType("topologicpy.TGraph")

    class FakeTGraph:
        def __init__(self):
            self._dictionary = {"id": "g", "ontology_class": "top:Graph"}
            self._vertices = [
                {"index": 0, "active": True, "dictionary": {"id": "v1", "ontology_class": "top:Vertex"}},
                {"index": 1, "active": False, "dictionary": {"id": "inactive"}},
            ]
            self._edges = [
                {"index": 0, "active": True, "dictionary": {"id": "e1", "ontology_class": "top:Edge"}}
            ]
            self.invalidated = False

        @staticmethod
        def _OntologySubjectFromDictionary(dictionary, fallback, namespacePrefix="inst"):
            return f"{namespacePrefix}:{dictionary.get('id', fallback)}"

        def _invalidate_cache(self):
            self.invalidated = True

    fake_mod.TGraph = FakeTGraph
    monkeypatch.setitem(sys.modules, "topologicpy.TGraph", fake_mod)

    graph = FakeTGraph()
    inferred = rs.RDFGraphByTriples([
        ("inst:v1", "rdf:type", "top:Vertex"),
        ("inst:v1", "rdf:type", "top:Topology"),
        ("inst:v1", "rdf:type", "bot:Element"),
        ("inst:e1", "rdf:type", "top:Relationship"),
    ], silent=True)

    if inferred is None:
        returned = rs.ApplyInferences(graph, inferred, silent=True)
        assert returned is graph
        assert graph.invalidated is False
        return

    returned = rs.ApplyInferences(graph, inferred, silent=True)
    assert returned is graph
    assert graph.invalidated is True
    assert graph._vertices[0]["dictionary"]["inferred_ontology_classes"] == ["top:Topology"]
    assert graph._vertices[0]["dictionary"]["inferred_bot_classes"] == ["bot:Element"]
    assert graph._edges[0]["dictionary"]["inferred_ontology_classes"] == ["top:Relationship"]
    assert "inferred_ontology_classes" not in graph._vertices[1]["dictionary"]


def test_proof_tgraph_with_fake_tgraph(monkeypatch, rs):
    fake_mod = types.ModuleType("topologicpy.TGraph")

    class FakeTGraph:
        def __init__(self, directed=False, allowSelfLoops=False, allowParallelEdges=True, dictionary=None):
            self.directed = directed
            self.dictionary = dictionary or {}
            self.vertices = []
            self.edges = []

        def AddVertex(self, dictionary=None):
            self.vertices.append(dictionary or {})
            return len(self.vertices) - 1

        def AddEdge(self, src, dst, directed=True, dictionary=None):
            self.edges.append((src, dst, directed, dictionary or {}))
            return len(self.edges) - 1

    fake_mod.TGraph = FakeTGraph
    monkeypatch.setitem(sys.modules, "topologicpy.TGraph", fake_mod)

    result = rs.Result(
        rs.RDFGraphByTriples([("inst:a", "rdf:type", "top:Vertex")], silent=True),
        rs.RDFGraphByTriples([("inst:a", "rdf:type", "top:Vertex")], silent=True),
    )
    proof_graph = rs.ProofTGraph(result, triple=("inst:a", "rdf:type", "top:Vertex"))
    assert isinstance(proof_graph, FakeTGraph)
    assert proof_graph.directed is True
    assert len(proof_graph.vertices) >= 2
    assert len(proof_graph.edges) >= 1


def test_proof_graph_figure_and_html_forward_normalized_triple(monkeypatch, rs):
    fake_mod = types.ModuleType("topologicpy.Plotly")
    calls = []

    class FakePlotly:
        @staticmethod
        def FigureByProofGraph(**kwargs):
            calls.append(("figure", kwargs))
            assert "subject" not in kwargs
            assert "predicate" not in kwargs
            assert "object" not in kwargs
            return {"figure": kwargs}

        @staticmethod
        def ProofGraphHTML(**kwargs):
            calls.append(("html", kwargs))
            assert "subject" not in kwargs
            assert "predicate" not in kwargs
            assert "object" not in kwargs
            return kwargs.get("path")

    fake_mod.Plotly = FakePlotly
    monkeypatch.setitem(sys.modules, "topologicpy.Plotly", fake_mod)

    fig = rs.ProofGraphFigure("result", subject="inst:a", predicate="rdf:type", object="top:Vertex", layout="tree")
    html = rs.ProofGraphHTML("result", subject="inst:a", predicate="rdf:type", object="top:Vertex", path="proof.html")

    assert fig["figure"]["triple"] == ("inst:a", "rdf:type", "top:Vertex")
    assert html == "proof.html"
    assert calls[0][0] == "figure"
    assert calls[1][0] == "html"


def test_rdf_graph_by_topology_uses_fake_ontology(monkeypatch, rs):
    fake_mod = types.ModuleType("topologicpy.Ontology")

    class FakeOntology:
        NAMESPACES = {"ex": "http://example.org/"}

        @staticmethod
        def GraphTriples(topology, **kwargs):
            return [("inst:g", "rdf:type", "top:Graph"), ("inst:g", "ex:label", "Graph")]

        @staticmethod
        def Triples(topology, **kwargs):
            return [("inst:t", "rdf:type", "top:Vertex")]

        @staticmethod
        def OntologyTriples(includeBOT=True):
            return [("top:Graph", "rdf:type", "owl:Class")]

        @staticmethod
        def PropertyQName(text):
            return text

        @staticmethod
        def CanonicalClass(text, defaultValue=None):
            return text if text is not None else defaultValue

    fake_mod.Ontology = FakeOntology
    monkeypatch.setitem(sys.modules, "topologicpy.Ontology", fake_mod)
    if not _rdflib_available(rs):
        assert rs.RDFGraphByTopology(object(), includeOntologyAxioms=True, silent=True) is None
        return
    graph = rs.RDFGraphByTopology(object(), includeOntologyAxioms=True, silent=True)
    assert graph is not None
    triples = _triple_strings(graph)
    assert any(s.endswith("/instance#g") for s, _, _ in triples)
    assert any(o.endswith("owl#Class") for _, _, o in triples)
