# Auto-generated PythonOCC backend parity test
# Copied from test_GQL.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.GQL.

These tests deliberately avoid depending on the internal lowercase GQL parser,
executor, or the full TGraph implementation. Where those collaborators are
needed, deterministic stand-ins are installed with monkeypatch.
"""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from topologicpy.GQL import GQL


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class FakeTGraph:
    """Small TGraph stand-in for GQL wrapper tests."""

    def __init__(self, vertices=None, edges=None, directed=True, dictionary=None):
        self.vertices = vertices or []
        self.edges = edges or []
        self.directed = bool(directed)
        self.dictionary = dict(dictionary or {})

    def SetDictionary(self, dictionary):
        self.dictionary = dict(dictionary or {})
        return self

    def SetVertexDictionary(self, index, dictionary):
        self.vertices[int(index)]["dictionary"] = dict(dictionary or {})
        return self

    def SetEdgeDictionary(self, index, dictionary):
        self.edges[int(index)]["dictionary"] = dict(dictionary or {})
        return self

    @staticmethod
    def Dictionary(graph):
        return dict(graph.dictionary)

    @staticmethod
    def Vertices(graph, asTopologic=False, active=True):
        return list(graph.vertices)

    @staticmethod
    def Edges(graph, asTopologic=False, active=True):
        return list(graph.edges)

    @staticmethod
    def ByVerticesEdges(
        vertices,
        edges,
        directed=True,
        allowSelfLoops=True,
        allowParallelEdges=True,
        dictionary=None,
        ontology=True,
        silent=True,
    ):
        vertex_records = []
        for i, vertex in enumerate(vertices or []):
            d = dict(vertex or {})
            vertex_records.append({"index": i, "dictionary": d})

        edge_records = []
        for i, edge in enumerate(edges or []):
            d = dict(edge or {})
            edge_records.append(
                {
                    "index": i,
                    "src": d.get("src"),
                    "dst": d.get("dst"),
                    "dictionary": d,
                }
            )

        return FakeTGraph(vertex_records, edge_records, directed=directed, dictionary=dictionary)

    @staticmethod
    def FromPython(graph, ontology=True):
        return FakeTGraph.ByVerticesEdges(
            vertices=graph.get("vertices", []),
            edges=graph.get("edges", []),
            directed=graph.get("directed", True),
            dictionary=graph.get("dictionary", {}),
            ontology=ontology,
        )


def _install_fake_tgraph(monkeypatch):
    module = types.ModuleType("topologicpy.TGraph")
    module.TGraph = FakeTGraph
    monkeypatch.setitem(sys.modules, "topologicpy.TGraph", module)
    return module


_DEFAULT_PARSE_RESULT = object()
_DEFAULT_EXECUTE_RESULT = object()


def _install_fake_gql_pipeline(
    monkeypatch,
    parse_result=_DEFAULT_PARSE_RESULT,
    execute_result=_DEFAULT_EXECUTE_RESULT,
    parse_exception=None,
    execute_exception=None,
    recorder=None,
):
    recorder = recorder if recorder is not None else {}
    parse_result = {"ast": True} if parse_result is _DEFAULT_PARSE_RESULT else parse_result
    execute_result = [{"ok": True}] if execute_result is _DEFAULT_EXECUTE_RESULT else execute_result

    parent_module = types.ModuleType("topologicpy.gql")
    parent_module.__path__ = []

    parser_module = types.ModuleType("topologicpy.gql.Parser")
    executor_module = types.ModuleType("topologicpy.gql.Executor")

    class Parser:
        @staticmethod
        def Parse(query, silent=False):
            recorder["query"] = query
            recorder["parse_silent"] = silent
            if parse_exception is not None:
                raise parse_exception
            return parse_result

    class Executor:
        @staticmethod
        def Execute(graph, ast, silent=False):
            recorder["graph"] = graph
            recorder["ast"] = ast
            recorder["execute_silent"] = silent
            if execute_exception is not None:
                raise execute_exception
            return execute_result

    parser_module.Parser = Parser
    executor_module.Executor = Executor

    monkeypatch.setitem(sys.modules, "topologicpy.gql", parent_module)
    monkeypatch.setitem(sys.modules, "topologicpy.gql.Parser", parser_module)
    monkeypatch.setitem(sys.modules, "topologicpy.gql.Executor", executor_module)
    return recorder


def test_ontology_class_normalisation_and_local_names():
    assert GQL._normalise_top_class(None) is None
    assert GQL._normalise_top_class("") is None
    assert GQL._normalise_top_class("top:Room") == "top:Room"
    assert GQL._normalise_top_class("Room") == "top:Room"
    assert GQL._normalise_top_class("room") == "room"
    assert GQL._normalise_top_class("http://w3id.org/topologicpy/#Cell") == "top:Cell"
    assert GQL._normalise_top_class("http://w3id.org/topologicpy#Cell") == "top:Cell"
    assert GQL._ontology_local_name("top:Cell") == "Cell"
    assert GQL._ontology_local_name("http://w3id.org/topologicpy/#Space") == "Space"


def test_record_dictionary_preserves_structural_fields_and_metadata():
    record = {
        "id": "A",
        "index": 7,
        "src": "S",
        "representation": "ignore",
        "dictionary": {
            "id": "nested-id-should-not-override",
            "src": "nested-src-should-not-override",
            "label": "Room",
            "category": "node",
        },
    }
    merged = GQL._record_dictionary(record)
    assert merged["id"] == "A"
    assert merged["index"] == 7
    assert merged["src"] == "S"
    assert merged["label"] == "Room"
    assert merged["category"] == "node"
    assert "representation" not in merged


def test_safe_and_python_dictionary_use_merged_record_semantics():
    record = {"id": "A", "dictionary": {"label": "Room"}}
    assert GQL._safe_dictionary(record) == {"id": "A", "label": "Room"}
    assert GQL._python_dictionary(record) == {"id": "A", "label": "Room"}
    assert GQL._dictionary_value(record, "id") == "A"
    assert GQL._dictionary_value(record, "label") == "Room"
    assert GQL._dictionary_value(record, "missing", "fallback") == "fallback"


def test_normalize_ontology_labels_modes_and_invalid_inputs():
    query = "MATCH (n:top:Room)-[:top:adjacentTo]->(m:top:Space) RETURN n"
    assert GQL.NormalizeOntologyLabels(query) == "MATCH (n:Room)-[:adjacentTo]->(m:Space) RETURN n"
    assert GQL.NormalizeOntologyLabels(query, mode="none") == query
    assert GQL.NormalizeOntologyLabels(query, mode="off") == query
    assert GQL.NormalizeOntologyLabels(123) == 123
    assert GQL.NormalizeOntologyLabels(query, mode="unknown", silent=True) == query


def test_annotate_tgraph_fills_missing_graph_vertex_and_edge_metadata(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    graph = FakeTGraph(
        vertices=[
            {"index": 0, "dictionary": {"label": "top:Room"}},
            {"index": 1, "dictionary": {"ontology_class": "top:Existing"}},
        ],
        edges=[{"index": 0, "src": 0, "dst": 1, "dictionary": {"label": "adjacent"}}],
        dictionary={},
    )

    result = GQL._annotate_tgraph(graph, ontologyClass="BuildingGraph", generatedBy="unit-test")

    assert result is graph
    assert graph.dictionary["ontology_class"] == "top:BuildingGraph"
    assert graph.dictionary["category"] == "graph"
    assert graph.dictionary["generated_by"] == "unit-test"
    assert graph.vertices[0]["dictionary"]["ontology_class"] == "top:Room"
    assert graph.vertices[0]["dictionary"]["category"] == "node"
    assert graph.vertices[1]["dictionary"]["ontology_class"] == "top:Existing"
    assert graph.vertices[1]["dictionary"]["category"] == "node"
    assert graph.edges[0]["dictionary"]["ontology_class"] == "top:Relationship"
    assert graph.edges[0]["dictionary"]["category"] == "relationship"


def test_to_tgraph_from_working_graph_preserves_node_and_edge_record_metadata(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    working_graph = {
        "name": "WG",
        "directed": False,
        "dictionary": {"source": "gql"},
        "nodes": [
            {"id": "A", "index": 10, "dictionary": {"label": "Room", "ontology_class": "top:Space"}},
            {"id": "B", "dictionary": {"label": "Cell"}},
            {"id": "3.0", "dictionary": {"label": "Numeric"}},
        ],
        "relationships": [
            {"src": "A", "dst": "B", "dictionary": {"label": "adjacent"}},
            {"source": "B", "target": "3", "dictionary": {"weight": 2.5}},
            {"src": "missing", "dst": "B", "dictionary": {"label": "discard"}},
        ],
    }

    graph = GQL._to_tgraph(working_graph, ontology=True, generatedBy="test", silent=False)

    assert isinstance(graph, FakeTGraph)
    assert graph.directed is False
    assert graph.dictionary["name"] == "WG"
    assert graph.dictionary["source"] == "gql"
    assert graph.dictionary["ontology_class"] == "top:Graph"
    assert graph.vertices[0]["dictionary"]["id"] == "A"
    assert graph.vertices[0]["dictionary"]["index"] == 10
    assert graph.vertices[0]["dictionary"]["label"] == "Room"
    assert graph.vertices[0]["dictionary"]["ontology_class"] == "top:Space"
    assert graph.vertices[1]["dictionary"]["id"] == "B"
    assert len(graph.edges) == 2
    assert graph.edges[0]["src"] == 0
    assert graph.edges[0]["dst"] == 1
    assert graph.edges[0]["dictionary"]["label"] == "adjacent"
    assert graph.edges[1]["src"] == 1
    assert graph.edges[1]["dst"] == 2
    assert graph.edges[1]["dictionary"]["weight"] == 2.5


def test_to_tgraph_accepts_native_tgraph_python_format(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    native = {
        "type": "TGraph",
        "directed": True,
        "dictionary": {"id": "native"},
        "vertices": [{"id": "A"}, {"id": "B"}],
        "edges": [{"src": 0, "dst": 1, "label": "connects"}],
    }
    graph = GQL.TGraph(native, ontology=True, generatedBy="native-test", silent=False)
    assert isinstance(graph, FakeTGraph)
    assert graph.dictionary["id"] == "native"
    assert graph.dictionary["generated_by"] == "native-test"
    assert graph.edges[0]["src"] == 0
    assert graph.edges[0]["dst"] == 1


def test_to_tgraph_returns_same_tgraph_when_ontology_is_false(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    graph = FakeTGraph(dictionary={})
    assert GQL.TGraph(graph, ontology=False) is graph
    assert graph.dictionary == {}


def test_topologic_graph_alias_delegates_to_tgraph(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    graph = GQL.TopologicGraph({"nodes": ["A"], "relationships": []}, generatedBy="alias")
    assert isinstance(graph, FakeTGraph)
    assert graph.dictionary["generated_by"] == "alias"


def test_to_tgraph_invalid_inputs_return_none(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    assert GQL.TGraph(None, silent=True) is None
    assert GQL.TGraph("not-a-graph", silent=True) is None


def test_annotate_result_recursively_annotates_tgraphs(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    graph = FakeTGraph(dictionary={})
    result = GQL._annotate_result({"rows": [{"graph": graph}], "tgraph": graph}, ontology=True)
    assert result["rows"][0]["graph"].dictionary["generated_by"] == "GQL.Query"
    assert result["tgraph"].dictionary["ontology_class"] == "top:Graph"
    unchanged = GQL._annotate_result({"graph": graph}, ontology=False)
    assert unchanged["graph"] is graph


def test_ontology_terms_extracts_unique_nested_terms(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    graph = FakeTGraph(
        vertices=[{"index": 0, "dictionary": {"ontology_class": "top:Room", "category": "node", "label": "R"}}],
        edges=[{"index": 0, "dictionary": {"ontology_class": "top:Relationship", "category": "relationship"}}],
        dictionary={"ontology_class": "top:Graph", "category": "graph", "label": "G"},
    )
    result = GQL.OntologyTerms([graph, {"ontology_class": "top:Room", "category": "node", "label": "R"}])
    assert result["ontology_classes"] == ["top:Graph", "top:Room", "top:Relationship"]
    assert result["categories"] == ["graph", "node", "relationship"]
    assert result["labels"] == ["G", "R"]


def test_query_rejects_invalid_query_before_importing_parser():
    assert GQL.Query({}, None, silent=True) is None
    assert GQL.Query({}, "   ", silent=True) is None


def test_query_returns_none_when_parser_executor_import_fails(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("topologicpy.gql"):
            raise ImportError("forced missing gql package")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert GQL.Query({}, "MATCH (n) RETURN n", silent=True) is None


def test_query_success_normalizes_labels_and_executes(monkeypatch):
    _install_fake_tgraph(monkeypatch)
    graph = FakeTGraph(dictionary={})
    execute_result = {"graph": graph, "rows": [{"ontology_class": "top:Room"}]}
    recorder = _install_fake_gql_pipeline(monkeypatch, execute_result=execute_result)

    result = GQL.Query(graph, "MATCH (n:top:Room) RETURN n", ontology=True, silent=False)

    assert recorder["query"] == "MATCH (n:Room) RETURN n"
    assert recorder["graph"] is graph
    assert recorder["ast"] == {"ast": True}
    assert result["graph"].dictionary["generated_by"] == "GQL.Query"
    assert result["rows"][0]["ontology_class"] == "top:Room"


def test_query_can_disable_ontology_label_normalisation(monkeypatch):
    recorder = _install_fake_gql_pipeline(monkeypatch, execute_result=[{"ok": True}])
    result = GQL.Query({}, "MATCH (n:top:Room) RETURN n", normalizeOntologyLabels=False, ontology=False)
    assert recorder["query"] == "MATCH (n:top:Room) RETURN n"
    assert result == [{"ok": True}]


def test_query_parse_none_exception_and_execute_exception_return_none(monkeypatch):
    _install_fake_gql_pipeline(monkeypatch, parse_result=None)
    assert GQL.Query({}, "MATCH (n) RETURN n", silent=True) is None

    _install_fake_gql_pipeline(monkeypatch, parse_exception=RuntimeError("parse failed"))
    assert GQL.Query({}, "MATCH (n) RETURN n", silent=True) is None

    _install_fake_gql_pipeline(monkeypatch, execute_exception=RuntimeError("execute failed"))
    assert GQL.Query({}, "MATCH (n) RETURN n", silent=True) is None


def test_mutate_is_query_alias(monkeypatch):
    recorder = _install_fake_gql_pipeline(monkeypatch, execute_result={"graph": {"nodes": []}, "rows": []})
    result = GQL.Mutate({}, "CREATE (n:top:Room)", ontology=False)
    assert recorder["query"] == "CREATE (n:Room)"
    assert result == {"graph": {"nodes": []}, "rows": []}
