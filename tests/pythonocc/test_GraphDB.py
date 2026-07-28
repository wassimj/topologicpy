# Auto-generated PythonOCC backend parity test
# Copied from test_GraphDB.py


"""Unit tests for topologicpy.GraphDB."""

from __future__ import annotations

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

import sys
import types

import pytest

from topologicpy.GraphDB import GraphDB


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class RecordingManager:
    def __init__(self):
        self.calls = []

    def exec(self, query, parameters, write=False):
        self.calls.append((query, parameters, write))
        return [{"query": query, "parameters": parameters, "write": write}]


class PlainManager:
    def __init__(self):
        self.calls = []


class FakeKuzu:
    calls = []

    @staticmethod
    def EnsureSchema(manager, silent=False):
        FakeKuzu.calls.append(("EnsureSchema", manager, {"silent": silent}))
        return True

    @staticmethod
    def UpsertGraph(
        manager,
        graph,
        graphIDKey="graph_id",
        vertexIDKey="id",
        custom=None,
        silent=False,
    ):
        kwargs = {
            "graph": graph,
            "graphIDKey": graphIDKey,
            "vertexIDKey": vertexIDKey,
            "custom": custom,
            "silent": silent,
        }
        FakeKuzu.calls.append(("UpsertGraph", manager, kwargs))
        return graphIDKey

    @staticmethod
    def ByCSVPath(manager, path, graphIDHeader="graph_id", tolerance=0.0001, silent=False):
        kwargs = {"path": path, "graphIDHeader": graphIDHeader, "tolerance": tolerance, "silent": silent}
        FakeKuzu.calls.append(("ByCSVPath", manager, kwargs))
        return {"path": path, "graphIDHeader": graphIDHeader}

    @staticmethod
    def GraphByID(manager, graphID, silent=False):
        FakeKuzu.calls.append(("GraphByID", manager, {"graphID": graphID, "silent": silent}))
        return {"graph_id": graphID}

    @staticmethod
    def GraphsByQuery(manager, query, parameters=None, silent=False):
        FakeKuzu.calls.append(("GraphsByQuery", manager, {"query": query, "parameters": parameters, "silent": silent}))
        return [{"query": query, "parameters": parameters}]

    @staticmethod
    def DeleteGraph(manager, graphID, silent=False):
        FakeKuzu.calls.append(("DeleteGraph", manager, {"graphID": graphID, "silent": silent}))
        return graphID == "delete-me"

    @staticmethod
    def EmptyDatabase(manager, dropSchema=False, recreateSchema=True, silent=False):
        kwargs = {"dropSchema": dropSchema, "recreateSchema": recreateSchema, "silent": silent}
        FakeKuzu.calls.append(("EmptyDatabase", manager, kwargs))
        return dropSchema and recreateSchema

    @staticmethod
    def ListGraphs(manager, where=None, limit=100, offset=0, silent=False):
        kwargs = {"where": where, "limit": limit, "offset": offset, "silent": silent}
        FakeKuzu.calls.append(("ListGraphs", manager, kwargs))
        return [kwargs]

    @staticmethod
    def FetchAllPairs(manager, undirected=True, silent=False):
        FakeKuzu.calls.append(("FetchAllPairs", manager, {"undirected": undirected, "silent": silent}))
        return [{"undirected": undirected}]

    @staticmethod
    def CandidateCountsForLabels(manager, labels, excludeLabels=None, limit=50, silent=False):
        kwargs = {"labels": labels, "excludeLabels": excludeLabels, "limit": limit, "silent": silent}
        FakeKuzu.calls.append(("CandidateCountsForLabels", manager, kwargs))
        return [kwargs]

    @staticmethod
    def MaxNeighborsForLabel(manager, label, silent=False):
        FakeKuzu.calls.append(("MaxNeighborsForLabel", manager, {"label": label, "silent": silent}))
        return 7

    @staticmethod
    def FindBestExampleForLabel(manager, label, attachTo=None, silent=False):
        kwargs = {"label": label, "attachTo": attachTo, "silent": silent}
        FakeKuzu.calls.append(("FindBestExampleForLabel", manager, kwargs))
        return kwargs

    @staticmethod
    def Execute(manager, query, parameters=None, write=False, silent=False):
        kwargs = {"query": query, "parameters": parameters, "write": write, "silent": silent}
        FakeKuzu.calls.append(("Execute", manager, kwargs))
        return [kwargs]


class FakeNeo4j:
    calls = []

    @staticmethod
    def Execute(manager, query, parameters=None, write=False, database=None, silent=False):
        kwargs = {"query": query, "parameters": parameters, "write": write, "database": database, "silent": silent}
        FakeNeo4j.calls.append(("Execute", manager, kwargs))
        return [kwargs]

    @staticmethod
    def EnsureSchema(manager, database=None, silent=False):
        FakeNeo4j.calls.append(("EnsureSchema", manager, {"database": database, "silent": silent}))
        return True


class FakeLadybugDB:
    calls = []

    @staticmethod
    def Execute(manager, query, parameters=None, write=False, silent=False):
        kwargs = {"query": query, "parameters": parameters, "write": write, "silent": silent}
        FakeLadybugDB.calls.append(("Execute", manager, kwargs))
        return [kwargs]


def _install_backend_modules(monkeypatch):
    FakeKuzu.calls = []
    FakeNeo4j.calls = []
    FakeLadybugDB.calls = []

    kuzu_module = types.ModuleType("topologicpy.Kuzu")
    kuzu_module.Kuzu = FakeKuzu
    neo4j_module = types.ModuleType("topologicpy.Neo4j")
    neo4j_module.Neo4j = FakeNeo4j
    ladybug_module = types.ModuleType("topologicpy.LadybugDB")
    ladybug_module.LadybugDB = FakeLadybugDB

    monkeypatch.setitem(sys.modules, "topologicpy.Kuzu", kuzu_module)
    monkeypatch.setitem(sys.modules, "topologicpy.Neo4j", neo4j_module)
    monkeypatch.setitem(sys.modules, "topologicpy.LadybugDB", ladybug_module)


def test_by_parameters_provider_aliases_and_options_are_copied():
    options = {"a": 1}
    graphdb = GraphDB.ByParameters(" Neo ", manager="driver", database="neo4j", options=options, silent=True)

    assert graphdb == {"provider": "neo4j", "manager": "driver", "database": "neo4j", "options": {"a": 1}}
    assert graphdb["options"] is not options
    assert GraphDB.ByParameters("ladybug", manager="m", silent=True)["provider"] == "ladybugdb"
    assert GraphDB.ByParameters("kùzu", manager="m", silent=True)["provider"] == "kuzu"
    assert GraphDB.ByParameters(None, silent=True) is None
    assert GraphDB.ByParameters("unsupported", silent=True) is None
    assert GraphDB.ByParameters("kuzu", options=object(), silent=True) is None


def test_accessors_are_defensive_and_normalise_provider():
    graphdb = {"provider": "LBUG", "manager": "m", "database": "db", "options": {"x": 1}}

    assert GraphDB.Provider(graphdb, silent=True) == "ladybugdb"
    assert GraphDB.Manager(graphdb, silent=True) == "m"
    assert GraphDB.Database(graphdb, silent=True) == "db"
    assert GraphDB.Options(graphdb, silent=True) == {"x": 1}
    assert GraphDB.Options({"provider": "kuzu"}, silent=True) == {}
    assert GraphDB.Provider(None, silent=True) is None
    assert GraphDB.Manager(None, silent=True) is None
    assert GraphDB.Database(None, silent=True) is None
    assert GraphDB.Options(None, silent=True) is None


def test_internal_validation_helpers():
    assert GraphDB._normalise_provider(" neo4j ") == "neo4j"
    assert GraphDB._normalise_provider("") is None
    assert GraphDB._valid_provider("kuzu") is True
    assert GraphDB._valid_provider("bad") is False
    assert GraphDB._is_descriptor({"provider": "kuzu"}) is True
    assert GraphDB._is_descriptor({}) is False
    assert GraphDB._copy_options({"a": 1}, silent=True) == {"a": 1}
    assert GraphDB._copy_options(None, silent=True) == {}
    assert GraphDB._copy_options(object(), silent=True) is None


def test_filter_call_kwargs_respects_method_signature():
    def strict(manager, graph, silent=False):
        return graph

    def flexible(manager, **kwargs):
        return kwargs

    assert GraphDB._filter_call_kwargs(strict, {"graph": "g", "silent": True, "extra": 1}) == {
        "graph": "g",
        "silent": True,
    }
    assert GraphDB._filter_call_kwargs(flexible, {"extra": 1}) == {"extra": 1}


def test_backend_resolution_uses_provider_modules(monkeypatch):
    _install_backend_modules(monkeypatch)

    assert GraphDB._backend(GraphDB.ByParameters("kuzu", manager="m", silent=True), silent=True) is FakeKuzu
    assert GraphDB._backend(GraphDB.ByParameters("neo4j", manager="m", silent=True), silent=True) is FakeNeo4j
    assert GraphDB._backend(GraphDB.ByParameters("ladybugdb", manager="m", silent=True), silent=True) is FakeLadybugDB
    assert GraphDB._backend({"provider": "bad"}, silent=True) is None


def test_call_injects_options_filters_kwargs_and_preserves_explicit_overrides(monkeypatch):
    _install_backend_modules(monkeypatch)
    manager = PlainManager()
    graphdb = GraphDB.ByParameters(
        "kuzu",
        manager=manager,
        database="should_not_reach_kuzu",
        options={
            "graphIDKey": "from_options",
            "vertexIDKey": "from_options_vertex",
            "custom": "from_options",
            "ontologyGraphClass": "top:Graph",
            "database": "also_should_not_reach_kuzu",
            "unsupported": "drop-me",
        },
        silent=True,
    )

    result = GraphDB._call(graphdb, "UpsertGraph", graph="graph-object", graphIDKey="explicit", silent=True)

    assert result == "explicit"
    assert FakeKuzu.calls[-1][0] == "UpsertGraph"
    kwargs = FakeKuzu.calls[-1][2]
    assert kwargs == {
        "graph": "graph-object",
        "graphIDKey": "explicit",
        "vertexIDKey": "from_options_vertex",
        "custom": "from_options",
        "silent": True,
    }


def test_call_handles_missing_manager_missing_method_and_non_callable_backend(monkeypatch):
    _install_backend_modules(monkeypatch)

    graphdb_without_manager = GraphDB.ByParameters("kuzu", manager=None, silent=True)
    assert GraphDB._call(graphdb_without_manager, "EnsureSchema", silent=True) is None
    assert GraphDB._call(GraphDB.ByParameters("kuzu", manager=PlainManager(), silent=True), "Missing", silent=True) is None
    assert GraphDB._call(GraphDB.ByParameters("kuzu", manager=PlainManager(), silent=True), "", silent=True) is None

    old = getattr(FakeKuzu, "NotCallable", None)
    FakeKuzu.NotCallable = "not callable"
    try:
        assert GraphDB._call(GraphDB.ByParameters("kuzu", manager=PlainManager(), silent=True), "NotCallable", silent=True) is None
    finally:
        if old is None:
            delattr(FakeKuzu, "NotCallable")
        else:
            FakeKuzu.NotCallable = old


def test_schema_and_persistence_wrappers_dispatch_to_backend(monkeypatch):
    _install_backend_modules(monkeypatch)
    graphdb = GraphDB.ByParameters("kuzu", manager=PlainManager(), silent=True)

    assert GraphDB.EnsureSchema(graphdb, silent=True) is True
    assert GraphDB.ByCSVPath(graphdb, "/tmp/csv", graphIDHeader="gid", tolerance=0.25, silent=True) == {
        "path": "/tmp/csv",
        "graphIDHeader": "gid",
    }
    assert GraphDB.DeleteGraph(graphdb, "delete-me", silent=True) is True
    assert GraphDB.DeleteGraph(graphdb, "keep-me", silent=True) is False
    assert GraphDB.EmptyDatabase(graphdb, dropSchema=True, recreateSchema=True, silent=True) is True
    assert GraphDB.ListGraphs(graphdb, where={"label": "A"}, limit=5, offset=2, silent=True)[0]["limit"] == 5
    assert GraphDB.FetchAllPairs(graphdb, undirected=False, silent=True) == [{"undirected": False}]
    assert GraphDB.CandidateCountsForLabels(graphdb, ["A"], excludeLabels=["B"], limit=3, silent=True)[0]["limit"] == 3
    assert GraphDB.MaxNeighborsForLabel(graphdb, "Room", silent=True) == 7
    assert GraphDB.FindBestExampleForLabel(graphdb, "Room", attachTo="Door", silent=True)["attachTo"] == "Door"


def test_upsert_graph_rejects_none_and_dispatches_graph(monkeypatch):
    _install_backend_modules(monkeypatch)
    graphdb = GraphDB.ByParameters("kuzu", manager=PlainManager(), options={"ontology": False}, silent=True)

    assert GraphDB.UpsertGraph(graphdb, None, silent=True) is None
    assert GraphDB.UpsertGraph(graphdb, "graph-object", graphIDKey="gid", vertexIDKey="node_id", silent=True) == "gid"
    kwargs = FakeKuzu.calls[-1][2]
    assert kwargs["graph"] == "graph-object"
    assert kwargs["graphIDKey"] == "gid"
    assert kwargs["vertexIDKey"] == "node_id"


def test_graph_retrieval_wrappers_annotate_results_non_destructively(monkeypatch):
    _install_backend_modules(monkeypatch)
    graphdb = GraphDB.ByParameters("kuzu", manager=PlainManager(), silent=True)

    assert GraphDB.GraphByID(graphdb, "g1", ontology=False, silent=True) == {"graph_id": "g1"}
    assert GraphDB.GraphsByQuery(graphdb, "MATCH (n)", parameters={"x": 1}, ontology=False, silent=True) == [
        {"query": "MATCH (n)", "parameters": {"x": 1}}
    ]


def test_execute_uses_manager_exec_for_kuzu_and_ladybugdb(monkeypatch):
    _install_backend_modules(monkeypatch)
    manager = RecordingManager()
    graphdb = GraphDB.ByParameters("kuzu", manager=manager, silent=True)

    result = GraphDB.Execute(graphdb, "MATCH (n)", parameters={"x": 1}, write=True, silent=True)

    assert result == [{"query": "MATCH (n)", "parameters": {"x": 1}, "write": True}]
    assert manager.calls == [("MATCH (n)", {"x": 1}, True)]

    ladybug_manager = RecordingManager()
    ladybug = GraphDB.ByParameters("ladybugdb", manager=ladybug_manager, silent=True)
    assert GraphDB.Query(ladybug, "SELECT *", parameters={"a": 2}, silent=True)[0]["parameters"] == {"a": 2}


def test_execute_falls_back_to_backend_execute_when_manager_has_no_exec(monkeypatch):
    _install_backend_modules(monkeypatch)
    graphdb = GraphDB.ByParameters("kuzu", manager=PlainManager(), silent=True)

    result = GraphDB.Execute(graphdb, "MATCH (n)", parameters={"x": 2}, write=False, silent=True)

    assert result == [{"query": "MATCH (n)", "parameters": {"x": 2}, "write": False, "silent": True}]
    assert FakeKuzu.calls[-1][0] == "Execute"


def test_execute_dispatches_neo4j_with_database_context(monkeypatch):
    _install_backend_modules(monkeypatch)
    graphdb = GraphDB.ByParameters("neo4j", manager=PlainManager(), database="neo4j", silent=True)

    result = GraphDB.Execute(graphdb, "MATCH (n)", parameters={"x": 3}, write=False, silent=True)

    assert result == [{"query": "MATCH (n)", "parameters": {"x": 3}, "write": False, "database": "neo4j", "silent": True}]


def test_execute_validates_query_parameters_and_manager(monkeypatch):
    _install_backend_modules(monkeypatch)
    graphdb = GraphDB.ByParameters("kuzu", manager=RecordingManager(), silent=True)

    assert GraphDB.Execute(graphdb, "", silent=True) is None
    assert GraphDB.Execute(graphdb, None, silent=True) is None
    assert GraphDB.Execute(graphdb, "MATCH (n)", parameters=[("x", 1)], silent=True) is None
    assert GraphDB.Execute(GraphDB.ByParameters("kuzu", manager=None, silent=True), "MATCH (n)", silent=True) is None
    assert GraphDB.Execute({"provider": "bad", "manager": RecordingManager()}, "MATCH (n)", silent=True) is None


def test_query_and_ontology_query_helpers(monkeypatch):
    _install_backend_modules(monkeypatch)
    manager = RecordingManager()
    graphdb = GraphDB.ByParameters("kuzu", manager=manager, silent=True)

    assert GraphDB.Query(graphdb, "MATCH (n)", parameters={"a": 1}, silent=True)[0]["write"] is False
    rows = GraphDB.VerticesByOntologyClass(graphdb, "top:Room", limit="5", silent=True)
    assert rows[0]["parameters"] == {"ontologyClass": "top:Room", "limit": 5}
    rows = GraphDB.VerticesByCategory(graphdb, "space", limit=2, silent=True)
    assert rows[0]["parameters"] == {"category": "space", "limit": 2}
    assert GraphDB.VerticesByOntologyClass(graphdb, "", silent=True) is None
    assert GraphDB.VerticesByOntologyClass(graphdb, "top:Room", limit="bad", silent=True) is None
    assert GraphDB.VerticesByCategory(graphdb, "", silent=True) is None
    assert GraphDB.VerticesByCategory(graphdb, "space", limit="bad", silent=True) is None
    assert GraphDB.VerticesByOntologyClass(GraphDB.ByParameters("ladybugdb", manager=manager, silent=True), "top:Room", silent=True) is None


def test_ontology_keys_and_option_lookup():
    keys = GraphDB.OntologyKeys()

    assert "ontology_class" in keys
    assert "generated_by" in keys
    graphdb = GraphDB.ByParameters("kuzu", manager="m", options={"ontology": False}, silent=True)
    assert GraphDB._ontology_option(graphdb, "ontology", defaultValue=True) is False
    assert GraphDB._ontology_option(graphdb, "missing", defaultValue="fallback") == "fallback"


def test_tgraph_ontology_annotation_sets_graph_vertex_and_edge_metadata(monkeypatch):
    class FakeTGraph:
        def __init__(self):
            self.dictionary = {}
            self._vertices = [
                {"index": 0, "dictionary": {}},
                {"index": 1, "dictionary": {"id": "existing"}},
            ]
            self._edges = [{"index": 0, "dictionary": {}}]

        def SetDictionary(self, dictionary):
            self.dictionary = dict(dictionary)
            return self

        def SetVertexDictionary(self, index, dictionary):
            self._vertices[index]["dictionary"] = dict(dictionary)
            return self

        def SetEdgeDictionary(self, index, dictionary):
            self._edges[index]["dictionary"] = dict(dictionary)
            return self

        @staticmethod
        def Dictionary(graph):
            return graph.dictionary

        @staticmethod
        def Vertices(graph, asTopologic=False, active=True):
            return graph._vertices

        @staticmethod
        def Edges(graph, asTopologic=False, active=True):
            return graph._edges

    tgraph_module = types.ModuleType("topologicpy.TGraph")
    tgraph_module.TGraph = FakeTGraph
    monkeypatch.setitem(sys.modules, "topologicpy.TGraph", tgraph_module)

    graph = FakeTGraph()
    result = GraphDB._annotate_graph_ontology(
        graph,
        graphClass="top:CustomGraph",
        vertexClass="top:CustomNode",
        edgeClass="top:CustomRel",
        generatedBy="unit-test",
        silent=True,
    )

    assert result is graph
    assert graph.dictionary["ontology_class"] == "top:CustomGraph"
    assert graph.dictionary["category"] == "graph"
    assert graph.dictionary["generated_by"] == "unit-test"
    assert graph._vertices[0]["dictionary"]["ontology_class"] == "top:CustomNode"
    assert graph._vertices[0]["dictionary"]["category"] == "node"
    assert graph._vertices[0]["dictionary"]["id"] == 0
    assert graph._vertices[1]["dictionary"]["id"] == "existing"
    assert graph._edges[0]["dictionary"]["ontology_class"] == "top:CustomRel"
    assert graph._edges[0]["dictionary"]["category"] == "relationship"


def test_annotate_graph_result_handles_lists_and_ontology_false(monkeypatch):
    class FakeTGraph:
        def __init__(self):
            self.dictionary = {}
            self._vertices = []
            self._edges = []

        def SetDictionary(self, dictionary):
            self.dictionary = dict(dictionary)
            return self

        @staticmethod
        def Dictionary(graph):
            return graph.dictionary

        @staticmethod
        def Vertices(graph, asTopologic=False, active=True):
            return graph._vertices

        @staticmethod
        def Edges(graph, asTopologic=False, active=True):
            return graph._edges

    tgraph_module = types.ModuleType("topologicpy.TGraph")
    tgraph_module.TGraph = FakeTGraph
    monkeypatch.setitem(sys.modules, "topologicpy.TGraph", tgraph_module)

    graph = FakeTGraph()
    assert GraphDB._annotate_graph_result(graph, ontology=False, silent=True) is graph

    graph_a = FakeTGraph()
    graph_b = FakeTGraph()
    result = GraphDB._annotate_graph_result([graph_a, graph_b], generatedBy="unit-test", silent=True)

    assert result == [graph_a, graph_b]
    assert graph_a.dictionary["generated_by"] == "unit-test"
    assert graph_b.dictionary["generated_by"] == "unit-test"
