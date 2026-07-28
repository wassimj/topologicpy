


"""Unit tests for topologicpy.Neo4j.

These tests use in-memory fake Neo4j and TopologicPy support modules. They do
not require a running Neo4j server, credentials, the neo4j Python package, or
TopologicCore.
"""

from __future__ import annotations

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

import json
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


# -----------------------------------------------------------------------------
# Fake neo4j package installed before importing topologicpy.Neo4j.
# -----------------------------------------------------------------------------


class FakeNode:
    def __init__(self, element_id="n0", labels=None, **props):
        self.element_id = element_id
        self.id = element_id
        self.labels = set(labels or [])
        self._props = dict(props)

    def items(self):
        return self._props.items()

    def __getitem__(self, key):
        return self._props[key]


class FakeRelationship:
    def __init__(self, element_id="r0", start_node=None, end_node=None, rel_type="REL", **props):
        self.element_id = element_id
        self.id = element_id
        self.start_node = start_node
        self.end_node = end_node
        self.type = rel_type
        self._props = dict(props)

    def items(self):
        return self._props.items()

    def __getitem__(self, key):
        return self._props[key]


class FakePath:
    def __init__(self, nodes=None, relationships=None):
        self.nodes = list(nodes or [])
        self.relationships = list(relationships or [])


class FakeGraphDatabase:
    created = []

    @staticmethod
    def driver(url, auth=None):
        driver = FakeDriver(handler=lambda cypher, parameters: [])
        driver.url = url
        driver.auth = auth
        FakeGraphDatabase.created.append(driver)
        return driver


neo4j_pkg = types.ModuleType("neo4j")
neo4j_pkg.GraphDatabase = FakeGraphDatabase
neo4j_graph_pkg = types.ModuleType("neo4j.graph")
neo4j_graph_pkg.Node = FakeNode
neo4j_graph_pkg.Relationship = FakeRelationship
neo4j_graph_pkg.Path = FakePath
sys.modules["neo4j"] = neo4j_pkg
sys.modules["neo4j.graph"] = neo4j_graph_pkg

from topologicpy import Neo4j as neo4j_module
from topologicpy.Neo4j import Neo4j


# -----------------------------------------------------------------------------
# Fake Neo4j driver/session/transaction.
# -----------------------------------------------------------------------------


class FakeRecord(dict):
    def keys(self):
        return super().keys()


class FakeTx:
    def __init__(self, driver):
        self.driver = driver

    def run(self, cypher, parameters=None):
        params = dict(parameters or {})
        self.driver.calls.append({"cypher": cypher, "parameters": params})
        if self.driver.raise_on and self.driver.raise_on in cypher:
            raise RuntimeError("forced query failure")
        result = self.driver.handler(cypher, params)
        if result is None:
            return []
        return result


class FakeSession:
    def __init__(self, driver, **kwargs):
        self.driver = driver
        self.kwargs = kwargs
        self.driver.sessions.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, fn):
        self.driver.write_count += 1
        return fn(FakeTx(self.driver))

    def execute_read(self, fn):
        self.driver.read_count += 1
        return fn(FakeTx(self.driver))

    def write_transaction(self, fn):
        return self.execute_write(fn)

    def read_transaction(self, fn):
        return self.execute_read(fn)


class FakeDriver:
    def __init__(self, handler=None):
        self.handler = handler or (lambda cypher, parameters: [])
        self.calls = []
        self.sessions = []
        self.closed = False
        self.read_count = 0
        self.write_count = 0
        self.raise_on = None
        self.verified = []

    def session(self, **kwargs):
        return FakeSession(self, **kwargs)

    def close(self):
        self.closed = True

    def verify_connectivity(self, **kwargs):
        self.verified.append(kwargs)


# -----------------------------------------------------------------------------
# Fake minimal TopologicPy modules used by graph conversion methods.
# -----------------------------------------------------------------------------


def install_fake_topologic_modules(monkeypatch):
    class FakeVertex(dict):
        pass

    class FakeEdge(dict):
        pass

    class FakeGraph(dict):
        pass

    class Dictionary:
        @staticmethod
        def ValueAtKey(dictionary, key, default=None):
            if isinstance(dictionary, dict):
                return dictionary.get(key, default)
            return default

        @staticmethod
        def PythonDictionary(dictionary):
            return dict(dictionary or {})

        @staticmethod
        def ByPythonDictionary(dictionary):
            return dict(dictionary or {})

        @staticmethod
        def SetValueAtKey(dictionary, key, value):
            d = dict(dictionary or {})
            d[key] = value
            return d

        @staticmethod
        def ByKeyValue(key, value):
            return {key: value}

    class Topology:
        @staticmethod
        def Dictionary(topology):
            return dict(getattr(topology, "dictionary", topology.get("dictionary", {})) if isinstance(topology, dict) else getattr(topology, "dictionary", {}))

        @staticmethod
        def SetDictionary(topology, dictionary, silent=True):
            if isinstance(topology, dict):
                topology["dictionary"] = dict(dictionary or {})
            else:
                topology.dictionary = dict(dictionary or {})
            return topology

        @staticmethod
        def UUID(topology):
            d = Topology.Dictionary(topology)
            return d.get("uuid") or d.get("graph_id")

        @staticmethod
        def IsInstance(topology, type_name):
            return isinstance(topology, FakeGraph) and str(type_name).lower() == "graph"

    class Vertex:
        @staticmethod
        def ByCoordinates(x, y, z):
            return FakeVertex(x=float(x), y=float(y), z=float(z), dictionary={})

    class Edge:
        @staticmethod
        def ByStartVertexEndVertex(a, b):
            return FakeEdge(start=a, end=b, dictionary={})

    class Graph:
        @staticmethod
        def ByVerticesEdges(vertices, edges, ontology=True):
            return FakeGraph(vertices=list(vertices or []), edges=list(edges or []), dictionary={})

        @staticmethod
        def MeshData(graph, mantissa=6):
            return dict(graph.get("mesh_data", {}))

        @staticmethod
        def Vertices(graph):
            return list(graph.get("vertices", []))

        @staticmethod
        def Edges(graph):
            return list(graph.get("edges", []))

        @staticmethod
        def ByCSVPath(**kwargs):
            return [FakeGraph(dictionary={"graph_id": "csv_graph", "label": "CSV"}, mesh_data={"vertices": [], "edges": []})]

    class TGraph:
        def __init__(self, dictionary=None, vertices=None, edges=None):
            self._dictionary = dict(dictionary or {})
            self._vertices = list(vertices or [])
            self._edges = list(edges or [])
            self.vertex_updates = []
            self.edge_updates = []

        def SetDictionary(self, dictionary):
            self._dictionary = dict(dictionary or {})
            return self

        def SetVertexDictionary(self, index, dictionary):
            self.vertex_updates.append((index, dict(dictionary or {})))
            for vertex in self._vertices:
                if vertex.get("index") == index:
                    vertex["dictionary"] = dict(dictionary or {})
            return self

        def SetEdgeDictionary(self, index, dictionary):
            self.edge_updates.append((index, dict(dictionary or {})))
            for edge in self._edges:
                if edge.get("index") == index:
                    edge["dictionary"] = dict(dictionary or {})
            return self

        @staticmethod
        def Dictionary(graph):
            return dict(graph._dictionary)

        @staticmethod
        def Vertices(graph, asTopologic=False, active=True):
            return list(graph._vertices)

        @staticmethod
        def Edges(graph, asTopologic=False, active=True):
            return list(graph._edges)

        @staticmethod
        def Coordinates(graph, index, default=None):
            for vertex in graph._vertices:
                if vertex.get("index") == index:
                    d = vertex.get("dictionary", {})
                    if all(k in d for k in ("x", "y", "z")):
                        return [d["x"], d["y"], d["z"]]
            return default

        @staticmethod
        def ByGraph(graph, ontology=True):
            return graph

        @staticmethod
        def ByVerticesEdges(vertices=None, edges=None, dictionary=None, ontology=True, silent=True):
            return TGraph(dictionary=dictionary or {}, vertices=vertices or [], edges=edges or [])

        @staticmethod
        def ByCSVPath(**kwargs):
            return [TGraph(dictionary={"graph_id": "csv_tgraph", "label": "CSV"}, vertices=[], edges=[])]

    modules = {
        "topologicpy.Dictionary": ("Dictionary", Dictionary),
        "topologicpy.Topology": ("Topology", Topology),
        "topologicpy.Vertex": ("Vertex", Vertex),
        "topologicpy.Edge": ("Edge", Edge),
        "topologicpy.Graph": ("Graph", Graph),
        "topologicpy.TGraph": ("TGraph", TGraph),
    }
    for mod_name, (attr_name, cls) in modules.items():
        mod = types.ModuleType(mod_name)
        setattr(mod, attr_name, cls)
        monkeypatch.setitem(sys.modules, mod_name, mod)
    return types.SimpleNamespace(
        FakeVertex=FakeVertex,
        FakeEdge=FakeEdge,
        FakeGraph=FakeGraph,
        Dictionary=Dictionary,
        Topology=Topology,
        Vertex=Vertex,
        Edge=Edge,
        Graph=Graph,
        TGraph=TGraph,
    )


def schema_ok_handler(cypher, parameters):
    if "RETURN count(g) > 0 AS exists" in cypher:
        return [FakeRecord(exists=False)]
    if "RETURN g.id AS id" in cypher:
        return [FakeRecord(id=parameters.get("id"))]
    return []


# -----------------------------------------------------------------------------
# Tests.
# -----------------------------------------------------------------------------


def test_private_helpers_and_property_extractors_are_defensive():
    assert json.loads(neo4j_module._json_dumps({"x": {1, 2}}))["x"] in ([1, 2], [2, 1])
    assert neo4j_module._json_loads('{"a": 1}') == {"a": 1}
    assert neo4j_module._json_loads("bad", {"fallback": True}) == {"fallback": True}
    assert neo4j_module._unique_preserve_order([[1], [1], {"a": 2}, {"a": 2}]) == [[1], {"a": 2}]
    assert Neo4j._sanitize_identifier("9 bad-name!", default="X") == "_9_bad_name_"

    node = FakeNode("n1", labels=["Room"], name="A")
    rel = FakeRelationship("r1", rel_type="ADJ", weight=2)
    assert Neo4j._node_properties(node)["labels"] == ["Room"]
    assert Neo4j._node_properties(node)["id"] == "n1"
    assert Neo4j._relationship_properties(rel)["type"] == "ADJ"
    assert Neo4j._relationship_properties(rel)["id"] == "r1"


def test_connect_manager_close_and_invalid_driver(monkeypatch):
    fake_gdb = type("GDB", (), {"driver": staticmethod(FakeGraphDatabase.driver)})
    monkeypatch.setattr(neo4j_module, "GraphDatabase", fake_gdb)

    driver = Neo4j.Connect("bolt://host", "neo4j", "pw", database="neo4j", silent=True)
    assert driver.url == "bolt://host"
    assert driver.auth == ("neo4j", "pw")
    assert driver.verified[-1] == {"database": "neo4j"}

    manager = Neo4j.Manager("bolt://host2", "user", "pw", silent=True)
    assert manager.url == "bolt://host2"
    assert manager.verified[-1] == {}
    assert Neo4j.Close(manager, silent=True) is True
    assert manager.closed is True
    assert Neo4j.Close(object(), silent=True) is False


def test_execute_query_and_batch_execute_use_read_write_transactions():
    driver = FakeDriver(handler=lambda cypher, params: [FakeRecord(value=params.get("x", 0))])

    read_rows = Neo4j.Query(driver, "RETURN $x AS value", parameters={"x": 7}, database="db", silent=True)
    write_rows = Neo4j.Execute(driver, "CREATE (n)", write=True, silent=True)

    assert read_rows == [FakeRecord(value=7)]
    assert write_rows == [FakeRecord(value=0)]
    assert driver.read_count == 1
    assert driver.write_count == 1
    assert driver.sessions[0] == {"database": "db"}
    assert Neo4j.Execute(object(), "RETURN 1", silent=True) is None
    assert Neo4j.Execute(driver, "", silent=True) is None

    batches = []

    def batch_handler(cypher, params):
        batches.append(list(params.get("rows", [])))
        return []

    batch_driver = FakeDriver(handler=batch_handler)
    assert Neo4j.BatchExecute(batch_driver, "UNWIND $rows AS row RETURN row", [1, 2, 3, 4, 5], batchSize=2, silent=True) is True
    assert batches == [[1, 2], [3, 4], [5]]
    assert Neo4j.BatchExecute(batch_driver, "RETURN 1", [], silent=True) is True
    assert Neo4j.BatchExecute(batch_driver, "RETURN 1", "bad", silent=True) is False


def test_schema_database_management_index_constraint_and_counts():
    def handler(cypher, params):
        if "SHOW CONSTRAINTS" in cypher:
            return [FakeRecord(name="c_one")]
        if "SHOW INDEXES" in cypher:
            return [FakeRecord(name="i_one")]
        if "count(n) AS count" in cypher:
            return [FakeRecord(count=3)]
        if "count(r) AS count" in cypher:
            return [FakeRecord(count=4)]
        if "db.labels" in cypher:
            return [FakeRecord(label="Graph"), FakeRecord(label="Vertex")]
        if "db.relationshipTypes" in cypher:
            return [FakeRecord(relationshipType="Edge")]
        return []

    driver = FakeDriver(handler=handler)
    assert Neo4j.EnsureSchema(driver, database="db", silent=True) is True
    assert any("vertex_uid_unique" in call["cypher"] for call in driver.calls)

    assert Neo4j.EmptyDatabase(driver, dropSchema=True, recreateSchema=False, database="db", silent=True) is True
    assert any("DROP CONSTRAINT `c_one`" in call["cypher"] for call in driver.calls)
    assert any("DROP INDEX `i_one`" in call["cypher"] for call in driver.calls)
    assert Neo4j.Reset(driver, silent=True) is True

    assert Neo4j.CreateIndex(driver, "Bad Label", "some-property", silent=True) is True
    assert "CREATE INDEX" in driver.calls[-1]["cypher"]
    assert "Bad_Label" in driver.calls[-1]["cypher"]
    assert Neo4j.CreateConstraint(driver, "Node", ["graph id", "id"], unique=True, silent=True) is True
    assert "REQUIRE (n.graph_id, n.id) IS UNIQUE" in driver.calls[-1]["cypher"]
    assert Neo4j.CreateConstraint(driver, "Node", ["a", "b"], unique=False, silent=True) is False

    assert Neo4j.CountNodes(driver, label="Graph", silent=True) == 3
    assert Neo4j.CountRelationships(driver, relationshipType="Edge", silent=True) == 4
    assert Neo4j.Labels(driver, silent=True) == ["Graph", "Vertex"]
    assert Neo4j.RelationshipTypes(driver, silent=True) == ["Edge"]
    assert Neo4j.Info(driver, silent=True)["nodeCount"] == 3


def test_schema_returns_false_when_any_statement_fails():
    def handler(cypher, params):
        if "vertex_uid_unique" in cypher:
            raise RuntimeError("schema failure")
        return []

    driver = FakeDriver(handler=handler)
    assert Neo4j.EnsureSchema(driver, silent=True) is False


def test_query_helpers_match_delete_validate_schema_listgraphs_and_dataframe(monkeypatch):
    node = FakeNode("n1", labels=["Vertex"], id="v1", label="Room")

    def handler(cypher, params):
        if "MATCH (n:Vertex)" in cypher and "RETURN n" in cypher:
            return [FakeRecord(n=node)]
        if "duplicate" in cypher or "WITH n.topologic_id AS id" in cypher:
            return [FakeRecord(count=1)]
        if "WHERE n.topologic_id IS NULL" in cypher:
            return [FakeRecord(count=2)]
        if "WHERE NOT (n)--()" in cypher:
            return [FakeRecord(count=3)]
        if "SHOW INDEXES" in cypher:
            return [FakeRecord(name="idx")]
        if "SHOW CONSTRAINTS" in cypher:
            return [FakeRecord(name="con")]
        if "MATCH (g:Graph)" in cypher and "RETURN g.id AS id" in cypher:
            return [FakeRecord(id="g1", label="A", num_nodes=2, num_edges=1, props='{"k": 1}')]
        if "RETURN $x AS x" in cypher:
            return [FakeRecord(x=5, n=node)]
        return []

    driver = FakeDriver(handler=handler)
    assert Neo4j.MatchNodes(driver, label="Vertex", properties={"label": "Room"}, silent=True)[0]["label"] == "Room"
    assert Neo4j.DeleteNodes(driver, label="Vertex", properties={"label": "Room"}, silent=True) is True
    assert Neo4j.DeleteRelationships(driver, relationshipType="Edge", silent=True) is True
    assert Neo4j.Validate(driver, silent=True) == {"duplicateNodeIds": 1, "missingNodeIds": 2, "orphanNodes": 3}
    assert Neo4j.Schema(driver, silent=True) == {"indexes": [{"name": "idx"}], "constraints": [{"name": "con"}]}
    assert Neo4j.ListGraphs(driver, where={"label": "A", "min_nodes": 1}, limit=-5, offset=-1, silent=True)[0]["id"] == "g1"
    assert driver.calls[-1]["parameters"]["limit"] == 0
    assert driver.calls[-1]["parameters"]["offset"] == 0

    class FakeDataFrame:
        def __init__(self, rows):
            self.rows = rows

    pandas_mod = types.ModuleType("pandas")
    pandas_mod.DataFrame = FakeDataFrame
    monkeypatch.setitem(sys.modules, "pandas", pandas_mod)
    df = Neo4j.ToDataFrame(driver, "RETURN $x AS x", parameters={"x": 5}, silent=True)
    assert isinstance(df, FakeDataFrame)
    assert df.rows[0]["x"] == 5


def test_upsert_tgraph_uses_vertex_uid_and_counts_bidirectional_self_loop(monkeypatch):
    fakes = install_fake_topologic_modules(monkeypatch)
    graph = fakes.TGraph(
        dictionary={"graph_id": "g1", "label": "Demo"},
        vertices=[
            {"index": 0, "dictionary": {"id": "a", "label": "Room", "x": 0, "y": 0, "z": 0}},
            {"index": 1, "dictionary": {"id": "b", "label": "Door", "x": 1, "y": 0, "z": 0}},
        ],
        edges=[
            {"index": 0, "src": 0, "dst": 1, "dictionary": {"label": "ADJ"}},
            {"index": 1, "src": 1, "dst": 1, "dictionary": {"label": "SELF"}},
        ],
    )
    driver = FakeDriver(handler=schema_ok_handler)

    gid = Neo4j.UpsertGraph(driver, graph, bidirectional=True, ontology=True, silent=True)
    assert gid == "g1"

    graph_create = next(call for call in driver.calls if "CREATE (g:Graph" in call["cypher"])
    assert graph_create["parameters"]["num_nodes"] == 2
    assert graph_create["parameters"]["num_edges"] == 3

    vertex_batch = next(call for call in driver.calls if "CREATE (:Vertex" in call["cypher"])
    vertex_rows = vertex_batch["parameters"]["rows"]
    assert vertex_rows[0]["uid"] == "g1:a"
    assert vertex_rows[1]["uid"] == "g1:b"
    assert json.loads(vertex_rows[0]["props"])["_db_id"] == "g1:a"

    edge_batch = next(call for call in driver.calls if "CREATE (a)-[:Edge" in call["cypher"])
    edge_rows = edge_batch["parameters"]["rows"]
    assert len(edge_rows) == 3
    assert {tuple((r["a_uid"], r["b_uid"])) for r in edge_rows} == {("g1:a", "g1:b"), ("g1:b", "g1:a"), ("g1:b", "g1:b")}
    assert "MATCH (a:Vertex {uid: row.a_uid})" in edge_batch["cypher"]


def test_upsert_duplicate_overwrite_delete_and_invalid_batch(monkeypatch):
    fakes = install_fake_topologic_modules(monkeypatch)
    graph = fakes.TGraph(
        dictionary={"graph_id": "g2", "label": "Existing"},
        vertices=[{"index": 0, "dictionary": {"id": "a", "label": "A", "x": 0, "y": 0, "z": 0}}],
        edges=[],
    )

    def exists_handler(cypher, params):
        if "RETURN count(g) > 0 AS exists" in cypher:
            return [FakeRecord(exists=True)]
        if "RETURN g.id AS id" in cypher:
            return [FakeRecord(id=params.get("id"))]
        return []

    driver = FakeDriver(handler=exists_handler)
    assert Neo4j.UpsertGraph(driver, graph, overwrite=False, silent=True) is None
    assert Neo4j.UpsertGraph(driver, graph, overwrite=True, silent=True) == "g2"
    assert any("DELETE r" in call["cypher"] for call in driver.calls)

    fail_driver = FakeDriver(handler=lambda cypher, params: [])
    fail_driver.raise_on = "MATCH (v:Vertex)"
    assert Neo4j.DeleteGraph(fail_driver, "g", silent=True) is False


def test_graph_by_id_reconstructs_topologic_graph(monkeypatch):
    fakes = install_fake_topologic_modules(monkeypatch)

    def handler(cypher, params):
        if "MATCH (g:Graph {id:$id})" in cypher:
            return [FakeRecord(id="g1", label="Demo", ontology_class="top:Graph", category="graph", uri=None, source=None, generated_by="test", num_nodes=2, num_edges=1, props='{"label":"Demo"}')]
        if "MATCH (v:Vertex)" in cypher and "ORDER BY v.id" in cypher:
            return [
                FakeRecord(id="a", label="Room", ontology_class="top:Node", category="node", uri=None, ifc_class=None, ifc_guid=None, x=0, y=0, z=0, props='{"id":"a"}'),
                FakeRecord(id="b", label="Door", ontology_class="top:Node", category="node", uri=None, ifc_class=None, ifc_guid=None, x=1, y=0, z=0, props='{"id":"b"}'),
            ]
        if "MATCH (a:Vertex)-[r:Edge]->(b:Vertex)" in cypher:
            return [FakeRecord(a_id="a", b_id="b", label="ADJ", ontology_class="top:Relationship", category="relationship", uri=None, props='{"label":"ADJ"}')]
        return []

    graph = Neo4j.GraphByID(FakeDriver(handler=handler), "g1", ontology=True, silent=True)
    assert isinstance(graph, fakes.FakeGraph)
    assert len(graph["vertices"]) == 2
    assert len(graph["edges"]) == 1
    assert graph["dictionary"]["generated_by"] == "Neo4j.GraphByID"
    assert graph["vertices"][0]["dictionary"]["id"] == "a"
    assert graph["edges"][0]["dictionary"]["label"] == "ADJ"


def test_graphs_by_query_and_tograph_collect_nodes_relationships_paths(monkeypatch):
    fakes = install_fake_topologic_modules(monkeypatch)
    n1 = FakeNode("n1", labels=["Vertex"], label="A", x=0, y=0, z=0)
    n2 = FakeNode("n2", labels=["Vertex"], label="B", x=1, y=0, z=0)
    rel = FakeRelationship("r1", start_node=n1, end_node=n2, rel_type="Edge", weight=3)
    path = FakePath(nodes=[n1, n2], relationships=[rel])
    driver = FakeDriver(handler=lambda cypher, params: [FakeRecord(result=path), FakeRecord(extra=[n1, rel])])

    graphs = Neo4j.GraphsByQuery(driver, "MATCH p RETURN p", silent=True)
    assert len(graphs) == 1
    assert isinstance(graphs[0], fakes.FakeGraph)
    assert len(graphs[0]["vertices"]) == 2
    assert len(graphs[0]["edges"]) == 1
    assert graphs[0]["edges"][0]["dictionary"]["type"] == "Edge"

    graph = Neo4j.ToGraph(driver, cypher="MATCH p RETURN p", silent=True)
    assert isinstance(graph, fakes.FakeGraph)
    assert len(graph["vertices"]) == 2
    assert len(graph["edges"]) == 1
    assert Neo4j.ToGraph(FakeDriver(handler=lambda cypher, params: []), graphID="g123", silent=True) is None


def test_corpus_analytics_and_ontology_queries_decode_rows():
    def handler(cypher, params):
        if "RETURN a.label AS a_label" in cypher:
            return [FakeRecord(a_label="Room", b_label="Door", count=2), FakeRecord(a_label="Door", b_label="Room", count=3)]
        if "max_degree" in cypher:
            return [FakeRecord(max_degree=5)]
        if "b_label_lc AS label" in cypher:
            return [FakeRecord(label="door", count=4), FakeRecord(label="ignore", count=99)]
        if "b.id AS id" in cypher and "a.label AS attach_label" in cypher:
            return [FakeRecord(id="v1", graph_id="g", label="Door", x=1, y=2, z=3, props='{"p":1}', attach_label="Room", frequency=7)]
        if "MATCH (g:Graph)" in cypher and "g.ontology_class = $ontology_class" in cypher:
            return [FakeRecord(id="g", label="G", ontology_class="top:Graph", category="graph", uri=None, num_nodes=1, num_edges=0, props='{"a":2}')]
        if "MATCH (v:Vertex)" in cypher and "v.ontology_class = $ontology_class" in cypher:
            return [FakeRecord(id="v", graph_id="g", label="V", ontology_class="top:Node", category="node", uri=None, ifc_class=None, ifc_guid=None, x=0, y=0, z=0, props='{"b":3}')]
        if "MATCH (g:Graph) RETURN coalesce" in cypher:
            return [FakeRecord(ontology_class="top:Graph", count=1)]
        if "MATCH (v:Vertex) RETURN coalesce" in cypher:
            return [FakeRecord(ontology_class="top:Node", count=2)]
        if "MATCH ()-[r:Edge]->() RETURN coalesce" in cypher:
            return [FakeRecord(ontology_class="top:Relationship", count=3)]
        return []

    driver = FakeDriver(handler=handler)
    assert Neo4j.FetchAllPairs(driver, undirected=True, silent=True) == [{"a_label": "Door", "b_label": "Room", "pair": ["Door", "Room"], "count": 5}]
    assert Neo4j.FetchAllPairs(driver, undirected=False, silent=True)[0]["count"] == 2
    assert Neo4j.MaxNeighborsForLabel(driver, "Room", silent=True) == 5
    candidates = Neo4j.CandidateCountsForLabels(driver, ["Room", "Room"], excludeLabels=["ignore"], limit=-5, silent=True)
    assert candidates == [{"label": "door", "count": 4}]
    assert driver.calls[-1]["parameters"]["limit"] == 1
    example = Neo4j.FindBestExampleForLabel(driver, "Door", attachTo="Room", silent=True)
    assert example["props"] == {"p": 1}
    assert Neo4j.GraphsByOntologyClass(driver, "top:Graph", silent=True)[0]["props"] == {"a": 2}
    assert Neo4j.VerticesByOntologyClass(driver, "top:Node", graphID="g", silent=True)[0]["props"] == {"b": 3}
    assert Neo4j.OntologySummary(driver, silent=True) == {
        "graphs": [{"ontology_class": "top:Graph", "count": 1}],
        "vertices": [{"ontology_class": "top:Node", "count": 2}],
        "edges": [{"ontology_class": "top:Relationship", "count": 3}],
    }


def test_csv_import_and_compatibility_wrappers_forward_parameters(monkeypatch):
    fakes = install_fake_topologic_modules(monkeypatch)
    driver = FakeDriver(handler=schema_ok_handler)
    result = Neo4j.ByCSVPath(driver, "/tmp/fake_csv", database="db", silent=True)
    assert result == {"graphs_upserted": 1, "graph_ids": ["csv_tgraph"]}

    graph = fakes.TGraph(dictionary={"graph_id": "wrap"}, vertices=[], edges=[])
    called = []

    def fake_upsert(manager, graph_arg, **kwargs):
        called.append(kwargs)
        return "wrap"

    monkeypatch.setattr(Neo4j, "UpsertGraph", staticmethod(fake_upsert))
    assert Neo4j.ByGraph(driver, graph, graphID="explicit", nodeLabelKey="node_label", relationshipTypeKey="rel_type", bidirectional=False, overwrite=False, database="db", silent=True) is driver
    assert called[-1]["vertexLabelKey"] == "node_label"
    assert called[-1]["edgeLabelKey"] == "rel_type"
    assert called[-1]["bidirectional"] is False
    assert called[-1]["overwrite"] is False
    assert called[-1]["database"] == "db"

    assert Neo4j.MergeGraph(driver, graph, nodeLabelKey="nl", relationshipTypeKey="rt", bidirectional=False, silent=True) is driver
    assert called[-1]["vertexLabelKey"] == "nl"
    assert called[-1]["edgeLabelKey"] == "rt"
    assert called[-1]["bidirectional"] is False


def test_neighborhood_validates_inputs_and_delegates_to_tograph(monkeypatch):
    driver = FakeDriver(handler=lambda c, p: [])
    assert Neo4j.Neighborhood(object(), "n", silent=True) is None
    assert Neo4j.Neighborhood(driver, "", silent=True) is None
    assert Neo4j.Neighborhood(driver, "n", depth=0, silent=True) is None

    captured = {}

    def fake_to_graph(driver_arg, **kwargs):
        captured.update(kwargs)
        return "graph"

    monkeypatch.setattr(Neo4j, "ToGraph", staticmethod(fake_to_graph))
    assert Neo4j.Neighborhood(driver, "node-1", depth=2, silent=True) == "graph"
    assert "*1..2" in captured["cypher"]
    assert captured["parameters"] == {"nodeId": "node-1"}
