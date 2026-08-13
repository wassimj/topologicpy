# Auto-generated PythonOCC backend parity test
# Copied from test_GraphRAG.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

import json
import sys
import types

import pytest

from topologicpy.GraphRAG import GraphRAG, _GraphRAGConfig


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class FakeTGraph:
    def __init__(self, directed=False, allowSelfLoops=True, allowParallelEdges=True, dictionary=None):
        self.directed = directed
        self.allowSelfLoops = allowSelfLoops
        self.allowParallelEdges = allowParallelEdges
        self.dictionary = dict(dictionary or {})
        self._vertices = []
        self._edges = []

    def AddVertex(self, dictionary=None):
        idx = len(self._vertices)
        d = dict(dictionary or {})
        record = {
            "index": idx,
            "id": d.get("id", f"n{idx}"),
            "label": d.get("label", f"Node{idx}"),
            "dictionary": d,
        }
        self._vertices.append(record)
        return idx

    def AddEdge(self, src, dst, dictionary=None):
        if src is None or dst is None:
            return None
        if src == dst and not self.allowSelfLoops:
            return None
        if not self.allowParallelEdges:
            for edge in self._edges:
                if {edge.get("src"), edge.get("dst")} == {src, dst}:
                    return None
        idx = len(self._edges)
        d = dict(dictionary or {})
        record = {
            "index": idx,
            "src": int(src),
            "dst": int(dst),
            "label": d.get("label"),
            "dictionary": d,
        }
        self._edges.append(record)
        return idx

    def SetDictionary(self, dictionary):
        self.dictionary = dict(dictionary or {})
        return self

    @staticmethod
    def Vertices(graph, *args, **kwargs):
        return list(graph._vertices)

    @staticmethod
    def Edges(graph, *args, **kwargs):
        return list(graph._edges)

    @staticmethod
    def Coordinates(graph, index, default=None):
        try:
            d = graph._vertices[int(index)].get("dictionary", {}) or {}
            return [d.get("x"), d.get("y"), d.get("z")]
        except Exception:
            return default

    @staticmethod
    def AdjacentVertices(graph, index, mode="all"):
        out = []
        for edge in graph._edges:
            if edge.get("src") == index:
                out.append(edge.get("dst"))
            elif edge.get("dst") == index:
                out.append(edge.get("src"))
        return out

    @staticmethod
    def VertexIndex(graph, vertex):
        if isinstance(vertex, int):
            return vertex if 0 <= vertex < len(graph._vertices) else None
        if isinstance(vertex, dict):
            idx = vertex.get("index")
            if isinstance(idx, int):
                return idx
        for i, candidate in enumerate(graph._vertices):
            if candidate is vertex:
                return i
        return None

    @staticmethod
    def Vertex(graph, index):
        return graph._vertices[int(index)]

    @staticmethod
    def Edge(graph, index):
        return graph._edges[int(index)]


def install_fake_tgraph(monkeypatch):
    module = types.ModuleType("topologicpy.TGraph")
    module.TGraph = FakeTGraph
    monkeypatch.setitem(sys.modules, "topologicpy.TGraph", module)
    return FakeTGraph


def install_fake_graphdb(monkeypatch):
    class FakeGraphDB:
        calls = []

        @staticmethod
        def CandidateCountsForLabels(graphdb, labels, excludeLabels=None, limit=50, silent=False):
            FakeGraphDB.calls.append(("CandidateCountsForLabels", list(labels or []), limit))
            return list((graphdb or {}).get("candidates", []))[:limit]

        @staticmethod
        def MaxNeighborsForLabel(graphdb, label, silent=False):
            FakeGraphDB.calls.append(("MaxNeighborsForLabel", label))
            return (graphdb or {}).get("max", {}).get(label)

        @staticmethod
        def FetchAllPairs(graphdb, undirected=True, silent=False):
            FakeGraphDB.calls.append(("FetchAllPairs", undirected))
            return list((graphdb or {}).get("pairs", []))

        @staticmethod
        def FindBestExampleForLabel(graphdb, label, attachTo=None, silent=False):
            FakeGraphDB.calls.append(("FindBestExampleForLabel", label, attachTo))
            return (graphdb or {}).get("best", {}).get(label)

    module = types.ModuleType("topologicpy.GraphDB")
    module.GraphDB = FakeGraphDB
    monkeypatch.setitem(sys.modules, "topologicpy.GraphDB", module)
    return FakeGraphDB


class FakeLLMHandle:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []


def install_fake_llm(monkeypatch):
    class FakeLLM:
        @staticmethod
        def Prompt(llm, prompt, **kwargs):
            llm.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
            if llm.responses:
                return llm.responses.pop(0)
            return '{"action": "stop", "reason": "no more responses"}'

    module = types.ModuleType("topologicpy.LLM")
    module.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "topologicpy.LLM", module)
    return FakeLLM


def make_grag(**kwargs):
    defaults = {"silent": True}
    defaults.update(kwargs)
    return GraphRAG.ByParameters(**defaults)


def make_fake_tgraph_with_two_nodes():
    graph = FakeTGraph(directed=False, allowParallelEdges=False, dictionary={"label": "seed"})
    graph._vertices = [
        {
            "index": 0,
            "id": "A",
            "label": "Living",
            "dictionary": {"x": 1.0, "y": 2.0, "z": 3.0},
        },
        {
            "index": 1,
            "dictionary": {"id": "B", "label": "Kitchen", "x": 4.0, "y": 5.0, "z": 6.0},
        },
    ]
    graph._edges = [
        {"index": 0, "src": 0, "dst": 1, "label": "door", "dictionary": {"weight": 2}},
    ]
    return graph


def test_by_parameters_coerces_numeric_values_and_preserves_configuration():
    grag = GraphRAG.ByParameters(
        graphdb="db",
        llm="llm",
        promptContext="Context",
        tolerance="not-a-number",
        maxCandidates="0",
        maxPairs=None,
        ontology=False,
        silent=True,
    )

    assert isinstance(grag, _GraphRAGConfig)
    assert grag.graphdb == "db"
    assert grag.llm == "llm"
    assert grag.promptContext == "Context"
    assert grag.tolerance == pytest.approx(0.0001)
    assert grag.maxCandidates == 1
    assert grag.maxPairs == 40
    assert grag.ontology is False
    assert "_GraphRAGConfig" in repr(grag)


def test_record_dictionary_merges_top_level_and_nested_metadata():
    record = {
        "index": 7,
        "id": "top-id",
        "label": "TopLabel",
        "dictionary": {"id": "nested-id", "category": "node"},
        "representation": "discard-me",
    }

    merged = GraphRAG._record_dictionary(record)

    assert merged["index"] == 7
    assert merged["id"] == "nested-id"
    assert merged["label"] == "TopLabel"
    assert merged["category"] == "node"
    assert "representation" not in merged
    assert GraphRAG._vertex_value(record, "id") == "nested-id"


def test_json_coercion_candidate_labels_pairs_and_max_neighbor_values():
    assert GraphRAG._coerce_json('```JSON\n{"action": "stop"}\n```') == {"action": "stop"}
    assert GraphRAG._coerce_json('prefix [{"action": "stop"}] suffix') == [{"action": "stop"}]
    assert GraphRAG._coerce_json("not json") == {}

    candidates = [
        {"label": "Kitchen"},
        {"candidate": "Bath"},
        {"neighbor_label": "Hall"},
        ("Bedroom", 4),
        "Study",
        None,
        "Study",
    ]
    assert GraphRAG._candidate_labels(candidates) == ["Kitchen", "Bath", "Hall", "Bedroom", "Study"]

    pairs = [
        {"a": "Kitchen", "b": "Living"},
        {"src": "Garage", "dst": "Street"},
        ("Bath", "Bedroom"),
    ]
    assert GraphRAG._filter_pairs_by_labels(pairs, ["living"], limit=3) == [pairs[0]]
    assert GraphRAG._max_neighbor_value({"max_neighbors": [{"value": "3"}]}) == 3
    assert GraphRAG._max_neighbor_value("4.0") == 4


def test_normalize_action_aliases_and_action_schema_are_consistent():
    add = GraphRAG.NormalizeAction({"action": "add vertex", "b_label": "Bedroom", "a_id": "n1"})
    assert add["action"] == "add_node"
    assert add["label"] == "Bedroom"
    assert add["attach_to_id"] == "n1"

    connect = GraphRAG.NormalizeAction({"action": "link", "src": "n1", "target": "n2", "label": "adjacent"})
    assert connect["action"] == "connect"
    assert connect["a_id"] == "n1"
    assert connect["b_id"] == "n2"
    assert connect["edge_label"] == "adjacent"
    assert "label" not in connect
    assert connect["_raw_label"] == "adjacent"

    remove = GraphRAG.NormalizeAction({"action": "delete edge", "from_id": "n1", "to_id": "n2"})
    assert remove["action"] == "remove_edge"
    assert remove["a_id"] == "n1"
    assert remove["b_id"] == "n2"

    schema = GraphRAG.ActionSchema()
    assert set(schema["properties"]["action"]["enum"]) == {"add_node", "connect", "remove_node", "remove_edge", "stop"}


def test_prompt_handles_incomplete_summaries_and_includes_matrix_actions():
    grag = make_grag(promptContext="Graph assistant")
    summary = {
        "nodes": [{"id": "n1"}],
        "edges": [{"src": "n1"}],
    }

    prompt = GraphRAG.Prompt(grag, summary, evidence={}, description="add a room", silent=True)
    payload = json.loads(prompt[prompt.index("{"):])

    assert payload["target_description"] == "add a room"
    assert payload["current_graph"]["nodes"][0]["id"] == "n1"
    assert "remove_node" in payload["allowed_actions"]
    assert "remove_edge" in payload["allowed_actions"]


def test_summarize_tgraph_preserves_top_level_record_metadata(monkeypatch):
    install_fake_tgraph(monkeypatch)
    graph = make_fake_tgraph_with_two_nodes()
    grag = make_grag()

    summary = GraphRAG.SummarizeGraph(grag, graph, silent=True)

    assert summary["num_nodes"] == 2
    assert summary["num_edges"] == 1
    assert summary["nodes"][0]["id"] == "A"
    assert summary["nodes"][0]["label"] == "Living"
    assert summary["nodes"][0]["degree"] == 1
    assert summary["nodes"][0]["x"] == 1.0
    assert summary["nodes"][1]["id"] == "B"
    assert summary["edges"][0]["src"] == "A"
    assert summary["edges"][0]["dst"] == "B"
    assert summary["edges"][0]["label"] == "door"
    assert summary["edges"][0]["props"]["weight"] == 2


def test_matrix_seed_state_deduplicates_ids_and_matrix_summary_uses_custom_keys(monkeypatch):
    install_fake_tgraph(monkeypatch)
    graph = FakeTGraph()
    graph._vertices = [
        {"index": 0, "dictionary": {"id": "same", "label": "Room"}},
        {"index": 1, "dictionary": {"id": "same", "label": "Room"}},
    ]
    graph._edges = [{"index": 0, "src": 0, "dst": 1, "dictionary": {"label": "adjacent"}}]
    grag = make_grag(ontologyClassKey="oclass", categoryKey="cat")

    state = GraphRAG._matrix_seed_state(grag, graph, silent=True)
    summary = GraphRAG._matrix_summary(state, grag=grag)

    assert [node["id"] for node in state["nodes"]] == ["same", "n1"]
    # The duplicated source ids are intentionally de-duplicated before matrix
    # edge reconstruction, so the ambiguous original edge is not recreated.
    assert state["matrix"][0][1] == 0
    assert summary["nodes"][0]["props"]["oclass"] == "top:Node"
    assert summary["nodes"][0]["category"] == "node"

    state["matrix"][0][1] = 1
    state["matrix"][1][0] = 1
    state["edge_labels"][("n1", "same")] = "adjacent"
    summary = GraphRAG._matrix_summary(state, grag=grag)
    assert summary["edges"][0]["label"] == "adjacent"
    assert summary["edges"][0]["props"]["cat"] == "relationship"


def test_evidence_uses_graphdb_and_computes_expandable_nodes(monkeypatch):
    FakeGraphDB = install_fake_graphdb(monkeypatch)
    graphdb = {
        "candidates": [{"label": "Bedroom"}, {"candidate": "Bath"}],
        "max": {"Living": {"max_neighbors": "2"}, "Kitchen": 1},
        "pairs": [{"a": "Living", "b": "Bedroom"}, {"a": "Garage", "b": "Drive"}],
    }
    grag = make_grag(graphdb=graphdb, maxCandidates=2, maxPairs=5)
    summary = {
        "nodes": [
            {"id": "A", "label": "Living", "degree": 1},
            {"id": "B", "label": "Kitchen", "degree": 1},
        ],
        "edges": [],
    }

    evidence = GraphRAG.Evidence(grag, summary, silent=True)

    assert evidence["candidate_counts"] == graphdb["candidates"]
    assert evidence["max_neighbors"]["Living"] == {"max_neighbors": "2"}
    assert evidence["pairs"] == [{"a": "Living", "b": "Bedroom"}]
    assert [node["label"] for node in evidence["expandable_nodes"]] == ["Living"]
    assert ("CandidateCountsForLabels", ["Living", "Kitchen"], 2) in FakeGraphDB.calls


def test_pick_action_uses_llm_and_parses_fenced_json(monkeypatch):
    install_fake_llm(monkeypatch)
    llm = FakeLLMHandle(['```JSON\n{"action": "add", "label": "Bedroom", "id": "n2", "reason": "needed"}\n```'])
    grag = make_grag(llm=llm)

    action = GraphRAG.PickAction(grag, {"nodes": [], "edges": []}, evidence={}, description="target", silent=True)

    assert action["action"] == "add_node"
    assert action["label"] == "Bedroom"
    assert action["id"] == "n2"
    assert llm.calls
    assert llm.calls[0]["kwargs"]["temperature"] == 0
    assert "target" in llm.calls[0]["prompt"]


def test_matrix_label_helpers_and_corpus_resolution(monkeypatch):
    install_fake_graphdb(monkeypatch)
    graphdb = {
        "best": {"living room": {"canonical_label": "Living Room"}},
        "pairs": [{"label_a": "Living Room", "label_b": "Kitchen"}],
    }
    grag = make_grag(graphdb=graphdb)

    assert GraphRAG._matrix_label_signature("livingRoom") == "livingroom"
    assert "living room" in GraphRAG._matrix_label_variants("livingRoom")
    assert GraphRAG._matrix_extract_label_from_best_example({"node": {"label": "Kitchen"}}) == "Kitchen"
    assert GraphRAG._matrix_resolve_corpus_label(grag, "Living Room", fallback="Living") == "Living Room"
    assert GraphRAG._matrix_pair_values({"source_label": "A", "target_label": "B"}) == ["A", "B"]
    assert GraphRAG._matrix_corpus_pair_supported(grag, "kitchen", "living_room", silent=True) is True


def test_matrix_apply_action_connect_and_remove_edge_respect_corpus_support(monkeypatch):
    install_fake_graphdb(monkeypatch)
    grag = make_grag(graphdb={"max": {"Living": 3, "Kitchen": 3}, "pairs": []})
    state = {
        "nodes": [
            {"id": "A", "label": "Living", "props": {"id": "A", "label": "Living"}},
            {"id": "B", "label": "Kitchen", "props": {"id": "B", "label": "Kitchen"}},
        ],
        "matrix": [[0, 0], [0, 0]],
        "edge_labels": {},
    }

    result = GraphRAG._matrix_apply_action(grag, state, {"action": "connect", "a_id": "A", "b_id": "B", "edge_label": "adjacent"}, silent=True)
    assert result["ok"] is True
    assert state["matrix"][0][1] == 1
    assert state["edge_labels"][("A", "B")] == "adjacent"

    refused = GraphRAG._matrix_apply_action(
        grag,
        state,
        {"action": "remove_edge", "a_id": "A", "b_id": "B"},
        evidence={"pairs": [{"a": "Living", "b": "Kitchen"}]},
        silent=True,
    )
    assert refused["ok"] is False
    assert "Refused" in refused["message"]
    assert state["matrix"][0][1] == 1

    removed = GraphRAG._matrix_apply_action(grag, state, {"action": "remove_edge", "a_id": "A", "b_id": "B"}, evidence={"pairs": []}, silent=True)
    assert removed["ok"] is True
    assert state["matrix"][0][1] == 0


def test_matrix_apply_action_add_and_remove_node():
    grag = make_grag(graphdb=None)
    state = {"nodes": [], "matrix": [], "edge_labels": {}}

    added = GraphRAG._matrix_apply_action(grag, state, {"action": "add_node", "label": "Study", "id": "S"}, silent=True)
    assert added["ok"] is True
    assert state["nodes"][0]["id"] == "S"
    assert state["nodes"][0]["props"]["ontology_class"] == "top:Node"

    duplicate = GraphRAG._matrix_apply_action(grag, state, {"action": "add_node", "label": "Study", "id": "S"}, silent=True)
    assert duplicate["ok"] is True
    assert state["nodes"][1]["id"] == "n1"

    removed = GraphRAG._matrix_apply_action(grag, state, {"action": "remove_node", "id": "S"}, silent=True)
    assert removed["ok"] is True
    assert [node["id"] for node in state["nodes"]] == ["n1"]


def test_materialise_graph_creates_tgraph_with_ontology_metadata(monkeypatch):
    TGraph = install_fake_tgraph(monkeypatch)
    grag = make_grag()
    state = {
        "nodes": [
            {"id": "A", "label": "Living", "x": 0, "y": 0, "z": 0, "props": {}},
            {"id": "B", "label": "Kitchen", "x": 1, "y": 0, "z": 0, "props": {}},
        ],
        "matrix": [[0, 1], [1, 0]],
        "edge_labels": {("A", "B"): "adjacent"},
    }

    graph = GraphRAG._matrix_materialise_graph(grag, state, silent=True)

    assert isinstance(graph, TGraph)
    assert graph.dictionary["ontology_class"] == "top:KnowledgeGraph"
    assert len(graph._vertices) == 2
    assert len(graph._edges) == 1
    assert graph._vertices[0]["dictionary"]["label"] == "Living"
    assert graph._edges[0]["dictionary"]["label"] == "adjacent"


def test_apply_action_tgraph_add_node_and_connect(monkeypatch):
    install_fake_tgraph(monkeypatch)
    graph = FakeTGraph(directed=False, allowParallelEdges=False)
    graph.AddVertex(dictionary={"id": "A", "label": "Living", "x": 0, "y": 0, "z": 0})
    grag = make_grag()

    result = GraphRAG.ApplyAction(
        grag,
        graph,
        {"action": "add_node", "label": "Kitchen", "id": "B", "attach_to_id": "A", "edge_label": "adjacent"},
        silent=True,
    )

    assert result["ok"] is True
    assert len(graph._vertices) == 2
    assert len(graph._edges) == 1
    assert graph._edges[0]["dictionary"]["label"] == "adjacent"

    connect_result = GraphRAG.ApplyAction(grag, graph, {"action": "connect", "a_id": "A", "b_id": "B"}, silent=True)
    assert connect_result["ok"] is False
    assert "already" in connect_result["message"].lower() or "edge" in connect_result["message"].lower()


def test_generate_add_node_then_stop_materialises_graph(monkeypatch):
    install_fake_tgraph(monkeypatch)
    install_fake_llm(monkeypatch)
    initial = FakeTGraph(directed=False, allowParallelEdges=False)
    initial.AddVertex(dictionary={"id": "A", "label": "Living", "x": 0, "y": 0, "z": 0})
    llm = FakeLLMHandle([
        '{"action": "add_node", "label": "Kitchen", "id": "B", "attach_to_id": "A", "edge_label": "adjacent", "reason": "needed"}',
        '{"action": "stop", "reason": "done"}',
    ])
    grag = make_grag(llm=llm)

    result = GraphRAG.Generate(grag, initial, description="Add kitchen", maxSteps=3, automatic=True, verbose=False, silent=True)

    assert result["ok"] is True
    assert result["status"] == "stopped"
    assert result["num_steps"] == 2
    assert len(result["graph"]._vertices) == 2
    assert len(result["graph"]._edges) == 1
    assert result["matrix_state"]["matrix"][0][1] == 1


def test_generate_ignores_bad_llm_until_patience_exhausted(monkeypatch):
    install_fake_tgraph(monkeypatch)
    install_fake_llm(monkeypatch)
    initial = FakeTGraph(directed=False)
    initial.AddVertex(dictionary={"id": "A", "label": "Living"})
    llm = FakeLLMHandle(["not json", "also not json"])
    grag = make_grag(llm=llm)

    result = GraphRAG.Generate(grag, initial, maxSteps=5, patience=2, automatic=True, verbose=False, silent=True)

    assert result["ok"] is True
    assert result["status"] == "patience_exhausted"
    assert [step["status"] for step in result["steps"]] == ["ignored_bad_llm_response", "ignored_bad_llm_response"]


def test_approve_action_and_small_helpers(monkeypatch):
    install_fake_tgraph(monkeypatch)
    graph = FakeTGraph()
    graph.AddVertex(dictionary={"id": "n1", "label": "Room"})
    grag = make_grag()

    assert GraphRAG.ApproveAction({"action": "stop"}, approvalFunction=lambda action, message: "accept", silent=True) == "accept"
    assert GraphRAG.ApproveAction({"action": "stop"}, approvalFunction=lambda action, message: "bad", silent=True) == "ignore"
    assert GraphRAG.ApproveAction({"action": "stop"}, silent=True) == "ignore"

    assert GraphRAG._find_vertex_by_id(grag, graph, "n1")["label"] == "Room"
    assert GraphRAG._find_vertex_by_label(grag, graph, "room")["id"] == "n1"
    assert GraphRAG._matrix_unique_id({"nodes": [{"id": "n1"}]}, suggested_id="n1") == "n2"
    assert GraphRAG._matrix_find_index_by_label({"nodes": [{"label": "Living Room"}]}, "living_room") == 0
    assert GraphRAG.list_working_nodes_edges(grag, graph, silent=True)["nodes"][0]["id"] == "n1"
