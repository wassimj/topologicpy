import json
import sys
import types

import pytest

from topologicpy.GraphRAG import GraphRAG


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

    assert grag.graphdb == "db"
    assert grag.llm == "llm"
    assert grag.promptContext == "Context"
    assert grag.tolerance == pytest.approx(0.0001)
    assert grag.maxCandidates == 1
    assert grag.maxPairs == 40
    assert grag.ontology is False






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


