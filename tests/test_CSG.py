"""Unit tests for topologicpy.CSG."""

from __future__ import annotations

import math

import pytest

CSG = pytest.importorskip("topologicpy.CSG").CSG
Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Face = pytest.importorskip("topologicpy.Face").Face
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary
Graph = pytest.importorskip("topologicpy.Graph").Graph


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x=0, y=0, z=0):
    try:
        return Vertex.ByCoordinates(x, y, z, silent=True)
    except TypeError:
        try:
            return Vertex.ByCoordinates(x, y, z)
        except TypeError:
            return Vertex.ByCoordinates([x, y, z])


def _face(width=1, length=1):
    try:
        return Face.Rectangle(width=width, length=length, silent=True)
    except TypeError:
        return Face.Rectangle(width=width, length=length)


def _set_dict(topology, values):
    d = Dictionary.ByKeysValues(list(values.keys()), list(values.values()))
    try:
        return Topology.SetDictionary(topology, d, silent=True)
    except TypeError:
        return Topology.SetDictionary(topology, d)

def test_public_methods_reject_invalid_graph_inputs():
    face = _face()
    assert CSG.AddTopologyVertex(object(), face, silent=True) is None
    assert CSG.AddOperationVertex(object(), "union", {"id": "A"}, {"id": "B"}, silent=True) is None
    assert CSG.Connect(object(), _v(0, 0, 0), _v(1, 0, 0), silent=True) is None
    assert CSG.Invoke(object(), silent=True) is None
    assert CSG.Topologies(object(), silent=True) is None

def test_add_topology_vertex_requires_valid_topology():
    assert CSG.AddTopologyVertex(None, None, silent=True) is None
    assert CSG.AddTopologyVertex(None, "not a topology", silent=True) is None

def test_connect_rejects_invalid_vertices():
    assert CSG.Connect(None, None, _v(1, 0, 0), silent=True) is None
    assert CSG.Connect(None, _v(0, 0, 0), None, silent=True) is None

def test_topologies_returns_empty_list_for_graph_without_brep_nodes():
    a = _set_dict(_v(0, 0, 0), {"id": "A"})
    b = _set_dict(_v(1, 0, 0), {"id": "B"})
    graph = CSG.Connect(None, a, b, silent=True)

    if graph is None:
        pytest.skip("Graph backend did not create a graph from two vertices.")

    result = CSG.Topologies(graph, silent=True)
    assert result == []

def test_invoke_rejects_graphs_with_no_single_root_when_empty_graph_supported():
    from topologicpy.TGraph import TGraph

    graph = CSG.Init(asTGraph=True, silent=True)

    assert isinstance(graph, TGraph)
    assert TGraph.Vertices(graph) == []
    assert TGraph.Edges(graph) == []
    assert CSG.Invoke(graph, silent=True) is None

def test_invoke_simple_union_graph_smoke():
    from topologicpy.TGraph import TGraph

    face_a = _face(width=1, length=1)

    try:
        face_b = Topology.Translate(
            face_a,
            x=0.5,
            y=0.0,
            z=0.0,
            silent=True,
        )
    except TypeError:
        face_b = Topology.Translate(
            face_a,
            0.5,
            0.0,
            0.0,
        )

    graph = CSG.Init(asTGraph=True, silent=True)

    assert isinstance(graph, TGraph)

    a = CSG.AddTopologyVertex(
        graph,
        face_a,
        silent=True,
    )

    b = CSG.AddTopologyVertex(
        graph,
        face_b,
        silent=True,
    )

    op = CSG.AddOperationVertex(
        graph,
        "union",
        a,
        b,
        silent=True,
    )

    assert a is not None
    assert b is not None
    assert op is not None

    graph = CSG.Connect(
        graph,
        a,
        op,
        silent=True,
    )

    assert isinstance(graph, TGraph)

    graph = CSG.Connect(
        graph,
        b,
        op,
        silent=True,
    )

    assert isinstance(graph, TGraph)

    result = CSG.Invoke(
        graph,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology")

def _public_dictionary(topology):
    dictionary = Topology.Dictionary(topology, silent=True)
    return Dictionary.PythonDictionary(dictionary, silent=True) or {}


def test_add_topology_vertex_creates_metadata_vertex_without_graph():
    face = _face(width=1, length=1)
    vertex = CSG.AddTopologyVertex(None, face, silent=True)

    assert Topology.IsInstance(vertex, "Vertex")
    d = _public_dictionary(vertex)
    assert d["type"] == "topology"
    assert d["brepType"] == Topology.Type(face)
    assert d["brepTypeString"].lower() == Topology.TypeAsString(face).lower()
    assert isinstance(d["brep"], str) and len(d["brep"]) > 0
    assert isinstance(d["id"], str) and len(d["id"]) > 0


def test_add_operation_vertex_validates_operation_and_operand_ids():
    a = CSG.AddTopologyVertex(None, _face(width=1, length=1), silent=True)
    b = CSG.AddTopologyVertex(None, _face(width=1, length=1), silent=True)

    assert CSG.AddOperationVertex(None, "invalid", a, b, silent=True) is None

    vertex = CSG.AddOperationVertex(None, "intersect", a, b, silent=True)
    assert Topology.IsInstance(vertex, "Vertex")

    d = _public_dictionary(vertex)
    assert d["type"] == "operation"
    assert d["operation"] == "intersection"
    assert isinstance(d["a_id"], str) and d["a_id"]
    assert isinstance(d["b_id"], str) and d["b_id"]
    assert isinstance(d["id"], str) and d["id"]

def test_connect_creates_graph_from_two_vertices():
    a = _set_dict(_v(0, 0, 0), {"id": "A"})
    b = _set_dict(_v(1, 0, 0), {"id": "B"})

    graph = CSG.Connect(None, a, b, silent=True)

    assert graph is not None
    assert len(Graph.Vertices(graph) or []) >= 2
    assert len(Graph.Edges(graph) or []) >= 1

def test_init_returns_graph_when_backend_supports_empty_graphs():
    from topologicpy.TGraph import TGraph

    graph = CSG.Init(
        asTGraph=True,
        silent=True,
    )

    assert isinstance(graph, TGraph)
    assert TGraph.Vertices(graph) == []
    assert TGraph.Edges(graph) == []
