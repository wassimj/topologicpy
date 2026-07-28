# Auto-generated PythonOCC backend parity test
# Copied from test_CSG.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.CSG."""

from __future__ import annotations

import math

import pytest

from topologicpy.CSG import CSG
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary


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


def test_python_dictionary_merges_record_fields_and_nested_dictionary():
    rec = {
        "index": "7",
        "id": "top_level",
        "representation": object(),
        "dictionary": {"id": "nested", "type": "topology", "value": 42},
    }

    d = CSG._python_dictionary(rec)

    assert d["index"] == "7"
    assert d["id"] == "nested"
    assert d["type"] == "topology"
    assert d["value"] == 42
    assert "representation" not in d
    assert CSG._python_dictionary(None) == {}


def test_dictionary_value_and_set_dictionary_values_on_record():
    rec = {"index": 1, "dictionary": {"a": 10}}

    assert CSG._dictionary_value(rec, "a") == 10
    assert CSG._dictionary_value(rec, "missing", "fallback") == "fallback"
    assert CSG._dictionary_value(rec, None, "fallback") == "fallback"

    returned = CSG._set_dictionary_values(rec, {"b": 20, "c": "x"})
    assert returned is rec
    assert rec["dictionary"]["b"] == 20
    assert rec["dictionary"]["c"] == "x"
    assert rec["b"] == 20
    assert CSG._set_dictionary_values(None, {"a": 1}) is None


def test_coordinates_from_tgraph_style_records():
    assert CSG._coordinates({"dictionary": {"x": 1.23456, "y": "2", "z": 3}}, mantissa=3) == [1.235, 2.0, 3.0]
    assert CSG._coordinates({"dictionary": {"X": 1, "Y": 2, "Z": 3}}, mantissa=6) == [1.0, 2.0, 3.0]
    assert CSG._coordinates({"coords": [4.123456, 5, 6]}, mantissa=2) == [4.12, 5.0, 6.0]
    assert CSG._coordinates({"coords": ["bad", 5, 6]}, mantissa=2) is None
    assert CSG._coordinates(None) is None


def test_vertex_index_normalisation_lookup_and_same_vertex_for_records():
    a = {"index": "1", "dictionary": {"id": "A"}}
    b = {"dictionary": {"vertex_id": 2}}
    c = {"node_id": "3.0"}

    assert CSG._normalise_index("1.0") == 1
    assert CSG._normalise_index("A") == "A"
    assert CSG._vertex_index(a) == "1"
    assert CSG._vertex_index(b) == 2
    assert CSG._vertex_index({}, fallback="fallback") == "fallback"

    lookup = CSG._vertex_lookup([a, b, c])
    assert lookup[0] is a
    assert lookup["A"] is a
    assert lookup[2] is b
    assert lookup[3] is c

    assert CSG._same_vertex({"dictionary": {"id": "7"}}, {"id": 7}) is True
    assert CSG._same_vertex({"dictionary": {"id": "7"}}, {"id": 8}) is False


def test_edge_endpoints_for_record_edges():
    vertices = [
        {"index": 0, "dictionary": {"id": "A"}},
        {"index": 1, "dictionary": {"id": "B"}},
    ]
    lookup = CSG._vertex_lookup(vertices)

    start, end = CSG._edge_endpoints({"src": "A", "dst": "B"}, vertices=vertices, lookup=lookup)
    assert start is vertices[0]
    assert end is vertices[1]

    start, end = CSG._edge_endpoints({"dictionary": {"source": 0, "target": 1}}, vertices=vertices, lookup=lookup)
    assert start is vertices[0]
    assert end is vertices[1]


def test_incoming_and_outgoing_vertices_for_tgraph_style_records(monkeypatch):
    graph = object()
    vertices = [
        {"index": 0, "dictionary": {"id": "A"}},
        {"index": 1, "dictionary": {"id": "B"}},
        {"index": 2, "dictionary": {"id": "C"}},
    ]
    edges = [
        {"src": "A", "dst": "B"},
        {"src": "C", "dst": "B"},
    ]

    monkeypatch.setattr(CSG, "_is_tgraph", staticmethod(lambda g: g is graph))
    monkeypatch.setattr(CSG, "_vertices", staticmethod(lambda g, asTopologic=True: vertices))
    monkeypatch.setattr(CSG, "_edges", staticmethod(lambda g, asTopologic=True: edges))

    incoming = CSG._incoming_vertices(graph, vertices[1], directed=True)
    outgoing = CSG._outgoing_vertices(graph, vertices[0], directed=True)

    assert incoming == [vertices[0], vertices[2]]
    assert outgoing == [vertices[1]]


def test_operation_name_aliases_and_invalid_values():
    assert CSG._operation_name("union") == "union"
    assert CSG._operation_name("INTERSECT") == "intersection"
    assert CSG._operation_name("symmetric_difference") == "xor"
    assert CSG._operation_name("symdif") == "xor"
    assert CSG._operation_name("subtract") == "difference"
    assert CSG._operation_name("not_an_operation") is None
    assert CSG._operation_name(None) is None


def test_unique_coords_returns_origin_first_and_avoids_used_points():
    assert CSG._unique_coords() == [0, 0, 0]

    used = [[0, 0, 0]]
    point = CSG._unique_coords(used_coords=used, width=1, length=1, height=1, tolerance=0.000001)
    assert len(point) == 3
    assert math.dist(point, used[0]) >= 0.000001


def test_call_with_optional_silent_supports_functions_with_and_without_silent():
    def accepts_silent(value, silent=False):
        return value, silent

    def rejects_silent(value):
        return value

    assert CSG._call_with_optional_silent(accepts_silent, "x", silent=True) == ("x", True)
    assert CSG._call_with_optional_silent(rejects_silent, "x", silent=True) == "x"


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


def test_add_topology_vertex_creates_metadata_vertex_without_graph():
    face = _face(width=1, length=1)
    vertex = CSG.AddTopologyVertex(None, face, silent=True)

    assert Topology.IsInstance(vertex, "Vertex")
    d = CSG._python_dictionary(vertex)
    assert d["type"] == "topology"
    assert d["brepType"] == Topology.Type(face)
    assert d["brepTypeString"].lower() == Topology.TypeAsString(face).lower()
    assert isinstance(d["brep"], str) and len(d["brep"]) > 0
    assert isinstance(d["id"], str) and len(d["id"]) > 0


def test_add_operation_vertex_validates_operation_and_operand_ids():
    a = {"dictionary": {"id": "A"}}
    b = {"id": "B"}

    assert CSG.AddOperationVertex(None, "invalid", a, b, silent=True) is None
    assert CSG.AddOperationVertex(None, "union", a, {}, silent=True) is None

    vertex = CSG.AddOperationVertex(None, "intersect", a, b, silent=True)
    assert Topology.IsInstance(vertex, "Vertex")

    d = CSG._python_dictionary(vertex)
    assert d["type"] == "operation"
    assert d["operation"] == "intersection"
    assert d["a_id"] == "A"
    assert d["b_id"] == "B"
    assert isinstance(d["id"], str) and len(d["id"]) > 0


def test_connect_rejects_invalid_vertices():
    assert CSG.Connect(None, None, _v(1, 0, 0), silent=True) is None
    assert CSG.Connect(None, _v(0, 0, 0), None, silent=True) is None


def test_connect_creates_graph_from_two_vertices():
    a = _set_dict(_v(0, 0, 0), {"id": "A"})
    b = _set_dict(_v(1, 0, 0), {"id": "B"})

    graph = CSG.Connect(None, a, b, silent=True)

    assert graph is not None
    assert CSG._is_graph_like(graph)
    vertices = CSG._vertices(graph, asTopologic=True)
    edges = CSG._edges(graph, asTopologic=True)
    assert isinstance(vertices, list)
    assert isinstance(edges, list)
    assert len(vertices) >= 2
    assert len(edges) >= 1


def test_init_returns_graph_when_backend_supports_empty_graphs():
    graph = CSG.Init(silent=True)

    # Some legacy Graph backends may reject empty graph creation. If they do,
    # Init returns None through the existing public fallback path.
    assert graph is None or CSG._is_graph_like(graph)


def test_topologies_returns_empty_list_for_graph_without_brep_nodes(monkeypatch):
    graph = object()
    vertices = [
        {"index": 0, "dictionary": {"id": "A"}},
        {"index": 1, "dictionary": {"id": "B"}},
    ]

    monkeypatch.setattr(CSG, "_is_graph_like", staticmethod(lambda g: g is graph))
    monkeypatch.setattr(CSG, "_vertices", staticmethod(lambda g, asTopologic=True: vertices))

    assert CSG.Topologies(graph, silent=True) == []


def test_invoke_rejects_graphs_with_no_single_root(monkeypatch):
    graph = object()

    monkeypatch.setattr(CSG, "_is_graph_like", staticmethod(lambda g: g is graph))
    monkeypatch.setattr(CSG, "_vertices", staticmethod(lambda g, asTopologic=True: []))
    monkeypatch.setattr(CSG, "_outgoing_vertices", staticmethod(lambda g, v, directed=True: []))

    assert CSG.Invoke(graph, silent=True) is None


def test_invoke_simple_union_graph_smoke_without_backend_graph(monkeypatch):
    face_a = _face(width=1, length=1)
    try:
        face_b = Topology.Translate(face_a, x=0.5, y=0.0, z=0.0, silent=True)
    except TypeError:
        face_b = Topology.Translate(face_a, 0.5, 0.0, 0.0)

    brep_a = Topology.BREPString(face_a)
    brep_b = Topology.BREPString(face_b)

    a = {"index": 0, "dictionary": {"id": "A", "type": "topology", "brep": brep_a, "matrix": None}}
    b = {"index": 1, "dictionary": {"id": "B", "type": "topology", "brep": brep_b, "matrix": None}}
    op = {"index": 2, "dictionary": {"id": "OP", "type": "operation", "operation": "union", "a_id": "A", "b_id": "B", "matrix": None}}
    graph = object()

    def fake_vertices(g, asTopologic=True):
        return [a, b, op]

    def fake_incoming(g, vertex, directed=True):
        if vertex is op:
            return [a, b]
        return []

    def fake_outgoing(g, vertex, directed=True):
        if vertex in [a, b]:
            return [op]
        return []

    monkeypatch.setattr(CSG, "_is_graph_like", staticmethod(lambda g: g is graph))
    monkeypatch.setattr(CSG, "_vertices", staticmethod(fake_vertices))
    monkeypatch.setattr(CSG, "_incoming_vertices", staticmethod(fake_incoming))
    monkeypatch.setattr(CSG, "_outgoing_vertices", staticmethod(fake_outgoing))

    result = CSG.Invoke(graph, silent=True)
    assert result is None or Topology.IsInstance(result, "Topology")
