# Auto-generated PythonOCC backend parity test
# Copied from test_Context.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Context."""

import pytest

pytest.importorskip("topologicpy.Core")

from topologicpy.Context import Context
from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _vertex(x=0, y=0, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _assert_context(context):
    assert context is not None
    assert Topology.IsInstance(context, "Context")


def _assert_topology(topology):
    assert topology is not None
    assert Topology.IsInstance(topology, "Topology")


def test_by_topology_parameters_creates_context_for_vertex():
    vertex = _vertex(1, 2, 3)

    context = Context.ByTopologyParameters(vertex)

    _assert_context(context)


def test_by_topology_parameters_accepts_explicit_parameters_for_edge():
    start = _vertex(0, 0, 0)
    end = _vertex(10, 0, 0)
    edge = Edge.ByVertices(start, end, silent=True)

    context = Context.ByTopologyParameters(edge, u=0.25, v=0.5, w=0.75)

    _assert_context(context)


def test_topology_returns_topology_from_context():
    vertex = _vertex(4, 5, 6)
    context = Context.ByTopologyParameters(vertex)

    topology = Context.Topology(context)

    _assert_topology(topology)
    assert Topology.IsInstance(topology, "Vertex")


def test_topology_round_trip_preserves_edge_topology_type():
    edge = Edge.ByVertices(_vertex(0, 0, 0), _vertex(1, 0, 0), silent=True)
    context = Context.ByTopologyParameters(edge)

    topology = Context.Topology(context)

    _assert_topology(topology)
    assert Topology.IsInstance(topology, "Edge")


def test_by_topology_parameters_rejects_invalid_topology():
    assert Context.ByTopologyParameters(None) is None
    assert Context.ByTopologyParameters("not a topology") is None


def test_topology_rejects_invalid_context():
    assert Context.Topology(None) is None
    assert Context.Topology("not a context") is None


def test_invalid_context_does_not_break_subsequent_valid_context_creation():
    assert Context.Topology(None) is None

    vertex = _vertex(0, 0, 0)
    context = Context.ByTopologyParameters(vertex)

    _assert_context(context)
    _assert_topology(Context.Topology(context))
