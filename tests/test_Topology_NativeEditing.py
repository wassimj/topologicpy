"""Tests for Tranche 16 native topology editing and exact curve/surface survival."""

import math
import os

import pytest

from topologicpy.Cell import Cell
from topologicpy.Cluster import Cluster
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _two_edge_wire():
    v0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    v1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)
    v2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)
    e0 = Edge.ByVertices(v0, v1, silent=True)
    e1 = Edge.ByVertices(v1, v2, silent=True)
    wire = Wire.ByEdges([e0, e1], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire, e0, e1, v0, v1, v2


def _arc_tail_wire():
    arc = Edge.Arc(
        radius=2.0,
        fromAngle=0.0,
        toAngle=90.0,
        silent=True,
    )
    assert Topology.IsInstance(arc, "Edge")
    end = Edge.EndVertex(arc)
    tail_end = Vertex.ByCoordinates(0.0, 3.0, 0.0)
    tail = Edge.ByVertices(end, tail_end, silent=True)
    wire = Wire.ByEdges([arc, tail], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire, arc, tail, tail_end


def test_remove_edges_accepts_single_edge_argument():
    wire, e0, e1, *_ = _two_edge_wire()
    result = Topology.RemoveEdges(wire, e1, silent=True)
    assert result is not None
    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1


def test_remove_vertices_accepts_single_vertex_and_cascades_incident_edge():
    wire, e0, e1, v0, v1, v2 = _two_edge_wire()
    result = Topology.RemoveVertices(wire, v2, silent=True)
    assert result is not None
    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1


def test_remove_faces_accepts_single_face_argument():
    cell = Cell.Box(width=2.0, length=2.0, height=2.0, silent=True)
    faces = Topology.Faces(cell, silent=True) or []
    assert len(faces) == 6

    result = Topology.RemoveFaces(cell, faces[0], silent=True)
    assert result is not None

    remaining = Topology.Faces(result, silent=True) or []
    assert len(remaining) == 5


def test_remove_edit_validation_and_noop_are_non_throwing():
    wire, e0, e1, *_ = _two_edge_wire()

    assert Topology.RemoveEdges(None, e0, silent=True) is None
    assert Topology.RemoveFaces(None, None, silent=True) is None
    assert Topology.RemoveVertices(None, None, silent=True) is None

    assert Topology.RemoveEdges(wire, None, silent=True) is wire
    assert Topology.RemoveVertices(wire, [], silent=True) is wire
    assert Topology.RemoveEdges(wire, e0, tolerance="bad", silent=True) is None


@pytest.mark.pythonocc_only
def test_native_remove_edge_preserves_surviving_arc_exactly():
    wire, arc, tail, _ = _arc_tail_wire()
    expected_length = Edge.Length(arc, mantissa=None, silent=True)

    result = Topology.RemoveEdges(wire, tail, silent=True)
    assert result is not None

    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1
    survivor = edges[0]

    assert Edge.IsLinear(survivor, silent=True) is False
    assert math.isclose(
        Edge.Length(survivor, mantissa=None, silent=True),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )


@pytest.mark.pythonocc_only
def test_native_remove_vertex_preserves_surviving_arc_exactly():
    wire, arc, tail, tail_end = _arc_tail_wire()
    expected_length = Edge.Length(arc, mantissa=None, silent=True)

    result = Topology.RemoveVertices(wire, tail_end, silent=True)
    assert result is not None

    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1
    survivor = edges[0]

    assert Edge.IsLinear(survivor, silent=True) is False
    assert math.isclose(
        Edge.Length(survivor, mantissa=None, silent=True),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )


@pytest.mark.pythonocc_only
def test_native_remove_face_preserves_cylindrical_surface_exactly():
    cell = Cell.Cylinder(
        radius=1.5,
        height=2.5,
        uSides=32,
        vSides=1,
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")

    faces = Topology.Faces(cell, silent=True) or []
    planar = [face for face in faces if Face.IsPlanar(face, silent=True)]
    curved = [face for face in faces if not Face.IsPlanar(face, silent=True)]

    assert len(planar) >= 2
    assert len(curved) >= 1

    expected_curved_area = sum(
        Face.Area(face, mantissa=None, silent=True)
        for face in curved
    )

    result = Topology.RemoveFaces(cell, planar[0], silent=True)
    assert result is not None

    remaining = Topology.Faces(result, silent=True) or []
    remaining_curved = [
        face for face in remaining
        if not Face.IsPlanar(face, silent=True)
    ]

    assert len(remaining_curved) >= 1
    assert math.isclose(
        sum(Face.Area(face, mantissa=None, silent=True) for face in remaining_curved),
        expected_curved_area,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )


@pytest.mark.pythonocc_only
def test_shapeless_cluster_falls_back_without_losing_surviving_curve():
    arc_a = Edge.Arc(
        radius=1.0,
        fromAngle=0.0,
        toAngle=90.0,
        silent=True,
    )
    arc_b = Topology.Translate(
        Edge.Arc(
            radius=2.0,
            fromAngle=0.0,
            toAngle=120.0,
            silent=True,
        ),
        x=10.0,
        silent=True,
    )

    cluster = Cluster.ByTopologies([arc_a, arc_b], silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    assert Topology.OCCTShape(cluster, silent=True) is None

    expected_length = Edge.Length(arc_b, mantissa=None, silent=True)

    result = Topology.RemoveEdges(cluster, arc_a, silent=True)
    assert result is not None

    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1
    survivor = edges[0]

    assert Edge.IsLinear(survivor, silent=True) is False
    assert math.isclose(
        Edge.Length(survivor, mantissa=None, silent=True),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )
