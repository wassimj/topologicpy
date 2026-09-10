"""Focused tests for low-risk generic Topology query robustness."""

import math
import os

import pytest

from topologicpy.Cell import Cell
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _circle_wire(z=0.0, radius=1.0):
    edge = Edge.Circle(
        origin=Vertex.ByCoordinates(0.0, 0.0, z),
        radius=radius,
        silent=True,
    )
    assert Topology.IsInstance(edge, "Edge")
    wire = Wire.ByEdges([edge], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


def _quarter_cylinder_face(radius=1.0, height=2.0):
    s2 = math.sqrt(2.0) / 2.0
    cps = [
        [Vertex.ByCoordinates(radius, 0.0, 0.0), Vertex.ByCoordinates(radius, 0.0, height)],
        [Vertex.ByCoordinates(radius, radius, 0.0), Vertex.ByCoordinates(radius, radius, height)],
        [Vertex.ByCoordinates(0.0, radius, 0.0), Vertex.ByCoordinates(0.0, radius, height)],
    ]
    weights = [[1.0, 1.0], [s2, s2], [1.0, 1.0]]
    return Face.ByNurbsParameters(
        cps,
        weights=weights,
        uKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 1.0, 1.0],
        isRational=True,
        uDegree=2,
        vDegree=1,
        silent=True,
    )


def test_internal_vertex_planar_face_and_cell_are_strictly_internal():
    face = Face.Rectangle(width=4.0, length=3.0, silent=True)
    cell = Cell.Box(width=4.0, length=3.0, height=2.0, silent=True)

    iv_face = Topology.InternalVertex(face, silent=True)
    iv_cell = Topology.InternalVertex(cell, silent=True)

    assert Topology.IsInstance(iv_face, "Vertex")
    assert Topology.IsInstance(iv_cell, "Vertex")
    assert Vertex.IsInternal(iv_face, face, tolerance=1.0e-4, silent=True)
    assert Vertex.IsInternal(iv_cell, cell, tolerance=1.0e-4, silent=True)


def test_internal_vertex_timeout_is_retained_for_compatibility_but_not_used():
    face = Face.Rectangle(width=2.0, length=2.0, silent=True)
    result = Topology.InternalVertex(face, timeout=0, silent=True)
    assert Topology.IsInstance(result, "Vertex")
    assert Vertex.IsInternal(result, face, tolerance=1.0e-4, silent=True)


def test_internal_vertex_validation_is_non_throwing():
    assert Topology.InternalVertex(None, silent=True) is None
    assert Topology.InternalVertex(Face.Rectangle(silent=True), tolerance="bad", silent=True) is None
    assert Topology.InternalVertex(Face.Rectangle(silent=True), tolerance=0.0, silent=True) is None


def test_open_vertices_of_open_linear_wire_are_endpoints():
    v0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    v1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)
    v2 = Vertex.ByCoordinates(2.0, 1.0, 0.0)
    wire = Wire.ByVertices([v0, v1, v2], close=False, silent=True)
    assert Topology.IsInstance(wire, "Wire")

    open_vertices = Topology.OpenVertices(wire, silent=True)
    assert isinstance(open_vertices, list)
    assert len(open_vertices) == 2


@pytest.mark.pythonocc_only
def test_internal_vertex_on_curved_nurbs_face_is_internal():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    result = Topology.InternalVertex(face, silent=True)
    assert Topology.IsInstance(result, "Vertex")
    assert Vertex.IsInternal(result, face, tolerance=1.0e-4, silent=True)


@pytest.mark.pythonocc_only
def test_open_edges_of_exact_cylindrical_shell_remain_curved():
    shell = Shell.ByWires(
        [_circle_wire(0.0, 1.0), _circle_wire(2.0, 1.0)],
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(shell, "Shell")

    open_edges = Topology.OpenEdges(shell, silent=True)
    assert isinstance(open_edges, list)
    assert len(open_edges) == 2
    assert all(Edge.IsLinear(edge, silent=True) is False for edge in open_edges)


@pytest.mark.pythonocc_only
def test_external_boundary_of_circular_face_preserves_circle():
    face = Face.ByWire(_circle_wire(0.0, 2.0), silent=True)
    assert Topology.IsInstance(face, "Face")

    boundary = Topology.ExternalBoundary(face, silent=True)
    assert Topology.IsInstance(boundary, "Wire")
    edges = Wire.Edges(boundary, silent=True)
    assert isinstance(edges, list) and len(edges) == 1
    assert Edge.IsClosed(edges[0], silent=True)
    assert Edge.IsLinear(edges[0], silent=True) is False
    assert math.isclose(
        Edge.Length(edges[0], mantissa=None, silent=True),
        4.0 * math.pi,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )


@pytest.mark.pythonocc_only
def test_shortest_edge_to_circle_uses_actual_curve_not_topological_vertices():
    center = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    circle = Edge.Circle(radius=2.5, silent=True)
    assert Topology.IsInstance(circle, "Edge")

    shortest = Topology.ShortestEdge(center, circle, silent=True)
    assert Topology.IsInstance(shortest, "Edge")
    assert math.isclose(
        Edge.Length(shortest, mantissa=None, silent=True),
        2.5,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )
