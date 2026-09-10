"""Focused tests for generic Topology native wrappers and curve-aware planarity."""

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


def _xyz(vertex):
    return [float(v) for v in Vertex.Coordinates(vertex, mantissa=9)]


def _close(a, b, tol=1.0e-6):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


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


def test_isplanar_uses_curve_geometry_not_only_topological_vertices():
    arc = Edge.Arc(radius=2.0, fromAngle=15.0, toAngle=165.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    assert Topology.IsPlanar(arc, silent=True) is True

    bezier = Edge.Bezier(
        [
            Vertex.ByCoordinates(0.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 2.0, 0.0),
            Vertex.ByCoordinates(2.0, 0.0, 0.0),
        ],
        silent=True,
    )
    assert Topology.IsInstance(bezier, "Edge")
    assert Topology.IsPlanar(bezier, silent=True) is True

    helix = Edge.Helix(radius=1.0, height=2.0, turns=1.25, sides=24, silent=True)
    assert Topology.IsInstance(helix, "Edge")
    assert Topology.IsPlanar(helix, tolerance=1.0e-4, silent=True) is False


def test_isplanar_handles_faces_and_higher_dimensional_topologies():
    face = Face.Rectangle(width=4.0, length=3.0, silent=True)
    cell = Cell.Box(width=4.0, length=3.0, height=2.0, silent=True)

    assert Topology.IsPlanar(face, silent=True) is True
    assert Topology.IsPlanar(cell, silent=True) is False


@pytest.mark.pythonocc_only
def test_isplanar_detects_actual_curved_surface():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False
    assert Topology.IsPlanar(face, silent=True) is False


@pytest.mark.pythonocc_only
def test_isplanar_detects_curved_shell_even_when_vertices_are_insufficient():
    e0 = Edge.Circle(radius=1.0, silent=True)
    e1 = Topology.Translate(e0, z=2.0, silent=True)
    w0 = Wire.ByEdges([e0], silent=True)
    w1 = Wire.ByEdges([e1], silent=True)

    from topologicpy.Shell import Shell
    shell = Shell.ByWires([w0, w1], polyhedron=False, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert Topology.IsPlanar(shell, silent=True) is False


@pytest.mark.pythonocc_only
def test_copy_and_deepcopy_preserve_curved_edge_geometry_and_direction():
    arc = Edge.Arc(
        radius=3.0,
        fromAngle=20.0,
        toAngle=140.0,
        silent=True
    )
    assert Topology.IsInstance(arc, "Edge")

    shallow = Topology.Copy(
        arc,
        deep=False,
        silent=True
    )
    deep = Topology.DeepCopy(
        arc,
        silent=True
    )

    for result in (shallow, deep):
        assert Topology.IsInstance(result, "Edge")
        assert Edge.IsLinear(result, silent=True) is False

        assert math.isclose(
            Edge.Length(
                result,
                mantissa=None,
                silent=True
            ),
            Edge.Length(
                arc,
                mantissa=None,
                silent=True
            ),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )

        assert _close(
            _xyz(Edge.StartVertex(result)),
            _xyz(Edge.StartVertex(arc))
        )

        assert _close(
            _xyz(Edge.EndVertex(result)),
            _xyz(Edge.EndVertex(arc))
        )


@pytest.mark.pythonocc_only
def test_occt_shape_roundtrip_preserves_nurbs_surface():
    face = _quarter_cylinder_face(radius=1.5, height=2.25)
    assert Topology.IsInstance(face, "Face")

    shape = Topology.OCCTShape(face, silent=True)
    assert shape is not None

    result = Topology.ByOCCTShape(shape, silent=True)
    assert Topology.IsInstance(result, "Face")
    assert Face.IsPlanar(result, silent=True) is False
    assert math.isclose(
        Face.Area(result, mantissa=None, silent=True),
        Face.Area(face, mantissa=None, silent=True),
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )


@pytest.mark.pythonocc_only
def test_occtshape_of_lightweight_cluster_returns_none_without_restructuring_cluster():
    face = Face.Rectangle(silent=True)
    cluster = Cluster.ByTopologies([face], silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    assert Topology.OCCTShape(cluster, silent=True) is None
    assert len(Topology.Faces(cluster) or []) == 1


def test_center_of_mass_and_centroid_alias_are_consistent():
    origin = Vertex.ByCoordinates(3.0, -4.0, 5.0)
    cell = Cell.Box(
        origin=origin,
        width=2.0,
        length=4.0,
        height=6.0,
        placement="center",
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")

    center = Topology.CenterOfMass(cell, silent=True)
    centroid = Topology.Centroid(cell, silent=True)

    assert Topology.IsInstance(center, "Vertex")
    assert Topology.IsInstance(centroid, "Vertex")
    assert _close(_xyz(center), [3.0, -4.0, 5.0])
    assert _close(_xyz(centroid), [3.0, -4.0, 5.0])


def test_native_query_validation_is_non_throwing():
    assert Topology.OCCTShape(None, silent=True) is None
    assert Topology.ByOCCTShape(None, silent=True) is None
    assert Topology.CenterOfMass(None, silent=True) is None
    assert Topology.Centroid(None, silent=True) is None
    assert Topology.DeepCopy(None, silent=True) is None
    assert Topology.IsPlanar(None, silent=True) is None
    assert Topology.IsPlanar(Face.Rectangle(silent=True), tolerance=0.0, silent=True) is None
