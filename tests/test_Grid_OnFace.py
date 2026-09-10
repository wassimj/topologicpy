import math
import os

import pytest

from topologicpy.Cell import Cell
from topologicpy.Dictionary import Dictionary
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Grid import Grid
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


IS_PYTHONOCC = "pythonocc" in os.environ.get(
    "TOPOLOGICPY_CORE_BACKEND", ""
).lower()


def _value(topology, key):
    d = Topology.Dictionary(topology, silent=True)
    return Dictionary.ValueAtKey(d, key, None)


def _cylinder_face():
    cell = Cell.Cylinder(
        radius=2.0,
        height=5.0,
        uSides=32,
        vSides=1,
        polyhedron=False,
        silent=True,
    )
    for face in Topology.Faces(cell, silent=True) or []:
        if Face.IsPlanar(face, silent=True) is False:
            return face
    return None


def _nurbs_face():
    w = 1.0 / math.sqrt(2.0)
    points = [
        [
            Vertex.ByCoordinates(1, 0, 0),
            Vertex.ByCoordinates(1, 1, 0),
            Vertex.ByCoordinates(0, 1, 0),
        ],
        [
            Vertex.ByCoordinates(1, 0, 3),
            Vertex.ByCoordinates(1, 1, 3),
            Vertex.ByCoordinates(0, 1, 3),
        ],
    ]
    return Face.ByNurbsParameters(
        controlPoints=points,
        weights=[[1, w, 1], [1, w, 1]],
        uKnots=[0, 0, 1, 1],
        vKnots=[0, 0, 0, 1, 1, 1],
        isRational=True,
        uDegree=1,
        vDegree=2,
        silent=True,
    )


def test_onface_planar_divisions_are_straight_and_clipped():
    face = Face.Rectangle(width=8, length=6, silent=True)
    grid = Grid.OnFace(
        face,
        uDivisions=4,
        vDivisions=3,
        includeBoundary=True,
        silent=True,
    )
    assert Topology.IsInstance(grid, "Cluster")
    edges = Topology.Edges(grid, silent=True) or []
    assert len(edges) == 9
    assert all(Edge.IsLinear(e, silent=True) for e in edges)
    assert all(_value(e, "grid_mode") == "planar" for e in edges)


def test_onface_planar_explicit_values():
    face = Face.Rectangle(width=10, length=4, silent=True)
    grid = Grid.OnFace(
        face,
        uValues=[0.25, 0.5, 0.75],
        vValues=[0.5],
        silent=True,
    )
    edges = Topology.Edges(grid, silent=True) or []
    assert len(edges) == 4
    u = sorted(
        float(_value(e, "grid_parameter"))
        for e in edges
        if _value(e, "grid_axis") == "u"
    )
    assert u == [0.25, 0.5, 0.75]


@pytest.mark.pythonocc_only
def test_onface_auto_switches_to_surface_grid_on_cylinder():
    face = _cylinder_face()
    assert Topology.IsInstance(face, "Face")

    grid = Grid.OnFace(
        face,
        uDivisions=4,
        vDivisions=3,
        includeBoundary=False,
        silent=True,
    )
    assert Topology.IsInstance(grid, "Cluster")

    edges = Topology.Edges(grid, silent=True) or []
    assert edges
    assert any(Edge.IsLinear(e, silent=True) is False for e in edges)
    assert all(_value(e, "grid_mode") == "surface" for e in edges)
    assert all(_value(e, "grid_geometry") == "isocurve" for e in edges)


@pytest.mark.pythonocc_only
def test_onface_nurbs_grid_contains_exact_curved_isocurves():
    face = _nurbs_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False

    grid = Grid.OnFace(
        face,
        uValues=[0.25, 0.5, 0.75],
        vValues=[0.25, 0.5, 0.75],
        mode="surface",
        silent=True,
    )
    assert Topology.IsInstance(grid, "Cluster")

    edges = Topology.Edges(grid, silent=True) or []
    assert edges
    assert any(Edge.IsLinear(e, silent=True) is False for e in edges)
    assert all(_value(e, "grid_geometry") == "isocurve" for e in edges)


@pytest.mark.pythonocc_only
def test_onface_forced_planar_rejects_curved_face():
    assert Grid.OnFace(
        _cylinder_face(),
        mode="planar",
        uDivisions=3,
        vDivisions=3,
        silent=True,
    ) is None


def test_onface_validation_is_non_throwing():
    assert Grid.OnFace(None, silent=True) is None

    face = Face.Rectangle(width=4, length=3, silent=True)
    assert Grid.OnFace(face, mode="bad", silent=True) is None
    assert Grid.OnFace(face, spacing=0, silent=True) is None
