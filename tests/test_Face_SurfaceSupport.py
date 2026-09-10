"""Tranche 07: Face surface evaluation and exact PythonOCC NURBS-surface support."""

import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _xyz(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _close(a, b, tol=2.0e-5):
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


def _dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _mag(a):
    return math.sqrt(_dot(a, a))


def _planar_rectangle():
    return Face.Rectangle(
        origin=Vertex.ByCoordinates(0.0, 0.0, 0.0),
        width=4.0,
        length=2.0,
        placement="lowerleft",
        direction=[0, 0, 1],
        silent=True,
    )


def _quarter_cylinder_nurbs():
    s2 = math.sqrt(2.0) / 2.0
    control_points = [
        [Vertex.ByCoordinates(1.0, 0.0, 0.0), Vertex.ByCoordinates(1.0, 0.0, 1.0)],
        [Vertex.ByCoordinates(1.0, 1.0, 0.0), Vertex.ByCoordinates(1.0, 1.0, 1.0)],
        [Vertex.ByCoordinates(0.0, 1.0, 0.0), Vertex.ByCoordinates(0.0, 1.0, 1.0)],
    ]
    weights = [
        [1.0, 1.0],
        [s2, s2],
        [1.0, 1.0],
    ]
    return Face.ByNurbsParameters(
        control_points,
        weights=weights,
        uKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 1.0, 1.0],
        isRational=True,
        uDegree=2,
        vDegree=1,
        silent=True,
    )


def test_planar_face_uv_roundtrip_and_orientation():
    face = _planar_rectangle()
    assert Topology.IsInstance(face, "Face")

    point = Face.VertexByParameters(face, u=0.25, v=0.75, silent=True)
    assert Topology.IsInstance(point, "Vertex")
    assert _close(_xyz(point), [1.0, 1.5, 0.0], tol=2.0e-5)

    uv = Face.VertexParameters(face, point, outputType="uv", mantissa=None, silent=True)
    assert isinstance(uv, list) and len(uv) == 2
    assert math.isclose(uv[0], 0.25, abs_tol=2.0e-5)
    assert math.isclose(uv[1], 0.75, abs_tol=2.0e-5)


def test_planar_face_normal_tangents_and_planarity():
    face = _planar_rectangle()
    normal = Face.NormalAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    tangents = Face.TangentsAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)

    assert isinstance(normal, list) and len(normal) == 3
    assert isinstance(tangents, dict)
    tu = tangents.get("u")
    tv = tangents.get("v")
    assert isinstance(tu, list) and isinstance(tv, list)
    assert math.isclose(_mag(normal), 1.0, abs_tol=2.0e-6)
    assert math.isclose(_mag(tu), 1.0, abs_tol=2.0e-6)
    assert math.isclose(_mag(tv), 1.0, abs_tol=2.0e-6)
    assert abs(_dot(normal, tu)) <= 2.0e-5
    assert abs(_dot(normal, tv)) <= 2.0e-5
    assert normal[2] > 0.999
    assert Face.IsPlanar(face, silent=True) is True

    one_tangent = Face.TangentAtParameters(face, 0.5, 0.5, axis="u", mantissa=None, silent=True)
    assert isinstance(one_tangent, list) and len(one_tangent) == 3
    assert abs(abs(_dot(one_tangent, tu)) - 1.0) <= 2.0e-5

    # Boundary UV values exercise the one-sided finite-difference fallback used
    # by TopologicCore when native surface derivatives are unavailable.
    corner_tangents = Face.TangentsAtParameters(face, 0.0, 0.0, mantissa=None, silent=True)
    assert isinstance(corner_tangents, dict)
    corner_tu = corner_tangents.get("u")
    corner_tv = corner_tangents.get("v")
    assert isinstance(corner_tu, list) and isinstance(corner_tv, list)
    assert math.isclose(_mag(corner_tu), 1.0, abs_tol=2.0e-6)
    assert math.isclose(_mag(corner_tv), 1.0, abs_tol=2.0e-6)
    assert abs(_dot(normal, corner_tu)) <= 2.0e-5
    assert abs(_dot(normal, corner_tv)) <= 2.0e-5


def test_planar_face_curvature_is_zero():
    face = _planar_rectangle()
    curvature = Face.CurvatureAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    assert isinstance(curvature, dict)
    for key in ("maximum", "minimum", "mean", "gaussian"):
        assert key in curvature
        assert abs(float(curvature[key])) <= 2.0e-4


def test_surface_query_validation_does_not_raise():
    face = _planar_rectangle()
    assert Face.NormalAtParameters(face, u=-0.1, v=0.5, silent=True) is None
    assert Face.TangentAtParameters(face, u=0.5, v=0.5, axis="bad", silent=True) is None
    assert Face.TangentsAtParameters(face, u=1.1, v=0.5, silent=True) is None
    assert Face.CurvatureAtParameters(None, silent=True) is None
    assert Face.IsPlanar(None, silent=True) is None


@pytest.mark.topologiccore_only
def test_topologiccore_nurbs_surface_construction_is_explicitly_unsupported():
    cps = [
        [Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(0, 1, 0)],
        [Vertex.ByCoordinates(1, 0, 0), Vertex.ByCoordinates(1, 1, 0)],
    ]
    assert Face.ByNurbsParameters(cps, uDegree=1, vDegree=1, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_planar_bspline_surface_is_exact_and_planar():
    cps = [
        [Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(0, 2, 0)],
        [Vertex.ByCoordinates(4, 0, 0), Vertex.ByCoordinates(4, 2, 0)],
    ]
    face = Face.ByNurbsParameters(
        cps,
        uDegree=1,
        vDegree=1,
        uKnots=[0, 0, 1, 1],
        vKnots=[0, 0, 1, 1],
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is True
    center = Face.VertexByParameters(face, 0.5, 0.5, silent=True)
    assert _close(_xyz(center), [2.0, 1.0, 0.0], tol=2.0e-6)

    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_BSplineSurface

    assert BRepAdaptor_Surface(face.shape, True).GetType() == GeomAbs_BSplineSurface


@pytest.mark.pythonocc_only
def test_pythonocc_rational_quarter_cylinder_geometry_and_planarity():
    face = _quarter_cylinder_nurbs()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False

    s2 = math.sqrt(2.0) / 2.0
    center = Face.VertexByParameters(face, 0.5, 0.5, silent=True)
    assert Topology.IsInstance(center, "Vertex")
    assert _close(_xyz(center), [s2, s2, 0.5], tol=3.0e-6)

    uv = Face.VertexParameters(face, center, mantissa=None, silent=True)
    assert isinstance(uv, list) and len(uv) == 2
    assert math.isclose(uv[0], 0.5, abs_tol=3.0e-6)
    assert math.isclose(uv[1], 0.5, abs_tol=3.0e-6)


@pytest.mark.pythonocc_only
def test_pythonocc_quarter_cylinder_normal_tangents_and_curvature():
    face = _quarter_cylinder_nurbs()
    assert Topology.IsInstance(face, "Face")

    normal = Face.NormalAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    tangents = Face.TangentsAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    curvature = Face.CurvatureAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)

    assert isinstance(normal, list) and len(normal) == 3
    assert isinstance(tangents, dict)
    assert isinstance(curvature, dict)

    s2 = math.sqrt(2.0) / 2.0
    radial = [s2, s2, 0.0]
    assert abs(abs(_dot(normal, radial)) - 1.0) <= 3.0e-6

    tu = tangents["u"]
    tv = tangents["v"]
    assert abs(_dot(tu, radial)) <= 3.0e-6
    assert abs(abs(tv[2]) - 1.0) <= 3.0e-6
    assert abs(_dot(normal, tu)) <= 3.0e-6
    assert abs(_dot(normal, tv)) <= 3.0e-6

    principal = sorted([abs(float(curvature["maximum"])), abs(float(curvature["minimum"]))])
    assert math.isclose(principal[0], 0.0, abs_tol=2.0e-6)
    assert math.isclose(principal[1], 1.0, rel_tol=2.0e-6, abs_tol=2.0e-6)
    assert math.isclose(float(curvature["gaussian"]), 0.0, abs_tol=2.0e-6)
    assert curvature["isUmbilic"] is False


@pytest.mark.pythonocc_only
def test_pythonocc_normaledge_uses_local_surface_normal():
    face = _quarter_cylinder_nurbs()
    normal_edge = Face.NormalEdge(face, length=0.5, silent=True)
    assert Topology.IsInstance(normal_edge, "Edge")
    assert math.isclose(Edge.Length(normal_edge, mantissa=6, silent=True), 0.5, abs_tol=2.0e-6)
