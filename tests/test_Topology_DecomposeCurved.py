"""Regression tests for curvature-aware Topology.Decompose classification."""

import math
import os

import pytest

from topologicpy.Cell import Cell
from topologicpy.Cluster import Cluster
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _gentle_vertical_nurbs_face():
    # A vertical quadratic NURBS patch with a shallow bow in Y.
    # Its surface is genuinely non-planar, but its normals remain tightly
    # clustered around a horizontal mean direction, so it has a clear overall
    # vertical orientation.
    cps = [
        [Vertex.ByCoordinates(0.0, 0.0, 0.0), Vertex.ByCoordinates(0.0, 0.0, 3.0)],
        [Vertex.ByCoordinates(1.0, 0.25, 0.0), Vertex.ByCoordinates(1.0, 0.25, 3.0)],
        [Vertex.ByCoordinates(2.0, 0.0, 0.0), Vertex.ByCoordinates(2.0, 0.0, 3.0)],
    ]
    weights = [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    return Face.ByNurbsParameters(
        cps,
        weights=weights,
        uKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 1.0, 1.0],
        isRational=False,
        uDegree=2,
        vDegree=1,
        silent=True,
    )


def _quarter_cylinder_face():
    s2 = math.sqrt(2.0) / 2.0
    cps = [
        [Vertex.ByCoordinates(1.0, 0.0, 0.0), Vertex.ByCoordinates(1.0, 0.0, 2.0)],
        [Vertex.ByCoordinates(1.0, 1.0, 0.0), Vertex.ByCoordinates(1.0, 1.0, 2.0)],
        [Vertex.ByCoordinates(0.0, 1.0, 0.0), Vertex.ByCoordinates(0.0, 1.0, 2.0)],
    ]
    weights = [
        [1.0, 1.0],
        [s2, s2],
        [1.0, 1.0],
    ]
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


def test_planar_box_classification_and_existing_keys_are_unchanged():
    cell = Cell.Box(width=4.0, length=3.0, height=2.0, silent=True)
    result = Topology.Decompose(cell, silent=True)

    assert isinstance(result, dict)
    assert len(result["externalVerticalFaces"]) == 4
    assert len(result["topHorizontalFaces"]) == 1
    assert len(result["bottomHorizontalFaces"]) == 1
    assert len(result["externalInclinedFaces"]) == 0

    # New categories are additive; ordinary planar geometry must not migrate
    # into them.
    assert result["externalCurvedFaces"] == []
    assert result["internalCurvedFaces"] == []
    assert result["freeCurvedFaces"] == []
    assert result["curvedFaces"] == []


def test_curved_category_keys_are_always_present():
    cell = Cell.Box(silent=True)
    result = Topology.Decompose(cell, silent=True)

    for key in (
        "externalCurvedFaces",
        "internalCurvedFaces",
        "freeCurvedFaces",
        "externalCurvedApertures",
        "internalCurvedApertures",
        "freeCurvedApertures",
        "curvedFaces",
    ):
        assert key in result
        assert isinstance(result[key], list)


@pytest.mark.pythonocc_only
def test_gently_curved_surface_retains_overall_vertical_classification():
    face = _gentle_vertical_nurbs_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False

    cluster = Cluster.ByTopologies([face], silent=True)
    result = Topology.Decompose(cluster, normalSpreadAngle=30.0, silent=True)

    assert face in result["freeVerticalFaces"]
    assert face not in result["freeCurvedFaces"]
    assert len(result["curvedFaces"]) == 0


@pytest.mark.pythonocc_only
def test_normal_spread_threshold_can_force_same_surface_into_curved_category():
    face = _gentle_vertical_nurbs_face()
    cluster = Cluster.ByTopologies([face], silent=True)

    result = Topology.Decompose(cluster, normalSpreadAngle=3.0, silent=True)

    assert face in result["freeCurvedFaces"]
    assert face not in result["freeVerticalFaces"]


@pytest.mark.pythonocc_only
def test_strongly_curved_patch_is_classified_as_curved():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False

    cluster = Cluster.ByTopologies([face], silent=True)
    result = Topology.Decompose(cluster, normalSpreadAngle=30.0, silent=True)

    assert face in result["freeCurvedFaces"]
    assert face not in result["freeVerticalFaces"]
    assert face in result["curvedFaces"]


@pytest.mark.pythonocc_only
def test_external_cylindrical_face_is_classified_as_external_curved():
    cell = Cell.Cylinder(
        radius=1.0,
        height=2.0,
        uSides=32,
        vSides=1,
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")

    result = Topology.Decompose(cell, normalSpreadAngle=30.0, silent=True)

    # The two planar end caps remain horizontal. The periodic lateral surface
    # has no coherent mean normal and must be in the curved category.
    assert len(result["topHorizontalFaces"]) == 1
    assert len(result["bottomHorizontalFaces"]) == 1
    assert len(result["externalCurvedFaces"]) >= 1
    assert all(face in result["curvedFaces"] for face in result["externalCurvedFaces"])
