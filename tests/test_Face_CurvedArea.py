"""Correctness tests for Face.Area on planar and genuinely curved faces."""

import math
import os

import pytest

from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "topologic_core").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _quarter_cylinder_nurbs(radius=1.0, height=1.0):
    s2 = math.sqrt(2.0) / 2.0
    r = float(radius)
    h = float(height)
    control_points = [
        [Vertex.ByCoordinates(r, 0.0, 0.0), Vertex.ByCoordinates(r, 0.0, h)],
        [Vertex.ByCoordinates(r, r, 0.0), Vertex.ByCoordinates(r, r, h)],
        [Vertex.ByCoordinates(0.0, r, 0.0), Vertex.ByCoordinates(0.0, r, h)],
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


def _circle_wire(z=0.0, radius=1.0):
    edge = Edge.Circle(
        origin=Vertex.ByCoordinates(0.0, 0.0, float(z)),
        radius=float(radius),
        placement="center",
        silent=True,
    )
    assert Topology.IsInstance(edge, "Edge")
    wire = Wire.ByEdges([edge], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


def test_face_area_planar_rectangle_remains_correct_and_unrounded_mode_works():
    face = Face.Rectangle(
        origin=Vertex.ByCoordinates(0.0, 0.0, 0.0),
        width=4.0,
        length=2.0,
        placement="lowerleft",
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    area = Face.Area(face, mantissa=None, silent=True)
    assert isinstance(area, float)
    assert math.isclose(area, 8.0, rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_face_area_validation_respects_silent():
    assert Face.Area(None, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_exact_quarter_cylinder_nurbs_area():
    radius = 2.0
    height = 3.0
    face = _quarter_cylinder_nurbs(radius=radius, height=height)
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False

    area = Face.Area(face, mantissa=None, silent=True)
    expected = 0.5 * math.pi * radius * height
    assert isinstance(area, float)
    assert math.isclose(area, expected, rel_tol=1.0e-8, abs_tol=1.0e-8)


@pytest.mark.pythonocc_only
def test_pythonocc_face_area_fixes_curved_shell_faces_that_previously_returned_zero():
    radius = 1.0
    height = 2.0
    shell = Shell.ByWires(
        [_circle_wire(0.0, radius), _circle_wire(height, radius)],
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(shell, "Shell")

    faces = Topology.Faces(shell, silent=True) or []
    assert len(faces) >= 1
    assert any(Face.IsPlanar(face, silent=True) is False for face in faces)

    areas = [Face.Area(face, mantissa=None, silent=True) for face in faces]
    assert all(isinstance(value, float) and value > 0.0 for value in areas)

    total = sum(areas)
    expected = 2.0 * math.pi * radius * height
    assert math.isclose(total, expected, rel_tol=1.0e-8, abs_tol=1.0e-8)
