import math
import os
from pathlib import Path

import pytest

rhino3dm = pytest.importorskip("rhino3dm")

from topologicpy.Core import Core
from topologicpy.Dictionary import Dictionary
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Rhino import Rhino
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


FIXTURE = Path(__file__).parent / "assets" / "representative_nurbs_test.3dm"


def _by_name(topologies):
    result = {}
    for topology in topologies:
        dictionary = Topology.Dictionary(topology)
        name = Dictionary.ValueAtKey(dictionary, "name")
        result[name] = topology
    return result


def test_open_nurbs_knots_are_expanded_for_occt():
    file = rhino3dm.File3dm.Read(str(FIXTURE))
    curve = file.Objects[0].Geometry
    expanded = Rhino._ExpandedKnots(curve.Knots)
    assert len(expanded) == len(curve.Points) + curve.Degree + 1
    assert expanded[:4] == [0.0, 0.0, 0.0, 0.0]
    assert expanded[-4:] == [2.0, 2.0, 2.0, 2.0]


def test_rational_control_points_are_dehomogenized():
    file = rhino3dm.File3dm.Read(str(FIXTURE))
    circle = file.Objects[2].Geometry
    point = circle.Points[1]
    x, y, z = Rhino._PointCoordinates(point)
    assert x == pytest.approx(point.X / point.W)
    assert y == pytest.approx(point.Y / point.W)
    assert z == pytest.approx(point.Z / point.W)


def test_imported_curves_match_rhino_at_sample_parameters():
    file = rhino3dm.File3dm.Read(str(FIXTURE))
    for index in range(4):
        rhino_curve = file.Objects[index].Geometry
        edge = Rhino._NurbsCurve(rhino_curve, tolerance=0.001, silent=True)
        assert Topology.IsInstance(edge, "Edge")
        domain = rhino_curve.Domain
        for parameter in (0.1, 0.37, 0.83):
            rhino_parameter = domain.T0 + (domain.T1 - domain.T0) * parameter
            rhino_point = rhino_curve.PointAt(rhino_parameter)
            imported_point = Edge.VertexByParameter(edge, parameter)
            x, y, z = Vertex.Coordinates(imported_point, mantissa=12)
            error = math.sqrt(
                (x - rhino_point.X) ** 2
                + (y - rhino_point.Y) ** 2
                + (z - rhino_point.Z) ** 2
            )
            assert error < 1.0e-9


def test_representative_3dm_import():
    topologies = Topology.By3DMPath(str(FIXTURE), silent=True)
    named = _by_name(topologies)

    for name in (
        "01_Open_NURBS",
        "02_Closed_NURBS",
        "03_Rational_Circle",
        "04_Rational_Ellipse",
        "06_Trimmed_Surface_With_Hole",
        "07_Closed_Polysurface",
    ):
        assert name in named

    assert Edge.Length(named["03_Rational_Circle"], mantissa=8) == pytest.approx(
        2.0 * math.pi * 8.0, abs=1.0e-7
    )
    assert Topology.IsInstance(named["06_Trimmed_Surface_With_Hole"], "Face")
    assert len(Topology.Edges(named["06_Trimmed_Surface_With_Hole"], silent=True)) == 5
    trimmed_area = Face.Area(
        named["06_Trimmed_Surface_With_Hole"], mantissa=None, silent=True
    )
    assert 400.0 < trimmed_area < 500.0
    assert len(Face.InternalBoundaries(named["06_Trimmed_Surface_With_Hole"])) == 1
    assert Topology.IsInstance(named["07_Closed_Polysurface"], "Cell")
    prism_faces = Topology.Faces(named["07_Closed_Polysurface"], silent=True)
    assert len(prism_faces) == 6
    assert len(Topology.Edges(named["07_Closed_Polysurface"], silent=True)) == 12
    for prism_face in prism_faces:
        mesh = Topology.Tessellate(prism_face, silent=True)
        assert isinstance(mesh, dict)
        assert len(mesh.get("faces", [])) > 0

    # Exact free-standing NURBS surfaces require the PythonOCC backend. The
    # TopologicCore fallback currently exposes curve construction but not the
    # array types needed by its native Face.BySurface overload.
    if type(Core.Backend()).__name__ == "PythonOCCBackend":
        assert "05_Untrimmed_NURBS_Surface" in named
        assert "08_Singular_Sphere_Surface" in named
        assert Topology.IsInstance(named["05_Untrimmed_NURBS_Surface"], "Face")
        assert Topology.IsInstance(named["08_Singular_Sphere_Surface"], "Face")


def test_filters_and_metadata():
    topologies = Rhino.By3DMPath(
        str(FIXTURE),
        objectNames=["03_Rational_Circle"],
        silent=True,
    )
    assert len(topologies) == 1
    dictionary = Topology.Dictionary(topologies[0])
    assert Dictionary.ValueAtKey(dictionary, "name") == "03_Rational_Circle"
    assert Dictionary.ValueAtKey(dictionary, "rhino_type") == "NurbsCurve"
    assert isinstance(Dictionary.ValueAtKey(dictionary, "color"), list)


def test_missing_path_returns_empty_list():
    assert Rhino.By3DMPath("does_not_exist.3dm", silent=True) == []
