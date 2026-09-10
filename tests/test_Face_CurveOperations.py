"""Surface-operation regressions complementing test_Face_SurfaceSupport."""

import pytest

from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire

TOL = 1.0e-4


def _curved_nurbs_face():
    z = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    cps = [
        [Vertex.ByCoordinates(float(i), float(j), z[i][j]) for j in range(4)]
        for i in range(4)
    ]
    return Face.ByNurbsParameters(
        controlPoints=cps,
        uDegree=3,
        vDegree=3,
        tolerance=TOL,
        silent=True,
    )


def test_plane_equation_remains_supported_for_planar_face():
    face = Face.Rectangle(width=3.0, length=2.0, silent=True)
    equation = Face.PlaneEquation(face, mantissa=6)
    assert isinstance(equation, dict)
    assert set(equation).issuperset({"a", "b", "c", "d"})


def test_angle_between_planar_faces_remains_supported():
    a = Face.Rectangle(width=2, length=2, direction=[0, 0, 1], silent=True)
    b = Face.Rectangle(width=2, length=2, direction=[1, 0, 0], silent=True)
    assert Face.Angle(a, b, mantissa=6) == pytest.approx(90.0, abs=1e-5)


@pytest.mark.pythonocc_only
def test_nurbs_trim_and_complement_conserve_area_and_surface_curvature():
    face = _curved_nurbs_face()
    vertices = [
        Face.VertexByParameters(face, 0.2, 0.2, tolerance=TOL, silent=True),
        Face.VertexByParameters(face, 0.8, 0.2, tolerance=TOL, silent=True),
        Face.VertexByParameters(face, 0.8, 0.8, tolerance=TOL, silent=True),
        Face.VertexByParameters(face, 0.2, 0.8, tolerance=TOL, silent=True),
    ]
    trim = Wire.ByVertices(vertices, close=True, tolerance=TOL, silent=True)

    # TrimByWire's public signature intentionally remains compact.
    inside = Face.TrimByWire(face, trim, reverse=False)
    outside = Face.TrimByWire(face, trim, reverse=True)

    assert Topology.IsInstance(inside, "Face")
    assert Topology.IsInstance(outside, "Face")

    area = Face.Area(face, mantissa=None, silent=True)
    combined = (
        Face.Area(inside, mantissa=None, silent=True)
        + Face.Area(outside, mantissa=None, silent=True)
    )
    assert combined == pytest.approx(area, rel=2e-5, abs=2e-5)
    assert Face.IsPlanar(inside, tolerance=TOL, silent=True) is False
    assert Face.IsPlanar(outside, tolerance=TOL, silent=True) is False
