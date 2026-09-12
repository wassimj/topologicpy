import math
import pytest

from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Topology import Topology

TOL = 1.0e-4


def V(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _coords(v):
    return Vertex.Coordinates(v, mantissa=None)


def _unit(values):
    m = math.sqrt(sum(float(x) * float(x) for x in values))
    return [float(x) / m for x in values]


def _rotate_x(values, angle_deg):
    x, y, z = [float(v) for v in values]
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    return [x, c * y - s * z, s * y + c * z]


def _assert_vec(actual, expected, tol=1.0e-6):
    assert actual is not None
    assert len(actual) == 3
    for a, e in zip(actual, expected):
        assert float(a) == pytest.approx(float(e), abs=tol)


def test_face_circle_preserves_historical_faceted_default():
    radius = 2.0
    sides = 7
    face = Face.Circle(radius=radius, sides=sides, tolerance=TOL)
    assert Topology.IsInstance(face, "Face")
    expected = 0.5 * sides * radius * radius * math.sin(2.0 * math.pi / sides)
    assert Face.Area(face, mantissa=None) == pytest.approx(expected, rel=1.0e-8, abs=1.0e-8)


@pytest.mark.skipif(
    Topology._IsTopologicCoreBackend(),
    reason="Requires exact PythonOCC/OCCT curve or surface support.",
)
def test_face_circle_exact_mode_preserves_curved_boundary():
    face = Face.Circle(radius=2.0, sides=4, polyline=False, tolerance=TOL, silent=True)
    assert Topology.IsInstance(face, "Face")
    boundary = Face.ExternalBoundary(face, silent=True)
    assert Topology.IsInstance(boundary, "Wire")
    assert Wire.IsPolyline(boundary) is False
    assert Face.Area(face, mantissa=None) == pytest.approx(math.pi * 4.0, rel=1.0e-8, abs=1.0e-8)


def test_face_bywire_preserves_exact_circle():
    wire = Wire.Circle(radius=2.0, sides=1, polyline=False, silent=True)
    face = Face.ByWire(wire, tolerance=TOL, silent=True)
    assert Topology.IsInstance(face, "Face")
    boundary = Face.ExternalBoundary(face, silent=True)
    assert Topology.IsInstance(boundary, "Wire")
    assert Wire.IsPolyline(boundary) is False
    assert Face.Area(face, mantissa=None) == pytest.approx(math.pi * 4.0, rel=1.0e-8, abs=1.0e-8)


@pytest.mark.skipif(
    Topology._IsTopologicCoreBackend(),
    reason="Requires exact PythonOCC/OCCT curve or surface support.",
)
def test_face_ellipse_exact_mode_preserves_rational_curves():
    face = Face.Ellipse(width=4.0, length=2.0, sides=4, polyline=False, tolerance=TOL, silent=True)
    assert Topology.IsInstance(face, "Face")
    boundary = Face.ExternalBoundary(face, silent=True)
    assert Topology.IsInstance(boundary, "Wire")
    assert Wire.IsPolyline(boundary) is False
    assert Face.Area(face, mantissa=None) == pytest.approx(math.pi * 2.0 * 1.0, rel=1.0e-8, abs=1.0e-8)


@pytest.mark.skipif(
    Topology._IsTopologicCoreBackend(),
    reason="Requires exact PythonOCC/OCCT curve or surface support.",
)
def test_face_bywires_preserves_curved_outer_and_inner_boundaries():
    outer = Wire.Circle(radius=3.0, sides=4, polyline=False, silent=True)
    inner = Wire.Ellipse(width=2.0, length=1.0, sides=4, polyline=False, silent=True)
    face = Face.ByWires(outer, [inner], tolerance=TOL, silent=True)
    assert Topology.IsInstance(face, "Face")

    external = Face.ExternalBoundary(face, silent=True)
    internals = Face.InternalBoundaries(face) or []
    assert Topology.IsInstance(external, "Wire")
    assert Wire.IsPolyline(external) is False
    assert len(internals) == 1
    assert Wire.IsPolyline(internals[0]) is False

    expected = math.pi * 3.0 * 3.0 - math.pi * 1.0 * 0.5
    assert Face.Area(face, mantissa=None) == pytest.approx(expected, rel=1.0e-8, abs=1.0e-8)


def _nurbs_face():
    xs = [-2.0, -0.6666666667, 0.6666666667, 2.0]
    ys = [-2.0, -0.6666666667, 0.6666666667, 2.0]
    z = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.2, 0.7, 0.0],
        [0.0, 0.4, 1.4, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    cps = [[V(x, y, z[i][j]) for j, y in enumerate(ys)] for i, x in enumerate(xs)]
    return Face.ByNurbsParameters(
        controlPoints=cps,
        weights=None,
        uKnots=None,
        vKnots=None,
        isRational=False,
        isUPeriodic=False,
        isVPeriodic=False,
        uDegree=3,
        vDegree=3,
        tolerance=TOL,
        silent=True,
    )


@pytest.mark.skipif(
    Topology._IsTopologicCoreBackend(),
    reason="Requires exact PythonOCC/OCCT curve or surface support.",
)
def test_pythonocc_surface_evaluation_respects_face_location():
    face = _nurbs_face()
    assert Topology.IsInstance(face, "Face")

    u, v = 0.31, 0.63
    point0 = Face.VertexByParameters(face, u=u, v=v, tolerance=TOL, silent=True)
    normal0 = Face.NormalAtParameters(face, u=u, v=v, mantissa=None, tolerance=TOL, silent=True)
    tangents0 = Face.TangentsAtParameters(face, u=u, v=v, mantissa=None, tolerance=TOL, silent=True)
    assert Topology.IsInstance(point0, "Vertex")
    assert normal0 is not None
    assert isinstance(tangents0, dict)

    angle = 37.0
    moved = Topology.Rotate(face, origin=V(0, 0, 0), axis=[1, 0, 0], angle=angle, silent=True)
    moved = Topology.Translate(moved, x=5.0, y=-3.0, z=2.0, silent=True)
    assert Topology.IsInstance(moved, "Face")

    point1 = Face.VertexByParameters(moved, u=u, v=v, tolerance=TOL, silent=True)
    normal1 = Face.NormalAtParameters(moved, u=u, v=v, mantissa=None, tolerance=TOL, silent=True)
    tangents1 = Face.TangentsAtParameters(moved, u=u, v=v, mantissa=None, tolerance=TOL, silent=True)

    expected_point = _rotate_x(_coords(point0), angle)
    expected_point = [expected_point[0] + 5.0, expected_point[1] - 3.0, expected_point[2] + 2.0]
    _assert_vec(_coords(point1), expected_point, tol=2.0e-6)
    _assert_vec(normal1, _unit(_rotate_x(normal0, angle)), tol=2.0e-6)
    _assert_vec(tangents1["u"], _unit(_rotate_x(tangents0["u"], angle)), tol=2.0e-6)
    _assert_vec(tangents1["v"], _unit(_rotate_x(tangents0["v"], angle)), tol=2.0e-6)

    uv = Face.VertexParameters(moved, point1, outputType="uv", mantissa=None, tolerance=TOL, silent=True)
    assert uv[0] == pytest.approx(u, abs=2.0e-6)
    assert uv[1] == pytest.approx(v, abs=2.0e-6)


@pytest.mark.skipif(
    Topology._IsTopologicCoreBackend(),
    reason="Requires exact PythonOCC/OCCT curve or surface support.",
)
def test_pythonocc_normaledge_uses_local_surface_normal():
    face = _nurbs_face()
    assert Topology.IsInstance(face, "Face")

    edge = Face.NormalEdge(face, length=2.0, tolerance=TOL, silent=True)
    assert Topology.IsInstance(edge, "Edge")

    start = Edge.StartVertex(edge)
    end = Edge.EndVertex(edge)
    assert Topology.IsInstance(start, "Vertex")
    assert Topology.IsInstance(end, "Vertex")

    uv = Face.VertexParameters(face, start, outputType="uv", mantissa=None, tolerance=TOL, silent=True)
    assert uv is not None and len(uv) == 2

    expected = Face.NormalAtParameters(
        face,
        u=uv[0],
        v=uv[1],
        outputType="xyz",
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )
    direction = Edge.Direction(edge, mantissa=None)
    _assert_vec(direction, expected, tol=2.0e-6)
    assert Edge.Length(edge, mantissa=None) == pytest.approx(2.0, abs=2.0e-6)
