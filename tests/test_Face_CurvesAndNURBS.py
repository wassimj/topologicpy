import math

import pytest

from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


TOL = 1.0e-4


def _xyz(vertex):
    return Vertex.Coordinates(vertex, mantissa=None)


def _length(vector):
    return math.sqrt(sum(value * value for value in vector))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _make_curved_nurbs_face():
    z_values = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    control_points = []

    for i in range(4):
        row = []
        for j in range(4):
            row.append(
                Vertex.ByCoordinates(
                    float(i),
                    float(j),
                    z_values[i][j],
                )
            )
        control_points.append(row)

    return Face.ByNurbsParameters(
        controlPoints=control_points,
        uDegree=3,
        vDegree=3,
        tolerance=TOL,
        silent=True,
    )


def _make_planar_nurbs_face():
    control_points = []

    for i in range(4):
        row = []
        for j in range(4):
            row.append(
                Vertex.ByCoordinates(
                    float(i),
                    float(j),
                    2.0,
                )
            )
        control_points.append(row)

    return Face.ByNurbsParameters(
        controlPoints=control_points,
        uDegree=3,
        vDegree=3,
        tolerance=TOL,
        silent=True,
    )


def test_nurbs_face_is_one_true_curved_face():
    face = _make_curved_nurbs_face()

    assert Topology.IsInstance(face, "Face")
    assert len(Topology.Faces(face, silent=True) or []) == 1
    assert Face.IsPlanar(face, tolerance=TOL, silent=True) is False
    assert Face.Area(face, mantissa=None, silent=True) > 9.0


def test_nurbs_face_uv_evaluation_preserves_corner_points():
    face = _make_curved_nurbs_face()

    p00 = Face.VertexByParameters(
        face,
        u=0.0,
        v=0.0,
        tolerance=TOL,
        silent=True,
    )
    p11 = Face.VertexByParameters(
        face,
        u=1.0,
        v=1.0,
        tolerance=TOL,
        silent=True,
    )

    assert _xyz(p00) == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-6)
    assert _xyz(p11) == pytest.approx([3.0, 3.0, 0.0], abs=1.0e-6)

    center = Face.VertexByParameters(
        face,
        u=0.5,
        v=0.5,
        tolerance=TOL,
        silent=True,
    )

    uv = Face.VertexParameters(
        face,
        center,
        outputType="uv",
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    assert uv == pytest.approx([0.5, 0.5], abs=1.0e-5)


def test_nurbs_normal_and_tangents_are_local_surface_directions():
    face = _make_curved_nurbs_face()

    normal = Face.NormalAtParameters(
        face,
        u=0.35,
        v=0.4,
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    tangents = Face.TangentsAtParameters(
        face,
        u=0.35,
        v=0.4,
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    assert isinstance(normal, list)
    assert len(normal) == 3
    assert _length(normal) == pytest.approx(1.0, abs=1.0e-6)

    assert isinstance(tangents, dict)
    assert set(tangents.keys()) >= {"u", "v"}

    tangent_u = tangents["u"]
    tangent_v = tangents["v"]

    assert _length(tangent_u) == pytest.approx(1.0, abs=2.0e-5)
    assert _length(tangent_v) == pytest.approx(1.0, abs=2.0e-5)

    assert _dot(normal, tangent_u) == pytest.approx(0.0, abs=2.0e-4)
    assert _dot(normal, tangent_v) == pytest.approx(0.0, abs=2.0e-4)

    assert Face.TangentAtParameters(
        face,
        u=0.35,
        v=0.4,
        axis="u",
        mantissa=None,
        tolerance=TOL,
        silent=True,
    ) == pytest.approx(tangent_u, abs=2.0e-5)

    assert Face.TangentAtParameters(
        face,
        u=0.35,
        v=0.4,
        axis="v",
        mantissa=None,
        tolerance=TOL,
        silent=True,
    ) == pytest.approx(tangent_v, abs=2.0e-5)


def test_nurbs_curvature_returns_complete_finite_result():
    face = _make_curved_nurbs_face()

    curvature = Face.CurvatureAtParameters(
        face,
        u=0.35,
        v=0.4,
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    assert isinstance(curvature, dict)

    for key in ["maximum", "minimum", "mean", "gaussian"]:
        assert key in curvature
        assert math.isfinite(float(curvature[key]))

    assert "maximumDirection" in curvature
    assert "minimumDirection" in curvature
    assert "isUmbilic" in curvature
    assert isinstance(curvature["isUmbilic"], bool)


def test_planar_nurbs_surface_is_recognized_as_planar():
    face = _make_planar_nurbs_face()

    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, tolerance=TOL, silent=True) is True

    equation = Face.PlaneEquation(
        face,
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    assert isinstance(equation, dict)

    # Every sampled point must satisfy the returned plane equation.
    for u, v in [(0.0, 0.0), (0.2, 0.7), (0.5, 0.5), (1.0, 1.0)]:
        x, y, z = _xyz(
            Face.VertexByParameters(
                face,
                u=u,
                v=v,
                tolerance=TOL,
                silent=True,
            )
        )
        value = (
            equation["a"] * x
            + equation["b"] * y
            + equation["c"] * z
            + equation["d"]
        )
        assert value == pytest.approx(0.0, abs=2.0e-5)


def test_plane_equation_and_global_angle_reject_curved_faces():
    curved = _make_curved_nurbs_face()
    planar = Face.Rectangle(
        width=3.0,
        length=3.0,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(planar, "Face")

    assert Face.PlaneEquation(
        curved,
        tolerance=TOL,
        silent=True,
    ) is None

    assert Face.Angle(
        curved,
        planar,
        tolerance=TOL,
        silent=True,
    ) is None


def test_angle_between_planar_faces_remains_supported():
    face_a = Face.Rectangle(
        width=2.0,
        length=2.0,
        direction=[0.0, 0.0, 1.0],
        tolerance=TOL,
        silent=True,
    )

    face_b = Face.Rectangle(
        width=2.0,
        length=2.0,
        direction=[1.0, 0.0, 0.0],
        tolerance=TOL,
        silent=True,
    )

    angle = Face.Angle(
        face_a,
        face_b,
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    assert angle == pytest.approx(90.0, abs=1.0e-5)


def test_surface_aware_nurbs_trim_and_reverse_conserve_area():
    face = _make_curved_nurbs_face()

    trim_vertices = [
        Face.VertexByParameters(face, 0.2, 0.2, tolerance=TOL, silent=True),
        Face.VertexByParameters(face, 0.8, 0.2, tolerance=TOL, silent=True),
        Face.VertexByParameters(face, 0.8, 0.8, tolerance=TOL, silent=True),
        Face.VertexByParameters(face, 0.2, 0.8, tolerance=TOL, silent=True),
    ]

    assert all(
        Topology.IsInstance(vertex, "Vertex")
        for vertex in trim_vertices
    )

    trim_wire = Wire.ByVertices(
        trim_vertices,
        close=True,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(trim_wire, "Wire")

    trimmed = Face.TrimByWire(
        face,
        trim_wire,
        reverse=False,
        tolerance=TOL,
        silent=True,
    )

    complement = Face.TrimByWire(
        face,
        trim_wire,
        reverse=True,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(trimmed, "Face")
    assert Topology.IsInstance(complement, "Face")

    original_area = Face.Area(
        face,
        mantissa=None,
        silent=True,
    )
    trimmed_area = Face.Area(
        trimmed,
        mantissa=None,
        silent=True,
    )
    complement_area = Face.Area(
        complement,
        mantissa=None,
        silent=True,
    )

    assert trimmed_area > 0.0
    assert complement_area > 0.0

    assert trimmed_area + complement_area == pytest.approx(
        original_area,
        rel=2.0e-5,
        abs=2.0e-5,
    )

    # Both pieces must retain the non-planar supporting geometry.
    assert Face.IsPlanar(trimmed, tolerance=TOL, silent=True) is False
    assert Face.IsPlanar(complement, tolerance=TOL, silent=True) is False


def test_normal_edge_uses_local_normal_on_curved_face():
    face = _make_curved_nurbs_face()

    normal_edge = Face.NormalEdge(
        face,
        length=1.0,
        tolerance=TOL,
        silent=True,
    )

    assert Topology.IsInstance(normal_edge, "Edge")

    start = Topology.Vertices(normal_edge, silent=True)[0]
    end = Topology.Vertices(normal_edge, silent=True)[-1]

    sx, sy, sz = _xyz(start)
    ex, ey, ez = _xyz(end)

    vector = [ex - sx, ey - sy, ez - sz]

    assert _length(vector) == pytest.approx(1.0, abs=2.0e-5)

    uv = Face.VertexParameters(
        face,
        start,
        outputType="uv",
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    local_normal = Face.NormalAtParameters(
        face,
        u=uv[0],
        v=uv[1],
        mantissa=None,
        tolerance=TOL,
        silent=True,
    )

    unit_vector = [
        value / _length(vector)
        for value in vector
    ]

    assert unit_vector == pytest.approx(local_normal, abs=2.0e-4)
