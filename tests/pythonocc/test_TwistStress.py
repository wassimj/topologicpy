import math
import os
import random

import pytest

from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology


TOLERANCE = 0.0001
POINT_TOLERANCE = 3.0e-5
SEED = 20260815
RANDOM_CASES = int(os.environ.get("TOPOLOGICPY_TWIST_STRESS_CASES", "100"))


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _coords(vertex):
    return (
        float(Vertex.X(vertex, mantissa=9)),
        float(Vertex.Y(vertex, mantissa=9)),
        float(Vertex.Z(vertex, mantissa=9)),
    )


def _points(topology):
    return [_coords(v) for v in (Topology.Vertices(topology, silent=True) or [])]


def _counts(topology):
    return (
        len(Topology.Cells(topology, silent=True) or []),
        len(Topology.Faces(topology, silent=True) or []),
        len(Topology.Edges(topology, silent=True) or []),
        len(Topology.Vertices(topology, silent=True) or []),
    )


def _dedupe_points(points, tol=POINT_TOLERANCE):
    result = []
    for p in points:
        if not any(
            math.dist(p, q) <= tol
            for q in result
        ):
            result.append(p)
    return result


def _assert_point_sets_close(expected, actual, tol=POINT_TOLERANCE):
    expected = _dedupe_points(expected, tol=tol)
    actual = _dedupe_points(actual, tol=tol)

    assert len(actual) == len(expected), (
        f"Unique vertex count mismatch: expected {len(expected)}, "
        f"actual {len(actual)}.\nExpected={expected}\nActual={actual}"
    )

    unmatched = list(actual)

    for p in expected:
        best_index = None
        best_distance = None

        for i, q in enumerate(unmatched):
            distance = math.dist(p, q)

            if best_distance is None or distance < best_distance:
                best_index = i
                best_distance = distance

        assert best_distance is not None
        assert best_distance <= tol, (
            f"Expected point {p} has no corresponding result point within "
            f"{tol}. Nearest distance={best_distance}; actual={actual}"
        )

        unmatched.pop(best_index)


def _rotate_about_z(point, origin_xy, angle_degrees, ang_tolerance):
    x, y, z = point
    ox, oy = origin_xy

    if abs(float(angle_degrees)) < float(ang_tolerance):
        return (x, y, z)

    angle = math.radians(float(angle_degrees))
    c = math.cos(angle)
    s = math.sin(angle)

    dx = x - ox
    dy = y - oy

    return (
        ox + dx * c - dy * s,
        oy + dx * s + dy * c,
        z,
    )


def _expected_twist_points(
    points,
    origin_xyz,
    angle_range,
    ang_tolerance=0.01,
    z_tolerance=TOLERANCE,
):
    if not points:
        return []

    z_values = [p[2] for p in points]
    z_min = min(z_values)
    z_max = max(z_values)
    height = z_max - z_min

    if abs(height) <= z_tolerance:
        return list(points)

    a0 = float(angle_range[0])
    a1 = float(angle_range[1])

    ox = float(origin_xyz[0])
    oy = float(origin_xyz[1])

    result = []

    for point in points:
        ht = (point[2] - z_min) / height
        angle = a0 + ht * (a1 - a0)
        result.append(
            _rotate_about_z(
                point,
                (ox, oy),
                angle,
                ang_tolerance,
            )
        )

    return result


def _edge():
    return Edge.ByVertices(
        [
            _vertex(-1.4, 0.3, -0.8),
            _vertex(2.2, -1.1, 2.6),
        ],
        tolerance=TOLERANCE,
        silent=True,
    )


def _wire():
    # An open 3-D manifold Wire. Each Edge remains a valid straight Edge after
    # its two endpoints receive their independently interpolated twist angles.
    vertices = [
        _vertex(-2.0, -0.5, -1.5),
        _vertex(-0.7, 1.2, -0.3),
        _vertex(1.1, 0.6, 1.4),
        _vertex(2.4, -1.0, 3.0),
    ]

    return Wire.ByVertices(
        vertices,
        close=False,
        tolerance=TOLERANCE,
        silent=True,
    )


def _triangle_face():
    # Three transformed vertices always define a planar Face unless degenerate.
    face = Face.ByVertices(
        [
            _vertex(-1.7, -0.8, -1.1),
            _vertex(2.0, -0.2, 0.6),
            _vertex(0.3, 2.1, 2.7),
        ],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


def _quad_face():
    face = Face.ByVertices(
        [
            _vertex(-2.0, -1.0, -1.0),
            _vertex(2.0, -1.0, -1.0),
            _vertex(2.0, 1.0, 2.0),
            _vertex(-2.0, 1.0, 2.0),
        ],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


def _face_with_hole():
    outer = Wire.Rectangle(
        origin=_vertex(0.0, 0.0, 0.0),
        width=4.0,
        length=3.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )

    inner = Wire.Rectangle(
        origin=_vertex(0.35, -0.2, 0.0),
        width=1.1,
        length=0.8,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )

    face = Face.ByWires(
        outer,
        [inner],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")

    # Incline the entire holed Face so its vertices have varying Z.
    face = Topology.Rotate(
        face,
        origin=_vertex(0.0, 0.0, 0.0),
        axis=[1.0, 0.25, 0.0],
        angle=37.0,
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


def _cell():
    cell = Cell.Prism(
        origin=_vertex(0.4, -0.7, 1.1),
        width=2.4,
        length=1.7,
        height=3.2,
        uSides=1,
        vSides=1,
        wSides=1,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")
    return cell


def _shell():
    shell = Cell.ExternalBoundary(_cell())
    assert Topology.IsInstance(shell, "Shell")
    return shell


def _cellcomplex():
    cc = CellComplex.Prism(
        origin=_vertex(-0.6, 0.8, -0.3),
        width=2.8,
        length=2.2,
        height=2.6,
        uSides=2,
        vSides=2,
        wSides=2,
        placement="center",
        tolerance=TOLERANCE,
    )
    assert Topology.IsInstance(cc, "CellComplex")
    return cc


def _mixed_cluster():
    cell = _cell()

    wire = Topology.Translate(
        _wire(),
        x=5.0,
        y=0.0,
        z=0.2,
        transferDictionaries=False,
        silent=True,
    )

    edge = Topology.Translate(
        _edge(),
        x=-4.5,
        y=1.5,
        z=-0.4,
        transferDictionaries=False,
        silent=True,
    )

    vertex = _vertex(0.0, 5.0, 2.0)

    cluster = Cluster.ByTopologies(
        [cell, wire, edge, vertex],
        silent=True,
    )
    assert Topology.IsInstance(cluster, "Cluster")
    return cluster


DIRECT_FACTORIES = {
    "edge": _edge,
    "wire": _wire,
    "triangle_face": _triangle_face,
}

TRIANGULATED_FACTORIES = {
    "quad_face": _quad_face,
    "face_with_hole": _face_with_hole,
    "shell": _shell,
    "cell": _cell,
    "cellcomplex": _cellcomplex,
    "cluster": _mixed_cluster,
}


@pytest.fixture(scope="session", autouse=True)
def _pythonocc_backend_only():
    backend = Core.Backend()
    assert backend is not None
    assert backend.__class__.__name__ == "PythonOCCBackend", (
        "test_TwistStress.py must be run with the PythonOCC backend. "
        f"Active backend: {backend.__class__.__name__}"
    )


@pytest.mark.parametrize("factory_name", list(DIRECT_FACTORIES))
@pytest.mark.parametrize(
    "origin_xyz,angle_range",
    [
        ((0.0, 0.0, 0.0), [0.0, 90.0]),
        ((1.3, -2.1, 0.5), [-45.0, 120.0]),
        ((-2.5, 3.2, -1.0), [270.0, -135.0]),
        ((0.4, 0.7, 2.0), [720.0, -540.0]),
        ((-1.1, -0.9, 0.0), [15.0, 15.0]),
    ],
)
def test_twist_direct_coordinate_oracle(factory_name, origin_xyz, angle_range):
    topology = DIRECT_FACTORIES[factory_name]()
    original_type = Topology.TypeAsString(topology)
    original_counts = _counts(topology)
    original_points = _points(topology)

    expected_points = _expected_twist_points(
        original_points,
        origin_xyz,
        angle_range,
        ang_tolerance=0.01,
    )

    result = Topology.Twist(
        topology,
        origin=_vertex(*origin_xyz),
        angleRange=angle_range,
        triangulate=False,
        mantissa=9,
        angTolerance=0.01,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology"), (
        f"Twist returned invalid result for {factory_name}; "
        f"origin={origin_xyz}, angleRange={angle_range}"
    )
    assert Topology.TypeAsString(result) == original_type
    assert _counts(result) == original_counts
    _assert_point_sets_close(expected_points, _points(result))


@pytest.mark.parametrize("factory_name", list(DIRECT_FACTORIES))
def test_twist_direct_random_stress(factory_name):
    rng = random.Random(f"{SEED}:{factory_name}")

    for i in range(RANDOM_CASES):
        topology = DIRECT_FACTORIES[factory_name]()
        original_type = Topology.TypeAsString(topology)
        original_counts = _counts(topology)
        original_points = _points(topology)

        origin_xyz = (
            rng.uniform(-8.0, 8.0),
            rng.uniform(-8.0, 8.0),
            rng.uniform(-3.0, 3.0),
        )

        angle_range = [
            rng.uniform(-1080.0, 1080.0),
            rng.uniform(-1080.0, 1080.0),
        ]

        expected_points = _expected_twist_points(
            original_points,
            origin_xyz,
            angle_range,
            ang_tolerance=0.01,
        )

        result = Topology.Twist(
            topology,
            origin=_vertex(*origin_xyz),
            angleRange=angle_range,
            triangulate=False,
            mantissa=9,
            angTolerance=0.01,
            tolerance=TOLERANCE,
            silent=True,
        )

        assert Topology.IsInstance(result, "Topology"), (
            f"Twist failed for {factory_name}/random/{i}; "
            f"origin={origin_xyz}, angleRange={angle_range}"
        )
        assert Topology.TypeAsString(result) == original_type
        assert _counts(result) == original_counts
        _assert_point_sets_close(expected_points, _points(result))


@pytest.mark.parametrize("factory_name", list(TRIANGULATED_FACTORIES))
@pytest.mark.parametrize(
    "origin_xyz,angle_range",
    [
        ((0.0, 0.0, 0.0), [0.0, 30.0]),
        ((1.0, -0.8, 0.4), [-25.0, 40.0]),
        ((-1.4, 1.7, -0.3), [35.0, -20.0]),
    ],
)
def test_twist_triangulated_preserves_expected_type_counts_and_points(
    factory_name,
    origin_xyz,
    angle_range,
):
    topology = TRIANGULATED_FACTORIES[factory_name]()

    triangulated = Topology.Triangulate(
        topology,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(triangulated, "Topology")

    expected_type = Topology.TypeAsString(triangulated)
    expected_counts = _counts(triangulated)
    triangulated_points = _points(triangulated)

    expected_points = _expected_twist_points(
        triangulated_points,
        origin_xyz,
        angle_range,
        ang_tolerance=0.01,
    )

    result = Topology.Twist(
        topology,
        origin=_vertex(*origin_xyz),
        angleRange=angle_range,
        triangulate=True,
        mantissa=9,
        angTolerance=0.01,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology"), (
        f"Triangulated Twist returned invalid result for {factory_name}; "
        f"origin={origin_xyz}, angleRange={angle_range}"
    )
    assert Topology.TypeAsString(result) == expected_type, (
        f"Triangulated Twist changed type for {factory_name}: "
        f"{expected_type} -> {Topology.TypeAsString(result)}"
    )
    assert _counts(result) == expected_counts, (
        f"Triangulated Twist changed topology counts for {factory_name}: "
        f"{expected_counts} -> {_counts(result)}"
    )
    _assert_point_sets_close(expected_points, _points(result))


@pytest.mark.parametrize(
    "factory",
    [
        _edge,
        _wire,
        _triangle_face,
        _quad_face,
        _face_with_hole,
        _shell,
        _cell,
        _cellcomplex,
        _mixed_cluster,
    ],
)
def test_twist_zero_angle_range_returns_original(factory):
    topology = factory()

    result = Topology.Twist(
        topology,
        angleRange=[0.0, 0.0],
        triangulate=False,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert result is topology


def test_twist_zero_height_returns_original():
    face = Face.Rectangle(
        origin=_vertex(0.3, -0.4, 2.5),
        width=3.0,
        length=2.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")

    result = Topology.Twist(
        face,
        origin=_vertex(0.0, 0.0, 0.0),
        angleRange=[15.0, 85.0],
        triangulate=False,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert result is face


@pytest.mark.parametrize("factory_name", list(DIRECT_FACTORIES))
def test_twist_angles_below_ang_tolerance_leave_geometry_unchanged(factory_name):
    topology = DIRECT_FACTORIES[factory_name]()
    original_points = _points(topology)

    result = Topology.Twist(
        topology,
        origin=_vertex(1.0, -2.0, 0.0),
        angleRange=[0.001, -0.005],
        triangulate=False,
        angTolerance=0.01,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology")
    assert Topology.TypeAsString(result) == Topology.TypeAsString(topology)
    _assert_point_sets_close(original_points, _points(result))


@pytest.mark.parametrize(
    "angle_range",
    [
        None,
        [],
        [1.0],
        [1.0, 2.0, 3.0],
        ["bad", 30.0],
        [30.0, None],
        "0,90",
    ],
)
def test_twist_rejects_invalid_angle_range(angle_range):
    topology = _edge()

    result = Topology.Twist(
        topology,
        angleRange=angle_range,
        silent=True,
    )

    assert result is None


def test_twist_default_origin_matches_centroid_oracle():
    topology = _triangle_face()

    centroid = Topology.Centroid(topology)
    assert Topology.IsInstance(centroid, "Vertex")

    origin_xyz = _coords(centroid)
    angle_range = [-35.0, 70.0]

    expected_points = _expected_twist_points(
        _points(topology),
        origin_xyz,
        angle_range,
        ang_tolerance=0.01,
    )

    result = Topology.Twist(
        topology,
        origin=None,
        angleRange=angle_range,
        triangulate=False,
        mantissa=9,
        angTolerance=0.01,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Face")
    _assert_point_sets_close(expected_points, _points(result))


@pytest.mark.parametrize(
    "factory",
    [
        _quad_face,
        _face_with_hole,
        _shell,
        _cell,
    ],
)
def test_nontriangulated_warped_topology_is_not_coerced_by_pythonocc(factory):
    """
    A Z-varying twist can turn originally planar quadrilateral boundaries into
    warped polygons. PythonOCC must not be rescued by Topology.Fix into a
    fabricated topology. A native failure or a naturally valid result is
    acceptable; if a result is returned, it must contain the exact transformed
    vertices predicted by the twist mapping.
    """
    topology = factory()
    origin_xyz = (0.2, -0.3, 0.0)
    angle_range = [0.0, 55.0]

    expected_points = _expected_twist_points(
        _points(topology),
        origin_xyz,
        angle_range,
        ang_tolerance=0.01,
    )

    result = Topology.Twist(
        topology,
        origin=_vertex(*origin_xyz),
        angleRange=angle_range,
        triangulate=False,
        mantissa=9,
        angTolerance=0.01,
        tolerance=TOLERANCE,
        silent=True,
    )

    if result is not None:
        assert Topology.IsInstance(result, "Topology")
        _assert_point_sets_close(expected_points, _points(result))
