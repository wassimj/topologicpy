import math
import os
import random

import pytest

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
POINT_TOLERANCE = 1.0e-6
SEED = 20260815
RANDOM_CASES = int(os.environ.get("TOPOLOGICPY_ROTATE_STRESS_CASES", "250"))


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


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
    shell = Shell.ByFaces(
        Topology.Faces(_cell(), silent=True),
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )
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


def _branching_wire():
    c = _vertex(0.25, -0.35, 0.6)
    endpoints = [
        _vertex(2.0, -0.2, 0.9),
        _vertex(-1.4, 1.3, 0.2),
        _vertex(0.1, -1.8, 1.7),
        _vertex(0.8, 0.4, -1.1),
    ]
    edges = [
        Edge.ByVertices([c, p], tolerance=TOLERANCE, silent=True)
        for p in endpoints
    ]
    wire = Wire.ByEdges(edges, tolerance=TOLERANCE, silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


def _face_with_hole():
    outer = Wire.Rectangle(
        origin=_vertex(0.2, -0.1, 0.5),
        width=4.0,
        length=3.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    inner = Wire.Rectangle(
        origin=_vertex(0.55, -0.3, 0.5),
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
    return face


def _cluster():
    face = Face.Rectangle(
        origin=_vertex(-2.2, 0.7, 0.4),
        width=1.3,
        length=0.9,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    edge = Edge.ByVertices(
        [_vertex(1.7, -0.8, 0.2), _vertex(2.8, 0.4, 1.6)],
        tolerance=TOLERANCE,
        silent=True,
    )
    point = _vertex(0.1, 2.4, -0.9)
    cluster = Cluster.ByTopologies([face, edge, point], silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    return cluster


FACTORIES = {
    "vertex": lambda: _vertex(1.2, -0.7, 2.3),
    "edge": lambda: Edge.ByVertices(
        [_vertex(-1.2, 0.5, 2.1), _vertex(2.4, -0.8, -0.3)],
        tolerance=TOLERANCE,
        silent=True,
    ),
    "wire": lambda: Wire.Rectangle(
        origin=_vertex(0.4, -0.6, 0.7),
        width=2.7,
        length=1.6,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    ),
    "branching_wire": _branching_wire,
    "face": lambda: Face.Rectangle(
        origin=_vertex(-0.3, 0.9, 1.2),
        width=2.5,
        length=1.4,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    ),
    "face_with_hole": _face_with_hole,
    "shell": _shell,
    "cell": _cell,
    "cellcomplex": _cellcomplex,
    "cluster": _cluster,
}


def _coords(vertex):
    return (
        float(Vertex.X(vertex, mantissa=12)),
        float(Vertex.Y(vertex, mantissa=12)),
        float(Vertex.Z(vertex, mantissa=12)),
    )


def _points(topology):
    if Topology.IsInstance(topology, "Vertex"):
        return [_coords(topology)]
    return [_coords(v) for v in (Topology.Vertices(topology, silent=True) or [])]


def _counts(topology):
    if not Topology.IsInstance(topology, "Topology"):
        return None
    return (
        len(Topology.Cells(topology, silent=True) or []),
        len(Topology.Faces(topology, silent=True) or []),
        len(Topology.Edges(topology, silent=True) or []),
        len(Topology.Vertices(topology, silent=True) or []),
    )


def _distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def _rotate_point(point, origin, axis, angle_degrees):
    """Independent right-hand Rodrigues rotation used as the geometry oracle."""
    ax, ay, az = axis
    mag = math.sqrt(ax * ax + ay * ay + az * az)
    assert mag > 0.0
    kx, ky, kz = ax / mag, ay / mag, az / mag

    vx = point[0] - origin[0]
    vy = point[1] - origin[1]
    vz = point[2] - origin[2]

    theta = math.radians(angle_degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    dot = kx * vx + ky * vy + kz * vz
    cx = ky * vz - kz * vy
    cy = kz * vx - kx * vz
    cz = kx * vy - ky * vx

    rx = vx * c + cx * s + kx * dot * (1.0 - c)
    ry = vy * c + cy * s + ky * dot * (1.0 - c)
    rz = vz * c + cz * s + kz * dot * (1.0 - c)

    return (
        rx + origin[0],
        ry + origin[1],
        rz + origin[2],
    )


def _assert_point_sets_close(expected, actual, tolerance=POINT_TOLERANCE):
    assert len(actual) == len(expected), (
        f"Vertex-count mismatch in point-set comparison: "
        f"expected {len(expected)}, actual {len(actual)}"
    )

    remaining = list(actual)
    for expected_point in expected:
        nearest_index = min(
            range(len(remaining)),
            key=lambda i: _distance(expected_point, remaining[i]),
        )
        nearest_distance = _distance(expected_point, remaining[nearest_index])
        assert nearest_distance <= tolerance, (
            f"Expected rotated vertex {expected_point} has no matching result vertex. "
            f"Nearest distance = {nearest_distance:.12g}; tolerance = {tolerance}."
        )
        remaining.pop(nearest_index)

    assert not remaining


def _random_axis(rng):
    while True:
        axis = (
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
        )
        if math.sqrt(sum(v * v for v in axis)) > 1.0e-6:
            return axis


def _random_case(rng):
    origin = (
        rng.uniform(-10.0, 10.0),
        rng.uniform(-10.0, 10.0),
        rng.uniform(-10.0, 10.0),
    )
    axis = _random_axis(rng)

    # Stay outside the public angTolerance no-op region so every case actually
    # exercises the active backend rotation implementation.
    while True:
        angle = rng.uniform(-1440.0, 1440.0)
        if abs(angle) >= 0.01:
            break

    return origin, axis, angle


def _exercise_rotation(topology, origin_xyz, axis, angle, label):
    original_type = Topology.TypeAsString(topology)
    original_counts = _counts(topology)
    original_points = _points(topology)

    origin = _vertex(*origin_xyz)
    result = Topology.Rotate(
        topology,
        origin=origin,
        axis=list(axis),
        angle=angle,
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology"), (
        f"Rotate returned an invalid result for {label}; "
        f"origin={origin_xyz}, axis={axis}, angle={angle}"
    )
    assert Topology.TypeAsString(result) == original_type, (
        f"Rotate changed topology type for {label}: "
        f"{original_type} -> {Topology.TypeAsString(result)}; "
        f"origin={origin_xyz}, axis={axis}, angle={angle}"
    )
    assert _counts(result) == original_counts, (
        f"Rotate changed subtopology counts for {label}: "
        f"{original_counts} -> {_counts(result)}; "
        f"origin={origin_xyz}, axis={axis}, angle={angle}"
    )

    expected_points = [
        _rotate_point(point, origin_xyz, axis, angle)
        for point in original_points
    ]
    _assert_point_sets_close(expected_points, _points(result))

    # Reverse the exact rotation. This also verifies that repeated wrapping and
    # rigid transformation preserve a recoverable topology.
    restored = Topology.Rotate(
        result,
        origin=origin,
        axis=list(axis),
        angle=-angle,
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(restored, "Topology"), (
        f"Inverse Rotate failed for {label}; "
        f"origin={origin_xyz}, axis={axis}, angle={angle}"
    )
    assert Topology.TypeAsString(restored) == original_type
    assert _counts(restored) == original_counts
    _assert_point_sets_close(original_points, _points(restored), tolerance=2.0e-6)



@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_rotate_deterministic_stress(factory_name):
    topology = FACTORIES[factory_name]()

    axes = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (-2.0, 3.0, 5.0),
        (0.125, -7.0, 2.5),
    ]
    origins = [
        (0.0, 0.0, 0.0),
        (1.25, -2.5, 0.75),
        (-3.1, 4.2, 2.7),
    ]
    angles = [
        0.001,
        0.01,
        1.0,
        45.0,
        89.999,
        90.0,
        179.999,
        180.0,
        270.0,
        359.999,
        360.0,
        -45.0,
        -180.0,
        720.0,
        -720.0,
    ]

    case_index = 0
    for axis in axes:
        for origin in origins:
            for angle in angles:
                _exercise_rotation(
                    topology,
                    origin,
                    axis,
                    angle,
                    label=f"{factory_name}/deterministic/{case_index}",
                )
                case_index += 1


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_rotate_random_stress(factory_name):
    topology = FACTORIES[factory_name]()
    rng = random.Random(f"{SEED}:{factory_name}")

    for case_index in range(RANDOM_CASES):
        origin, axis, angle = _random_case(rng)
        _exercise_rotation(
            topology,
            origin,
            axis,
            angle,
            label=f"{factory_name}/random/{case_index}",
        )


def test_rotate_repeated_full_revolution_cellcomplex():
    """Exercise 360 consecutive rotations on a shared-topology model."""
    topology = _cellcomplex()
    original_type = Topology.TypeAsString(topology)
    original_counts = _counts(topology)
    original_points = _points(topology)

    origin = _vertex(1.7, -0.9, 2.2)
    axis = [1.0, -2.0, 3.0]

    result = topology
    for i in range(360):
        result = Topology.Rotate(
            result,
            origin=origin,
            axis=axis,
            angle=1.0,
            transferDictionaries=False,
            tolerance=TOLERANCE,
            silent=True,
        )
        assert Topology.IsInstance(result, "Topology"), f"Rotation failed at iteration {i + 1}"
        assert Topology.TypeAsString(result) == original_type
        assert _counts(result) == original_counts

    _assert_point_sets_close(original_points, _points(result), tolerance=1.0e-5)
