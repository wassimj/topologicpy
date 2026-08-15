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
POINT_TOLERANCE = 2.0e-5
SEED = 20260815
RANDOM_CASES = int(os.environ.get("TOPOLOGICPY_TAPER_STRESS_CASES", "100"))


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _inclined_wire():
    wire = Wire.Rectangle(
        origin=_vertex(0.4, -0.5, 0.7),
        width=2.8,
        length=1.6,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    wire = Topology.Rotate(
        wire,
        origin=_vertex(0.1, 0.2, -0.3),
        axis=[1.0, 0.3, 0.0],
        angle=37.0,
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(wire, "Wire")
    return wire


def _inclined_face():
    face = Face.Rectangle(
        origin=_vertex(-0.3, 0.8, 0.6),
        width=2.7,
        length=1.9,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    face = Topology.Rotate(
        face,
        origin=_vertex(0.2, -0.4, 0.1),
        axis=[1.0, -0.2, 0.1],
        angle=41.0,
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


def _inclined_face_with_hole():
    outer = Wire.Rectangle(
        origin=_vertex(0.2, -0.1, 0.4),
        width=4.0,
        length=3.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    inner = Wire.Rectangle(
        origin=_vertex(0.55, -0.3, 0.4),
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
    face = Topology.Rotate(
        face,
        origin=_vertex(-0.3, 0.5, 0.0),
        axis=[0.8, 0.25, 0.1],
        angle=33.0,
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
    cell = _cell()
    shell = Cell.ExternalBoundary(cell)
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


def _cluster():
    cell = _cell()
    wire = _inclined_wire()
    edge = Edge.ByVertices(
        [_vertex(-3.0, 1.1, -1.2), _vertex(-2.1, 2.0, 2.6)],
        tolerance=TOLERANCE,
        silent=True,
    )
    cluster = Cluster.ByTopologies([cell, wire, edge], silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    return cluster


FACTORIES = {
    "edge": lambda: Edge.ByVertices(
        [_vertex(-1.2, 0.5, -1.1), _vertex(2.4, -0.8, 2.7)],
        tolerance=TOLERANCE,
        silent=True,
    ),
    "wire": _inclined_wire,
    "face": _inclined_face,
    "face_with_hole": _inclined_face_with_hole,
    "shell": _shell,
    "cell": _cell,
    "cellcomplex": _cellcomplex,
    "cluster": _cluster,
}


# An inclined Face with an internal boundary generally becomes non-planar
# under a Z-varying taper. A single untriangulated Face is therefore not a
# valid expected result for that case. It is tested separately below using
# triangulate=True.
NON_DEGENERATE_FACTORIES = [
    "edge",
    "wire",
    "face",
    "shell",
    "cell",
    "cellcomplex",
    "cluster",
]


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


def _deduplicate_points(points, tolerance=1.0e-6):
    result = []
    for point in points:
        if not any(_distance(point, existing) <= tolerance for existing in result):
            result.append(point)
    return result


def _assert_point_sets_close(expected, actual, tolerance=POINT_TOLERANCE):
    expected = _deduplicate_points(expected, tolerance=tolerance * 0.25)
    actual = _deduplicate_points(actual, tolerance=tolerance * 0.25)

    assert len(actual) == len(expected), (
        f"Unique vertex-count mismatch: expected {len(expected)}, actual {len(actual)}.\n"
        f"Expected: {expected}\nActual: {actual}"
    )

    remaining = list(actual)
    for expected_point in expected:
        nearest_index = min(
            range(len(remaining)),
            key=lambda i: _distance(expected_point, remaining[i]),
        )
        nearest_distance = _distance(expected_point, remaining[nearest_index])
        assert nearest_distance <= tolerance, (
            f"Expected tapered vertex {expected_point} has no matching result vertex. "
            f"Nearest distance = {nearest_distance:.12g}; tolerance = {tolerance}."
        )
        remaining.pop(nearest_index)


def _clamped_ratio_range(ratio_range):
    return [min(1.0, ratio_range[0]), min(1.0, ratio_range[1])]


def _expected_taper_points(points, origin_xyz, ratio_range):
    ratio_range = _clamped_ratio_range(ratio_range)
    z_min = min(p[2] for p in points)
    z_max = max(p[2] for p in points)
    height = z_max - z_min
    assert height > TOLERANCE

    ox, oy, _ = origin_xyz
    expected = []

    for x, y, z in points:
        ht = (z - z_min) / height
        rt = ratio_range[0] + ht * (ratio_range[1] - ratio_range[0])
        expected.append(
            (
                x + (ox - x) * rt,
                y + (oy - y) * rt,
                z,
            )
        )

    return expected


def _exercise_taper(topology, origin_xyz, ratio_range, label, expect_counts=True):
    original_type = Topology.TypeAsString(topology)
    original_counts = _counts(topology)
    original_points = _points(topology)

    result = Topology.Taper(
        topology,
        origin=_vertex(*origin_xyz),
        ratioRange=list(ratio_range),
        triangulate=False,
        mantissa=9,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology"), (
        f"Taper returned an invalid result for {label}; "
        f"origin={origin_xyz}, ratioRange={ratio_range}"
    )

    assert Topology.TypeAsString(result) == original_type, (
        f"Taper changed topology type for {label}: "
        f"{original_type} -> {Topology.TypeAsString(result)}; "
        f"origin={origin_xyz}, ratioRange={ratio_range}"
    )

    if expect_counts:
        assert _counts(result) == original_counts, (
            f"Taper changed subtopology counts for non-degenerate case {label}: "
            f"{original_counts} -> {_counts(result)}; "
            f"origin={origin_xyz}, ratioRange={ratio_range}"
        )

    expected_points = _expected_taper_points(
        original_points,
        origin_xyz,
        ratio_range,
    )
    _assert_point_sets_close(expected_points, _points(result))


@pytest.fixture(scope="session", autouse=True)
def _pythonocc_backend_only():
    backend = Core.Backend()
    assert backend is not None
    assert backend.__class__.__name__ == "PythonOCCBackend", (
        "test_TaperStress.py must be run with the PythonOCC backend. "
        f"Active backend: {backend.__class__.__name__}"
    )


@pytest.mark.parametrize("factory_name", NON_DEGENERATE_FACTORIES)
def test_taper_deterministic_non_degenerate(factory_name):
    topology = FACTORIES[factory_name]()

    origins = [
        (0.0, 0.0, 0.0),
        (1.25, -2.5, 0.75),
        (-3.1, 4.2, 2.7),
    ]

    ratio_ranges = [
        [0.0, 0.25],
        [0.1, 0.6],
        [0.75, 0.2],
        [-0.25, 0.5],
        [-0.8, -0.1],
        [-0.5, 0.85],
    ]

    index = 0
    for origin in origins:
        for ratio_range in ratio_ranges:
            _exercise_taper(
                topology,
                origin,
                ratio_range,
                label=f"{factory_name}/deterministic/{index}",
                expect_counts=True,
            )
            index += 1


@pytest.mark.parametrize("factory_name", NON_DEGENERATE_FACTORIES)
def test_taper_random_non_degenerate(factory_name):
    topology = FACTORIES[factory_name]()
    rng = random.Random(f"{SEED}:{factory_name}")

    for index in range(RANDOM_CASES):
        origin = (
            rng.uniform(-8.0, 8.0),
            rng.uniform(-8.0, 8.0),
            rng.uniform(-3.0, 3.0),
        )

        # Keep the random stress suite away from full collapse (ratio == 1)
        # so topology-count preservation is a valid invariant.
        ratio_range = [
            rng.uniform(-1.25, 0.9),
            rng.uniform(-1.25, 0.9),
        ]

        _exercise_taper(
            topology,
            origin,
            ratio_range,
            label=f"{factory_name}/random/{index}",
            expect_counts=True,
        )


@pytest.mark.parametrize("factory_name", ["wire", "face", "shell", "cell", "cellcomplex"])
def test_taper_endpoint_collapse(factory_name):
    """A ratio of 1 at one end is legal and may reduce lower-dimensional counts."""
    topology = FACTORIES[factory_name]()

    for ratio_range in ([0.0, 1.0], [1.0, 0.0], [-0.5, 1.0]):
        _exercise_taper(
            topology,
            origin_xyz=(0.3, -0.4, 1.7),
            ratio_range=ratio_range,
            label=f"{factory_name}/collapse/{ratio_range}",
            expect_counts=False,
        )


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_taper_zero_range_returns_input(factory_name):
    topology = FACTORIES[factory_name]()
    result = Topology.Taper(
        topology,
        ratioRange=[0, 0],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert result is topology


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_taper_fully_degenerate_range_returns_input(factory_name):
    topology = FACTORIES[factory_name]()
    result = Topology.Taper(
        topology,
        ratioRange=[1, 1],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert result is topology


def test_taper_zero_height_returns_input():
    """A horizontal topology has no Z interval over which to interpolate taper."""
    face = Face.Rectangle(
        origin=_vertex(0.0, 0.0, 2.0),
        width=3.0,
        length=2.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    result = Topology.Taper(
        face,
        ratioRange=[0.0, 0.75],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert result is face


@pytest.mark.parametrize("factory_name", ["face", "face_with_hole", "shell", "cell", "cellcomplex"])
def test_taper_triangulated_preserves_type_and_expected_vertices(factory_name):
    topology = FACTORIES[factory_name]()
    origin_xyz = (0.4, -0.7, 0.2)
    ratio_range = [-0.25, 0.55]

    triangulated = Topology.Triangulate(
        topology,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(triangulated, "Topology")

    expected_type = Topology.TypeAsString(triangulated)
    expected_points = _expected_taper_points(
        _points(triangulated),
        origin_xyz,
        ratio_range,
    )

    result = Topology.Taper(
        topology,
        origin=_vertex(*origin_xyz),
        ratioRange=ratio_range,
        triangulate=True,
        mantissa=9,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology")
    assert Topology.TypeAsString(result) == expected_type
    _assert_point_sets_close(expected_points, _points(result))

def test_taper_inclined_face_with_hole_requires_triangulation():
    """
    An inclined planar Face with a hole generally becomes non-planar when the
    taper ratio varies with Z. With triangulate=False, Taper must not fabricate
    a single Face through Fix/SelfMerge coercion.
    """
    topology = FACTORIES["face_with_hole"]()

    result = Topology.Taper(
        topology,
        origin=_vertex(0.0, 0.0, 0.0),
        ratioRange=[0.0, 0.25],
        triangulate=False,
        mantissa=9,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert result is None or not Topology.IsInstance(result, "Face")


def test_taper_inclined_face_with_hole_succeeds_when_triangulated():
    """
    The same non-rigid deformation is valid when the holed Face is triangulated
    first, because each triangle can remain an independently valid Face.
    """
    topology = FACTORIES["face_with_hole"]()
    origin_xyz = (0.0, 0.0, 0.0)
    ratio_range = [0.0, 0.25]

    triangulated = Topology.Triangulate(
        topology,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(triangulated, "Topology")

    expected_type = Topology.TypeAsString(triangulated)
    expected_points = _expected_taper_points(
        _points(triangulated),
        origin_xyz,
        ratio_range,
    )

    result = Topology.Taper(
        topology,
        origin=_vertex(*origin_xyz),
        ratioRange=ratio_range,
        triangulate=True,
        mantissa=9,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology")
    assert Topology.TypeAsString(result) == expected_type
    _assert_point_sets_close(expected_points, _points(result))

