import math
import os
import random

import pytest

from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology


TOLERANCE = 0.0001
SEED = 20260815
RANDOM_CASES = int(os.environ.get("TOPOLOGICPY_DISTANCE_STRESS_CASES", "100"))


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _edge(a, b):
    e = Edge.ByVertices(
        [_vertex(*a), _vertex(*b)],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(e, "Edge")
    return e


def _rectangle_face(width=4.0, length=4.0, z=0.0):
    wire = Wire.Rectangle(
        origin=_vertex(0.0, 0.0, z),
        width=width,
        length=length,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    face = Face.ByWire(
        wire,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


def _holed_face():
    outer = Wire.Rectangle(
        origin=_vertex(0.0, 0.0, 0.0),
        width=6.0,
        length=6.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    inner = Wire.Rectangle(
        origin=_vertex(0.0, 0.0, 0.0),
        width=2.0,
        length=2.0,
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


def _box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)):
    cx, cy, cz = center
    sx, sy, sz = size

    cell = Cell.Prism(
        origin=_vertex(cx, cy, cz),
        width=sx,
        length=sy,
        height=sz,
        uSides=1,
        vSides=1,
        wSides=1,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")
    return cell


def _point_segment_distance(p, a, b):
    px, py, pz = p
    ax, ay, az = a
    bx, by, bz = b

    ab = (bx - ax, by - ay, bz - az)
    ap = (px - ax, py - ay, pz - az)
    ab2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2

    if ab2 <= 1.0e-20:
        return math.dist(p, a)

    t = (
        ap[0] * ab[0]
        + ap[1] * ab[1]
        + ap[2] * ab[2]
    ) / ab2

    t = max(0.0, min(1.0, t))

    q = (
        ax + t * ab[0],
        ay + t * ab[1],
        az + t * ab[2],
    )

    return math.dist(p, q)


def _actual_aabb(cell):
    vertices = Topology.Vertices(cell, silent=True) or []
    assert vertices

    coordinates = [
        (
            float(Vertex.X(vertex, mantissa=12)),
            float(Vertex.Y(vertex, mantissa=12)),
            float(Vertex.Z(vertex, mantissa=12)),
        )
        for vertex in vertices
    ]

    return (
        (
            min(point[0] for point in coordinates),
            min(point[1] for point in coordinates),
            min(point[2] for point in coordinates),
        ),
        (
            max(point[0] for point in coordinates),
            max(point[1] for point in coordinates),
            max(point[2] for point in coordinates),
        ),
    )


def _aabb_distance_from_bounds(bounds_a, bounds_b):
    minimum_a, maximum_a = bounds_a
    minimum_b, maximum_b = bounds_b

    gaps = []

    for amin, amax, bmin, bmax in zip(
        minimum_a,
        maximum_a,
        minimum_b,
        maximum_b,
    ):
        if amax < bmin:
            gap = bmin - amax
        elif bmax < amin:
            gap = amin - bmax
        else:
            gap = 0.0

        gaps.append(gap)

    return math.sqrt(sum(g * g for g in gaps))



def test_vertex_vertex_3_4_5():
    a = _vertex(0.0, 0.0, 0.0)
    b = _vertex(3.0, 4.0, 0.0)

    assert Topology.ShortestDistance(a, b, silent=True) == 5.0


def test_vertex_edge_perpendicular_distance():
    p = _vertex(0.0, 2.0, 0.0)
    e = _edge((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert math.isclose(
        Topology.ShortestDistance(p, e, mantissa=9, silent=True),
        2.0,
        abs_tol=1.0e-9,
    )


def test_vertex_edge_endpoint_distance():
    p = _vertex(4.0, 3.0, 0.0)
    e = _edge((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))

    expected = math.sqrt(13.0)

    assert math.isclose(
        Topology.ShortestDistance(p, e, mantissa=9, silent=True),
        expected,
        abs_tol=1.0e-9,
    )


def test_crossing_edges_distance_zero():
    a = _edge((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    b = _edge((0.0, -3.0, 0.0), (0.0, 3.0, 0.0))

    assert Topology.ShortestDistance(a, b, silent=True) == 0.0


def test_skew_edges_known_distance():
    a = _edge((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    b = _edge((0.0, -2.0, 3.0), (0.0, 2.0, 3.0))

    assert math.isclose(
        Topology.ShortestDistance(a, b, mantissa=9, silent=True),
        3.0,
        abs_tol=1.0e-9,
    )


def test_vertex_face_normal_distance():
    face = _rectangle_face(width=6.0, length=4.0, z=0.0)
    p = _vertex(0.5, -0.5, 7.0)

    assert math.isclose(
        Topology.ShortestDistance(p, face, mantissa=9, silent=True),
        7.0,
        abs_tol=1.0e-9,
    )


def test_vertex_on_face_distance_zero():
    face = _rectangle_face(width=6.0, length=4.0, z=0.0)
    p = _vertex(1.0, 1.0, 0.0)

    assert Topology.ShortestDistance(p, face, silent=True) == 0.0


def test_vertex_in_face_hole_uses_trimmed_face():
    face = _holed_face()
    p = _vertex(0.0, 0.0, 0.0)

    # The point lies in the 2x2 hole, so its shortest distance to the
    # material portion of the coplanar Face is 1.0.
    assert math.isclose(
        Topology.ShortestDistance(p, face, mantissa=9, silent=True),
        1.0,
        abs_tol=1.0e-9,
    )


def test_vertex_over_face_hole_distance_to_hole_boundary():
    face = _holed_face()
    p = _vertex(0.0, 0.0, 3.0)

    # Closest material is a corner/edge of the square hole:
    # horizontal distance 1, vertical distance 3.
    expected = math.sqrt(10.0)

    assert math.isclose(
        Topology.ShortestDistance(p, face, mantissa=9, silent=True),
        expected,
        abs_tol=1.0e-9,
    )


def test_parallel_faces_known_distance():
    a = _rectangle_face(width=4.0, length=4.0, z=0.0)
    b = _rectangle_face(width=4.0, length=4.0, z=3.0)

    assert math.isclose(
        Topology.ShortestDistance(a, b, mantissa=9, silent=True),
        3.0,
        abs_tol=1.0e-9,
    )


def test_separated_cells_known_gap():
    a = _box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    b = _box(center=(0.0, 0.0, 5.0), size=(2.0, 2.0, 2.0))

    assert math.isclose(
        Topology.ShortestDistance(a, b, mantissa=9, silent=True),
        3.0,
        abs_tol=1.0e-9,
    )


def test_touching_cells_distance_zero():
    a = _box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    b = _box(center=(2.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))

    assert Topology.ShortestDistance(a, b, silent=True) == 0.0


def test_overlapping_cells_distance_zero():
    a = _box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    b = _box(center=(1.0, 0.5, 0.25), size=(2.0, 2.0, 2.0))

    assert Topology.ShortestDistance(a, b, silent=True) == 0.0


def test_contained_vertex_cell_distance_zero():
    cell = _box(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0))
    p = _vertex(0.25, -0.5, 0.75)

    assert Topology.ShortestDistance(p, cell, silent=True) == 0.0


def test_distance_is_symmetric():
    face = _holed_face()
    p = _vertex(0.0, 0.0, 3.0)

    ab = Topology.ShortestDistance(p, face, mantissa=9, silent=True)
    ba = Topology.ShortestDistance(face, p, mantissa=9, silent=True)

    assert math.isclose(ab, ba, abs_tol=1.0e-9)


def test_tolerance_normalizes_small_distance_to_zero():
    a = _vertex(0.0, 0.0, 0.0)
    b = _vertex(0.00005, 0.0, 0.0)

    assert Topology.ShortestDistance(
        a,
        b,
        tolerance=0.0001,
        silent=True,
    ) == 0.0


def test_distance_above_tolerance_is_not_zero():
    a = _vertex(0.0, 0.0, 0.0)
    b = _vertex(0.0002, 0.0, 0.0)

    assert Topology.ShortestDistance(
        a,
        b,
        mantissa=7,
        tolerance=0.0001,
        silent=True,
    ) == 0.0002


def test_mantissa_rounding():
    a = _vertex(0.0, 0.0, 0.0)
    b = _vertex(1.0, 1.0, 0.0)

    assert Topology.ShortestDistance(
        a,
        b,
        mantissa=4,
        silent=True,
    ) == round(math.sqrt(2.0), 4)


def test_cluster_to_vertex_distance():
    cluster = Cluster.ByTopologies(
        [
            _vertex(-5.0, 0.0, 0.0),
            _vertex(2.0, 0.0, 0.0),
        ],
        silent=True,
    )
    assert Topology.IsInstance(cluster, "Cluster")

    target = _vertex(5.0, 0.0, 0.0)

    assert math.isclose(
        Topology.ShortestDistance(cluster, target, mantissa=9, silent=True),
        3.0,
        abs_tol=1.0e-9,
    )


def test_cluster_to_cluster_distance():
    a = Cluster.ByTopologies(
        [
            _vertex(-10.0, 0.0, 0.0),
            _edge((-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ],
        silent=True,
    )
    b = Cluster.ByTopologies(
        [
            _vertex(8.0, 0.0, 0.0),
            _edge((2.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        ],
        silent=True,
    )

    assert Topology.IsInstance(a, "Cluster")
    assert Topology.IsInstance(b, "Cluster")

    # Closest pair: edge ending at -1 to edge beginning at +2.
    assert math.isclose(
        Topology.ShortestDistance(a, b, mantissa=9, silent=True),
        3.0,
        abs_tol=1.0e-9,
    )


def test_mixed_cluster_touching_distance_zero():
    cluster = Cluster.ByTopologies(
        [
            _edge((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            _vertex(10.0, 10.0, 10.0),
        ],
        silent=True,
    )
    p = _vertex(0.0, 0.0, 0.0)

    assert Topology.ShortestDistance(cluster, p, silent=True) == 0.0


@pytest.mark.parametrize("case_index", range(10))
def test_random_vertex_edge_oracle(case_index):
    rng = random.Random(f"{SEED}:vertex-edge:{case_index}")

    for _ in range(max(1, RANDOM_CASES // 10)):
        a = (
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
        )
        b = (
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
        )

        if math.dist(a, b) < 0.1:
            b = (b[0] + 1.0, b[1], b[2])

        p = (
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
        )

        expected = _point_segment_distance(p, a, b)

        actual = Topology.ShortestDistance(
            _vertex(*p),
            _edge(a, b),
            mantissa=9,
            silent=True,
        )

        assert actual is not None
        assert math.isclose(
            actual,
            expected,
            rel_tol=1.0e-8,
            abs_tol=1.0e-8,
        )


@pytest.mark.parametrize("case_index", range(5))
def test_random_axis_aligned_cells_oracle(case_index):
    rng = random.Random(f"{SEED}:cells:{case_index}")

    cases = max(1, min(RANDOM_CASES // 10, 10))

    for _ in range(cases):
        center_a = (
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
        )
        center_b = (
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
        )
        size_a = (
            rng.uniform(0.5, 3.0),
            rng.uniform(0.5, 3.0),
            rng.uniform(0.5, 3.0),
        )
        size_b = (
            rng.uniform(0.5, 3.0),
            rng.uniform(0.5, 3.0),
            rng.uniform(0.5, 3.0),
        )

        cell_a = _box(center_a, size_a)
        cell_b = _box(center_b, size_b)

        # Use the actual constructed Cell bounds rather than the nominal
        # center/size inputs. TopologicPy constructors may round generated
        # coordinates according to their mantissa policy, and the distance
        # oracle must evaluate the geometry that was actually created.
        expected = _aabb_distance_from_bounds(
            _actual_aabb(cell_a),
            _actual_aabb(cell_b),
        )

        actual = Topology.ShortestDistance(
            cell_a,
            cell_b,
            mantissa=8,
            silent=True,
        )

        assert actual is not None

        if expected <= TOLERANCE:
            expected = 0.0

        assert math.isclose(
            actual,
            expected,
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        )


def test_generic_fallback_only_when_distance_capability_absent(monkeypatch):
    a = _vertex(0.0, 0.0, 0.0)
    b = _vertex(3.0, 4.0, 0.0)

    original_has_attribute = Core.HasAttribute

    def _has_attribute(namespace, name):
        if namespace == "TopologyUtility" and name == "Distance":
            return False
        return original_has_attribute(namespace, name)

    monkeypatch.setattr(
        Core,
        "HasAttribute",
        staticmethod(_has_attribute),
    )

    assert Topology.ShortestDistance(
        a,
        b,
        mantissa=9,
        silent=True,
    ) == 5.0


@pytest.mark.parametrize(
    "case",
    [
        "none_a",
        "none_b",
        "string_a",
        "list_b",
    ],
)
def test_rejects_invalid_inputs(case):
    vertex = _vertex(0.0, 0.0, 0.0)

    if case == "none_a":
        invalid_a, invalid_b = None, vertex
    elif case == "none_b":
        invalid_a, invalid_b = vertex, None
    elif case == "string_a":
        invalid_a, invalid_b = "bad", vertex
    else:
        invalid_a, invalid_b = vertex, []

    assert Topology.ShortestDistance(
        invalid_a,
        invalid_b,
        silent=True,
    ) is None
