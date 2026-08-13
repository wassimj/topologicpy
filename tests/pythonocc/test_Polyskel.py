# Auto-generated PythonOCC backend parity test
# Copied from test_Polyskel.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Polyskel."""

import math
import sys
import types

import pytest

from topologicpy import Polyskel as ps


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _xy(point):
    return (round(float(point.x), 6), round(float(point.y), 6))


def test_point2_vector_arithmetic_and_equality():
    a = ps.Point2(1, 2)
    b = ps.Point2(4, 6)

    assert tuple(a) == (1.0, 2.0)
    assert _xy(a + b) == (5.0, 8.0)
    assert _xy(b - a) == (3.0, 4.0)
    assert _xy(a * 2) == (2.0, 4.0)
    assert _xy(2 * a) == (2.0, 4.0)
    assert _xy(b / 2) == (2.0, 3.0)
    assert _xy(-a) == (-1.0, -2.0)
    assert abs(ps.Point2(3, 4)) == pytest.approx(5.0)
    assert a.dot(b) == pytest.approx(16.0)
    assert a.cross(b) == pytest.approx(-2.0)
    assert a.distance(b) == pytest.approx(5.0)
    assert ps.Point2(0, 0).normalized() == ps.Point2(0, 0)
    assert ps.Point2(3, 4).normalized().distance(ps.Point2(0, 0)) == pytest.approx(1.0)
    assert ps.Point2(1, 1) == ps.Point2(1 + ps.EPSILON / 2, 1 - ps.EPSILON / 2)
    assert hash(ps.Point2(1, 1)) == hash(ps.Point2(1, 1))


def test_line_ray_and_segment_intersections():
    x_axis = ps.Line2(ps.Point2(0, 0), ps.Point2(2, 0))
    diagonal = ps.Line2(ps.Point2(1, -1), ps.Point2(1, 1))
    inter = x_axis.intersect(diagonal)
    assert inter == ps.Point2(1, 0)
    assert x_axis.distance(ps.Point2(1, 3)) == pytest.approx(3.0)

    ray_a = ps.Ray2(ps.Point2(0, 0), ps.Point2(1, 1))
    ray_b = ps.Ray2(ps.Point2(1, 0), ps.Point2(-1, 1))
    assert ray_a.intersect(ray_b) == ps.Point2(0.5, 0.5)
    assert ps.Ray2(ps.Point2(0, 0), ps.Point2(1, 0)).intersect(
        ps.Ray2(ps.Point2(0, 1), ps.Point2(1, 0))
    ) is None

    seg_a = ps.LineSegment2(ps.Point2(0, 0), ps.Point2(2, 0))
    seg_b = ps.LineSegment2(ps.Point2(1, -1), ps.Point2(1, 1))
    seg_c = ps.LineSegment2(ps.Point2(3, -1), ps.Point2(3, 1))
    assert seg_a.intersect(seg_b) == ps.Point2(1, 0)
    assert seg_a.intersect(seg_c) is None
    assert seg_a.on_segment(ps.Point2(1, 0))
    assert not seg_a.on_segment(ps.Point2(3, 0))


def test_distance_normalize_and_bisector_with_coordinate_pairs():
    assert ps.distance((0, 0), (3, 4)) == pytest.approx(5.0)
    n = ps.normalize((3, 4))
    print("**** n is *****", n)
    assert isinstance(n, ps.Point2)
    assert _xy(n) == (0.6, 0.8)

    b = ps.bisector((1, 0), (0, 0), (0, 1))
    assert isinstance(b, ps.Point2)
    assert b.x == pytest.approx(math.sqrt(0.5))
    assert b.y == pytest.approx(math.sqrt(0.5))


def test_normalize_contour_removes_closure_duplicates_and_collinear_vertices():
    contour = [
        (0, 0),
        (1, 0),
        (2, 0),  # collinear duplicate direction on bottom edge
        (2, 1),
        (0, 1),
        (0, 0),  # closing point
    ]
    cleaned = ps._normalize_contour(contour)
    assert [_xy(p) for p in cleaned] == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    assert ps._signed_area(cleaned) == pytest.approx(2.0)
    assert list(ps._window([])) == []


def test_event_queue_orders_events_and_ignores_none():
    queue = ps._EventQueue()
    queue.put(None)
    assert queue.empty()
    queue.put(ps._EdgeEvent(3.0, ps.Point2(0, 0), None, None))
    queue.put(ps._EdgeEvent(1.0, ps.Point2(0, 0), None, None))
    queue.put_all([None, ps._SplitEvent(2.0, ps.Point2(0, 0), None, None)])

    assert queue.peek().distance == pytest.approx(1.0)
    assert queue.get().distance == pytest.approx(1.0)
    assert queue.get().distance == pytest.approx(2.0)
    assert queue.get().distance == pytest.approx(3.0)
    assert queue.empty()


def test_point_in_polygon_valid_skeleton_and_merge_sources():
    square = ps._normalize_contour([(0, 0), (2, 0), (2, 2), (0, 2)])
    assert ps._point_in_polygon(ps.Point2(1, 1), square)
    assert ps._point_in_polygon(ps.Point2(0, 1), square)
    assert not ps._point_in_polygon(ps.Point2(3, 1), square)

    skeleton = [
        ps.Subtree(ps.Point2(1, 1), 1.0, [ps.Point2(1, 0)]),
        ps.Subtree(ps.Point2(1 + ps.EPSILON / 2, 1), 0.5, [ps.Point2(2, 1)]),
    ]
    merged = ps._merge_sources(skeleton)
    assert len(merged) == 1
    assert merged[0].height == pytest.approx(0.5)
    assert len(merged[0].sinks) == 2
    assert ps._valid_skeleton(merged, square, [])
    assert not ps._valid_skeleton([ps.Subtree(ps.Point2(3, 3), 1, [])], square, [])


def test_fallback_convex_skeleton_square_is_deterministic():
    square = ps._normalize_contour([(0, 0), (2, 0), (2, 2), (0, 2)])
    skeleton = ps._fallback_convex_skeleton(square)
    assert len(skeleton) == 4
    assert {_xy(st.source) for st in skeleton} == {(1.0, 1.0)}
    assert sorted(_xy(st.sinks[0]) for st in skeleton) == [
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 2.0),
        (2.0, 1.0),
    ]
    assert all(st.height == pytest.approx(1.0) for st in skeleton)


def test_skeletonize_square_and_alias_return_subtrees_with_point_output():
    polygon = [(0, 0), (2, 0), (2, 2), (0, 2)]
    skeleton = ps.skeletonize(polygon, asTopologic=False, silent=True)
    alias = ps.Skeletonize(polygon, asTopologic=False, silent=True)

    assert skeleton == alias
    assert len(skeleton) >= 1
    assert all(isinstance(st, ps.Subtree) for st in skeleton)
    assert all(isinstance(st.source, ps.Point2) for st in skeleton)
    assert all(st.height >= 0 for st in skeleton)
    assert all(st.sinks for st in skeleton)
    assert {_xy(st.source) for st in skeleton} == {(1.0, 1.0)}


def test_skeletonize_accepts_point2_inputs_closed_contours_and_holes():
    outer = [ps.Point2(0, 0), ps.Point2(4, 0), ps.Point2(4, 4), ps.Point2(0, 4), ps.Point2(0, 0)]
    hole = [(1, 1), (1, 2), (2, 2), (2, 1)]
    skeleton = ps.skeletonize(outer, holes=[hole], asTopologic=False, silent=True)
    assert isinstance(skeleton, list)
    assert all(isinstance(st.source, ps.Point2) for st in skeleton)
    assert all(math.isfinite(st.source.x) and math.isfinite(st.source.y) for st in skeleton)


def test_skeletonize_invalid_inputs_and_fallback_disabled():
    assert ps.skeletonize([(0, 0), (1, 0)], asTopologic=False, silent=True) == []
    assert ps.skeletonize(None, asTopologic=False, silent=True) == []

    # A degenerate but three-point input should not raise. With fallback disabled
    # it can legitimately return an empty result.
    result = ps.skeletonize([(0, 0), (1, 0), (2, 0)], asTopologic=False, fallback=False, silent=True)
    assert isinstance(result, list)


def test_skeletonize_concave_polygon_stays_finite_and_inside_fallback_domain():
    polygon = [(0, 0), (4, 0), (4, 1), (2, 1), (2, 3), (0, 3)]
    outer = ps._normalize_contour(polygon)
    skeleton = ps.skeletonize(polygon, asTopologic=False, silent=True)
    assert skeleton
    assert all(math.isfinite(st.source.x) and math.isfinite(st.source.y) for st in skeleton)
    assert all(ps._point_in_polygon(st.source, outer, include_boundary=True) for st in skeleton)


def test_optional_topologic_vertex_bridge(monkeypatch):
    class FakeVertexObject:
        def __init__(self, x, y, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class FakeVertex:
        @staticmethod
        def ByCoordinates(x, y, z=0.0):
            return FakeVertexObject(x, y, z)

        @staticmethod
        def X(vertex):
            return vertex.x

        @staticmethod
        def Y(vertex):
            return vertex.y

    vertex_module = types.ModuleType("topologicpy.Vertex")
    vertex_module.Vertex = FakeVertex
    monkeypatch.setitem(sys.modules, "topologicpy.Vertex", vertex_module)

    topo_point = FakeVertex.ByCoordinates(3, 4, 0)
    assert isinstance(ps.normalize(topo_point), FakeVertexObject)
    assert FakeVertex.X(ps.normalize(topo_point)) == pytest.approx(0.6)
    assert ps.distance(topo_point, FakeVertex.ByCoordinates(0, 0, 0)) == pytest.approx(5.0)

    skeleton = ps.skeletonize(
        [
            FakeVertex.ByCoordinates(0, 0, 0),
            FakeVertex.ByCoordinates(2, 0, 0),
            FakeVertex.ByCoordinates(2, 2, 0),
            FakeVertex.ByCoordinates(0, 2, 0),
        ],
        asTopologic=True,
        silent=True,
    )
    assert skeleton
    assert all(isinstance(st.source, FakeVertexObject) for st in skeleton)
    assert all(isinstance(sink, FakeVertexObject) for st in skeleton for sink in st.sinks)


def test_public_exports_are_present():
    for name in [
        "EPSILON",
        "Point2",
        "Ray2",
        "Line2",
        "LineSegment2",
        "Subtree",
        "bisector",
        "distance",
        "normalize",
        "skeletonize",
        "Skeletonize",
    ]:
        assert name in ps.__all__
        assert hasattr(ps, name)
