# Auto-generated PythonOCC backend parity test
# Copied from test_BVH.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.BVH."""

import math

import pytest

BVH_MODULE = pytest.importorskip("topologicpy.BVH")
AABB = BVH_MODULE.AABB
BVH = BVH_MODULE.BVH

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Topology = pytest.importorskip("topologicpy.Topology").Topology


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x=0, y=0, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _assert_topology(topology):
    assert Topology.IsInstance(topology, "Topology")


def _sample_vertices():
    return [_v(0, 0, 0), _v(10, 0, 0), _v(20, 0, 0)]


@pytest.fixture
def sample_bvh():
    bvh = BVH.ByTopologies(_sample_vertices(), maxLeafSize=1, tolerance=0.0, silent=True)
    assert isinstance(bvh, BVH)
    return bvh


def test_aabb_from_points_pad_extent_center_and_contains_point():
    box = AABB.from_points([(0, 1, 2), (4, 5, 6), (-1, 3, 8)], pad=0.5)

    assert box.minx == pytest.approx(-1.5)
    assert box.miny == pytest.approx(0.5)
    assert box.minz == pytest.approx(1.5)
    assert box.maxx == pytest.approx(4.5)
    assert box.maxy == pytest.approx(5.5)
    assert box.maxz == pytest.approx(8.5)
    assert box.extent() == pytest.approx((6.0, 5.0, 7.0))
    assert box.center() == pytest.approx((1.5, 3.0, 5.0))
    assert box.contains_point((0, 1, 2))
    assert not box.contains_point((5, 1, 2))


def test_aabb_from_empty_points_returns_origin_degenerate_box():
    box = AABB.from_points([])

    assert box.minx == pytest.approx(0)
    assert box.miny == pytest.approx(0)
    assert box.minz == pytest.approx(0)
    assert box.maxx == pytest.approx(0)
    assert box.maxy == pytest.approx(0)
    assert box.maxz == pytest.approx(0)
    assert box.extent() == pytest.approx((0, 0, 0))


def test_aabb_union_and_overlap_predicates():
    a = AABB(0, 0, 0, 1, 1, 1)
    b = AABB(0.5, 0.5, 0.5, 2, 2, 2)
    c = AABB(3, 3, 3, 4, 4, 4)

    union = AABB.union(a, b)

    assert union.minx == pytest.approx(0)
    assert union.miny == pytest.approx(0)
    assert union.minz == pytest.approx(0)
    assert union.maxx == pytest.approx(2)
    assert union.maxy == pytest.approx(2)
    assert union.maxz == pytest.approx(2)
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)


def test_aabb_ray_intersect_hits_and_misses():
    box = AABB(0, 0, 0, 1, 1, 1)

    hit, tmin, tmax = box.ray_intersect((-1, 0.5, 0.5), (1, 0, 0))
    assert hit is True
    assert tmin == pytest.approx(1)
    assert tmax == pytest.approx(2)

    miss, _, _ = box.ray_intersect((-1, 2, 0.5), (1, 0, 0))
    assert miss is False


def test_by_topologies_builds_tree_from_list_and_varargs():
    vertices = _sample_vertices()

    bvh_from_list = BVH.ByTopologies(vertices, maxLeafSize=1, tolerance=0.0, silent=True)
    bvh_from_args = BVH.ByTopologies(vertices[0], vertices[1], vertices[2], maxLeafSize=2, tolerance=0.0, silent=True)

    for bvh in [bvh_from_list, bvh_from_args]:
        assert isinstance(bvh, BVH)
        assert len(bvh.items) == 3
        assert len(bvh.bboxes) == 3
        assert len(bvh.centroids) == 3
        assert len(bvh.nodes) >= 1
        assert BVH.Depth(bvh) >= 1


def test_by_topologies_handles_invalid_inputs():
    assert BVH.ByTopologies([], silent=True) is None
    assert BVH.ByTopologies(None, "not a topology", silent=True) is None


def test_depth_and_query_aabb_on_empty_bvh():
    bvh = BVH()

    assert BVH.Depth(bvh) == 0
    assert BVH.QueryAABB(bvh, AABB(-1, -1, -1, 1, 1, 1)) == []


def test_query_aabb_returns_overlapping_item_indices(sample_bvh):
    near_origin = AABB(-1, -1, -1, 1, 1, 1)
    around_second = AABB(9, -1, -1, 11, 1, 1)
    broad = AABB(-1, -1, -1, 21, 1, 1)
    far = AABB(100, 100, 100, 101, 101, 101)

    assert set(BVH.QueryAABB(sample_bvh, near_origin)) == {0}
    assert set(BVH.QueryAABB(sample_bvh, around_second)) == {1}
    assert set(BVH.QueryAABB(sample_bvh, broad)) == {0, 1, 2}
    assert BVH.QueryAABB(sample_bvh, far) == []


def test_clashes_returns_candidate_topologies(sample_bvh):
    query_vertex = _v(10, 0, 0)
    hits = BVH.Clashes(sample_bvh, query_vertex, tolerance=0.0, silent=True)

    assert isinstance(hits, list)
    assert len(hits) == 1
    assert hits[0] is sample_bvh.items[1] or Topology.IsSame(hits[0], sample_bvh.items[1])


def test_clashes_accepts_nested_topology_inputs(sample_bvh):
    query_vertices = [[_v(0, 0, 0)], [_v(20, 0, 0)]]
    hits = BVH.Clashes(sample_bvh, query_vertices, tolerance=0.0, silent=True)

    assert isinstance(hits, list)
    assert len(hits) == 2
    assert {sample_bvh.items.index(hit) for hit in hits if hit in sample_bvh.items} == {0, 2}


def test_clashes_handles_invalid_inputs(sample_bvh):
    assert BVH.Clashes(sample_bvh, None, "invalid", silent=True) is None


def test_nearest_returns_topology_with_nearest_aabb_centroid(sample_bvh):
    nearest = BVH.Nearest(sample_bvh, _v(9.9, 0, 0), silent=True)

    assert nearest is sample_bvh.items[1] or Topology.IsSame(nearest, sample_bvh.items[1])


def test_nearest_handles_invalid_or_empty_inputs(sample_bvh):
    assert BVH.Nearest(None, _v(0, 0, 0), silent=True) is None
    assert BVH.Nearest(sample_bvh, None, silent=True) is None
    assert BVH.Nearest(BVH(), _v(0, 0, 0), silent=True) is None


def test_raycast_returns_candidate_indices_for_valid_ray(sample_bvh):
    hits = BVH.Raycast(sample_bvh, _v(-1, 0, 0), [1, 0, 0], silent=True)

    assert isinstance(hits, list)
    assert set(hits) == {0, 1, 2}


def test_raycast_handles_invalid_inputs(sample_bvh):
    assert BVH.Raycast(None, _v(0, 0, 0), [1, 0, 0], silent=True) is None
    assert BVH.Raycast(sample_bvh, None, [1, 0, 0], silent=True) is None
    assert BVH.Raycast(sample_bvh, _v(0, 0, 0), "not a vector", silent=True) is None


def test_bvh_can_index_edges_as_topologies():
    a = _v(0, 0, 0)
    b = _v(1, 0, 0)
    c = _v(10, 0, 0)
    d = _v(11, 0, 0)
    edge_a = Edge.ByVertices([a, b], silent=True)
    edge_b = Edge.ByVertices([c, d], silent=True)

    _assert_topology(edge_a)
    _assert_topology(edge_b)

    bvh = BVH.ByTopologies([edge_a, edge_b], maxLeafSize=1, tolerance=0.0, silent=True)
    hits = BVH.Clashes(bvh, Edge.ByVertices([_v(0.25, 0, 0), _v(0.75, 0, 0)], silent=True), tolerance=0.0, silent=True)

    assert isinstance(bvh, BVH)
    assert isinstance(hits, list)
    assert len(hits) == 1
    assert hits[0] is edge_a or Topology.IsSame(hits[0], edge_a)
