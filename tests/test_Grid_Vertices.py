import math
import os
import pytest

from topologicpy.Edge import Edge
from topologicpy.Grid import Grid
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


def _edge(a, b):
    return Edge.ByVertices([Vertex.ByCoordinates(*a), Vertex.ByCoordinates(*b)])


def _coordinates(vertices):
    return {tuple(round(x, 6) for x in Vertex.Coordinates(v)) for v in vertices}


@pytest.fixture(params=[False, True], ids=["backend", "analytical"])
def analytical(request, monkeypatch):
    if request.param:
        monkeypatch.setattr(Topology, "_IsTopologicCoreBackend", staticmethod(lambda: True))


def test_unsplit_grid_endpoints_and_intersections(analytical):
    edges = [_edge((-3, y, 0), (3, y, 0)) for y in (-2, -1, 1, 2)]
    edges += [_edge((x, -3, 0), (x, 3, 0)) for x in (-2, -1, 1, 2)]
    result = Grid.Vertices(edges)
    assert len(result) == 32
    expected = {(x, y, 0) for x in (-2, -1, 1, 2) for y in (-2, -1, 1, 2)}
    expected |= {(-3, y, 0) for y in (-2, -1, 1, 2)}
    expected |= {(3, y, 0) for y in (-2, -1, 1, 2)}
    expected |= {(x, -3, 0) for x in (-2, -1, 1, 2)}
    expected |= {(x, 3, 0) for x in (-2, -1, 1, 2)}
    assert _coordinates(result) == expected


def test_multiway_crossing_is_unique_in_3d(analytical):
    edges = [_edge((-2, 0, -2), (2, 0, 2)),
             _edge((0, -2, -2), (0, 2, 2)),
             _edge((-2, -2, 0), (2, 2, 0))]
    result = Grid.Vertices(edges)
    assert len(result) == 7
    assert (0, 0, 0) in _coordinates(result)


def test_skew_edges_do_not_intersect(analytical):
    result = Grid.Vertices([_edge((-2, 0, 0), (2, 0, 0)),
                            _edge((0, -2, 1), (0, 2, 1))])
    assert len(result) == 4


def test_overlaps_and_duplicate_edges(analytical):
    edge = _edge((0, 0, 0), (4, 0, 0))
    result = Grid.Vertices([edge, edge, _edge((1, 0, 0), (3, 0, 0))])
    assert _coordinates(result) == {(0, 0, 0), (1, 0, 0), (3, 0, 0), (4, 0, 0)}
    assert len(result) == 4


def test_tolerance_deduplicates_across_bucket_boundaries(analytical):
    result = Grid.Vertices([_edge((0.000099, 0, 0), (1, 0, 0)),
                            _edge((0.000101, 0, 0), (0, 1, 0))])
    assert len(result) == 3


def test_topology_and_single_edge_inputs(analytical):
    from topologicpy.Cluster import Cluster
    edges = [_edge((-1, 0, 0), (1, 0, 0)), _edge((0, -1, 0), (0, 1, 0))]
    assert len(Grid.Vertices(Cluster.ByTopologies(edges))) == 5
    assert len(Grid.Vertices(edges[0])) == 2
    assert Grid.Vertices([]) == []
    assert Grid.Vertices(None, silent=True) is None
    assert Grid.Vertices([edges[0], None], silent=True) is None
    assert Grid.Vertices(edges, tolerance=math.inf, silent=True) is None


@pytest.mark.skipif("pythonocc" not in os.environ.get("TOPOLOGICPY_CORE_BACKEND", ""),
                    reason="Exact curved Edges require PythonOCC")
def test_exact_curved_edge_intersections_preserve_input():
    circle = Edge.Circle(radius=2, placement="center", silent=True)
    diameter = _edge((-3, 0, 0), (3, 0, 0))
    length = Edge.Length(circle)
    result = Grid.Vertices([circle, diameter])
    points = _coordinates(result)
    assert {(-2, 0, 0), (2, 0, 0), (-3, 0, 0), (3, 0, 0)} <= points
    assert len(points) == len(result)
    assert Edge.Length(circle) == pytest.approx(length)
    assert Edge.Length(circle) == pytest.approx(4 * math.pi)


@pytest.mark.skipif("pythonocc" not in os.environ.get("TOPOLOGICPY_CORE_BACKEND", ""),
                    reason="Exact curved surface grids require PythonOCC")
def test_grid_on_curved_face_finds_unsplit_crossings():
    from topologicpy.Cell import Cell
    from topologicpy.Face import Face

    cylinder = Cell.Cylinder(radius=2, height=5, polyhedron=False, silent=True)
    face = next(f for f in Topology.Faces(cylinder) if not Face.IsPlanar(f, silent=True))
    grid = Grid.OnFace(face, uDivisions=4, vDivisions=4)
    endpoints = Grid.Vertices([edge for edge in Topology.Edges(grid)][:1])
    assert endpoints
    points = Grid.Vertices(grid)
    original = _coordinates(Topology.Vertices(grid))
    assert original <= _coordinates(points)
    assert len(points) > len(original)
    for vertex in points:
        x, y, z = Vertex.Coordinates(vertex)
        assert x*x+y*y == pytest.approx(4, abs=0.0001)
    assert _coordinates(Topology.Vertices(grid)) == original
