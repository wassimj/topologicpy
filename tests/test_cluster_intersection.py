"""Regression coverage for intersections with shapeless Cluster operands."""
import pytest

from topologicpy.Cluster import Cluster
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


def rectangle(min_x=-70, max_x=-60, min_y=-20, max_y=5):
    return Face.ByWire(Wire.ByVertices([
        Vertex.ByCoordinates(min_x, min_y),
        Vertex.ByCoordinates(max_x, min_y),
        Vertex.ByCoordinates(max_x, max_y),
        Vertex.ByCoordinates(min_x, max_y),
    ], close=True))


def crossing_edge():
    return Edge.ByStartVertexEndVertex(
        Vertex.ByCoordinates(-120, -10, 0),
        Vertex.ByCoordinates(-20, -10, 0),
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_edge_face_cluster_intersection(reverse, nested):
    face, edge = rectangle(), crossing_edge()
    obstacle = Cluster.ByTopologies([face])
    if nested:
        obstacle = Cluster.ByTopologies([obstacle])
    direct = Topology.Intersect(edge, face)
    result = Topology.Intersect(obstacle, edge) if reverse else Topology.Intersect(edge, obstacle)
    assert direct is not None
    assert result is not None
    assert sum(Edge.Length(e) for e in Topology.Edges(result)) == pytest.approx(10)
    assert sorted(Vertex.X(v) for v in Topology.Vertices(result)) == pytest.approx([-70, -60])


@pytest.mark.parametrize("wrap_edge", [False, True])
def test_cluster_intersection_preserves_disconnected_hits(wrap_edge):
    edge = crossing_edge()
    if wrap_edge:
        edge = Cluster.ByTopologies([edge])
    obstacles = Cluster.ByTopologies([rectangle(), rectangle(-50, -40), rectangle(0, 10)])
    result = Topology.Intersect(edge, obstacles)
    assert result is not None
    assert sorted(Edge.Length(e) for e in Topology.Edges(result)) == pytest.approx([10, 10])
    assert sorted(Vertex.X(v) for v in Topology.Vertices(result)) == pytest.approx([-70, -60, -50, -40])


def test_disjoint_cluster_intersection_is_none():
    assert Topology.Intersect(crossing_edge(), Cluster.ByTopologies([rectangle(0, 10)])) is None


def test_wire_face_cluster_intersection():
    wire = Wire.ByVertices([
        Vertex.ByCoordinates(-120, -10),
        Vertex.ByCoordinates(-20, -10),
    ], close=False)
    result = Topology.Intersect(wire, Cluster.ByTopologies([rectangle()]))
    assert result is not None
    assert sum(Edge.Length(e) for e in Topology.Edges(result)) == pytest.approx(10)


@pytest.mark.parametrize("cluster_obstacle", [False, True])
def test_straighten_avoids_obstacles(cluster_obstacle):
    obstacle = rectangle()
    if cluster_obstacle:
        obstacle = Cluster.ByTopologies([obstacle])
    route = Wire.ByVertices([
        Vertex.ByCoordinates(-120, -10),
        Vertex.ByCoordinates(-80, 10),
        Vertex.ByCoordinates(-50, 10),
        Vertex.ByCoordinates(-20, -10),
    ], close=False)
    host = rectangle(-130, -10, -30, 30)
    direct = Wire.Straighten(route, host)
    assert len(Topology.Edges(direct)) == 1
    result = Wire.Straighten(route, host, obstacles=[obstacle])
    assert result is not None
    assert len(Topology.Edges(result)) > 1
    for edge in Topology.Edges(result):
        assert Topology.Intersect(edge, rectangle()) is None
