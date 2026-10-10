"""Physical wall coverage versus genuine isovist occlusion discontinuities."""
import math
import pytest
from topologicpy.Face import Face
from topologicpy.Wire import Wire
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary


def polygon(points):
    return Face.ByVertices([Vertex.ByCoordinates(x,y,0) for x,y in points], silent=True)


def metric(face, key):
    return Dictionary.ValueAtKey(Topology.Dictionary(face), key)


@pytest.mark.parametrize("point", [(0,0), (1,0.5), (-1.8,-1.5)])
def test_convex_room_has_no_occluded_boundary(point):
    room = Face.Rectangle(width=4, length=4)
    iso = Face.Isovist(room, Vertex.ByCoordinates(*point,0), metrics=True, silent=True)
    assert iso is not None
    assert metric(iso, "occlusivity") == 0
    assert metric(iso, "occluded_length") == 0
    assert metric(iso, "closed_perimeter") == metric(iso, "perimeter")
    assert all(not metric(edge,"occlusive") for edge in Topology.Edges(iso))


def test_concave_room_retains_genuine_occluded_boundary():
    room = polygon([(0,0),(6,0),(6,2),(2,2),(2,6),(0,6)])
    iso = Face.Isovist(room, Vertex.ByCoordinates(5,1,0), metrics=True, silent=True)
    # Sightline through (2,2) reaches x=0 at y=8/3.
    expected = math.sqrt(40)/3
    assert metric(iso,"occluded_length") == pytest.approx(expected, abs=5e-4)
    assert metric(iso,"occlusivity") == pytest.approx(expected/metric(iso,"perimeter"), abs=1e-4)
    flags = [metric(edge,"occlusive") for edge in Topology.Edges(iso)]
    assert any(flags) and not all(flags)


def test_obstacle_silhouette_has_two_occlusion_discontinuities():
    room = Face.Rectangle(width=10, length=10)
    obstacle = Wire.ByVertices([Vertex.ByCoordinates(x,y,0) for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]], close=True)
    iso = Face.Isovist(room, Vertex.ByCoordinates(-3,0,0), obstacles=[obstacle], metrics=True, silent=True)
    # Silhouette rays extend from (-1,+/-1) to (5,+/-4).
    assert metric(iso,"occluded_length") == pytest.approx(2*math.sqrt(45), abs=1e-3)
    assert 0 < metric(iso,"occlusivity") < 1
    assert sum(metric(edge,"occlusive") for edge in Topology.Edges(iso)) == 2


def test_physical_dictionary_transfer_preserves_occlusion_classification():
    room = Face.Rectangle(width=4,length=4)
    for edge in Topology.Edges(room):
        Topology.SetDictionary(edge, Dictionary.ByKeysValues(["wall_tag","occlusive"],["wall", True]))
    iso = Face.Isovist(room, Vertex.ByCoordinates(0,0,0), transferDictionaries=True, metrics=True, silent=True)
    assert metric(iso,"occlusivity") == 0
    assert all(metric(edge,"wall_tag") == "wall" for edge in Topology.Edges(iso))
    assert all(not metric(edge,"occlusive") for edge in Topology.Edges(iso))


def test_hole_boundary_is_treated_as_physical_obstacle():
    outer = Face.ExternalBoundary(Face.Rectangle(width=10,length=10))
    hole = Face.ExternalBoundary(Face.Rectangle(width=2,length=2))
    room = Face.ByWires(outer,[hole])
    iso = Face.Isovist(room,Vertex.ByCoordinates(-3,0,0),metrics=True,silent=True)
    assert metric(iso,"occluded_length") == pytest.approx(2*math.sqrt(45),abs=1e-3)
    assert sum(bool(metric(edge,"occlusive")) for edge in Topology.Edges(iso)) == 2


def test_flatten_roundtrip_and_tilted_isovist_preserve_wall_metadata():
    room = Face.Rectangle(width=4,length=4,origin=Vertex.ByCoordinates(10,20,30),direction=[1,1,1])
    for edge in Topology.Edges(room):
        Topology.SetDictionary(edge,Dictionary.ByKeysValues(["wall_tag"],["physical"]))
    origin = Topology.Centroid(room)
    normal = Face.Normal(room)
    flat = Topology.Flatten(room, origin=origin, direction=normal)
    restored = Topology.Unflatten(flat, origin=origin, direction=normal)
    assert all(metric(e,"wall_tag") == "physical" for e in Topology.Edges(restored))
    iso = Face.Isovist(room,origin,metrics=True,transferDictionaries=True,silent=True)
    assert iso is not None
    assert metric(iso,"occlusivity") == 0
    assert all(metric(e,"wall_tag") == "physical" for e in Topology.Edges(iso))
