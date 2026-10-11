import math
import pytest
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary

def value(t,k):
    return Dictionary.ValueAtKey(Topology.Dictionary(t),k)

def max_distance(iso,origin):
    p=Vertex.Coordinates(origin)
    return max(math.dist(p,Vertex.Coordinates(v)) for v in Topology.Vertices(iso))

def test_circle_limit_and_metrics():
    origin=Vertex.ByCoordinates(0,0,0)
    iso=Face.Isovist(Face.Rectangle(width=100,length=100),origin,radius=3,metrics=True)
    assert iso is not None
    assert max_distance(iso,origin)<=3.0001
    assert Face.Area(iso)==pytest.approx(64*.5*9*math.sin(2*math.pi/64),abs=.005)
    assert value(iso,'occluded_length')==0
    assert value(iso,'closed_perimeter')==pytest.approx(0,abs=.001)
    assert value(iso,'range_limited_length')==pytest.approx(value(iso,'perimeter'),abs=.001)
    assert all(value(e,'range_limited') for e in Topology.Edges(iso))

def test_none_and_large_radius_preserve_room():
    room=Face.Rectangle(width=4,length=4); p=Vertex.ByCoordinates(0,0,0)
    default=Face.Isovist(room,p,metrics=True)
    explicit=Face.Isovist(room,p,radius=None,metrics=True)
    large=Face.Isovist(room,p,radius=100,metrics=True)
    assert Face.Area(default)==Face.Area(explicit)
    assert Face.Area(large)==pytest.approx(Face.Area(default),abs=.001)
    assert value(large,'range_limited_length')==0
    assert value(large,'occluded_length')==0

@pytest.mark.parametrize('radius',[0,-1,float('nan'),float('inf'),True,'bad'])
def test_invalid_radius(radius):
    assert Face.Isovist(Face.Rectangle(),Vertex.ByCoordinates(0,0,0),radius=radius,silent=True) is None

@pytest.mark.parametrize('sides',[0,7,8.5,True])
def test_invalid_sides(sides):
    assert Face.Isovist(Face.Rectangle(),Vertex.ByCoordinates(0,0,0),radius=1,radiusSides=sides,silent=True) is None

def test_partial_fov_and_tilted_plane():
    room=Face.Rectangle(width=20,length=20,origin=Vertex.ByCoordinates(10,20,30),direction=[1,1,1])
    p=Topology.Centroid(room)
    iso=Face.Isovist(room,p,radius=2,fov=90)
    assert iso is not None
    assert max_distance(iso,p)<=2.0001
    assert Face.Area(iso)==pytest.approx(math.pi,rel=.01)

def test_wall_and_range_classification():
    room=Face.Rectangle(width=10,length=10)
    p=Vertex.ByCoordinates(4,0,0)
    iso=Face.Isovist(room,p,radius=2,metrics=True)
    assert iso is not None
    assert value(iso,'occluded_length')==0
    assert value(iso,'closed_perimeter')>0
    assert 0<value(iso,'range_limited_length')<value(iso,'perimeter')

def test_obstacle_blocks_within_radius():
    from topologicpy.Wire import Wire
    room=Face.Rectangle(width=30,length=30)
    obstacle=Face.ExternalBoundary(Face.Rectangle(width=2,length=2))
    p=Vertex.ByCoordinates(-3,0,0)
    iso=Face.Isovist(room,p,obstacles=[obstacle],radius=6,metrics=True)
    assert iso is not None
    assert max_distance(iso,p)<=6.0001
    assert value(iso,'occluded_length')>0
    assert value(iso,'range_limited_length')>0
    assert not Vertex.IsInternal(Vertex.ByCoordinates(2,0,0),iso)
