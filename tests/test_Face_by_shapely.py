import pytest
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

@pytest.mark.parametrize('reverse',[False,True])
def test_holes_coordinates_and_orientation(reverse):
    exterior=[(0,0),(10,0),(10,8),(0,8)]
    holes=[[(1,1),(3,1),(3,3),(1,3)],[(6,4),(8,4),(8,6),(6,6)]]
    if reverse:exterior.reverse();[h.reverse() for h in holes]
    polygon=Polygon(exterior,holes)
    face=Face.ByShapely(polygon,z=7.5,silent=True)
    assert Topology.IsInstance(face,'Face')
    assert Face.Area(face,mantissa=None)==pytest.approx(polygon.area,abs=1e-7)
    assert len(Face.InternalBoundaries(face))==2
    actual={(Vertex.X(v,mantissa=None),Vertex.Y(v,mantissa=None),Vertex.Z(v,mantissa=None)) for v in Topology.Vertices(face)}
    assert actual=={(x,y,7.5) for ring in [exterior,*holes] for x,y in ring}

def test_concave_and_shapely_z_projection():
    p=Polygon([(0,0,1),(4,0,2),(4,1,3),(1,1,4),(1,4,5),(0,4,6)])
    face=Face.ByShapely(p,silent=True)
    assert Face.Area(face,mantissa=None)==pytest.approx(7.)
    assert all(Vertex.Z(v)==0 for v in Topology.Vertices(face))

@pytest.mark.parametrize('value',[None,Polygon(),Polygon([(0,0),(2,2),(0,2),(2,0)]),MultiPolygon([Polygon([(0,0),(1,0),(0,1)])]),LineString([(0,0),(1,1)]),Point(0,0)])
def test_invalid_geometries(value):
    assert Face.ByShapely(value,silent=True) is None

@pytest.mark.parametrize('options',[{'z':float('nan')},{'z':float('inf')},{'tolerance':0},{'tolerance':-1},{'tolerance':float('nan')},{'tolerance':True}])
def test_invalid_options(options):
    assert Face.ByShapely(Polygon([(0,0),(1,0),(0,1)]),silent=True,**options) is None
