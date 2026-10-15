"""Offset topology changes preserve all surviving planar regions."""
import math
import pytest
from topologicpy.Face import Face
from topologicpy.Wire import Wire
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary

DUMBBELL=[(0,0),(4,0),(4,1.5),(6,1.5),(6,0),(10,0),(10,4),(6,4),(6,2.5),(4,2.5),(4,4),(0,4)]

def wire(points,tilt=False):
    def xyz(x,y):return [10+x,20+.8*y,30+.6*y] if tilt else [x,y,0]
    return Wire.ByVertices([Vertex.ByCoordinates(*xyz(x,y)) for x,y in points],close=True,silent=True)

def rectangle(x,y,w,h):return [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]

def regions(result):
    return Topology.Faces(result,silent=True) or [] if result is not None else []

def area(result):return sum(Face.Area(f,mantissa=None) for f in regions(result))

@pytest.mark.parametrize('tilt',[False,True])
@pytest.mark.parametrize('invert',[False,True])
def test_narrow_neck_splits_into_faces_and_preserves_plane(tilt,invert,capsys):
    source=Face.ByWire(wire(DUMBBELL,tilt),silent=True)
    if invert:source=Face.Invert(source,silent=True)
    result=Face.ByOffset(source,offset=.6)
    assert Topology.IsInstance(result,'Cluster')
    faces=regions(result)
    assert len(faces)==2
    assert area(result)==pytest.approx(15.68,abs=1e-5)
    output=capsys.readouterr().out
    assert 'Warning' in output and '2 separate regions' in output
    n=Face.Normal(source)
    for face in faces:
        assert sum(a*b for a,b in zip(n,Face.Normal(face)))>.99
        if tilt:
            for vertex in Topology.Vertices(face):
                x,y,z=Vertex.Coordinates(vertex)
                assert z-30==pytest.approx(.75*(y-20),abs=1e-5)


def test_silent_and_face_dictionaries_on_every_region(capsys):
    source=Face.ByWire(wire(DUMBBELL),silent=True)
    source=Topology.SetDictionary(source,Dictionary.ByPythonDictionary({'label':'floor'}))
    result=Face.ByOffset(source,offset=.6,transferDictionaries=True,silent=True)
    assert len(regions(result))==2
    assert capsys.readouterr().out==''
    for item in [result]+regions(result):
        assert Dictionary.ValueAtKey(Topology.Dictionary(item),'label')=='floor'


def test_expanding_hole_can_split_face():
    source=Face.ByWires(wire(rectangle(0,0,10,10)),[wire(rectangle(4,2,2,6))],silent=True)
    result=Face.ByOffset(source,offset=1.1,silent=True)
    assert len(regions(result))==2
    assert area(result)==pytest.approx(28.08,abs=1e-5)
    assert all(not Face.InternalBoundaries(f) for f in regions(result))


def test_holes_merge_without_discarding_region():
    source=Face.ByWires(wire(rectangle(0,0,12,8)),[wire(rectangle(3,3,2,2)),wire(rectangle(7,3,2,2))],silent=True)
    result=Face.ByOffset(source,offset=1.1,silent=True)
    assert Topology.IsInstance(result,'Face')
    assert len(Face.InternalBoundaries(result))==1
    assert area(result)==pytest.approx(22.4,abs=1e-5)

@pytest.mark.parametrize('offset',[2,2.5,5])
def test_complete_erosion_does_not_resurrect_a_face(offset,capsys):
    source=Face.ByWire(wire(rectangle(0,0,4,4)),silent=True)
    assert Face.ByOffset(source,offset=offset,silent=True) is None
    assert capsys.readouterr().out==''


def test_outward_offset_consumes_small_holes():
    source=Face.ByWires(wire(rectangle(0,0,10,10)),[wire(rectangle(4.5,4.5,1,1))],silent=True)
    result=Face.ByOffset(source,offset=-.6,silent=True)
    assert Topology.IsInstance(result,'Face')
    assert len(Face.InternalBoundaries(result))==0
    assert area(result)==pytest.approx(125.44,abs=1e-5)


def test_single_region_and_reverse_remain_compatible():
    source=Face.ByWire(wire(rectangle(0,0,4,4)),silent=True)
    assert Topology.IsInstance(Face.ByOffset(source,offset=.5,silent=True),'Face')
    assert area(Face.ByOffset(source,offset=.5,silent=True))==pytest.approx(9)
    assert area(Face.ByOffset(source,offset=.5,reverse=True,silent=True))==pytest.approx(25)
    assert area(Face.ByOffset(source,offset=0,silent=True))==pytest.approx(16)

@pytest.mark.pythonocc_only
def test_exact_curved_hole_splits_region_without_tessellation(capsys):
    hole=Wire.Circle(origin=Vertex.ByCoordinates(5,3,0),radius=1.5,sides=1,polyline=False,silent=True)
    source=Face.ByWires(wire(rectangle(0,0,10,6)),[hole],silent=True)
    result=Face.ByOffset(source,offset=1.0)
    assert len(regions(result))==2
    expected=32-(6+12.5*math.asin(.8))
    assert area(result)==pytest.approx(expected,abs=1e-5)
    from topologicpy.Edge import Edge
    assert any(not Edge.IsLinear(e) for f in regions(result) for e in Topology.Edges(f))
    assert '2 separate regions' in capsys.readouterr().out


def test_variable_offsets_preserved_and_native_regions_can_split():
    boundary=wire(rectangle(0,0,10,10))
    for edge in Topology.Edges(boundary):
        Topology.SetDictionary(edge,Dictionary.ByPythonDictionary({'offset':1.0}))
    source=Face.ByWires(boundary,[wire(rectangle(4,2,2,6))],silent=True)
    result=Face.ByOffset(source,offset=1.1,silent=True)
    assert len(regions(result))==2
    assert area(result)==pytest.approx(30.4,abs=1e-5)


def test_by_offset_area_still_returns_single_region():
    source=Face.ByWire(wire(rectangle(0,0,4,4)),silent=True)
    result=Face.ByOffsetArea(source,area=9,defaultMinOffset=.5,defaultMaxOffset=.5,maxIterations=1,silent=True)
    assert Topology.IsInstance(result,'Face')
    assert area(result)==pytest.approx(9)


def test_by_offset_area_accepts_multiple_regions(capsys):
    source=Face.ByWire(wire(DUMBBELL),silent=True)
    result=Face.ByOffsetArea(source,area=15.68,defaultMinOffset=.6,defaultMaxOffset=.6,maxIterations=1,silent=False)
    assert Topology.IsInstance(result,'Cluster')
    assert len(regions(result))==2
    assert area(result)==pytest.approx(15.68,abs=1e-5)
    assert capsys.readouterr().out.count('2 separate regions')==1
