"""Clearance is a geometric constraint, independent of grid phase and display tiles."""
import math
import copy
from pathlib import Path
import pytest
from shapely.geometry import Polygon,Point,LineString
from shapely.ops import unary_union
from topologicpy.Face import Face
from topologicpy.Wire import Wire
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology
from topologicpy.Edge import Edge
from topologicpy.Grid import Grid
from topologicpy.TGraph import TGraph
from topologicpy.Navigation import Navigation

DUMBBELL=[(0,0),(4,0),(4,1.5),(6,1.5),(6,0),(10,0),(10,4),(6,4),(6,2.5),(4,2.5),(4,4),(0,4)]

def face(points,z=0,slope=0):
    return Face.ByVertices([Vertex.ByCoordinates(x,y,z+slope*x) for x,y in points],silent=True)

def rect(x=0,y=0,w=4,h=4,z=0,slope=0):return face([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],z,slope)

def route(graph,a,b):
    vertices=TGraph.Vertices(graph, asTopologic=True)
    start=min(vertices,key=lambda v:math.dist(Vertex.Coordinates(v),a))
    end=min(vertices,key=lambda v:math.dist(Vertex.Coordinates(v),b))
    return TGraph.ShortestPath(graph,start,end,silent=True)

def assert_grid_inside(grid,domain,tolerance=2e-5):
    assert grid is not None
    for edge in Topology.Edges(grid):
        for t in (0,.25,.5,.75,1):
            point=Vertex.Coordinates(Edge.VertexByParameter(edge,t))
            assert domain.buffer(tolerance).covers(Point(point[:2]))


def test_onface_clips_original_lattice_and_labels():
    source=rect()
    plain=Grid.OnFace(source,spacing=1,includeBoundary=True,silent=True)
    grid=Grid.OnFace(source,spacing=1,includeBoundary=True,clearance=.3,silent=True)
    assert_grid_inside(grid,Polygon([(0,0),(4,0),(4,4),(0,4)]).buffer(-.3,join_style='mitre'))
    plain_labels={(Navigation._Metadata(e)['grid_axis'],Navigation._Metadata(e)['grid_index']):Navigation._Metadata(e)['grid_parameter'] for e in Topology.Edges(plain)}
    for edge in Topology.Edges(grid):
        data=Navigation._Metadata(edge)
        assert data['grid_parameter']==plain_labels[data['grid_axis'],data['grid_index']]


def test_onface_keeps_multiple_offset_regions():
    grid=Grid.OnFace(face(DUMBBELL),spacing=.5,clearance=.6,silent=True)
    allowed=Polygon(DUMBBELL).buffer(-.6,join_style='mitre')
    assert_grid_inside(grid,allowed)
    points=[Vertex.Coordinates(v)[0] for v in Topology.Vertices(grid)]
    assert min(points)<3 and max(points)>7


def test_onface_clearance_in_tilted_face_plane():
    source=face([(0,0),(4,0),(4,4),(0,4)],slope=1)
    grid=Grid.OnFace(source,spacing=.5,clearance=.4,silent=True)
    boundaries=[Face.ExternalBoundary(source)]
    assert grid is not None
    for v in Topology.Vertices(grid):
        assert min(Vertex.Distance(v,w) for w in boundaries)>=.4-2e-5


def test_onfaces_merges_slab_seams_and_uses_xy_clearance():
    a=rect();b=rect(x=4)
    grid=Grid.OnFaces([a,b],spacing=1,origin=[.5,.5],clearance=.6)
    allowed=Polygon([(0,0),(8,0),(8,4),(0,4)]).buffer(-.6,join_style='mitre')
    assert_grid_inside(grid,allowed)
    assert any(abs(Vertex.Coordinates(v)[0]-4)<1e-5 for v in Topology.Vertices(grid))
    slope=rect(slope=1)
    projected=Grid.OnFaces([slope],spacing=.5,clearance=.4)
    assert_grid_inside(projected,Polygon([(0,0),(4,0),(4,4),(0,4)]).buffer(-.4,join_style='mitre'))


@pytest.mark.parametrize('mode',['navigation','visibility'])
def test_narrow_passage_closes_without_bridging(mode):
    model=Navigation.ByFaces([face(DUMBBELL)])
    assert route(Navigation.Graph(model,spacing=1,mode=mode,clearance=.4),[1,2,0],[9,2,0])
    graph=Navigation.Graph(model,spacing=1,mode=mode,clearance=.6)
    assert not route(graph,[1,2,0],[9,2,0])
    allowed=Polygon(DUMBBELL).buffer(-.6,join_style='mitre')
    for edge in TGraph.Edges(graph, asTopologic=True):
        line=LineString([Vertex.Coordinates(Edge.StartVertex(edge))[:2],Vertex.Coordinates(Edge.EndVertex(edge))[:2]])
        assert allowed.buffer(2e-5).covers(line)


def test_navigation_keeps_original_tiles_with_shifted_boundary_points():
    model=Navigation.ByFaces([rect()])
    graph=Navigation.Graph(model,spacing=1,clearance=.75)
    assert len(TGraph.Vertices(graph, asTopologic=True))==16
    assert sum(Face.Area(f) for f in Navigation.Tiles(model,graph))==pytest.approx(16)
    assert min(min(Vertex.Coordinates(v)[:2]) for v in TGraph.Vertices(graph, asTopologic=True))>=.75
    assert any(Vertex.Coordinates(v)[:2]==[1.5,1.5] for v in TGraph.Vertices(graph, asTopologic=True))
    assert Navigation._Metadata(graph)['clearance']==.75


def test_obstacle_holes_expand_before_edges_are_built():
    source=rect(w=8,h=8);obstacle=rect(x=3,y=3,w=2,h=2)
    model=Navigation.ByFaces([source],obstacles=[obstacle])
    graph=Navigation.Graph(model,spacing=1,mode='visibility',clearance=.6)
    outer=Polygon([(0,0),(8,0),(8,8),(0,8)])
    domain=outer.difference(Polygon([(3,3),(5,3),(5,5),(3,5)])).buffer(-.6,join_style='mitre')
    for edge in TGraph.Edges(graph, asTopologic=True):
        line=LineString([Vertex.Coordinates(Edge.StartVertex(edge))[:2],Vertex.Coordinates(Edge.EndVertex(edge))[:2]])
        assert domain.buffer(2e-5).covers(line)


def test_step_portals_stay_open_without_offsetting_each_slab():
    model=Navigation.ByFaces([rect(),rect(x=4,z=.17)],storeyIDs=['a','a'])
    assert route(Navigation.Graph(model,spacing=1,clearance=.6),[1,2,0],[7,2,.17])


@pytest.mark.parametrize('value',[-.1,float('nan'),float('inf')])
def test_invalid_clearance(value):
    f=rect()
    assert Grid.OnFace(f,clearance=value,silent=True) is None
    with pytest.raises(ValueError):Grid.OnFaces([f],clearance=value)
    with pytest.raises(ValueError):Navigation.Graph(Navigation.ByFaces([f]),clearance=value)


def test_complete_erosion_and_failed_offsets_never_fall_back(monkeypatch):
    f=rect()
    assert Grid.OnFace(f,clearance=3,silent=True) is None
    assert Grid.OnFaces([f],clearance=3) is None
    with pytest.raises(ValueError,match='clearance'):Navigation.Graph(Navigation.ByFaces([f]),clearance=3)
    monkeypatch.setattr(Face,'ByOffset',lambda *args,**kwargs:None)
    assert Grid.OnFace(f,clearance=.1,silent=True) is None
    assert Grid.OnFaces([f],clearance=.1) is None
    with pytest.raises(ValueError,match='clearance'):Navigation.Graph(Navigation.ByFaces([f]),clearance=.1)


@pytest.mark.pythonocc_only
def test_curved_surface_clearance_is_explicitly_unsupported():
    from topologicpy.Cell import Cell
    cylinder=Cell.Cylinder(radius=2,height=4,polyhedron=False,silent=True)
    surface=next(f for f in Topology.Faces(cylinder) if Face.IsPlanar(f,silent=True) is False)
    assert Grid.OnFace(surface,clearance=.2,silent=True) is None


@pytest.fixture(scope='module')
def demo():return Navigation.ByIFCPath(Path(__file__).parent/'fixtures'/'navigation-three-run.ifc')


@pytest.mark.parametrize('kind',['stair','ramp','lift'])
def test_building_connections_remain_with_body_clearance(demo,kind):
    model=copy.copy(demo)
    groups={c.get('group') for c in model.connectors if c['type']==kind}
    model.connectors=[c for c in model.connectors if c['type']==kind or (c.get('kind')=='landing' and c.get('group') in groups)]
    graph=Navigation.Graph(model,spacing=1.5,clearance=.25)
    assert route(graph,[.5,.5,0],[11,8,3])


def test_zero_clearance_is_backward_compatible(demo):
    a=TGraph.MeshData(Navigation.Graph(demo,spacing=2))
    b=TGraph.MeshData(Navigation.Graph(demo,spacing=2,clearance=0))
    # Core does not guarantee edge enumeration order between equivalent graphs.
    # Compare the full geometry/connectivity, retaining duplicate multiplicity.
    def canonical(data):
        vertices=[tuple(p) for p in data['vertices']]
        edges=[tuple(sorted((vertices[i],vertices[j]))) for i,j in data['edges']]
        return sorted(vertices),sorted(edges)
    assert canonical(a)==canonical(b)

@pytest.mark.parametrize('kind',['stair','ramp','explicit_ramp'])
def test_narrow_circulation_width_closes_without_shrinking_treads(kind):
    if kind=='stair':
        floors=[rect(x=-4),rect(x=3,z=3)]
        treads=[rect(x=i/6,y=1.8,w=1/6,h=.4,z=(i+1)/6) for i in range(18)]
        connector=Navigation.Stair(treads)
        upper=[5,2,3]
    else:
        floors=[rect(x=-4),rect(x=60,z=3)]
        connector=Navigation.Ramp([rect(x=0,y=1.8,w=60,h=.4,slope=.05)],centreline=[[0,2,0],[60,2,3]] if kind=='explicit_ramp' else None)
        upper=[62,2,3]
    model=Navigation.ByFaces(floors,connectors=[connector])
    assert route(Navigation.Graph(model,spacing=1,clearance=.1),[-2,2,0],upper)
    graph=Navigation.Graph(model,spacing=1,clearance=.25)
    assert not route(graph,[-2,2,0],upper)
    assert not any(Navigation._Metadata(v)['kind'] in ('tread','ramp') for v in TGraph.Vertices(graph, asTopologic=True))


def test_tgraph_clearance_and_tiles():
    model=Navigation.ByFaces([rect()])
    graph=Navigation.Graph(model,clearance=.75)
    assert isinstance(graph,TGraph)
    assert len(TGraph.Vertices(graph, asTopologic=True))==16
    assert sum(Face.Area(f) for f in Navigation.Tiles(model,graph))==pytest.approx(16)
    assert TGraph.Dictionary(graph)['clearance']==.75


@pytest.mark.pythonocc_only
def test_planar_curved_boundary_clearance_keeps_exact_arcs():
    circle=Wire.Circle(radius=2,sides=1,polyline=False,silent=True)
    source=Face.ByWire(circle,silent=True)
    grid=Grid.OnFace(source,spacing=.5,clearance=.3,silent=True)
    assert grid is not None
    assert all(Vertex.Distance(v,circle)>=.3-2e-5 for v in Topology.Vertices(grid))


def test_duplex_stairs_keep_a_clearance_route():
    path=Path(__file__).parents[1]/'assets'/'IFC'/'Ifc2x3_Duplex_Architecture.ifc'
    if not path.exists():pytest.skip('Optional duplex IFC unavailable')
    model=Navigation.ByIFCPath(path)
    graph=Navigation.Graph(model,spacing=1.5,clearance=.2)
    vertices=TGraph.Vertices(graph, asTopologic=True)
    groups={Navigation._Metadata(v)['group'] for v in vertices if Navigation._Metadata(v)['kind']=='tread'}
    assert len(groups)==2
    for group in groups:
        treads=[v for v in vertices if Navigation._Metadata(v)['group']==group and Navigation._Metadata(v)['kind']=='tread']
        low=min(treads,key=lambda v:Vertex.Coordinates(v)[2]);high=max(treads,key=lambda v:Vertex.Coordinates(v)[2])
        assert TGraph.ShortestPath(graph,low,high,silent=True)
        for z,endpoint in [(0,low),(3.1,high)]:
            candidates=[v for v in vertices if Navigation._Metadata(v)['kind']=='floor' and abs(Vertex.Coordinates(v)[2]-z)<.15]
            nearest=min(candidates,key=lambda v:math.dist(Vertex.Coordinates(v),Vertex.Coordinates(endpoint)))
            assert TGraph.ShortestPath(graph,endpoint,nearest,silent=True)


@pytest.mark.pythonocc_only
def test_consumed_circular_face_has_no_clearance_grid():
    circle=Wire.Circle(radius=2,sides=1,polyline=False,silent=True)
    source=Face.ByWire(circle,silent=True)
    assert Grid.OnFace(source,spacing=.5,clearance=3,silent=True) is None


@pytest.mark.pythonocc_only
def test_native_wire_respects_small_explicit_tolerance():
    points=[(0,0),(1,0),(1,.00001),(1,1),(0,1)]
    loop=Wire.ByVertices([Vertex.ByCoordinates(x,y,0) for x,y in points],close=True,tolerance=1e-7,silent=True)
    assert loop is not None and Wire.IsClosed(loop)
    assert len(Topology.Edges(loop))==5
    assert min(Edge.Length(e) for e in Topology.Edges(loop))<.0001


@pytest.mark.pythonocc_only
def test_native_closed_offset_curve_repairs_seam_without_changing_geometry():
    from topologicpy.Core import Core
    circle=Wire.Circle(radius=2,sides=1,polyline=False,silent=True)
    source=Face.ByWire(circle,silent=True)
    inset=Face.ByOffset(source,offset=.3,silent=True)
    assert Core.FaceUtility.IsValid(inset)
    assert Face.Area(inset,mantissa=None)==pytest.approx(math.pi*1.7**2,abs=1e-6)
    assert Face.Area(source,mantissa=None)==pytest.approx(math.pi*4,abs=1e-6)
    assert all(not Edge.IsLinear(e) for e in Topology.Edges(inset))


def test_lift_door_portal_uses_footprint_and_respects_opening_width():
    floors=[rect(),rect(z=3)]
    lift=Navigation.Lift([[4,2,0],[4,2,3]],[[5,2,0],[5,2,3]],footprint=rect(x=4,y=1,w=2,h=2))
    obstacles=[]
    for z in [0,3]:
        obstacles.extend([rect(x=3.9,y=0,w=.2,h=1.6,z=z),rect(x=3.9,y=2.4,w=.2,h=1.6,z=z)])
    model=Navigation.ByFaces(floors,obstacles=obstacles,connectors=[lift])
    assert route(Navigation.Graph(model,spacing=1,clearance=.25),[1,2,0],[1,2,3])
    assert not route(Navigation.Graph(model,spacing=1,clearance=.45),[1,2,0],[1,2,3])
