"""Geometry-first navigation and real IFC connectivity regressions."""
import copy
import math
from pathlib import Path

import pytest
from topologicpy.Navigation import Navigation, NavigationModel
from topologicpy.IFC import IFC
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Graph import Graph
from topologicpy.Grid import Grid
from topologicpy.Topology import Topology


def rectangle(x=0,y=0,w=4,h=4,z=0,slope=0):
    return Face.ByVertices([Vertex.ByCoordinates(a,b,z+slope*a) for a,b in [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]])


def route(graph, start, end):
    vertices=Graph.Vertices(graph)
    a=min(vertices,key=lambda v:math.dist(Vertex.Coordinates(v),start))
    b=min(vertices,key=lambda v:math.dist(Vertex.Coordinates(v),end))
    result=Graph.ShortestPath(graph,a,b,returnVertices=True,silent=True)
    return [vertices[i] for i in result[1]] if isinstance(result,tuple) else None


def test_single_face_standard_graph_tiles_and_routing():
    model=Navigation.ByFaces([rectangle()])
    assert isinstance(model,NavigationModel)
    graph=Navigation.Graph(model)
    assert len(Graph.Vertices(graph))==16
    assert len(Graph.Edges(graph))==24
    assert route(graph,[.5,.5,0],[3.5,3.5,0])
    tiles=Navigation.Tiles(model,graph)
    assert len(tiles)==16
    assert sum(Face.Area(f) for f in tiles)==pytest.approx(16)
    ids={Navigation._Metadata(v)['navigationID'] for v in Graph.Vertices(graph)}
    assert {Navigation._Metadata(f)['navigationID'] for f in tiles}==ids
    assert all(Navigation._Metadata(v)['source']=='0' for v in Graph.Vertices(graph))
    model.faces=[]
    assert len(Navigation.Tiles(model,graph))==16


@pytest.mark.parametrize('rise,gap,limit,expected',[(.17,0,.17,True),(.171,0,.17,False),(.171,0,.2,True),(0,.01,.17,False)])
def test_steps_without_gap_bridging(rise,gap,limit,expected):
    model=Navigation.ByFaces([rectangle(),rectangle(x=4+gap,z=rise)],storeyIDs=['a','a'])
    graph=Navigation.Graph(model,spacing=2,maxStepHeight=limit)
    assert bool(route(graph,[1,1,0],[7+gap,1,rise]))==expected


def test_obstacles_split_graph_and_eye_faces():
    model=Navigation.ByFaces([rectangle()],obstacles=[rectangle(x=1.8,w=.4)])
    graph=Navigation.Graph(model,spacing=1)
    assert not route(graph,[.5,.5,0],[3.5,.5,0])
    faces=Navigation.IsovistFaces(model,eyeHeight=1.7)
    assert sum(Face.Area(f) for f in faces)==pytest.approx(14.4)
    assert all(abs(p[2]-1.7)<1e-6 for f in faces for p in Navigation._Record(f)['outer'])


def test_projected_shared_grid_on_slopes_and_rotated_frame():
    flat=rectangle();slope=rectangle(x=4,slope=.05)
    grid=Grid.OnFaces([flat,slope],spacing=1,origin=[.25,.25])
    assert grid is not None
    points=[Vertex.Coordinates(v) for v in Topology.Vertices(grid)]
    assert all(abs(z-(.05*x if x>4.00001 else 0))<1e-5 for x,y,z in points if abs(x-4)>1e-5)
    model=Navigation.ByFaces([flat])
    a=Navigation.Graph(model,spacing=1,origin=[.2,.3],direction=[1,1])
    assert route(a,[.5,.5,0],[3.5,3.5,0])
    assert sum(Face.Area(f) for f in Navigation.Tiles(model,a))==pytest.approx(16,abs=1e-4)


def test_explicit_ramp_centreline_and_lift():
    floors=[rectangle(),rectangle(z=3)]
    ramp=Navigation.Ramp(centreline=[[2,2,0],[2,2,1.5],[2,2,3]],spacing=.7)
    model=Navigation.ByFaces(floors,connectors=[ramp])
    assert route(Navigation.Graph(model),[.5,.5,0],[.5,.5,3])
    lift=Navigation.Lift([[2,2,0],[2,2,3]],[[2.2,2,0],[2.2,2,3]])
    model=Navigation.ByFaces(floors,connectors=[lift])
    graph=Navigation.Graph(model)
    assert route(graph,[.5,.5,0],[.5,.5,3])
    assert len(Navigation.Tiles(model,graph))==32
    assert not any(Navigation._Metadata(v)['tileIDs'] for v in Graph.Vertices(graph) if Navigation._Metadata(v)['kind']=='lift')


@pytest.mark.parametrize('kwargs',[{'spacing':0},{'spacing':float('nan')},{'maxHorizontalGap':.01},{'maxStepHeight':-1},{'direction':[0,0]},{'mode':'unknown'}])
def test_invalid_graph_options(kwargs):
    with pytest.raises(ValueError):Navigation.Graph(Navigation.ByFaces([rectangle()]),**kwargs)


@pytest.fixture(scope='module')
def ifc_model():
    return Navigation.ByIFCPath(Path(__file__).parent/'fixtures'/'navigation-three-run.ifc')


def test_ifc_parses_once(monkeypatch):
    model=IFC.ModelByPath(str(Path(__file__).parent/'fixtures'/'navigation-three-run.ifc'))
    def forbidden(*args,**kwargs):raise AssertionError('IFC must not be reparsed')
    monkeypatch.setattr(IFC,'ModelByPath',forbidden)
    nav=IFC.NavigationModel(model)
    assert len(nav.storeys)==2
    assert nav.metadata['unit']=='m'


@pytest.mark.parametrize('kind',['stair','ramp','lift'])
def test_ifc_connectors_independently_route(ifc_model,kind):
    # Geometry handles are reused; caller-editable lists are independent.
    model=copy.copy(ifc_model);model._ifc=copy.copy(ifc_model._ifc)
    groups={c['group'] for c in model.connectors if c['type']==kind}
    model.connectors=[c for c in model.connectors if c['type']==kind or (c.get('kind')=='landing' and c['group'] in groups)]
    if kind!='lift':model._ifc={**model._ifc,'lifts':[],'doors':[]}
    graph=Navigation.Graph(model,spacing=1.5)
    points=route(graph,[.5,.5,0],[11,8,3])
    assert points
    kinds=[Navigation._Metadata(v)['kind'] for v in points]
    if kind=='ramp':
        assert kinds.count('landing')==2
        assert sum(math.dist(Vertex.Coordinates(a),Vertex.Coordinates(b)) for a,b in zip(points,points[1:]))>60
    assert len(Navigation.Tiles(model,graph))>0


def test_ifc_storey_isolation_and_isovist_heights(ifc_model):
    sid=ifc_model.storeys[1]['id']
    graph=Navigation.Graph(ifc_model,spacing=2,storey=sid)
    assert all(Navigation._Metadata(v)['storey']==sid for v in Graph.Vertices(graph))
    faces=Navigation.IsovistFaces(ifc_model)
    assert {round(Navigation._Record(f)['plane'][2],3) for f in faces}=={1.6,2.8,3.4,4.6,4.75}
    assert max(p[2] for f in ifc_model.faces for p in Navigation._Record(f)['outer'])<3.2


@pytest.mark.parametrize('spacing',[1.5,4])
def test_duplex_stairs(spacing):
    path=Path(__file__).parents[1]/'assets'/'IFC'/'Ifc2x3_Duplex_Architecture.ifc'
    if not path.exists():pytest.skip('Optional duplex IFC asset unavailable')
    model=Navigation.ByIFCPath(path)
    model._ifc={**model._ifc,'lifts':[],'doors':[]}
    groups={c['group'] for c in model.connectors if c['type']=='stair'}
    connected=0
    for group in groups:
        subset=copy.copy(model);subset.connectors=[c for c in model.connectors if c['group']==group]
        if len([c for c in subset.connectors if c.get('kind')=='tread'])<10:continue
        graph=Navigation.Graph(subset,spacing=spacing)
        vertices=Graph.Vertices(graph)
        stair=[v for v in vertices if Navigation._Metadata(v)['kind']=='tread']
        if not stair:continue
        low=min(stair,key=lambda v:Vertex.Coordinates(v)[2]);high=max(stair,key=lambda v:Vertex.Coordinates(v)[2])
        path=Graph.ShortestPath(graph,low,high,returnVertices=True,silent=True)
        assert path
        for elevation,endpoint in [(0,low),(3.1,high)]:
            candidates=[v for v in vertices if Navigation._Metadata(v)['kind']=='floor' and abs(Vertex.Coordinates(v)[2]-elevation)<.15]
            nearest=min(candidates,key=lambda v:math.dist(Vertex.Coordinates(v),Vertex.Coordinates(endpoint)))
            assert Graph.ShortestPath(graph,endpoint,nearest,silent=True)
        connected+=1
    assert connected==2


def test_holes_remain_obstacles_and_tiles_preserve_area():
    from topologicpy.Wire import Wire
    def wire(points):return Wire.ByVertices([Vertex.ByCoordinates(x,y,0) for x,y in points],close=True)
    face=Face.ByWires(wire([(0,0),(6,0),(6,6),(0,6)]),[wire([(2,2),(4,2),(4,4),(2,4)])])
    model=Navigation.ByFaces([face]);graph=Navigation.Graph(model)
    assert sum(Face.Area(f) for f in Navigation.Tiles(model,graph))==pytest.approx(32)
    assert len(Navigation._Record(Navigation.IsovistFaces(model)[0])['holes'])==1
    assert route(graph,[.5,.5,0],[5.5,5.5,0])


def test_tgraph_routing_analysis_and_tiles():
    from topologicpy.TGraph import TGraph
    model=Navigation.ByFaces([rectangle()])
    graph=Navigation.Graph(model,graphType='tgraph')
    assert isinstance(graph,TGraph)
    assert TGraph.ShortestPath(graph,0,15,edgeKey='Length',silent=True)
    assert len(Navigation.Tiles(model,graph))==16
    assert TGraph.DegreeCentrality(graph) is not None


def test_malformed_and_custom_connector_sampling():
    ramp=Navigation.Ramp([rectangle(w=12,h=1.5,slope=.05)],spacing=2)
    model=Navigation.ByFaces([rectangle()],connectors=[ramp])
    record=Navigation._Data(model)['connectors'][0]
    samples=Navigation._RampSamples(record,spacing=2)
    distances=[math.dist(a[0],b[0]) for a,b in zip(samples,samples[1:])]
    assert all(d==pytest.approx(2) for d in distances[:-1])
    empty=Navigation.Ramp([rectangle()])
    model.connectors=[empty]
    assert len(Graph.Vertices(Navigation.Graph(model)))==16
    with pytest.raises(ValueError):Navigation.Ramp(centreline=[[0,0,0],[1,0,0]],spacing=0)


def test_ifc_obstacle_selection_and_exclusions():
    path=Path(__file__).parent/'fixtures'/'navigation-three-run.ifc'
    model=Navigation.ByIFCPath(path,obstacleTypes=[],excludeTypes=['IfcRampFlight'])
    assert not model.obstacles
    assert not any(c['type']=='ramp' for c in model.connectors)


def test_imported_connectors_and_obstacles_are_editable(ifc_model):
    model=copy.copy(ifc_model)
    model.connectors=[]
    graph=Navigation.Graph(model,spacing=2)
    assert not route(graph,[.5,.5,0],[11,8,3])
    model.obstacles=[]
    assert not Navigation._UseSolidObstacles(model)
    assert sum(Face.Area(f) for f in Navigation.IsovistFaces(model))>0


def test_surveyed_coordinates():
    model=Navigation.ByFaces([rectangle(x=1000000,y=2000000,slope=.05)])
    graph=Navigation.Graph(model,spacing=2)
    assert len(Graph.Vertices(graph))==4
    assert len(Navigation.Tiles(model,graph))==4


def test_wire_centreline_ordering():
    from topologicpy.Wire import Wire
    points=[[0,0,0],[3,0,1],[3,4,2]]
    wire=Wire.ByVertices([Vertex.ByCoordinates(*p) for p in points],close=False)
    descriptor=Navigation.Ramp(centreline=wire)
    actual=descriptor['centreline']
    assert actual==points or actual==points[::-1]
    assert [3,0,1] in Navigation._SampleLine(actual,None,.15)


def test_all_ifc_tiles_reference_existing_vertices(ifc_model):
    graph=Navigation.Graph(ifc_model,spacing=1.5)
    ids={Navigation._Metadata(v)['navigationID'] for v in Graph.Vertices(graph)}
    tiles=Navigation.Tiles(ifc_model,graph)
    assert {Navigation._Metadata(f)['navigationID'] for f in tiles}.issubset(ids)
    assert Navigation.IsovistFaces(ifc_model,eyeHeight=2)


def test_maximum_slab_slope_selection():
    path=Path(__file__).parent/'fixtures'/'navigation-three-run.ifc'
    source=IFC.ModelByPath(str(path))
    for entity in IFC.Entities(source).values():
        if entity.type=='IFCRAMPFLIGHT':
            entity.type='IFCSLAB'
            args=list(entity.args);args[8]='FLOOR';entity.args=args
    flat=IFC.NavigationModel(source,maxSlope=0)
    sloped=IFC.NavigationModel(source,maxSlope=45)
    assert all(math.hypot(*Navigation._Record(f)['plane'][:2])<1e-6 for f in flat.faces)
    assert sum(math.hypot(*Navigation._Record(f)['plane'][:2])>.01 for f in sloped.faces)==3
    assert not any(c.get('kind')=='floor' for c in flat.connectors)


def test_overlapping_coplanar_slabs_form_one_grid():
    model=Navigation.ByFaces([rectangle(),rectangle(x=2)],storeyIDs=['a','a'])
    graph=Navigation.Graph(model)
    assert len(Graph.Vertices(graph))==24
    assert sum(Face.Area(f) for f in Navigation.Tiles(model,graph))==pytest.approx(24)
    assert all(Navigation._Metadata(v)['sourceIDs']==['0','1'] for v in Graph.Vertices(graph))


@pytest.mark.parametrize('graph_type',['graph','tgraph'])
def test_coincident_lift_doors_preserve_tile_associations(graph_type):
    model=Navigation.ByFaces([rectangle(),rectangle(z=3)],connectors=[
        Navigation.Lift([[.5,.5,0],[.5,.5,3]],[[1,.5,0],[1,.5,3]])])
    graph=Navigation.Graph(model,graphType=graph_type)
    tiles=Navigation.Tiles(model,graph)
    assert len(tiles)==32
    if graph_type=='graph':
        vertices=Graph.Vertices(graph)
        assert len(vertices)==34
        ids={Navigation._Metadata(v)['navigationID'] for v in vertices}
        assert {Navigation._Metadata(t)['navigationID'] for t in tiles}.issubset(ids)
        assert route(graph,[.5,.5,0],[.5,.5,3])
    else:
        from topologicpy.TGraph import TGraph
        assert len(TGraph.Vertices(graph))==34
        assert TGraph.ShortestPath(graph,0,16,silent=True)


@pytest.mark.parametrize('identifier',['42','1e3','null','true','"quoted"'])
def test_navigation_identifiers_survive_native_dictionary_round_trip(identifier):
    model=Navigation.ByFaces([rectangle()],storeyIDs=[identifier])
    model.connectors=[Navigation.Stair(rectangle(x=4,w=1,h=1),storey=identifier,source=identifier,group=identifier)]
    graph=Navigation.Graph(model,spacing=2,storey=identifier)
    metadata=[Navigation._Metadata(v) for v in Graph.Vertices(graph)]
    assert metadata
    assert all(d['storey']==identifier and d['storeyIDs']==[identifier] for d in metadata)
    stair=next(d for d in metadata if d['kind']=='tread')
    assert stair['source']==identifier
    assert stair['group']==identifier
    assert Navigation.Tiles(model,graph,storey=identifier)


@pytest.mark.parametrize('decoded',[False,True])
def test_navigation_tiles_accept_json_and_decoded_metadata(decoded):
    import json
    from topologicpy.TGraph import TGraph
    model=Navigation.ByFaces([rectangle()])
    graph=Navigation.Graph(model,spacing=2,graphType='tgraph')
    metadata=TGraph.Dictionary(graph)
    if decoded:metadata['navigationTiles']=json.loads(metadata['navigationTiles'])
    graph.SetDictionary(metadata)
    tiles=Navigation.Tiles(model,graph)
    assert len(tiles)==4
    assert sum(Face.Area(f) for f in tiles)==pytest.approx(16)
