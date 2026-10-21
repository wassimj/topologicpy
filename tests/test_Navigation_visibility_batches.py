"""Navigation must use TGraph visibility and retain physical floor constraints."""
import math
import numpy as np
import pytest
from shapely.geometry import Point, LineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from topologicpy.Navigation import Navigation
from topologicpy.Face import Face


def reference(nodes, footprints, planes, domains, count, rise, eps):
    """Scalar pre-acceleration predicate, kept as an independent oracle."""
    tree=STRtree(footprints);result=set()
    for i in range(count):
        for j in range(i+1,count):
            if nodes[i]['storey']!=nodes[j]['storey']:continue
            line=LineString([nodes[i]['point'][:2],nodes[j]['point'][:2]])
            if line.length<eps or not domains[nodes[i]['storey']].buffer(eps*.1).covers(line):continue
            traversed=[int(k) for k in tree.query(line) if footprints[int(k)].intersection(line).length>eps]
            if not traversed:continue
            intervals=[]
            for k in traversed:
                part=footprints[k].intersection(line)
                for g in [part] if part.geom_type=='LineString' else getattr(part,'geoms',[]):
                    if g.geom_type!='LineString':continue
                    ts=sorted(line.project(Point(e))/line.length for e in [g.coords[0],g.coords[-1]])
                    intervals.append((*ts,k))
            intervals.sort();valid=True
            for left,right in zip(intervals,intervals[1:]):
                at=line.interpolate((left[1]+right[0])/2,normalized=True)
                a,b=planes[left[2]],planes[right[2]]
                if abs((a[0]-b[0])*at.x+(a[1]-b[1])*at.y+a[2]-b[2])>rise+eps:
                    valid=False;break
            if valid:result.add((i,j))
    return result


def conservative_reference(*args):
    nodes,shapes,planes,domains,count,rise,eps=args
    result=set()
    for i,j in reference(*args):
        line=LineString([nodes[i]['point'][:2],nodes[j]['point'][:2]])
        region=domains[nodes[i]['storey']].simplify(1e-6,preserve_topology=True)
        polygons=[region] if region.geom_type=='Polygon' else list(region.geoms)
        corners=[Point(p) for poly in polygons for ring in [poly.exterior,*poly.interiors] for p in list(ring.coords)[:-1]]
        if any(line.distance(p)<=eps and eps<line.project(p)<line.length-eps for p in corners):continue
        result.add((i,j))
    return result


def scene(kind):
    nodes=[];footprints=[];planes=[];domains={}
    for level in range(2 if kind=='storeys' else 1):
        pieces=[]
        for x in range(5):
            for y in range(4):
                if kind in ('hole','disconnected') and x==2 and (kind=='disconnected' or y in (1,2)):continue
                poly=box(x,y,x+1,y+1);slope=.1 if kind=='slope' else 0
                height=(.4 if x>=3 else 0) if kind=='step' else level*3
                planes.append([slope,0,height]);footprints.append(poly);pieces.append(poly)
                nodes.append({'point':[x+.5,y+.5,slope*(x+.5)+height],'storey':str(level)})
        domains[str(level)]=unary_union(pieces)
    return nodes,footprints,planes,domains,len(nodes),.17,1e-5


@pytest.mark.parametrize('kind',['open','hole','disconnected','slope','step','storeys'])
@pytest.mark.parametrize('batch',[1,7,4096])
def test_shared_matches_conservative_reference(kind,batch):
    args=scene(kind)
    assert set(Navigation._VisibilityLinks(*args,batchSize=batch))==conservative_reference(*args)


def test_corner_tangency_uses_tgraph_conservative_convention():
    # The segment touches the missing upper-right region only at (0,0).
    support=box(-1,-1,1,1).difference(box(0,0,1,1))
    nodes=[{'point':[-.5,.5,0],'storey':'0'},
           {'point':[.5,-.5,0],'storey':'0'}]
    shapes=[box(-1,0,0,1),box(0,-1,1,0)]
    args=(nodes,shapes,[[0,0,0],[0,0,0]],{'0':support},2,.17,1e-5)
    assert reference(*args)=={(0,1)}
    assert set(Navigation._VisibilityLinks(*args))==set()


def test_candidate_limit_fails_before_geometry_allocation(monkeypatch):
    import shapely
    args=scene('open')
    monkeypatch.setattr(shapely,'linestrings',lambda *a,**k:pytest.fail('allocated geometry before limit check'))
    with pytest.raises(ValueError,match='candidate pairs'):
        list(Navigation._VisibilityLinks(*args,maxPairs=2))


def test_public_visibility_edge_budget():
    model=Navigation.ByFaces([Face.Rectangle(width=4,length=4)])
    with pytest.raises(ValueError,match='edge limit'):
        Navigation.Graph(model,mode='visibility',maxVisibilityEdges=2)


@pytest.mark.parametrize('option',['visibilityBatchSize','maxVisibilityPairs','maxVisibilityEdges'])
@pytest.mark.parametrize('value',[0,-1,True,1.5,float('nan')])
def test_visibility_budget_validation(option,value):
    model=Navigation.ByFaces([Face.Rectangle(width=2,length=2)])
    with pytest.raises(ValueError,match='positive integer'):
        Navigation.Graph(model,**{option:value})


def test_navigation_calls_shared_public_engine(monkeypatch):
    from topologicpy.TGraph import TGraph
    real=TGraph.VisibilityGraph;calls=[]
    def shared(*a,**kw):
        calls.append(kw);return real(*a,**kw)
    monkeypatch.setattr(TGraph,'VisibilityGraph',shared)
    model=Navigation.ByFaces([Face.Rectangle(width=4,length=4)])
    graph=Navigation.Graph(model,mode='visibility')
    assert calls and TGraph.Size(graph)>0
    assert calls[0]['trim'] and not calls[0]['includeBoundaryVertices']
    assert not calls[0]['storeEdgeRepresentations']


@pytest.mark.parametrize('directed',[False,True])
def test_shared_engine_budgets(directed):
    from topologicpy.TGraph import TGraph
    from topologicpy.Vertex import Vertex
    host=Face.Rectangle(width=4,length=4)
    points=[Vertex.ByCoordinates(x,y,0) for x,y in [(-1,-1),(1,-1),(-1,1),(1,1)]]
    options=dict(vertices=points,includeBoundaryVertices=False,bidirectional=not directed,storeEdgeRepresentations=False,ontology=False)
    with pytest.raises(ValueError,match='candidate pairs'):
        TGraph.VisibilityGraph(host,maxCandidatePairs=2,**options)
    with pytest.raises(ValueError,match='edge limit'):
        TGraph.VisibilityGraph(host,maxEdges=1,**options)


def test_flat_navigation_edges_match_tgraph_visibility():
    from topologicpy.TGraph import TGraph
    from topologicpy.Vertex import Vertex
    host=Navigation._Face(Navigation._Surface(box(0,0,6,6).difference(box(2,2,4,4)),[0,0,0],'0'))
    navigation=Navigation.Graph(Navigation.ByFaces([host]),mode='visibility')
    points=TGraph.MeshData(navigation)['vertices']
    direct=TGraph.VisibilityGraph(host,vertices=[Vertex.ByCoordinates(*p) for p in points],includeBoundaryVertices=False,trim=True,storeEdgeRepresentations=False,ontology=False)
    def edges(graph):return {tuple(sorted((e['src'],e['dst']))) for e in TGraph.Edges(graph)}
    assert edges(navigation)==edges(direct)


def test_navigation_compiled_shared_engine_matches_python():
    from topologicpy.TGraph import TGraph
    model=Navigation.ByFaces([Face.Rectangle(width=4,length=4)])
    a=Navigation.Graph(model,mode='visibility',useNumba=False)
    b=Navigation.Graph(model,mode='visibility',useNumba=True)
    def edges(graph):return {tuple(sorted((e['src'],e['dst']))) for e in TGraph.Edges(graph)}
    assert edges(a)==edges(b)


@pytest.mark.parametrize('option',['maxCandidatePairs','maxEdges'])
@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_shared_resource_limit_validation(option,value):
    from topologicpy.TGraph import TGraph
    with pytest.raises(ValueError,match='positive integer'):
        TGraph.VisibilityGraph(Face.Rectangle(width=2,length=2),**{option:value})
