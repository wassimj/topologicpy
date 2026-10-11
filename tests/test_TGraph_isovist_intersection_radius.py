import math
import pytest
from shapely.geometry import box
from topologicpy.TGraph import TGraph
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology

def pairs(g):
    assert g is not None
    return [(e['src'],e['dst']) for e in g._edges]

def test_generates_and_forwards_radius(monkeypatch):
    room=Face.Rectangle(width=30,length=30)
    origins=[Vertex.ByCoordinates(x,0,0) for x in [-6,-5,0,6]]
    called=[]; original=Face.Isovist
    def spy(*args,**kwargs):
        called.append(kwargs['radius']);return original(*args,**kwargs)
    monkeypatch.setattr(Face,'Isovist',spy)
    g=TGraph.IsovistIntersectionGraph(room,origins=origins,radius=2)
    assert called==[2]*4
    assert pairs(g)==[(0,1)]
    for i,o in enumerate(origins):
        p=Vertex.Coordinates(o)
        assert max(math.dist(p,Vertex.Coordinates(v)) for v in Topology.Vertices(TGraph.VertexOriginator(g,i)))<=2.0001

@pytest.mark.parametrize('metrics',[True,False])
def test_precomputed_clipping_and_distance_boundary(metrics):
    ps=[box(-100,-100,100,100)]*4
    origins=[[0,0,0],[1,0,0],[4,0,0],[8,0,0]]
    g=TGraph.IsovistIntersectionGraph(ps,origins=origins,radius=2,computeOverlap=metrics,batchSize=1)
    assert pairs(g)==[(0,1),(1,2)] # exactly 2r apart gives no positive area

def test_index_matches_unbounded_reference():
    origins=[Vertex.ByCoordinates(i*.7,j*.7,0) for i in range(4) for j in range(4)]
    room=Face.Rectangle(width=20,length=20)
    isovists=[Face.Isovist(room,o,radius=1) for o in origins]
    reference=TGraph.IsovistIntersectionGraph(isovists,origins=origins)
    bounded=TGraph.IsovistIntersectionGraph(face=room,origins=origins,radius=1,workers=1,batchSize=3)
    assert pairs(bounded)==pairs(reference)
    for a,b in zip(bounded._edges,reference._edges):
        assert a['dictionary']['overlap_area']==pytest.approx(b['dictionary']['overlap_area'],abs=1e-5)

def test_radius_requires_real_origins():
    assert TGraph.IsovistIntersectionGraph([box(0,0,1,1)],radius=1,silent=True) is None
    assert TGraph.IsovistIntersectionGraph(face=Face.Rectangle(),origins=[],radius=1) is not None
    for r in [0,-1,float('inf'),float('nan')]:
        assert TGraph.IsovistIntersectionGraph(face=Face.Rectangle(),origins=[[0,0,0]],radius=r,silent=True) is None
