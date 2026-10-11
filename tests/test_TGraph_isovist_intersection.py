import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box
from topologicpy.TGraph import TGraph

def pairs(g):
    assert g is not None
    return [(e['src'],e['dst']) for e in g._edges]

@pytest.mark.parametrize('metrics',[True,False])
@pytest.mark.parametrize('batch',[1,7,256])
def test_oracle(metrics,batch):
    rng=np.random.default_rng(87)
    ps=[box(x,y,x+w,y+h) for x,y,w,h in rng.uniform(0.1,8,(35,4))]
    ps += [ps[0],box(30,30,31,31),box(31,30,32,31),box(32,31,33,32)]
    expected=[(i,j) for i in range(len(ps)) for j in range(i+1,len(ps)) if ps[i].intersection(ps[j]).area>0]
    g=TGraph.IsovistIntersectionGraph(ps,computeOverlap=metrics,batchSize=batch)
    assert pairs(g)==expected
    assert TGraph.Compile(g)['n']==len(ps)
    for e in g._edges:
        if metrics:
            a=ps[e['src']].intersection(ps[e['dst']]).area
            assert e['dictionary']['overlap_area']==pytest.approx(a)

def test_holes_containment_threshold():
    p=Polygon([(0,0),(10,0),(10,10),(0,10)],holes=[[(3,3),(7,3),(7,7),(3,7)]])
    ps=[p,box(4,4,6,6),box(1,1,2,2),p]
    assert pairs(TGraph.IsovistIntersectionGraph(ps))==[(0,2),(0,3),(2,3)]
    assert pairs(TGraph.IsovistIntersectionGraph(ps,minOverlap=1,computeOverlap=False))==[(0,3)]

def test_bad_inputs():
    assert pairs(TGraph.IsovistIntersectionGraph([]))==[]
    for kwargs in [dict(tolerance=0),dict(minOverlap=-1),dict(batchSize=0),dict(origins=[]),dict(origins=[[np.nan,0,0]])]:
        assert TGraph.IsovistIntersectionGraph([box(0,0,1,1)],silent=True,**kwargs) is None
    assert TGraph.IsovistIntersectionGraph([Polygon([(0,0),(1,1),(0,1),(1,0)])],silent=True) is None

def test_concave_shared_visibility_without_origin_visibility():
    a=Polygon([(0,0),(4,0),(4,1),(1,1),(1,4),(0,4)])
    b=Polygon([(3,0),(7,0),(7,4),(6,4),(6,1),(3,1)])
    g=TGraph.IsovistIntersectionGraph([a,b],origins=[[.5,3,0],[6.5,3,0]])
    assert pairs(g)==[(0,1)]
    assert g._vertices[0]['dictionary']['y']==3
    assert g._edges[0]['dictionary']['overlap_area']==pytest.approx(1)
    assert TGraph.AddEdge(g,0,1) is None

def test_native_hole_and_origins():
    from topologicpy.Face import Face
    from topologicpy.Wire import Wire
    from topologicpy.Vertex import Vertex
    def ring(coords):
        return Wire.ByVertices([Vertex.ByCoordinates(x,y,0) for x,y in coords],close=True)
    a=Face.ByWires(ring([(0,0),(10,0),(10,10),(0,10)]),[ring([(3,3),(7,3),(7,7),(3,7)])])
    b=Face.ByWire(ring([(4,4),(6,4),(6,6),(4,6)]))
    c=Face.ByWire(ring([(1,1),(2,1),(2,2),(1,2)]))
    g=TGraph.IsovistIntersectionGraph([a,b,c],origins=[Vertex.ByCoordinates(1,1,0),Vertex.ByCoordinates(5,5,0),Vertex.ByCoordinates(1.5,1.5,0)])
    assert pairs(g)==[(0,2)]
    assert g._vertices[0]['dictionary']['isovist_area']==pytest.approx(84)

@pytest.mark.parametrize('metrics',[True,False])
def test_parallel_matches_serial(metrics):
    ps=[box(i*.01,0,2+i*.01,2) for i in range(100)]
    serial=TGraph.IsovistIntersectionGraph(ps,workers=1,computeOverlap=metrics)
    threaded=TGraph.IsovistIntersectionGraph(ps,workers=4,computeOverlap=metrics)
    assert serial._edges==threaded._edges
    assert len(threaded._edges)==4950

def test_native_vertical_plane_and_originator():
    from topologicpy.Face import Face
    from topologicpy.Wire import Wire
    from topologicpy.Vertex import Vertex
    def face(x):
        return Face.ByWire(Wire.ByVertices([Vertex.ByCoordinates(x+a,b,5) for a,b in [(0,0),(2,0),(2,2),(0,2)]],close=True))
    from topologicpy.Topology import Topology
    ps=[Topology.Rotate(face(x),axis=[1,0,0],angle=90) for x in [0,1,4]]
    g=TGraph.IsovistIntersectionGraph(ps)
    assert pairs(g)==[(0,1)]
    assert g._edges[0]['dictionary']['overlap_area']==pytest.approx(2)
    assert TGraph.VertexOriginator(g,0) is ps[0]
    ps[2]=Topology.Translate(ps[2],y=1)
    assert TGraph.IsovistIntersectionGraph(ps,silent=True) is None
