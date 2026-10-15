"""Located edges must retain world coordinates in navigation graphs."""
import math
import pytest
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Grid import Grid
from topologicpy.Topology import Topology
from topologicpy.TGraph import TGraph


@pytest.mark.parametrize('start,end',[
    ([1,2,0],[1,8,0]),([11,2,0],[11,8,0]),([3,7,2],[9,11,5]),
])
def test_located_straight_edge_parameter(start,end):
    edge=Edge.ByVertices([Vertex.ByCoordinates(*start),Vertex.ByCoordinates(*end)])
    for u in [.1,.5,.9]:
        actual=Vertex.Coordinates(Edge.VertexByParameter(edge,u),mantissa=None)
        expected=[a+u*(b-a) for a,b in zip(start,end)]
        assert math.dist(actual,expected)<1e-6


def test_sliced_navigation_edge_representatives():
    face=Face.Rectangle(width=12,length=10)
    shell=Topology.Slice(face,Grid.OnFace(face,spacing=2,uAlignment='start',vAlignment='start'))
    graph=TGraph.ByTopology(shell,direct=False,viaSharedTopologies=True,useInternalVertex=True,ontology=False)
    assert graph is not None
    for vertex in TGraph.Vertices(graph):
        if vertex.get('originator_type')!='Edge':continue
        expected=Vertex.Coordinates(Topology.Centroid(vertex['originator']))
        actual=[vertex['dictionary'][key] for key in ['x','y','z']]
        assert math.dist(actual,expected)<1e-6
    path,cost=TGraph.ShortestPath(graph,[-5,-4,0],[5,4,0],useAStar=True,returnCost=True)
    assert path
    records={v['index']:v['dictionary'] for v in TGraph.Vertices(graph)}
    actual=sum(math.dist([records[a][k] for k in ['x','y','z']],[records[b][k] for k in ['x','y','z']]) for a,b in zip(path,path[1:]))
    assert abs(cost-actual)<1e-6


def test_curved_edge_keeps_exact_curve_evaluation():
    edge=Edge.Arc(radius=3,fromAngle=0,toAngle=90)
    midpoint=Vertex.Coordinates(Edge.VertexByParameter(edge,.5),mantissa=None)
    assert math.dist(midpoint,[3/math.sqrt(2),3/math.sqrt(2),0])<1e-6
