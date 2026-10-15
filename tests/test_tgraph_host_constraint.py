"""A planar host constrains representative graph connections, preserving nodes."""
import math
import pytest
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Topology import Topology
from topologicpy.Grid import Grid
from topologicpy.TGraph import TGraph


def floor():
    outer=Wire.ByVertices([Vertex.ByCoordinates(*p,0) for p in [[0,0],[12,0],[12,10],[0,10]]])
    hole=Wire.ByVertices([Vertex.ByCoordinates(*p,0) for p in [[4,3],[8,3],[8,7],[4,7]]])
    return Face.ByWires(outer,[hole])


@pytest.mark.parametrize('rotation',[0,55])
def test_host_constrains_routes_in_host_plane(rotation):
    host=floor()
    if rotation:host=Topology.Rotate(host,origin=Vertex.ByCoordinates(0,0,0),axis=[1,0,0],angle=rotation)
    shell=Topology.Slice(host,Grid.OnFace(host,spacing=4,uAlignment='start',vAlignment='start'))
    plain=TGraph.ByTopology(shell,direct=False,viaSharedTopologies=True,useInternalVertex=True,ontology=False)
    graph=TGraph.ByTopology(shell,direct=False,viaSharedTopologies=True,useInternalVertex=True,ontology=False,host=host)
    assert graph is not None
    assert len(TGraph.Vertices(graph))==len(TGraph.Vertices(plain))
    assert len(TGraph.Edges(graph))<=len(TGraph.Edges(plain))
    vertices=TGraph.Vertices(graph,asTopologic=True,useRepresentation=True)
    mapping={record['index']:v for record,v in zip(TGraph.Vertices(graph),vertices)}
    from topologicpy.Edge import Edge
    for edge in TGraph.Edges(graph):
        line=Edge.ByVertices([mapping[edge['src']],mapping[edge['dst']]])
        assert Topology.Contains(host,line,tolerance=.0001,silent=True)


def test_invalid_host_is_rejected():
    host=floor()
    graph=TGraph.ByTopology(host,host=Vertex.ByCoordinates(0,0,0),silent=True)
    assert graph is None
