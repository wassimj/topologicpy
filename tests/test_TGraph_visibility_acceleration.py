"""Visibility acceleration must preserve corner rules and graph records."""
import builtins
import pytest
from topologicpy.TGraph import TGraph
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Face import Face

def scene():
    def ring(points):
        return Wire.ByVertices([Vertex.ByCoordinates(x,y,0) for x,y in points],close=True)
    floor=Face.ByWires(ring([(0,0),(10,0),(10,10),(0,10)]),[ring([(4,4),(6,4),(6,6),(4,6)])])
    points=[(2,6),(6,2),(2,2),(8,8),(1,1),(9,1),(4,4),(6,4)]
    return floor,[Vertex.ByCoordinates(x,y,0) for x,y in points]

def records(g):
    return [(e['index'],e['src'],e['dst'],e['directed'],e['dictionary']['length']) for e in g._edges]

@pytest.mark.parametrize('allow',[False,True])
@pytest.mark.parametrize('directed',[False,True])
@pytest.mark.parametrize('neighbors',[None,3])
def test_acceleration_matches_python(allow,directed,neighbors):
    pytest.importorskip('numba')
    floor,points=scene()
    options=dict(vertices=points,includeBoundaryVertices=False,trim=True,allowBoundaryPaths=allow,
                 bidirectional=not directed,maxNeighbors=neighbors,storeEdgeRepresentations=False,ontology=False)
    expected=TGraph.VisibilityGraph(floor,**options)
    accelerated=TGraph.VisibilityGraph(floor,useNumba=True,**options)
    assert accelerated._dictionary['visibility_engine']==('numba_full' if neighbors is None else 'angular_event_sweep')
    assert records(accelerated)==records(expected)
    pairs={(min(e['src'],e['dst']),max(e['src'],e['dst'])) for e in accelerated._edges}
    if neighbors is None:
        assert (0,1) not in pairs # Tangency at a corner.
        assert (2,3) not in pairs # Crosses the hole through corners.
        assert (4,5) in pairs

def test_optional_accelerator_failure_uses_python(monkeypatch):
    floor,points=scene()
    options=dict(vertices=points,includeBoundaryVertices=False,trim=True,storeEdgeRepresentations=False,ontology=False)
    expected=TGraph.VisibilityGraph(floor,**options)
    original_import=builtins.__import__
    def unavailable(name,*args,**kwargs):
        if name=='topologicpy.TGraphVisibility': raise ImportError('optional accelerator unavailable')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',unavailable)
    actual=TGraph.VisibilityGraph(floor,useNumba=True,**options)
    assert records(actual)==records(expected)
