import json, math
from pathlib import Path
import pytest
from topologicpy.TGraph import TGraph

def graph_from_endpoints(endpoints):
    graph=TGraph(allowSelfLoops=False)
    points=[]
    for p,q in endpoints:
        indices=[]
        for point in (p,q):
            index=next((i for i,other in enumerate(points) if math.dist(point,other)<=1e-5),None)
            if index is None:
                index=len(points);points.append(point)
                graph.AddVertex(dict(zip(('x','y','z'),point)))
            indices.append(index)
        graph.AddEdge(*indices)
    return graph

def test_barnsbury_saved_choice():
    folder=Path(__file__).resolve().parent
    geometry=folder/'fixtures/depthmap/barnsbury_segment_connections.json'
    reference=folder/'fixtures/depthmap/barnsbury_choice_reference.json'
    fixture=json.loads(geometry.read_text()); expected=json.loads(reference.read_text())
    g=graph_from_endpoints(fixture['endpoints'])
    choice=TGraph.AngularChoice(g,method='depthmap',algorithm='tulip',mantissa=None)
    assert choice==expected['choice']
    assert sum(choice)==160863
    assert [r['dictionary']['choice'] for r in g._edges]==choice
    nach=TGraph.AngularChoice(g,method='nach',algorithm='tulip',mantissa=None)
    # Depthmap's saved float32 derived column and CSV decimal precision.
    assert nach==pytest.approx(expected['nach'],abs=1.2e-7)
    assert TGraph.AngularChoice(g,method='depthmap',mantissa=None)!=choice

def test_tulip_chain_and_disconnected():
    g=graph_from_endpoints([((0,0,0),(2,0,0)),((2,0,0),(2,3,0)),
                           ((2,3,0),(6,3,0)),((20,0,0),(21,0,0))])
    assert TGraph.AngularChoice(g,method='depthmap',algorithm='tulip')==[0,2,0,0]
    assert TGraph.AngularChoice(g,normalize=False)==[0,1,0,0]

@pytest.mark.parametrize('kwargs',[{'radius':1.5},{'weighting':'invalid'},
    {'method':'betweenness'},{'tulipBins':True},{'tulipBins':0},
    {'tulipBins':1.5},{'tulipBins':1000000},{'mode':'invalid'},
    {'radiusType':'invalid'},{'algorithm':'invalid'}])
def test_tulip_invalid(kwargs):
    options=dict(method='depthmap',algorithm='tulip',silent=True);options.update(kwargs)
    assert TGraph.AngularChoice(TGraph(),**options) is None

def test_tulip_empty_and_directed():
    assert TGraph.AngularChoice(TGraph(),method='depthmap',algorithm='tulip')==[]
    g=TGraph(directed=True)
    a=g.AddVertex({'x':0,'y':0,'z':0});b=g.AddVertex({'x':1,'y':0,'z':0})
    g.AddEdge(a,b)
    assert TGraph.AngularChoice(g,method='depthmap',algorithm='tulip',silent=True) is None

if __name__=='__main__':
    raise SystemExit(pytest.main([__file__,'-q','-p','no:cacheprovider','-o','addopts=']))
