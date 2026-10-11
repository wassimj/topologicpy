import random
import pytest
from topologicpy.TGraph import TGraph

def reference(g,source,mode):
    depths={source:0};queue=[source]
    for v in queue:
        for u in TGraph.AdjacentIndices(g,v,mode):
            if u not in depths:depths[u]=depths[v]+1;queue.append(u)
    return depths

@pytest.mark.parametrize('mode',['out','in','all','unknown'])
@pytest.mark.parametrize('cached',[False,True])
def test_random_depths_and_means(mode,cached):
    rng=random.Random(84)
    for trial in range(8):
        g=TGraph(directed=True,allowSelfLoops=True,allowParallelEdges=True)
        for i in range(12):g.AddVertex()
        for _ in range(30):g.AddEdge(rng.randrange(10),rng.randrange(10),directed=rng.choice([True,False]))
        g._vertices[3]['active']=False;g._edges[2]['active']=False
        if cached:TGraph.AnalysisContext(g)
        expected={}
        for source in TGraph._ActiveVertexIndices(g):
            depths=reference(g,source,mode)
            assert TGraph.DepthMap(g,source,mode)==depths
            assert list(TGraph.DepthMap(g,source,mode))==list(depths)
            expected[source]=sum(depths.values())/(len(depths)-1) if len(depths)>1 else -1.
        assert TGraph.MeanDepth(g,mode=mode,key='mean')==expected
        assert all(g._vertices[i]['dictionary']['mean']==v for i,v in expected.items())
        assert TGraph.MeanDepth(g,sources=[0,{'index':11},3,-1],mode=mode)=={i:expected[i] for i in (0,11)}

def test_cache_outputs_and_mutations():
    g=TGraph()
    for _ in range(4):g.AddVertex()
    g.AddEdge(0,1);g.AddEdge(1,2)
    ctx=TGraph.AnalysisContext(g)
    assert TGraph.DepthMap(g,0)=={0:0,1:1,2:2}
    g._vertices[1]['dictionary']['depth']=999
    TGraph.DepthMap(g,0)
    assert g._vertices[1]['dictionary']['depth']==1
    expected=TGraph.MeanDepth(g,key='mean')
    g._vertices[0]['dictionary']['mean']=999
    assert TGraph.MeanDepth(g,key='mean')==expected
    assert g._vertices[0]['dictionary']['mean']==1.5
    TGraph.DepthMap(g,0,key='other')
    assert g._vertices[2]['dictionary']['other']==2
    g._edges[1]['active']=False
    assert TGraph.DepthMap(g,0)=={0:0,1:1}
    assert TGraph.MeanDepth(g)[0]==1.
    g._edges[1]['active']=True
    g.AddEdge(2,3)
    assert TGraph.DepthMap(g,0)[3]==3
    g._vertices[2]['active']=False
    assert TGraph.DepthMap(g,0)=={0:0,1:1}
    assert TGraph.DepthMap(g,2)=={}
    ctx.Clear()
    assert TGraph.MeanDepth(g,sources=[])=={}

def test_empty_invalid_and_default():
    assert TGraph.MeanDepth(None)=={}
    assert TGraph.DepthMap(TGraph())=={}
    g=TGraph();g.AddVertex();g.AddVertex();g._vertices[0]['active']=False
    assert TGraph.DepthMap(g)=={1:0}
    assert TGraph.MeanDepth(g)=={1:-1.}

def test_shared_preparation(monkeypatch):
    g=TGraph()
    for _ in range(5):g.AddVertex()
    for i in range(4):g.AddEdge(i,i+1)
    original=TGraph.AdjacentIndices;calls=[]
    def counted(*args,**kwargs):
        calls.append(args[1]);return original(*args,**kwargs)
    monkeypatch.setattr(TGraph,'AdjacentIndices',staticmethod(counted))
    TGraph.MeanDepth(g)
    assert len(calls)==5
    calls.clear();TGraph.AnalysisContext(g)
    TGraph.DepthMap(g,0);TGraph.DepthMap(g,1)
    assert len(calls)==5
