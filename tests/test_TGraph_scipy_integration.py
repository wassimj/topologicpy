import pytest
from topologicpy.TGraph import TGraph

@pytest.mark.parametrize('mode',['out','in','all'])
@pytest.mark.parametrize('radius',[None,0,1,2.5])
def test_scipy_matches_python(monkeypatch,mode,radius):
    g=TGraph(directed=True,allowSelfLoops=True,allowParallelEdges=True)
    for i in range(72):g.AddVertex()
    for i in range(65):
        g.AddEdge(i,i+1,directed=(i%3!=0))
        if i%4==0:g.AddEdge(i,i+2,directed=False)
    g.AddEdge(0,1);g.AddEdge(2,2)
    g._vertices[20]['active']=False;g._edges[7]['active']=False
    fast=TGraph._ScipyUnweightedDistanceSummaries
    assert fast([[i+1] if i<71 else [] for i in range(72)],radius) is not None
    options=dict(key='result',colorKey=None,mantissa=None,radius=radius,silent=True)
    actual=[TGraph.Integration(g,method=m,**options) for m in ['hh','closeness']]
    actual.append(TGraph.ClosenessCentrality(g,mode=mode,normalize=False,**options))
    monkeypatch.setattr(TGraph,'_ScipyUnweightedDistanceSummaries',staticmethod(lambda *a:None))
    expected=[TGraph.Integration(g,method=m,**options) for m in ['hh','closeness']]
    expected.append(TGraph.ClosenessCentrality(g,mode=mode,normalize=False,**options))
    assert actual==expected
    monkeypatch.setattr(TGraph,'_ScipyUnweightedDistanceSummaries',staticmethod(fast))
    TGraph.AnalysisContext(g)
    assert TGraph.Integration(g,method='hh',**options)==actual[0]
    g._vertices[0]['dictionary']['result']=999
    assert TGraph.Integration(g,method='hh',**options)==actual[0]
    assert g._vertices[0]['dictionary']['result']==actual[0][0]
    g._edges[0]['active']=False
    changed=TGraph.Integration(g,method='hh',**options)
    g._analysis_context.Detach()
    monkeypatch.setattr(TGraph,'_ScipyUnweightedDistanceSummaries',staticmethod(lambda *a:None))
    assert TGraph.Integration(g,method='hh',**options)==changed

def test_batch_boundary():
    adjacency=[[i+1] if i<269 else [] for i in range(270)]
    result=TGraph._ScipyUnweightedDistanceSummaries(adjacency)
    assert result is not None
    for i,(total,count,_) in enumerate(result):
        assert count==270-i
        assert total==(269-i)*(270-i)/2

def test_acceleration_failure_falls_back(monkeypatch):
    import scipy.sparse.csgraph
    def fail(*args,**kwargs):raise RuntimeError('Unavailable traversal')
    monkeypatch.setattr(scipy.sparse.csgraph,'dijkstra',fail)
    g=TGraph()
    for _ in range(65):g.AddVertex()
    for i in range(64):g.AddEdge(i,i+1)
    assert TGraph._ScipyUnweightedDistanceSummaries([[1]]*65) is None
    values=TGraph.Integration(g,method='hh',key=None,colorKey=None)
    assert len(values)==65 and all(v>0 for v in values)
