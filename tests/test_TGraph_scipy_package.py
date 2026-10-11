import pytest,random
from topologicpy.TGraph import TGraph

def graph():
    g=TGraph(directed=True,allowSelfLoops=True,allowParallelEdges=True)
    rng=random.Random(84)
    for i in range(76):g.AddVertex({'x':i,'y':i%7,'z':0})
    for i in range(68):g.AddEdge(i,i+1,directed=bool(i%2),dictionary={'weight':i%5,'decimal':(i%7)*.1,'positive':1+i%5})
    for i in range(110):g.AddEdge(rng.randrange(69),rng.randrange(69),directed=bool(i%2),dictionary={'weight':i%5,'decimal':(i%7)*.1,'positive':1+i%5})
    g._vertices[15]['active']=False;g._edges[10]['active']=False
    return g

@pytest.mark.parametrize('name,options',[
    ('MeanDepth',{}),('MeanDepth',{'sources':[0,3,75]}),('Diameter',{}),('Diameter',{'mode':'out'}),
    ('ClosenessCentrality',{'weightKey':'weight','radius':2.5}),('ClosenessCentrality',{'weightKey':'decimal','mode':'in'}),
    ('PageRank',{}),('EigenvectorCentrality',{}),('EigenvectorCentrality',{'mode':'in'}),
    ('AccessibilityCentrality',{'step':2}),('AccessibilityCentrality',{'step':0}),
    ('ShortestPathTree',{'source':0,'edgeKey':'hop','includePaths':True,'includeEdges':True,'useSciPy':True}),
    ('ShortestPathTree',{'source':0,'edgeKey':'positive','includePaths':True,'includeEdges':True,'useSciPy':True}),
    ('ShortestPathTree',{'source':0,'edgeKey':'weight','includePaths':True,'includeEdges':True}),
    ('ShortestPathTree',{'source':0,'edgeKey':'decimal','includePaths':True}),
    ('ShortestPathsFromSource',{'source':0,'targets':[3,7,75],'edgeKey':'hop','returnCost':True,'useSciPy':True})])
def test_matches_fallback(monkeypatch,name,options):
    g=graph();fn=getattr(TGraph,name)
    if name in ['ClosenessCentrality','AccessibilityCentrality']:options=dict(options,colorKey=None)
    accelerated=fn(g,**options)
    monkeypatch.setattr(TGraph,'_ScipyOperation',staticmethod(lambda *a,**k:None))
    reference=fn(g,**options)
    if isinstance(reference,list):assert accelerated==pytest.approx(reference,abs=1e-6)
    elif name=='MeanDepth':assert accelerated==pytest.approx(reference,abs=1e-12)
    else:assert accelerated==reference

def test_mst_exact_identity(monkeypatch):
    g=graph()
    for edge in g._edges:edge['dictionary']['id']=edge['index']
    actual=TGraph.MinimumSpanningTree(g)
    monkeypatch.setattr(TGraph,'_ScipyOperation',staticmethod(lambda *a,**k:None))
    expected=TGraph.MinimumSpanningTree(g)
    assert [(e['src'],e['dst'],e['dictionary']['id']) for e in actual._edges]==[(e['src'],e['dst'],e['dictionary']['id']) for e in expected._edges]

def test_kernels_selected_and_cache(monkeypatch):
    calculate = TGraph._ScipyCalculate
    g=graph()
    _,_,adj=TGraph._DepthAdjacency(g)
    assert calculate(g,'summaries',adj) is not None
    c=TGraph.Compile(g)
    assert calculate(g,'pagerank',c,.85,10,1e-9) is not None
    assert calculate(g,'eigenvector',adj,10,1e-9) is not None
    assert calculate(g,'mst',c) is None  # Small graphs retain inexpensive Kruskal.
    vertices=list(range(76));routing={i:[(i+1,i)] for i in range(75)};routing[75]=[]
    assert calculate(g,'routing',vertices,routing,{i:1 for i in range(75)},0) is not None
    assert calculate(g,'routing',vertices,routing,{i:1.1 for i in range(75)},0) is None
    assert calculate(g,'accessibility',[{(i+1)%76:1.} for i in range(76)],2) is not None
    context=TGraph.AnalysisContext(g)
    first=TGraph.MeanDepth(g);TGraph.Diameter(g)
    g._edges[0]['active']=False
    actual=TGraph.MeanDepth(g)
    context.Detach()
    monkeypatch.setattr(TGraph,'_ScipyOperation',staticmethod(lambda *a,**k:None))
    assert actual==TGraph.MeanDepth(g)

def test_connected_diameter_and_weight_mutation(monkeypatch):
    g=TGraph()
    for i in range(70):g.AddVertex()
    for i in range(69):g.AddEdge(i,i+1,dictionary={'cost':1.})
    assert TGraph.Diameter(g)==69
    ctx=TGraph.AnalysisContext(g)
    TGraph.ClosenessCentrality(g,weightKey='cost',key='cc',colorKey=None)
    g._edges[0]['dictionary']['cost']=4.
    actual=TGraph.ClosenessCentrality(g,weightKey='cost',key='cc',colorKey=None)
    g._vertices[0]['dictionary']['cc']=999
    assert TGraph.ClosenessCentrality(g,weightKey='cost',key='cc',colorKey=None)==actual
    assert g._vertices[0]['dictionary']['cc']==actual[0]
    ctx.Detach()
    monkeypatch.setattr(TGraph,'_ScipyOperation',staticmethod(lambda *a,**k:None))
    assert TGraph.ClosenessCentrality(g,weightKey='cost',key='cc',colorKey=None)==actual

def test_large_mst_ties_parallel_negative_zero_and_metadata(monkeypatch):
    g=TGraph(allowParallelEdges=True)
    for i in range(1000):g.AddVertex({'id':i})
    for i in range(999):g.AddEdge(i,i+1,dictionary={'weight':i%5-2,'id':i})
    for i in range(998):g.AddEdge(i,i+2,dictionary={'weight':i%3,'id':1000+i})
    g.AddEdge(0,1,dictionary={'weight':-2,'id':'parallel'})
    actual=TGraph.MinimumSpanningTree(g)
    monkeypatch.setattr(TGraph,'_ScipyOperation',staticmethod(lambda *a,**k:None))
    expected=TGraph.MinimumSpanningTree(g)
    assert [(e['src'],e['dst'],e['dictionary']) for e in actual._edges]==[(e['src'],e['dst'],e['dictionary']) for e in expected._edges]

def test_sparse_routing_cache_parent_ties_and_mutation(monkeypatch):
    g=graph();ctx=TGraph.AnalysisContext(g)
    options=dict(edgeKey='positive',includePaths=True,includeEdges=True,useSciPy=True)
    TGraph.ShortestPathTree(g,0,**options)
    actual=TGraph.ShortestPathTree(g,1,**options)
    assert ctx.Info()['namespaces']['scipy_routing_matrix']['hits']>0
    g._edges[0]['dictionary']['positive']=5
    changed=TGraph.ShortestPathTree(g,1,**options)
    ctx.Detach();monkeypatch.setattr(TGraph,'_ScipyOperation',staticmethod(lambda *a,**k:None))
    assert changed==TGraph.ShortestPathTree(g,1,**options)
