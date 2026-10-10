import copy,math
import pytest
from topologicpy.TGraph import TGraph

def graph(directed=False):
    g=TGraph(directed=directed,allowParallelEdges=True)
    for i,p in enumerate([(0,0,0),(1,0,0),(1,1,0),(2,1,0),(3,1,0)]):
        g.AddVertex({'x':p[0],'y':p[1],'z':p[2],'label':str(i),'group':i%2})
    for u,v in [(0,1),(1,2),(2,0),(2,3),(3,4)]:g.AddEdge(u,v,dictionary={'weight':1,'capacity':2})
    return g

def comparable(value):
    if isinstance(value,TGraph):
        return (value._directed,comparable(value._vertices),comparable(value._edges))
    if isinstance(value,dict):return {k:comparable(v) for k,v in value.items() if k not in ('representation','originator')}
    if isinstance(value,(list,tuple)):return [comparable(v) for v in value]
    return value

CASES=[
 ('Degree',{'index':0}),('DegreeCentrality',{'colorKey':None}),('DegreeSequence',{}),('DegreeMatrix',{}),
 ('MaximumDelta',{}),('MinimumDelta',{}),('Connectivity',{'colorKey':None}),('Diameter',{}),
 ('BreadthFirstSearch',{'source':0}),('DepthFirstSearch',{'source':0}),('DepthMap',{'source':0}),
 ('ConnectedComponents',{}),('IsConnected',{}),('IsTree',{}),('IsBipartite',{}),('IsComplete',{}),
 ('LocalClusteringCoefficient',{}),('AverageClusteringCoefficient',{}),('GlobalClusteringCoefficient',{}),
 ('Bridges',{}),('CutVertices',{}),('BiconnectedComponents',{}),('Leaves',{}),('IsolatedVertices',{}),
 ('AdjacencyMatrix',{}),('Laplacian',{}),('FiedlerVector',{}),('FiedlerVectorPartition',{}),
 ('PageRank',{}),('EigenvectorCentrality',{}),('EigenVectorCentrality',{'colorKey':None}),('AccessibilityCentrality',{'colorKey':None}),
 ('BetweennessCentrality',{'colorKey':None}),('ClosenessCentrality',{'colorKey':None}),('Choice',{}),('Integration',{'colorKey':None,'method':'hh'}),
 ('AngularChoice',{'algorithm':'tulip','method':'depthmap'}),('AngularIntegration',{'algorithm':'tulip','method':'depthmap'}),
 ('AngularTulipAnalysis',{'writeValues':False}),('AngularConnectivity',{'colorKey':None,'method':'depthmap'}),
 ('ShortestPath',{'source':0,'target':4,'edgeKey':'hop'}),('ShortestPathTree',{'source':0,'edgeKey':'hop'}),
 ('ShortestPathsFromSource',{'source':0,'targets':[3,4],'edgeKey':'hop'}),
 ('TopologicalDistance',{'vertexA':0,'vertexB':4}),('Depth',{'vertex':4,'source':0}),
 ('LineGraph',{}),('Quotient',{'key':'group'}),('MinimumSpanningTree',{}),('Subgraph',{'vertices':[0,1,2]}),
 ('InducedSubgraph',{'vertices':[0,1,2]}),('KHopsSubgraph',{'vertices':[0],'k':2}),
 ('Tree',{'vertex':0}),('Complement',{}),('Complete',{}),('WLFeatures',{'key':'label'}),
 ('AABB',{}),('MetricDistance',{'vertexA':0,'vertexB':4}),('Distance',{'vertexA':0,'vertexB':4}),
 ('MeshData',{}),('MaximumFlow',{'source':0,'sink':4}),('MinimumCut',{'source':0,'target':4}),
 ('EdgeConnectivity',{'source':0,'target':4}),('VertexConnectivity',{'source':0,'target':4}),
 ('ChromaticNumber',{}),('BetweennessPartition',{'m':2}),
]

@pytest.mark.parametrize('name,kwargs',CASES)
def test_cached_method_matches_uncached_and_reuses(name,kwargs):
    g=graph();fn=getattr(TGraph,name)
    expected=fn(g,**kwargs)
    context=TGraph.AnalysisContext(g)
    first=fn(g,**kwargs)
    assert comparable(first)==comparable(expected)
    hits=context.Info()['hits']
    second=fn(g,**kwargs)
    assert comparable(second)==comparable(first)
    assert context.Info()['hits']>hits

@pytest.mark.parametrize('directed',[False,True])
@pytest.mark.parametrize('radius',[None,0,1,2.5])
@pytest.mark.parametrize('weightKey',[None,'weight','length'])
def test_ordinary_shared_results(directed,radius,weightKey):
    g=graph(directed);ctx=TGraph.AnalysisContext(g)
    for useEdges in (False,True):
        actual=TGraph.BetweennessCentrality(g,useEdges=useEdges,weightKey=weightKey,radius=radius,key=None,colorKey=None,mantissa=None)
        ctx.Detach()
        expected=TGraph.BetweennessCentrality(g,useEdges=useEdges,weightKey=weightKey,radius=radius,key=None,colorKey=None,mantissa=None)
        g._analysis_context=ctx
        assert actual==expected
    for mode in ('out','in','all'):
        actual=TGraph.ClosenessCentrality(g,weightKey=weightKey,radius=radius,mode=mode,key=None,colorKey=None,mantissa=None)
        ctx.Detach()
        expected=TGraph.ClosenessCentrality(g,weightKey=weightKey,radius=radius,mode=mode,key=None,colorKey=None,mantissa=None)
        g._analysis_context=ctx
        assert actual==expected

def test_both_betweenness_results_and_distance_summaries_reused():
    g=graph();ctx=TGraph.AnalysisContext(g)
    TGraph.BetweennessCentrality(g,key=None,colorKey=None)
    TGraph.BetweennessCentrality(g,useEdges=True,key=None,colorKey=None)
    assert ctx.Info()['namespaces']['brandes']['hits']==1
    TGraph.ClosenessCentrality(g,key=None,colorKey=None)
    TGraph.Integration(g,method='hh',key=None,colorKey=None)
    assert ctx.Info()['namespaces']['ordinary_distances']['hits']==2

def test_angular_all_measures_reuse_one_traversal(monkeypatch):
    g=graph();ctx=TGraph.AnalysisContext(g)
    original=TGraph._AngularTulipValues;calls=[]
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(TGraph,'_AngularTulipValues',staticmethod(counted))
    TGraph.AngularChoice(g,method='depthmap',algorithm='tulip')
    TGraph.AngularIntegration(g,method='depthmap',algorithm='tulip')
    TGraph.AngularChoice(g,method='nach',algorithm='tulip')
    TGraph.AngularIntegration(g,method='nain',algorithm='tulip')
    assert len(calls)==1

@pytest.mark.parametrize('radiusType,radius',[('angular',None),('angular',1),('metric',2),('topological',2)])
@pytest.mark.parametrize('weighting',[None,'length'])
@pytest.mark.parametrize('algorithm',['exact','tulip'])
def test_angular_cached_variants_preserve_all_results(radiusType,radius,weighting,algorithm):
    g=graph();ctx=TGraph.AnalysisContext(g)
    opts=dict(radiusType=radiusType,radius=radius,weighting=weighting,algorithm=algorithm,mantissa=None)
    for name,method in [('AngularIntegration','depthmap'),('AngularChoice','depthmap'),('AngularIntegration','nain'),('AngularChoice','nach')]:
        actual=getattr(TGraph,name)(g,method=method,**opts)
        ctx.Detach();expected=getattr(TGraph,name)(g,method=method,**opts)
        g._analysis_context=ctx
        assert actual==expected
    namespace='angular_exact' if algorithm=='exact' else 'angular_tulip'
    assert ctx.Info()['namespaces'][namespace]['hits']==3

@pytest.mark.parametrize('returnTree',[False,True])
@pytest.mark.parametrize('returnEdges',[False,True])
def test_batch_paths_only_reconstruct_requested_targets(returnTree,returnEdges):
    g=graph();ctx=TGraph.AnalysisContext(g)
    result=TGraph.ShortestPathsFromSource(g,0,targets=[3,4],edgeKey='hop',returnTree=returnTree,returnEdges=returnEdges)
    for target in (3,4):
        expected=TGraph.ShortestPath(g,0,target,edgeKey='hop',returnEdges=returnEdges)
        assert comparable(result[target])==comparable(expected)
    if returnTree:assert 'paths' in result['_tree']

def test_clustering_uses_one_triangle_calculation():
    g=graph();ctx=TGraph.AnalysisContext(g)
    TGraph.LocalClusteringCoefficient(g)
    TGraph.GlobalClusteringCoefficient(g)
    assert ctx.Info()['namespaces']['clustering_stats']['hits']==1

def test_structural_measures_use_one_low_link_calculation():
    g=graph();ctx=TGraph.AnalysisContext(g)
    TGraph.Bridges(g);TGraph.CutVertices(g);TGraph.BiconnectedComponents(g)
    assert ctx.Info()['namespaces']['low_link']=={'hits':2,'misses':1}

def test_direct_weight_edits_and_metadata_only_reuse():
    g=graph();ctx=TGraph.AnalysisContext(g)
    a=TGraph.Compile(g,useNumpy=False,useSciPy=False)
    g.SetVertexValue(0,'display_name','changed')
    b=TGraph.Compile(g,useNumpy=False,useSciPy=False)
    assert ctx.Info()['namespaces']['compiled']['hits']==1
    TGraph.EdgeDictionary(g,0,copy=False)['weight']=9
    c=TGraph.Compile(g,useNumpy=False,useSciPy=False)
    assert c['edges'][0]['weight']==9 and a['edges'][0]['weight']==1

@pytest.mark.parametrize('change',[
 lambda g:g.AddEdge(0,4),lambda g:g.RemoveEdge(0),lambda g:g.RemoveVertex(1),
 lambda g:TGraph.SetVertexCoordinates(g,0,x=-5),
 lambda g:TGraph.VertexDictionary(g,0,copy=False).__setitem__('x',-4),
 lambda g:TGraph.EdgeDictionary(g,0,copy=False).__setitem__('weight',8)])
def test_live_mutation_matches_uncached(change):
    g=graph();ctx=TGraph.AnalysisContext(g)
    TGraph.ClosenessCentrality(g,weightKey='length',key=None,colorKey=None)
    TGraph.BetweennessCentrality(g,weightKey='weight',key=None,colorKey=None)
    change(g)
    actual=(TGraph.ClosenessCentrality(g,weightKey='length',key=None,colorKey=None),TGraph.BetweennessCentrality(g,weightKey='weight',key=None,colorKey=None))
    ctx.Detach()
    expected=(TGraph.ClosenessCentrality(g,weightKey='length',key=None,colorKey=None),TGraph.BetweennessCentrality(g,weightKey='weight',key=None,colorKey=None))
    assert actual==expected

def test_cache_result_and_tree_cannot_be_poisoned():
    g=graph();ctx=TGraph.AnalysisContext(g)
    result=TGraph.ShortestPathTree(g,0,edgeKey='hop');expected=copy.deepcopy(result)
    result['distance'][4]=-100
    assert TGraph.ShortestPathTree(g,0,edgeKey='hop')==expected
    lg=TGraph.LineGraph(g);lg.SetVertexValue(0,'label','poison')
    assert TGraph.LineGraph(g)._vertices[0]['dictionary']['label']!='poison'

def test_routing_live_records_are_current_on_cache_hits():
    g=graph();ctx=TGraph.AnalysisContext(g)
    TGraph.ShortestPath(g,0,4,edgeKey='hop',returnVertices=True,returnEdges=True)
    g.SetVertexValue(0,'label','new label')
    route,vertices,edges=TGraph.ShortestPath(g,0,4,edgeKey='hop',returnVertices=True,returnEdges=True)
    assert vertices[0] is g._vertices[0] and vertices[0]['dictionary']['label']=='new label'
    assert all(e is g._edges[e['index']] for e in edges)

def test_distinct_target_batches_reuse_tree():
    g=graph();ctx=TGraph.AnalysisContext(g)
    TGraph.ShortestPathsFromSource(g,0,targets=[3],edgeKey='hop')
    TGraph.ShortestPathsFromSource(g,0,targets=[4],edgeKey='hop')
    assert ctx.Info()['namespaces']['ShortestPathTree']['hits']==1

def test_outputs_replayed_and_only_selected_vertices_updated():
    g=graph();ctx=TGraph.AnalysisContext(g)
    a=TGraph.LocalClusteringCoefficient(g,vertices=[0],key='result')
    g.SetVertexValue(3,'result',999)
    TGraph.VertexDictionary(g,0,copy=False).pop('result')
    assert TGraph.LocalClusteringCoefficient(g,vertices=[0],key='result')==a
    assert g._vertices[0]['dictionary']['result']==a[0]
    assert g._vertices[3]['dictionary']['result']==999

def test_limits_clear_detach_disabled():
    g=graph();ctx=TGraph.AnalysisContext(g,maxEntries=2,maxBytes=100000,maxSourceTrees=1)
    for source in range(5):TGraph.ShortestPathTree(g,source,edgeKey='hop')
    info=ctx.Info();assert info['entries']<=2 and info['estimatedBytes']<=100000 and info['evictions']>0
    assert sum(k[0]=='ShortestPathTree' for k in ctx._entries)<=1
    TGraph.InvalidateCache(g);assert ctx.Info()['entries']==0
    ctx.enabled=False;TGraph.DegreeSequence(g);assert ctx.Info()['entries']==0
    ctx.Detach();assert g._analysis_context is None

def test_multiple_compiled_configurations_retained():
    g=graph();ctx=TGraph.AnalysisContext(g)
    for key in ('weight','cost','weight'):TGraph.Compile(g,weightKey=key,useNumpy=False,useSciPy=False)
    assert ctx.Info()['namespaces']['compiled']=={'hits':1,'misses':2}

def test_callable_filter_bypasses_results():
    g=graph();ctx=TGraph.AnalysisContext(g);state={'blocked':False}
    def permitted(edge):return not state['blocked']
    first=TGraph.ShortestPath(g,0,4,edgeFilter=permitted,edgeKey='hop')
    state['blocked']=True
    assert TGraph.ShortestPath(g,0,4,edgeFilter=permitted,edgeKey='hop') is None
    assert first is not None and ctx.Info()['bypasses']>=2

def test_comparisons_track_both_graphs():
    a=graph();b=graph();ctx=TGraph.AnalysisContext(a)
    for name,options in [('WLKernel',{}),('HopperKernel',{}),('WeightedJaccardSimilarity',{}),('Compare',{}),('IsIsomorphic',{})]:
        fn=getattr(TGraph,name);first=fn(a,b,**options);hits=ctx.Info()['hits']
        assert comparable(fn(a,b,**options))==comparable(first)
        assert ctx.Info()['hits']>hits
    b.RemoveEdge(0)
    actual=TGraph.WLKernel(a,b)
    ctx.Detach();assert actual==TGraph.WLKernel(a,b)

def test_geometry_tokens_and_independent_graph_outputs(monkeypatch):
    g=graph();ctx=TGraph.AnalysisContext(g);calls=[]
    def construct(edges,**kwargs):calls.append(1);return graph()
    monkeypatch.setattr(TGraph,'AxialGraph',staticmethod(construct))
    edges=[]
    with pytest.raises(ValueError):ctx.ComputeGeometry('AxialGraph',edges)
    a=ctx.ComputeGeometry('AxialGraph',edges,dependencyToken=1)
    a.RemoveEdge(0)
    b=ctx.ComputeGeometry('AxialGraph',edges,dependencyToken=1)
    assert len(calls)==1 and TGraph.Size(b)==5
    ctx.ComputeGeometry('AxialGraph',edges,dependencyToken=2)
    assert len(calls)==2

def test_semantic_inference_reused_and_mutating_reasoning_bypassed(monkeypatch):
    import sys,types
    module=sys.modules[TGraph.__module__];calls=[]
    reasoner=types.SimpleNamespace(
        RDFGraphByTopology=lambda *a,**kw:{'before':1},
        Infer=lambda *a,**kw:calls.append(1) or {'after':2},
        Result=lambda before,after,**kw:{'before':before,'after':after},
        ApplyInferences=lambda g,*a,**kw:g.SetVertexValue(0,'inferred',True))
    monkeypatch.setattr(module,'_tgraph_import_reasoner',lambda:reasoner)
    g=graph();ctx=TGraph.AnalysisContext(g)
    first=TGraph.InferOntology(g,returnResult=True)
    assert TGraph.InferOntology(g,returnResult=True)==first and len(calls)==1
    TGraph.Reason(g,applyToGraph=True)
    TGraph.Reason(g,applyToGraph=True)
    assert len(calls)==3 and ctx.Info()['bypasses']>=2

def test_explicit_combined_centralities():
    g=graph();ctx=TGraph.AnalysisContext(g)
    values=ctx.Centralities()
    assert set(values)=={'degree','closeness','betweenness'}
    assert ctx.Info()['namespaces']['ordinary_distances']['hits']==1

@pytest.mark.parametrize('limit',[-1,1.5,True])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):TGraph.AnalysisContext(graph(),maxBytes=limit)
