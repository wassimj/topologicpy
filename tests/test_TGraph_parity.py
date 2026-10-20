"""Behavioral parity additions: exercise algorithms and public result contracts."""
import ast
import csv
import io
import math
import sys
import types
from pathlib import Path

import pytest
from topologicpy.TGraph import TGraph


def graph(points=None, pairs=None, **kwargs):
    points = points or [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    attributes = kwargs.pop('vertexDictionaries', [{} for _ in points])
    attributes = [dict(d, **dict(zip(('x','y','z'),p))) for p,d in zip(points,attributes)]
    return TGraph.ByEdgeIndexPairs(len(points), pairs if pairs is not None else [[0,1],[1,2]],
                                  vertexDictionaries=attributes, ontology=False, **kwargs)


def operands():
    a = graph(vertexDictionaries=[{'id':'a','name':'A'}, {'id':'b','name':'old'}, {'id':'c'}],
              edgeDictionaries=[{'style':'old'},{'style':'original'}])
    b = graph([[10,0,0],[11,0,0],[12,0,0]], [[0,1],[1,2]],
              vertexDictionaries=[{'id':'a','name':'new'}, {'id':'b','extra':1}, {'id':'d'}],
              edgeDictionaries=[{'style':'new'},{'style':'added'}])
    return a,b


def test_overlay_overwrite_and_preserve_inputs():
    a,b=operands()
    before_a=TGraph.ToPython(a); before_b=TGraph.ToPython(b)
    imposed=TGraph.Impose(a,b,vertexKeys=['id'])
    imprinted=TGraph.Imprint(a,b,vertexKeys=['id'])
    assert (TGraph.Order(imposed),TGraph.Size(imposed)) == (4,3)
    assert (TGraph.Order(imprinted),TGraph.Size(imprinted)) == (3,2)
    assert TGraph.Vertices(imprinted)[0]['dictionary']['name']=='new'
    assert TGraph.Coordinates(imprinted,0)==[0,0,0]
    assert TGraph.Coordinates(imposed,0)==[10,0,0]
    assert TGraph.Edges(imprinted)[0]['dictionary']['style']=='new'
    assert TGraph.Edges(imprinted)[1]['dictionary']['style']=='original'
    assert TGraph.ToPython(a)==before_a and TGraph.ToPython(b)==before_b


@pytest.mark.parametrize('method,size,order', [('Union',3,4),('Merge',3,4),('Intersect',1,2),
    ('Difference',1,3),('SymmetricDifference',2,4),('XOR',2,4)])
def test_matching_options_consistent(method,size,order):
    a,b=operands();g=getattr(TGraph,method)(a,b,vertexKeys=['id'])
    assert TGraph.Size(g)==size and TGraph.Order(g)==order
    assert all(e['src'] in TGraph.ActiveVertexIndices(g) and e['dst'] in TGraph.ActiveVertexIndices(g) for e in TGraph.Edges(g))


def test_coordinate_tolerance_centroid_and_ambiguity():
    a=graph([[0,0,0],[1,0,0]],[[0,1]])
    b=graph([[0.02,0,0],[1.02,0,0]],[[0,1]])
    g=TGraph.Union(a,b,matchBy='coordinates',tolerance=.03,useCentroid=True)
    assert TGraph.Order(g)==2 and TGraph.Size(g)==1
    assert TGraph.Coordinates(g,0)==pytest.approx([.01,0,0])
    assert TGraph.Order(TGraph.Union(a,b,matchBy='coordinates',tolerance=.001))==4
    duplicate=graph([[0,0,0],[0,0,0]],[],vertexDictionaries=[{'id':'same'},{'id':'same'}])
    single=graph([[0,0,0]],[],vertexDictionaries=[{'id':'same'}])
    with pytest.raises(ValueError,match='Ambiguous'):
        TGraph.Intersect(duplicate,single,vertexKeys='id',ambiguous='error')
    with pytest.warns(RuntimeWarning,match='Ambiguous'):
        assert TGraph.Order(TGraph.Union(duplicate,single,vertexKeys='id'))==3
    with pytest.raises(ValueError):TGraph.Union(a,b,tolerance=-1)


def test_parallel_edges_and_direction_survive_overlay():
    a=graph([[0,0,0],[1,0,0]],[[0,1],[0,1],[1,0]],directed=True,allowParallelEdges=True)
    b=graph([[0,0,0],[1,0,0]],[[0,1]],directed=True)
    g=TGraph.Imprint(a,b,matchBy='coordinates')
    assert TGraph.Size(g)==3
    assert sum(e['src']==1 for e in TGraph.Edges(g))==1


def test_k_shortest_weighted_parallel_filters_and_records():
    g=graph([[0,0,0],[1,0,0],[0,1,0],[2,0,0]],[[0,1],[1,3],[0,2],[2,3],[0,3]],
            directed=True,edgeDictionaries=[{'w':1},{'w':1},{'w':1},{'w':2},{'w':9}])
    paths=TGraph.KShortestPaths(g,0,3,k=5,edgeKey='w',returnEdges=True,returnCost=True)
    assert [p[0] for p in paths]==[[0,1,3],[0,2,3],[0,3]]
    assert [p[-1] for p in paths]==[2,3,9]
    assert [[e['index'] for e in p[1]] for p in paths]==[[0,1],[2,3],[4]]
    assert TGraph.KShortestPaths(g,3,0,edgeKey='w')==[]
    assert TGraph.KShortestPaths(g,0,3,vertexFilter=lambda v:v['index']!=1,edgeKey='w')[0]==[0,2,3]
    assert TGraph.KShortestPaths(g,0,3,edgeFilter=lambda e:e['index']!=4,edgeKey='w')==[[0,1,3],[0,2,3]]
    assert TGraph.KShortestPaths(g,0,0)==[[0]]
    assert TGraph.KShortestPaths(g,0,3,k=0)==[]
    assert TGraph.KShortestPaths(g,0,3,timeLimit=0,returnInfo=True)['timedOut']
    assert TGraph.KShortestPaths(g,0,3,k=1,returnVertices=True)[0][1][0]['index']==0
    g._edges[0]['dictionary']['w']=-1
    with pytest.raises(ValueError):TGraph.KShortestPaths(g,0,3,edgeKey='w')


def test_k_shortest_matches_exhaustive_simple_paths():
    import random
    rng=random.Random(41)
    for _ in range(15):
        pairs=[[u,v] for u in range(5) for v in range(5) if u!=v and rng.random()<.35]
        weights=[{'w':rng.randrange(0,5)} for _ in pairs]
        g=graph([[i,0,0] for i in range(5)],pairs,directed=True,edgeDictionaries=weights)
        expected=[]
        def visit(u,path,cost):
            if u==4:expected.append((cost,list(path)));return
            for e in TGraph.IncidentEdges(g,u,mode='out'):
                v=e['dst']
                if v not in path:visit(v,path+[v],cost+e['dictionary']['w'])
        visit(0,[0],0)
        actual=TGraph.KShortestPaths(g,0,4,k=100,edgeKey='w',returnCost=True)
        assert sorted((p[1],p[0]) for p in actual)==sorted(expected)
        assert [p[1] for p in actual]==sorted(p[1] for p in actual)


def test_longest_exact_and_search_status():
    g=graph([[0,0,0],[1,0,0],[0,1,0]],[[0,1],[1,2],[2,0]])
    assert len(TGraph.LongestPath(g,strategy='diameter'))==2
    info=TGraph.LongestPath(g,source=0,target=1,strategy='exact',returnInfo=True)
    assert info['path']==[0,2,1] and info['cost']==2 and info['optimal']
    assert not TGraph.LongestPath(g,returnInfo=True)['optimal']
    info=TGraph.LongestPath(g,source=0,target=1,timeLimit=0,returnInfo=True)
    assert info['timedOut'] and not info['completed'] and not info['optimal']
    TGraph.SetEdgeValue(g,0,'w',10)
    assert TGraph.LongestPath(g,source=0,target=1,edgeKey='w')==[0,1]


def test_total_metric_depth_and_disconnected_policy():
    g=graph([[0,0,0],[2,0,0],[5,0,0],[9,0,0]],[[0,1],[1,2]])
    assert TGraph.DepthMap(g,0)=={0:0,1:1,2:2}
    assert TGraph.DepthMap(g,method='total',key=None)=={0:3,1:2,2:3,3:0}
    assert TGraph.DepthMap(g,sources=[0,2],method='total',distanceType='metric',key='total')=={0:7,2:8}
    assert TGraph.DepthMap(g,0,distanceType='metric',key=None)=={0:0,1:2,2:5}
    assert math.isinf(TGraph.DepthMap(g,method='total',unreachable='infinity')[0])
    with pytest.raises(ValueError):TGraph.DepthMap(g,method='total',unreachable='raise')
    assert TGraph.DepthMap(g,0,distanceType='metric',radius=2)=={0:0,1:2}


@pytest.mark.parametrize('remove_vertices',[False,True])
def test_remove_isolated_components_no_cascade(remove_vertices):
    g=graph([[i,0,0] for i in range(6)],[[0,1],[2,3],[3,4],[5,5]])
    TGraph.RemoveIsolatedEdges(g,removeVertices=remove_vertices)
    assert TGraph.Size(g)==3
    assert TGraph.Order(g)==(4 if remove_vertices else 6)
    assert all(e['index']!=0 for e in TGraph.Edges(g))


def test_weighted_laplacian_normalized_and_loops():
    g=graph([[0,0,0],[1,0,0],[2,0,0]],[[0,1],[0,1],[0,0]],allowParallelEdges=True,
            edgeDictionaries=[{'w':2},{'w':3},{'w':5}])
    assert TGraph.Laplacian(g,weightKey='w')==[[5,-5,0],[-5,5,0],[0,0,0]]
    result=TGraph.Laplacian(g,normalized=True,weightKey='w')
    assert result[0]==pytest.approx([.5,-1/math.sqrt(2),0])
    assert result[2]==[0,0,0]
    d=graph([[0,0,0],[1,0,0]],[[0,1]],directed=True)
    assert TGraph.Laplacian(d,mode='out')==[[1,-1],[0,0]]
    assert TGraph.Laplacian(d,mode='in')==[[0,0],[-1,1]]


def test_weighted_degree_strength_and_integration():
    g=graph([[0,0,0],[2,0,0],[5,0,0]],[[0,1],[1,2]],edgeDictionaries=[{'w':2},{'w':3}])
    assert TGraph.Connectivity(g,weightKey='w')==[2,5,3]
    assert TGraph.VertexDegree(g,1,weightKey='w')==5
    values=TGraph.Integration(g,weightKey='w',normalize=False,colorKey=None)
    expected=TGraph.ClosenessCentrality(g,weightKey='w',normalize=False,colorKey=None)
    assert values==expected
    assert TGraph.Integration(g,distanceType='metric',normalize=False,colorKey=None)==expected
    with pytest.raises(ValueError):TGraph.Integration(g,method='hh',weightKey='w')


def test_flow_asymmetric_capacity_details_and_annotations():
    g=graph([[0,0,0],[1,0,0]],[[0,1]],edgeDictionaries=[{'fwd':7,'bwd':2}])
    assert TGraph.MaximumFlow(g,0,1,edgeKeyFwd='fwd',edgeKeyBwd='bwd')==7
    details=TGraph.MaximumFlow(g,1,0,edgeKeyFwd='fwd',edgeKeyBwd='bwd',
                              returnDetails=True,flowKey='flow',residualKey='residual')
    assert details['value']==2
    d=TGraph.Edges(g)[0]['dictionary']
    assert d['flow']['backward']==2 and d['residual']['backward']==0
    assert d['residual']['forward']==7
    assert TGraph.MaximumFlow(g,1,0,edgeKeyFwd='fwd',edgeKeyBwd='bwd',bidirectional=False)==0


def test_matrix_dictionary_validation_directional_weights_and_csv(tmp_path):
    g=TGraph.ByAdjacencyMatrix([[0,7],[2,0]],dictionaries=[{'label':'a,b'},{'label':'"b"'}],
                               edgeKeyFwd='fwd',edgeKeyBwd='bwd',ontology=False)
    assert TGraph.Size(g)==1
    assert TGraph.Edges(g)[0]['dictionary']['bwd']==2
    assert TGraph.MaximumFlow(g,1,0,edgeKeyFwd='fwd',edgeKeyBwd='bwd')==2
    text=TGraph.AdjacencyMatrixCSVString(g,edgeKeyFwd='fwd',edgeKeyBwd='bwd',bidirectional=True)
    assert list(csv.reader(io.StringIO(text)))==[['0','7'],['2','0']]
    path=tmp_path/'matrix.csv';path.write_text('0,7\n2,0\n')
    loaded=TGraph.ByAdjacencyMatrixCSVPath(str(path),dictionaries=[{'id':'a'},{'id':'b'}],
                                         edgeKeyFwd='fwd',edgeKeyBwd='bwd',ontology=False)
    assert TGraph.Size(loaded)==1
    for matrix in ([[0,1],[0]], [[0,float('nan')],[1,0]]):
        with pytest.raises(ValueError):TGraph.ByAdjacencyMatrix(matrix)
    with pytest.raises(ValueError):TGraph.ByAdjacencyMatrix([[0]],dictionaries=[])


def test_sorted_views_annotations_and_compact_mesh():
    g=graph([[i,0,0] for i in range(5)],[[0,1],[1,2],[3,4]],
            vertexDictionaries=[{'rank':3},{'rank':1},{},{'rank':2},{'rank':4}],
            edgeDictionaries=[{'rank':3},{'rank':1},{'rank':2}])
    assert [v['index'] for v in TGraph.Vertices(g,sortBy='rank')]==[1,3,0,4,2]
    assert [v['index'] for v in TGraph.Vertices(g,sortBy='rank',reverse=True)]==[4,0,3,1,2]
    assert [e['index'] for e in TGraph.Edges(g,sortBy='rank')]==[1,2,0]
    assert [v['index'] for v in TGraph.Vertices(g)]==list(range(5))
    assert len(TGraph.Bridges(g,key='bridge'))==3
    assert [v['index'] for v in TGraph.CutVertices(g,key='cut')]==[1]
    assert TGraph.ConnectedComponents(g,key='component')==[[0,1,2],[3,4]]
    assert all(e['dictionary']['component'] in (1,2) for e in TGraph.Edges(g))
    g.RemoveVertex(1)
    data=TGraph.MeshData(g,includeDictionaries=True,includeIndices=True)
    assert data['vertexIndices']==[0,2,3,4] and data['edges']==[[2,3]]
    assert data['edgeIndices']==[2]
    data['vertexDictionaries'][0]['rank']=999
    assert TGraph.VertexDictionary(g,0)['rank']==3


def test_graphviz_styling_escaped_dot_and_object(monkeypatch):
    g=graph(vertexDictionaries=[{'label':'a"\\b','shape':'box'},{'label':'B'},{'label':'C'}],
            edgeDictionaries=[{'color':'red'},{'color':'blue'}])
    dot=TGraph.GraphVizGraph(g,layout='neato',vertexShapeKey='shape',edgeColorKey='color',
                            vertexAttributes={'style':'filled'},graphAttributes={'splines':'ortho'})
    assert 'layout="neato"' in dot and 'shape="box"' in dot and 'color="red"' in dot
    assert 'a\\"\\\\b' in dot
    class Source:
        def __init__(self,source,engine):self.source=source;self.engine=engine
    monkeypatch.setitem(sys.modules,'graphviz',types.SimpleNamespace(Source=Source))
    obj=TGraph.GraphVizGraph(g,asObject=True,layout='neato')
    assert obj.engine=='neato' and obj.source.startswith('graph')
    with pytest.raises(ValueError):TGraph.GraphVizGraph(g,vertexAttributes={'x];danger':'bad'})


def test_pyvis_styling_explicit_output_and_missing_dependency(monkeypatch):
    class Network:
        def __init__(self,**kwargs):self.options=kwargs;self.nodes=[];self.edges=[]
        def add_node(self,index,**kwargs):self.nodes.append((index,kwargs))
        def add_edge(self,source,target,**kwargs):self.edges.append((source,target,kwargs))
    monkeypatch.setitem(sys.modules,'pyvis',types.ModuleType('pyvis'))
    monkeypatch.setitem(sys.modules,'pyvis.network',types.SimpleNamespace(Network=Network))
    g=graph(vertexDictionaries=[{'size':12,'color':'red','group':'room','label':'<script>'},{},{}],
            edgeDictionaries=[{'capacity':8},{'capacity':2}])
    net=TGraph.PyvisGraph(g,vertexSizeKey='size',vertexColorKey='color',vertexGroupKey='group',
                          edgeWeightKey='capacity',selectMenu=True,filterMenu=True)
    assert net.nodes[0][1]['size']==12 and net.nodes[0][1]['group']=='room'
    assert '&lt;script&gt;' in net.nodes[0][1]['title']
    assert net.edges[0][2]['value']==8 and net.options['select_menu']
    monkeypatch.setitem(sys.modules,'pyvis.network',None)
    with pytest.raises(ImportError,match='topologicpy\\[graph\\]'):TGraph.PyvisGraph(g)


def test_angular_betweenness_alias_marks_correct_records():
    g=graph([[0,0,0],[1,0,0],[2,0,0],[2,1,0]],[[0,1],[1,2],[2,3]])
    expected=TGraph.AngularChoice(g,normalize=True,key='expected')
    actual=TGraph.AngularBetweenness(g,key='alias',colorKey=None)
    assert actual==expected
    assert [e['dictionary']['alias'] for e in TGraph.Edges(g)]==expected


def test_exact_longest_handles_long_chains_without_recursion():
    n=1200
    g=graph([[i,0,0] for i in range(n)],[[i,i+1] for i in range(n-1)])
    result=TGraph.LongestPath(g,source=0,target=n-1,timeLimit=5,returnInfo=True)
    assert result['path']==list(range(n)) and result['optimal']


def test_parallel_zero_cost_alternative_routes_and_inactive_indices():
    g=graph([[i,0,0] for i in range(4)],[[0,2],[0,2],[2,3],[0,3]],
            directed=True,allowParallelEdges=True,
            edgeDictionaries=[{'w':0},{'w':1},{'w':0},{'w':3}])
    g.RemoveVertex(1)
    result=TGraph.KShortestPaths(g,0,3,edgeKey='w',returnEdges=True,returnCost=True)
    assert [p[-1] for p in result]==[0,1,3]
    assert [[e['index'] for e in p[1]] for p in result]==[[0,2],[1,2],[3]]
    assert TGraph.DepthMap(g,method='total',weightKey='w',sources=[0],key=None)=={0:0}


def test_sorted_geometric_views_keep_record_incidence():
    from topologicpy.Vertex import Vertex
    from topologicpy.Edge import Edge
    g=graph([[0,0,0],[1,0,0],[2,0,0]],[[0,1],[1,2]],
            vertexDictionaries=[{'rank':2},{'rank':3},{'rank':1}],
            edgeDictionaries=[{'rank':2},{'rank':1}])
    vertices=TGraph.Vertices(g,sortBy='rank',asTopologic=True)
    assert [Vertex.X(v) for v in vertices]==[2,0,1]
    edges=TGraph.Edges(g,sortBy='rank',asTopologic=True)
    assert [Vertex.X(Edge.StartVertex(e)) for e in edges]==[1,0]
    assert TGraph.MeshData(g)['edges']==[[0,1],[1,2]]


def test_degree_self_loop_and_directed_strength():
    g=graph([[0,0,0],[1,0,0]],[[0,0],[0,1]],directed=True,
            edgeDictionaries=[{'w':3},{'w':2}])
    assert TGraph.VertexDegree(g,0,mode='all',weightKey='w')==8
    assert TGraph.VertexDegree(g,0,mode='out',weightKey='w')==5
    assert TGraph.VertexDegree(g,0,mode='in',weightKey='w')==3


def test_legacy_union_positional_silent_and_invalid_depth():
    a,b=operands()
    assert isinstance(TGraph.Union(a,b,True),TGraph)
    assert TGraph.DepthMap(None,method='total')=={}


def test_excluded_json_and_compilation_contracts_unchanged():
    g=graph()
    assert isinstance(TGraph.JSONData(g),dict)
    assert isinstance(TGraph.JSONString(g),str)
    assert isinstance(TGraph.Compile(g),dict)
