import math
import random
import pytest
from topologicpy.TGraph import TGraph

def graph_by_segments(endpoints, directed=False):
    graph = TGraph(directed=directed)
    vertices = {}
    for p, q in endpoints:
        for point in (p, q):
            if point not in vertices:
                vertices[point] = graph.AddVertex(dict(zip(('x','y','z'),point)))
        graph.AddEdge(vertices[p],vertices[q])
    return graph

def chain():
    return graph_by_segments([((0,0,0),(2,0,0)),((2,0,0),(2,3,0)),((2,3,0),(6,3,0))])

def test_formulas_and_ordered_choice():
    g=chain()
    assert TGraph.AngularIntegration(g,method='nain',mantissa=None)==pytest.approx([3**1.2/5,3**1.2/4,3**1.2/5])
    assert TGraph.AngularChoice(g,method='depthmap')==[0,2,0]
    assert TGraph.AngularChoice(g,normalize=False)==[0,1,0]
    assert TGraph.AngularChoice(g,method='nach',mantissa=None)==pytest.approx([0,math.log(3)/math.log(5),0])
    assert TGraph.AngularChoice(g,method='nach',normalize=False)==TGraph.AngularChoice(g,method='nach')
    assert TGraph.AngularIntegration(g,method='nain',normalize=False)==TGraph.AngularIntegration(g,method='nain')

def test_length_weighting_hand_calculation():
    g=chain()
    assert TGraph.AngularIntegration(g,method='depthmap',weighting='length',mantissa=None)==pytest.approx([81/11,81/6,81/7])
    assert TGraph.AngularIntegration(g,method='nain',weighting='length',mantissa=None)==pytest.approx([9**1.2/13,9**1.2/8,9**1.2/9])
    # Ordered endpoint halves: A=6+8; B=6+12+2*8; C=8+12.
    assert TGraph.AngularChoice(g,method='depthmap',weighting='length')==[14,34,20]
    assert TGraph.AngularChoice(g,method='nach',weighting='length',mantissa=None)==pytest.approx(
        [math.log(15)/math.log(14),math.log(35)/math.log(9),math.log(21)/math.log(10)])

@pytest.mark.parametrize('radiusType,radius,expected',[
    ('topological',0,[-1,-1,-1]),('topological',1,[4,4.5,4]),
    ('topological',2,[3,4.5,3]),('metric',2.5,[4,4,-1]),
    ('metric',3.5,[4,4.5,4]),('metric',6,[3,4.5,3]),
    ('angular',1,[4,4.5,4])])
def test_radius_boundaries(radiusType,radius,expected):
    assert TGraph.AngularIntegration(chain(),method='depthmap',radiusType=radiusType,radius=radius)==expected

@pytest.mark.parametrize('method',['depthmap','nain','closeness'])
def test_global_radius_type_does_not_change_results(method):
    g=chain()
    assert TGraph.AngularIntegration(g,method=method,radiusType='metric')==TGraph.AngularIntegration(g,method=method)
    assert TGraph.AngularIntegration(g,method=method,radiusType='topological')==TGraph.AngularIntegration(g,method=method)

def test_isolated_and_zero_depth():
    g=graph_by_segments([((0,0,0),(1,0,0)),((1,0,0),(2,0,0)),((5,0,0),(6,0,0))])
    assert TGraph.AngularIntegration(g,method='nain',mantissa=None)==pytest.approx([2**1.2/2,2**1.2/2,-1])
    assert TGraph.AngularChoice(g,method='nach')==[0,0,0]
    assert g._edges[2]['dictionary']['cc_color']=='#7f7f7f'

@pytest.mark.parametrize('kwargs',[{'radiusType':'bad'},{'weighting':'bad'},{'radius':True},{'radius':-1},
    {'radius':float('nan')},{'radius':float('inf')},{'mode':'bad'},{'method':'bad'}])
def test_validation(kwargs):
    assert TGraph.AngularIntegration(chain(),silent=True,**kwargs) is None
    assert TGraph.AngularChoice(chain(),silent=True,**kwargs) is None

def test_empty_graph_and_inactive_records():
    assert TGraph.AngularIntegration(TGraph(),method='nain')==[]
    assert TGraph.AngularChoice(TGraph(),method='nach',radiusType='metric',radius=5)==[]
    g=chain()
    g._edges[2]['active']=False
    assert len(TGraph.AngularIntegration(g,method='nain',weighting='length'))==2
    assert 'integration' not in g._edges[2]['dictionary']

def test_segment_graph_and_directed_modes():
    from topologicpy.Edge import Edge
    from topologicpy.Vertex import Vertex
    endpoints=[((0,0,0),(2,0,0)),((2,0,0),(2,3,0)),((2,3,0),(6,3,0))]
    sg=TGraph.SegmentGraph([Edge.ByVertices(Vertex.ByCoordinates(*p),Vertex.ByCoordinates(*q)) for p,q in endpoints])
    for fn,method in [(TGraph.AngularIntegration,'nain'),(TGraph.AngularChoice,'nach')]:
        for radiusType,radius in [('metric',3.5),('topological',2),('angular',None)]:
            kwargs=dict(method=method,weighting='length',radiusType=radiusType,radius=radius)
            assert fn(sg,**kwargs)==fn(chain(),**kwargs)
    assert all('integration' in v['dictionary'] for v in sg._vertices)
    g=graph_by_segments(endpoints,directed=True)
    assert TGraph.AngularIntegration(g,method='depthmap',mode='out')==[3,4,-1]
    assert TGraph.AngularIntegration(g,method='depthmap',mode='in')==[-1,4,3]
    assert TGraph.AngularChoice(g,method='depthmap',mode='out')==[0,1,0]

@pytest.mark.parametrize('radiusType',['metric','topological'])
def test_pareto_retains_feasible_higher_angular_route(radiusType):
    # State 3 needs two labels: the cheaper angular path consumes the radius.
    states=[(i,1) for i in range(6)]
    adj=[[(1,0),(3,2)],[(2,0)],[(3,0)],[(4,1)],[(5,1)],[]]
    lengths=[1]*6
    result=TGraph._AngularBoundedPaths(states,[[i] for i in range(6)],adj,lengths,0,3,radiusType)
    _,groups,dist,_,_,_,_,_=result
    assert min(dist[i] for i in groups[5])==4

@pytest.mark.parametrize('radiusType',['metric','topological'])
@pytest.mark.parametrize('seed',range(12))
def test_bounded_paths_against_exhaustive_simple_routes(seed,radiusType):
    rng=random.Random(seed)
    n=6;states=[(i,1) for i in range(n)];starts=[[i] for i in range(n)]
    # Directed acyclic networks permit exhaustive independent enumeration.
    adj=[[(j,rng.choice([0,0.5,1])) for j in range(i+1,n) if rng.random()<0.6] for i in range(n)]
    lengths=[rng.choice([1,2,3]) for _ in range(n)];radius=4
    for source in range(n):
        paths=[[] for _ in range(n)]
        def walk(v,angle,resource,path):
            paths[v].append((angle,resource,len(path)-1,path))
            for w,cost in adj[v]:
                step=1 if radiusType=='topological' else (lengths[v]+lengths[w])/2
                if resource+step<=radius:
                    walk(w,angle+cost,resource+step,path+[w])
        walk(source,0,0,[source])
        _,groups,dist,hops,sigma,pred,order,resources=TGraph._AngularBoundedPaths(states,starts,adj,lengths,source,radius,radiusType)
        for target in range(n):
            if not paths[target]:
                assert not groups[target]
                continue
            best=min(row[:3] for row in paths[target])
            selected=[i for i in groups[target] if (dist[i],resources[i],hops[i])==best]
            assert selected
            assert sum(sigma[i] for i in selected)==sum(row[:3]==best for row in paths[target])

@pytest.mark.parametrize('radiusType,radius',[('metric',4),('topological',3)])
@pytest.mark.parametrize('weighting',[None,'length'])
def test_bounded_centrality_against_independent_route_enumeration(radiusType,radius,weighting):
    endpoints=[((0,0,0),(1,0,0)),((1,0,0),(2,0,0)),((2,0,0),(2,1,0)),
               ((1,0,0),(1,2,0)),((1,2,0),(2,1,0)),((2,1,0),(3,1,0))]
    graph=graph_by_segments(endpoints)
    _,states,starts,adj,_=TGraph._AngularStateData(graph)
    n=len(endpoints);lengths=[math.dist(p,q) for p,q in endpoints]
    weights=lengths if weighting else [1]*n
    depths=[];nain=[];choice=[0.0]*n
    for source in range(n):
        routes=[[] for _ in range(n)]
        def visit(state,angle,resource,path):
            segment=states[state][0]
            if segment!=source:
                routes[segment].append((angle,resource,len(path)-1,path))
            for target,cost in adj[state]:
                if states[target][0] in [states[s][0] for s in path]:
                    continue
                step=1 if radiusType=='topological' else (lengths[segment]+lengths[states[target][0]])/2
                if resource+step<=radius+1e-12:
                    visit(target,angle+cost,resource+step,path+[target])
        for root in starts[source]:
            visit(root,0,0,[root])
        depth=0;mass=weights[source];reached=0
        for target,paths in enumerate(routes):
            if not paths:
                continue
            best=min(p[:3] for p in paths)
            paths=[p for p in paths if p[:3]==best]
            reached+=1;mass+=weights[target];depth+=weights[target]*best[0]
            demand=weights[source]*weights[target]
            for p in paths:
                for state in p[3][1:-1]:
                    choice[states[state][0]]+=demand/len(paths)
            if weighting:
                choice[source]+=demand/2;choice[target]+=demand/2
        depths.append(depth)
        nain.append(mass**1.2/(depth+2) if reached else -1)
    kwargs=dict(radiusType=radiusType,radius=radius,weighting=weighting,mantissa=None)
    assert TGraph.AngularIntegration(graph,method='nain',**kwargs)==pytest.approx(nain)
    assert TGraph.AngularChoice(graph,method='depthmap',**kwargs)==pytest.approx(choice)
    assert TGraph.AngularChoice(graph,method='nach',**kwargs)==pytest.approx(
        [math.log1p(c)/math.log(d+3) for c,d in zip(choice,depths)])
