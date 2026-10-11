import pytest
from topologicpy.TGraph import TGraph

@pytest.mark.parametrize('directed',[False,True])
@pytest.mark.parametrize('radius',[None,0,1])
def test_diamond_and_isolated_vertex(directed,radius):
    g=TGraph(directed=directed)
    for _ in range(5):g.AddVertex()
    for a,b in [(0,1),(0,2),(1,3),(2,3)]:g.AddEdge(a,b)
    options=dict(radius=radius,nxCompatible=False,key=None,colorKey=None,mantissa=None)
    expected=[0.,.5,.5,0.,0.] if directed else [.5,.5,.5,.5,0.]
    edge_expected=[1.5]*4 if directed else [2.]*4
    if radius==0:expected=[0.]*5;edge_expected=[0.]*4
    elif radius==1:expected=[0.]*5;edge_expected=[1.]*4
    assert TGraph.BetweennessCentrality(g,**options)==expected
    assert TGraph.BetweennessCentrality(g,useEdges=True,**options)==edge_expected
    context=TGraph.AnalysisContext(g)
    assert TGraph.BetweennessCentrality(g,**options)==expected
    assert TGraph.BetweennessCentrality(g,useEdges=True,**options)==edge_expected

def test_parallel_edges_keep_distinct_route_shares():
    g=TGraph(directed=True,allowParallelEdges=True,allowSelfLoops=True)
    for _ in range(3):g.AddVertex()
    g.AddEdge(0,1);g.AddEdge(0,1);g.AddEdge(1,2);g.AddEdge(1,1)
    options=dict(nxCompatible=False,key=None,colorKey=None,mantissa=None)
    assert TGraph.BetweennessCentrality(g,**options)==[0.,1.,0.]
    assert TGraph.BetweennessCentrality(g,useEdges=True,**options)==[1.,1.,2.,0.]
