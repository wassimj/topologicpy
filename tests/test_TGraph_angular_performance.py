import ast,random,math,copy,json
from pathlib import Path
import pytest
from topologicpy.TGraph import TGraph

def graph():
    g=TGraph()
    for p in [(0,0,0),(2,0,0),(2,3,0),(6,3,0)]:g.AddVertex(dict(zip(('x','y','z'),p)))
    for pair in [(0,1),(1,2),(2,3)]:g.AddEdge(*pair)
    return g

@pytest.mark.parametrize('radiusType,radius',[('angular',None),('angular',1),('metric',3),('topological',2)])
@pytest.mark.parametrize('weighting',[None,'length'])
def test_combined_matches_individual_methods(radiusType,radius,weighting):
    g=graph();options=dict(radiusType=radiusType,radius=radius,weighting=weighting,mantissa=None)
    expected={key:fn(g,method=method,algorithm='tulip',**options) for key,fn,method in [
        ('choice',TGraph.AngularChoice,'depthmap'),('integration',TGraph.AngularIntegration,'depthmap'),
        ('nain',TGraph.AngularIntegration,'nain'),('nach',TGraph.AngularChoice,'nach')]}
    result=TGraph.AngularTulipAnalysis(g,**options)
    for key in expected:
        assert result[key]==expected[key]
        assert [r['dictionary'][key] for r in g._edges]==result[key]
    assert len(result['nodeCount'])==3

def test_combined_is_one_pass_and_can_avoid_writes(monkeypatch):
    g=graph();before=copy.deepcopy(g._edges)
    original=TGraph._AngularTulipValues;calls=[]
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(TGraph,'_AngularTulipValues',staticmethod(counted))
    result=TGraph.AngularTulipAnalysis(g,writeValues=False)
    assert len(calls)==1 and g._edges==before
    assert result['choice']==[0,2,0]
    assert result['nodeCount']==[3,3,3]
    g._edges[1]['active']=False
    result=TGraph.AngularTulipAnalysis(g,writeValues=False)
    assert result['nodeCount']==[1,1] and result['integration']==[-1,-1]

def test_combined_colors_optional_empty_and_errors():
    g=graph()
    result=TGraph.AngularTulipAnalysis(g,colorize=False)
    assert all('bc_color' not in r['dictionary'] and 'cc_color' not in r['dictionary'] for r in g._edges)
    assert TGraph.AngularTulipAnalysis(TGraph())=={key:[] for key in ('choice','integration','nain','nach','nodeCount','totalDepth','totalWeight')}
    assert TGraph.AngularTulipAnalysis(g,radius=1.5,silent=True) is None

@pytest.mark.parametrize('run_index',range(9))
def test_combined_matches_saved_depthmap_configuration(run_index):
    root=Path(__file__).parent
    geometry=root/'fixtures/depthmap/barnsbury_segment_connections.json'
    reference=root/'fixtures/depthmap/barnsbury_tulip_reference.json'
    data=json.loads(geometry.read_text());run=json.loads(reference.read_text())['runs'][run_index]
    g=TGraph();points=[]
    for p,q in data['endpoints']:
        ends=[]
        for point in (p,q):
            i=next((i for i,v in enumerate(points) if math.dist(point,v)<=1e-5),None)
            if i is None:
                i=len(points);points.append(point);g.AddVertex(dict(zip(('x','y','z'),point)))
            ends.append(i)
        g.AddEdge(*ends)
    result=TGraph.AngularTulipAnalysis(g,**{k:run[k] for k in ('radius','radiusType','weighting','tulipBins')},mantissa=None,writeValues=False)
    for key in ('choice','integration','nain','nach'):
        assert result[key]==pytest.approx(run['expected'][key],rel=2e-7,abs=2e-7)
    assert result['nodeCount']==run['expected']['nodeCount']

def test_named_scale_resolved_once(monkeypatch):
    from topologicpy.Color import Color
    original=Color.ColorScale;names=[]
    def counted(scale='viridis',silent=False):
        if isinstance(scale,str):names.append(scale)
        return original(scale,silent=silent)
    monkeypatch.setattr(Color,'ColorScale',staticmethod(counted))
    TGraph._AngularWriteValues([{} for _ in range(5)],[0,.1,.2,.5,1],None,'color','viridis')
    assert names==['viridis']

@pytest.mark.parametrize('scale',['viridis','syntax','syntax_r','default',[[0,'#000000'],[1,'#ffffff']]])
@pytest.mark.parametrize('mode',['linear','log','sqrt'])
def test_color_values_unchanged(scale,mode):
    from topologicpy.Color import Color
    values=[-1,0,.1,.5,1]
    records=[{} for _ in values]
    TGraph._AngularWriteValues(records,values,'value','color',scale,colorScaleMode=mode)
    expected=[]
    for v in values:
        if v<0:expected.append('#7f7f7f');continue
        ratio=math.log1p(v)/math.log(2) if mode=='log' else (math.sqrt(v) if mode=='sqrt' else v)
        expected.append(Color.AnyToHex(Color.ByValueInRange(ratio,colorScale=scale)))
    assert [r['dictionary']['color'] for r in records]==expected

if __name__=='__main__':
    raise SystemExit(pytest.main([__file__,'-q','-p','no:cacheprovider','-o','addopts=']))
