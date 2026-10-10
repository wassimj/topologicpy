import math,json
from pathlib import Path
import pytest
from topologicpy.TGraph import TGraph

def rows():
    segments=[dict(Ref=10,x1=0,y1=0,x2=2,y2=0,**{'Segment Length':2}),
        dict(Ref=30,x1=2.0002,y1=0,x2=2.0002,y2=3,**{'Segment Length':3}),
        dict(Ref=50,x1=2.0002,y1=3,x2=6.0002,y2=3,**{'Segment Length':4})]
    connections=[dict(refA=10,refB=30,ss_weight=1,for_back=0,dir=1),
        dict(refA=30,refB=10,ss_weight=1,for_back=1,dir=-1),
        dict(refA=30,refB=50,ss_weight=1,for_back=0,dir=1),
        dict(refA=50,refB=30,ss_weight=1,for_back=1,dir=-1)]
    return segments,connections

def test_import_preserves_gap_link_saved_length_and_reference_order():
    segments,connections=rows()
    segments[0]['Segment Length']=5
    g=TGraph.ByDepthmapSegmentData(list(reversed(segments)),connections)
    assert g is not None
    assert [v['dictionary']['depthmap_ref'] for v in g._vertices]==[10,30,50]
    options=dict(method='depthmap',algorithm='tulip',mantissa=None)
    assert TGraph.AngularChoice(g,**options)==[0,2,0]
    assert TGraph.AngularChoice(g,weighting='length',**options)==[35,67,32]
    assert TGraph.AngularIntegration(g,weighting='length',**options)==pytest.approx([144/11,144/9,144/13])
    # The default exact engine also honours imported topology and saved lengths.
    assert TGraph.AngularChoice(g,method='depthmap',weighting='length')==[35,67,32]

def test_import_copy_relationship_removal_and_inactive_nodes():
    g=TGraph.ByDepthmapSegmentData(*rows())
    copied=TGraph.Copy(g)
    assert TGraph.AngularChoice(copied,method='depthmap',algorithm='tulip')==[0,2,0]
    copied.RemoveEdge(0)
    assert TGraph.AngularChoice(copied,method='depthmap',algorithm='tulip',weighting='length')==[0,12,12]
    g._vertices[1]['active']=False
    assert TGraph.AngularIntegration(g,method='nain',algorithm='tulip')==[-1,-1]
    assert TGraph.AngularChoice(g,method='depthmap',algorithm='tulip')==[0,0]

@pytest.mark.parametrize('kind',['duplicate_ref','unknown_ref','invalid_side','invalid_direction',
    'negative_cost','nan_cost','bad_length','zero_geometry','duplicate_connection','missing_reverse'])
def test_import_rejects_invalid_data(kind):
    segments,connections=rows()
    if kind=='duplicate_ref':segments.append(dict(segments[0]))
    elif kind=='unknown_ref':connections[0]['refB']=99
    elif kind=='invalid_side':connections[0]['for_back']=2
    elif kind=='invalid_direction':connections[0]['dir']=0
    elif kind=='negative_cost':connections[0]['ss_weight']=-1
    elif kind=='nan_cost':connections[0]['ss_weight']=float('nan')
    elif kind=='bad_length':segments[0]['Segment Length']=0
    elif kind=='zero_geometry':segments[0]['x2']=0
    elif kind=='duplicate_connection':connections.append(dict(connections[0]))
    elif kind=='missing_reverse':connections.pop()
    assert TGraph.ByDepthmapSegmentData(segments,connections,silent=True) is None

def test_empty_import():
    g=TGraph.ByDepthmapSegmentData([],[])
    assert TGraph.AngularChoice(g,method='depthmap',algorithm='tulip')==[]

def test_import_original_barnsbury_connections():
    root=Path(__file__).parent
    geometry=root/'fixtures/depthmap/barnsbury_segment_connections.json'
    reference=root/'fixtures/depthmap/barnsbury_tulip_reference.json'
    fixture=json.loads(geometry.read_text())
    segments=[dict(Ref=i,x1=p[0],y1=p[1],x2=q[0],y2=q[1],
                   **{'Segment Length':math.dist(p,q)}) for i,(p,q) in enumerate(fixture['endpoints'])]
    connections=[dict(refA=a,refB=b,ss_weight=w,for_back=0 if d==1 else 1,dir=e)
                 for a,d,b,e,w in fixture['transitions']]
    g=TGraph.ByDepthmapSegmentData(segments,connections)
    assert g is not None
    assert TGraph.AngularConnectivity(g,method='depthmap',mantissa=None)==pytest.approx(fixture['angular_connectivity'],rel=2e-6)
    expected=json.loads(reference.read_text())['runs'][0]['expected']
    options=dict(algorithm='tulip',method='depthmap',mantissa=None)
    assert TGraph.AngularChoice(g,**options)==expected['choice']
    assert TGraph.AngularIntegration(g,**options)==pytest.approx(expected['integration'],rel=1e-7)

if __name__=='__main__':
    raise SystemExit(pytest.main([__file__,'-q','-p','no:cacheprovider','-o','addopts=']))
