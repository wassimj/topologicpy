"""Large saved DepthmapX references and float32 queue-order regressions."""
import json
from pathlib import Path
import pytest
from topologicpy.TGraph import TGraph

FIXTURES = Path(__file__).parent / 'fixtures' / 'depthmap'

@pytest.mark.parametrize('run_index', range(3))
def test_extended_barnsbury_saved_depthmap_output(run_index):
    data = json.loads((FIXTURES / 'barnsbury_extended_tulip_reference.json').read_text())
    graph = TGraph.ByDepthmapSegmentData(data['segments'], data['connections'])
    assert graph is not None
    run = data['runs'][run_index]
    context = TGraph.AnalysisContext(graph)
    options = {k: run[k] for k in ('radius', 'radiusType', 'weighting', 'tulipBins')}
    result = TGraph.AngularTulipAnalysis(graph, **options, mantissa=None, writeValues=False)
    assert len(result['choice']) == 3434
    for key in ('choice', 'integration', 'nodeCount'):
        assert result[key] == pytest.approx(run['expected'][key], rel=2e-7, abs=2e-7)
    assert TGraph.AngularTulipAnalysis(graph, **options, mantissa=None, writeValues=False) == result
    assert context.Info()['namespaces']['AngularTulipAnalysis']['hits'] == 1

@pytest.mark.parametrize('case_index', range(12))
def test_float32_metric_tie_order_at_exactness_boundary(case_index):
    data = json.loads((FIXTURES / 'tulip_float32_regression.json').read_text())
    case = data['cases'][case_index]
    assert TGraph._AngularTulipValues(data['transitions'], case['lengths'], **case['options']) == case['expected']

@pytest.mark.parametrize('change', ['geometry', 'stored_length', 'stored_turn', 'topology'])
def test_angular_cache_live_edits_match_fresh_analysis(change):
    segments = [dict(Ref=i, x1=p[0], y1=p[1], x2=q[0], y2=q[1], **{'Segment Length':length})
                for i, (p,q,length) in enumerate([((0,0),(2,0),2), ((2,0),(2,3),3), ((2,3),(6,3),4)])]
    connections = [dict(refA=a, refB=b, ss_weight=1, for_back=d, dir=e)
                   for a,b,d,e in [(0,1,0,1),(1,0,1,-1),(1,2,0,1),(2,1,1,-1)]]
    if change == 'geometry':
        graph = TGraph()
        for x,y in [(0,0),(2,0),(2,3),(6,3)]:
            graph.AddVertex(dict(x=x,y=y,z=0))
        for a,b in [(0,1),(1,2),(2,3)]:
            graph.AddEdge(a,b)
    else:
        graph = TGraph.ByDepthmapSegmentData(segments, connections)
    context = TGraph.AnalysisContext(graph)
    options = dict(radius=6, radiusType='metric', weighting='length', writeValues=False, mantissa=None)
    before = TGraph.AngularTulipAnalysis(graph, **options)
    if change == 'geometry':
        graph._vertices[0]['dictionary']['x'] = -.25
    elif change == 'stored_length':
        graph._dictionary['depthmap_segment_data']['lengths']['0'] = 5
    elif change == 'stored_turn':
        graph._dictionary['depthmap_segment_data']['transitions'][0][-1] = .5
    else:
        graph.RemoveEdge(0)
    actual = TGraph.AngularTulipAnalysis(graph, **options)
    assert actual != before
    assert context.Info()['namespaces']['angular_tulip']['misses'] == 2
    context.Detach()
    assert actual == TGraph.AngularTulipAnalysis(graph, **options)
