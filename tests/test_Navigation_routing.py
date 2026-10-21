"""Navigation query endpoints must respect saved physical support."""
import copy
import math
import pytest
from topologicpy.Navigation import Navigation
from topologicpy.TGraph import TGraph
from topologicpy.Vertex import Vertex
from test_Navigation import rectangle


def test_node_queries_delegate_and_do_not_mutate(monkeypatch):
    model = Navigation.ByFaces([rectangle()])
    graph = Navigation.Graph(model)
    before = copy.deepcopy(TGraph.ToPython(graph))
    original = TGraph.ShortestPath
    calls = []
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(TGraph, 'ShortestPath', spy)
    for _ in range(2):
        result = Navigation.ShortestPath(model, graph, 0, 15)
        assert result['status'] == 'ok'
        assert result['length'] == pytest.approx(6)
        assert result['cost'] == pytest.approx(6)
        assert result['nodeIDs'][0] == 0 and result['nodeIDs'][-1] == 15
        assert result['wire'] is not None
        assert result['snapDistances'] == [0, 0]
    assert len(calls) == 2
    assert TGraph.ToPython(graph) == before
    result = Navigation.ShortestPath(model, graph, 0, 0)
    assert result['coordinates'] == [[.5, .5, 0]]
    assert result['length'] == 0 and result['wire'] is None


def test_coordinate_and_vertex_endpoints_include_connection_cost():
    model = Navigation.ByFaces([rectangle()])
    graph = Navigation.Graph(model)
    result = Navigation.ShortestPath(model, graph, [.1, .2, 0], Vertex.ByCoordinates(3.8, 3.9, 0))
    assert result['status'] == 'ok'
    assert result['coordinates'][0] == [.1, .2, 0]
    assert result['coordinates'][-1] == [3.8, 3.9, 0]
    assert result['length'] == pytest.approx(math.dist([.1, .2], [3.8, 3.9]))
    assert result['cost'] == pytest.approx(result['length'])
    assert result['nodeIDs'] == [None, None]


def test_wall_blocks_nearest_node_and_disconnected_route():
    model = Navigation.ByFaces([rectangle()], obstacles=[rectangle(x=1.8, w=.4)])
    graph = Navigation.Graph(model)
    result = Navigation.ShortestPath(model, graph, [1.7, .4, 0], [2.3, .4, 0])
    assert result['status'] == 'disconnected'
    result = Navigation.ShortestPath(model, graph, [2, .4, 0], 0)
    assert result['status'] == 'endpoint_unreachable'


def test_route_around_obstacle_has_supported_segments():
    from shapely.geometry import LineString
    model = Navigation.ByFaces([rectangle()], obstacles=[rectangle(x=1.8, y=1, w=.4, h=2)])
    graph = Navigation.Graph(model, spacing=.5)
    result = Navigation.ShortestPath(model, graph, [.7, 2, 0], [3.3, 2, 0])
    assert result['status'] == 'ok'
    obstacle = Navigation._Shape(Navigation._Record(model.obstacles[0]))
    assert result['length'] > 2.6
    assert all(not LineString([a[:2], b[:2]]).crosses(obstacle)
               for a, b in zip(result['coordinates'], result['coordinates'][1:]))


def test_clearance_snapshot_and_snap_limit():
    model = Navigation.ByFaces([rectangle()])
    graph = Navigation.Graph(model, clearance=.3)
    assert Navigation.ShortestPath(model, graph, [.1, .5, 0], 0)['status'] == 'endpoint_unreachable'
    assert Navigation.ShortestPath(model, graph, [.31, .31, 0], 0,
                                   maxSnapDistance=.01)['status'] == 'endpoint_unreachable'
    model.faces = []
    assert Navigation.ShortestPath(model, graph, [.4, .4, 0], 0)['status'] == 'ok'
    restored = TGraph.Copy(graph)
    assert Navigation.ShortestPath(model, restored, [.4, .4, 0], 0)['status'] == 'ok'


def test_coincident_storeys_require_explicit_identifier():
    model = Navigation.ByFaces([rectangle(), rectangle()], storeyIDs=['a', 'b'])
    graph = Navigation.Graph(model)
    assert Navigation.ShortestPath(model, graph, [.5, .5, 0], 0)['status'] == 'ambiguous_endpoint'
    assert Navigation.ShortestPath(model, graph, [.2, .2, 0], 0)['status'] == 'ambiguous_endpoint'
    result = Navigation.ShortestPath(model, graph, [.2, .2, 0], [.8, .8, 0], startStorey='a', endStorey='b')
    assert result['status'] == 'disconnected'
    assert Navigation.ShortestPath(model, graph, [.2, .2, 0], 0, startStorey='a')['status'] == 'ok'
    with pytest.raises(ValueError, match='storey'):
        Navigation.ShortestPath(model, graph, 0, 1, startStorey='b')


def test_lift_and_sloped_floor_queries():
    lift = Navigation.Lift([[.5, .5, 0], [.5, .5, 3]], [[1, .5, 0], [1, .5, 3]])
    model = Navigation.ByFaces([rectangle(), rectangle(z=3)], connectors=[lift])
    graph = Navigation.Graph(model)
    result = Navigation.ShortestPath(model, graph, [.2, .2, 0], [.2, .2, 3])
    assert result['status'] == 'ok'
    assert any(p[2] == 3 for p in result['coordinates'])
    model = Navigation.ByFaces([rectangle(slope=.1)])
    graph = Navigation.Graph(model)
    assert Navigation.ShortestPath(model, graph, [.2, .2, .02], [3.8, 3.8, .38])['status'] == 'ok'
    assert Navigation.ShortestPath(model, graph, [.2, .2, .2], 0)['status'] == 'endpoint_unreachable'


@pytest.mark.parametrize('kwargs', [dict(maxSnapDistance=-1), dict(maxSnapDistance=float('inf')), dict(tolerance=0)])
def test_invalid_query_options(kwargs):
    model = Navigation.ByFaces([rectangle()])
    graph = Navigation.Graph(model)
    with pytest.raises(ValueError):
        Navigation.ShortestPath(model, graph, 0, 1, **kwargs)


def test_custom_cost_requires_existing_nodes():
    model = Navigation.ByFaces([rectangle()])
    graph = Navigation.Graph(model)
    with pytest.raises(ValueError, match='edgeKey'):
        Navigation.ShortestPath(model, graph, [.1, .1, 0], 0, edgeKey='hop')
    assert Navigation.ShortestPath(model, graph, 0, 15, edgeKey='hop')['cost'] == 6
    with pytest.raises(ValueError, match='active'):
        Navigation.ShortestPath(model, graph, 999, 0)


@pytest.mark.parametrize('clearance', [0, .1])
def test_stair_surface_endpoints_keep_tread_ownership(clearance):
    floors = [rectangle(w=1, h=1), rectangle(x=2, w=1, h=1, z=.3)]
    stair = Navigation.Stair(rectangle(x=1, w=1, h=1, z=.15), group='stair', storey='a')
    model = Navigation.ByFaces(floors, connectors=[stair], storeyIDs=['a', 'a'])
    graph = Navigation.Graph(model, spacing=.5, clearance=clearance)
    tread = next(v['index'] for v in graph._vertices if v['dictionary'].get('kind') == 'tread')
    result = Navigation.ShortestPath(model, graph, [1.3, .4, .15], tread)
    assert result['status'] == 'ok'
    assert result['coordinates'][0] == [1.3, .4, .15]
    metadata = {v['index']: v['dictionary'] for v in graph._vertices}
    assert metadata[result['nodeIDs'][1]]['kind'] == 'tread'
    assert Navigation.ShortestPath(model, graph, [1.3, .4, .3], 0)['status'] == 'endpoint_unreachable'


def test_ramp_centreline_endpoints_use_partial_edge_length():
    model = Navigation.ByFaces([rectangle(), rectangle(z=3)], connectors=[
        Navigation.Ramp(centreline=[[2, 2, 0], [2, 2, 3]], spacing=3)])
    graph = Navigation.Graph(model)
    result = Navigation.ShortestPath(model, graph, [2, 2, .8], [2, 2, 2.2], straighten=True)
    assert result['status'] == 'ok'
    assert result['length'] == pytest.approx(1.4)
    assert result['coordinates'] == [[2, 2, .8], [2, 2, 2.2]]
    assert Navigation.ShortestPath(model, graph, [2.05, 2, .8], 0)['status'] == 'endpoint_unreachable'
    assert Navigation.ShortestPath(model, graph, [2, 2, .8], 0,
                                   maxSnapDistance=.01)['status'] == 'endpoint_unreachable'


def test_lift_shaft_and_footprint_endpoints():
    lift = Navigation.Lift([[.5, .5, 0], [.5, .5, 3]], [[1, .5, 0], [1, .5, 3]],
                           footprint=rectangle(x=.5, w=1, h=1))
    # Cabin footprint extends outside the actual floor, so this exercises cabin support.
    model = Navigation.ByFaces([rectangle(w=.6), rectangle(w=.6, z=3)], connectors=[lift])
    graph = Navigation.Graph(model, spacing=.5)
    result = Navigation.ShortestPath(model, graph, [1, .5, .7], [1, .5, 2.3], straighten=True)
    assert result['status'] == 'ok'
    assert result['length'] == pytest.approx(1.6)
    assert Navigation.ShortestPath(model, graph, [1.1, .6, 0], 0)['status'] == 'ok'
    assert Navigation.ShortestPath(model, graph, [1.1, .6, 1], 0)['status'] == 'endpoint_unreachable'
    restored = TGraph.Copy(graph)
    assert Navigation.ShortestPath(model, restored, [1, .5, 1], 0)['status'] == 'ok'


def test_straightening_shortens_floor_route_and_calls_wire(monkeypatch):
    from topologicpy.Wire import Wire
    model = Navigation.ByFaces([rectangle()])
    graph = Navigation.Graph(model)
    calls = []
    original = Wire.Straighten
    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(Wire, 'Straighten', spy)
    result = Navigation.ShortestPath(model, graph, 0, 15, straighten=True)
    assert result['status'] == 'ok' and calls
    assert result['length'] == pytest.approx(math.sqrt(18))
    assert result['cost'] == pytest.approx(result['length'])
    assert result['nodeIDs'] == [0, 15]


@pytest.mark.parametrize('clearance', [0, .2])
def test_straightening_preserves_obstacle_and_clearance(clearance):
    import json
    from shapely.geometry import LineString
    model = Navigation.ByFaces([rectangle()], obstacles=[rectangle(x=1.8, y=1, w=.4, h=2)])
    graph = Navigation.Graph(model, spacing=.5, clearance=clearance)
    mesh = TGraph.MeshData(graph, includeIndices=True)
    start = min(mesh['vertexIndices'], key=lambda i: math.dist(TGraph.Coordinates(graph, i), [.5, 2, 0]))
    end = min(mesh['vertexIndices'], key=lambda i: math.dist(TGraph.Coordinates(graph, i), [3.5, 2, 0]))
    before = Navigation.ShortestPath(model, graph, start, end)
    result = Navigation.ShortestPath(model, graph, start, end, straighten=True)
    assert result['status'] == 'ok'
    assert result['length'] <= before['length'] + 1e-5
    records = json.loads(TGraph.Dictionary(graph)['navigationRouting'])['surfaces']
    support = Navigation._Shape(records[0])
    assert all(support.covers(LineString([a[:2], b[:2]]))
               for a, b in zip(result['coordinates'], result['coordinates'][1:]))


def test_invalid_straightener_output_is_rejected(monkeypatch):
    from topologicpy.Wire import Wire
    model = Navigation.ByFaces([rectangle()], obstacles=[rectangle(x=1.8, y=1, w=.4, h=2)])
    graph = Navigation.Graph(model, spacing=.5)
    mesh = TGraph.MeshData(graph, includeIndices=True)
    start = min(mesh['vertexIndices'], key=lambda i: math.dist(TGraph.Coordinates(graph, i), [.5, 2, 0]))
    end = min(mesh['vertexIndices'], key=lambda i: math.dist(TGraph.Coordinates(graph, i), [3.5, 2, 0]))
    def unsafe(wire, *args, **kwargs):
        vertices = Wire.Vertices(wire)
        return Wire.ByVertices([vertices[0], vertices[-1]], close=False)
    monkeypatch.setattr(Wire, 'Straighten', unsafe)
    before = Navigation.ShortestPath(model, graph, start, end)
    result = Navigation.ShortestPath(model, graph, start, end, straighten=True)
    assert result['coordinates'] == before['coordinates']


def test_straightening_keeps_lift_edges_and_storey_changes():
    model = Navigation.ByFaces([rectangle(), rectangle(z=3)], connectors=[
        Navigation.Lift([[.5, .5, 0], [.5, .5, 3]], [[1, .5, 0], [1, .5, 3]])])
    graph = Navigation.Graph(model)
    result = Navigation.ShortestPath(model, graph, [3.5, 3.5, 0], [3.5, 3.5, 3], straighten=True)
    assert result['status'] == 'ok'
    assert [1, .5, 0] in result['coordinates'] and [1, .5, 3] in result['coordinates']
    assert [.5, .5, 0] in result['coordinates'] and [.5, .5, 3] in result['coordinates']
    with pytest.raises(ValueError, match='Length'):
        Navigation.ShortestPath(model, graph, 0, 1, edgeKey='hop', straighten=True)


@pytest.mark.parametrize('clearance', [0, .1])
def test_sloped_ramp_surface_coordinates(clearance):
    ramp = Navigation.Ramp(rectangle(x=1, w=2, h=1, z=-.1, slope=.1),
                           storey='a', group='ramp', spacing=.4)
    model = Navigation.ByFaces([rectangle(w=1, h=1), rectangle(x=3, w=1, h=1, z=.2)],
                               storeyIDs=['a', 'a'], connectors=[ramp])
    graph = Navigation.Graph(model, spacing=.5, clearance=clearance)
    ramp_node = next(v['index'] for v in graph._vertices if v['dictionary'].get('kind') == 'ramp')
    result = Navigation.ShortestPath(model, graph, [1.7, .3, .07], ramp_node, straighten=True)
    assert result['status'] == 'ok'
    assert result['coordinates'][0] == [1.7, .3, .07]
    assert any(i is not None and graph._vertices[i]['dictionary'].get('kind') == 'ramp'
               for i in result['nodeIDs'][1:])


def test_crossing_unconnected_centrelines_are_ambiguous():
    model = Navigation.ByFaces([rectangle(), rectangle(z=3)], connectors=[
        Navigation.Ramp(centreline=[[1, 2, 1.5], [3, 2, 1.5]], spacing=2),
        Navigation.Ramp(centreline=[[2, 1, 1.5], [2, 3, 1.5]], spacing=2)])
    graph = Navigation.Graph(model)
    assert Navigation.ShortestPath(model, graph, [2, 2, 1.5], 0)['status'] == 'ambiguous_endpoint'


@pytest.mark.parametrize('kind', ['stair', 'ramp', 'lift'])
def test_real_ifc_connector_edges_survive_straightening(kind):
    from pathlib import Path
    model = Navigation.ByIFCPath(Path(__file__).parent/'fixtures'/'navigation-three-run.ifc')
    groups = {c['group'] for c in model.connectors if c['type'] == kind}
    model.connectors = [c for c in model.connectors
                        if c['type'] == kind or c.get('kind') == 'landing' and c['group'] in groups]
    if kind != 'lift':
        model._ifc = {**model._ifc, 'lifts': [], 'doors': []}
    graph = Navigation.Graph(model, spacing=1.5)
    coords = {v['index']: TGraph.Coordinates(graph, v['index']) for v in graph._vertices}
    start = min(coords, key=lambda i: math.dist(coords[i], [.5, .5, 0]))
    end = min(coords, key=lambda i: math.dist(coords[i], [11, 8, 3]))
    before = Navigation.ShortestPath(model, graph, start, end)
    after = Navigation.ShortestPath(model, graph, start, end, straighten=True)
    assert before['status'] == after['status'] == 'ok'
    protected = [(coords[e['src']], coords[e['dst']]) for e in graph._edges
                 if e['dictionary']['kind'] not in ('floor', 'visibility')
                 and e['src'] in before['nodeIDs'] and e['dst'] in before['nodeIDs']]
    segments = list(zip(after['coordinates'], after['coordinates'][1:]))
    for a, b in protected:
        assert (a, b) in segments or (b, a) in segments
    assert after['length'] <= before['length']+1e-5



def test_connector_partial_edges_preserve_one_way_direction():
    model = Navigation.ByFaces([rectangle(), rectangle(z=3)], connectors=[
        Navigation.Lift([[.5, .5, 0], [.5, .5, 3]], [[1, .5, 0], [1, .5, 3]])])
    graph = Navigation.Graph(model)
    edge = next(e for e in graph._edges if e['dictionary']['kind'] == 'lift')
    edge['active'] = False
    edge['dictionary']['active'] = False
    graph.AddEdge(edge['src'], edge['dst'], directed=True, dictionary={'Length': 3, 'kind': 'lift'})
    result = Navigation.ShortestPath(model, graph, [1, .5, .7], [1, .5, 2.3])
    assert result['status'] == 'ok' and result['length'] == pytest.approx(1.6)
    assert Navigation.ShortestPath(model, graph, [1, .5, 2.3], [1, .5, .7])['status'] == 'disconnected'


def test_straightening_respects_coordinate_connection_limit():
    model = Navigation.ByFaces([rectangle()])
    graph = Navigation.Graph(model)
    result = Navigation.ShortestPath(model, graph, [.1, .1, 0], [3.9, 3.9, 0],
                                     maxSnapDistance=1, straighten=True)
    assert result['status'] == 'ok'
    assert max(result['snapDistances']) <= 1
    assert len(result['coordinates']) > 2
