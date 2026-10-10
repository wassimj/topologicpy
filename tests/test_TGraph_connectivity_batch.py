import pytest
from topologicpy.TGraph import TGraph


@pytest.mark.parametrize('attached', [False, True])
@pytest.mark.parametrize('mode', ['all', 'in', 'out'])
def test_connectivity_compiles_once_and_preserves_degrees(monkeypatch, attached, mode):
    graph = TGraph(directed=True, allowParallelEdges=True, allowSelfLoops=True)
    for index in range(5):
        graph.AddVertex({'x': index, 'y': 0, 'z': 0})
    for source, target in [(0, 1), (0, 1), (1, 2), (2, 2), (3, 0)]:
        graph.AddEdge(source, target)
    expected = [float(TGraph.Degree(graph, index, mode)) for index in TGraph.ActiveVertexIndices(graph)]
    if attached:
        TGraph.AnalysisContext(graph)
    original = TGraph.Compile
    calls = []

    def compile_once(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(TGraph, 'Compile', staticmethod(compile_once))
    assert TGraph.Connectivity(graph, key=None, mode=mode) == expected
    assert len(calls) == 1
    calls.clear()
    graph._edges[0]['active'] = False
    graph._vertices[4]['active'] = False
    result = TGraph.Connectivity(graph, key=None, mode=mode)
    assert len(calls) == 1
    assert result == [float(TGraph.Degree(graph, index, mode)) for index in TGraph.ActiveVertexIndices(graph)]
