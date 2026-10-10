import sys
import pytest
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.TGraph import TGraph


def rectangle_with_hole():
    outer = Wire.ByVertices([Vertex.ByCoordinates(x, y, 0) for x, y in [(0, 0), (10, 0), (10, 10), (0, 10)]], close=True)
    hole = Wire.ByVertices([Vertex.ByCoordinates(x, y, 0) for x, y in [(4, 4), (6, 4), (6, 6), (4, 6)]], close=True)
    return Face.ByWires(outer, [hole])


@pytest.mark.parametrize('bidirectional', [False, True])
@pytest.mark.parametrize('leaf_size', [1, 100])
def test_visibility_rejects_hole_crossings_before_trimming(bidirectional, leaf_size):
    face = rectangle_with_hole()
    samples = [Vertex.ByCoordinates(x, y, 0) for x, y in [(1, 5), (9, 5)]]
    trim_calls = []
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == '_segment_inside_host_face':
            trim_calls.append(1)

    try:
        sys.setprofile(profile)
        graph = TGraph.VisibilityGraph(face, vertices=samples, includeBoundaryVertices=False,
                                       trim=True, bidirectional=bidirectional, leafSize=leaf_size,
                                       storeEdgeRepresentations=False, ontology=False)
    finally:
        sys.setprofile(previous)
    assert graph is not None
    assert len(TGraph.Vertices(graph)) == 2
    assert not TGraph.Edges(graph)
    assert not trim_calls


@pytest.mark.parametrize('bidirectional', [False, True])
def test_visibility_preserves_clear_route_and_length(bidirectional):
    samples = [Vertex.ByCoordinates(x, y, 0) for x, y in [(1, 1), (9, 1)]]
    graph = TGraph.VisibilityGraph(rectangle_with_hole(), vertices=samples,
                                  includeBoundaryVertices=False, trim=True,
                                  bidirectional=bidirectional,
                                  storeEdgeRepresentations=False, ontology=False)
    edges = TGraph.Edges(graph)
    assert len(edges) == (1 if bidirectional else 2)
    assert {(e['src'], e['dst']) for e in edges} == ({(0, 1)} if bidirectional else {(0, 1), (1, 0)})
    assert all(e['dictionary']['length'] == pytest.approx(8) for e in edges)
