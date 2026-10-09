"""Checks containment, sampled reduction and original-plane axial output."""
import pytest
from shapely.geometry import Polygon, LineString
from topologicpy.Face import Face
from topologicpy.Wire import Wire
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Topology import Topology
from topologicpy.TGraph import TGraph


def face_from(points):
    return Face.ByWire(Wire.ByVertices([Vertex.ByCoordinates(*p) for p in points], close=True))


def coords(edge):
    return [Vertex.Coordinates(Edge.StartVertex(edge), mantissa=10),
            Vertex.Coordinates(Edge.EndVertex(edge), mantissa=10)]


def test_rectangle_reduces_and_alias_is_identical():
    face = Face.Rectangle(width=6, length=4)
    assert Face.AxialLines is Face.AxialEdges
    report = Face.AxialEdges(face, returnReport=True)
    assert report is not None and report["approximate"] is True
    assert report["sampleCount"] > 0
    assert len(report["edges"]) == 1
    assert report["candidateCount"] > 1
    assert report["uncoveredSamples"] == []
    assert len(Face.AxialEdges(face, reduce=False)) == report["candidateCount"]
    assert TGraph.Order(TGraph.AxialGraph(report["edges"])) == 1


def test_concave_space_edges_stay_inside_and_map_connects():
    points = [(0,0,0), (6,0,0), (6,2,0), (2,2,0), (2,6,0), (0,6,0)]
    face = face_from(points)
    report = Face.AxialEdges(face, returnReport=True)
    assert report is not None
    domain = Polygon([(x,y) for x,y,z in points])
    for edge in report["edges"]:
        assert domain.buffer(1e-8).covers(LineString([(p[0], p[1]) for p in coords(edge)]))
    graph = TGraph.AxialGraph(report["edges"])
    assert len(TGraph.ConnectedComponents(graph)) == 1


def test_obstacle_and_hole_are_not_crossed():
    face = Face.Rectangle(width=8, length=8)
    obstacle = Face.Rectangle(width=2, length=2)
    hole_wire = Face.ExternalBoundary(obstacle)
    holed = Face.ByWires(Face.ExternalBoundary(face), [hole_wire])
    domain = Polygon([(-4,-4),(4,-4),(4,4),(-4,4)], holes=[[(-1,-1),(1,-1),(1,1),(-1,1)]])
    for boundary, obstacles in [(face, [obstacle]), (holed, None)]:
        report = Face.AxialEdges(boundary, obstacles=obstacles, returnReport=True)
        assert report is not None and report["edges"]
        for edge in report["edges"]:
            assert domain.buffer(1e-8).covers(LineString([(p[0], p[1]) for p in coords(edge)]))


def test_vertical_plane_output_is_not_flattened():
    face = face_from([(3,0,0),(3,5,0),(3,5,4),(3,0,4)])
    edges = Face.AxialEdges(face)
    assert edges
    assert all(abs(p[0]-3) < 1e-8 for edge in edges for p in coords(edge))


def test_obstacle_can_split_free_space_into_components():
    face = Face.Rectangle(width=8, length=4)
    obstacle = Face.Rectangle(width=1, length=6)
    report = Face.AxialEdges(face, obstacles=[obstacle], returnReport=True)
    assert report is not None and report["componentCount"] == 2
    graph = TGraph.AxialGraph(report["edges"])
    assert len(TGraph.ConnectedComponents(graph)) == 2


@pytest.mark.parametrize("kwargs", [{"maxCandidates":0}, {"maxCandidates":True},
                                   {"maxSamples":0}, {"maxSamples":1.5},
                                   {"samplingDistance":0}, {"samplingDistance":float("nan")}])
def test_invalid_sampling_and_limits_fail_explicitly(kwargs):
    assert Face.AxialEdges(Face.Rectangle(), silent=True, **kwargs) is None


def test_non_coplanar_obstacle_is_rejected():
    obstacle = Face.Rectangle(origin=Vertex.ByCoordinates(0,0,3))
    assert Face.AxialEdges(Face.Rectangle(), obstacles=[obstacle], silent=True) is None


@pytest.mark.parametrize("reduce", [True, False])
def test_candidate_limit_returns_bounded_map_and_warning(reduce, capsys):
    report = Face.AxialEdges(Face.Rectangle(width=6, length=4), maxCandidates=2,
                            reduce=reduce, returnReport=True)
    assert report and report["edges"]
    assert report["candidateCount"] == 2
    assert len(report["edges"]) <= 2
    assert report["truncated"] is True
    assert report["limitsReached"] == ["maxCandidates"]
    assert "Warning" in capsys.readouterr().out
    if reduce:
        assert report["coverageComplete"] is True
        assert report["connected"] is True
    else:
        assert report["coverageComplete"] is None
        assert report["connected"] is None


def test_sample_limit_coarsens_and_returns_map():
    report = Face.AxialEdges(Face.Rectangle(width=6, length=4), maxSamples=1,
                            samplingDistance=0.0002, returnReport=True, silent=True)
    assert report and report["edges"]
    assert report["sampleCount"] == 1
    assert report["limitsReached"] == ["maxSamples"]
    assert report["samplingDistance"] > report["requestedSamplingDistance"]


def test_limited_candidates_are_spread_across_components():
    face = Face.Rectangle(width=8, length=4)
    obstacle = Face.Rectangle(width=1, length=6)
    report = Face.AxialEdges(face, obstacles=[obstacle], maxCandidates=2,
                            reduce=False, returnReport=True, silent=True)
    assert len(report["edges"]) == 2
    graph = TGraph.AxialGraph(report["edges"])
    assert len(TGraph.ConnectedComponents(graph)) == 2


def test_incomplete_coverage_returns_edges_and_truthful_report():
    face = Face.Rectangle(width=8, length=4)
    obstacle = Face.Rectangle(width=1, length=6)
    report = Face.AxialEdges(face, obstacles=[obstacle], maxCandidates=1,
                            returnReport=True, silent=True)
    assert len(report["edges"]) == 1
    assert report["uncoveredSamples"]
    assert report["coverageComplete"] is False
    assert report["connected"] is False
    assert sorted(report["componentConnected"]) == [False, True]
    assert all(len(point) == 3 for point in report["uncoveredSamples"])


def test_unlimited_report_and_empty_domain_are_not_marked_truncated():
    face = Face.Rectangle(width=6, length=4)
    report = Face.AxialEdges(face, returnReport=True, silent=True)
    assert report["truncated"] is False
    assert report["coverageComplete"] is True
    assert report["connected"] is True
    empty = Face.AxialEdges(face, obstacles=[face], returnReport=True, silent=True)
    assert empty["edges"] == []
    assert empty["truncated"] is False


def test_sample_budget_can_report_an_unsampled_component():
    face = Face.Rectangle(width=8, length=4)
    obstacle = Face.Rectangle(width=1, length=6)
    report = Face.AxialEdges(face, obstacles=[obstacle], maxSamples=1,
                            returnReport=True, silent=True)
    assert report["edges"]
    assert len(report["samples"]) == report["sampleCount"] == 1
    assert report["coverageComplete"] is False
    assert report["componentConnected"] == [True, False]


@pytest.mark.parametrize("holed", [False, True])
def test_report_matches_independent_visibility_and_connectivity_checks(holed):
    if holed:
        face = Face.Rectangle(width=12, length=12)
        obstacles = [Face.Rectangle(origin=Vertex.ByCoordinates(x, y, 0), width=2, length=2)
                     for x, y in [(-3, 0), (2, -3), (2, 2)]]
        domain = Polygon([(-6,-6), (6,-6), (6,6), (-6,6)], holes=[
            [(x-1,y-1), (x+1,y-1), (x+1,y+1), (x-1,y+1)]
            for x, y in [(-3,0), (2,-3), (2,2)]])
    else:
        points = [(0,0,0), (7,0,0), (7,2,0), (3,2,0), (3,7,0), (0,7,0)]
        face, obstacles = face_from(points), None
        domain = Polygon([(x,y) for x,y,z in points])
    report = Face.AxialEdges(face, obstacles=obstacles, maxCandidates=80,
                            maxSamples=60, samplingDistance=1,
                            returnReport=True, silent=True)
    assert report and report["edges"]
    assert len(report["samples"]) == report["sampleCount"] <= 60
    assert report["evaluatedCandidateCount"] <= report["candidateCount"] <= 80
    viewpoints = []
    for edge in report["edges"]:
        a, b = coords(edge)
        for t in (0.5, 0.25, 0.75, 1e-6, 1-1e-6):
            viewpoints.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
    visible = [any(domain.buffer(1e-9).covers(LineString([view, point[:2]]))
                   for view in viewpoints) for point in report["samples"]]
    assert report["coverageComplete"] == all(visible)
    assert len(report["uncoveredSamples"]) == visible.count(False)
    graph = TGraph.AxialGraph(report["edges"])
    assert report["connected"] == (len(TGraph.ConnectedComponents(graph)) == 1)


def test_near_duplicate_candidates_are_removed_within_tolerance():
    # A shallow kink and an origin near a rounding boundary exercise rays
    # whose endpoints differ numerically while representing the same axis.
    face = face_from([(0.00005,0,0), (2,0.000001,0), (4,0,0), (4,4,0), (0.00005,4,0)])
    edges = Face.AxialEdges(face, reduce=False, silent=True, tolerance=0.0001)
    assert edges
    import math
    endpoints = [coords(e) for e in edges]
    for i, (a, b) in enumerate(endpoints):
        for c, d in endpoints[i+1:]:
            duplicate = ((math.dist(a,c) <= 0.0001 and math.dist(b,d) <= 0.0001) or
                         (math.dist(a,d) <= 0.0001 and math.dist(b,c) <= 0.0001))
            assert not duplicate
    assert TGraph.AxialGraph(edges) is not None
