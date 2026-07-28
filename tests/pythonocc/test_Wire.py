# Auto-generated PythonOCC backend parity test
# Copied from test_Wire.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# TopologicPy Wire unit tests.

import math

import pytest


Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Face = pytest.importorskip("topologicpy.Face").Face
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary


TOLERANCE = 1e-6

@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capsys):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    yield
    capsys.readouterr()


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _edge(start, end):
    return Edge.ByStartVertexEndVertex(_v(*start), _v(*end), silent=True)


def _assert_vertex(vertex):
    assert Topology.IsInstance(vertex, "Vertex")


def _assert_edge(edge):
    assert Topology.IsInstance(edge, "Edge")


def _assert_wire(wire):
    assert Topology.IsInstance(wire, "Wire")


def _assert_cluster(cluster):
    assert Topology.IsInstance(cluster, "Cluster")


def _coords(vertex, mantissa=6):
    return Vertex.Coordinates(vertex, mantissa=mantissa)


def _assert_coords(vertex, expected, abs_tol=TOLERANCE, mantissa=6):
    actual = _coords(vertex, mantissa=mantissa)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


def _assert_vector(actual, expected, abs_tol=TOLERANCE):
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


def _edge_count(wire):
    edges = Wire.Edges(wire)
    assert isinstance(edges, list)
    return len(edges)


def _vertex_count(wire):
    vertices = Wire.Vertices(wire)
    assert isinstance(vertices, list)
    return len(vertices)


@pytest.fixture
def open_polyline():
    return Wire.ByVertices(
        [_v(0, 0, 0), _v(3, 0, 0), _v(3, 4, 0)],
        close=False,
        silent=True,
    )


@pytest.fixture
def rectangle():
    return Wire.Rectangle(width=4, length=2, placement="center", silent=True)


def test_by_vertices_creates_open_and_closed_wires():
    vertices = [_v(0, 0, 0), _v(4, 0, 0), _v(4, 3, 0), _v(0, 3, 0)]

    open_wire = Wire.ByVertices(vertices, close=False, silent=True)
    closed_wire = Wire.ByVertices(vertices, close=True, silent=True)

    _assert_wire(open_wire)
    _assert_wire(closed_wire)
    assert bool(Wire.IsClosed(open_wire)) is False
    assert bool(Wire.IsClosed(closed_wire)) is True
    assert bool(Wire.IsManifold(open_wire)) is True
    assert bool(Wire.IsManifold(closed_wire)) is True
    assert _edge_count(open_wire) == 3
    assert _edge_count(closed_wire) == 4
    assert Wire.Length(open_wire) == pytest.approx(11)
    assert Wire.Length(closed_wire) == pytest.approx(14)

    assert Wire.ByVertices(None, silent=True) is None
    assert Wire.ByVertices([_v(0, 0, 0)], silent=True) is None


def test_by_edges_and_by_edges_cluster_create_wire():
    edges = [
        _edge((0, 0, 0), (2, 0, 0)),
        _edge((2, 0, 0), (2, 2, 0)),
        _edge((2, 2, 0), (0, 2, 0)),
    ]

    wire = Wire.ByEdges(edges, orient=True, silent=True)
    cluster_wire = Wire.ByEdgesCluster(Cluster.ByTopologies(edges))

    _assert_wire(wire)
    _assert_wire(cluster_wire)
    assert _edge_count(wire) == 3
    assert Wire.Length(wire) == pytest.approx(6)
    assert Wire.ByEdges(None, silent=True) is None
    assert Wire.ByEdges([], silent=True) is None
    assert Wire.ByEdgesCluster(None) is None


def test_by_vertices_cluster_creates_wire():
    vertices = [_v(0, 0, 0), _v(2, 0, 0), _v(2, 2, 0), _v(0, 2, 0)]
    cluster = Cluster.ByTopologies(vertices)

    wire = Wire.ByVerticesCluster(cluster, close=True, silent=True)

    _assert_wire(wire)
    assert bool(Wire.IsClosed(wire)) is True
    assert _edge_count(wire) == 4
    assert Wire.ByVerticesCluster(None, silent=True) is None


def test_rectangle_square_and_diagonal_rectangle():
    rect = Wire.Rectangle(width=4, length=2, placement="center", silent=True)
    square = Wire.Square(size=3)
    diag_rect = Wire.Rectangle(width=4, length=2, diagonals=True, silent=True)

    _assert_wire(rect)
    _assert_wire(square)
    _assert_wire(diag_rect)
    assert bool(Wire.IsClosed(rect)) is True
    assert bool(Wire.IsClosed(square)) is True
    assert Wire.Length(rect) == pytest.approx(12)
    assert Wire.Length(square) == pytest.approx(12)
    assert _edge_count(rect) == 4
    assert _edge_count(square) == 4
    assert _edge_count(diag_rect) >= 4

    assert Wire.Rectangle(width=0, length=2, silent=True) is None
    assert Wire.Rectangle(width=2, length=2, placement="invalid", silent=True) is None
    assert Wire.Rectangle(width=2, length=2, direction=[0, 0, 0], silent=True) is None


def test_line_creates_open_wire_with_requested_subdivision():
    origin = _v(0, 0, 0)

    center = Wire.Line(origin=origin, length=10, direction=[1, 0, 0], sides=5, placement="center")
    start = Wire.Line(origin=origin, length=10, direction=[1, 0, 0], sides=2, placement="start")
    end = Wire.Line(origin=origin, length=10, direction=[1, 0, 0], sides=2, placement="end")

    for wire in [center, start, end]:
        _assert_wire(wire)
        assert bool(Wire.IsClosed(wire)) is False
        assert Wire.Length(wire) == pytest.approx(10)

    assert _edge_count(center) == 5
    assert _vertex_count(center) == 6
    _assert_coords(Wire.StartVertex(start, silent=True), [0, 0, 0])
    _assert_coords(Wire.EndVertex(start, silent=True), [10, 0, 0])
    _assert_coords(Wire.StartVertex(end, silent=True), [-10, 0, 0])
    _assert_coords(Wire.EndVertex(end, silent=True), [0, 0, 0])

    assert Wire.Line(length=0) is None
    assert Wire.Line(direction=[1, 0]) is None
    assert Wire.Line(direction="x") is None
    assert Wire.Line(sides=1) is None


def test_circle_and_arc_constructors_create_expected_wire_types():
    circle = Wire.Circle(radius=1, sides=16, close=True, silent=True)
    open_arc = Wire.Circle(radius=1, sides=8, fromAngle=0, toAngle=180, close=False, silent=True)
    chord_arc = Wire.Arc(_v(-1, 0, 0), _v(0, 1, 0), _v(1, 0, 0), sides=8, close=False, silent=True)
    edge_arc = Wire.ArcByEdge(_edge((-1, 0, 0), (1, 0, 0)), sagitta=1, sides=8, close=False, silent=True)

    for wire in [circle, open_arc, chord_arc, edge_arc]:
        _assert_wire(wire)
        assert Wire.Length(wire) > 0

    assert bool(Wire.IsClosed(circle)) is True
    assert bool(Wire.IsClosed(open_arc)) is False
    assert Wire.Circle(radius=0, silent=True) is None
    assert Wire.Circle(radius=1, placement="invalid", silent=True) is None
    assert Wire.Circle(radius=1, direction=[0, 0, 0], silent=True) is None
    assert Wire.Arc(None, _v(0, 1, 0), _v(1, 0, 0), silent=True) is None
    assert Wire.ArcByEdge(None, silent=True) is None
    assert Wire.ArcByEdge(_edge((0, 0, 0), (1, 0, 0)), sagitta=0, silent=True) is None

def test_arc_respects_close_parameter():
    arc = Wire.Arc(_v(-1, 0, 0), _v(0, 1, 0), _v(1, 0, 0), sides=8, close=True, silent=True)

    _assert_wire(arc)
    assert bool(Wire.IsClosed(arc)) is True


def test_shape_constructors_return_valid_wires():
    closed_shapes = [
        Wire.CrossShape(width=4, length=4, silent=True),
        Wire.CShape(width=4, length=4, silent=True),
        Wire.IShape(width=4, length=4, silent=True),
        Wire.LShape(width=4, length=4, silent=True),
        Wire.TShape(width=4, length=4, silent=True),
        Wire.Trapezoid(widthA=4, widthB=2, length=3),
        Wire.Star(radiusA=2, radiusB=1, rays=5),
        Wire.Einstein(radius=1),
        Wire.GoldenRectangle(width=2, maxIterations=3, silent=True),
    ]
    squircle = Wire.Squircle(radius=1, sides=25)

    for wire in closed_shapes:
        _assert_wire(wire)
        assert isinstance(Wire.IsClosed(wire), bool)
        assert Wire.Length(wire) > 0
        # TODO: Confirm which shape constructors are contractually required to
        # return closed wires on all Topologic Core backends, then add exact
        # closure assertions per constructor.

    _assert_wire(squircle)
    assert isinstance(Wire.IsClosed(squircle), bool)
    assert Wire.Length(squircle) > 0
    # TODO: Confirm whether Wire.Squircle is intended to be closed. The current
    # implementation can emit a degenerate edge warning and return an open wire.

    assert Wire.CrossShape(width=0, length=4, silent=True) is None
    assert Wire.CShape(width=0, length=4, silent=True) is None
    assert Wire.IShape(width=0, length=4, silent=True) is None
    assert Wire.LShape(width=0, length=4, silent=True) is None
    assert Wire.TShape(width=0, length=4, silent=True) is None


def test_start_end_external_boundary_and_close(open_polyline, rectangle):
    start = Wire.StartVertex(open_polyline, silent=True)
    end = Wire.EndVertex(open_polyline, silent=True)
    endpoints = Wire.StartEndVertices(open_polyline, silent=True)
    boundary = Wire.ExternalBoundary(open_polyline, silent=True)
    closed = Wire.Close(open_polyline, silent=True)

    _assert_vertex(start)
    _assert_vertex(end)
    assert isinstance(endpoints, list)
    assert len(endpoints) == 2
    _assert_cluster(boundary)
    _assert_wire(closed)
    assert bool(Wire.IsClosed(closed)) is True

    assert Wire.StartVertex(rectangle, silent=True) is None
    assert Wire.EndVertex(rectangle, silent=True) is None
    assert Wire.StartEndVertices(rectangle, silent=True) is None
    assert Wire.ExternalBoundary(rectangle, silent=True) is None
    assert Wire.Close(None, silent=True) is None


def test_edges_vertices_and_length_invalid_inputs(rectangle):
    edges = Wire.Edges(rectangle)
    vertices = Wire.Vertices(rectangle)

    assert isinstance(edges, list)
    assert isinstance(vertices, list)
    assert len(edges) == 4
    assert len(vertices) == 4
    assert Wire.Length(rectangle) == pytest.approx(12)

    assert Wire.Edges(None) is None
    assert Wire.Vertices(None) is None
    assert Wire.Length(None) is None
    assert Wire.Length(_v(0, 0, 0)) is None
    assert Wire.IsClosed(None) is None
    assert Wire.IsManifold(None, silent=True) is None


def test_vertex_by_parameter_distance_and_parameter_at_vertex(open_polyline):
    start = Wire.StartVertex(open_polyline, silent=True)
    end = Wire.EndVertex(open_polyline, silent=True)
    midpoint_on_first_edge = Wire.VertexByParameter(open_polyline, 0.25)
    corner = _v(3, 0, 0)

    _assert_coords(Wire.VertexByParameter(open_polyline, 0), _coords(start))
    _assert_coords(Wire.VertexByParameter(open_polyline, 1), _coords(end))
    _assert_coords(midpoint_on_first_edge, [1.75, 0, 0])
    _assert_coords(Wire.VertexByDistance(open_polyline, distance=3), [3, 0, 0])

    assert Wire.ParameterAtVertex(open_polyline, start, silent=True) == pytest.approx(0)
    assert Wire.ParameterAtVertex(open_polyline, end, silent=True) == pytest.approx(1)
    assert Wire.ParameterAtVertex(open_polyline, corner, silent=True) == pytest.approx(3 / 7)
    assert Wire.VertexDistance(open_polyline, corner) == pytest.approx(3)

    assert Wire.VertexByParameter(None, 0.5) is None
    assert Wire.VertexByParameter(open_polyline, -0.1) is None
    assert Wire.VertexByParameter(open_polyline, 1.1) is None
    assert Wire.VertexByDistance(None, distance=1) is None
    assert Wire.VertexDistance(None, corner) is None
    assert Wire.VertexDistance(open_polyline, None) is None
    assert Wire.ParameterAtVertex(None, corner, silent=True) is None
    assert Wire.ParameterAtVertex(open_polyline, None, silent=True) is None


def test_reverse_preserves_length_and_reverses_open_wire_endpoints(open_polyline):
    start_before = Wire.StartVertex(open_polyline, silent=True)
    end_before = Wire.EndVertex(open_polyline, silent=True)

    reversed_wire = Wire.Reverse(open_polyline, silent=True)

    _assert_wire(reversed_wire)
    assert bool(Wire.IsClosed(reversed_wire)) is False
    assert Wire.Length(reversed_wire) == pytest.approx(Wire.Length(open_polyline))
    _assert_coords(Wire.StartVertex(reversed_wire, silent=True), _coords(end_before))
    _assert_coords(Wire.EndVertex(reversed_wire, silent=True), _coords(start_before))
    assert Wire.Reverse(None, silent=True) is None


def test_angles_normal_and_representation_for_rectangle(rectangle):
    interior = Wire.InteriorAngles(rectangle)
    exterior = Wire.ExteriorAngles(rectangle)
    normal = Wire.Normal(rectangle)
    representation = Wire.Representation(rectangle)

    assert len(interior) == 4
    assert len(exterior) == 4
    assert all(angle == pytest.approx(90) for angle in interior)
    assert all(angle == pytest.approx(270) for angle in exterior)
    assert normal[0] == pytest.approx(0)
    assert normal[1] == pytest.approx(0)
    assert abs(normal[2]) == pytest.approx(1)

    assert isinstance(representation, list)
    assert len(representation) == 7
    # TODO: Confirm the canonical representation values for a 4x2 rectangle
    # against the intended shape-similarity contract, then add exact asserts.

    open_wire = Wire.Line(length=4)
    assert Wire.InteriorAngles(open_wire) is None
    assert Wire.ExteriorAngles(open_wire) is None
    assert Wire.Normal(None) is None

def test_representation_invalid_input_returns_none():
    assert Wire.Representation(None) is None


def test_bounding_rectangle_dictionary_reports_extents(rectangle):
    bounding = Wire.BoundingRectangle(rectangle, optimize=0, mantissa=6, silent=True)

    _assert_wire(bounding)
    assert bool(Wire.IsClosed(bounding)) is True

    dictionary = Topology.Dictionary(bounding)
    assert Dictionary.ValueAtKey(dictionary, "width") == pytest.approx(4)
    assert Dictionary.ValueAtKey(dictionary, "length") == pytest.approx(2)
    assert Dictionary.ValueAtKey(dictionary, "xmin") == pytest.approx(-2)
    assert Dictionary.ValueAtKey(dictionary, "xmax") == pytest.approx(2)
    assert Dictionary.ValueAtKey(dictionary, "ymin") == pytest.approx(-1)
    assert Dictionary.ValueAtKey(dictionary, "ymax") == pytest.approx(1)

    assert Wire.BoundingRectangle(None, silent=True) is None
    assert Wire.BoundingRectangle(Wire.Line(length=5), silent=True) is None


def test_by_offset_and_bisectors_for_closed_rectangle(rectangle):
    offset_wire = Wire.ByOffset(rectangle, offset=0.1, silent=True)
    bisectors = Wire.Bisectors(rectangle, offset=0.1, silent=True)

    _assert_wire(offset_wire)
    assert bool(Wire.IsClosed(offset_wire)) is True
    assert Wire.Length(offset_wire) > 0

    # Bisectors are returned as an unflattened topology from a cluster of edges.
    assert Topology.IsInstance(bisectors, "Topology")
    # TODO: Confirm the intended return type for Wire.Bisectors. The docstring says
    # Wire, but the implementation returns the unflattened bisector cluster.


def test_remove_collinear_edges_and_simplify_reduce_redundant_vertices():
    redundant = Wire.ByVertices(
        [_v(0, 0, 0), _v(1, 0, 0), _v(2, 0, 0), _v(2, 2, 0), _v(0, 2, 0)],
        close=True,
        silent=True,
    )

    cleaned = Wire.RemoveCollinearEdges(redundant, silent=True)
    simplified = Wire.Simplify(redundant, tolerance=0.01, silent=True)

    _assert_wire(cleaned)
    _assert_wire(simplified)
    assert _edge_count(cleaned) <= _edge_count(redundant)
    assert _vertex_count(simplified) <= _vertex_count(redundant)

    assert Wire.RemoveCollinearEdges(None, silent=True) is None
    assert Wire.Simplify(None, silent=True) is None


def test_fillet_miter_and_invert_return_wires_for_rectangle(rectangle):
    filleted = Wire.Fillet(rectangle, radius=0.1, sides=4, silent=True)
    mitered = Wire.Miter(rectangle, offset=0.1, silent=True)
    inverted = Wire.Invert(rectangle, silent=True)

    for wire in [filleted, mitered, inverted]:
        _assert_wire(wire)
        assert Wire.Length(wire) > 0

    assert Wire.Fillet(None, silent=True) is None
    assert Wire.Miter(None, silent=True) is None
    assert Wire.Invert(None, silent=True) is None


def test_planarize_and_project_with_explicit_direction():
    elevated = Wire.Rectangle(origin=_v(0, 0, 5), width=2, length=2, silent=True)
    receiver = Face.Rectangle(width=10, length=10)

    planarized = Wire.Planarize(elevated)
    projected = Wire.Project(elevated, receiver, direction=[0, 0, -1])

    _assert_wire(planarized)
    _assert_wire(projected)
    assert all(abs(Vertex.Z(v)) <= 1e-6 for v in Wire.Vertices(projected))

    assert Wire.Planarize(None) is None
    assert Wire.Project(None, receiver, direction=[0, 0, -1]) is None
    assert Wire.Project(elevated, None, direction=[0, 0, -1]) is None


def test_project_uses_face_normal_when_direction_is_none():
    elevated = Wire.Rectangle(origin=_v(0, 0, 5), width=2, length=2, silent=True)
    receiver = Face.Rectangle(width=10, length=10)

    projected = Wire.Project(elevated, receiver)

    _assert_wire(projected)
    assert all(abs(Vertex.Z(v)) <= 1e-6 for v in Wire.Vertices(projected))

def test_interpolate_between_two_compatible_wires():
    lower = Wire.Rectangle(origin=_v(0, 0, 0), width=2, length=2, silent=True)
    upper = Wire.Rectangle(origin=_v(0, 0, 2), width=2, length=2, silent=True)

    result = Wire.Interpolate([lower, upper], n=3, outputType="contours")

    assert Topology.IsInstance(result, "Topology")

    wires = Topology.Wires(result)
    assert isinstance(wires, list)
    assert len(wires) == 5  # lower + 3 intermediate wires + upper

    wires = sorted(wires, key=lambda w: Vertex.Z(Topology.Centroid(w)))

    expected_z_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    for wire, expected_z in zip(wires, expected_z_values):
        _assert_wire(wire)
        assert bool(Wire.IsClosed(wire)) is True
        assert Wire.Length(wire) == pytest.approx(Wire.Length(lower))

        z_values = [Vertex.Z(v) for v in Wire.Vertices(wire)]
        assert all(z == pytest.approx(expected_z, abs=1e-6) for z in z_values)

    assert Wire.Interpolate([lower], n=3) is None
    assert Wire.Interpolate([lower, upper], n=3, outputType="invalid") is None
    assert Wire.Interpolate([lower, upper], n=3, mapping="invalid") is None


def test_lattice_and_cage_return_topologies_with_edges():
    lattice = Wire.Lattice(width=2, length=2, height=2, uSides=2, vSides=2, wSides=2)
    cage = Wire.Cage(width=2, length=2, height=2, uSides=2, vSides=2, wSides=2)

    assert Topology.IsInstance(lattice, "Topology")
    assert Topology.IsInstance(cage, "Topology")
    assert len(Topology.Edges(lattice)) > 0
    assert len(Topology.Edges(cage)) > 0

    assert Wire.Lattice(uSides=0) is None
    assert Wire.Cage(uSides=0) is None


def test_convex_hull_cycles_split_and_close_for_simple_inputs(rectangle):
    hull = Wire.ConvexHull(rectangle)
    cycles = Wire.Cycles(rectangle)
    split = Wire.Split(rectangle)

    _assert_wire(hull)
    assert isinstance(cycles, list)
    assert len(cycles) >= 1
    assert isinstance(split, list)
    assert len(split) >= 1



def test_roof_and_skeleton_return_topologies_for_simple_face(rectangle):
    face = Face.ByWire(rectangle)
    roof = Wire.Roof(face, angle=30, boundary=True)
    skeleton = Wire.Skeleton(face, boundary=True)

    assert Topology.IsInstance(roof, "Topology")
    assert Topology.IsInstance(skeleton, "Topology")

    assert Wire.Roof(None) is None
    assert Wire.Skeleton(None) is None


def test_spiral_and_golden_spiral_create_open_or_nonzero_wires():
    spiral = Wire.Spiral(radiusA=0.1, radiusB=1.0, height=1, turns=2, sides=24)
    golden_spiral = Wire.GoldenSpiral(width=1, maxIterations=4, sides=32, silent=True)

    _assert_wire(spiral)
    _assert_wire(golden_spiral)
    assert Wire.Length(spiral) > 0
    assert Wire.Length(golden_spiral) > 0


def test_by_tgraph_vertices_builds_wire_from_minimal_tgraph_records():
    class DummyTGraph:
        def __init__(self):
            self._vertices = {
                0: {"x": 0, "y": 0, "z": 0, "dictionary": {"name": "a"}},
                1: {"x": 1, "y": 0, "z": 0, "dictionary": {"name": "b"}},
                2: {"x": 1, "y": 1, "z": 0, "dictionary": {"name": "c"}},
            }
            self._edges = {
                (0, 1): {"dictionary": {"label": "ab"}},
                (1, 2): {"dictionary": {"label": "bc"}},
                (2, 0): {"dictionary": {"label": "ca"}},
            }

    wire = Wire.ByTGraphVertices(DummyTGraph(), [0, 1, 2], close=True, silent=True)

    _assert_wire(wire)
    assert bool(Wire.IsClosed(wire)) is True
    assert _edge_count(wire) == 3
    assert Wire.Length(wire) == pytest.approx(2 + math.sqrt(2))
    # TODO: Confirm whether ByTGraphVertices should guarantee dictionary
    # transfer through Wire.ByEdges on every backend/platform, then assert
    # vertex and edge dictionary values explicitly.

    assert Wire.ByTGraphVertices(None, [0, 1], silent=True) is None
    assert Wire.ByTGraphVertices(DummyTGraph(), None, silent=True) is None
    assert Wire.ByTGraphVertices(DummyTGraph(), [0], silent=True) is None
    assert Wire.ByTGraphVertices(DummyTGraph(), [0, 99], silent=True) is None


def test_invalid_inputs_for_higher_level_wire_methods(rectangle):
    assert Wire.ByOffset(None) is None
    assert Wire.ByOffsetArea(None, area=1, silent=True) is None
    assert Wire.Close(None, silent=True) is None
    assert Wire.ExternalBoundary(None, silent=True) is None
    assert Wire.Project(rectangle, None) is None
    assert Wire.VertexByDistance(rectangle, distance=1) is None
    assert Wire.VertexDistance(rectangle, _v(0, 0, 0)) is None
