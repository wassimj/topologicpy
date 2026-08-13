# Auto-generated PythonOCC backend parity test
# Copied from test_Edge.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# TopologicPy Edge unit tests.

import math

import pytest


Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Face = pytest.importorskip("topologicpy.Face").Face
Topology = pytest.importorskip("topologicpy.Topology").Topology


TOLERANCE = 1e-6

@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _edge(start, end):
    return Edge.ByStartVertexEndVertex(_v(*start), _v(*end), silent=True)


def _assert_vertex(vertex):
    assert Topology.IsInstance(vertex, "Vertex")


def _assert_edge(edge):
    assert Topology.IsInstance(edge, "Edge")


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


def _start(edge):
    return Edge.StartVertex(edge, silent=True)


def _end(edge):
    return Edge.EndVertex(edge, silent=True)


@pytest.fixture
def x_edge():
    return _edge((0, 0, 0), (10, 0, 0))


@pytest.fixture
def y_edge():
    return _edge((0, 0, 0), (0, 10, 0))


def test_by_start_vertex_end_vertex_creates_edge_and_preserves_orientation():
    start = _v(0, 0, 0)
    end = _v(3, 4, 0)

    edge = Edge.ByStartVertexEndVertex(start, end, silent=True)

    _assert_edge(edge)
    _assert_coords(Edge.StartVertex(edge, silent=True), [0, 0, 0])
    _assert_coords(Edge.EndVertex(edge, silent=True), [3, 4, 0])
    assert Edge.Length(edge) == pytest.approx(5)


def test_by_start_vertex_end_vertex_rejects_invalid_or_degenerate_input():
    start = _v(0, 0, 0)
    same_location = _v(0, 0, 0)
    near_location = _v(0.00001, 0, 0)

    assert Edge.ByStartVertexEndVertex(None, start, silent=True) is None
    assert Edge.ByStartVertexEndVertex(start, None, silent=True) is None
    assert Edge.ByStartVertexEndVertex(start, same_location, silent=True) is None
    assert Edge.ByStartVertexEndVertex(start, near_location, tolerance=0.001, silent=True) is None


def test_by_vertices_accepts_lists_nested_lists_and_varargs():
    start = _v(0, 0, 0)
    middle = _v(5, 0, 0)
    end = _v(10, 0, 0)

    examples = [
        Edge.ByVertices([start, end], silent=True),
        Edge.ByVertices(start, end, silent=True),
        Edge.ByVertices([[start, middle], [end]], silent=True),
    ]

    for edge in examples:
        _assert_edge(edge)
        _assert_coords(_start(edge), [0, 0, 0])
        _assert_coords(_end(edge), [10, 0, 0])

    assert Edge.ByVertices([], silent=True) is None
    assert Edge.ByVertices([start], silent=True) is None


def test_by_vertices_cluster_uses_first_and_last_cluster_vertices():
    vertices = [_v(0, 0, 0), _v(5, 0, 0), _v(10, 0, 0)]
    cluster = Cluster.ByTopologies(vertices)

    edge = Edge.ByVerticesCluster(cluster)

    _assert_edge(edge)
    assert Edge.Length(edge) == pytest.approx(10)
    assert Edge.ByVerticesCluster(None) is None


def test_by_origin_direction_length_creates_expected_edge():
    origin = _v(1, 2, 3)
    edge = Edge.ByOriginDirectionLength(origin=origin, direction=[0, 1, 0], length=5, silent=True)

    _assert_edge(edge)
    _assert_coords(_start(edge), [1, 2, 3])
    _assert_coords(_end(edge), [1, 7, 3])
    assert Edge.Length(edge) == pytest.approx(5)
    assert Edge.ByOriginDirectionLength(origin=origin, length=0, silent=True) is None


def test_line_creates_center_start_and_end_placements():
    origin = _v(0, 0, 0)

    center = Edge.Line(origin=origin, length=4, direction=[1, 0, 0], placement="center")
    start = Edge.Line(origin=origin, length=4, direction=[1, 0, 0], placement="start")
    end = Edge.Line(origin=origin, length=4, direction=[1, 0, 0], placement="end")

    _assert_coords(_start(center), [-2, 0, 0])
    _assert_coords(_end(center), [2, 0, 0])
    _assert_coords(_start(start), [0, 0, 0])
    _assert_coords(_end(start), [4, 0, 0])
    _assert_coords(_start(end), [-4, 0, 0])
    _assert_coords(_end(end), [0, 0, 0])

    assert Edge.Line(origin=origin, length=0) is None
    assert Edge.Line(origin=origin, direction=[1, 0]) is None
    assert Edge.Line(origin=origin, direction="x") is None
    assert Edge.Line(origin=origin, placement="invalid") is None


def test_accessors_return_start_end_and_vertices(x_edge):
    start = Edge.StartVertex(x_edge, silent=True)
    end = Edge.EndVertex(x_edge, silent=True)
    vertices = Edge.Vertices(x_edge, silent=True)

    _assert_vertex(start)
    _assert_vertex(end)
    assert isinstance(vertices, list)
    assert len(vertices) == 2
    _assert_coords(start, [0, 0, 0])
    _assert_coords(end, [10, 0, 0])

    assert Edge.StartVertex(None, silent=True) is None
    assert Edge.EndVertex(None, silent=True) is None
    assert Edge.Vertices(None, silent=True) is None


def test_length_quadrance_direction_angle_and_spread(x_edge, y_edge):
    reverse_x = Edge.Reverse(x_edge, silent=True)

    assert Edge.Length(x_edge) == pytest.approx(10)
    assert Edge.Quadrance(x_edge) == pytest.approx(100)
    assert Edge.Direction(x_edge) == [1, 0, 0]
    assert Edge.Direction(reverse_x) == [-1, 0, 0]
    assert Edge.Angle(x_edge, y_edge) == pytest.approx(90)
    assert Edge.Angle(x_edge, reverse_x, bracket=True) == pytest.approx(0)
    assert Edge.Spread(x_edge, y_edge) == pytest.approx(1)
    assert Edge.Spread(x_edge, reverse_x, bracket=True) == pytest.approx(0)

    assert Edge.Length(None) is None
    assert Edge.Quadrance(None) is None
    assert Edge.Direction(None) is None
    assert Edge.Angle(None, y_edge) is None
    assert Edge.Spread(x_edge, None) is None


def test_equation2d_reports_horizontal_sloped_and_vertical_lines():
    horizontal = _edge((0, 2, 0), (10, 2, 0))
    diagonal = _edge((0, 1, 0), (2, 5, 0))
    vertical = _edge((3, -1, 0), (3, 7, 0))

    assert Edge.Equation2D(horizontal) == {
        "slope": 0,
        "x_intercept": None,
        "y_intercept": 2,
    }
    assert Edge.Equation2D(diagonal) == {
        "slope": 2,
        "x_intercept": None,
        "y_intercept": 1,
    }
    assert Edge.Equation2D(vertical) == {
        "slope": float("inf"),
        "x_intercept": 3,
        "y_intercept": None,
    }


def test_vertex_by_parameter_and_distance_return_expected_coordinates(x_edge):
    _assert_coords(Edge.VertexByParameter(x_edge, u=0), [0, 0, 0])
    _assert_coords(Edge.VertexByParameter(x_edge, u=1), [10, 0, 0])
    _assert_coords(Edge.VertexByParameter(x_edge, u=0.25), [2.5, 0, 0])

    _assert_coords(Edge.VertexByDistance(x_edge, distance=3), [3, 0, 0])
    _assert_coords(Edge.VertexByDistance(x_edge, distance=-2, origin=_end(x_edge)), [8, 0, 0])
    assert Edge.VertexByParameter(None, u=0.5) is None
    assert Edge.VertexByDistance(None, distance=1) is None


def test_parameter_at_vertex_returns_u_parameter_for_points_on_edge(x_edge):
    start = _start(x_edge)
    end = _end(x_edge)
    middle = Edge.VertexByParameter(x_edge, u=0.5)
    outside = _v(0, 1, 0)

    assert Edge.ParameterAtVertex(x_edge, start, silent=True) == pytest.approx(0)
    assert Edge.ParameterAtVertex(x_edge, end, silent=True) == pytest.approx(1)
    assert Edge.ParameterAtVertex(x_edge, middle, silent=True) == pytest.approx(0.5)
    assert Edge.ParameterAtVertex(x_edge, outside, silent=True) is None
    assert Edge.ParameterAtVertex(None, start, silent=True) is None
    assert Edge.ParameterAtVertex(x_edge, None, silent=True) is None


def test_reverse_and_index_behaviour(x_edge):
    same_coordinates = _edge((0, 0, 0), (10, 0, 0))
    reversed_coordinates = _edge((10, 0, 0), (0, 0, 0))
    unrelated = _edge((0, 1, 0), (10, 1, 0))

    reversed_edge = Edge.Reverse(x_edge, silent=True)

    _assert_edge(reversed_edge)
    _assert_coords(_start(reversed_edge), [10, 0, 0])
    _assert_coords(_end(reversed_edge), [0, 0, 0])
    assert Edge.Index(x_edge, [unrelated, same_coordinates]) == 1
    assert Edge.Index(x_edge, [reversed_coordinates]) == 0
    assert Edge.Index(x_edge, [x_edge], strict=True) == 0
    assert Edge.Index(None, [x_edge]) is None
    assert Edge.Index(x_edge, None) is None


def test_normalize_normal_edge_and_normal_vector(x_edge):
    normalized = Edge.Normalize(x_edge, silent=True)
    normal_edge = Edge.NormalEdge(x_edge, length=3, u=0.5, silent=True)

    _assert_edge(normalized)
    assert Edge.Length(normalized) == pytest.approx(1)
    assert Edge.Direction(normalized) == [1, 0, 0]

    _assert_edge(normal_edge)
    assert Edge.Length(normal_edge) == pytest.approx(3)
    _assert_coords(_start(normal_edge), [5, 0, 0])
    _assert_vector(Edge.Normal(x_edge), [0, 1, 0])

    assert Edge.Normalize(None, silent=True) is None
    assert Edge.Normal(None) is None
    assert Edge.NormalEdge(None, silent=True) is None
    assert Edge.NormalEdge(x_edge, length=0, silent=True) is None


def test_normalize_use_end_vertex_returns_unit_edge(x_edge):
    normalized_to_end = Edge.Normalize(x_edge, useEndVertex=True, silent=True)

    _assert_edge(normalized_to_end)
    assert Edge.Length(normalized_to_end) == pytest.approx(1)
    _assert_coords(_end(normalized_to_end), [10, 0, 0])


def test_extend_trim_and_set_length_change_lengths_and_endpoint_positions(x_edge):
    extended = Edge.Extend(x_edge, distance=4, bothSides=True, silent=True)
    extended_from_end = Edge.Extend(x_edge, distance=2, bothSides=False, reverse=False, silent=True)
    trimmed = Edge.Trim(x_edge, distance=4, bothSides=True, silent=True)
    trimmed_from_end = Edge.Trim(x_edge, distance=2, bothSides=False, reverse=False, silent=True)
    set_length = Edge.SetLength(x_edge, length=4, bothSides=True)

    assert Edge.Length(extended) == pytest.approx(14)
    _assert_coords(_start(extended), [-2, 0, 0])
    _assert_coords(_end(extended), [12, 0, 0])

    assert Edge.Length(extended_from_end) == pytest.approx(12)
    _assert_coords(_start(extended_from_end), [0, 0, 0])
    _assert_coords(_end(extended_from_end), [12, 0, 0])

    assert Edge.Length(trimmed) == pytest.approx(6)
    _assert_coords(_start(trimmed), [2, 0, 0])
    _assert_coords(_end(trimmed), [8, 0, 0])

    assert Edge.Length(trimmed_from_end) == pytest.approx(8)
    _assert_coords(_start(trimmed_from_end), [0, 0, 0])
    _assert_coords(_end(trimmed_from_end), [8, 0, 0])

    assert Edge.Length(set_length) == pytest.approx(4)
    assert Edge.Extend(None, silent=True) is None
    assert Edge.Trim(None, silent=True) is None
    assert Edge.SetLength(None) is None


def test_by_offset2d_offsets_left_of_edge_in_xy_plane(x_edge):
    offset_edge = Edge.ByOffset2D(x_edge, offset=1)

    _assert_edge(offset_edge)
    _assert_coords(_start(offset_edge), [0, 1, 0])
    _assert_coords(_end(offset_edge), [10, 1, 0])
    assert Edge.Length(offset_edge) == pytest.approx(10)

    # TODO: Add an assertion for the intended behaviour when a non-XY edge is offset.


def test_intersect2d_handles_crossing_shared_endpoint_and_parallel_edges():
    horizontal = _edge((0, 0, 0), (10, 0, 0))
    vertical = _edge((5, -5, 0), (5, 5, 0))
    shared_a = _edge((0, 0, 0), (1, 0, 0))
    shared_b = _edge((0, 0, 0), (0, 1, 0))
    parallel = _edge((0, 1, 0), (10, 1, 0))

    _assert_coords(Edge.Intersect2D(horizontal, vertical, silent=True), [5, 0, 0])
    _assert_coords(Edge.Intersect2D(shared_a, shared_b, silent=True), [0, 0, 0])
    assert Edge.Intersect2D(horizontal, parallel, silent=True) is None


def test_collinear_parallel_and_coplanar_predicates():
    x_axis = _edge((0, 0, 0), (10, 0, 0))
    x_axis_extension = _edge((20, 0, 0), (30, 0, 0))
    x_parallel = _edge((0, 1, 0), (10, 1, 0))
    y_axis = _edge((0, 0, 0), (0, 10, 0))
    skew = _edge((0, 0, 1), (0, 10, 1))

    assert bool(Edge.IsCollinear(x_axis, x_axis_extension)) is True
    assert bool(Edge.IsCollinear(x_axis, x_parallel)) is False
    assert bool(Edge.IsParallel(x_axis, x_parallel)) is True
    assert bool(Edge.IsParallel(x_axis, y_axis)) is False
    assert bool(Edge.IsCoplanar(x_axis, y_axis)) is True
    assert bool(Edge.IsCoplanar(x_axis, skew)) is False

    assert Edge.IsCollinear(None, x_axis) is None
    assert Edge.IsParallel(x_axis, None) is None
    assert Edge.IsCoplanar(None, x_axis) is None


def test_connection_joins_closest_endpoints():
    edge_a = _edge((0, 0, 0), (1, 0, 0))
    edge_b = _edge((5, 0, 0), (5, 1, 0))

    connection = Edge.Connection(edge_a, edge_b, silent=True)

    _assert_edge(connection)
    _assert_coords(_start(connection), [1, 0, 0])
    _assert_coords(_end(connection), [5, 0, 0])
    assert Edge.Length(connection) == pytest.approx(4)


def test_bisect_creates_expected_bisector_for_perpendicular_edges():
    x_unit = _edge((0, 0, 0), (1, 0, 0))
    y_unit = _edge((0, 0, 0), (0, 1, 0))

    bisector = Edge.Bisect(x_unit, y_unit, length=2, placement=1, silent=True)

    _assert_edge(bisector)
    _assert_coords(_start(bisector), [0, 0, 0])
    assert Edge.Length(bisector) == pytest.approx(2)
    direction = Edge.Direction(bisector, mantissa=6)
    assert direction[0] == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert direction[1] == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert direction[2] == pytest.approx(0, abs=1e-6)

    separated = _edge((10, 0, 0), (11, 0, 0))
    assert Edge.Bisect(x_unit, separated, silent=True) is None


def test_by_face_normal_creates_edge_with_requested_length():
    face = Face.Rectangle(width=4, length=6)
    normal_edge = Edge.ByFaceNormal(face, length=3)

    _assert_edge(normal_edge)
    assert Edge.Length(normal_edge) == pytest.approx(3)
    direction = Edge.Direction(normal_edge)
    assert abs(direction[0]) == pytest.approx(0, abs=TOLERANCE)
    assert abs(direction[1]) == pytest.approx(0, abs=TOLERANCE)
    assert abs(direction[2]) == pytest.approx(1, abs=TOLERANCE)
    assert Edge.ByFaceNormal(None) is None


def test_external_boundary_returns_cluster_of_end_vertices(x_edge):
    boundary = Edge.ExternalBoundary(x_edge, silent=True)

    assert Topology.IsInstance(boundary, "Cluster")
    vertices = Topology.Vertices(boundary)
    assert len(vertices) == 2
    assert Edge.ExternalBoundary(None, silent=True) is None


def test_extend_to_edge_extends_to_intersection_with_second_edge():
    edge_a = _edge((0, 0, 0), (10, 0, 0))
    edge_b = _edge((20, -10, 0), (20, 10, 0))

    extended = Edge.ExtendToEdge(edge_a, edge_b, silent=True)

    _assert_edge(extended)
    assert Edge.Length(extended) == pytest.approx(20)
    _assert_coords(_start(extended), [0, 0, 0])
    _assert_coords(_end(extended), [20, 0, 0])


def test_trim_by_edge_trims_to_intersection_with_second_edge():
    edge_a = _edge((0, 0, 0), (10, 0, 0))
    edge_b = _edge((5, -10, 0), (5, 10, 0))

    trimmed = Edge.TrimByEdge(edge_a, edge_b, silent=True)
    reversed_trimmed = Edge.TrimByEdge(edge_a, edge_b, reverse=True, silent=True)

    _assert_edge(trimmed)
    assert Edge.Length(trimmed) == pytest.approx(5)
    _assert_coords(_start(trimmed), [0, 0, 0])
    _assert_coords(_end(trimmed), [5, 0, 0])

    _assert_edge(reversed_trimmed)
    assert Edge.Length(reversed_trimmed) == pytest.approx(5)
    _assert_coords(_start(reversed_trimmed), [10, 0, 0])
    _assert_coords(_end(reversed_trimmed), [5, 0, 0])

    assert Edge.TrimByEdge(None, edge_b, silent=True) is None
    assert Edge.TrimByEdge(edge_a, None, silent=True) is None


def test_align2d_returns_4x4_transformation_matrix():
    source = _edge((0, 0, 0), (2, 0, 0))
    target = _edge((0, 0, 0), (0, 4, 0))

    matrix = Edge.Align2D(source, target)

    assert isinstance(matrix, list)
    assert len(matrix) == 4
    assert all(isinstance(row, list) and len(row) == 4 for row in matrix)
    assert all(isinstance(value, (int, float)) for row in matrix for value in row)
    # TODO: Verify the exact expected transformation matrix after confirming
    # the intended alignment convention: centroid-to-centroid or start-to-start.
