# Copyright (C) 2026
# TopologicPy Edge unit and regression tests.

"""
Consolidated tests for topologicpy.Edge.

The suite is backend-neutral unless a test is explicitly marked
``pythonocc_only`` or ``topologiccore_only``. Tests are organized by API theme
rather than by historical development tranche.
"""

import math

import pytest


Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Face = pytest.importorskip("topologicpy.Face").Face
Topology = pytest.importorskip("topologicpy.Topology").Topology
Core = pytest.importorskip("topologicpy.Core").Core


TOLERANCE = 1.0e-6


# ============================================================================
# Shared helpers and fixtures
# ============================================================================

def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _edge(start, end):
    return Edge.ByStartVertexEndVertex(_v(*start), _v(*end), silent=True)


def _assert_vertex(vertex):
    assert Topology.IsInstance(vertex, "Vertex")


def _assert_edge(edge):
    assert Topology.IsInstance(edge, "Edge")


def _coords(vertex, mantissa=None):
    return Vertex.Coordinates(vertex, mantissa=mantissa)


def _xyz(vertex):
    return _coords(vertex, mantissa=None)


def _assert_coords(vertex, expected, abs_tol=TOLERANCE, mantissa=6):
    actual = _coords(vertex, mantissa=mantissa)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


def _assert_xyz(vertex, expected, tol=TOLERANCE):
    actual = _coords(vertex, mantissa=None)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=tol)


def _assert_vector(actual, expected, abs_tol=TOLERANCE):
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


def _close(a, b, tol=1.0e-5):
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


def _close_xyz(a, b, tol=TOLERANCE):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _distance(a, b):
    pa = _coords(a, mantissa=None)
    pb = _coords(b, mantissa=None)
    return math.sqrt(sum((float(pa[i]) - float(pb[i])) ** 2 for i in range(3)))


def _norm(vector):
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


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


def _classification_line_edge():
    return Edge.ByStartVertexEndVertex(
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(3.0, 4.0, 0.0),
    )


def _parameter_line_edge():
    return Edge.ByStartVertexEndVertex(
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(4.0, 0.0, 0.0),
    )


def _wrap_occ_edge(shape):
    edge = Core.Edge.ByOcctShape(shape)
    assert edge is not None
    return edge


def _occ_circle_edge(radius=2.0):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

    circle = gp_Circ(
        gp_Ax2(
            gp_Pnt(0.0, 0.0, 0.0),
            gp_Dir(0.0, 0.0, 1.0),
        ),
        float(radius),
    )
    return _wrap_occ_edge(BRepBuilderAPI_MakeEdge(circle).Edge())


def _occ_bezier_shape(points):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.Geom import Geom_BezierCurve
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.gp import gp_Pnt

    poles = TColgp_Array1OfPnt(1, len(points))
    for index, (x, y, z) in enumerate(points, start=1):
        poles.SetValue(index, gp_Pnt(float(x), float(y), float(z)))
    return BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)).Edge()


def _occ_bezier_edge(points):
    return _wrap_occ_edge(_occ_bezier_shape(points))


def _occ_quadratic_bezier_edge(reversed_orientation=False):
    shape = _occ_bezier_shape(
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
        ]
    )
    if reversed_orientation:
        shape = shape.Reversed()
    return _wrap_occ_edge(shape)


def _quarter_circle_nurbs(radius=2.0):
    control_points = [
        Vertex.ByCoordinates(radius, 0.0, 0.0),
        Vertex.ByCoordinates(radius, radius, 0.0),
        Vertex.ByCoordinates(0.0, radius, 0.0),
    ]
    return Edge.ByNurbsParameters(
        controlPoints=control_points,
        weights=[1.0, math.sqrt(0.5), 1.0],
        knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        isRational=True,
        isPeriodic=False,
        degree=2,
        silent=True,
    )


# ============================================================================
# Construction, accessors, and basic topology
# ============================================================================


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


def test_external_boundary_returns_cluster_of_end_vertices(x_edge):
    boundary = Edge.ExternalBoundary(x_edge, silent=True)

    assert Topology.IsInstance(boundary, "Cluster")
    vertices = Topology.Vertices(boundary)
    assert len(vertices) == 2
    assert Edge.ExternalBoundary(None, silent=True) is None


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


# ============================================================================
# Measurements, parameters, and classification
# ============================================================================


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


def test_linear_edge_parameter_round_trip_and_tangent():
    edge = _parameter_line_edge()
    vertex = Edge.VertexByParameter(edge, u=0.25)

    assert vertex is not None
    assert _coords(vertex) == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, vertex, mantissa=9) == pytest.approx(0.25, abs=1.0e-7)
    assert Edge.TangentAtParameter(edge, u=0.25, mantissa=9) == pytest.approx(
        [1.0, 0.0, 0.0], abs=1.0e-7
    )


def test_parameter_endpoints_return_edge_endpoints():
    edge = _parameter_line_edge()
    start = Edge.StartVertex(edge)
    end = Edge.EndVertex(edge)

    assert _coords(Edge.VertexByParameter(edge, 0.0)) == pytest.approx(_coords(start), abs=1.0e-9)
    assert _coords(Edge.VertexByParameter(edge, 1.0)) == pytest.approx(_coords(end), abs=1.0e-9)
    assert Edge.ParameterAtVertex(edge, start) == pytest.approx(0.0, abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, end) == pytest.approx(1.0, abs=1.0e-7)


def test_parameter_methods_validate_inputs():
    edge = _parameter_line_edge()
    off_edge = Vertex.ByCoordinates(1.0, 1.0, 0.0)

    assert Edge.VertexByParameter(None, 0.5, silent=True) is None
    assert Edge.VertexByParameter(edge, "not-a-number", silent=True) is None
    assert Edge.VertexByParameter(edge, -0.1, silent=True) is None
    assert Edge.VertexByParameter(edge, 1.1, silent=True) is None
    assert Edge.ParameterAtVertex(None, off_edge, silent=True) is None
    assert Edge.ParameterAtVertex(edge, None, silent=True) is None
    assert Edge.ParameterAtVertex(edge, off_edge, silent=True) is None
    assert Edge.TangentAtParameter(None, silent=True) is None
    assert Edge.TangentAtParameter(edge, u="not-a-number", silent=True) is None


def test_edge_islinear_and_isclosed_for_straight_edge():
    edge = _classification_line_edge()
    assert edge is not None
    assert Edge.IsLinear(edge) is True
    assert Edge.IsClosed(edge) is False
    assert math.isclose(Edge.Length(edge), 5.0, rel_tol=0.0, abs_tol=1.0e-6)


def test_edge_islinear_isclosed_validate_inputs():
    assert Edge.IsLinear(None, silent=True) is None
    assert Edge.IsClosed(None, silent=True) is None
    edge = _classification_line_edge()
    assert Edge.IsLinear(edge, tolerance=0.0, silent=True) is None
    assert Edge.IsClosed(edge, tolerance=0.0, silent=True) is None
    assert Edge.IsLinear(edge, tolerance=float("inf"), silent=True) is None
    assert Edge.IsClosed(edge, tolerance=float("nan"), silent=True) is None


def test_length_none_returns_unrounded_float_and_direction_handles_closed_edge():
    arc = Edge.Arc(radius=1.0, fromAngle=0.0, toAngle=90.0, silent=True)
    length = Edge.Length(arc, mantissa=None, silent=True)
    assert isinstance(length, float)
    assert math.isclose(length, math.pi/2.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    circle = Edge.Circle(radius=1.0, silent=True)
    assert Edge.Direction(circle, mantissa=None, silent=True) is None


# ============================================================================
# Linear editing, intersections, and transformations
# ============================================================================


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
    normalized_to_end = Edge.Normalize(x_edge, useEndVertex=True, silent=True)
    normal_edge = Edge.NormalEdge(x_edge, length=3, u=0.5, silent=True)

    _assert_edge(normalized)
    assert Edge.Length(normalized) == pytest.approx(1)
    assert Edge.Direction(normalized) == [1, 0, 0]

    _assert_edge(normalized_to_end)
    assert Edge.Length(normalized_to_end) == pytest.approx(1)
    _assert_coords(_end(normalized_to_end), [10, 0, 0])

    _assert_edge(normal_edge)
    assert Edge.Length(normal_edge) == pytest.approx(3)
    _assert_coords(_start(normal_edge), [5, 0, 0])
    _assert_vector(Edge.Normal(x_edge), [0, 1, 0])

    assert Edge.Normalize(None, silent=True) is None
    assert Edge.Normal(None) is None
    assert Edge.NormalEdge(None, silent=True) is None
    assert Edge.NormalEdge(x_edge, length=0, silent=True) is None


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


def test_linear_only_operations_refuse_to_flatten_curves():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=120.0, silent=True)
    assert Edge.SetLength(arc, length=5.0, silent=True) is None
    assert Edge.Extend(arc, distance=1.0, silent=True) is None
    assert Edge.Normalize(arc, silent=True) is None


# ============================================================================
# Circles, arcs, and NURBS curves
# ============================================================================


def test_edge_circle_creates_closed_non_linear_exact_length():
    radius = 2.0
    circle = Edge.Circle(radius=radius, silent=True)

    assert Topology.IsInstance(circle, "Edge")
    assert Edge.IsClosed(circle, silent=True) is True
    assert Edge.IsLinear(circle, silent=True) is False
    assert math.isclose(Edge.Length(circle, mantissa=9), 2.0 * math.pi * radius, rel_tol=1e-7, abs_tol=1e-7)


def test_edge_circle_center_placement_and_parameter_samples():
    origin = Vertex.ByCoordinates(10.0, -3.0, 7.0)
    radius = 3.0
    circle = Edge.Circle(origin=origin, radius=radius, direction=[0, 0, 1], placement="center", silent=True)

    assert Topology.IsInstance(circle, "Edge")
    for u in (0.0, 0.25, 0.5, 0.75):
        vertex = Edge.VertexByParameter(circle, u=u, silent=True)
        assert Topology.IsInstance(vertex, "Vertex")
        assert math.isclose(_distance(vertex, origin), radius, rel_tol=1e-7, abs_tol=1e-7)
        assert math.isclose(Vertex.Z(vertex, mantissa=9), 7.0, rel_tol=0.0, abs_tol=1e-7)


def test_edge_circle_corner_placement():
    origin = Vertex.ByCoordinates(5.0, 8.0, 0.0)
    radius = 2.0
    circle = Edge.Circle(origin=origin, radius=radius, placement="lowerleft", silent=True)

    assert Topology.IsInstance(circle, "Edge")
    samples = [Edge.VertexByParameter(circle, u=u, silent=True) for u in (0.0, 0.25, 0.5, 0.75)]
    xs = [Vertex.X(v, mantissa=9) for v in samples]
    ys = [Vertex.Y(v, mantissa=9) for v in samples]
    assert math.isclose(min(xs), 5.0, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(min(ys), 8.0, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(max(xs), 9.0, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(max(ys), 12.0, rel_tol=0.0, abs_tol=1e-7)


def test_edge_circle_oriented_plane():
    origin = Vertex.ByCoordinates(1.0, 2.0, 3.0)
    radius = 1.5
    circle = Edge.Circle(origin=origin, radius=radius, direction=[0, 1, 0], placement="center", silent=True)

    assert Topology.IsInstance(circle, "Edge")
    for u in (0.0, 0.25, 0.5, 0.75):
        vertex = Edge.VertexByParameter(circle, u=u, silent=True)
        assert Topology.IsInstance(vertex, "Vertex")
        # Plane normal is +Y, so every point lies at y=2.
        assert math.isclose(Vertex.Y(vertex, mantissa=9), 2.0, rel_tol=0.0, abs_tol=1e-7)
        assert math.isclose(_distance(vertex, origin), radius, rel_tol=1e-7, abs_tol=1e-7)


def test_edge_circle_validates_inputs():
    assert Edge.Circle(origin="not a vertex", silent=True) is None
    assert Edge.Circle(radius=0.0, silent=True) is None
    assert Edge.Circle(radius=1e-6, tolerance=1e-4, silent=True) is None
    assert Edge.Circle(direction=[0, 0, 0], silent=True) is None
    assert Edge.Circle(direction=[0, 1], silent=True) is None
    assert Edge.Circle(placement="banana", silent=True) is None
    assert Edge.Circle(tolerance=0.0, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_circle_is_native_occt_circle():
    circle = Edge.Circle(radius=2.5, silent=True)
    assert Topology.IsInstance(circle, "Edge")

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Circle

    adaptor = BRepAdaptor_Curve(circle.shape)
    assert adaptor.GetType() == GeomAbs_Circle


def test_edge_arc_quarter_circle_exact_geometry():
    radius = 2.0
    arc = Edge.Arc(radius=radius, fromAngle=0.0, toAngle=90.0, silent=True)

    assert Topology.IsInstance(arc, "Edge")
    assert Edge.IsClosed(arc, silent=True) is False
    assert Edge.IsLinear(arc, silent=True) is False
    assert math.isclose(
        Edge.Length(arc, mantissa=9),
        0.5 * math.pi * radius,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )
    _assert_xyz(Edge.StartVertex(arc, silent=True), [radius, 0.0, 0.0])
    _assert_xyz(Edge.EndVertex(arc, silent=True), [0.0, radius, 0.0])
    expected = radius / math.sqrt(2.0)
    _assert_xyz(Edge.VertexByParameter(arc, u=0.5, silent=True), [expected, expected, 0.0], tol=2.0e-6)


def test_edge_arc_wraps_angles_counter_clockwise():
    radius = 1.5
    arc = Edge.Arc(radius=radius, fromAngle=300.0, toAngle=60.0, silent=True)

    assert Topology.IsInstance(arc, "Edge")
    expected_length = radius * math.radians(120.0)
    assert math.isclose(Edge.Length(arc, mantissa=9), expected_length, rel_tol=1.0e-6, abs_tol=1.0e-6)

    a0 = math.radians(300.0)
    a1 = math.radians(60.0)
    _assert_xyz(Edge.StartVertex(arc, silent=True), [radius * math.cos(a0), radius * math.sin(a0), 0.0], tol=2.0e-6)
    _assert_xyz(Edge.EndVertex(arc, silent=True), [radius * math.cos(a1), radius * math.sin(a1), 0.0], tol=2.0e-6)


def test_edge_arc_start_placement_and_oriented_plane():
    origin = Vertex.ByCoordinates(10.0, -3.0, 7.0)
    radius = 1.25
    arc = Edge.Arc(
        origin=origin,
        radius=radius,
        fromAngle=0.0,
        toAngle=90.0,
        direction=[0.0, 1.0, 0.0],
        placement="start",
        silent=True,
    )

    assert Topology.IsInstance(arc, "Edge")
    assert _distance(Edge.StartVertex(arc, silent=True), origin) <= 2.0e-6
    assert math.isclose(Edge.Length(arc, mantissa=9), 0.5 * math.pi * radius, rel_tol=1.0e-6, abs_tol=1.0e-6)

    # The arc plane normal is +Y, so every sampled point lies in y = origin.y.
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        vertex = Edge.VertexByParameter(arc, u=u, silent=True)
        assert Topology.IsInstance(vertex, "Vertex")
        assert math.isclose(Vertex.Y(vertex, mantissa=9), -3.0, rel_tol=0.0, abs_tol=2.0e-6)


def test_edge_arc_validates_inputs():
    assert Edge.Arc(origin="not a vertex", silent=True) is None
    assert Edge.Arc(radius=0.0, silent=True) is None
    assert Edge.Arc(radius=1.0e-6, tolerance=1.0e-4, silent=True) is None
    assert Edge.Arc(fromAngle=10.0, toAngle=10.0, silent=True) is None
    assert Edge.Arc(fromAngle=0.0, toAngle=360.0, silent=True) is None
    assert Edge.Arc(direction=[0.0, 0.0, 0.0], silent=True) is None
    assert Edge.Arc(direction=[0.0, 1.0], silent=True) is None
    assert Edge.Arc(placement="banana", silent=True) is None
    assert Edge.Arc(tolerance=0.0, silent=True) is None


def test_edge_bynurbsparameters_exact_quarter_circle():
    radius = 2.0
    edge = _quarter_circle_nurbs(radius)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsClosed(edge, silent=True) is False
    assert Edge.IsLinear(edge, silent=True) is False
    assert math.isclose(
        Edge.Length(edge, mantissa=9),
        0.5 * math.pi * radius,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

    _assert_xyz(Edge.StartVertex(edge, silent=True), [radius, 0.0, 0.0])
    _assert_xyz(Edge.EndVertex(edge, silent=True), [0.0, radius, 0.0])
    midpoint = Edge.VertexByParameter(edge, u=0.5, silent=True)
    expected = radius / math.sqrt(2.0)
    _assert_xyz(midpoint, [expected, expected, 0.0], tol=2.0e-6)


def test_edge_bynurbsparameters_nonrational_quadratic():
    control_points = [
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 1.0, 0.0),
        Vertex.ByCoordinates(2.0, 0.0, 0.0),
    ]
    edge = Edge.ByNurbsParameters(
        controlPoints=control_points,
        knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        degree=2,
        isRational=False,
        silent=True,
    )

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    _assert_xyz(Edge.VertexByParameter(edge, u=0.5, silent=True), [1.0, 0.5, 0.0], tol=2.0e-6)
    assert Edge.Length(edge, mantissa=9) > 2.0


def test_edge_bynurbsparameters_validates_inputs():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)
    p2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)

    assert Edge.ByNurbsParameters([], silent=True) is None
    assert Edge.ByNurbsParameters([p0, p1], degree=2, silent=True) is None
    assert Edge.ByNurbsParameters([p0, p1, p2], weights=[1.0, 1.0], degree=2, silent=True) is None
    assert Edge.ByNurbsParameters([p0, p1, p2], weights=[1.0, 0.0, 1.0], degree=2, silent=True) is None
    assert Edge.ByNurbsParameters(
        [p0, p1, p2],
        knots=[0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
        degree=2,
        silent=True,
    ) is None


@pytest.mark.pythonocc_only
def test_pythonocc_nurbs_and_arc_use_native_curve_types():
    nurbs = _quarter_circle_nurbs(2.0)
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=90.0, silent=True)
    assert Topology.IsInstance(nurbs, "Edge")
    assert Topology.IsInstance(arc, "Edge")

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_Circle

    assert BRepAdaptor_Curve(nurbs.shape).GetType() == GeomAbs_BSplineCurve
    assert BRepAdaptor_Curve(arc.shape).GetType() == GeomAbs_Circle


# ============================================================================
# Bezier, spline, conic, and helix construction
# ============================================================================


def test_bezier_quadratic_geometry_and_parameterization():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 2.0, 0.0)
    p2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)

    edge = Edge.Bezier([p0, p1, p2], silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsClosed(edge, silent=True) is False
    assert Edge.IsLinear(edge, silent=True) is False
    assert _close_xyz(_coords(Edge.StartVertex(edge)), [0.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.EndVertex(edge)), [2.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.VertexByParameter(edge, 0.5, silent=True)), [1.0, 1.0, 0.0])


def test_bezier_collinear_control_points_remain_linear():
    points = [
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 0.0, 0.0),
        Vertex.ByCoordinates(3.0, 0.0, 0.0),
        Vertex.ByCoordinates(4.0, 0.0, 0.0),
    ]
    edge = Edge.Bezier(points, silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, tolerance=1.0e-6, silent=True) is True
    assert math.isclose(Edge.Length(edge, mantissa=9), 4.0, rel_tol=1.0e-8, abs_tol=1.0e-8)


def test_rational_bezier_exact_quarter_circle():
    s2 = math.sqrt(2.0) / 2.0
    points = [
        Vertex.ByCoordinates(1.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 1.0, 0.0),
        Vertex.ByCoordinates(0.0, 1.0, 0.0),
    ]
    edge = Edge.Bezier(points, weights=[1.0, s2, 1.0], silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    mid = _coords(Edge.VertexByParameter(edge, 0.5, silent=True))
    assert _close_xyz(mid, [s2, s2, 0.0], tol=1.0e-6)
    assert math.isclose(Edge.Length(edge, mantissa=9), math.pi / 2.0, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_bezier_reversed_control_points_preserve_direction():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 2.0, 0.0)
    p2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)

    edge = Edge.Bezier([p2, p1, p0], silent=True)

    assert Topology.IsInstance(edge, "Edge")
    assert _close_xyz(_coords(Edge.StartVertex(edge)), [2.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.EndVertex(edge)), [0.0, 0.0, 0.0])
    assert _close_xyz(_coords(Edge.VertexByParameter(edge, 0.5, silent=True)), [1.0, 1.0, 0.0])
    tangent = Edge.TangentAtParameter(edge, 0.25, mantissa=None, silent=True)
    assert tangent is not None
    assert tangent[0] < 0.0


def test_bezier_validates_inputs():
    p0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    p1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)

    assert Edge.Bezier(None, silent=True) is None
    assert Edge.Bezier([p0], silent=True) is None
    assert Edge.Bezier([p0, None, p1], silent=True) is None
    assert Edge.Bezier([p0, p1], weights=[1.0], silent=True) is None
    assert Edge.Bezier([p0, p1], weights=[1.0, 0.0], silent=True) is None
    assert Edge.Bezier([p0, p1], weights=[1.0, float("nan")], silent=True) is None
    assert Edge.Bezier([p0, p1], tolerance=0.0, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_bezier_is_native_bspline_curve():
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve

    edge = Edge.Bezier(
        [
            Vertex.ByCoordinates(0.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 2.0, 0.0),
            Vertex.ByCoordinates(2.0, 0.0, 0.0),
        ],
        silent=True,
    )
    assert Topology.IsInstance(edge, "Edge")
    adaptor = BRepAdaptor_Curve(edge.shape)
    assert adaptor.GetType() == GeomAbs_BSplineCurve


def test_bycurve_creates_single_non_linear_bspline_edge():
    points = [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(1, 2, 0),
        Vertex.ByCoordinates(3, 2, 0),
        Vertex.ByCoordinates(4, 0, 0),
    ]
    edge = Edge.ByCurve(points, degree=3, silent=True)
    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    assert _close(_xyz(Edge.StartVertex(edge, silent=True)), [0, 0, 0])
    assert _close(_xyz(Edge.EndVertex(edge, silent=True)), [4, 0, 0])


def test_parabola_is_exact_quadratic_conic():
    f = 0.75
    edge = Edge.Parabola(focalLength=f, fromParameter=-1.5, toParameter=1.5, silent=True)
    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    for u in [0.0, 0.2, 0.5, 0.8, 1.0]:
        x, y, z = _xyz(Edge.VertexByParameter(edge, u, silent=True))
        assert abs(z) <= 1.0e-7
        assert math.isclose(y, x*x/(4.0*f), rel_tol=2.0e-6, abs_tol=2.0e-6)


def test_hyperbola_is_exact_rational_conic_on_both_branches():
    a, b = 2.0, 0.8
    for branch, sign in [("right", 1.0), ("left", -1.0)]:
        edge = Edge.Hyperbola(a=a, b=b, fromParameter=-0.8, toParameter=0.8, branch=branch, silent=True)
        assert Topology.IsInstance(edge, "Edge")
        assert Edge.IsLinear(edge, silent=True) is False
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x, y, z = _xyz(Edge.VertexByParameter(edge, u, silent=True))
            assert sign*x > 0.0
            assert abs(z) <= 1.0e-7
            value = x*x/(a*a) - y*y/(b*b)
            assert math.isclose(value, 1.0, rel_tol=3.0e-6, abs_tol=3.0e-6)


def test_helix_is_one_curved_edge_with_expected_endpoints_and_length():
    radius, height, turns = 1.25, 3.0, 1.5
    edge = Edge.Helix(radius=radius, height=height, turns=turns, sides=20, placement="base", silent=True)
    assert Topology.IsInstance(edge, "Edge")
    assert Edge.IsLinear(edge, silent=True) is False
    start = _xyz(Edge.StartVertex(edge, silent=True))
    end = _xyz(Edge.EndVertex(edge, silent=True))
    expected_end = [radius * math.cos(2*math.pi*turns), radius * math.sin(2*math.pi*turns), height]
    assert _close(start, [radius, 0.0, 0.0], tol=2.0e-5)
    assert _close(end, expected_end, tol=2.0e-5)
    analytic = math.sqrt((2.0*math.pi*radius*turns)**2 + height**2)
    actual = Edge.Length(edge, mantissa=None, silent=True)
    assert actual is not None
    assert math.isclose(actual, analytic, rel_tol=3.0e-4, abs_tol=3.0e-4)


@pytest.mark.pythonocc_only
def test_pythonocc_special_curves_are_native_bspline_edges():
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve

    curves = [
        Edge.ByCurve([
            Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(1, 2, 0),
            Vertex.ByCoordinates(3, 2, 0), Vertex.ByCoordinates(4, 0, 0),
        ], silent=True),
        Edge.Parabola(silent=True),
        Edge.Hyperbola(silent=True),
        Edge.Helix(sides=12, silent=True),
    ]
    assert all(Topology.IsInstance(e, "Edge") for e in curves)
    for edge in curves:
        adaptor = BRepAdaptor_Curve(edge.shape)
        assert adaptor.GetType() == GeomAbs_BSplineCurve


# ============================================================================
# Curved-edge classification, frames, distance, and integrity
# ============================================================================


@pytest.mark.pythonocc_only
def test_pythonocc_closed_circle_is_closed_and_not_linear():
    edge = _occ_circle_edge(radius=2.0)
    assert Edge.IsClosed(edge) is True
    assert Edge.IsLinear(edge) is False
    assert math.isclose(Edge.Length(edge), 4.0 * math.pi, rel_tol=1.0e-6, abs_tol=1.0e-6)


@pytest.mark.pythonocc_only
def test_pythonocc_straight_bezier_is_geometrically_linear():
    edge = _occ_bezier_edge([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ])
    assert Edge.IsClosed(edge) is False
    assert Edge.IsLinear(edge) is True
    assert math.isclose(Edge.Length(edge), 2.0, rel_tol=1.0e-6, abs_tol=1.0e-6)


@pytest.mark.pythonocc_only
def test_pythonocc_curved_bezier_is_not_linear():
    edge = _occ_bezier_edge([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
    ])
    assert Edge.IsClosed(edge) is False
    assert Edge.IsLinear(edge) is False
    assert Edge.Length(edge) > 2.0


@pytest.mark.pythonocc_only
def test_pythonocc_collinear_backtracking_bezier_is_not_linear():
    edge = _occ_bezier_edge([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ])
    assert Edge.IsClosed(edge) is False
    assert Edge.IsLinear(edge) is False
    assert Edge.Length(edge) > 1.0


def test_curve_normal_is_unit_and_perpendicular_to_tangent():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=180.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    tangent = Edge.TangentAtParameter(arc, u=0.5, mantissa=None, silent=True)
    normal = Edge.NormalAtParameter(arc, u=0.5, mantissa=None, silent=True)
    assert tangent is not None and normal is not None
    assert math.isclose(_norm(tangent), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert math.isclose(_norm(normal), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert abs(_dot(tangent, normal)) <= 2.0e-5
    # Midpoint of the upper semicircle is (0, 2, 0); principal normal points inward.
    midpoint = _xyz(Edge.VertexByParameter(arc, 0.5, silent=True))
    radial_inward = [-midpoint[0], -midpoint[1], -midpoint[2]]
    rmag = _norm(radial_inward)
    radial_inward = [x/rmag for x in radial_inward]
    assert _dot(normal, radial_inward) > 0.999


def test_normal_and_normaledge_use_local_curve_frame():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=180.0, silent=True)
    normal = Edge.Normal(arc, silent=True)
    assert normal is not None
    assert math.isclose(_norm(normal), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
    nedge = Edge.NormalEdge(arc, length=2.5, u=0.5, silent=True)
    assert Topology.IsInstance(nedge, "Edge")
    assert math.isclose(Edge.Length(nedge, mantissa=6, silent=True), 2.5, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert _close(_xyz(Edge.StartVertex(nedge, silent=True)), _xyz(Edge.VertexByParameter(arc, 0.5, silent=True)))


def test_vertex_by_distance_uses_curvilinear_distance_on_arc():
    radius = 2.0
    arc = Edge.Arc(radius=radius, fromAngle=0.0, toAngle=180.0, silent=True)
    half_length = 0.5 * Edge.Length(arc, mantissa=None, silent=True)
    point = Edge.VertexByDistance(arc, distance=half_length, mantissa=None, silent=True)
    expected = Edge.VertexByParameter(arc, 0.5, silent=True)
    assert Topology.IsInstance(point, "Vertex")
    assert _close(_xyz(point), _xyz(expected), tol=2.0e-5)


def test_vertex_by_distance_wraps_on_closed_circle():
    circle = Edge.Circle(radius=1.5, silent=True)
    circumference = Edge.Length(circle, mantissa=None, silent=True)
    quarter = Edge.VertexByDistance(circle, distance=1.25*circumference, mantissa=None, silent=True)
    expected = Edge.VertexByParameter(circle, 0.25, silent=True)
    assert Topology.IsInstance(quarter, "Vertex")
    assert _close(_xyz(quarter), _xyz(expected), tol=5.0e-5)


@pytest.mark.pythonocc_only
def test_reverse_preserves_curved_geometry_and_orientation():
    arc = Edge.Arc(radius=3.0, fromAngle=20.0, toAngle=140.0, silent=True)
    rev = Edge.Reverse(arc, silent=True)
    assert Topology.IsInstance(rev, "Edge")
    assert Edge.IsLinear(rev, silent=True) is False
    assert math.isclose(Edge.Length(rev, mantissa=None, silent=True), Edge.Length(arc, mantissa=None, silent=True), rel_tol=1.0e-8, abs_tol=1.0e-8)
    assert _close(_xyz(Edge.StartVertex(rev, silent=True)), _xyz(Edge.EndVertex(arc, silent=True)))
    assert _close(_xyz(Edge.EndVertex(rev, silent=True)), _xyz(Edge.StartVertex(arc, silent=True)))
    assert _close(_xyz(Edge.VertexByParameter(rev, 0.5, silent=True)), _xyz(Edge.VertexByParameter(arc, 0.5, silent=True)), tol=2.0e-6)
    ta = Edge.TangentAtParameter(arc, 0.5, mantissa=None, silent=True)
    tr = Edge.TangentAtParameter(rev, 0.5, mantissa=None, silent=True)
    assert _dot(ta, tr) < -0.999999


@pytest.mark.topologiccore_only
def test_topologiccore_unsupported_curve_operations_do_not_flatten():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=120.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    assert Edge.IsLinear(arc, silent=True) is False
    # TopologicCore exposes NURBS construction/evaluation but not an exact
    # curve-reversal or curve-trim API. Returning None is safer than silently
    # rebuilding the operation as a straight chord or sampled approximation.
    assert Edge.Reverse(arc, silent=True) is None
    assert Edge.Trim(arc, distance=0.5, bothSides=True, silent=True) is None


# ============================================================================
# Curve parameterization and trimming
# ============================================================================


@pytest.mark.pythonocc_only
def test_pythonocc_quadratic_bezier_parameter_evaluation():
    edge = _occ_quadratic_bezier_edge()

    quarter = Edge.VertexByParameter(edge, u=0.25)
    middle = Edge.VertexByParameter(edge, u=0.5)

    assert _coords(quarter) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(middle) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, quarter, mantissa=9) == pytest.approx(0.25, abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, middle, mantissa=9) == pytest.approx(0.5, abs=1.0e-7)


@pytest.mark.pythonocc_only
def test_pythonocc_quadratic_bezier_tangent_uses_curve_derivative():
    edge = _occ_quadratic_bezier_edge()

    expected_quarter = [2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 0.0]
    assert Edge.TangentAtParameter(edge, u=0.25, mantissa=9) == pytest.approx(
        expected_quarter, abs=1.0e-7
    )
    assert Edge.TangentAtParameter(edge, u=0.5, mantissa=9) == pytest.approx(
        [1.0, 0.0, 0.0], abs=1.0e-7
    )


@pytest.mark.pythonocc_only
def test_pythonocc_reversed_curve_respects_topological_orientation():
    edge = _occ_quadratic_bezier_edge(reversed_orientation=True)

    quarter = Edge.VertexByParameter(edge, u=0.25)
    assert _coords(quarter) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, quarter, mantissa=9) == pytest.approx(0.25, abs=1.0e-7)

    expected_tangent = [-2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 0.0]
    assert Edge.TangentAtParameter(edge, u=0.25, mantissa=9) == pytest.approx(
        expected_tangent, abs=1.0e-7
    )


def test_trim_by_parameters_linear_forward():
    edge = _parameter_line_edge()
    trimmed = Edge.TrimByParameters(edge, 0.25, 0.75)

    assert trimmed is not None
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([3.0, 0.0, 0.0], abs=1.0e-7)
    assert Edge.Length(trimmed, mantissa=9) == pytest.approx(2.0, abs=1.0e-7)
    assert Edge.IsLinear(trimmed)


def test_trim_by_parameters_linear_reverse_direction():
    edge = _parameter_line_edge()
    trimmed = Edge.TrimByParameters(edge, 0.75, 0.25)

    assert trimmed is not None
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([3.0, 0.0, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-7)
    assert Edge.Length(trimmed, mantissa=9) == pytest.approx(2.0, abs=1.0e-7)


def test_trim_by_parameters_identity_full_reverse_and_validation():
    edge = _parameter_line_edge()

    assert Edge.TrimByParameters(edge, 0.0, 1.0) is edge

    reversed_edge = Edge.TrimByParameters(edge, 1.0, 0.0)
    assert reversed_edge is not None
    assert _coords(Edge.StartVertex(reversed_edge)) == pytest.approx([4.0, 0.0, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(reversed_edge)) == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-7)

    assert Edge.TrimByParameters(None, 0.25, 0.75, silent=True) is None
    assert Edge.TrimByParameters(edge, "bad", 0.75, silent=True) is None
    assert Edge.TrimByParameters(edge, -0.1, 0.75, silent=True) is None
    assert Edge.TrimByParameters(edge, 0.25, 1.1, silent=True) is None
    assert Edge.TrimByParameters(edge, 0.5, 0.5, silent=True) is None
    assert Edge.TrimByParameters(edge, 0.25, 0.75, tolerance=0, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_trimmed_bezier_preserves_curve_geometry():
    edge = _occ_quadratic_bezier_edge()
    trimmed = Edge.TrimByParameters(edge, 0.25, 0.75)

    assert trimmed is not None
    assert Edge.IsLinear(trimmed) is False
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.VertexByParameter(trimmed, 0.5)) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)
    assert Edge.Length(trimmed, mantissa=9) == pytest.approx(1.040228819, abs=1.0e-7)


@pytest.mark.pythonocc_only
def test_pythonocc_reverse_trimmed_bezier_preserves_curve_and_direction():
    edge = _occ_quadratic_bezier_edge()
    trimmed = Edge.TrimByParameters(edge, 0.75, 0.25)

    assert trimmed is not None
    assert Edge.IsLinear(trimmed) is False
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.VertexByParameter(trimmed, 0.5)) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)
    assert Edge.TangentAtParameter(trimmed, 0.5, mantissa=9) == pytest.approx([-1.0, 0.0, 0.0], abs=1.0e-7)


@pytest.mark.pythonocc_only
def test_pythonocc_trim_on_reversed_source_uses_topological_parameters():
    edge = _occ_quadratic_bezier_edge(reversed_orientation=True)
    trimmed = Edge.TrimByParameters(edge, 0.25, 0.75)

    assert trimmed is not None
    assert Edge.IsLinear(trimmed) is False
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.VertexByParameter(trimmed, 0.5)) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)


@pytest.mark.pythonocc_only
def test_distance_trim_preserves_arc_and_exact_remaining_length():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=180.0, silent=True)
    original = Edge.Length(arc, mantissa=None, silent=True)
    trimmed = Edge.Trim(arc, distance=1.0, bothSides=True, silent=True)
    assert Topology.IsInstance(trimmed, "Edge")
    assert Edge.IsLinear(trimmed, silent=True) is False
    assert math.isclose(Edge.Length(trimmed, mantissa=None, silent=True), original - 1.0, rel_tol=2.0e-5, abs_tol=2.0e-5)
    assert _close(_xyz(Edge.VertexByParameter(trimmed, 0.5, silent=True)), _xyz(Edge.VertexByParameter(arc, 0.5, silent=True)), tol=3.0e-5)
