# Copyright (C) 2026
# Wassim Jabi
#
# Unified TopologicPy Face regression and unit test suite.

"""
Consolidated tests for TopologicPy.Face.

The suite is organised by behaviour rather than by historical tranche/file.
Backend-specific tests use the shared ``pythonocc_only`` and
``topologiccore_only`` pytest markers defined by tests/conftest.py.
"""

import math

import pytest


Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary


GEOMETRY_TOLERANCE = 1.0e-6
SURFACE_TOLERANCE = 1.0e-4


# ============================================================================
# Shared fixtures and geometry helpers
# ============================================================================


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _assert_vertex(vertex):
    assert Topology.IsInstance(vertex, "Vertex")


def _assert_edge(edge):
    assert Topology.IsInstance(edge, "Edge")


def _assert_wire(wire):
    assert Topology.IsInstance(wire, "Wire")


def _assert_face(face):
    assert Topology.IsInstance(face, "Face")


def _coords(vertex, mantissa=None):
    return Vertex.Coordinates(vertex, mantissa=mantissa)


def _xyz(vertex):
    return [float(value) for value in Vertex.Coordinates(vertex, mantissa=None)]


def _assert_coords(vertex, expected, abs_tol=GEOMETRY_TOLERANCE, mantissa=6):
    actual = _coords(vertex, mantissa=mantissa)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


def _assert_vector(actual, expected, abs_tol=GEOMETRY_TOLERANCE):
    assert actual is not None
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert float(value) == pytest.approx(float(target), abs=abs_tol)


def _close(a, b, tol=2.0e-5):
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


def _dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _mag(values):
    return math.sqrt(_dot(values, values))


def _unit(values):
    magnitude = math.sqrt(sum(float(value) * float(value) for value in values))
    return [float(value) / magnitude for value in values]


def _reverse(vector):
    return [-vector[0], -vector[1], -vector[2]]


def _rotate_x(values, angle_deg):
    x, y, z = [float(value) for value in values]
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return [x, c * y - s * z, s * y + c * z]


def _planar_rectangle():
    return Face.Rectangle(
        origin=_v(0.0, 0.0, 0.0),
        width=4.0,
        length=2.0,
        placement="lowerleft",
        direction=[0, 0, 1],
        silent=True,
    )


def _quarter_cylinder_nurbs(radius=1.0, height=1.0):
    s2 = math.sqrt(2.0) / 2.0
    radius = float(radius)
    height = float(height)
    control_points = [
        [_v(radius, 0.0, 0.0), _v(radius, 0.0, height)],
        [_v(radius, radius, 0.0), _v(radius, radius, height)],
        [_v(0.0, radius, 0.0), _v(0.0, radius, height)],
    ]
    weights = [
        [1.0, 1.0],
        [s2, s2],
        [1.0, 1.0],
    ]
    return Face.ByNurbsParameters(
        control_points,
        weights=weights,
        uKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 1.0, 1.0],
        isRational=True,
        uDegree=2,
        vDegree=1,
        silent=True,
    )


def _curved_nurbs_face():
    z = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    control_points = [
        [_v(float(i), float(j), z[i][j]) for j in range(4)]
        for i in range(4)
    ]
    return Face.ByNurbsParameters(
        controlPoints=control_points,
        uDegree=3,
        vDegree=3,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )


def _general_nurbs_face():
    xs = [-2.0, -0.6666666667, 0.6666666667, 2.0]
    ys = [-2.0, -0.6666666667, 0.6666666667, 2.0]
    z = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.2, 0.7, 0.0],
        [0.0, 0.4, 1.4, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    control_points = [
        [_v(x, y, z[i][j]) for j, y in enumerate(ys)]
        for i, x in enumerate(xs)
    ]
    return Face.ByNurbsParameters(
        controlPoints=control_points,
        weights=None,
        uKnots=None,
        vKnots=None,
        isRational=False,
        isUPeriodic=False,
        isVPeriodic=False,
        uDegree=3,
        vDegree=3,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )


def _circle_wire(z=0.0, radius=1.0):
    edge = Edge.Circle(
        origin=_v(0.0, 0.0, float(z)),
        radius=float(radius),
        placement="center",
        silent=True,
    )
    _assert_edge(edge)
    wire = Wire.ByEdges([edge], silent=True)
    _assert_wire(wire)
    return wire


@pytest.fixture
def rectangle_face():
    return Face.Rectangle(width=4, length=2, placement="center", silent=True)


@pytest.fixture
def square_face():
    return Face.Square(size=2)


@pytest.fixture
def holed_face():
    outer = Wire.Rectangle(width=4, length=4, placement="center", silent=True)
    inner = Wire.Rectangle(width=1, length=1, placement="center", silent=True)
    return Face.ByWires(outer, [inner], silent=True)


# ============================================================================
# Construction, primitive factories, and topology conversion
# ============================================================================


def test_rectangle_square_circle_and_basic_area(rectangle_face, square_face):
    circle = Face.Circle(radius=1, sides=32)
    lowerleft = Face.Rectangle(
        origin=_v(0, 0, 0),
        width=4,
        length=2,
        placement="lowerleft",
        silent=True,
    )

    for face in [rectangle_face, square_face, circle, lowerleft]:
        _assert_face(face)
        assert Face.Area(face) > 0

    assert Face.Area(rectangle_face) == pytest.approx(8)
    assert Face.Area(square_face) == pytest.approx(4)
    assert Face.Rectangle(width=0, length=2, silent=True) is None
    assert Face.Rectangle(width=2, length=2, placement="invalid", silent=True) is None
    assert Face.Rectangle(width=2, length=2, direction=[0, 0, 0], silent=True) is None
    assert Face.Circle(radius=0) is None
    assert Face.Area(None) is None


def test_by_vertices_wire_edges_and_clusters_create_faces():
    vertices = [_v(0, 0, 0), _v(4, 0, 0), _v(4, 2, 0), _v(0, 2, 0)]
    wire = Wire.ByVertices(vertices, close=True, silent=True)
    edges = Wire.Edges(wire)
    vertex_cluster = Cluster.ByTopologies(vertices)
    edge_cluster = Cluster.ByTopologies(edges)

    faces = [
        Face.ByVertices(vertices, silent=True),
        Face.ByWire(wire, silent=True),
        Face.ByEdges(edges, silent=True),
        Face.ByVerticesCluster(vertex_cluster, silent=True),
        Face.ByEdgesCluster(edge_cluster, silent=True),
    ]

    for face in faces:
        _assert_face(face)
        assert Face.Area(face) == pytest.approx(8)

    assert Face.ByVertices(None, silent=True) is None
    assert Face.ByVertices([_v(0, 0, 0), _v(1, 0, 0)], silent=True) is None
    assert Face.ByWire(None, silent=True) is None
    assert Face.ByEdges(None, silent=True) is None
    assert Face.ByEdges([], silent=True) is None
    assert Face.ByVerticesCluster(None, silent=True) is None
    assert Face.ByEdgesCluster(None, silent=True) is None


def test_shape_constructors_return_valid_faces():
    constructors = [
        Face.CrossShape(width=4, length=4, silent=True),
        Face.CShape(width=4, length=4, silent=True),
        Face.IShape(width=4, length=4, silent=True),
        Face.LShape(width=4, length=4, silent=True),
        Face.TShape(width=4, length=4, silent=True),
        Face.Trapezoid(widthA=4, widthB=2, length=3),
        Face.Star(radiusA=2, radiusB=1, rays=5),
        Face.Ellipse(width=4, length=2, sides=32, polyline=True),
        Face.Einstein(radius=1),
        Face.NorthArrow(radius=1),
    ]

    for face in constructors:
        _assert_face(face)
        assert Face.Area(face) > 0

    assert Face.CrossShape(width=0, length=4, silent=True) is None
    assert Face.CShape(width=0, length=4, silent=True) is None
    assert Face.IShape(width=0, length=4, silent=True) is None
    assert Face.LShape(width=0, length=4, silent=True) is None
    assert Face.TShape(width=0, length=4, silent=True) is None


def test_hollow_section_constructors_return_faces():
    faces = [
        Face.CHS(radius=2, thickness=0.5, sides=24, silent=True),
        Face.Ring(radius=2, thickness=0.5, sides=24, silent=True),
        Face.RHS(width=4, length=3, thickness=0.25, silent=True),
        Face.SHS(size=4, thickness=0.25, silent=True),
    ]

    for face in faces:
        _assert_face(face)
        assert Face.Area(face) > 0
        assert len(Face.InternalBoundaries(face)) >= 1

    assert Face.CHS(radius=1, thickness=1, silent=True) is None
    assert Face.Ring(radius=1, thickness=1, silent=True) is None
    assert Face.RHS(width=1, length=1, thickness=0.5, silent=True) is None
    assert Face.SHS(size=1, thickness=0.5, silent=True) is None


def test_rectangle_by_plane_equation_creates_oriented_face():
    equation = {"a": 0, "b": 0, "c": 1, "d": 0}
    face = Face.RectangleByPlaneEquation(width=2, length=3, equation=equation)

    _assert_face(face)
    assert Face.Area(face) == pytest.approx(6)


def test_invalid_inputs_for_selected_shape_factories():
    assert Face.Ellipse(width=0, length=1) is None
    assert Face.Star(radiusA=0, radiusB=1) is None
    assert Face.Trapezoid(widthA=0, widthB=1, length=1) is None
    assert Face.Square(size=0) is None


# ============================================================================
# Boundaries, accessors, area, compactness, and convexity
# ============================================================================


def test_by_wires_and_internal_boundary_accessors(holed_face):
    _assert_face(holed_face)

    external = Face.ExternalBoundary(holed_face, silent=True)
    internal = Face.InternalBoundaries(holed_face)
    wires = Face.Wires(holed_face)
    alias = Face.Wire(holed_face)

    _assert_wire(external)
    _assert_wire(alias)
    assert isinstance(internal, list)
    assert len(internal) == 1
    assert isinstance(wires, list)
    assert len(wires) >= 2
    assert Face.Area(holed_face) < 16

    outer = Wire.Rectangle(width=4, length=4, placement="center", silent=True)
    assert Face.ByWires(outer, None, silent=True) is None
    assert Face.ByWires(None, [], silent=True) is None
    assert Face.ByWiresCluster(outer, None, silent=True) is not None
    assert Face.ByWiresCluster(None, None, silent=True) is None


def test_add_internal_boundaries_accepts_lists_and_clusters(rectangle_face):
    hole = Wire.Rectangle(width=1, length=1, placement="center", silent=True)
    face_from_list = Face.AddInternalBoundaries(rectangle_face, [hole])
    face_from_cluster = Face.AddInternalBoundariesCluster(
        rectangle_face,
        Cluster.ByTopologies([hole]),
    )

    for face in [face_from_list, face_from_cluster]:
        _assert_face(face)
        assert isinstance(Face.InternalBoundaries(face), list)
        assert len(Face.InternalBoundaries(face)) >= 1

    assert Face.AddInternalBoundaries(None, [hole]) is None
    assert Face.AddInternalBoundaries(rectangle_face, None) == rectangle_face
    assert Face.AddInternalBoundariesCluster(None, Cluster.ByTopologies([hole])) is None
    assert Face.AddInternalBoundariesCluster(rectangle_face, None) == rectangle_face


def test_edges_vertices_external_boundary_and_wires(rectangle_face):
    edges = Face.Edges(rectangle_face)
    vertices = Face.Vertices(rectangle_face)
    external = Face.ExternalBoundary(rectangle_face, silent=True)
    wires = Face.Wires(rectangle_face)

    assert isinstance(edges, list)
    assert isinstance(vertices, list)
    assert isinstance(wires, list)
    assert len(edges) == 4
    assert len(vertices) == 4
    assert len(wires) >= 1
    _assert_wire(external)

    assert Face.Edges(None) is None
    assert Face.Vertices(None) is None
    assert Face.ExternalBoundary(None, silent=True) is None
    assert Face.InternalBoundaries(None) is None
    assert Face.Wires(None) is None


def test_interior_exterior_angles_and_compactness(rectangle_face):
    interior = Face.InteriorAngles(rectangle_face)
    exterior = Face.ExteriorAngles(rectangle_face)
    compactness = Face.Compactness(rectangle_face)

    assert len(interior) == 4
    assert len(exterior) == 4
    assert all(angle == pytest.approx(90) for angle in interior)
    assert all(angle == pytest.approx(270) for angle in exterior)
    assert 0 < compactness <= 1

    assert Face.InteriorAngles(None) is None
    assert Face.ExteriorAngles(None) is None


def test_compactness_invalid_input_returns_none():
    assert Face.Compactness(None) is None


def test_isconvex_true_for_rectangle(rectangle_face):
    assert bool(Face.IsConvex(rectangle_face, silent=True)) is True
    assert Face.IsConvex(None, silent=True) is None


def test_isconvex_false_for_concave_l_shape():
    concave = Face.LShape(width=4, length=4, a=1, b=1, silent=True)
    assert bool(Face.IsConvex(concave, silent=True)) is False


def test_face_area_planar_rectangle_remains_correct_and_unrounded_mode_works():
    face = Face.Rectangle(
        origin=_v(0.0, 0.0, 0.0),
        width=4.0,
        length=2.0,
        placement="lowerleft",
        silent=True,
    )
    _assert_face(face)
    area = Face.Area(face, mantissa=None, silent=True)
    assert isinstance(area, float)
    assert math.isclose(area, 8.0, rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_face_area_validation_respects_silent():
    assert Face.Area(None, silent=True) is None


# ============================================================================
# Planar geometry, normals, angles, orientation, and internal points
# ============================================================================


def test_normal_normal_edge_plane_equation_angle_and_coplanarity(rectangle_face):
    raised = Topology.Translate(rectangle_face, 0, 0, 1)
    vertical = Face.Rectangle(width=4, length=2, direction=[0, 1, 0], silent=True)

    normal = Face.Normal(rectangle_face)
    equation = Face.PlaneEquation(rectangle_face)
    normal_edge = Face.NormalEdge(rectangle_face, length=2, silent=True)

    assert len(normal) == 3
    assert abs(normal[0]) == pytest.approx(0)
    assert abs(normal[1]) == pytest.approx(0)
    assert abs(normal[2]) == pytest.approx(1)

    assert set(equation.keys()) == {"a", "b", "c", "d"}
    assert abs(equation["a"]) == pytest.approx(0)
    assert abs(equation["b"]) == pytest.approx(0)
    assert abs(equation["c"]) == pytest.approx(1)
    assert equation["d"] == pytest.approx(0)

    _assert_edge(normal_edge)
    assert Edge.Length(normal_edge) == pytest.approx(2)

    assert Face.Angle(rectangle_face, rectangle_face) == pytest.approx(0)
    assert Face.Angle(rectangle_face, vertical) == pytest.approx(90)
    assert bool(Face.IsCoplanar(rectangle_face, rectangle_face)) is True
    assert bool(Face.IsCoplanar(rectangle_face, raised)) is False

    assert Face.Normal(None) is None
    assert Face.NormalEdge(None, silent=True) is None
    assert Face.NormalEdge(rectangle_face, length=0, silent=True) is None
    assert Face.PlaneEquation(None) is None
    assert Face.Angle(None, rectangle_face) is None
    assert Face.IsCoplanar(None, rectangle_face) is None


def test_plane_equation_remains_supported_for_planar_face():
    face = Face.Rectangle(width=3.0, length=2.0, silent=True)
    equation = Face.PlaneEquation(face, mantissa=6)
    assert isinstance(equation, dict)
    assert set(equation).issuperset({"a", "b", "c", "d"})


def test_angle_between_planar_faces_remains_supported():
    a = Face.Rectangle(width=2, length=2, direction=[0, 0, 1], silent=True)
    b = Face.Rectangle(width=2, length=2, direction=[1, 0, 0], silent=True)
    assert Face.Angle(a, b, mantissa=6) == pytest.approx(90.0, abs=1.0e-5)


def test_facing_toward_and_compass_angle_use_face_normal(rectangle_face):
    normal = Face.Normal(rectangle_face)
    vertical_face = Face.Rectangle(width=4, length=2, direction=[1, 0, 0], silent=True)

    assert bool(Face.FacingToward(rectangle_face, direction=normal)) is True
    assert bool(Face.FacingToward(rectangle_face, direction=_reverse(normal))) is False

    assert Face.CompassAngle(rectangle_face) is None
    assert isinstance(Face.CompassAngle(vertical_face), (float, int))
    assert Face.CompassAngle(None) is None


def test_internal_vertex_and_third_vertex_are_valid(rectangle_face):
    internal = Face.InternalVertex(rectangle_face, silent=True)
    third = Face.ThirdVertex(rectangle_face, silent=True)

    _assert_vertex(internal)
    _assert_vertex(third)
    assert bool(Vertex.IsInternal(internal, rectangle_face, silent=True)) is True
    assert Face.InternalVertex(None, silent=True) is None
    assert Face.ThirdVertex(None, silent=True) is None


# ============================================================================
# Parametric surface evaluation and differential geometry
# ============================================================================


def test_vertex_by_parameters_and_vertex_parameters(rectangle_face):
    center = Face.VertexByParameters(rectangle_face, 0.5, 0.5)
    params = Face.VertexParameters(rectangle_face, center)

    _assert_vertex(center)
    assert len(params) == 2
    assert params[0] == pytest.approx(0.5, abs=1.0e-5)
    assert params[1] == pytest.approx(0.5, abs=1.0e-5)
    assert Face.VertexParameters(rectangle_face, center, outputType="u") == [
        pytest.approx(0.5, abs=1.0e-5)
    ]
    assert Face.VertexParameters(rectangle_face, center, outputType="v") == [
        pytest.approx(0.5, abs=1.0e-5)
    ]

    assert Face.VertexByParameters(None, 0.5, 0.5) is None
    assert Face.VertexParameters(None, center) is None
    assert Face.VertexParameters(rectangle_face, None) is None


def test_planar_face_uv_roundtrip_and_orientation():
    face = _planar_rectangle()
    _assert_face(face)

    point = Face.VertexByParameters(face, u=0.25, v=0.75, silent=True)
    _assert_vertex(point)
    assert _close(_xyz(point), [1.0, 1.5, 0.0], tol=2.0e-5)

    uv = Face.VertexParameters(face, point, outputType="uv", mantissa=None, silent=True)
    assert isinstance(uv, list) and len(uv) == 2
    assert math.isclose(uv[0], 0.25, abs_tol=2.0e-5)
    assert math.isclose(uv[1], 0.75, abs_tol=2.0e-5)


def test_planar_face_normal_tangents_and_planarity():
    face = _planar_rectangle()
    normal = Face.NormalAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    tangents = Face.TangentsAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)

    assert isinstance(normal, list) and len(normal) == 3
    assert isinstance(tangents, dict)
    tangent_u = tangents.get("u")
    tangent_v = tangents.get("v")
    assert isinstance(tangent_u, list) and isinstance(tangent_v, list)
    assert math.isclose(_mag(normal), 1.0, abs_tol=2.0e-6)
    assert math.isclose(_mag(tangent_u), 1.0, abs_tol=2.0e-6)
    assert math.isclose(_mag(tangent_v), 1.0, abs_tol=2.0e-6)
    assert abs(_dot(normal, tangent_u)) <= 2.0e-5
    assert abs(_dot(normal, tangent_v)) <= 2.0e-5
    assert normal[2] > 0.999
    assert Face.IsPlanar(face, silent=True) is True

    one_tangent = Face.TangentAtParameters(
        face,
        0.5,
        0.5,
        axis="u",
        mantissa=None,
        silent=True,
    )
    assert isinstance(one_tangent, list) and len(one_tangent) == 3
    assert abs(abs(_dot(one_tangent, tangent_u)) - 1.0) <= 2.0e-5

    corner_tangents = Face.TangentsAtParameters(
        face,
        0.0,
        0.0,
        mantissa=None,
        silent=True,
    )
    assert isinstance(corner_tangents, dict)
    corner_u = corner_tangents.get("u")
    corner_v = corner_tangents.get("v")
    assert isinstance(corner_u, list) and isinstance(corner_v, list)
    assert math.isclose(_mag(corner_u), 1.0, abs_tol=2.0e-6)
    assert math.isclose(_mag(corner_v), 1.0, abs_tol=2.0e-6)
    assert abs(_dot(normal, corner_u)) <= 2.0e-5
    assert abs(_dot(normal, corner_v)) <= 2.0e-5


def test_planar_face_curvature_is_zero():
    face = _planar_rectangle()
    curvature = Face.CurvatureAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    assert isinstance(curvature, dict)
    for key in ("maximum", "minimum", "mean", "gaussian"):
        assert key in curvature
        assert abs(float(curvature[key])) <= 2.0e-4


def test_surface_query_validation_does_not_raise():
    face = _planar_rectangle()
    assert Face.NormalAtParameters(face, u=-0.1, v=0.5, silent=True) is None
    assert Face.TangentAtParameters(face, u=0.5, v=0.5, axis="bad", silent=True) is None
    assert Face.TangentsAtParameters(face, u=1.1, v=0.5, silent=True) is None
    assert Face.CurvatureAtParameters(None, silent=True) is None
    assert Face.IsPlanar(None, silent=True) is None


# ============================================================================
# Exact curves, NURBS surfaces, curved area, and transformed support geometry
# ============================================================================


def test_face_circle_preserves_historical_faceted_default():
    radius = 2.0
    sides = 7
    face = Face.Circle(radius=radius, sides=sides, tolerance=SURFACE_TOLERANCE)
    _assert_face(face)
    expected = 0.5 * sides * radius * radius * math.sin(2.0 * math.pi / sides)
    assert Face.Area(face, mantissa=None) == pytest.approx(expected, rel=1.0e-8, abs=1.0e-8)


@pytest.mark.pythonocc_only
def test_face_circle_exact_mode_preserves_curved_boundary():
    face = Face.Circle(
        radius=2.0,
        sides=4,
        polyline=False,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    _assert_face(face)
    boundary = Face.ExternalBoundary(face, silent=True)
    _assert_wire(boundary)
    assert Wire.IsPolyline(boundary) is False
    assert Face.Area(face, mantissa=None) == pytest.approx(math.pi * 4.0, rel=1.0e-8, abs=1.0e-8)


def test_face_bywire_preserves_exact_circle():
    wire = Wire.Circle(radius=2.0, sides=1, polyline=False, silent=True)
    face = Face.ByWire(wire, tolerance=SURFACE_TOLERANCE, silent=True)
    _assert_face(face)
    boundary = Face.ExternalBoundary(face, silent=True)
    _assert_wire(boundary)
    assert Wire.IsPolyline(boundary) is False
    assert Face.Area(face, mantissa=None) == pytest.approx(math.pi * 4.0, rel=1.0e-8, abs=1.0e-8)


@pytest.mark.pythonocc_only
def test_face_ellipse_exact_mode_preserves_rational_curves():
    face = Face.Ellipse(
        width=4.0,
        length=2.0,
        sides=4,
        polyline=False,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    _assert_face(face)
    boundary = Face.ExternalBoundary(face, silent=True)
    _assert_wire(boundary)
    assert Wire.IsPolyline(boundary) is False
    assert Face.Area(face, mantissa=None) == pytest.approx(math.pi * 2.0, rel=1.0e-8, abs=1.0e-8)


@pytest.mark.pythonocc_only
def test_face_bywires_preserves_curved_outer_and_inner_boundaries():
    outer = Wire.Circle(radius=3.0, sides=4, polyline=False, silent=True)
    inner = Wire.Ellipse(width=2.0, length=1.0, sides=4, polyline=False, silent=True)
    face = Face.ByWires(outer, [inner], tolerance=SURFACE_TOLERANCE, silent=True)
    _assert_face(face)

    external = Face.ExternalBoundary(face, silent=True)
    internals = Face.InternalBoundaries(face) or []
    _assert_wire(external)
    assert Wire.IsPolyline(external) is False
    assert len(internals) == 1
    assert Wire.IsPolyline(internals[0]) is False

    expected = math.pi * 3.0 * 3.0 - math.pi * 1.0 * 0.5
    assert Face.Area(face, mantissa=None) == pytest.approx(expected, rel=1.0e-8, abs=1.0e-8)


@pytest.mark.topologiccore_only
def test_topologiccore_nurbs_surface_construction_is_explicitly_unsupported():
    control_points = [
        [_v(0, 0, 0), _v(0, 1, 0)],
        [_v(1, 0, 0), _v(1, 1, 0)],
    ]
    assert Face.ByNurbsParameters(control_points, uDegree=1, vDegree=1, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_planar_bspline_surface_is_exact_and_planar():
    control_points = [
        [_v(0, 0, 0), _v(0, 2, 0)],
        [_v(4, 0, 0), _v(4, 2, 0)],
    ]
    face = Face.ByNurbsParameters(
        control_points,
        uDegree=1,
        vDegree=1,
        uKnots=[0, 0, 1, 1],
        vKnots=[0, 0, 1, 1],
        silent=True,
    )
    _assert_face(face)
    assert Face.IsPlanar(face, silent=True) is True
    center = Face.VertexByParameters(face, 0.5, 0.5, silent=True)
    assert _close(_xyz(center), [2.0, 1.0, 0.0], tol=2.0e-6)

    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_BSplineSurface

    assert BRepAdaptor_Surface(face.shape, True).GetType() == GeomAbs_BSplineSurface


@pytest.mark.pythonocc_only
def test_pythonocc_rational_quarter_cylinder_geometry_and_planarity():
    face = _quarter_cylinder_nurbs()
    _assert_face(face)
    assert Face.IsPlanar(face, silent=True) is False

    s2 = math.sqrt(2.0) / 2.0
    center = Face.VertexByParameters(face, 0.5, 0.5, silent=True)
    _assert_vertex(center)
    assert _close(_xyz(center), [s2, s2, 0.5], tol=3.0e-6)

    uv = Face.VertexParameters(face, center, mantissa=None, silent=True)
    assert isinstance(uv, list) and len(uv) == 2
    assert math.isclose(uv[0], 0.5, abs_tol=3.0e-6)
    assert math.isclose(uv[1], 0.5, abs_tol=3.0e-6)


@pytest.mark.pythonocc_only
def test_pythonocc_quarter_cylinder_normal_tangents_and_curvature():
    face = _quarter_cylinder_nurbs()
    _assert_face(face)

    normal = Face.NormalAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    tangents = Face.TangentsAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)
    curvature = Face.CurvatureAtParameters(face, 0.5, 0.5, mantissa=None, silent=True)

    assert isinstance(normal, list) and len(normal) == 3
    assert isinstance(tangents, dict)
    assert isinstance(curvature, dict)

    s2 = math.sqrt(2.0) / 2.0
    radial = [s2, s2, 0.0]
    assert abs(abs(_dot(normal, radial)) - 1.0) <= 3.0e-6

    tangent_u = tangents["u"]
    tangent_v = tangents["v"]
    assert abs(_dot(tangent_u, radial)) <= 3.0e-6
    assert abs(abs(tangent_v[2]) - 1.0) <= 3.0e-6
    assert abs(_dot(normal, tangent_u)) <= 3.0e-6
    assert abs(_dot(normal, tangent_v)) <= 3.0e-6

    principal = sorted(
        [abs(float(curvature["maximum"])), abs(float(curvature["minimum"]))]
    )
    assert math.isclose(principal[0], 0.0, abs_tol=2.0e-6)
    assert math.isclose(principal[1], 1.0, rel_tol=2.0e-6, abs_tol=2.0e-6)
    assert math.isclose(float(curvature["gaussian"]), 0.0, abs_tol=2.0e-6)
    assert curvature["isUmbilic"] is False


@pytest.mark.pythonocc_only
def test_pythonocc_surface_evaluation_respects_face_location():
    face = _general_nurbs_face()
    _assert_face(face)

    u, v = 0.31, 0.63
    point0 = Face.VertexByParameters(face, u=u, v=v, tolerance=SURFACE_TOLERANCE, silent=True)
    normal0 = Face.NormalAtParameters(
        face,
        u=u,
        v=v,
        mantissa=None,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    tangents0 = Face.TangentsAtParameters(
        face,
        u=u,
        v=v,
        mantissa=None,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    _assert_vertex(point0)
    assert normal0 is not None
    assert isinstance(tangents0, dict)

    angle = 37.0
    moved = Topology.Rotate(face, origin=_v(0, 0, 0), axis=[1, 0, 0], angle=angle, silent=True)
    moved = Topology.Translate(moved, x=5.0, y=-3.0, z=2.0, silent=True)
    _assert_face(moved)

    point1 = Face.VertexByParameters(moved, u=u, v=v, tolerance=SURFACE_TOLERANCE, silent=True)
    normal1 = Face.NormalAtParameters(
        moved,
        u=u,
        v=v,
        mantissa=None,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    tangents1 = Face.TangentsAtParameters(
        moved,
        u=u,
        v=v,
        mantissa=None,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )

    expected_point = _rotate_x(_coords(point0), angle)
    expected_point = [
        expected_point[0] + 5.0,
        expected_point[1] - 3.0,
        expected_point[2] + 2.0,
    ]
    _assert_vector(_coords(point1), expected_point, abs_tol=2.0e-6)
    _assert_vector(normal1, _unit(_rotate_x(normal0, angle)), abs_tol=2.0e-6)
    _assert_vector(tangents1["u"], _unit(_rotate_x(tangents0["u"], angle)), abs_tol=2.0e-6)
    _assert_vector(tangents1["v"], _unit(_rotate_x(tangents0["v"], angle)), abs_tol=2.0e-6)

    uv = Face.VertexParameters(
        moved,
        point1,
        outputType="uv",
        mantissa=None,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    assert uv[0] == pytest.approx(u, abs=2.0e-6)
    assert uv[1] == pytest.approx(v, abs=2.0e-6)


@pytest.mark.pythonocc_only
def test_pythonocc_quarter_cylinder_normaledge_uses_local_surface_normal():
    face = _quarter_cylinder_nurbs()
    normal_edge = Face.NormalEdge(face, length=0.5, silent=True)
    _assert_edge(normal_edge)
    assert math.isclose(Edge.Length(normal_edge, mantissa=6, silent=True), 0.5, abs_tol=2.0e-6)


@pytest.mark.pythonocc_only
def test_pythonocc_general_nurbs_normaledge_uses_local_surface_normal():
    face = _general_nurbs_face()
    _assert_face(face)

    edge = Face.NormalEdge(face, length=2.0, tolerance=SURFACE_TOLERANCE, silent=True)
    _assert_edge(edge)

    start = Edge.StartVertex(edge)
    end = Edge.EndVertex(edge)
    _assert_vertex(start)
    _assert_vertex(end)

    uv = Face.VertexParameters(
        face,
        start,
        outputType="uv",
        mantissa=None,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    assert uv is not None and len(uv) == 2

    expected = Face.NormalAtParameters(
        face,
        u=uv[0],
        v=uv[1],
        outputType="xyz",
        mantissa=None,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )
    direction = Edge.Direction(edge, mantissa=None)
    _assert_vector(direction, expected, abs_tol=2.0e-6)
    assert Edge.Length(edge, mantissa=None) == pytest.approx(2.0, abs=2.0e-6)


@pytest.mark.pythonocc_only
def test_pythonocc_exact_quarter_cylinder_nurbs_area():
    radius = 2.0
    height = 3.0
    face = _quarter_cylinder_nurbs(radius=radius, height=height)
    _assert_face(face)
    assert Face.IsPlanar(face, silent=True) is False

    area = Face.Area(face, mantissa=None, silent=True)
    expected = 0.5 * math.pi * radius * height
    assert isinstance(area, float)
    assert math.isclose(area, expected, rel_tol=1.0e-8, abs_tol=1.0e-8)


@pytest.mark.pythonocc_only
def test_pythonocc_face_area_fixes_curved_shell_faces_that_previously_returned_zero():
    radius = 1.0
    height = 2.0
    shell = Shell.ByWires(
        [_circle_wire(0.0, radius), _circle_wire(height, radius)],
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(shell, "Shell")

    faces = Topology.Faces(shell, silent=True) or []
    assert len(faces) >= 1
    assert any(Face.IsPlanar(face, silent=True) is False for face in faces)

    areas = [Face.Area(face, mantissa=None, silent=True) for face in faces]
    assert all(isinstance(value, float) and value > 0.0 for value in areas)

    total = sum(areas)
    expected = 2.0 * math.pi * radius * height
    assert math.isclose(total, expected, rel_tol=1.0e-8, abs_tol=1.0e-8)


# ============================================================================
# Editing, offsets, projection, trimming, filleting, and simplification
# ============================================================================


def test_bounding_rectangle_preserves_area_metadata(rectangle_face):
    bounding = Face.BoundingRectangle(rectangle_face, optimize=0)

    _assert_face(bounding)
    assert Face.Area(bounding) == pytest.approx(Face.Area(rectangle_face))

    dictionary = Topology.Dictionary(bounding)
    assert Dictionary.ValueAtKey(dictionary, "width") == pytest.approx(4)
    assert Dictionary.ValueAtKey(dictionary, "length") == pytest.approx(2)

    assert Face.BoundingRectangle(None) is None


def test_offset_and_thickened_wire_create_faces():
    boundary = Wire.ByVertices(
        [_v(-2, -1, 0), _v(2, -1, 0), _v(2, 1, 0), _v(-2, 1, 0)],
        close=True,
        silent=True,
    )
    source = Face.ByWire(boundary, silent=True)
    offset = Face.ByOffset(source, offset=0.1, smooth=True, silent=True)

    polyline = Wire.ByVertices(
        [_v(0, 0, 0), _v(2, 0, 0), _v(2, 1, 0)],
        close=False,
        silent=True,
    )
    thickened = Face.ByThickenedWire(polyline, offsetA=0.5, offsetB=0.5, silent=True)

    _assert_face(source)
    _assert_face(offset)
    _assert_face(thickened)
    assert Face.Area(source) == pytest.approx(8)
    assert Face.Area(offset) > 0
    assert Face.Area(offset) != pytest.approx(Face.Area(source))
    assert Face.Area(thickened) > 0

    assert Face.ByOffset(None, smooth=True, silent=True) is None
    assert Face.ByThickenedWire(None, silent=True) is None


def test_invert_planarize_project_and_trim(rectangle_face):
    elevated = Topology.Translate(rectangle_face, 0, 0, 5)
    receiver = Face.Rectangle(width=10, length=10, silent=True)
    cutter = Wire.Rectangle(width=1, length=1, placement="center", silent=True)

    inverted = Face.Invert(rectangle_face, silent=True)
    planarized = Face.Planarize(elevated)
    projected = Face.Project(elevated, receiver, direction=[0, 0, -1])
    trimmed = Face.TrimByWire(receiver, cutter)

    for face in [inverted, planarized, projected, trimmed]:
        _assert_face(face)

    assert Face.Area(inverted) == pytest.approx(Face.Area(rectangle_face))
    assert Face.Invert(None, silent=True) is None
    assert Face.Planarize(None) is None
    assert Face.Project(None, receiver) is None
    assert Face.Project(elevated, None) is None
    assert Face.TrimByWire(None, cutter) is None
    assert Face.TrimByWire(receiver, None) == receiver


def test_fillet_simplify_and_remove_collinear_edges_return_faces():
    redundant = Face.ByVertices(
        [_v(0, 0, 0), _v(1, 0, 0), _v(2, 0, 0), _v(2, 2, 0), _v(0, 2, 0)],
        silent=True,
    )

    filleted = Face.Fillet(redundant, radius=0.1, sides=4, silent=True)
    simplified = Face.Simplify(redundant, tolerance=0.01, silent=True)
    cleaned = Face.RemoveCollinearEdges(redundant, silent=True)

    _assert_face(simplified)
    assert Face.Area(simplified) > 0

    if Topology._IsTopologicCoreBackend():
        assert filleted is None
        assert cleaned is None
    else:
        for face in [filleted, cleaned]:
            _assert_face(face)
            assert Face.Area(face) > 0

    assert Face.Fillet(None, silent=True) is None
    assert Face.Simplify(None, silent=True) is None
    assert Face.RemoveCollinearEdges(None, silent=True) is None


@pytest.mark.pythonocc_only
def test_nurbs_trim_and_complement_conserve_area_and_surface_curvature():
    face = _curved_nurbs_face()
    vertices = [
        Face.VertexByParameters(face, 0.2, 0.2, tolerance=SURFACE_TOLERANCE, silent=True),
        Face.VertexByParameters(face, 0.8, 0.2, tolerance=SURFACE_TOLERANCE, silent=True),
        Face.VertexByParameters(face, 0.8, 0.8, tolerance=SURFACE_TOLERANCE, silent=True),
        Face.VertexByParameters(face, 0.2, 0.8, tolerance=SURFACE_TOLERANCE, silent=True),
    ]
    trim = Wire.ByVertices(
        vertices,
        close=True,
        tolerance=SURFACE_TOLERANCE,
        silent=True,
    )

    inside = Face.TrimByWire(face, trim, reverse=False)
    outside = Face.TrimByWire(face, trim, reverse=True)

    _assert_face(inside)
    _assert_face(outside)

    area = Face.Area(face, mantissa=None, silent=True)
    combined = Face.Area(inside, mantissa=None, silent=True) + Face.Area(
        outside,
        mantissa=None,
        silent=True,
    )
    assert combined == pytest.approx(area, rel=2.0e-5, abs=2.0e-5)
    assert Face.IsPlanar(inside, tolerance=SURFACE_TOLERANCE, silent=True) is False
    assert Face.IsPlanar(outside, tolerance=SURFACE_TOLERANCE, silent=True) is False


# ============================================================================
# Triangulation and shell conversion
# ============================================================================


def test_triangulate_rectangle_returns_face_list(rectangle_face):
    triangles = Face.Triangulate(rectangle_face, mode=0, silent=True)

    assert isinstance(triangles, list)
    assert len(triangles) >= 1
    for face in triangles:
        _assert_face(face)
        assert Face.Area(face) > 0
        assert len(Topology.Vertices(face, silent=True) or []) == 3

    assert Face.Triangulate(None, mode=0, silent=True) is None


def test_face_triangulate_is_thin_topology_wrapper(rectangle_face, monkeypatch):
    original = Topology.Triangulate
    called = {"value": False}

    def wrapped(*args, **kwargs):
        called["value"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(Topology, "Triangulate", wrapped)

    triangles = Face.Triangulate(
        rectangle_face,
        mode=1,
        meshSize=0.25,
        silent=True,
    )

    assert called["value"] is True
    assert isinstance(triangles, list)
    assert len(triangles) >= 1
    assert all(
        len(Topology.Vertices(triangle, silent=True) or []) == 3
        for triangle in triangles
    )


def test_by_shell_recovers_face_from_simple_shell(rectangle_face):
    shell = Shell.ByFaces([rectangle_face], silent=True)
    recovered = Face.ByShell(shell, silent=True)

    _assert_face(recovered)
    assert Face.Area(recovered) == pytest.approx(Face.Area(rectangle_face))
    assert Face.ByShell(None, silent=True) is None
