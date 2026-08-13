# Auto-generated PythonOCC backend parity test
# Copied from test_Face.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# TopologicPy Face unit tests.

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


TOLERANCE = 1e-6

@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _assert_vertex(vertex):
    assert Topology.IsInstance(vertex, "Vertex")


def _assert_edge(edge):
    assert Topology.IsInstance(edge, "Edge")


def _assert_wire(wire):
    assert Topology.IsInstance(wire, "Wire")


def _assert_face(face):
    assert Topology.IsInstance(face, "Face")


def _assert_shell(shell):
    assert Topology.IsInstance(shell, "Shell")


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


def _reverse(vector):
    return [-vector[0], -vector[1], -vector[2]]


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


def test_rectangle_square_circle_and_basic_area(rectangle_face, square_face):
    circle = Face.Circle(radius=1, sides=32)
    lowerleft = Face.Rectangle(origin=_v(0, 0, 0), width=4, length=2, placement="lowerleft", silent=True)

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
    face_from_cluster = Face.AddInternalBoundariesCluster(rectangle_face, Cluster.ByTopologies([hole]))

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


def test_facing_toward_and_compass_angle_use_face_normal(rectangle_face):
    normal = Face.Normal(rectangle_face)
    vertical_face = Face.Rectangle(width=4, length=2, direction=[1, 0, 0], silent=True)

    assert bool(Face.FacingToward(rectangle_face, direction=normal)) is True
    assert bool(Face.FacingToward(rectangle_face, direction=_reverse(normal))) is False

    # A horizontal face has a vertical normal, so its horizontal compass angle is undefined.
    assert Face.CompassAngle(rectangle_face) is None
    assert isinstance(Face.CompassAngle(vertical_face), (float, int))
    assert Face.CompassAngle(None) is None


def test_vertex_by_parameters_and_vertex_parameters(rectangle_face):
    center = Face.VertexByParameters(rectangle_face, 0.5, 0.5)
    params = Face.VertexParameters(rectangle_face, center)

    _assert_vertex(center)
    assert len(params) == 2
    assert params[0] == pytest.approx(0.5, abs=1e-5)
    assert params[1] == pytest.approx(0.5, abs=1e-5)
    assert Face.VertexParameters(rectangle_face, center, outputType="u") == [pytest.approx(0.5, abs=1e-5)]
    assert Face.VertexParameters(rectangle_face, center, outputType="v") == [pytest.approx(0.5, abs=1e-5)]

    assert Face.VertexByParameters(None, 0.5, 0.5) is None
    assert Face.VertexParameters(None, center) is None
    assert Face.VertexParameters(rectangle_face, None) is None


def test_internal_vertex_and_third_vertex_are_valid(rectangle_face):
    internal = Face.InternalVertex(rectangle_face, silent=True)
    third = Face.ThirdVertex(rectangle_face, silent=True)

    _assert_vertex(internal)
    _assert_vertex(third)
    assert bool(Vertex.IsInternal(internal, rectangle_face, silent=True)) is True
    assert Face.InternalVertex(None, silent=True) is None
    assert Face.ThirdVertex(None, silent=True) is None


def test_bounding_rectangle_preserves_area_metadata(rectangle_face):
    bounding = Face.BoundingRectangle(rectangle_face, optimize=0)

    _assert_face(bounding)
    assert Face.Area(bounding) == pytest.approx(Face.Area(rectangle_face))

    dictionary = Topology.Dictionary(bounding)
    assert Dictionary.ValueAtKey(dictionary, "width") == pytest.approx(4)
    assert Dictionary.ValueAtKey(dictionary, "length") == pytest.approx(2)

    assert Face.BoundingRectangle(None) is None


def test_offset_and_thickened_wire_create_faces(rectangle_face):
    offset = Face.ByOffset(rectangle_face, offset=0.1, silent=True)
    polyline = Wire.ByVertices(
        [_v(0, 0, 0), _v(2, 0, 0), _v(2, 1, 0)],
        close=False,
        silent=True,
    )
    thickened = Face.ByThickenedWire(polyline, offsetA=0.5, offsetB=0.5, silent=True)

    _assert_face(offset)
    _assert_face(thickened)
    assert Face.Area(offset) > 0
    assert Face.Area(thickened) > 0

    assert Face.ByOffset(None, silent=True) is None
    assert Face.ByThickenedWire(None, silent=True) is None


def test_invert_harmonize_planarize_project_and_trim(rectangle_face):
    elevated = Topology.Translate(rectangle_face, 0, 0, 5)
    receiver = Face.Rectangle(width=10, length=10, silent=True)
    cutter = Wire.Rectangle(width=1, length=1, placement="center", silent=True)

    inverted = Face.Invert(rectangle_face, silent=True)
    harmonized = Face.Harmonize(rectangle_face, silent=True)
    planarized = Face.Planarize(elevated)
    projected = Face.Project(elevated, receiver, direction=[0, 0, -1])
    trimmed = Face.TrimByWire(receiver, cutter)

    for face in [inverted, harmonized, planarized, projected, trimmed]:
        _assert_face(face)

    assert Face.Area(inverted) == pytest.approx(Face.Area(rectangle_face))
    assert Face.Invert(None, silent=True) is None
    assert Face.Harmonize(None, silent=True) is None
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

    for face in [filleted, simplified, cleaned]:
        _assert_face(face)
        assert Face.Area(face) > 0

    assert Face.Fillet(None, silent=True) is None
    assert Face.Simplify(None, silent=True) is None
    assert Face.RemoveCollinearEdges(None, silent=True) is None


def test_shape_constructors_return_valid_faces():
    constructors = [
        Face.CrossShape(width=4, length=4, silent=True),
        Face.CShape(width=4, length=4, silent=True),
        Face.IShape(width=4, length=4, silent=True),
        Face.LShape(width=4, length=4, silent=True),
        Face.TShape(width=4, length=4, silent=True),
        Face.Trapezoid(widthA=4, widthB=2, length=3),
        Face.Star(radiusA=2, radiusB=1, rays=5),
        Face.Ellipse(width=4, length=2, sides=32),
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


def test_triangulate_rectangle_returns_face_list(rectangle_face):
    triangles = Face.Triangulate(rectangle_face, mode=0, silent=True)

    assert isinstance(triangles, list)
    assert len(triangles) >= 1
    for face in triangles:
        _assert_face(face)
        assert Face.Area(face) > 0

    assert Face.Triangulate(None, mode=0, silent=True) is None


def test_by_shell_recovers_face_from_simple_shell(rectangle_face):
    shell = Shell.ByFaces([rectangle_face], silent=True)
    recovered = Face.ByShell(shell, silent=True)

    _assert_face(recovered)
    assert Face.Area(recovered) == pytest.approx(Face.Area(rectangle_face))
    assert Face.ByShell(None, silent=True) is None


def test_isconvex_true_for_rectangle(rectangle_face):
    assert bool(Face.IsConvex(rectangle_face, silent=True)) is True
    assert Face.IsConvex(None, silent=True) is None


def test_isconvex_false_for_concave_l_shape():
    concave = Face.LShape(width=4, length=4, a=1, b=1, silent=True)
    assert bool(Face.IsConvex(concave, silent=True)) is False


def test_invalid_inputs_for_selected_shape_factories():
    assert Face.Star(radiusA=0, radiusB=1) is None
    assert Face.Trapezoid(widthA=0, widthB=1, length=1) is None
    assert Face.Square(size=0) is None


def test_ellipse_invalid_dimensions_return_none():
    assert Face.Ellipse(width=0, length=1) is None
