# Auto-generated PythonOCC backend parity test
# Copied from test_Vertex.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# TopologicPy Vertex unit tests.

import math
import random

import pytest


Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
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


def _assert_vertex(vertex):
    assert Topology.IsInstance(vertex, "Vertex")


def _coords(vertex, mantissa=6):
    return Vertex.Coordinates(vertex, mantissa=mantissa)


def _assert_coords(vertex, expected, abs_tol=TOLERANCE, mantissa=6):
    actual = _coords(vertex, mantissa=mantissa)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


def _dictionary_value(dictionary, key):
    value = Dictionary.ValueAtKey(dictionary, key)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _signed_area_xy(vertices):
    xy = [(Vertex.X(v), Vertex.Y(v)) for v in vertices]
    area = 0.0
    for i, (x1, y1) in enumerate(xy):
        x2, y2 = xy[(i + 1) % len(xy)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


@pytest.fixture
def xy_square_vertices():
    return [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(1, 0, 0),
        Vertex.ByCoordinates(1, 1, 0),
        Vertex.ByCoordinates(0, 1, 0),
    ]


@pytest.fixture
def xy_face():
    return Face.Rectangle(width=10, length=10)


def test_by_coordinates_accepts_positional_sequence_and_keywords():
    examples = [
        Vertex.ByCoordinates(1, 2, 3),
        Vertex.ByCoordinates([1, 2, 3]),
        Vertex.ByCoordinates((1, 2, 3)),
        Vertex.ByCoordinates(x=1, y=2, z=3),
    ]

    for vertex in examples:
        _assert_vertex(vertex)
        _assert_coords(vertex, [1, 2, 3])


def test_by_coordinates_defaults_missing_coordinates_to_zero():
    _assert_coords(Vertex.ByCoordinates(), [0, 0, 0])
    _assert_coords(Vertex.ByCoordinates(5), [5, 0, 0])
    _assert_coords(Vertex.ByCoordinates([5, 6]), [5, 6, 0])


def test_by_coordinates_rejects_invalid_arguments():
    assert Vertex.ByCoordinates(1, 2, 3, 4) is None
    assert Vertex.ByCoordinates([1, 2, 3, 4]) is None
    assert Vertex.ByCoordinates("not-a-number", 0, 0) is None
    assert Vertex.ByCoordinates(1, x=2) is None


def test_point_and_origin_create_expected_vertices():
    _assert_coords(Vertex.Point(2, 3, 4), [2, 3, 4])
    _assert_coords(Vertex.Origin(), [0, 0, 0])


def test_coordinate_accessors_and_coordinate_output_types():
    vertex = Vertex.ByCoordinates(1.23456, 2.34567, 3.45678)

    assert Vertex.X(vertex, mantissa=2) == pytest.approx(1.23)
    assert Vertex.Y(vertex, mantissa=2) == pytest.approx(2.35)
    assert Vertex.Z(vertex, mantissa=2) == pytest.approx(3.46)
    assert Vertex.Coordinates(vertex, outputType="xyz", mantissa=1) == [1.2, 2.3, 3.5]
    assert Vertex.Coordinates(vertex, outputType="zyx", mantissa=1) == [3.5, 2.3, 1.2]
    assert Vertex.Coordinates(vertex, outputType="xy", mantissa=1) == [1.2, 2.3]
    assert Vertex.Coordinates(vertex, outputType="x", mantissa=1) == [1.2]

    matrix = Vertex.Coordinates(vertex, outputType="matrix", mantissa=1)
    assert matrix == [
        [1, 0, 0, 1.2],
        [0, 1, 0, 2.3],
        [0, 0, 1, 3.5],
        [0, 0, 0, 1],
    ]


def test_coordinate_accessors_reject_non_vertices():
    assert Vertex.Coordinates(None) is None
    assert Vertex.X(None, silent=True) is None
    assert Vertex.Y(None, silent=True) is None
    assert Vertex.Z(None, silent=True) is None


def test_align_coordinates_snaps_to_nearby_coordinate_lists_and_transfers_dictionary():
    vertex = Vertex.ByCoordinates(0.99996, 2.00004, 3.00003)
    dictionary = Dictionary.ByPythonDictionary({"name": "snap-source", "index": 7})
    vertex = Topology.SetDictionary(vertex, dictionary)

    aligned = Vertex.AlignCoordinates(
        vertex,
        xList=[1.0, 5.0],
        yList=[2.0, 6.0],
        zList=[3.0, 7.0],
        xEpsilon=0.001,
        yEpsilon=0.001,
        zEpsilon=0.001,
        transferDictionary=True,
        mantissa=6,
        silent=True,
    )

    _assert_vertex(aligned)
    _assert_coords(aligned, [1, 2, 3])
    out_dictionary = Topology.Dictionary(aligned)
    assert _dictionary_value(out_dictionary, "name") == "snap-source"
    assert _dictionary_value(out_dictionary, "index") == 7


def test_align_coordinates_allows_partial_coordinate_lists_after_fix():
    vertex = Vertex.ByCoordinates(0.99996, 2.25, 3.5)
    aligned = Vertex.AlignCoordinates(
        vertex,
        xList=[1.0],
        xEpsilon=0.001,
        silent=True,
    )
    _assert_coords(aligned, [1.0, 2.25, 3.5])


def test_centroid_returns_average_vertex(xy_square_vertices):
    centroid = Vertex.Centroid(xy_square_vertices)
    _assert_vertex(centroid)
    _assert_coords(centroid, [0.5, 0.5, 0])
    assert Vertex.Centroid([xy_square_vertices[0]]) is xy_square_vertices[0]
    assert Vertex.Centroid([]) is None
    assert Vertex.Centroid("not-a-list") is None


def test_counterclockwise_and_clockwise_2d_sort_by_orientation(xy_square_vertices):
    unordered = [xy_square_vertices[i] for i in [2, 0, 3, 1]]

    ccw = Vertex.CounterClockwise2D(list(unordered))
    cw = Vertex.Clockwise2D(list(unordered))

    assert isinstance(ccw, list)
    assert isinstance(cw, list)
    assert _signed_area_xy(ccw) > 0
    assert _signed_area_xy(cw) < 0


def test_are_collinear_handles_true_false_and_duplicate_cases():
    line = [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(1, 1, 1),
        Vertex.ByCoordinates(2, 2, 2),
        Vertex.ByCoordinates(2, 2, 2),
    ]
    not_line = line[:2] + [Vertex.ByCoordinates(2, 2, 3)]

    assert bool(Vertex.AreCollinear(line)) is True
    assert bool(Vertex.AreCollinear(not_line)) is False
    assert bool(Vertex.AreCollinear(line[:2])) is True
    assert Vertex.AreCollinear("not-a-list") is None


def test_are_coplanar_detects_planar_and_non_planar_vertices(xy_square_vertices):
    assert bool(Vertex.AreCoplanar(xy_square_vertices, silent=True)) is True

    non_planar = list(xy_square_vertices) + [Vertex.ByCoordinates(0, 0, 1)]
    assert bool(Vertex.AreCoplanar(non_planar, silent=True)) is False
    assert Vertex.AreCoplanar("not-a-list", silent=True) is None
    assert Vertex.AreCoplanar(xy_square_vertices[:2], silent=True) is None


def test_normal_and_plane_equation_for_xy_plane(xy_square_vertices):
    normal = Vertex.Normal(xy_square_vertices, mantissa=6, silent=True)
    assert normal is not None
    assert abs(normal[0]) == pytest.approx(0, abs=TOLERANCE)
    assert abs(normal[1]) == pytest.approx(0, abs=TOLERANCE)
    assert abs(normal[2]) == pytest.approx(1, abs=TOLERANCE)

    equation = Vertex.PlaneEquation(xy_square_vertices, mantissa=6)
    assert abs(equation["a"]) == pytest.approx(0, abs=TOLERANCE)
    assert abs(equation["b"]) == pytest.approx(0, abs=TOLERANCE)
    assert abs(equation["c"]) == pytest.approx(1, abs=TOLERANCE)
    assert equation["d"] == pytest.approx(0, abs=TOLERANCE)


def test_ipsilateral_and_same_side_methods(xy_face):
    positive_side = [
        Vertex.ByCoordinates(1, 1, 1),
        Vertex.ByCoordinates(2, 2, 2),
        Vertex.ByCoordinates(3, 3, 3),
    ]
    mixed_sides = [
        Vertex.ByCoordinates(1, 1, 1),
        Vertex.ByCoordinates(2, 2, -1),
    ]

    assert bool(Vertex.AreIpsilateral(positive_side, xy_face)) is True
    assert bool(Vertex.AreOnSameSide(positive_side, xy_face)) is True
    assert bool(Vertex.AreIpsilateral(mixed_sides, xy_face)) is False
    assert bool(Vertex.AreOnSameSide(mixed_sides, xy_face)) is False

    cluster = Cluster.ByTopologies(positive_side)
    assert bool(Vertex.AreIpsilateralCluster(cluster, xy_face)) is True
    assert bool(Vertex.AreOnSameSideCluster(cluster, xy_face)) is True


def test_distance_and_quadrance_to_vertex_and_edge():
    origin = Vertex.ByCoordinates(0, 0, 0)
    other = Vertex.ByCoordinates(3, 4, 0)
    edge = Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(0, 4, 0), Vertex.ByCoordinates(4, 0, 0))

    assert Vertex.Distance(origin, other) == pytest.approx(5)
    assert Vertex.Quadrance(origin, other) == pytest.approx(25)
    assert Vertex.Distance(origin, edge) == pytest.approx(math.sqrt(8), abs=1e-6)
    assert Vertex.Quadrance(origin, edge) == pytest.approx(8, abs=1e-6)
    assert Vertex.Distance(None, other) is None
    assert Vertex.Quadrance(origin, None) is None


def test_project_and_perpendicular_distance_to_face():
    face = Face.Rectangle(width=20, length=20)
    vertex = Vertex.ByCoordinates(1, 1, 7)

    projected = Vertex.Project(vertex, face)
    _assert_vertex(projected)
    _assert_coords(projected, [1, 1, 0])
    assert Vertex.PerpendicularDistance(vertex, face) == pytest.approx(7)
    assert Vertex.Distance(vertex, face) == pytest.approx(7)
    assert Vertex.Quadrance(vertex, face) == pytest.approx(49)


def test_by_offset_2d_relative_to_edge_uses_left_hand_normal_and_preserves_z():
    edge = Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(10, 0, 0))
    vertex = Vertex.ByCoordinates(5, 0, 3)

    offset = Vertex.ByOffset2DRelativeToEdge(vertex, edge, offset=2)
    _assert_vertex(offset)
    _assert_coords(offset, [5, 2, 3])
    assert Vertex.ByOffset2DRelativeToEdge(None, edge) is None
    assert Vertex.ByOffset2DRelativeToEdge(vertex, None) is None


def test_index_finds_vertices_by_identity_or_coordinate_tolerance():
    v0 = Vertex.ByCoordinates(0, 0, 0)
    v1 = Vertex.ByCoordinates(1, 0, 0)
    v1_same_coords = Vertex.ByCoordinates(1, 0, 0)
    vertices = [v0, v1]

    assert Vertex.Index(v1, vertices, strict=True) == 1
    assert Vertex.Index(v1_same_coords, vertices, strict=False) == 1
    assert Vertex.Index(v1_same_coords, vertices, strict=True) is None
    assert Vertex.Index(None, vertices) is None
    assert Vertex.Index(v0, []) is None


def test_is_internal_peripheral_external_and_coincident_for_simple_topologies():
    edge = Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(10, 0, 0))
    start = Vertex.ByCoordinates(0, 0, 0)
    middle = Vertex.ByCoordinates(5, 0, 0)
    outside = Vertex.ByCoordinates(5, 1, 0)

    assert bool(Vertex.IsCoincident(start, Edge.StartVertex(edge), silent=True)) is True
    assert bool(Vertex.IsInternal(middle, edge, silent=True)) is True
    assert bool(Vertex.IsPeripheral(start, edge, silent=True)) is True
    assert bool(Vertex.IsPeripheral(middle, edge, silent=True)) is False
    assert bool(Vertex.IsExternal(outside, edge, silent=True)) is True

    face = Face.Rectangle(width=10, length=10)
    inside_face = Vertex.ByCoordinates(1, 1, 0)
    outside_face = Vertex.ByCoordinates(20, 20, 0)

    assert bool(Vertex.IsInternal(inside_face, face, silent=True)) is True
    assert bool(Vertex.IsExternal(outside_face, face, silent=True)) is True
    # TODO: Add a boundary-specific IsPeripheral assertion after confirming
    # the canonical Face.Rectangle origin/corner convention used in fixtures.


def test_is_internal_identify_returns_matching_topology():
    edge = Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(10, 0, 0))
    middle = Vertex.ByCoordinates(5, 0, 0)

    status, item = Vertex.IsInternal(middle, edge, identify=True, silent=True)
    assert bool(status) is True
    assert Topology.IsInstance(item, "Edge")


def test_is_internal_2d_accepts_single_vertex_and_vertex_lists():
    face = Face.Rectangle(width=10, length=10)
    inside = Vertex.ByCoordinates(1, 1, 0)
    outside = Vertex.ByCoordinates(20, 20, 0)

    assert bool(Vertex.IsInternal2D(inside, face, includeBoundary=True, silent=True)) is True
    assert bool(Vertex.IsInternal2D(outside, face, includeBoundary=True, silent=True)) is False
    assert Vertex.IsInternal2D([inside, outside], face, includeBoundary=True, silent=True) == [True, False]
    # TODO: Add includeBoundary=False assertions after confirming the precise
    # rectangle boundary coordinates for the canonical fixture face.


def test_enclosing_edges_faces_and_cells_find_simple_containers():
    edge = Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(10, 0, 0))
    midpoint = Vertex.ByCoordinates(5, 0, 0)
    enclosing_edges = Vertex.EnclosingEdges(midpoint, edge, silent=True)
    assert isinstance(enclosing_edges, list)
    assert len(enclosing_edges) == 1
    assert Topology.IsInstance(enclosing_edges[0], "Edge")

    face = Face.Rectangle(width=10, length=10)
    face_point = Vertex.ByCoordinates(1, 1, 0)
    enclosing_faces = Vertex.EnclosingFaces(face_point, face, silent=True)
    assert isinstance(enclosing_faces, list)
    assert len(enclosing_faces) == 1
    assert Topology.IsInstance(enclosing_faces[0], "Face")

    cell_complex = CellComplex.Prism(height=2)
    cell_point = Vertex.ByCoordinates(0.25, 0.3, 0.75)
    enclosing_cells = Vertex.EnclosingCells(cell_point, cell_complex, silent=True)
    assert isinstance(enclosing_cells, list)
    assert len(enclosing_cells) == 1
    assert Topology.IsInstance(enclosing_cells[0], "Cell")

    # TODO: Add project-specific expected counts for multi-cell fixtures once stable
    # regression fixture geometry is available.


def test_nearest_vertex_uses_topology_vertices():
    vertices = [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(5, 0, 0),
        Vertex.ByCoordinates(10, 0, 0),
    ]
    cluster = Cluster.ByTopologies(vertices)
    query = Vertex.ByCoordinates(4.8, 0.1, 0)

    nearest = Vertex.NearestVertex(query, cluster)
    assert Vertex.Index(nearest, vertices) == 1
    nearest_without_kdtree = Vertex.NearestVertex(query, cluster, useKDTree=False)
    assert Vertex.Index(nearest_without_kdtree, vertices) == 1


def test_degree_incoming_edges_and_outgoing_edges_for_wire():
    source = Vertex.ByCoordinates(0, 0, 0)
    center = Vertex.ByCoordinates(1, 0, 0)
    target = Vertex.ByCoordinates(2, 0, 0)
    incoming = Edge.ByStartVertexEndVertex(source, center)
    outgoing = Edge.ByStartVertexEndVertex(center, target)
    wire = Wire.ByEdges([incoming, outgoing])

    assert Vertex.Degree(center, wire, topologyType="edge") == 2
    assert len(Vertex.IncomingEdges(center, wire)) == 1
    assert len(Vertex.OutgoingEdges(center, wire)) == 1


def test_fuse_and_weld_keep_list_length_and_merge_nearby_coordinates():
    vertices = [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(0.00001, 0, 0),
        Vertex.ByCoordinates(1, 0, 0),
    ]

    fused = Vertex.Fuse(vertices, tolerance=0.001)
    welded = Vertex.Weld(vertices, tolerance=0.001)

    assert len(fused) == len(vertices)
    assert len(welded) == len(vertices)
    assert Vertex.Distance(fused[0], fused[1]) == pytest.approx(0)
    assert Vertex.Distance(welded[0], welded[1]) == pytest.approx(0)
    assert Vertex.Fuse([]) is None


def test_interpolate_value_supports_topologic_vertices_and_tgraph_vertex_records():
    source_a = Vertex.ByCoordinates(0, 0, 0)
    source_b = Vertex.ByCoordinates(10, 0, 0)
    source_a = Topology.SetDictionary(source_a, Dictionary.ByPythonDictionary({"intensity": 10.0}))
    source_b = Topology.SetDictionary(source_b, Dictionary.ByPythonDictionary({"intensity": 30.0}))

    exact_target = Vertex.ByCoordinates(0, 0, 0)
    interpolated = Vertex.InterpolateValue(exact_target, [source_a, source_b], n=2, key="intensity")
    out_dictionary = Topology.Dictionary(interpolated)
    assert _dictionary_value(out_dictionary, "intensity") == pytest.approx(10.0)

    record = {"dictionary": {"x": 0, "y": 0, "z": 0}}
    source_records = [
        {"dictionary": {"x": 0, "y": 0, "z": 0, "intensity": 12.0}},
        {"dictionary": {"x": 10, "y": 0, "z": 0, "intensity": 32.0}},
    ]
    result_record = Vertex.InterpolateValue(record, source_records, n=2, key="intensity")
    assert result_record is record
    assert result_record["dictionary"]["intensity"] == pytest.approx(12.0)


def test_external_boundary_of_vertex_is_none():
    vertex = Vertex.ByCoordinates(0, 0, 0)
    assert Vertex.ExternalBoundary(vertex, silent=True) is None
    assert Vertex.ExternalBoundary(None, silent=True) is None


def test_transform_applies_4_by_4_matrix():
    vertex = Vertex.ByCoordinates(1, 2, 3)
    matrix = [
        [1, 0, 0, 10],
        [0, 1, 0, 20],
        [0, 0, 1, 30],
        [0, 0, 0, 1],
    ]

    transformed = Vertex.Transform(vertex, matrix)
    _assert_vertex(transformed)
    _assert_coords(transformed, [11, 22, 33])
    assert Vertex.Transform(vertex, [[1, 0], [0, 1]], silent=True) is None
    assert Vertex.Transform(None, matrix, silent=True) is None


def test_random_vertex_returns_noncoincident_vertex_inside_input_bounding_box():
    random.seed(12345)
    vertices = [Vertex.ByCoordinates(0, 0, 0), Vertex.ByCoordinates(10, 10, 10)]

    candidate = Vertex.RandomVertex(vertices, maxTries=100, pad=0, silent=True)

    _assert_vertex(candidate)
    x, y, z = Vertex.Coordinates(candidate)
    assert 0 <= x <= 10
    assert 0 <= y <= 10
    assert 0 <= z <= 10
    assert all(Vertex.Distance(candidate, v) > 0.0001 for v in vertices)


def test_separate_moves_coincident_vertices_apart():
    vertices = [
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(0, 0, 0),
        Vertex.ByCoordinates(1, 0, 0),
    ]

    separated = Vertex.Separate(vertices, minDistance=0.05, iterations=50, strength=0.5, silent=True)

    assert isinstance(separated, list)
    assert len(separated) == len(vertices)
    assert Vertex.Distance(separated[0], separated[1]) >= 0.05 - 1e-3
