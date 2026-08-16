# Copyright (C) 2026
# TopologicPy Shell unit tests.

import pytest


Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Topology = pytest.importorskip("topologicpy.Topology").Topology


TOLERANCE = 1e-6


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _assert_wire(wire):
    assert Topology.IsInstance(wire, "Wire")


def _assert_face(face):
    assert Topology.IsInstance(face, "Face")


def _assert_shell(shell):
    assert Topology.IsInstance(shell, "Shell")


def _assert_topology(topology):
    assert Topology.IsInstance(topology, "Topology")


def _face_area_sum(shell):
    faces = Shell.Faces(shell)
    assert isinstance(faces, list)
    return sum(Face.Area(face) for face in faces)


def _rectangle_wire(z=0, width=2, length=1):
    vertices = [
        _v(-width * 0.5, -length * 0.5, z),
        _v(width * 0.5, -length * 0.5, z),
        _v(width * 0.5, length * 0.5, z),
        _v(-width * 0.5, length * 0.5, z),
    ]
    return Wire.ByVertices(vertices, close=True, silent=True)


def _polyline_wire():
    return Wire.ByVertices(
        [_v(0, 0, 0), _v(2, 0, 0), _v(2, 1, 0)],
        close=False,
        silent=True,
    )


@pytest.fixture
def rectangle_shell():
    return Shell.Rectangle(width=4, length=2, uSides=2, vSides=1)


@pytest.fixture
def two_face_shell():
    f1 = Face.Rectangle(origin=_v(0, 0, 0), width=1, length=1, placement="lowerleft", silent=True)
    f2 = Face.Rectangle(origin=_v(1, 0, 0), width=1, length=1, placement="lowerleft", silent=True)
    return Shell.ByFaces([f1, f2], silent=True)


def test_by_faces_and_by_faces_cluster_create_shells(two_face_shell):
    _assert_shell(two_face_shell)
    assert len(Shell.Faces(two_face_shell)) == 2
    assert _face_area_sum(two_face_shell) == pytest.approx(2)

    faces = Shell.Faces(two_face_shell)
    cluster_shell = Shell.ByFacesCluster(Cluster.ByTopologies(faces), silent=True)

    _assert_shell(cluster_shell)
    assert len(Shell.Faces(cluster_shell)) == 2

    assert Shell.ByFaces(None, silent=True) is None
    assert Shell.ByFaces([], silent=True) is None
    assert Shell.ByFacesCluster(None, silent=True) is None


def test_rectangle_and_square_create_expected_gridded_shells(rectangle_shell):
    square = Shell.Square(size=2, uSides=2, vSides=2)

    _assert_shell(rectangle_shell)
    _assert_shell(square)

    assert len(Shell.Faces(rectangle_shell)) == 2
    assert len(Shell.Faces(square)) == 4
    assert _face_area_sum(rectangle_shell) == pytest.approx(8)
    assert _face_area_sum(square) == pytest.approx(4)


def test_accessors_return_faces_edges_vertices_and_wires(rectangle_shell):
    faces = Shell.Faces(rectangle_shell)
    edges = Shell.Edges(rectangle_shell)
    vertices = Shell.Vertices(rectangle_shell)
    wires = Shell.Wires(rectangle_shell)

    assert isinstance(faces, list)
    assert isinstance(edges, list)
    assert isinstance(vertices, list)
    assert isinstance(wires, list)
    assert len(faces) == 2
    assert len(edges) >= 4
    assert len(vertices) >= 4
    assert len(wires) >= 1

    assert Shell.Faces(None) is None
    assert Shell.Edges(None) is None
    assert Shell.Vertices(None) is None
    assert Shell.Wires(None) is None


def test_external_boundary_internal_boundaries_and_is_closed(rectangle_shell):
    external = Shell.ExternalBoundary(rectangle_shell, silent=True)
    internal = Shell.InternalBoundaries(rectangle_shell, silent=True)

    _assert_wire(external)
    assert isinstance(internal, list)
    assert len(internal) == 0
    assert bool(Shell.IsClosed(rectangle_shell)) is False

    assert Shell.ExternalBoundary(None, silent=True) is None
    assert Shell.InternalBoundaries(None, silent=True) is None


def test_is_closed_invalid_input_returns_none():
    assert Shell.IsClosed(None) is None


def test_is_on_boundary_for_external_boundary_vertex(rectangle_shell):
    boundary = Shell.ExternalBoundary(rectangle_shell, silent=True)
    boundary_vertex = Wire.Vertices(boundary)[0]

    assert bool(Shell.IsOnBoundary(rectangle_shell, boundary_vertex)) is True
    assert Shell.IsOnBoundary(None, boundary_vertex) is None
    assert Shell.IsOnBoundary(rectangle_shell, None) is None


def test_is_on_boundary_returns_false_for_non_boundary_vertex():
    shell = Shell.Rectangle(width=4, length=2, uSides=2, vSides=1)
    assert bool(Shell.IsOnBoundary(shell, _v(10, 10, 0))) is False


def test_internal_edges_returns_list_for_two_adjacent_faces(two_face_shell):
    internal_edges = Shell.InternalEdges(two_face_shell, silent=True)

    assert isinstance(internal_edges, list)
    # TODO: After Shell.InternalEdges grouping is confirmed, assert the exact
    # number and type of returned merged internal-edge groups.

    assert Shell.InternalEdges(None, silent=True) is None


def test_by_wires_and_by_wires_cluster_loft_between_matching_wires():
    bottom = _rectangle_wire(z=0, width=2, length=1)
    top = _rectangle_wire(z=1, width=2, length=1)

    shell = Shell.ByWires([bottom, top], triangulate=False, silent=True)
    cluster_shell = Shell.ByWiresCluster(Cluster.ByTopologies([bottom, top]), triangulate=False, silent=True)

    _assert_shell(shell)
    _assert_shell(cluster_shell)
    assert len(Shell.Faces(shell)) >= 4
    assert len(Shell.Faces(cluster_shell)) >= 4

    assert Shell.ByWires(None, silent=True) is None
    assert Shell.ByWiresCluster(None, silent=True) is None
    assert Shell.ByWires([bottom, Wire.Line(length=1)], triangulate=False, silent=True) is None


def test_circle_and_pie_create_shells_with_positive_face_area():
    circle = Shell.Circle(radius=1, sides=12)
    annular_pie = Shell.Pie(radiusA=2, radiusB=1, sides=12, rings=1, fromAngle=0, toAngle=180)

    for shell in [circle, annular_pie]:
        _assert_shell(shell)
        assert len(Shell.Faces(shell)) > 0
        assert _face_area_sum(shell) > 0

    assert Shell.Circle(radius=0, sides=12) is None
    assert Shell.Pie(radiusA=1, radiusB=1, sides=12) is None
    assert Shell.Pie(radiusA=1, radiusB=0, sides=2) is None
    assert Shell.Pie(radiusA=1, radiusB=0, fromAngle=0, toAngle=0) is None


def test_delaunay_and_voronoi_return_partition_topologies():
    face = Face.Rectangle(width=4, length=4, silent=True)
    vertices = [
        _v(-1, -1, 0),
        _v(1, -1, 0),
        _v(1, 1, 0),
        _v(-1, 1, 0),
        _v(0, 0, 0),
    ]

    delaunay = Shell.Delaunay(vertices, tolerance=0.0001)
    voronoi = Shell.Voronoi(vertices, face=face, tolerance=0.0001)

    _assert_topology(delaunay)
    _assert_topology(voronoi)
    assert len(Topology.Faces(delaunay, silent=True)) > 0
    assert len(Topology.Faces(voronoi, silent=True)) > 0

    assert Shell.Delaunay(None) is None
    assert Shell.Delaunay([_v(0, 0, 0), _v(1, 0, 0)]) is None
    assert Shell.Voronoi([_v(0, 0, 0)], face=face) is None


def test_surface_generators_return_shells_or_valid_topologies():
    rectangular_hp = Shell.HyperbolicParaboloidRectangularDomain(uSides=3, vSides=3)
    circular_hp = Shell.HyperbolicParaboloidCircularDomain(radius=1, sides=8, rings=3)
    paraboloid = Shell.Paraboloid(width=2, length=2, uSides=4, vSides=4)

    for topology in [rectangular_hp, circular_hp, paraboloid]:
        _assert_topology(topology)
        assert len(Topology.Faces(topology, silent=True)) > 0


def test_mobius_strip_constructor_and_invalid_parameters():
    mobius = Shell.MobiusStrip(radius=1, height=0.5, uSides=8, vSides=1, silent=True)

    _assert_shell(mobius)
    assert len(Shell.Faces(mobius)) > 0

    assert Shell.MobiusStrip(radius=0, silent=True) is None
    assert Shell.MobiusStrip(height=0, silent=True) is None
    assert Shell.MobiusStrip(uSides=2, silent=True) is None
    assert Shell.MobiusStrip(vSides=0, silent=True) is None
    assert Shell.MobiusStrip(direction=[0, 0], silent=True) is None


def test_golden_rectangle_constructor_and_invalid_parameters():
    shell = Shell.GoldenRectangle(width=2, maxIterations=3, includeSpiral=False, silent=True)

    _assert_shell(shell)
    assert len(Shell.Faces(shell)) >= 3

    assert Shell.GoldenRectangle(width=0, silent=True) is None
    assert Shell.GoldenRectangle(width=1, maxIterations=0, silent=True) is None
    assert Shell.GoldenRectangle(width=1, maxIterations=3, includeSpiral=True, sides=2, silent=True) is None
    assert Shell.GoldenRectangle(width=1, placement="invalid", silent=True) is None
    assert Shell.GoldenRectangle(width=1, direction=[0, 0], silent=True) is None


def test_planarize_remove_collinear_edges_simplify_and_self_merge(rectangle_shell):
    planarized = Shell.Planarize(rectangle_shell)
    cleaned = Shell.RemoveCollinearEdges(rectangle_shell, silent=True)
    simplified = Shell.Simplify(rectangle_shell, tolerance=0.01)
    merged_face = Shell.SelfMerge(rectangle_shell)

    _assert_topology(planarized)
    _assert_shell(cleaned)
    _assert_shell(simplified)
    _assert_face(merged_face)

    assert Face.Area(merged_face) == pytest.approx(8)

    assert Shell.Planarize(None) is None
    assert Shell.RemoveCollinearEdges(None, silent=True) is None
    assert Shell.Simplify(None) is None
    assert Shell.SelfMerge(None) is None


def test_by_thickened_wire_creates_shell_from_non_collinear_polyline():
    shell = Shell.ByThickenedWire(_polyline_wire(), offsetA=0.25, offsetB=0.25)

    _assert_shell(shell)
    assert len(Shell.Faces(shell)) > 0

    assert Shell.ByThickenedWire(None) is None


def test_roof_and_skeleton_reject_invalid_face_input():
    assert Shell.Roof(None) is None
    assert Shell.Skeleton(None) is None


def test_roof_and_skeleton_create_topologies_for_rectangle_face():
    face = Face.Rectangle(
        width=4,
        length=2,
        silent=True,
    )

    roof = Shell.Roof(
        face,
        angle=30,
    )

    skeleton = Shell.Skeleton(
        face,
    )

    _assert_shell(roof)
    _assert_shell(skeleton)

    roof_faces = Topology.Faces(
        roof,
        silent=True,
    )

    skeleton_faces = Topology.Faces(
        skeleton,
        silent=True,
    )

    assert isinstance(roof_faces, list)
    assert isinstance(skeleton_faces, list)

    assert len(roof_faces) > 0
    assert len(skeleton_faces) > 0
