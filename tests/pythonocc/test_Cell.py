# Auto-generated PythonOCC backend parity test
# Copied from test_Cell.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# TopologicPy Cell unit tests.

import pytest


Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Cell = pytest.importorskip("topologicpy.Cell").Cell
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
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


def _assert_cell(cell):
    assert Topology.IsInstance(cell, "Cell")


def _assert_topology(topology):
    assert Topology.IsInstance(topology, "Topology")


def _volume(cell):
    volume = Cell.Volume(cell)
    assert volume is not None
    return abs(volume)


def _area(cell):
    area = Cell.SurfaceArea(cell)
    assert area is not None
    return abs(area)


def _rectangle_wire(z=0, width=2, length=2):
    vertices = [
        _v(-width * 0.5, -length * 0.5, z),
        _v(width * 0.5, -length * 0.5, z),
        _v(width * 0.5, length * 0.5, z),
        _v(-width * 0.5, length * 0.5, z),
    ]
    return Wire.ByVertices(vertices, close=True, silent=True)


@pytest.fixture
def cube_cell():
    return Cell.Cube(size=2)


@pytest.fixture
def rectangular_prism():
    return Cell.Prism(width=2, length=3, height=4)


def test_prism_box_and_cube_create_cells_with_expected_measures(rectangular_prism):
    box = Cell.Box(width=2, length=3, height=4)
    cube = Cell.Cube(size=2)

    for cell in [rectangular_prism, box, cube]:
        _assert_cell(cell)
        assert _volume(cell) > 0
        assert _area(cell) > 0

    assert Cell.Volume(rectangular_prism) == pytest.approx(24)
    assert Cell.Volume(box) == pytest.approx(24)
    assert Cell.Volume(cube) == pytest.approx(8)
    assert Cell.SurfaceArea(rectangular_prism) == pytest.approx(52)
    assert Cell.Area(cube) == pytest.approx(24)


def test_accessors_on_cube_return_faces_edges_vertices_wires_and_shells(cube_cell):
    faces = Cell.Faces(cube_cell)
    edges = Cell.Edges(cube_cell)
    vertices = Cell.Vertices(cube_cell)
    wires = Cell.Wires(cube_cell)
    shells = Cell.Shells(cube_cell)
    external_boundary = Cell.ExternalBoundary(cube_cell, silent=True)
    internal_boundaries = Cell.InternalBoundaries(cube_cell)

    assert isinstance(faces, list)
    assert isinstance(edges, list)
    assert isinstance(vertices, list)
    assert isinstance(wires, list)
    assert isinstance(shells, list)
    assert isinstance(internal_boundaries, list)

    assert len(faces) == 6
    assert len(edges) >= 12
    assert len(vertices) >= 8
    assert len(wires) >= 1
    assert len(shells) >= 1
    assert len(internal_boundaries) == 0
    _assert_shell(external_boundary)


def test_invalid_accessors_return_none_or_empty_lists():
    assert Cell.Faces(None) is None
    assert Cell.Edges(None) is None
    assert Cell.Vertices(None) is None
    assert Cell.Wires(None) is None
    assert Cell.Shells(None) is None
    assert Cell.ExternalBoundary(None, silent=True) is None
    assert Cell.InternalBoundaries(None) == []
    assert Cell.Volume(None) is None


def test_internal_vertex_containment_status_and_boundary_checks(cube_cell):
    internal = Cell.InternalVertex(cube_cell, silent=True)
    outside = _v(10, 10, 10)
    boundary = _v(1, 0, 0)

    _assert_vertex(internal)
    assert Cell.ContainmentStatus(cube_cell, internal) == 0
    assert Cell.ContainmentStatus(cube_cell, outside) == 2
    assert bool(Cell.IsOnBoundary(cube_cell, outside)) is False
    assert isinstance(Cell.IsOnBoundary(cube_cell, boundary), bool)

    assert Cell.InternalVertex(None, silent=True) is None
    assert Cell.ContainmentStatus(None, internal) is None
    assert Cell.ContainmentStatus(cube_cell, None) is None
    assert Cell.IsOnBoundary(None, internal) is None
    assert Cell.IsOnBoundary(cube_cell, None) is None


def test_decompose_cube_classifies_faces(cube_cell):
    decomposition = Cell.Decompose(cube_cell)

    expected_keys = {
        "verticalFaces",
        "topHorizontalFaces",
        "bottomHorizontalFaces",
        "inclinedFaces",
        "verticalApertures",
        "topHorizontalApertures",
        "bottomHorizontalApertures",
        "inclinedApertures",
    }

    assert isinstance(decomposition, dict)
    assert set(decomposition.keys()) == expected_keys
    assert len(decomposition["verticalFaces"]) == 4
    assert len(decomposition["topHorizontalFaces"]) == 1
    assert len(decomposition["bottomHorizontalFaces"]) == 1
    assert len(decomposition["inclinedFaces"]) == 0
    assert Cell.Decompose(None) is None


def test_reconstruct_cell_from_faces_cluster_and_shell(cube_cell):
    faces = Cell.Faces(cube_cell)
    face_cluster = Cluster.ByTopologies(faces)
    shell = Cell.Shells(cube_cell)[0]

    from_faces = Cell.ByFaces(faces, silent=True)
    from_cluster = Cell.ByFacesCluster(face_cluster, silent=True)
    from_shell = Cell.ByShell(shell, silent=True)

    for cell in [from_faces, from_cluster, from_shell]:
        _assert_cell(cell)
        assert Cell.Volume(cell) == pytest.approx(Cell.Volume(cube_cell))

    assert Cell.ByFaces(None, silent=True) is None
    assert Cell.ByFaces([], silent=True) is None
    assert Cell.ByFacesCluster(None, silent=True) is None
    assert Cell.ByShell(None, silent=True) is None


def test_by_shells_reconstructs_external_boundary_only(cube_cell):
    external = Cell.ExternalBoundary(cube_cell, silent=True)
    reconstructed = Cell.ByShells(external, [], silent=True)

    _assert_cell(reconstructed)
    assert Cell.Volume(reconstructed) == pytest.approx(Cell.Volume(cube_cell))
    assert Cell.ByShells(None, [], silent=True) is None


def test_by_thickened_face_creates_expected_volume():
    face = Face.Rectangle(width=2, length=3, silent=True)

    one_sided = Cell.ByThickenedFace(face, thickness=4, bothSides=False, silent=True)
    both_sides = Cell.ByThickenedFace(face, thickness=4, bothSides=True, silent=True)

    for cell in [one_sided, both_sides]:
        _assert_cell(cell)
        assert Cell.Volume(cell) == pytest.approx(24)

    assert Cell.ByThickenedFace(None, silent=True) is None
    assert Cell.ByThickenedFace(face, thickness=0, silent=True) is None
    assert Cell.ByThickenedFace(face, thickness=1, wSides=0, silent=True) is None


def test_by_thickened_shell_creates_cell_from_open_shell():
    shell = Shell.Rectangle(width=2, length=3, uSides=1, vSides=1)

    cell = Cell.ByThickenedShell(shell, thickness=4, bothSides=False, silent=True)

    _assert_cell(cell)
    assert Cell.Volume(cell) == pytest.approx(24)
    assert Cell.ByThickenedShell(None, silent=True) is None


def test_by_wires_and_by_wires_cluster_create_lofted_cell():
    lower = _rectangle_wire(z=0, width=2, length=2)
    upper = _rectangle_wire(z=2, width=2, length=2)

    cell = Cell.ByWires([lower, upper], triangulate=False, silent=True)
    cluster_cell = Cell.ByWiresCluster(Cluster.ByTopologies([lower, upper]), triangulate=False)

    for item in [cell, cluster_cell]:
        _assert_cell(item)
        assert Cell.Volume(item) == pytest.approx(8)

    assert Cell.ByWires(None, silent=True) is None
    assert Cell.ByWires([lower], silent=True) is None
    assert Cell.ByWiresCluster(None) is None


def test_offset_remove_collinear_edges_and_external_boundary(cube_cell):
    offset = Cell.ByOffset(cube_cell, offset=0.1)
    cleaned = Cell.RemoveCollinearEdges(cube_cell, silent=True)

    _assert_topology(offset)
    _assert_cell(cleaned)
    assert _volume(offset) > 0
    assert Cell.Volume(cleaned) == pytest.approx(Cell.Volume(cube_cell))
    assert Cell.RemoveCollinearEdges(None, silent=True) is None


def test_compactness_for_cube(cube_cell):
    sphere_reference = Cell.Compactness(cube_cell, reference="sphere")
    cube_reference = Cell.Compactness(cube_cell, reference="cube")

    assert 0 < sphere_reference <= 1
    assert cube_reference == pytest.approx(1)


def test_cylinder_cone_sphere_and_capsule_constructors_return_cells():
    constructors = [
        Cell.Cylinder(radius=1, height=2, uSides=8, vSides=1),
        Cell.Cone(baseRadius=1, topRadius=0, height=2, uSides=8, vSides=1),
        Cell.Cone(baseRadius=1, topRadius=0.5, height=2, uSides=8, vSides=1),
        Cell.Sphere(radius=1, uSides=8, vSides=4, silent=True),
        Cell.Capsule(radius=0.5, height=2, uSides=8, vSidesEnds=3, vSidesMiddle=1),
    ]

    for cell in constructors:
        _assert_cell(cell)
        assert _volume(cell) > 0

    assert Cell.Sphere(radius=0, silent=True) is None


def test_polyhedron_constructors_return_cells():
    constructors = [
        Cell.Tetrahedron(length=1, depth=0, silent=True),
        Cell.Octahedron(radius=1, silent=True),
        Cell.Icosahedron(radius=1, silent=True),
        Cell.Dodecahedron(radius=1, silent=True),
    ]

    for cell in constructors:
        _assert_cell(cell)
        assert _volume(cell) > 0


def test_torus_paraboloid_and_hyperboloid_return_cells_or_valid_topologies():
    constructors = [
        Cell.Torus(majorRadius=1, minorRadius=0.25, uSides=8, vSides=4),
        Cell.Paraboloid(width=2, length=2, height=1, uSides=4, vSides=4, silent=True),
        Cell.Hyperboloid(baseRadius=0.75, topRadius=0.75, height=1, sides=8, silent=True),
    ]

    for topology in constructors:
        _assert_topology(topology)
        assert len(Topology.Faces(topology, silent=True)) > 0


def test_section_shape_constructors_return_cells():
    constructors = [
        Cell.CrossShape(width=4, length=4, height=1, silent=True),
        Cell.CShape(width=4, length=4, height=1, silent=True),
        Cell.IShape(width=4, length=4, height=1, silent=True),
        Cell.LShape(width=4, length=4, height=1, silent=True),
        Cell.TShape(width=4, length=4, height=1, silent=True),
        Cell.RHS(width=4, length=3, height=2, thickness=0.25, silent=True),
        Cell.SHS(size=4, height=2, thickness=0.25, silent=True),
        Cell.CHS(radius=1, height=2, thickness=0.2, sides=12, silent=True),
        Cell.Tube(radius=1, height=2, thickness=0.2, sides=12, silent=True),
    ]

    for cell in constructors:
        _assert_cell(cell)
        assert _volume(cell) > 0

    assert Cell.CrossShape(width=0, length=4, height=1, silent=True) is None
    assert Cell.CShape(width=0, length=4, height=1, silent=True) is None
    assert Cell.IShape(width=0, length=4, height=1, silent=True) is None
    assert Cell.LShape(width=0, length=4, height=1, silent=True) is None
    assert Cell.TShape(width=0, length=4, height=1, silent=True) is None
    assert Cell.RHS(width=1, length=1, height=1, thickness=0.5, silent=True) is None
    assert Cell.SHS(size=1, height=1, thickness=0.5, silent=True) is None
    assert Cell.CHS(radius=1, height=1, thickness=0.5, silent=True) is None
    assert Cell.Tube(radius=1, height=1, thickness=0.5, silent=True) is None


def test_pipe_returns_expected_dictionary():
    edge = Edge.ByStartVertexEndVertex(_v(0, 0, 0), _v(0, 0, 2), silent=True)

    result = Cell.Pipe(edge, radius=0.25, sides=8)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"pipe", "endcapA", "endcapB"}
    _assert_cell(result["pipe"])
    assert result["endcapA"] is None
    assert result["endcapB"] is None
    assert Cell.Pipe(None) is None


def test_sets_classifies_cells_inside_super_cell():
    cell_a = Cell.Cube(origin=_v(-1, 0, 0), size=1)
    cell_b = Cell.Cube(origin=_v(1, 0, 0), size=1)
    super_cell = Cell.Cube(size=6)

    sets = Cell.Sets([cell_a, cell_b], [super_cell])

    assert isinstance(sets, list)
    assert len(sets) == 1
    assert len(sets[0]) == 2
    assert Cell.Sets(None, [super_cell]) is None
    assert Cell.Sets([cell_a], None) is None


def test_roof_returns_cell_for_rectangular_face_or_none_without_raising():
    face = Face.Rectangle(width=4, length=2, silent=True)

    roof = Cell.Roof(face, angle=30)

    if roof is not None:
        _assert_cell(roof)
        assert _volume(roof) > 0
    assert Cell.Roof(None) is None
