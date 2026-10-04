# Copyright (C) 2026
# Wassim Jabi
#
# Unified TopologicPy Topology regression and unit test suite.

"""
Consolidated tests for TopologicPy.Topology and its generic topology-query,
editing, geometry, relationship, meshing, and persistence pathways.

The suite is backend-neutral unless a test is explicitly marked
``pythonocc_only``. The shared tests/conftest.py selects and restores the active
backend for each test.
"""

import json
import math
import os
import uuid
import zipfile
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import pytest


Aperture = pytest.importorskip("topologicpy.Aperture").Aperture
Cell = pytest.importorskip("topologicpy.Cell").Cell
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
Context = pytest.importorskip("topologicpy.Context").Context
Core = pytest.importorskip("topologicpy.Core").Core
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary
Edge = pytest.importorskip("topologicpy.Edge").Edge
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Topology = pytest.importorskip("topologicpy.Topology").Topology
Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Wire = pytest.importorskip("topologicpy.Wire").Wire


ACTIVE_BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").strip().lower()
if not ACTIVE_BACKEND:
    _backend_name = type(Core.Backend()).__name__.lower()
    ACTIVE_BACKEND = "pythonocc" if "pythonocc" in _backend_name else "topologic_core"

IS_PYTHONOCC = ACTIVE_BACKEND == "pythonocc"
IS_TOPOLOGIC_CORE = ACTIVE_BACKEND == "topologic_core"

GEOMETRY_TOLERANCE = 1.0e-6
RELATIONSHIP_TOLERANCE = 1.0e-3
QUERY_TOLERANCE = 1.0e-4
QUERY_COORDINATE_MANTISSA = 9

# ============================================================================
# Shared fixtures and reusable geometry helpers
# ============================================================================

def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)

def _coords(vertex, mantissa=6):
    return Vertex.Coordinates(vertex, mantissa=mantissa)

def _assert_coords(vertex, expected, abs_tol=GEOMETRY_TOLERANCE, mantissa=6):
    assert Topology.IsInstance(vertex, "Vertex")
    actual = _coords(vertex, mantissa=mantissa)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)

def _assert_topology(topology):
    assert Topology.IsInstance(topology, "Topology")

def _dict_value(dictionary, key, default=None):
    try:
        value = Dictionary.ValueAtKey(dictionary, key, default)
    except TypeError:
        value = Dictionary.ValueAtKey(dictionary, key)
        if value is None:
            value = default
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value

@pytest.fixture
def square_face():
    return Face.Rectangle(width=2, length=2, silent=True)

@pytest.fixture
def simple_cell():
    return Cell.Prism(width=2, length=2, height=2, silent=True)

@pytest.fixture
def mixed_cluster(square_face):
    edge = Edge.ByVertices([_v(0, 0, 0), _v(1, 0, 0)], silent=True)
    wire = Wire.ByVertices(
        [_v(0, 0, 0), _v(1, 0, 0), _v(1, 1, 0)],
        close=False,
        silent=True,
    )
    vertex = _v(5, 0, 0)
    return Cluster.ByTopologies([vertex, edge, wire, square_face], silent=True)

def _quarter_cylinder_face(radius=1.0, height=2.0):
    s2 = math.sqrt(2.0) / 2.0
    cps = [
        [Vertex.ByCoordinates(radius, 0.0, 0.0), Vertex.ByCoordinates(radius, 0.0, height)],
        [Vertex.ByCoordinates(radius, radius, 0.0), Vertex.ByCoordinates(radius, radius, height)],
        [Vertex.ByCoordinates(0.0, radius, 0.0), Vertex.ByCoordinates(0.0, radius, height)],
    ]
    weights = [
        [1.0, 1.0],
        [s2, s2],
        [1.0, 1.0],
    ]
    return Face.ByNurbsParameters(
        cps,
        weights=weights,
        uKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 1.0, 1.0],
        isRational=True,
        uDegree=2,
        vDegree=1,
        silent=True,
    )

def _xyz(vertex):
    return [float(v) for v in Vertex.Coordinates(vertex, mantissa=9)]

def _coords_close(a, b, tol=1.0e-6):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))

def _circle_wire(z=0.0, radius=1.0):
    edge = Edge.Circle(
        origin=Vertex.ByCoordinates(0.0, 0.0, z),
        radius=radius,
        silent=True,
    )
    assert Topology.IsInstance(edge, "Edge")
    wire = Wire.ByEdges([edge], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


# ============================================================================
# Type system, construction, accessors, and basic geometry
# ============================================================================

def test_type_id_mapping_and_invalid_names():
    expected = {
        "vertex": 1,
        "edge": 2,
        "wire": 4,
        "face": 8,
        "shell": 16,
        "cell": 32,
        "cellcomplex": 64,
        "cluster": 128,
        "aperture": 256,
        "context": 512,
        "dictionary": 1024,
        "graph": 2048,
        "tgraph": 2048,
        "topology": 4096,
    }

    for name, type_id in expected.items():
        assert Topology.TypeID(name) == type_id
        assert Topology.TypeID(name.upper()) == type_id

    assert Topology.TypeID(None) is None
    assert Topology.TypeID("not_a_topology_type") is None

def test_is_instance_type_and_type_as_string_for_basic_topologies(square_face, simple_cell):
    vertex = _v(0, 0, 0)
    edge = Edge.ByVertices([_v(0, 0, 0), _v(1, 0, 0)], silent=True)
    wire = Wire.Rectangle(width=1, length=1, silent=True)
    shell = Cell.ExternalBoundary(simple_cell, silent=True)
    cluster = Cluster.ByTopologies([vertex, edge, square_face], silent=True)

    examples = [
        (vertex, "Vertex"),
        (edge, "Edge"),
        (wire, "Wire"),
        (square_face, "Face"),
        (shell, "Shell"),
        (simple_cell, "Cell"),
        (cluster, "Cluster"),
    ]

    for topology, name in examples:
        assert Topology.IsInstance(topology, name)
        assert Topology.IsInstance(topology, "Topology")
        assert Topology.Type(topology) == Topology.TypeID(name)
        assert Topology.TypeAsString(topology, silent=True).lower() == name.lower()

    assert bool(Topology.IsInstance(None, "Topology")) is False
    assert Topology.TypeAsString(None, silent=True) is None

def test_by_geometry_creates_requested_topology_types():
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    faces = [[0, 1, 2, 3]]

    vertex = Topology.ByGeometry(vertices[:1], topologyType="vertex", silent=True)
    edge = Topology.ByGeometry(vertices[:2], edges=[[0, 1]], topologyType="edge", silent=True)
    wire = Topology.ByGeometry(vertices, edges=edges, topologyType="wire", silent=True)
    face = Topology.ByGeometry(vertices, faces=faces, topologyType="face", silent=True)
    cluster = Topology.ByGeometry(vertices, edges=edges, faces=faces, silent=True)

    assert Topology.IsInstance(vertex, "Vertex")
    assert Topology.IsInstance(edge, "Edge")
    assert Topology.IsInstance(wire, "Wire")
    assert Topology.IsInstance(face, "Face")
    assert Topology.IsInstance(cluster, "Topology")

    assert Topology.ByGeometry([], silent=True) is None
    assert Topology.ByGeometry(vertices, edges=[[0, 99]], topologyType="edge", silent=True) is None

def test_topology_accessors_respect_dimension(square_face, simple_cell):
    assert len(Topology.Vertices(square_face, silent=True)) >= 4
    assert len(Topology.Edges(square_face, silent=True)) >= 4
    assert len(Topology.Wires(square_face, silent=True)) >= 1
    assert len(Topology.Faces(square_face, silent=True)) == 1
    assert Topology.Cells(square_face, silent=True) == []
    assert Topology.Shells(square_face, silent=True) == []
    assert Topology.CellComplexes(square_face, silent=True) == []

    assert len(Topology.Vertices(simple_cell, silent=True)) >= 8
    assert len(Topology.Edges(simple_cell, silent=True)) >= 12
    assert len(Topology.Faces(simple_cell, silent=True)) >= 6
    assert len(Topology.Shells(simple_cell, silent=True)) >= 1
    assert len(Topology.Cells(simple_cell, silent=True)) == 1
    assert Topology.CellComplexes(simple_cell, silent=True) == []

def test_invalid_accessor_inputs_return_none():
    assert Topology.Vertices(None, silent=True) is None
    assert Topology.Edges(None, silent=True) is None
    assert Topology.Wires(None, silent=True) is None
    assert Topology.Faces(None, silent=True) is None
    assert Topology.Shells(None, silent=True) is None
    assert Topology.Cells(None, silent=True) is None
    assert Topology.CellComplexes(None, silent=True) is None
    assert Topology.Clusters(None, silent=True) is None
    assert Topology.ExternalBoundary(None, silent=True) is None
    assert Topology.Centroid(None, silent=True) is None
    assert Topology.VerticesCentroid(None, silent=True) is None

def test_external_boundary_dispatches_to_type_specific_implementations(square_face, simple_cell, mixed_cluster):
    edge = Edge.ByVertices([_v(0, 0, 0), _v(1, 0, 0)], silent=True)
    open_wire = Wire.Line(length=1, silent=True)
    closed_wire = Wire.Rectangle(width=1, length=1, silent=True)

    assert Topology.IsInstance(Topology.ExternalBoundary(edge, silent=True), "Cluster")
    assert Topology.IsInstance(Topology.ExternalBoundary(open_wire, silent=True), "Cluster")
    assert Topology.ExternalBoundary(closed_wire, silent=True) is None
    assert Topology.IsInstance(Topology.ExternalBoundary(square_face, silent=True), "Wire")
    assert Topology.IsInstance(Topology.ExternalBoundary(simple_cell, silent=True), "Shell")
    assert Topology.IsInstance(Topology.ExternalBoundary(mixed_cluster, silent=True), "Topology")

def test_centroid_center_of_mass_and_vertices_centroid(square_face):
    centroid = Topology.Centroid(square_face, silent=True)
    center_of_mass = Topology.CenterOfMass(square_face)
    vertices_centroid = Topology.VerticesCentroid(square_face, silent=True)

    _assert_coords(centroid, [0, 0, 0])
    _assert_coords(center_of_mass, [0, 0, 0])
    _assert_coords(vertices_centroid, [0, 0, 0])

    assert Topology.CenterOfMass(None) is None

def test_center_of_mass_and_centroid_alias_are_consistent():
    origin = Vertex.ByCoordinates(3.0, -4.0, 5.0)
    cell = Cell.Box(
        origin=origin,
        width=2.0,
        length=4.0,
        height=6.0,
        placement="center",
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")

    center = Topology.CenterOfMass(cell, silent=True)
    centroid = Topology.Centroid(cell, silent=True)

    assert Topology.IsInstance(center, "Vertex")
    assert Topology.IsInstance(centroid, "Vertex")
    assert _coords_close(_xyz(center), [3.0, -4.0, 5.0])
    assert _coords_close(_xyz(centroid), [3.0, -4.0, 5.0])

def test_bounding_box_returns_valid_topology_with_dictionary(square_face, simple_cell):
    for topology in [square_face, simple_cell]:
        bbox = Topology.BoundingBox(topology, optimize=0, silent=True)
        _assert_topology(bbox)
        dictionary = Topology.Dictionary(bbox, silent=True)
        keys = Dictionary.Keys(dictionary)
        assert isinstance(keys, list)
        assert "xrot" in keys
        assert "yrot" in keys
        assert "zrot" in keys

    assert Topology.BoundingBox(None, silent=True) is None


# ============================================================================
# Dictionaries, filtering, grouping, and metadata
# ============================================================================

def test_dictionary_set_add_and_uuid_round_trip():
    vertex = _v(1, 2, 3)
    vertex = Topology.SetDictionary(vertex, Dictionary.ByPythonDictionary({"name": "alpha"}), silent=True)
    vertex = Topology.AddDictionary(vertex, Dictionary.ByPythonDictionary({"level": 2}))

    dictionary = Topology.Dictionary(vertex, silent=True)
    assert _dict_value(dictionary, "name") == "alpha"
    assert _dict_value(dictionary, "level") == 2

    uuid_a = Topology.UUID(vertex, uuidKey="uuid", silent=True)
    uuid_b = Topology.UUID(vertex, uuidKey="uuid", silent=True)
    assert uuid_a == uuid_b
    assert str(uuid.UUID(uuid_a)) == uuid_a

    assert Topology.Dictionary(None, silent=True) is None
    assert Topology.SetDictionary(None, Dictionary.ByPythonDictionary({}), silent=True) is None
    assert Topology.AddDictionary(vertex, None) is None
    assert Topology.UUID(None, silent=True) is None

def test_filter_supports_type_and_dictionary_searches():
    a = Topology.SetDictionary(_v(0, 0, 0), Dictionary.ByPythonDictionary({"zone": "public lobby", "tag": "A-01"}), silent=True)
    b = Topology.SetDictionary(_v(1, 0, 0), Dictionary.ByPythonDictionary({"zone": "private office", "tag": "B-02"}), silent=True)
    edge = Edge.ByVertices([a, b], silent=True)
    items = [a, b, edge]

    vertices = Topology.Filter(items, topologyType="vertex")
    public = Topology.Filter(items, topologyType="vertex", searchType="contains", key="zone", value="public")
    wildcard = Topology.Filter(items, topologyType="vertex", searchType="equal to", key="tag", value="A-*")

    assert len(vertices["filtered"]) == 2
    assert len(vertices["other"]) == 1
    assert public["filtered"] == [a]
    assert wildcard["filtered"] == [a]
    assert Topology.Filter(None) is None

def test_bin_by_dictionary_key_returns_groups_and_counts():
    a = Topology.SetDictionary(_v(0, 0, 0), Dictionary.ByPythonDictionary({"group": "A", "data": [1, 2]}), silent=True)
    b = Topology.SetDictionary(_v(1, 0, 0), Dictionary.ByPythonDictionary({"group": "A", "data": [1, 2]}), silent=True)
    c = Topology.SetDictionary(_v(2, 0, 0), Dictionary.ByPythonDictionary({"group": "B", "data": [2, 3]}), silent=True)
    d = _v(3, 0, 0)

    groups, counts = Topology.BinByDictionaryKey([a, b, c, d], key="group", return_counts=True)
    assert counts["A"] == 2
    assert counts["B"] == 1
    assert counts["__MISSING__"] == 1
    assert groups["A"] == [a, b]

    data_groups, _ = Topology.BinByDictionaryKey([a, b, c], key="data")
    assert sorted(len(group) for group in data_groups.values()) == [1, 2]

def test_cluster_by_keys_groups_topologies_with_matching_dictionary_values():
    a = Topology.SetDictionary(_v(0, 0, 0), Dictionary.ByPythonDictionary({"zone": "A"}), silent=True)
    b = Topology.SetDictionary(_v(1, 0, 0), Dictionary.ByPythonDictionary({"zone": "A"}), silent=True)
    c = Topology.SetDictionary(_v(2, 0, 0), Dictionary.ByPythonDictionary({"zone": "B"}), silent=True)

    groups = Topology.ClusterByKeys([a, b, c], "zone", silent=True)

    assert isinstance(groups, list)
    assert sorted(len(group) for group in groups) == [1, 2]
    assert Topology.ClusterByKeys(None, "zone", silent=True) is None
    assert Topology.ClusterByKeys([], "zone", silent=True) is None
    assert Topology.ClusterByKeys([a], silent=True) is None

def test_inherit_transfers_dictionary_from_enclosing_source():
    source = Face.Rectangle(width=10, length=10, silent=True)
    source = Topology.SetDictionary(source, Dictionary.ByPythonDictionary({"zone": "source"}), silent=True)
    target = Face.Rectangle(width=1, length=1, silent=True)

    inherited = Topology.Inherit([target], [source], keys=["zone"], silent=True)

    assert isinstance(inherited, list)
    assert len(inherited) == 1
    assert _dict_value(Topology.Dictionary(inherited[0], silent=True), "zone") == "source"

    assert Topology.Inherit([], [source], silent=True) is None
    assert Topology.Inherit([target], [], silent=True) is None


# ============================================================================
# Geometry extraction, meshing, and tessellation
# ============================================================================

def _validate_mesh(mesh):
    assert isinstance(mesh, dict)
    assert mesh["schema"] == "topologicpy.mesh/1"

    vertices = mesh["vertices"]
    faces = mesh["faces"]
    cells = mesh["cells"]

    assert isinstance(vertices, list)
    assert isinstance(faces, list)
    assert cells == []

    assert mesh["verts"] is vertices
    assert mesh["tris"] == faces
    assert mesh["quads"] == []
    assert mesh["tets"] == []

    for face in faces:
        assert len(face) == 3
        assert len(set(face)) == 3

        for index in face:
            assert 0 <= index < len(vertices)

    metadata = mesh["metadata"]

    assert metadata["vertexCount"] == len(
        vertices
    )
    assert metadata["faceCount"] == len(
        faces
    )
    assert metadata["triangleCount"] == len(
        faces
    )

def test_geometry_and_mesh_data_export_basic_structure(square_face, simple_cell):
    geometry = Topology.Geometry(square_face, silent=True)
    mesh_data = Topology.MeshData(simple_cell, mode=1, silent=True)

    for data in [geometry, mesh_data]:
        assert isinstance(data, dict)
        assert isinstance(data.get("vertices"), list)
        assert isinstance(data.get("edges"), list)
        assert isinstance(data.get("faces"), list)
        assert len(data["vertices"]) > 0

    assert Topology.Geometry(None, silent=True) is None
    assert Topology.MeshData(None, silent=True) is None

def test_mesh_to_topologies_creates_vertices_faces_and_cells():
    mesh = Topology.MeshToTopologies(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 1, 2]],
        tets=[[0, 1, 2, 3]],
        silent=True,
    )

    assert isinstance(mesh, dict)
    assert len(mesh["vertices"]) == 4
    assert len(mesh["faces"]) == 1
    assert len(mesh["cells"]) == 1
    assert all(Topology.IsInstance(v, "Vertex") for v in mesh["vertices"])
    assert all(Topology.IsInstance(f, "Face") for f in mesh["faces"])
    assert all(Topology.IsInstance(c, "Cell") for c in mesh["cells"])

@pytest.mark.pythonocc_only
def test_geometry_and_meshdata_tessellate_curved_face_without_failure():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")

    geometry = Topology.Geometry(face, triangulate=True, silent=True)
    assert isinstance(geometry, dict)
    assert isinstance(geometry.get("vertices"), list) and len(geometry["vertices"]) >= 3
    assert isinstance(geometry.get("faces"), list) and len(geometry["faces"]) >= 1

    mesh = Topology.MeshData(face, mode=0, silent=True)
    assert isinstance(mesh, dict)
    assert isinstance(mesh.get("vertices"), list) and len(mesh["vertices"]) >= 3
    assert isinstance(mesh.get("faces"), list) and len(mesh["faces"]) >= 1

@pytest.mark.pythonocc_only
def test_triangulate_curved_face_intentionally_returns_planar_triangles():
    face = _quarter_cylinder_face()
    result = Topology.Triangulate(face, silent=True)
    assert Topology.IsInstance(result, "Topology")

    faces = Topology.Faces(result) or []
    if Topology.IsInstance(result, "Face"):
        faces = [result]
    assert len(faces) >= 1
    for triangle in faces:
        vertices = Topology.Vertices(triangle) or []
        assert len(vertices) == 3
        assert Face.IsPlanar(triangle, silent=True) is True

def test_tessellate_planar_face_returns_canonical_triangle_mesh():
    face = Face.Rectangle(
        width=4.0,
        length=3.0,
        silent=True,
    )

    assert Topology.IsInstance(
        face,
        "Face",
    )

    mesh = Topology.Tessellate(
        face,
        silent=True,
    )

    _validate_mesh(mesh)

    assert len(mesh["faces"]) >= 2

    assert mesh["metadata"]["source"] in (
        "occt",
        "topologic_core",
    )

def test_tessellate_validation_is_non_throwing():
    assert (
        Topology.Tessellate(
            None,
            silent=True,
        )
        is None
    )

    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    assert (
        Topology.Tessellate(
            face,
            quality="bad",
            silent=True,
        )
        is None
    )

    assert (
        Topology.Tessellate(
            face,
            linearDeflection=0,
            silent=True,
        )
        is None
    )

    assert (
        Topology.Tessellate(
            face,
            angularDeflection=180,
            silent=True,
        )
        is None
    )

def test_tessellate_box_produces_surface_triangles_only():
    cell = Cell.Box(
        width=2.0,
        length=3.0,
        height=4.0,
        silent=True,
    )

    mesh = Topology.Tessellate(
        cell,
        silent=True,
    )

    _validate_mesh(mesh)

    assert len(mesh["faces"]) >= 12
    assert mesh["cells"] == []

@pytest.mark.pythonocc_only
def test_tessellate_exact_cylindrical_shell_preserves_curved_surface():
    c0 = Edge.Circle(
        radius=1.0,
        silent=True,
    )

    c1 = Topology.Translate(
        Edge.Circle(
            radius=1.0,
            silent=True,
        ),
        z=2.0,
        silent=True,
    )
    w0 = Wire.ByEdges(
        [c0],
        silent=True,
    )

    w1 = Wire.ByEdges(
        [c1],
        silent=True,
    )

    shell = Shell.ByWires(
        [w0, w1],
        polyhedron=False,
        silent=True,
    )

    assert Topology.IsInstance(
        shell,
        "Shell",
    )

    coarse = Topology.Tessellate(
        shell,
        quality="coarse",
        remesh=True,
        silent=True,
    )

    fine = Topology.Tessellate(
        shell,
        quality="fine",
        remesh=True,
        silent=True,
    )

    _validate_mesh(coarse)
    _validate_mesh(fine)

    assert coarse["metadata"]["source"] == "occt"
    assert fine["metadata"]["source"] == "occt"

    assert len(coarse["faces"]) > 0
    assert len(fine["faces"]) >= len(
        coarse["faces"]
    )

    for x, y, z in fine["vertices"]:
        radius = math.sqrt(
            x * x
            + y * y
        )

        assert math.isclose(
            radius,
            1.0,
            rel_tol=1.0e-4,
            abs_tol=1.0e-4,
        )

        assert (
            -1.0e-6
            <= z
            <= 2.0 + 1.0e-6
        )

@pytest.mark.pythonocc_only
def test_tessellate_welding_reduces_or_preserves_vertex_count():
    cell = Cell.Cylinder(
        radius=1.0,
        height=2.0,
        uSides=24,
        vSides=1,
        polyhedron=False,
        silent=True,
    )

    welded = Topology.Tessellate(
        cell,
        quality="medium",
        weld=True,
        remesh=True,
        silent=True,
    )

    unwelded = Topology.Tessellate(
        cell,
        quality="medium",
        weld=False,
        remesh=True,
        silent=True,
    )

    _validate_mesh(welded)
    _validate_mesh(unwelded)

    assert len(
        welded["vertices"]
    ) <= len(
        unwelded["vertices"]
    )


# ============================================================================
# Transforms, copying, native shapes, and curve preservation
# ============================================================================

def test_copy_preserves_type_and_geometry(square_face):
    copied = Topology.Copy(square_face)

    assert Topology.IsInstance(copied, "Face")
    assert copied is not square_face
    assert len(Topology.Vertices(copied, silent=True)) == len(Topology.Vertices(square_face, silent=True))
    assert Topology.Copy(None) is None

def test_translate_move_rotate_scale_and_place_vertices():
    vertex = _v(1, 2, 3)

    translated = Topology.Translate(vertex, x=1, y=-2, z=3, silent=True)
    moved = Topology.Move(vertex, x=-1, y=-2, z=-3)
    rotated = Topology.Rotate(_v(1, 0, 0), origin=_v(0, 0, 0), axis=[0, 0, 1], angle=90, silent=True)
    scaled = Topology.Scale(_v(1, 2, 3), origin=_v(0, 0, 0), x=2, y=3, z=4, silent=True)
    placed = Topology.Place(_v(1, 1, 1), originA=_v(1, 1, 1), originB=_v(5, 6, 7))

    _assert_coords(translated, [2, 0, 6])
    _assert_coords(moved, [0, 0, 0])
    _assert_coords(rotated, [0, 1, 0], abs_tol=1e-5)
    _assert_coords(scaled, [2, 6, 12])
    _assert_coords(placed, [5, 6, 7])

    assert Topology.Translate(None, silent=True) is None
    assert Topology.Rotate(None, silent=True) is None
    assert Topology.Scale(None, silent=True) is None
    assert Topology.Place(None) is None

def test_translate_preserves_dictionary_when_requested():
    vertex = Topology.SetDictionary(_v(1, 2, 3), Dictionary.ByPythonDictionary({"id": "source"}), silent=True)
    translated = Topology.Translate(vertex, x=1, y=1, z=1, transferDictionaries=True, silent=True)

    assert _dict_value(Topology.Dictionary(translated, silent=True), "id") == "source"

def test_transform_preserves_arc_geometry_and_start_end_direction():
    arc = Edge.Arc(radius=2.0, fromAngle=0.0, toAngle=90.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    original_length = Edge.Length(arc, mantissa=None, silent=True)

    matrix = [
        [0.0, -1.0, 0.0, 10.0],
        [1.0,  0.0, 0.0, 20.0],
        [0.0,  0.0, 1.0,  5.0],
        [0.0,  0.0, 0.0,  1.0],
    ]
    transformed = Topology.Transform(arc, matrix, silent=True)
    assert Topology.IsInstance(transformed, "Edge")
    assert Edge.IsLinear(transformed, silent=True) is False
    assert math.isclose(
        Edge.Length(transformed, mantissa=None, silent=True),
        original_length,
        rel_tol=2e-6,
        abs_tol=2e-6,
    )

    # Original start is (2,0,0). After +90deg about Z and translation it is
    # (10,22,5). This explicitly guards topological start->end orientation.
    start = _xyz(Edge.StartVertex(transformed))
    assert all(abs(a-b) <= 2e-6 for a,b in zip(start, [10.0, 22.0, 5.0]))

@pytest.mark.pythonocc_only
def test_transform_preserves_nurbs_surface_and_area():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False
    area = Face.Area(face, mantissa=None, silent=True)

    matrix = [
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 0.0,-1.0, 3.0],
        [0.0, 1.0, 0.0,-2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    transformed = Topology.Transform(face, matrix, silent=True)
    assert Topology.IsInstance(transformed, "Face")
    assert Face.IsPlanar(transformed, silent=True) is False
    assert math.isclose(
        Face.Area(transformed, mantissa=None, silent=True),
        area,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

@pytest.mark.pythonocc_only
def test_copy_and_deepcopy_preserve_curved_edge_geometry_and_direction():
    arc = Edge.Arc(
        radius=3.0,
        fromAngle=20.0,
        toAngle=140.0,
        silent=True
    )
    assert Topology.IsInstance(arc, "Edge")

    shallow = Topology.Copy(
        arc,
        deep=False,
        silent=True
    )
    deep = Topology.DeepCopy(
        arc,
        silent=True
    )

    for result in (shallow, deep):
        assert Topology.IsInstance(result, "Edge")
        assert Edge.IsLinear(result, silent=True) is False

        assert math.isclose(
            Edge.Length(
                result,
                mantissa=None,
                silent=True
            ),
            Edge.Length(
                arc,
                mantissa=None,
                silent=True
            ),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )

        assert _coords_close(
            _xyz(Edge.StartVertex(result)),
            _xyz(Edge.StartVertex(arc))
        )

        assert _coords_close(
            _xyz(Edge.EndVertex(result)),
            _xyz(Edge.EndVertex(arc))
        )

@pytest.mark.pythonocc_only
def test_occt_shape_roundtrip_preserves_nurbs_surface():
    face = _quarter_cylinder_face(radius=1.5, height=2.25)
    assert Topology.IsInstance(face, "Face")

    shape = Topology.OCCTShape(face, silent=True)
    assert shape is not None

    result = Topology.ByOCCTShape(shape, silent=True)
    assert Topology.IsInstance(result, "Face")
    assert Face.IsPlanar(result, silent=True) is False
    assert math.isclose(
        Face.Area(result, mantissa=None, silent=True),
        Face.Area(face, mantissa=None, silent=True),
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )

@pytest.mark.pythonocc_only
def test_occtshape_of_lightweight_cluster_returns_none_without_restructuring_cluster():
    face = Face.Rectangle(silent=True)
    cluster = Cluster.ByTopologies([face], silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    assert Topology.OCCTShape(cluster, silent=True) is None
    assert len(Topology.Faces(cluster) or []) == 1

def test_native_query_validation_is_non_throwing():
    assert Topology.OCCTShape(None, silent=True) is None
    assert Topology.ByOCCTShape(None, silent=True) is None
    assert Topology.CenterOfMass(None, silent=True) is None
    assert Topology.Centroid(None, silent=True) is None
    assert Topology.DeepCopy(None, silent=True) is None
    assert Topology.IsPlanar(None, silent=True) is None
    assert Topology.IsPlanar(Face.Rectangle(silent=True), tolerance=0.0, silent=True) is None


# ============================================================================
# Planarity and decomposition
# ============================================================================

def _gentle_vertical_nurbs_face():
    # A vertical quadratic NURBS patch with a shallow bow in Y.
    # Its surface is genuinely non-planar, but its normals remain tightly
    # clustered around a horizontal mean direction, so it has a clear overall
    # vertical orientation.
    cps = [
        [Vertex.ByCoordinates(0.0, 0.0, 0.0), Vertex.ByCoordinates(0.0, 0.0, 3.0)],
        [Vertex.ByCoordinates(1.0, 0.25, 0.0), Vertex.ByCoordinates(1.0, 0.25, 3.0)],
        [Vertex.ByCoordinates(2.0, 0.0, 0.0), Vertex.ByCoordinates(2.0, 0.0, 3.0)],
    ]
    weights = [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    return Face.ByNurbsParameters(
        cps,
        weights=weights,
        uKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 1.0, 1.0],
        isRational=False,
        uDegree=2,
        vDegree=1,
        silent=True,
    )

def test_decompose_returns_expected_logical_keys(simple_cell):
    decomposition = Topology.Decompose(simple_cell, silent=True)

    expected_keys = {
        "cells",
        "externalVerticalFaces",
        "topHorizontalFaces",
        "bottomHorizontalFaces",
        "internalHorizontalFaces",
        "verticalFaces",
        "horizontalFaces",
        "inclinedFaces",
    }

    assert isinstance(decomposition, dict)
    assert expected_keys.issubset(set(decomposition.keys()))
    assert len(decomposition["cells"]) == 1
    assert len(decomposition["verticalFaces"]) >= 4
    assert len(decomposition["horizontalFaces"]) >= 2

def test_isplanar_uses_curve_geometry_not_only_topological_vertices():
    arc = Edge.Arc(radius=2.0, fromAngle=15.0, toAngle=165.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    assert Topology.IsPlanar(arc, silent=True) is True

    bezier = Edge.Bezier(
        [
            Vertex.ByCoordinates(0.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 2.0, 0.0),
            Vertex.ByCoordinates(2.0, 0.0, 0.0),
        ],
        silent=True,
    )
    assert Topology.IsInstance(bezier, "Edge")
    assert Topology.IsPlanar(bezier, silent=True) is True

    helix = Edge.Helix(radius=1.0, height=2.0, turns=1.25, sides=24, silent=True)
    assert Topology.IsInstance(helix, "Edge")
    assert Topology.IsPlanar(helix, tolerance=1.0e-4, silent=True) is False

def test_isplanar_handles_faces_and_higher_dimensional_topologies():
    face = Face.Rectangle(width=4.0, length=3.0, silent=True)
    cell = Cell.Box(width=4.0, length=3.0, height=2.0, silent=True)

    assert Topology.IsPlanar(face, silent=True) is True
    assert Topology.IsPlanar(cell, silent=True) is False

@pytest.mark.pythonocc_only
def test_isplanar_detects_actual_curved_surface():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False
    assert Topology.IsPlanar(face, silent=True) is False

@pytest.mark.pythonocc_only
def test_isplanar_detects_curved_shell_even_when_vertices_are_insufficient():
    e0 = Edge.Circle(radius=1.0, silent=True)
    e1 = Topology.Translate(e0, z=2.0, silent=True)
    w0 = Wire.ByEdges([e0], silent=True)
    w1 = Wire.ByEdges([e1], silent=True)

    shell = Shell.ByWires([w0, w1], polyhedron=False, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert Topology.IsPlanar(shell, silent=True) is False

def test_planar_box_classification_and_existing_keys_are_unchanged():
    cell = Cell.Box(width=4.0, length=3.0, height=2.0, silent=True)
    result = Topology.Decompose(cell, silent=True)

    assert isinstance(result, dict)
    assert len(result["externalVerticalFaces"]) == 4
    assert len(result["topHorizontalFaces"]) == 1
    assert len(result["bottomHorizontalFaces"]) == 1
    assert len(result["externalInclinedFaces"]) == 0

    # New categories are additive; ordinary planar geometry must not migrate
    # into them.
    assert result["externalCurvedFaces"] == []
    assert result["internalCurvedFaces"] == []
    assert result["freeCurvedFaces"] == []
    assert result["curvedFaces"] == []

def test_curved_category_keys_are_always_present():
    cell = Cell.Box(silent=True)
    result = Topology.Decompose(cell, silent=True)

    for key in (
        "externalCurvedFaces",
        "internalCurvedFaces",
        "freeCurvedFaces",
        "externalCurvedApertures",
        "internalCurvedApertures",
        "freeCurvedApertures",
        "curvedFaces",
    ):
        assert key in result
        assert isinstance(result[key], list)

@pytest.mark.pythonocc_only
def test_gently_curved_surface_retains_overall_vertical_classification():
    face = _gentle_vertical_nurbs_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False

    cluster = Cluster.ByTopologies([face], silent=True)
    result = Topology.Decompose(cluster, normalSpreadAngle=30.0, silent=True)

    assert face in result["freeVerticalFaces"]
    assert face not in result["freeCurvedFaces"]
    assert len(result["curvedFaces"]) == 0

@pytest.mark.pythonocc_only
def test_normal_spread_threshold_can_force_same_surface_into_curved_category():
    face = _gentle_vertical_nurbs_face()
    cluster = Cluster.ByTopologies([face], silent=True)

    result = Topology.Decompose(cluster, normalSpreadAngle=3.0, silent=True)

    assert face in result["freeCurvedFaces"]
    assert face not in result["freeVerticalFaces"]

@pytest.mark.pythonocc_only
def test_strongly_curved_patch_is_classified_as_curved():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    assert Face.IsPlanar(face, silent=True) is False

    cluster = Cluster.ByTopologies([face], silent=True)
    result = Topology.Decompose(cluster, normalSpreadAngle=30.0, silent=True)

    assert face in result["freeCurvedFaces"]
    assert face not in result["freeVerticalFaces"]
    assert face in result["curvedFaces"]

@pytest.mark.pythonocc_only
def test_external_cylindrical_face_is_classified_as_external_curved():
    cell = Cell.Cylinder(
        radius=1.0,
        height=2.0,
        uSides=32,
        vSides=1,
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")

    result = Topology.Decompose(cell, normalSpreadAngle=30.0, silent=True)

    # The two planar end caps remain horizontal. The periodic lateral surface
    # has no coherent mean normal and must be in the curved category.
    assert len(result["topHorizontalFaces"]) == 1
    assert len(result["bottomHorizontalFaces"]) == 1
    assert len(result["externalCurvedFaces"]) >= 1
    assert all(face in result["curvedFaces"] for face in result["externalCurvedFaces"])


# ============================================================================
# Merging, booleans, slicing, imposing, and imprinting
# ============================================================================

def _grid_edge(x1, y1, x2, y2):
    return Edge.ByVertices(
        Vertex.ByCoordinates(float(x1), float(y1), 0.0),
        Vertex.ByCoordinates(float(x2), float(y2), 0.0),
        silent=True,
    )

def test_self_merge_and_merge_all_return_valid_topologies():
    edge_a = Edge.ByVertices([_v(0, 0, 0), _v(1, 0, 0)], silent=True)
    edge_b = Edge.ByVertices([_v(1, 0, 0), _v(2, 0, 0)], silent=True)

    merged = Topology.SelfMerge(
        Cluster.ByTopologies([edge_a, edge_b], silent=True),
        silent=True,
    )
    assert Topology.IsInstance(merged, "Wire")

    merged_all = Topology.MergeAll([edge_a, edge_b], silent=True)
    _assert_topology(merged_all)

    cell = Cell.Prism(
        origin=_v(0, 0, 0),
        width=1,
        length=1,
        height=1,
        placement="lowerleft",
        silent=True,
    )
    faces = Topology.Faces(cell, silent=True)
    edges = Topology.Edges(cell, silent=True)

    singleton = Topology.SelfMerge(
        Cluster.ByTopologies([cell], silent=True),
        silent=True,
    )
    assert Topology.IsInstance(singleton, "Cell")

    duplicate_face = Topology.SelfMerge(
        Cluster.ByTopologies([faces[0], faces[0]], silent=True),
        silent=True,
    )
    assert Topology.IsInstance(duplicate_face, "Face")

    # Deliberately redundant, mixed-dimensional input:
    # - cell appears twice
    # - faces[0] is already part of cell
    # - edges[0] is already part of faces[0]/cell
    #
    # PythonOCC canonicalizes this to a Cell, whereas native TopologicCore may
    # preserve a valid CellComplex. The backend-neutral contract here is that
    # SelfMerge returns a valid topology rather than enforcing identical
    # canonicalization across kernels.
    redundant = Topology.SelfMerge(
        Cluster.ByTopologies(
            [cell, faces[0], edges[0], cell],
            silent=True,
        ),
        silent=True,
    )
    _assert_topology(redundant)

    face_soup = Topology.SelfMerge(
        Cluster.ByTopologies(faces, silent=True),
        silent=True,
    )
    assert Topology.IsInstance(face_soup, "Cell")

    adjacent = Cell.Prism(
        origin=_v(1, 0, 0),
        width=1,
        length=1,
        height=1,
        placement="lowerleft",
        silent=True,
    )
    cell_complex = Topology.SelfMerge(
        Cluster.ByTopologies([cell, adjacent], silent=True),
        silent=True,
    )
    assert Topology.IsInstance(cell_complex, "CellComplex")
    assert len(Topology.Cells(cell_complex, silent=True)) == 2

    disconnected = Cell.Prism(
        origin=_v(5, 0, 0),
        width=1,
        length=1,
        height=1,
        placement="lowerleft",
        silent=True,
    )

def test_boolean_operations_return_topology_or_none_for_simple_faces():
    face_a = Face.Rectangle(origin=_v(0, 0, 0), width=2, length=2, silent=True)
    face_b = Face.Rectangle(origin=_v(1, 0, 0), width=2, length=2, silent=True)

    for operation in [Topology.Intersect, Topology.Difference, Topology.Union, Topology.SymDif, Topology.XOR]:
        result = operation(face_a, face_b, silent=True)
        assert result is None or Topology.IsInstance(result, "Topology")

    assert Topology.Intersect(None, face_b, silent=True) is None
    assert Topology.Difference(None, face_b, silent=True) is None
    union_identity = Topology.Union(None, face_b, silent=True)
    assert Topology.IsSame(union_identity, face_b)

def test_slice_impose_and_imprint_return_topology_or_none(simple_cell):
    cutter = Face.Rectangle(width=3, length=3, silent=True)

    for operation in [Topology.Slice, Topology.Impose, Topology.Imprint]:
        result = operation(simple_cell, cutter, silent=True)
        assert result is None or Topology.IsInstance(result, "Topology")

    assert Topology.Slice(None, cutter, silent=True) is None
    assert Topology.Impose(None, cutter, silent=True) is None
    assert Topology.Imprint(None, cutter, silent=True) is None

def test_slice_face_with_grid_edges_produces_well_formed_shell():
    face = Face.Rectangle(width=6.0, length=6.0, placement="lowerleft", silent=True)
    assert Topology.IsInstance(face, "Face")

    # A 2 x 2 set of full-span interior cutters divides the face into a 3 x 3 grid.
    cutters = [
        _grid_edge(2.0, -1.0, 2.0, 7.0),
        _grid_edge(4.0, -1.0, 4.0, 7.0),
        _grid_edge(-1.0, 2.0, 7.0, 2.0),
        _grid_edge(-1.0, 4.0, 7.0, 4.0),
    ]
    assert all(Topology.IsInstance(edge, "Edge") for edge in cutters)
    grid = Cluster.ByTopologies(cutters, silent=True)
    assert Topology.IsInstance(grid, "Cluster")

    sliced = Topology.Slice(face, grid, tolerance=1.0e-4, silent=True)

    assert Topology.IsInstance(sliced, "Shell")
    faces = Topology.Faces(sliced, silent=True)
    edges = Topology.Edges(sliced, silent=True)
    vertices = Topology.Vertices(sliced, silent=True)

    # A coherent 3 x 3 planar subdivision has 9 faces, 24 shared edges and 16
    # shared vertices. These counts catch duplicated/unshared subtopologies.
    assert len(faces) == 9
    assert len(edges) == 24
    assert len(vertices) == 16

    areas = [Face.Area(f, mantissa=9) for f in faces]
    assert all(area is not None and area > 0.0 for area in areas)
    assert all(math.isclose(area, 4.0, rel_tol=1.0e-7, abs_tol=1.0e-7) for area in areas)
    assert math.isclose(sum(areas), 36.0, rel_tol=1.0e-7, abs_tol=1.0e-7)

    # Euler characteristic for a connected planar disk: V - E + F = 1.
    assert len(vertices) - len(edges) + len(faces) == 1


# ============================================================================
# Editing and simplification
# ============================================================================

def _rect_face(x0, y0, x1, y1, z=0.0):
    vertices = [
        Vertex.ByCoordinates(x0, y0, z),
        Vertex.ByCoordinates(x1, y0, z),
        Vertex.ByCoordinates(x1, y1, z),
        Vertex.ByCoordinates(x0, y1, z),
    ]
    wire = Wire.ByVertices(vertices, close=True, silent=True)
    assert Topology.IsInstance(wire, "Wire")
    face = Face.ByWire(wire, silent=True)
    assert Topology.IsInstance(face, "Face")
    return face

def _split_rectangle_face():
    vertices = [
        Vertex.ByCoordinates(0.0, 0.0, 0.0),
        Vertex.ByCoordinates(1.0, 0.0, 0.0),
        Vertex.ByCoordinates(2.0, 0.0, 0.0),
        Vertex.ByCoordinates(2.0, 1.0, 0.0),
        Vertex.ByCoordinates(0.0, 1.0, 0.0),
    ]
    wire = Wire.ByVertices(vertices, close=True, silent=True)
    face = Face.ByWire(wire, silent=True)
    assert Topology.IsInstance(face, "Face")
    return face

def _semicircle_face_with_split_diameter():
    arc = Edge.Arc(radius=1.0, fromAngle=0.0, toAngle=180.0, silent=True)
    assert Topology.IsInstance(arc, "Edge")
    a = Edge.EndVertex(arc)
    b = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    c = Edge.StartVertex(arc)
    e1 = Edge.ByVertices([a, b], silent=True)
    e2 = Edge.ByVertices([b, c], silent=True)
    wire = Wire.ByEdges([arc, e1, e2], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    face = Face.ByWire(wire, silent=True)
    assert Topology.IsInstance(face, "Face")
    return face

def _two_edge_wire():
    v0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    v1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)
    v2 = Vertex.ByCoordinates(2.0, 0.0, 0.0)
    e0 = Edge.ByVertices(v0, v1, silent=True)
    e1 = Edge.ByVertices(v1, v2, silent=True)
    wire = Wire.ByEdges([e0, e1], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire, e0, e1, v0, v1, v2

def _arc_tail_wire():
    arc = Edge.Arc(
        radius=2.0,
        fromAngle=0.0,
        toAngle=90.0,
        silent=True,
    )
    assert Topology.IsInstance(arc, "Edge")
    end = Edge.EndVertex(arc)
    tail_end = Vertex.ByCoordinates(0.0, 3.0, 0.0)
    tail = Edge.ByVertices(end, tail_end, silent=True)
    wire = Wire.ByEdges([arc, tail], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire, arc, tail, tail_end

def test_remove_collinear_coplanar_edges_faces_and_cleanup(square_face, simple_cell):
    cleaned_face = Topology.RemoveCollinearEdges(square_face, silent=True)
    clean_cell = Topology.RemoveCoplanarFaces(simple_cell, silent=True)
    fixed = Topology.Fix(square_face, topologyType="face")
    cleaned = Topology.Cleanup(Topology.Copy(square_face))

    _assert_topology(cleaned_face)
    if IS_TOPOLOGIC_CORE:
        assert clean_cell is None
    else:
        _assert_topology(clean_cell)
    assert fixed is None or Topology.IsInstance(fixed, "Topology")
    assert cleaned is None or Topology.IsInstance(cleaned, "Topology")

    assert Topology.RemoveCollinearEdges(None, silent=True) is None
    assert Topology.RemoveCoplanarFaces(None, silent=True) is None
    assert Topology.Cleanup("not-a-topology") is None

def test_remove_collinear_edges_still_simplifies_polyhedral_face():
    face = _split_rectangle_face()
    before = Topology.Edges(face) or []
    assert len(before) == 5

    result = Topology.RemoveCollinearEdges(
        face,
        polyhedron=True,
        silent=True,
    )
    assert Topology.IsInstance(result, "Face")
    after = Topology.Edges(result) or []
    assert len(after) == 4
    assert math.isclose(Face.Area(result, mantissa=None, silent=True), 2.0, rel_tol=1e-6, abs_tol=1e-6)

def test_curve_safe_remove_collinear_edges_never_flattens_arc():
    face = _semicircle_face_with_split_diameter()
    before = Topology.Edges(face) or []
    assert len(before) == 3
    assert sum(Edge.IsLinear(edge, silent=True) is False for edge in before) == 1

    result = Topology.RemoveCollinearEdges(
        face,
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(result, "Face")

    after = Topology.Edges(result) or []
    curved = [edge for edge in after if Edge.IsLinear(edge, silent=True) is False]
    assert len(curved) == 1

    if IS_PYTHONOCC:
        # Native KeepShape protection allows only the split straight diameter
        # to simplify while retaining the exact arc.
        assert len(after) == 2
    else:
        # TopologicCore deliberately preserves the mixed topology unchanged
        # rather than rebuilding and flattening the curve.
        assert len(after) == 3

    assert math.isclose(
        Edge.Length(curved[0], mantissa=None, silent=True),
        math.pi,
        rel_tol=2e-6,
        abs_tol=2e-6,
    )

def test_remove_coplanar_faces_still_merges_adjacent_planar_faces():
    f1 = _rect_face(0.0, 0.0, 1.0, 1.0)
    f2 = _rect_face(1.0, 0.0, 2.0, 1.0)
    shell = Shell.ByFaces([f1, f2], silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert len(Topology.Faces(shell) or []) == 2

    result = Topology.RemoveCoplanarFaces(
        shell,
        silent=True,
    )

    if not IS_PYTHONOCC:
        assert result is None
        return

    assert Topology.IsInstance(result, "Topology")
    faces = Topology.Faces(result) or []
    if Topology.IsInstance(result, "Face"):
        faces = [result]
    assert len(faces) == 1
    assert math.isclose(Face.Area(faces[0], mantissa=None, silent=True), 2.0, rel_tol=1e-6, abs_tol=1e-6)

@pytest.mark.pythonocc_only
def test_remove_coplanar_faces_preserves_and_unifies_cylindrical_surfaces():
    shell_a = Shell.ByWires(
        [_circle_wire(0.0), _circle_wire(1.0)],
        polyhedron=False,
        silent=True,
    )
    shell_b = Shell.ByWires(
        [_circle_wire(1.0), _circle_wire(2.0)],
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(shell_a, "Shell")
    assert Topology.IsInstance(shell_b, "Shell")

    faces = (Topology.Faces(shell_a) or []) + (Topology.Faces(shell_b) or [])
    shell = Shell.ByFaces(faces, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    before = Topology.Faces(shell) or []
    assert len(before) >= 2
    assert all(Face.IsPlanar(face, silent=True) is False for face in before)

    area_before = sum(Face.Area(face, mantissa=None, silent=True) for face in before)
    result = Topology.RemoveCoplanarFaces(
        shell,
        silent=True,
    )
    assert Topology.IsInstance(result, "Topology")
    after = Topology.Faces(result) or []
    if Topology.IsInstance(result, "Face"):
        after = [result]

    # Adjacent faces on the same cylindrical support surface may be
    # legitimately unified into a single curved face.
    assert len(after) >= 1
    assert len(after) <= len(before)

    # The operation must not flatten the cylindrical geometry.
    assert all(
        Face.IsPlanar(face, silent=True) is False
        for face in after
    )

    # The total curved surface area must be preserved.
    area_after = sum(
        Face.Area(
            face,
            mantissa=None,
            silent=True
        )
        for face in after
    )

    assert math.isclose(
        area_after,
        area_before,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

def test_remove_collinear_edges_validates_polyhedron_flag():
    face = _split_rectangle_face()

    assert Topology.RemoveCollinearEdges(
        face,
        polyhedron="False",
        silent=True
    ) is None

def test_remove_edges_accepts_single_edge_argument():
    wire, e0, e1, *_ = _two_edge_wire()
    result = Topology.RemoveEdges(wire, e1, silent=True)
    assert result is not None
    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1

def test_remove_vertices_accepts_single_vertex_and_cascades_incident_edge():
    wire, e0, e1, v0, v1, v2 = _two_edge_wire()
    result = Topology.RemoveVertices(wire, v2, silent=True)
    assert result is not None
    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1

def test_remove_faces_accepts_single_face_argument():
    cell = Cell.Box(width=2.0, length=2.0, height=2.0, silent=True)
    faces = Topology.Faces(cell, silent=True) or []
    assert len(faces) == 6

    result = Topology.RemoveFaces(cell, faces[0], silent=True)
    assert result is not None

    remaining = Topology.Faces(result, silent=True) or []
    assert len(remaining) == 5

def test_remove_edit_validation_and_noop_are_non_throwing():
    wire, e0, e1, *_ = _two_edge_wire()

    assert Topology.RemoveEdges(None, e0, silent=True) is None
    assert Topology.RemoveFaces(None, None, silent=True) is None
    assert Topology.RemoveVertices(None, None, silent=True) is None

    assert Topology.RemoveEdges(wire, None, silent=True) is wire
    assert Topology.RemoveVertices(wire, [], silent=True) is wire
    assert Topology.RemoveEdges(wire, e0, tolerance="bad", silent=True) is None

@pytest.mark.pythonocc_only
def test_native_remove_edge_preserves_surviving_arc_exactly():
    wire, arc, tail, _ = _arc_tail_wire()
    expected_length = Edge.Length(arc, mantissa=None, silent=True)

    result = Topology.RemoveEdges(wire, tail, silent=True)
    assert result is not None

    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1
    survivor = edges[0]

    assert Edge.IsLinear(survivor, silent=True) is False
    assert math.isclose(
        Edge.Length(survivor, mantissa=None, silent=True),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )

@pytest.mark.pythonocc_only
def test_native_remove_vertex_preserves_surviving_arc_exactly():
    wire, arc, tail, tail_end = _arc_tail_wire()
    expected_length = Edge.Length(arc, mantissa=None, silent=True)

    result = Topology.RemoveVertices(wire, tail_end, silent=True)
    assert result is not None

    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1
    survivor = edges[0]

    assert Edge.IsLinear(survivor, silent=True) is False
    assert math.isclose(
        Edge.Length(survivor, mantissa=None, silent=True),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )

@pytest.mark.pythonocc_only
def test_native_remove_face_preserves_cylindrical_surface_exactly():
    cell = Cell.Cylinder(
        radius=1.5,
        height=2.5,
        uSides=32,
        vSides=1,
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")

    faces = Topology.Faces(cell, silent=True) or []
    planar = [face for face in faces if Face.IsPlanar(face, silent=True)]
    curved = [face for face in faces if not Face.IsPlanar(face, silent=True)]

    assert len(planar) >= 2
    assert len(curved) >= 1

    expected_curved_area = sum(
        Face.Area(face, mantissa=None, silent=True)
        for face in curved
    )

    result = Topology.RemoveFaces(cell, planar[0], silent=True)
    assert result is not None

    remaining = Topology.Faces(result, silent=True) or []
    remaining_curved = [
        face for face in remaining
        if not Face.IsPlanar(face, silent=True)
    ]

    assert len(remaining_curved) >= 1
    assert math.isclose(
        sum(Face.Area(face, mantissa=None, silent=True) for face in remaining_curved),
        expected_curved_area,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )

@pytest.mark.pythonocc_only
def test_shapeless_cluster_falls_back_without_losing_surviving_curve():
    arc_a = Edge.Arc(
        radius=1.0,
        fromAngle=0.0,
        toAngle=90.0,
        silent=True,
    )
    arc_b = Topology.Translate(
        Edge.Arc(
            radius=2.0,
            fromAngle=0.0,
            toAngle=120.0,
            silent=True,
        ),
        x=10.0,
        silent=True,
    )

    cluster = Cluster.ByTopologies([arc_a, arc_b], silent=True)
    assert Topology.IsInstance(cluster, "Cluster")
    assert Topology.OCCTShape(cluster, silent=True) is None

    expected_length = Edge.Length(arc_b, mantissa=None, silent=True)

    result = Topology.RemoveEdges(cluster, arc_a, silent=True)
    assert result is not None

    edges = Topology.Edges(result, silent=True) or []
    assert len(edges) == 1
    survivor = edges[0]

    assert Edge.IsLinear(survivor, silent=True) is False
    assert math.isclose(
        Edge.Length(survivor, mantissa=None, silent=True),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )


# ============================================================================
# Topological queries, incidence, and navigation
# ============================================================================

# Deterministic two-cell octahedral query model and expected incidence cases.

QUERY_WEST = (-1.0, 0.0, 0.0)

QUERY_SOUTH = (0.0, -1.0, 0.0)

QUERY_EAST = (1.0, 0.0, 0.0)

QUERY_NORTH = (0.0, 1.0, 0.0)

QUERY_TOP = (0.0, 0.0, 1.0)

QUERY_BOTTOM = (0.0, 0.0, -1.0)

QUERY_SHARED_KEYS = ("vertices", "edges", "wires", "faces")

QUERY_SHARED_TYPES = {
    "vertices": "vertex",
    "edges": "edge",
    "wires": "wire",
    "faces": "face",
}

QUERY_SUBTOPOLOGY_CASES = (
    # Vertex
    ("vertex", "vertex", 1),

    # Edge
    ("edge", "vertex", 2),
    ("edge", "edge", 1),

    # Wire
    ("wire", "vertex", 4),
    ("wire", "edge", 4),
    ("wire", "wire", 1),

    # Face
    ("face", "vertex", 4),
    ("face", "edge", 4),
    ("face", "wire", 1),
    ("face", "face", 1),

    # Shell
    ("shell", "vertex", 5),
    ("shell", "edge", 8),
    ("shell", "wire", 5),
    ("shell", "face", 5),
    ("shell", "shell", 1),

    # Cell
    ("cell", "vertex", 5),
    ("cell", "edge", 8),
    ("cell", "wire", 5),
    ("cell", "face", 5),
    ("cell", "shell", 1),
    ("cell", "cell", 1),

    # CellComplex
    ("cellcomplex", "vertex", 6),
    ("cellcomplex", "edge", 12),
    ("cellcomplex", "wire", 9),
    ("cellcomplex", "face", 9),
    ("cellcomplex", "shell", 2),
    ("cellcomplex", "cell", 2),
    ("cellcomplex", "cellcomplex", 1),
)

QUERY_SUPERTOPOLOGY_CASES = (
    # Explicit supertopology types from an equatorial vertex.
    ("vertex", "edge", 4, "edge"),
    ("vertex", "wire", 5, "wire"),
    ("vertex", "face", 5, "face"),
    ("vertex", "shell", 2, "shell"),
    ("vertex", "cell", 2, "cell"),
    ("vertex", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from an equatorial edge.
    ("edge", "wire", 3, "wire"),
    ("edge", "face", 3, "face"),
    ("edge", "shell", 2, "shell"),
    ("edge", "cell", 2, "cell"),
    ("edge", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from the shared equatorial wire.
    ("wire", "face", 1, "face"),
    ("wire", "shell", 2, "shell"),
    ("wire", "cell", 2, "cell"),
    ("wire", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from the shared equatorial face.
    ("face", "shell", 2, "shell"),
    ("face", "cell", 2, "cell"),
    ("face", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from one constituent shell and cell.
    ("shell", "cell", 1, "cell"),
    ("shell", "cellcomplex", 1, "cellcomplex"),
    ("cell", "cellcomplex", 1, "cellcomplex"),

    # Inferred immediate supertopology type.
    ("vertex", None, 4, "edge"),
    ("edge", None, 3, "wire"),
    ("wire", None, 1, "face"),
    ("face", None, 2, "shell"),
    ("shell", None, 1, "cell"),
    ("cell", None, 1, "cellcomplex"),
)

QUERY_ADJACENCY_CASES = (
    # Same-dimensional adjacency follows the expected boundary relation.
    ("vertex", "vertex", 4),
    ("edge", "edge", 6),
    ("wire", "wire", 8),
    ("face", "face", 8),
    ("shell", "shell", 1),
    ("cell", "cell", 1),
)

QUERY_SHARED_CASES = (
    # Two equatorial edges meeting at the west vertex.
    (
        "edge_west_south",
        "edge_west_north",
        {"vertices": 1, "edges": 0, "wires": 0, "faces": 0},
    ),

    # The shared equatorial face and one triangular side face.
    (
        "shared_face",
        "top_side_face",
        {"vertices": 2, "edges": 1, "wires": 0, "faces": 0},
    ),

    # Their external wires share the same equatorial edge.
    (
        "shared_wire",
        "top_side_wire",
        {"vertices": 2, "edges": 1, "wires": 0, "faces": 0},
    ),

    # The two cell shells share the complete equatorial face.
    (
        "top_shell",
        "bottom_shell",
        {"vertices": 4, "edges": 4, "wires": 1, "faces": 1},
    ),

    # The two cells share the complete equatorial face.
    (
        "top_cell",
        "bottom_cell",
        {"vertices": 4, "edges": 4, "wires": 1, "faces": 1},
    ),

    # A constituent cell shares all of its boundary topology with its host.
    (
        "host",
        "top_cell",
        {"vertices": 5, "edges": 8, "wires": 5, "faces": 5},
    ),
)

def _query_coordinates(vertex) -> Tuple[float, float, float]:
    """Return a vertex's rounded XYZ coordinates."""
    coordinates = Vertex.Coordinates(
        vertex,
        outputType="xyz",
        mantissa=QUERY_COORDINATE_MANTISSA,
    )
    assert isinstance(coordinates, list)
    assert len(coordinates) == 3
    return tuple(float(value) for value in coordinates)

def _query_coordinate_key(
    coordinates: Sequence[float],
) -> Tuple[float, float, float]:
    """Return a stable rounded coordinate key."""
    return tuple(
        round(float(value), QUERY_COORDINATE_MANTISSA)
        for value in coordinates
    )

def _query_vertex_signature(topology) -> Tuple[Tuple[float, float, float], ...]:
    """Return a topology signature formed from its sorted vertex coordinates."""
    vertices = Topology.Vertices(topology, silent=True)
    assert isinstance(vertices, list)
    return tuple(
        sorted(_query_coordinate_key(_query_coordinates(vertex)) for vertex in vertices)
    )

def _query_target_signature(
    coordinates: Iterable[Sequence[float]],
) -> Tuple[Tuple[float, float, float], ...]:
    """Return a sorted coordinate signature for target coordinates."""
    return tuple(sorted(_query_coordinate_key(point) for point in coordinates))

def _query_centroid_z(topology) -> float:
    """Return the Z coordinate of a topology centroid."""
    centroid = Topology.Centroid(topology)
    assert Topology.IsInstance(centroid, "Vertex")
    return float(Vertex.Z(centroid, mantissa=QUERY_COORDINATE_MANTISSA))

def _query_type_name(topology) -> str:
    """Return a normalized Topologic type name."""
    try:
        type_name = Topology.TypeAsString(topology, silent=True)
    except TypeError:
        type_name = Topology.TypeAsString(topology)

    assert isinstance(type_name, str)
    return type_name.lower()

def _query_actual_type_names(topologies) -> List[str]:
    """Return normalized type names for diagnostics."""
    if not isinstance(topologies, list):
        return [type(topologies).__name__]
    return [_query_type_name(topology) for topology in topologies]

def _assert_query_topology_list(
    result,
    expected_count: int,
    expected_type: str,
    context: str,
):
    """Assert that a result is a list of the expected size and topology type."""
    assert isinstance(result, list), (
        f"{context} did not return a list. "
        f"Returned Python type: {type(result).__name__}."
    )
    assert len(result) == expected_count, (
        f"{context} returned {len(result)} objects; "
        f"expected {expected_count}. "
        f"Returned Topologic types: {_query_actual_type_names(result)}."
    )

    invalid_indices = [
        index
        for index, topology in enumerate(result)
        if not Topology.IsInstance(topology, expected_type)
    ]
    assert not invalid_indices, (
        f"{context} returned objects of the wrong type at indices "
        f"{invalid_indices}. Expected {expected_type!r}; "
        f"returned types: {_query_actual_type_names(result)}."
    )

    actual_types = [_query_type_name(topology) for topology in result]
    assert all(type_name == expected_type.lower() for type_name in actual_types), (
        f"{context} returned unexpected type names. "
        f"Expected only {expected_type!r}; returned {actual_types}."
    )

def _query_find_by_signature(
    topologies: Sequence,
    target_coordinates: Iterable[Sequence[float]],
    topology_type: str,
):
    """Find exactly one topology with the requested vertex-coordinate signature."""
    target = _query_target_signature(target_coordinates)
    matches = [
        topology
        for topology in topologies
        if Topology.IsInstance(topology, topology_type)
        and _query_vertex_signature(topology) == target
    ]
    assert len(matches) == 1, (
        f"Could not uniquely locate a {topology_type} with signature {target}. "
        f"Found {len(matches)} matches."
    )
    return matches[0]

@lru_cache(maxsize=1)
def _query_model() -> Dict[str, object]:
    """Create and index the deterministic query test model."""
    origin = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    assert Topology.IsInstance(origin, "Vertex")

    host = CellComplex.Octahedron(
        origin=origin,
        radius=1.0,
        direction=[0.0, 0.0, 1.0],
        placement="center",
        tolerance=QUERY_TOLERANCE,
    )
    assert Topology.IsInstance(host, "CellComplex")

    vertices = Topology.Vertices(host, silent=True)
    edges = Topology.Edges(host, silent=True)
    faces = Topology.Faces(host, silent=True)
    shells = Topology.Shells(host, silent=True)
    cells = Topology.Cells(host, silent=True)

    _assert_query_topology_list(vertices, 6, "vertex", "Fixture vertices")
    _assert_query_topology_list(edges, 12, "edge", "Fixture edges")
    _assert_query_topology_list(faces, 9, "face", "Fixture faces")
    _assert_query_topology_list(shells, 2, "shell", "Fixture shells")
    _assert_query_topology_list(cells, 2, "cell", "Fixture cells")

    west = _query_find_by_signature(vertices, [QUERY_WEST], "vertex")
    south = _query_find_by_signature(vertices, [QUERY_SOUTH], "vertex")
    east = _query_find_by_signature(vertices, [QUERY_EAST], "vertex")
    north = _query_find_by_signature(vertices, [QUERY_NORTH], "vertex")
    top = _query_find_by_signature(vertices, [QUERY_TOP], "vertex")
    bottom = _query_find_by_signature(vertices, [QUERY_BOTTOM], "vertex")

    edge_west_south = _query_find_by_signature(
        edges,
        [QUERY_WEST, QUERY_SOUTH],
        "edge",
    )
    edge_west_north = _query_find_by_signature(
        edges,
        [QUERY_WEST, QUERY_NORTH],
        "edge",
    )

    shared_face = _query_find_by_signature(
        faces,
        [QUERY_WEST, QUERY_SOUTH, QUERY_EAST, QUERY_NORTH],
        "face",
    )
    top_side_face = _query_find_by_signature(
        faces,
        [QUERY_TOP, QUERY_WEST, QUERY_SOUTH],
        "face",
    )

    top_cells = [cell for cell in cells if _query_centroid_z(cell) > QUERY_TOLERANCE]
    bottom_cells = [cell for cell in cells if _query_centroid_z(cell) < -QUERY_TOLERANCE]
    assert len(top_cells) == 1
    assert len(bottom_cells) == 1
    top_cell = top_cells[0]
    bottom_cell = bottom_cells[0]

    top_shells = Topology.Shells(top_cell, silent=True)
    bottom_shells = Topology.Shells(bottom_cell, silent=True)
    _assert_query_topology_list(top_shells, 1, "shell", "Top-cell shells")
    _assert_query_topology_list(bottom_shells, 1, "shell", "Bottom-cell shells")
    top_shell = top_shells[0]
    bottom_shell = bottom_shells[0]

    shared_wires = Topology.Wires(shared_face, silent=True)
    top_side_wires = Topology.Wires(top_side_face, silent=True)
    _assert_query_topology_list(shared_wires, 1, "wire", "Shared-face wires")
    _assert_query_topology_list(top_side_wires, 1, "wire", "Top-side-face wires")
    shared_wire = shared_wires[0]
    top_side_wire = top_side_wires[0]

    return {
        "cellcomplex": host,
        "vertex": west,
        "edge": edge_west_south,
        "wire": shared_wire,
        "face": shared_face,
        "shell": top_shell,
        "cell": top_cell,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "top": top,
        "bottom": bottom,
        "edge_west_south": edge_west_south,
        "edge_west_north": edge_west_north,
        "shared_wire": shared_wire,
        "top_side_wire": top_side_wire,
        "shared_face": shared_face,
        "top_side_face": top_side_face,
        "top_shell": top_shell,
        "bottom_shell": bottom_shell,
        "top_cell": top_cell,
        "bottom_cell": bottom_cell,
        "host": host,
    }

def test_shared_topology_helpers_return_expected_collection_shapes():
    edge = Edge.ByVertices([_v(0, 0, 0), _v(1, 0, 0)], silent=True)
    face = Face.Rectangle(width=2, length=2, silent=True)

    shared = Topology.SharedTopologies(edge, face)
    assert isinstance(shared, dict)
    assert set(shared.keys()) == {"vertices", "edges", "wires", "faces"}
    assert isinstance(Topology.SharedVertices(edge, face), list)
    assert isinstance(Topology.SharedEdges(edge, face), list)
    assert isinstance(Topology.SharedWires(edge, face), list)
    assert isinstance(Topology.SharedFaces(edge, face), list)

    assert Topology.SharedTopologies(None, face) is None
    assert Topology.SharedVertices(None, face) is None

def test_subtopologies_and_supertopologies_for_face_and_cell(square_face, simple_cell):
    vertices = Topology.SubTopologies(square_face, subTopologyType="vertex", silent=True)
    edges = Topology.SubTopologies(square_face, subTopologyType="edge", silent=True)
    faces = Topology.SubTopologies(simple_cell, subTopologyType="face", silent=True)

    assert isinstance(vertices, list)
    assert isinstance(edges, list)
    assert isinstance(faces, list)
    assert len(vertices) >= 4
    assert len(edges) >= 4
    assert len(faces) >= 6

    face = faces[0]
    super_cells = Topology.SuperTopologies(face, hostTopology=simple_cell, topologyType="cell")
    assert isinstance(super_cells, list)
    assert len(super_cells) >= 1

    assert Topology.SubTopologies(None, subTopologyType="vertex", silent=True) is None
    assert Topology.SuperTopologies(None, hostTopology=simple_cell, topologyType="cell") is None

def test_select_subtopology_and_degree_on_cell(simple_cell):
    selector = Topology.Centroid(simple_cell, silent=True)
    selected_cell = Topology.SelectSubTopology(simple_cell, selector, subTopologyType="cell")
    selected_face = Topology.SelectSubTopology(simple_cell, selector, subTopologyType="face")

    assert selected_cell is None or Topology.IsInstance(selected_cell, "Cell")
    assert selected_face is None or Topology.IsInstance(selected_face, "Face")

    face = Topology.Faces(simple_cell, silent=True)[0]
    degree = Topology.Degree(face, hostTopology=simple_cell)
    assert isinstance(degree, int)
    assert degree >= 1

    assert Topology.SelectSubTopology(None, selector, subTopologyType="cell") is None
    assert Topology.Degree(None, hostTopology=simple_cell) is None

def test_open_topology_helpers_return_lists(simple_cell, square_face):
    assert isinstance(Topology.OpenFaces(simple_cell), list)
    assert isinstance(Topology.OpenEdges(square_face), list)
    assert isinstance(Topology.OpenVertices(square_face), list)

    assert Topology.OpenFaces(None) is None
    assert Topology.OpenEdges(None) is None
    assert Topology.OpenVertices(None) is None

def test_internal_vertex_planar_face_and_cell_are_strictly_internal():
    face = Face.Rectangle(width=4.0, length=3.0, silent=True)
    cell = Cell.Box(width=4.0, length=3.0, height=2.0, silent=True)

    iv_face = Topology.InternalVertex(face, silent=True)
    iv_cell = Topology.InternalVertex(cell, silent=True)

    assert Topology.IsInstance(iv_face, "Vertex")
    assert Topology.IsInstance(iv_cell, "Vertex")
    assert Vertex.IsInternal(iv_face, face, tolerance=1.0e-4, silent=True)
    assert Vertex.IsInternal(iv_cell, cell, tolerance=1.0e-4, silent=True)

def test_internal_vertex_timeout_is_retained_for_compatibility_but_not_used():
    face = Face.Rectangle(width=2.0, length=2.0, silent=True)
    result = Topology.InternalVertex(face, timeout=0, silent=True)
    assert Topology.IsInstance(result, "Vertex")
    assert Vertex.IsInternal(result, face, tolerance=1.0e-4, silent=True)

def test_internal_vertex_validation_is_non_throwing():
    assert Topology.InternalVertex(None, silent=True) is None
    assert Topology.InternalVertex(Face.Rectangle(silent=True), tolerance="bad", silent=True) is None
    assert Topology.InternalVertex(Face.Rectangle(silent=True), tolerance=0.0, silent=True) is None

def test_open_vertices_of_open_linear_wire_are_endpoints():
    v0 = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    v1 = Vertex.ByCoordinates(1.0, 0.0, 0.0)
    v2 = Vertex.ByCoordinates(2.0, 1.0, 0.0)
    wire = Wire.ByVertices([v0, v1, v2], close=False, silent=True)
    assert Topology.IsInstance(wire, "Wire")

    open_vertices = Topology.OpenVertices(wire, silent=True)
    assert isinstance(open_vertices, list)
    assert len(open_vertices) == 2

@pytest.mark.pythonocc_only
def test_internal_vertex_on_curved_nurbs_face_is_internal():
    face = _quarter_cylinder_face()
    assert Topology.IsInstance(face, "Face")
    result = Topology.InternalVertex(face, silent=True)
    assert Topology.IsInstance(result, "Vertex")
    assert Vertex.IsInternal(result, face, tolerance=1.0e-4, silent=True)

@pytest.mark.pythonocc_only
def test_open_edges_of_exact_cylindrical_shell_remain_curved():
    shell = Shell.ByWires(
        [_circle_wire(0.0, 1.0), _circle_wire(2.0, 1.0)],
        polyhedron=False,
        silent=True,
    )
    assert Topology.IsInstance(shell, "Shell")

    open_edges = Topology.OpenEdges(shell, silent=True)
    assert isinstance(open_edges, list)
    assert len(open_edges) == 2
    assert all(Edge.IsLinear(edge, silent=True) is False for edge in open_edges)

@pytest.mark.pythonocc_only
def test_external_boundary_of_circular_face_preserves_circle():
    face = Face.ByWire(_circle_wire(0.0, 2.0), silent=True)
    assert Topology.IsInstance(face, "Face")

    boundary = Topology.ExternalBoundary(face, silent=True)
    assert Topology.IsInstance(boundary, "Wire")
    edges = Wire.Edges(boundary, silent=True)
    assert isinstance(edges, list) and len(edges) == 1
    assert Edge.IsClosed(edges[0], silent=True)
    assert Edge.IsLinear(edges[0], silent=True) is False
    assert math.isclose(
        Edge.Length(edges[0], mantissa=None, silent=True),
        4.0 * math.pi,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

@pytest.mark.pythonocc_only
def test_shortest_edge_to_circle_uses_actual_curve_not_topological_vertices():
    center = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    circle = Edge.Circle(radius=2.5, silent=True)
    assert Topology.IsInstance(circle, "Edge")

    shortest = Topology.ShortestEdge(center, circle, silent=True)
    assert Topology.IsInstance(shortest, "Edge")
    assert math.isclose(
        Edge.Length(shortest, mantissa=None, silent=True),
        2.5,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

@pytest.mark.parametrize(
    "source_name, requested_type, expected_count",
    QUERY_SUBTOPOLOGY_CASES,
)
def test_subtopologies(source_name, requested_type, expected_count):
    """Test the number and type returned by Topology.SubTopologies."""
    model = _query_model()
    source = model[source_name]

    result = Topology.SubTopologies(
        source,
        subTopologyType=requested_type,
        silent=True,
    )

    _assert_query_topology_list(
        result,
        expected_count,
        requested_type,
        (
            "Topology.SubTopologies"
            f"({source_name}, {requested_type})"
        ),
    )

@pytest.mark.parametrize(
    "source_name, requested_type, expected_count, expected_type",
    QUERY_SUPERTOPOLOGY_CASES,
)
def test_supertopologies(
    source_name,
    requested_type,
    expected_count,
    expected_type,
):
    """Test the number and type returned by Topology.SuperTopologies."""
    model = _query_model()
    source = model[source_name]
    host = model["host"]

    result = Topology.SuperTopologies(
        source,
        hostTopology=host,
        topologyType=requested_type,
        silent=True,
    )

    requested_label = requested_type if requested_type is not None else "inferred"
    _assert_query_topology_list(
        result,
        expected_count,
        expected_type,
        (
            "Topology.SuperTopologies"
            f"({source_name}, {requested_label})"
        ),
    )

@pytest.mark.parametrize(
    "source_name, requested_type, expected_count",
    QUERY_ADJACENCY_CASES,
)
def test_adjacent_topologies(
    source_name,
    requested_type,
    expected_count,
):
    """Test the number and type returned by Topology.AdjacentTopologies."""
    model = _query_model()
    source = model[source_name]
    host = model["host"]

    result = Topology.AdjacentTopologies(
        source,
        hostTopology=host,
        topologyType=requested_type,
        silent=True,
    )

    _assert_query_topology_list(
        result,
        expected_count,
        requested_type,
        (
            "Topology.AdjacentTopologies"
            f"({source_name}, {requested_type})"
        ),
    )

@pytest.mark.parametrize(
    "topology_a_name, topology_b_name, expected_counts",
    QUERY_SHARED_CASES,
)
def test_shared_topologies(
    topology_a_name,
    topology_b_name,
    expected_counts,
):
    """Test Topology.SharedTopologies and its four typed convenience methods."""
    model = _query_model()
    topology_a = model[topology_a_name]
    topology_b = model[topology_b_name]

    result = Topology.SharedTopologies(
        topology_a,
        topology_b,
        silent=True,
    )

    assert isinstance(result, dict), (
        "Topology.SharedTopologies did not return a dictionary. "
        f"Returned Python type: {type(result).__name__}."
    )
    assert set(result.keys()) == set(QUERY_SHARED_KEYS), (
        "Topology.SharedTopologies returned unexpected dictionary keys. "
        f"Expected {QUERY_SHARED_KEYS}; returned {tuple(result.keys())}."
    )

    for key in QUERY_SHARED_KEYS:
        _assert_query_topology_list(
            result[key],
            expected_counts[key],
            QUERY_SHARED_TYPES[key],
            (
                "Topology.SharedTopologies"
                f"({topology_a_name}, {topology_b_name})[{key!r}]"
            ),
        )

    typed_methods = {
        "vertices": Topology.SharedVertices,
        "edges": Topology.SharedEdges,
        "wires": Topology.SharedWires,
        "faces": Topology.SharedFaces,
    }

    for key, method in typed_methods.items():
        typed_result = method(
            topology_a,
            topology_b,
            silent=True,
        )
        _assert_query_topology_list(
            typed_result,
            expected_counts[key],
            QUERY_SHARED_TYPES[key],
            (
                f"{method.__qualname__}"
                f"({topology_a_name}, {topology_b_name})"
            ),
        )

def test_navigation_shared_topologies_use_native_identity():

    cell_complex = CellComplex.Prism(
        origin=Vertex.Origin(),
        width=2.0,
        length=1.0,
        height=1.0,
        uSides=2,
        vSides=1,
        wSides=1,
        placement="center",
        silent=True,
    )

    assert Topology.IsInstance(
        cell_complex,
        "CellComplex",
    )

    cells = Topology.Cells(
        cell_complex,
        silent=True,
    )

    assert len(cells) == 2

    shared = Topology.SharedTopologies(
        cells[0],
        cells[1],
        silent=True,
    )

    assert isinstance(shared, dict)
    assert set(shared.keys()) == {
        "vertices",
        "edges",
        "wires",
        "faces",
    }

    assert len(shared["faces"]) == 1
    assert len(shared["edges"]) == 4
    assert len(shared["vertices"]) == 4

    assert len(
        Topology.SharedFaces(
            cells[0],
            cells[1],
            silent=True,
        )
    ) == 1

    assert len(
        Topology.SharedEdges(
            cells[0],
            cells[1],
            silent=True,
        )
    ) == 4

    assert len(
        Topology.SharedVertices(
            cells[0],
            cells[1],
            silent=True,
        )
    ) == 4

    # Independently constructed but coincident geometry is not shared topology.
    cell_a = Cell.Prism(
        origin=Vertex.Origin(),
        width=1.0,
        length=1.0,
        height=1.0,
        placement="center",
        silent=True,
    )

    cell_b = Cell.Prism(
        origin=Vertex.Origin(),
        width=1.0,
        length=1.0,
        height=1.0,
        placement="center",
        silent=True,
    )

    coincident = Topology.SharedTopologies(
        cell_a,
        cell_b,
        silent=True,
    )

    assert coincident == {
        "vertices": [],
        "edges": [],
        "wires": [],
        "faces": [],
    }

def test_navigation_adjacent_cells_and_face_ancestors():

    cell_complex = CellComplex.Prism(
        origin=Vertex.Origin(),
        width=2.0,
        length=1.0,
        height=1.0,
        uSides=2,
        vSides=1,
        wSides=1,
        placement="center",
        silent=True,
    )

    cells = Topology.Cells(
        cell_complex,
        silent=True,
    )

    assert len(cells) == 2

    adjacent = Topology.AdjacentTopologies(
        cells[0],
        cell_complex,
        topologyType="cell",
        silent=True,
    )

    assert len(adjacent) == 1

    assert Topology.IsSame(
        adjacent[0],
        cells[1],
        silent=True,
    )

    shared_faces = Topology.SharedFaces(
        cells[0],
        cells[1],
        silent=True,
    )

    assert len(shared_faces) == 1

    incident_cells = Topology.AdjacentTopologies(
        shared_faces[0],
        cell_complex,
        topologyType="cell",
        silent=True,
    )

    assert len(incident_cells) == 2

    faces_of_cell = Topology.AdjacentTopologies(
        cells[0],
        cell_complex,
        topologyType="face",
        silent=True,
    )

    assert len(faces_of_cell) == 6

def test_navigation_degree_uses_native_incidence():

    cell_complex = CellComplex.Prism(
        origin=Vertex.Origin(),
        width=2.0,
        length=1.0,
        height=1.0,
        uSides=2,
        vSides=1,
        wSides=1,
        placement="center",
        silent=True,
    )

    cells = Topology.Cells(
        cell_complex,
        silent=True,
    )

    shared_faces = Topology.SharedFaces(
        cells[0],
        cells[1],
        silent=True,
    )

    assert len(shared_faces) == 1

    assert (
        Topology.Degree(
            shared_faces[0],
            cell_complex,
            silent=True,
        )
        == 2
    )

    external_faces = [
        face
        for face in Topology.Faces(
            cell_complex,
            silent=True,
        )
        if Topology.Degree(
            face,
            cell_complex,
            silent=True,
        )
        == 1
    ]

    assert len(external_faces) > 0

def test_navigation_open_topologies_follow_incidence():

    # Open Wire: exactly two end vertices have degree 1.
    v0 = Vertex.ByCoordinates(
        0,
        0,
        0,
    )

    v1 = Vertex.ByCoordinates(
        1,
        0,
        0,
    )

    v2 = Vertex.ByCoordinates(
        2,
        0,
        0,
    )

    e0 = Edge.ByVertices(
        [v0, v1],
        silent=True,
    )

    e1 = Edge.ByVertices(
        [v1, v2],
        silent=True,
    )

    wire = Wire.ByEdges(
        [e0, e1],
        silent=True,
    )

    open_vertices = Topology.OpenVertices(
        wire,
        silent=True,
    )

    assert len(open_vertices) == 2

    # A standalone Face has boundary edges incident to only one Face.
    face = Face.Rectangle(
        origin=Vertex.Origin(),
        width=2.0,
        length=2.0,
        placement="center",
        silent=True,
    )

    open_edges = Topology.OpenEdges(
        face,
        silent=True,
    )

    assert len(open_edges) == 4

    # The Face itself is not incident to a Cell.
    open_faces = Topology.OpenFaces(
        face,
        silent=True,
    )

    assert len(open_faces) == 1
    assert Topology.IsSame(
        open_faces[0],
        face,
        silent=True,
    )

def test_navigation_select_subtopology():

    face = Face.Rectangle(
        origin=Vertex.Origin(),
        width=2.0,
        length=2.0,
        placement="center",
        silent=True,
    )

    vertices = Topology.Vertices(
        face,
        silent=True,
    )

    assert len(vertices) == 4

    selected_vertex = Topology.SelectSubTopology(
        face,
        vertices[0],
        subTopologyType="vertex",
        silent=True,
    )

    assert Topology.IsInstance(
        selected_vertex,
        "Vertex",
    )

    assert Topology.IsSame(
        selected_vertex,
        vertices[0],
        silent=True,
    )

    selected_face = Topology.SelectSubTopology(
        face,
        Vertex.Origin(),
        subTopologyType="face",
        silent=True,
    )

    assert Topology.IsInstance(
        selected_face,
        "Face",
    )

    assert Topology.IsSame(
        selected_face,
        face,
        silent=True,
    )


# ============================================================================
# Distances, spatial relationships, ordering, and extrema
# ============================================================================

def test_shortest_edge_shortest_distance_and_shortest_edges_for_vertices():
    a = _v(0, 0, 0)
    b = _v(3, 4, 0)
    wire = Wire.Rectangle(width=2, length=1, silent=True)

    shortest_edge = Topology.ShortestEdge(a, b, silent=True)
    distance = Topology.ShortestDistance(a, b, silent=True)
    shortest_edges = Topology.ShortestEdges(wire, silent=True)

    assert Topology.IsInstance(shortest_edge, "Edge")
    assert distance == pytest.approx(5, abs=GEOMETRY_TOLERANCE)
    assert isinstance(shortest_edges, list)
    assert len(shortest_edges) >= 1
    assert all(Topology.IsInstance(edge, "Edge") for edge in shortest_edges)

    assert Topology.ShortestEdge(None, b, silent=True) is None
    assert Topology.ShortestDistance(None, b, silent=True) is None
    assert Topology.ShortestEdges(None, silent=True) is None

def test_largest_smallest_longest_shortest_helpers(square_face):
    largest_faces = Topology.LargestFaces(square_face, silent=True)
    smallest_faces = Topology.SmallestFaces(square_face, silent=True)
    longest_edges = Topology.LongestEdges(square_face, silent=True)
    shortest_edges = Topology.ShortestEdges(square_face, silent=True)
    shortest_edge = Topology.ShortestEdge(_v(0, 0, 0), _v(1, 0, 0), silent=True)

    assert isinstance(largest_faces, list)
    assert isinstance(smallest_faces, list)
    assert isinstance(longest_edges, list)
    assert isinstance(shortest_edges, list)
    assert len(largest_faces) >= 1
    assert len(smallest_faces) >= 1
    assert len(longest_edges) >= 1
    assert len(shortest_edges) >= 1
    assert all(Topology.IsInstance(face, "Face") for face in largest_faces)
    assert all(Topology.IsInstance(face, "Face") for face in smallest_faces)
    assert all(Topology.IsInstance(edge, "Edge") for edge in longest_edges)
    assert all(Topology.IsInstance(edge, "Edge") for edge in shortest_edges)
    assert Topology.IsInstance(shortest_edge, "Edge")

def test_sort_by_selectors_orders_topologies_by_selector_location():
    left = Face.Rectangle(origin=_v(0, 0, 0), width=1, length=1, silent=True)
    right = Face.Rectangle(origin=_v(10, 0, 0), width=1, length=1, silent=True)
    selector_left = _v(0, 0, 0)
    selector_right = _v(10, 0, 0)

    result = Topology.SortBySelectors([right, left], [selector_left, selector_right], exclusive=True)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"sorted", "unsorted"}
    assert len(result["sorted"]) == 2
    assert len(result["unsorted"]) == 0
    assert Topology.IsSame(result["sorted"][0], left)
    assert Topology.IsSame(result["sorted"][1], right)

def test_spatial_relationship_wrappers_return_booleans_or_none(square_face):
    same_face = Topology.Copy(square_face)
    far_face = Topology.Translate(square_face, x=10, y=0, z=0, silent=True)

    assert isinstance(Topology.Equals(square_face, same_face, silent=True), bool)
    assert isinstance(Topology.Disjoint(square_face, far_face, silent=True), bool)
    assert isinstance(Topology.Touches(square_face, far_face, silent=True), bool)
    assert isinstance(Topology.Overlaps(square_face, same_face, silent=True), bool)
    assert isinstance(Topology.Crosses(square_face, far_face, silent=True), bool)

    assert Topology.Equals(None, square_face, silent=True) is None
    assert Topology.Disjoint(None, square_face, silent=True) is None


# ============================================================================
# Contents, contexts, apertures, and relationship persistence
# ============================================================================

def _relationship_box():
    """Return a simple 2 x 2 x 2 cell centred at the origin."""
    origin = Vertex.ByCoordinates(0, 0, 0)
    cell = Cell.Box(
        origin=origin,
        width=2,
        length=2,
        height=2,
        placement="center",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")
    return cell

def _relationship_top_face(cell):
    """Return the horizontal face with the greatest centroid Z coordinate."""
    faces = Topology.Faces(cell, silent=True)
    assert isinstance(faces, list)
    assert len(faces) == 6

    return max(
        faces,
        key=lambda face: Vertex.Z(Topology.Centroid(face)),
    )

def _relationship_same_face_from_parent(cell, reference_face):
    """Re-traverse the parent and recover the same underlying face."""
    faces = Topology.Faces(cell, silent=True)
    matches = [
        face
        for face in faces
        if Topology.IsSame(face, reference_face, silent=True)
    ]

    assert len(matches) == 1
    return matches[0]

def _relationship_inset_face(face, scale=0.25, x=0.0, y=0.0):
    """Create a smaller coplanar face inside the input horizontal face."""
    centroid = Topology.Centroid(face)

    inset = Topology.Scale(
        face,
        origin=centroid,
        x=scale,
        y=scale,
        z=1.0,
        transferDictionaries=False,
        silent=True,
    )
    assert Topology.IsInstance(inset, "Face")

    if x != 0.0 or y != 0.0:
        inset = Topology.Translate(
            inset,
            x=x,
            y=y,
            z=0.0,
            transferDictionaries=False,
            silent=True,
        )
        assert Topology.IsInstance(inset, "Face")

    return inset

def _relationship_aperture_type(topology):
    """Return the value of the TopologicPy aperture marker, if present."""
    dictionary = Topology.Dictionary(topology)
    return Dictionary.ValueAtKey(dictionary, "type", None)

def test_add_content_contents_contexts_and_remove_content(square_face):
    content = _v(0, 0, 0)
    with_content = Topology.AddContent(square_face, content)
    contents = Topology.Contents(with_content)

    assert Topology.IsInstance(with_content, "Topology")
    assert isinstance(contents, list)
    assert len(contents) >= 1
    assert Topology.Contexts(contents[0]) is not None

    without_content = Topology.RemoveContent(with_content, contents[0])
    assert Topology.IsInstance(without_content, "Topology")

    assert Topology.AddContent(None, content) is None
    assert Topology.Contents(None) is None
    assert Topology.Contexts(None) is None

def test_aperture_queries_return_empty_lists_when_none_exist(square_face):
    assert Topology.Apertures(square_face, silent=True) == []
    assert Topology.Apertures(square_face, subTopologyType="all", silent=True) == []
    assert Topology.ApertureTopologies(square_face) == []

    assert Topology.Apertures(None, silent=True) is None
    assert Topology.ApertureTopologies(None) is None

def test_add_content_to_face_survives_parent_retraversal():
    """Core regression: content attached to a face must survive a fresh wrapper."""
    cell = _relationship_box()
    face_before = _relationship_top_face(cell)
    content = _relationship_inset_face(face_before)

    returned_cell = Topology.AddContent(
        cell,
        content,
        subTopologyType="face",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    assert Topology.IsSame(returned_cell, cell, silent=True)

    face_after = _relationship_same_face_from_parent(cell, face_before)

    # This is the invariant that the old PythonOCC wrapper-local implementation
    # violated.
    contents = Topology.Contents(face_after, silent=True)

    assert isinstance(contents, list)
    assert len(contents) == 1
    assert Topology.IsInstance(contents[0], "Face")
    assert Face.Area(contents[0]) == pytest.approx(
        Face.Area(content),
        abs=1e-6,
    )

def test_add_aperture_to_self_is_retrievable():
    """An aperture added directly to a topology must be returned by Apertures."""
    host = Face.Rectangle(width=2, length=2, silent=True)
    aperture = Face.Rectangle(width=0.5, length=0.5, silent=True)

    host = Topology.AddApertures(
        host,
        [aperture],
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(host, silent=True)

    assert isinstance(apertures, list)
    assert len(apertures) == 1
    assert Topology.IsInstance(apertures[0], "Face")
    assert str(_relationship_aperture_type(apertures[0])).lower() == "aperture"

def test_add_aperture_to_face_survives_parent_retraversal():
    """Main regression: an aperture on a cell face must survive re-traversal."""
    cell = _relationship_box()
    face_before = _relationship_top_face(cell)
    aperture = _relationship_inset_face(face_before)

    returned_cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    assert Topology.IsSame(returned_cell, cell, silent=True)

    # Force a fresh traversal of the cell faces rather than keeping the wrapper
    # that AddApertures originally modified.
    face_after = _relationship_same_face_from_parent(cell, face_before)

    apertures = Topology.Apertures(face_after, silent=True)

    assert isinstance(apertures, list)
    assert len(apertures) == 1
    assert Topology.IsInstance(apertures[0], "Face")
    assert str(_relationship_aperture_type(apertures[0])).lower() == "aperture"

def test_parent_can_collect_apertures_from_face_subtopologies():
    """The parent-level convenience query must find face-hosted apertures."""
    cell = _relationship_box()
    target_face = _relationship_top_face(cell)
    aperture = _relationship_inset_face(target_face)

    cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert isinstance(apertures, list)
    assert len(apertures) == 1
    assert Topology.IsInstance(apertures[0], "Face")
    assert str(_relationship_aperture_type(apertures[0])).lower() == "aperture"

def test_two_apertures_on_same_face_when_not_exclusive():
    """exclusive=False must allow multiple apertures on one subtopology."""
    cell = _relationship_box()
    target_face = _relationship_top_face(cell)

    aperture_a = _relationship_inset_face(target_face, scale=0.20, x=-0.45)
    aperture_b = _relationship_inset_face(target_face, scale=0.20, x=0.45)

    cell = Topology.AddApertures(
        cell,
        [aperture_a, aperture_b],
        exclusive=False,
        subTopologyType="face",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert isinstance(apertures, list)
    assert len(apertures) == 2
    assert all(
        str(_relationship_aperture_type(aperture)).lower() == "aperture"
        for aperture in apertures
    )

def test_exclusive_allows_only_one_aperture_per_face():
    """exclusive=True must identify a face by topology identity, not wrapper id."""
    cell = _relationship_box()
    target_face = _relationship_top_face(cell)

    aperture_a = _relationship_inset_face(target_face, scale=0.20, x=-0.45)
    aperture_b = _relationship_inset_face(target_face, scale=0.20, x=0.45)

    cell = Topology.AddApertures(
        cell,
        [aperture_a, aperture_b],
        exclusive=True,
        subTopologyType="face",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert isinstance(apertures, list)
    assert len(apertures) == 1

def test_aperture_is_not_attached_to_unrelated_faces():
    """Only the geometrically matching face should receive the aperture."""
    cell = _relationship_box()
    target_face = _relationship_top_face(cell)
    aperture = _relationship_inset_face(target_face)

    cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    faces = Topology.Faces(cell, silent=True)
    counts = [
        len(Topology.Apertures(face, silent=True))
        for face in faces
    ]

    assert sorted(counts) == [0, 0, 0, 0, 0, 1]

def test_repeated_aperture_queries_do_not_duplicate_relationships():
    """Hydrating a fresh wrapper repeatedly must not create duplicate links."""
    cell = _relationship_box()
    target_face = _relationship_top_face(cell)
    aperture = _relationship_inset_face(target_face)

    cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=RELATIONSHIP_TOLERANCE,
        silent=True,
    )

    first = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )
    second = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )
    third = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert len(first) == 1
    assert len(second) == 1
    assert len(third) == 1


# ============================================================================
# BREP, JSON, XYZ, OBJ, STEP, and TPY persistence
# ============================================================================

def _set_python_dictionary(topology, values):
    dictionary = Dictionary.ByPythonDictionary(
        values,
        silent=True,
    )

    return Topology.SetDictionary(
        topology,
        dictionary,
        silent=True,
    )

def _python_dictionary(topology):
    dictionary = Topology.Dictionary(
        topology,
        silent=True,
    )

    return Dictionary.PythonDictionary(
        dictionary,
        silent=True,
    ) or {}

def test_brep_string_round_trip_for_face(square_face):
    brep = Topology.BREPString(square_face)
    assert isinstance(brep, str)
    assert len(brep) > 0

    rebuilt = Topology.ByBREPString(brep, silent=True)
    _assert_topology(rebuilt)
    assert Topology.TypeAsString(rebuilt, silent=True).lower() == "face"

    assert Topology.BREPString(None) is None
    assert Topology.ByBREPString(None, silent=True) is None

@pytest.mark.pythonocc_only
def test_brep_string_is_raw_occt_brep():
    cell = Cell.Prism()
    brep = Topology.BREPString(cell)

    assert isinstance(brep, str)
    assert brep.strip()
    assert not brep.lstrip().startswith("{")
    assert brep.lstrip().startswith("DBRep_DrawableShape")

    rebuilt = Topology.ByBREPString(brep)
    assert Topology.IsInstance(rebuilt, "Cell")

@pytest.mark.pythonocc_only
def test_export_to_brep_writes_raw_occt_brep(tmp_path):
    cell = Cell.Prism()
    path = tmp_path / "cell.brep"

    result = Topology.ExportToBREP(cell, str(path), overwrite=True)
    assert result is True

    text = path.read_text(encoding="utf-8")
    assert text.strip()
    assert not text.lstrip().startswith("{")
    assert text.lstrip().startswith("DBRep_DrawableShape")

def test_json_string_round_trip_for_vertex():
    vertex = Topology.SetDictionary(_v(1, 2, 3), Dictionary.ByPythonDictionary({"name": "json-vertex"}), silent=True)
    json_string = Topology.JSONString(vertex)

    assert isinstance(json_string, str)
    loaded = json.loads(json_string)
    assert isinstance(loaded, list)
    assert any(record.get("type") == "Vertex" for record in loaded)

    rebuilt = Topology.ByJSONString(json_string, silent=True)
    assert isinstance(rebuilt, list)
    assert len(rebuilt) >= 1
    assert any(Topology.IsInstance(item, "Vertex") for item in rebuilt)

    assert Topology.ByJSONString("not json", silent=True) is None

def test_xyz_path_imports_frames(tmp_path):
    xyz_path = tmp_path / "points.xyz"
    xyz_path.write_text("3\nFrame 1\nA 0 0 0\nB 1 0 0\nC 0 1 0\n", encoding="utf-8")

    frames = Topology.ByXYZPath(str(xyz_path))

    assert isinstance(frames, list)
    assert len(frames) == 1
    assert Topology.IsInstance(frames[0], "Cluster")
    assert len(Topology.Vertices(frames[0], silent=True)) == 3

    assert Topology.ByXYZPath(None) is None
    assert Topology.ByXYZPath(str(tmp_path / "missing.xyz")) is None

def test_export_to_obj_and_brep_create_files(tmp_path, square_face):
    brep_path = tmp_path / "face.brep"
    obj_path = tmp_path / "face.obj"

    brep_status = Topology.ExportToBREP(square_face, str(brep_path), overwrite=True)
    try:
        obj_status = Topology.ExportToOBJ(square_face, path=str(obj_path), overwrite=True, silent=True)
    except Exception:
        obj_status = None

    assert brep_status is True
    assert brep_path.exists()
    assert brep_path.stat().st_size > 0

    # OBJ export depends on mesh/material support that is optional across the
    # public backend contract. When supported it must create a non-empty file.
    if obj_status is True:
        assert obj_path.exists()
        assert obj_path.stat().st_size > 0
    else:
        assert obj_status in (None, False)

    assert Topology.ExportToBREP(None, str(tmp_path / "bad.brep"), overwrite=True) is None
    try:
        bad_obj_status = Topology.ExportToOBJ(None, path=str(tmp_path / "bad.obj"), overwrite=True, silent=True)
    except Exception:
        bad_obj_status = None
    assert bad_obj_status is None

def test_json_export_and_import_path_round_trip(tmp_path):
    vertex = _v(1, 2, 3)
    path = tmp_path / "vertex.json"

    status = Topology.ExportToJSON(vertex, str(path), overwrite=True)
    imported = Topology.ByJSONPath(str(path), silent=True)

    assert status is True
    assert path.exists()
    assert isinstance(imported, list)
    assert len(imported) >= 1
    assert any(Topology.IsInstance(item, "Vertex") for item in imported)

    assert Topology.ByJSONPath(None, silent=True) is None

def test_step_entry_points_validate_without_throwing(tmp_path):
    bad = tmp_path / "bad.xyz"

    assert (
        Topology.Save(
            None,
            bad,
            silent=True,
        )
        is False
    )

    assert (
        Topology.Load(
            bad,
            silent=True,
        )
        is None
    )

    assert (
        Topology.ExportToSTEP(
            None,
            tmp_path / "bad.step",
            silent=True,
        )
        is False
    )

    assert (
        Topology.BySTEPPath(
            tmp_path / "missing.step",
            silent=True,
        )
        is None
    )

def test_generic_save_rejects_unregistered_extension(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=3.0,
        silent=True,
    )

    assert (
        Topology.Save(
            face,
            tmp_path / "face.xyz",
            silent=True,
        )
        is False
    )

@pytest.mark.pythonocc_only
def test_step_roundtrip_preserves_exact_arc_geometry(tmp_path):
    edge = Edge.Arc(
        radius=3.0,
        fromAngle=15.0,
        toAngle=145.0,
        silent=True,
    )

    assert Topology.IsInstance(
        edge,
        "Edge",
    )

    expected_length = Edge.Length(
        edge,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "arc.step"

    assert Topology.ExportToSTEP(
        edge,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.BySTEPPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Edge",
    )

    assert (
        Edge.IsLinear(
            result,
            silent=True,
        )
        is False
    )

    assert math.isclose(
        Edge.Length(
            result,
            mantissa=None,
            silent=True,
        ),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )

@pytest.mark.pythonocc_only
def test_step_roundtrip_preserves_nurbs_surface(tmp_path):
    # Exact rational quarter-cylinder patch.
    w = 1.0 / math.sqrt(2.0)

    control_points = [
        [
            Vertex.ByCoordinates(1.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 1.0, 0.0),
            Vertex.ByCoordinates(0.0, 1.0, 0.0),
        ],
        [
            Vertex.ByCoordinates(1.0, 0.0, 2.0),
            Vertex.ByCoordinates(1.0, 1.0, 2.0),
            Vertex.ByCoordinates(0.0, 1.0, 2.0),
        ],
    ]

    weights = [
        [1.0, w, 1.0],
        [1.0, w, 1.0],
    ]

    face = Face.ByNurbsParameters(
        controlPoints=control_points,
        weights=weights,
        uKnots=[0.0, 0.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        isRational=True,
        uDegree=1,
        vDegree=2,
        silent=True,
    )

    assert Topology.IsInstance(
        face,
        "Face",
    )

    assert (
        Face.IsPlanar(
            face,
            silent=True,
        )
        is False
    )

    expected_area = Face.Area(
        face,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "surface.step"

    assert Topology.ExportToSTEP(
        face,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.BySTEPPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Face",
    )

    assert (
        Face.IsPlanar(
            result,
            silent=True,
        )
        is False
    )

    assert math.isclose(
        Face.Area(
            result,
            mantissa=None,
            silent=True,
        ),
        expected_area,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

@pytest.mark.pythonocc_only
def test_generic_save_load_routes_step_codec(tmp_path):
    cell = Cell.Cylinder(
        radius=1.25,
        height=3.0,
        uSides=24,
        vSides=1,
        polyhedron=False,
        silent=True,
    )

    assert Topology.IsInstance(
        cell,
        "Cell",
    )

    path = tmp_path / "cylinder.stp"

    assert Topology.Save(
        cell,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.Load(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Cell",
    )

    curved_faces = [
        face
        for face in (
            Topology.Faces(
                result,
                silent=True,
            )
            or []
        )
        if not Face.IsPlanar(
            face,
            silent=True,
        )
    ]

    assert len(curved_faces) >= 1

@pytest.mark.pythonocc_only
def test_step_overwrite_contract(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=3.0,
        silent=True,
    )

    path = tmp_path / "overwrite.step"

    assert Topology.ExportToSTEP(
        face,
        path,
        overwrite=False,
        silent=True,
    )

    assert (
        Topology.ExportToSTEP(
            face,
            path,
            overwrite=False,
            silent=True,
        )
        is False
    )

    assert Topology.ExportToSTEP(
        face,
        path,
        overwrite=True,
        silent=True,
    )

def test_tpy_parent_dictionary_roundtrip(tmp_path):
    face = Face.Rectangle(
        width=4.0,
        length=3.0,
        silent=True,
    )

    face = _set_python_dictionary(
        face,
        {
            "name": "Room A",
            "number": 17,
            "active": True,
            "values": [1, 2.5, "x"],
        },
    )

    # Capture TopologicPy's actual stored dictionary representation before
    # persistence. Some backends represent Python bool values as integer 0/1.
    expected_dictionary = _python_dictionary(
        face
    )

    path = tmp_path / "parent.tpy"

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Face",
    )

    dictionary = _python_dictionary(
        result
    )

    assert dictionary == expected_dictionary
    assert dictionary["name"] == "Room A"
    assert dictionary["number"] == 17
    assert dictionary["values"] == [1, 2.5, "x"]

def test_generic_save_load_routes_tpy_codec(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    face = _set_python_dictionary(
        face,
        {
            "id": "generic-save",
        },
    )

    path = tmp_path / "generic.tpy"

    assert Topology.Save(
        face,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.Load(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Face",
    )

    assert (
        _python_dictionary(
            result
        ).get("id")
        == "generic-save"
    )

def test_tpy_subtopology_dictionary_roundtrip(tmp_path):
    cell = Cell.Box(
        width=4.0,
        length=3.0,
        height=2.0,
        silent=True,
    )

    faces = Topology.Faces(
        cell,
        silent=True,
    ) or []

    edges = Topology.Edges(
        cell,
        silent=True,
    ) or []

    assert len(faces) >= 1
    assert len(edges) >= 1

    _set_python_dictionary(
        faces[0],
        {
            "saved_face": "face-0",
        },
    )

    _set_python_dictionary(
        edges[0],
        {
            "saved_edge": "edge-0",
        },
    )

    path = tmp_path / "subtopologies.tpy"

    assert Topology.ExportToTPY(
        cell,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Cell",
    )

    loaded_faces = Topology.Faces(
        result,
        silent=True,
    ) or []

    loaded_edges = Topology.Edges(
        result,
        silent=True,
    ) or []

    assert any(
        _python_dictionary(face).get(
            "saved_face"
        )
        == "face-0"
        for face in loaded_faces
    )

    assert any(
        _python_dictionary(edge).get(
            "saved_edge"
        )
        == "edge-0"
        for edge in loaded_edges
    )

def test_tpy_content_relationship_roundtrip(tmp_path):
    host = Face.Rectangle(
        width=6.0,
        length=4.0,
        silent=True,
    )

    content = Edge.ByVertices(
        [
            Vertex.ByCoordinates(
                -1.0,
                0.0,
                0.0,
            ),
            Vertex.ByCoordinates(
                1.0,
                0.0,
                0.0,
            ),
        ],
        silent=True,
    )

    content = _set_python_dictionary(
        content,
        {
            "role": "content",
        },
    )

    host = Topology.AddContent(
        host,
        content,
        subTopologyType="self",
        silent=True,
    )

    contents = Topology.Contents(
        host,
        silent=True,
    ) or []

    assert len(contents) == 1

    path = tmp_path / "contents.tpy"

    assert Topology.ExportToTPY(
        host,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    loaded_contents = Topology.Contents(
        result,
        silent=True,
    ) or []

    assert len(loaded_contents) == 1

    assert (
        _python_dictionary(
            loaded_contents[0]
        ).get("role")
        == "content"
    )

    contexts = Topology.Contexts(
        loaded_contents[0],
        silent=True,
    ) or []

    assert len(contexts) >= 1

def test_tpy_aperture_relationship_roundtrip(tmp_path):
    host = Face.Rectangle(
        width=6.0,
        length=4.0,
        silent=True,
    )

    aperture_topology = Face.Rectangle(
        origin=Vertex.ByCoordinates(
            0.0,
            0.0,
            0.0,
        ),
        width=1.0,
        length=1.0,
        silent=True,
    )

    aperture_topology = _set_python_dictionary(
        aperture_topology,
        {
            "role": "aperture",
        },
    )

    context = Context.ByTopologyParameters(
        host,
        u=0.25,
        v=0.75,
        w=0.5,
    )

    aperture = Aperture.ByTopologyContext(
        aperture_topology,
        context,
    )

    assert aperture is not None

    assert len(
        Topology.Apertures(
            host,
            silent=True,
        )
        or []
    ) >= 1

    path = tmp_path / "aperture.tpy"

    assert Topology.ExportToTPY(
        host,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    apertures = Topology.Apertures(
        result,
        silent=True,
    ) or []

    assert len(apertures) >= 1

    assert any(
        _python_dictionary(aperture).get(
            "role"
        )
        == "aperture"
        for aperture in apertures
    )

@pytest.mark.pythonocc_only
def test_tpy_exact_arc_roundtrip(tmp_path):
    arc = Edge.Arc(
        radius=3.0,
        fromAngle=20.0,
        toAngle=160.0,
        silent=True,
    )

    expected_length = Edge.Length(
        arc,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "arc.tpy"

    assert Topology.ExportToTPY(
        arc,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Edge",
    )

    assert (
        Edge.IsLinear(
            result,
            silent=True,
        )
        is False
    )

    assert math.isclose(
        Edge.Length(
            result,
            mantissa=None,
            silent=True,
        ),
        expected_length,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )

@pytest.mark.pythonocc_only
def test_tpy_exact_nurbs_face_roundtrip(tmp_path):
    w = 1.0 / math.sqrt(2.0)

    control_points = [
        [
            Vertex.ByCoordinates(
                1.0,
                0.0,
                0.0,
            ),
            Vertex.ByCoordinates(
                1.0,
                1.0,
                0.0,
            ),
            Vertex.ByCoordinates(
                0.0,
                1.0,
                0.0,
            ),
        ],
        [
            Vertex.ByCoordinates(
                1.0,
                0.0,
                2.0,
            ),
            Vertex.ByCoordinates(
                1.0,
                1.0,
                2.0,
            ),
            Vertex.ByCoordinates(
                0.0,
                1.0,
                2.0,
            ),
        ],
    ]

    face = Face.ByNurbsParameters(
        controlPoints=control_points,
        weights=[
            [1.0, w, 1.0],
            [1.0, w, 1.0],
        ],
        uKnots=[
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        vKnots=[
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
        ],
        isRational=True,
        uDegree=1,
        vDegree=2,
        silent=True,
    )

    expected_area = Face.Area(
        face,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "nurbs.tpy"

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Face",
    )

    assert (
        Face.IsPlanar(
            result,
            silent=True,
        )
        is False
    )

    assert math.isclose(
        Face.Area(
            result,
            mantissa=None,
            silent=True,
        ),
        expected_area,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )

def test_tpy_shapeless_cluster_roundtrip(tmp_path):
    edge_a = Edge.ByVertices(
        [
            Vertex.ByCoordinates(
                0,
                0,
                0,
            ),
            Vertex.ByCoordinates(
                1,
                0,
                0,
            ),
        ],
        silent=True,
    )

    edge_b = Edge.ByVertices(
        [
            Vertex.ByCoordinates(
                0,
                1,
                0,
            ),
            Vertex.ByCoordinates(
                1,
                1,
                0,
            ),
        ],
        silent=True,
    )

    cluster = Cluster.ByTopologies(
        [
            edge_a,
            edge_b,
        ],
        silent=True,
    )

    assert Topology.IsInstance(
        cluster,
        "Cluster",
    )

    path = tmp_path / "cluster.tpy"

    assert Topology.ExportToTPY(
        cluster,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Cluster",
    )

    assert len(
        Topology.Edges(
            result,
            silent=True,
        )
        or []
    ) == 2

def test_tpy_overwrite_contract(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    path = tmp_path / "overwrite.tpy"

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=False,
        silent=True,
    )

    assert (
        Topology.ExportToTPY(
            face,
            path,
            overwrite=False,
            silent=True,
        )
        is False
    )

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=True,
        silent=True,
    )

def test_tpy_corrupt_checksum_is_rejected(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    good = tmp_path / "good.tpy"
    bad = tmp_path / "bad.tpy"

    assert Topology.ExportToTPY(
        face,
        good,
        overwrite=True,
        silent=True,
    )

    with zipfile.ZipFile(
        good,
        "r",
    ) as source:
        manifest = json.loads(
            source.read(
                "manifest.json"
            ).decode("utf-8")
        )

        geometry_path = (
            manifest["objects"][0][
                "geometry"
            ]["path"]
        )

        members = {
            name: source.read(name)
            for name in source.namelist()
        }

    members[
        geometry_path
    ] = (
        members[geometry_path]
        + b"\n# corrupted\n"
    )

    with zipfile.ZipFile(
        bad,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as target:
        for name, data in members.items():
            target.writestr(
                name,
                data,
            )

    assert (
        Topology.ByTPYPath(
            bad,
            silent=True,
        )
        is None
    )


# Four cube arrangements x eight Booleans: independent type/measure contracts.
import _boolean_cube_matrix as _cube_matrix


def _assert_cube_matrix_geometry(result, expected):
    expected_type, dimension, measure = expected
    if expected_type is None:
        assert result is None, "Expected a certified empty geometry result"
        return
    assert result is not None
    assert Topology.TypeAsString(result) == expected_type
    if dimension == 3:
        cells = _cube_matrix.members(result, "Cell")
        assert cells, "Expected 3D material"
        actual = sum(Cell.Volume(cell, mantissa=12, silent=True) for cell in cells)
        assert actual == pytest.approx(measure, rel=1e-6, abs=1e-9)
    elif dimension == 2:
        faces = _cube_matrix.members(result, "Face")
        assert faces and not _cube_matrix.members(result, "Cell")
        actual = sum(Face.Area(face, mantissa=12, silent=True) for face in faces)
        assert actual == pytest.approx(measure, rel=1e-6, abs=1e-9)
    elif dimension == 1:
        assert Topology.IsInstance(result, "Edge")
        assert Edge.Length(result, mantissa=12) == pytest.approx(measure, rel=1e-6, abs=1e-9)
    elif dimension == 0:
        assert Topology.IsInstance(result, "Vertex")
        assert Vertex.Coordinates(result, mantissa=12) == pytest.approx(measure, abs=1e-6)
    else:
        raise AssertionError("Unknown expected dimension")


@pytest.mark.pythonocc_only
@pytest.mark.parametrize("state,operation", _cube_matrix.CASES, ids=_cube_matrix.CASE_IDS)
def test_boolean_cube_matrix_type_and_measure(state, operation):
    a, b = _cube_matrix.make_inputs(state)
    result = _cube_matrix.run_operation(a, b, operation)
    _assert_cube_matrix_geometry(result, _cube_matrix.EXPECTED_GEOMETRY[state][operation])


@pytest.mark.pythonocc_only
@pytest.mark.parametrize("origin,expected", [
    ((1, 1, 0), ("Edge", 1, 1)),
    ((1, 1, 1), ("Vertex", 0, (1, 1, 1))),
    ((0, 0, 0), ("Cell", 3, 1)),
], ids=["edge-contact", "vertex-contact", "identical-cubes"])
def test_intersect_cube_contact_dimensions_and_measure(origin, expected):
    result = Topology.Intersect(_cube_matrix.cube(), _cube_matrix.cube(origin),
                                tolerance=_cube_matrix.TOLERANCE, silent=True)
    _assert_cube_matrix_geometry(result, expected)


@pytest.mark.pythonocc_only
def test_intersect_c_shape_returns_two_face_cluster_with_expected_area():
    a = Cell.CShape(origin=Vertex.ByCoordinates(0, 0, 0), placement="bottom")
    b = Cell.Prism(origin=Vertex.ByCoordinates(1, 0, 0), placement="bottom")
    result = Topology.Intersect(a, b, tolerance=1e-6, silent=True)
    _assert_cube_matrix_geometry(result, ("Cluster", 2, 0.5))
    children = Cluster.Topologies(result)
    assert len(children) == 2
    assert all(Topology.IsInstance(child, "Face") for child in children)
