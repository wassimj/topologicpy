# Copyright (C) 2026
# TopologicPy Topology unit tests.

import json
import uuid

import pytest


pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("tqdm")

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Cell = pytest.importorskip("topologicpy.Cell").Cell
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


def _v(x, y, z=0):
    return Vertex.ByCoordinates(x, y, z)


def _coords(vertex, mantissa=6):
    return Vertex.Coordinates(vertex, mantissa=mantissa)


def _assert_coords(vertex, expected, abs_tol=TOLERANCE, mantissa=6):
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


def test_brep_string_round_trip_for_face(square_face):
    brep = Topology.BREPString(square_face)
    assert isinstance(brep, str)
    assert len(brep) > 0

    rebuilt = Topology.ByBREPString(brep, silent=True)
    _assert_topology(rebuilt)
    assert Topology.TypeAsString(rebuilt, silent=True).lower() == "face"

    assert Topology.BREPString(None) is None
    assert Topology.ByBREPString(None, silent=True) is None


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


def test_copy_preserves_type_and_geometry(square_face):
    copied = Topology.Copy(square_face)

    assert Topology.IsInstance(copied, "Face")
    assert copied is not square_face
    assert len(Topology.Vertices(copied, silent=True)) == len(Topology.Vertices(square_face, silent=True))
    assert Topology.Copy(None) is None


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


def test_self_merge_and_merge_all_return_valid_topologies():
    edge_a = Edge.ByVertices([_v(0, 0, 0), _v(1, 0, 0)], silent=True)
    edge_b = Edge.ByVertices([_v(1, 0, 0), _v(2, 0, 0)], silent=True)
    cluster = Cluster.ByTopologies([edge_a, edge_b], silent=True)

    merged = Topology.SelfMerge(cluster, silent=True)
    merged_all = Topology.MergeAll([edge_a, edge_b], silent=True)

    _assert_topology(merged)
    _assert_topology(merged_all)
    assert Topology.SelfMerge(None, silent=True) is None
    assert Topology.MergeAll(None, silent=True) is None


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


def test_remove_collinear_coplanar_edges_faces_and_cleanup(square_face, simple_cell):
    cleaned_face = Topology.RemoveCollinearEdges(square_face, silent=True)
    clean_cell = Topology.RemoveCoplanarFaces(simple_cell, silent=True)
    fixed = Topology.Fix(square_face, topologyType="face")
    cleaned = Topology.Cleanup(Topology.Copy(square_face))

    _assert_topology(cleaned_face)
    _assert_topology(clean_cell)
    assert fixed is None or Topology.IsInstance(fixed, "Topology")
    assert cleaned is None or Topology.IsInstance(cleaned, "Topology")

    assert Topology.RemoveCollinearEdges(None, silent=True) is None
    assert Topology.RemoveCoplanarFaces(None, silent=True) is None
    assert Topology.Cleanup("not-a-topology") is None


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


def test_shortest_edge_shortest_distance_and_shortest_edges_for_vertices():
    a = _v(0, 0, 0)
    b = _v(3, 4, 0)
    wire = Wire.Rectangle(width=2, length=1, silent=True)

    shortest_edge = Topology.ShortestEdge(a, b, silent=True)
    distance = Topology.ShortestDistance(a, b, silent=True)
    shortest_edges = Topology.ShortestEdges(wire, silent=True)

    assert Topology.IsInstance(shortest_edge, "Edge")
    assert distance == pytest.approx(5, abs=TOLERANCE)
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

def test_navigation_shared_topologies_use_native_identity():
    from topologicpy.Cell import Cell
    from topologicpy.CellComplex import CellComplex
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

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
    from topologicpy.CellComplex import CellComplex
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

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
    from topologicpy.CellComplex import CellComplex
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

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
    from topologicpy.Edge import Edge
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    from topologicpy.Wire import Wire

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
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

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

def test_remove_faces_preserves_nurbs_surface():
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_BSplineSurface

    from topologicpy.Cluster import Cluster
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

    z = [
        [0, 0, 0, 0],
        [0, 4, 2, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 0],
    ]

    control_points = []

    for i in range(4):
        row = []

        for j in range(4):
            row.append(
                Vertex.ByCoordinates(
                    i,
                    j,
                    z[i][j],
                )
            )

        control_points.append(row)

    nurbs = Face.ByNurbsParameters(
        controlPoints=control_points,
        uDegree=3,
        vDegree=3,
        silent=True,
    )

    planar = Face.Rectangle(
        origin=Vertex.ByCoordinates(
            10,
            0,
            0,
        ),
        width=1,
        length=1,
        silent=True,
    )

    cluster = Cluster.ByTopologies(
        [nurbs, planar]
    )

    result = Topology.RemoveFaces(
        cluster,
        [planar],
        silent=True,
    )

    faces = Topology.Faces(
        result,
        silent=True,
    )

    assert len(faces) == 1

    shape = Topology.OCCTShape(
        faces[0],
        silent=True,
    )

    adaptor = BRepAdaptor_Surface(shape)

    assert (
        adaptor.GetType()
        == GeomAbs_BSplineSurface
    )

def test_remove_edge_from_face_cascades_to_face_removal():
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

    face = Face.Rectangle(
        origin=Vertex.Origin(),
        width=2,
        length=2,
        silent=True,
    )

    edge = Topology.Edges(
        face,
        silent=True,
    )[0]

    result = Topology.RemoveEdges(
        face,
        [edge],
        silent=True,
    )

    assert result is None

def test_remove_vertex_from_face_cascades_to_face_removal():
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

    face = Face.Rectangle(
        origin=Vertex.Origin(),
        width=2,
        length=2,
        silent=True,
    )

    vertex = Topology.Vertices(
        face,
        silent=True,
    )[0]

    result = Topology.RemoveVertices(
        face,
        [vertex],
        silent=True,
    )

    assert result is None

def test_remove_collinear_edges_native_polyhedron():
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    from topologicpy.Wire import Wire

    wire = Wire.ByVertices(
        [
            Vertex.ByCoordinates(0, 0, 0),
            Vertex.ByCoordinates(1, 0, 0),
            Vertex.ByCoordinates(2, 0, 0),
            Vertex.ByCoordinates(2, 1, 0),
            Vertex.ByCoordinates(0, 1, 0),
        ],
        close=True,
        silent=True,
    )

    assert len(
        Topology.Edges(
            wire,
            silent=True,
        )
    ) == 5

    simplified = (
        Topology.RemoveCollinearEdges(
            wire,
            polyhedron=True,
            silent=True,
        )
    )

    assert Topology.IsInstance(
        simplified,
        "Wire",
    )

    assert len(
        Topology.Edges(
            simplified,
            silent=True,
        )
    ) == 4

def test_remove_coplanar_faces_native_polyhedron():
    from topologicpy.Face import Face
    from topologicpy.Shell import Shell
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

    face_a = Face.Rectangle(
        origin=Vertex.ByCoordinates(
            -0.5,
            0,
            0,
        ),
        width=1,
        length=1,
        silent=True,
    )

    face_b = Face.Rectangle(
        origin=Vertex.ByCoordinates(
            0.5,
            0,
            0,
        ),
        width=1,
        length=1,
        silent=True,
    )

    shell = Shell.ByFaces(
        [face_a, face_b],
        silent=True,
    )

    assert len(
        Topology.Faces(
            shell,
            silent=True,
        )
    ) == 2

    simplified = (
        Topology.RemoveCoplanarFaces(
            shell,
            polyhedron=True,
            silent=True,
        )
    )

    assert len(
        Topology.Faces(
            simplified,
            silent=True,
        )
    ) == 1

