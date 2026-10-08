"""PythonOCC integration regressions for lightweight and nested Clusters."""
import pytest

from topologicpy.Cell import Cell
from topologicpy.Cluster import Cluster
from topologicpy.Dictionary import Dictionary
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


pytestmark = pytest.mark.pythonocc_only


def rectangle(x=0, split=False):
    coords = [(x, 0), (x + 2, 0), (x + 2, 2), (x, 2)]
    if split:
        coords.insert(1, (x + 1, 0))
    return Face.ByWire(Wire.ByVertices([Vertex.ByCoordinates(*p) for p in coords], close=True))


def nested(member):
    return Cluster.ByTopologies([Cluster.ByTopologies([member])])


def cells_of(topology):
    return [topology] if Topology.IsInstance(topology, "Cell") else Topology.Cells(topology)


def faces_of(topology):
    return [topology] if Topology.IsInstance(topology, "Face") else Topology.Faces(topology)


@pytest.mark.parametrize("operation,volume", [
    ("Intersect", 4), ("Difference", 4), ("XOR", 8), ("Union", 12), ("Merge", 12),
])
@pytest.mark.parametrize("wrap_a,wrap_b", [(True, False), (False, True), (True, True)])
def test_boolean_nested_cluster_operands(operation, volume, wrap_a, wrap_b):
    a = Cell.Prism(width=2, length=2, height=2)
    b = Cell.Prism(origin=Vertex.ByCoordinates(1, 0, 0), width=2, length=2, height=2)
    if wrap_a:
        a = nested(a)
    if wrap_b:
        b = nested(b)
    result = getattr(Topology, operation)(a, b)
    assert result is not None
    assert sum(abs(Cell.Volume(cell)) for cell in cells_of(result)) == pytest.approx(volume)


@pytest.mark.parametrize("operation", ["Union", "Merge"])
def test_cluster_boolean_transfers_root_dictionaries(operation):
    a = nested(rectangle())
    b = rectangle(1)
    Topology.SetDictionary(a, Dictionary.ByKeysValues(["left"], ["A"]))
    Topology.SetDictionary(b, Dictionary.ByKeysValues(["right"], ["B"]))
    result = getattr(Topology, operation)(a, b, tranDict=True)
    assert result is not None
    dictionary = Topology.Dictionary(result)
    assert Dictionary.ValueAtKey(dictionary, "left") == "A"
    assert Dictionary.ValueAtKey(dictionary, "right") == "B"


def test_occt_shape_contains_every_nested_member_without_mutation():
    from OCC.Core.TopoDS import TopoDS_Iterator
    face = rectangle()
    edge = Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(10, 0), Vertex.ByCoordinates(12, 0))
    inner = Cluster.ByTopologies([face])
    cluster = Cluster.ByTopologies([inner, edge])
    shape = Topology.OCCTShape(cluster)
    assert shape is not None and not shape.IsNull()
    iterator = TopoDS_Iterator(shape)
    assert iterator.More()
    nested_iterator = TopoDS_Iterator(iterator.Value())
    assert nested_iterator.Value().IsSame(Topology.OCCTShape(face))
    iterator.Next()
    assert iterator.Value().IsSame(Topology.OCCTShape(edge))
    iterator.Next()
    assert not iterator.More()
    assert cluster.shape is None and inner.shape is None
    assert cluster.Topologies() == [inner, edge]


def test_native_bounds_include_all_dimensions_of_nested_cluster():
    cluster = Cluster.ByTopologies([nested(rectangle()), Vertex.ByCoordinates(10, 7, 3)])
    assert cluster.BoundingBoxNative() == pytest.approx([0, 0, 0, 10, 7, 3])
    obb = cluster.BoundingBoxOBBNative()
    assert obb is not None
    for point in ([0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [10, 7, 3]):
        delta = [v - c for v, c in zip(point, obb["center"])]
        for axis, half in zip(("xdir", "ydir", "zdir"), obb["half_sizes"]):
            projection = sum(d * a for d, a in zip(delta, obb[axis]))
            assert abs(projection) <= half + 1e-6


def test_native_cluster_bounds_use_curve_extrema():
    circle = Edge.Circle(radius=2)
    cluster = nested(circle)
    assert cluster.BoundingBoxNative() == pytest.approx([-2, -2, 0, 2, 2, 0], abs=1e-6)


def test_nested_cluster_tessellation_keeps_disconnected_faces():
    cluster = Cluster.ByTopologies([nested(rectangle()), rectangle(10)])
    mesh = Topology.Tessellate(cluster, quality="coarse")
    assert mesh is not None and mesh["faces"]
    assert min(p[0] for p in mesh["vertices"]) == pytest.approx(0)
    assert max(p[0] for p in mesh["vertices"]) == pytest.approx(12)
    area = 0
    for indices in mesh["faces"]:
        a, b, c = [mesh["vertices"][i] for i in indices]
        area += abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / 2
    assert area == pytest.approx(8)


@pytest.mark.parametrize("method,kind", [
    ("RemoveFacesNative", "face"), ("RemoveEdgesNative", "edge"), ("RemoveVerticesNative", "vertex"),
])
def test_native_removal_preserves_unrelated_members_and_root_metadata(method, kind):
    face = rectangle()
    edge = Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(10, 0), Vertex.ByCoordinates(12, 0))
    vertex = Vertex.ByCoordinates(20, 0)
    target = {"face": face, "edge": edge, "vertex": vertex}[kind]
    inner = Cluster.ByTopologies([face, edge, vertex])
    cluster = Cluster.ByTopologies([inner])
    Topology.SetDictionary(cluster, Dictionary.ByKeysValues(["marker"], ["root"]))
    status, result = getattr(cluster, method)([target])
    assert status is True and result is not None
    assert Topology.IsInstance(result, "Cluster")
    surviving_inner = result.Topologies()[0]
    assert Topology.IsInstance(surviving_inner, "Cluster")
    survivors = surviving_inner.Topologies()
    expected = [member for member in [face, edge, vertex] if member is not target]
    assert len(survivors) == len(expected)
    assert all(a is b for a, b in zip(survivors, expected))
    assert Dictionary.ValueAtKey(Topology.Dictionary(result), "marker") == "root"
    assert inner.Topologies() == [face, edge, vertex]


@pytest.mark.parametrize("method", ["RemoveFacesNative", "RemoveEdgesNative", "RemoveVerticesNative"])
def test_native_complete_member_deletion(method):
    member = {"RemoveFacesNative": rectangle,
              "RemoveEdgesNative": lambda: Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(0, 0), Vertex.ByCoordinates(1, 0)),
              "RemoveVerticesNative": lambda: Vertex.ByCoordinates(0, 0)}[method]()
    status, result = getattr(nested(member), method)([member])
    assert status is True and result is None


@pytest.mark.parametrize("method", ["RemoveFaces", "RemoveEdges", "RemoveVertices"])
def test_public_cluster_removal_uses_native_path(method, monkeypatch):
    def forbid_fallback(*args, **kwargs):
        raise AssertionError("Cluster removal entered the legacy reconstruction path")
    monkeypatch.setattr(Topology, f"_Legacy{method}_BackendV1", forbid_fallback)
    target = {"RemoveFaces": rectangle,
              "RemoveEdges": lambda: Edge.ByStartVertexEndVertex(Vertex.ByCoordinates(0, 0), Vertex.ByCoordinates(1, 0)),
              "RemoveVertices": lambda: Vertex.ByCoordinates(0, 0)}[method]()
    retained = Vertex.ByCoordinates(20, 0)
    cluster = Cluster.ByTopologies([nested(target), retained])
    result = getattr(Topology, method)(cluster, [target])
    assert result is not None and Topology.IsInstance(result, "Cluster")
    assert result.Topologies() == [retained]


def test_native_edit_recovers_uncached_compound_members():
    face, vertex = rectangle(), Vertex.ByCoordinates(20, 0)
    cluster = Cluster.ByTopologies([face, vertex])
    imported = type(cluster)(shape=Topology.OCCTShape(cluster), topologies=[])
    status, result = imported.RemoveFacesNative([face])
    assert status is True and result is not None
    assert len(result.Topologies()) == 1
    assert Topology.IsSame(result.Topologies()[0], vertex)


def test_cluster_collinear_cleanup_preserves_free_vertex_and_nested_structure():
    face = rectangle(split=True)
    vertex = Vertex.ByCoordinates(20, 0)
    cluster = Cluster.ByTopologies([nested(face), vertex])
    result = Topology.RemoveCollinearEdges(cluster, polyhedron=False)
    assert result is not None and Topology.IsInstance(result, "Cluster")
    assert result.Topologies()[1] is vertex
    cleaned = result.Topologies()[0].Topologies()[0].Topologies()[0]
    assert len(Topology.Edges(cleaned)) == 4
    assert Face.Area(cleaned) == pytest.approx(4)
    assert len(Topology.Edges(face)) == 5


def test_native_cluster_coplanar_cleanup_preserves_unrelated_vertex():
    shell = Shell.ByFaces([rectangle(), rectangle(2)])
    vertex = Vertex.ByCoordinates(20, 0)
    cluster = Cluster.ByTopologies([nested(shell), vertex])
    status, result = cluster.RemoveCoplanarFacesNative()
    assert status is True and result is not None
    assert result.Topologies()[1] is vertex
    faces = faces_of(result.Topologies()[0])
    assert len(faces) == 1
    assert sum(Face.Area(face) for face in faces) == pytest.approx(8)


def test_public_coplanar_face_soup_retains_sewing_behavior():
    result = Topology.RemoveCoplanarFaces(Cluster.ByTopologies([rectangle(), rectangle(2)]))
    assert result is not None
    faces = faces_of(result)
    assert len(faces) == 1
    assert Face.Area(faces[0]) == pytest.approx(8)


def test_backend_cleanup_reaches_nested_geometry():
    cluster = nested(rectangle(split=True))
    result = cluster.Cleanup()
    assert result is not None and Topology.IsInstance(result, "Cluster")
    faces = faces_of(result)
    assert len(faces) == 1
    assert len(Topology.Edges(faces[0])) == 4
    assert Face.Area(faces[0]) == pytest.approx(4)


def test_cluster_brep_round_trip_preserves_geometry():
    cluster = Cluster.ByTopologies([nested(rectangle()), Vertex.ByCoordinates(10, 7, 3)])
    restored = Topology.ByBREPString(Topology.BREPString(cluster))
    assert restored is not None
    assert restored.BoundingBoxNative() == pytest.approx(cluster.BoundingBoxNative())


def test_curved_member_survives_cluster_simplification():
    face = Face.ByWire(Wire.ByEdges([Edge.Circle(radius=2)]))
    assert face is not None
    before = Face.Area(face, mantissa=None)
    status, result = nested(face).RemoveCollinearEdgesNative(polyhedron=False)
    assert status is True and result is not None
    after = faces_of(result)[0]
    assert Face.Area(after, mantissa=None) == pytest.approx(before, rel=1e-6)
    assert any(Edge.IsLinear(edge) is False for edge in Topology.Edges(after))
