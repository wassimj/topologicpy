"""Native integration tests for public transform/copy provenance.

Run with the PythonOCC backend. Endpoint assertions use native identity, never
geometric matching. These tests intentionally include dictionary-free shapes.
"""
import math

import pytest

from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Core import Core
from topologicpy.Dictionary import Dictionary
from topologicpy.Provenance import Provenance
from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


pytestmark = pytest.mark.skipif(
    type(Core.Backend()).__name__ != "PythonOCCBackend",
    reason="Exact modifier provenance requires PythonOCC",
)


def box():
    return Cell.Prism(width=2, length=3, height=4, silent=True)


def same(a, b):
    return a is not None and b is not None and Topology.IsSame(a, b, silent=True)


def check_correspondence(source, result, provenance, operation):
    assert Topology.IsInstance(result, "Topology")
    assert isinstance(provenance, Provenance)
    assert provenance.supported
    assert provenance.History()
    assert provenance.operation == operation
    assert {r["operation"] for r in provenance.History()} == {operation}
    for kind, getter in [("Vertex", Topology.Vertices), ("Edge", Topology.Edges),
                         ("Face", Topology.Faces), ("Cell", Topology.Cells)]:
        sources = getter(source, silent=True) or []
        targets = getter(result, silent=True) or []
        assert len(sources) == len(targets)
        records = provenance.Records(topologyType=kind)
        for entity in sources:
            successors = [r["result"] for r in records
                          if same(r.get("source"), entity) and r.get("result") is not None]
            unique = []
            for successor in successors:
                if not any(same(successor, t) for t in unique):
                    unique.append(successor)
            assert len(unique) == 1, (operation, kind, "missing or multiple successors")
            assert any(same(unique[0], t) for t in targets)
        for entity in targets:
            origins = [r["source"] for r in records if same(r.get("result"), entity)]
            assert any(same(origin, s) for origin in origins for s in sources)
        if targets:
            graph = provenance.Graph(topologyType=kind)
            assert isinstance(graph, TGraph)
            assert TGraph.Order(graph) > 0
            assert TGraph.Size(graph) > 0
    assert any(same(r.get("source"), source) and same(r.get("result"), result)
               for r in provenance.History()), "Root correspondence is missing"


@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize("operation,kwargs", [
    ("Translate", {"x": 1.5, "y": -2, "z": 3}),
    ("Rotate", {"axis": [0, 0, 1], "angle": 37}),
    ("Scale", {"x": 2, "y": 2, "z": 2}),
    ("Scale", {"x": 2, "y": .5, "z": 1.5}),
    ("Scale", {"x": -1, "y": 2, "z": .5}),
])
def test_transform_correspondence_without_dictionary_dependency(operation, kwargs, transfer):
    source = box()
    result, provenance = getattr(Topology, operation)(
        source, **kwargs, transferDictionaries=transfer,
        silent=True, returnProvenance=True,
    )
    check_correspondence(source, result, provenance, operation)
    if operation == "Scale":
        assert math.isclose(abs(Cell.Volume(result)),
                            abs(Cell.Volume(source) * kwargs["x"] * kwargs["y"] * kwargs["z"]),
                            rel_tol=1e-5)


@pytest.mark.parametrize("deep", [False, True])
def test_copy_is_independent_and_captures_dictionary_free_subshapes(deep):
    source = box()
    result, provenance = Topology.Copy(source, deep=deep, silent=True, returnProvenance=True)
    assert not same(source, result)
    check_correspondence(source, result, provenance, "Copy")


@pytest.mark.parametrize("operation,kwargs", [
    ("Translate", {"x": 0, "y": 0, "z": 0}),
    ("Rotate", {"angle": 0}),
    ("Scale", {"x": 1, "y": 1, "z": 1}),
])
def test_identity_operations_produce_history(operation, kwargs):
    source = box()
    result, provenance = getattr(Topology, operation)(
        source, **kwargs, transferDictionaries=False, silent=True, returnProvenance=True)
    check_correspondence(source, result, provenance, operation)
    if result is source:
        assert {r["relation"] for r in provenance.History()} == {"unchanged"}


@pytest.mark.parametrize("factor", [0, float("nan"), float("inf"), "invalid"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_invalid_scale_is_an_explicit_failure(factor, axis):
    result, provenance = Topology.Scale(box(), **{axis: factor}, silent=True, returnProvenance=True)
    assert result is None
    assert not provenance.supported
    assert provenance.metadata["reason"] == "operation_failed"
    assert not provenance.History()


def test_scaling_about_an_offset_origin():
    source = box()
    origin = Vertex.ByCoordinates(3, -2, 5)
    result, provenance = Topology.Scale(source, origin=origin, x=2, y=3, z=.5,
                                        transferDictionaries=False, silent=True, returnProvenance=True)
    check_correspondence(source, result, provenance, "Scale")
    origin_xyz = Vertex.Coordinates(origin)
    for record in provenance.Records(topologyType="Vertex"):
        if record.get("sourceType") != "Vertex" or record.get("result") is None:
            continue
        a = Vertex.Coordinates(record["source"], mantissa=12)
        b = Vertex.Coordinates(record["result"], mantissa=12)
        expected = [o + s * (v-o) for o, s, v in zip(origin_xyz, [2, 3, .5], a)]
        assert all(math.isclose(x, y, abs_tol=1e-7) for x, y in zip(expected, b))


def test_dictionary_transfer_false_does_not_write_metadata():
    source = box()
    source = Topology.SetDictionary(source, Dictionary.ByKeysValues(["marker"], ["source-only"]))
    face = Topology.Faces(source)[0]
    Topology.SetDictionary(face, Dictionary.ByKeysValues(["face_marker"], ["source-only"]))
    for operation in ["Translate", "Rotate", "Scale"]:
        kwargs = {"Translate": {"x": 1}, "Rotate": {"angle": 25}, "Scale": {"x": 2}}[operation]
        result, provenance = getattr(Topology, operation)(
            source, **kwargs, transferDictionaries=False, returnProvenance=True, silent=True)
        assert provenance.supported
        assert "marker" not in (Dictionary.Keys(Topology.Dictionary(result)) or [])
        for target in Topology.Faces(result):
            assert "face_marker" not in (Dictionary.Keys(Topology.Dictionary(target)) or [])


def test_translate_scale_rotate_boolean_copy_composes():
    source = box()
    moved, p1 = Topology.Translate(source, x=1, transferDictionaries=False, returnProvenance=True)
    scaled, p2 = Topology.Scale(moved, x=2, y=1, z=1, transferDictionaries=False, returnProvenance=True)
    rotated, p3 = Topology.Rotate(scaled, angle=90, transferDictionaries=False, returnProvenance=True)
    # Cut one end; retain nonzero material so the original cell has surviving descendants.
    cutter = Cell.Prism(origin=Vertex.ByCoordinates(0, 5, 0), width=10, length=4, height=10)
    cut, p4 = Topology.Difference(rotated, cutter, returnProvenance=True, silent=True)
    result, p5 = Topology.Copy(cut, deep=False, returnProvenance=True)
    histories = [p1, p2, p3, p4, p5]
    assert all(p.supported for p in histories)
    composed = Provenance.Compose(*histories)
    assert composed.metadata["stageCount"] == 5
    assert any(same(r.get("source"), source) for c in Topology.Cells(result)
               for r in composed.Origins(c))
    assert any(same(r.get("result"), result) for r in composed.Descendants(source))
    graph = composed.Graph(topologyType="Cell")
    assert {e["dictionary"].get("operation") for e in TGraph.Edges(graph)} >= {
        "Translate", "Scale", "Rotate", "Copy"}


def test_inverse_transforms_preserve_derivation_origins():
    source = box()
    a, p1 = Topology.Scale(source, x=2, y=3, z=.5, returnProvenance=True)
    b, p2 = Topology.Scale(a, x=.5, y=1/3, z=2, returnProvenance=True)
    assert math.isclose(Cell.Volume(source), Cell.Volume(b), rel_tol=1e-6)
    assert any(same(r.get("source"), source) for r in Provenance.Compose(p1, p2).Origins(b))


@pytest.mark.parametrize("operation", ["Translate", "Scale", "Rotate", "Copy"])
def test_default_return_remains_a_topology(operation):
    assert Topology.IsInstance(getattr(Topology, operation)(box(), silent=True), "Topology")


@pytest.mark.parametrize("operation", ["Translate", "Scale", "Rotate", "Copy"])
def test_invalid_topology_returns_an_unsupported_pair(operation):
    result, provenance = getattr(Topology, operation)(None, silent=True, returnProvenance=True)
    assert result is None
    assert isinstance(provenance, Provenance)
    assert not provenance.supported


@pytest.mark.parametrize("kind", ["Vertex", "Edge", "Face"])
@pytest.mark.parametrize("operation", ["Translate", "Scale", "Rotate", "Copy"])
def test_lower_dimensional_roots(kind, operation):
    cell = box()
    getter = {"Vertex": Topology.Vertices, "Edge": Topology.Edges, "Face": Topology.Faces}[kind]
    source = getter(cell)[0]
    kwargs = {"Translate": {"x": 1}, "Scale": {"x": 2, "y": 3, "z": .5},
              "Rotate": {"angle": 30}, "Copy": {}}[operation]
    result, provenance = getattr(Topology, operation)(source, **kwargs, returnProvenance=True, silent=True)
    assert Topology.IsInstance(result, kind)
    assert provenance.supported
    assert any(same(r.get("source"), source) and same(r.get("result"), result)
               for r in provenance.History())


@pytest.mark.parametrize("operation", ["Translate", "Scale", "Rotate", "Copy"])
def test_cluster_root_and_member_correspondence(operation):
    a, b = box(), Cell.Prism(origin=Vertex.ByCoordinates(8, 0, 0))
    source = Cluster.ByTopologies([a, b])
    kwargs = {"Translate": {"x": 1}, "Scale": {"x": 2},
              "Rotate": {"angle": 30}, "Copy": {}}[operation]
    if operation != "Copy":
        kwargs["transferDictionaries"] = False
    result, provenance = getattr(Topology, operation)(source, **kwargs, returnProvenance=True, silent=True)
    assert Topology.IsInstance(result, "Cluster")
    assert provenance.supported
    assert len(Topology.Cells(result)) == 2
    assert any(same(r.get("source"), source) and same(r.get("result"), result)
               for r in provenance.History())
    for cell in Topology.Cells(result):
        assert any(same(r.get("source"), original) for original in [a, b]
                   for r in provenance.Origins(cell))
