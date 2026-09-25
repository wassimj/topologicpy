"""Integration tests for BRepGraph Tranche 1 in the PythonOCC backend."""

import pytest

pytest.importorskip("OCC.Core.BRepGraph")

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

from topologicpy.pythonocc_backend.topology import Topology


def _same_shape_set(left, right):
    left_shapes = [getattr(item, "shape", None) for item in left or []]
    right_shapes = [getattr(item, "shape", None) for item in right or []]
    if len(left_shapes) != len(right_shapes):
        return False
    used = set()
    for shape_a in left_shapes:
        matched = False
        for i, shape_b in enumerate(right_shapes):
            if i in used or shape_a is None or shape_b is None:
                continue
            try:
                same = bool(shape_a.IsSame(shape_b))
            except Exception:
                same = False
            if same:
                used.add(i)
                matched = True
                break
        if not matched:
            return False
    return True


def _box_model():
    shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    topology = Topology.ByOcctShape(shape)
    assert topology is not None
    return topology


def test_backend_sub_super_shared_and_adjacent_queries_use_exact_incidence(monkeypatch):
    monkeypatch.delenv("TOPOLOGICPY_DISABLE_BREPGRAPH", raising=False)
    box = _box_model()

    faces = Topology.Faces(box) or []
    assert len(faces) == 6

    face = faces[0]
    ancestors = Topology.SuperTopologies(face, box, "cell") or []
    assert len(ancestors) == 1
    assert ancestors[0].shape.IsSame(box.shape)

    shared_edges = Topology.SharedTopologies(face, box, "edge") or []
    assert len(shared_edges) == 4

    adjacent_faces = face.AdjacentFaces(box) or []
    assert len(adjacent_faces) == 4


def test_brepgraph_and_legacy_paths_are_semantically_equivalent(monkeypatch):
    box = _box_model()

    monkeypatch.delenv("TOPOLOGICPY_DISABLE_BREPGRAPH", raising=False)
    graph_faces = Topology.Faces(box) or []
    graph_super = Topology.SuperTopologies(graph_faces[0], box, "cell") or []
    graph_shared = Topology.SharedTopologies(graph_faces[0], box, "edge") or []
    graph_adjacent = graph_faces[0].AdjacentFaces(box) or []

    monkeypatch.setenv("TOPOLOGICPY_DISABLE_BREPGRAPH", "1")
    legacy_faces = Topology.Faces(box) or []
    # Match the same face by native identity; list ordering must not be assumed.
    legacy_face = next(
        face for face in legacy_faces if face.shape.IsSame(graph_faces[0].shape)
    )
    legacy_super = Topology.SuperTopologies(legacy_face, box, "cell") or []
    legacy_shared = Topology.SharedTopologies(legacy_face, box, "edge") or []
    legacy_adjacent = legacy_face.AdjacentFaces(box) or []

    assert _same_shape_set(graph_faces, legacy_faces)
    assert _same_shape_set(graph_super, legacy_super)
    assert _same_shape_set(graph_shared, legacy_shared)
    assert _same_shape_set(graph_adjacent, legacy_adjacent)
