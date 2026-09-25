"""Focused tests for the private OCCT 8 BRepGraph index layer."""

import pytest

pytest.importorskip("OCC.Core.BRepGraph")

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepGraph import BRepGraph_ChildExplorer, BRepGraph_NodeId, brepgraph
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer

from topologicpy.pythonocc_backend._brepgraph import (
    BRepGraphIndex,
    is_available,
    shared_shapes,
)


def _first_subshape(shape, shape_type):
    explorer = TopExp_Explorer(shape, shape_type)
    assert explorer.More()
    return explorer.Current()


def test_brepgraph_index_is_available_on_pythonocc_8(monkeypatch):
    monkeypatch.delenv("TOPOLOGICPY_DISABLE_BREPGRAPH", raising=False)
    assert is_available() is True


def test_brepgraph_index_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TOPOLOGICPY_DISABLE_BREPGRAPH", "1")
    assert is_available() is False
    monkeypatch.delenv("TOPOLOGICPY_DISABLE_BREPGRAPH", raising=False)


def test_raw_brepgraph_child_explorer_walks_box():
    """Verify wrapped NodeAt/CurrentParent access without using Current().

    pythonocc-core 8.0.1 does not expose BRepGraphInc::NodeInstance as a
    Python proxy, so ChildExplorer.Current() is intentionally not used.
    """
    graph = brepgraph()
    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    add_result = graph.Shapes().Add(box)
    assert add_result.IsOk()
    assert graph.Topo().Faces().Nb() == 6

    explorer = BRepGraph_ChildExplorer(graph, add_result.TopologyRoot)
    face_nodes = set()
    emitted = 0

    while explorer.More():
        depth = int(explorer.Depth())
        assert depth >= 2

        # OCCT 8.0.1 implementation: NodeAt(0) is stack frame 1, i.e. the
        # first node below the explicit root.  Therefore the current emitted
        # node is NodeAt(Depth() - 2).
        node = explorer.NodeAt(depth - 2)
        parent = explorer.CurrentParent()

        assert node.IsValid()
        assert parent.IsValid()
        assert graph.Topo().Gen().IsActive(node)
        assert graph.Topo().Gen().IsActive(parent)

        emitted += 1
        if node.NodeKind == BRepGraph_NodeId.Kind_Face:
            face_nodes.add(int(node.Index))
        explorer.Next()

    assert emitted > 0
    assert len(face_nodes) == 6


def test_brepgraph_subtopology_and_supertopology_queries():
    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    index = BRepGraphIndex(box)
    assert index.valid

    faces = index.subshapes(box, TopAbs_FACE)
    assert faces is not None
    assert len(faces) == 6

    ancestors = index.super_shapes(faces[0], TopAbs_SOLID)
    assert ancestors is not None
    assert len(ancestors) == 1
    assert ancestors[0].IsSame(box)


def test_brepgraph_shared_topology_query():
    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    face = _first_subshape(box, TopAbs_FACE)

    face_edges = BRepGraphIndex(face).subshapes(face, TopAbs_EDGE)
    shared = shared_shapes(face, box, TopAbs_EDGE)

    assert face_edges is not None
    assert shared is not None
    assert len(shared) == len(face_edges) == 4


def test_brepgraph_same_dimension_face_adjacency():
    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    index = BRepGraphIndex(box)
    faces = index.subshapes(box, TopAbs_FACE)
    assert faces is not None and len(faces) == 6

    adjacent = index.adjacent_shapes(faces[0], TopAbs_FACE)
    assert adjacent is not None
    assert len(adjacent) == 4
