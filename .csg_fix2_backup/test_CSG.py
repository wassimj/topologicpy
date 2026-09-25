"""Tests for the provenance-aware TGraph CSG expression engine."""

from __future__ import annotations

import pytest

CSG = pytest.importorskip("topologicpy.CSG").CSG
TGraph = pytest.importorskip("topologicpy.TGraph").TGraph
Topology = pytest.importorskip("topologicpy.Topology").Topology
Cell = pytest.importorskip("topologicpy.Cell").Cell
Vertex = pytest.importorskip("topologicpy.Vertex").Vertex


def _box(x=0.0, y=0.0, z=0.0, size=1.0):
    box = Cell.Prism(width=size, length=size, height=size, silent=True)
    if x or y or z:
        try:
            box = Topology.Translate(box, x=x, y=y, z=z, silent=True)
        except TypeError:
            box = Topology.Translate(box, x, y, z)
    return box


def _active_records(graph):
    return TGraph.Vertices(graph, copy=False, asTopologic=False, active=True)


def test_csg_expression_is_tgraph_and_operand_order_lives_on_edges():
    graph = CSG.Init()
    assert isinstance(graph, TGraph)

    a = CSG.Source(graph, _box(), name="A")
    b = CSG.Source(graph, _box(x=0.5), name="B")
    op = CSG.Difference(graph, a, b, name="A minus B")

    assert isinstance(a, int)
    assert isinstance(b, int)
    assert isinstance(op, int)

    report = CSG.Validate(graph)
    assert report["valid"] is True
    assert report["roots"] == [op]

    incoming = TGraph.IncomingEdges(graph, op)
    assert len(incoming) == 2
    operand_roles = sorted((edge["dictionary"]["operand"], edge["src"]) for edge in incoming)
    assert operand_roles == [(0, a), (1, b)]

    op_record = TGraph.Vertex(graph, op, copy=False, active=True, asTopologic=False)
    assert "a_id" not in op_record["dictionary"]
    assert "b_id" not in op_record["dictionary"]
    assert "brep" not in op_record["dictionary"]


def test_evaluate_and_cache_reuse_unchanged_expression():
    graph = CSG.Init()
    a = CSG.Source(graph, _box(), name="A")
    b = CSG.Source(graph, _box(x=0.4), name="B")
    op = CSG.Union(graph, a, b)

    first = CSG.Evaluate(graph, op, silent=True)
    assert Topology.IsInstance(first, "Topology")
    status1 = CSG.Status(graph)

    second = CSG.Evaluate(graph, op, silent=True)
    assert second is first
    status2 = CSG.Status(graph)
    assert status2["cacheHits"] > status1["cacheHits"]


def test_set_source_invalidates_only_downstream_cache():
    graph = CSG.Init()
    a = CSG.Source(graph, _box(), name="A")
    b = CSG.Source(graph, _box(x=0.4), name="B")
    c = CSG.Source(graph, _box(x=3.0), name="C")
    ab = CSG.Union(graph, a, b)
    root = CSG.Union(graph, ab, c)

    result1 = CSG.Evaluate(graph, root, silent=True)
    assert Topology.IsInstance(result1, "Topology")
    assert CSG.Result(graph, c) is not None

    replacement = _box(x=0.7)
    assert CSG.SetSource(graph, b, replacement, silent=True)
    assert CSG.Result(graph, b) is None
    assert CSG.Result(graph, ab) is None
    assert CSG.Result(graph, root) is None
    # C is independent of B and remains cached.
    assert CSG.Result(graph, c) is not None

    result2 = CSG.Evaluate(graph, root, silent=True)
    assert Topology.IsInstance(result2, "Topology")


def test_brep_is_created_only_at_persistence_boundary():
    graph = CSG.Init()
    a = CSG.Source(graph, _box(), name="A")
    record = TGraph.Vertex(graph, a, copy=False, active=True, asTopologic=False)
    assert "brep" not in record["dictionary"]

    data = CSG.Data(graph)
    assert data["schema"] == CSG.SCHEMA
    source_data = next(item for item in data["vertices"] if item["index"] == a)
    assert isinstance(source_data.get("brep"), str)
    assert len(source_data["brep"]) > 0

    rebuilt = CSG.ByData(data, silent=True)
    assert isinstance(rebuilt, TGraph)
    report = CSG.Validate(rebuilt)
    assert report["valid"] is True
    result = CSG.Evaluate(rebuilt, silent=True)
    assert Topology.IsInstance(result, "Topology")


def test_transform_is_an_explicit_expression_node():
    graph = CSG.Init()
    a = CSG.Source(graph, _box(), name="A")
    matrix = [
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    moved = CSG.Transform(graph, a, matrix)
    assert CSG.Validate(graph)["valid"] is True
    result = CSG.Evaluate(graph, moved, silent=True)
    assert Topology.IsInstance(result, "Topology")


def test_invalid_arity_is_rejected_at_construction():
    graph = CSG.Init()
    a = CSG.Source(graph, _box())
    assert CSG.Operation(graph, "difference", [a], silent=True) is None
    assert CSG.Operation(graph, "transform", [a, a], matrix=[[1, 0, 0, 0]] * 4, silent=True) is None


def test_pythonocc_csg_captures_exact_lineage_with_brepgraph():
    pytest.importorskip("OCC.Core.BRepGraph")

    graph = CSG.Init()
    a = CSG.Source(graph, _box(size=2.0), name="host")
    b = CSG.Source(graph, _box(x=0.75, size=1.0), name="tool")
    cut = CSG.Difference(graph, a, b)

    result = CSG.Evaluate(graph, cut, lineage=True, silent=True)
    assert Topology.IsInstance(result, "Topology")

    history = CSG.History(graph, operation=cut)
    assert history
    assert any(record.get("usedBRepGraph") is True for record in history)
    assert {record.get("relation") for record in history} & {"modified", "generated", "unchanged", "deleted"}

    materialised = [record.get("result") for record in history if record.get("result") is not None]
    assert materialised
    origins = CSG.Origins(graph, materialised[0], operation=cut)
    assert isinstance(origins, list)

    lineage_graph = CSG.LineageGraph(graph, operation=cut)
    assert isinstance(lineage_graph, TGraph)
    assert TGraph.Order(lineage_graph) > 0


def test_multiple_roots_require_explicit_target():
    graph = CSG.Init()
    a = CSG.Source(graph, _box())
    b = CSG.Source(graph, _box(x=3.0))
    report = CSG.Validate(graph)
    assert report["valid"] is True
    assert len(report["roots"]) == 2
    assert CSG.Evaluate(graph, silent=True) is None
    assert Topology.IsInstance(CSG.Evaluate(graph, a, silent=True), "Topology")
