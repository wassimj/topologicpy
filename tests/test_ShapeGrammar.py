from __future__ import annotations

import math
import pytest

from topologicpy.ShapeGrammar import ShapeGrammar
from topologicpy.Cell import Cell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


def _box(width=2.0, length=2.0, height=2.0, **kwargs):
    return Cell.Prism(width=width, length=length, height=height, **kwargs)


def _translation(x=0.0, y=0.0, z=0.0):
    return [
        [1.0, 0.0, 0.0, float(x)],
        [0.0, 1.0, 0.0, float(y)],
        [0.0, 0.0, 1.0, float(z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _centroid_xyz(topology):
    c = Topology.Centroid(topology)
    return (Vertex.X(c), Vertex.Y(c), Vertex.Z(c))


def test_operation_surface_is_canonical_and_deterministic():
    sg = ShapeGrammar()
    assert sg.OperationTitles() == [
        "Replace", "Transform", "Union", "Difference", "Symmetric Difference",
        "Intersect", "Merge", "Slice", "Impose", "Imprint", "Divide",
    ]
    assert sg.OperationByTitle("xor")["operation"] == "xor"
    assert sg.OperationByTitle("Symmetric Difference")["operation"] == "xor"
    assert sg.OperationByTitle("common")["operation"] == "intersection"
    assert sg.OperationByTitle("not-an-operation") is None


def test_add_rule_returns_stable_id_and_public_descriptor():
    sg = ShapeGrammar(title="Grammar")
    pattern = _box()
    output = _box(0.5, 0.5, 0.5)
    rule = sg.AddRule(pattern, output, title="Pocket", operation="Difference", metadata={"kind": "test"})
    assert rule == 0
    d = sg.Rule(rule)
    assert d["id"] == 0
    assert d["title"] == "Pocket"
    assert d["canonicalOperation"] == "difference"
    assert d["metadata"] == {"kind": "test"}
    assert len(sg.Rules()) == 1


def test_rule_validation_is_operation_specific():
    sg = ShapeGrammar()
    pattern = _box()
    output = _box(0.5, 0.5, 0.5)
    assert sg.AddRule(pattern, None, operation="Difference", silent=True) is None
    assert sg.AddRule(pattern, output, operation="Transform", matrix=_translation(1), silent=True) is None
    assert sg.AddRule(pattern, None, operation="Transform", silent=True) is None
    assert sg.AddRule(pattern, None, operation="Divide", uSides=0, silent=True) is None
    assert sg.AddRule(pattern, output, operation="Replace") == 0


def test_compile_builds_rule_index():
    sg = ShapeGrammar()
    pattern = _box()
    sg.AddRule(pattern, _box(0.5, 0.5, 0.5), operation="Replace")
    summary = sg.Compile()
    assert summary["rules"] == 1
    assert summary["compiledRules"] == 1
    assert summary["types"] == 1


def test_applicable_rules_matches_translated_similar_target():
    sg = ShapeGrammar()
    pattern = _box()
    rule = sg.AddRule(pattern, _box(0.5, 0.5, 0.5), title="Replace", operation="Replace")
    target = Topology.Translate(pattern, 5, -2, 3)
    matches = sg.ApplicableRules(target)
    assert matches
    assert matches[0]["rule"] == rule
    assert len(matches[0]["matrix"]) == 4


def test_matching_cache_avoids_second_similarity_test():
    sg = ShapeGrammar()
    pattern = _box()
    sg.AddRule(pattern, _box(0.5, 0.5, 0.5), operation="Replace")
    target = Topology.Translate(pattern, 1, 2, 3)
    assert sg.ApplicableRules(target)
    first = sg.Status()
    assert sg.ApplicableRules(target)
    second = sg.Status()
    assert second["matchCacheHits"] == first["matchCacheHits"] + 1
    assert second["similarityTests"] == first["similarityTests"]


def test_replace_maps_output_into_target_frame():
    sg = ShapeGrammar()
    pattern = _box(2, 2, 2)
    output = _box(1, 1, 1)
    rule = sg.AddRule(pattern, output, operation="Replace")
    target = Topology.Translate(pattern, 7, -4, 2)
    result = sg.ApplyRule(target, rule)
    assert Topology.IsInstance(result, "Topology")
    cx, cy, cz = _centroid_xyz(result)
    tx, ty, tz = _centroid_xyz(target)
    assert cx == pytest.approx(tx, abs=1e-4)
    assert cy == pytest.approx(ty, abs=1e-4)
    assert cz == pytest.approx(tz, abs=1e-4)


@pytest.mark.parametrize(
    "operation",
    ["Union", "Difference", "Symmetric Difference", "Intersect", "Merge", "Slice", "Impose", "Imprint"],
)
def test_boolean_operations_execute_against_actual_target(operation):
    sg = ShapeGrammar()
    pattern = _box(2, 2, 2)
    tool = _box(1, 1, 3)
    rule = sg.AddRule(pattern, tool, operation=operation)
    target = Topology.Translate(pattern, 3, 0, 0)
    result = sg.ApplyRule(target, rule)
    assert result is not None
    assert Topology.IsInstance(result, "Topology")


def test_rule_preparation_matrix_and_match_matrix_are_composed_once():
    sg = ShapeGrammar()
    pattern = _box(2, 2, 2)
    output = _box(0.5, 0.5, 0.5)
    rule = sg.AddRule(pattern, output, operation="Replace", matrix=_translation(0.5, 0, 0))
    target = Topology.Translate(pattern, 10, 0, 0)
    result = sg.ApplyRule(target, rule)
    cx, _, _ = _centroid_xyz(result)
    tx, _, _ = _centroid_xyz(target)
    assert cx == pytest.approx(tx + 0.5, abs=1e-4)


def test_transform_rule_is_expressed_in_rule_local_frame():
    sg = ShapeGrammar()
    pattern = _box()
    rule = sg.AddRule(pattern, None, operation="Transform", matrix=_translation(1, 0, 0))
    target = Topology.Translate(pattern, 10, 0, 0)
    before = _centroid_xyz(target)
    result = sg.ApplyRule(target, rule)
    after = _centroid_xyz(result)
    assert after[0] == pytest.approx(before[0] + 1, abs=1e-4)


def test_divide_compiles_rule_local_cutters_and_slices_target():
    sg = ShapeGrammar()
    pattern = _box(4, 4, 4)
    rule = sg.AddRule(pattern, None, operation="Divide", uSides=2, vSides=2, wSides=2)
    target = Topology.Translate(pattern, 5, 0, 0)
    result = sg.ApplyRule(target, rule)
    assert result is not None
    cells = Topology.Cells(result, silent=True) or []
    assert len(cells) >= 2


def test_boolean_call_prefers_public_trandict_contract(monkeypatch):
    sg = ShapeGrammar()
    calls = []

    def fake_difference(a, b, **kwargs):
        calls.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(Topology, "Difference", staticmethod(fake_difference))

    assert sg._boolean_call("difference", object(), object(), silent=True) == "ok"
    assert len(calls) == 1
    assert calls[0].get("tranDict") is True
    assert calls[0].get("silent") is True


def test_apply_cache_reuses_result_and_records_application_event():
    sg = ShapeGrammar()
    pattern = _box()
    rule = sg.AddRule(pattern, _box(0.5, 0.5, 0.5), operation="Difference")
    target = Topology.Translate(pattern, 2, 0, 0)
    first = sg.ApplyRule(target, rule)
    status1 = sg.Status()
    second = sg.ApplyRule(target, rule)
    status2 = sg.Status()
    assert second is first
    assert status2["applyCacheHits"] == status1["applyCacheHits"] + 1
    assert status2["applications"] == 2
    assert sg.Application()["cached"] is True


def test_replace_identity_has_explicit_unchanged_lineage():
    sg = ShapeGrammar()
    pattern = _box()
    output = _box(1, 1, 1)
    rule = sg.AddRule(pattern, output, operation="Replace")
    result = sg.ApplyRule(pattern, rule, matrix=ShapeGrammar._identity_matrix(), lineage=True)
    assert result is output
    history = sg.History(application=sg.Status()["lastApplication"])
    assert history
    assert any(record["relation"] == "unchanged" for record in history)


def test_pythonocc_boolean_history_uses_brepgraph_when_available():
    try:
        from topologicpy.pythonocc_backend._brepgraph import is_available
    except Exception:
        pytest.skip("PythonOCC BRepGraph adapter unavailable")
    if not is_available():
        pytest.skip("BRepGraph unavailable/disabled")

    sg = ShapeGrammar()
    pattern = _box(2, 2, 2)
    rule = sg.AddRule(pattern, _box(1, 1, 3), operation="Difference")
    target = Topology.Translate(pattern, 2, 0, 0)

    assert sg.ApplyRule(target, rule, lineage=True) is not None
    history = sg.History(application=sg.Status()["lastApplication"])
    assert history

    # The tool alignment itself also produces exact transform history.  The
    # important regression check is that the subsequent Boolean contributes
    # its own native Difference records rather than merely inheriting those
    # transform records.
    difference_history = [
        record
        for record in history
        if record.get("operation") == "Difference"
    ]
    assert difference_history
    assert any(record.get("usedBRepGraph") is True for record in difference_history)
    assert {
        record.get("relation")
        for record in difference_history
    } & {"modified", "generated", "unchanged", "deleted"}

    # A volumetric cut should expose at least one Face relationship from the
    # Boolean history. OCCT may classify the interface as modified or generated,
    # so the test deliberately does not require one specific relation.
    assert any(
        record.get("sourceType") == "Face"
        or record.get("resultType") == "Face"
        for record in difference_history
    )


def test_history_queries_and_lineage_graph():
    sg = ShapeGrammar()
    pattern = _box(2, 2, 2)
    rule = sg.AddRule(pattern, _box(1, 1, 3), operation="Difference")
    target = Topology.Translate(pattern, 2, 0, 0)
    result = sg.ApplyRule(target, rule, lineage=True)
    app = sg.Status()["lastApplication"]
    assert result is not None
    assert isinstance(sg.History(application=app), list)
    assert isinstance(sg.GeneratedBy(app), list)
    assert isinstance(sg.ModifiedBy(app), list)
    assert isinstance(sg.DeletedBy(app), list)
    graph = sg.LineageGraph(app)
    assert graph is not None


def test_derivation_graph_tracks_rule_application_sequence():
    sg = ShapeGrammar()
    pattern = _box()
    rule = sg.AddRule(pattern, _box(0.75, 0.75, 0.75), operation="Replace")
    target = Topology.Translate(pattern, 1, 0, 0)
    first = sg.ApplyRule(target, rule)
    # Force is intentional: the replacement is not required to match the pattern.
    second = sg.ApplyRule(first, rule, matrix=ShapeGrammar._identity_matrix(), force=True)
    assert second is not None
    graph = sg.DerivationGraph()
    assert graph is not None


def test_json_roundtrip_preserves_rules_not_runtime_state():
    sg = ShapeGrammar(title="Roundtrip", description="test")
    pattern = _box()
    rule = sg.AddRule(pattern, _box(0.5, 0.5, 0.5), title="R", operation="Difference", metadata={"a": 1})
    target = Topology.Translate(pattern, 1, 0, 0)
    assert sg.ApplyRule(target, rule) is not None
    assert sg.Status()["applications"] == 1

    string = sg.JSONString()
    restored = ShapeGrammar.ByJSONString(string)
    assert restored is not None
    assert restored.title == "Roundtrip"
    assert restored.Rule(rule)["title"] == "R"
    assert restored.Rule(rule)["metadata"] == {"a": 1}
    assert restored.Status()["applications"] == 0


def test_remove_rule_invalidates_runtime_caches():
    sg = ShapeGrammar()
    pattern = _box()
    rule = sg.AddRule(pattern, _box(0.5, 0.5, 0.5), operation="Replace")
    target = Topology.Translate(pattern, 1, 0, 0)
    assert sg.ApplicableRules(target)
    assert sg.Status()["matchCacheEntries"] > 0
    assert sg.RemoveRule(rule)
    assert sg.Rule(rule) is None
    assert sg.Status()["matchCacheEntries"] == 0


def test_public_exports():
    import topologicpy.ShapeGrammar as module
    assert module.__all__ == ["ShapeGrammar"]
