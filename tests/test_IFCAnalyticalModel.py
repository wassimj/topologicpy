"""Release tests: analytical reconstruction must work without any external engine."""
from copy import deepcopy
from pathlib import Path
import builtins
import json
import os
import subprocess
import sys

import pytest
from shapely import set_precision
from shapely.geometry import box, Point, mapping

from topologicpy.IFC import IFC, IFCFastEntity
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Cell import Cell

DATA = Path(__file__).with_name("data")
SOURCE = DATA/"analytical-three-rooms.ifc"


def _bounds(topology):
    p = [Vertex.Coordinates(v) for v in Topology.Vertices(topology)]
    return [[min(v[i] for v in p) for i in range(3)], [max(v[i] for v in p) for i in range(3)]]


def _site(offset=40., declared=True):
    model = IFC.Model(str(SOURCE), silent=True)
    e = deepcopy(model.entities)
    shift = max(e)+1
    def remap(v):
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "REF":
            return ("REF", v[1]+shift)
        if isinstance(v, list):
            return [remap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(remap(x) for x in v)
        return v
    for i, entity in model.entities.items():
        args = remap(deepcopy(entity.args))
        if entity.type == "IFCCARTESIANPOINT":
            args[0][0] += offset
        if len(args) > 4 and isinstance(args[0], str):
            args[0] = "second:"+args[0]
        e[i+shift] = IFCFastEntity(i+shift, entity.type, args)
    if declared:
        for n, ids in enumerate((list(model.entities), [i+shift for i in model.entities])):
            i = max(e)+1
            e[i] = IFCFastEntity(i, "IFCBUILDING", [f"building-{n}", None, f"Building {n}", None, None, None, None, None, "ELEMENT", 0., 0., None])
            children = [("REF", j) for j in ids if e[j].type in {"IFCSPACE", "IFCWALL", "IFCSLAB", "IFCOPENINGELEMENT", "IFCWINDOW"}]
            e[i+1] = IFCFastEntity(i+1, "IFCRELAGGREGATES", [f"aggregate-{n}", None, None, None, ("REF", i), children])
    return IFC.ModelByEntities(e, silent=True)


def _plan(outlines=None):
    source = IFC.AnalyticalModel(str(SOURCE), coordinateFrame="local", verify=False)
    plan = deepcopy(source.editable_plan)
    outlines = outlines or [box(0, 0, 5, 8), box(5, 0, 10, 8)]
    template = plan["rooms"][0]
    plan["rooms"] = []
    for i, polygon in enumerate(outlines):
        plan["rooms"].append({**deepcopy(template), "id": f"room-{i}", "source_space_id": f"room-{i}", "name": f"Room {i}",
            "footprint": mapping(polygon), "reference_footprint": mapping(polygon), "reference_area": polygon.area,
            "area": polygon.area, "bottom": 0., "top": 4., "reference_bottom": 0., "reference_top": 4., "air_top": 4.,
            "exterior_cap": True, "storey": 0})
    plan.update(origin=[0., 0., 0.], reference_openings=[], source_boundaries=[], construction_evidence=[], fill_types={})
    plan["fidelity"].update(reference_spaces=2, eligible_source_space_records=2, coincident_source_aliases={})
    return plan


def test_self_contained_reconstruction_and_readiness(tmp_path, monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == "bim2bem" or name.startswith("bim2bem.") or name == "ifcopenshell" or name.startswith("ifcopenshell."):
            raise AssertionError("Standalone analytical reconstruction imported an external engine")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setenv("TOPOLOGICPY_ANALYTICAL_ENGINE", str(tmp_path/"does-not-exist"))
    monkeypatch.chdir(tmp_path)
    model = IFC.Model(str(SOURCE), silent=True)
    before = repr(model.entities)
    result = IFC.AnalyticalModel(model)
    assert result.valid and len(result.spaces) == 3
    assert len(result.cell_complexes) == 1
    assert sum(s["analytical_volume"] for s in result.spaces) == pytest.approx(192.)
    assert result.readiness["geometry_verified"] and not result.readiness["simulation_ready"]
    assert repr(model.entities) == before
    assert type(result).__module__ == "topologicpy.IFC"
    assert _bounds(result.topology)[0] == pytest.approx([0., 0., 0.])
    assert _bounds(result.topology)[1] == pytest.approx([10., 6., 3.2])


@pytest.mark.parametrize("offset,declared", [(40., True), (10., True), (40., False)])
def test_independent_buildings_even_when_touching(tmp_path, offset, declared):
    result = IFC.AnalyticalModel(_site(offset, declared))
    assert len(result.building_models) == 2 and len(result.cell_complexes) == 2
    assert all(Topology.IsInstance(c, "CellComplex") for c in result.cell_complexes)
    assert [len(Topology.Cells(c)) for c in result.cell_complexes] == [3, 3]
    assert IFC.AnalyticalModelSave(result, tmp_path)
    restored = IFC.AnalyticalModelLoad(tmp_path)
    assert len(restored.cell_complexes) == 2
    assert len(Topology.Cells(restored.topology)) == 6
    local = IFC.AnalyticalModelLoad(tmp_path, coordinateFrame="local")
    assert local.origin == pytest.approx(result.editable_plan["origin"] if any(result.origin) else result.building_models[0].editable_plan["origin"])


def test_saved_world_geometry_and_local_editing_frame(tmp_path):
    result = IFC.AnalyticalModel(str(SOURCE))
    IFC.AnalyticalModelSave(result, tmp_path)
    world = IFC.AnalyticalModelLoad(tmp_path)
    local = IFC.AnalyticalModelLoad(tmp_path, coordinateFrame="local")
    assert _bounds(world.topology)[0] == pytest.approx([0., 0., 0.])
    origin = result.editable_plan["origin"]
    assert local.origin == pytest.approx(origin)
    assert _bounds(local.topology)[0] == pytest.approx([-v for v in origin])
    assert len(world.openings) == len(result.openings)
    assert IFC.AnalyticalModelSave(local, tmp_path/"resaved")
    reread = IFC.AnalyticalModelLoad(tmp_path/"resaved")
    assert _bounds(reread.topology)[0] == pytest.approx([0., 0., 0.])


def test_rotated_placements_and_millimetres():
    model = IFC.Model(str(DATA/"analytical-three-rooms-mm.ifc"), silent=True)
    e, i = model.entities, max(model.entities)+1
    e[i] = IFCFastEntity(i, "IFCCARTESIANPOINT", [[100000., 200000., 3000.]])
    e[i+1] = IFCFastEntity(i+1, "IFCDIRECTION", [[0., 0., 1.]])
    e[i+2] = IFCFastEntity(i+2, "IFCDIRECTION", [[0., 1., 0.]])
    e[i+3] = IFCFastEntity(i+3, "IFCAXIS2PLACEMENT3D", [("REF", i), ("REF", i+1), ("REF", i+2)])
    e[i+4] = IFCFastEntity(i+4, "IFCLOCALPLACEMENT", [None, ("REF", i+3)])
    for product in e.values():
        if product.type in {"IFCSPACE", "IFCWALL", "IFCSLAB", "IFCOPENINGELEMENT", "IFCWINDOW"}:
            product.args[5] = ("REF", i+4)
    result = IFC.AnalyticalModel(model)
    assert result.valid
    assert _bounds(result.topology)[0] == pytest.approx([94., 200., 3.], abs=.02)
    assert _bounds(result.topology)[1] == pytest.approx([100., 210., 6.2], abs=.02)


@pytest.mark.parametrize("curved", [False, True])
def test_mansard_and_curved_facade_preserve_spatial_volume(curved):
    outlines = [box(0, 0, 5, 8), box(5, 0, 10, 8)]
    if curved:
        outlines[1] = set_precision(outlines[1].union(Point(10, 4).buffer(1, resolution=12)), .01)
    plan = _plan(outlines)
    patches = [(box(0, 0, 11, 2), [0, 1, 3]), (box(0, 2, 11, 4), [0, .5, 4]),
               (box(0, 4, 11, 6), [0, -.5, 8]), (box(0, 6, 11, 8), [0, -1, 11])]
    plan["exterior_envelope"] = {"facets": [{"footprint": mapping(p), "plane": c} for p, c in patches]}
    result = IFC.AnalyticalModelByPlan(plan)
    assert result.fidelity["exterior_fit_applied"]
    expected = []
    for outline in outlines:
        volume = 0.
        for patch, (a, b, c) in patches:
            r = set_precision(outline, 0).intersection(patch)
            volume += r.area*(a*r.centroid.x+b*r.centroid.y+c)
        expected.append(volume)
    assert len(result.spaces) == 2
    assert sorted(s["analytical_volume"] for s in result.spaces) == pytest.approx(sorted(expected), abs=.001)


def test_step_is_not_blended_or_split_into_extra_cells():
    plan = _plan()
    plan["exterior_envelope"] = {"facets": [{"footprint": mapping(box(0, 0, 5, 8)), "plane": [0, 0, 3]},
                                            {"footprint": mapping(box(5, 0, 10, 8)), "plane": [0, 0, 5]}]}
    result = IFC.AnalyticalModelByPlan(plan)
    assert result.fidelity["exterior_fit_applied"]
    assert len(result.spaces) == 2
    assert sorted(s["analytical_volume"] for s in result.spaces) == pytest.approx([120., 200.])


def test_leaning_facade_preserves_full_cells():
    plan = _plan([box(0, 0, 5, 8), box(5, 0, 11, 8)])
    for r in plan["rooms"]:
        r.pop("exterior_cap", None)
    plan["facade_envelope"] = {"walls": [{"points": [[10, 0, 0], [10, 8, 0], [11, 8, 4], [11, 0, 4]], "faces": [[0, 1, 2, 3]]}], "maximum_displacement_m": 1.5}
    result = IFC.AnalyticalModelByPlan(plan)
    assert result.fidelity["exterior_fit_applied"]
    assert sorted(s["analytical_volume"] for s in result.spaces) == pytest.approx([160., 176.], abs=.001)


def test_invalid_inputs_are_explicit():
    assert not IFC.AnalyticalModel("nonexistent-ifc-file.ifc").valid
    with pytest.raises(ValueError):
        IFC.AnalyticalModel(str(SOURCE), coordinateFrame="unknown")
    with pytest.raises(ValueError):
        IFC.AnalyticalModel(str(SOURCE), options={"role_overrides": {"X": "invalid-role"}})
    with pytest.raises(TypeError):
        IFC.AnalyticalModelByPlan(42)


def test_shared_boundary_has_common_physical_and_virtual_patches():
    plan = _plan()
    plan["source_boundaries"] = []
    for owner in ["room-0", "room-1"]:
        for kind, low, high in [("physical", 0, 2), ("virtual", 2, 4)]:
            plan["source_boundaries"].append({"id": owner+kind, "space": owner,
                "element": "wall" if kind == "physical" else None, "physical_or_virtual": kind,
                "internal_or_external": "internal", "points": [[5, 0, low], [5, 8, low], [5, 8, high], [5, 0, high]]})
    result = IFC.AnalyticalModelByPlan(plan)
    patches = [p for p in result.boundary_patches if len(p["owners"]) == 2]
    assert len(patches) == 2
    assert {p["physical_or_virtual"] for p in patches} == {"physical", "virtual"}
    assert sum(p["area"] for p in patches) == pytest.approx(32.)
    assert all(p["semantics_verified"] for p in patches)


def test_partial_building_failure_preserves_the_usable_building(monkeypatch, tmp_path):
    original = IFC._Analytical_redraw_generate_building
    def partial(evidence, *args, **kwargs):
        if evidence.statistics["building_id"] == "building-1":
            raise ValueError("Unsupported building fixture")
        return original(evidence, *args, **kwargs)
    monkeypatch.setattr(IFC, "_Analytical_redraw_generate_building", staticmethod(partial))
    result = IFC.AnalyticalModel(_site())
    assert result.valid and len(result.building_models) == 2
    assert len(result.cell_complexes) == 1
    assert not result.readiness["geometry_verified"]
    IFC.AnalyticalModelSave(result, tmp_path)
    restored = IFC.AnalyticalModelLoad(tmp_path)
    assert restored.valid and len(restored.building_models) == 2
    assert len(restored.cell_complexes) == 1


def test_native_ifc_import_remains_lazy():
    import topologicpy
    package_parent = str(Path(topologicpy.__file__).resolve().parent.parent)
    script = '''import builtins
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'bim2bem','numpy','shapely','scipy','ifcopenshell'}:
        raise AssertionError('Eager optional import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
from topologicpy.IFC import IFC
assert callable(IFC.AnalyticalModel)
'''
    env = {**os.environ, "PYTHONPATH": package_parent}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_missing_space_records_produce_an_explicitly_inferred_draft():
    model = IFC.Model(str(SOURCE), silent=True)
    model.entities = {i: e for i, e in model.entities.items() if e.type != "IFCSPACE"}
    result = IFC.AnalyticalModel(model)
    assert result.valid and len(result.spaces) >= 2
    assert any(d.code == "ROOMS_INFERRED" for d in result.diagnostics)
    assert not result.readiness["simulation_ready"]


def test_missing_construction_does_not_discard_rooms(monkeypatch):
    from topologicpy.IFC import IFCFastTopology
    original = IFCFastTopology.MeshDataByProduct
    def partial(product, *args, **kwargs):
        if product.type == "IFCWALL" and product.args[2] == "west":
            return None
        return original(product, *args, **kwargs)
    monkeypatch.setattr(IFCFastTopology, "MeshDataByProduct", staticmethod(partial))
    result = IFC.AnalyticalModel(str(SOURCE))
    assert result.valid and len(result.spaces) == 3
    assert any(d.code == "PARTIAL_REFERENCE_EVIDENCE" for d in result.diagnostics)


def test_literal_step_semicolon_and_role_override(tmp_path):
    text = SOURCE.read_text()
    position = text.rfind("ENDSEC;")
    text = text[:position]+"#99999=IFCPROPERTYSINGLEVALUE('Finish',$,IFCTEXT('tile; finish'),$);\n"+text[position:]
    path = tmp_path/"literal.ifc"
    path.write_text(text)
    result = IFC.AnalyticalModel(str(path), options={"role_overrides": {"A": "void"}})
    assert result.valid
    assert "A" not in {s["source_space_id"] for s in result.spaces}
    assert any(e["source"] == "A" and e["role"] == "void" for e in result.fidelity["excluded_source_spaces"])
