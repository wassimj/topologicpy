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



def _plan(outlines=None):
    # This tests manual plan assembly. Its input must not depend on the
    # source reconstruction strategy used by IFC.AnalyticalModel.
    outlines = outlines or [box(0, 0, 5, 8), box(5, 0, 10, 8)]
    rooms = []
    for i, polygon in enumerate(outlines):
        rooms.append(dict(id=f"room-{i}", source_space_id=f"room-{i}",
            name=f"Room {i}", footprint=mapping(polygon),
            reference_footprint=mapping(polygon), reference_area=polygon.area,
            reference_method="explicit_space", confidence={"score": 1.0},
            area=polygon.area, bottom=0., top=4., reference_bottom=0.,
            reference_top=4., air_top=4., exterior_cap=True, storey=0))
    return dict(rooms=rooms, origin=[0., 0., 0.], reference_openings=[],
        source_boundaries=[], construction_evidence=[], fill_types={},
        fidelity=dict(reference_spaces=2, eligible_source_space_records=2,
            coincident_source_aliases={}), warnings=[])


@pytest.mark.pythonocc_only
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


@pytest.mark.pythonocc_only
def test_step_is_not_blended_or_split_into_extra_cells():
    plan = _plan()
    plan["exterior_envelope"] = {"facets": [{"footprint": mapping(box(0, 0, 5, 8)), "plane": [0, 0, 3]},
                                            {"footprint": mapping(box(5, 0, 10, 8)), "plane": [0, 0, 5]}]}
    result = IFC.AnalyticalModelByPlan(plan)
    assert result.fidelity["exterior_fit_applied"]
    assert len(result.spaces) == 2
    assert sorted(s["analytical_volume"] for s in result.spaces) == pytest.approx([120., 200.])


@pytest.mark.pythonocc_only
def test_leaning_facade_preserves_full_cells():
    plan = _plan([box(0, 0, 5, 8), box(5, 0, 11, 8)])
    for r in plan["rooms"]:
        r.pop("exterior_cap", None)
    plan["facade_envelope"] = {"walls": [{"points": [[10, 0, 0], [10, 8, 0], [11, 8, 4], [11, 0, 4]], "faces": [[0, 1, 2, 3]]}], "maximum_displacement_m": 1.5}
    result = IFC.AnalyticalModelByPlan(plan)
    assert result.fidelity["exterior_fit_applied"]
    assert sorted(s["analytical_volume"] for s in result.spaces) == pytest.approx([160., 176.], abs=.001)


@pytest.mark.pythonocc_only
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

