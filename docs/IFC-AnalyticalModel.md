# Analytical spatial models from IFC

`IFC.AnalyticalModel` generates a reviewable spatial model using the native IFC
parser and geometry interpreter. Its implementation is included in `IFC.py`.
There is no separate reconstruction package, local source registration, project
path or IfcOpenShell requirement. NumPy, SciPy and Shapely are already TopologicPy
dependencies and load only when analytical reconstruction is requested.

The feature is experimental. Closed, shared geometry is not a certification of
IFC second-level boundaries, construction properties or energy predictions.

## Basic use

```python
from topologicpy.IFC import IFC

result = IFC.AnalyticalModel(
    "building.ifc",
    options={"geometry_tolerance": 0.02, "drafting_grid": 0.01},
    coordinateFrame="world",
)

for building in result.building_models:
    print(building.building_id, building.readiness)
    print(building.fidelity)
    for diagnostic in building.diagnostics:
        print(diagnostic.code, diagnostic.message)

complexes = result.cell_complexes
IFC.AnalyticalModelSave(result, "analytical-model")
restored = IFC.AnalyticalModelLoad("analytical-model")
```

The input can be a path or an existing `IFCModel`. Source entities are not
modified. The result exposes:

- `topology`: one building topology, or a Cluster of independent buildings.
- `cell_complexes`: actual CellComplexes only; an unconnected draft is not
  presented as a verified CellComplex.
- `building_models`: individual results, including unusable building entries.
- `spaces`, `boundaries`, `boundary_patches`, `openings`: geometry and provenance.
- `editable_plan`: reconstruction geometry, reference contours and assumptions.
- `fidelity`, `diagnostics`, `statistics`, `readiness` and `report()`.

## Separate buildings

IFC building membership has priority. Where membership is missing, proximity
and disconnected spatial components provide a recorded inference. Layouts are
generated separately. Even touching declared buildings are not Boolean-unioned
into one CellComplex. A failed building does not discard usable buildings, but
coverage and readiness continue to report the failure.

Site exports contain `buildings/001`, `buildings/002`, etc., with independent
geometry, metadata, plans and reports. The root BREP is a convenience collection;
it does not create cross-building shared faces.

## Coordinates and editing

The public API defaults to IFC engineering/world coordinates in metres. Nested
placements, rotations and declared length units are respected. This is the
native importer's coordinate convention; GIS/map-conversion reprojection is not
performed. Use `coordinateFrame="local"` for the calculation frame.

The editable plan has its own `origin`, even when the returned topology is in
world coordinates. Add that origin to plan coordinates to obtain IFC-world
metres. Keep BREP and metadata together when exchanging a saved model.

```python
plan = result.editable_plan
# Edit room footprints/heights, or edit the saved editable-plan.json file.
edited = IFC.AnalyticalModelByPlan(plan, coordinateFrame="world")
IFC.AnalyticalModelSave(edited, "edited-model")
```

Rebuilding recomputes fidelity and checks common interfaces. Save/load verifies
room identities, volumes, closed shell incidence, shared native faces and
apertures. The loader also supports the earlier local-coordinate BREP sidecar
convention. New exports are co-located with the IFC in world metres.

## Settings

Settings are supplied as an `options` dictionary. Lengths are metres; areas are
square metres.

| Setting | Default | Purpose |
|---|---:|---|
| `tolerance` | 0.00001 | Native geometry tolerance |
| `geometry_tolerance` | 0.02 | Exterior curve approximation target |
| `drafting_grid` | 0.01 | Common planar arrangement resolution |
| `minimum_cell_height` | 0.4 | Reject collapsed/thin spatial regions |
| `minimum_fragment_area` | 0.03 | Exclude drafting fragments |
| `include_outdoor` | False | Include explicitly selected outdoor spaces |
| `role_overrides` | {} | Source-space GUID or uppercase label to role |

Roles include `room`, `atrium`, `shaft`, `stair`, `plenum`, `outdoor`, `void`,
`roof` and `area_measurement`. Overrides precede structured IFC evidence, which
precedes reviewable multilingual name rules. Roof-area, void and aggregate-area
records are not automatically converted to rooms. Tiny closure pockets remain
diagnostics, rather than extra spatial cells.

Declared storeys anchor elevations. Exterior bottom floors use outer slab faces;
internal slab levels use mid-surfaces. Missing construction evidence produces
an explicitly inferred envelope. Ceilings constrain a separate prism estimate
of room-air volume: analytical floor-to-floor volume is not silently labelled
as conditioned air volume. Classified atria and shafts can span floors when
their extents are consistent; conflicts remain review items.

## Exterior fitting and boundaries

Roof skins are reconstructed as piecewise facets. Curved and mansard profiles,
intersecting skins and roof steps are supported; steps create boundary faces,
not interpolated ramps or additional cells. Supported leaning/irregular wall
facets constrain common exterior vertices. Strictly coplanar faces with the same
spatial owners are coalesced. A fit must preserve the designed cell count and
minimum height/volume. Failed fits retain regularised rooms with a diagnostic.

Existing space-boundary geometry constrains layouts and boundary classification.
Analytical patches subdivide native face records by source construction and
physical/virtual support; both owners use the same patch geometry. Patches
retain source IDs, available material layers and their evidence basis. They are
metadata subdivisions, not newly exported `IfcRelSpaceBoundary2ndLevel` entities.
Curtain-wall assembly evidence uses available panel children. Openings are
projected and clipped to host facets, including sloping hosts.

Unknown ground/exterior exposure, unresolved openings and incomplete materials
are reported. A material name/layer does not certify thermal properties.

## Fidelity and limitations

Readiness distinguishes draft, verified geometry and supported boundary semantics.
`simulation_ready` remains false. Loads, schedules, HVAC, thermophysical data and
reviewed energy conversion remain separate requirements.

Footprint, source coverage, opening placement and roof source-surface sample errors
are reported separately. Confidence is a drafting heuristic, not a probability.
Agreement with a source IFC is not independent physical ground truth. Curvature
cannot be recovered more accurately than the source mesh, and the requested
tolerance is a target rather than an accuracy guarantee.

Re-entrant skins, occupied overhangs, very fine facade details and conflicting
source geometry remain limitations. Wall-cap details below the configured spatial
height do not trigger whole-building deformation. All approximations and fallback
decisions require review before simulation.

## Tests and real-file validation

Run the public API and native importer regressions:

```bash
python -m pytest tests/test_IFCAnalyticalModel.py tests/test_IFC.py -o addopts=""
```

Known-answer tests cover separate/touching buildings, partial failures, placement
rotations and millimetres, persisted world/local coordinates, roofs and roof steps,
curved/leaning facades, boundary patches, missing evidence and valid STEP strings.
They explicitly block external-engine and IfcOpenShell imports.

Four optional real-file cases are pinned in
`tests/data/ifc_analytical_real_models.csv`, including source URLs and hashes.
Download/extract the files separately and name them `<id>.ifc`, then run:

```bash
python tools/check_ifc_analytical_models.py --fixtures /path/to/ifc-files --output /path/to/results
```

The runner is offline, isolates each conversion, enforces a timeout and verifies
source integrity. It requires verified persistence and majority source coverage,
and checks the Institute's roof fit. The real files remain source-agreement cases;
independent energy-model review is still required.
