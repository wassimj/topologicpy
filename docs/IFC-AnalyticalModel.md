# IFC analytical spatial models

`IFC.AnalyticalModel` reconstructs simplified spatial volumes using evidence
strictly descended from an IFC building and storey. It returns one verified
CellComplex per usable building and reports approximate room and boundary
fidelity. The result is a geometric approximation, not an energy simulation.

```python
from topologicpy.IFC import IFC

result = IFC.AnalyticalModel("building.ifc", coordinateFrame="local")
for building in result.building_models:
    print(building.building_id, building.statistics, building.diagnostics)
```

The input can be an IFC path or `IFCModel`. `coordinateFrame` accepts `local`
or `world` (the default). Native geometry closure and shared-face verification
are always performed; `verify` is accepted for API compatibility.

## Options

| Option | Default | Meaning |
|---|---:|---|
| `minimum_clear_width` | 0.6 m | Minimum clear room width |
| `minimum_clear_height` | 1.8 m | Minimum clear room height |
| `simplification` | 0.1 m | Geometric simplification |
| `drafting_grid` | 0.05 m | Initial partition grid |
| `tolerance` | 0.00001 m | Native modelling tolerance |
| `network_wall_thickness` | 0.35 m | Shared floor-line contraction reach |
| `network_angular_tolerance` | 2.5° | Direction-preservation threshold |
| `role_overrides` | `{}` | Overrides for spatial roles |

Room divisions, volume, roof detail, and apertures may differ from the IFC.
`IFC.AnalyticalModelByPlan` remains available for rebuilding an editable
plan; `IFC.AnalyticalModelSave` and `IFC.AnalyticalModelLoad` handle persistence.
