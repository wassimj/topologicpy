# Universal TopologicPy stress tests

These stress tests are intentionally backend-blind. They exercise the public
TopologicPy API and use analytic geometry, topology invariants, metric preservation,
round trips, and deterministic random seeds as their oracles.

## Geometry kernel

Run against PythonOCC using the repository's standard test environment.

## Stress volume controls

The original environment variables are retained:

- `TOPOLOGICPY_ROTATE_STRESS_CASES`
- `TOPOLOGICPY_DISTANCE_STRESS_CASES`
- `TOPOLOGICPY_TRANSFORM_STRESS_CASES`
- `TOPOLOGICPY_TRANSLATE_STRESS_CASES`
- `TOPOLOGICPY_TRIANGULATE_STRESS_CASES`
- `TOPOLOGICPY_TWIST_STRESS_CASES`

