# Universal TopologicPy stress tests

These stress tests are intentionally backend-blind. They exercise the public
TopologicPy API and use analytic geometry, topology invariants, metric preservation,
round trips, and deterministic random seeds as their oracles.

## Backend policy

Run the same suite in each backend job/process. Do not switch Core backends inside
the test suite itself; Core backend state is process-global and stress tests should
measure the backend selected by the surrounding test environment.

Examples:

- TopologicCore job: run with the environment/configuration that selects TopologicCore.
- PythonOCC job: set `TOPOLOGICPY_CORE_BACKEND=pythonocc` before pytest starts.

This design also allows future backends to run the exact same files without adding
backend names, skips, xfails, or backend-specific expected values.

## Stress volume controls

The original environment variables are retained:

- `TOPOLOGICPY_ROTATE_STRESS_CASES`
- `TOPOLOGICPY_DISTANCE_STRESS_CASES`
- `TOPOLOGICPY_TRANSFORM_STRESS_CASES`
- `TOPOLOGICPY_TRANSLATE_STRESS_CASES`
- `TOPOLOGICPY_TRIANGULATE_STRESS_CASES`
- `TOPOLOGICPY_TWIST_STRESS_CASES`

## Deliberate change

Three ShortestDistance native-failure tests were removed from the stress layer.
Their expected behaviour is intentionally backend-policy-dependent: TopologicCore may
fall back to its compatibility algorithm after a native distance failure, while newer
backends expose native failure. Those belong in focused backend-policy unit tests, not
in a universal stress suite.
