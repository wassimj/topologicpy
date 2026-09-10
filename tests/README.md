# TopologicPy canonical test suite

This directory is the single canonical public-API test suite for TopologicPy.
It replaces both the former current and donor test trees.

## Principles

1. **One suite, both backends.** The same tests are collected for PythonOCC and
   TopologicCore. Backend-specific exact/native capabilities use centralized
   markers rather than separate test trees.
2. **Current API is authoritative.** Historical donor tests are retained only
   when they still express a valid current contract.
3. **No silent geometric approximation.** PythonOCC exact curve/NURBS tests
   assert preservation of exact geometry. TopologicCore capability tests assert
   explicit unsupported behavior where appropriate.
4. **Public contracts first.** Tests should prefer TopologicPy public methods and
   semantic/topological invariants over backend implementation details.
5. **Regressions stay permanent.** The production-risk Slice, WireByPath,
   shapeless-Cluster, periodic seam, native editing, STEP/TPY, and curved-grid
   cases are retained explicitly.

## Backend selection

Plain pytest selects the first installed backend, preferring PythonOCC:

```bash
python -m pytest tests
```

Run every installed backend in isolated subprocesses:

```bash
python tests/run_backends.py
```

Explicit runs:

```bash
python -m pytest tests --backend pythonocc
python -m pytest tests --backend topologic_core
```

`-n auto` may be added when pytest-xdist is installed.

## Backend capability markers

Use:

```python
@pytest.mark.pythonocc_only
```

for exact/native PythonOCC capability tests, and:

```python
@pytest.mark.topologiccore_only
```

for tests of an intentionally TopologicCore-specific fallback contract.

Do not compute backend-specific `skipif` expressions at module import time.
The root `conftest.py` applies these markers after backend selection, preventing
collection-order/backend-environment errors.

Known backend **defects**, as opposed to intentional capability differences,
remain centralized in `backend_exceptions.json`.

## Test layers

- `test_<Module>.py`: broad public API contracts.
- focused `*_Curve*`, `*_Native*`, `*_STEP`, `*_TPY`, `*_Tessellate`, and
  `*_regressions` modules: high-value exactness/regression contracts.
- `stress_tests/`: backend-neutral deterministic stress tests.

## Donor decisions

Restored from donor because broad coverage was missing:

- Cell
- Cluster
- Shell
- Topology
- Plotly
- CellComplex regressions
- stress tests
- advanced public STEP cases
- high-level curved Wire constructors
- selected Face curved operations
- selected meshing regressions

Deliberately not restored:

- old `Grid.Square`, `Rectangular`, `ByDivisions`, `Structural`, `TileLayout`,
  and `Vertices` tests: `Grid.OnFace` is now the only public Grid method;
- `Edge.AdjacentEdges` donor test: that convenience API was intentionally not
  restored;
- old `SemanticManager` / `Content` tests: that architecture was intentionally
  abandoned; current Topology/Aperture relationship tests are authoritative;
- stale `Topology.RemoveCoplanarFaces(..., polyhedron=...)`: the public method
  intentionally uses the mature shapeless-container-aware path and has no
  `polyhedron` argument;
- private STEP codec-helper tests and private mesh-helper implementation tests.

## Critical permanent sentinels

The suite must continue to protect:

- `test_Topology_SliceGridRegression.py`;
- `test_TGraph_WireByPathDirection.py`;
- native RemoveEdges/RemoveVertices/RemoveFaces exactness;
- periodic seam-aware OpenEdges;
- curved/NURBS Face differential geometry and area;
- exact curved Shell/Cell/CellComplex construction;
- STEP exact BREP exchange;
- TPY BREP + dictionary + Content/Aperture persistence;
- `Grid.OnFace` exact surface isocurves under PythonOCC.

The external production floor-plan Slice checkpoint remains **2184 Faces** and
should be rerun manually at major topology-kernel milestones.
