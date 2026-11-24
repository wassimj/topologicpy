# TopologicPy canonical test suite

This directory is the single canonical public-API test suite for TopologicPy.
It replaces both the former current and donor test trees.

## Principles

1. PythonOCC is the sole geometry kernel. Exact curve, NURBS and native topology contracts remain mandatory.
2. Tests assert public API semantics and geometry invariants; do not silently approximate geometry or weaken assertions.
3. Keep production-risk regression fixtures permanently.

## Running the suite

Install PythonOCC in the active environment as described in the repository README, then install the test extras:

```bash
python -m pip install -e ".[test]"
python -m pytest tests
```

For sequential execution use `python -m pytest tests -o addopts="" -v`.
The `pythonocc_only` marker identifies native contracts; it no longer selects a backend.

## Test layers

- `test_<Module>.py`: broad public API contracts.
- focused `*_Curve*`, `*_Native*`, `*_STEP`, `*_TPY`, `*_Tessellate`, and
  `*_regressions` modules: high-value exactness/regression contracts.
- `test_Ontology_conformance.py`: end-to-end RDF export contracts (see below).
- `stress_tests/`: backend-neutral deterministic stress tests.

## Ontology export contracts

`test_Ontology.py` unit-tests the vocabulary helpers in isolation.
`test_Ontology_conformance.py` complements it by exercising *real* exports
against the *shipped* `ontology/topologicpy.ttl`, guarding the invariants that
keep the export layer and the ontology in sync:

- **Conformance.** Every `top:` term a real TGraph or IFC export emits must be
  declared in the ontology, including when unknown dictionary keys are present
  (they must route to `dict:`, never `top:`).
- **Round-trip.** A TTL export re-imported through `KnowledgeGraph` must
  preserve triple, node and relationship counts and literal datatypes.
- **SHACL.** A real export must satisfy the Node/Relationship/Graph contract
  shapes (`startsAt`/`endsAt` resolving to `Node`, typed coordinates, etc.).
- **Competency questions.** Node/relationship counts, absence of dangling
  endpoints, adjacency traversal, and IFC `ifcClass`/`ifcGUID` retrieval must
  answer via SPARQL.
- **Coherence.** The shipped ontology must parse, have no dangling `top:`
  domain/range targets, and survive a pure-Python OWL-RL closure with no term
  entailed to be `owl:Nothing`.

These tests use `pytest.importorskip` and skip cleanly when `rdflib`, `pyshacl`,
`owlrl`, `ifcopenshell`, or a geometry backend are unavailable. `rdflib`,
`pyshacl` and `owlrl` are declared in the `test` optional-dependency extra so
the core contracts run in CI; the IFC tests additionally require `ifcopenshell`
and skip without it.

Full DL reasoning (HermiT, Pellet) and the external pitfall/FAIR scanners
(OOPS!, FOOPS!) are intentionally **not** part of the automated suite: they
require Java or live web services and are non-deterministic in CI. Run them as
an occasional manual audit when the vocabulary changes materially.

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
