# PythonOCC Backend Test Parity Plan

## Current State

| Backend | Test Functions | Modules Covered |
|---------|----------------|-----------------|
| TopologicCore | 675 | 43 modules |
| PythonOCC | 29 | 8 modules |
| **Gap** | **646** | **35 modules** |

## Priority 1: Core Geometry (Critical Path)

These modules are foundational and must work correctly:

| Module | TopologicCore Tests | PythonOCC Tests | Gap |
|--------|---------------------|-----------------|-----|
| Vertex | 30 | ~5 | 25 |
| Edge | 25 | ~3 | 22 |
| Wire | 27 | ~4 | 23 |
| Face | 24 | ~4 | 20 |
| Shell | 18 | ~4 | 14 |
| Cell | 19 | ~3 | 16 |
| CellComplex | 15 | ~2 | 13 |
| Cluster | 15 | ~1 | 14 |
| Topology | 38 | ~2 | 36 |

## Priority 2: Operations & Algorithms

| Module | TopologicCore Tests | PythonOCC Tests | Gap |
|--------|---------------------|-----------------|-----|
| Dictionary | 28 | ~3 | 25 |
| Graph | 20 | ~1 | 19 |
| Boolean/CSG | 19 | ~1 | 18 |
| Transformations | 12 | ~1 | 11 |
| GeometricProperties | 31 | 0 | 31 |

## Priority 3: Domain-Specific

| Module | TopologicCore Tests | PythonOCC Tests | Gap |
|--------|---------------------|-----------------|-----|
| IFC | 10 | 0 | 10 |
| EnergyModel | 18 | 0 | 18 |
| Ontology | 17 | 0 | 17 |
| GraphDB | 18 | 0 | 18 |
| KnowledgeGraph | 13 | 0 | 13 |

## Implementation Strategy

### Phase 1: Backend-Agnostic Test Wrapper

Create a test runner that can execute the same tests against both backends:

```python
# tests/conftest.py addition
@pytest.fixture(params=["topologiccore", "pythonocc"], 
                ids=["TopologicCore", "PythonOCC"])
def backend(request):
    if request.param == "pythonocc":
        try:
            from topologicpy.pythonocc_backend import PythonOCCBackend
            Core.SetBackend(PythonOCCBackend())
        except ImportError:
            pytest.skip("PythonOCC not installed")
    else:
        Core.ResetBackend()
    yield
    Core.ResetBackend()
```

### Phase 2: Port Critical Tests

Generate PythonOCC versions of Priority 1 tests:

1. Vertex tests (30 tests)
2. Edge tests (25 tests)
3. Wire tests (27 tests)
4. Face tests (24 tests)
5. Shell tests (18 tests)
6. Cell tests (19 tests)
7. CellComplex tests (15 tests)
8. Topology tests (38 tests)

### Phase 3: Add Missing Module Tests

For modules with 0 PythonOCC tests, create equivalent test files.

## Test Generation Strategy

For each TopologicCore test, create a PythonOCC equivalent:

1. Copy test structure
2. Add backend switch fixture
3. Adjust tolerances if needed (OCCT vs native)
4. Run parity check

## Files to Create

```
tests/
├── pythonocc/
│   ├── test_vertex.py
│   ├── test_edge.py
│   ├── test_wire.py
│   ├── test_face.py
│   ├── test_shell.py
│   ├── test_cell.py
│   ├── test_cell_complex.py
│   ├── test_topology.py
│   ├── test_dictionary.py
│   ├── test_graph.py
│   └── ...
```

## Success Criteria

- [ ] 100% test function parity for Priority 1 modules
- [ ] 100% test function parity for Priority 2 modules
- [ ] All tests pass on both backends
- [ ] CI pipeline runs both backend test suites
