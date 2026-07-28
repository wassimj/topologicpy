# PythonOCC Backend Parity Tests

These tests verify that the PythonOCC backend produces identical results to the TopologicCore backend.

## Running Tests

```bash
# Run all PythonOCC parity tests
TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/ -v

# Run a specific test file
TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_vertex.py -v

# Run with coverage
TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/ --cov=topologicpy.pythonocc_backend
```

## Test Coverage

| Module | Test File | Tests |
|--------|-----------|-------|
| Vertex | test_vertex.py | 20 |
| Edge | test_edge.py | 20 |
| Wire | test_wire.py | 15 |
| Face | test_face.py | 15 |
| Cell | test_cell.py | 15 |
| Topology | test_topology.py | 20 |
| Dictionary | test_dictionary.py | 15 |
| Graph | test_graph.py | 15 |
| **Total** | | **135** |

## Comparison with TopologicCore

| Backend | Test Functions | Coverage |
|---------|----------------|----------|
| TopologicCore | 675 | 100% |
| PythonOCC (before) | 29 | 4% |
| PythonOCC (after) | 164 | 24% |

## Parity Gap

We've closed 135 tests of the 646-test gap. Remaining modules to test:

- Shell, CellComplex, Cluster
- Boolean/CSG operations
- Transformations
- GeometricProperties
- IFC, EnergyModel, Ontology
- GraphDB, KnowledgeGraph, etc.

## Adding New Tests

1. Copy a test from `tests/test_*.py`
2. Add `os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"` at the top
3. Run with `TOPOLOGICPY_CORE_BACKEND=pythonocc pytest`
4. Adjust tolerances if needed (OCCT may have different numerical behavior)
