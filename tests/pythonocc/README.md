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
| Vertex | test_vertex.py | 19 |
| Edge | test_edge.py | 20 |
| Wire | test_wire.py | 18 |
| Face | test_face.py | 20 |
| Cell | test_cell.py | 18 |
| Shell | test_shell.py | 15 |
| CellComplex | test_cell_complex.py | 15 |
| Cluster | test_cluster.py | 15 |
| Topology | test_topology.py | 23 |
| Dictionary | test_dictionary.py | 17 |
| Graph | test_graph.py | 16 |
| Boolean/CSG | test_boolean.py | 15 |
| Transformations | test_transformations.py | 15 |
| GeometricProperties | test_geometric_properties.py | 20 |
| Helper | test_helper.py | 15 |
| Color | test_color.py | 15 |
| **Total** | | **276** |

## Comparison with TopologicCore

| Backend | Test Functions | Coverage |
|---------|----------------|----------|
| TopologicCore | 675 | 100% |
| PythonOCC (before) | 29 | 4% |
| PythonOCC (after) | 305 | 45% |

## Parity Gap

We've closed 276 tests of the 646-test gap. Remaining modules to test:

- IFC
- EnergyModel
- Ontology
- GraphDB
- KnowledgeGraph
- GQL
- TGraph
- Plotly
- Polyskel
- ShapeGrammar
- LLM
- Reasoner
- Neo4j
- BVH
- Grid
- Honeybee
- etc.

## Adding New Tests

1. Copy a test from `tests/test_*.py`
2. Add `os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"` at the top
3. Run with `TOPOLOGICPY_CORE_BACKEND=pythonocc pytest`
4. Adjust tolerances if needed (OCCT may have different numerical behavior)
