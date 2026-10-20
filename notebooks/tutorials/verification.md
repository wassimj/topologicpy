# Execution verification

**16 of 16 notebooks passed**, without skipped code cells or saved error outputs.

Verified on 16 October 2026 using Windows x64, Python 3.12.14, TopologicPy 0.9.86, PythonOCCBackend and pythonocc-core 8.0.1.1. Source revision: `4be76a6eb4c6aa7f3935ceac95606c7704105be9`.

Each notebook ran through nbclient in a new Jupyter kernel, with an isolated working directory and copies of the supplied data. Notebook structure was validated. Every notebook explicitly asserts the requested version/backend; geometry, routing, database queries, energy-model conversion and training have result checks. The source of every executed cell was compared with the delivered notebook. No helper functions or classes are defined in notebook code, and no tutorial imports or uses the legacy TopologicPy Graph class or Kuzu. Every spatial graph is constructed with TGraph.

| Notebook | Code cells | Code lines | Fresh-kernel total (s) | Cell execution (s) | Result |
|---|---:|---:|---:|---:|---|
| [01_basics.ipynb](01_basics.ipynb) | 3 | 26 | 22.83 | 2.84 | Pass |
| [02_geometry_and_construction.ipynb](02_geometry_and_construction.ipynb) | 3 | 24 | 23.62 | 3.81 | Pass |
| [03_topological_queries.ipynb](03_topological_queries.ipynb) | 3 | 24 | 23.84 | 3.91 | Pass |
| [04_dictionaries.ipynb](04_dictionaries.ipynb) | 3 | 24 | 23.74 | 3.82 | Pass |
| [05_visualisation.ipynb](05_visualisation.ipynb) | 3 | 23 | 24.88 | 4.89 | Pass |
| [06_meshing.ipynb](06_meshing.ipynb) | 3 | 28 | 23.00 | 3.17 | Pass |
| [07_parametric_design.ipynb](07_parametric_design.ipynb) | 3 | 27 | 22.94 | 3.11 | Pass |
| [08_spatial_analysis.ipynb](08_spatial_analysis.ipynb) | 3 | 23 | 19.41 | 3.10 | Pass |
| [09_graphs_and_networks.ipynb](09_graphs_and_networks.ipynb) | 3 | 29 | 22.08 | 2.29 | Pass |
| [10_data_exchange.ipynb](10_data_exchange.ipynb) | 3 | 28 | 22.90 | 3.16 | Pass |
| [11_graph_databases.ipynb](11_graph_databases.ipynb) | 4 | 53 | 22.02 | 2.30 | Pass |
| [12_ifc_and_bim.ipynb](12_ifc_and_bim.ipynb) | 3 | 27 | 23.17 | 3.34 | Pass |
| [13_space_syntax.ipynb](13_space_syntax.ipynb) | 3 | 30 | 22.33 | 2.56 | Pass |
| [14_energy_analysis.ipynb](14_energy_analysis.ipynb) | 3 | 35 | 23.87 | 3.92 | Pass |
| [15_graph_machine_learning.ipynb](15_graph_machine_learning.ipynb) | 3 | 62 | 36.29 | 16.05 | Pass |
| [16_integrated_workflow.ipynb](16_integrated_workflow.ipynb) | 3 | 40 | 22.82 | 3.00 | Pass |

Fresh-kernel totals include kernel startup and shutdown overhead. Cell execution excludes that overhead. Installation time is excluded. Code line counts include imports, version/backend checks, comments and blank lines; no notebook has more than 80 code lines. Most have three code cells.

## Scope of the evidence

This verifies the supplied cells and data in the recorded environment. It is not a promise for other operating systems, different package builds, edited cells or optional exercises. The existing environment had an editable installation of this checkout; this run did not recreate the entire Conda environment from scratch. Setup files pin the main tested versions, and the [machine-readable report](verification.json) records dependency versions, input hashes and timings.

The Plotly notebook produced inline figure data and an offline HTML export; rendering still depends on the notebook viewer and its trust settings. IFC extraction verifies a synthetic room, not arbitrary production IFC files. The RDF example verifies a local dataset and SPARQL round trip, not an external database service. Energy verification covers analytical-model creation and OSM export, not annual simulation. The graph-learning example checks finite, decreasing training loss and prediction shape on toy data, not generalisation or predictive accuracy.

Notebook outputs are cleared in the delivered sources so files remain small and easy to review. Run Restart Kernel and Run All to regenerate them. To repeat verification, install the documented environment and run `python notebooks/tutorials/verify.py` from the repository root. Its generated `verification-local.json` is ignored by Git.
