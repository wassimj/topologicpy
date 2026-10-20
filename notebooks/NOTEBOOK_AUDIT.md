# Notebook audit and tutorial replacement

Audit date: 16 October 2026. Source checkout: `4be76a6eb4c6aa7f3935ceac95606c7704105be9`.

## Scope and method

Inventoried the 78 existing notebooks under `notebooks/`, excluding checkpoint directories, before adding the new tutorials. Read notebook JSON, counted code cells/lines, scanned code for path/install/service patterns and inspected saved error outputs. Reviewed the local v0.9.86 API implementations and executed the replacement examples against pythonocc. The old notebooks were **not all executed**: this is a static legacy audit and a runtime-verified replacement sequence, not certification of the legacy collection.

The valid legacy notebooks contain 20,129 code lines. The invalid notebook is excluded from that line total. Automated flags overlap and identify review risks, not proof that an entire notebook is unusable:

- 69 contain machine-specific path patterns.
- 37 contain more than 150 code lines.
- 31 contain package-installation commands or strings.
- 9 contain saved error outputs.
- 3 reference external service configuration.
- 1 has invalid notebook JSON: `ACADIA23_BuildingGenerator.ipynb`, parser error at line 186, column 51.

For example, `ACADIA23_BGR_TrainingValidationTesting.ipynb` modifies `sys.path`, reads datasets from `C:/Users/wassimj/...`, and writes a trained model to that user's Desktop. Such examples need more than an import update to become portable beginner tutorials.

## Curriculum decisions

The new entry point is [tutorials/README.md](tutorials/README.md), with one short, self-contained gateway notebook per agreed theme in the user's exact order. Examples use direct API calls, ordinary loops, small synthetic data and explicit checks. They avoid helper functions/classes, long benchmarks, downloaded datasets and in-notebook installation. The supplied exercises are optional and are not included in the execution guarantee.

All spatial graph creation uses **TGraph**, as requested. No new tutorial uses TopologicPy's legacy Graph class or Kuzu. Database coverage uses a local RDF/SPARQL store with a TGraph as its source. The PyG Data object is a tensor container converted from TGraph, rather than another spatial modelling API.

Legacy workshops, benchmarks, RAG prototypes, trained-model files and specialist demonstrations remain unchanged for reference. They are not presented as lessons in the new sequence. Retaining them also preserves original research material and existing links. A future deletion or archive migration can be considered separately.

The intended thematic mapping below is a filename-based triage aid; it is not an assertion that every notebook fits exclusively into one group. All rows have the same disposition: **retain as legacy reference; use the new numbered tutorial as the beginner entry point**. Root-folder non-notebook assets were left untouched; the new tutorials carry their own documented inputs.

## Version-specific findings

Execution of early replacement drafts exposed two API issues in this checkout: the legacy Graph choice alias passes an unsupported `method` argument to its betweenness implementation, and the Kuzu convenience wrapper uses `AS exists`, rejected as a reserved identifier by Kuzu 0.11.3. These were observed during testing, not inferred from notebook names. Following the user's direction, both classes were removed from the replacement design, and no library-source workaround was introduced.

Energy model defaults resolve template/weather paths beneath the installed module, which do not match this checkout's asset location. Tutorial 14 supplies explicit paths to bundled copies and verifies the result. The IFC tutorial uses the current built-in importer and a 24-entity local STEP fixture. No IfcOpenShell dependency is needed for that notebook.

Only notebook code, sample inputs, setup documentation and verification support were added. No public library API was modified. See [verification.md](tutorials/verification.md) for execution results, exact environment and limits.

## Original notebook inventory

Code line counts include comments and blank lines. Saved outputs were not counted as code. The malformed JSON file cannot be counted reliably.

| Original notebook | Code cells | Code lines | Suggested theme | Static review flags |
|---|---:|---:|---|---|
| [ACADIA23_BGR_TrainingValidationTesting.ipynb](ACADIA23_BGR_TrainingValidationTesting.ipynb) | 14 | 73 | Graph Machine Learning | machine-specific paths |
| [ACADIA23_BuildingGenerator.ipynb](ACADIA23_BuildingGenerator.ipynb) | 0 | 0 | Parametric Design and Optimisation | invalid notebook JSON |
| [ACADIA23_Day1.ipynb](ACADIA23_Day1.ipynb) | 18 | 56 | Basics | machine-specific paths; installs in notebook |
| [ACADIA23_GML_D02_Topologicpy_Tutorial.ipynb](ACADIA23_GML_D02_Topologicpy_Tutorial.ipynb) | 12 | 85 | Basics | machine-specific paths; installs in notebook |
| [ACADIA23_GraphCSVImport.ipynb](ACADIA23_GraphCSVImport.ipynb) | 4 | 53 | Data Exchange and Interoperability | machine-specific paths; installs in notebook |
| [AdjacentVerticesByCompassDirection.ipynb](AdjacentVerticesByCompassDirection.ipynb) | 4 | 70 | Topological Queries and Relationships | machine-specific paths |
| [AlignVertices.ipynb](AlignVertices.ipynb) | 8 | 160 | Spatial and Geometric Analysis | machine-specific paths; long example (>150 code lines) |
| [ANN-new.ipynb](ANN-new.ipynb) | 16 | 84 | Graph Machine Learning | machine-specific paths; installs in notebook |
| [Betweenness_Centrality.ipynb](Betweenness_Centrality.ipynb) | 1 | 111 | Graphs and Networks | No heuristic flags |
| [Building_Tower.ipynb](Building_Tower.ipynb) | 13 | 88 | Parametric Design and Optimisation | machine-specific paths |
| [BVH.ipynb](BVH.ipynb) | 2 | 53 | Spatial and Geometric Analysis | machine-specific paths |
| [Chessboard.ipynb](Chessboard.ipynb) | 7 | 420 | Parametric Design and Optimisation | machine-specific paths; long example (>150 code lines) |
| [Clustering.ipynb](Clustering.ipynb) | 6 | 54 | Graph Machine Learning | machine-specific paths; installs in notebook |
| [CombinatorialSubComponents.ipynb](CombinatorialSubComponents.ipynb) | 11 | 90 | Topological Queries and Relationships | machine-specific paths |
| [CSG.ipynb](CSG.ipynb) | 9 | 42 | Geometry and Construction | machine-specific paths |
| [DGL-Classification-CSV.ipynb](DGL-Classification-CSV.ipynb) | 14 | 65 | Graph Machine Learning | machine-specific paths |
| [DGL-Classification-DGCNN.ipynb](DGL-Classification-DGCNN.ipynb) | 14 | 62 | Graph Machine Learning | machine-specific paths |
| [Edge_Face_Metrics.ipynb](Edge_Face_Metrics.ipynb) | 1 | 36 | Spatial and Geometric Analysis | machine-specific paths |
| [EnergyModel.ipynb](EnergyModel.ipynb) | 13 | 169 | Energy Analysis | machine-specific paths; installs in notebook; saved error output; long example (>150 code lines) |
| [ExportToGLB.ipynb](ExportToGLB.ipynb) | 2 | 643 | Data Exchange and Interoperability | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [FaceByOffsetArea.ipynb](FaceByOffsetArea.ipynb) | 5 | 203 | Spatial and Geometric Analysis | machine-specific paths; installs in notebook; saved error output; long example (>150 code lines) |
| [GA.ipynb](GA.ipynb) | 4 | 139 | Parametric Design and Optimisation | machine-specific paths; installs in notebook |
| [GQL.ipynb](GQL.ipynb) | 29 | 467 | Graph Databases | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [Graph_Bridges_Cuts.ipynb](Graph_Bridges_Cuts.ipynb) | 3 | 155 | Graphs and Networks | machine-specific paths; long example (>150 code lines) |
| [Graph_Partition.ipynb](Graph_Partition.ipynb) | 2 | 184 | Graphs and Networks | machine-specific paths; long example (>150 code lines) |
| [Graph_TGraph_NetworkX_Benchmark.ipynb](Graph_TGraph_NetworkX_Benchmark.ipynb) | 13 | 661 | Graphs and Networks | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [GraphBooleans.ipynb](GraphBooleans.ipynb) | 4 | 168 | Graphs and Networks | machine-specific paths; long example (>150 code lines) |
| [GraphByIFCPath.ipynb](GraphByIFCPath.ipynb) | 9 | 84 | IFC and BIM | machine-specific paths |
| [GraphGrammars.ipynb](GraphGrammars.ipynb) | 1 | 23 | Parametric Design and Optimisation | machine-specific paths |
| [GraphIsomrphismMatching.ipynb](GraphIsomrphismMatching.ipynb) | 12 | 565 | Graphs and Networks | machine-specific paths; saved error output; long example (>150 code lines) |
| [GraphRAG01.ipynb](GraphRAG01.ipynb) | 3 | 26 | Graph Databases | installs in notebook |
| [GraphRAG02.ipynb](GraphRAG02.ipynb) | 9 | 190 | Graph Databases | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [GraphRAG03.ipynb](GraphRAG03.ipynb) | 3 | 320 | Graph Databases | machine-specific paths; long example (>150 code lines) |
| [GraphRAGTopologicPy_POC-3_Claude.ipynb](GraphRAGTopologicPy_POC-3_Claude.ipynb) | 18 | 921 | Graph Databases | machine-specific paths; long example (>150 code lines); external service/configuration |
| [GraphViz.ipynb](GraphViz.ipynb) | 9 | 196 | Visualisation and Communication | machine-specific paths; long example (>150 code lines) |
| [IFC4-RV-Exporter.ipynb](IFC4-RV-Exporter.ipynb) | 14 | 1679 | IFC and BIM | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [IFC_Explainable_Semantic_Building_Graph_Tutorial.ipynb](IFC_Explainable_Semantic_Building_Graph_Tutorial.ipynb) | 20 | 850 | IFC and BIM | machine-specific paths; saved error output; long example (>150 code lines) |
| [IFC_Healing_Semantic_Relationships.ipynb](IFC_Healing_Semantic_Relationships.ipynb) | 11 | 189 | IFC and BIM | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [IFC_SpatialRelationshipsGraph.ipynb](IFC_SpatialRelationshipsGraph.ipynb) | 12 | 86 | IFC and BIM | machine-specific paths; installs in notebook |
| [import_HBJSON.ipynb](import_HBJSON.ipynb) | 5 | 54 | Energy Analysis | machine-specific paths; installs in notebook |
| [Interpolation.ipynb](Interpolation.ipynb) | 8 | 53 | Spatial and Geometric Analysis | machine-specific paths |
| [Isovist_Metrics.fixed.ipynb](Isovist_Metrics.fixed.ipynb) | 3 | 153 | Space Syntax Analysis | machine-specific paths; long example (>150 code lines) |
| [Isovist_Metrics.ipynb](Isovist_Metrics.ipynb) | 3 | 153 | Space Syntax Analysis | machine-specific paths; long example (>150 code lines) |
| [Kuzu.ipynb](Kuzu.ipynb) | 15 | 100 | Graph Databases | machine-specific paths; installs in notebook |
| [LineGraph.ipynb](LineGraph.ipynb) | 17 | 1144 | Graphs and Networks | machine-specific paths; saved error output; long example (>150 code lines) |
| [MEP.ipynb](MEP.ipynb) | 4 | 66 | IFC and BIM | machine-specific paths |
| [MEP01.ipynb](MEP01.ipynb) | 10 | 156 | IFC and BIM | machine-specific paths; long example (>150 code lines) |
| [MEP02.ipynb](MEP02.ipynb) | 4 | 60 | IFC and BIM | machine-specific paths |
| [mesh_plot.ipynb](mesh_plot.ipynb) | 3 | 50 | Meshing and Discretisation | machine-specific paths |
| [MSD.ipynb](MSD.ipynb) | 9 | 235 | Space Syntax Analysis | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [MSD02.ipynb](MSD02.ipynb) | 11 | 433 | Space Syntax Analysis | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [Neo4j.ipynb](Neo4j.ipynb) | 16 | 213 | Graph Databases | machine-specific paths; long example (>150 code lines); external service/configuration |
| [NEO4J_House_Export.ipynb](NEO4J_House_Export.ipynb) | 11 | 435 | Graph Databases | machine-specific paths; installs in notebook; saved error output; long example (>150 code lines); external service/configuration |
| [Outposts_Graph.ipynb](Outposts_Graph.ipynb) | 10 | 66 | Graphs and Networks | machine-specific paths |
| [pagerank.ipynb](pagerank.ipynb) | 10 | 71 | Graphs and Networks | machine-specific paths |
| [PDF_Import.ipynb](PDF_Import.ipynb) | 3 | 61 | Data Exchange and Interoperability | machine-specific paths |
| [PrincipalAxis_CanonicalMatrix.ipynb](PrincipalAxis_CanonicalMatrix.ipynb) | 6 | 69 | Spatial and Geometric Analysis | machine-specific paths |
| [Provenance.ipynb](Provenance.ipynb) | 1 | 546 | Dictionaries and Semantic Data | long example (>150 code lines) |
| [PyG-new.ipynb](PyG-new.ipynb) | 17 | 117 | Graph Machine Learning | machine-specific paths; installs in notebook |
| [PyG_BGR_Classification.ipynb](PyG_BGR_Classification.ipynb) | 14 | 108 | Graph Machine Learning | machine-specific paths |
| [PythonOCC_benchmark_01.ipynb](PythonOCC_benchmark_01.ipynb) | 1 | 768 | Basics | long example (>150 code lines) |
| [Pyvis.ipynb](Pyvis.ipynb) | 9 | 47 | Visualisation and Communication | machine-specific paths; installs in notebook |
| [QuotientGraph.ipynb](QuotientGraph.ipynb) | 2 | 63 | Graphs and Networks | machine-specific paths |
| [RDF_BOT_Export.ipynb](RDF_BOT_Export.ipynb) | 12 | 465 | Graph Databases | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [RDF_BOT_Import.ipynb](RDF_BOT_Import.ipynb) | 10 | 61 | Graph Databases | machine-specific paths |
| [Roof.ipynb](Roof.ipynb) | 3 | 27 | Geometry and Construction | No heuristic flags |
| [Semantic_MEP_Space_Coordination_Reasoner.ipynb](Semantic_MEP_Space_Coordination_Reasoner.ipynb) | 15 | 780 | IFC and BIM | machine-specific paths; long example (>150 code lines) |
| [ShapeGrammar_WithProvenance_Tutorial.ipynb](ShapeGrammar_WithProvenance_Tutorial.ipynb) | 32 | 396 | Parametric Design and Optimisation | long example (>150 code lines) |
| [ShellByDisjointFaces.ipynb](ShellByDisjointFaces.ipynb) | 2 | 70 | Geometry and Construction | machine-specific paths; installs in notebook |
| [Shortest_Paths_With_Capacities.ipynb](Shortest_Paths_With_Capacities.ipynb) | 14 | 1864 | Graphs and Networks | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [Space_Syntax_Gallery.ipynb](Space_Syntax_Gallery.ipynb) | 21 | 316 | Space Syntax Analysis | long example (>150 code lines) |
| [Space_Syntax_Gallery_2.ipynb](Space_Syntax_Gallery_2.ipynb) | 22 | 318 | Space Syntax Analysis | saved error output; long example (>150 code lines) |
| [StreetNetworkAnalysis.ipynb](StreetNetworkAnalysis.ipynb) | 15 | 269 | Space Syntax Analysis | machine-specific paths; installs in notebook; long example (>150 code lines) |
| [SubgraphMatching.ipynb](SubgraphMatching.ipynb) | 6 | 112 | Graphs and Networks | machine-specific paths |
| [TopologicPy_Reasoner_Demo.ipynb](TopologicPy_Reasoner_Demo.ipynb) | 10 | 105 | IFC and BIM | machine-specific paths; installs in notebook |
| [Unsupervised_Learning.ipynb](Unsupervised_Learning.ipynb) | 38 | 271 | Graph Machine Learning | machine-specific paths; installs in notebook; saved error output; long example (>150 code lines) |
| [WireByHoleArea.ipynb](WireByHoleArea.ipynb) | 5 | 228 | Spatial and Geometric Analysis | machine-specific paths; installs in notebook; saved error output; long example (>150 code lines) |
| [WireByOffset_VariableOffsets.ipynb](WireByOffset_VariableOffsets.ipynb) | 6 | 106 | Spatial and Geometric Analysis | machine-specific paths; installs in notebook |
