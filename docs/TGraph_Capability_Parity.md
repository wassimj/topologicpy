# TGraph capabilities carried forward from Graph

All examples use `from topologicpy.TGraph import TGraph`. The legacy Graph class
is not imported. JSONData/JSONString and graph compilation APIs are unchanged.
No method installs an optional dependency automatically.

## Graph operations and matching

`Union`, `Merge`, `Intersect`, `Difference`, `SymmetricDifference`, `XOR`, `Impose`
and `Imprint` accept these matching options:

- `vertexKeys`: one attribute key or a sequence; all values must be present.
- `edgeKeys`: optional attribute keys in addition to endpoints and directedness.
- `matchBy`: `"keys"`, `"coordinates"`, or `"auto"`. Explicit keys take priority;
  coordinate matching uses Euclidean `tolerance` in model units. Auto matching
  checks shared identifiers (`id`, `uuid`, `guid`, `uid`, `name`, `label`) before
  coordinates.
- `useCentroid=True`: place matched vertices at the mean of their coordinates.
- `ambiguous="separate"`: leave ambiguous cross-graph identities separate and
  warn, unless `silent=True`. `ambiguous="error"` raises `ValueError`.

Matching is one-to-one across inputs. Coincident vertices within one input are
never collapsed. Parallel edge identities are paired one-to-one and directed
edges match only the same orientation. Inputs are not modified.

```python
combined = TGraph.Union(a, b, vertexKeys="room_id")
common = TGraph.Intersect(a, b, matchBy="coordinates", tolerance=0.001)
updated = TGraph.Impose(a, b, vertexKeys="room_id")
annotated = TGraph.Imprint(a, b, vertexKeys="room_id")
```

Union retains A's conflicting attributes and fills missing values from B.
Impose combines both structures with B's conflicting attributes taking precedence.
Imprint retains A's structure and applies B's attributes to matching elements;
it adds no B-only elements and retains A's coordinates unless useCentroid=True.
Difference removes matching edges from A, retaining
its vertices. SymmetricDifference/XOR retain unmatched edges from both inputs.

Calls to pre-existing operations without new matching options retain their
established identity rules. Supply matching options for consistent shared rules.
Impose and Imprint now implement their distinct documented operations.

## Paths

`ShortestPaths` remains the existing batch source-target operation.
`KShortestPaths` uses Yen's algorithm for distinct simple routes ordered by cost:

```python
routes = TGraph.KShortestPaths(
    graph, source=0, target=12, k=5, edgeKey="travel_time", returnCost=True)
# Each result is (vertex_indices, cost).
```

Use `mode="out"`, `"in"`, or `"all"`, plus optional `vertexKey`, `edgeFilter`,
`vertexFilter`, `edgeCostFunc` and `vertexCostFunc`. Callbacks receive records.
Vertex costs are charged on arrival, excluding the source. Costs must be finite
and non-negative. Parallel edge sequences count as distinct routes. Optional
`returnVertices` and `returnEdges` append records using the ShortestPath convention.
`timeLimit` limits the search; `returnInfo=True` returns `paths`, `completed` and
`timedOut`. A timed-out result contains only the already ranked prefix.

```python
result = TGraph.LongestPath(
    graph, source=0, target=12, strategy="exact", edgeKey="length",
    timeLimit=2, returnInfo=True)
# path, cost, strategy, completed, timedOut, optimal
```

LongestPath `strategy="auto"` retains the previous diameter approximation when
no endpoints are supplied and uses exact simple-path DFS when endpoints are
specified. `strategy="diameter"` explicitly selects the longest shortest path;
its `optimal` result is always false. `strategy="exact"`/`"dfs"` is exponential
and can time out. A completed exact search is optimal; an interrupted one returns
the best completed source-target route found so far. `vertexKey`, `edgeKey` and
`costKey` (an edge-key alias) select costs. With no edge key, edges cost one.

## Depth, weighted analysis, and cleanup

```python
depths = TGraph.DepthMap(graph, source=0)  # existing single-source traversal
sums = TGraph.DepthMap(graph, method="total", sources=[0, 4],
                       distanceType="metric", key="total_depth")
```

DepthMap returns a stable-index dictionary. `method="total"` defaults to all
active roots. `distanceType="metric"` uses lengths; `weightKey` overrides costs.
For disconnected graphs, `unreachable="ignore"` sums each reachable component,
`"infinity"` gives infinite total depth, and `"raise"` rejects the calculation.
`radius` is an inclusive distance cutoff. `key=None` disables annotation.

- `Connectivity(..., weightKey="capacity")` and
  `VertexDegree(..., weightKey="capacity")` calculate weighted degree. Missing
  weights count as one. Loops count twice in total/undirected degree and once per
  directed in/out degree. Existing Connectivity normalization and colors apply.
- `Integration(..., weightKey="travel_time", method="closeness")` uses weighted
  distances. `distanceType="metric"` selects geometric lengths. The existing
  `method="hh"`/`"depthmap"` implements topological Hillier-Hanson integration;
  weighted/metric requests in that mode raise an explicit error.
- `Laplacian(..., weightKey="weight", normalized=True)` returns
  `D^-1/2 (D-A) D^-1/2`, with zero-degree scaling set to zero. Parallel weights
  sum; loops cancel in D-A. Undirected edges contribute both directions, loops
  once. Directed `mode="out"` uses outgoing adjacency, `"in"` transposes it,
  and `"all"` symmetrizes by addition. Directed normalization uses the same
  algebra; it is not Chung's directed Laplacian.
- `RemoveIsolatedEdges(..., removeVertices=True, removeInvalid=True)` removes
  standalone edge components using original active incidence. It never cascades
  through a chain. Parallel-edge components and loops are retained. Invalid-edge
  cleanup remains independently controllable with `removeInvalid`.
- `AngularBetweenness` aliases AngularChoice. It supports result/color keys and
  forwards radius, mode and angular-algorithm options. Results belong to segment
  graph vertices or ordinary graph edges, matching AngularChoice.

## Maximum flow

```python
value = TGraph.MaximumFlow(graph, source, sink)  # numeric default preserved
result = TGraph.MaximumFlow(
    graph, source, sink, edgeKeyFwd="capacity_forward",
    edgeKeyBwd="capacity_backward", flowKey="flow", residualKey="residual",
    returnDetails=True)
```

Separate capacity keys are supported, with `bidirectional` or per-edge `bidirKey`
overrides. A backward capacity key enables reverse traversal unless overridden.
Detailed results include per-arc edge identity, orientation, capacity, flow and
residual capacity. Optional edge annotations use `forward`/`backward` entries.
Parallel arcs retain their separate identities.

## Matrices, views, and annotations

- `ByAdjacencyMatrix` and `ByAdjacencyMatrixCSVPath` accept `dictionaries` or
  `vertexDictionaries`, exactly one per row. They validate square finite numeric
  matrices. CSV input reports malformed files and returns None. In undirected
  mode, `edgeKeyFwd`/`edgeKeyBwd` store upper/lower triangular weights on one edge
  per unordered pair. Directed matrices create individual traversal arcs.
- `AdjacencyMatrixCSVString` serializes AdjacencyMatrix using standard CSV quoting
  and exposes the same matrix options.
- `Vertices(..., sortBy="label", reverse=True)` and `Edges(...)` sort views;
  sequences of keys are supported, missing values always come last, and stored
  indices/incidence stay unchanged. Geometry conversion follows the view order.
- `Bridges(..., key="bridge")` and `CutVertices(..., key="cut")` optionally mark
  all corresponding active records with 1/0, preserving their result lists.
- `ConnectedComponents(..., key="component")` labels vertices and edges with
  one-based component IDs. Cross-group directed edges receive zero. Existing
  mode and returned component-list behavior are preserved.
- `MeshData(..., includeDictionaries=True, includeIndices=True)` adds copied
  dictionaries and stable IDs in the same order as coordinates and compact
  incidence. The default coordinates/edges-only result is unchanged.

## Graphviz and Pyvis

GraphVizGraph returns DOT text by default, without requiring Graphviz. It supports
`layout`, fixed styling arguments, and `graphAttributes`, `vertexAttributes`,
`edgeAttributes` dictionaries. Their `*AttributeKeys` counterparts map DOT
attribute names to dictionary keys. Keys override constants. Labels and attribute
values are quoted. `asObject=True` returns an optional `graphviz.Source` object.
ExportToGraphVizGraph forwards the same DOT options and writes text explicitly.

```python
dot = TGraph.GraphVizGraph(
    graph, layout="neato", vertexShapeKey="shape", edgeColorKey="color",
    vertexAttributes={"style": "filled"}, vertexFillColorKey="fill")
network = TGraph.PyvisGraph(
    graph, vertexSizeKey="area", vertexColorKey="color",
    vertexGroupKey="category", edgeWeightKey="capacity",
    selectMenu=True, filterMenu=True)
network.write_html("graph.html")
```

PyvisGraph returns a Network and never writes HTML itself. Size/color/group/weight
keys override constants; vertexGroups can map categories through colorScale.
Selection/filter menus and neighbor highlighting are configurable. Optional
Graphviz objects and Pyvis require `python -m pip install "topologicpy[graph]"`;
a missing package raises ImportError with that hint.

Pyvis constructor/menu options follow its [official API documentation](https://pyvis.readthedocs.io/en/latest/documentation.html).
