topologicpy.TGraph module
=========================

Axial and segment graphs
------------------------

``TGraph.AxialGraph(edges)`` represents each complete axial edge as a node and
each valid 3D contact as a graph edge. Projected crossings on different floors
do not connect. ``TGraph.SegmentGraph(edges)`` splits the input edges at contacts
and represents each resulting segment as a node. Both constructors require
straight, non-degenerate edges. Axial graphs accept overlapping collinear
interiors, duplicates and contained Edges, connecting each pair once while
retaining a separate node and originator for each supplied Edge. Segment graphs
require overlapping interiors to be merged before splitting.
They construct graphs from supplied geometry; generation belongs to
``Face.AxialEdges`` (also available as ``Face.AxialLines``).

Axial nodes retain the input edge. Segment nodes retain their generated segment,
with stable ``edge_id`` and ``parent_edge_id`` UUIDs and a live
``parent_originator`` reference to the input edge. Segment connections store
``angular_weight`` in quarter-turn units. Ordinary centrality methods can use
this key directly, and the angular wrappers recognize the segment graph tag.

``Choice`` computes vertex betweenness; ``Integration`` computes closeness-based
integration. The latter does not implement Hillier-Hanson normalization.
``Choice(normalize=False)`` returns raw counts, while True applies NetworkX
betweenness normalization.

For segment analysis, one may also provide a junction graph whose edges represent segments.
``AngularChoice``, ``AngularBetweenness``, ``AngularIntegration``, and
``AngularConnectivity`` return values in active-edge order and store results
on those edges. With a ``SegmentGraph``, they instead return active-node values
and store results on those nodes. This corrects the previous aliases, which operated on vertices.
Angular integration is closeness-based rather than NAIN. Segment connectivity
counts adjacent segments; it does not weight turning angles.

Closeness and betweenness accept an inclusive ``radius`` cutoff. None retains
global analysis. A cutoff uses hops without a weight key, accumulated costs
with a weight key, or quarter-turn units for angular analysis. Radius limits
source-to-destination shortest-path distance; it does not select an induced
neighborhood around each measured vertex. Normalization continues to use the
full graph order. These methods retain their undirected analysis convention.

Angular cost uses three-dimensional deflection: straight continuation costs
zero in principle and a right turn costs one. Straight transitions use a
positive floor of 1e-9 quarter-turn units to prevent zero-cost cycles in
shortest-path counting. Consequently, even a straight transition is outside
a radius of zero. Collinear paths can have very large unnormalized angular
closeness values; the floor is an approximation rather than a physical cost.

Originators and colouring source geometry
----------------------------------------

An originator is distinct from the node's display representation. ``AddVertex``
accepts ``originator=topology`` and automatically links a topology input or
representation when no serialized link already exists. ``ByTopology`` and
``BySpatialRelationships`` link the original topology even when its graph
representation is a representative point. ``ByVerticesEdges`` retains source
links independently of its ``storeRepresentations`` option.

The record stores ``originator``, ``originator_id`` and ``originator_type``;
the latter two are also mirrored in its dictionary. Identity uses the persistent
``Topology.UUID`` stored under ``uuid`` on the source. Creating a link creates
that UUID if it is absent. Coincident geometry and graph indices never determine
identity. ``SetVertexOriginator`` assigns or clears a link, ``VertexOriginator``
retrieves one source, and ``Originators`` returns sources in active-node order.

``Copy``, ``Subgraph``, tree construction and spanning-tree construction retain
live source references. ``ToPython(includeRepresentations=True)`` and
``FromPython`` preserve references in memory. Default JSON-compatible export
retains UUID/type metadata independently of mutable analysis dictionaries,
but cannot retain live Python objects. Reconnect with
``Originators(graph, originators=source_topologies)`` after import. Distinct
objects with the same UUID are ambiguous and are rejected when resolving an
explicit source list. Geometry copies inheriting a UUID must therefore be
disambiguated by the caller. Synthetic nodes with no single source, such as
aggregate quotient nodes, require an explicit source link before transfer.

``TransferDictionariesToOriginators`` merges selected node fields into source
dictionaries and preserves existing unrelated fields and UUIDs. Missing or
ambiguous links fail preflight before dictionary writes. For repeated sources,
identical values are accepted; conflicting values require an explicit
``aggregation`` (first, last, sum, mean, min or max). ``overwrite=False`` keeps
existing source values. ``returnReport=True`` reports updates and errors.
Preflight is atomic; a backend write failure can report partial updates.

For example::

    edges = Face.AxialEdges(face)
    graph = TGraph.AxialGraph(edges=edges)
    TGraph.BetweennessCentrality(graph, key="choice", colorKey="color")
    coloured_edges = TGraph.TransferDictionariesToOriginators(
        graph, keys=["choice", "color"])

.. automodule:: topologicpy.TGraph
   :members:
   :undoc-members:
   :show-inheritance:
