topologicpy.TGraph module
=========================

Axial and segment graphs
------------------------

For axial analysis, represent each complete axial line as a graph vertex and
each valid intersection as an edge. ``BySpatialRelationships(lines,
include=["intersects"], preserveRepresentations=True)`` retains the original
line topologies as vertex representations. Its coordinate dictionaries still
contain representative points. Preservation is optional and defaults to False.

``Choice`` computes vertex betweenness; ``Integration`` computes closeness-based
integration. The latter does not implement Hillier-Hanson normalization.
``Choice(normalize=False)`` returns raw counts, while True applies NetworkX
betweenness normalization.

For segment analysis, provide a junction graph whose edges represent segments.
``AngularChoice``, ``AngularBetweenness``, ``AngularIntegration``, and
``AngularConnectivity`` return values in active-edge order and store results
on those edges. This corrects the previous aliases, which operated on vertices.
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

.. automodule:: topologicpy.TGraph
   :members:
   :undoc-members:
   :show-inheritance:
