topologicpy.Face module
=======================

Axial edges
-----------

``Face.AxialEdges`` generates maximal straight visibility edges within a planar
polygonal face, respecting holes and optional coplanar polygonal face/wire
obstacles. ``Face.AxialLines`` is an alias of the same function. Results retain
the original face plane, including tilted and vertical planes. Curved boundaries
and non-planar input are rejected.

The first implementation is an approximate sampled map. It generates candidates
from boundary directions and rays between interior samples and polygon corners.
Reduction covers sampled witnesses by visibility from a distributed working set
of candidate edges, expanding it if necessary for coverage. It uses the full
candidate set to add connecting candidates, and removes redundant candidates while retaining sampled
coverage and connectivity within each free-space component. Visibility coverage
is not a proof of complete movement-axis coverage, nor a classical fewest-line
map or a DepthmapX-compatible all-lines algorithm. Narrow features may require
smaller ``samplingDistance``. The default spacing is one sixth of the largest
bounding dimension. The result is not guaranteed to be invariant under sampling
changes or polygon decomposition details.

``reduce=False`` returns the candidate edges. ``returnReport=True`` returns a
dictionary containing ``edges``, ``candidateCount``, ``sampleCount``,
``uncoveredSamples``, ``componentCount``, ``samplingDistance`` and
``approximate=True``. Sample and candidate budgets return the available map
instead of failing when a limit is exceeded. Sampling is coarsened and distributed
when necessary; candidates are generated round-robin across regions and seeds,
with architectural directions preceding corner rays.
Near-duplicate candidates are removed by endpoint distance within ``tolerance``,
including reversed Edges and points on adjacent spatial-hash cells.
``maxCandidates`` caps the
candidate set before reduction, so the returned map may contain fewer Edges.
A warning identifies a partial result unless ``silent=True``.

The report includes ``truncated``, ``limitsReached``, ``coverageComplete``,
``connected`` and ``componentConnected``. Uncovered samples are returned as 3D
coordinates in the original plane. Coverage completeness refers to retained
samples, not all points in the Face. With ``reduce=False``, coverage and
connectivity are not evaluated and these fields are ``None``.
Retained witness coordinates are available as ``samples`` for inspecting coverage.
The requested and actual spacing are reported separately as ``requestedSamplingDistance``
and ``samplingDistance``. ``evaluatedCandidateCount`` records how many candidates
needed visibility evaluation during reduction. Increasing a budget or reducing
the requested spacing can improve an approximate map; budgets bound storage,
not runtime.

Each output edge has its own ``uuid``/``edge_id``, the source face's
``source_face_id`` and an ``axial_index``. Regeneration creates new source edges
and therefore new UUIDs. Use ``TGraph.AxialGraph(edges=edges)`` to analyse them,
then transfer selected node dictionary values back with
``TGraph.TransferDictionariesToOriginators`` for colouring.

.. automodule:: topologicpy.Face
   :members:
   :undoc-members:
   :show-inheritance:
