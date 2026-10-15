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

Offset regions
--------------

``Face.ByOffset`` returns a ``Face`` when one region survives and a ``Cluster``
containing all Faces when the offset separates the region. A multi-region result
prints a warning with the number of regions unless ``silent=True``. Complete
erosion returns ``None``; it does not return an inverted remnant.

Uniform polygonal offsets can close narrow necks, merge holes, remove shrinking
holes, or let expanding holes meet the exterior. They are resolved in the Face's
own plane, including vertical and tilted planes, with mitred joins. Single-region
cases retain the existing Wire offset path when it agrees with the region result.
Curved and variable offsets retain their exact Wire construction; collisions
between otherwise valid curved boundaries are resolved with native face
Booleans, without tessellation. Unsupported/self-intersecting curved or variable
Wire offsets can still fail and return ``None``.

With ``transferDictionaries=True``, the source Face dictionary is transferred to
every returned Face and the Cluster. The existing Wire path retains its source
Edge/Vertex dictionary transfer; newly resolved boundaries do not always have
one-to-one source Edge/Vertex correspondences.

.. code-block:: python

   result = Face.ByOffset(face, offset=0.3)
   regions = Topology.Faces(result) if result is not None else []

``Face.ByOffsetArea`` optimises the combined area of all surviving regions and
uses the same return convention. Optimisation trials suppress diagnostics; the
final multi-region result emits the warning.

.. automodule:: topologicpy.Face
   :members:
   :undoc-members:
   :show-inheritance:


Transparent holes in isovists
----------------------------

``Face.Isovist`` distinguishes non-walkable holes from opaque obstacles.
Its existing ``obstacles`` list accepts additional closed Wires; the new
``holes`` list accepts additional transparent exclusion Wires. Both reject
viewpoints in or on them, but only obstacles stop sightlines. Visible area
includes transparent holes and must not be used as a navigation domain.

Internal boundary Wires use the dictionary key ``isovistType`` by default,
with case-insensitive values ``hole`` or ``obstacle``. Set
``boundaryTypeKey`` to use another key. Missing labels follow
``defaultBoundaryType="obstacle"`` (the previous behaviour); choose
``defaultBoundaryType="hole"`` to see across all unlabelled internal voids.
Explicit input lists determine their members' roles regardless of labels.
Opaque obstacles take precedence wherever the two types overlap.

For example::

    isovist = Face.Isovist(room, viewpoint,
                          holes=[atrium], obstacles=[column],
                          boundaryTypeKey="isovistType",
                          defaultBoundaryType="hole", metrics=True)

The geometry is evaluated in the face plane, including tilted faces.
Transparent hole edges do not contribute physical wall dictionaries or
occlusion/closed-perimeter metrics. Existing positional arguments remain
unchanged; the new parameters are appended to the signature.

To classify an existing internal boundary individually::

    inner = Face.InternalBoundaries(room)[0]
    Topology.SetDictionary(inner,
                          Dictionary.ByPythonDictionary({"isovistType": "hole"}))
    isovist = Face.Isovist(room, viewpoint)
