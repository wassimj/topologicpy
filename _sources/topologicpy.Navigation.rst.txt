Navigation
==========

The navigation API separates source extraction, editable geometry, grid
construction and graph analysis. It does not depend on SpatialLab or a web
framework. Navigation sampling and geometry helpers reside in ``Navigation.py``; IFC mesh adaptation and extraction helpers reside in ``IFC.py`` as private class methods. Existing ``Grid.OnFace`` and ``Graph.NavigationGraph`` are unchanged.

Quick start
-----------

.. code-block:: python

   from topologicpy.IFC import IFC
   from topologicpy.Navigation import Navigation

   source = IFC.ModelByPath("building.ifc")
   model = IFC.NavigationModel(source)
   graph = Navigation.Graph(model, spacing=1.0)
   tiles = Navigation.Tiles(model, graph)
   eye_faces = Navigation.IsovistFaces(model, eyeHeight=1.6)

``Navigation.ByIFCPath(path)`` combines the first two calls. Parsing happens once;
changing graph spacing does not reparse the IFC. A 2D face uses the same model:

.. code-block:: python

   model = Navigation.ByFaces([face], obstacles=[obstacle_footprint])
   graph = Navigation.Graph(model, spacing=0.5)

Model and geometry
------------------

``NavigationModel.faces`` and ``obstacles`` contain actual Topologic Faces;
``connectors`` contain geometry descriptors made by ``Stair``, ``Ramp`` and
``Lift``. ``storeys`` contains dictionaries with ``id``, ``name`` and
``elevation``. Face dictionaries carry ``storey`` and ``source`` identifiers.
Users may edit the geometry and construct another graph. Replacing the imported obstacle footprint list overrides solid sectioning with the supplied footprint geometry. Footprint obstacles
passed to ``ByFaces`` are assigned to the nearest elevation unless they already
carry a storey identifier. ``storeyIDs`` can explicitly assign the walking faces.

Coordinates use world Z as up. Faces must have linear boundaries, be planar and have a nonzero XY
projection; horizontal and sloping faces are supported. IFC coordinates are
normalized to metres. Direct geometry inputs retain their model units, so
spacing, step heights and eye heights must use those same units. Intrinsic grids
on curved surfaces remain available through ``Grid.OnFace``.

IFC extraction excludes roofs and uses containment before elevation fallback.
``maxSlope`` selects candidate slab slopes in degrees (default 45). An upward
surface alone is not proof of operational walkability. ``obstacleTypes`` can
restrict physical obstacle classes; ``None`` uses all supported physical
obstacles. ``excludeTypes`` excludes selected IFC classes from extraction.
Furniture and opening geometry use conservative world-aligned boxes; opening
boxes cut their hosts. The model metadata includes an ``omittedSurfaces`` list for native face conversion failures, without printing warnings. Unsupported geometry is best effort and disconnected
components remain disconnected.

Connectors
----------

* ``Navigation.Stair(topology_or_faces, ...)`` selects horizontal tread/landing
  surfaces. A single descriptor groups its faces into one assembly. Separate
  descriptors may share ``group`` to join a stair and its landings.
* ``Navigation.Ramp(topology_or_faces, spacingRatio=0.15, ...)`` samples the
  inferred centreline at a fraction of its 3D length. Absolute ``spacing``
  overrides the ratio. Endpoints and detected flat landings are retained.
  Disjoint or non-monotonic geometry may not yield a centreline; supply an
  ordered ``centreline`` (XYZ points, Vertices, or a linear Wire) to bypass inference.
  Supplied centreline corners are always retained. Without surface geometry,
  explicit centreline nodes have no colour tiles.
* ``Navigation.Lift(doors, cabinCentres, storeys=None, ...)`` accepts matching
  ordered lists, with one door and cabin point per level. Door and cabin points
  must share an elevation. Cabin points connect in elevation order; doors
  attach to floor grids. Lifts have no dedicated colour tiles.

Graph construction
------------------

``Navigation.Graph`` returns a standard Topologic graph. Set ``graphType="tgraph"`` to return a ``TGraph`` with the same nodes, edges and metadata. Existing ``Graph``
analysis and shortest-path methods operate on it. Edge dictionary ``Length`` is
Euclidean distance in the model units, so geometric routing uses the usual
``Graph.ShortestPath(..., edgeKey="Length")``.

The default graph connects adjacent clipped grid cells. ``mode="visibility"``
adds same-storey visibility connections while checking surface discontinuities.
``origin`` and ``direction`` define a shared XY grid frame. ``storey`` restricts
construction to one storey before routing, preventing detours through other
levels. The model itself is not changed.

The default maximum vertical slab step is 0.17 model units. Horizontal gaps are
never bridged (apart from numerical tolerance): ``maxHorizontalGap`` must be
zero. Explicit connectors can describe intentional bridges. Tread-to-tread
connections require touching geometry from the same connector assembly;
extracted stair rises above ``maxStairRiser`` (default 0.35 model units) remain disconnected. Connector
attachments use the nearest reachable floor-grid point. Invalid spacing,
unsupported modes and empty walking geometry raise explicit exceptions.

Overlapping coplanar floor faces are merged before gridding. ``sourceIDs``
retains the contributing source identifiers for each merged region. Coincident
connector and grid points share one vertex; ``kinds`` and ``storeyIDs`` retain
their combined classifications.

Vertices carry ``navigationID``, ``storey``, ``source``, ``kind`` and ``tileIDs``.
Edges carry ``Length`` and ``kind``. ``Navigation.Tiles(model, graph)`` returns
Topologic Faces with ``navigationID`` and ``tileID`` dictionaries. Tile geometry
is stored as JSON in the graph dictionary so it survives graph serialization;
associations describe the graph at construction time, even if the source model
is later edited. A node may own multiple tiles. Graphs without navigation tile
metadata return an empty tile list.

``Navigation.IsovistFaces`` returns strictly horizontal faces at every walking
elevation plus eye height. IFC solids are sectioned again at the requested eye
height. Direct planar obstacles are treated as vertical footprints. Results can
be restricted to a storey and passed to existing planar isovist methods.

``Grid.OnFaces`` produces shared, clipped grid-line geometry on several planar
faces. It uses the same XY projection convention as navigation, whereas
``Grid.OnFace`` retains its existing surface-grid behavior.

Clearance
---------

.. code-block:: python

   graph = Navigation.Graph(model, spacing=1.0, clearance=0.3)

Clearance is a horizontal XY margin in model units (metres for IFC models).
The default zero preserves existing construction. Obstacles are subtracted
first, then physically joined walking regions are offset using ``Face.ByOffset``.
Internal slab boundaries and permitted small steps remain connected. A narrow
passage can close or split a region; no links are fabricated across that split.
If no floor region survives, graph construction raises ``ValueError``.

The grid origin and cell spacing stay fixed. Boundary samples move into the
surviving portion of their original cell. Original display tiles remain attached
to those samples, so an inset does not automatically create an uncoloured band
along every wall. Consumed cells disappear; cells split into several components
retain separate tiles. Where a concave cell prevents a straight connection,
extra bend nodes keep the route inside the valid cleared region. These junctions
have no dedicated colour tiles and participate in graph analyses normally.

Stair and ramp width is tested across the assembled passage, retaining internal
tread and landing connections. Attachment corridors are checked for width and
obstacles when choosing reachable floor samples. A connector that is too narrow
remains disconnected. Supplied ramp surfaces are used even with an explicit
centreline; geometry-free centrelines and lift cabin edges do not supply width
information and retain their explicit connections. ``Navigation.Lift`` accepts an optional ``footprint`` Face to preserve door portals during clearance; the IFC adapter supplies the shaft footprint automatically. This is a geometric XY
constraint, not an accessibility or headroom assessment. Isovist faces are
unchanged; their grid can independently use ``Grid.OnFace`` clearance if desired.

The margin is stored in the graph dictionary as ``clearance`` and applies to
both native Graph and TGraph output. All implementation helpers remain within
``Navigation.py``, ``Grid.py`` and the corresponding existing backend class files.

Isovist boundary roles
----------------------

``Navigation.IsovistFaces`` labels every returned internal Wire using
``isovistType="hole"`` for a genuine floor void and
``isovistType="obstacle"`` for a region intersecting an eye-level obstacle.
IFC solids are sectioned at the requested eye height; generic planar
obstacles retain their vertical-footprint interpretation. Merged void and
obstacle boundaries are classified as opaque as a whole, even if the
obstacle occupies only part of the enclosed region.

``Face.Isovist`` reads these labels automatically: it rejects viewpoints in
both types but sees across transparent holes. The source walking faces and
the navigation graph remain unchanged, and the returned eye-level faces
still exclude both types from grid sampling. Pass ``boundaryTypeKey`` to
both methods when using a custom dictionary key::

   eye_faces = Navigation.IsovistFaces(model, boundaryTypeKey="visibilityRole")
   visible = Face.Isovist(eye_faces[0], viewpoint,
                         boundaryTypeKey="visibilityRole")

API reference
-------------

.. automodule:: topologicpy.Navigation
   :members:
   :undoc-members:
   :show-inheritance:
