topologicpy.Daylight module
==========================

Direct daylight and preliminary Waldram assessment
-------------------------------------------------

``Daylight`` provides direct sky calculations using TopologicPy geometry and
its existing BVH visibility engine. It does not determine legal rights,
infringement, compensation, planning compliance or total daylight factor.

All distances use model units; the examples assume metres. Factors and
thresholds use **percent**: ``threshold=0.2`` means 0.2%, not 20%.
A percentage-point change is distinct from a relative percentage change.

Quick start
-----------

.. code-block:: python

   from topologicpy.Daylight import Daylight
   from topologicpy.Cell import Cell
   from topologicpy.Face import Face
   from topologicpy.Topology import Topology
   from topologicpy.Vertex import Vertex

   cell = Cell.Prism(Vertex.ByCoordinates(0, 0, 0), width=4, length=4,
                     height=3, placement="lowerleft")
   window = Face.ByVertices([
       Vertex.ByCoordinates(1, 0, 1), Vertex.ByCoordinates(3, 0, 1),
       Vertex.ByCoordinates(3, 0, 2.5), Vertex.ByCoordinates(1, 0, 2.5)
   ])
   cell = Topology.AddApertures(cell, [window], subTopologyType="face")
   room = Daylight.PrepareRoom(cell)
   plane = Daylight.WorkingPlane(cell, height=0.85)
   grid = Daylight.SensorGrid(plane, spacing=0.3)

   # Retained surrounding geometry appears in both scenes. Do NOT include
   # an additional uncut copy of the selected room Cell.
   before_scene = Daylight.PrepareScene([])
   obstruction = Face.ByVertices([
       Vertex.ByCoordinates(-5, -1, 0), Vertex.ByCoordinates(9, -1, 0),
       Vertex.ByCoordinates(9, -1, 9), Vertex.ByCoordinates(-5, -1, 9)
   ])
   after_scene = Daylight.PrepareScene([obstruction])
   before = Daylight.Waldram(room, before_scene, grid=grid)
   after = Daylight.Waldram(room, after_scene, grid=grid)
   comparison = Daylight.Compare(before, after)
   regions = Daylight.Contours(before, levels=[0.2])
   diagram = Daylight.WaldramDiagram(before, sensorIndex=0)
   data = Daylight.ResultData(before, includeGeometry=True)

Scene roles belong to the calling application: Before uses retained geometry
plus removals; After uses retained geometry plus proposals. Prepared rooms
provide real window openings; raw topologies are opaque. Display opacity and
visibility do not affect calculations. All analytical surfaces are two-sided.

Method
------

``SkyComponent`` uses equal-solid-angle hemisphere midpoint quadrature and
projected incidence weights. It supports uniform and CIE standard overcast
relative luminance. The reference is unobstructed external horizontal
illuminance under the same sky. ``SkyFactor`` specialises this for horizontal
sensors and a uniform sky. These methods are useful outdoors or where apertures
are large; coarse angular patches may miss a narrow window altogether.

``Waldram`` instead integrates over the room's actual apertures. For a window
triangle with area A, sensor-to-triangle distance r, aperture incidence cosine
c_w and upward sensor incidence cosine c_h, its midpoint contribution is::

    skyFactorPercent = 100 * A * abs(c_w) * max(c_h, 0) / (pi * r**2)

Native tessellation retains aperture holes. Uniform subdivision refines its
triangles; full-scene ray queries reject occluded contributions. Overlapping
coplanar windows are unioned, and nearer aperture intersections prevent
projected windows being counted more than once. Windows may be vertical,
inclined or horizontal. All room/context Faces must currently be planar with
straight polygon boundaries. Unsupported geometry raises an error.

Windows are read from TopologicPy aperture attachments on the Cell or its
Faces. Preparation checks coplanarity and complete host containment before
subtracting windows from opaque analytical walls. It does not interpret
aperture metadata alone as a geometric hole.

The uniform-sky method excludes reflected light, glazing/frame losses, material
transmission and direct sunlight. It is a geometric sky-factor assessment, not
an indoor illuminance or annual daylight simulation. ``SkyComponent`` with an
overcast distribution remains a **direct sky component**, not total daylight
factor. Full reflected-light and annual metrics are outside this version.

Sampling and sensitivity
------------------------

Working-plane height defaults to 0.85 above the room's lowest vertex elevation.
Use ``referenceElevation`` explicitly for stepped or sloping floors. The plane
must cut the room interior. Multiple disconnected section Faces are supported.

Grid cells are clipped to the domain; each receives an interior sensor and its
actual area. Threshold area is a **piecewise-constant sampled-cell estimate**.
It is not a continuous exact contour-area integral. Nonconvex domains and holes
are retained. Grid partition area conservation is checked.

``resolution`` accepts ``low``, ``medium``, ``high`` and ``very_high``. Aperture
triangles have respectively 2, 3, 4 or 5 subdivision levels, each replacing one
triangle by four. The default convergence check evaluates one additional level,
returns the finer values and reports their difference from the requested level.
It does not certify convergence or an error bound. An explicit triangle-count
limit prevents unbounded angular subdivision.

To examine spatial sensitivity, repeat ``Waldram`` with a finer grid over the
same working plane, then call ``Convergence(coarse, fine)``. Large rooms and
small openings may need both finer spatial and angular sampling. Tiny losses
or threshold crossings should not be interpreted without this check.

``Compare`` requires the same room, grid, method, resolution and refinement
settings. It reports area gained/lost, before/after threshold percentages,
percentage-point change and crossing of a configurable reference (default 50%).
There is deliberately no legal pass/fail or automated risk classification.

Contours and diagrams
---------------------

``Contours`` unions above/below-threshold grid cells into native Faces. Its
boundary Wires include the domain perimeter and holes. These are staircase
sampled-region boundaries, **not smooth interpolated isolines**. The reported
areas agree with the numerical threshold estimate.

``WaldramDiagram`` maps aperture triangles to an XY sheet with azimuth in
radians and y = sin(altitude)^2 / 2. The projection's Jacobian corresponds to
horizontal projected solid angle. Each patch carries its midpoint visibility
and its original numerical contribution. Sheet triangles approximate curved
projected edges; their geometric areas are not used to recompute sky factors.
Horizon-straddling or degenerate projected patches can be omitted and are
counted. Azimuth seam patches may extend beyond the nominal sheet width.
Numeric sky factor remains the complete calculation even if a sheet patch is
omitted. This is an explanatory sampled diagram, not an exact silhouette.

Records and persistence
-----------------------

Prepared rooms, grids and scenes are snapshots, immutable by convention.
Rebuild after edits; do not change record arrays or geometry in place.
Results retain method version, identities, coordinates, areas, settings,
assumptions, warnings, refinement metadata and summary values. The private
``_diagram`` cache holds live ray geometry.

``ResultData`` produces JSON-compatible data without native geometry/cache.
``includeGeometry=True`` adds working-plane and clipped-cell BREP strings.
Recalculate from source geometry to restore a diagram cache. The result export
is not a complete project or prepared-scene persistence format.

Failures use ``ValueError`` rather than silent alternative calculations.
Cancellation callbacks raise ``InterruptedError`` during angular integration
and ray tracing. Native geometry preparation is synchronous and not interruptible.

API reference
-------------

.. automodule:: topologicpy.Daylight
   :members:
   :undoc-members:
   :show-inheritance:
