topologicpy.Grid module
=======================

``Grid.Vertices(grid)`` returns unique endpoints and intersections of grid
Edges, including crossings that have not been split into separate Edges.
It accepts the topology returned by ``Grid.OnFace`` or a list of Edges.
PythonOCC processes curved Edges in a single native batch; straight
TopologicCore Edges use analytical 3D intersections with bounding-box pruning.
Coincident points are deduplicated using the model-unit ``tolerance``.

Clearance
---------

``Grid.OnFace(..., clearance=0.3)`` clips the original planar grid to the inset
region(s) returned by ``Face.ByOffset``. Outer boundaries move inward and holes
expand. Distances use the Face plane and model units. Grid positions, divisions,
labels and spacing are computed on the original Face; clearance does not recenter
or rescale the grid. All surviving regions are retained. An exhausted or failed
offset returns ``None``. Positive clearance on intrinsic surface grids is
explicitly unsupported, while ``clearance=0`` retains existing surface behavior.

``Grid.OnFaces(..., clearance=0.3)`` measures the margin in horizontal XY, matching
its projected grid definition. Continuous adjoining patches are offset together
so internal slab seams do not create gaps. Holes are included in the offset.
This method retains its linear-boundary requirement. ``includeBoundary=True``
includes the clipped region boundaries; the lattice still uses the supplied
shared origin and direction.

The default is ``clearance=0.0``. Negative or non-finite values are invalid.
``Grid.Vertices`` needs no separate clearance parameter because it derives
points from the resulting clipped grid.

Patterned grids
---------------

``Grid.OnPlane`` creates a finite grid in a rectangular domain on XY by default.
``width`` and ``length`` specify its dimensions; ``origin`` is its centre and
``direction`` is its normal. ``Grid.OnFace`` clips the same patterns to a planar
Face and its holes. Both accept ``silent`` and use model-unit dimensions.
Existing OnFace calls retain their original planar and intrinsic-surface behavior.

Use ``gridType="rectangular"`` for square or independently spaced U/V grids,
``"hexagonal"`` for regular hexagons, ``"diagrid"`` for two crossing diagonal
families, or ``"polar"`` for complete concentric rings and radial lines. Hexagon
``cellSize`` is the side length. Diagrid ``angle`` is measured from local X;
``uSpacing`` and ``vSpacing`` measure perpendicular family separation.
Polar ``angularSpacing`` is in degrees and must divide 360 exactly. Circles
are native curves; ``approximate=True`` explicitly permits sampled rings when
exact circles cannot be constructed by the selected backend.

``output="edges"`` returns a Cluster, ``output="vertices"`` a list of unique
Vertices (including crossings), and ``output="faces"`` a list of partition Faces.
The Face output includes partial cells at the clipping boundary. Empty Edge
results are None; empty Vertex/Face results are empty lists. Invalid input
returns None, with messages suppressed when ``silent=True``.

``projection="local"`` is the OnFace default: dimensions are measured in the
face plane. ``projection="xy"`` projects a horizontal pattern onto a nonvertical
planar Face; XY spacing remains constant on slopes. ``rotation`` rotates the
pattern in degrees; it does not rotate the clipping Face. ``origin`` anchors the
lattice, while ``centre`` can override the polar centre. Clearance clips to
inset regions without shifting the original pattern; OnFace clearance remains
a distance in the face plane, including when the pattern is projected from XY.
Patterned construction does not accept intrinsic surface mode or legacy
U/V divisions, explicit values or labels. Use the existing OnFace interface for
those options. All private implementation helpers remain in Grid.py.

Examples::

    from topologicpy.Grid import Grid

    # Equal spacing on the XY plane.
    edges = Grid.OnPlane(width=12, length=8, spacing=1, silent=True)

    # Independent X/Y spacing; return the partition Faces.
    faces = Grid.OnPlane(width=12, length=8, uSpacing=2, vSpacing=1,
                         output="faces", silent=True)

    # Hexagonal cells on an existing planar Face.
    faces = Grid.OnFace(face, gridType="hexagonal", cellSize=1,
                        orientation="pointy-top", output="faces", silent=True)

    # Diamond diagrid, rotated within the plane.
    vertices = Grid.OnPlane(gridType="diagrid", angle=45, rotation=15,
                            output="vertices", silent=True)

    # Complete polar pattern, clipped to the Face.
    edges = Grid.OnFace(face, gridType="polar", radialSpacing=1,
                        angularSpacing=30, centre=[0, 0, 0], silent=True)

    # Horizontal lattice projected onto a sloping Face.
    edges = Grid.OnFace(face, gridType="rectangular", projection="xy",
                        spacing=1, origin=[0, 0, 0], silent=True)

.. automodule:: topologicpy.Grid
   :members:
   :undoc-members:
   :show-inheritance:
