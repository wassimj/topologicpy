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

.. automodule:: topologicpy.Grid
   :members:
   :undoc-members:
   :show-inheritance:
