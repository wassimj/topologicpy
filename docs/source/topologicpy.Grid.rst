topologicpy.Grid module
=======================

``Grid.Vertices(grid)`` returns unique endpoints and intersections of grid
Edges, including crossings that have not been split into separate Edges.
It accepts the topology returned by ``Grid.OnFace`` or a list of Edges.
PythonOCC processes curved Edges in a single native batch; straight
TopologicCore Edges use analytical 3D intersections with bounding-box pruning.
Coincident points are deduplicated using the model-unit ``tolerance``.

.. automodule:: topologicpy.Grid
   :members:
   :undoc-members:
   :show-inheritance:
