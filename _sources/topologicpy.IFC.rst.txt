topologicpy.IFC module
======================

Geometry units
--------------

``IFC.Topologies`` and ``IFC.MeshData``, their file/path/entity wrappers, and
``IFC.AnalyticalFaces``, ``IFC.ThermalCellComplex``, ``IFC.AccessGraph``, and
``IFC.AdjacencyGraph`` convert project-assigned IFC length units to metres by
default. SI prefixes and conversion-based units such as feet are supported.
Dimensions and placement translations use the same conversion. Source STEP
entities, properties, quantities, and round-trip ``IFC.Export`` remain in their
original IFC units.

Fragments without unit assignments assume metres for compatibility. Invalid,
unsupported, or conflicting declared length units fail explicitly instead of
silently assuming a unit. Public import methods return ``None`` on failure;
low-level mesh methods raise ``ValueError`` for invalid units or multipliers.

The ``scale`` argument is deprecated and defaults to ``None``. An explicit,
finite, positive value preserves the legacy raw-coordinate multiplier and
**overrides** automatic unit conversion. For example, ``scale=0.001`` converts
raw millimetre coordinates to metres once, while ``scale=1.0`` preserves raw
coordinates. New callers should omit ``scale`` and rely on the IFC units.
Geometric tolerances, proximity thresholds, and thermal enclosure margins use
output coordinate units, which are metres with automatic conversion. Semantic
quantity values are source values, not converted geometry measurements.

``Topology.ByIFCFile`` and ``Topology.ByIFCPath`` use the same defaults and
forward ``scale``, ``topologyType``, and ``circleSides`` consistently.

Public API
----------

``IFCFastTopology`` remains available for compatibility and internal geometry
parsing. Its topology, path-mesh, and summary entry points delegate to ``IFC``;
it is excluded from this public listing to avoid repeated method names.

.. automodule:: topologicpy.IFC
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: IFCFastTopology
