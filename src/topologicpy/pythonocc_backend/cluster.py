from __future__ import annotations

from dataclasses import dataclass, field

from .topology import Topology, _downward_wrappers, _is_null_shape

try:
    from .topology import _merge_backend_dictionaries
except Exception:  # pragma: no cover - compatibility with older topology backends
    def _merge_backend_dictionaries(a, b):
        result = dict(a or {}) if isinstance(a, dict) else {}
        if isinstance(b, dict):
            result.update(b)
        return result

try:
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopAbs import (
        TopAbs_COMPOUND,
        TopAbs_COMPSOLID,
        TopAbs_SOLID,
        TopAbs_SHELL,
        TopAbs_FACE,
        TopAbs_WIRE,
        TopAbs_EDGE,
        TopAbs_VERTEX,
    )
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator, topods
except Exception:  # pragma: no cover - permits import when PythonOCC is unavailable
    BRep_Builder = None
    TopAbs_COMPOUND = None
    TopAbs_COMPSOLID = None
    TopAbs_SOLID = None
    TopAbs_SHELL = None
    TopAbs_FACE = None
    TopAbs_WIRE = None
    TopAbs_EDGE = None
    TopAbs_VERTEX = None
    TopoDS_Compound = None
    TopoDS_Iterator = None
    topods = None


def _flatten_python_inputs(values):
    """Flattens Python list/tuple containers without flattening Topology objects."""
    result = []

    def walk(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            result.append(value)

    walk(values)
    return result


@dataclass(eq=False)
class Cluster(Topology):
    """
    PythonOCC backend representation of a Topologic Cluster.

    A Cluster is represented natively as a shallow ``TopoDS_Compound``. When the
    Cluster is constructed from Python Topology wrappers, those direct wrappers are
    also retained so direct-member queries can preserve wrapper metadata such as
    dictionaries. Descendant queries are resolved from the OCCT Compound itself.
    """

    topologies: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def ByTopologies(
        topologies,
        transferDictionaries: bool = False,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None,
        **kwargs,
    ):
        """
        Creates a Cluster from one or more backend Topologies.

        Parameters
        ----------
        topologies : Topology or list
            One or more Topology wrappers, optionally nested in lists/tuples.
        transferDictionaries : bool, optional
            If True and ``dictionary`` is not supplied, merges dictionaries from
            direct input Topologies onto the Cluster. Default is False.
        dictionary : dict, optional
            Explicit Cluster dictionary. Default is None.
        contents, contexts, apertures : list, optional
            Backend relationship collections preserved on the Cluster wrapper.

        Returns
        -------
        Cluster or None
            A Cluster backed by a native ``TopoDS_Compound``.
        """
        if BRep_Builder is None or TopoDS_Compound is None:
            return None

        if "transferDictionary" in kwargs:
            transferDictionaries = bool(kwargs["transferDictionary"])

        items = []
        for item in _flatten_python_inputs(topologies):
            if not isinstance(item, Topology):
                continue
            shape = getattr(item, "shape", None)
            if _is_null_shape(shape):
                continue
            items.append(item)

        if not items:
            return None

        try:
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for item in items:
                builder.Add(compound, item.shape)
        except Exception:
            return None

        result_dictionary = dictionary
        if result_dictionary is None and transferDictionaries:
            result_dictionary = {}
            for item in items:
                try:
                    result_dictionary = _merge_backend_dictionaries(
                        result_dictionary,
                        Topology.GetDictionary(item),
                    )
                except Exception:
                    pass

        return Cluster(
            shape=compound,
            topologies=list(items),
            dictionary=result_dictionary,
            contents=list(contents) if contents else [],
            contexts=list(contexts) if contexts else [],
            apertures=list(apertures) if apertures else [],
        )

    @staticmethod
    def ByOcctShape(
        shape,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None,
    ):
        """
        Wraps an existing OCCT Compound as a Cluster.

        Direct Python wrappers are not available in this path and are therefore
        materialized lazily from the Compound when queried.
        """
        if _is_null_shape(shape) or topods is None or TopAbs_COMPOUND is None:
            return None

        try:
            if shape.ShapeType() != TopAbs_COMPOUND:
                return None
            compound = topods.Compound(shape)
            if _is_null_shape(compound):
                return None
        except Exception:
            return None

        return Cluster(
            shape=compound,
            topologies=[],
            dictionary=dictionary,
            contents=list(contents) if contents else [],
            contexts=list(contexts) if contexts else [],
            apertures=list(apertures) if apertures else [],
        )

    # ------------------------------------------------------------------
    # Internal queries
    # ------------------------------------------------------------------

    def _direct_topologies(self):
        """Returns only the direct children of the Cluster Compound."""
        # Preserve original wrappers/dictionaries when this Cluster was built in
        # Python rather than reconstructed from a raw OCCT shape.
        if self.topologies:
            return list(self.topologies)

        shape = getattr(self, "shape", None)
        if _is_null_shape(shape) or TopoDS_Iterator is None:
            return []

        result = []
        try:
            iterator = TopoDS_Iterator(shape)
            while iterator.More():
                child_shape = iterator.Value()
                if child_shape is not None and not _is_null_shape(child_shape):
                    if TopAbs_COMPOUND is not None and child_shape.ShapeType() == TopAbs_COMPOUND:
                        child = Cluster.ByOcctShape(child_shape)
                    else:
                        child = Topology.ByOcctShape(child_shape)
                    if child is not None:
                        result.append(child)
                iterator.Next()
        except Exception:
            return []
        return result

    def _query(self, shape_type):
        """Returns descendant wrappers of the requested OCCT shape type."""
        shape = getattr(self, "shape", None)
        if not _is_null_shape(shape) and shape_type is not None:
            try:
                return _downward_wrappers(self, shape_type)
            except Exception:
                pass

        # Compatibility fallback for any legacy shapeless Cluster wrapper.
        result = []
        for topology in self.topologies or []:
            try:
                if shape_type == TopAbs_VERTEX:
                    values = topology.Vertices()
                elif shape_type == TopAbs_EDGE:
                    values = topology.Edges()
                elif shape_type == TopAbs_WIRE:
                    values = topology.Wires()
                elif shape_type == TopAbs_FACE:
                    values = topology.Faces()
                elif shape_type == TopAbs_SHELL:
                    values = topology.Shells()
                elif shape_type == TopAbs_SOLID:
                    values = topology.Cells()
                elif shape_type == TopAbs_COMPSOLID:
                    values = topology.CellComplexes()
                elif shape_type == TopAbs_COMPOUND:
                    values = topology.Clusters()
                else:
                    values = []
            except Exception:
                values = []
            if values:
                result.extend(values)
        return result

    @staticmethod
    def _return_or_extend(result, output):
        """Supports Python return-list and Topologic output-list calling styles."""
        result = list(result or [])
        if output is not None:
            output.extend(result)
            return 0
        return result

    # ------------------------------------------------------------------
    # Direct and descendant queries
    # ------------------------------------------------------------------

    def Topologies(self, hostTopology=None, topologies=None):
        """Returns the direct child Topologies of the Compound."""
        return Cluster._return_or_extend(self._direct_topologies(), topologies)

    def Vertices(self, hostTopology=None, vertices=None):
        """Returns all descendant Vertices."""
        return Cluster._return_or_extend(self._query(TopAbs_VERTEX), vertices)

    def Edges(self, hostTopology=None, edges=None):
        """Returns all descendant Edges."""
        return Cluster._return_or_extend(self._query(TopAbs_EDGE), edges)

    def Wires(self, hostTopology=None, wires=None):
        """Returns all descendant Wires."""
        return Cluster._return_or_extend(self._query(TopAbs_WIRE), wires)

    def Faces(self, hostTopology=None, faces=None):
        """Returns all descendant Faces."""
        return Cluster._return_or_extend(self._query(TopAbs_FACE), faces)

    def Shells(self, hostTopology=None, shells=None):
        """Returns all descendant Shells."""
        return Cluster._return_or_extend(self._query(TopAbs_SHELL), shells)

    def Cells(self, hostTopology=None, cells=None):
        """Returns all descendant Cells."""
        return Cluster._return_or_extend(self._query(TopAbs_SOLID), cells)

    def CellComplexes(self, hostTopology=None, cellComplexes=None):
        """Returns all descendant CellComplexes."""
        return Cluster._return_or_extend(self._query(TopAbs_COMPSOLID), cellComplexes)

    def Clusters(self, hostTopology=None, clusters=None):
        """Returns nested descendant Clusters, excluding this Cluster itself."""
        return Cluster._return_or_extend(self._query(TopAbs_COMPOUND), clusters)


# Compatibility aliases used by parts of the algorithm layer and direct Core callers.
Cluster.ByTopologiesCluster = staticmethod(
    lambda topologys, transferDictionaries=False: Cluster.ByTopologies(
        topologys,
        transferDictionaries=transferDictionaries,
    )
)


def _cluster_free_topologies(cluster, tolerance: float = 0.0001):
    """Returns direct Cluster members without performing geometry operations."""
    if not isinstance(cluster, Cluster):
        return []
    return cluster.Topologies() or []


Cluster.FreeTopologies = staticmethod(_cluster_free_topologies)
ClusterUtility = Cluster
