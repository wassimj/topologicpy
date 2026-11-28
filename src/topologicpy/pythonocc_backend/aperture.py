from __future__ import annotations

from .topology import Topology, _topology_type_name
from .dictionary import Dictionary


_GEOMETRIC_TYPES = {"Vertex", "Edge", "Wire", "Face", "Shell", "Cell", "CellComplex", "Cluster"}


def _is_geometry(topology):
    return isinstance(topology, Topology) and _topology_type_name(topology) in _GEOMETRIC_TYPES


class Aperture:
    """A semantic namespace; never a shapeless native topology wrapper."""

    @staticmethod
    def IsAperture(topology):
        if not _is_geometry(topology):
            return False
        value = Dictionary.ValueAtKey(topology.GetDictionary(), "type")
        return isinstance(value, str) and value.casefold() == "aperture"

    @staticmethod
    def ByTopologyContext(topology, context):
        from .context import Context
        if not isinstance(topology, Topology) or not isinstance(context, Context):
            return None
        host = context.Topology()
        if not _is_geometry(host) or not _is_geometry(topology) or Topology.IsSame(host, topology):
            return None
        dictionary = Dictionary(topology.GetDictionary())
        dictionary.SetValueAtKey("type", "aperture")
        topology.SetDictionary(dictionary)
        host.AddContent(topology, context)
        return topology

    @staticmethod
    def Topology(aperture):
        return aperture if Aperture.IsAperture(aperture) else None
