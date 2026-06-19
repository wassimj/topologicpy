from __future__ import annotations

from dataclasses import dataclass, field
from .topology import Topology
from .helpers import unique_by_uuid


@dataclass(eq=False)
class Cluster(Topology):
    topologies: list = field(default_factory=list)

    @staticmethod
    def ByTopologies(topologies, transferDictionaries=False):
        topologies = [t for t in (topologies or []) if isinstance(t, Topology)]
        if not topologies:
            return None
        return Cluster(shape=None, topologies=topologies)

    def Topologies(self, topologies=None):
        result = list(getattr(self, "topologies", []) or [])
        if topologies is not None:
            topologies.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        result = []
        for topology in self.topologies:
            temp = []
            topology.Vertices(None, temp)
            result.extend(temp)
        result = unique_by_uuid(result)
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = []
        for topology in self.topologies:
            temp = []
            topology.Edges(None, temp)
            result.extend(temp)
        result = unique_by_uuid(result)
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Wires(self, hostTopology=None, wires=None):
        result = []
        for topology in self.topologies:
            temp = []
            topology.Wires(None, temp)
            result.extend(temp)
        result = unique_by_uuid(result)
        if wires is not None:
            wires.extend(result)
            return 0
        return result

    def Faces(self, hostTopology=None, faces=None):
        result = []
        for topology in self.topologies:
            temp = []
            topology.Faces(None, temp)
            result.extend(temp)
        result = unique_by_uuid(result)
        if faces is not None:
            faces.extend(result)
            return 0
        return result

# ---------------------------------------------------------------------------
# Explicit unsupported Cluster API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _cluster_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Cluster.{name}", return_value)
    return _method


Cluster.ByTopologiesCluster = staticmethod(_cluster_not_implemented("ByTopologiesCluster"))
Cluster.SelfMerge = _cluster_not_implemented("SelfMerge")
Cluster.FreeTopologies = _cluster_not_implemented("FreeTopologies", [])
