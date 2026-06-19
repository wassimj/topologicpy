from __future__ import annotations

from dataclasses import dataclass
from .topology import Topology
from .occ_utils import make_occ_vertex
from .helpers import distance3, same_vertex, unique_by_uuid


@dataclass(eq=False)
class Vertex(Topology):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def ByCoordinates(x=0.0, y=0.0, z=0.0):
        x = float(x); y = float(y); z = float(z)
        return Vertex(shape=make_occ_vertex(x, y, z), x=x, y=y, z=z)

    @staticmethod
    def ByPoint(point):
        try:
            return Vertex.ByCoordinates(point.X(), point.Y(), point.Z())
        except Exception:
            return None

    def X(self):
        return float(self.x)

    def Y(self):
        return float(self.y)

    def Z(self):
        return float(self.z)

    def Coordinates(self):
        return [self.x, self.y, self.z]

    def Vertices(self, hostTopology=None, vertices=None):
        result = [self]
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result


class VertexUtility:
    @staticmethod
    def Distance(vertexA, vertexB):
        if isinstance(vertexA, Vertex) and isinstance(vertexB, Vertex):
            return distance3(vertexA, vertexB)
        return None

    @staticmethod
    def AdjacentEdges(vertex, topology, edges):
        from .edge import Edge
        from .graph import Graph
        if not isinstance(vertex, Vertex):
            return 1
        result = []
        if isinstance(topology, Graph):
            for edge in topology.edges:
                if same_vertex(edge.start, vertex) or same_vertex(edge.end, vertex):
                    result.append(edge)
        elif isinstance(topology, Topology):
            temp = []
            topology.Edges(None, temp)
            for edge in temp:
                if isinstance(edge, Edge) and (same_vertex(edge.start, vertex) or same_vertex(edge.end, vertex)):
                    result.append(edge)
        edges.extend(unique_by_uuid(result))
        return 0

# ---------------------------------------------------------------------------
# Explicit unsupported Vertex API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _vertex_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Vertex.{name}", return_value)
    return _method


def _vertex_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"VertexUtility.{name}", return_value)
    return _method


Vertex.ByCoordinatesString = staticmethod(_vertex_not_implemented("ByCoordinatesString"))
Vertex.Origin = staticmethod(_vertex_not_implemented("Origin"))
Vertex.Project = staticmethod(_vertex_not_implemented("Project"))
Vertex.Fuse = _vertex_not_implemented("Fuse")
VertexUtility.NearestVertex = staticmethod(_vertex_utility_not_implemented("NearestVertex"))
VertexUtility.ParameterAtVertex = staticmethod(_vertex_utility_not_implemented("ParameterAtVertex"))
