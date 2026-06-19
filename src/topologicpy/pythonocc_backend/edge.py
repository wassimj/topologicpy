from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .topology import Topology
from .vertex import Vertex
from .occ_utils import make_occ_edge
from .helpers import distance3, same_vertex


@dataclass(eq=False)
class Edge(Topology):
    start: Optional[Vertex] = None
    end: Optional[Vertex] = None

    @staticmethod
    def ByStartVertexEndVertex(startVertex, endVertex):
        if not isinstance(startVertex, Vertex) or not isinstance(endVertex, Vertex):
            return None
        if same_vertex(startVertex, endVertex):
            return None
        return Edge(shape=make_occ_edge(startVertex, endVertex), start=startVertex, end=endVertex)

    @staticmethod
    def ByVertices(vertices):
        if vertices is None or len(vertices) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertices[0], vertices[-1])

    def StartVertex(self):
        return self.start

    def EndVertex(self):
        return self.end

    def Vertices(self, hostTopology=None, vertices=None):
        result = [v for v in [self.start, self.end] if isinstance(v, Vertex)]
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = [self]
        if edges is not None:
            edges.extend(result)
            return 0
        return result


class EdgeUtility:
    @staticmethod
    def Length(edge):
        if isinstance(edge, Edge) and isinstance(edge.start, Vertex) and isinstance(edge.end, Vertex):
            return distance3(edge.start, edge.end)
        return None

    @staticmethod
    def PointAtParameter(edge, parameter):
        if not isinstance(edge, Edge):
            return None
        parameter = float(parameter)
        return Vertex.ByCoordinates(
            edge.start.x + (edge.end.x - edge.start.x) * parameter,
            edge.start.y + (edge.end.y - edge.start.y) * parameter,
            edge.start.z + (edge.end.z - edge.start.z) * parameter,
        )

    @staticmethod
    def ParameterAtPoint(edge, vertex):
        if not isinstance(edge, Edge) or not isinstance(vertex, Vertex):
            return None
        length2 = (
            (edge.end.x - edge.start.x) ** 2
            + (edge.end.y - edge.start.y) ** 2
            + (edge.end.z - edge.start.z) ** 2
        )
        if length2 == 0:
            return 0
        return (
            (vertex.x - edge.start.x) * (edge.end.x - edge.start.x)
            + (vertex.y - edge.start.y) * (edge.end.y - edge.start.y)
            + (vertex.z - edge.start.z) * (edge.end.z - edge.start.z)
        ) / length2

# ---------------------------------------------------------------------------
# Explicit unsupported Edge API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _edge_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Edge.{name}", return_value)
    return _method


def _edge_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"EdgeUtility.{name}", return_value)
    return _method


Edge.ByCurve = staticmethod(_edge_not_implemented("ByCurve"))
Edge.ByStartVertexEndVertexTolerance = staticmethod(_edge_not_implemented("ByStartVertexEndVertexTolerance"))
EdgeUtility.Angle = staticmethod(_edge_utility_not_implemented("Angle"))
EdgeUtility.NormalAtParameter = staticmethod(_edge_utility_not_implemented("NormalAtParameter"))
EdgeUtility.Trim = staticmethod(_edge_utility_not_implemented("Trim"))
