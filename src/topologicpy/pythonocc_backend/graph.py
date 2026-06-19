from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from .helpers import new_uuid, same_vertex, unique_by_uuid
from .dictionary import Dictionary
from .vertex import Vertex
from .edge import Edge


@dataclass(eq=False)
class Graph:
    vertices: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    dictionary: Any = field(default_factory=dict)
    _uuid: str = field(default_factory=new_uuid)

    def __hash__(self):
        return hash(self._uuid)

    def GetTypeAsString(self):
        return "Graph"

    def SetDictionary(self, dictionary):
        if dictionary is None:
            self.dictionary = {}
        elif isinstance(dictionary, (dict, Dictionary)):
            self.dictionary = dictionary
        elif hasattr(dictionary, "_data") and isinstance(dictionary._data, dict):
            self.dictionary = dictionary
        elif hasattr(dictionary, "data") and isinstance(dictionary.data, dict):
            self.dictionary = dictionary
        else:
            self.dictionary = {}
        return self

    def GetDictionary(self):
        if self.dictionary is None:
            self.dictionary = {}
        return self.dictionary

    def Dictionary(self):
        if self.dictionary is None:
            self.dictionary = {}
        return self.dictionary

    @staticmethod
    def ByVerticesEdges(vertices, edges):
        vertices = [v for v in (vertices or []) if isinstance(v, Vertex)]
        edges = [e for e in (edges or []) if isinstance(e, Edge)]
        for edge in edges:
            if edge.start not in vertices:
                vertices.append(edge.start)
            if edge.end not in vertices:
                vertices.append(edge.end)
        return Graph(vertices=unique_by_uuid(vertices), edges=unique_by_uuid(edges))

    @staticmethod
    def ByVertices(vertices):
        return Graph.ByVerticesEdges(vertices, [])

    def Vertices(self, vertices=None):
        result = list(getattr(self, "vertices", []) or [])
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Edges(self, edges=None, tolerance=0.0001):
        result = list(getattr(self, "edges", []) or [])
        if edges is not None and isinstance(edges, list):
            edges.extend(result)
            return 0
        return result

    def AdjacentVertices(self, vertex, vertices=None):
        result = []
        if isinstance(vertex, Vertex):
            for edge in self.edges:
                if same_vertex(edge.start, vertex):
                    result.append(edge.end)
                elif same_vertex(edge.end, vertex):
                    result.append(edge.start)
        result = unique_by_uuid(result)
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Edge(self, vertexA, vertexB, tolerance=0.0001):
        for edge in self.edges:
            if (
                same_vertex(edge.start, vertexA, tolerance)
                and same_vertex(edge.end, vertexB, tolerance)
            ) or (
                same_vertex(edge.start, vertexB, tolerance)
                and same_vertex(edge.end, vertexA, tolerance)
            ):
                return edge
        return None

    def AddVertex(self, vertex):
        if isinstance(vertex, Vertex):
            self.vertices = unique_by_uuid(self.vertices + [vertex])
        return self

    def AddEdge(self, edge):
        if isinstance(edge, Edge):
            self.edges = unique_by_uuid(self.edges + [edge])
            self.vertices = unique_by_uuid(self.vertices + [edge.start, edge.end])
        return self


class GraphUtility:
    pass

# ---------------------------------------------------------------------------
# Explicit unsupported Graph API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _graph_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Graph.{name}", return_value)
    return _method


def _graph_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"GraphUtility.{name}", return_value)
    return _method


Graph.ByTopology = staticmethod(_graph_not_implemented("ByTopology"))
Graph.ByAdjacencyMatrix = staticmethod(_graph_not_implemented("ByAdjacencyMatrix"))
Graph.RemoveVertex = _graph_not_implemented("RemoveVertex")
Graph.RemoveEdge = _graph_not_implemented("RemoveEdge")
Graph.ShortestPath = _graph_not_implemented("ShortestPath")
GraphUtility.AdjacentVertices = staticmethod(_graph_utility_not_implemented("AdjacentVertices"))
GraphUtility.ShortestPath = staticmethod(_graph_utility_not_implemented("ShortestPath"))
