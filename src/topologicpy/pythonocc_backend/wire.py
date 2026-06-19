from __future__ import annotations

from dataclasses import dataclass, field
from .topology import Topology
from .vertex import Vertex
from .edge import Edge
from .occ_utils import make_occ_wire
from .helpers import same_vertex, unique_by_uuid


@dataclass(eq=False)
class Wire(Topology):
    edges: list = field(default_factory=list)

    @staticmethod
    def ByEdges(edges, tolerance=0.0001):
        if edges is None:
            return None
        edges = [e for e in edges if isinstance(e, Edge)]
        if not edges:
            return None
        ordered = Wire._order_edges(edges, tolerance=tolerance)
        if ordered is None:
            ordered = edges
        return Wire(shape=make_occ_wire(ordered), edges=ordered)

    @staticmethod
    def ByVertices(vertices, close=False, tolerance=0.0001):
        if vertices is None:
            return None
        vertices = [v for v in vertices if isinstance(v, Vertex)]
        if len(vertices) < 2:
            return None
        edges = []
        for a, b in zip(vertices[:-1], vertices[1:]):
            if not same_vertex(a, b, tolerance):
                e = Edge.ByStartVertexEndVertex(a, b)
                if e is not None:
                    edges.append(e)
        if close and len(vertices) > 2 and not same_vertex(vertices[-1], vertices[0], tolerance):
            e = Edge.ByStartVertexEndVertex(vertices[-1], vertices[0])
            if e is not None:
                edges.append(e)
        if not edges:
            return None
        return Wire.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def _order_edges(edges, tolerance=0.0001):
        if not edges:
            return []
        unused = list(edges)
        ordered = [unused.pop(0)]
        while unused:
            last = ordered[-1].end
            found_index = None
            found_edge = None
            for i, edge in enumerate(unused):
                if same_vertex(edge.start, last, tolerance):
                    found_index = i
                    found_edge = edge
                    break
                if same_vertex(edge.end, last, tolerance):
                    found_index = i
                    found_edge = Edge.ByStartVertexEndVertex(edge.end, edge.start)
                    if found_edge is not None:
                        found_edge.dictionary = edge.dictionary
                    break
            if found_index is None:
                return None
            ordered.append(found_edge)
            unused.pop(found_index)
            if len(ordered) > len(edges) + 1:
                return None
        return ordered

    def Edges(self, hostTopology=None, edges=None):
        result = list(getattr(self, "edges", []) or [])
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        result = []
        for edge in getattr(self, "edges", []) or []:
            if isinstance(edge, Edge):
                result.extend([edge.start, edge.end])
        result = unique_by_uuid([v for v in result if isinstance(v, Vertex)])
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Wires(self, hostTopology=None, wires=None):
        result = [self]
        if wires is not None:
            wires.extend(result)
            return 0
        return result

    def IsClosed(self, tolerance=0.0001):
        edges = getattr(self, "edges", []) or []
        if not edges:
            return False
        return same_vertex(edges[0].start, edges[-1].end, tolerance)


class WireUtility:
    @staticmethod
    def IsClosed(wire):
        if isinstance(wire, Wire):
            return wire.IsClosed()
        return False

# ---------------------------------------------------------------------------
# Explicit unsupported Wire API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _wire_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Wire.{name}", return_value)
    return _method


def _wire_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"WireUtility.{name}", return_value)
    return _method


Wire.ByEdgesCluster = staticmethod(_wire_not_implemented("ByEdgesCluster"))
Wire.ByWires = staticmethod(_wire_not_implemented("ByWires"))
Wire.Reverse = _wire_not_implemented("Reverse")
WireUtility.Length = staticmethod(_wire_utility_not_implemented("Length"))
WireUtility.Cycles = staticmethod(_wire_utility_not_implemented("Cycles", []))
WireUtility.Split = staticmethod(_wire_utility_not_implemented("Split"))
