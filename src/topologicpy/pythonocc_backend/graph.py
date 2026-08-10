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
        # Dedupe by coordinates, not just uuid/shape-identity: the edge's
        # start/end wrappers and the caller's vertex wrappers for the SAME
        # point are distinct Python objects (different uuids), so `in` fails
        # and unique_by_uuid cannot collapse them. Without this a graph of
        # 3 vertices reports 5-7 vertices (shared endpoints duplicated), which
        # breaks Vertex/edge adjacency and degree queries.
        deduped = []
        seen_coords = set()
        for v in vertices:
            try:
                key = (round(float(v.x) / 1e-4), round(float(v.y) / 1e-4), round(float(v.z) / 1e-4))
            except Exception:
                key = None
            if key is not None:
                if key in seen_coords:
                    continue
                seen_coords.add(key)
            deduped.append(v)
        return Graph(vertices=unique_by_uuid(deduped), edges=unique_by_uuid(edges))

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

    def AddVertices(self, vertices, tolerance=0.0001):
        new_vertices = [v for v in (vertices or []) if isinstance(v, Vertex)]
        if new_vertices:
            self.vertices = unique_by_uuid(self.vertices + new_vertices)
        return self

    def RemoveVertices(self, vertices):
        to_remove = [v for v in (vertices or []) if isinstance(v, Vertex)]
        if not to_remove:
            return self
        self.vertices = [
            v for v in self.vertices
            if not any(same_vertex(v, r) for r in to_remove)
        ]
        self.edges = [
            e for e in self.edges
            if not any(same_vertex(e.start, r) or same_vertex(e.end, r) for r in to_remove)
        ]
        return self

    def RemoveEdges(self, edges, tolerance=0.0001):
        to_remove = [e for e in (edges or []) if isinstance(e, Edge)]
        if not to_remove:
            return self

        def _same_edge(a, b):
            return (
                same_vertex(a.start, b.start, tolerance) and same_vertex(a.end, b.end, tolerance)
            ) or (
                same_vertex(a.start, b.end, tolerance) and same_vertex(a.end, b.start, tolerance)
            )

        self.edges = [
            e for e in self.edges
            if not any(_same_edge(e, r) for r in to_remove)
        ]
        return self

    def Connect(self, verticesA, verticesB, tolerance=0.0001):
        verticesA = [v for v in (verticesA or []) if isinstance(v, Vertex)]
        verticesB = [v for v in (verticesB or []) if isinstance(v, Vertex)]
        for va, vb in zip(verticesA, verticesB):
            if same_vertex(va, vb, tolerance):
                continue
            existing = self.Edge(va, vb, tolerance=tolerance)
            if existing is not None:
                continue
            edge = Edge.ByStartVertexEndVertex(va, vb)
            if edge is not None:
                self.AddEdge(edge)
        return self

    def ContainsVertex(self, vertex, tolerance=0.0001):
        if not isinstance(vertex, Vertex):
            return False
        return any(same_vertex(v, vertex, tolerance) for v in self.vertices)

    def ContainsEdge(self, edge, tolerance=0.0001):
        if not isinstance(edge, Edge):
            return False
        for e in self.edges:
            if (
                same_vertex(e.start, edge.start, tolerance)
                and same_vertex(e.end, edge.end, tolerance)
            ) or (
                same_vertex(e.start, edge.end, tolerance)
                and same_vertex(e.end, edge.start, tolerance)
            ):
                return True
        return False

    def _degree(self, vertex, tolerance=0.0001):
        degree = 0
        for edge in self.edges:
            if same_vertex(edge.start, vertex, tolerance):
                degree += 1
            if same_vertex(edge.end, vertex, tolerance):
                degree += 1
        return degree

    def DegreeSequence(self, sequence=None):
        result = sorted((self._degree(v) for v in self.vertices), reverse=True)
        if sequence is not None:
            sequence.extend(result)
            return 0
        return result

    def Density(self):
        n = len(self.vertices)
        m = len(self.edges)
        if n < 2:
            return 0.0
        return float(2.0 * m) / float(n * (n - 1))

    def IsolatedVertices(self, vertices=None):
        result = [v for v in self.vertices if self._degree(v) == 0]
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def AllPaths(self, vertexA, vertexB, searchLimitFlag=True, timeLimit=10, output=None):
        import time as _time
        from .wire import Wire

        if output is None:
            output = []
        if not isinstance(vertexA, Vertex) or not isinstance(vertexB, Vertex):
            return 0

        adjacency = {}
        for edge in self.edges:
            adjacency.setdefault(id(edge.start), []).append((edge.end, edge))
            adjacency.setdefault(id(edge.end), []).append((edge.start, edge))

        start_key = None
        for v in self.vertices:
            if same_vertex(v, vertexA):
                start_key = id(v)
                break
        if start_key is None:
            return 0

        end_time = _time.time() + timeLimit if searchLimitFlag else float("inf")
        max_paths = 1000

        results = []
        visited_ids = set()

        def _dfs(current_vertex, path_edges, visited_ids):
            if _time.time() > end_time or len(results) >= max_paths:
                return
            if same_vertex(current_vertex, vertexB):
                if path_edges:
                    results.append(list(path_edges))
                return
            neighbors = adjacency.get(id(current_vertex), [])
            for next_vertex, edge in neighbors:
                if id(next_vertex) in visited_ids:
                    continue
                visited_ids.add(id(next_vertex))
                path_edges.append(edge)
                _dfs(next_vertex, path_edges, visited_ids)
                path_edges.pop()
                visited_ids.discard(id(next_vertex))
                if _time.time() > end_time or len(results) >= max_paths:
                    return

        visited_ids.add(start_key)
        _dfs(vertexA, [], visited_ids)

        for path_edges in results:
            oriented = self._oriented_edge_chain(path_edges, vertexA)
            wire = Wire.ByEdges(oriented if oriented is not None else path_edges)
            if wire is not None:
                output.append(wire)

        return 0

    @staticmethod
    def _oriented_edge_chain(path_edges, start_vertex, tolerance=0.0001):
        """
        Reorient a connected edge chain head-to-tail: reuse forward edges, replace
        backward ones with a reversed copy (dictionary preserved).
        """
        if not path_edges:
            return None
        oriented = []
        current = start_vertex
        for edge in path_edges:
            if same_vertex(edge.start, current, tolerance):
                oriented.append(edge)
                current = edge.end
            elif same_vertex(edge.end, current, tolerance):
                flipped = Edge.ByStartVertexEndVertex(edge.end, edge.start)
                if flipped is None:
                    return None
                flipped.dictionary = edge.dictionary
                oriented.append(flipped)
                current = edge.start
            else:
                return None
        return oriented

    def Topology(self):
        from .cluster import Cluster

        topologies = list(self.vertices) + list(self.edges)
        return Cluster.ByTopologies(topologies)

    def Path(self, vertexA, vertexB, tolerance=0.0001):
        from .wire import Wire

        paths = []
        self.AllPaths(vertexA, vertexB, True, 10, paths)
        if not paths:
            return None
        # Prefer the shortest path (fewest edges) as a reasonable default.
        return min(paths, key=lambda w: len(getattr(w, "edges", []) or []))

    def IsComplete(self):
        n = len(self.vertices)
        if n < 2:
            return True
        max_edges = n * (n - 1) / 2.0
        # Count unique undirected edges.
        seen = set()
        for e in self.edges:
            seen.add(frozenset((id(e.start), id(e.end))))
        return len(seen) >= max_edges

    def IsErdoesGallai(self, sequence):
        seq = sorted([s for s in (sequence or [])], reverse=True)
        n = len(seq)
        if n == 0:
            return True
        if sum(seq) % 2 != 0:
            return False
        for k in range(1, n + 1):
            lhs = sum(seq[:k])
            rhs = k * (k - 1) + sum(min(seq[i], k) for i in range(k, n))
            if lhs > rhs:
                return False
        return True

    def MaximumDelta(self):
        if not self.vertices:
            return 0
        return max(self._degree(v) for v in self.vertices)

    def MinimumDelta(self):
        if not self.vertices:
            return 0
        return min(self._degree(v) for v in self.vertices)

    def GetGUID(self):
        return self._uuid

class GraphUtility:
    pass


# ---------------------------------------------------------------------------
# Graph construction / functional mutation / routing
# ---------------------------------------------------------------------------

def _graph_copy(graph):
    """Return a shallow copy of a backend Graph (new uuid, shared sub-objects)."""
    import copy as _copy
    new = _copy.copy(graph)
    new._uuid = new_uuid()
    return new


@staticmethod
def _ByTopology(topology, silent: bool = False):
    """
    Build a backend Graph from any topology: collect its vertices and edges.
    Mirrors topologicpy.Graph.ByTopology's behaviour (vertices + edges only).
    """
    from .vertex import Vertex
    from .edge import Edge

    if topology is None:
        return None
    # Backend-native objects expose Vertices/Edges instance methods that
    # accept an out-parameter list; call the wrapper-free path directly.
    verts = []
    try:
        _ = topology.Vertices(None, verts)
    except Exception:
        verts = []
    edges = []
    try:
        _ = topology.Edges(None, edges)
    except Exception:
        edges = []
    if not verts and not edges:
        # fall back to edges' endpoints
        for e in edges:
            verts.append(e.start)
            verts.append(e.end)
    verts = [v for v in verts if isinstance(v, Vertex)]
    edges = [e for e in edges if isinstance(e, Edge)]
    if not verts and not edges:
        return None
    return Graph.ByVerticesEdges(verts, edges)


@staticmethod
def _ByAdjacencyMatrix(adjacencyMatrix, dictionaries=None, silent: bool = False):
    """
    Build a backend Graph from an adjacency matrix. Delegates to the real
    (pure-Python) topologicpy wrapper implementation so behaviour matches
    the reference backend exactly.
    """
    from topologicpy.Graph import Graph as _WrapperGraph
    return _WrapperGraph.ByAdjacencyMatrix(
        adjacencyMatrix=adjacencyMatrix,
        dictionaries=dictionaries,
        silent=silent,
    )


@staticmethod
def _RemoveVertex(graph, vertex, silent: bool = False):
    """
    Functional vertex removal: return a new backend Graph with `vertex`
    (and its incident edges) removed. Mirrors Graph.RemoveVertex.
    """
    if not isinstance(graph, Graph):
        return None
    if isinstance(vertex, Vertex):
        vertex = [vertex]
    if not isinstance(vertex, (list, tuple)) or len(vertex) == 0:
        return None
    new_graph = _graph_copy(graph)
    new_graph.RemoveVertices(list(vertex))
    return new_graph


@staticmethod
def _RemoveEdge(graph, *edges, tolerance: float = 0.0001, silent: bool = False):
    """
    Functional edge removal: return a new backend Graph with the given
    edges removed. Mirrors Graph.RemoveEdge.
    """
    if not isinstance(graph, Graph):
        return None
    edge_list = [e for e in edges if isinstance(e, Edge)]
    if len(edge_list) == 0:
        return None
    new_graph = _graph_copy(graph)
    new_graph.RemoveEdges(list(edge_list), tolerance=tolerance)
    return new_graph


@staticmethod
def _ShortestPath(graph, vertexA, vertexB, vertexKey: str = "",
                    edgeKey: str = "Length", tolerance: float = 0.0001,
                    silent: bool = False):
    """
    Return the shortest path between vertexA and vertexB as a Wire, using
    Dijkstra over edge weights (edgeKey dictionary value, else Euclidean
    length). Returns None if no path exists. Mirrors Graph.ShortestPath.
    """
    from .wire import Wire

    if not isinstance(graph, Graph):
        return None
    if not isinstance(vertexA, Vertex) or not isinstance(vertexB, Vertex):
        return None

    # Build adjacency: map vertex -> list of (neighbour, edge, weight)
    adj = {}
    for v in graph.vertices:
        adj.setdefault(id(v), [])

    def _weight(edge):
        if edgeKey:
            try:
                from .dictionary import Dictionary
                from .topology import Topology
                d = Topology.Dictionary(edge)
                w = Dictionary.ValueAtKey(d, edgeKey)
                if w is not None:
                    try:
                        return float(w)
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            return float(Edge.Length(edge))
        except Exception:
            return 1.0

    for e in graph.edges:
        sa, ea = id(e.start), id(e.end)
        w = _weight(e)
        adj.setdefault(sa, []).append((ea, e, w))
        adj.setdefault(ea, []).append((sa, e, w))

    start_id = None
    end_id = None
    for v in graph.vertices:
        if same_vertex(v, vertexA, tolerance):
            start_id = id(v)
        if same_vertex(v, vertexB, tolerance):
            end_id = id(v)
    if start_id is None or end_id is None:
        return None

    import heapq
    dist = {start_id: 0.0}
    prev_edge = {start_id: None}
    visited = set()
    pq = [(0.0, start_id)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end_id:
            break
        for (nbr, edge, w) in adj.get(u, []):
            if nbr in visited:
                continue
            nd = d + w
            if nd < dist.get(nbr, float("inf")):
                dist[nbr] = nd
                prev_edge[nbr] = edge
                heapq.heappush(pq, (nd, nbr))

    if end_id not in prev_edge:
        return None  # unreachable

    # Walk back collecting edges
    path_edges = []
    cur = end_id
    while cur != start_id:
        e = prev_edge.get(cur)
        if e is None:
            return None
        path_edges.append(e)
        # step to the other endpoint
        if id(e.start) == cur:
            cur = id(e.end)
        else:
            cur = id(e.start)
    path_edges.reverse()

    oriented = Graph._oriented_edge_chain(path_edges, vertexA, tolerance=tolerance)
    wire = Wire.ByEdges(oriented if oriented is not None else path_edges,
                         tolerance=tolerance)
    return wire


@staticmethod
def _AdjacentVertices(graph, vertex, vertices=None, tolerance: float = 0.0001):
    """GraphUtility.AdjacentVertices -> delegate to instance method."""
    result = graph.AdjacentVertices(vertex) if isinstance(graph, Graph) else []
    if vertices is not None:
        vertices.extend(result)
        return 0
    return result


@staticmethod
def _UtilityShortestPath(graph, vertexA, vertexB, vertexKey: str = "",
                           edgeKey: str = "Length", tolerance: float = 0.0001,
                           silent: bool = False):
    """GraphUtility.ShortestPath -> delegate to Graph.ShortestPath."""
    return Graph._ShortestPath(graph, vertexA, vertexB,
                               vertexKey=vertexKey, edgeKey=edgeKey,
                               tolerance=tolerance, silent=silent)


Graph.ByTopology = staticmethod(_ByTopology)
Graph.ByAdjacencyMatrix = staticmethod(_ByAdjacencyMatrix)
Graph.RemoveVertex = staticmethod(_RemoveVertex)
Graph.RemoveEdge = staticmethod(_RemoveEdge)
Graph.ShortestPath = staticmethod(_ShortestPath)
GraphUtility.AdjacentVertices = staticmethod(_AdjacentVertices)
GraphUtility.ShortestPath = staticmethod(_UtilityShortestPath)
