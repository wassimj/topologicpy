# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import csv
import io
import json
import math


class TGraph:
    PHASE2_WEIGHTED_MATRIX_FIX = True
    """
    A topology-first graph class.

    TGraph stores graph incidence independently from geometry:
    - vertices are stable indexed records with Python dictionaries;
    - edges are src/dst index records with Python dictionaries;
    - optional vertex representations can be any topology;
    - optional edge representations can be topologic edges, wires, curves, or control data.
    """

    __slots__ = (
        "_vertices",
        "_edges",
        "_out_edges",
        "_in_edges",
        "_incident_edges",
        "_edge_lookup",
        "_directed",
        "_allow_self_loops",
        "_allow_parallel_edges",
        "_dictionary",
        "_version",
        "_compiled",
    )

    def __init__(
        self,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
    ):
        self._vertices: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []
        self._out_edges: Dict[int, Set[int]] = {}
        self._in_edges: Dict[int, Set[int]] = {}
        self._incident_edges: Dict[int, Set[int]] = {}
        self._edge_lookup: Dict[Tuple[int, int, bool], Set[int]] = {}
        self._directed = bool(directed)
        self._allow_self_loops = bool(allowSelfLoops)
        self._allow_parallel_edges = bool(allowParallelEdges)
        self._dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
        self._version = 0
        self._compiled = None

    def __repr__(self) -> str:
        kind = "directed" if self._directed else "bidirectional"
        return f"TGraph(vertices={TGraph.Order(self)}, edges={TGraph.Size(self)}, {kind})"

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------



    @staticmethod
    def ByAdjacencyDictionary(
        adjacencyDictionary: Optional[Dict[Any, Iterable[Any]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        vertexLabelKey: str = "label",
        edgeWeightKey: str = "weight",
    ) -> "TGraph":
        adjacencyDictionary = adjacencyDictionary or {}
        g = TGraph(directed=directed, allowSelfLoops=allowSelfLoops,
                   allowParallelEdges=allowParallelEdges, dictionary=dictionary)
        label_to_index: Dict[Any, int] = {}

        def ensure(label: Any) -> int:
            if label not in label_to_index:
                label_to_index[label] = g.AddVertex(dictionary={vertexLabelKey: label})
            return label_to_index[label]

        for src_label, neighbors in adjacencyDictionary.items():
            ensure(src_label)
            for item in neighbors or []:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    dst_label = item[0]
                    weight = item[1] if len(item) > 1 else None
                else:
                    dst_label = item
                    weight = None
                ensure(dst_label)

        for src_label, neighbors in adjacencyDictionary.items():
            src = label_to_index[src_label]
            for item in neighbors or []:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    dst_label = item[0]
                    weight = item[1] if len(item) > 1 else None
                else:
                    dst_label = item
                    weight = None
                ed = {}
                if weight is not None:
                    ed[edgeWeightKey] = weight
                g.AddEdge(src, label_to_index[dst_label], dictionary=ed)
        return g


    @staticmethod
    def ByTriples(
        triples: Optional[List[Tuple[Any, Any, Any]]] = None,
        directed: bool = True,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        vertexLabelKey: str = "label",
        edgeLabelKey: str = "label",
    ) -> "TGraph":
        g = TGraph(directed=directed, allowSelfLoops=allowSelfLoops,
                   allowParallelEdges=allowParallelEdges, dictionary=dictionary)
        label_to_index: Dict[Any, int] = {}

        def ensure(label: Any) -> int:
            if label not in label_to_index:
                label_to_index[label] = g.AddVertex(dictionary={vertexLabelKey: label})
            return label_to_index[label]

        for triple in triples or []:
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
            src_label, rel, dst_label = triple[0], triple[1], triple[2]
            src = ensure(src_label)
            dst = ensure(dst_label)
            g.AddEdge(src, dst, directed=True, dictionary={edgeLabelKey: rel, "relationship": rel})
        return g

    @staticmethod
    def ByVerticesEdges(
        vertices: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        silent: bool = False,
    ) -> "TGraph":
        g = TGraph(directed=directed, allowSelfLoops=allowSelfLoops,
                   allowParallelEdges=allowParallelEdges, dictionary=dictionary)
        topology_to_index = {}
        for item in vertices or []:
            d = TGraph._TopologyDictionaryToPython(item)
            idx = g.AddVertex(dictionary=d, representation=item)
            topology_to_index[id(item)] = idx
        for item in edges or []:
            d = TGraph._TopologyDictionaryToPython(item)
            src = d.get("src")
            dst = d.get("dst")
            if not isinstance(src, int) or not isinstance(dst, int):
                try:
                    from topologicpy.Topology import Topology
                    ev = Topology.Vertices(item, silent=True)
                    if ev and len(ev) >= 2:
                        src = topology_to_index.get(id(ev[0]))
                        dst = topology_to_index.get(id(ev[1]))
                except Exception:
                    src = None
                    dst = None
            if isinstance(src, int) and isinstance(dst, int):
                g.AddEdge(src, dst, dictionary=d, representation=item)
        return g

    @staticmethod
    def ByJSONData(data: Dict[str, Any]) -> Optional["TGraph"]:
        return TGraph.FromPython(data)

    @staticmethod
    def ByJSONString(jsonString: str) -> Optional["TGraph"]:
        if not isinstance(jsonString, str):
            return None
        try:
            return TGraph.ByJSONData(json.loads(jsonString))
        except Exception:
            return None

    @staticmethod
    def ByNetworkXGraph(nxGraph: Any, directed: Optional[bool] = None,
                        allowSelfLoops: bool = True, allowParallelEdges: bool = True) -> Optional["TGraph"]:
        if nxGraph is None:
            return None
        try:
            is_directed = nxGraph.is_directed() if directed is None else bool(directed)
            is_multi = nxGraph.is_multigraph()
        except Exception:
            return None
        g = TGraph(directed=is_directed, allowSelfLoops=allowSelfLoops, allowParallelEdges=allowParallelEdges or is_multi)
        node_to_index = {}
        try:
            for node, data in nxGraph.nodes(data=True):
                d = dict(data) if isinstance(data, dict) else {}
                d.setdefault("label", node)
                node_to_index[node] = g.AddVertex(dictionary=d)
            if is_multi:
                for u, v, key, data in nxGraph.edges(keys=True, data=True):
                    d = dict(data) if isinstance(data, dict) else {}
                    d.setdefault("key", key)
                    g.AddEdge(node_to_index[u], node_to_index[v], directed=is_directed, dictionary=d)
            else:
                for u, v, data in nxGraph.edges(data=True):
                    d = dict(data) if isinstance(data, dict) else {}
                    g.AddEdge(node_to_index[u], node_to_index[v], directed=is_directed, dictionary=d)
            return g
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------------------

    @staticmethod
    def IsInstance(graph: Any) -> bool:
        return isinstance(graph, TGraph)

    @staticmethod
    def Order(graph: "TGraph") -> int:
        if not isinstance(graph, TGraph):
            return 0
        return sum(1 for v in graph._vertices if v.get("active", True))

    @staticmethod
    def Size(graph: "TGraph") -> int:
        if not isinstance(graph, TGraph):
            return 0
        return sum(1 for e in graph._edges if e.get("active", True))

    @staticmethod
    def IsDirected(graph: "TGraph") -> bool:
        return bool(graph._directed) if isinstance(graph, TGraph) else False

    @staticmethod
    def Dictionary(graph: "TGraph") -> Dict[str, Any]:
        return dict(graph._dictionary) if isinstance(graph, TGraph) else {}

    def SetDictionary(self, dictionary: Optional[Dict[str, Any]] = None) -> "TGraph":
        self._dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
        self._invalidate_cache()
        return self

    # ---------------------------------------------------------------------
    # Dictionary conversion
    # ---------------------------------------------------------------------

    @staticmethod
    def _TopologyDictionaryToPython(topology: Any) -> Dict[str, Any]:
        if topology is None:
            return {}
        try:
            from topologicpy.Topology import Topology
            d = Topology.Dictionary(topology)
        except Exception:
            d = None
        return TGraph._DictionaryToPython(d)

    @staticmethod
    def _DictionaryToPython(dictionary: Any) -> Dict[str, Any]:
        if dictionary is None:
            return {}
        if isinstance(dictionary, dict):
            return dict(dictionary)
        try:
            from topologicpy.Dictionary import Dictionary
            keys = Dictionary.Keys(dictionary) or []
        except Exception:
            keys = []
        result = {}
        for key in keys:
            try:
                from topologicpy.Dictionary import Dictionary
                result[key] = Dictionary.ValueAtKey(dictionary, key, None)
            except TypeError:
                try:
                    from topologicpy.Dictionary import Dictionary
                    result[key] = Dictionary.ValueAtKey(dictionary, key)
                except Exception:
                    result[key] = None
            except Exception:
                result[key] = None
        return result

    @staticmethod
    def _PythonToDictionary(data: Optional[Dict[str, Any]]) -> Any:
        if not isinstance(data, dict) or len(data) == 0:
            return None
        try:
            from topologicpy.Dictionary import Dictionary
            keys = list(data.keys())
            values = [data[k] for k in keys]
            return Dictionary.ByKeysValues(keys, values)
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _validate_vertex_index(self, index: int, active: bool = True) -> bool:
        if not isinstance(index, int) or index < 0 or index >= len(self._vertices):
            return False
        return True if not active else bool(self._vertices[index].get("active", True))

    def _validate_edge_index(self, index: int, active: bool = True) -> bool:
        if not isinstance(index, int) or index < 0 or index >= len(self._edges):
            return False
        return True if not active else bool(self._edges[index].get("active", True))

    def _edge_key(self, src: int, dst: int, directed: bool) -> Tuple[int, int, bool]:
        if directed:
            return (src, dst, True)
        a, b = (src, dst) if src <= dst else (dst, src)
        return (a, b, False)

    def _register_edge_adjacency(self, edge_index: int, src: int, dst: int, directed: bool) -> None:
        self._out_edges.setdefault(src, set()).add(edge_index)
        self._in_edges.setdefault(dst, set()).add(edge_index)
        self._incident_edges.setdefault(src, set()).add(edge_index)
        self._incident_edges.setdefault(dst, set()).add(edge_index)
        if not directed:
            self._out_edges.setdefault(dst, set()).add(edge_index)
            self._in_edges.setdefault(src, set()).add(edge_index)
        key = self._edge_key(src, dst, directed)
        self._edge_lookup.setdefault(key, set()).add(edge_index)

    def _unregister_edge_adjacency(self, edge_index: int) -> None:
        if not self._validate_edge_index(edge_index, active=False):
            return
        e = self._edges[edge_index]
        src, dst, directed = e["src"], e["dst"], e["directed"]
        for table in [self._out_edges, self._in_edges, self._incident_edges]:
            for key in [src, dst]:
                if key in table:
                    table[key].discard(edge_index)
        key = self._edge_key(src, dst, directed)
        if key in self._edge_lookup:
            self._edge_lookup[key].discard(edge_index)
            if not self._edge_lookup[key]:
                del self._edge_lookup[key]

    @staticmethod
    def _as_index(vertex: Union[int, Dict[str, Any]]) -> Optional[int]:
        if isinstance(vertex, int):
            return vertex
        if isinstance(vertex, dict):
            idx = vertex.get("index")
            return idx if isinstance(idx, int) else None
        return None

    def _invalidate_cache(self) -> None:
        """Invalidates the compiled integer-indexed kernel."""
        self._version += 1
        self._compiled = None

    # ---------------------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------------------

    def AddVertex(self, dictionary: Optional[Dict[str, Any]] = None, representation: Any = None) -> int:
        index = len(self._vertices)
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d["index"] = index
        d.setdefault("active", True)
        record = {"index": index, "dictionary": d, "representation": representation, "active": True}
        self._vertices.append(record)
        self._out_edges.setdefault(index, set())
        self._in_edges.setdefault(index, set())
        self._incident_edges.setdefault(index, set())
        self._invalidate_cache()
        return index

    def AddVertices(self, vertices: Iterable[Any], dictionaries: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        result = []
        dictionaries = dictionaries or []
        for i, item in enumerate(vertices or []):
            d = dictionaries[i] if i < len(dictionaries) and isinstance(dictionaries[i], dict) else None
            if d is None and isinstance(item, dict):
                d = item
                rep = None
            else:
                rep = item
            result.append(self.AddVertex(dictionary=d, representation=rep))
        return result

    def AddEdge(self, src: int, dst: int, directed: Optional[bool] = None,
                dictionary: Optional[Dict[str, Any]] = None, representation: Any = None) -> Optional[int]:
        if not self._validate_vertex_index(src) or not self._validate_vertex_index(dst):
            return None
        if src == dst and not self._allow_self_loops:
            return None
        edge_directed = self._directed if directed is None else bool(directed)
        key = self._edge_key(src, dst, edge_directed)
        if not self._allow_parallel_edges and key in self._edge_lookup:
            return None
        index = len(self._edges)
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d.update({"index": index, "src": src, "dst": dst, "directed": edge_directed, "active": True})
        record = {"index": index, "src": src, "dst": dst, "directed": edge_directed,
                  "dictionary": d, "representation": representation, "active": True}
        self._edges.append(record)
        self._register_edge_adjacency(index, src, dst, edge_directed)
        self._invalidate_cache()
        return index

    def AddEdgeByIndex(self, index: Union[List[int], Tuple[int, int]], dictionary: Optional[Dict[str, Any]] = None,
                       directed: Optional[bool] = None, representation: Any = None) -> Optional[int]:
        if not isinstance(index, (list, tuple)) or len(index) < 2:
            return None
        return self.AddEdge(index[0], index[1], directed=directed, dictionary=dictionary, representation=representation)

    def RemoveEdge(self, edge: Union[int, Dict[str, Any]], silent: bool = False) -> "TGraph":
        idx = TGraph._as_index(edge)
        if idx is None and isinstance(edge, dict):
            idx = edge.get("index")
        if not self._validate_edge_index(idx, active=True):
            return self
        self._unregister_edge_adjacency(idx)
        self._edges[idx]["active"] = False
        self._edges[idx]["dictionary"]["active"] = False
        self._invalidate_cache()
        return self

    def RemoveVertex(self, vertex: Union[int, Dict[str, Any]], silent: bool = False) -> "TGraph":
        idx = TGraph._as_index(vertex)
        if not self._validate_vertex_index(idx, active=True):
            return self
        for eid in list(self._incident_edges.get(idx, set())):
            if self._validate_edge_index(eid, active=True):
                self.RemoveEdge(eid, silent=True)
        self._vertices[idx]["active"] = False
        self._vertices[idx]["dictionary"]["active"] = False
        self._invalidate_cache()
        return self

    # ---------------------------------------------------------------------
    # Accessors and dictionaries
    # ---------------------------------------------------------------------

    @staticmethod
    def Vertices(graph: "TGraph", asTopologic: bool = False, useRepresentation: bool = True,
                 activeOnly: bool = True, mantissa: int = 6, tolerance: float = 0.0001,
                 silent: bool = False) -> List[Any]:
        if not isinstance(graph, TGraph):
            return []
        records = [v for v in graph._vertices if (not activeOnly or v.get("active", True))]
        if not asTopologic:
            return [dict(v, dictionary=dict(v.get("dictionary", {}))) for v in records]
        result = []
        n = max(1, len(records))
        for i, record in enumerate(records):
            d = dict(record.get("dictionary", {}))
            rep = record.get("representation")
            v = None
            if useRepresentation and rep is not None:
                try:
                    from topologicpy.Topology import Topology
                    if Topology.IsInstance(rep, "Vertex"):
                        v = rep
                    elif Topology.IsInstance(rep, "Topology"):
                        v = Topology.CenterOfMass(rep)
                except Exception:
                    v = None
            if v is None:
                x, y, z = d.get("x"), d.get("y"), d.get("z")
                if not all(isinstance(c, (int, float)) for c in [x, y, z]):
                    angle = 2.0 * math.pi * float(i) / float(n)
                    x, y, z = math.cos(angle), math.sin(angle), 0.0
                try:
                    from topologicpy.Vertex import Vertex
                    v = Vertex.ByCoordinates(round(float(x), mantissa), round(float(y), mantissa), round(float(z), mantissa))
                except Exception:
                    v = None
            if v is not None:
                try:
                    from topologicpy.Topology import Topology
                    td = TGraph._PythonToDictionary(d)
                    v = Topology.SetDictionary(v, td, silent=True)
                except Exception:
                    pass
                result.append(v)
        return result

    @staticmethod
    def Edges(graph: "TGraph", asTopologic: bool = False, useRepresentation: bool = True,
              activeOnly: bool = True, segmentCurves: bool = True, mantissa: int = 6,
              tolerance: float = 0.0001, silent: bool = False) -> List[Any]:
        if not isinstance(graph, TGraph):
            return []
        records = [e for e in graph._edges if (not activeOnly or e.get("active", True))]
        if not asTopologic:
            return [dict(e, dictionary=dict(e.get("dictionary", {}))) for e in records]
        vertices = TGraph.Vertices(graph, asTopologic=True, useRepresentation=True, activeOnly=False,
                                   mantissa=mantissa, tolerance=tolerance, silent=silent)
        result = []
        for record in records:
            src, dst = record.get("src"), record.get("dst")
            d = dict(record.get("dictionary", {}))
            rep = record.get("representation")
            topology = None
            if useRepresentation and rep is not None:
                topology = TGraph._EdgeRepresentationToTopology(rep, dictionary=d, segmentCurves=segmentCurves,
                                                                tolerance=tolerance, silent=silent)
            if topology is None:
                if not isinstance(src, int) or not isinstance(dst, int) or src >= len(vertices) or dst >= len(vertices):
                    continue
                if src == dst:
                    continue
                try:
                    from topologicpy.Edge import Edge
                    from topologicpy.Vertex import Vertex
                    from topologicpy.Topology import Topology
                    if Vertex.Distance(vertices[src], vertices[dst]) <= tolerance:
                        continue
                    topology = Edge.ByStartVertexEndVertex(vertices[src], vertices[dst], tolerance=tolerance)
                    if topology is not None:
                        topology = Topology.SetDictionary(topology, TGraph._PythonToDictionary(d), silent=True)
                except Exception:
                    topology = None
            if topology is not None:
                result.append(topology)
        return result

    @staticmethod
    def Vertex(graph: "TGraph", index: int, asTopologic: bool = False) -> Optional[Any]:
        if not isinstance(graph, TGraph) or not graph._validate_vertex_index(index):
            return None
        if not asTopologic:
            v = graph._vertices[index]
            return dict(v, dictionary=dict(v.get("dictionary", {})))
        verts = TGraph.Vertices(graph, asTopologic=True, activeOnly=False)
        return verts[index] if index < len(verts) else None

    @staticmethod
    def Edge(graph: "TGraph", index: int) -> Optional[Dict[str, Any]]:
        if not isinstance(graph, TGraph) or not graph._validate_edge_index(index):
            return None
        e = graph._edges[index]
        return dict(e, dictionary=dict(e.get("dictionary", {})))

    def VertexDictionary(self, index: int) -> Dict[str, Any]:
        if not self._validate_vertex_index(index, active=False):
            return {}
        return dict(self._vertices[index].get("dictionary", {}))

    def EdgeDictionary(self, index: int) -> Dict[str, Any]:
        if not self._validate_edge_index(index, active=False):
            return {}
        return dict(self._edges[index].get("dictionary", {}))

    def SetVertexDictionary(self, index: int, dictionary: Optional[Dict[str, Any]] = None) -> "TGraph":
        if not self._validate_vertex_index(index, active=False):
            return self
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d["index"] = index
        d.setdefault("active", self._vertices[index].get("active", True))
        self._vertices[index]["dictionary"] = d
        self._invalidate_cache()
        return self

    def SetEdgeDictionary(self, index: int, dictionary: Optional[Dict[str, Any]] = None) -> "TGraph":
        if not self._validate_edge_index(index, active=False):
            return self
        e = self._edges[index]
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d.update({"index": index, "src": e["src"], "dst": e["dst"], "directed": e["directed"], "active": e.get("active", True)})
        self._edges[index]["dictionary"] = d
        self._invalidate_cache()
        return self

    # Static aliases for TopologicPy style
    @staticmethod
    def VertexDictionaryStatic(graph: "TGraph", index: int) -> Dict[str, Any]:
        return graph.VertexDictionary(index) if isinstance(graph, TGraph) else {}

    # ---------------------------------------------------------------------
    # Graph queries
    # ---------------------------------------------------------------------





    @staticmethod
    def AdjacentVertices(graph: "TGraph", vertex: Union[int, Dict[str, Any]], mode: str = "out") -> List[Dict[str, Any]]:
        idx = TGraph._as_index(vertex)
        return [TGraph.Vertex(graph, i) for i in TGraph.AdjacentIndices(graph, idx, mode=mode)] if isinstance(graph, TGraph) else []

    @staticmethod
    def IncidentEdges(graph: "TGraph", index: int, mode: str = "all") -> List[Dict[str, Any]]:
        if not isinstance(graph, TGraph) or not graph._validate_vertex_index(index):
            return []
        mode = str(mode).lower()
        if mode == "out":
            ids = graph._out_edges.get(index, set())
        elif mode == "in":
            ids = graph._in_edges.get(index, set())
        else:
            ids = graph._incident_edges.get(index, set())
        return [TGraph.Edge(graph, i) for i in sorted(ids) if graph._validate_edge_index(i)]

    @staticmethod
    def AdjacentEdges(graph: "TGraph", edge: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return []
        idx = TGraph._as_index(edge)
        if idx is None and isinstance(edge, dict):
            idx = edge.get("index")
        if not graph._validate_edge_index(idx):
            return []
        e = graph._edges[idx]
        ids = set(graph._incident_edges.get(e["src"], set())) | set(graph._incident_edges.get(e["dst"], set()))
        ids.discard(idx)
        return [TGraph.Edge(graph, i) for i in sorted(ids) if graph._validate_edge_index(i)]

    @staticmethod
    def IncomingVertices(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return TGraph.AdjacentVertices(graph, vertex, mode="in")

    @staticmethod
    def OutgoingVertices(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return TGraph.AdjacentVertices(graph, vertex, mode="out")

    @staticmethod
    def IncomingEdges(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return TGraph.IncidentEdges(graph, TGraph._as_index(vertex), mode="in")

    @staticmethod
    def OutgoingEdges(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return TGraph.IncidentEdges(graph, TGraph._as_index(vertex), mode="out")



    @staticmethod
    def HasEdge(graph: "TGraph", src: int, dst: int, directed: Optional[bool] = None) -> bool:
        if not isinstance(graph, TGraph) or not graph._validate_vertex_index(src) or not graph._validate_vertex_index(dst):
            return False
        edge_directed = graph._directed if directed is None else bool(directed)
        key = graph._edge_key(src, dst, edge_directed)
        return any(graph._validate_edge_index(i) for i in graph._edge_lookup.get(key, set()))

    @staticmethod
    def EdgesBetween(graph: "TGraph", src: int, dst: int, directed: Optional[bool] = None) -> List[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return []
        edge_directed = graph._directed if directed is None else bool(directed)
        ids = graph._edge_lookup.get(graph._edge_key(src, dst, edge_directed), set())
        return [TGraph.Edge(graph, i) for i in sorted(ids) if graph._validate_edge_index(i)]

    @staticmethod
    def EdgeBetween(graph: "TGraph", src: int, dst: int, directed: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        edges = TGraph.EdgesBetween(graph, src, dst, directed=directed)
        return edges[0] if edges else None

    @staticmethod
    def VertexIndex(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> Optional[int]:
        idx = TGraph._as_index(vertex)
        return idx if isinstance(graph, TGraph) and graph._validate_vertex_index(idx) else None

    @staticmethod
    def EdgeIndex(graph: "TGraph", edge: Union[int, Dict[str, Any]]) -> Optional[int]:
        idx = TGraph._as_index(edge)
        if idx is None and isinstance(edge, dict):
            idx = edge.get("index")
        return idx if isinstance(graph, TGraph) and graph._validate_edge_index(idx) else None

    @staticmethod
    def VertexByKeyValue(graph: "TGraph", key: str = None, value: Any = None, silent: bool = False) -> Optional[Dict[str, Any]]:
        if not isinstance(graph, TGraph) or key is None:
            return None
        for v in graph._vertices:
            if v.get("active", True) and v.get("dictionary", {}).get(key) == value:
                return TGraph.Vertex(graph, v["index"])
        return None

    # ---------------------------------------------------------------------
    # Matrices and adjacency exports
    # ---------------------------------------------------------------------



    @staticmethod
    def AdjacencyDictionary(graph: "TGraph", vertexLabelKey: str = "label", includeWeights: bool = False,
                            edgeKey: str = "weight") -> Dict[Any, List[Any]]:
        if not isinstance(graph, TGraph):
            return {}
        result = {}
        for v in graph._vertices:
            if not v.get("active", True):
                continue
            idx = v["index"]
            label = v.get("dictionary", {}).get(vertexLabelKey, idx)
            row = []
            for nb in TGraph.AdjacentIndices(graph, idx):
                nb_label = graph._vertices[nb].get("dictionary", {}).get(vertexLabelKey, nb)
                if includeWeights:
                    e = TGraph.EdgeBetween(graph, idx, nb)
                    w = e.get("dictionary", {}).get(edgeKey, 1) if e else 1
                    row.append((nb_label, w))
                else:
                    row.append(nb_label)
            result[label] = row
        return result

    @staticmethod
    def AdjacencyMatrixCSVString(graph: "TGraph", **kwargs) -> str:
        matrix = TGraph.AdjacencyMatrix(graph, **kwargs)
        try:
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerows(matrix)
            return out.getvalue()
        except Exception:
            return ""

    # ---------------------------------------------------------------------
    # Traversal and connectivity
    # ---------------------------------------------------------------------




    @staticmethod
    def IsolatedVertices(graph: "TGraph") -> List[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return []
        return [TGraph.Vertex(graph, v["index"]) for v in graph._vertices if v.get("active", True) and TGraph.Degree(graph, v["index"]) == 0]

    @staticmethod
    def Leaves(graph: "TGraph") -> List[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return []
        return [TGraph.Vertex(graph, v["index"]) for v in graph._vertices if v.get("active", True) and TGraph.Degree(graph, v["index"]) == 1]

    # ---------------------------------------------------------------------
    # Compatibility / utility methods
    # ---------------------------------------------------------------------

    @staticmethod
    def Copy(graph: "TGraph") -> Optional["TGraph"]:
        return TGraph.FromPython(TGraph.ToPython(graph, includeRepresentations=True)) if isinstance(graph, TGraph) else None

    @staticmethod
    def Subgraph(graph: "TGraph", vertices: Iterable[int], induced: bool = True) -> Optional["TGraph"]:
        if not isinstance(graph, TGraph):
            return None
        old_indices = [i for i in vertices if graph._validate_vertex_index(i)]
        mapping = {old: new for new, old in enumerate(old_indices)}
        g = TGraph(directed=graph._directed, allowSelfLoops=graph._allow_self_loops,
                   allowParallelEdges=graph._allow_parallel_edges, dictionary=graph._dictionary)
        for old in old_indices:
            rec = graph._vertices[old]
            g.AddVertex(dictionary=dict(rec.get("dictionary", {})), representation=rec.get("representation"))
        if induced:
            allowed = set(old_indices)
            for e in graph._edges:
                if not e.get("active", True):
                    continue
                if e["src"] in allowed and e["dst"] in allowed:
                    g.AddEdge(mapping[e["src"]], mapping[e["dst"]], directed=e["directed"],
                              dictionary=dict(e.get("dictionary", {})), representation=e.get("representation"))
        return g

    @staticmethod
    def Coordinates(graph: "TGraph", vertex: Union[int, Dict[str, Any]], default: Optional[List[float]] = None) -> Optional[List[float]]:
        if not isinstance(graph, TGraph):
            return default
        idx = TGraph._as_index(vertex)
        if not graph._validate_vertex_index(idx):
            return default
        d = graph._vertices[idx].get("dictionary", {})
        if all(isinstance(d.get(k), (int, float)) for k in ["x", "y", "z"]):
            return [float(d["x"]), float(d["y"]), float(d["z"])]
        rep = graph._vertices[idx].get("representation")
        if rep is not None:
            try:
                from topologicpy.Vertex import Vertex
                from topologicpy.Topology import Topology
                if Topology.IsInstance(rep, "Vertex"):
                    return [float(x) for x in Vertex.Coordinates(rep)]
                if Topology.IsInstance(rep, "Topology"):
                    c = Topology.CenterOfMass(rep)
                    return [float(x) for x in Vertex.Coordinates(c)]
            except Exception:
                pass
        return default

    @staticmethod
    def NearestVertex(graph: "TGraph", x: float = 0, y: float = 0, z: float = 0, vertex: Any = None) -> Optional[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return None
        if vertex is not None:
            try:
                from topologicpy.Vertex import Vertex
                x, y, z = Vertex.Coordinates(vertex)
            except Exception:
                pass
        best = None
        best_d2 = None
        for rec in graph._vertices:
            if not rec.get("active", True):
                continue
            c = TGraph.Coordinates(graph, rec["index"])
            if c is None:
                continue
            d2 = (c[0] - x) ** 2 + (c[1] - y) ** 2 + (c[2] - z) ** 2
            if best is None or d2 < best_d2:
                best = rec["index"]
                best_d2 = d2
        return TGraph.Vertex(graph, best) if best is not None else None

    @staticmethod
    def AABB(graph: "TGraph", pad: float = 0.0) -> Optional[Any]:
        if not isinstance(graph, TGraph):
            return None
        pts = [TGraph.Coordinates(graph, v["index"]) for v in graph._vertices if v.get("active", True)]
        pts = [p for p in pts if p is not None]
        if not pts:
            return None
        try:
            from topologicpy.BVH import AABB
            return AABB.from_points(pts=pts, pad=pad)
        except Exception:
            xs, ys, zs = zip(*pts)
            return {"xmin": min(xs) - pad, "xmax": max(xs) + pad,
                    "ymin": min(ys) - pad, "ymax": max(ys) + pad,
                    "zmin": min(zs) - pad, "zmax": max(zs) + pad}

    @staticmethod
    def _EdgeLength(graph: "TGraph", edge: Dict[str, Any], mantissa: int = 6, tolerance: float = 0.0001) -> float:
        c1 = TGraph.Coordinates(graph, edge.get("src"))
        c2 = TGraph.Coordinates(graph, edge.get("dst"))
        if c1 is None or c2 is None:
            return 1.0
        return round(math.dist(c1, c2), mantissa)

    @staticmethod
    def NetworkXGraph(graph: "TGraph") -> Optional[Any]:
        if not isinstance(graph, TGraph):
            return None
        try:
            import networkx as nx
            G = nx.DiGraph() if graph._directed else nx.Graph()
            if graph._allow_parallel_edges:
                G = nx.MultiDiGraph() if graph._directed else nx.MultiGraph()
            for v in graph._vertices:
                if v.get("active", True):
                    d = dict(v.get("dictionary", {}))
                    G.add_node(v["index"], **d)
            for e in graph._edges:
                if e.get("active", True):
                    d = dict(e.get("dictionary", {}))
                    G.add_edge(e["src"], e["dst"], **d)
            return G
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Topologic representation/view methods
    # ---------------------------------------------------------------------

    @staticmethod
    def _EdgeRepresentationToTopology(representation: Any, dictionary: Optional[Dict[str, Any]] = None,
                                      segmentCurves: bool = True, tolerance: float = 0.0001,
                                      silent: bool = False) -> Optional[Any]:
        if representation is None:
            return None
        d = dictionary if isinstance(dictionary, dict) else {}
        try:
            from topologicpy.Topology import Topology
            if Topology.IsInstance(representation, "Topology"):
                return Topology.SetDictionary(representation, TGraph._PythonToDictionary(d), silent=True)
        except Exception:
            pass
        control_points = None
        if isinstance(representation, dict):
            if str(representation.get("type", "")).lower() in ["bezier", "polyline", "wire", "curve"]:
                control_points = representation.get("control_points")
        elif isinstance(representation, list):
            control_points = representation
        if control_points and segmentCurves:
            return TGraph._ControlPointsToWire(control_points, dictionary=d, tolerance=tolerance, silent=silent)
        return None

    @staticmethod
    def _ControlPointsToWire(controlPoints: List[Any], dictionary: Optional[Dict[str, Any]] = None,
                             tolerance: float = 0.0001, silent: bool = False) -> Optional[Any]:
        vertices = []
        for p in controlPoints:
            v = None
            try:
                from topologicpy.Topology import Topology
                if Topology.IsInstance(p, "Vertex"):
                    v = p
            except Exception:
                pass
            if v is None and isinstance(p, (list, tuple)) and len(p) >= 3:
                try:
                    from topologicpy.Vertex import Vertex
                    v = Vertex.ByCoordinates(float(p[0]), float(p[1]), float(p[2]))
                except Exception:
                    pass
            if v is not None:
                vertices.append(v)
        if len(vertices) < 2:
            return None
        edges = []
        try:
            from topologicpy.Edge import Edge
            for i in range(len(vertices) - 1):
                e = Edge.ByStartVertexEndVertex(vertices[i], vertices[i + 1], tolerance=tolerance)
                if e is not None:
                    edges.append(e)
        except Exception:
            return None
        if not edges:
            return None
        try:
            from topologicpy.Wire import Wire
            from topologicpy.Topology import Topology
            w = Wire.ByEdges(edges, tolerance=tolerance)
            return Topology.SetDictionary(w, TGraph._PythonToDictionary(dictionary), silent=True)
        except Exception:
            return None

    @staticmethod
    def Topology(graph: "TGraph", includeVertices: bool = True, includeEdges: bool = True,
                 useRepresentations: bool = True, segmentCurves: bool = True,
                 tolerance: float = 0.0001, silent: bool = False) -> Optional[Any]:
        if not isinstance(graph, TGraph):
            return None
        topologies = []
        if includeVertices:
            topologies.extend(TGraph.Vertices(graph, asTopologic=True, useRepresentation=useRepresentations,
                                             tolerance=tolerance, silent=silent))
        if includeEdges:
            topologies.extend(TGraph.Edges(graph, asTopologic=True, useRepresentation=useRepresentations,
                                          segmentCurves=segmentCurves, tolerance=tolerance, silent=silent))
        if not topologies:
            return None
        try:
            from topologicpy.Cluster import Cluster
            return Cluster.ByTopologies(topologies, silent=silent)
        except TypeError:
            try:
                from topologicpy.Cluster import Cluster
                return Cluster.ByTopologies(topologies)
            except Exception:
                return None
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------

    @staticmethod
    def ToPython(graph: "TGraph", includeRepresentations: bool = False) -> Dict[str, Any]:
        if not isinstance(graph, TGraph):
            return {}
        vertices = []
        for v in graph._vertices:
            record = {"index": v["index"], "active": v.get("active", True),
                      "dictionary": dict(v.get("dictionary", {}))}
            if includeRepresentations:
                record["representation"] = v.get("representation")
            vertices.append(record)
        edges = []
        for e in graph._edges:
            record = {"index": e["index"], "src": e["src"], "dst": e["dst"],
                      "directed": e["directed"], "active": e.get("active", True),
                      "dictionary": dict(e.get("dictionary", {}))}
            if includeRepresentations:
                record["representation"] = e.get("representation")
            edges.append(record)
        return {"type": "TGraph", "directed": graph._directed,
                "allowSelfLoops": graph._allow_self_loops,
                "allowParallelEdges": graph._allow_parallel_edges,
                "dictionary": dict(graph._dictionary), "vertices": vertices, "edges": edges}

    @staticmethod
    def FromPython(data: Dict[str, Any]) -> Optional["TGraph"]:
        if not isinstance(data, dict):
            return None
        g = TGraph(directed=bool(data.get("directed", False)),
                   allowSelfLoops=bool(data.get("allowSelfLoops", True)),
                   allowParallelEdges=bool(data.get("allowParallelEdges", False)),
                   dictionary=data.get("dictionary", {}))
        vertices = sorted([v for v in data.get("vertices", []) if isinstance(v, dict)], key=lambda x: x.get("index", 0))
        next_expected = 0
        for v in vertices:
            target = v.get("index", next_expected)
            while next_expected < target:
                idx = g.AddVertex(dictionary={"active": False})
                g._vertices[idx]["active"] = False
                g._vertices[idx]["dictionary"]["active"] = False
                next_expected += 1
            idx = g.AddVertex(dictionary=v.get("dictionary", {}), representation=v.get("representation"))
            g._vertices[idx]["active"] = bool(v.get("active", True))
            g._vertices[idx]["dictionary"]["active"] = g._vertices[idx]["active"]
            next_expected += 1
        for e in data.get("edges", []):
            if not isinstance(e, dict):
                continue
            idx = g.AddEdge(e.get("src"), e.get("dst"), directed=e.get("directed"),
                            dictionary=e.get("dictionary", {}), representation=e.get("representation"))
            if idx is not None and not bool(e.get("active", True)):
                g.RemoveEdge(idx, silent=True)
        return g

    @staticmethod
    def JSONData(graph: "TGraph", includeRepresentations: bool = False) -> Dict[str, Any]:
        return TGraph.ToPython(graph, includeRepresentations=includeRepresentations)

    @staticmethod
    def JSONString(graph: "TGraph", indent: Optional[int] = None, includeRepresentations: bool = False) -> str:
        try:
            return json.dumps(TGraph.JSONData(graph, includeRepresentations=includeRepresentations), indent=indent)
        except Exception:
            return "{}"

    # ---------------------------------------------------------------------
    # Phase 3 graph algorithms
    # ---------------------------------------------------------------------

    @staticmethod
    def _Phase3ActiveVertexIndices(graph: "TGraph") -> List[int]:
        if not isinstance(graph, TGraph):
            return []
        return [v["index"] for v in graph._vertices if v.get("active", True)]

    @staticmethod
    def _Phase3ActiveEdges(graph: "TGraph") -> List[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return []
        return [e for e in graph._edges if e.get("active", True)]

    @staticmethod
    def _Phase3SetVertexValue(graph: "TGraph", index: int, key: Optional[str], value: Any) -> None:
        if not isinstance(graph, TGraph) or key is None:
            return
        if not graph._validate_vertex_index(index):
            return
        graph._vertices[index].setdefault("dictionary", {})[key] = value

    @staticmethod
    def _Phase3EdgeWeight(edge: Dict[str, Any], edgeKey: Optional[str] = None, default: float = 1.0) -> float:
        if not isinstance(edge, dict):
            return float(default)
        if edgeKey is None:
            return float(default)
        d = edge.get("dictionary", {})
        if not isinstance(d, dict):
            return float(default)
        value = d.get(edgeKey, default)
        try:
            value = float(value)
        except Exception:
            value = float(default)
        return value if value > 0 else float(default)

    @staticmethod
    def _Phase3Neighbors(graph: "TGraph", index: int, mode: str = "out") -> List[int]:
        return TGraph.AdjacentIndices(graph, index, mode=mode)

    @staticmethod
    def _Phase3UndirectedAdjacency(graph: "TGraph") -> Dict[int, Set[int]]:
        adjacency = {i: set() for i in TGraph._Phase3ActiveVertexIndices(graph)}
        for e in TGraph._Phase3ActiveEdges(graph):
            src = e.get("src")
            dst = e.get("dst")
            if src in adjacency and dst in adjacency:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
        return adjacency

    @staticmethod
    def _Phase3WeightedEdges(graph: "TGraph", edgeKey: Optional[str] = None) -> List[Tuple[float, int, int, int]]:
        result = []
        for e in TGraph._Phase3ActiveEdges(graph):
            src = e.get("src")
            dst = e.get("dst")
            if not graph._validate_vertex_index(src) or not graph._validate_vertex_index(dst):
                continue
            weight = TGraph._Phase3EdgeWeight(e, edgeKey=edgeKey, default=1.0)
            result.append((weight, src, dst, e.get("index")))
        return result



    @staticmethod
    def Density(graph: "TGraph", includeSelfLoops: bool = False) -> float:
        """
        Returns graph density using active vertices and active edges.
        """

        if not isinstance(graph, TGraph):
            return 0.0
        n = TGraph.Order(graph)
        m = TGraph.Size(graph)
        if n <= 1:
            return 0.0
        if graph._directed:
            possible = n * n if includeSelfLoops and graph._allow_self_loops else n * (n - 1)
        else:
            possible = n * (n + 1) / 2.0 if includeSelfLoops and graph._allow_self_loops else n * (n - 1) / 2.0
        return float(m) / float(possible) if possible > 0 else 0.0


    @staticmethod
    def IsComplete(graph: "TGraph", includeSelfLoops: bool = False) -> bool:
        """
        Returns True if every active vertex is connected to every other active vertex.
        """

        if not isinstance(graph, TGraph):
            return False
        vertices = TGraph._Phase3ActiveVertexIndices(graph)
        for i in vertices:
            for j in vertices:
                if i == j and not includeSelfLoops:
                    continue
                if i == j and includeSelfLoops:
                    if not TGraph.HasEdge(graph, i, j, directed=graph._directed):
                        return False
                    continue
                if graph._directed:
                    if not TGraph.HasEdge(graph, i, j, directed=True):
                        return False
                else:
                    if not TGraph.HasEdge(graph, i, j, directed=False):
                        return False
        return True

    @staticmethod
    def IsTree(graph: "TGraph") -> bool:
        """
        Returns True if the active graph is an undirected tree.
        """

        if not isinstance(graph, TGraph):
            return False
        n = TGraph.Order(graph)
        if n == 0:
            return True
        if not TGraph.IsConnected(graph, mode="all"):
            return False
        non_loop_edges = 0
        for e in TGraph._Phase3ActiveEdges(graph):
            if e.get("src") != e.get("dst"):
                non_loop_edges += 1
            else:
                return False
        return non_loop_edges == n - 1

    @staticmethod
    def IsBipartite(graph: "TGraph") -> bool:
        """
        Returns True if the active graph is bipartite, treating edges as undirected.
        """

        if not isinstance(graph, TGraph):
            return False
        adjacency = TGraph._Phase3UndirectedAdjacency(graph)
        color = {}
        for start in sorted(adjacency.keys()):
            if start in color:
                continue
            color[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adjacency[u]:
                    if u == v:
                        return False
                    if v not in color:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        return False
        return True

    @staticmethod
    def Bridges(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Returns bridge edge records, treating the graph as undirected.

        Parallel edges are not bridges because removing one still leaves another
        connection between the same vertex pair.
        """

        if not isinstance(graph, TGraph):
            return []
        vertices = TGraph._Phase3ActiveVertexIndices(graph)
        adjacency_edges: Dict[int, List[Tuple[int, int]]] = {v: [] for v in vertices}
        pair_counts: Dict[Tuple[int, int], int] = {}

        for e in TGraph._Phase3ActiveEdges(graph):
            u = e.get("src")
            v = e.get("dst")
            eid = e.get("index")
            if u == v:
                continue
            if u not in adjacency_edges or v not in adjacency_edges:
                continue
            a, b = (u, v) if u <= v else (v, u)
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
            adjacency_edges[u].append((v, eid))
            adjacency_edges[v].append((u, eid))

        visited = set()
        disc = {}
        low = {}
        parent_edge = {}
        time_counter = [0]
        bridges = []

        def dfs(u: int) -> None:
            visited.add(u)
            disc[u] = low[u] = time_counter[0]
            time_counter[0] += 1
            for v, eid in adjacency_edges[u]:
                if eid == parent_edge.get(u, None):
                    continue
                if v not in visited:
                    parent_edge[v] = eid
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    a, b = (u, v) if u <= v else (v, u)
                    if low[v] > disc[u] and pair_counts.get((a, b), 0) == 1:
                        bridges.append(TGraph.Edge(graph, eid))
                else:
                    low[u] = min(low[u], disc[v])

        for v in vertices:
            if v not in visited:
                dfs(v)

        return bridges

    @staticmethod
    def CutVertices(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Returns articulation vertex records, treating the graph as undirected.
        """

        if not isinstance(graph, TGraph):
            return []
        adjacency = TGraph._Phase3UndirectedAdjacency(graph)
        visited = set()
        disc = {}
        low = {}
        parent = {}
        articulation = set()
        time_counter = [0]

        def dfs(u: int) -> None:
            visited.add(u)
            disc[u] = low[u] = time_counter[0]
            time_counter[0] += 1
            children = 0
            for v in adjacency[u]:
                if v == u:
                    continue
                if v not in visited:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if u not in parent and children > 1:
                        articulation.add(u)
                    if u in parent and low[v] >= disc[u]:
                        articulation.add(u)
                elif parent.get(u, None) != v:
                    low[u] = min(low[u], disc[v])

        for v in sorted(adjacency.keys()):
            if v not in visited:
                dfs(v)

        return [TGraph.Vertex(graph, i) for i in sorted(articulation)]




    # ---------------------------------------------------------------------
    # Phase 4: Topology / spatial graph builders
    # ---------------------------------------------------------------------

    @staticmethod
    def _Phase4ValueAtKey(data: Any, key: str, default: Any = None) -> Any:
        if key is None:
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        try:
            from topologicpy.Dictionary import Dictionary
            return Dictionary.ValueAtKey(data, key, default)
        except TypeError:
            try:
                from topologicpy.Dictionary import Dictionary
                value = Dictionary.ValueAtKey(data, key)
                return default if value is None else value
            except Exception:
                return default
        except Exception:
            return default

    @staticmethod
    def _Phase4SetValue(data: Optional[Dict[str, Any]], key: str, value: Any) -> Dict[str, Any]:
        d = dict(data) if isinstance(data, dict) else {}
        d[key] = value
        return d

    @staticmethod
    def _Phase4TopologyType(topology: Any) -> str:
        if topology is None:
            return "None"
        try:
            from topologicpy.Topology import Topology
            return str(Topology.TypeAsString(topology))
        except Exception:
            return type(topology).__name__

    @staticmethod
    def _Phase4Coordinates(topology: Any, useInternalVertex: bool = False, mantissa: int = 6, tolerance: float = 0.0001) -> Optional[List[float]]:
        if topology is None:
            return None
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Vertex import Vertex
            v = None
            if Topology.IsInstance(topology, "Vertex"):
                v = topology
            elif useInternalVertex:
                try:
                    v = Topology.InternalVertex(topology, tolerance=tolerance)
                except Exception:
                    v = None
            if v is None:
                try:
                    v = Topology.CenterOfMass(topology)
                except Exception:
                    v = None
            if v is None:
                return None
            return [round(float(c), mantissa) for c in Vertex.Coordinates(v)]
        except Exception:
            return None

    @staticmethod
    def _Phase4BREPString(topology: Any) -> Optional[str]:
        try:
            from topologicpy.Topology import Topology
            return Topology.BREPString(topology)
        except Exception:
            return None

    @staticmethod
    def _Phase4TopologyDictionary(topology: Any, storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001, useInternalVertex: bool = False) -> Dict[str, Any]:
        d = TGraph._TopologyDictionaryToPython(topology)
        d["topology_type"] = TGraph._Phase4TopologyType(topology)
        coords = TGraph._Phase4Coordinates(topology, useInternalVertex=useInternalVertex, mantissa=mantissa, tolerance=tolerance)
        if coords is not None:
            d.setdefault("x", coords[0])
            d.setdefault("y", coords[1])
            d.setdefault("z", coords[2])
        if storeBREP:
            brep = TGraph._Phase4BREPString(topology)
            if brep is not None:
                d["brep"] = brep
        return d

    @staticmethod
    def _Phase4Topologies(topology: Any, topologyType: str, free: bool = False, tolerance: float = 0.0001) -> List[Any]:
        if topology is None:
            return []
        topologyType = str(topologyType or "").lower()
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Cluster import Cluster
            if topologyType == "cellcomplex":
                return Topology.CellComplexes(topology, silent=True) or []
            if topologyType == "cell":
                return (Cluster.FreeCells(topology, tolerance=tolerance) if free else Topology.Cells(topology, silent=True)) or []
            if topologyType == "shell":
                return (Cluster.FreeShells(topology, tolerance=tolerance) if free else Topology.Shells(topology, silent=True)) or []
            if topologyType == "face":
                return (Cluster.FreeFaces(topology, tolerance=tolerance) if free else Topology.Faces(topology, silent=True)) or []
            if topologyType == "wire":
                return (Cluster.FreeWires(topology, tolerance=tolerance) if free else Topology.Wires(topology, silent=True)) or []
            if topologyType == "edge":
                return (Cluster.FreeEdges(topology, tolerance=tolerance) if free else Topology.Edges(topology, silent=True)) or []
            if topologyType == "vertex":
                return (Cluster.FreeVertices(topology, tolerance=tolerance) if free else Topology.Vertices(topology, silent=True)) or []
        except TypeError:
            try:
                from topologicpy.Topology import Topology
                if topologyType == "cellcomplex":
                    return Topology.CellComplexes(topology) or []
                if topologyType == "cell":
                    return Topology.Cells(topology) or []
                if topologyType == "shell":
                    return Topology.Shells(topology) or []
                if topologyType == "face":
                    return Topology.Faces(topology) or []
                if topologyType == "wire":
                    return Topology.Wires(topology) or []
                if topologyType == "edge":
                    return Topology.Edges(topology) or []
                if topologyType == "vertex":
                    return Topology.Vertices(topology) or []
            except Exception:
                return []
        except Exception:
            return []
        return []

    @staticmethod
    def _Phase4Apertures(topology: Any) -> List[Any]:
        try:
            from topologicpy.Topology import Topology
            return Topology.Apertures(topology) or []
        except Exception:
            return []

    @staticmethod
    def _Phase4Contents(topology: Any) -> List[Any]:
        try:
            from topologicpy.Topology import Topology
            return Topology.Contents(topology, silent=True) or []
        except TypeError:
            try:
                from topologicpy.Topology import Topology
                return Topology.Contents(topology) or []
            except Exception:
                return []
        except Exception:
            return []

    @staticmethod
    def _Phase4TopologyFromAperture(topology: Any) -> Any:
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Aperture import Aperture
            if Topology.IsInstance(topology, "Aperture"):
                return Aperture.Topology(topology)
        except Exception:
            pass
        return topology

    @staticmethod
    def _Phase4BoundaryKey(topology: Any, boundaryType: str, mantissa: int = 6) -> Any:
        boundaryType = str(boundaryType or "").lower()
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Vertex import Vertex
            vertices = Topology.Vertices(topology, silent=True) or []
            coords = []
            for v in vertices:
                coords.append(tuple(round(float(c), mantissa) for c in Vertex.Coordinates(v)))
            if coords:
                return (boundaryType, tuple(sorted(coords)))
        except Exception:
            pass
        return (boundaryType, id(topology))

    @staticmethod
    def _Phase4Children(topology: Any, childType: str, tolerance: float = 0.0001) -> List[Any]:
        return TGraph._Phase4Topologies(topology, childType, free=False, tolerance=tolerance)

    @staticmethod
    def _Phase4OwnerBoundaryMaps(owners: List[Any], childType: str, mantissa: int = 6, tolerance: float = 0.0001) -> Tuple[Dict[Any, List[int]], Dict[Any, List[Any]], Dict[int, List[Tuple[Any, Any]]]]:
        boundary_to_owner_indices: Dict[Any, List[int]] = {}
        boundary_to_refs: Dict[Any, List[Any]] = {}
        owner_to_boundaries: Dict[int, List[Tuple[Any, Any]]] = {}
        seen_owner_boundary = set()

        for owner_index, owner in enumerate(owners):
            for child in TGraph._Phase4Children(owner, childType, tolerance=tolerance):
                bk = TGraph._Phase4BoundaryKey(child, childType, mantissa=mantissa)
                if (owner_index, bk) not in seen_owner_boundary:
                    boundary_to_owner_indices.setdefault(bk, []).append(owner_index)
                    owner_to_boundaries.setdefault(owner_index, []).append((bk, child))
                    seen_owner_boundary.add((owner_index, bk))
                boundary_to_refs.setdefault(bk, []).append(child)
        return boundary_to_owner_indices, boundary_to_refs, owner_to_boundaries

    @staticmethod
    def _Phase4AddTopologyVertex(graph: "TGraph", topology: Any, category: Any = None, label: Any = None,
                                 storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001,
                                 useInternalVertex: bool = False, extra: Optional[Dict[str, Any]] = None) -> int:
        d = TGraph._Phase4TopologyDictionary(topology, storeBREP=storeBREP, mantissa=mantissa, tolerance=tolerance, useInternalVertex=useInternalVertex)
        if category is not None:
            d["category"] = category
        if label is not None:
            d.setdefault("label", label)
        if isinstance(extra, dict):
            d.update(extra)
        return graph.AddVertex(dictionary=d, representation=topology)

    @staticmethod
    def _Phase4AddRelationship(graph: "TGraph", src: int, dst: int, relationship: str, category: Any = None,
                               source: Any = None, dictionary: Optional[Dict[str, Any]] = None,
                               directed: Optional[bool] = None) -> Optional[int]:
        if src is None or dst is None:
            return None
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d["relationship"] = relationship
        if category is not None:
            d["category"] = category
        if source is not None:
            sd = TGraph._TopologyDictionaryToPython(TGraph._Phase4TopologyFromAperture(source))
            for k, v in sd.items():
                d.setdefault(k, v)
            d.setdefault("source_topology_type", TGraph._Phase4TopologyType(TGraph._Phase4TopologyFromAperture(source)))
        return graph.AddEdge(src, dst, directed=directed, dictionary=d, representation=source)

    @staticmethod
    def _Phase4ProcessCollection(graph: "TGraph", host: Any, ownerType: str, childType: str,
                                 direct: bool = True, directApertures: bool = False,
                                 viaSharedTopologies: bool = False, viaSharedApertures: bool = False,
                                 toExteriorTopologies: bool = False, toExteriorApertures: bool = False,
                                 toContents: bool = False, toOutposts: bool = False,
                                 outpostLookup: Optional[Dict[Any, Any]] = None,
                                 idKey: str = "TOPOLOGIC_ID", outpostsKey: str = "outposts",
                                 storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001,
                                 useInternalVertex: bool = False) -> None:
        owners = TGraph._Phase4Topologies(host, ownerType, free=False, tolerance=tolerance)
        owner_vertex = {}
        for i, owner in enumerate(owners):
            owner_vertex[i] = TGraph._Phase4AddTopologyVertex(graph, owner, category=0, label=ownerType,
                                                              storeBREP=storeBREP, mantissa=mantissa,
                                                              tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                              extra={"role": "owner"})

        boundary_to_owners, boundary_to_refs, owner_to_boundaries = TGraph._Phase4OwnerBoundaryMaps(owners, childType, mantissa=mantissa, tolerance=tolerance)

        if direct or directApertures:
            seen = set()
            for bk, incident in boundary_to_owners.items():
                if len(incident) < 2:
                    continue
                refs = boundary_to_refs.get(bk, []) or []
                has_aperture = any(TGraph._Phase4Apertures(ref) for ref in refs)
                for a_i in range(len(incident) - 1):
                    for b_i in range(a_i + 1, len(incident)):
                        oi = incident[a_i]
                        oj = incident[b_i]
                        src = owner_vertex.get(oi)
                        dst = owner_vertex.get(oj)
                        if src is None or dst is None:
                            continue
                        pair = (min(src, dst), max(src, dst), bk)
                        if direct and pair not in seen:
                            seen.add(pair)
                            TGraph._Phase4AddRelationship(graph, src, dst, "Direct", category=0,
                                                           source=refs[0] if refs else None, directed=False)
                        if directApertures and has_aperture:
                            aps = []
                            for ref in refs:
                                aps.extend(TGraph._Phase4Apertures(ref))
                            source = aps[0] if aps else (refs[0] if refs else None)
                            ap_pair = pair + ("aperture",)
                            if ap_pair not in seen:
                                seen.add(ap_pair)
                                TGraph._Phase4AddRelationship(graph, src, dst, "Direct_Apertures", category=2,
                                                               source=source, directed=False)

        boundary_vertex = {}
        aperture_vertex = {}

        def boundary_index(boundary: Any, category: Any, relationship_label: str) -> int:
            bk = TGraph._Phase4BoundaryKey(boundary, childType, mantissa=mantissa)
            key = (category, bk)
            if key not in boundary_vertex:
                boundary_vertex[key] = TGraph._Phase4AddTopologyVertex(graph, boundary, category=category,
                                                                        label=relationship_label,
                                                                        storeBREP=storeBREP, mantissa=mantissa,
                                                                        tolerance=tolerance,
                                                                        useInternalVertex=useInternalVertex,
                                                                        extra={"role": "boundary"})
            return boundary_vertex[key]

        def aperture_index(aperture: Any, category: Any, relationship_label: str) -> int:
            ap_topology = TGraph._Phase4TopologyFromAperture(aperture)
            key = (category, id(aperture))
            if key not in aperture_vertex:
                aperture_vertex[key] = TGraph._Phase4AddTopologyVertex(graph, ap_topology, category=category,
                                                                        label=relationship_label,
                                                                        storeBREP=storeBREP, mantissa=mantissa,
                                                                        tolerance=tolerance,
                                                                        useInternalVertex=useInternalVertex,
                                                                        extra={"role": "aperture"})
            return aperture_vertex[key]

        for owner_index, owner in enumerate(owners):
            src = owner_vertex.get(owner_index)
            if src is None:
                continue
            for bk, child in owner_to_boundaries.get(owner_index, []):
                is_shared = len(boundary_to_owners.get(bk, [])) > 1
                apertures = TGraph._Phase4Apertures(child)
                if is_shared and viaSharedTopologies:
                    dst = boundary_index(child, 1, "Shared Topology")
                    TGraph._Phase4AddRelationship(graph, src, dst, "Via_Shared_Topologies", category=1, source=child, directed=False)
                if is_shared and viaSharedApertures:
                    for aperture in apertures:
                        dst = aperture_index(aperture, 2, "Shared Aperture")
                        TGraph._Phase4AddRelationship(graph, src, dst, "Via_Shared_Apertures", category=2, source=aperture, directed=False)
                if (not is_shared) and toExteriorTopologies:
                    dst = boundary_index(child, 3, "Exterior Topology")
                    TGraph._Phase4AddRelationship(graph, src, dst, "To_Exterior_Topologies", category=3, source=child, directed=False)
                if (not is_shared) and toExteriorApertures:
                    for aperture in apertures:
                        dst = aperture_index(aperture, 4, "Exterior Aperture")
                        TGraph._Phase4AddRelationship(graph, src, dst, "To_Exterior_Apertures", category=4, source=aperture, directed=False)

            if toContents:
                for content in TGraph._Phase4Contents(owner):
                    content = TGraph._Phase4TopologyFromAperture(content)
                    dst = TGraph._Phase4AddTopologyVertex(graph, content, category=5, label="Content",
                                                          storeBREP=storeBREP, mantissa=mantissa,
                                                          tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                          extra={"role": "content"})
                    TGraph._Phase4AddRelationship(graph, src, dst, "To_Contents", category=5, source=content, directed=False)

            if toOutposts and isinstance(outpostLookup, dict):
                d = graph.VertexDictionary(src)
                ids = d.get(outpostsKey, [])
                if ids is None:
                    ids = []
                if not isinstance(ids, list):
                    ids = [ids]
                for outpost_id in ids:
                    outpost = outpostLookup.get(outpost_id)
                    if outpost is None:
                        continue
                    dst = TGraph._Phase4AddTopologyVertex(graph, outpost, category=6, label="Outpost",
                                                          storeBREP=storeBREP, mantissa=mantissa,
                                                          tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                          extra={"role": "outpost"})
                    TGraph._Phase4AddRelationship(graph, src, dst, "To_Outposts", category=6, source=outpost, directed=False)

    @staticmethod
    def _Phase4ProcessSingle(graph: "TGraph", topology: Any, childType: str,
                             toExteriorTopologies: bool = False, toExteriorApertures: bool = False,
                             toContents: bool = False, toOutposts: bool = False,
                             outpostLookup: Optional[Dict[Any, Any]] = None,
                             outpostsKey: str = "outposts", storeBREP: bool = False,
                             mantissa: int = 6, tolerance: float = 0.0001,
                             useInternalVertex: bool = False) -> None:
        src = TGraph._Phase4AddTopologyVertex(graph, topology, category=0, label=TGraph._Phase4TopologyType(topology),
                                              storeBREP=storeBREP, mantissa=mantissa, tolerance=tolerance,
                                              useInternalVertex=useInternalVertex, extra={"role": "owner"})
        if toExteriorTopologies or toExteriorApertures:
            for child in TGraph._Phase4Children(topology, childType, tolerance=tolerance):
                if toExteriorTopologies:
                    dst = TGraph._Phase4AddTopologyVertex(graph, child, category=3, label="Exterior Topology",
                                                          storeBREP=storeBREP, mantissa=mantissa,
                                                          tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                          extra={"role": "boundary"})
                    TGraph._Phase4AddRelationship(graph, src, dst, "To_Exterior_Topologies", category=3, source=child, directed=False)
                if toExteriorApertures:
                    for aperture in TGraph._Phase4Apertures(child):
                        ap_topology = TGraph._Phase4TopologyFromAperture(aperture)
                        dst = TGraph._Phase4AddTopologyVertex(graph, ap_topology, category=4, label="Exterior Aperture",
                                                              storeBREP=storeBREP, mantissa=mantissa,
                                                              tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                              extra={"role": "aperture"})
                        TGraph._Phase4AddRelationship(graph, src, dst, "To_Exterior_Apertures", category=4, source=aperture, directed=False)
        if toContents:
            for content in TGraph._Phase4Contents(topology):
                content = TGraph._Phase4TopologyFromAperture(content)
                dst = TGraph._Phase4AddTopologyVertex(graph, content, category=5, label="Content",
                                                      storeBREP=storeBREP, mantissa=mantissa,
                                                      tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                      extra={"role": "content"})
                TGraph._Phase4AddRelationship(graph, src, dst, "To_Contents", category=5, source=content, directed=False)
        if toOutposts and isinstance(outpostLookup, dict):
            d = graph.VertexDictionary(src)
            ids = d.get(outpostsKey, [])
            if ids is None:
                ids = []
            if not isinstance(ids, list):
                ids = [ids]
            for outpost_id in ids:
                outpost = outpostLookup.get(outpost_id)
                if outpost is None:
                    continue
                dst = TGraph._Phase4AddTopologyVertex(graph, outpost, category=6, label="Outpost",
                                                      storeBREP=storeBREP, mantissa=mantissa,
                                                      tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                      extra={"role": "outpost"})
                TGraph._Phase4AddRelationship(graph, src, dst, "To_Outposts", category=6, source=outpost, directed=False)

    @staticmethod
    def _Phase4AllSubtopologies(topology: Any, tolerance: float = 0.0001) -> List[Any]:
        result = []
        for topology_type in ["CellComplex", "Cell", "Shell", "Face", "Wire", "Edge", "Vertex"]:
            result.extend(TGraph._Phase4Topologies(topology, topology_type, free=False, tolerance=tolerance))
        return result

    @staticmethod
    def _Phase4OutpostLookup(topologies: List[Any], idKey: str = "TOPOLOGIC_ID") -> Dict[Any, Any]:
        lookup = {}
        key_l = str(idKey).lower()
        for topology in topologies:
            d = TGraph._TopologyDictionaryToPython(topology)
            for k, v in d.items():
                if str(k).lower() == key_l and v is not None and v not in lookup:
                    lookup[v] = topology
        return lookup

    @staticmethod
    def ByTopology(
        topology: Any,
        direct: bool = True,
        directApertures: bool = False,
        viaSharedTopologies: bool = False,
        viaSharedApertures: bool = False,
        toExteriorTopologies: bool = False,
        toExteriorApertures: bool = False,
        toContents: bool = False,
        toOutposts: bool = False,
        idKey: str = "TOPOLOGIC_ID",
        outpostsKey: str = "outposts",
        useInternalVertex: bool = False,
        storeBREP: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates a topology-first TGraph from a Topologic topology.

        The returned graph stores incidence and dictionaries only. Topologic
        objects are stored as optional representations on vertex and edge records.
        No topologic_core.Graph or geometric graph edges are created internally.
        """

        try:
            from topologicpy.Topology import Topology
            from topologicpy.Cluster import Cluster
        except Exception:
            if not silent:
                print("TGraph.ByTopology - Error: Could not import Topology/Cluster. Returning None.")
            return None

        try:
            if not Topology.IsInstance(topology, "Topology"):
                if not silent:
                    print("TGraph.ByTopology - Error: The input topology is not valid. Returning None.")
                return None
        except Exception:
            return None

        g = TGraph(directed=False, allowSelfLoops=True, allowParallelEdges=True,
                   dictionary=dictionary if isinstance(dictionary, dict) else {"generated_by": "TGraph.ByTopology"})

        all_topologies = TGraph._Phase4AllSubtopologies(topology, tolerance=tolerance)
        outposts = TGraph._Phase4OutpostLookup(all_topologies, idKey=idKey)

        if Topology.IsInstance(topology, "CellComplex"):
            TGraph._Phase4ProcessCollection(g, topology, "Cell", "Face", direct, directApertures,
                                             viaSharedTopologies, viaSharedApertures,
                                             toExteriorTopologies, toExteriorApertures,
                                             toContents, toOutposts, outposts, idKey, outpostsKey,
                                             storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Cell"):
            TGraph._Phase4ProcessSingle(g, topology, "Face", toExteriorTopologies, toExteriorApertures,
                                        toContents, toOutposts, outposts, outpostsKey,
                                        storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Shell"):
            TGraph._Phase4ProcessCollection(g, topology, "Face", "Edge", direct, directApertures,
                                             viaSharedTopologies, viaSharedApertures,
                                             toExteriorTopologies, toExteriorApertures,
                                             toContents, toOutposts, outposts, idKey, outpostsKey,
                                             storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Face"):
            TGraph._Phase4ProcessSingle(g, topology, "Edge", toExteriorTopologies, toExteriorApertures,
                                        toContents, toOutposts, outposts, outpostsKey,
                                        storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Wire"):
            TGraph._Phase4ProcessCollection(g, topology, "Edge", "Vertex", direct, directApertures,
                                             viaSharedTopologies, viaSharedApertures,
                                             toExteriorTopologies, toExteriorApertures,
                                             toContents, toOutposts, outposts, idKey, outpostsKey,
                                             storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Edge"):
            TGraph._Phase4ProcessSingle(g, topology, "Vertex", toExteriorTopologies, toExteriorApertures,
                                        toContents, toOutposts, outposts, outpostsKey,
                                        storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Vertex"):
            TGraph._Phase4AddTopologyVertex(g, topology, category=0, label="Vertex",
                                             storeBREP=storeBREP, mantissa=mantissa,
                                             tolerance=tolerance, useInternalVertex=useInternalVertex,
                                             extra={"role": "owner"})
        elif Topology.IsInstance(topology, "Cluster"):
            for sub in TGraph._Phase4Topologies(topology, "CellComplex", free=False, tolerance=tolerance):
                TGraph._Phase4ProcessCollection(g, sub, "Cell", "Face", direct, directApertures,
                                                 viaSharedTopologies, viaSharedApertures,
                                                 toExteriorTopologies, toExteriorApertures,
                                                 toContents, toOutposts, outposts, idKey, outpostsKey,
                                                 storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Phase4Topologies(topology, "Cell", free=True, tolerance=tolerance):
                TGraph._Phase4ProcessSingle(g, sub, "Face", toExteriorTopologies, toExteriorApertures,
                                            toContents, toOutposts, outposts, outpostsKey,
                                            storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Phase4Topologies(topology, "Shell", free=True, tolerance=tolerance):
                TGraph._Phase4ProcessCollection(g, sub, "Face", "Edge", direct, directApertures,
                                                 viaSharedTopologies, viaSharedApertures,
                                                 toExteriorTopologies, toExteriorApertures,
                                                 toContents, toOutposts, outposts, idKey, outpostsKey,
                                                 storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Phase4Topologies(topology, "Face", free=True, tolerance=tolerance):
                TGraph._Phase4ProcessSingle(g, sub, "Edge", toExteriorTopologies, toExteriorApertures,
                                            toContents, toOutposts, outposts, outpostsKey,
                                            storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Phase4Topologies(topology, "Wire", free=True, tolerance=tolerance):
                TGraph._Phase4ProcessCollection(g, sub, "Edge", "Vertex", direct, directApertures,
                                                 viaSharedTopologies, viaSharedApertures,
                                                 toExteriorTopologies, toExteriorApertures,
                                                 toContents, toOutposts, outposts, idKey, outpostsKey,
                                                 storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Phase4Topologies(topology, "Edge", free=True, tolerance=tolerance):
                TGraph._Phase4ProcessSingle(g, sub, "Vertex", toExteriorTopologies, toExteriorApertures,
                                            toContents, toOutposts, outposts, outpostsKey,
                                            storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Phase4Topologies(topology, "Vertex", free=True, tolerance=tolerance):
                TGraph._Phase4AddTopologyVertex(g, sub, category=0, label="Vertex",
                                                 storeBREP=storeBREP, mantissa=mantissa,
                                                 tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                 extra={"role": "owner"})

        return g

    @staticmethod
    def AccessGraph(
        topology: Any,
        key: str = None,
        includeTypes: list = None,
        excludeTypes: list = None,
        viaSharedApertures: bool = False,
        toExteriorApertures: bool = False,
        useInternalVertex: bool = False,
        includeIsolatedVertices: bool = True,
        storeBREP: bool = False,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates an access graph from topology, optionally filtering aperture edges
        or aperture vertices by dictionary value.
        """

        directApertures = not viaSharedApertures
        graph = TGraph.ByTopology(
            topology,
            direct=False,
            directApertures=directApertures,
            viaSharedTopologies=False,
            viaSharedApertures=viaSharedApertures,
            toExteriorTopologies=False,
            toExteriorApertures=toExteriorApertures,
            useInternalVertex=useInternalVertex,
            storeBREP=storeBREP,
            mantissa=mantissa,
            tolerance=tolerance,
            silent=silent,
        )
        if graph is None or key is None:
            return graph

        include_set = {str(x).strip().lower() for x in includeTypes or [] if x is not None}
        exclude_set = {str(x).strip().lower() for x in excludeTypes or [] if x is not None}

        def passes(record: Dict[str, Any]) -> bool:
            d = record.get("dictionary", {}) if isinstance(record, dict) else {}
            value = d.get(key, None)
            value_l = "" if value is None else str(value).strip().lower()
            if include_set and value_l not in include_set:
                return False
            if exclude_set and value_l in exclude_set:
                return False
            return True

        if viaSharedApertures:
            for vertex in list(TGraph.Vertices(graph)):
                d = vertex.get("dictionary", {})
                if d.get("role") == "aperture" and not passes(vertex):
                    graph.RemoveVertex(vertex, silent=True)
        else:
            for edge in list(TGraph.Edges(graph)):
                d = edge.get("dictionary", {})
                relationship = str(d.get("relationship", "")).lower()
                if "aperture" in relationship and not passes(edge):
                    graph.RemoveEdge(edge, silent=True)

        if not includeIsolatedVertices:
            for vertex in list(TGraph.IsolatedVertices(graph)):
                graph.RemoveVertex(vertex, silent=True)
        return graph

    @staticmethod
    def BySpatialRelationships(
        topologies: List[Any],
        relationship: str = "intersects",
        directed: bool = False,
        allowSelfLoops: bool = False,
        storeBREP: bool = False,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates a TGraph from a list of topologies by testing a spatial relationship.

        Supported relationship values include: "intersects", "touches", "contains",
        "within", "overlaps", and "distance". For "distance", an edge is created
        when the distance is less than or equal to tolerance.
        """

        if not isinstance(topologies, list):
            return None
        g = TGraph(directed=directed, allowSelfLoops=allowSelfLoops, allowParallelEdges=False,
                   dictionary={"generated_by": "TGraph.BySpatialRelationships", "relationship": relationship})
        for t in topologies:
            TGraph._Phase4AddTopologyVertex(g, t, category=0, label=TGraph._Phase4TopologyType(t),
                                             storeBREP=storeBREP, mantissa=mantissa, tolerance=tolerance,
                                             extra={"role": "spatial_object"})
        rel = str(relationship or "intersects").strip().lower()

        def related(a: Any, b: Any) -> bool:
            if a is None or b is None:
                return False
            try:
                from topologicpy.Topology import Topology
                if rel in ["distance", "near", "nearest"]:
                    try:
                        return float(Topology.Distance(a, b)) <= tolerance
                    except Exception:
                        try:
                            ca = TGraph._Phase4Coordinates(a, mantissa=mantissa, tolerance=tolerance)
                            cb = TGraph._Phase4Coordinates(b, mantissa=mantissa, tolerance=tolerance)
                            return ca is not None and cb is not None and math.dist(ca, cb) <= tolerance
                        except Exception:
                            return False
                if rel in ["intersects", "intersect"]:
                    r = Topology.Intersect(a, b)
                    return r is not None
                if rel in ["touches", "touch"]:
                    try:
                        return bool(Topology.Touches(a, b))
                    except Exception:
                        r = Topology.Intersect(a, b)
                        return r is not None
                if rel in ["contains", "contain"]:
                    try:
                        return bool(Topology.Contains(a, b))
                    except Exception:
                        return False
                if rel in ["within", "inside"]:
                    try:
                        return bool(Topology.Contains(b, a))
                    except Exception:
                        return False
                if rel in ["overlaps", "overlap"]:
                    try:
                        return bool(Topology.Overlaps(a, b))
                    except Exception:
                        r = Topology.Intersect(a, b)
                        return r is not None
            except Exception:
                return False
            return False

        n = len(topologies)
        for i in range(n):
            start = i if allowSelfLoops else i + 1
            for j in range(start, n):
                if i == j and not allowSelfLoops:
                    continue
                if related(topologies[i], topologies[j]):
                    TGraph._Phase4AddRelationship(g, i, j, relationship=rel, category="spatial", source=None, directed=directed)
        return g

    @staticmethod
    def VisibilityGraph(
        face: Any,
        vertices: Optional[List[Any]] = None,
        obstacles: Optional[List[Any]] = None,
        bidirectional: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates a lightweight visibility graph from viewpoint vertices on/in a face.

        This method stores visibility as incidence only. Edge geometry is optional
        and generated only as a representation when possible.
        """

        try:
            from topologicpy.Topology import Topology
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
        except Exception:
            if not silent:
                print("TGraph.VisibilityGraph - Error: Could not import TopologicPy classes. Returning None.")
            return None

        if vertices is None:
            try:
                vertices = Topology.Vertices(face, silent=True) or []
            except Exception:
                vertices = []
        if not isinstance(vertices, list):
            return None
        obstacles = obstacles or []
        g = TGraph(directed=not bidirectional, allowSelfLoops=False, allowParallelEdges=False,
                   dictionary={"generated_by": "TGraph.VisibilityGraph"})
        for v in vertices:
            d = TGraph._Phase4TopologyDictionary(v, mantissa=6, tolerance=tolerance)
            d["role"] = "viewpoint"
            g.AddVertex(dictionary=d, representation=v)

        def visible(a: Any, b: Any) -> Tuple[bool, Any]:
            try:
                if Vertex.Distance(a, b) <= tolerance:
                    return False, None
                e = Edge.ByStartVertexEndVertex(a, b, tolerance=tolerance)
                if e is None:
                    return False, None
                try:
                    clipped = Topology.Intersect(e, face)
                    if clipped is None:
                        return False, None
                except Exception:
                    pass
                for obstacle in obstacles:
                    try:
                        inter = Topology.Intersect(e, obstacle)
                        if inter is not None:
                            return False, None
                    except Exception:
                        continue
                return True, e
            except Exception:
                return False, None

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                ok, rep = visible(vertices[i], vertices[j])
                if ok:
                    g.AddEdge(i, j, directed=not bidirectional, dictionary={"relationship": "visibility"}, representation=rep)
                    if not bidirectional and g._directed:
                        g.AddEdge(j, i, directed=True, dictionary={"relationship": "visibility"}, representation=rep)
        return g

    @staticmethod
    def NavigationGraph(
        face: Any,
        vertices: Optional[List[Any]] = None,
        obstacles: Optional[List[Any]] = None,
        bidirectional: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Alias-style constructor for a 2D line-of-sight navigation graph.
        """

        graph = TGraph.VisibilityGraph(face, vertices=vertices, obstacles=obstacles,
                                       bidirectional=bidirectional, tolerance=tolerance, silent=silent)
        if isinstance(graph, TGraph):
            graph._dictionary["generated_by"] = "TGraph.NavigationGraph"
        return graph

    @staticmethod
    def ByIFCPath(*args, **kwargs) -> Optional["TGraph"]:
        """
        Placeholder for IFC-backed TGraph construction.

        IFC import should create vertices and edges as incidence records and store
        IFC/Topologic objects only as optional representations. This placeholder
        intentionally avoids duplicating the legacy geometry-first importer.
        """

        if not kwargs.get("silent", False):
            print("TGraph.ByIFCPath - Warning: IFC import is scheduled for a later Phase 4/5 pass. Returning None.")
        return None

    @staticmethod
    def ByIFCFile(*args, **kwargs) -> Optional["TGraph"]:
        """
        Placeholder for IFC-backed TGraph construction from an IFC file object.
        """

        if not kwargs.get("silent", False):
            print("TGraph.ByIFCFile - Warning: IFC import is scheduled for a later Phase 4/5 pass. Returning None.")
        return None


    # ---------------------------------------------------------------------
    # TGraph v2 compiled integer-indexed kernel
    # ---------------------------------------------------------------------

    @staticmethod
    def InvalidateCache(graph: "TGraph") -> Optional["TGraph"]:
        """
        Invalidates the compiled integer-indexed kernel of the input graph.
        """
        if not isinstance(graph, TGraph):
            return None
        graph._invalidate_cache()
        return graph



    @staticmethod
    def _V2SetVertexValue(graph: "TGraph", stable_index: int, key: Optional[str], value: Any) -> None:
        if key is None or not isinstance(graph, TGraph):
            return
        if graph._validate_vertex_index(stable_index, active=False):
            graph._vertices[stable_index].setdefault("dictionary", {})[key] = value

    @staticmethod
    def _V2BFSCompact(adj: List[List[int]], source: int) -> Tuple[List[int], List[int]]:
        n = len(adj)
        dist = [-1] * n
        parent = [-1] * n
        dist[source] = 0
        q = deque([source])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        return dist, parent

    @staticmethod
    def ActiveVertexIndices(graph: "TGraph") -> List[int]:
        c = TGraph.Compile(graph)
        return list(c["vertices"]) if isinstance(c, dict) else []

    @staticmethod
    def ActiveEdgeIndices(graph: "TGraph") -> List[int]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        return [e["edge_index"] for e in c["edges"]]

    @staticmethod
    def AdjacentIndices(graph: "TGraph", index: int, mode: str = "out") -> List[int]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict) or index not in c["position"]:
            return []
        p = c["position"][index]
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        return [c["vertices"][q] for q in adj[p]]

    @staticmethod
    def AdjacencyList(graph: "TGraph", mode: str = "out") -> List[List[int]]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        vertices = c["vertices"]
        return [[vertices[j] for j in row] for row in adj]

    @staticmethod
    def Degree(graph: "TGraph", index: int, mode: str = "all") -> int:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict) or index not in c["position"]:
            return 0
        p = c["position"][index]
        return len(TGraph.CompiledAdjacency(graph, mode=mode)[p])

    @staticmethod
    def DegreeSequence(graph: "TGraph", mode: str = "all") -> List[int]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        return [len(row) for row in adj]



    @staticmethod
    def DepthFirstSearch(graph: "TGraph", source: int, mode: str = "out") -> List[int]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict) or source not in c["position"]:
            return []
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        start = c["position"][source]
        visited = set()
        order = []
        stack = [start]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            order.append(c["vertices"][u])
            for v in reversed(adj[u]):
                if v not in visited:
                    stack.append(v)
        return order

    @staticmethod
    def IsConnected(graph: "TGraph", mode: str = "all") -> bool:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return False
        if c["n"] <= 1:
            return True
        return len(TGraph.ConnectedComponents(graph, mode=mode)) == 1

    @staticmethod
    def ClosenessCentrality(graph: "TGraph", mode: str = "out", normalize: bool = True,
                            key: str = "closeness_centrality", mantissa: int = 6) -> List[float]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        n = c["n"]
        values = []
        for s in range(n):
            dist, _ = TGraph._V2BFSCompact(adj, s)
            reachable_distances = [d for d in dist if d >= 0]
            reachable = len(reachable_distances)
            if reachable <= 1:
                value = 0.0
            else:
                total = sum(reachable_distances)
                value = float(reachable - 1) / float(total) if total > 0 else 0.0
                if normalize and n > 1:
                    value *= float(reachable - 1) / float(n - 1)
            value = round(value, mantissa)
            values.append(value)
            TGraph._V2SetVertexValue(graph, c["vertices"][s], key, value)
        return values

    @staticmethod
    def BetweennessCentrality(graph: "TGraph", mode: str = "out", normalize: bool = True,
                              endpoints: bool = False, key: str = "betweenness_centrality",
                              mantissa: int = 6) -> List[float]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        n = c["n"]
        cb = [0.0] * n
        for s in range(n):
            stack = []
            pred = [[] for _ in range(n)]
            sigma = [0.0] * n
            dist = [-1] * n
            sigma[s] = 1.0
            dist[s] = 0
            q = deque([s])
            while q:
                v = q.popleft()
                stack.append(v)
                for w in adj[v]:
                    if dist[w] < 0:
                        q.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            if endpoints:
                cb[s] += sum(1 for d in dist if d >= 0) - 1
            delta = [0.0] * n
            while stack:
                w = stack.pop()
                coeff = (1.0 + delta[w]) / sigma[w] if sigma[w] else 0.0
                for v in pred[w]:
                    delta[v] += sigma[v] * coeff
                if w != s:
                    cb[w] += delta[w] + (1.0 if endpoints and dist[w] >= 0 else 0.0)
        if normalize and n > 2:
            scale = 1.0 / float((n - 1) * (n - 2))
            cb = [v * scale for v in cb]
        elif not graph._directed and mode == "all":
            cb = [v * 0.5 for v in cb]
        values = []
        for i, value in enumerate(cb):
            value = round(value, mantissa)
            values.append(value)
            TGraph._V2SetVertexValue(graph, c["vertices"][i], key, value)
        return values

    @staticmethod
    def Diameter(graph: "TGraph", mode: str = "all") -> Optional[int]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return None
        n = c["n"]
        if n == 0:
            return None
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        diameter = 0
        for s in range(n):
            dist, _ = TGraph._V2BFSCompact(adj, s)
            if any(d < 0 for d in dist):
                return None
            diameter = max(diameter, max(dist))
        return diameter

    @staticmethod
    def MinimumSpanningTree(graph: "TGraph", edgeKey: str = "weight") -> Optional["TGraph"]:
        c = TGraph.Compile(graph, weightKey=edgeKey)
        if not isinstance(c, dict):
            return None
        mst = TGraph(directed=False, allowSelfLoops=False, allowParallelEdges=False, dictionary=TGraph.Dictionary(graph))
        for stable in c["vertices"]:
            rec = graph._vertices[stable]
            mst.AddVertex(dictionary=dict(rec.get("dictionary", {})), representation=rec.get("representation"))
        parent = list(range(c["n"]))
        rank = [0] * c["n"]
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True
        for ce in sorted(c["edges"], key=lambda e: (e["weight"], e["edge_index"])):
            if ce["src"] == ce["dst"]:
                continue
            if union(ce["src"], ce["dst"]):
                e = graph._edges[ce["edge_index"]]
                mst.AddEdge(ce["src"], ce["dst"], directed=False,
                            dictionary=dict(e.get("dictionary", {})), representation=e.get("representation"))
        return mst


    # ---------------------------------------------------------------------
    # TGraph v2.1 hot-path optimisation layer
    # ---------------------------------------------------------------------

    PHASE21_HOTPATH_OPTIMISATION = True

    @staticmethod
    def ByDictionaries(
        vertexDictionaries: Optional[List[Dict[str, Any]]] = None,
        edgeDictionaries: Optional[List[Dict[str, Any]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
    ) -> "TGraph":
        """
        Creates a TGraph from vertex and edge dictionaries using the v2.1 bulk path.
        """

        vertexDictionaries = vertexDictionaries or []
        edgeDictionaries = edgeDictionaries or []
        pairs = []
        eds = []
        edge_reps = []
        for d in edgeDictionaries:
            if not isinstance(d, dict):
                continue
            src = d.get("src")
            dst = d.get("dst")
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            ed = dict(d)
            rep = ed.pop("representation", None)
            pairs.append((src, dst))
            eds.append(ed)
            edge_reps.append(rep)
        return TGraph.ByEdgeIndexPairs(
            len(vertexDictionaries),
            pairs,
            vertexDictionaries=vertexDictionaries,
            edgeDictionaries=eds,
            directed=directed,
            allowSelfLoops=allowSelfLoops,
            allowParallelEdges=allowParallelEdges,
            dictionary=dictionary,
            representations={"edges": edge_reps},
        )

    @staticmethod
    def ByAdjacencyList(
        adjacencyList: Optional[List[Iterable[int]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
    ) -> "TGraph":
        """
        Creates a TGraph from an adjacency list using the v2.1 bulk path.
        """

        adjacencyList = adjacencyList or []
        pairs = []
        for src, row in enumerate(adjacencyList):
            if row is None:
                continue
            for dst in row:
                if isinstance(dst, int):
                    pairs.append((src, dst))
        vertex_dicts = [{"index": i} for i in range(len(adjacencyList))]
        return TGraph.ByEdgeIndexPairs(
            len(adjacencyList),
            pairs,
            vertexDictionaries=vertex_dicts,
            directed=directed,
            allowSelfLoops=allowSelfLoops,
            allowParallelEdges=allowParallelEdges,
            dictionary=dictionary,
        )

    @staticmethod
    def ByAdjacencyMatrix(
        adjacencyMatrix: Optional[List[List[Any]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        edgeKey: str = "weight",
    ) -> "TGraph":
        """
        Creates a TGraph from an adjacency matrix using the v2.1 bulk path.
        """

        matrix = adjacencyMatrix or []
        n = len(matrix)
        pairs = []
        eds = []
        for i, row in enumerate(matrix):
            if not isinstance(row, list):
                continue
            for j, value in enumerate(row[:n]):
                if value in [0, None, False]:
                    continue
                pairs.append((i, j))
                d = {}
                if value != 1:
                    d[edgeKey] = value
                eds.append(d)
        vertex_dicts = [{"index": i} for i in range(n)]
        return TGraph.ByEdgeIndexPairs(
            n,
            pairs,
            vertexDictionaries=vertex_dicts,
            edgeDictionaries=eds,
            directed=directed,
            allowSelfLoops=allowSelfLoops,
            allowParallelEdges=allowParallelEdges,
            dictionary=dictionary,
        )

    @staticmethod
    def _V21AdjacencyCompact(graph: "TGraph", mode: str = "out", weightKey: str = "weight") -> Tuple[Optional[Dict[str, Any]], List[List[int]]]:
        c = TGraph.Compile(graph, weightKey=weightKey)
        if not isinstance(c, dict):
            return None, []
        mode = str(mode).lower()
        if mode == "in":
            return c, c["adj_in"]
        if mode == "all":
            return c, c["adj_all"]
        return c, c["adj_out"]

    @staticmethod
    def BreadthFirstSearch(graph: "TGraph", source: int, mode: str = "out") -> List[int]:
        """
        Returns BFS traversal order using compiled adjacency.
        """

        c, adj = TGraph._V21AdjacencyCompact(graph, mode=mode)
        if not isinstance(c, dict) or source not in c["position"]:
            return []
        start = c["position"][source]
        n = c["n"]
        seen = [False] * n
        seen[start] = True
        q = deque([start])
        order = []
        vertices = c["vertices"]
        while q:
            u = q.popleft()
            order.append(vertices[u])
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        return order

    @staticmethod
    def ConnectedComponents(graph: "TGraph", mode: str = "all") -> List[List[int]]:
        """
        Returns connected components using compiled adjacency.
        """

        c, adj = TGraph._V21AdjacencyCompact(graph, mode=mode)
        if not isinstance(c, dict):
            return []
        n = c["n"]
        seen = [False] * n
        vertices = c["vertices"]
        components = []
        for start in range(n):
            if seen[start]:
                continue
            q = deque([start])
            seen[start] = True
            comp = []
            while q:
                u = q.popleft()
                comp.append(vertices[u])
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
            components.append(comp)
        return components

    @staticmethod
    def AdjacencyMatrix(graph: "TGraph", vertexKey: str = None, reverse: bool = False,
                        edgeKeyFwd: str = None, edgeKeyBwd: str = None, bidirKey: str = None,
                        bidirectional: bool = None, useEdgeIndex: bool = False,
                        useEdgeLength: bool = False, mantissa: int = 6, tolerance: float = 0.0001) -> List[List[Any]]:
        """
        Returns an adjacency matrix using active edge records and compact ordering.
        """

        c = TGraph.Compile(graph, weightKey=edgeKeyFwd or "weight")
        if not isinstance(c, dict):
            return []

        order = list(c["vertices"])
        if vertexKey is not None:
            order.sort(key=lambda i: graph._vertices[i].get("dictionary", {}).get(vertexKey), reverse=reverse)
        elif reverse:
            order.reverse()

        pos_out = {stable: i for i, stable in enumerate(order)}
        n = len(order)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        default_bidir = (not graph._directed) if bidirectional is None else bool(bidirectional)

        for e in graph._edges:
            if not e.get("active", True):
                continue
            src_stable = e.get("src")
            dst_stable = e.get("dst")
            if src_stable not in pos_out or dst_stable not in pos_out:
                continue
            d = e.get("dictionary", {}) if isinstance(e.get("dictionary", {}), dict) else {}
            if useEdgeIndex:
                value_fwd = e["index"] + 1
                value_bwd = value_fwd
            elif useEdgeLength:
                value_fwd = value_bwd = TGraph._EdgeLength(graph, e, mantissa=mantissa, tolerance=tolerance)
            else:
                value_fwd = d.get(edgeKeyFwd, 1) if edgeKeyFwd is not None else 1
                value_bwd = d.get(edgeKeyBwd, 1) if edgeKeyBwd is not None else value_fwd
            matrix[pos_out[src_stable]][pos_out[dst_stable]] = value_fwd
            if bidirKey is not None:
                bidir = bool(d.get(bidirKey, default_bidir))
            else:
                bidir = default_bidir or not e.get("directed", False)
            if bidir:
                matrix[pos_out[dst_stable]][pos_out[src_stable]] = value_bwd
        return matrix


    # ---------------------------------------------------------------------
    # TGraph v2.2 optional acceleration layer
    # ---------------------------------------------------------------------

    PHASE22_ACCELERATION = True
    _V22_NUMBA_BFS_PARENT = None

    @staticmethod
    def Compile(graph: "TGraph", weightKey: str = "weight", force: bool = False,
                useNumpy: bool = True, useSciPy: bool = True, useNumba: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns a v2.2 compiled integer-indexed kernel.

        The kernel always contains compact Python adjacency rows. When available,
        it also contains NumPy CSR-style arrays and optional SciPy CSR matrices.
        """
        if not isinstance(graph, TGraph):
            return None

        cached = graph._compiled
        cache_key = (weightKey, bool(useNumpy), bool(useSciPy))
        if not force and isinstance(cached, dict):
            if cached.get("version") == graph._version and cached.get("cache_key") == cache_key:
                return cached

        vertices = [v["index"] for v in graph._vertices if v.get("active", True)]
        vertices.sort()
        n = len(vertices)
        pos = {stable: i for i, stable in enumerate(vertices)}

        adj_out_sets = [set() for _ in range(n)]
        adj_in_sets = [set() for _ in range(n)]
        adj_all_sets = [set() for _ in range(n)]
        compiled_edges: List[Dict[str, Any]] = []
        edge_lookup_compact: Dict[Tuple[int, int, bool], List[int]] = {}

        src_expanded = []
        dst_expanded = []
        weight_expanded = []

        for e in graph._edges:
            if not e.get("active", True):
                continue
            src_stable = e.get("src")
            dst_stable = e.get("dst")
            if src_stable not in pos or dst_stable not in pos:
                continue
            src = pos[src_stable]
            dst = pos[dst_stable]
            directed = bool(e.get("directed", graph._directed))
            d = e.get("dictionary", {}) if isinstance(e.get("dictionary", {}), dict) else {}
            weight = d.get(weightKey, 1.0) if weightKey is not None else 1.0
            try:
                weight = float(weight)
            except Exception:
                weight = 1.0

            edge_id = e.get("index")
            ce_index = len(compiled_edges)
            compiled_edges.append({
                "compiled_index": ce_index,
                "edge_index": edge_id,
                "src": src,
                "dst": dst,
                "src_stable": src_stable,
                "dst_stable": dst_stable,
                "directed": directed,
                "weight": weight,
            })

            adj_out_sets[src].add(dst)
            adj_in_sets[dst].add(src)
            adj_all_sets[src].add(dst)
            adj_all_sets[dst].add(src)
            src_expanded.append(src)
            dst_expanded.append(dst)
            weight_expanded.append(weight)

            if not directed:
                adj_out_sets[dst].add(src)
                adj_in_sets[src].add(dst)
                src_expanded.append(dst)
                dst_expanded.append(src)
                weight_expanded.append(weight)

            key = (src, dst, directed) if directed else ((src, dst, False) if src <= dst else (dst, src, False))
            edge_lookup_compact.setdefault(key, []).append(edge_id)

        adj_out = [sorted(row) for row in adj_out_sets]
        adj_in = [sorted(row) for row in adj_in_sets]
        adj_all = [sorted(row) for row in adj_all_sets]
        degree_out = [len(row) for row in adj_out]
        degree_in = [len(row) for row in adj_in]
        degree_all = [len(row) for row in adj_all]

        compiled = {
            "version": graph._version,
            "cache_key": cache_key,
            "weightKey": weightKey,
            "directed": graph._directed,
            "n": n,
            "vertices": vertices,
            "position": pos,
            "edges": compiled_edges,
            "adj_out": adj_out,
            "adj_in": adj_in,
            "adj_all": adj_all,
            "degree_out": degree_out,
            "degree_in": degree_in,
            "degree_all": degree_all,
            "edge_lookup": edge_lookup_compact,
            "numpy_available": False,
            "scipy_available": False,
            "numba_requested": bool(useNumba),
        }

        if useNumpy:
            try:
                import numpy as _np
                def _csr_arrays(adj_rows):
                    indptr = _np.empty(len(adj_rows) + 1, dtype=_np.int64)
                    total = 0
                    indptr[0] = 0
                    for i, row in enumerate(adj_rows):
                        total += len(row)
                        indptr[i + 1] = total
                    indices = _np.empty(total, dtype=_np.int64)
                    cursor = 0
                    for row in adj_rows:
                        ln = len(row)
                        if ln:
                            indices[cursor:cursor + ln] = row
                            cursor += ln
                    return indptr, indices
                compiled["indptr_out"], compiled["indices_out"] = _csr_arrays(adj_out)
                compiled["indptr_in"], compiled["indices_in"] = _csr_arrays(adj_in)
                compiled["indptr_all"], compiled["indices_all"] = _csr_arrays(adj_all)
                compiled["src_expanded"] = _np.asarray(src_expanded, dtype=_np.int64)
                compiled["dst_expanded"] = _np.asarray(dst_expanded, dtype=_np.int64)
                compiled["weight_expanded"] = _np.asarray(weight_expanded, dtype=float)
                compiled["degree_out_np"] = _np.asarray(degree_out, dtype=_np.int64)
                compiled["degree_in_np"] = _np.asarray(degree_in, dtype=_np.int64)
                compiled["degree_all_np"] = _np.asarray(degree_all, dtype=_np.int64)
                compiled["numpy_available"] = True
            except Exception as exc:
                compiled["numpy_error"] = f"{type(exc).__name__}: {exc}"

        if useNumpy and useSciPy and compiled.get("numpy_available", False):
            try:
                import numpy as _np
                from scipy import sparse as _sparse
                data = _np.ones_like(compiled["src_expanded"], dtype=float)
                weights = compiled["weight_expanded"]
                shape = (n, n)
                compiled["csr_unweighted"] = _sparse.csr_matrix((data, (compiled["src_expanded"], compiled["dst_expanded"])), shape=shape)
                compiled["csr_weighted"] = _sparse.csr_matrix((weights, (compiled["src_expanded"], compiled["dst_expanded"])), shape=shape)
                compiled["scipy_available"] = True
            except Exception as exc:
                compiled["scipy_error"] = f"{type(exc).__name__}: {exc}"

        graph._compiled = compiled
        return compiled

    @staticmethod
    def CompiledAdjacency(graph: "TGraph", mode: str = "out", weightKey: str = "weight") -> List[List[int]]:
        c = TGraph.Compile(graph, weightKey=weightKey)
        if not isinstance(c, dict):
            return []
        mode = str(mode).lower()
        if mode == "in":
            return c["adj_in"]
        if mode == "all":
            return c["adj_all"]
        return c["adj_out"]

    @staticmethod
    def _V22NumbaBFSParent():
        try:
            if TGraph._V22_NUMBA_BFS_PARENT is not None:
                return TGraph._V22_NUMBA_BFS_PARENT
            import numpy as _np
            from numba import njit
            @njit(cache=False)
            def _bfs_parent(indptr, indices, source, target):
                n = indptr.shape[0] - 1
                visited = _np.zeros(n, dtype=_np.uint8)
                parent = _np.full(n, -1, dtype=_np.int64)
                queue = _np.empty(n, dtype=_np.int64)
                head = 0
                tail = 0
                visited[source] = 1
                queue[tail] = source
                tail += 1
                found = source == target
                while head < tail and not found:
                    u = queue[head]
                    head += 1
                    for k in range(indptr[u], indptr[u + 1]):
                        v = indices[k]
                        if visited[v] == 0:
                            visited[v] = 1
                            parent[v] = u
                            if v == target:
                                found = True
                                break
                            queue[tail] = v
                            tail += 1
                return parent, found
            TGraph._V22_NUMBA_BFS_PARENT = _bfs_parent
            return _bfs_parent
        except Exception:
            return None


    @staticmethod
    def PageRank(graph: "TGraph", damping: float = 0.85, iterations: int = 100,
                 tolerance: float = 1e-9, key: str = "pagerank", mantissa: int = 6,
                 useNumpy: bool = True) -> List[float]:
        c = TGraph.Compile(graph, useNumpy=useNumpy, useSciPy=False)
        if not isinstance(c, dict):
            return []
        n = c["n"]
        if n == 0:
            return []
        damping = max(0.0, min(1.0, float(damping)))

        if useNumpy and c.get("numpy_available", False):
            try:
                import numpy as _np
                src = c["src_expanded"]
                dst = c["dst_expanded"]
                out_deg = c["degree_out_np"].astype(float)
                rank = _np.full(n, 1.0 / n, dtype=float)
                base = (1.0 - damping) / n
                active_sources = out_deg[src] > 0
                src_active = src[active_sources]
                dst_active = dst[active_sources]
                for _ in range(max(1, int(iterations))):
                    new_rank = _np.full(n, base, dtype=float)
                    contrib = rank[src_active] / out_deg[src_active]
                    _np.add.at(new_rank, dst_active, damping * contrib)
                    dangling = rank[out_deg == 0].sum()
                    if dangling:
                        new_rank += damping * dangling / n
                    diff = float(_np.abs(new_rank - rank).sum())
                    rank = new_rank
                    if diff <= tolerance:
                        break
                values = [round(float(v), mantissa) for v in rank.tolist()]
                if key is not None:
                    for i, value in enumerate(values):
                        graph._vertices[c["vertices"][i]].setdefault("dictionary", {})[key] = value
                return values
            except Exception:
                pass

        adj = c["adj_out"]
        rank = [1.0 / n] * n
        base = (1.0 - damping) / n
        for _ in range(max(1, int(iterations))):
            new_rank = [base] * n
            dangling = 0.0
            for u, nbrs in enumerate(adj):
                if not nbrs:
                    dangling += rank[u]
                else:
                    share = rank[u] / float(len(nbrs))
                    add = damping * share
                    for v in nbrs:
                        new_rank[v] += add
            if dangling:
                share = damping * dangling / float(n)
                for i in range(n):
                    new_rank[i] += share
            diff = sum(abs(new_rank[i] - rank[i]) for i in range(n))
            rank = new_rank
            if diff <= tolerance:
                break
        values = [round(v, mantissa) for v in rank]
        if key is not None:
            for i, value in enumerate(values):
                graph._vertices[c["vertices"][i]].setdefault("dictionary", {})[key] = value
        return values



    # ---------------------------------------------------------------------
    # TGraph v2.3 hot-path correction layer
    # ---------------------------------------------------------------------

    PHASE23_HOTPATH_CORRECTION = True
    _V23_NUMBA_BFS_ALL_PARENT = None

    @staticmethod
    def AccelerationReport() -> Dict[str, Any]:
        """
        Returns information about optional acceleration libraries available to TGraph.
        """
        report = {
            "phase22": True,
            "phase23": True,
            "phase24": True,
            "numpy": False,
            "scipy": False,
            "numba": False,
        }
        try:
            import numpy as _np  # noqa: F401
            report["numpy"] = True
            report["numpy_version"] = getattr(_np, "__version__", None)
        except Exception:
            pass
        try:
            import scipy as _sp  # noqa: F401
            report["scipy"] = True
            report["scipy_version"] = getattr(_sp, "__version__", None)
        except Exception:
            pass
        try:
            import numba as _nb  # noqa: F401
            report["numba"] = True
            report["numba_version"] = getattr(_nb, "__version__", None)
        except Exception:
            pass
        return report

    @staticmethod
    def ByEdgeIndexPairs(
        order: int,
        edgeIndexPairs: Optional[Iterable[Union[Tuple[int, int], List[int]]]] = None,
        vertexDictionaries: Optional[List[Dict[str, Any]]] = None,
        edgeDictionaries: Optional[List[Dict[str, Any]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        representations: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        lean: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates a TGraph from integer edge pairs using a v2.3 bulk path.

        If lean is True, minimal vertex and edge dictionaries are created. This is
        intended for algorithmic/benchmark workloads where rich per-record
        dictionaries and representations are unnecessary.
        """
        if not isinstance(order, int) or order < 0:
            if not silent:
                print("TGraph.ByEdgeIndexPairs - Error: order must be a non-negative integer. Returning None.")
            return None

        g = TGraph(directed=directed, allowSelfLoops=allowSelfLoops,
                   allowParallelEdges=allowParallelEdges, dictionary=dictionary)

        edge_pairs = list(edgeIndexPairs or [])
        vertexDictionaries = vertexDictionaries or []
        edgeDictionaries = edgeDictionaries or []
        representations = representations if isinstance(representations, dict) else {}
        vertex_reps = representations.get("vertices", []) if isinstance(representations.get("vertices", []), list) else []
        edge_reps = representations.get("edges", []) if isinstance(representations.get("edges", []), list) else []

        g._vertices = []
        g._edges = []
        g._out_edges = {i: set() for i in range(order)}
        g._in_edges = {i: set() for i in range(order)}
        g._incident_edges = {i: set() for i in range(order)}
        g._edge_lookup = {}

        vertices_out = g._vertices
        if lean:
            for i in range(order):
                vertices_out.append({"index": i, "dictionary": {"index": i}, "representation": None, "active": True})
        else:
            for i in range(order):
                d = dict(vertexDictionaries[i]) if i < len(vertexDictionaries) and isinstance(vertexDictionaries[i], dict) else {}
                d["index"] = i
                d.setdefault("active", True)
                rep = vertex_reps[i] if i < len(vertex_reps) else None
                vertices_out.append({"index": i, "dictionary": d, "representation": rep, "active": True})

        seen = set()
        out_edges = g._out_edges
        in_edges = g._in_edges
        incident_edges = g._incident_edges
        edge_lookup = g._edge_lookup
        edges_out = g._edges

        for k, pair in enumerate(edge_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            src, dst = pair[0], pair[1]
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            if src < 0 or dst < 0 or src >= order or dst >= order:
                continue
            if src == dst and not allowSelfLoops:
                continue

            if lean:
                edge_directed = directed
            else:
                ed0 = edgeDictionaries[k] if k < len(edgeDictionaries) and isinstance(edgeDictionaries[k], dict) else {}
                edge_directed = bool(ed0.get("directed", directed))

            if edge_directed:
                key = (src, dst, True)
            else:
                key = (src, dst, False) if src <= dst else (dst, src, False)
            if not allowParallelEdges and key in seen:
                continue
            seen.add(key)

            index = len(edges_out)
            if lean:
                ed = {"index": index, "src": src, "dst": dst, "directed": edge_directed}
                rep = None
            else:
                ed = dict(edgeDictionaries[k]) if k < len(edgeDictionaries) and isinstance(edgeDictionaries[k], dict) else {}
                ed.update({"index": index, "src": src, "dst": dst, "directed": edge_directed, "active": True})
                rep = edge_reps[k] if k < len(edge_reps) else None

            edges_out.append({
                "index": index,
                "src": src,
                "dst": dst,
                "directed": edge_directed,
                "dictionary": ed,
                "representation": rep,
                "active": True,
            })

            out_edges[src].add(index)
            in_edges[dst].add(index)
            incident_edges[src].add(index)
            incident_edges[dst].add(index)
            if not edge_directed:
                out_edges[dst].add(index)
                in_edges[src].add(index)
            edge_lookup.setdefault(key, set()).add(index)

        g._invalidate_cache()
        return g

    @staticmethod
    def DegreeCentrality(graph: "TGraph", mode: str = "all", normalize: bool = True,
                         key: str = "degree_centrality", mantissa: int = 6) -> List[float]:
        """
        Returns degree centrality values using compiled degree arrays.

        If key is None, this is a read-only hot path and no vertex dictionaries
        are touched. This is the recommended mode for benchmarking and numeric
        workflows.
        """
        c = TGraph.Compile(graph, useNumpy=True, useSciPy=False)
        if not isinstance(c, dict):
            return []
        n = c["n"]
        if n == 0:
            return []
        denom = float(max(1, n - 1))
        mode_l = str(mode).lower()
        degrees = c["degree_in"] if mode_l == "in" else (c["degree_out"] if mode_l == "out" else c["degree_all"])

        if key is None:
            if normalize:
                return [float(d) / denom for d in degrees]
            return [float(d) for d in degrees]

        values = [round((float(d) / denom if normalize else float(d)), mantissa) for d in degrees]
        vertices = c["vertices"]
        for i, value in enumerate(values):
            graph._vertices[vertices[i]].setdefault("dictionary", {})[key] = value
        return values

    @staticmethod
    def _V23NumbaBFSAllParent():
        try:
            if TGraph._V23_NUMBA_BFS_ALL_PARENT is not None:
                return TGraph._V23_NUMBA_BFS_ALL_PARENT
            import numpy as _np
            from numba import njit

            @njit(cache=False)
            def _bfs_all_parent(indptr, indices, source):
                n = indptr.shape[0] - 1
                visited = _np.zeros(n, dtype=_np.uint8)
                parent = _np.full(n, -1, dtype=_np.int64)
                dist = _np.full(n, -1, dtype=_np.int64)
                queue = _np.empty(n, dtype=_np.int64)
                head = 0
                tail = 0
                visited[source] = 1
                dist[source] = 0
                queue[tail] = source
                tail += 1
                while head < tail:
                    u = queue[head]
                    head += 1
                    nd = dist[u] + 1
                    for k in range(indptr[u], indptr[u + 1]):
                        v = indices[k]
                        if visited[v] == 0:
                            visited[v] = 1
                            parent[v] = u
                            dist[v] = nd
                            queue[tail] = v
                            tail += 1
                return parent, dist

            TGraph._V23_NUMBA_BFS_ALL_PARENT = _bfs_all_parent
            return _bfs_all_parent
        except Exception:
            return None

    @staticmethod
    def _Phase23AdjacencyKeys(mode: str) -> Tuple[str, str, str]:
        mode_l = str(mode).lower()
        if mode_l == "in":
            return "adj_in", "indptr_in", "indices_in"
        if mode_l == "all":
            return "adj_all", "indptr_all", "indices_all"
        return "adj_out", "indptr_out", "indices_out"

    @staticmethod
    def ShortestPath(graph: "TGraph", source: int, target: int, mode: str = "out", useNumba: bool = False) -> Optional[List[int]]:
        """
        Returns the unweighted shortest path as stable vertex indices.
        """
        use_np = bool(useNumba)
        c = TGraph.Compile(graph, useNumpy=use_np, useSciPy=False, useNumba=useNumba)
        if not isinstance(c, dict):
            return None
        pos = c["position"]
        if source not in pos or target not in pos:
            return None
        s = pos[source]
        t = pos[target]
        if s == t:
            return [source]

        adj_key, indptr_key, indices_key = TGraph._Phase23AdjacencyKeys(mode)

        if useNumba and c.get("numpy_available", False):
            bfs = TGraph._V22NumbaBFSParent()
            if bfs is not None and indptr_key in c and indices_key in c:
                try:
                    parent, found = bfs(c[indptr_key], c[indices_key], s, t)
                    if not found:
                        return None
                    path = [t]
                    cur = t
                    while cur != s:
                        cur = int(parent[cur])
                        if cur < 0:
                            return None
                        path.append(cur)
                    path.reverse()
                    verts = c["vertices"]
                    return [verts[i] for i in path]
                except Exception:
                    pass

        adj = c[adj_key]
        n = c["n"]
        visited = bytearray(n)
        parent = [-1] * n
        queue = [0] * n
        head = 0
        tail = 0
        visited[s] = 1
        queue[tail] = s
        tail += 1
        found = False
        while head < tail:
            u = queue[head]
            head += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = 1
                    parent[v] = u
                    if v == t:
                        found = True
                        head = tail
                        break
                    queue[tail] = v
                    tail += 1
            if found:
                break
        if not found:
            return None
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            if cur < 0:
                return None
            path.append(cur)
        path.reverse()
        verts = c["vertices"]
        return [verts[i] for i in path]




    # ---------------------------------------------------------------------
    # Phase 2.4: adaptive path queries and shortest-path trees
    # ---------------------------------------------------------------------

    PHASE24_ADAPTIVE_PATHS = True
    PHASE24_HOTPATH_CORRECTION = True  # Backwards-compatible marker for v2.4 benchmark harnesses.
    _V24_NUMBA_BFS_TREE = None

    @staticmethod
    def _V24NumbaBFSTree():
        """
        Returns a cached Numba BFS tree kernel.

        The kernel operates on compact CSR-style adjacency arrays and returns
        compact parent and distance arrays. Distances are unweighted hop counts;
        unreachable vertices have distance -1 and parent -1.
        """
        try:
            if TGraph._V24_NUMBA_BFS_TREE is not None:
                return TGraph._V24_NUMBA_BFS_TREE
            import numpy as _np
            from numba import njit

            @njit(cache=False)
            def _bfs_tree(indptr, indices, source):
                n = indptr.shape[0] - 1
                visited = _np.zeros(n, dtype=_np.uint8)
                parent = _np.full(n, -1, dtype=_np.int64)
                distance = _np.full(n, -1, dtype=_np.int64)
                queue = _np.empty(n, dtype=_np.int64)
                head = 0
                tail = 0
                visited[source] = 1
                distance[source] = 0
                queue[tail] = source
                tail += 1

                while head < tail:
                    u = queue[head]
                    head += 1
                    nd = distance[u] + 1
                    for k in range(indptr[u], indptr[u + 1]):
                        v = indices[k]
                        if visited[v] == 0:
                            visited[v] = 1
                            parent[v] = u
                            distance[v] = nd
                            queue[tail] = v
                            tail += 1
                return parent, distance

            TGraph._V24_NUMBA_BFS_TREE = _bfs_tree
            return _bfs_tree
        except Exception:
            return None

    @staticmethod
    def ShortestPathTree(graph: "TGraph", source: int, mode: str = "out", useNumba: bool = False,
                         includePaths: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns a shortest-path tree from a source vertex.

        Parameters
        ----------
        graph : TGraph
            The input graph.
        source : int
            The stable source vertex index.
        mode : str , optional
            Traversal mode. Options are "out", "in", or "all". Default is "out".
        useNumba : bool , optional
            If True and Numba is available, use the compiled Numba BFS tree kernel.
            Default is False.
        includePaths : bool , optional
            If True, include a dictionary of explicit paths from source to every
            reachable vertex. Default is False.

        Returns
        -------
        dict or None
            A dictionary with keys: source, mode, parent, distance, reachable,
            and optionally paths. Parent and distance are keyed by stable vertex
            index. Unreachable vertices have parent None and distance -1.
        """
        if not isinstance(graph, TGraph):
            return None

        mode = str(mode).lower()
        if mode not in ("out", "in", "all"):
            mode = "out"

        c = TGraph.Compile(graph, useNumpy=useNumba, useSciPy=False, useNumba=useNumba)
        if not isinstance(c, dict):
            return None

        pos = c.get("position", {})
        if source not in pos:
            return None

        s = pos[source]
        n = c.get("n", 0)
        verts = c.get("vertices", [])

        parent_compact = None
        distance_compact = None

        if useNumba and c.get("numpy_available", False):
            indptr_key = "indptr_in" if mode == "in" else ("indptr_all" if mode == "all" else "indptr_out")
            indices_key = "indices_in" if mode == "in" else ("indices_all" if mode == "all" else "indices_out")
            bfs_tree = TGraph._V24NumbaBFSTree()
            if bfs_tree is not None and indptr_key in c and indices_key in c:
                try:
                    parent_compact, distance_compact = bfs_tree(c[indptr_key], c[indices_key], s)
                except Exception:
                    parent_compact = None
                    distance_compact = None

        if parent_compact is None or distance_compact is None:
            adj_key = "adj_in" if mode == "in" else ("adj_all" if mode == "all" else "adj_out")
            adj = c.get(adj_key, [])
            parent = [-1] * n
            distance = [-1] * n
            queue = [0] * max(1, n)
            head = 0
            tail = 0
            distance[s] = 0
            queue[tail] = s
            tail += 1

            while head < tail:
                u = queue[head]
                head += 1
                nd = distance[u] + 1
                for v in adj[u]:
                    if distance[v] < 0:
                        distance[v] = nd
                        parent[v] = u
                        queue[tail] = v
                        tail += 1

            parent_compact = parent
            distance_compact = distance

        parent_stable: Dict[int, Optional[int]] = {}
        distance_stable: Dict[int, int] = {}
        reachable: List[int] = []

        for i, stable in enumerate(verts):
            try:
                d = int(distance_compact[i])
            except Exception:
                d = -1
            distance_stable[stable] = d
            if d >= 0:
                reachable.append(stable)
            try:
                p = int(parent_compact[i])
            except Exception:
                p = -1
            parent_stable[stable] = None if p < 0 else verts[p]

        result: Dict[str, Any] = {
            "source": source,
            "mode": mode,
            "parent": parent_stable,
            "distance": distance_stable,
            "reachable": reachable,
        }

        if includePaths:
            paths: Dict[int, Optional[List[int]]] = {}
            for target in verts:
                if distance_stable.get(target, -1) < 0:
                    paths[target] = None
                    continue
                if target == source:
                    paths[target] = [source]
                    continue
                path = [target]
                cur = target
                ok = True
                while cur != source:
                    cur_parent = parent_stable.get(cur, None)
                    if cur_parent is None:
                        ok = False
                        break
                    path.append(cur_parent)
                    cur = cur_parent
                if ok:
                    path.reverse()
                    paths[target] = path
                else:
                    paths[target] = None
            result["paths"] = paths

        return result

    @staticmethod
    def ShortestPathsFromSource(graph: "TGraph", source: int, targets: Optional[Iterable[int]] = None,
                                mode: str = "out", useNumba: bool = False,
                                returnTree: bool = False) -> Dict[int, Optional[List[int]]]:
        """
        Returns shortest paths from one source to one or more targets.

        This v2.4 override uses ShortestPathTree so that one BFS serves all
        requested targets. If returnTree is True, the returned dictionary also
        contains a reserved key "_tree" with the raw tree dictionary.
        """
        tree = TGraph.ShortestPathTree(graph, source, mode=mode, useNumba=useNumba, includePaths=False)
        if not isinstance(tree, dict):
            return {}

        if targets is None:
            target_list = list(tree.get("distance", {}).keys())
        else:
            target_list = list(targets)

        parent = tree.get("parent", {})
        distance = tree.get("distance", {})

        def _path_to(target: int) -> Optional[List[int]]:
            if target not in distance or distance.get(target, -1) < 0:
                return None
            if target == source:
                return [source]
            path = [target]
            cur = target
            while cur != source:
                cur = parent.get(cur, None)
                if cur is None:
                    return None
                path.append(cur)
            path.reverse()
            return path

        result: Dict[int, Optional[List[int]]] = {t: _path_to(t) for t in target_list}
        if returnTree:
            result["_tree"] = tree  # type: ignore[index]
        return result

    @staticmethod
    def _Phase24PairGroupingStats(pairs: Iterable[Tuple[int, int]]) -> Dict[str, Any]:
        pair_list = [p for p in list(pairs or []) if isinstance(p, (list, tuple)) and len(p) >= 2]
        by_source: Dict[int, int] = {}
        for s, _ in pair_list:
            by_source[s] = by_source.get(s, 0) + 1
        pair_count = len(pair_list)
        unique_sources = len(by_source)
        average_targets_per_source = float(pair_count) / float(unique_sources) if unique_sources > 0 else 0.0
        max_targets_per_source = max(by_source.values()) if by_source else 0
        return {
            "pair_count": pair_count,
            "unique_sources": unique_sources,
            "average_targets_per_source": average_targets_per_source,
            "max_targets_per_source": max_targets_per_source,
            "source_counts": dict(by_source),
        }

    @staticmethod
    def ShortestPaths(graph: "TGraph", pairs: Iterable[Tuple[int, int]], mode: str = "out", useNumba: bool = False,
                      grouped: Any = "auto", groupThreshold: float = 1.5) -> List[Optional[List[int]]]:
        """
        Returns shortest paths for source/target pairs.

        Parameters
        ----------
        grouped : bool or str , optional
            If False, use independent shortest-path calls.
            If True, group pairs by source and use one BFS per source.
            If "auto", group only when the average number of targets per source
            is at least groupThreshold. Default is "auto".
        groupThreshold : float , optional
            Average targets/source threshold used when grouped="auto".
            Default is 1.5.
        """
        pair_list = list(pairs or [])
        if not pair_list:
            return []

        stats = TGraph._Phase24PairGroupingStats(pair_list)
        if str(grouped).lower() == "auto":
            use_grouped = stats["average_targets_per_source"] >= float(groupThreshold)
        else:
            use_grouped = bool(grouped)

        if not use_grouped:
            result = []
            for pair in pair_list:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    result.append(None)
                else:
                    result.append(TGraph.ShortestPath(graph, pair[0], pair[1], mode=mode, useNumba=useNumba))
            return result

        by_source: Dict[int, List[Tuple[int, int]]] = {}
        for i, pair in enumerate(pair_list):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            s, t = pair[0], pair[1]
            by_source.setdefault(s, []).append((i, t))

        result: List[Optional[List[int]]] = [None] * len(pair_list)
        for s, items in by_source.items():
            targets = [t for _, t in items]
            paths = TGraph.ShortestPathsFromSource(graph, s, targets=targets, mode=mode, useNumba=useNumba)
            for i, t in items:
                result[i] = paths.get(t, None)
        return result

    @staticmethod
    def EigenvectorCentrality(
        graph: "TGraph",
        mode: str = "out",
        iterations: int = 100,
        tolerance: float = 1e-9,
        key: str = "eigenvector_centrality",
        mantissa: int = 6,
    ) -> List[float]:
        """
        Computes eigenvector centrality using shifted power iteration.

        Conventions
        -----------
        mode="out"
            Propagates centrality to successor vertices. This matches
            nx.eigenvector_centrality(G) for directed graphs in the current
            TGraph update convention.

        mode="in"
            Propagates centrality to predecessor vertices. This matches
            nx.eigenvector_centrality(G.reverse(copy=True)) for directed graphs.

        mode="all"
            Treats the graph as undirected.
        """
        if not isinstance(graph, TGraph):
            return []

        try:
            vertices = TGraph.ActiveVertexIndices(graph)
        except Exception:
            try:
                vertices = TGraph._Phase3ActiveVertexIndices(graph)
            except Exception:
                vertices = []

        n = len(vertices)
        if n == 0:
            return []

        mode = str(mode).lower()
        if mode not in ("out", "in", "all"):
            mode = "out"

        vertex_set = set(vertices)
        adjacency = {}
        for v in vertices:
            try:
                nbrs = TGraph.AdjacentIndices(graph, v, mode=mode)
            except Exception:
                nbrs = []
            adjacency[v] = [u for u in nbrs if u in vertex_set]

        x = {v: 1.0 / math.sqrt(float(n)) for v in vertices}
        max_iterations = max(1, int(iterations))
        tol = float(tolerance)

        for _ in range(max_iterations):
            x_last = x
            x_new = {v: x_last[v] for v in vertices}  # (A + I) shift.
            for v in vertices:
                xv = x_last[v]
                if xv == 0.0:
                    continue
                for u in adjacency[v]:
                    x_new[u] += xv
            norm = math.sqrt(sum(value * value for value in x_new.values()))
            if norm <= 0.0:
                x_new = {v: 1.0 / math.sqrt(float(n)) for v in vertices}
            else:
                inv_norm = 1.0 / norm
                x_new = {v: value * inv_norm for v, value in x_new.items()}
            diff = math.sqrt(sum((x_new[v] - x_last[v]) ** 2 for v in vertices))
            x = x_new
            if diff <= tol:
                break

        values = []
        for v in vertices:
            value = round(float(x[v]), mantissa)
            values.append(value)
            if key is not None:
                try:
                    TGraph._Phase3SetVertexValue(graph, v, key, value)
                except Exception:
                    try:
                        d = graph._vertices[v].get("dictionary", {})
                        if isinstance(d, dict):
                            d[key] = value
                    except Exception:
                        pass
        return values

    @staticmethod
    def WarmUpAcceleration(graph: "TGraph", mode: str = "all", useNumba: bool = True) -> Dict[str, Any]:
        """
        Builds accelerated structures and optionally triggers Numba compilation.
        """
        c = TGraph.Compile(graph, useNumpy=True, useSciPy=True, useNumba=useNumba, force=True)
        report = TGraph.AccelerationReport()
        report["compiled"] = isinstance(c, dict)
        report["phase24"] = True
        if useNumba and isinstance(c, dict) and c.get("n", 0) >= 2:
            try:
                verts = c["vertices"]
                _ = TGraph.ShortestPath(graph, verts[0], verts[min(1, len(verts)-1)], mode=mode, useNumba=True)
                _ = TGraph.ShortestPathTree(graph, verts[0], mode=mode, useNumba=True)
                _ = TGraph.ShortestPathsFromSource(graph, verts[0], targets=[verts[min(1, len(verts)-1)]], mode=mode, useNumba=True)
                report["numba_warmed"] = True
            except Exception as exc:
                report["numba_warmed"] = False
                report["numba_warm_error"] = f"{type(exc).__name__}: {exc}"
        return report
