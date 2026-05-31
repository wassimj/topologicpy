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
              tolerance: float = 0.0001, silent: bool = False,
              selfLoopMode: str = "circle", selfLoopRadius: float = 0.25,
              selfLoopMajorRadius: Optional[float] = None, selfLoopMinorRadius: Optional[float] = None,
              selfLoopSides: int = 32, selfLoopNormal: Optional[List[float]] = None,
              sagittaKey: str = "sagitta") -> List[Any]:
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

                # A self-loop is valid graph incidence but cannot be represented
                # as a straight Topologic Edge. Generate a circular/elliptical
                # Wire representation unless an explicit representation already
                # handled it above.
                if src == dst:
                    topology = TGraph._SelfLoopToWire(
                        vertex=vertices[src],
                        dictionary=d,
                        representation=rep,
                        mode=selfLoopMode,
                        radius=selfLoopRadius,
                        majorRadius=selfLoopMajorRadius,
                        minorRadius=selfLoopMinorRadius,
                        sides=selfLoopSides,
                        normal=selfLoopNormal,
                        tolerance=tolerance,
                        silent=silent,
                    )
                else:
                    # Optional visual arc representation through a sagitta. This
                    # is drawing geometry only; the graph edge remains pure
                    # incidence between src and dst.
                    topology = TGraph._SagittaArcToWire(
                        vertexA=vertices[src],
                        vertexB=vertices[dst],
                        dictionary=d,
                        representation=rep,
                        sagittaKey=sagittaKey,
                        tolerance=tolerance,
                        silent=silent,
                    )

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
    def _VectorNormalised(vector: Optional[List[float]], default: Optional[List[float]] = None) -> List[float]:
        default = default if isinstance(default, list) and len(default) >= 3 else [0.0, 0.0, 1.0]
        if not isinstance(vector, (list, tuple)) or len(vector) < 3:
            vector = default
        try:
            x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
        except Exception:
            x, y, z = float(default[0]), float(default[1]), float(default[2])
        length = math.sqrt(x*x + y*y + z*z)
        if length <= 0.0:
            return [float(default[0]), float(default[1]), float(default[2])]
        return [x/length, y/length, z/length]

    @staticmethod
    def _VectorCross(a: List[float], b: List[float]) -> List[float]:
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        ]

    @staticmethod
    def _VectorDot(a: List[float], b: List[float]) -> float:
        return float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

    @staticmethod
    def _FrameFromNormal(normal: Optional[List[float]] = None) -> Tuple[List[float], List[float], List[float]]:
        n = TGraph._VectorNormalised(normal, default=[0.0, 0.0, 1.0])
        ref = [1.0, 0.0, 0.0]
        if abs(TGraph._VectorDot(n, ref)) > 0.9:
            ref = [0.0, 1.0, 0.0]
        u = TGraph._VectorNormalised(TGraph._VectorCross(n, ref), default=[1.0, 0.0, 0.0])
        v = TGraph._VectorNormalised(TGraph._VectorCross(n, u), default=[0.0, 1.0, 0.0])
        return u, v, n

    @staticmethod
    def _VertexCoordinates(vertex: Any) -> Optional[List[float]]:
        try:
            from topologicpy.Vertex import Vertex
            coords = Vertex.Coordinates(vertex)
            if coords and len(coords) >= 3:
                return [float(coords[0]), float(coords[1]), float(coords[2])]
        except Exception:
            return None
        return None

    @staticmethod
    def _SelfLoopToWire(vertex: Any, dictionary: Optional[Dict[str, Any]] = None,
                        representation: Any = None, mode: str = "circle", radius: float = 0.25,
                        majorRadius: Optional[float] = None, minorRadius: Optional[float] = None,
                        sides: int = 32, normal: Optional[List[float]] = None,
                        tolerance: float = 0.0001, silent: bool = False) -> Optional[Any]:
        """
        Returns a circular or elliptical Wire representation for a self-loop.

        The self-loop remains graph incidence. This method only creates a visual
        Topologic representation when asTopologic=True is requested.
        """
        d = dictionary if isinstance(dictionary, dict) else {}
        rep = representation if isinstance(representation, dict) else {}
        mode = str(rep.get("mode", rep.get("self_loop_mode", rep.get("type", mode)))).lower()
        if mode in ("selfloop", "self_loop", "loop"):
            mode = str(rep.get("shape", d.get("self_loop_mode", mode))).lower()
        if mode not in ("circle", "ellipse"):
            mode = str(d.get("self_loop_mode", mode)).lower()
        if mode not in ("circle", "ellipse"):
            mode = "circle"

        def _number(*values, default=0.25):
            for value in values:
                try:
                    if value is not None:
                        return float(value)
                except Exception:
                    pass
            return float(default)

        radius = _number(rep.get("radius"), d.get("self_loop_radius"), radius, default=0.25)
        major = _number(rep.get("major_radius"), rep.get("majorRadius"), d.get("self_loop_major_radius"), majorRadius, radius, default=radius)
        minor = _number(rep.get("minor_radius"), rep.get("minorRadius"), d.get("self_loop_minor_radius"), minorRadius, radius*0.65, default=radius*0.65)
        if mode == "circle":
            major = minor = radius
        try:
            sides = int(rep.get("sides", d.get("self_loop_sides", sides)))
        except Exception:
            sides = 32
        sides = max(8, sides)
        normal = rep.get("normal", d.get("self_loop_normal", normal))

        centre = TGraph._VertexCoordinates(vertex)
        if centre is None:
            return None
        u, v, _ = TGraph._FrameFromNormal(normal)
        points = []
        for i in range(sides):
            angle = 2.0 * math.pi * float(i) / float(sides)
            ca = math.cos(angle)
            sa = math.sin(angle)
            points.append([
                centre[0] + major * ca * u[0] + minor * sa * v[0],
                centre[1] + major * ca * u[1] + minor * sa * v[1],
                centre[2] + major * ca * u[2] + minor * sa * v[2],
            ])
        points.append(points[0])
        return TGraph._ControlPointsToWire(points, dictionary=d, tolerance=tolerance, silent=silent)

    @staticmethod
    def _SagittaArcToWire(vertexA: Any, vertexB: Any, dictionary: Optional[Dict[str, Any]] = None,
                          representation: Any = None, sagittaKey: str = "sagitta",
                          tolerance: float = 0.0001, silent: bool = False) -> Optional[Any]:
        """
        Returns a polyline Wire approximating an arc controlled by a sagitta.

        The arc is visual representation only. If no sagitta is present, returns
        None so the caller can create a straight Topologic edge.
        """
        d = dictionary if isinstance(dictionary, dict) else {}
        rep = representation if isinstance(representation, dict) else {}
        sagitta = rep.get(sagittaKey, rep.get("sagitta", d.get(sagittaKey, d.get("sagitta", None))))
        try:
            sagitta = float(sagitta)
        except Exception:
            return None
        if abs(sagitta) <= tolerance:
            return None
        a = TGraph._VertexCoordinates(vertexA)
        b = TGraph._VertexCoordinates(vertexB)
        if a is None or b is None:
            return None
        chord = [b[0]-a[0], b[1]-a[1], b[2]-a[2]]
        length = math.sqrt(chord[0]*chord[0] + chord[1]*chord[1] + chord[2]*chord[2])
        if length <= tolerance:
            return None
        tangent = [chord[0]/length, chord[1]/length, chord[2]/length]
        normal = rep.get("normal", d.get("arc_normal", d.get("normal", [0.0, 0.0, 1.0])))
        normal = TGraph._VectorNormalised(normal, default=[0.0, 0.0, 1.0])
        perp = TGraph._VectorCross(normal, tangent)
        if math.sqrt(sum(x*x for x in perp)) <= tolerance:
            _, perp, _ = TGraph._FrameFromNormal(normal)
        perp = TGraph._VectorNormalised(perp, default=[0.0, 1.0, 0.0])
        mid = [(a[i] + b[i]) * 0.5 + sagitta * perp[i] for i in range(3)]
        try:
            sides = int(rep.get("sides", d.get("arc_sides", d.get("sides", 16))))
        except Exception:
            sides = 16
        sides = max(4, sides)
        points = []
        for i in range(sides + 1):
            t = float(i) / float(sides)
            # Quadratic Bezier through a, mid-control, b.
            omt = 1.0 - t
            points.append([
                omt*omt*a[j] + 2.0*omt*t*mid[j] + t*t*b[j]
                for j in range(3)
            ])
        return TGraph._ControlPointsToWire(points, dictionary=d, tolerance=tolerance, silent=silent)

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
    # Graph algorithms
    # ---------------------------------------------------------------------

    @staticmethod
    def _ActiveVertexIndices(graph: "TGraph") -> List[int]:
        if not isinstance(graph, TGraph):
            return []
        return [v["index"] for v in graph._vertices if v.get("active", True)]

    @staticmethod
    def _ActiveEdges(graph: "TGraph") -> List[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return []
        return [e for e in graph._edges if e.get("active", True)]

    @staticmethod
    def _SetVertexValue(graph: "TGraph", index: int, key: Optional[str], value: Any) -> None:
        if not isinstance(graph, TGraph) or key is None:
            return
        if not graph._validate_vertex_index(index):
            return
        graph._vertices[index].setdefault("dictionary", {})[key] = value

    @staticmethod
    def _EdgeWeightValue(edge: Dict[str, Any], edgeKey: Optional[str] = None, default: float = 1.0) -> float:
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
    def _NeighborIndices(graph: "TGraph", index: int, mode: str = "out") -> List[int]:
        return TGraph.AdjacentIndices(graph, index, mode=mode)

    @staticmethod
    def _UndirectedAdjacency(graph: "TGraph") -> Dict[int, Set[int]]:
        adjacency = {i: set() for i in TGraph._ActiveVertexIndices(graph)}
        for e in TGraph._ActiveEdges(graph):
            src = e.get("src")
            dst = e.get("dst")
            if src in adjacency and dst in adjacency:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
        return adjacency

    @staticmethod
    def _WeightedEdges(graph: "TGraph", edgeKey: Optional[str] = None) -> List[Tuple[float, int, int, int]]:
        result = []
        for e in TGraph._ActiveEdges(graph):
            src = e.get("src")
            dst = e.get("dst")
            if not graph._validate_vertex_index(src) or not graph._validate_vertex_index(dst):
                continue
            weight = TGraph._EdgeWeightValue(e, edgeKey=edgeKey, default=1.0)
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
        vertices = TGraph._ActiveVertexIndices(graph)
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
        for e in TGraph._ActiveEdges(graph):
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
        adjacency = TGraph._UndirectedAdjacency(graph)
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
        vertices = TGraph._ActiveVertexIndices(graph)
        adjacency_edges: Dict[int, List[Tuple[int, int]]] = {v: [] for v in vertices}
        pair_counts: Dict[Tuple[int, int], int] = {}

        for e in TGraph._ActiveEdges(graph):
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
        adjacency = TGraph._UndirectedAdjacency(graph)
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
    # Topology / spatial graph builders
    # ---------------------------------------------------------------------

    @staticmethod
    def _ValueAtKey(data: Any, key: str, default: Any = None) -> Any:
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
    def _SetValue(data: Optional[Dict[str, Any]], key: str, value: Any) -> Dict[str, Any]:
        d = dict(data) if isinstance(data, dict) else {}
        d[key] = value
        return d

    @staticmethod
    def _TopologyType(topology: Any) -> str:
        if topology is None:
            return "None"
        try:
            from topologicpy.Topology import Topology
            return str(Topology.TypeAsString(topology))
        except Exception:
            return type(topology).__name__

    @staticmethod
    def _TopologyCoordinates(topology: Any, useInternalVertex: bool = False, mantissa: int = 6, tolerance: float = 0.0001) -> Optional[List[float]]:
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
    def _BREPString(topology: Any) -> Optional[str]:
        try:
            from topologicpy.Topology import Topology
            return Topology.BREPString(topology)
        except Exception:
            return None

    @staticmethod
    def _TopologyDictionary(topology: Any, storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001, useInternalVertex: bool = False) -> Dict[str, Any]:
        d = TGraph._TopologyDictionaryToPython(topology)
        d["topology_type"] = TGraph._TopologyType(topology)
        coords = TGraph._TopologyCoordinates(topology, useInternalVertex=useInternalVertex, mantissa=mantissa, tolerance=tolerance)
        if coords is not None:
            d.setdefault("x", coords[0])
            d.setdefault("y", coords[1])
            d.setdefault("z", coords[2])
        if storeBREP:
            brep = TGraph._BREPString(topology)
            if brep is not None:
                d["brep"] = brep
        return d

    @staticmethod
    def _Topologies(topology: Any, topologyType: str, free: bool = False, tolerance: float = 0.0001) -> List[Any]:
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
    def _Apertures(topology: Any) -> List[Any]:
        try:
            from topologicpy.Topology import Topology
            return Topology.Apertures(topology) or []
        except Exception:
            return []

    @staticmethod
    def _Contents(topology: Any) -> List[Any]:
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
    def _TopologyFromAperture(topology: Any) -> Any:
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Aperture import Aperture
            if Topology.IsInstance(topology, "Aperture"):
                return Aperture.Topology(topology)
        except Exception:
            pass
        return topology

    @staticmethod
    def _BoundaryKey(topology: Any, boundaryType: str, mantissa: int = 6) -> Any:
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
    def _Children(topology: Any, childType: str, tolerance: float = 0.0001) -> List[Any]:
        return TGraph._Topologies(topology, childType, free=False, tolerance=tolerance)

    @staticmethod
    def _OwnerBoundaryMaps(owners: List[Any], childType: str, mantissa: int = 6, tolerance: float = 0.0001) -> Tuple[Dict[Any, List[int]], Dict[Any, List[Any]], Dict[int, List[Tuple[Any, Any]]]]:
        boundary_to_owner_indices: Dict[Any, List[int]] = {}
        boundary_to_refs: Dict[Any, List[Any]] = {}
        owner_to_boundaries: Dict[int, List[Tuple[Any, Any]]] = {}
        seen_owner_boundary = set()

        for owner_index, owner in enumerate(owners):
            for child in TGraph._Children(owner, childType, tolerance=tolerance):
                bk = TGraph._BoundaryKey(child, childType, mantissa=mantissa)
                if (owner_index, bk) not in seen_owner_boundary:
                    boundary_to_owner_indices.setdefault(bk, []).append(owner_index)
                    owner_to_boundaries.setdefault(owner_index, []).append((bk, child))
                    seen_owner_boundary.add((owner_index, bk))
                boundary_to_refs.setdefault(bk, []).append(child)
        return boundary_to_owner_indices, boundary_to_refs, owner_to_boundaries

    @staticmethod
    def _AddTopologyVertex(graph: "TGraph", topology: Any, category: Any = None, label: Any = None,
                                 storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001,
                                 useInternalVertex: bool = False, extra: Optional[Dict[str, Any]] = None) -> int:
        d = TGraph._TopologyDictionary(topology, storeBREP=storeBREP, mantissa=mantissa, tolerance=tolerance, useInternalVertex=useInternalVertex)
        if category is not None:
            d["category"] = category
        if label is not None:
            d.setdefault("label", label)
        if isinstance(extra, dict):
            d.update(extra)
        return graph.AddVertex(dictionary=d, representation=topology)

    @staticmethod
    def _AddRelationship(graph: "TGraph", src: int, dst: int, relationship: str, category: Any = None,
                               source: Any = None, dictionary: Optional[Dict[str, Any]] = None,
                               directed: Optional[bool] = None) -> Optional[int]:
        if src is None or dst is None:
            return None
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d["relationship"] = relationship
        if category is not None:
            d["category"] = category
        if source is not None:
            sd = TGraph._TopologyDictionaryToPython(TGraph._TopologyFromAperture(source))
            for k, v in sd.items():
                d.setdefault(k, v)
            d.setdefault("source_topology_type", TGraph._TopologyType(TGraph._TopologyFromAperture(source)))
        return graph.AddEdge(src, dst, directed=directed, dictionary=d, representation=source)

    @staticmethod
    def _ProcessTopologyCollection(graph: "TGraph", host: Any, ownerType: str, childType: str,
                                 direct: bool = True, directApertures: bool = False,
                                 viaSharedTopologies: bool = False, viaSharedApertures: bool = False,
                                 toExteriorTopologies: bool = False, toExteriorApertures: bool = False,
                                 toContents: bool = False, toOutposts: bool = False,
                                 outpostLookup: Optional[Dict[Any, Any]] = None,
                                 idKey: str = "TOPOLOGIC_ID", outpostsKey: str = "outposts",
                                 storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001,
                                 useInternalVertex: bool = False) -> None:
        owners = TGraph._Topologies(host, ownerType, free=False, tolerance=tolerance)
        owner_vertex = {}
        for i, owner in enumerate(owners):
            owner_vertex[i] = TGraph._AddTopologyVertex(graph, owner, category=0, label=ownerType,
                                                              storeBREP=storeBREP, mantissa=mantissa,
                                                              tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                              extra={"role": "owner"})

        boundary_to_owners, boundary_to_refs, owner_to_boundaries = TGraph._OwnerBoundaryMaps(owners, childType, mantissa=mantissa, tolerance=tolerance)

        if direct or directApertures:
            seen = set()
            for bk, incident in boundary_to_owners.items():
                if len(incident) < 2:
                    continue
                refs = boundary_to_refs.get(bk, []) or []
                has_aperture = any(TGraph._Apertures(ref) for ref in refs)
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
                            TGraph._AddRelationship(graph, src, dst, "Direct", category=0,
                                                           source=refs[0] if refs else None, directed=False)
                        if directApertures and has_aperture:
                            aps = []
                            for ref in refs:
                                aps.extend(TGraph._Apertures(ref))
                            source = aps[0] if aps else (refs[0] if refs else None)
                            ap_pair = pair + ("aperture",)
                            if ap_pair not in seen:
                                seen.add(ap_pair)
                                TGraph._AddRelationship(graph, src, dst, "Direct_Apertures", category=2,
                                                               source=source, directed=False)

        boundary_vertex = {}
        aperture_vertex = {}

        def boundary_index(boundary: Any, category: Any, relationship_label: str) -> int:
            bk = TGraph._BoundaryKey(boundary, childType, mantissa=mantissa)
            key = (category, bk)
            if key not in boundary_vertex:
                boundary_vertex[key] = TGraph._AddTopologyVertex(graph, boundary, category=category,
                                                                        label=relationship_label,
                                                                        storeBREP=storeBREP, mantissa=mantissa,
                                                                        tolerance=tolerance,
                                                                        useInternalVertex=useInternalVertex,
                                                                        extra={"role": "boundary"})
            return boundary_vertex[key]

        def aperture_index(aperture: Any, category: Any, relationship_label: str) -> int:
            ap_topology = TGraph._TopologyFromAperture(aperture)
            key = (category, id(aperture))
            if key not in aperture_vertex:
                aperture_vertex[key] = TGraph._AddTopologyVertex(graph, ap_topology, category=category,
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
                apertures = TGraph._Apertures(child)
                if is_shared and viaSharedTopologies:
                    dst = boundary_index(child, 1, "Shared Topology")
                    TGraph._AddRelationship(graph, src, dst, "Via_Shared_Topologies", category=1, source=child, directed=False)
                if is_shared and viaSharedApertures:
                    for aperture in apertures:
                        dst = aperture_index(aperture, 2, "Shared Aperture")
                        TGraph._AddRelationship(graph, src, dst, "Via_Shared_Apertures", category=2, source=aperture, directed=False)
                if (not is_shared) and toExteriorTopologies:
                    dst = boundary_index(child, 3, "Exterior Topology")
                    TGraph._AddRelationship(graph, src, dst, "To_Exterior_Topologies", category=3, source=child, directed=False)
                if (not is_shared) and toExteriorApertures:
                    for aperture in apertures:
                        dst = aperture_index(aperture, 4, "Exterior Aperture")
                        TGraph._AddRelationship(graph, src, dst, "To_Exterior_Apertures", category=4, source=aperture, directed=False)

            if toContents:
                for content in TGraph._Contents(owner):
                    content = TGraph._TopologyFromAperture(content)
                    dst = TGraph._AddTopologyVertex(graph, content, category=5, label="Content",
                                                          storeBREP=storeBREP, mantissa=mantissa,
                                                          tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                          extra={"role": "content"})
                    TGraph._AddRelationship(graph, src, dst, "To_Contents", category=5, source=content, directed=False)

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
                    dst = TGraph._AddTopologyVertex(graph, outpost, category=6, label="Outpost",
                                                          storeBREP=storeBREP, mantissa=mantissa,
                                                          tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                          extra={"role": "outpost"})
                    TGraph._AddRelationship(graph, src, dst, "To_Outposts", category=6, source=outpost, directed=False)

    @staticmethod
    def _ProcessSingleTopology(graph: "TGraph", topology: Any, childType: str,
                             toExteriorTopologies: bool = False, toExteriorApertures: bool = False,
                             toContents: bool = False, toOutposts: bool = False,
                             outpostLookup: Optional[Dict[Any, Any]] = None,
                             outpostsKey: str = "outposts", storeBREP: bool = False,
                             mantissa: int = 6, tolerance: float = 0.0001,
                             useInternalVertex: bool = False) -> None:
        src = TGraph._AddTopologyVertex(graph, topology, category=0, label=TGraph._TopologyType(topology),
                                              storeBREP=storeBREP, mantissa=mantissa, tolerance=tolerance,
                                              useInternalVertex=useInternalVertex, extra={"role": "owner"})
        if toExteriorTopologies or toExteriorApertures:
            for child in TGraph._Children(topology, childType, tolerance=tolerance):
                if toExteriorTopologies:
                    dst = TGraph._AddTopologyVertex(graph, child, category=3, label="Exterior Topology",
                                                          storeBREP=storeBREP, mantissa=mantissa,
                                                          tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                          extra={"role": "boundary"})
                    TGraph._AddRelationship(graph, src, dst, "To_Exterior_Topologies", category=3, source=child, directed=False)
                if toExteriorApertures:
                    for aperture in TGraph._Apertures(child):
                        ap_topology = TGraph._TopologyFromAperture(aperture)
                        dst = TGraph._AddTopologyVertex(graph, ap_topology, category=4, label="Exterior Aperture",
                                                              storeBREP=storeBREP, mantissa=mantissa,
                                                              tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                              extra={"role": "aperture"})
                        TGraph._AddRelationship(graph, src, dst, "To_Exterior_Apertures", category=4, source=aperture, directed=False)
        if toContents:
            for content in TGraph._Contents(topology):
                content = TGraph._TopologyFromAperture(content)
                dst = TGraph._AddTopologyVertex(graph, content, category=5, label="Content",
                                                      storeBREP=storeBREP, mantissa=mantissa,
                                                      tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                      extra={"role": "content"})
                TGraph._AddRelationship(graph, src, dst, "To_Contents", category=5, source=content, directed=False)
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
                dst = TGraph._AddTopologyVertex(graph, outpost, category=6, label="Outpost",
                                                      storeBREP=storeBREP, mantissa=mantissa,
                                                      tolerance=tolerance, useInternalVertex=useInternalVertex,
                                                      extra={"role": "outpost"})
                TGraph._AddRelationship(graph, src, dst, "To_Outposts", category=6, source=outpost, directed=False)

    @staticmethod
    def _AllSubtopologies(topology: Any, tolerance: float = 0.0001) -> List[Any]:
        result = []
        for topology_type in ["CellComplex", "Cell", "Shell", "Face", "Wire", "Edge", "Vertex"]:
            result.extend(TGraph._Topologies(topology, topology_type, free=False, tolerance=tolerance))
        return result

    @staticmethod
    def _OutpostLookup(topologies: List[Any], idKey: str = "TOPOLOGIC_ID") -> Dict[Any, Any]:
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

        all_topologies = TGraph._AllSubtopologies(topology, tolerance=tolerance)
        outposts = TGraph._OutpostLookup(all_topologies, idKey=idKey)

        if Topology.IsInstance(topology, "CellComplex"):
            TGraph._ProcessTopologyCollection(g, topology, "Cell", "Face", direct, directApertures,
                                             viaSharedTopologies, viaSharedApertures,
                                             toExteriorTopologies, toExteriorApertures,
                                             toContents, toOutposts, outposts, idKey, outpostsKey,
                                             storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Cell"):
            TGraph._ProcessSingleTopology(g, topology, "Face", toExteriorTopologies, toExteriorApertures,
                                        toContents, toOutposts, outposts, outpostsKey,
                                        storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Shell"):
            TGraph._ProcessTopologyCollection(g, topology, "Face", "Edge", direct, directApertures,
                                             viaSharedTopologies, viaSharedApertures,
                                             toExteriorTopologies, toExteriorApertures,
                                             toContents, toOutposts, outposts, idKey, outpostsKey,
                                             storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Face"):
            TGraph._ProcessSingleTopology(g, topology, "Edge", toExteriorTopologies, toExteriorApertures,
                                        toContents, toOutposts, outposts, outpostsKey,
                                        storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Wire"):
            TGraph._ProcessTopologyCollection(g, topology, "Edge", "Vertex", direct, directApertures,
                                             viaSharedTopologies, viaSharedApertures,
                                             toExteriorTopologies, toExteriorApertures,
                                             toContents, toOutposts, outposts, idKey, outpostsKey,
                                             storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Edge"):
            TGraph._ProcessSingleTopology(g, topology, "Vertex", toExteriorTopologies, toExteriorApertures,
                                        toContents, toOutposts, outposts, outpostsKey,
                                        storeBREP, mantissa, tolerance, useInternalVertex)
        elif Topology.IsInstance(topology, "Vertex"):
            TGraph._AddTopologyVertex(g, topology, category=0, label="Vertex",
                                             storeBREP=storeBREP, mantissa=mantissa,
                                             tolerance=tolerance, useInternalVertex=useInternalVertex,
                                             extra={"role": "owner"})
        elif Topology.IsInstance(topology, "Cluster"):
            for sub in TGraph._Topologies(topology, "CellComplex", free=False, tolerance=tolerance):
                TGraph._ProcessTopologyCollection(g, sub, "Cell", "Face", direct, directApertures,
                                                 viaSharedTopologies, viaSharedApertures,
                                                 toExteriorTopologies, toExteriorApertures,
                                                 toContents, toOutposts, outposts, idKey, outpostsKey,
                                                 storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Topologies(topology, "Cell", free=True, tolerance=tolerance):
                TGraph._ProcessSingleTopology(g, sub, "Face", toExteriorTopologies, toExteriorApertures,
                                            toContents, toOutposts, outposts, outpostsKey,
                                            storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Topologies(topology, "Shell", free=True, tolerance=tolerance):
                TGraph._ProcessTopologyCollection(g, sub, "Face", "Edge", direct, directApertures,
                                                 viaSharedTopologies, viaSharedApertures,
                                                 toExteriorTopologies, toExteriorApertures,
                                                 toContents, toOutposts, outposts, idKey, outpostsKey,
                                                 storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Topologies(topology, "Face", free=True, tolerance=tolerance):
                TGraph._ProcessSingleTopology(g, sub, "Edge", toExteriorTopologies, toExteriorApertures,
                                            toContents, toOutposts, outposts, outpostsKey,
                                            storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Topologies(topology, "Wire", free=True, tolerance=tolerance):
                TGraph._ProcessTopologyCollection(g, sub, "Edge", "Vertex", direct, directApertures,
                                                 viaSharedTopologies, viaSharedApertures,
                                                 toExteriorTopologies, toExteriorApertures,
                                                 toContents, toOutposts, outposts, idKey, outpostsKey,
                                                 storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Topologies(topology, "Edge", free=True, tolerance=tolerance):
                TGraph._ProcessSingleTopology(g, sub, "Vertex", toExteriorTopologies, toExteriorApertures,
                                            toContents, toOutposts, outposts, outpostsKey,
                                            storeBREP, mantissa, tolerance, useInternalVertex)
            for sub in TGraph._Topologies(topology, "Vertex", free=True, tolerance=tolerance):
                TGraph._AddTopologyVertex(g, sub, category=0, label="Vertex",
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
            TGraph._AddTopologyVertex(g, t, category=0, label=TGraph._TopologyType(t),
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
                            ca = TGraph._TopologyCoordinates(a, mantissa=mantissa, tolerance=tolerance)
                            cb = TGraph._TopologyCoordinates(b, mantissa=mantissa, tolerance=tolerance)
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
                    TGraph._AddRelationship(g, i, j, relationship=rel, category="spatial", source=None, directed=directed)
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
            d = TGraph._TopologyDictionary(v, mantissa=6, tolerance=tolerance)
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
            print("TGraph.ByIFCPath - Warning: IFC import is not available in this TGraph implementation. Returning None.")
        return None

    @staticmethod
    def ByIFCFile(*args, **kwargs) -> Optional["TGraph"]:
        """
        Placeholder for IFC-backed TGraph construction from an IFC file object.
        """

        if not kwargs.get("silent", False):
            print("TGraph.ByIFCFile - Warning: IFC import is not available in this TGraph implementation. Returning None.")
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
    def _DijkstraStateetVertexValue(graph: "TGraph", stable_index: int, key: Optional[str], value: Any) -> None:
        if key is None or not isinstance(graph, TGraph):
            return
        if graph._validate_vertex_index(stable_index, active=False):
            graph._vertices[stable_index].setdefault("dictionary", {})[key] = value

    @staticmethod
    def _BFSCompiledStateompact(adj: List[List[int]], source: int) -> Tuple[List[int], List[int]]:
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
        mode_l = str(mode).lower()
        if mode_l == "in":
            return int(c["degree_in"][p])
        if mode_l == "out":
            return int(c["degree_out"][p])
        return int(c["degree_all"][p])

    @staticmethod
    def DegreeSequence(graph: "TGraph", mode: str = "all") -> List[int]:
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        mode_l = str(mode).lower()
        if mode_l == "in":
            return [int(d) for d in c["degree_in"]]
        if mode_l == "out":
            return [int(d) for d in c["degree_out"]]
        return [int(d) for d in c["degree_all"]]



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
            dist, _ = TGraph._BFSCompiledStateompact(adj, s)
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
            TGraph._DijkstraStateetVertexValue(graph, c["vertices"][s], key, value)
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
            TGraph._DijkstraStateetVertexValue(graph, c["vertices"][i], key, value)
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
            dist, _ = TGraph._BFSCompiledStateompact(adj, s)
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
    def _AdjacencyMatrixFastArraydjacencyCompact(graph: "TGraph", mode: str = "out", weightKey: str = "weight") -> Tuple[Optional[Dict[str, Any]], List[List[int]]]:
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

        c, adj = TGraph._AdjacencyMatrixFastArraydjacencyCompact(graph, mode=mode)
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

        c, adj = TGraph._AdjacencyMatrixFastArraydjacencyCompact(graph, mode=mode)
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
    # Optional acceleration layer
    # ---------------------------------------------------------------------

    _NUMBA_BFS_PARENT = None

    @staticmethod
    def Compile(graph: "TGraph", weightKey: str = "weight", force: bool = False,
                useNumpy: bool = True, useSciPy: bool = True, useNumba: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns a compiled integer-indexed kernel.

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
        degree_out_counts = [0 for _ in range(n)]
        degree_in_counts = [0 for _ in range(n)]
        degree_all_counts = [0 for _ in range(n)]
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

            if directed:
                degree_out_counts[src] += 1
                degree_in_counts[dst] += 1
                if src == dst:
                    degree_all_counts[src] += 2
                else:
                    degree_all_counts[src] += 1
                    degree_all_counts[dst] += 1
            else:
                if src == dst:
                    # NetworkX-compatible undirected self-loop degree.
                    degree_out_counts[src] += 2
                    degree_in_counts[src] += 2
                    degree_all_counts[src] += 2
                else:
                    degree_out_counts[src] += 1
                    degree_out_counts[dst] += 1
                    degree_in_counts[src] += 1
                    degree_in_counts[dst] += 1
                    degree_all_counts[src] += 1
                    degree_all_counts[dst] += 1

            if not directed:
                adj_out_sets[dst].add(src)
                adj_in_sets[src].add(dst)
                if src != dst:
                    src_expanded.append(dst)
                    dst_expanded.append(src)
                    weight_expanded.append(weight)

            key = (src, dst, directed) if directed else ((src, dst, False) if src <= dst else (dst, src, False))
            edge_lookup_compact.setdefault(key, []).append(edge_id)

        adj_out = [sorted(row) for row in adj_out_sets]
        adj_in = [sorted(row) for row in adj_in_sets]
        adj_all = [sorted(row) for row in adj_all_sets]
        degree_out = degree_out_counts
        degree_in = degree_in_counts
        degree_all = degree_all_counts
        pagerank_out_degree = [len(row) for row in adj_out]

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
                compiled["pagerank_out_degree_np"] = _np.asarray(pagerank_out_degree, dtype=_np.int64)
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
    def _NumbaShortestPathKernelumbaBFSParent():
        try:
            if TGraph._NUMBA_BFS_PARENT is not None:
                return TGraph._NUMBA_BFS_PARENT
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
            TGraph._NUMBA_BFS_PARENT = _bfs_parent
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
                out_deg = c.get("pagerank_out_degree_np", c["degree_out_np"]).astype(float)
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
    # Compiled-kernel construction and hot-path helpers
    # ---------------------------------------------------------------------

    _NUMBA_BFS_ALL_PARENT = None

    @staticmethod
    def AccelerationReport() -> Dict[str, Any]:
        """
        Returns information about optional acceleration libraries available to TGraph.
        """
        report = {
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
    def _NumbaBFSMultiTargetKernelumbaBFSAllParent():
        try:
            if TGraph._NUMBA_BFS_ALL_PARENT is not None:
                return TGraph._NUMBA_BFS_ALL_PARENT
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

            TGraph._NUMBA_BFS_ALL_PARENT = _bfs_all_parent
            return _bfs_all_parent
        except Exception:
            return None

    @staticmethod
    def _CompiledAdjacencyKeys(mode: str) -> Tuple[str, str, str]:
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

        adj_key, indptr_key, indices_key = TGraph._CompiledAdjacencyKeys(mode)

        if useNumba and c.get("numpy_available", False):
            bfs = TGraph._NumbaShortestPathKernelumbaBFSParent()
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
    # Adaptive path queries and shortest-path trees
    # ---------------------------------------------------------------------

    _NUMBA_BFS_TREE = None

    @staticmethod
    def _NumbaBFSTreeKernelumbaBFSTree():
        """
        Returns a cached Numba BFS tree kernel.

        The kernel operates on compact CSR-style adjacency arrays and returns
        compact parent and distance arrays. Distances are unweighted hop counts;
        unreachable vertices have distance -1 and parent -1.
        """
        try:
            if TGraph._NUMBA_BFS_TREE is not None:
                return TGraph._NUMBA_BFS_TREE
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

            TGraph._NUMBA_BFS_TREE = _bfs_tree
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
            bfs_tree = TGraph._NumbaBFSTreeKernelumbaBFSTree()
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
    def _PairGroupingStats(pairs: Iterable[Tuple[int, int]]) -> Dict[str, Any]:
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

        stats = TGraph._PairGroupingStats(pair_list)
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
                vertices = TGraph._ActiveVertexIndices(graph)
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
                    TGraph._SetVertexValue(graph, v, key, value)
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

    # ---------------------------------------------------------------------
    # v3.0 parity: LineGraph and CSV import/export
    # ---------------------------------------------------------------------

    @staticmethod
    def LineGraph(
        graph: "TGraph",
        directed: Optional[bool] = None,
        transferDictionaries: bool = True,
        vertexLabelKey: str = "label",
        originalEdgeKey: str = "original_edge_index",
        sharedVertexKey: str = "shared_vertex_index",
        relationshipKey: str = "relationship",
    ) -> Optional["TGraph"]:
        """
        Returns the line graph of the input graph.

        In the returned graph, each active edge of the input graph becomes a
        vertex. Two line-graph vertices are connected when their corresponding
        input edges are adjacent.

        Conventions
        -----------
        Undirected input graph
            Two line-graph vertices are adjacent if the corresponding input
            edges share at least one endpoint. The line graph is undirected.

        Directed input graph
            There is a directed edge from line-graph vertex e1 to line-graph
            vertex e2 when dst(e1) == src(e2). This matches the standard
            directed line-graph convention used by NetworkX.

        Self-loops
            A self-loop is treated as an edge incident to its source/destination
            vertex. In a directed graph, a self-loop can connect to following
            edges according to dst(e1) == src(e2), including itself only when
            self-loops are allowed in the returned line graph.

        Parameters
        ----------
        graph : TGraph
            The input graph.
        directed : bool , optional
            If specified, sets whether the returned line graph is directed.
            If None, the input graph's directedness is used.
        transferDictionaries : bool , optional
            If True, transfers input edge dictionaries to line-graph vertices.
            Default is True.
        vertexLabelKey : str , optional
            Key for a readable line-graph vertex label. Default is "label".
        originalEdgeKey : str , optional
            Key storing the original edge index. Default is "original_edge_index".
        sharedVertexKey : str , optional
            Key storing the shared original vertex index on line-graph edges.
            Default is "shared_vertex_index".
        relationshipKey : str , optional
            Key storing the relationship label on line-graph edges. Default is
            "relationship".

        Returns
        -------
        TGraph
            The line graph.
        """

        if not isinstance(graph, TGraph):
            return None

        output_directed = graph._directed if directed is None else bool(directed)
        active_edges = [e for e in graph._edges if e.get("active", True)]

        lg = TGraph(
            directed=output_directed,
            allowSelfLoops=graph._allow_self_loops,
            allowParallelEdges=graph._allow_parallel_edges,
            dictionary={
                "source": "TGraph.LineGraph",
                "source_order": TGraph.Order(graph),
                "source_size": TGraph.Size(graph),
                "source_directed": graph._directed,
            },
        )

        edge_to_vertex = {}
        for e in active_edges:
            eidx = e.get("index")
            d = dict(e.get("dictionary", {})) if transferDictionaries else {}
            d[originalEdgeKey] = eidx
            d["src"] = e.get("src")
            d["dst"] = e.get("dst")
            d["directed"] = e.get("directed", graph._directed)
            d.setdefault(vertexLabelKey, f"e{eidx}")
            edge_to_vertex[eidx] = lg.AddVertex(dictionary=d, representation=e.get("representation"))

        if graph._directed:
            # Directed line graph: (u, v) -> (v, w).
            outgoing_by_src = {}
            for e in active_edges:
                outgoing_by_src.setdefault(e.get("src"), []).append(e)

            for e1 in active_edges:
                e1idx = e1.get("index")
                v = e1.get("dst")
                for e2 in outgoing_by_src.get(v, []):
                    e2idx = e2.get("index")
                    if e1idx == e2idx and not lg._allow_self_loops:
                        continue
                    dictionary = {
                        relationshipKey: "directed_edge_adjacency",
                        sharedVertexKey: v,
                        "from_original_edge_index": e1idx,
                        "to_original_edge_index": e2idx,
                    }
                    lg.AddEdge(edge_to_vertex[e1idx], edge_to_vertex[e2idx], directed=output_directed, dictionary=dictionary)
        else:
            # Undirected line graph: connect any two edges sharing an endpoint.
            incident_by_vertex = {}
            for e in active_edges:
                incident_by_vertex.setdefault(e.get("src"), []).append(e)
                if e.get("dst") != e.get("src"):
                    incident_by_vertex.setdefault(e.get("dst"), []).append(e)

            seen_pairs = set()
            for shared_vertex, incident_edges in incident_by_vertex.items():
                m = len(incident_edges)
                for i in range(m):
                    for j in range(i + 1, m):
                        e1 = incident_edges[i]
                        e2 = incident_edges[j]
                        e1idx = e1.get("index")
                        e2idx = e2.get("index")
                        a, b = sorted((e1idx, e2idx))
                        pair = (a, b)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        dictionary = {
                            relationshipKey: "edge_adjacency",
                            sharedVertexKey: shared_vertex,
                            "edge_a_index": a,
                            "edge_b_index": b,
                        }
                        lg.AddEdge(edge_to_vertex[a], edge_to_vertex[b], directed=output_directed, dictionary=dictionary)

        return lg

    @staticmethod
    def VerticesCSVString(graph: "TGraph", includeInactive: bool = False) -> str:
        """
        Returns a CSV string for the graph's vertex records.
        """

        if not isinstance(graph, TGraph):
            return ""

        keys = set(["index", "active"])
        rows = []
        for v in graph._vertices:
            if not includeInactive and not v.get("active", True):
                continue
            d = dict(v.get("dictionary", {}))
            d["index"] = v.get("index")
            d["active"] = v.get("active", True)
            rows.append(d)
            keys.update(d.keys())

        fieldnames = ["index", "active"] + sorted(k for k in keys if k not in ("index", "active"))
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        return out.getvalue()

    @staticmethod
    def EdgesCSVString(graph: "TGraph", includeInactive: bool = False) -> str:
        """
        Returns a CSV string for the graph's edge records.
        """

        if not isinstance(graph, TGraph):
            return ""

        keys = set(["index", "src", "dst", "directed", "active"])
        rows = []
        for e in graph._edges:
            if not includeInactive and not e.get("active", True):
                continue
            d = dict(e.get("dictionary", {}))
            d["index"] = e.get("index")
            d["src"] = e.get("src")
            d["dst"] = e.get("dst")
            d["directed"] = e.get("directed", graph._directed)
            d["active"] = e.get("active", True)
            rows.append(d)
            keys.update(d.keys())

        fieldnames = ["index", "src", "dst", "directed", "active"] + sorted(k for k in keys if k not in ("index", "src", "dst", "directed", "active"))
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        return out.getvalue()

    @staticmethod
    def CSVData(graph: "TGraph", includeInactive: bool = False) -> Dict[str, Any]:
        """
        Returns CSV serialisation data for a graph.

        Returns a dictionary with keys:
        - metadata
        - vertices_csv
        - edges_csv
        """

        if not isinstance(graph, TGraph):
            return {}
        return {
            "metadata": {
                "type": "TGraphCSV",
                "directed": graph._directed,
                "allowSelfLoops": graph._allow_self_loops,
                "allowParallelEdges": graph._allow_parallel_edges,
                "dictionary": dict(graph._dictionary),
            },
            "vertices_csv": TGraph.VerticesCSVString(graph, includeInactive=includeInactive),
            "edges_csv": TGraph.EdgesCSVString(graph, includeInactive=includeInactive),
        }

    @staticmethod
    def _CSVValue(value: Any) -> Any:
        """
        Converts a CSV string value into a Python value where appropriate.
        """

        if value is None:
            return None
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text == "":
            return None
        low = text.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low == "none" or low == "null":
            return None
        try:
            if any(ch in text for ch in [".", "e", "E"]):
                return float(text)
            return int(text)
        except Exception:
            pass
        try:
            return json.loads(text)
        except Exception:
            return text

    @staticmethod
    def ByCSVStrings(
        verticesCSVString: str,
        edgesCSVString: str,
        metadata: Optional[Dict[str, Any]] = None,
        directed: Optional[bool] = None,
        allowSelfLoops: Optional[bool] = None,
        allowParallelEdges: Optional[bool] = None,
    ) -> Optional["TGraph"]:
        """
        Constructs a TGraph from vertex and edge CSV strings.
        """

        if not isinstance(verticesCSVString, str) or not isinstance(edgesCSVString, str):
            return None

        metadata = metadata if isinstance(metadata, dict) else {}
        graph_directed = bool(metadata.get("directed", False)) if directed is None else bool(directed)
        graph_allow_self = bool(metadata.get("allowSelfLoops", True)) if allowSelfLoops is None else bool(allowSelfLoops)
        graph_allow_parallel = bool(metadata.get("allowParallelEdges", False)) if allowParallelEdges is None else bool(allowParallelEdges)
        graph_dictionary = metadata.get("dictionary", {}) if isinstance(metadata.get("dictionary", {}), dict) else {}

        g = TGraph(
            directed=graph_directed,
            allowSelfLoops=graph_allow_self,
            allowParallelEdges=graph_allow_parallel,
            dictionary=graph_dictionary,
        )

        reader = csv.DictReader(io.StringIO(verticesCSVString))
        vertex_rows = []
        for row in reader:
            converted = {k: TGraph._CSVValue(v) for k, v in row.items() if k is not None}
            vertex_rows.append(converted)

        vertex_rows.sort(key=lambda r: int(r.get("index", len(vertex_rows))) if r.get("index") is not None else len(vertex_rows))
        index_map = {}
        for row in vertex_rows:
            old_index = row.get("index")
            active = bool(row.get("active", True))
            d = {k: v for k, v in row.items() if k not in ("index", "active") and v is not None}
            new_index = g.AddVertex(dictionary=d)
            if old_index is not None:
                index_map[int(old_index)] = new_index
            if not active:
                g.RemoveVertex(new_index)

        reader = csv.DictReader(io.StringIO(edgesCSVString))
        for row in reader:
            converted = {k: TGraph._CSVValue(v) for k, v in row.items() if k is not None}
            src = converted.get("src")
            dst = converted.get("dst")
            if src is None or dst is None:
                continue
            src = index_map.get(int(src), int(src))
            dst = index_map.get(int(dst), int(dst))
            edge_directed = converted.get("directed", graph_directed)
            active = bool(converted.get("active", True))
            d = {k: v for k, v in converted.items() if k not in ("index", "src", "dst", "directed", "active") and v is not None}
            eid = g.AddEdge(src, dst, directed=bool(edge_directed), dictionary=d)
            if eid is not None and not active:
                g.RemoveEdge(eid)

        return g


    @staticmethod
    def _CSVFlatten(items: Any) -> List[Any]:
        """
        Flattens nested lists/tuples for CSV feature-key handling.
        """

        if items is None:
            return []
        if not isinstance(items, (list, tuple)):
            return [items]
        result = []
        for item in items:
            if isinstance(item, (list, tuple)):
                result.extend(TGraph._CSVFlatten(item))
            else:
                result.append(item)
        return result

    @staticmethod
    def _CSVFeatureHeaders(prefix: str, featureKeys: List[Any]) -> List[str]:
        """
        Returns feature headers matching Graph.ExportGraphsToCSV convention.

        If a supplied feature key already includes the requested prefix, it is
        returned unchanged. This prevents headers such as ``feat_feat_area``
        when callers pass keys that are already named ``feat_area``.
        """

        headers = []
        prefix = str(prefix) if prefix is not None else "feat"
        prefix_with_sep = prefix + "_"

        for key in featureKeys or []:
            key = str(key)
            if key.startswith(prefix_with_sep):
                headers.append(key)
            else:
                headers.append(f"{prefix}_{key}")
        return headers

    @staticmethod
    def _CSVLastGraphID(csvPath: str) -> int:
        """
        Returns the last graph id from the first column of a CSV file.
        """

        import os
        if not os.path.exists(csvPath):
            return -1
        try:
            with open(csvPath, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    return -1
                chunk = 4096
                data = b""
                pos = size
                while pos > 0:
                    step = chunk if pos >= chunk else pos
                    pos -= step
                    f.seek(pos, os.SEEK_SET)
                    data = f.read(step) + data
                    lines = data.splitlines()
                    if len(lines) >= 2:
                        break
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    s = line.decode("utf-8", errors="ignore")
                    first = s.split(",", 1)[0].strip()
                    if first == "" or first.lower() == "graph_id":
                        continue
                    try:
                        return int(float(first))
                    except Exception:
                        return -1
        except Exception:
            return -1
        return -1

    @staticmethod
    def _CSVMaskFromDictionaryOrRatio(
        dictionary: Optional[Dict[str, Any]],
        maskKey: Optional[str],
        trainMax: int,
        validateMax: int,
        counts: Dict[str, int],
    ) -> Tuple[bool, bool, bool]:
        """
        Returns train/validate/test booleans using Graph.ExportGraphsToCSV logic.
        """

        if maskKey is not None and isinstance(dictionary, dict):
            value = dictionary.get(maskKey)
            if value in [0, 1, 2, "0", "1", "2"]:
                value = int(value)
                if value == 0:
                    counts["train"] += 1
                    return True, False, False
                if value == 1:
                    counts["val"] += 1
                    return False, True, False
                counts["test"] += 1
                return False, False, True

        if counts["train"] < trainMax:
            counts["train"] += 1
            return True, False, False
        if counts["val"] < validateMax:
            counts["val"] += 1
            return False, True, False
        counts["test"] += 1
        return False, False, True

    @staticmethod
    def _CSVFeatureValues(dictionary: Optional[Dict[str, Any]], featureKeys: List[Any], mantissa: int = 6) -> List[float]:
        """
        Returns a stable numeric feature vector. Missing/invalid values become 0.0.
        """

        if not featureKeys:
            return []
        d = dictionary if isinstance(dictionary, dict) else {}
        values = []
        for key in featureKeys:
            try:
                value = d.get(key, None)
                if value is None:
                    values.append(0.0)
                else:
                    values.append(round(float(value), mantissa))
            except Exception:
                values.append(0.0)
        return values

    @staticmethod
    def _CSVLabelValue(dictionary: Optional[Dict[str, Any]], key: str, defaultValue: Any) -> Any:
        """
        Returns label value from dictionary with fallback.
        """

        d = dictionary if isinstance(dictionary, dict) else {}
        value = d.get(key, None)
        return defaultValue if value is None else value

    @staticmethod
    def ExportToCSV(
        graph,
        path,

        graphLabelKey="label",
        defaultGraphLabel=0,
        graphFeaturesKeys=None,
        graphIDHeader="graph_id",
        graphLabelHeader="label",
        graphFeaturesHeader="feat",

        edgeLabelKey="label",
        defaultEdgeLabel=0,
        edgeFeaturesKeys=None,
        edgeSRCHeader="src_id",
        edgeDSTHeader="dst_id",
        edgeLabelHeader="label",
        edgeFeaturesHeader="feat",
        edgeTrainMaskHeader="train_mask",
        edgeValidateMaskHeader="val_mask",
        edgeTestMaskHeader="test_mask",
        edgeMaskKey="mask",
        edgeTrainRatio=0.8,
        edgeValidateRatio=0.1,
        edgeTestRatio=0.1,
        bidirectional=True,

        nodeLabelKey="label",
        defaultNodeLabel=0,
        nodeFeaturesKeys=None,
        nodeIDHeader="node_id",
        nodeLabelHeader="label",
        nodeFeaturesHeader="feat",
        nodeTrainMaskHeader="train_mask",
        nodeValidateMaskHeader="val_mask",
        nodeTestMaskHeader="test_mask",
        nodeMaskKey="mask",
        nodeTrainRatio=0.8,
        nodeValidateRatio=0.1,
        nodeTestRatio=0.1,

        mantissa=6,
        tolerance=0.0001,
        overwrite=False,
        silent=False,
    ):
        """
        Exports the input TGraph or list of TGraphs into CSV files compatible
        with the Graph.ExportToCSV / Graph.ExportGraphsToCSV PyG format.

        The exported directory contains:
        - graphs.csv
        - nodes.csv
        - edges.csv
        - meta.yaml
        """

        if not isinstance(graph, list):
            graph = [graph]
        graph = [g for g in graph if isinstance(g, TGraph)]
        if len(graph) < 1:
            if not silent:
                print("TGraph.ExportToCSV - Error: The input graph parameter does not contain any valid TGraphs. Returning None")
            return None

        return TGraph.ExportGraphsToCSV(
            graphs=graph,
            path=path,
            graphLabelKey=graphLabelKey,
            defaultGraphLabel=defaultGraphLabel,
            graphFeaturesKeys=graphFeaturesKeys,
            graphIDHeader=graphIDHeader,
            graphLabelHeader=graphLabelHeader,
            graphFeaturesHeader=graphFeaturesHeader,

            edgeLabelKey=edgeLabelKey,
            defaultEdgeLabel=defaultEdgeLabel,
            edgeFeaturesKeys=edgeFeaturesKeys,
            edgeSRCHeader=edgeSRCHeader,
            edgeDSTHeader=edgeDSTHeader,
            edgeLabelHeader=edgeLabelHeader,
            edgeFeaturesHeader=edgeFeaturesHeader,
            edgeTrainMaskHeader=edgeTrainMaskHeader,
            edgeValidateMaskHeader=edgeValidateMaskHeader,
            edgeTestMaskHeader=edgeTestMaskHeader,
            edgeMaskKey=edgeMaskKey,
            edgeTrainRatio=edgeTrainRatio,
            edgeValidateRatio=edgeValidateRatio,
            edgeTestRatio=edgeTestRatio,
            bidirectional=bidirectional,

            nodeLabelKey=nodeLabelKey,
            defaultNodeLabel=defaultNodeLabel,
            nodeFeaturesKeys=nodeFeaturesKeys,
            nodeIDHeader=nodeIDHeader,
            nodeLabelHeader=nodeLabelHeader,
            nodeFeaturesHeader=nodeFeaturesHeader,
            nodeTrainMaskHeader=nodeTrainMaskHeader,
            nodeValidateMaskHeader=nodeValidateMaskHeader,
            nodeTestMaskHeader=nodeTestMaskHeader,
            nodeMaskKey=nodeMaskKey,
            nodeTrainRatio=nodeTrainRatio,
            nodeValidateRatio=nodeValidateRatio,
            nodeTestRatio=nodeTestRatio,

            mantissa=mantissa,
            tolerance=tolerance,
            overwrite=overwrite,
            silent=silent,
        )

    @staticmethod
    def ExportGraphsToCSV(
        graphs,
        path,
        graphLabelKey="label",
        defaultGraphLabel=0,
        graphFeaturesKeys=None,
        graphIDHeader="graph_id",
        graphLabelHeader="label",
        graphFeaturesHeader="feat",

        edgeLabelKey="label",
        defaultEdgeLabel=0,
        edgeFeaturesKeys=None,
        edgeSRCHeader="src_id",
        edgeDSTHeader="dst_id",
        edgeLabelHeader="label",
        edgeFeaturesHeader="feat",
        edgeTrainMaskHeader="train_mask",
        edgeValidateMaskHeader="val_mask",
        edgeTestMaskHeader="test_mask",
        edgeMaskKey="mask",
        edgeTrainRatio=0.8,
        edgeValidateRatio=0.1,
        edgeTestRatio=0.1,
        bidirectional=True,

        nodeLabelKey="label",
        defaultNodeLabel=0,
        nodeFeaturesKeys=None,
        nodeIDHeader="node_id",
        nodeLabelHeader="label",
        nodeFeaturesHeader="feat",
        nodeTrainMaskHeader="train_mask",
        nodeValidateMaskHeader="val_mask",
        nodeTestMaskHeader="test_mask",
        nodeMaskKey="mask",
        nodeTrainRatio=0.8,
        nodeValidateRatio=0.1,
        nodeTestRatio=0.1,
        mantissa=6,
        tolerance=0.0001,
        overwrite=False,
        silent=False,
    ):
        """
        Batch-exports a list of TGraphs to CSV files compatible with the
        existing Graph.ExportGraphsToCSV / PyG pipeline.

        Files written
        -------------
        graphs.csv
            graph_id, label, feat_<graph feature key>...
        nodes.csv
            graph_id, node_id, label, train_mask, val_mask, test_mask,
            feat_<node feature key>..., X, Y, Z
        edges.csv
            graph_id, src_id, dst_id, label, train_mask, val_mask, test_mask,
            feat_<edge feature key>...
        meta.yaml
            PyG-style dataset metadata.
        """

        import os
        import csv as _csv
        import math as _math
        import random as _random

        def _err(message):
            if not silent:
                print(message)
            return None

        if graphs is None or not isinstance(graphs, list) or len(graphs) == 0:
            return _err("TGraph.ExportGraphsToCSV - Error: 'graphs' must be a non-empty list. Returning None.")

        graphs = [g for g in graphs if isinstance(g, TGraph)]
        if len(graphs) == 0:
            return _err("TGraph.ExportGraphsToCSV - Error: 'graphs' does not contain valid TGraphs. Returning None.")

        if abs(float(nodeTrainRatio) + float(nodeValidateRatio) + float(nodeTestRatio) - 1.0) > 0.001:
            return _err("TGraph.ExportGraphsToCSV - Error: node train/val/test ratios must add up to 1. Returning None.")
        if abs(float(edgeTrainRatio) + float(edgeValidateRatio) + float(edgeTestRatio) - 1.0) > 0.001:
            return _err("TGraph.ExportGraphsToCSV - Error: edge train/val/test ratios must add up to 1. Returning None.")

        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            return _err("TGraph.ExportGraphsToCSV - Error: Could not create output folder. Returning None.")

        graphsCSV = os.path.join(path, "graphs.csv")
        nodesCSV = os.path.join(path, "nodes.csv")
        edgesCSV = os.path.join(path, "edges.csv")

        if overwrite is False:
            if not os.path.exists(graphsCSV):
                return _err("TGraph.ExportGraphsToCSV - Error: overwrite=False but graphs.csv not found. Returning None.")
            if not os.path.exists(nodesCSV):
                return _err("TGraph.ExportGraphsToCSV - Error: overwrite=False but nodes.csv not found. Returning None.")
            if not os.path.exists(edgesCSV):
                return _err("TGraph.ExportGraphsToCSV - Error: overwrite=False but edges.csv not found. Returning None.")

        startGraphID = 0 if overwrite else (TGraph._CSVLastGraphID(graphsCSV) + 1)

        graphFeatureKeys = TGraph._CSVFlatten([] if graphFeaturesKeys is None else graphFeaturesKeys)
        nodeFeatureKeys = TGraph._CSVFlatten([] if nodeFeaturesKeys is None else nodeFeaturesKeys)
        edgeFeatureKeys = TGraph._CSVFlatten([] if edgeFeaturesKeys is None else edgeFeaturesKeys)

        graphFeatureHeaders = TGraph._CSVFeatureHeaders(graphFeaturesHeader, graphFeatureKeys)
        nodeFeatureHeaders = TGraph._CSVFeatureHeaders(nodeFeaturesHeader, nodeFeatureKeys)
        edgeFeatureHeaders = TGraph._CSVFeatureHeaders(edgeFeaturesHeader, edgeFeatureKeys)

        graphsHeader = [graphIDHeader, graphLabelHeader] + graphFeatureHeaders
        nodesHeader = [
            graphIDHeader,
            nodeIDHeader,
            nodeLabelHeader,
            nodeTrainMaskHeader,
            nodeValidateMaskHeader,
            nodeTestMaskHeader,
        ] + nodeFeatureHeaders + ["X", "Y", "Z"]
        edgesHeader = [
            graphIDHeader,
            edgeSRCHeader,
            edgeDSTHeader,
            edgeLabelHeader,
            edgeTrainMaskHeader,
            edgeValidateMaskHeader,
            edgeTestMaskHeader,
        ] + edgeFeatureHeaders

        graphsMode = "w" if overwrite else "a"
        nodesMode = "w" if overwrite else "a"
        edgesMode = "w" if overwrite else "a"

        try:
            with open(graphsCSV, graphsMode, newline="", encoding="utf-8") as fg, \
                 open(nodesCSV, nodesMode, newline="", encoding="utf-8") as fn, \
                 open(edgesCSV, edgesMode, newline="", encoding="utf-8") as fe:

                wg = _csv.writer(fg)
                wn = _csv.writer(fn)
                we = _csv.writer(fe)

                if overwrite:
                    wg.writerow(graphsHeader)
                    wn.writerow(nodesHeader)
                    we.writerow(edgesHeader)

                for offset, graph in enumerate(graphs):
                    graphID = startGraphID + offset
                    graphDictionary = dict(graph._dictionary) if isinstance(graph._dictionary, dict) else {}
                    graphLabel = TGraph._CSVLabelValue(graphDictionary, graphLabelKey, defaultGraphLabel)
                    graphFeatures = TGraph._CSVFeatureValues(graphDictionary, graphFeatureKeys, mantissa=mantissa)
                    wg.writerow([graphID, graphLabel] + graphFeatures)

                    activeVertices = [v for v in graph._vertices if v.get("active", True)]
                    if len(activeVertices) < 3:
                        return _err(f"TGraph.ExportGraphsToCSV - Error: graph {graphID} is too small (<3 vertices). Returning None.")

                    vertices = _random.sample(activeVertices, len(activeVertices))
                    nodeCount = len(vertices)
                    nodeTrainMax = max(1, _math.floor(nodeCount * float(nodeTrainRatio)))
                    nodeValidateMax = max(1, _math.floor(nodeCount * float(nodeValidateRatio)))
                    nodeCounts = {"train": 0, "val": 0, "test": 0}

                    vertexIndexToNodeID = {}
                    for nodeID, vertexRecord in enumerate(vertices):
                        vertexIndex = vertexRecord.get("index")
                        vertexIndexToNodeID[vertexIndex] = nodeID
                        vertexDictionary = dict(vertexRecord.get("dictionary", {}))
                        nodeLabel = TGraph._CSVLabelValue(vertexDictionary, nodeLabelKey, defaultNodeLabel)
                        trainMask, validateMask, testMask = TGraph._CSVMaskFromDictionaryOrRatio(
                            vertexDictionary,
                            nodeMaskKey,
                            nodeTrainMax,
                            nodeValidateMax,
                            nodeCounts,
                        )
                        nodeFeatures = TGraph._CSVFeatureValues(vertexDictionary, nodeFeatureKeys, mantissa=mantissa)
                        coordinates = TGraph.Coordinates(graph, vertexIndex, default=[0.0, 0.0, 0.0])
                        if coordinates is None:
                            coordinates = [0.0, 0.0, 0.0]
                        x = round(float(coordinates[0]), mantissa)
                        y = round(float(coordinates[1]), mantissa)
                        z = round(float(coordinates[2]), mantissa)
                        wn.writerow([graphID, nodeID, nodeLabel, trainMask, validateMask, testMask] + nodeFeatures + [x, y, z])

                    activeEdges = [e for e in graph._edges if e.get("active", True)]
                    edgeCount = len(activeEdges)
                    edgeTrainMax = _math.floor(edgeCount * float(edgeTrainRatio))
                    edgeValidateMax = _math.floor(edgeCount * float(edgeValidateRatio))
                    edgeCounts = {"train": 0, "val": 0, "test": 0}

                    for edgeRecord in activeEdges:
                        src = edgeRecord.get("src")
                        dst = edgeRecord.get("dst")
                        if src not in vertexIndexToNodeID or dst not in vertexIndexToNodeID:
                            continue
                        edgeDictionary = dict(edgeRecord.get("dictionary", {}))
                        edgeLabel = TGraph._CSVLabelValue(edgeDictionary, edgeLabelKey, defaultEdgeLabel)
                        trainMask, validateMask, testMask = TGraph._CSVMaskFromDictionaryOrRatio(
                            edgeDictionary,
                            edgeMaskKey,
                            edgeTrainMax,
                            edgeValidateMax,
                            edgeCounts,
                        )
                        edgeFeatures = TGraph._CSVFeatureValues(edgeDictionary, edgeFeatureKeys, mantissa=mantissa)
                        srcID = vertexIndexToNodeID[src]
                        dstID = vertexIndexToNodeID[dst]
                        we.writerow([graphID, srcID, dstID, edgeLabel, trainMask, validateMask, testMask] + edgeFeatures)
                        if bidirectional:
                            we.writerow([graphID, dstID, srcID, edgeLabel, trainMask, validateMask, testMask] + edgeFeatures)

            with open(os.path.join(path, "meta.yaml"), "w", encoding="utf-8") as yamlFile:
                yamlFile.write(
                    "dataset_name: topologic_dataset\n"
                    "edge_data:\n- file_name: edges.csv\n"
                    "node_data:\n- file_name: nodes.csv\n"
                    "graph_data:\n  file_name: graphs.csv\n"
                )
            return True
        except Exception as exc:
            return _err(f"TGraph.ExportGraphsToCSV - Error: {exc}. Returning None.")

    @staticmethod
    def ByCSVPath(
        path: str,
        graphIDHeader: str = "graph_id",
        graphLabelHeader: str = "label",
        edgeSRCHeader: str = "src_id",
        edgeDSTHeader: str = "dst_id",
        edgeLabelHeader: str = "label",
        nodeIDHeader: str = "node_id",
        nodeLabelHeader: str = "label",
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = True,
        silent: bool = False,
    ):
        """
        Imports TGraphs from Graph/PyG-compatible CSV files.

        If the directory contains graphs.csv, nodes.csv, and edges.csv, returns
        a list of TGraphs, one per graph_id. This mirrors the dataset-oriented
        CSV structure used by Graph.ExportGraphsToCSV and the PyG class.

        For backwards compatibility with the earlier TGraph-only CSV export,
        if metadata.json, vertices.csv, and edges.csv are found instead, returns
        a single TGraph from those files.
        """

        import os
        import csv as _csv

        def _err(message):
            if not silent:
                print(message)
            return None

        if not isinstance(path, str):
            return None

        graphsCSV = os.path.join(path, "graphs.csv")
        nodesCSV = os.path.join(path, "nodes.csv")
        edgesCSV = os.path.join(path, "edges.csv")

        if os.path.exists(graphsCSV) and os.path.exists(nodesCSV) and os.path.exists(edgesCSV):
            try:
                graphRows = []
                with open(graphsCSV, newline="", encoding="utf-8") as f:
                    for row in _csv.DictReader(f):
                        graphRows.append({k: TGraph._CSVValue(v) for k, v in row.items() if k is not None})

                nodeRowsByGraph = {}
                with open(nodesCSV, newline="", encoding="utf-8") as f:
                    for row in _csv.DictReader(f):
                        converted = {k: TGraph._CSVValue(v) for k, v in row.items() if k is not None}
                        gid = converted.get(graphIDHeader)
                        nodeRowsByGraph.setdefault(gid, []).append(converted)

                edgeRowsByGraph = {}
                with open(edgesCSV, newline="", encoding="utf-8") as f:
                    for row in _csv.DictReader(f):
                        converted = {k: TGraph._CSVValue(v) for k, v in row.items() if k is not None}
                        gid = converted.get(graphIDHeader)
                        edgeRowsByGraph.setdefault(gid, []).append(converted)

                result = []
                graphRows.sort(key=lambda r: int(r.get(graphIDHeader, 0)))
                for graphRow in graphRows:
                    gid = graphRow.get(graphIDHeader)
                    graphDictionary = {k: v for k, v in graphRow.items() if k != graphIDHeader and v is not None}
                    g = TGraph(
                        directed=directed,
                        allowSelfLoops=allowSelfLoops,
                        allowParallelEdges=allowParallelEdges,
                        dictionary=graphDictionary,
                    )

                    nodeRows = sorted(nodeRowsByGraph.get(gid, []), key=lambda r: int(r.get(nodeIDHeader, 0)))
                    nodeIDToVertexIndex = {}
                    for nodeRow in nodeRows:
                        nodeID = int(nodeRow.get(nodeIDHeader, len(nodeIDToVertexIndex)))
                        d = {}
                        for k, v in nodeRow.items():
                            if k in [graphIDHeader, nodeIDHeader, "X", "Y", "Z"]:
                                continue
                            if v is not None:
                                d[k] = v
                        if nodeLabelHeader in nodeRow:
                            d.setdefault("label", nodeRow.get(nodeLabelHeader))
                        for coordKey in ["X", "Y", "Z"]:
                            if coordKey in nodeRow and nodeRow.get(coordKey) is not None:
                                d[coordKey.lower()] = float(nodeRow.get(coordKey))
                        nodeIDToVertexIndex[nodeID] = g.AddVertex(dictionary=d)

                    for edgeRow in edgeRowsByGraph.get(gid, []):
                        srcID = edgeRow.get(edgeSRCHeader)
                        dstID = edgeRow.get(edgeDSTHeader)
                        if srcID is None or dstID is None:
                            continue
                        srcID = int(srcID)
                        dstID = int(dstID)
                        if srcID not in nodeIDToVertexIndex or dstID not in nodeIDToVertexIndex:
                            continue
                        d = {}
                        for k, v in edgeRow.items():
                            if k in [graphIDHeader, edgeSRCHeader, edgeDSTHeader]:
                                continue
                            if v is not None:
                                d[k] = v
                        if edgeLabelHeader in edgeRow:
                            d.setdefault("label", edgeRow.get(edgeLabelHeader))
                        g.AddEdge(nodeIDToVertexIndex[srcID], nodeIDToVertexIndex[dstID], dictionary=d)
                    result.append(g)
                return result
            except Exception as exc:
                return _err(f"TGraph.ByCSVPath - Error: {exc}. Returning None.")

        # Backwards compatibility with earlier TGraph v3.0 record CSV format.
        try:
            from pathlib import Path
            folder = Path(path)
            metadataPath = folder / "metadata.json"
            verticesPath = folder / "vertices.csv"
            edgesPath = folder / "edges.csv"
            if metadataPath.exists() and verticesPath.exists() and edgesPath.exists():
                metadata = json.loads(metadataPath.read_text(encoding="utf-8"))
                verticesCSV = verticesPath.read_text(encoding="utf-8")
                edgesCSV = edgesPath.read_text(encoding="utf-8")
                return TGraph.ByCSVStrings(verticesCSV, edgesCSV, metadata=metadata)
        except Exception:
            pass

        return _err("TGraph.ByCSVPath - Error: Could not find compatible CSV files. Returning None.")


    # ---------------------------------------------------------------------
    # v3.0 ontology / RDF / JSON-LD parity
    # ---------------------------------------------------------------------


    @staticmethod
    def _OntologyConfig() -> Dict[str, Any]:
        """Returns ontology constants, using topologicpy.Ontology when available."""
        fallback_namespaces = {
            "bot": "https://w3id.org/bot#",
            "brick": "https://brickschema.org/schema/Brick#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "top": "http://w3id.org/topologicpy#",
        }
        fallback_top_to_bot = {
            "top:Site": "bot:Site",
            "top:Building": "bot:Building",
            "top:Storey": "bot:Storey",
            "top:Zone": "bot:Zone",
            "top:Space": "bot:Space",
            "top:Room": "bot:Space",
            "top:Element": "bot:Element",
            "top:Wall": "bot:Element",
            "top:Door": "bot:Element",
            "top:Window": "bot:Element",
            "top:Slab": "bot:Element",
            "top:Roof": "bot:Element",
            "top:Column": "bot:Element",
            "top:Beam": "bot:Element",
            "top:Interface": "bot:Interface",
        }
        fallback_categories = {
            "top:Graph": "graph",
            "top:SpatialGraph": "graph",
            "top:AdjacencyGraph": "graph",
            "top:VisibilityGraph": "graph",
            "top:CirculationGraph": "graph",
            "top:ConnectivityGraph": "graph",
            "top:KnowledgeGraph": "graph",
            "top:Node": "topology",
            "top:Vertex": "topology",
            "top:Relationship": "topology",
            "top:Edge": "topology",
            "top:Space": "space",
            "top:Room": "space",
            "top:Wall": "element",
            "top:Door": "element",
            "top:Window": "element",
            "top:Element": "element",
        }
        fallback_ifc = {
            "IfcProject": "top:Project",
            "IfcSite": "top:Site",
            "IfcBuilding": "top:Building",
            "IfcBuildingStorey": "top:Storey",
            "IfcSpace": "top:Space",
            "IfcZone": "top:Zone",
            "IfcWall": "top:Wall",
            "IfcWallStandardCase": "top:Wall",
            "IfcCurtainWall": "top:Wall",
            "IfcDoor": "top:Door",
            "IfcWindow": "top:Window",
            "IfcSlab": "top:Slab",
            "IfcRoof": "top:Roof",
            "IfcColumn": "top:Column",
            "IfcBeam": "top:Beam",
            "IfcOpeningElement": "top:Opening",
            "IfcFurnishingElement": "top:Furniture",
            "IfcFurniture": "top:Furniture",
            "IfcDistributionElement": "top:Equipment",
            "IfcBuildingElementProxy": "top:Element",
        }
        fallback_aliases = {
            "startsAt": "hasStartVertex",
            "endsAt": "hasEndVertex",
            "connectedTo": "connectsTo",
            "x": "hasX",
            "y": "hasY",
            "z": "hasZ",
            "length": "hasLength",
            "area": "hasArea",
            "volume": "hasVolume",
            "mantissa": "hasMantissa",
            "unit": "hasUnit",
        }
        config = {
            "namespaces": fallback_namespaces,
            "top_to_bot": fallback_top_to_bot,
            "categories": fallback_categories,
            "ifc_to_top": fallback_ifc,
            "aliases": fallback_aliases,
        }
        try:
            from topologicpy.Ontology import Ontology
            config["namespaces"] = dict(getattr(Ontology, "NAMESPACES", fallback_namespaces))
            config["top_to_bot"] = dict(getattr(Ontology, "TOP_TO_BOT", fallback_top_to_bot))
            config["categories"] = dict(getattr(Ontology, "TOP_CATEGORIES", fallback_categories))
            config["ifc_to_top"] = dict(getattr(Ontology, "IFC_TO_TOP", fallback_ifc))
            config["aliases"] = dict(getattr(Ontology, "PROPERTY_ALIASES", fallback_aliases))
        except Exception:
            pass
        return config

    @staticmethod
    def _OntologyDictionary(graph: "TGraph", element: str = "graph", index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not isinstance(graph, TGraph):
            return None
        element = str(element or "graph").lower()
        if element in ("graph", "g"):
            if not isinstance(graph._dictionary, dict):
                graph._dictionary = {}
            return graph._dictionary
        if element in ("vertex", "node", "v"):
            if not graph._validate_vertex_index(index, active=False):
                return None
            return graph._vertices[index].setdefault("dictionary", {})
        if element in ("edge", "relationship", "e"):
            if not graph._validate_edge_index(index, active=False):
                return None
            return graph._edges[index].setdefault("dictionary", {})
        return None

    @staticmethod
    def _OntologyGet(graph: "TGraph", key: str, defaultValue: Any = None, element: str = "graph", index: Optional[int] = None) -> Any:
        d = TGraph._OntologyDictionary(graph, element=element, index=index)
        if not isinstance(d, dict):
            return defaultValue
        value = d.get(key, defaultValue)
        return defaultValue if value is None else value

    @staticmethod
    def _OntologySet(graph: "TGraph", key: str, value: Any, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        if key is None:
            return None
        d = TGraph._OntologyDictionary(graph, element=element, index=index)
        if not isinstance(d, dict):
            return None
        d[str(key)] = value
        if isinstance(graph, TGraph):
            graph._invalidate_cache()
        return graph

    @staticmethod
    def _OntologySafeString(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    @staticmethod
    def _OntologySafeLocalName(value: Any) -> str:
        import re
        if value is None:
            return "unnamed"
        s = str(value).strip()
        if s == "":
            return "unnamed"
        s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
        if s == "":
            s = "unnamed"
        if s[0].isdigit():
            s = "id_" + s
        return s

    @staticmethod
    def _OntologyRDFLiteral(value: Any) -> str:
        if isinstance(value, bool):
            return '"' + str(value).lower() + '"^^xsd:boolean'
        if isinstance(value, int) and not isinstance(value, bool):
            return '"' + str(value) + '"^^xsd:integer'
        if isinstance(value, float):
            return '"' + repr(float(value)) + '"^^xsd:double'
        return '"' + TGraph._OntologySafeString(value) + '"'

    @staticmethod
    def _OntologyExpandQName(qname: str, defaultValue: Any = None) -> Any:
        if not isinstance(qname, str) or ":" not in qname:
            return defaultValue
        prefix, local = qname.split(":", 1)
        ns = TGraph._OntologyConfig()["namespaces"].get(prefix)
        if ns is None:
            return defaultValue
        return ns + local

    @staticmethod
    def _OntologyPropertyQName(key: str, defaultPrefix: str = "top") -> Optional[str]:
        if key is None:
            return None
        key = str(key).strip()
        if key == "":
            return None
        if ":" in key:
            return key
        aliases = TGraph._OntologyConfig()["aliases"]
        key = aliases.get(key, key)
        return str(defaultPrefix) + ":" + TGraph._OntologySafeLocalName(key)

    @staticmethod
    def _OntologySubjectFromDictionary(dictionary: Dict[str, Any], fallback: str, namespacePrefix: str = "inst") -> str:
        d = dictionary if isinstance(dictionary, dict) else {}
        for key in ("uri", "ifc_guid", "global_id", "guid", "label", "name"):
            value = d.get(key, None)
            if value not in (None, ""):
                if key == "uri" and ":" in str(value):
                    return str(value)
                return namespacePrefix + ":" + TGraph._OntologySafeLocalName(value)
        return namespacePrefix + ":" + TGraph._OntologySafeLocalName(fallback)

    @staticmethod
    def OntologyClass(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        return TGraph._OntologyGet(graph, "ontology_class", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def SetOntologyClass(graph: "TGraph", ontologyClass: str, element: str = "graph", index: Optional[int] = None,
                         setCategory: bool = True, setURI: bool = True) -> Optional["TGraph"]:
        if not isinstance(ontologyClass, str) or ontologyClass.strip() == "":
            return None
        ontologyClass = ontologyClass.strip()
        graph = TGraph._OntologySet(graph, "ontology_class", ontologyClass, element=element, index=index)
        if graph is None:
            return None
        if setCategory:
            category = TGraph.CategoryByOntologyClass(ontologyClass, defaultValue=None)
            if category is not None:
                TGraph.SetOntologyCategory(graph, category, element=element, index=index)
        if setURI:
            uri = TGraph._OntologyExpandQName(ontologyClass, defaultValue=None)
            if uri is not None:
                TGraph._OntologySet(graph, "ontology_uri", uri, element=element, index=index)
        return graph

    @staticmethod
    def OntologyCategory(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        return TGraph._OntologyGet(graph, "category", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def SetOntologyCategory(graph: "TGraph", category: str, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        if not isinstance(category, str) or category.strip() == "":
            return None
        return TGraph._OntologySet(graph, "category", category.strip(), element=element, index=index)

    @staticmethod
    def OntologyLabel(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        return TGraph._OntologyGet(graph, "label", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def SetOntologyLabel(graph: "TGraph", label: Any, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        return TGraph._OntologySet(graph, "label", label, element=element, index=index)

    @staticmethod
    def OntologyURI(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        return TGraph._OntologyGet(graph, "uri", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def SetOntologyURI(graph: "TGraph", uri: str, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        if not isinstance(uri, str) or uri.strip() == "":
            return None
        return TGraph._OntologySet(graph, "uri", uri.strip(), element=element, index=index)

    @staticmethod
    def CategoryByOntologyClass(ontologyClass: str, defaultValue: Any = None) -> Any:
        if ontologyClass is None:
            return defaultValue
        return TGraph._OntologyConfig()["categories"].get(str(ontologyClass).strip(), defaultValue)

    @staticmethod
    def BOTClassByOntologyClass(ontologyClass: str, defaultValue: Any = None) -> Any:
        if ontologyClass is None:
            return defaultValue
        ontologyClass = str(ontologyClass).strip()
        config = TGraph._OntologyConfig()
        if ontologyClass in config["top_to_bot"]:
            return config["top_to_bot"][ontologyClass]
        return defaultValue

    @staticmethod
    def OntologyClassByIFCClass(ifcClass: str, defaultValue: Any = "top:Element") -> Any:
        if ifcClass is None:
            return defaultValue
        return TGraph._OntologyConfig()["ifc_to_top"].get(str(ifcClass).strip(), defaultValue)

    @staticmethod
    def AnnotateOntology(
        graph: "TGraph",
        ontologyClass: Optional[str] = None,
        category: Optional[str] = None,
        label: Any = None,
        uri: Optional[str] = None,
        source: Any = None,
        derivedFrom: Any = None,
        generatedBy: Any = None,
        element: str = "graph",
        index: Optional[int] = None,
    ) -> Optional["TGraph"]:
        if not isinstance(graph, TGraph):
            return None
        if ontologyClass is not None:
            if TGraph.SetOntologyClass(graph, ontologyClass, element=element, index=index) is None:
                return None
        if category is not None:
            TGraph.SetOntologyCategory(graph, category, element=element, index=index)
        if label is not None:
            TGraph.SetOntologyLabel(graph, label, element=element, index=index)
        if uri is not None:
            TGraph.SetOntologyURI(graph, uri, element=element, index=index)
        if source is not None:
            TGraph._OntologySet(graph, "source", source, element=element, index=index)
        if derivedFrom is not None:
            TGraph._OntologySet(graph, "derived_from", derivedFrom, element=element, index=index)
        if generatedBy is not None:
            TGraph._OntologySet(graph, "generated_by", generatedBy, element=element, index=index)
        return graph

    @staticmethod
    def AnnotateIFC(
        graph: "TGraph",
        ifcClass: Optional[str] = None,
        ifcGUID: Optional[str] = None,
        ifcName: Optional[str] = None,
        source: Optional[str] = None,
        element: str = "graph",
        index: Optional[int] = None,
    ) -> Optional["TGraph"]:
        if not isinstance(graph, TGraph):
            return None
        if ifcClass is not None:
            TGraph._OntologySet(graph, "ifc_class", ifcClass, element=element, index=index)
            ontologyClass = TGraph.OntologyClassByIFCClass(ifcClass, defaultValue=None)
            if ontologyClass is not None:
                TGraph.SetOntologyClass(graph, ontologyClass, element=element, index=index)
        if ifcGUID is not None:
            TGraph._OntologySet(graph, "ifc_guid", ifcGUID, element=element, index=index)
        if ifcName is not None:
            TGraph.SetOntologyLabel(graph, ifcName, element=element, index=index)
        if source is not None:
            TGraph._OntologySet(graph, "source", source, element=element, index=index)
        return graph

    @staticmethod
    def NormalizeOntologyDictionaries(
        graph: "TGraph",
        labelKeys: Optional[List[str]] = None,
        categoryKeys: Optional[List[str]] = None,
        ifcClassKeys: Optional[List[str]] = None,
        ifcGUIDKeys: Optional[List[str]] = None,
        includeGraph: bool = True,
        includeVertices: bool = True,
        includeEdges: bool = True,
    ) -> Optional["TGraph"]:
        if not isinstance(graph, TGraph):
            return None
        labelKeys = labelKeys or ["name", "Name", "LongName", "ifc_name", "label"]
        categoryKeys = categoryKeys or ["category", "type", "ObjectType"]
        ifcClassKeys = ifcClassKeys or ["ifc_class", "IfcClass", "class", "type"]
        ifcGUIDKeys = ifcGUIDKeys or ["ifc_guid", "GlobalId", "global_id", "guid"]

        def _first(d, keys):
            for k in keys:
                if isinstance(d, dict) and d.get(k, None) not in (None, ""):
                    return d.get(k)
            return None

        targets = []
        if includeGraph:
            targets.append(("graph", None, graph._dictionary))
        if includeVertices:
            for v in graph._vertices:
                targets.append(("vertex", v.get("index"), v.get("dictionary", {})))
        if includeEdges:
            for e in graph._edges:
                targets.append(("edge", e.get("index"), e.get("dictionary", {})))

        for element, index, d in targets:
            label = _first(d, labelKeys)
            category = _first(d, categoryKeys)
            ifcClass = _first(d, ifcClassKeys)
            ifcGUID = _first(d, ifcGUIDKeys)
            if label is not None:
                TGraph.SetOntologyLabel(graph, label, element=element, index=index)
            if category is not None:
                TGraph.SetOntologyCategory(graph, str(category).lower(), element=element, index=index)
            if ifcClass is not None:
                TGraph.AnnotateIFC(graph, ifcClass=ifcClass, element=element, index=index)
            if ifcGUID is not None:
                TGraph._OntologySet(graph, "ifc_guid", ifcGUID, element=element, index=index)
        return graph

    @staticmethod
    def OntologyTriples(
        graph: "TGraph",
        includeVertices: bool = True,
        includeEdges: bool = True,
        includeDictionaries: bool = True,
        includeBOT: bool = True,
        namespacePrefix: str = "inst",
    ) -> List[Tuple[str, str, str]]:
        if not isinstance(graph, TGraph):
            return []
        triples: List[Tuple[str, str, str]] = []
        graph_subject = TGraph._OntologySubjectFromDictionary(graph._dictionary, "graph", namespacePrefix=namespacePrefix)
        graph_class = graph._dictionary.get("ontology_class", "top:Graph")
        triples.append((graph_subject, "rdf:type", graph_class))
        if includeBOT:
            botClass = TGraph.BOTClassByOntologyClass(graph_class)
            if botClass is not None:
                triples.append((graph_subject, "rdf:type", botClass))

        def _dictionary_triples(subject, d, default_class=None):
            local = []
            d = d if isinstance(d, dict) else {}
            ontologyClass = d.get("ontology_class", default_class)
            if ontologyClass is not None and (subject, "rdf:type", ontologyClass) not in triples:
                local.append((subject, "rdf:type", ontologyClass))
                if includeBOT:
                    botClass = TGraph.BOTClassByOntologyClass(ontologyClass)
                    if botClass is not None:
                        local.append((subject, "rdf:type", botClass))
            label = d.get("label", None)
            if label is not None:
                local.append((subject, "rdfs:label", TGraph._OntologyRDFLiteral(label)))
            category = d.get("category", None)
            if category is not None:
                local.append((subject, "top:category", TGraph._OntologyRDFLiteral(category)))
            if includeDictionaries:
                skip = {"ontology_class", "ontology_uri", "label", "category", "uri"}
                for key, value in d.items():
                    if key in skip or value is None:
                        continue
                    predicate = TGraph._OntologyPropertyQName(key)
                    if predicate is None:
                        continue
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            local.append((subject, predicate, TGraph._OntologyRDFLiteral(item)))
                    else:
                        local.append((subject, predicate, TGraph._OntologyRDFLiteral(value)))
            return local

        triples.extend(_dictionary_triples(graph_subject, graph._dictionary, default_class="top:Graph"))

        vertex_subjects: Dict[int, str] = {}
        if includeVertices:
            for v in graph._vertices:
                if not v.get("active", True):
                    continue
                idx = v.get("index")
                d = dict(v.get("dictionary", {}))
                subject = TGraph._OntologySubjectFromDictionary(d, f"vertex_{idx}", namespacePrefix=namespacePrefix)
                vertex_subjects[idx] = subject
                triples.append((graph_subject, "top:hasNode", subject))
                triples.extend(_dictionary_triples(subject, d, default_class="top:Node"))
                coords = TGraph.Coordinates(graph, idx, default=None)
                if coords is not None:
                    triples.append((subject, "top:hasX", TGraph._OntologyRDFLiteral(float(coords[0]))))
                    triples.append((subject, "top:hasY", TGraph._OntologyRDFLiteral(float(coords[1]))))
                    triples.append((subject, "top:hasZ", TGraph._OntologyRDFLiteral(float(coords[2]))))

        if includeEdges:
            for e in graph._edges:
                if not e.get("active", True):
                    continue
                idx = e.get("index")
                src = e.get("src")
                dst = e.get("dst")
                d = dict(e.get("dictionary", {}))
                subject = TGraph._OntologySubjectFromDictionary(d, f"edge_{idx}", namespacePrefix=namespacePrefix)
                triples.append((graph_subject, "top:hasRelationship", subject))
                triples.extend(_dictionary_triples(subject, d, default_class="top:Relationship"))
                sv = vertex_subjects.get(src, namespacePrefix + ":" + TGraph._OntologySafeLocalName(f"vertex_{src}"))
                tv = vertex_subjects.get(dst, namespacePrefix + ":" + TGraph._OntologySafeLocalName(f"vertex_{dst}"))
                triples.append((subject, "top:hasStartVertex", sv))
                triples.append((subject, "top:hasEndVertex", tv))
                triples.append((sv, "top:connectsTo", tv))
                if not e.get("directed", graph._directed):
                    triples.append((tv, "top:connectsTo", sv))
        return triples

    @staticmethod
    def TurtleFromTriples(
        triples: List[Tuple[str, str, str]],
        namespaces: Optional[Dict[str, str]] = None,
        instanceNamespace: str = "http://w3id.org/topologicpy/instance#",
        includeHeader: bool = True,
    ) -> str:
        namespaces = dict(namespaces or TGraph._OntologyConfig()["namespaces"])
        if "inst" not in namespaces:
            namespaces["inst"] = instanceNamespace
        lines: List[str] = []
        if includeHeader:
            for prefix, uri in namespaces.items():
                lines.append(f"@prefix {prefix}: <{uri}> .")
            lines.append("")
        for triple in triples or []:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            s, p, o = triple
            if s is None or p is None or o is None:
                continue
            lines.append(f"{s} {p} {o} .")
        return "\n".join(lines) + "\n"

    @staticmethod
    def TTLString(
        graph: "TGraph",
        includeVertices: bool = True,
        includeEdges: bool = True,
        includeDictionaries: bool = True,
        includeBOT: bool = True,
        namespacePrefix: str = "inst",
        instanceNamespace: str = "http://w3id.org/topologicpy/instance#",
        silent: bool = False,
    ) -> Optional[str]:
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.TTLString - Error: The input graph is not a valid TGraph. Returning None.")
            return None
        triples = TGraph.OntologyTriples(
            graph,
            includeVertices=includeVertices,
            includeEdges=includeEdges,
            includeDictionaries=includeDictionaries,
            includeBOT=includeBOT,
            namespacePrefix=namespacePrefix,
        )
        return TGraph.TurtleFromTriples(triples, instanceNamespace=instanceNamespace)

    @staticmethod
    def RDFString(*args, **kwargs) -> Optional[str]:
        """Alias for TTLString for compatibility with Graph/Ontology workflows."""
        return TGraph.TTLString(*args, **kwargs)

    @staticmethod
    def BOTString(*args, **kwargs) -> Optional[str]:
        """Returns a Turtle string with BOT alignment triples enabled."""
        kwargs["includeBOT"] = True
        return TGraph.TTLString(*args, **kwargs)

    @staticmethod
    def ExportTTL(graph: "TGraph", path: str, silent: bool = False, **kwargs) -> Optional[str]:
        if not isinstance(path, str) or path.strip() == "":
            if not silent:
                print("TGraph.ExportTTL - Error: The input path is not a valid string. Returning None.")
            return None
        ttl = TGraph.TTLString(graph, silent=silent, **kwargs)
        if ttl is None:
            return None
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(ttl)
            return path
        except Exception as exc:
            if not silent:
                print(f"TGraph.ExportTTL - Error: {exc}. Returning None.")
            return None

    @staticmethod
    def JSONLDData(
        graph: "TGraph",
        includeVertices: bool = True,
        includeEdges: bool = True,
        includeDictionaries: bool = True,
        includeBOT: bool = True,
        namespacePrefix: str = "inst",
        instanceNamespace: str = "http://w3id.org/topologicpy/instance#",
    ) -> Dict[str, Any]:
        if not isinstance(graph, TGraph):
            return {}
        namespaces = dict(TGraph._OntologyConfig()["namespaces"])
        namespaces.setdefault("inst", instanceNamespace)
        triples = TGraph.OntologyTriples(
            graph,
            includeVertices=includeVertices,
            includeEdges=includeEdges,
            includeDictionaries=includeDictionaries,
            includeBOT=includeBOT,
            namespacePrefix=namespacePrefix,
        )
        nodes: Dict[str, Dict[str, Any]] = {}
        for s, p, o in triples:
            node = nodes.setdefault(str(s), {"@id": str(s)})
            if p == "rdf:type":
                node.setdefault("@type", [])
                if o not in node["@type"]:
                    node["@type"].append(o)
                continue
            value: Any = str(o)
            if isinstance(o, str) and o.startswith('"'):
                value = o
            predicate = str(p)
            node.setdefault(predicate, [])
            if value not in node[predicate]:
                node[predicate].append(value)
        for node in nodes.values():
            if "@type" in node and len(node["@type"]) == 1:
                node["@type"] = node["@type"][0]
            for k in list(node.keys()):
                if k not in ("@id", "@type") and isinstance(node[k], list) and len(node[k]) == 1:
                    node[k] = node[k][0]
        return {"@context": namespaces, "@graph": list(nodes.values())}

    @staticmethod
    def JSONLDString(graph: "TGraph", indent: Optional[int] = 2, **kwargs) -> str:
        try:
            return json.dumps(TGraph.JSONLDData(graph, **kwargs), indent=indent)
        except Exception:
            return "{}"

    @staticmethod
    def ExportJSONLD(graph: "TGraph", path: str, indent: Optional[int] = 2, silent: bool = False, **kwargs) -> Optional[str]:
        if not isinstance(path, str) or path.strip() == "":
            if not silent:
                print("TGraph.ExportJSONLD - Error: The input path is not a valid string. Returning None.")
            return None
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(TGraph.JSONLDString(graph, indent=indent, **kwargs))
            return path
        except Exception as exc:
            if not silent:
                print(f"TGraph.ExportJSONLD - Error: {exc}. Returning None.")
            return None

    @staticmethod
    def ValidateOntology(
        graph: "TGraph",
        requireClass: bool = True,
        requireVertexClasses: bool = False,
        requireEdgeClasses: bool = False,
        requireLabels: bool = False,
        checkClassKnown: bool = True,
        checkCategory: bool = True,
        silent: bool = False,
    ) -> Dict[str, Any]:
        report = {"ok": False, "errors": [], "warnings": [], "graph": {}, "vertices": [], "edges": []}
        if not isinstance(graph, TGraph):
            report["errors"].append("The input graph is not a valid TGraph.")
            return report
        config = TGraph._OntologyConfig()

        def _validate_dict(d, label, require_class, require_label):
            r = {"ok": False, "errors": [], "warnings": [], "dictionary": dict(d if isinstance(d, dict) else {})}
            ontologyClass = r["dictionary"].get("ontology_class")
            category = r["dictionary"].get("category")
            elementLabel = r["dictionary"].get("label")
            if require_class and (ontologyClass is None or str(ontologyClass).strip() == ""):
                r["errors"].append("Missing ontology_class.")
            if require_label and (elementLabel is None or str(elementLabel).strip() == ""):
                r["errors"].append("Missing label.")
            if ontologyClass not in (None, ""):
                ontologyClass = str(ontologyClass).strip()
                if ":" in ontologyClass:
                    prefix = ontologyClass.split(":", 1)[0]
                    if prefix not in config["namespaces"]:
                        r["errors"].append(f"Unknown ontology_class prefix: {prefix}.")
                known = ontologyClass in config["categories"] or TGraph._OntologyExpandQName(ontologyClass) is not None
                if checkClassKnown and not known:
                    r["warnings"].append(f"ontology_class is not known and cannot be expanded: {ontologyClass}.")
                expected = TGraph.CategoryByOntologyClass(ontologyClass, defaultValue=None)
                if checkCategory and expected is not None and category not in (None, "") and str(category).lower() != str(expected).lower():
                    r["warnings"].append(f"Category '{category}' does not match inferred category '{expected}' for {ontologyClass}.")
            r["ok"] = len(r["errors"]) == 0
            for error in r["errors"]:
                report["errors"].append(f"{label}: {error}")
            for warning in r["warnings"]:
                report["warnings"].append(f"{label}: {warning}")
            return r

        report["graph"] = _validate_dict(graph._dictionary, "Graph", requireClass, requireLabels)
        for v in graph._vertices:
            if not v.get("active", True):
                continue
            idx = v.get("index")
            r = _validate_dict(v.get("dictionary", {}), f"Vertex {idx}", requireVertexClasses, requireLabels)
            r["index"] = idx
            report["vertices"].append(r)
        for e in graph._edges:
            if not e.get("active", True):
                continue
            idx = e.get("index")
            r = _validate_dict(e.get("dictionary", {}), f"Edge {idx}", requireEdgeClasses, False)
            r["index"] = idx
            if not graph._validate_vertex_index(e.get("src"), active=False) or not graph._validate_vertex_index(e.get("dst"), active=False):
                r["errors"].append("Could not resolve edge start/end vertices.")
                r["ok"] = False
                report["errors"].append(f"Edge {idx}: Could not resolve edge start/end vertices.")
            report["edges"].append(r)
        report["ok"] = len(report["errors"]) == 0
        if not silent and (report["errors"] or report["warnings"]):
            for error in report["errors"]:
                print("TGraph.ValidateOntology - Error:", error)
            for warning in report["warnings"]:
                print("TGraph.ValidateOntology - Warning:", warning)
        return report

    @staticmethod
    def ValidateTTLString(ttlString: str, silent: bool = False) -> Dict[str, Any]:
        report = {"ok": False, "errors": [], "warnings": [], "triple_count": 0}
        if not isinstance(ttlString, str) or ttlString.strip() == "":
            report["errors"].append("The input ttlString is not a valid string.")
            return report
        try:
            import rdflib
        except Exception:
            report["warnings"].append("RDFLib is not installed, so syntax validation could not be performed.")
            report["ok"] = True
            return report
        try:
            g = rdflib.Graph()
            g.parse(data=ttlString, format="turtle")
            report["triple_count"] = len(g)
            report["ok"] = True
            return report
        except Exception as exc:
            report["errors"].append(str(exc))
            if not silent:
                print("TGraph.ValidateTTLString - Error:", exc)
            return report


    # ---------------------------------------------------------------------
    # API parity convenience methods
    # ---------------------------------------------------------------------


    @staticmethod
    def IsEmpty(graph: "TGraph") -> bool:
        """Returns True if the input graph has no active vertices."""
        return not isinstance(graph, TGraph) or TGraph.Order(graph) == 0

    @staticmethod
    def DegreeMatrix(graph: "TGraph", mode: str = "all") -> List[List[int]]:
        """Returns the degree matrix of the input graph."""
        if not isinstance(graph, TGraph):
            return []
        degrees = TGraph.DegreeSequence(graph, mode=mode)
        n = len(degrees)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i, d in enumerate(degrees):
            matrix[i][i] = d
        return matrix

    @staticmethod
    def EigenVectorCentrality(
        graph: "TGraph",
        normalize: bool = False,
        key: str = "eigen_vector_centrality",
        colorKey: str = "evc_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> List[float]:
        """Compatibility alias for EigenvectorCentrality."""
        values = TGraph.EigenvectorCentrality(
            graph,
            mode="all" if not TGraph.IsDirected(graph) else "out",
            iterations=1000,
            tolerance=tolerance,
            key=key,
            mantissa=mantissa,
        )
        if normalize and values:
            max_value = max(abs(v) for v in values)
            if max_value > 0:
                values = [round(float(v) / max_value, mantissa) for v in values]
                if isinstance(graph, TGraph) and key is not None:
                    for idx, v_idx in enumerate(TGraph.ActiveVertexIndices(graph)):
                        TGraph._SetVertexValue(graph, v_idx, key, values[idx])
        return values

    @staticmethod
    def Triples(graph: "TGraph", **kwargs) -> List[Tuple[str, str, str]]:
        """Compatibility alias for OntologyTriples."""
        return TGraph.OntologyTriples(graph, **kwargs)

    @staticmethod
    def ExportToRDF(graph: "TGraph", path: str, overwrite: bool = True, silent: bool = False, **kwargs) -> Optional[str]:
        """Exports the input graph to an RDF/Turtle file."""
        import os
        if path is None:
            if not silent:
                print("TGraph.ExportToRDF - Error: The input path is None. Returning None.")
            return None
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("TGraph.ExportToRDF - Error: File exists and overwrite is False. Returning None.")
            return None
        return TGraph.ExportTTL(graph, path=path, silent=silent, **kwargs)

    @staticmethod
    def ExportToBOT(graph: "TGraph", path: str, overwrite: bool = False, silent: bool = False, **kwargs) -> Optional[str]:
        """Exports the input graph to a BOT/Turtle file."""
        import os
        if path is None:
            if not silent:
                print("TGraph.ExportToBOT - Error: The input path is None. Returning None.")
            return None
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("TGraph.ExportToBOT - Error: File exists and overwrite is False. Returning None.")
            return None
        try:
            data = TGraph.BOTString(graph, **kwargs)
            if data is None:
                return None
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            return path
        except Exception as exc:
            if not silent:
                print("TGraph.ExportToBOT - Error:", exc)
            return None

    @staticmethod
    def ExportToJSONLD(graph: "TGraph", path: str, indent: int = 2, sortKeys: bool = False, overwrite: bool = False, silent: bool = False, **kwargs) -> Optional[str]:
        """Exports the input graph to a JSON-LD file."""
        import os
        if path is None:
            if not silent:
                print("TGraph.ExportToJSONLD - Error: The input path is None. Returning None.")
            return None
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("TGraph.ExportToJSONLD - Error: File exists and overwrite is False. Returning None.")
            return None
        return TGraph.ExportJSONLD(graph, path=path, indent=indent, sortKeys=sortKeys, silent=silent, **kwargs)

    @staticmethod
    def ExportToJSON(
        graph: "TGraph",
        path: str,
        propertiesKey: str = "properties",
        verticesKey: str = "vertices",
        edgesKey: str = "edges",
        vertexLabelKey: str = "",
        edgeLabelKey: str = "",
        xKey: str = "x",
        yKey: str = "y",
        zKey: str = "z",
        indent: int = 4,
        sortKeys: bool = False,
        mantissa: int = 6,
        overwrite: bool = False,
    ) -> Optional[str]:
        """Exports the input graph to a JSON file."""
        import os
        if not isinstance(graph, TGraph) or path is None:
            return None
        if os.path.exists(path) and not overwrite:
            return None
        try:
            data = {
                propertiesKey: dict(graph._dictionary),
                verticesKey: [],
                edgesKey: [],
            }
            for v in graph._vertices:
                if not v.get("active", True):
                    continue
                d = dict(v.get("dictionary", {}))
                rec = dict(d)
                rec.setdefault("id", v.get("index"))
                c = TGraph.Coordinates(graph, v.get("index"), default=None)
                if c is not None:
                    rec[xKey] = round(float(c[0]), mantissa)
                    rec[yKey] = round(float(c[1]), mantissa)
                    rec[zKey] = round(float(c[2]), mantissa)
                if vertexLabelKey and vertexLabelKey in d:
                    rec["label"] = d.get(vertexLabelKey)
                data[verticesKey].append(rec)
            for e in graph._edges:
                if not e.get("active", True):
                    continue
                d = dict(e.get("dictionary", {}))
                rec = dict(d)
                rec.setdefault("id", e.get("index"))
                rec.setdefault("source", e.get("src"))
                rec.setdefault("target", e.get("dst"))
                rec.setdefault("directed", e.get("directed"))
                if edgeLabelKey and edgeLabelKey in d:
                    rec["label"] = d.get(edgeLabelKey)
                data[edgesKey].append(rec)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, sort_keys=sortKeys)
            return path
        except Exception:
            return None

    @staticmethod
    def ByJSONDictionary(
        jsonDictionary: Dict[str, Any],
        xKey: str = "x",
        yKey: str = "y",
        zKey: str = "z",
        vertexIDKey: str = None,
        edgeSourceKey: str = "source",
        edgeTargetKey: str = "target",
        edgeIDKey: str = None,
        graphPropsKey: str = "properties",
        verticesKey: str = "vertices",
        edgesKey: str = "edges",
        ontology: bool = True,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """Creates a TGraph from a JSON dictionary with graph, vertex, and edge records."""
        if not isinstance(jsonDictionary, dict):
            if not silent:
                print("TGraph.ByJSONDictionary - Error: The input jsonDictionary is not a dictionary. Returning None.")
            return None
        gd = jsonDictionary.get(graphPropsKey, {})
        g = TGraph(directed=False, allowSelfLoops=True, allowParallelEdges=True, dictionary=gd if isinstance(gd, dict) else {})
        id_to_index = {}
        vertices = jsonDictionary.get(verticesKey, []) or []
        for i, rec in enumerate(vertices):
            d = dict(rec) if isinstance(rec, dict) else {}
            vid = d.get(vertexIDKey, d.get("id", i)) if vertexIDKey is not None else d.get("id", i)
            idx = g.AddVertex(dictionary=d)
            id_to_index[vid] = idx
        for rec in jsonDictionary.get(edgesKey, []) or []:
            if not isinstance(rec, dict):
                continue
            src_id = rec.get(edgeSourceKey, rec.get("src", rec.get("source")))
            dst_id = rec.get(edgeTargetKey, rec.get("dst", rec.get("target")))
            src = id_to_index.get(src_id, src_id if isinstance(src_id, int) else None)
            dst = id_to_index.get(dst_id, dst_id if isinstance(dst_id, int) else None)
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            d = dict(rec)
            directed = bool(d.get("directed", g._directed))
            g.AddEdge(src, dst, directed=directed, dictionary=d)
            if directed:
                g._directed = True
        if ontology:
            try:
                TGraph.NormalizeOntologyDictionaries(g, silent=True)
            except Exception:
                pass
        return g

    @staticmethod
    def ByJSONFile(file: Any, **kwargs) -> Optional["TGraph"]:
        """Imports a TGraph from an open JSON file object or file-like object."""
        try:
            data = json.load(file)
        except Exception:
            return None
        return TGraph.ByJSONDictionary(data, **kwargs)

    @staticmethod
    def ByJSONPath(path: str, silent: bool = False, **kwargs) -> Optional["TGraph"]:
        """Imports a TGraph from a JSON file path."""
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return TGraph.ByJSONFile(f, silent=silent, **kwargs)
        except Exception as exc:
            if not silent:
                print("TGraph.ByJSONPath - Error:", exc)
            return None

    @staticmethod
    def ExportGraphToCSV(graph: "TGraph", path: str, graphLabel: Any, graphFeatures: str = "", overwrite: bool = False, **kwargs) -> Optional[bool]:
        """Compatibility wrapper for exporting a single graph to PyG-compatible CSV files."""
        if not isinstance(graph, TGraph):
            return None
        graph = TGraph.Copy(graph)
        graph._dictionary[kwargs.get("graphLabelKey", "label")] = graphLabel
        if isinstance(graphFeatures, str) and graphFeatures.strip() != "":
            values = [v.strip() for v in graphFeatures.split(",")]
            for i, value in enumerate(values):
                try:
                    graph._dictionary[f"feat_{i}"] = round(float(value), kwargs.get("mantissa", 6))
                except Exception:
                    graph._dictionary[f"feat_{i}"] = value
            kwargs.setdefault("graphFeaturesKeys", [f"feat_{i}" for i in range(len(values))])
        return TGraph.ExportToCSV(graph, path=path, overwrite=overwrite, **kwargs)

    @staticmethod
    def ExportToAdjacencyMatrixCSV(adjacencyMatrix: List[List[Any]], path: str) -> Optional[str]:
        """Exports an adjacency matrix to a CSV file."""
        if path is None or adjacencyMatrix is None:
            return None
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(adjacencyMatrix)
            return path
        except Exception:
            return None

    @staticmethod
    def ByAdjacencyMatrixCSVPath(path: str, dictionaries: list = None, ontology: bool = True, silent: bool = False) -> Optional["TGraph"]:
        """Creates a TGraph from an adjacency matrix CSV file."""
        if path is None:
            return None
        try:
            import csv
            matrix = []
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    parsed = []
                    for value in row:
                        value = value.strip()
                        if value == "":
                            parsed.append(0)
                        else:
                            try:
                                fv = float(value)
                                parsed.append(int(fv) if fv.is_integer() else fv)
                            except Exception:
                                parsed.append(value)
                    matrix.append(parsed)
            g = TGraph.ByAdjacencyMatrix(matrix, directed=False, dictionary={})
            if isinstance(dictionaries, list):
                for i, d in enumerate(dictionaries[:len(g._vertices)]):
                    if isinstance(d, dict):
                        g.SetVertexDictionary(i, d)
            return g
        except Exception as exc:
            if not silent:
                print("TGraph.ByAdjacencyMatrixCSVPath - Error:", exc)
            return None

    @staticmethod
    def MeshData(graph: "TGraph", activeOnly: bool = True) -> Dict[str, Any]:
        """Returns simple mesh-like data for graph vertices and edges."""
        if not isinstance(graph, TGraph):
            return {"vertices": [], "edges": []}
        coords = []
        index_map = {}
        for rec in graph._vertices:
            if activeOnly and not rec.get("active", True):
                continue
            idx = rec.get("index")
            c = TGraph.Coordinates(graph, idx, default=None)
            if c is None:
                c = [0.0, 0.0, 0.0]
            index_map[idx] = len(coords)
            coords.append(c)
        edges = []
        for e in graph._edges:
            if activeOnly and not e.get("active", True):
                continue
            src = e.get("src")
            dst = e.get("dst")
            if src in index_map and dst in index_map:
                edges.append([index_map[src], index_map[dst]])
        return {"vertices": coords, "edges": edges}

    @staticmethod
    def ByMeshData(vertices, edges, vertexDictionaries=None, edgeDictionaries=None, ontology: bool = True, tolerance: float = 0.0001) -> "TGraph":
        """Creates a TGraph from mesh-like vertices and edge index pairs."""
        g = TGraph(directed=False, allowSelfLoops=True, allowParallelEdges=True)
        vertexDictionaries = vertexDictionaries or []
        for i, v in enumerate(vertices or []):
            d = vertexDictionaries[i] if i < len(vertexDictionaries) and isinstance(vertexDictionaries[i], dict) else {}
            d = dict(d)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                d.setdefault("x", float(v[0])); d.setdefault("y", float(v[1])); d.setdefault("z", float(v[2]))
            g.AddVertex(dictionary=d)
        edgeDictionaries = edgeDictionaries or []
        for i, pair in enumerate(edges or []):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            d = edgeDictionaries[i] if i < len(edgeDictionaries) and isinstance(edgeDictionaries[i], dict) else {}
            g.AddEdge(int(pair[0]), int(pair[1]), dictionary=d)
        return g

    @staticmethod
    def LocalClusteringCoefficient(graph: "TGraph", vertices: list = None, key: str = "local_clustering_coefficient", mantissa: int = 6, silent: bool = False) -> List[float]:
        """Returns local clustering coefficients for the selected vertices."""
        if not isinstance(graph, TGraph):
            return []
        selected = [TGraph._as_index(v) for v in vertices] if vertices is not None else TGraph.ActiveVertexIndices(graph)
        selected = [v for v in selected if graph._validate_vertex_index(v)]
        result = []
        for v in selected:
            nbrs = TGraph.AdjacentIndices(graph, v, mode="all")
            nbrs = [u for u in nbrs if u != v]
            k = len(nbrs)
            if k < 2:
                coeff = 0.0
            else:
                links = 0
                nbr_set = set(nbrs)
                for i, a in enumerate(nbrs):
                    for b in nbrs[i+1:]:
                        if TGraph.HasEdge(graph, a, b, directed=False) or TGraph.HasEdge(graph, b, a, directed=True) or TGraph.HasEdge(graph, a, b, directed=True):
                            links += 1
                coeff = (2.0 * links) / float(k * (k - 1))
            coeff = round(coeff, mantissa)
            result.append(coeff)
            if key is not None:
                TGraph._SetVertexValue(graph, v, key, coeff)
        return result

    @staticmethod
    def AverageClusteringCoefficient(graph: "TGraph", mantissa: int = 6, silent: bool = False) -> float:
        """Returns the average clustering coefficient of the input graph."""
        values = TGraph.LocalClusteringCoefficient(graph, key=None, mantissa=mantissa, silent=silent)
        if not values:
            return 0.0
        return round(sum(values) / float(len(values)), mantissa)

    @staticmethod
    def GlobalClusteringCoefficient(graph: "TGraph") -> float:
        """Returns the global clustering coefficient / transitivity of the input graph."""
        if not isinstance(graph, TGraph):
            return 0.0
        vertices = TGraph.ActiveVertexIndices(graph)
        triangles3 = 0
        triples = 0
        for v in vertices:
            nbrs = [u for u in TGraph.AdjacentIndices(graph, v, mode="all") if u != v]
            k = len(nbrs)
            triples += k * (k - 1)
            nbr_set = set(nbrs)
            for a in nbrs:
                for b in nbrs:
                    if a != b and (TGraph.HasEdge(graph, a, b, directed=False) or TGraph.HasEdge(graph, a, b, directed=True)):
                        triangles3 += 1
        if triples == 0:
            return 0.0
        return triangles3 / float(triples)

    @staticmethod
    def AllPaths(graph: "TGraph", vertexA, vertexB, timeLimit=10, silent: bool = False) -> List[List[int]]:
        """Returns all simple paths between two vertices within a time limit."""
        import time
        if not isinstance(graph, TGraph):
            return []
        src = TGraph._as_index(vertexA)
        dst = TGraph._as_index(vertexB)
        if not graph._validate_vertex_index(src) or not graph._validate_vertex_index(dst):
            return []
        deadline = time.time() + max(0.0, float(timeLimit))
        result = []
        stack = [(src, [src], {src})]
        while stack:
            if time.time() > deadline:
                break
            u, path, seen = stack.pop()
            if u == dst:
                result.append(path)
                continue
            for v in reversed(TGraph.AdjacentIndices(graph, u, mode="out" if graph._directed else "all")):
                if v not in seen:
                    stack.append((v, path + [v], seen | {v}))
        return result

    @staticmethod
    def Path(graph: "TGraph", vertexA, vertexB, tolerance: float = 0.0001, silent: bool = False) -> List[int]:
        """Returns a shortest path between two vertices as a list of vertex indices."""
        return TGraph.ShortestPath(graph, TGraph._as_index(vertexA), TGraph._as_index(vertexB), mode="out" if TGraph.IsDirected(graph) else "all")

    @staticmethod
    def ShortestPathViaVertices(graph: "TGraph", startVertex, endVertex, vertices: list = None, tolerance: float = 0.0001, silent: bool = False) -> Optional[List[int]]:
        """Returns a head-to-tail shortest path that visits the requested intermediate vertices in order."""
        if not isinstance(graph, TGraph):
            return None
        sequence = [TGraph._as_index(startVertex)] + [TGraph._as_index(v) for v in (vertices or [])] + [TGraph._as_index(endVertex)]
        if any(not graph._validate_vertex_index(v) for v in sequence):
            return None
        mode = "out" if graph._directed else "all"
        final = []
        seen = set()
        for a, b in zip(sequence[:-1], sequence[1:]):
            segment = TGraph.ShortestPath(graph, a, b, mode=mode)
            if segment is None or len(segment) == 0:
                return None
            if not final:
                final.extend(segment)
                seen.update(segment[:-1])
            else:
                for v in segment[1:]:
                    if v in seen and v != b:
                        return None
                    final.append(v)
                    seen.add(v)
        return final

    @staticmethod
    def Tree(graph: "TGraph", vertex=None, mode: str = "all", silent: bool = False) -> Optional["TGraph"]:
        """Creates a tree graph rooted at the input vertex using a shortest-path tree."""
        if not isinstance(graph, TGraph):
            return None
        source = TGraph._as_index(vertex) if vertex is not None else (TGraph.ActiveVertexIndices(graph)[0] if TGraph.ActiveVertexIndices(graph) else None)
        if source is None or not graph._validate_vertex_index(source):
            return None
        tree = TGraph.ShortestPathTree(graph, source, mode=mode)
        if not isinstance(tree, dict):
            return None
        parents = tree.get("parent", {})
        g = TGraph(directed=True, allowSelfLoops=False, allowParallelEdges=False, dictionary={"root": source})
        active = TGraph.ActiveVertexIndices(graph)
        old_to_new = {}
        for old in active:
            old_to_new[old] = g.AddVertex(dictionary=dict(graph._vertices[old].get("dictionary", {})))
        for child, parent in parents.items():
            if parent is not None and parent in old_to_new and child in old_to_new:
                g.AddEdge(old_to_new[parent], old_to_new[child], directed=True, dictionary={"relationship": "tree_edge"})
        return g

    @staticmethod
    def WikiString(graph: "TGraph", vertexKey: str = "id", vertexLabelKey: str = "label", vertexTypeKey: str = "type", edgeKey: str = "predicate", titleKey: str = None, includeDictionaries: bool = True, includeBacklinks: bool = True, tolerance: float = 0.0001, silent: bool = False) -> str:
        """Returns a simple Obsidian-style markdown wiki representation of the graph."""
        if not isinstance(graph, TGraph):
            return ""
        lines = ["# TGraph Wiki", ""]
        for v in graph._vertices:
            if not v.get("active", True):
                continue
            d = v.get("dictionary", {})
            title = d.get(titleKey, d.get(vertexLabelKey, d.get(vertexKey, f"vertex_{v['index']}"))) if titleKey else d.get(vertexLabelKey, d.get(vertexKey, f"vertex_{v['index']}"))
            lines.extend([f"## {title}", ""])
            if includeDictionaries:
                for k, value in sorted(d.items()):
                    lines.append(f"- **{k}**: {value}")
            nbrs = TGraph.AdjacentIndices(graph, v["index"], mode="all")
            if nbrs:
                lines.append("- **Adjacent**: " + ", ".join(f"[[vertex_{n}]]" for n in nbrs))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def ExportToWiki(graph: "TGraph", path: str, overwrite: bool = True, silent: bool = False, **kwargs) -> Optional[str]:
        """Exports a simple Obsidian-style markdown wiki representation of the graph."""
        import os
        if path is None:
            return None
        if os.path.exists(path) and not overwrite:
            return None
        try:
            text = TGraph.WikiString(graph, **kwargs)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return path
        except Exception as exc:
            if not silent:
                print("TGraph.ExportToWiki - Error:", exc)
            return None
