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
        """
        Initializes a TGraph object.

        Parameters
        ----------
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.

        Returns
        -------
        None
            None.
        """
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
        """
        Returns a string representation of this TGraph.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The string representation of this TGraph.
        """
        kind = "directed" if self._directed else "bidirectional"
        return f"TGraph(vertices={TGraph.Order(self)}, edges={TGraph.Size(self)}, {kind})"

    @staticmethod
    def AABB(graph: "TGraph", pad: float = 0.0) -> Optional[Any]:
        """
        Returns the axis-aligned bounding box of the active vertices in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        pad : float , optional
            The amount by which to pad the bounding box. Default is 0.0.

        Returns
        -------
        dict or topologicpy.BVH.AABB or None
            The axis-aligned bounding box of the active graph vertices.
        """
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
    def AccelerationReport() -> Dict[str, Any]:
        """
        Returns a report indicating whether optional acceleration libraries are available.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary reporting optional acceleration library availability and versions.
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
        Returns an access graph derived from the input topology.

        Parameters
        ----------
        topology : Any
            The input Topologic topology.
        key : str , optional
            The dictionary key to use. Default is None.
        includeTypes : list , optional
            The topology or category types to include. Default is None.
        excludeTypes : list , optional
            The topology or category types to exclude. Default is None.
        viaSharedApertures : bool , optional
            If set to True, relationships through shared apertures are included. Default is
            False.
        toExteriorApertures : bool , optional
            If set to True, relationships to exterior apertures are included. Default is False.
        useInternalVertex : bool , optional
            If set to True, an internal vertex is used when deriving topology coordinates.
            Default is False.
        includeIsolatedVertices : bool , optional
            If set to True, isolated vertices are retained. Default is True.
        storeBREP : bool , optional
            If set to True, BREP strings are stored in dictionaries where possible. Default is
            False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
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
    def AccessibilityCentrality(graph: "TGraph", step: int = 2, normalize: bool = False,
                                key: str = "accessibility_centrality", colorKey: str = "ac_color",
                                colorScale: str = "viridis", mantissa: int = 6,
                                tolerance: float = 0.0001, silent: bool = False) -> List[float]:
        """
        Computes accessibility centrality values for the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        step : int , optional
            The input step value. Default is 2.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is False.
        key : str , optional
            The dictionary key to use. Default is 'accessibility_centrality'.
        colorKey : str , optional
            The dictionary key under which computed color values are stored. Default is
            'ac_color'.
        colorScale : str , optional
            The Plotly color scale to use. Default is 'viridis'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[float]
            The resulting accessibility centrality list.
        """
        if not isinstance(graph, TGraph):
            return []
        vertices = TGraph._ActiveVertexIndices(graph)
        n = len(vertices)
        if n == 0:
            return []
        pos = {v:i for i,v in enumerate(vertices)}
        P = [[0.0 for _ in range(n)] for _ in range(n)]
        for v in vertices:
            ns = [u for u in TGraph.AdjacentIndices(graph, v, mode="all") if u in pos]
            if ns:
                w = 1.0/len(ns)
                for u in ns:
                    P[pos[v]][pos[u]] += w
        def mm(A,B):
            m=len(A); C=[[0.0]*m for _ in range(m)]
            for i in range(m):
                for k in range(m):
                    if A[i][k]:
                        aik=A[i][k]
                        for j in range(m):
                            C[i][j]+=aik*B[k][j]
            return C
        M = [[float(i==j) for j in range(n)] for i in range(n)]
        for _ in range(max(0,int(step))):
            M = mm(M,P)
        vals=[]
        import math as _math
        for row in M:
            probs=[p for p in row if p>0]
            entropy=-sum(p*_math.log(p) for p in probs)
            vals.append(_math.exp(entropy))
        if normalize and vals:
            mn,mx=min(vals),max(vals)
            vals=[0.0 if abs(mx-mn)<=1e-12 else (v-mn)/(mx-mn) for v in vals]
        vals=[round(v,mantissa) for v in vals]
        for idx,value in zip(vertices,vals):
            graph._vertices[idx]["dictionary"][key]=value
        return vals

    @staticmethod
    def ActiveEdgeIndices(graph: "TGraph") -> List[int]:
        """
        Returns the indices of the active edges in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[int]
            The resulting active edge indices list.
        """
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        return [e["edge_index"] for e in c["edges"]]

    @staticmethod
    def _ActiveEdges(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Returns the active edge records of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting active edges list.
        """
        if not isinstance(graph, TGraph):
            return []
        return [e for e in graph._edges if e.get("active", True)]

    @staticmethod
    def _ActiveVertexIndices(graph: "TGraph") -> List[int]:
        """
        Returns the active vertex indices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[int]
            The resulting active vertex indices list.
        """
        if not isinstance(graph, TGraph):
            return []
        return [v["index"] for v in graph._vertices if v.get("active", True)]

    @staticmethod
    def ActiveVertexIndices(graph: "TGraph") -> List[int]:
        """
        Returns the indices of the active vertices in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[int]
            The resulting active vertex indices list.
        """
        c = TGraph.Compile(graph)
        return list(c["vertices"]) if isinstance(c, dict) else []

    def AddEdge(self, src: Any, dst: Any = None, directed: Optional[bool] = None,
                dictionary: Optional[Dict[str, Any]] = None, representation: Any = None,
                transferVertexDictionaries: bool = False, transferEdgeDictionaries: bool = False,
                tolerance: float = 0.0001, silent: bool = False) -> Optional[int]:
        """
        Adds an edge to this TGraph and returns its index.

        Parameters
        ----------
        src : Any
            The source vertex index.
        dst : Any , optional
            The destination vertex index. Default is None.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        representation : Any , optional
            The optional representation object to store with the graph record. Default is None.
        transferVertexDictionaries : bool , optional
            If set to True, vertex dictionaries are transferred to created Topologic vertices.
            Default is False.
        transferEdgeDictionaries : bool , optional
            If set to True, edge dictionaries are transferred to created Topologic edges.
            Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int or None
            The index of the created edge, or None if no edge was created.
        """
        edge_item = src if dst is None else None
        if edge_item is not None and not isinstance(edge_item, int):
            d = dict(dictionary) if isinstance(dictionary, dict) else {}
            if transferEdgeDictionaries and edge_item is not None:
                d.update(TGraph._TopologyDictionaryToPython(edge_item))
            try:
                from topologicpy.Edge import Edge
                sv = Edge.StartVertex(edge_item)
                ev = Edge.EndVertex(edge_item)
            except Exception:
                sv = ev = None
            if sv is not None and ev is not None:
                sidx = None; didx = None
                try:
                    from topologicpy.Topology import Topology
                    from topologicpy.Vertex import Vertex
                    for rec in self._vertices:
                        if not rec.get("active", True):
                            continue
                        rep = rec.get("representation", None)
                        if rep is sv or (rep is not None and Topology.IsSame(rep, sv)):
                            sidx = rec.get("index")
                        if rep is ev or (rep is not None and Topology.IsSame(rep, ev)):
                            didx = rec.get("index")
                    if sidx is None:
                        sidx = self.AddVertex(dictionary=TGraph._TopologyDictionaryToPython(sv), representation=sv)
                    if didx is None:
                        didx = self.AddVertex(dictionary=TGraph._TopologyDictionaryToPython(ev), representation=ev)
                except Exception:
                    pass
                if isinstance(sidx, int) and isinstance(didx, int):
                    src = sidx; dst = didx; representation = edge_item; dictionary = d
        if isinstance(src, (list, tuple)) and dst is None and len(src) >= 2:
            src, dst = src[0], src[1]
        src_idx = TGraph.VertexIndex(self, src)
        dst_idx = TGraph.VertexIndex(self, dst)
        if src_idx is None or dst_idx is None:
            return None
        if src_idx == dst_idx and not self._allow_self_loops:
            return None
        edge_directed = self._directed if directed is None else bool(directed)
        key = self._edge_key(src_idx, dst_idx, edge_directed)
        if not self._allow_parallel_edges and key in self._edge_lookup:
            return None
        index = len(self._edges)
        d = dict(dictionary) if isinstance(dictionary, dict) else TGraph._DictionaryToPython(dictionary)
        d.update({"index": index, "src": src_idx, "dst": dst_idx, "directed": edge_directed, "active": True})
        record = {"index": index, "src": src_idx, "dst": dst_idx, "directed": edge_directed,
                  "dictionary": d, "representation": representation, "active": True}
        self._edges.append(record)
        self._register_edge_adjacency(index, src_idx, dst_idx, edge_directed)
        self._invalidate_cache()
        return index

    def AddEdgeByIndex(self, index: Union[List[int], Tuple[int, int]], dictionary: Optional[Dict[str, Any]] = None,
                       directed: Optional[bool] = None, representation: Any = None, silent: bool = False) -> Optional[int]:
        """
        Adds an edge to this TGraph using a pair of vertex indices and returns the edge index.

        Parameters
        ----------
        index : Union[List[int], Tuple[int, int]]
            The input index.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.
        representation : Any , optional
            The optional representation object to store with the graph record. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int or None
            The index of the created edge, or None if no edge was created.
        """
        if not isinstance(index, (list, tuple)) or len(index) < 2:
            return None
        return self.AddEdge(index[0], index[1], directed=directed, dictionary=dictionary, representation=representation, silent=silent)

    @staticmethod
    def _AddRelationship(graph: "TGraph", src: int, dst: int, relationship: str, category: Any = None,
                               source: Any = None, dictionary: Optional[Dict[str, Any]] = None,
                               directed: Optional[bool] = None) -> Optional[int]:
        """
        Adds an internal relationship edge between two vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        src : int
            The source vertex index.
        dst : int
            The destination vertex index.
        relationship : str
            The input relationship value.
        category : Any , optional
            The ontology category value. Default is None.
        source : Any , optional
            The input source vertex, vertex index, or source identifier. Default is None.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.

        Returns
        -------
        Optional[int]
            The resulting add relationship index or count.
        """
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
    def _AddTopologyVertex(graph: "TGraph", topology: Any, category: Any = None, label: Any = None,
                                 storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001,
                                 useInternalVertex: bool = False, extra: Optional[Dict[str, Any]] = None) -> int:
        """
        Adds a topology-derived vertex to the input TGraph and returns its index.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        topology : Any
            The input Topologic topology.
        category : Any , optional
            The ontology category value. Default is None.
        label : Any , optional
            The label value. Default is None.
        storeBREP : bool , optional
            If set to True, BREP strings are stored in dictionaries where possible. Default is
            False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        useInternalVertex : bool , optional
            If set to True, an internal vertex is used when deriving topology coordinates.
            Default is False.
        extra : Optional[Dict[str, Any]] , optional
            The input extra value. Default is None.

        Returns
        -------
        int
            The resulting add topology vertex index or count.
        """
        d = TGraph._TopologyDictionary(topology, storeBREP=storeBREP, mantissa=mantissa, tolerance=tolerance, useInternalVertex=useInternalVertex)
        if category is not None:
            d["category"] = category
        if label is not None:
            d.setdefault("label", label)
        if isinstance(extra, dict):
            d.update(extra)
        return graph.AddVertex(dictionary=d, representation=topology)

    def AddVertex(self, dictionary: Optional[Dict[str, Any]] = None, representation: Any = None,
                  tolerance: float = 0.0001, silent: bool = False) -> int:
        """
        Adds a vertex to this TGraph and returns its index.

        Parameters
        ----------
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        representation : Any , optional
            The optional representation object to store with the graph record. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int
            The index of the created vertex.
        """
        item = dictionary
        d = None
        rep = representation
        if isinstance(item, dict):
            d = dict(item)
        elif item is not None:
            rep = item if representation is None else representation
            d = TGraph._TopologyDictionaryToPython(item)
            try:
                from topologicpy.Topology import Topology
                from topologicpy.Vertex import Vertex
                if Topology.IsInstance(item, "Vertex"):
                    c = Vertex.Coordinates(item)
                    if c and len(c) >= 3:
                        d.setdefault("x", float(c[0])); d.setdefault("y", float(c[1])); d.setdefault("z", float(c[2]))
            except Exception:
                pass
        else:
            d = {}
        index = len(self._vertices)
        d = dict(d) if isinstance(d, dict) else {}
        d["index"] = index
        d.setdefault("active", True)
        record = {"index": index, "dictionary": d, "representation": rep, "active": True}
        self._vertices.append(record)
        self._out_edges.setdefault(index, set())
        self._in_edges.setdefault(index, set())
        self._incident_edges.setdefault(index, set())
        self._invalidate_cache()
        return index

    @staticmethod
    def AddVertexByData(graph: "TGraph", dictionary: Optional[Dict[str, Any]] = None,
                        x: float = None, y: float = None, z: float = None,
                        silent: bool = False) -> Optional["TGraph"]:
        """
        Adds a vertex with coordinate data to the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        x : float , optional
            The X coordinate. Default is None.
        y : float , optional
            The Y coordinate. Default is None.
        z : float , optional
            The Z coordinate. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None
        d = dict(dictionary) if isinstance(dictionary, dict) else TGraph._DictionaryToPython(dictionary)
        import random as _random
        if x is None: x = _random.uniform(0, 100)
        if y is None: y = _random.uniform(0, 100)
        if z is None: z = _random.uniform(0, 100)
        d.update({"x": float(x), "y": float(y), "z": float(z)})
        rep = None
        try:
            from topologicpy.Vertex import Vertex
            rep = Vertex.ByCoordinates(float(x), float(y), float(z))
        except Exception:
            rep = None
        graph.AddVertex(dictionary=d, representation=rep)
        return graph

    def AddVertices(self, vertices: Iterable[Any], dictionaries: Optional[List[Dict[str, Any]]] = None,
                    tolerance: float = 0.0001, silent: bool = False) -> List[int]:
        """
        Adds multiple vertices to this TGraph and returns their indices.

        Parameters
        ----------
        vertices : Iterable[Any]
            The input vertices or vertex indices.
        dictionaries : Optional[List[Dict[str, Any]]] , optional
            The input list of dictionaries. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The indices of the created vertices.
        """
        result = []
        dictionaries = dictionaries or []
        for i, item in enumerate(vertices or []):
            if i < len(dictionaries) and isinstance(dictionaries[i], dict):
                d = dict(dictionaries[i])
                rep = item
                result.append(self.AddVertex(dictionary=d, representation=rep, tolerance=tolerance, silent=silent))
            else:
                result.append(self.AddVertex(dictionary=item, representation=None, tolerance=tolerance, silent=silent))
        return result

    @staticmethod
    def AdjacencyDictionary(graph: "TGraph", vertexLabelKey: str = "label", includeWeights: bool = False,
                            edgeKey: str = "weight") -> Dict[Any, List[Any]]:
        """
        Returns an adjacency dictionary representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        includeWeights : bool , optional
            If set to True, edge weights are included in the output. Default is False.
        edgeKey : str , optional
            The edge dictionary key to use. Default is 'weight'.

        Returns
        -------
        Dict[Any, List[Any]]
            The resulting adjacency dictionary list.
        """
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
    def AdjacencyList(graph: "TGraph", mode: str = "out") -> List[List[int]]:
        """
        Returns an adjacency list representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.

        Returns
        -------
        List[List[int]]
            The resulting adjacency list list.
        """
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return []
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        vertices = c["vertices"]
        return [[vertices[j] for j in row] for row in adj]

    @staticmethod
    def AdjacencyMatrix(graph: "TGraph", vertexKey: str = None, reverse: bool = False,
                        edgeKeyFwd: str = None, edgeKeyBwd: str = None, bidirKey: str = None,
                        bidirectional: bool = None, useEdgeIndex: bool = False,
                        useEdgeLength: bool = False, mantissa: int = 6, tolerance: float = 0.0001) -> List[List[Any]]:
        """
        Returns an adjacency matrix representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexKey : str , optional
            The vertex dictionary key to use. Default is None.
        reverse : bool , optional
            If set to True, the output ordering is reversed. Default is False.
        edgeKeyFwd : str , optional
            The input edge key fwd value. Default is None.
        edgeKeyBwd : str , optional
            The input edge key bwd value. Default is None.
        bidirKey : str , optional
            The dictionary key to use. Default is None.
        bidirectional : bool , optional
            The input bidirectional value. Default is None.
        useEdgeIndex : bool , optional
            If set to True, the corresponding option is enabled. Default is False.
        useEdgeLength : bool , optional
            If set to True, the corresponding option is enabled. Default is False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        List[List[Any]]
            The resulting adjacency matrix list.
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

    @staticmethod
    def _AdjacencyMatrixFastArraydjacencyCompact(graph: "TGraph", mode: str = "out", weightKey: str = "weight") -> Tuple[Optional[Dict[str, Any]], List[List[int]]]:
        """
        Returns the compiled graph data and compact adjacency array for the requested adjacency mode.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        weightKey : str , optional
            The edge dictionary key to use as a weight. Default is 'weight'.

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], List[List[int]]]
            The resulting adjacency matrix fast arraydjacency compact list.
        """
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
    def AdjacencyMatrixFigure(
        graph: "TGraph",
        vertexKey: str = None,
        showZero: bool = False,
        zeroChar: str = "·",
        zeroColor: str = "rgba(0,0,0,0)",
        valueColor: str = "rgba(0,0,0,0.05)",
        diagonalHighlight: bool = True,
        diagonalColor: str = "rgba(0,0,0,0.08)",
        title: str = None,
        cellSize: int = 24,
        fontFamily: str = "Arial",
        fontSize: int = 12,
        fontColor: str = "black",
        backgroundColor: str = "white",
        headerColor: str = "rgba(230,230,230,1)",
        reverse: bool = False,
        edgeKeyFwd: str = None,
        edgeKeyBwd: str = None,
        bidirKey: str = None,
        bidirectional: bool = True,
        useEdgeIndex: bool = False,
        useEdgeLength: bool = False,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns a Plotly table figure representing the adjacency matrix of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexKey : str , optional
            The vertex dictionary key to use. Default is None.
        showZero : bool , optional
            If set to True, show zero are shown. Default is False.
        zeroChar : str , optional
            The input zero char value. Default is '·'.
        zeroColor : str , optional
            The color value to use. Default is 'rgba(0,0,0,0)'.
        valueColor : str , optional
            The color value to use. Default is 'rgba(0,0,0,0.05)'.
        diagonalHighlight : bool , optional
            The input diagonal highlight value. Default is True.
        diagonalColor : str , optional
            The color value to use. Default is 'rgba(0,0,0,0.08)'.
        title : str , optional
            The optional figure title. Default is None.
        cellSize : int , optional
            The input cell size value. Default is 24.
        fontFamily : str , optional
            The input font family value. Default is 'Arial'.
        fontSize : int , optional
            The input font size value. Default is 12.
        fontColor : str , optional
            The color value to use. Default is 'black'.
        backgroundColor : str , optional
            The color value to use. Default is 'white'.
        headerColor : str , optional
            The color value to use. Default is 'rgba(230,230,230,1)'.
        reverse : bool , optional
            If set to True, the output ordering is reversed. Default is False.
        edgeKeyFwd : str , optional
            The input edge key fwd value. Default is None.
        edgeKeyBwd : str , optional
            The input edge key bwd value. Default is None.
        bidirKey : str , optional
            The dictionary key to use. Default is None.
        bidirectional : bool , optional
            The input bidirectional value. Default is True.
        useEdgeIndex : bool , optional
            If set to True, the corresponding option is enabled. Default is False.
        useEdgeLength : bool , optional
            If set to True, the corresponding option is enabled. Default is False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Any
            The resulting adjacency matrix figure object or value.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.AdjacencyMatrixFigure - Error: The input graph is not a valid TGraph. Returning None.")
            return None
        try:
            import plotly.graph_objects as go
        except Exception:
            if not silent:
                print("TGraph.AdjacencyMatrixFigure - Error: Plotly is not installed. Returning None.")
            return None

        matrix = TGraph.AdjacencyMatrix(
            graph,
            vertexKey=vertexKey,
            reverse=reverse,
            edgeKeyFwd=edgeKeyFwd,
            edgeKeyBwd=edgeKeyBwd,
            bidirKey=bidirKey,
            bidirectional=bidirectional,
            useEdgeIndex=useEdgeIndex,
            useEdgeLength=useEdgeLength,
            mantissa=mantissa,
            tolerance=tolerance,
        )
        vertices = TGraph.Vertices(graph, asTopologic=False, activeOnly=True)
        labels = []
        for i, v in enumerate(vertices):
            d = v.get("dictionary", {}) if isinstance(v, dict) else {}
            label = d.get(vertexKey, None) if vertexKey is not None else None
            labels.append(str(label if label is not None else d.get("label", d.get("index", i))))

        display = []
        fills = []
        for r, row in enumerate(matrix):
            display_row = []
            fill_row = []
            for c, value in enumerate(row):
                is_zero = value in [0, 0.0, None, "", False]
                display_row.append(zeroChar if is_zero and not showZero else value)
                if diagonalHighlight and r == c:
                    fill_row.append(diagonalColor)
                else:
                    fill_row.append(zeroColor if is_zero else valueColor)
            display.append(display_row)
            fills.append(fill_row)

        # Plotly Table expects columns, not rows.
        cell_values = [[display[r][c] for r in range(len(display))] for c in range(len(display[0]) if display else 0)]
        cell_fills = [[fills[r][c] for r in range(len(fills))] for c in range(len(fills[0]) if fills else 0)]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=labels,
                        fill_color=headerColor,
                        align="center",
                        font=dict(family=fontFamily, size=fontSize, color=fontColor),
                    ),
                    cells=dict(
                        values=cell_values,
                        fill_color=cell_fills,
                        align="center",
                        font=dict(family=fontFamily, size=fontSize, color=fontColor),
                        height=max(12, int(cellSize)),
                    ),
                )
            ]
        )
        fig.update_layout(
            title=title,
            paper_bgcolor=backgroundColor,
            plot_bgcolor=backgroundColor,
            font=dict(family=fontFamily, size=fontSize, color=fontColor),
            margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        )
        return fig

    @staticmethod
    def AdjacentEdges(graph: "TGraph", edge: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns the active edges adjacent to the input edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edge : Union[int, Dict[str, Any]]
            The input edge, edge index, or edge record.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting adjacent edges list.
        """
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
    def AdjacentIndices(graph: "TGraph", index: int, mode: str = "out") -> List[int]:
        """
        Returns the indices of vertices adjacent to the input vertex index.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        index : int
            The input index.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.

        Returns
        -------
        List[int]
            The resulting adjacent indices list.
        """
        c = TGraph.Compile(graph)
        if not isinstance(c, dict) or index not in c["position"]:
            return []
        p = c["position"][index]
        adj = TGraph.CompiledAdjacency(graph, mode=mode)
        return [c["vertices"][q] for q in adj[p]]

    @staticmethod
    def AdjacentVertices(graph: "TGraph", vertex: Union[int, Dict[str, Any]], mode: str = "out") -> List[Dict[str, Any]]:
        """
        Returns the vertices adjacent to the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting adjacent vertices list.
        """
        idx = TGraph._as_index(vertex)
        return [TGraph.Vertex(graph, i) for i in TGraph.AdjacentIndices(graph, idx, mode=mode)] if isinstance(graph, TGraph) else []

    @staticmethod
    def AllPaths(graph: "TGraph", vertexA, vertexB, timeLimit=10, silent: bool = False) -> List[List[int]]:
        """
        Returns simple paths between two vertices of the input TGraph within a time limit.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexA : Any
            The first input vertex or vertex index.
        vertexB : Any
            The second input vertex or vertex index.
        timeLimit : Any , optional
            The maximum time, in seconds, allowed for the search. Default is 10.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[List[int]]
            The resulting all paths list.
        """
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
    def AnnotateIFC(
        graph: "TGraph",
        ifcClass: Optional[str] = None,
        ifcGUID: Optional[str] = None,
        ifcName: Optional[str] = None,
        source: Optional[str] = None,
        element: str = "graph",
        index: Optional[int] = None,
    ) -> Optional["TGraph"]:
        """
        Annotates a graph, vertex, or edge dictionary with IFC metadata.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        ifcClass : Optional[str] , optional
            The IFC class value. Default is None.
        ifcGUID : Optional[str] , optional
            The IFC GlobalId value. Default is None.
        ifcName : Optional[str] , optional
            The IFC name value. Default is None.
        source : Optional[str] , optional
            The input source vertex, vertex index, or source identifier. Default is None.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
        """
        Annotates a graph, vertex, or edge dictionary with ontology metadata.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        ontologyClass : Optional[str] , optional
            The ontology class value. Default is None.
        category : Optional[str] , optional
            The ontology category value. Default is None.
        label : Any , optional
            The label value. Default is None.
        uri : Optional[str] , optional
            The URI value. Default is None.
        source : Any , optional
            The input source vertex, vertex index, or source identifier. Default is None.
        derivedFrom : Any , optional
            The provenance value identifying the source object. Default is None.
        generatedBy : Any , optional
            The provenance value identifying the generating method. Default is None.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
    def _as_index(vertex: Union[int, Dict[str, Any]]) -> Optional[int]:
        """
        Resolves a vertex or edge record to its stored integer index.

        Parameters
        ----------
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.

        Returns
        -------
        Optional[int]
            The resulting as index index or count.
        """
        if isinstance(vertex, int):
            return vertex
        if isinstance(vertex, dict):
            idx = vertex.get("index")
            return idx if isinstance(idx, int) else None
        return None

    @staticmethod
    def AverageClusteringCoefficient(graph: "TGraph", mantissa: int = 6, silent: bool = False) -> float:
        """
        Returns the average local clustering coefficient of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The resulting average clustering coefficient value.
        """
        values = TGraph.LocalClusteringCoefficient(graph, key=None, mantissa=mantissa, silent=silent)
        if not values:
            return 0.0
        return round(float(sum(values)) / float(len(values)), mantissa)

    @staticmethod
    def BetweennessCentrality(
        graph: "TGraph",
        weightKey: str = None,
        normalize: bool = False,
        nxCompatible: bool = True,
        useEdges: bool = False,
        edgeKey: str = None,
        angular: bool = False,
        angularWeightKey: str = "angular_weight",
        key: str = "betweenness_centrality",
        colorKey: str = "bc_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 0.001,
        silent: bool = False,
    ) -> Optional[List[float]]:
        """
        Computes the betweenness centrality of the input TGraph and stores the result
        in the dictionary of each vertex or edge.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        weightKey : str , optional
            If set to None, each edge is assumed to have a weight of 1. If set to
            "length" or "distance", the geometric length of each edge is used as its
            weight. If set to any other value, the value associated with that key in
            each edge dictionary is used as the edge weight. Default is None.
        normalize : bool , optional
            If set to True, the values are normalized between 0 and 1. Default is False.
        nxCompatible : bool , optional
            If set to True, NetworkX-style normalization is applied and the normalize
            input is ignored. Default is True.
        useEdges : bool , optional
            If set to True, the calculation uses the edges rather than the vertices.
            Default is False.
        edgeKey : str , optional
            If not None, the value associated with that key in each edge dictionary is
            used to bundle the edges into one entity for the calculation. Otherwise,
            each edge segment is assumed to be an independent entity. Default is None.
        angular : bool , optional
            If set to True, the calculation uses angular weights between adjacent edge
            segments. This option is valid only when useEdges is set to True.
            Default is False.
        angularWeightKey : str , optional
            The dictionary key under which to store the computed angular weight on the
            line graph edges. Default is "angular_weight".
        key : str , optional
            The desired dictionary key name under which to store the calculated value.
            Default is "betweenness_centrality".
        colorKey : str , optional
            The desired dictionary key name under which to store the calculated color.
            Default is "bc_color".
        colorScale : str , optional
            The desired color scale name to use for colors. Default is "viridis".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The list of centralities in the order matching the vertices or edges as requested.
        """

        import heapq
        import math
        import numbers
        from collections import deque, defaultdict

        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.BetweennessCentrality - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        if angular and not useEdges:
            if not silent:
                print("TGraph.BetweennessCentrality - Error: The angular option is valid only when useEdges is set to True. Returning None.")
            return None

        if angular and edgeKey is not None:
            if not silent:
                print("TGraph.BetweennessCentrality - Error: The angular option is not compatible with edge bundling through edgeKey. Returning None.")
            return None

        # ---------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------

        def _active_vertex_indices(g):
            return [v.get("index") for v in g._vertices if v.get("active", True)]

        def _active_edge_records(g):
            return [e for e in g._edges if e.get("active", True)]

        def _unwrap(x):
            if isinstance(x, list) and len(x) == 1:
                return x[0]
            return x

        def _as_float(x, default=0.0):
            x = _unwrap(x)
            if isinstance(x, numbers.Number):
                return float(x)
            try:
                return float(x)
            except Exception:
                return float(default)

        def _round(x):
            if mantissa is None or mantissa < 0:
                return float(x)
            return round(float(x), mantissa)

        def _normalize_flat(vals):
            if not vals:
                return []
            xs = [float(v) for v in vals]
            mn = min(xs)
            mx = max(xs)
            eps = tolerance if tolerance and tolerance > 0 else 1e-12
            if abs(mx - mn) < eps:
                return [0.0 for _ in xs]
            return [(x - mn) / (mx - mn) for x in xs]

        def _color(value, minValue, maxValue):
            try:
                from topologicpy.Color import Color
                return Color.AnyToHex(
                    Color.ByValueInRange(
                        value,
                        minValue=minValue,
                        maxValue=maxValue,
                        colorScale=colorScale,
                    )
                )
            except Exception:
                return None

        def _color_range(vals, unit_range=False):
            if not vals:
                return 0.0, 1.0
            if unit_range:
                return 0.0, 1.0
            mn = min(float(v) for v in vals)
            mx = max(float(v) for v in vals)
            eps = tolerance if tolerance and tolerance > 0 else 1e-12
            if abs(mx - mn) < eps:
                mx = mn + eps
            return mn, mx

        def _apply_values_to_vertices(g, vertexIndices, values, unit_range=False):
            if not vertexIndices or not values:
                return

            mn, mx = _color_range(values, unit_range=unit_range)

            for vertexIndex, value in zip(vertexIndices, values):
                if not g._validate_vertex_index(vertexIndex, active=False):
                    continue

                d = g._vertices[vertexIndex].setdefault("dictionary", {})
                v = float(value)

                if key is not None:
                    d[key] = v

                if colorKey is not None:
                    c = _color(v, mn, mx)
                    if c is not None:
                        d[colorKey] = c

        def _apply_values_to_edges(g, edgeRecords, values, unit_range=False):
            if not edgeRecords or not values:
                return

            mn, mx = _color_range(values, unit_range=unit_range)

            for edgeRecord, value in zip(edgeRecords, values):
                edgeIndex = edgeRecord.get("index", None)

                if not g._validate_edge_index(edgeIndex, active=False):
                    continue

                d = g._edges[edgeIndex].setdefault("dictionary", {})
                v = float(value)

                if key is not None:
                    d[key] = v

                if colorKey is not None:
                    c = _color(v, mn, mx)
                    if c is not None:
                        d[colorKey] = c

        def _edge_length(g, edgeRecord):
            src = edgeRecord.get("src", None)
            dst = edgeRecord.get("dst", None)

            c1 = TGraph.Coordinates(g, src, default=None)
            c2 = TGraph.Coordinates(g, dst, default=None)

            if c1 is None or c2 is None:
                return 1.0

            try:
                return float(math.dist(c1, c2))
            except Exception:
                return 1.0

        def _edge_weight(g, edgeRecord):
            if weightKey is None:
                return 1.0

            if isinstance(weightKey, str):
                wl = weightKey.lower()
                if ("len" in wl) or ("dis" in wl):
                    return _edge_length(g, edgeRecord)

            d = edgeRecord.get("dictionary", {})
            if isinstance(d, dict):
                return _as_float(d.get(weightKey, None), default=1.0)

            return 1.0

        def _vector_from_shared_vertex(g, edgeRecord, sharedVertexIndex):
            src = edgeRecord.get("src", None)
            dst = edgeRecord.get("dst", None)

            c_shared = TGraph.Coordinates(g, sharedVertexIndex, default=None)

            if c_shared is None:
                return None

            if src == sharedVertexIndex:
                c_other = TGraph.Coordinates(g, dst, default=None)
            elif dst == sharedVertexIndex:
                c_other = TGraph.Coordinates(g, src, default=None)
            else:
                return None

            if c_other is None:
                return None

            return [
                float(c_other[0]) - float(c_shared[0]),
                float(c_other[1]) - float(c_shared[1]),
                float(c_other[2]) - float(c_shared[2]),
            ]

        def _angle_between_vectors(a, b):
            if a is None or b is None:
                return None

            la = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
            lb = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])

            if la <= 0.0 or lb <= 0.0:
                return None

            dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb)
            dot = max(-1.0, min(1.0, dot))

            return math.degrees(math.acos(dot))

        def _shared_vertex_index(edgeA, edgeB):
            a_src = edgeA.get("src", None)
            a_dst = edgeA.get("dst", None)
            b_src = edgeB.get("src", None)
            b_dst = edgeB.get("dst", None)

            if a_src == b_src or a_src == b_dst:
                return a_src

            if a_dst == b_src or a_dst == b_dst:
                return a_dst

            return None

        def _set_angular_weights_on_line_graph(inputGraph, lineGraph):
            originalEdges = _active_edge_records(inputGraph)
            idToEdge = {}

            for i, edgeRecord in enumerate(originalEdges):
                edgeIndex = edgeRecord.get("index", None)
                if edgeIndex is None:
                    continue

                inputGraph._edges[edgeIndex].setdefault("dictionary", {})["u_edge_id"] = i
                idToEdge[i] = inputGraph._edges[edgeIndex]

            for lineEdgeRecord in _active_edge_records(lineGraph):
                src = lineEdgeRecord.get("src", None)
                dst = lineEdgeRecord.get("dst", None)

                if src is None or dst is None:
                    continue

                if not lineGraph._validate_vertex_index(src, active=False):
                    continue

                if not lineGraph._validate_vertex_index(dst, active=False):
                    continue

                dsrc = lineGraph._vertices[src].get("dictionary", {})
                ddst = lineGraph._vertices[dst].get("dictionary", {})

                idA = dsrc.get("u_edge_id", None)
                idB = ddst.get("u_edge_id", None)

                edgeA = idToEdge.get(idA, None)
                edgeB = idToEdge.get(idB, None)

                if edgeA is None or edgeB is None:
                    continue

                shared = _shared_vertex_index(edgeA, edgeB)

                if shared is None:
                    continue

                vecA = _vector_from_shared_vertex(inputGraph, edgeA, shared)
                vecB = _vector_from_shared_vertex(inputGraph, edgeB, shared)

                angle = _angle_between_vectors(vecA, vecB)

                if angle is None:
                    continue

                w = _round(float(angle) / 90.0)
                edgeIndex = lineEdgeRecord.get("index", None)

                if lineGraph._validate_edge_index(edgeIndex, active=False):
                    lineGraph._edges[edgeIndex].setdefault("dictionary", {})[angularWeightKey] = w

        # ---------------------------------------------------------------------
        # Angular edge mode
        # ---------------------------------------------------------------------

        if useEdges and angular:
            edgeRecords = _active_edge_records(graph)

            if not edgeRecords:
                return []

            for i, edgeRecord in enumerate(edgeRecords):
                edgeIndex = edgeRecord.get("index", None)
                if graph._validate_edge_index(edgeIndex, active=False):
                    graph._edges[edgeIndex].setdefault("dictionary", {})["u_edge_id"] = i

            try:
                lineGraph = TGraph.LineGraph(graph, transferEdgeDictionaries=True)
            except TypeError:
                try:
                    lineGraph = TGraph.LineGraph(graph)
                except Exception:
                    lineGraph = None
            except Exception:
                lineGraph = None

            if not isinstance(lineGraph, TGraph):
                if not silent:
                    print("TGraph.BetweennessCentrality - Error: Could not create a line graph. Returning None.")
                return None

            _set_angular_weights_on_line_graph(graph, lineGraph)

            _ = TGraph.BetweennessCentrality(
                lineGraph,
                weightKey=angularWeightKey,
                normalize=normalize,
                nxCompatible=nxCompatible,
                useEdges=False,
                edgeKey=None,
                angular=False,
                angularWeightKey=angularWeightKey,
                key=key,
                colorKey=colorKey,
                colorScale=colorScale,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=silent,
            )

            idToValue = {}

            for lineVertex in _active_vertex_indices(lineGraph):
                d = lineGraph._vertices[lineVertex].get("dictionary", {})
                eid = d.get("u_edge_id", None)
                value = d.get(key, None)
                if eid is not None and value is not None:
                    idToValue[eid] = value

            out_vals = []

            for i, edgeRecord in enumerate(edgeRecords):
                edgeIndex = edgeRecord.get("index", None)
                value = _round(idToValue.get(i, 0.0))
                out_vals.append(value)

                if graph._validate_edge_index(edgeIndex, active=False):
                    d = graph._edges[edgeIndex].setdefault("dictionary", {})
                    d.pop("u_edge_id", None)
                    if key is not None:
                        d[key] = value

            unit_range_for_color = True if nxCompatible or normalize else False
            _apply_values_to_edges(graph, edgeRecords, out_vals, unit_range=unit_range_for_color)

            return out_vals

        # ---------------------------------------------------------------------
        # Build graph adjacency once.
        # Graph.py treats the input graph as undirected for this calculation.
        # ---------------------------------------------------------------------

        vertexIndices = _active_vertex_indices(graph)
        edgeRecords = _active_edge_records(graph)

        if not vertexIndices:
            return []

        if not edgeRecords:
            if useEdges:
                return []

            out_vals = [_round(0.0) for _ in vertexIndices]
            _apply_values_to_vertices(graph, vertexIndices, out_vals, unit_range=True)
            return out_vals

        vertexIndexToPosition = {vertexIndex: i for i, vertexIndex in enumerate(vertexIndices)}
        n_nodes = len(vertexIndices)

        adj = defaultdict(list)

        for localEdgeIndex, edgeRecord in enumerate(edgeRecords):
            srcIndex = edgeRecord.get("src", None)
            dstIndex = edgeRecord.get("dst", None)

            if srcIndex == dstIndex:
                continue

            if srcIndex not in vertexIndexToPosition or dstIndex not in vertexIndexToPosition:
                continue

            u = vertexIndexToPosition[srcIndex]
            v = vertexIndexToPosition[dstIndex]
            w = _edge_weight(graph, edgeRecord)

            try:
                w = float(w)
                if w <= 0.0:
                    w = 1.0
            except Exception:
                w = 1.0

            # Undirected adjacency to match Graph.py behaviour.
            adj[u].append((v, w, localEdgeIndex))
            adj[v].append((u, w, localEdgeIndex))

        # ---------------------------------------------------------------------
        # Brandes betweenness centrality.
        # ---------------------------------------------------------------------

        def _brandes_unweighted():
            CBv = [0.0] * n_nodes
            CBe = [0.0] * len(edgeRecords)

            for s in range(n_nodes):
                S = []
                P = [[] for _ in range(n_nodes)]
                sigma = [0.0] * n_nodes
                dist = [-1] * n_nodes

                sigma[s] = 1.0
                dist[s] = 0

                Q = deque([s])

                while Q:
                    v = Q.popleft()
                    S.append(v)
                    dv = dist[v]

                    for w, _weight, edgeLocalIndex in adj.get(v, []):
                        if dist[w] < 0:
                            Q.append(w)
                            dist[w] = dv + 1

                        if dist[w] == dv + 1:
                            sigma[w] += sigma[v]
                            P[w].append((v, edgeLocalIndex))

                delta = [0.0] * n_nodes

                while S:
                    w = S.pop()

                    for v, edgeLocalIndex in P[w]:
                        if sigma[w] == 0.0:
                            c = 0.0
                        else:
                            c = (sigma[v] / sigma[w]) * (1.0 + delta[w])

                        CBe[edgeLocalIndex] += c
                        delta[v] += c

                    if w != s:
                        CBv[w] += delta[w]

            # Undirected graph: each shortest path is counted twice.
            return [x / 2.0 for x in CBv], [x / 2.0 for x in CBe]

        def _brandes_weighted():
            CBv = [0.0] * n_nodes
            CBe = [0.0] * len(edgeRecords)
            eq_eps = 1e-12

            for s in range(n_nodes):
                S = []
                P = [[] for _ in range(n_nodes)]
                sigma = [0.0] * n_nodes
                dist = [float("inf")] * n_nodes
                settled = [False] * n_nodes

                sigma[s] = 1.0
                dist[s] = 0.0

                heap = [(0.0, s)]

                while heap:
                    dv, v = heapq.heappop(heap)

                    if settled[v]:
                        continue

                    if dv > dist[v] + eq_eps:
                        continue

                    settled[v] = True
                    S.append(v)

                    for w, weight, edgeLocalIndex in adj.get(v, []):
                        vw_dist = dv + float(weight)

                        if vw_dist < dist[w] - eq_eps:
                            dist[w] = vw_dist
                            heapq.heappush(heap, (vw_dist, w))
                            sigma[w] = sigma[v]
                            P[w] = [(v, edgeLocalIndex)]

                        elif abs(vw_dist - dist[w]) <= eq_eps:
                            sigma[w] += sigma[v]
                            P[w].append((v, edgeLocalIndex))

                            if not settled[w]:
                                heapq.heappush(heap, (dist[w], w))

                delta = [0.0] * n_nodes

                while S:
                    w = S.pop()

                    for v, edgeLocalIndex in P[w]:
                        if sigma[w] == 0.0:
                            c = 0.0
                        else:
                            c = (sigma[v] / sigma[w]) * (1.0 + delta[w])

                        CBe[edgeLocalIndex] += c
                        delta[v] += c

                    if w != s:
                        CBv[w] += delta[w]

            # Undirected graph: each shortest path is counted twice.
            return [x / 2.0 for x in CBv], [x / 2.0 for x in CBe]

        if weightKey is None:
            CBv, CBe = _brandes_unweighted()
        else:
            CBv, CBe = _brandes_weighted()

        # ---------------------------------------------------------------------
        # Normalization.
        # ---------------------------------------------------------------------

        def _nx_scale_vertex(vals):
            n = n_nodes

            if n <= 2:
                return [0.0 for _ in vals]

            scale = 2.0 / float((n - 1) * (n - 2))
            return [float(v) * scale for v in vals]

        def _nx_scale_edge(vals):
            n = n_nodes

            if n <= 1:
                return [0.0 for _ in vals]

            scale = 2.0 / float(n * (n - 1))
            return [float(v) * scale for v in vals]

        # ---------------------------------------------------------------------
        # Edge mode.
        # ---------------------------------------------------------------------

        if useEdges:
            out_vals = [float(v) for v in CBe]

            if edgeKey is not None:
                group_sum = {}
                group_id = []

                for i, edgeRecord in enumerate(edgeRecords):
                    d = edgeRecord.get("dictionary", {})
                    gid = d.get(edgeKey, None) if isinstance(d, dict) else None

                    if gid is None:
                        gid = "__edge_" + str(i)

                    group_id.append(gid)
                    group_sum[gid] = group_sum.get(gid, 0.0) + float(out_vals[i])

                out_vals = [float(group_sum[group_id[i]]) for i in range(len(edgeRecords))]

            if nxCompatible:
                out_vals = _nx_scale_edge(out_vals)
                unit_range_for_color = True
            else:
                if normalize:
                    out_vals = _normalize_flat(out_vals)
                    unit_range_for_color = True
                else:
                    unit_range_for_color = False

            out_vals = [_round(v) for v in out_vals]

            _apply_values_to_edges(
                graph,
                edgeRecords,
                out_vals,
                unit_range=unit_range_for_color,
            )

            return out_vals

        # ---------------------------------------------------------------------
        # Vertex mode.
        # ---------------------------------------------------------------------

        out_vals = [float(v) for v in CBv]

        if nxCompatible:
            out_vals = _nx_scale_vertex(out_vals)
            unit_range_for_color = True
        else:
            if normalize:
                out_vals = _normalize_flat(out_vals)
                unit_range_for_color = True
            else:
                unit_range_for_color = False

        out_vals = [_round(v) for v in out_vals]

        _apply_values_to_vertices(
            graph,
            vertexIndices,
            out_vals,
            unit_range=unit_range_for_color,
        )

        return out_vals

    @staticmethod
    def BetweennessPartition(graph: "TGraph", n: int = 2, m: int = 10, key: str = "partition", tolerance: float = 0.0001, silent: bool = False) -> List[int]:
        """
        Partitions the input TGraph by iteratively removing high-betweenness edges.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        n : int , optional
            The input count, order, or requested number of groups. Default is 2.
        m : int , optional
            The input count or maximum number of iterations/removals. Default is 10.
        key : str , optional
            The dictionary key to use. Default is 'partition'.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[int]
            The resulting betweenness partition list.
        """
        if not isinstance(graph, TGraph):
            return []

        target_components = max(1, int(n))
        max_removals = max(0, int(m))
        working = TGraph.Copy(graph)
        if not isinstance(working, TGraph):
            return []

        for _ in range(max_removals):
            comps = TGraph.ConnectedComponents(working, mode="all")
            if len(comps) >= target_components:
                break
            edge_bc = TGraph._NativeEdgeBetweenness(working)
            if not edge_bc:
                break
            best_pair = min(edge_bc.keys(), key=lambda pair: (-edge_bc[pair], pair[0], pair[1]))
            edges = TGraph.EdgesBetween(working, best_pair[0], best_pair[1], directed=False)
            if not edges:
                break
            working.RemoveEdge(edges[0].get("index"), silent=True)

        comps = TGraph.ConnectedComponents(working, mode="all")
        vertices = TGraph.ActiveVertexIndices(graph)
        assignment = {v: 0 for v in vertices}
        for cid, comp in enumerate(comps):
            for v in comp:
                assignment[v] = cid

        values = []
        for v in vertices:
            cid = int(assignment.get(v, 0))
            values.append(cid)
            if key is not None:
                TGraph._SetVertexValue(graph, v, key, cid)
        return values

    @staticmethod
    def _BFSCompiledStateompact(adj: List[List[int]], source: int) -> Tuple[List[int], List[int]]:
        """
        Returns breadth-first-search distance and parent arrays for a compact adjacency array.

        Parameters
        ----------
        adj : List[List[int]]
            The input adj value.
        source : int
            The input source vertex, vertex index, or source identifier.

        Returns
        -------
        Tuple[List[int], List[int]]
            The resulting bfscompiled stateompact list.
        """
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
    def BOTClassByOntologyClass(ontologyClass: str, defaultValue: Any = None) -> Any:
        """
        Returns the BOT class corresponding to the input TopologicPy ontology class.

        Parameters
        ----------
        ontologyClass : str
            The ontology class value.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        Any
            The resulting botclass by ontology class object or value.
        """
        if ontologyClass is None:
            return defaultValue
        ontologyClass = str(ontologyClass).strip()
        config = TGraph._OntologyConfig()
        if ontologyClass in config["top_to_bot"]:
            return config["top_to_bot"][ontologyClass]
        return defaultValue

    @staticmethod
    def BOTString(*args, **kwargs) -> Optional[str]:
        """
        Returns a BOT-compatible TTL string representation of the input TGraph.

        Parameters
        ----------
        *args : Any , optional
            Additional positional arguments.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting botstring string.
        """
        kwargs["includeBOT"] = True
        return TGraph.TTLString(*args, **kwargs)

    @staticmethod
    def BreadthFirstSearch(graph: "TGraph", source: int, mode: str = "out") -> List[int]:
        """
        Returns the breadth-first traversal order from the input source vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        source : int
            The input source vertex, vertex index, or source identifier.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.

        Returns
        -------
        List[int]
            The resulting breadth first search list.
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
    def _BREPString(topology: Any) -> Optional[str]:
        """
        Returns a BREP string representation of the input topology.

        Parameters
        ----------
        topology : Any
            The input Topologic topology.

        Returns
        -------
        Optional[str]
            The resulting brepstring string.
        """
        try:
            from topologicpy.Topology import Topology
            return Topology.BREPString(topology)
        except Exception:
            return None

    @staticmethod
    def Bridges(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Returns the bridge edges of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting bridges list.
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
    def ByAdjacencyDictionary(
        adjacencyDictionary: Optional[Dict[Any, Iterable[Any]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        vertexLabelKey: str = "label",
        edgeWeightKey: str = "weight",
        ontology: bool = True,
    ) -> "TGraph":
        """
        Creates a TGraph from an adjacency dictionary.

        Parameters
        ----------
        adjacencyDictionary : Optional[Dict[Any, Iterable[Any]]] , optional
            The input adjacency dictionary value. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is 'weight'.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:Graph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.ByAdjacencyDictionary", ontology=ontology, silent=True)

    @staticmethod
    def ByAdjacencyList(
        adjacencyList: Optional[List[Iterable[int]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        ontology: bool = True,
    ) -> "TGraph":
        """
        Creates a TGraph from an adjacency list.

        Parameters
        ----------
        adjacencyList : Optional[List[Iterable[int]]] , optional
            The input adjacency list value. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
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
            ontology=ontology,
        )

    @staticmethod
    def ByAdjacencyMatrix(
        adjacencyMatrix: Optional[List[List[Any]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        edgeKey: str = "weight",
        ontology: bool = True,
    ) -> "TGraph":
        """
        Creates a TGraph from an adjacency matrix.

        Parameters
        ----------
        adjacencyMatrix : Optional[List[List[Any]]] , optional
            The input adjacency matrix value. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        edgeKey : str , optional
            The edge dictionary key to use. Default is 'weight'.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
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
            ontology=ontology,
        )

    @staticmethod
    def ByDictionaries(
        vertexDictionaries: Optional[List[Dict[str, Any]]] = None,
        edgeDictionaries: Optional[List[Dict[str, Any]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        ontology: bool = True,
    ) -> "TGraph":
        """
        Creates a TGraph from vertex and edge dictionaries.

        Parameters
        ----------
        vertexDictionaries : Optional[List[Dict[str, Any]]] , optional
            The input vertex dictionaries value. Default is None.
        edgeDictionaries : Optional[List[Dict[str, Any]]] , optional
            The input edge dictionaries value. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
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
            ontology=ontology,
        )

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
        buildEdgeLookup: bool = True,
        ontology: bool = True,
    ) -> Optional["TGraph"]:
        """
        Creates a TGraph from a vertex count and edge index pairs.

        Parameters
        ----------
        order : int
            The input order value.
        edgeIndexPairs : Optional[Iterable[Union[Tuple[int, int], List[int]]]] , optional
            The input edge index pairs value. Default is None.
        vertexDictionaries : Optional[List[Dict[str, Any]]] , optional
            The input vertex dictionaries value. Default is None.
        edgeDictionaries : Optional[List[Dict[str, Any]]] , optional
            The input edge dictionaries value. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        representations : Optional[Dict[str, Any]] , optional
            The input representations value. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        lean : bool , optional
            The input lean value. Default is False.
        buildEdgeLookup : bool , optional
            The input build edge lookup value. Default is True.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if lean:
            g = TGraph._ByEdgeIndexPairsLeanFast(
                order=order,
                edgeIndexPairs=edgeIndexPairs,
                directed=directed,
                allowSelfLoops=allowSelfLoops,
                allowParallelEdges=allowParallelEdges,
                dictionary=dictionary,
                silent=silent,
                buildEdgeLookup=buildEdgeLookup,
                inputUnique=False,
            )
            return TGraph._OntologyAnnotateGraph(
                g, graphClass="top:Graph", vertexClass="top:Node", edgeClass="top:Relationship",
                generatedBy="TGraph.ByEdgeIndexPairs", ontology=ontology, silent=silent)

        if not isinstance(order, int) or order < 0:
            if not silent:
                print("TGraph.ByEdgeIndexPairs - Error: order must be a non-negative integer. Returning None.")
            return None

        g = TGraph(directed=directed, allowSelfLoops=allowSelfLoops,
                   allowParallelEdges=allowParallelEdges, dictionary=dictionary)

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

        for k, pair in enumerate(edgeIndexPairs or []):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            src, dst = pair[0], pair[1]
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            if src < 0 or dst < 0 or src >= order or dst >= order:
                continue
            if src == dst and not allowSelfLoops:
                continue

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

        g._dictionary["__edge_lookup_valid__"] = True
        g._invalidate_cache()
        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:Graph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.ByEdgeIndexPairs", ontology=ontology, silent=silent)

    @staticmethod
    def _ByEdgeIndexPairsLeanFast(
        order: int,
        edgeIndexPairs: Optional[Iterable[Union[Tuple[int, int], List[int]]]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        buildEdgeLookup: bool = False,
        inputUnique: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates a TGraph from edge index pairs using a lean internal construction path.

        Parameters
        ----------
        order : int
            The input order value.
        edgeIndexPairs : Optional[Iterable[Union[Tuple[int, int], List[int]]]] , optional
            The input edge index pairs value. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        buildEdgeLookup : bool , optional
            The input build edge lookup value. Default is False.
        inputUnique : bool , optional
            The input input unique value. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(order, int) or order < 0:
            if not silent:
                print("TGraph._ByEdgeIndexPairsLeanFast - Error: order must be a non-negative integer. Returning None.")
            return None

        g = TGraph(directed=directed, allowSelfLoops=allowSelfLoops,
                   allowParallelEdges=allowParallelEdges, dictionary=dictionary)

        g._vertices = [
            {"index": i, "dictionary": {"index": i}, "representation": None, "active": True}
            for i in range(order)
        ]
        out_sets = [set() for _ in range(order)]
        in_sets = [set() for _ in range(order)]
        incident_sets = [set() for _ in range(order)]
        edge_lookup = {} if buildEdgeLookup else None
        edges_out = []
        seen = set() if (not allowParallelEdges and not inputUnique) else None

        edge_directed = bool(directed)
        append_edge = edges_out.append

        for pair in edgeIndexPairs or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            src, dst = pair[0], pair[1]
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            if src < 0 or dst < 0 or src >= order or dst >= order:
                continue
            if src == dst and not allowSelfLoops:
                continue

            if edge_directed:
                key = (src, dst, True)
            else:
                key = (src, dst, False) if src <= dst else (dst, src, False)

            if seen is not None:
                if key in seen:
                    continue
                seen.add(key)

            index = len(edges_out)
            append_edge({
                "index": index,
                "src": src,
                "dst": dst,
                "directed": edge_directed,
                "dictionary": {},
                "representation": None,
                "active": True,
            })

            out_sets[src].add(index)
            in_sets[dst].add(index)
            incident_sets[src].add(index)
            incident_sets[dst].add(index)
            if not edge_directed:
                out_sets[dst].add(index)
                in_sets[src].add(index)
            if edge_lookup is not None:
                edge_lookup.setdefault(key, set()).add(index)

        g._edges = edges_out
        g._out_edges = {i: out_sets[i] for i in range(order)}
        g._in_edges = {i: in_sets[i] for i in range(order)}
        g._incident_edges = {i: incident_sets[i] for i in range(order)}
        g._edge_lookup = edge_lookup if edge_lookup is not None else {}
        g._dictionary["__edge_lookup_valid__"] = bool(buildEdgeLookup)
        g._invalidate_cache()
        return g

    @staticmethod
    def ByIFCFile(
        file,
        importMode: str = "topology",
        clean: bool = False,
        storeBREP: bool = False,
        useInternalVertex: bool = False,
        includeTypes: list = None,
        excludeTypes: list = None,
        includeRels: list = None,
        excludeRels: list = None,
        dictionaryMode: str = "basic",
        xMin: float = -0.5,
        yMin: float = -0.5,
        zMin: float = -0.5,
        xMax: float = 0.5,
        yMax: float = 0.5,
        zMax: float = 0.5,
        vertexColorKey: str = "color",
        edgeColorKey: str = "color",
        colorScale: str = "viridis",
        epsilon: float = 0.01,
        angTolerance: float = 0.1,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates a TGraph from an IFC file.

        Supported relationship extraction is convention-based and includes IFC
        relationships with Relating* and Related* attributes, including but not
        limited to IfcRelAggregates, IfcRelNests, IfcRelAssignsToGroup,
        IfcRelContainedInSpatialStructure, IfcRelFillsElement, IfcRelVoidsElement,
        IfcRelDefinesByProperties, IfcRelAssociatesMaterial, and IfcRelDefinesByType.

        Parameters
        ----------
        file : ifcopenshell.file or str
            The input IFC file object or path to an IFC file.
        importMode : str , optional
            Import mode. Options are "topology", "geometry", "triples", and "semantic".
            Default is "topology".
        clean : bool , optional
            If set to True, coplanar faces and collinear edges are removed in geometry
            mode where applicable. Default is False.
        storeBREP : bool , optional
            In geometry mode, if set to True, stores the BREP string of each imported
            topology under the key "BREP". Default is False.
        useInternalVertex : bool , optional
            In geometry mode, if set to True, uses an internal vertex to represent each
            topology. Otherwise, uses its centroid. Default is False.
        includeTypes : list , optional
            A list of IFC entity types to include as primary graph vertices. Default is
            None.
        excludeTypes : list , optional
            A list of IFC entity types to exclude from the graph. Default is None.
        includeRels : list , optional
            A list of IFC relationship types to include. Default is None.
        excludeRels : list , optional
            A list of IFC relationship types to exclude. Default is None.
        dictionaryMode : str , optional
            Options are "none", "basic", "psets", "all", or "full". Default is "basic".
        xMin : float , optional
            Desired minimum X coordinate for topology layout. Default is -0.5.
        yMin : float , optional
            Desired minimum Y coordinate for topology layout. Default is -0.5.
        zMin : float , optional
            Desired minimum Z coordinate for topology layout. Default is -0.5.
        xMax : float , optional
            Desired maximum X coordinate for topology layout. Default is 0.5.
        yMax : float , optional
            Desired maximum Y coordinate for topology layout. Default is 0.5.
        zMax : float , optional
            Desired maximum Z coordinate for topology layout. Default is 0.5.
        vertexColorKey : str , optional
            The desired vertex dictionary key under which to store the calculated color.
            Default is "color".
        edgeColorKey : str , optional
            The desired edge dictionary key under which to store the calculated color.
            Default is "color".
        colorScale : str , optional
            The desired color scale name. Default is "viridis".
        epsilon : float , optional
            Desired tolerance for removing coplanar faces. Default is 0.01.
        angTolerance : float , optional
            Angular tolerance for removing collinear edges. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The created TGraph, or None if the operation fails.
        """

        import math

        try:
            from topologicpy.Vertex import Vertex
            from topologicpy.Topology import Topology
            from topologicpy.IFC import IFC
            from topologicpy.Color import Color
        except Exception as exc:
            if not silent:
                print(f"TGraph.ByIFCFile - Error: Could not import required TopologicPy classes. {exc} Returning None.")
            return None

        try:
            import ifcopenshell
        except Exception:
            ifcopenshell = None

        # ---------------------------------------------------------------------
        # Validate colorscale.
        # ---------------------------------------------------------------------

        color_scales = [
            "aggrnyl", "agsunset", "algae", "amp", "armyrose", "balance",
            "blackbody", "bluered", "blues", "blugrn", "bluyl", "brbg",
            "brwnyl", "bugn", "bupu", "burg", "burgyl", "cividis", "curl",
            "darkmint", "deep", "delta", "dense", "earth", "edge", "electric",
            "emrld", "fall", "geyser", "gnbu", "gray", "greens", "greys",
            "haline", "hot", "hsv", "ice", "icefire", "inferno", "jet",
            "magenta", "magma", "matter", "mint", "mrybm", "mygbm", "oranges",
            "orrd", "oryel", "oxy", "peach", "phase", "picnic", "pinkyl",
            "piyg", "plasma", "plotly3", "portland", "prgn", "pubu", "pubugn",
            "puor", "purd", "purp", "purples", "purpor", "rainbow", "rdbu",
            "rdgy", "rdpu", "rdylbu", "rdylgn", "redor", "reds", "solar",
            "spectral", "speed", "sunset", "sunsetdark", "teal", "tealgrn",
            "tealrose", "tempo", "temps", "thermal", "tropic", "turbid",
            "turbo", "twilight", "viridis", "ylgn", "ylgnbu", "ylorbr",
            "ylorrd"
        ]
        color_scales += [s + "_r" for s in color_scales]

        if not isinstance(colorScale, str):
            colorScale = "viridis"

        colorScale = colorScale.lower()

        if colorScale not in color_scales:
            if not silent:
                print(f"TGraph.ByIFCFile - Error: Unknown Plotly colorscale ({colorScale}). Returning None.")
            return None

        # ---------------------------------------------------------------------
        # Input normalisation.
        # ---------------------------------------------------------------------

        includeTypes = includeTypes or []
        excludeTypes = excludeTypes or []
        includeRels = includeRels or []
        excludeRels = excludeRels or []

        mode_aliases = {
            "relationship": "topology",
            "relationships": "topology",
            "rel": "topology",
            "rels": "topology",
            "topological": "topology",
            "entity": "topology",
            "entities": "topology",
            "geom": "geometry",
            "geometric": "geometry",
            "triple": "triples",
            "triples": "triples",
            "rdf": "triples",
            "semantic": "semantic",
            "semantics": "semantic",
            "knowledge": "semantic",
            "kg": "semantic",
        }

        importMode = (importMode or "topology").lower().strip()
        importMode = mode_aliases.get(importMode, importMode)

        if importMode not in ["geometry", "topology", "triples", "semantic"]:
            if not silent:
                print(f"TGraph.ByIFCFile - Warning: Unsupported mode '{importMode}'. Falling back to 'topology'.")
            importMode = "topology"

        dictionaryMode = (dictionaryMode or "basic").lower().strip()

        if isinstance(file, str):
            if ifcopenshell is None:
                if not silent:
                    print("TGraph.ByIFCFile - Error: ifcopenshell is not installed. Returning None.")
                return None
            try:
                file = ifcopenshell.open(file)
            except Exception as exc:
                if not silent:
                    print(f"TGraph.ByIFCFile - Error: Could not open IFC file. {exc} Returning None.")
                return None

        if file is None:
            if not silent:
                print("TGraph.ByIFCFile - Error: The input file is None. Returning None.")
            return None

        include_type_set = {str(s).lower() for s in includeTypes}
        exclude_type_set = {str(s).lower() for s in excludeTypes}
        include_rel_set = {str(s).lower() for s in includeRels}
        exclude_rel_set = {str(s).lower() for s in excludeRels}

        # ---------------------------------------------------------------------
        # Basic IFC helpers.
        # ---------------------------------------------------------------------

        def _entity_global_id(entity):
            try:
                gid = getattr(entity, "GlobalId", None)
                return gid if gid not in ["", None] else None
            except Exception:
                return None

        def _entity_type(entity):
            try:
                if hasattr(entity, "is_a") and callable(entity.is_a):
                    return entity.is_a()
            except Exception:
                pass
            try:
                et = getattr(entity, "type", None)
                if et is not None:
                    return et() if callable(et) else et
            except Exception:
                pass
            if isinstance(entity, dict):
                return entity.get("type") or entity.get("Type") or entity.get("ifc_type") or entity.get("IfcType")
            return None

        def _entity_name(entity):
            try:
                name = getattr(entity, "Name", None)
                return name if name not in ["", None] else None
            except Exception:
                return None

        def _entity_id(entity):
            try:
                if hasattr(entity, "id") and callable(entity.id):
                    return entity.id()
            except Exception:
                pass
            try:
                eid = getattr(entity, "id", None)
                return eid() if callable(eid) else eid
            except Exception:
                return None

        def _entity_key(entity):
            gid = _entity_global_id(entity)
            if gid:
                return str(gid)

            eid = _entity_id(entity)
            if eid is not None:
                try:
                    return f"#{int(eid)}"
                except Exception:
                    return f"#{eid}"

            return None

        def _passes_type_filter(entity, include_set=None, exclude_set=None):
            include_set = include_set if include_set is not None else include_type_set
            exclude_set = exclude_set if exclude_set is not None else exclude_type_set

            et = _entity_type(entity)
            if not et:
                return False

            et_l = str(et).lower()

            if et_l in exclude_set:
                return False

            if include_set and et_l not in include_set:
                return False

            return True

        def _flatten_psets(entity):
            result = {}

            if dictionaryMode not in ["psets", "all", "full"]:
                return result

            try:
                import ifcopenshell.util.element
                psets = ifcopenshell.util.element.get_psets(entity) or {}
            except Exception:
                return result

            for pset_name, pset_data in psets.items():
                if not isinstance(pset_data, dict):
                    continue
                for k, v in pset_data.items():
                    if k == "id":
                        continue
                    try:
                        result[f"{pset_name}.{k}"] = v
                    except Exception:
                        pass

            return result

        def _dictionary_from_ifc_entity(entity):
            if dictionaryMode in ["none", "no", "false"]:
                return {}

            d = {}

            gid = _entity_global_id(entity)
            if gid is not None:
                d["IFC_global_id"] = gid

            eid = _entity_id(entity)
            if eid is not None:
                d["IFC_id"] = eid

            key = _entity_key(entity)
            if key is not None:
                d["IFC_key"] = key

            name = _entity_name(entity)
            if name is not None:
                d["IFC_name"] = name

            etype = _entity_type(entity)
            if etype is not None:
                d["IFC_type"] = str(etype)

            if dictionaryMode in ["psets", "all", "full"]:
                d.update(_flatten_psets(entity))

            return d

        def _relationship_dictionary(ifc_rel):
            if dictionaryMode in ["none", "no", "false"]:
                return {}

            d = {}

            try:
                rel_gid = getattr(ifc_rel, "GlobalId", None)
            except Exception:
                rel_gid = None

            try:
                rel_id = ifc_rel.id()
            except Exception:
                rel_id = None

            try:
                rel_name = getattr(ifc_rel, "Name", None)
            except Exception:
                rel_name = None

            try:
                rel_type = ifc_rel.is_a()
            except Exception:
                rel_type = None

            if rel_gid is not None:
                d["IFC_global_id"] = rel_gid

            if rel_id is not None:
                d["IFC_id"] = rel_id

            if rel_name is not None:
                d["IFC_name"] = rel_name

            if rel_type is not None:
                d["IFC_type"] = str(rel_type)
                d["relationship"] = str(rel_type)

            return d

        def _topology_dictionary(topology):
            try:
                return TGraph._DictionaryToPython(Topology.Dictionary(topology))
            except Exception:
                return {}

        # ---------------------------------------------------------------------
        # RDF-like triples / semantic graph modes.
        # ---------------------------------------------------------------------

        if importMode in ["triples", "semantic"]:
            try:
                triples = IFC.Triples(
                    file,
                    includeRels=includeRels,
                    excludeRels=excludeRels,
                    includeMetadata=True,
                    silent=silent,
                )
            except TypeError:
                try:
                    triples = IFC.Triples(
                        file,
                        includeRels=includeRels,
                        excludeRels=excludeRels,
                        silent=silent,
                    )
                except Exception as exc:
                    if not silent:
                        print(f"TGraph.ByIFCFile - Error: Could not extract IFC triples. {exc} Returning None.")
                    return None
            except Exception as exc:
                if not silent:
                    print(f"TGraph.ByIFCFile - Error: Could not extract IFC triples. {exc} Returning None.")
                return None

            if not isinstance(triples, list):
                if not silent:
                    print("TGraph.ByIFCFile - Error: IFC.Triples did not return a valid list. Returning None.")
                return None

            g = TGraph(
                directed=True,
                allowSelfLoops=True,
                allowParallelEdges=False,
                dictionary={"generated_by": "TGraph.ByIFCFile", "import_mode": importMode},
            )

            label_to_index = {}

            def _triple_value(triple, *keys):
                if isinstance(triple, dict):
                    for k in keys:
                        if k in triple:
                            return triple.get(k)
                return None

            def _ensure_vertex(label, vtype=None):
                label = str(label)

                if label not in label_to_index:
                    d = {"id": label, "label": label}
                    if vtype is not None:
                        d["type"] = vtype
                    label_to_index[label] = g.AddVertex(dictionary=d)

                return label_to_index[label]

            for triple in triples:
                if isinstance(triple, dict):
                    subject = _triple_value(triple, "subject", "s")
                    predicate = _triple_value(triple, "predicate", "p")
                    obj = _triple_value(triple, "object", "o")
                elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
                    subject, predicate, obj = triple[0], triple[1], triple[2]
                else:
                    continue

                if subject is None or predicate is None or obj is None:
                    continue

                src = _ensure_vertex(subject)
                dst = _ensure_vertex(obj)
                g.AddEdge(
                    src,
                    dst,
                    directed=True,
                    dictionary={
                        "label": str(predicate),
                        "predicate": str(predicate),
                        "relationship": str(predicate),
                    },
                )

            return TGraph._OntologyAnnotateGraph(
                g,
                graphClass="top:KnowledgeGraph",
                vertexClass="top:Node",
                edgeClass="top:Relationship",
                generatedBy="TGraph.ByIFCFile",
                ontology=True,
                silent=True,
            )

        # ---------------------------------------------------------------------
        # Relationship extraction.
        # ---------------------------------------------------------------------

        def _relationship_endpoints(ifc_rel):
            if ifc_rel is None:
                return None, []

            try:
                rel_type = ifc_rel.is_a()
            except Exception:
                return None, []

            if rel_type == "IfcRelConnectsPorts":
                src = getattr(ifc_rel, "RelatingPort", None)
                dst = getattr(ifc_rel, "RelatedPort", None)
                return src, [dst] if dst is not None else []

            try:
                info = ifc_rel.get_info()
                attr_names = list(info.keys())
            except Exception:
                attr_names = []

            relating_attrs = []
            related_attrs = []

            for attr_name in attr_names:
                if not isinstance(attr_name, str):
                    continue
                if attr_name.startswith("Relating"):
                    relating_attrs.append(attr_name)
                elif attr_name.startswith("Related"):
                    related_attrs.append(attr_name)

            source = None

            for attr_name in relating_attrs:
                try:
                    value = getattr(ifc_rel, attr_name, None)
                except Exception:
                    value = None
                if value is not None:
                    source = value
                    break

            destinations = []

            for attr_name in related_attrs:
                try:
                    value = getattr(ifc_rel, attr_name, None)
                except Exception:
                    value = None

                if value is None:
                    continue

                if isinstance(value, (list, tuple)):
                    destinations.extend([v for v in value if v is not None])
                else:
                    destinations.append(value)

            return source, destinations

        def _get_relationships(ifc_file):
            relationships = []

            try:
                rels = ifc_file.by_type("IfcRelationship")
            except Exception:
                rels = []

            for rel in rels:
                try:
                    rel_type = rel.is_a()
                    rel_type_l = str(rel_type).lower()
                except Exception:
                    continue

                if rel_type_l in exclude_rel_set:
                    continue

                if include_rel_set and rel_type_l not in include_rel_set:
                    continue

                relationships.append(rel)

            return relationships

        def _is_vertex_candidate(entity):
            if entity is None:
                return False

            et = _entity_type(entity)
            if not et:
                return False

            if _entity_global_id(entity):
                return True

            allowed_without_guid = {
                "IfcMaterial",
                "IfcMaterialLayer",
                "IfcMaterialLayerSet",
                "IfcMaterialLayerSetUsage",
                "IfcMaterialProfile",
                "IfcMaterialProfileSet",
                "IfcMaterialProfileSetUsage",
                "IfcMaterialConstituent",
                "IfcMaterialConstituentSet",
                "IfcMaterialList",
                "IfcPropertySingleValue",
                "IfcPropertyEnumeratedValue",
                "IfcPropertyListValue",
                "IfcPropertyTableValue",
                "IfcPropertyBoundedValue",
                "IfcPropertyReferenceValue",
                "IfcPhysicalQuantity",
                "IfcQuantityArea",
                "IfcQuantityCount",
                "IfcQuantityLength",
                "IfcQuantityNumber",
                "IfcQuantityTime",
                "IfcQuantityVolume",
                "IfcQuantityWeight",
                "IfcRepresentationMap",
                "IfcClassification",
                "IfcClassificationReference",
                "IfcExternalReference",
                "IfcDocumentReference",
                "IfcLibraryReference",
            }

            return str(et) in allowed_without_guid

        # ---------------------------------------------------------------------
        # Synthetic layout.
        # ---------------------------------------------------------------------

        def _synthetic_coordinates(index, total_count):
            if total_count <= 1:
                return (
                    0.5 * (xMin + xMax),
                    0.5 * (yMin + yMax),
                    zMin,
                )

            cols = max(1, int(math.ceil(math.sqrt(total_count))))
            rows = max(1, int(math.ceil(float(total_count) / float(cols))))

            col = index % cols
            row = index // cols

            x = xMin if cols == 1 else xMin + (float(col) / float(cols - 1)) * (xMax - xMin)
            y = yMin if rows == 1 else yMin + (float(row) / float(rows - 1)) * (yMax - yMin)
            z = zMin

            return x, y, z

        # ---------------------------------------------------------------------
        # Graph entity collection.
        # ---------------------------------------------------------------------

        def _collect_graph_entities(relationships):
            entity_by_key = {}

            def _add_entity(entity, respect_include_filter=True):
                if not _is_vertex_candidate(entity):
                    return

                et = _entity_type(entity)

                if et and str(et).lower() in exclude_type_set:
                    return

                if respect_include_filter and not _passes_type_filter(entity):
                    return

                key = _entity_key(entity)

                if key is None:
                    return

                if key not in entity_by_key:
                    entity_by_key[key] = entity

            try:
                for entity in file.by_type("IfcRoot"):
                    _add_entity(entity, respect_include_filter=True)
            except Exception:
                pass

            for ifc_rel in relationships:
                source, destinations = _relationship_endpoints(ifc_rel)

                if source is not None:
                    _add_entity(source, respect_include_filter=False)

                for destination in destinations or []:
                    _add_entity(destination, respect_include_filter=False)

            return list(entity_by_key.values())

        def _add_key_aliases(key_to_index, index, entity=None, gid=None, step_id=None, explicit_key=None):
            keys = []

            if explicit_key is not None:
                keys.append(str(explicit_key))

            if entity is not None:
                try:
                    e_key = _entity_key(entity)
                    if e_key is not None:
                        keys.append(str(e_key))
                except Exception:
                    pass

                try:
                    e_gid = _entity_global_id(entity)
                    if e_gid:
                        keys.append(str(e_gid))
                except Exception:
                    pass

                try:
                    e_step_id = _entity_id(entity)
                    if e_step_id is not None:
                        keys.append(str(e_step_id))
                        keys.append(f"#{e_step_id}")
                except Exception:
                    pass

            if gid:
                keys.append(str(gid))

            if step_id is not None:
                keys.append(str(step_id))
                keys.append(f"#{step_id}")

            for k in keys:
                if k:
                    key_to_index[k] = index

            return key_to_index

        def _topology_entity_key(topology):
            td = _topology_dictionary(topology)

            gid = td.get("IFC_global_id", None)
            step_id = td.get("IFC_id", None)
            explicit_key = td.get("IFC_key", None)

            key = None

            if explicit_key not in [None, "", 0, "0"]:
                key = str(explicit_key)
            elif gid not in [None, "", 0, "0"]:
                key = str(gid)
            elif step_id not in [None, "", 0, "0"]:
                key = f"#{step_id}"

            return key, gid, step_id

        def _initialise_mesh_data_from_entities(entities):
            vertices = []
            vertex_dictionaries = []
            key_to_index = {}

            total_count = max(len(entities), 1)

            for i, entity in enumerate(entities):
                key = _entity_key(entity)

                if key is None:
                    continue

                x, y, z = _synthetic_coordinates(i, total_count)
                d = _dictionary_from_ifc_entity(entity)

                index = len(vertices)
                d["index"] = index
                d.setdefault("x", float(x))
                d.setdefault("y", float(y))
                d.setdefault("z", float(z))

                vertices.append([float(x), float(y), float(z)])
                vertex_dictionaries.append(d)

                _add_key_aliases(key_to_index, index, entity=entity, explicit_key=key)

            return vertices, vertex_dictionaries, key_to_index

        def _add_missing_entity_vertex(entity, vertices, vertex_dictionaries, key_to_index):
            key = _entity_key(entity)

            if key is None:
                return None

            index = key_to_index.get(key)

            if index is not None:
                return index

            x, y, z = _synthetic_coordinates(len(vertices), len(vertices) + 1)

            d = _dictionary_from_ifc_entity(entity)
            index = len(vertices)

            d["index"] = index
            d.setdefault("x", float(x))
            d.setdefault("y", float(y))
            d.setdefault("z", float(z))

            vertices.append([float(x), float(y), float(z)])
            vertex_dictionaries.append(d)

            _add_key_aliases(key_to_index, index, entity=entity, explicit_key=key)

            return index

        def _apply_geometry_coordinates(vertices, vertex_dictionaries, key_to_index):
            try:
                topologies = IFC.TopologiesByFile(
                    file,
                    includeTypes=includeTypes,
                    excludeTypes=excludeTypes,
                    dictionaryMode=dictionaryMode,
                    clean=clean,
                    epsilon=epsilon,
                    angTolerance=angTolerance,
                    tolerance=tolerance,
                    silent=silent,
                )
            except Exception as exc:
                if not silent:
                    print(f"TGraph.ByIFCFile - Warning: Could not import IFC geometry. {exc} Keeping synthetic coordinates.")
                topologies = []

            if topologies is None:
                topologies = []
            elif not isinstance(topologies, list):
                topologies = [topologies]

            for topology in topologies:
                if not Topology.IsInstance(topology, "Topology"):
                    continue

                key, gid, step_id = _topology_entity_key(topology)

                if key is None:
                    continue

                index = key_to_index.get(key)

                if index is None:
                    index = len(vertices)
                    vertices.append(None)
                    vertex_dictionaries.append({})
                    _add_key_aliases(key_to_index, index, gid=gid, step_id=step_id, explicit_key=key)

                try:
                    v = Topology.InternalVertex(topology) if useInternalVertex else Topology.Centroid(topology)
                except Exception:
                    v = None

                if not Topology.IsInstance(v, "Vertex"):
                    if not silent:
                        print(f"TGraph.ByIFCFile - Warning: Could not create a representative vertex for entity {key}. Keeping synthetic coordinates.")
                    continue

                td = _topology_dictionary(topology)

                if storeBREP:
                    try:
                        td["BREP"] = Topology.BREPString(topology)
                    except Exception:
                        if not silent:
                            print("TGraph.ByIFCFile - Warning: Could not store BREP string for one topology. Continuing.")

                try:
                    coords = Vertex.Coordinates(v)
                except Exception:
                    coords = None

                if coords is None or len(coords) < 3:
                    continue

                vertices[index] = [float(coords[0]), float(coords[1]), float(coords[2])]

                td["index"] = index
                td.setdefault("x", float(coords[0]))
                td.setdefault("y", float(coords[1]))
                td.setdefault("z", float(coords[2]))

                vertex_dictionaries[index] = td

                _add_key_aliases(key_to_index, index, gid=gid, step_id=step_id, explicit_key=key)

            for i, coords in enumerate(vertices):
                if coords is None:
                    coords = _synthetic_coordinates(i, max(len(vertices), 1))
                    vertices[i] = [float(coords[0]), float(coords[1]), float(coords[2])]

            return vertices, vertex_dictionaries, key_to_index

        def _build_edges_from_relationships(relationships, vertices, vertex_dictionaries, key_to_index):
            edges = []
            edge_dictionaries = []
            seen_edges = set()

            for ifc_rel in relationships:
                source, destinations = _relationship_endpoints(ifc_rel)

                if source is None or not destinations:
                    continue

                source_key = _entity_key(source)

                if source_key is None:
                    continue

                src_index = key_to_index.get(source_key)

                if src_index is None:
                    src_index = _add_missing_entity_vertex(source, vertices, vertex_dictionaries, key_to_index)

                if src_index is None:
                    if not silent:
                        try:
                            rel_label = f"{ifc_rel.is_a()} #{ifc_rel.id()}"
                        except Exception:
                            rel_label = str(ifc_rel)
                        print(f"TGraph.ByIFCFile - Warning: Source endpoint not found as graph vertex. Relationship: {rel_label}.")
                    continue

                rel_dict = _relationship_dictionary(ifc_rel)

                try:
                    rel_type = ifc_rel.is_a()
                except Exception:
                    rel_type = "IfcRelationship"

                for destination in destinations:
                    if destination is None:
                        continue

                    dest_key = _entity_key(destination)

                    if dest_key is None:
                        continue

                    dst_index = key_to_index.get(dest_key)

                    if dst_index is None:
                        dst_index = _add_missing_entity_vertex(destination, vertices, vertex_dictionaries, key_to_index)

                    if dst_index is None:
                        if not silent:
                            try:
                                rel_label = f"{ifc_rel.is_a()} #{ifc_rel.id()}"
                            except Exception:
                                rel_label = str(ifc_rel)
                            print(f"TGraph.ByIFCFile - Warning: Destination endpoint not found as graph vertex. Relationship: {rel_label}.")
                        continue

                    if src_index == dst_index:
                        continue

                    edge_key = (src_index, dst_index, str(rel_type))

                    if edge_key in seen_edges:
                        continue

                    seen_edges.add(edge_key)

                    edge_dict = dict(rel_dict)
                    edge_dict["src"] = src_index
                    edge_dict["dst"] = dst_index
                    edge_dict.setdefault("directed", False)

                    edges.append([src_index, dst_index])
                    edge_dictionaries.append(edge_dict)

            return edges, edge_dictionaries

        def _get_mesh_data_by_mode(mode):
            relationships = _get_relationships(file)

            entities = _collect_graph_entities(relationships)
            vertices, vertex_dictionaries, key_to_index = _initialise_mesh_data_from_entities(entities)

            if mode == "geometry":
                vertices, vertex_dictionaries, key_to_index = _apply_geometry_coordinates(
                    vertices,
                    vertex_dictionaries,
                    key_to_index,
                )

            edges, edge_dictionaries = _build_edges_from_relationships(
                relationships,
                vertices,
                vertex_dictionaries,
                key_to_index,
            )

            return vertices, edges, vertex_dictionaries, edge_dictionaries

        def _unique_sorted_types_from_dictionaries(dictionaries, default_type=None):
            type_set = set()

            for d in dictionaries:
                if not isinstance(d, dict):
                    continue

                ifc_type = d.get("IFC_type", None)

                if ifc_type not in [None, ""]:
                    type_set.add(str(ifc_type).lower())

            if not type_set and default_type:
                type_set.add(str(default_type).lower())

            return sorted(type_set)

        # ---------------------------------------------------------------------
        # Execute.
        # ---------------------------------------------------------------------

        vertices, edges, vertex_dictionaries, edge_dictionaries = _get_mesh_data_by_mode(importMode)

        element_types = _unique_sorted_types_from_dictionaries(vertex_dictionaries, default_type="Unknown")
        rel_types = _unique_sorted_types_from_dictionaries(edge_dictionaries, default_type="IfcRelationship")

        v_dicts = []

        for v_d in vertex_dictionaries:
            if not isinstance(v_d, dict):
                v_d = {}

            ifc_type = v_d.get("IFC_type", "Unknown")
            ifc_type = str(ifc_type).lower() if ifc_type is not None else "unknown"

            try:
                index = element_types.index(ifc_type)
            except Exception:
                index = 0

            maxValue = max(len(element_types) - 1, 1)

            try:
                vertexColor = Color.AnyToHex(
                    Color.ByValueInRange(
                        index,
                        minValue=0,
                        maxValue=maxValue,
                        colorScale=colorScale,
                    )
                )
                v_d[vertexColorKey] = vertexColor
            except Exception:
                pass

            v_dicts.append(v_d)

        e_dicts = []

        for e_d in edge_dictionaries:
            if not isinstance(e_d, dict):
                e_d = {}

            ifc_rel = e_d.get("IFC_type", "IfcRelationship")
            ifc_rel = str(ifc_rel).lower() if ifc_rel is not None else "ifcrelationship"

            try:
                index = rel_types.index(ifc_rel)
            except Exception:
                index = 0

            maxValue = max(len(rel_types) - 1, 1)

            try:
                edgeColor = Color.AnyToHex(
                    Color.ByValueInRange(
                        index,
                        minValue=0,
                        maxValue=maxValue,
                        colorScale=colorScale,
                    )
                )
                e_d[edgeColorKey] = edgeColor
            except Exception:
                pass

            e_dicts.append(e_d)

        try:
            tgraph = TGraph.ByMeshData(
                vertices,
                edges,
                vertexDictionaries=v_dicts,
                edgeDictionaries=e_dicts,
                ontology=True,
                tolerance=tolerance,
            )
        except Exception as exc:
            if not silent:
                print(f"TGraph.ByIFCFile - Error: Could not create TGraph in {importMode} mode. {exc} Returning None.")
            return None

        if not isinstance(tgraph, TGraph):
            return None

        try:
            tgraph._dictionary.setdefault("source", "IFC")
            tgraph._dictionary.setdefault("import_mode", importMode)
            tgraph._dictionary.setdefault("dictionary_mode", dictionaryMode)
            tgraph._dictionary.setdefault("generated_by", "TGraph.ByIFCFile")
        except Exception:
            pass

        return tgraph


    @staticmethod
    def ByIFCPath(
        path,
        importMode: str = "topology",
        clean: bool = False,
        storeBREP: bool = False,
        useInternalVertex: bool = False,
        dictionaryMode: str = "basic",
        includeTypes: list = None,
        excludeTypes: list = None,
        includeRels: list = None,
        excludeRels: list = None,
        xMin: float = -0.5,
        yMin: float = -0.5,
        zMin: float = -0.5,
        xMax: float = 0.5,
        yMax: float = 0.5,
        zMax: float = 0.5,
        vertexColorKey: str = "color",
        edgeColorKey: str = "color",
        colorScale: str = "viridis",
        epsilon: float = 0.01,
        angTolerance: float = 0.1,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Creates a TGraph from an IFC file path.

        Parameters
        ----------
        path : str
            The input IFC file path.
        importMode : str , optional
            Import mode. Options are "topology", "geometry", "triples", and "semantic".
            Default is "topology".
        clean : bool , optional
            If set to True, coplanar faces and collinear edges are removed in geometry
            mode where applicable. Default is False.
        storeBREP : bool , optional
            In geometry mode, if set to True, stores the BREP string of each imported
            topology under the key "BREP". Default is False.
        useInternalVertex : bool , optional
            In geometry mode, if set to True, uses an internal vertex to represent each
            topology. Otherwise, uses its centroid. Default is False.
        dictionaryMode : str , optional
            Options are "none", "basic", "psets", "all", or "full". Default is "basic".
        includeTypes : list , optional
            A list of IFC entity types to include. Default is None.
        excludeTypes : list , optional
            A list of IFC entity types to exclude. Default is None.
        includeRels : list , optional
            A list of IFC relationship types to include. Default is None.
        excludeRels : list , optional
            A list of IFC relationship types to exclude. Default is None.
        xMin : float , optional
            Desired minimum X coordinate for topology layout. Default is -0.5.
        yMin : float , optional
            Desired minimum Y coordinate for topology layout. Default is -0.5.
        zMin : float , optional
            Desired minimum Z coordinate for topology layout. Default is -0.5.
        xMax : float , optional
            Desired maximum X coordinate for topology layout. Default is 0.5.
        yMax : float , optional
            Desired maximum Y coordinate for topology layout. Default is 0.5.
        zMax : float , optional
            Desired maximum Z coordinate for topology layout. Default is 0.5.
        vertexColorKey : str , optional
            The desired vertex dictionary key under which to store the calculated color.
            Default is "color".
        edgeColorKey : str , optional
            The desired edge dictionary key under which to store the calculated color.
            Default is "color".
        colorScale : str , optional
            The desired color scale name. Default is "viridis".
        epsilon : float , optional
            Desired tolerance for removing coplanar faces. Default is 0.01.
        angTolerance : float , optional
            Angular tolerance for removing collinear edges. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The created TGraph, or None if the operation fails.
        """

        if not path:
            if not silent:
                print("TGraph.ByIFCPath - Error: The input path is not a valid path. Returning None.")
            return None

        try:
            import ifcopenshell
            import ifcopenshell.util.placement
            import ifcopenshell.util.element
            import ifcopenshell.util.shape
            import ifcopenshell.geom
        except Exception:
            if not silent:
                print("TGraph.ByIFCPath - Warning: Installing required ifcopenshell library.")
            try:
                import os
                os.system("pip install ifcopenshell")
            except Exception:
                try:
                    import os
                    os.system("pip install ifcopenshell --user")
                except Exception:
                    pass
            try:
                import ifcopenshell
                import ifcopenshell.util.placement
                import ifcopenshell.util.element
                import ifcopenshell.util.shape
                import ifcopenshell.geom
                if not silent:
                    print("TGraph.ByIFCPath - Warning: ifcopenshell library installed correctly.")
            except Exception:
                if not silent:
                    print("TGraph.ByIFCPath - Error: Could not import ifcopenshell. Please install ifcopenshell manually. Returning None.")
                return None

        try:
            file = ifcopenshell.open(path)
        except Exception as exc:
            if not silent:
                print(f"TGraph.ByIFCPath - Error: Could not open the IFC file. {exc} Returning None.")
            return None

        if file is None:
            if not silent:
                print("TGraph.ByIFCPath - Error: Could not open the IFC file. Returning None.")
            return None

        return TGraph.ByIFCFile(
            file,
            importMode=importMode,
            clean=clean,
            storeBREP=storeBREP,
            useInternalVertex=useInternalVertex,
            dictionaryMode=dictionaryMode,
            includeTypes=includeTypes,
            excludeTypes=excludeTypes,
            includeRels=includeRels,
            excludeRels=excludeRels,
            xMin=xMin,
            yMin=yMin,
            zMin=zMin,
            xMax=xMax,
            yMax=yMax,
            zMax=zMax,
            vertexColorKey=vertexColorKey,
            edgeColorKey=edgeColorKey,
            colorScale=colorScale,
            epsilon=epsilon,
            angTolerance=angTolerance,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def ByJSONData(data: Dict[str, Any], ontology: bool = True) -> Optional["TGraph"]:
        """
        Creates a TGraph from JSON-compatible Python data.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data dictionary.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        return TGraph.FromPython(data, ontology=ontology)

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
        """
        Creates a TGraph from a JSON-compatible dictionary.

        Parameters
        ----------
        jsonDictionary : Dict[str, Any]
            The input json dictionary value.
        xKey : str , optional
            The dictionary key to use. Default is 'x'.
        yKey : str , optional
            The dictionary key to use. Default is 'y'.
        zKey : str , optional
            The dictionary key to use. Default is 'z'.
        vertexIDKey : str , optional
            The dictionary key to use. Default is None.
        edgeSourceKey : str , optional
            The dictionary key to use. Default is 'source'.
        edgeTargetKey : str , optional
            The dictionary key to use. Default is 'target'.
        edgeIDKey : str , optional
            The dictionary key to use. Default is None.
        graphPropsKey : str , optional
            The dictionary key to use. Default is 'properties'.
        verticesKey : str , optional
            The dictionary key to use. Default is 'vertices'.
        edgesKey : str , optional
            The dictionary key to use. Default is 'edges'.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """

        if not isinstance(jsonDictionary, dict):
            if not silent:
                print("TGraph.ByJSONDictionary - Error: The input jsonDictionary is not a dictionary. Returning None.")
            return None

        gd = jsonDictionary.get(graphPropsKey, {})
        gd = dict(gd) if isinstance(gd, dict) else {}
        directed = bool(gd.get("directed", False))
        allow_self = bool(gd.get("allowSelfLoops", True))
        allow_parallel = bool(gd.get("allowParallelEdges", True))
        g = TGraph(directed=directed, allowSelfLoops=allow_self, allowParallelEdges=allow_parallel, dictionary=gd)
        id_to_index = {}

        vertices = jsonDictionary.get(verticesKey, []) or []
        for i, rec in enumerate(vertices):
            d = dict(rec) if isinstance(rec, dict) else {}
            vid = d.get(vertexIDKey, d.get("id", i)) if vertexIDKey is not None else d.get("id", i)
            if xKey in d:
                try:
                    d[xKey] = round(float(d[xKey]), mantissa)
                except Exception:
                    pass
            if yKey in d:
                try:
                    d[yKey] = round(float(d[yKey]), mantissa)
                except Exception:
                    pass
            if zKey in d:
                try:
                    d[zKey] = round(float(d[zKey]), mantissa)
                except Exception:
                    pass
            idx = g.AddVertex(dictionary=d)
            id_to_index[vid] = idx
            id_to_index[i] = idx

        for rec in jsonDictionary.get(edgesKey, []) or []:
            if not isinstance(rec, dict):
                continue
            src_id = rec.get(edgeSourceKey, rec.get("src", rec.get("source")))
            dst_id = rec.get(edgeTargetKey, rec.get("dst", rec.get("target")))
            src = id_to_index.get(src_id, src_id if isinstance(src_id, int) and g._validate_vertex_index(src_id) else None)
            dst = id_to_index.get(dst_id, dst_id if isinstance(dst_id, int) and g._validate_vertex_index(dst_id) else None)
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            d = dict(rec)
            directed_edge = bool(d.get("directed", g._directed))
            g.AddEdge(src, dst, directed=directed_edge, dictionary=d)
            if directed_edge:
                g._directed = True

        return TGraph._OntologyAnnotateGraph(
            g, graphClass=g._dictionary.get("ontology_class", "top:Graph"), vertexClass="top:Node",
            edgeClass="top:Relationship", generatedBy="TGraph.ByJSONDictionary", ontology=ontology, silent=silent)

    @staticmethod
    def ByJSONFile(file: Any, **kwargs) -> Optional["TGraph"]:
        """
        Creates a TGraph from a JSON file object.

        Parameters
        ----------
        file : Any
            The input file object.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        try:
            data = json.load(file)
        except Exception:
            return None
        return TGraph.ByJSONDictionary(data, **kwargs)

    @staticmethod
    def ByJSONPath(path: str, silent: bool = False, **kwargs) -> Optional["TGraph"]:
        """
        Creates a TGraph from a JSON file path.

        Parameters
        ----------
        path : str
            The input file path.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
    def ByJSONString(jsonString: str, ontology: bool = True) -> Optional["TGraph"]:
        """
        Creates a TGraph from a JSON string.

        Parameters
        ----------
        jsonString : str
            The input JSON string.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(jsonString, str):
            return None
        try:
            return TGraph.ByJSONData(json.loads(jsonString), ontology=ontology)
        except Exception:
            return None

    @staticmethod
    def ByMeshData(vertices, edges, vertexDictionaries=None, edgeDictionaries=None, ontology: bool = True, tolerance: float = 0.0001) -> "TGraph":
        """
        Creates a TGraph from mesh vertices, edges, and faces data.

        Parameters
        ----------
        vertices : list
            The input vertices or vertex indices.
        edges : list
            The input edges or edge records.
        vertexDictionaries : Any , optional
            The input vertex dictionaries value. Default is None.
        edgeDictionaries : Any , optional
            The input edge dictionaries value. Default is None.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:Graph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.ByMeshData", ontology=ontology, silent=True)

    @staticmethod
    def ByNetworkXGraph(nxGraph: Any, vertexID: str = "id", xKey: str = "x", yKey: str = "y", zKey: str = "z",
                        coordsKey: str = "coords", randomRange: Tuple[float, float] = (-1, 1),
                        mantissa: int = 6, tolerance: float = 0.0001,
                        directed: Optional[bool] = None, allowSelfLoops: bool = True,
                        allowParallelEdges: bool = True, ontology: bool = True) -> Optional["TGraph"]:
        """
        Creates a TGraph from a NetworkX graph.

        Parameters
        ----------
        nxGraph : Any
            The input nx graph value.
        vertexID : str , optional
            The input vertex id value. Default is 'id'.
        xKey : str , optional
            The dictionary key to use. Default is 'x'.
        yKey : str , optional
            The dictionary key to use. Default is 'y'.
        zKey : str , optional
            The dictionary key to use. Default is 'z'.
        coordsKey : str , optional
            The dictionary key to use. Default is 'coords'.
        randomRange : Tuple[float, float] , optional
            The input random range value. Default is (-1, 1).
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is True.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """

        if nxGraph is None:
            return None
        try:
            is_directed = nxGraph.is_directed() if directed is None else bool(directed)
            is_multi = nxGraph.is_multigraph()
        except Exception:
            return None

        graph_dictionary = {
            "source": "networkx",
            "directed": is_directed,
            "multigraph": bool(is_multi),
        }
        g = TGraph(directed=is_directed,
                   allowSelfLoops=allowSelfLoops,
                   allowParallelEdges=bool(allowParallelEdges or is_multi),
                   dictionary=graph_dictionary)
        node_to_index = {}

        try:
            for node, data in nxGraph.nodes(data=True):
                d = dict(data) if isinstance(data, dict) else {}
                d.setdefault(vertexID, node)
                d.setdefault("label", node)
                if coordsKey in d and isinstance(d.get(coordsKey), (list, tuple)) and len(d.get(coordsKey)) >= 3:
                    coords = d.get(coordsKey)
                    d.setdefault(xKey, round(float(coords[0]), mantissa))
                    d.setdefault(yKey, round(float(coords[1]), mantissa))
                    d.setdefault(zKey, round(float(coords[2]), mantissa))
                node_to_index[node] = g.AddVertex(dictionary=d)

            if is_multi:
                for u, v, key, data in nxGraph.edges(keys=True, data=True):
                    if u not in node_to_index or v not in node_to_index:
                        continue
                    d = dict(data) if isinstance(data, dict) else {}
                    d.setdefault("key", key)
                    g.AddEdge(node_to_index[u], node_to_index[v], directed=is_directed, dictionary=d)
            else:
                for u, v, data in nxGraph.edges(data=True):
                    if u not in node_to_index or v not in node_to_index:
                        continue
                    d = dict(data) if isinstance(data, dict) else {}
                    g.AddEdge(node_to_index[u], node_to_index[v], directed=is_directed, dictionary=d)
            return TGraph._OntologyAnnotateGraph(
                g, graphClass="top:Graph", vertexClass="top:Node", edgeClass="top:Relationship",
                generatedBy="TGraph.ByNetworkXGraph", ontology=ontology, silent=True)
        except Exception:
            return None

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
        Creates a spatial relationship graph from the input topologies.

        Parameters
        ----------
        topologies : List[Any]
            The input list of Topologic topologies.
        relationship : str , optional
            The input relationship value. Default is 'intersects'.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is False.
        storeBREP : bool , optional
            If set to True, BREP strings are stored in dictionaries where possible. Default is
            False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:SpatialGraph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.BySpatialRelationships", ontology=True, silent=silent)

    @staticmethod
    def ByTopology(
        topology,
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
        vertexCategoryKey: str = "category",
        edgeCategoryKey: str = "category",
        useInternalVertex: bool = False,
        storeBREP: bool = False,
        ontology: bool = True,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Creates a TGraph from a Topologic topology and selected relationship rules.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input Topologic topology.
        direct : bool , optional
            If set to True, direct topological relationships are included. Default is True.
        directApertures : bool , optional
            If set to True, direct aperture relationships are included. Default is False.
        viaSharedTopologies : bool , optional
            If set to True, relationships through shared topologies are included. Default is
            False.
        viaSharedApertures : bool , optional
            If set to True, relationships through shared apertures are included. Default is
            False.
        toExteriorTopologies : bool , optional
            If set to True, relationships to exterior topologies are included. Default is False.
        toExteriorApertures : bool , optional
            If set to True, relationships to exterior apertures are included. Default is False.
        toContents : bool , optional
            The input to contents value. Default is False.
        toOutposts : bool , optional
            The input to outposts value. Default is False.
        idKey : str , optional
            The dictionary key to use. Default is 'TOPOLOGIC_ID'.
        outpostsKey : str , optional
            The dictionary key to use. Default is 'outposts'.
        vertexCategoryKey : str , optional
            The dictionary key to use. Default is 'category'.
        edgeCategoryKey : str , optional
            The dictionary key to use. Default is 'category'.
        useInternalVertex : bool , optional
            If set to True, an internal vertex is used when deriving topology coordinates.
            Default is False.
        storeBREP : bool , optional
            If set to True, BREP strings are stored in dictionaries where possible. Default is
            False.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The resulting TGraph, or None if the operation fails.
        """

        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Aperture import Aperture

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("TGraph.ByTopology - Error: The input topology parameter is not a valid topology. Returning None.")
            return None

        vertexIndexKey = "index"
        edgeSrcKey = "src"
        edgeDstKey = "dst"

        offset = tolerance * 100.0

        graph_dictionary = {}
        try:
            graph_dictionary = TGraph._TopologyDictionaryToPython(topology)
        except Exception:
            graph_dictionary = {}

        if ontology:
            graph_dictionary.setdefault("ontology_class", "top:SpatialGraph")
            graph_dictionary.setdefault("category", "graph")
            graph_dictionary.setdefault("generated_by", "TGraph.ByTopology")

        graph = TGraph(
            directed=False,
            allowSelfLoops=False,
            allowParallelEdges=False,
            dictionary=graph_dictionary,
        )

        dictionary_cache = {}
        keys_cache = {}
        values_cache = {}
        brep_cache = {}
        representative_cache = {}
        apertures_cache = {}
        contents_cache = {}
        vertices_cache = {}
        edges_cache = {}
        faces_cache = {}
        incidence_cache = {}

        topology_vertex_index_by_object_id = {}
        representative_index_by_key = {}

        T_IsInstance = Topology.IsInstance
        T_CenterOfMass = Topology.CenterOfMass
        T_InternalVertex = Topology.InternalVertex
        T_Dictionary = Topology.Dictionary
        T_Apertures = Topology.Apertures
        T_Contents = Topology.Contents
        T_Vertices = Topology.Vertices
        T_Edges = Topology.Edges
        T_Faces = Topology.Faces
        T_Type = Topology.Type
        T_TypeAsString = Topology.TypeAsString
        T_BREPString = Topology.BREPString

        def _warn(message):
            if not silent:
                print(message)

        def _id(t):
            return id(t)

        def _keys(d):
            if d is None:
                return []
            did = id(d)
            if did in keys_cache:
                return keys_cache[did]
            try:
                ks = Dictionary.Keys(d) or []
            except Exception:
                ks = []
            keys_cache[did] = ks
            return ks

        def _values(d):
            if d is None:
                return []
            did = id(d)
            if did in values_cache:
                return values_cache[did]
            try:
                vs = Dictionary.Values(d) or []
            except Exception:
                vs = []
            values_cache[did] = vs
            return vs

        def _value_at_key(d, key, defaultValue=None):
            if d is None or key is None:
                return defaultValue
            try:
                return Dictionary.ValueAtKey(d, key, defaultValue)
            except TypeError:
                try:
                    value = Dictionary.ValueAtKey(d, key)
                    return defaultValue if value is None else value
                except Exception:
                    return defaultValue
            except Exception:
                return defaultValue

        def _case_key(d, key):
            if d is None or key is None:
                return None
            key_l = str(key).lower()
            for k in _keys(d):
                if str(k).lower() == key_l:
                    return k
            return None

        def _dictionary(t):
            if t is None:
                return None
            tid = _id(t)
            if tid in dictionary_cache:
                return dictionary_cache[tid]
            try:
                d = T_Dictionary(t)
            except Exception:
                d = None
            dictionary_cache[tid] = d
            return d

        def _dictionary_to_python(d):
            result = {}
            if d is None:
                return result
            for k in _keys(d):
                result[k] = _value_at_key(d, k, None)
            return result

        def _topology_dictionary_to_python(t):
            return _dictionary_to_python(_dictionary(t))

        def _brep_data(t):
            tid = _id(t)
            if tid in brep_cache:
                return brep_cache[tid]
            try:
                data = [T_BREPString(t), T_Type(t), T_TypeAsString(t)]
            except Exception:
                data = [None, None, None]
            brep_cache[tid] = data
            return data

        def _add_brep_to_python_dict(d, t):
            if not storeBREP:
                return d
            brep, brep_type, brep_type_string = _brep_data(t)
            if brep is not None:
                d["brep"] = brep
            if brep_type is not None:
                d["brepType"] = brep_type
            if brep_type_string is not None:
                d["brepTypeString"] = brep_type_string
            return d

        def _category_dictionary(t, category):
            d = _topology_dictionary_to_python(t)
            d[vertexCategoryKey] = category
            return _add_brep_to_python_dict(d, t)

        def _topology_from_aperture(t):
            if T_IsInstance(t, "Aperture"):
                try:
                    return Aperture.Topology(t)
                except Exception:
                    return t
            return t

        def _coordinates(v):
            try:
                return [
                    round(float(Vertex.X(v, mantissa=mantissa)), mantissa),
                    round(float(Vertex.Y(v, mantissa=mantissa)), mantissa),
                    round(float(Vertex.Z(v, mantissa=mantissa)), mantissa),
                ]
            except Exception:
                return None

        def _topology_identity_key(t, category=None, apply_offset=False):
            """
            Returns a stable construction key for a Topologic topology.

            Topologic subtopology accessors may return new Python wrapper
            objects for the same underlying geometric/topological entity. Using
            id(t) alone therefore creates duplicate TGraph vertices. This key
            is based on type and rounded vertex coordinates, with BREP fallback
            for higher-dimensional or unusual topologies.
            """
            t = _topology_from_aperture(t)
            if t is None:
                return (None, category, bool(apply_offset))

            try:
                type_string = T_TypeAsString(t)
            except Exception:
                type_string = type(t).__name__

            type_l = str(type_string).lower()

            try:
                if T_IsInstance(t, "Vertex"):
                    return ("Vertex", tuple(_coordinates(t) or []), category, bool(apply_offset))
            except Exception:
                pass

            try:
                vs = T_Vertices(t, silent=True) or []
            except TypeError:
                try:
                    vs = T_Vertices(t) or []
                except Exception:
                    vs = []
            except Exception:
                vs = []

            coords = []
            for v in vs:
                c = _coordinates(v)
                if c is not None:
                    coords.append(tuple(c))

            if coords:
                coords = tuple(sorted(coords))
                return (type_l, coords, category, bool(apply_offset))

            # Fallback. This is slower, but only used when vertices cannot be
            # extracted. Hashing avoids storing very large strings in the cache key.
            try:
                return (type_l, hash(T_BREPString(t)), category, bool(apply_offset))
            except Exception:
                return (type_l, _id(t), category, bool(apply_offset))

        def _add_vertex_record(representation, dictionary):
            d = dict(dictionary) if isinstance(dictionary, dict) else {}

            coords = _coordinates(representation)
            if coords is not None:
                d.setdefault("x", coords[0])
                d.setdefault("y", coords[1])
                d.setdefault("z", coords[2])

            index = graph.AddVertex(dictionary=d, representation=representation)
            graph._vertices[index]["dictionary"][vertexIndexKey] = index

            if representation is not None:
                topology_vertex_index_by_object_id[_id(representation)] = index

            return index

        def _representative_vertex_index(t, category=None, apply_offset=False, source_dictionary_topology=None):
            t = _topology_from_aperture(t)
            source = source_dictionary_topology if source_dictionary_topology is not None else t
            key = _topology_identity_key(t, category=category, apply_offset=apply_offset)

            if key in representative_index_by_key:
                return representative_index_by_key[key]

            if key in representative_cache:
                v = representative_cache[key]
            else:
                try:
                    if useInternalVertex:
                        v = T_InternalVertex(t, tolerance=tolerance)
                    else:
                        v = T_CenterOfMass(t)
                except Exception:
                    v = None

                if v is None:
                    return None

                if apply_offset:
                    try:
                        v = Vertex.ByCoordinates(
                            Vertex.X(v, mantissa=mantissa) + offset,
                            Vertex.Y(v, mantissa=mantissa) + offset,
                            Vertex.Z(v, mantissa=mantissa) + offset,
                        )
                    except Exception:
                        pass

                representative_cache[key] = v

            if category is not None:
                d = _category_dictionary(source, category)
            else:
                d = _topology_dictionary_to_python(source)

            idx = _add_vertex_record(v, d)
            representative_index_by_key[key] = idx
            return idx

        def _append_vertex_from_topology(t, category=None):
            if t is None:
                return None

            key = _topology_identity_key(t, category=category, apply_offset=False)
            if key in representative_index_by_key:
                return representative_index_by_key[key]

            d = _topology_dictionary_to_python(t)
            if category is not None:
                d[vertexCategoryKey] = category
            d = _add_brep_to_python_dict(d, t)

            idx = _add_vertex_record(t, d)
            representative_index_by_key[key] = idx
            return idx

        def _edge_dictionary(relationship, category, source_topology=None):
            source = _topology_from_aperture(source_topology)
            d = {}

            if source is not None:
                d.update(_topology_dictionary_to_python(source))

            d["relationship"] = relationship
            d[edgeCategoryKey] = category
            return d

        def _append_edge_by_indices(src, dst, relationship, category, source_topology=None):
            if src is None or dst is None:
                return None

            if src == dst:
                return None

            d = _edge_dictionary(relationship, category, source_topology)
            d[edgeSrcKey] = src
            d[edgeDstKey] = dst

            return graph.AddEdge(
                src,
                dst,
                directed=False,
                dictionary=d,
                representation=None,
            )

        def _append_edge(v1_index, v2_index, relationship, category, source_topology=None):
            return _append_edge_by_indices(v1_index, v2_index, relationship, category, source_topology=source_topology)

        def _apertures(t):
            tid = _id(t)
            if tid in apertures_cache:
                return apertures_cache[tid]
            try:
                aps = T_Apertures(t) or []
            except Exception:
                aps = []
            apertures_cache[tid] = aps
            return aps

        def _contents(t):
            tid = _id(t)
            if tid in contents_cache:
                return contents_cache[tid]

            try:
                cs = T_Contents(t, silent=True) or []
            except TypeError:
                try:
                    cs = T_Contents(t) or []
                except Exception:
                    cs = []
            except Exception:
                cs = []

            cs = [_topology_from_aperture(c) for c in cs if c is not None]
            contents_cache[tid] = cs
            return cs

        def _vertices(t):
            tid = _id(t)
            if tid in vertices_cache:
                return vertices_cache[tid]
            try:
                vs = T_Vertices(t, silent=True) or []
            except TypeError:
                try:
                    vs = T_Vertices(t) or []
                except Exception:
                    vs = []
            except Exception:
                vs = []
            vertices_cache[tid] = vs
            return vs

        def _edges(t):
            tid = _id(t)
            if tid in edges_cache:
                return edges_cache[tid]
            try:
                es = T_Edges(t, silent=True) or []
            except TypeError:
                try:
                    es = T_Edges(t) or []
                except Exception:
                    es = []
            except Exception:
                es = []
            edges_cache[tid] = es
            return es

        def _faces(t):
            tid = _id(t)
            if tid in faces_cache:
                return faces_cache[tid]
            try:
                fs = T_Faces(t, silent=True) or []
            except TypeError:
                try:
                    fs = T_Faces(t) or []
                except Exception:
                    fs = []
            except Exception:
                fs = []
            faces_cache[tid] = fs
            return fs

        def _coord_key(v):
            return (
                round(Vertex.X(v, mantissa=mantissa), mantissa),
                round(Vertex.Y(v, mantissa=mantissa), mantissa),
                round(Vertex.Z(v, mantissa=mantissa), mantissa),
            )

        def _vertex_key(v):
            return _coord_key(v)

        def _edge_key(e):
            return tuple(sorted([_coord_key(v) for v in _vertices(e)]))

        def _face_key(f):
            return tuple(sorted([_coord_key(v) for v in _vertices(f)]))

        def _boundary_key(t, boundary_type):
            if boundary_type == "Face":
                return _face_key(t)
            if boundary_type == "Edge":
                return _edge_key(t)
            if boundary_type == "Vertex":
                return _vertex_key(t)
            return _id(t)

        def _children(t, child_type):
            if child_type == "Face":
                return _faces(t)
            if child_type == "Edge":
                return _edges(t)
            if child_type == "Vertex":
                return _vertices(t)
            return []

        def _safe_list(value):
            return value if isinstance(value, list) else ([] if value is None else list(value))

        def _subtopologies(t, topology_type, free=False):
            try:
                if topology_type == "CellComplex":
                    return Topology.CellComplexes(t, silent=True) or []
                if topology_type == "Cell":
                    return (Cluster.FreeCells(t, tolerance=tolerance) if free else Topology.Cells(t, silent=True)) or []
                if topology_type == "Shell":
                    return (Cluster.FreeShells(t, tolerance=tolerance) if free else Topology.Shells(t, silent=True)) or []
                if topology_type == "Face":
                    return (Cluster.FreeFaces(t, tolerance=tolerance) if free else Topology.Faces(t, silent=True)) or []
                if topology_type == "Wire":
                    return (Cluster.FreeWires(t, tolerance=tolerance) if free else Topology.Wires(t, silent=True)) or []
                if topology_type == "Edge":
                    return (Cluster.FreeEdges(t, tolerance=tolerance) if free else Topology.Edges(t, silent=True)) or []
                if topology_type == "Vertex":
                    return (Cluster.FreeVertices(t, tolerance=tolerance) if free else Topology.Vertices(t, silent=True)) or []
            except TypeError:
                try:
                    if topology_type == "CellComplex":
                        return Topology.CellComplexes(t) or []
                    if topology_type == "Cell":
                        return Topology.Cells(t) or []
                    if topology_type == "Shell":
                        return Topology.Shells(t) or []
                    if topology_type == "Face":
                        return Topology.Faces(t) or []
                    if topology_type == "Wire":
                        return Topology.Wires(t) or []
                    if topology_type == "Edge":
                        return Topology.Edges(t) or []
                    if topology_type == "Vertex":
                        return Topology.Vertices(t) or []
                except Exception:
                    return []
            except Exception:
                return []
            return []

        def _build_incidence(host, owners, child_type):
            key = (_id(host), tuple(_id(o) for o in owners), child_type)
            if key in incidence_cache:
                return incidence_cache[key]

            boundary_to_owners = {}
            boundary_to_refs = {}
            boundary_to_owner_ids = {}

            for owner in owners:
                owner_id = _id(owner)

                for child in _children(owner, child_type):
                    bk = _boundary_key(child, child_type)

                    owner_ids = boundary_to_owner_ids.setdefault(bk, set())
                    if owner_id not in owner_ids:
                        owner_ids.add(owner_id)
                        boundary_to_owners.setdefault(bk, []).append(owner)

                    boundary_to_refs.setdefault(bk, []).append(child)

            incidence_cache[key] = (boundary_to_owners, boundary_to_refs)
            return boundary_to_owners, boundary_to_refs

        def _add_related_vertices(v_from_index, related_topologies, vertex_category, relationship, edge_category, apply_offset=False):
            if v_from_index is None:
                return

            for t in related_topologies:
                v_to_index = _representative_vertex_index(t, vertex_category, apply_offset=apply_offset)
                if v_to_index is None:
                    continue
                _append_edge(v_from_index, v_to_index, relationship, edge_category, source_topology=t)

        def _add_contents(v_from_index, contents):
            if v_from_index is None:
                return

            for content in contents:
                content = _topology_from_aperture(content)
                if content is None:
                    continue

                v_to_index = _representative_vertex_index(content, 5, apply_offset=True)
                if v_to_index is None:
                    continue

                _append_edge(
                    v_from_index,
                    v_to_index,
                    "To_Contents",
                    5,
                    source_topology=content,
                )

        def _outpost_lookup(topologies):
            lookup = {}
            id_key_l = str(idKey).lower()

            for t in topologies:
                d = _dictionary(t)
                k = None

                for key in _keys(d):
                    if str(key).lower() == id_key_l:
                        k = key
                        break

                if k is None:
                    continue

                value = _value_at_key(d, k, None)
                if value is not None and value not in lookup:
                    lookup[value] = t

            return lookup

        def _add_outposts(v_from_index, outpost_lookup):
            if v_from_index is None:
                return

            try:
                v_record = graph._vertices[v_from_index]
                d_python = v_record.get("dictionary", {})
            except Exception:
                return

            k = None
            outposts_key_l = str(outpostsKey).lower()

            for key in d_python.keys():
                if str(key).lower() == outposts_key_l:
                    k = key
                    break

            if k is None:
                return

            ids = d_python.get(k, [])
            if ids is None:
                return
            if not isinstance(ids, list):
                ids = [ids]

            for an_id in ids:
                outpost = outpost_lookup.get(an_id, None)
                if outpost is None:
                    continue
                v_to_index = _representative_vertex_index(outpost, 6)
                if v_to_index is None:
                    continue
                _append_edge(v_from_index, v_to_index, "To_Outposts", 6)

        def _classify_boundaries(owner, child_type, boundary_to_owners):
            shared_tops = []
            exterior_tops = []
            shared_aps = []
            exterior_aps = []

            for child in _children(owner, child_type):
                bk = _boundary_key(child, child_type)
                aps = _apertures(child)

                if len(boundary_to_owners.get(bk, [])) > 1:
                    shared_tops.append(child)
                    shared_aps.extend(aps)
                else:
                    exterior_tops.append(child)
                    exterior_aps.extend(aps)

            return shared_tops, exterior_tops, shared_aps, exterior_aps

        def _add_direct_edges_from_incidence(owners, child_type, boundary_to_owners, boundary_to_refs, require_aperture=False):
            owner_records = {}

            for owner in owners:
                owner_id = _id(owner)
                v_owner_index = _representative_vertex_index(owner, 0)
                if v_owner_index is None:
                    continue

                owner_records[owner_id] = {
                    "owner": owner,
                    "index": v_owner_index,
                }

            seen_pairs = set()

            for bk, incident_owners in boundary_to_owners.items():
                if not incident_owners:
                    continue

                unique_records = []
                seen_owner_ids = set()
                seen_indices = set()

                for owner in incident_owners:
                    owner_id = _id(owner)
                    if owner_id in seen_owner_ids:
                        continue

                    record = owner_records.get(owner_id, None)
                    if record is None:
                        continue

                    index = record["index"]
                    if index in seen_indices:
                        continue

                    seen_owner_ids.add(owner_id)
                    seen_indices.add(index)
                    unique_records.append(record)

                if len(unique_records) < 2:
                    continue

                refs = boundary_to_refs.get(bk, []) or []

                if require_aperture:
                    has_aperture = False
                    for ref in refs:
                        if _apertures(ref):
                            has_aperture = True
                            break
                    if not has_aperture:
                        continue

                relationship = "Direct_Apertures" if require_aperture else "Direct"
                edge_category = 2 if require_aperture else 0
                n = len(unique_records)

                for i in range(n - 1):
                    ri = unique_records[i]
                    src_index = ri["index"]

                    for j in range(i + 1, n):
                        rj = unique_records[j]
                        dst_index = rj["index"]

                        if src_index == dst_index:
                            continue

                        pair = (src_index, dst_index) if src_index <= dst_index else (dst_index, src_index)
                        pair = pair + (bk, "aperture" if require_aperture else "direct")

                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        source = None

                        if require_aperture:
                            for ref in refs:
                                aps = _apertures(ref)
                                if aps:
                                    source = aps[0]
                                    break
                        else:
                            source = refs[0] if refs else None

                        _append_edge_by_indices(
                            src_index,
                            dst_index,
                            relationship,
                            edge_category,
                            source_topology=source,
                        )

        def _process_collection(host, main_type, child_type, outpost_lookup):
            owners = _subtopologies(host, main_type)
            boundary_to_owners, boundary_to_refs = _build_incidence(host, owners, child_type)

            for owner in owners:
                _representative_vertex_index(owner, 0)

            if direct:
                _add_direct_edges_from_incidence(
                    owners,
                    child_type,
                    boundary_to_owners,
                    boundary_to_refs,
                    require_aperture=False,
                )

            if directApertures:
                _add_direct_edges_from_incidence(
                    owners,
                    child_type,
                    boundary_to_owners,
                    boundary_to_refs,
                    require_aperture=True,
                )

            if not any([viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts]):
                return

            for owner in owners:
                v_owner_index = _representative_vertex_index(owner, 0)

                shared_tops, exterior_tops, shared_aps, exterior_aps = _classify_boundaries(
                    owner,
                    child_type,
                    boundary_to_owners,
                )

                if viaSharedTopologies:
                    _add_related_vertices(v_owner_index, shared_tops, 1, "Via_Shared_Topologies", 1)

                    if toContents or toOutposts:
                        for shared_top in shared_tops:
                            v_shared_index = _representative_vertex_index(shared_top, 1)

                            if toContents:
                                _add_contents(v_shared_index, _contents(shared_top))
                            if toOutposts:
                                _add_outposts(v_shared_index, outpost_lookup)

                if viaSharedApertures:
                    _add_related_vertices(v_owner_index, shared_aps, 2, "Via_Shared_Apertures", 2, apply_offset=True)

                if toExteriorTopologies:
                    _add_related_vertices(v_owner_index, exterior_tops, 3, "To_Exterior_Topologies", 3)

                    if toContents or toOutposts:
                        for exterior_top in exterior_tops:
                            v_exterior_index = _representative_vertex_index(exterior_top, 3)

                            if toContents:
                                _add_contents(v_exterior_index, _contents(exterior_top))
                            if toOutposts:
                                _add_outposts(v_exterior_index, outpost_lookup)

                if toExteriorApertures:
                    _add_related_vertices(v_owner_index, exterior_aps, 4, "To_Exterior_Apertures", 4, apply_offset=True)

                if toContents:
                    _add_contents(v_owner_index, _contents(owner))

                if toOutposts:
                    _add_outposts(v_owner_index, outpost_lookup)

        def _process_single(t, child_type, outpost_lookup):
            v_index = _representative_vertex_index(t, 0)

            if not any([toExteriorTopologies, toExteriorApertures, toContents, toOutposts]):
                return

            exterior_tops = _children(t, child_type)
            exterior_aps = []

            for exterior_top in exterior_tops:
                exterior_aps.extend(_apertures(exterior_top))

            if toExteriorTopologies:
                _add_related_vertices(v_index, exterior_tops, 3, "To_Exterior_Topologies", 3)

                if toContents or toOutposts:
                    for exterior_top in exterior_tops:
                        v_exterior_index = _representative_vertex_index(exterior_top, 3)

                        if toContents:
                            _add_contents(v_exterior_index, _contents(exterior_top))
                        if toOutposts:
                            _add_outposts(v_exterior_index, outpost_lookup)

            if toExteriorApertures:
                _add_related_vertices(v_index, exterior_aps, 4, "To_Exterior_Apertures", 4, apply_offset=True)

            if toContents:
                _add_contents(v_index, _contents(t))

            if toOutposts:
                _add_outposts(v_index, outpost_lookup)

        def _process_vertex(t, outpost_lookup):
            v_index = _append_vertex_from_topology(t, 0)

            if toContents:
                _add_contents(v_index, _contents(t))

            if toOutposts:
                _add_outposts(v_index, outpost_lookup)

        def _safe_topology_list(fn, t):
            try:
                return fn(t, silent=True) or []
            except TypeError:
                try:
                    return fn(t) or []
                except Exception:
                    return []
            except Exception:
                return []

        def _all_subtopologies(t):
            return (
                _safe_topology_list(Topology.CellComplexes, t) +
                _safe_topology_list(Topology.Cells, t) +
                _safe_topology_list(Topology.Shells, t) +
                _safe_topology_list(Topology.Faces, t) +
                _safe_topology_list(Topology.Wires, t) +
                _safe_topology_list(Topology.Edges, t) +
                _safe_topology_list(Topology.Vertices, t)
            )

        others = _all_subtopologies(topology)
        outposts = _outpost_lookup(others)

        if T_IsInstance(topology, "CellComplex"):
            _process_collection(topology, "Cell", "Face", outposts)

        elif T_IsInstance(topology, "Cell"):
            _process_single(topology, "Face", outposts)

        elif T_IsInstance(topology, "Shell"):
            _process_collection(topology, "Face", "Edge", outposts)

        elif T_IsInstance(topology, "Face"):
            _process_single(topology, "Edge", outposts)

        elif T_IsInstance(topology, "Wire"):
            _process_collection(topology, "Edge", "Vertex", outposts)

        elif T_IsInstance(topology, "Edge"):
            _process_single(topology, "Vertex", outposts)

        elif T_IsInstance(topology, "Vertex"):
            _process_vertex(topology, outposts)

        elif T_IsInstance(topology, "Cluster"):
            c_cellComplexes = _safe_topology_list(Topology.CellComplexes, topology)

            try:
                c_cells = Cluster.FreeCells(topology, tolerance=tolerance) or []
            except Exception:
                c_cells = []

            try:
                c_shells = Cluster.FreeShells(topology, tolerance=tolerance) or []
            except Exception:
                c_shells = []

            try:
                c_faces = Cluster.FreeFaces(topology, tolerance=tolerance) or []
            except Exception:
                c_faces = []

            try:
                c_wires = Cluster.FreeWires(topology, tolerance=tolerance) or []
            except Exception:
                c_wires = []

            try:
                c_edges = Cluster.FreeEdges(topology, tolerance=tolerance) or []
            except Exception:
                c_edges = []

            try:
                c_vertices = Cluster.FreeVertices(topology, tolerance=tolerance) or []
            except Exception:
                c_vertices = []

            others = others + c_cellComplexes + c_cells + c_shells + c_faces + c_wires + c_edges + c_vertices
            outposts = _outpost_lookup(others)

            for t in c_cellComplexes:
                _process_collection(t, "Cell", "Face", outposts)
            for t in c_cells:
                _process_single(t, "Face", outposts)
            for t in c_shells:
                _process_collection(t, "Face", "Edge", outposts)
            for t in c_faces:
                _process_single(t, "Edge", outposts)
            for t in c_wires:
                _process_collection(t, "Edge", "Vertex", outposts)
            for t in c_edges:
                _process_single(t, "Vertex", outposts)
            for t in c_vertices:
                _process_vertex(t, outposts)

        else:
            return None

        if ontology:
            try:
                if hasattr(TGraph, "AnnotateOntology"):
                    TGraph.AnnotateOntology(
                        graph,
                        ontologyClass="top:SpatialGraph",
                        category="graph",
                        generatedBy="TGraph.ByTopology",
                        silent=True,
                    )
                else:
                    graph._dictionary["ontology_class"] = "top:SpatialGraph"
                    graph._dictionary["category"] = "graph"
                    graph._dictionary["generated_by"] = "TGraph.ByTopology"
            except Exception:
                graph._dictionary["ontology_class"] = "top:SpatialGraph"
                graph._dictionary["category"] = "graph"
                graph._dictionary["generated_by"] = "TGraph.ByTopology"

        return graph

    @staticmethod
    def ByTriples(
        triples: Optional[List[Tuple[Any, Any, Any]]] = None,
        directed: bool = True,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        vertexLabelKey: str = "label",
        edgeLabelKey: str = "label",
        ontology: bool = True,
    ) -> "TGraph":
        """
        Creates a directed TGraph from subject-relationship-object triples.

        Parameters
        ----------
        triples : Optional[List[Tuple[Any, Any, Any]]] , optional
            The input triples value. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is True.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        edgeLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:KnowledgeGraph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.ByTriples", ontology=ontology, silent=True)

    @staticmethod
    def ByVerticesEdges(
        vertices: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = False,
        dictionary: Optional[Dict[str, Any]] = None,
        vertexIDKey: str = "id",
        edgeSRCKey: str = "src",
        edgeDSTKey: str = "dst",
        edgeSourceKey: str = "source",
        edgeTargetKey: str = "target",
        edgeSRCIDKey: str = "src_id",
        edgeDSTIDKey: str = "dst_id",
        edgeSourceIDKey: str = "source_id",
        edgeTargetIDKey: str = "target_id",
        storeRepresentations: bool = True,
        allowGeometricFallback: bool = True,
        ambiguousEndpointPolicy: str = "warn",  # "warn", "skip", "first", "error"
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
        ontology: bool = True,
    ) -> "TGraph":
        """
        Creates a TGraph from input vertices and edges.

        Parameters
        ----------
        vertices : Optional[List[Any]] , optional
            The input vertices or vertex indices. Default is None.
        edges : Optional[List[Any]] , optional
            The input edges or edge records. Default is None.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is False.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        vertexIDKey : str , optional
            The dictionary key to use. Default is 'id'.
        edgeSRCKey : str , optional
            The dictionary key to use. Default is 'src'.
        edgeDSTKey : str , optional
            The dictionary key to use. Default is 'dst'.
        edgeSourceKey : str , optional
            The dictionary key to use. Default is 'source'.
        edgeTargetKey : str , optional
            The dictionary key to use. Default is 'target'.
        edgeSRCIDKey : str , optional
            The dictionary key to use. Default is 'src_id'.
        edgeDSTIDKey : str , optional
            The dictionary key to use. Default is 'dst_id'.
        edgeSourceIDKey : str , optional
            The dictionary key to use. Default is 'source_id'.
        edgeTargetIDKey : str , optional
            The dictionary key to use. Default is 'target_id'.
        storeRepresentations : bool , optional
            If set to True, input objects are stored as graph representations. Default is True.
        allowGeometricFallback : bool , optional
            The input allow geometric fallback value. Default is True.
        ambiguousEndpointPolicy : str , optional
            The input ambiguous endpoint policy value. Default is 'warn'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """

        vertices = vertices or []
        edges = edges or []

        g = TGraph(
            directed=directed,
            allowSelfLoops=allowSelfLoops,
            allowParallelEdges=allowParallelEdges,
            dictionary=dictionary,
        )

        ambiguousEndpointPolicy = str(ambiguousEndpointPolicy or "warn").lower()
        if ambiguousEndpointPolicy not in ("warn", "skip", "first", "error"):
            ambiguousEndpointPolicy = "warn"

        def _warn(message: str) -> None:
            if not silent:
                print(message)

        def _is_int(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        def _valid_index(index: Any) -> bool:
            return _is_int(index) and 0 <= index < len(vertices)

        def _dictionary_to_python(obj: Any) -> Dict[str, Any]:
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return dict(obj)
            try:
                return TGraph._TopologyDictionaryToPython(obj)
            except Exception:
                pass
            try:
                from topologicpy.Topology import Topology
                from topologicpy.Dictionary import Dictionary

                d = Topology.Dictionary(obj)
                keys = Dictionary.Keys(d) or []
                return {k: Dictionary.ValueAtKey(d, k, None) for k in keys}
            except TypeError:
                try:
                    from topologicpy.Topology import Topology
                    from topologicpy.Dictionary import Dictionary

                    d = Topology.Dictionary(obj)
                    keys = Dictionary.Keys(d) or []
                    return {k: Dictionary.ValueAtKey(d, k) for k in keys}
                except Exception:
                    return {}
            except Exception:
                return {}

        def _is_topologic_instance(obj: Any, type_name: str) -> bool:
            try:
                from topologicpy.Topology import Topology
                return bool(Topology.IsInstance(obj, type_name))
            except Exception:
                return False

        def _coordinates(obj: Any) -> Optional[List[float]]:
            if obj is None:
                return None

            if isinstance(obj, dict):
                if all(k in obj for k in ("x", "y", "z")):
                    try:
                        return [
                            round(float(obj["x"]), mantissa),
                            round(float(obj["y"]), mantissa),
                            round(float(obj["z"]), mantissa),
                        ]
                    except Exception:
                        pass

            try:
                from topologicpy.Vertex import Vertex
                from topologicpy.Topology import Topology

                if Topology.IsInstance(obj, "Vertex"):
                    c = Vertex.Coordinates(obj)
                    if c and len(c) >= 3:
                        return [
                            round(float(c[0]), mantissa),
                            round(float(c[1]), mantissa),
                            round(float(c[2]), mantissa),
                        ]

                if Topology.IsInstance(obj, "Topology"):
                    v = None
                    try:
                        v = Topology.CenterOfMass(obj)
                    except Exception:
                        v = None
                    if v is not None:
                        c = Vertex.Coordinates(v)
                        if c and len(c) >= 3:
                            return [
                                round(float(c[0]), mantissa),
                                round(float(c[1]), mantissa),
                                round(float(c[2]), mantissa),
                            ]
            except Exception:
                pass

            return None

        def _coord_key(coords: Optional[List[float]]) -> Optional[Tuple[int, int, int]]:
            if coords is None:
                return None
            try:
                scale = 1.0 / float(tolerance) if tolerance > 0 else 10000.0
                return (
                    int(round(float(coords[0]) * scale)),
                    int(round(float(coords[1]) * scale)),
                    int(round(float(coords[2]) * scale)),
                )
            except Exception:
                return None

        def _edge_start_end(edge_obj: Any) -> Tuple[Any, Any]:
            if edge_obj is None:
                return None, None

            if isinstance(edge_obj, dict):
                return None, None

            if isinstance(edge_obj, (list, tuple)) and len(edge_obj) >= 2:
                return edge_obj[0], edge_obj[1]

            try:
                from topologicpy.Edge import Edge
                return Edge.StartVertex(edge_obj), Edge.EndVertex(edge_obj)
            except Exception:
                pass

            try:
                from topologicpy.Topology import Topology
                sv = Topology.StartVertex(edge_obj)
                ev = Topology.EndVertex(edge_obj)
                return sv, ev
            except Exception:
                pass

            return None, None

        def _value_at_keys(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
            if not isinstance(d, dict):
                return default
            for key in keys:
                if key is not None and key in d:
                    return d.get(key)
            return default

        # ------------------------------------------------------------------
        # Add vertices and build endpoint-resolution indices.
        # ------------------------------------------------------------------

        object_id_to_index: Dict[int, int] = {}
        vertex_id_to_index: Dict[Any, int] = {}
        coord_to_indices: Dict[Tuple[int, int, int], List[int]] = {}

        for i, item in enumerate(vertices):
            d = _dictionary_to_python(item)

            coords = _coordinates(item)
            if coords is not None:
                d.setdefault("x", coords[0])
                d.setdefault("y", coords[1])
                d.setdefault("z", coords[2])

            representation = item if storeRepresentations else None
            if isinstance(item, dict):
                representation = item.get("representation", None) if storeRepresentations else None

            idx = g.AddVertex(dictionary=d, representation=representation)

            object_id_to_index[id(item)] = idx

            v_id = d.get(vertexIDKey, None)
            if v_id is not None:
                vertex_id_to_index[v_id] = idx

            # Also support common ID aliases without requiring the caller to set
            # vertexIDKey differently.
            for alias in ("id", "ID", "uuid", "guid", "ifc_guid", "GlobalId", "name", "label"):
                value = d.get(alias, None)
                if value is not None and value not in vertex_id_to_index:
                    vertex_id_to_index[value] = idx

            ck = _coord_key(coords)
            if ck is not None:
                coord_to_indices.setdefault(ck, []).append(idx)

        def _resolve_by_geometry(endpoint: Any, role: str, edge_index: int) -> Optional[int]:
            if not allowGeometricFallback:
                return None

            coords = _coordinates(endpoint)
            ck = _coord_key(coords)
            if ck is None:
                return None

            matches = coord_to_indices.get(ck, [])

            if len(matches) == 1:
                return matches[0]

            if len(matches) > 1:
                message = (
                    "TGraph.ByVerticesEdges - Warning: Ambiguous geometric endpoint "
                    f"for edge {edge_index}, role '{role}'. "
                    f"Coordinates {coords} match vertex indices {matches}."
                )

                if ambiguousEndpointPolicy == "error":
                    raise ValueError(message)

                if ambiguousEndpointPolicy == "first":
                    if not silent:
                        print(message + " Using first match.")
                    return matches[0]

                if ambiguousEndpointPolicy in ("warn", "skip"):
                    if ambiguousEndpointPolicy == "warn":
                        _warn(message + " Skipping edge.")
                    return None

            return None

        def _resolve_endpoint(
            endpoint: Any,
            edge_dict: Dict[str, Any],
            role: str,
            edge_index: int,
        ) -> Optional[int]:
            """
            Resolves an edge endpoint to a TGraph vertex index.

            role must be "src" or "dst".
            """

            if role == "src":
                index_keys = [edgeSRCKey, edgeSourceKey]
                id_keys = [edgeSRCIDKey, edgeSourceIDKey]
            else:
                index_keys = [edgeDSTKey, edgeTargetKey]
                id_keys = [edgeDSTIDKey, edgeTargetIDKey]

            # 1. Explicit index keys.
            for key in index_keys:
                value = edge_dict.get(key, None)
                if _valid_index(value):
                    return int(value)

            # 2. Explicit ID keys.
            for key in id_keys:
                value = edge_dict.get(key, None)
                if value in vertex_id_to_index:
                    return vertex_id_to_index[value]

            # 3. source/target may be IDs rather than indices.
            for key in index_keys:
                value = edge_dict.get(key, None)
                if value in vertex_id_to_index:
                    return vertex_id_to_index[value]

            # 4. Endpoint itself may be an index.
            if _valid_index(endpoint):
                return int(endpoint)

            # 5. Endpoint itself may be an ID.
            if endpoint in vertex_id_to_index:
                return vertex_id_to_index[endpoint]

            # 6. Object identity.
            if endpoint is not None and id(endpoint) in object_id_to_index:
                return object_id_to_index[id(endpoint)]

            # 7. Geometric fallback only if unambiguous.
            return _resolve_by_geometry(endpoint, role=role, edge_index=edge_index)

        # ------------------------------------------------------------------
        # Add edges.
        # ------------------------------------------------------------------

        for edge_index, edge_item in enumerate(edges):
            edge_dict = _dictionary_to_python(edge_item)
            representation = edge_item if storeRepresentations else None

            # Tuple/list pairs are interpreted as endpoint references unless they
            # look like Topologic control data.
            if isinstance(edge_item, (list, tuple)) and len(edge_item) >= 2:
                start_obj, end_obj = edge_item[0], edge_item[1]
                if not edge_dict:
                    edge_dict = {}
            else:
                start_obj, end_obj = _edge_start_end(edge_item)

            src = _resolve_endpoint(start_obj, edge_dict, "src", edge_index)
            dst = _resolve_endpoint(end_obj, edge_dict, "dst", edge_index)

            if src is None or dst is None:
                if not silent:
                    _warn(
                        "TGraph.ByVerticesEdges - Warning: Could not resolve "
                        f"endpoints for edge {edge_index}. Skipping edge."
                    )
                continue

            # Make endpoint metadata explicit in the stored edge dictionary.
            edge_dict.setdefault(edgeSRCKey, src)
            edge_dict.setdefault(edgeDSTKey, dst)

            g.AddEdge(
                src,
                dst,
                directed=directed,
                dictionary=edge_dict,
                representation=representation,
            )

        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:Graph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.ByVerticesEdges", ontology=ontology, silent=silent)

    @staticmethod
    def CategoryByOntologyClass(ontologyClass: str, defaultValue: Any = None) -> Any:
        """
        Returns the ontology category corresponding to the input ontology class.

        Parameters
        ----------
        ontologyClass : str
            The ontology class value.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        Any
            The resulting category by ontology class object or value.
        """
        if ontologyClass is None:
            return defaultValue
        return TGraph._OntologyConfig()["categories"].get(str(ontologyClass).strip(), defaultValue)





    # START CSV-RELATED METHODS
    @staticmethod
    def _CSVFlatten(items: Any) -> List[Any]:
        """
        Flattens nested lists/tuples for CSV feature-key handling.

        Parameters
        ----------
        items : Any
            The input item or nested list/tuple of items.

        Returns
        -------
        list
            The flattened list.
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
        Returns CSV feature header names.

        If a supplied feature key already includes the requested prefix, it is
        returned unchanged. This prevents headers such as ``feat_feat_area`` when
        callers pass keys that are already named ``feat_area``.

        Parameters
        ----------
        prefix : str
            The feature-column prefix.
        featureKeys : list
            The feature dictionary keys.

        Returns
        -------
        list
            The CSV feature headers.
        """

        headers = []
        prefix = str(prefix) if prefix is not None else "feat"
        prefix_with_sep = prefix + "_"

        for key in TGraph._CSVFlatten(featureKeys):
            key = str(key)
            if key.startswith(prefix_with_sep):
                headers.append(key)
            else:
                headers.append(f"{prefix}_{key}")
        return headers


    @staticmethod
    def _CSVFeatureKeysFromHeaders(headers: List[str], prefix: str) -> List[str]:
        """
        Derives feature keys from CSV headers.

        Parameters
        ----------
        headers : list
            The CSV headers.
        prefix : str
            The feature-column prefix.

        Returns
        -------
        list
            The derived feature keys.
        """

        if not isinstance(headers, list):
            return []

        prefix = str(prefix) if prefix is not None else "feat"
        prefix_with_sep = prefix + "_"

        keys = []
        for header in headers:
            if not isinstance(header, str):
                continue
            if header.startswith(prefix_with_sep):
                keys.append(header[len(prefix_with_sep):])
        return keys


    @staticmethod
    def _CSVFeatureValues(dictionary: Optional[Dict[str, Any]], featureKeys: List[Any], mantissa: int = 6) -> List[float]:
        """
        Returns a stable numeric feature vector. Missing or invalid values become 0.0.

        Parameters
        ----------
        dictionary : dict
            The source dictionary.
        featureKeys : list
            The dictionary keys to read.
        mantissa : int , optional
            The number of decimal places to round values to. Default is 6.

        Returns
        -------
        list
            The numeric feature vector.
        """

        if not featureKeys:
            return []

        d = dictionary if isinstance(dictionary, dict) else {}
        values = []

        for key in TGraph._CSVFlatten(featureKeys):
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
        Returns a label value from a dictionary with a fallback.

        Parameters
        ----------
        dictionary : dict
            The source dictionary.
        key : str
            The dictionary key.
        defaultValue : Any
            The default value.

        Returns
        -------
        Any
            The label value.
        """

        d = dictionary if isinstance(dictionary, dict) else {}

        if key is None:
            return defaultValue

        value = d.get(key, None)
        return defaultValue if value is None else value


    @staticmethod
    def _CSVBool(value: Any, default: bool = False) -> bool:
        """
        Converts common CSV boolean values to bool.

        Parameters
        ----------
        value : Any
            The input value.
        default : bool , optional
            The fallback value. Default is False.

        Returns
        -------
        bool
            The converted boolean value.
        """

        if isinstance(value, bool):
            return value

        if value is None:
            return default

        if isinstance(value, (int, float)):
            return bool(int(value))

        s = str(value).strip().lower()

        if s in ["1", "true", "t", "yes", "y"]:
            return True
        if s in ["0", "false", "f", "no", "n", ""]:
            return False

        return default


    @staticmethod
    def _CSVValue(value: Any) -> Any:
        """
        Converts a CSV string value into a Python value.

        Parameters
        ----------
        value : Any
            The input CSV value.

        Returns
        -------
        Any
            The converted value.
        """

        if value is None:
            return None

        if not isinstance(value, str):
            return value

        s = value.strip()

        if s == "":
            return None

        sl = s.lower()

        if sl in ["true", "t", "yes", "y"]:
            return True
        if sl in ["false", "f", "no", "n"]:
            return False
        if sl in ["none", "null"]:
            return None

        try:
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                import json
                return json.loads(s)
        except Exception:
            pass

        try:
            if "." not in s and "e" not in sl:
                return int(s)
        except Exception:
            pass

        try:
            return float(s)
        except Exception:
            return value


    @staticmethod
    def _CSVExportValue(value: Any) -> Any:
        """
        Converts a Python value to a CSV-safe scalar.

        Parameters
        ----------
        value : Any
            The input value.

        Returns
        -------
        Any
            The CSV-safe value.
        """

        if value is None:
            return ""

        if isinstance(value, bool):
            return 1 if value else 0

        if isinstance(value, (str, int, float)):
            return value

        try:
            import json
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)


    @staticmethod
    def _CSVMaskFromDictionaryOrRatio(
        dictionary: Optional[Dict[str, Any]],
        maskKey: Optional[str],
        trainMax: int,
        validateMax: int,
        counts: Dict[str, int],
    ) -> Tuple[bool, bool, bool]:
        """
        Returns train/validation/test booleans using either an explicit dictionary
        mask value or deterministic split counts.

        Parameters
        ----------
        dictionary : dict
            The source dictionary.
        maskKey : str
            The dictionary key to read. Values 0, 1, and 2 mean train, validation,
            and test respectively.
        trainMax : int
            Maximum number of items assigned to train before falling through.
        validateMax : int
            Maximum number of items assigned to validation before falling through.
        counts : dict
            Mutable split counts.

        Returns
        -------
        tuple
            A tuple of booleans: (train_mask, val_mask, test_mask).
        """

        d = dictionary if isinstance(dictionary, dict) else {}

        if maskKey is not None:
            value = d.get(maskKey, None)

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

            trainValue = d.get("train_mask", None)
            valValue = d.get("val_mask", None)
            testValue = d.get("test_mask", None)

            if trainValue is not None or valValue is not None or testValue is not None:
                train = TGraph._CSVBool(trainValue, default=False)
                val = TGraph._CSVBool(valValue, default=False)
                test = TGraph._CSVBool(testValue, default=False)

                if train:
                    counts["train"] += 1
                elif val:
                    counts["val"] += 1
                elif test:
                    counts["test"] += 1

                return train, val, test

        if counts["train"] < trainMax:
            counts["train"] += 1
            return True, False, False

        if counts["val"] < validateMax:
            counts["val"] += 1
            return False, True, False

        counts["test"] += 1
        return False, False, True


    @staticmethod
    def _CSVReadRows(path: str) -> List[Dict[str, Any]]:
        """
        Reads a CSV file into a list of dictionaries with converted Python values.

        Parameters
        ----------
        path : str
            The CSV file path.

        Returns
        -------
        list
            The converted row dictionaries.
        """

        import csv

        rows = []

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: TGraph._CSVValue(v) for k, v in row.items() if k is not None})

        return rows


    @staticmethod
    def _CSVWriteRows(path: str, headers: List[str], rows: List[Dict[str, Any]]) -> None:
        """
        Writes row dictionaries to CSV.

        Parameters
        ----------
        path : str
            The CSV file path.
        headers : list
            The CSV headers.
        rows : list
            The row dictionaries.

        Returns
        -------
        None
            None.
        """

        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()

            for row in rows:
                writer.writerow({k: TGraph._CSVExportValue(row.get(k, "")) for k in headers})

    @staticmethod
    def ByCSVPath(
        path: str,

        graphIDHeader: str = "graph_id",
        graphLabelHeader: str = "label",
        graphFeaturesHeader: str = "feat",
        graphFeaturesKeys: list = None,

        edgeSRCHeader: str = "src_id",
        edgeDSTHeader: str = "dst_id",
        edgeLabelHeader: str = "label",
        edgeTrainMaskHeader: str = "train_mask",
        edgeValidateMaskHeader: str = "val_mask",
        edgeTestMaskHeader: str = "test_mask",
        edgeFeaturesHeader: str = "feat",
        edgeFeaturesKeys: list = None,

        nodeIDHeader: str = "node_id",
        nodeLabelHeader: str = "label",
        nodeTrainMaskHeader: str = "train_mask",
        nodeValidateMaskHeader: str = "val_mask",
        nodeTestMaskHeader: str = "test_mask",
        nodeFeaturesHeader: str = "feat",
        nodeFeaturesKeys: list = None,
        nodeXHeader: str = "x",
        nodeYHeader: str = "y",
        nodeZHeader: str = "z",

        directed: bool = False,
        allowSelfLoops: bool = True,
        allowParallelEdges: bool = True,
        ontology: bool = True,
        silent: bool = False,
    ) -> Optional[List["TGraph"]]:
        """
        Creates one or more TGraphs from a PyTorch/PyG-ready CSV folder.

        The folder must contain:

        - graphs.csv
        - nodes.csv
        - edges.csv

        The method returns a list of TGraphs. Each graph, vertex, and edge receives
        dictionaries containing labels, masks, features, and source IDs where present.

        Parameters
        ----------
        path : str
            The input folder path containing graphs.csv, nodes.csv, and edges.csv.
        graphIDHeader : str , optional
            The graph ID column name. Default is "graph_id".
        graphLabelHeader : str , optional
            The graph label column name. Default is "label".
        graphFeaturesHeader : str , optional
            The graph feature column prefix. Default is "feat".
        graphFeaturesKeys : list , optional
            Graph feature keys. If None, feature columns are inferred from the prefix.
            Default is None.
        edgeSRCHeader : str , optional
            The edge source node ID column name. Default is "src_id".
        edgeDSTHeader : str , optional
            The edge destination node ID column name. Default is "dst_id".
        edgeLabelHeader : str , optional
            The edge label column name. Default is "label".
        edgeTrainMaskHeader : str , optional
            The edge train mask column name. Default is "train_mask".
        edgeValidateMaskHeader : str , optional
            The edge validation mask column name. Default is "val_mask".
        edgeTestMaskHeader : str , optional
            The edge test mask column name. Default is "test_mask".
        edgeFeaturesHeader : str , optional
            The edge feature column prefix. Default is "feat".
        edgeFeaturesKeys : list , optional
            Edge feature keys. If None, feature columns are inferred from the prefix.
            Default is None.
        nodeIDHeader : str , optional
            The node ID column name. Default is "node_id".
        nodeLabelHeader : str , optional
            The node label column name. Default is "label".
        nodeTrainMaskHeader : str , optional
            The node train mask column name. Default is "train_mask".
        nodeValidateMaskHeader : str , optional
            The node validation mask column name. Default is "val_mask".
        nodeTestMaskHeader : str , optional
            The node test mask column name. Default is "test_mask".
        nodeFeaturesHeader : str , optional
            The node feature column prefix. Default is "feat".
        nodeFeaturesKeys : list , optional
            Node feature keys. If None, feature columns are inferred from the prefix.
            Default is None.
        nodeXHeader : str , optional
            The node X-coordinate column name. Default is "x".
        nodeYHeader : str , optional
            The node Y-coordinate column name. Default is "y".
        nodeZHeader : str , optional
            The node Z-coordinate column name. Default is "z".
        directed : bool , optional
            If set to True, imported graph edges are treated as directed. Default is False.
        allowSelfLoops : bool , optional
            If set to True, self-loop edges are allowed. Default is True.
        allowParallelEdges : bool , optional
            If set to True, parallel edges are allowed. Default is True.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable.
            Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            A list of imported TGraphs, or None if the folder is invalid.
        """

        import os

        def _err(message):
            if not silent:
                print(message)
            return None

        if not isinstance(path, str) or path.strip() == "":
            return _err("TGraph.ByCSVPath - Error: The input path is not a valid string. Returning None.")

        graphsCSV = os.path.join(path, "graphs.csv")
        nodesCSV = os.path.join(path, "nodes.csv")
        edgesCSV = os.path.join(path, "edges.csv")

        if not (os.path.exists(graphsCSV) and os.path.exists(nodesCSV) and os.path.exists(edgesCSV)):
            # Backwards compatibility with lightweight TGraph record CSV format.
            metadataPath = os.path.join(path, "metadata.json")
            verticesPath = os.path.join(path, "vertices.csv")
            recordEdgesPath = os.path.join(path, "edges.csv")

            if os.path.exists(metadataPath) and os.path.exists(verticesPath) and os.path.exists(recordEdgesPath):
                try:
                    import json
                    with open(metadataPath, "r", encoding="utf-8") as f:
                        metadata = json.loads(f.read())
                    with open(verticesPath, "r", encoding="utf-8") as f:
                        verticesCSVString = f.read()
                    with open(recordEdgesPath, "r", encoding="utf-8") as f:
                        edgesCSVString = f.read()
                    g = TGraph.ByCSVStrings(verticesCSVString, edgesCSVString, metadata=metadata)
                    return [g] if isinstance(g, TGraph) else None
                except Exception as exc:
                    return _err(f"TGraph.ByCSVPath - Error: Could not read legacy record CSV files. {exc}. Returning None.")

            return _err("TGraph.ByCSVPath - Error: Could not find graphs.csv, nodes.csv, and edges.csv. Returning None.")

        try:
            graphRows = TGraph._CSVReadRows(graphsCSV)
            nodeRows = TGraph._CSVReadRows(nodesCSV)
            edgeRows = TGraph._CSVReadRows(edgesCSV)
        except Exception as exc:
            return _err(f"TGraph.ByCSVPath - Error: Could not read CSV files. {exc}. Returning None.")

        if not graphRows:
            return _err("TGraph.ByCSVPath - Error: graphs.csv contains no graph rows. Returning None.")

        graphHeaders = list(graphRows[0].keys()) if graphRows else []
        nodeHeaders = list(nodeRows[0].keys()) if nodeRows else []
        edgeHeaders = list(edgeRows[0].keys()) if edgeRows else []

        graphFeatureKeys = TGraph._CSVFlatten(graphFeaturesKeys)
        nodeFeatureKeys = TGraph._CSVFlatten(nodeFeaturesKeys)
        edgeFeatureKeys = TGraph._CSVFlatten(edgeFeaturesKeys)

        if not graphFeatureKeys:
            graphFeatureKeys = TGraph._CSVFeatureKeysFromHeaders(graphHeaders, graphFeaturesHeader)
        if not nodeFeatureKeys:
            nodeFeatureKeys = TGraph._CSVFeatureKeysFromHeaders(nodeHeaders, nodeFeaturesHeader)
        if not edgeFeatureKeys:
            edgeFeatureKeys = TGraph._CSVFeatureKeysFromHeaders(edgeHeaders, edgeFeaturesHeader)

        graphFeatureHeaders = TGraph._CSVFeatureHeaders(graphFeaturesHeader, graphFeatureKeys)
        nodeFeatureHeaders = TGraph._CSVFeatureHeaders(nodeFeaturesHeader, nodeFeatureKeys)
        edgeFeatureHeaders = TGraph._CSVFeatureHeaders(edgeFeaturesHeader, edgeFeatureKeys)

        nodeRowsByGraph = {}
        for row in nodeRows:
            graphID = row.get(graphIDHeader, None)
            nodeRowsByGraph.setdefault(graphID, []).append(row)

        edgeRowsByGraph = {}
        for row in edgeRows:
            graphID = row.get(graphIDHeader, None)
            edgeRowsByGraph.setdefault(graphID, []).append(row)

        result = []

        for graphRow in graphRows:
            graphID = graphRow.get(graphIDHeader, None)

            if graphID is None:
                if not silent:
                    print("TGraph.ByCSVPath - Warning: A graph row has no graph ID. Skipping.")
                continue

            graphDictionary = {}

            for k, v in graphRow.items():
                if v is not None:
                    graphDictionary[k] = v

            graphDictionary.setdefault(graphIDHeader, graphID)

            if graphLabelHeader in graphRow:
                graphDictionary.setdefault("label", graphRow.get(graphLabelHeader))

            graphFeatureValues = []
            for featureKey, featureHeader in zip(graphFeatureKeys, graphFeatureHeaders):
                value = graphRow.get(featureHeader, 0.0)
                try:
                    value = float(value)
                except Exception:
                    value = 0.0
                graphDictionary[str(featureKey)] = value
                graphFeatureValues.append(value)

            if graphFeatureKeys:
                graphDictionary["feat"] = graphFeatureValues
                graphDictionary["feat_keys"] = [str(k) for k in graphFeatureKeys]

            graphDictionary.setdefault("generated_by", "TGraph.ByCSVPath")
            graphDictionary.setdefault("source", path)

            g = TGraph(
                directed=directed,
                allowSelfLoops=allowSelfLoops,
                allowParallelEdges=allowParallelEdges,
                dictionary=graphDictionary,
            )

            nodeIDToVertexIndex = {}

            for nodeRow in nodeRowsByGraph.get(graphID, []):
                nodeID = nodeRow.get(nodeIDHeader, None)

                if nodeID is None:
                    if not silent:
                        print(f"TGraph.ByCSVPath - Warning: A node in graph {graphID} has no node ID. Skipping.")
                    continue

                d = {}

                for k, v in nodeRow.items():
                    if k == graphIDHeader:
                        continue
                    if v is not None:
                        d[k] = v

                d.setdefault(nodeIDHeader, nodeID)

                if nodeLabelHeader in nodeRow:
                    d.setdefault("label", nodeRow.get(nodeLabelHeader))

                d["train_mask"] = TGraph._CSVBool(nodeRow.get(nodeTrainMaskHeader, False), default=False)
                d["val_mask"] = TGraph._CSVBool(nodeRow.get(nodeValidateMaskHeader, False), default=False)
                d["test_mask"] = TGraph._CSVBool(nodeRow.get(nodeTestMaskHeader, False), default=False)

                nodeFeatureValues = []
                for featureKey, featureHeader in zip(nodeFeatureKeys, nodeFeatureHeaders):
                    value = nodeRow.get(featureHeader, 0.0)
                    try:
                        value = float(value)
                    except Exception:
                        value = 0.0
                    d[str(featureKey)] = value
                    nodeFeatureValues.append(value)

                if nodeFeatureKeys:
                    d["feat"] = nodeFeatureValues
                    d["feat_keys"] = [str(k) for k in nodeFeatureKeys]

                x = nodeRow.get(nodeXHeader, None)
                y = nodeRow.get(nodeYHeader, None)
                z = nodeRow.get(nodeZHeader, None)

                try:
                    x = 0.0 if x is None else float(x)
                except Exception:
                    x = 0.0
                try:
                    y = 0.0 if y is None else float(y)
                except Exception:
                    y = 0.0
                try:
                    z = 0.0 if z is None else float(z)
                except Exception:
                    z = 0.0

                d["x"] = x
                d["y"] = y
                d["z"] = z

                representation = None
                try:
                    from topologicpy.Vertex import Vertex
                    representation = Vertex.ByCoordinates(x, y, z)
                except Exception:
                    representation = None

                vertexIndex = g.AddVertex(dictionary=d, representation=representation)
                nodeIDToVertexIndex[nodeID] = vertexIndex
                nodeIDToVertexIndex[str(nodeID)] = vertexIndex

            for edgeRow in edgeRowsByGraph.get(graphID, []):
                srcID = edgeRow.get(edgeSRCHeader, None)
                dstID = edgeRow.get(edgeDSTHeader, None)

                if srcID is None or dstID is None:
                    if not silent:
                        print(f"TGraph.ByCSVPath - Warning: An edge in graph {graphID} has no valid src/dst ID. Skipping.")
                    continue

                srcIndex = nodeIDToVertexIndex.get(srcID, nodeIDToVertexIndex.get(str(srcID), None))
                dstIndex = nodeIDToVertexIndex.get(dstID, nodeIDToVertexIndex.get(str(dstID), None))

                if srcIndex is None or dstIndex is None:
                    if not silent:
                        print(f"TGraph.ByCSVPath - Warning: Could not resolve edge endpoints ({srcID}, {dstID}) in graph {graphID}. Skipping.")
                    continue

                d = {}

                for k, v in edgeRow.items():
                    if k == graphIDHeader:
                        continue
                    if v is not None:
                        d[k] = v

                d.setdefault(edgeSRCHeader, srcID)
                d.setdefault(edgeDSTHeader, dstID)

                if edgeLabelHeader in edgeRow:
                    d.setdefault("label", edgeRow.get(edgeLabelHeader))

                d["train_mask"] = TGraph._CSVBool(edgeRow.get(edgeTrainMaskHeader, False), default=False)
                d["val_mask"] = TGraph._CSVBool(edgeRow.get(edgeValidateMaskHeader, False), default=False)
                d["test_mask"] = TGraph._CSVBool(edgeRow.get(edgeTestMaskHeader, False), default=False)

                edgeFeatureValues = []
                for featureKey, featureHeader in zip(edgeFeatureKeys, edgeFeatureHeaders):
                    value = edgeRow.get(featureHeader, 0.0)
                    try:
                        value = float(value)
                    except Exception:
                        value = 0.0
                    d[str(featureKey)] = value
                    edgeFeatureValues.append(value)

                if edgeFeatureKeys:
                    d["feat"] = edgeFeatureValues
                    d["feat_keys"] = [str(k) for k in edgeFeatureKeys]

                g.AddEdge(srcIndex, dstIndex, directed=directed, dictionary=d)

            if TGraph.Order(g) < 1:
                if not silent:
                    print(f"TGraph.ByCSVPath - Warning: Graph id {graphID} has no vertices. Skipping.")
                continue

            if ontology:
                try:
                    g = TGraph._OntologyAnnotateGraph(
                        g,
                        graphClass=g._dictionary.get("ontology_class", "top:Graph"),
                        vertexClass="top:Node",
                        edgeClass="top:Relationship",
                        generatedBy="TGraph.ByCSVPath",
                        ontology=True,
                        silent=True,
                    )
                except Exception:
                    pass

            result.append(g)

        return result

    @staticmethod
    def ExportToCSV(
        graph,
        path,

        graphLabelKey: str = "label",
        defaultGraphLabel=0,
        graphFeaturesKeys: list = None,
        graphIDHeader: str = "graph_id",
        graphLabelHeader: str = "label",
        graphFeaturesHeader: str = "feat",

        edgeLabelKey: str = "label",
        defaultEdgeLabel=0,
        edgeFeaturesKeys: list = None,
        edgeSRCHeader: str = "src_id",
        edgeDSTHeader: str = "dst_id",
        edgeLabelHeader: str = "label",
        edgeFeaturesHeader: str = "feat",
        edgeTrainMaskHeader: str = "train_mask",
        edgeValidateMaskHeader: str = "val_mask",
        edgeTestMaskHeader: str = "test_mask",
        edgeMaskKey: str = "mask",
        edgeTrainRatio: float = 0.8,
        edgeValidateRatio: float = 0.1,
        edgeTestRatio: float = 0.1,
        bidirectional: bool = True,

        nodeLabelKey: str = "label",
        defaultNodeLabel=0,
        nodeFeaturesKeys: list = None,
        nodeIDHeader: str = "node_id",
        nodeLabelHeader: str = "label",
        nodeFeaturesHeader: str = "feat",
        nodeTrainMaskHeader: str = "train_mask",
        nodeValidateMaskHeader: str = "val_mask",
        nodeTestMaskHeader: str = "test_mask",
        nodeMaskKey: str = "mask",
        nodeTrainRatio: float = 0.8,
        nodeValidateRatio: float = 0.1,
        nodeTestRatio: float = 0.1,

        nodeXHeader: str = "x",
        nodeYHeader: str = "y",
        nodeZHeader: str = "z",

        mantissa: int = 6,
        overwrite: bool = False,
        silent: bool = False,
    ) -> Optional[bool]:
        """
        Exports one TGraph or a list of TGraphs to a PyTorch/PyG-ready CSV folder.

        This is the public CSV export method. It accepts either a single TGraph or a
        list of TGraphs and writes graphs.csv, nodes.csv, edges.csv, and meta.yaml.

        Parameters
        ----------
        graph : TGraph or list
            The input TGraph or list of TGraphs.
        path : str
            The output folder path.
        graphLabelKey : str , optional
            Graph dictionary key for graph labels. Default is "label".
        defaultGraphLabel : Any , optional
            Default graph label. Default is 0.
        graphFeaturesKeys : list , optional
            Graph feature dictionary keys. Default is None.
        graphIDHeader : str , optional
            Graph ID header. Default is "graph_id".
        graphLabelHeader : str , optional
            Graph label header. Default is "label".
        graphFeaturesHeader : str , optional
            Graph feature prefix. Default is "feat".
        edgeLabelKey : str , optional
            Edge dictionary key for edge labels. Default is "label".
        defaultEdgeLabel : Any , optional
            Default edge label. Default is 0.
        edgeFeaturesKeys : list , optional
            Edge feature dictionary keys. Default is None.
        edgeSRCHeader : str , optional
            Edge source node ID header. Default is "src_id".
        edgeDSTHeader : str , optional
            Edge destination node ID header. Default is "dst_id".
        edgeLabelHeader : str , optional
            Edge label header. Default is "label".
        edgeFeaturesHeader : str , optional
            Edge feature prefix. Default is "feat".
        edgeTrainMaskHeader : str , optional
            Edge train mask header. Default is "train_mask".
        edgeValidateMaskHeader : str , optional
            Edge validation mask header. Default is "val_mask".
        edgeTestMaskHeader : str , optional
            Edge test mask header. Default is "test_mask".
        edgeMaskKey : str , optional
            Edge dictionary key for split assignment. Values 0, 1, 2 mean train,
            validation, and test respectively. Default is "mask".
        edgeTrainRatio : float , optional
            Edge train ratio. Default is 0.8.
        edgeValidateRatio : float , optional
            Edge validation ratio. Default is 0.1.
        edgeTestRatio : float , optional
            Edge test ratio. Default is 0.1.
        bidirectional : bool , optional
            If set to True, writes both source-to-destination and destination-to-source
            rows for each non-self-loop edge. Default is True.
        nodeLabelKey : str , optional
            Vertex dictionary key for node labels. Default is "label".
        defaultNodeLabel : Any , optional
            Default node label. Default is 0.
        nodeFeaturesKeys : list , optional
            Vertex feature dictionary keys. Default is None.
        nodeIDHeader : str , optional
            Node ID header. Default is "node_id".
        nodeLabelHeader : str , optional
            Node label header. Default is "label".
        nodeFeaturesHeader : str , optional
            Node feature prefix. Default is "feat".
        nodeTrainMaskHeader : str , optional
            Node train mask header. Default is "train_mask".
        nodeValidateMaskHeader : str , optional
            Node validation mask header. Default is "val_mask".
        nodeTestMaskHeader : str , optional
            Node test mask header. Default is "test_mask".
        nodeMaskKey : str , optional
            Vertex dictionary key for split assignment. Values 0, 1, 2 mean train,
            validation, and test respectively. Default is "mask".
        nodeTrainRatio : float , optional
            Node train ratio. Default is 0.8.
        nodeValidateRatio : float , optional
            Node validation ratio. Default is 0.1.
        nodeTestRatio : float , optional
            Node test ratio. Default is 0.1.
        nodeXHeader : str , optional
            Node X-coordinate header. Default is "x".
        nodeYHeader : str , optional
            Node Y-coordinate header. Default is "y".
        nodeZHeader : str , optional
            Node Z-coordinate header. Default is "z".
        mantissa : int , optional
            The desired number of decimal places. Default is 6.
        overwrite : bool , optional
            If set to True, existing CSV files are overwritten. Default is False.
        silent : bool , optional
            If set to True, errors and warnings are suppressed. Default is False.

        Returns
        -------
        bool or None
            True if successful; otherwise None.
        """

        if isinstance(graph, TGraph):
            graphs = [graph]
        elif isinstance(graph, (list, tuple)):
            graphs = [g for g in graph if isinstance(g, TGraph)]
        else:
            if not silent:
                print("TGraph.ExportToCSV - Error: The input graph parameter is not a TGraph or a list of TGraphs. Returning None.")
            return None

        if len(graphs) < 1:
            if not silent:
                print("TGraph.ExportToCSV - Error: No valid TGraphs were found. Returning None.")
            return None

        return TGraph._ExportGraphsToCSV(
            graphs=graphs,
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

            nodeXHeader=nodeXHeader,
            nodeYHeader=nodeYHeader,
            nodeZHeader=nodeZHeader,

            mantissa=mantissa,
            overwrite=overwrite,
            silent=silent,
        )

    @staticmethod
    def _ExportGraphsToCSV(
        graphs,
        path,

        graphLabelKey: str = "label",
        defaultGraphLabel=0,
        graphFeaturesKeys: list = None,
        graphIDHeader: str = "graph_id",
        graphLabelHeader: str = "label",
        graphFeaturesHeader: str = "feat",

        edgeLabelKey: str = "label",
        defaultEdgeLabel=0,
        edgeFeaturesKeys: list = None,
        edgeSRCHeader: str = "src_id",
        edgeDSTHeader: str = "dst_id",
        edgeLabelHeader: str = "label",
        edgeFeaturesHeader: str = "feat",
        edgeTrainMaskHeader: str = "train_mask",
        edgeValidateMaskHeader: str = "val_mask",
        edgeTestMaskHeader: str = "test_mask",
        edgeMaskKey: str = "mask",
        edgeTrainRatio: float = 0.8,
        edgeValidateRatio: float = 0.1,
        edgeTestRatio: float = 0.1,
        bidirectional: bool = True,

        nodeLabelKey: str = "label",
        defaultNodeLabel=0,
        nodeFeaturesKeys: list = None,
        nodeIDHeader: str = "node_id",
        nodeLabelHeader: str = "label",
        nodeFeaturesHeader: str = "feat",
        nodeTrainMaskHeader: str = "train_mask",
        nodeValidateMaskHeader: str = "val_mask",
        nodeTestMaskHeader: str = "test_mask",
        nodeMaskKey: str = "mask",
        nodeTrainRatio: float = 0.8,
        nodeValidateRatio: float = 0.1,
        nodeTestRatio: float = 0.1,

        nodeXHeader: str = "x",
        nodeYHeader: str = "y",
        nodeZHeader: str = "z",

        mantissa: int = 6,
        overwrite: bool = False,
        silent: bool = False,
    ) -> Optional[bool]:
        """
        Private implementation helper for ExportToCSV.

        Parameters
        ----------
        graphs : list
            The list of TGraphs to export.
        path : str
            The output folder path.

        Returns
        -------
        bool or None
            True if successful; otherwise None.
        """

        import os
        import shutil

        def _err(message):
            if not silent:
                print(message)
            return None

        if not isinstance(graphs, (list, tuple)) or len(graphs) < 1:
            return _err("TGraph._ExportGraphsToCSV - Error: The input graphs parameter is not a valid list. Returning None.")

        graphs = [g for g in graphs if isinstance(g, TGraph)]

        if len(graphs) < 1:
            return _err("TGraph._ExportGraphsToCSV - Error: No valid TGraphs were found. Returning None.")

        if not isinstance(path, str) or path.strip() == "":
            return _err("TGraph._ExportGraphsToCSV - Error: The input path is not a valid string. Returning None.")

        graphFeatureKeys = TGraph._CSVFlatten(graphFeaturesKeys)
        nodeFeatureKeys = TGraph._CSVFlatten(nodeFeaturesKeys)
        edgeFeatureKeys = TGraph._CSVFlatten(edgeFeaturesKeys)

        graphFeatureHeaders = TGraph._CSVFeatureHeaders(graphFeaturesHeader, graphFeatureKeys)
        nodeFeatureHeaders = TGraph._CSVFeatureHeaders(nodeFeaturesHeader, nodeFeatureKeys)
        edgeFeatureHeaders = TGraph._CSVFeatureHeaders(edgeFeaturesHeader, edgeFeatureKeys)

        graphsCSV = os.path.join(path, "graphs.csv")
        nodesCSV = os.path.join(path, "nodes.csv")
        edgesCSV = os.path.join(path, "edges.csv")
        metaYAML = os.path.join(path, "meta.yaml")

        if os.path.exists(path):
            if overwrite:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    return _err("TGraph._ExportGraphsToCSV - Error: The input path exists and is not a folder. Returning None.")
            else:
                existing = [p for p in [graphsCSV, nodesCSV, edgesCSV, metaYAML] if os.path.exists(p)]
                if existing:
                    return _err("TGraph._ExportGraphsToCSV - Error: CSV files already exist and overwrite is False. Returning None.")

        os.makedirs(path, exist_ok=True)

        graphHeaders = [graphIDHeader, graphLabelHeader] + graphFeatureHeaders

        nodeHeaders = [
            graphIDHeader,
            nodeIDHeader,
            nodeLabelHeader,
            nodeTrainMaskHeader,
            nodeValidateMaskHeader,
            nodeTestMaskHeader,
            nodeXHeader,
            nodeYHeader,
            nodeZHeader,
        ] + nodeFeatureHeaders

        edgeHeaders = [
            graphIDHeader,
            edgeSRCHeader,
            edgeDSTHeader,
            edgeLabelHeader,
            edgeTrainMaskHeader,
            edgeValidateMaskHeader,
            edgeTestMaskHeader,
        ] + edgeFeatureHeaders

        graphRows = []
        nodeRows = []
        edgeRows = []

        for graphID, graph in enumerate(graphs):
            graphDictionary = graph._dictionary if isinstance(graph._dictionary, dict) else {}

            graphRow = {
                graphIDHeader: graphID,
                graphLabelHeader: TGraph._CSVLabelValue(graphDictionary, graphLabelKey, defaultGraphLabel),
            }

            for header, featureKey, value in zip(
                graphFeatureHeaders,
                graphFeatureKeys,
                TGraph._CSVFeatureValues(graphDictionary, graphFeatureKeys, mantissa=mantissa),
            ):
                graphRow[header] = value

            graphRows.append(graphRow)

            activeVertices = [v for v in graph._vertices if v.get("active", True)]
            activeEdges = [e for e in graph._edges if e.get("active", True)]

            vertexIndexToNodeID = {}
            nodeCount = len(activeVertices)

            nodeTrainMax = int(round(float(nodeTrainRatio) * float(nodeCount)))
            nodeValidateMax = int(round(float(nodeValidateRatio) * float(nodeCount)))
            nodeCounts = {"train": 0, "val": 0, "test": 0}

            for nodeID, vertexRecord in enumerate(activeVertices):
                vertexIndex = vertexRecord.get("index", None)
                vertexIndexToNodeID[vertexIndex] = nodeID

                d = vertexRecord.get("dictionary", {})
                d = d if isinstance(d, dict) else {}

                trainMask, valMask, testMask = TGraph._CSVMaskFromDictionaryOrRatio(
                    d,
                    nodeMaskKey,
                    nodeTrainMax,
                    nodeValidateMax,
                    nodeCounts,
                )

                coordinates = TGraph.Coordinates(graph, vertexIndex, default=None)

                if coordinates is None or len(coordinates) < 3:
                    x, y, z = 0.0, 0.0, 0.0
                else:
                    try:
                        x = round(float(coordinates[0]), mantissa)
                        y = round(float(coordinates[1]), mantissa)
                        z = round(float(coordinates[2]), mantissa)
                    except Exception:
                        x, y, z = 0.0, 0.0, 0.0

                nodeRow = {
                    graphIDHeader: graphID,
                    nodeIDHeader: nodeID,
                    nodeLabelHeader: TGraph._CSVLabelValue(d, nodeLabelKey, defaultNodeLabel),
                    nodeTrainMaskHeader: 1 if trainMask else 0,
                    nodeValidateMaskHeader: 1 if valMask else 0,
                    nodeTestMaskHeader: 1 if testMask else 0,
                    nodeXHeader: x,
                    nodeYHeader: y,
                    nodeZHeader: z,
                }

                for header, featureKey, value in zip(
                    nodeFeatureHeaders,
                    nodeFeatureKeys,
                    TGraph._CSVFeatureValues(d, nodeFeatureKeys, mantissa=mantissa),
                ):
                    nodeRow[header] = value

                nodeRows.append(nodeRow)

            exportEdges = []

            for edgeRecord in activeEdges:
                srcIndex = edgeRecord.get("src", None)
                dstIndex = edgeRecord.get("dst", None)

                if srcIndex not in vertexIndexToNodeID or dstIndex not in vertexIndexToNodeID:
                    continue

                exportEdges.append((edgeRecord, vertexIndexToNodeID[srcIndex], vertexIndexToNodeID[dstIndex], False))

                if bidirectional and srcIndex != dstIndex:
                    exportEdges.append((edgeRecord, vertexIndexToNodeID[dstIndex], vertexIndexToNodeID[srcIndex], True))

            edgeCount = len(exportEdges)
            edgeTrainMax = int(round(float(edgeTrainRatio) * float(edgeCount)))
            edgeValidateMax = int(round(float(edgeValidateRatio) * float(edgeCount)))
            edgeCounts = {"train": 0, "val": 0, "test": 0}

            for edgeRecord, srcNodeID, dstNodeID, isReverse in exportEdges:
                d = edgeRecord.get("dictionary", {})
                d = d if isinstance(d, dict) else {}

                trainMask, valMask, testMask = TGraph._CSVMaskFromDictionaryOrRatio(
                    d,
                    edgeMaskKey,
                    edgeTrainMax,
                    edgeValidateMax,
                    edgeCounts,
                )

                edgeRow = {
                    graphIDHeader: graphID,
                    edgeSRCHeader: srcNodeID,
                    edgeDSTHeader: dstNodeID,
                    edgeLabelHeader: TGraph._CSVLabelValue(d, edgeLabelKey, defaultEdgeLabel),
                    edgeTrainMaskHeader: 1 if trainMask else 0,
                    edgeValidateMaskHeader: 1 if valMask else 0,
                    edgeTestMaskHeader: 1 if testMask else 0,
                }

                for header, featureKey, value in zip(
                    edgeFeatureHeaders,
                    edgeFeatureKeys,
                    TGraph._CSVFeatureValues(d, edgeFeatureKeys, mantissa=mantissa),
                ):
                    edgeRow[header] = value

                edgeRows.append(edgeRow)

        try:
            TGraph._CSVWriteRows(graphsCSV, graphHeaders, graphRows)
            TGraph._CSVWriteRows(nodesCSV, nodeHeaders, nodeRows)
            TGraph._CSVWriteRows(edgesCSV, edgeHeaders, edgeRows)

            with open(metaYAML, "w", encoding="utf-8") as yamlFile:
                yamlFile.write(
                    "dataset_name: topologic_dataset\n"
                    "edge_data:\n"
                    "- file_name: edges.csv\n"
                    "node_data:\n"
                    "- file_name: nodes.csv\n"
                    "graph_data:\n"
                    "  file_name: graphs.csv\n"
                )

            return True

        except Exception as exc:
            return _err(f"TGraph._ExportGraphsToCSV - Error: {exc}. Returning None.")

    @staticmethod
    def ExportGraphsToCSV(*args, **kwargs):
        """
        Deprecated compatibility alias for TGraph.ExportToCSV.

        Use TGraph.ExportToCSV instead. ExportToCSV accepts either one TGraph or a
        list of TGraphs.

        Returns
        -------
        bool or None
            True if successful; otherwise None.
        """

        try:
            import warnings
            warnings.warn(
                "TGraph.ExportGraphsToCSV is deprecated. Use TGraph.ExportToCSV instead. "
                "ExportToCSV accepts either a single TGraph or a list of TGraphs.",
                DeprecationWarning,
                stacklevel=2,
            )
        except Exception:
            pass

        return TGraph.ExportToCSV(*args, **kwargs)

    @staticmethod
    def ExportGraphToCSV(graph: "TGraph", *args, **kwargs):
        """
        Deprecated compatibility alias for TGraph.ExportToCSV.

        Use TGraph.ExportToCSV instead.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.

        Returns
        -------
        bool or None
            True if successful; otherwise None.
        """

        try:
            import warnings
            warnings.warn(
                "TGraph.ExportGraphToCSV is deprecated. Use TGraph.ExportToCSV instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        except Exception:
            pass

        return TGraph.ExportToCSV(graph, *args, **kwargs)

    @staticmethod
    def CSVData(graph: "TGraph", includeInactive: bool = False) -> Dict[str, Any]:
        """
        Returns lightweight TGraph record-CSV data.

        This method is intended for TGraph-native record serialisation. For
        PyTorch/PyG-ready datasets, use ByCSVPath and ExportToCSV.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        includeInactive : bool , optional
            If set to True, inactive vertices and edges are included. Default is False.

        Returns
        -------
        dict
            A dictionary containing metadata, vertices_csv, and edges_csv.
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
    def VerticesCSVString(graph: "TGraph", includeInactive: bool = False) -> str:
        """
        Returns a lightweight CSV string of TGraph vertex records.

        This method is intended for TGraph-native record serialisation. For
        PyTorch/PyG-ready datasets, use ByCSVPath and ExportToCSV.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        includeInactive : bool , optional
            If set to True, inactive vertices are included. Default is False.

        Returns
        -------
        str
            The vertices CSV string.
        """

        if not isinstance(graph, TGraph):
            return ""

        import csv
        import io

        records = [v for v in graph._vertices if includeInactive or v.get("active", True)]

        baseHeaders = ["index", "active"]
        dictionaryHeaders = []

        for record in records:
            d = record.get("dictionary", {})
            if not isinstance(d, dict):
                continue
            for key in d.keys():
                if key not in baseHeaders and key not in dictionaryHeaders:
                    dictionaryHeaders.append(key)

        headers = baseHeaders + dictionaryHeaders

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for record in records:
            row = {
                "index": record.get("index", None),
                "active": 1 if record.get("active", True) else 0,
            }

            d = record.get("dictionary", {})
            if isinstance(d, dict):
                for key in dictionaryHeaders:
                    row[key] = TGraph._CSVExportValue(d.get(key, ""))

            writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def EdgesCSVString(graph: "TGraph", includeInactive: bool = False) -> str:
        """
        Returns a lightweight CSV string of TGraph edge records.

        This method is intended for TGraph-native record serialisation. For
        PyTorch/PyG-ready datasets, use ByCSVPath and ExportToCSV.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        includeInactive : bool , optional
            If set to True, inactive edges are included. Default is False.

        Returns
        -------
        str
            The edges CSV string.
        """

        if not isinstance(graph, TGraph):
            return ""

        import csv
        import io

        records = [e for e in graph._edges if includeInactive or e.get("active", True)]

        baseHeaders = ["index", "src", "dst", "directed", "active"]
        dictionaryHeaders = []

        for record in records:
            d = record.get("dictionary", {})
            if not isinstance(d, dict):
                continue
            for key in d.keys():
                if key not in baseHeaders and key not in dictionaryHeaders:
                    dictionaryHeaders.append(key)

        headers = baseHeaders + dictionaryHeaders

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for record in records:
            row = {
                "index": record.get("index", None),
                "src": record.get("src", None),
                "dst": record.get("dst", None),
                "directed": 1 if record.get("directed", graph._directed) else 0,
                "active": 1 if record.get("active", True) else 0,
            }

            d = record.get("dictionary", {})
            if isinstance(d, dict):
                for key in dictionaryHeaders:
                    row[key] = TGraph._CSVExportValue(d.get(key, ""))

            writer.writerow(row)

        return output.getvalue()

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
        Creates a TGraph from lightweight TGraph record-CSV strings.

        This method is intended for TGraph-native record serialisation. For
        PyTorch/PyG-ready datasets, use ByCSVPath and ExportToCSV.

        Parameters
        ----------
        verticesCSVString : str
            The vertices CSV string.
        edgesCSVString : str
            The edges CSV string.
        metadata : dict , optional
            Metadata dictionary. Default is None.
        directed : bool , optional
            Overrides the graph directed value in metadata. Default is None.
        allowSelfLoops : bool , optional
            Overrides the graph allowSelfLoops value in metadata. Default is None.
        allowParallelEdges : bool , optional
            Overrides the graph allowParallelEdges value in metadata. Default is None.

        Returns
        -------
        TGraph or None
            The created TGraph, or None if the operation fails.
        """

        if not isinstance(verticesCSVString, str) or not isinstance(edgesCSVString, str):
            return None

        import csv
        import io

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

        vertexRows = []

        reader = csv.DictReader(io.StringIO(verticesCSVString))
        for row in reader:
            vertexRows.append({k: TGraph._CSVValue(v) for k, v in row.items() if k is not None})

        def _vertex_sort_key(row):
            try:
                return int(row.get("index", 0))
            except Exception:
                return 0

        vertexRows.sort(key=_vertex_sort_key)

        indexMap = {}

        for row in vertexRows:
            oldIndex = row.get("index", None)
            active = TGraph._CSVBool(row.get("active", True), default=True)

            d = {
                k: v for k, v in row.items()
                if k not in ["index", "active"] and v is not None
            }

            newIndex = g.AddVertex(dictionary=d)

            if oldIndex is not None:
                try:
                    indexMap[int(oldIndex)] = newIndex
                except Exception:
                    indexMap[oldIndex] = newIndex

            if not active:
                try:
                    g.RemoveVertex(newIndex, silent=True)
                except TypeError:
                    try:
                        g.RemoveVertex(newIndex)
                    except Exception:
                        pass
                except Exception:
                    pass

        reader = csv.DictReader(io.StringIO(edgesCSVString))

        for row in reader:
            converted = {k: TGraph._CSVValue(v) for k, v in row.items() if k is not None}

            src = converted.get("src", None)
            dst = converted.get("dst", None)

            if src is None or dst is None:
                continue

            try:
                srcIndex = indexMap.get(int(src), int(src))
            except Exception:
                srcIndex = indexMap.get(src, src)

            try:
                dstIndex = indexMap.get(int(dst), int(dst))
            except Exception:
                dstIndex = indexMap.get(dst, dst)

            edgeDirected = TGraph._CSVBool(converted.get("directed", graph_directed), default=graph_directed)
            active = TGraph._CSVBool(converted.get("active", True), default=True)

            d = {
                k: v for k, v in converted.items()
                if k not in ["index", "src", "dst", "directed", "active"] and v is not None
            }

            edgeIndex = g.AddEdge(srcIndex, dstIndex, directed=edgeDirected, dictionary=d)

            if edgeIndex is not None and not active:
                try:
                    g.RemoveEdge(edgeIndex, silent=True)
                except TypeError:
                    try:
                        g.RemoveEdge(edgeIndex)
                    except Exception:
                        pass
                except Exception:
                    pass

        return TGraph._OntologyAnnotateGraph(
            g,
            graphClass=g._dictionary.get("ontology_class", "top:Graph"),
            vertexClass="top:Node",
            edgeClass="top:Relationship",
            generatedBy="TGraph.ByCSVStrings",
            ontology=True,
            silent=True,
        )

    @staticmethod
    def ExportToAdjacencyMatrixCSV(adjacencyMatrix: List[List[Any]], path: str) -> Optional[str]:
        """
        Exports an adjacency matrix to a CSV file.

        Parameters
        ----------
        adjacencyMatrix : list
            The adjacency matrix.
        path : str
            The output CSV path.

        Returns
        -------
        str or None
            The output path if successful; otherwise None.
        """

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
    def ByAdjacencyMatrixCSVPath(path: str, directed: bool = False, silent: bool = False) -> Optional["TGraph"]:
        """
        Creates a TGraph from an adjacency-matrix CSV file.

        Parameters
        ----------
        path : str
            The input adjacency-matrix CSV path.
        directed : bool , optional
            If set to True, graph edges are treated as directed. Default is False.
        silent : bool , optional
            If set to True, errors and warnings are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The created TGraph, or None if the operation fails.
        """

        if not isinstance(path, str) or path.strip() == "":
            if not silent:
                print("TGraph.ByAdjacencyMatrixCSVPath - Error: The input path is not a valid string. Returning None.")
            return None

        try:
            import csv

            matrix = []

            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    matrix.append([TGraph._CSVValue(v) for v in row])

            if len(matrix) < 1:
                if not silent:
                    print("TGraph.ByAdjacencyMatrixCSVPath - Error: The CSV file is empty. Returning None.")
                return None

            return TGraph.ByAdjacencyMatrix(matrix, directed=directed, silent=silent)

        except TypeError:
            try:
                return TGraph.ByAdjacencyMatrix(matrix, directed=directed)
            except Exception as exc:
                if not silent:
                    print(f"TGraph.ByAdjacencyMatrixCSVPath - Error: {exc}. Returning None.")
                return None

        except Exception as exc:
            if not silent:
                print(f"TGraph.ByAdjacencyMatrixCSVPath - Error: {exc}. Returning None.")
            return None

    # END CSV-RELATED METHODS




    @staticmethod
    def Choice(graph: "TGraph", normalize: bool = True, key: str = "choice", mantissa: int = 6,
               silent: bool = False) -> List[float]:
        """
        Selects vertices or edges from the input TGraph using weighted random choice.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.
        key : str , optional
            The dictionary key to use. Default is 'choice'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[float]
            The resulting choice list.
        """
        return TGraph.BetweennessCentrality(graph, mode="all", normalize=normalize, key=key, mantissa=mantissa)

    @staticmethod
    def ChromaticNumber(graph: "TGraph", maxColors: int = None, silent: bool = False) -> Optional[int]:
        """
        Returns an estimated chromatic number of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        maxColors : int , optional
            The input max colors value. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[int]
            The resulting chromatic number index or count.
        """
        if not isinstance(graph, TGraph):
            return None
        adjacency = TGraph._SimpleUndirectedNeighborSets(graph, includeSelfLoops=False)
        vertices = sorted(adjacency.keys(), key=lambda v: len(adjacency[v]), reverse=True)
        n = len(vertices)
        if n == 0:
            return 0
        if any(v in adjacency[v] for v in vertices):
            return None

        def can_color_with(k: int) -> bool:
            color = {}
            order = vertices
            def backtrack(pos: int) -> bool:
                if pos >= n:
                    return True
                v = order[pos]
                used = {color[u] for u in adjacency[v] if u in color}
                for c in range(k):
                    if c in used:
                        continue
                    color[v] = c
                    if backtrack(pos + 1):
                        return True
                    del color[v]
                return False
            return backtrack(0)

        limit = int(maxColors) if isinstance(maxColors, int) and maxColors > 0 else n
        exact_limit = min(limit, n)
        if n <= 18:
            for k in range(1, exact_limit + 1):
                if can_color_with(k):
                    return k
            return None

        # Greedy upper bound for larger graphs.
        color = {}
        for v in vertices:
            used = {color[u] for u in adjacency[v] if u in color}
            c = 0
            while c in used:
                c += 1
            color[v] = c
        chromatic = max(color.values()) + 1 if color else 0
        if isinstance(maxColors, int) and maxColors > 0 and chromatic > maxColors:
            return None
        return chromatic


    @staticmethod
    def ClosenessCentrality(
        graph: "TGraph",
        weightKey: str = None,
        normalize: bool = False,
        nxCompatible: bool = True,
        useEdges: bool = False,
        edgeKey: str = None,
        angular: bool = False,
        angularWeightKey: str = "angular_weight",
        key: str = "closeness_centrality",
        colorKey: str = "cc_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional[List[float]]:
        """
        Computes the closeness centrality of the input TGraph and stores the result
        in the dictionary of each vertex or edge.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        weightKey : str , optional
            If set to None, each edge is assumed to have a weight of 1. If set to
            "length" or "distance", the geometric length of each edge is used as its
            weight. If set to any other value, the value associated with that key in
            each edge dictionary is used as the edge weight. Default is None.
        normalize : bool , optional
            If set to True, the values are normalized between 0 and 1. Default is False.
        nxCompatible : bool , optional
            If set to True, the values are compatible with those derived from NetworkX.
            Default is True.
        useEdges : bool , optional
            If set to True, the calculation uses the edges rather than the vertices.
            Default is False.
        edgeKey : str , optional
            If not None, the value associated with that key in each edge dictionary is
            used to bundle the edges into one entity for the calculation. Otherwise,
            each edge segment is assumed to be an independent entity. Default is None.
        angular : bool , optional
            If set to True, the calculation uses angular weights between adjacent edge
            segments. This option is valid only when useEdges is set to True.
            Default is False.
        angularWeightKey : str , optional
            The dictionary key under which to store the computed angular weight on the
            line graph edges. Default is "angular_weight".
        key : str , optional
            The desired dictionary key name under which to store the calculated value.
            Default is "closeness_centrality".
        colorKey : str , optional
            The desired dictionary key name under which to store the calculated color.
            Default is "cc_color".
        colorScale : str , optional
            The desired color scale name to use for colors. Default is "viridis".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The list of centralities in the order matching the vertices or edges as requested.
        """

        import heapq
        import math
        import numbers
        from collections import deque

        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.ClosenessCentrality - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        if angular and not useEdges:
            if not silent:
                print("TGraph.ClosenessCentrality - Error: The angular option is valid only when useEdges is set to True. Returning None.")
            return None

        if angular and edgeKey is not None:
            if not silent:
                print("TGraph.ClosenessCentrality - Error: The angular option is not compatible with edge bundling through edgeKey. Returning None.")
            return None

        # ---------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------

        def _active_vertex_indices(g):
            return [v.get("index") for v in g._vertices if v.get("active", True)]

        def _active_edge_records(g):
            return [e for e in g._edges if e.get("active", True)]

        def _unwrap(x):
            if isinstance(x, list) and len(x) == 1:
                return x[0]
            return x

        def _as_float(x, default=0.0):
            x = _unwrap(x)
            if isinstance(x, numbers.Number):
                return float(x)
            try:
                return float(x)
            except Exception:
                return float(default)

        def _round(x):
            if mantissa is None or mantissa < 0:
                return float(x)
            return round(float(x), mantissa)

        def _normalize_flat(vals):
            if not vals:
                return []
            xs = [float(v) for v in vals]
            mn = min(xs)
            mx = max(xs)
            eps = tolerance if tolerance and tolerance > 0 else 1e-12
            if abs(mx - mn) < eps:
                return [0.0 for _ in xs]
            return [(x - mn) / (mx - mn) for x in xs]

        def _color(value, minValue, maxValue):
            try:
                from topologicpy.Color import Color
                return Color.AnyToHex(
                    Color.ByValueInRange(
                        value,
                        minValue=minValue,
                        maxValue=maxValue,
                        colorScale=colorScale,
                    )
                )
            except Exception:
                return None

        def _color_range(vals, unit_range=False):
            if not vals:
                return 0.0, 1.0
            if unit_range:
                return 0.0, 1.0
            mn = min(float(v) for v in vals)
            mx = max(float(v) for v in vals)
            eps = tolerance if tolerance and tolerance > 0 else 1e-12
            if abs(mx - mn) < eps:
                mx = mn + eps
            return mn, mx

        def _apply_values_to_vertices(g, vertexIndices, values, unit_range=False):
            if not vertexIndices or not values:
                return

            mn, mx = _color_range(values, unit_range=unit_range)

            for vertexIndex, value in zip(vertexIndices, values):
                if not g._validate_vertex_index(vertexIndex, active=False):
                    continue

                d = g._vertices[vertexIndex].setdefault("dictionary", {})
                v = float(value)

                if key is not None:
                    d[key] = v

                if colorKey is not None:
                    c = _color(v, mn, mx)
                    if c is not None:
                        d[colorKey] = c

        def _apply_values_to_edges(g, edgeRecords, values, unit_range=False):
            if not edgeRecords or not values:
                return

            mn, mx = _color_range(values, unit_range=unit_range)

            for edgeRecord, value in zip(edgeRecords, values):
                edgeIndex = edgeRecord.get("index", None)

                if not g._validate_edge_index(edgeIndex, active=False):
                    continue

                d = g._edges[edgeIndex].setdefault("dictionary", {})
                v = float(value)

                if key is not None:
                    d[key] = v

                if colorKey is not None:
                    c = _color(v, mn, mx)
                    if c is not None:
                        d[colorKey] = c

        def _edge_length(g, edgeRecord):
            srcIndex = edgeRecord.get("src", None)
            dstIndex = edgeRecord.get("dst", None)

            c1 = TGraph.Coordinates(g, srcIndex, default=None)
            c2 = TGraph.Coordinates(g, dstIndex, default=None)

            if c1 is None or c2 is None:
                return 1.0

            try:
                return float(math.dist(c1, c2))
            except Exception:
                return 1.0

        def _edge_weight(g, edgeRecord):
            if weightKey is None:
                return 1.0

            if isinstance(weightKey, str):
                wl = weightKey.lower()
                if ("len" in wl) or ("dis" in wl):
                    return _edge_length(g, edgeRecord)

            d = edgeRecord.get("dictionary", {})
            if isinstance(d, dict):
                return _as_float(d.get(weightKey, None), default=1.0)

            return 1.0

        def _vector_from_shared_vertex(g, edgeRecord, sharedVertexIndex):
            srcIndex = edgeRecord.get("src", None)
            dstIndex = edgeRecord.get("dst", None)

            c_shared = TGraph.Coordinates(g, sharedVertexIndex, default=None)

            if c_shared is None:
                return None

            if srcIndex == sharedVertexIndex:
                c_other = TGraph.Coordinates(g, dstIndex, default=None)
            elif dstIndex == sharedVertexIndex:
                c_other = TGraph.Coordinates(g, srcIndex, default=None)
            else:
                return None

            if c_other is None:
                return None

            return [
                float(c_other[0]) - float(c_shared[0]),
                float(c_other[1]) - float(c_shared[1]),
                float(c_other[2]) - float(c_shared[2]),
            ]

        def _angle_between_vectors(a, b):
            if a is None or b is None:
                return None

            la = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
            lb = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])

            if la <= 0.0 or lb <= 0.0:
                return None

            dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb)
            dot = max(-1.0, min(1.0, dot))

            return math.degrees(math.acos(dot))

        def _shared_vertex_index(edgeA, edgeB):
            a_src = edgeA.get("src", None)
            a_dst = edgeA.get("dst", None)
            b_src = edgeB.get("src", None)
            b_dst = edgeB.get("dst", None)

            if a_src == b_src or a_src == b_dst:
                return a_src

            if a_dst == b_src or a_dst == b_dst:
                return a_dst

            return None

        def _set_angular_weights_on_line_graph(inputGraph, lineGraph):
            originalEdges = _active_edge_records(inputGraph)
            idToEdge = {}

            for i, edgeRecord in enumerate(originalEdges):
                edgeIndex = edgeRecord.get("index", None)
                if edgeIndex is None:
                    continue

                inputGraph._edges[edgeIndex].setdefault("dictionary", {})["u_edge_id"] = i
                idToEdge[i] = inputGraph._edges[edgeIndex]

            for lineEdgeRecord in _active_edge_records(lineGraph):
                srcIndex = lineEdgeRecord.get("src", None)
                dstIndex = lineEdgeRecord.get("dst", None)

                if srcIndex is None or dstIndex is None:
                    continue

                if not lineGraph._validate_vertex_index(srcIndex, active=False):
                    continue

                if not lineGraph._validate_vertex_index(dstIndex, active=False):
                    continue

                dsrc = lineGraph._vertices[srcIndex].get("dictionary", {})
                ddst = lineGraph._vertices[dstIndex].get("dictionary", {})

                idA = dsrc.get("u_edge_id", None)
                idB = ddst.get("u_edge_id", None)

                edgeA = idToEdge.get(idA, None)
                edgeB = idToEdge.get(idB, None)

                if edgeA is None or edgeB is None:
                    continue

                shared = _shared_vertex_index(edgeA, edgeB)

                if shared is None:
                    continue

                vecA = _vector_from_shared_vertex(inputGraph, edgeA, shared)
                vecB = _vector_from_shared_vertex(inputGraph, edgeB, shared)

                angle = _angle_between_vectors(vecA, vecB)

                if angle is None:
                    continue

                w = _round(float(angle) / 90.0)
                edgeIndex = lineEdgeRecord.get("index", None)

                if lineGraph._validate_edge_index(edgeIndex, active=False):
                    lineGraph._edges[edgeIndex].setdefault("dictionary", {})[angularWeightKey] = w

        # ---------------------------------------------------------------------
        # Edge mode.
        # ---------------------------------------------------------------------

        if useEdges:
            edgeRecords = _active_edge_records(graph)

            if not edgeRecords:
                return []

            for i, edgeRecord in enumerate(edgeRecords):
                edgeIndex = edgeRecord.get("index", None)
                if graph._validate_edge_index(edgeIndex, active=False):
                    graph._edges[edgeIndex].setdefault("dictionary", {})["u_edge_id"] = i

            try:
                lineGraph = TGraph.LineGraph(graph, transferEdgeDictionaries=True)
            except TypeError:
                try:
                    lineGraph = TGraph.LineGraph(graph)
                except Exception:
                    lineGraph = None
            except Exception:
                lineGraph = None

            if not isinstance(lineGraph, TGraph):
                if not silent:
                    print("TGraph.ClosenessCentrality - Error: Could not create a line graph. Returning None.")
                return None

            if angular:
                _set_angular_weights_on_line_graph(graph, lineGraph)
                lineWeightKey = angularWeightKey
            else:
                lineWeightKey = weightKey

                if edgeKey is not None:
                    try:
                        lineGraph = TGraph.Quotient(
                            lineGraph,
                            key=edgeKey,
                            groupLabelKey="label",
                            transferDictionaries=True,
                        )
                    except TypeError:
                        try:
                            lineGraph = TGraph.Quotient(lineGraph, key=edgeKey)
                        except Exception:
                            lineGraph = None
                    except Exception:
                        lineGraph = None

                    if not isinstance(lineGraph, TGraph):
                        if not silent:
                            print("TGraph.ClosenessCentrality - Error: Could not create a quotient line graph. Returning None.")
                        return None

            _ = TGraph.ClosenessCentrality(
                lineGraph,
                weightKey=lineWeightKey,
                normalize=normalize,
                nxCompatible=nxCompatible,
                useEdges=False,
                edgeKey=None,
                angular=False,
                angularWeightKey=angularWeightKey,
                key=key,
                colorKey=colorKey,
                colorScale=colorScale,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=silent,
            )

            out_vals = []

            if edgeKey is None:
                idToValue = {}

                for lineVertexIndex in _active_vertex_indices(lineGraph):
                    d = lineGraph._vertices[lineVertexIndex].get("dictionary", {})
                    eid = d.get("u_edge_id", None)
                    value = d.get(key, None)

                    if eid is not None and value is not None:
                        idToValue[eid] = value

                for i, edgeRecord in enumerate(edgeRecords):
                    edgeIndex = edgeRecord.get("index", None)
                    value = _round(idToValue.get(i, 0.0))
                    out_vals.append(value)

                    if graph._validate_edge_index(edgeIndex, active=False):
                        d = graph._edges[edgeIndex].setdefault("dictionary", {})
                        d.pop("u_edge_id", None)
                        if key is not None:
                            d[key] = value

            else:
                groupValue = {}

                for lineVertexIndex in _active_vertex_indices(lineGraph):
                    d = lineGraph._vertices[lineVertexIndex].get("dictionary", {})
                    gid = d.get(edgeKey, d.get("label", None))
                    value = d.get(key, None)

                    if gid is not None and value is not None:
                        groupValue[gid] = value

                for edgeRecord in edgeRecords:
                    edgeIndex = edgeRecord.get("index", None)
                    d0 = edgeRecord.get("dictionary", {})
                    gid = d0.get(edgeKey, None) if isinstance(d0, dict) else None
                    value = _round(groupValue.get(gid, 0.0))
                    out_vals.append(value)

                    if graph._validate_edge_index(edgeIndex, active=False):
                        d = graph._edges[edgeIndex].setdefault("dictionary", {})
                        d.pop("u_edge_id", None)
                        if key is not None:
                            d[key] = value

            unit_range_for_color = True if normalize else False

            _apply_values_to_edges(
                graph,
                edgeRecords,
                out_vals,
                unit_range=unit_range_for_color,
            )

            return out_vals

        # ---------------------------------------------------------------------
        # Vertex mode.
        # Graph.py treats the input graph as undirected for this calculation.
        # ---------------------------------------------------------------------

        vertexIndices = _active_vertex_indices(graph)
        edgeRecords = _active_edge_records(graph)

        if not vertexIndices:
            return []

        n = len(vertexIndices)

        if n == 1:
            out_vals = [_round(0.0)]
            _apply_values_to_vertices(graph, vertexIndices, out_vals, unit_range=True)
            return out_vals

        vertexIndexToPosition = {vertexIndex: i for i, vertexIndex in enumerate(vertexIndices)}
        adj = [dict() for _ in range(n)]

        for edgeRecord in edgeRecords:
            srcIndex = edgeRecord.get("src", None)
            dstIndex = edgeRecord.get("dst", None)

            if srcIndex == dstIndex:
                continue

            if srcIndex not in vertexIndexToPosition or dstIndex not in vertexIndexToPosition:
                continue

            u = vertexIndexToPosition[srcIndex]
            v = vertexIndexToPosition[dstIndex]

            w = _edge_weight(graph, edgeRecord)

            try:
                w = float(w)
                if w <= 0.0:
                    w = 1.0
            except Exception:
                w = 1.0

            prev_uv = adj[u].get(v, None)
            if prev_uv is None or w < prev_uv:
                adj[u][v] = w
                adj[v][u] = w

        weighted = weightKey is not None

        def _bfs_sum(sourcePosition):
            dist = [-1] * n
            q = deque([sourcePosition])
            dist[sourcePosition] = 0

            reachable = 1
            total = 0.0

            while q:
                u = q.popleft()
                du = dist[u]

                for v in adj[u].keys():
                    if dist[v] == -1:
                        dist[v] = du + 1
                        reachable += 1
                        total += float(dist[v])
                        q.append(v)

            return total, reachable

        def _dijkstra_sum(sourcePosition):
            INF = float("inf")
            dist = [INF] * n
            dist[sourcePosition] = 0.0

            heap = [(0.0, sourcePosition)]

            while heap:
                du, u = heapq.heappop(heap)

                if du > dist[u]:
                    continue

                for v, w in adj[u].items():
                    nd = du + float(w)

                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))

            reachable = 0
            total = 0.0

            for d in dist:
                if d < INF:
                    reachable += 1
                    total += float(d)

            return total, reachable

        values = [0.0] * n

        for i in range(n):
            if weighted:
                total, reachable = _dijkstra_sum(i)
            else:
                total, reachable = _bfs_sum(i)

            s = max(reachable - 1, 0)

            if total > 0.0 and s > 0:
                if nxCompatible:
                    values[i] = (float(s) / float(n - 1)) * (float(s) / float(total))
                else:
                    values[i] = float(s) / float(total)
            else:
                values[i] = 0.0

        out_vals = _normalize_flat(values) if normalize else values
        out_vals = [_round(v) for v in out_vals]

        unit_range_for_color = True if normalize else False

        _apply_values_to_vertices(
            graph,
            vertexIndices,
            out_vals,
            unit_range=unit_range_for_color,
        )

        return out_vals

    @staticmethod
    def Color(graph: "TGraph", oldKey: str = "color", key: str = "color", maxColors: int = None, tolerance: float = 0.0001) -> Optional["TGraph"]:
        """
        Assigns color values to graph elements using a selected dictionary key and color scale.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        oldKey : str , optional
            The dictionary key to use. Default is 'color'.
        key : str , optional
            The dictionary key to use. Default is 'color'.
        maxColors : int , optional
            The input max colors value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None

        adjacency = TGraph._SimpleUndirectedNeighborSets(graph, includeSelfLoops=False)
        vertices = sorted(adjacency.keys(), key=lambda v: (-len(adjacency[v]), v))
        colors: Dict[int, int] = {}

        for v in vertices:
            used = set()
            for u in adjacency[v]:
                c = colors.get(u, None)
                if c is not None:
                    used.add(c)

            c = 0
            while c in used:
                c += 1

            if isinstance(maxColors, int) and maxColors > 0:
                c = c % maxColors

            colors[v] = c

        if key is not None:
            # Direct dictionary write-back is significantly faster than routing
            # through helper accessors for every vertex.
            for v, c in colors.items():
                if graph._validate_vertex_index(v):
                    graph._vertices[v].setdefault("dictionary", {})[key] = c

        return graph

    @staticmethod
    def Community(
        graph: "TGraph",
        key: str = "community",
        colorKey: str = "cp_color",
        colorScale: str = "viridis",
        algorithm: str = "louvain",
        weightKey: str = None,
        nClusters: int = None,
        maxIterations: int = 100,
        seed: int = 17,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> List[int]:
        """
        Partitions the input TGraph into communities and stores both the community
        id and a corresponding colour in each active vertex dictionary.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        key : str , optional
            The dictionary key under which to store the community id. Default is
            "community".
        colorKey : str , optional
            The dictionary key under which to store the community colour. Default is
            "cp_color".
        colorScale : str , optional
            The Plotly colour scale to use. Default is "viridis".
        algorithm : str , optional
            The community detection algorithm. Valid values are:
            "louvain", "spectral", "infomap", and "propagation". Default is "louvain".
        weightKey : str , optional
            The edge dictionary key to use as edge weight. If set to None, each edge
            has weight 1. If set to "length" or "distance", geometric edge length is
            used. Default is None.
        nClusters : int , optional
            Desired number of clusters for spectral clustering. If set to None, the
            number of clusters is estimated from the eigengap. Ignored by the other
            algorithms. Default is None.
        maxIterations : int , optional
            Maximum number of iterations. Default is 100.
        seed : int , optional
            Random seed used by stochastic or seeded algorithms. Default is 17.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Community ids
            are stored as integers. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The resulting community id list in active-vertex order.
        """
        return TGraph.CommunityPartition(
        graph = graph,
        key = key,
        colorKey = colorKey,
        colorScale = colorScale,
        algorithm = algoirthm,
        weightKey = weightKey,
        nClusters = nClusters,
        maxIterations = maxIterations,
        seed = seed,
        mantissa = mantissa,
        tolerance = tolerance,
        silent = silent
        )
    
    @staticmethod
    def CommunityPartition(
        graph: "TGraph",
        key: str = "community",
        colorKey: str = "cp_color",
        colorScale: str = "viridis",
        algorithm: str = "louvain",
        weightKey: str = None,
        nClusters: int = None,
        maxIterations: int = 100,
        seed: int = 17,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> List[int]:
        """
        Partitions the input TGraph into communities and stores the community id
        and corresponding colour in the dictionaries of active vertices. Active
        edge dictionaries are also annotated with community information and an
        "inter_community" boolean.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        key : str , optional
            The dictionary key under which to store the community id. Default is
            "community".
        colorKey : str , optional
            The dictionary key under which to store the community colour. Default is
            "cp_color".
        colorScale : str , optional
            The Plotly colour scale to use when assigning colours. Default is
            "viridis".
        algorithm : str , optional
            The community detection algorithm. Valid values are:

            - "louvain": Uses python-igraph's community_multilevel algorithm.
            - "spectral": Uses spectral clustering on the graph Laplacian followed
            by k-means.
            - "infomap": Uses the optional infomap package if installed.
            - "propagation": Uses weighted label propagation.

            Default is "louvain".
        weightKey : str , optional
            The edge dictionary key to use as edge weight. If set to None, each
            edge contributes a weight of 1 and duplicate undirected edges are
            aggregated by count. If set to "length", "distance", or "metric",
            geometric edge length is used. If set to any other string, that edge
            dictionary value is used. Default is None.
        nClusters : int , optional
            Desired number of clusters for spectral clustering. If set to None, the
            number of clusters is estimated from the eigengap. This parameter is
            used only by algorithm="spectral". Default is None.
        maxIterations : int , optional
            Maximum number of local iterations used by algorithm="propagation" and
            by the k-means phase of algorithm="spectral". Ignored by the
            python-igraph Louvain implementation. Used by algorithm="infomap" only
            if Infomap is unavailable and the method falls back to flow-weighted
            propagation. Default is 100.
        seed : int , optional
            Random seed used by order-sensitive algorithms and k-means
            initialisation. Default is 17.
        tolerance : float , optional
            Numerical tolerance used by spectral clustering and normalisation
            checks. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is
            False.

        Returns
        -------
        list
            The resulting community id list in active-vertex order.
        """

        import math
        import random

        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.CommunityPartition - Error: The input graph is not a valid TGraph. Returning [].")
            return []

        vertices = [v.get("index") for v in graph._vertices if v.get("active", True)]

        if not vertices:
            return []

        algorithm = str(algorithm or "louvain").strip().lower()

        aliases = {
            "label": "propagation",
            "label_propagation": "propagation",
            "label-propagation": "propagation",
            "propagate": "propagation",
            "lpa": "propagation",
            "spectral_clustering": "spectral",
            "spectral-clustering": "spectral",
            "sc": "spectral",
            "map": "infomap",
            "info_map": "infomap",
            "info-map": "infomap",
            "multilevel": "louvain",
            "community_multilevel": "louvain",
            "community-multilevel": "louvain",
        }

        algorithm = aliases.get(algorithm, algorithm)

        if algorithm not in ["louvain", "spectral", "infomap", "propagation"]:
            if not silent:
                print("TGraph.CommunityPartition - Error: algorithm must be one of: 'louvain', 'spectral', 'infomap', or 'propagation'. Returning [].")
            return []

        try:
            maxIterations = max(1, int(maxIterations))
        except Exception:
            maxIterations = 100

        eps = tolerance if tolerance and tolerance > 0 else 1e-12

        # ---------------------------------------------------------------------
        # Basic helpers
        # ---------------------------------------------------------------------

        def _as_float(value, default=1.0):
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            try:
                return float(value)
            except Exception:
                return float(default)

        def _edge_length(edgeRecord):
            src = edgeRecord.get("src", None)
            dst = edgeRecord.get("dst", None)

            c1 = TGraph.Coordinates(graph, src, default=None)
            c2 = TGraph.Coordinates(graph, dst, default=None)

            if c1 is None or c2 is None or len(c1) < 3 or len(c2) < 3:
                return 1.0

            try:
                return math.sqrt(
                    (float(c2[0]) - float(c1[0])) ** 2 +
                    (float(c2[1]) - float(c1[1])) ** 2 +
                    (float(c2[2]) - float(c1[2])) ** 2
                )
            except Exception:
                return 1.0

        def _edge_weight(edgeRecord):
            if weightKey is None:
                return 1.0

            if isinstance(weightKey, str) and weightKey.lower() in ["length", "distance", "metric"]:
                return max(0.0, _edge_length(edgeRecord))

            d = edgeRecord.get("dictionary", {})
            if isinstance(d, dict):
                return max(0.0, _as_float(d.get(weightKey, None), default=1.0))

            return 1.0

        def _build_weighted_edge_map():
            """
            Builds an undirected weighted edge map over active TGraph vertices.
            Duplicate undirected edges are aggregated as weights.
            """

            active = set(vertices)
            edge_weight_map = {}

            for edgeRecord in graph._edges:
                if not edgeRecord.get("active", True):
                    continue

                src = edgeRecord.get("src", None)
                dst = edgeRecord.get("dst", None)

                if src not in active or dst not in active:
                    continue

                if src == dst:
                    continue

                w = _edge_weight(edgeRecord)

                try:
                    w = float(w)
                except Exception:
                    w = 1.0

                if w <= 0:
                    continue

                u, v = (src, dst) if src <= dst else (dst, src)
                edge_weight_map[(u, v)] = edge_weight_map.get((u, v), 0.0) + w

            return edge_weight_map

        def _edge_map_to_adjacency(edge_weight_map, nodes):
            adj = {n: {} for n in nodes}

            for (u, v), w in edge_weight_map.items():
                if u == v:
                    adj.setdefault(u, {})
                    adj[u][u] = adj[u].get(u, 0.0) + float(w)
                else:
                    adj.setdefault(u, {})
                    adj.setdefault(v, {})
                    adj[u][v] = adj[u].get(v, 0.0) + float(w)
                    adj[v][u] = adj[v].get(u, 0.0) + float(w)

            return adj

        edge_weight_map = _build_weighted_edge_map()
        adj_original = _edge_map_to_adjacency(edge_weight_map, vertices)

        def _compact_labels(labelByVertex):
            unique = sorted(set(labelByVertex.get(v, v) for v in vertices), key=lambda x: str(x))
            remap = {old: i for i, old in enumerate(unique)}
            return {v: remap[labelByVertex.get(v, v)] for v in vertices}

        def _write_values(labelByVertex):
            compact = _compact_labels(labelByVertex)
            values = [int(compact[v]) for v in vertices]

            min_value = min(values) if values else 0
            max_value = max(values) if values else 1

            if min_value == max_value:
                color_min = min_value
                color_max = min_value + 1
            else:
                color_min = min_value
                color_max = max_value

            def _community_color(cid):
                try:
                    from topologicpy.Color import Color
                    return Color.AnyToHex(
                        Color.ByValueInRange(
                            cid,
                            minValue=color_min,
                            maxValue=color_max,
                            colorScale=colorScale,
                        )
                    )
                except Exception:
                    return None

            for v in vertices:
                if not graph._validate_vertex_index(v, active=False):
                    continue

                cid = int(compact[v])
                d = graph._vertices[v].setdefault("dictionary", {})

                if key is not None:
                    d[key] = cid

                if colorKey is not None:
                    color = _community_color(cid)
                    if color is not None:
                        d[colorKey] = color

            for edgeRecord in graph._edges:
                if not edgeRecord.get("active", True):
                    continue

                src = edgeRecord.get("src", None)
                dst = edgeRecord.get("dst", None)

                if src not in compact or dst not in compact:
                    continue

                community_a = int(compact[src])
                community_b = int(compact[dst])

                if community_a == community_b:
                    edge_community = community_a
                    edge_color_value = community_a
                    inter_community = False
                else:
                    edge_community = -1
                    edge_color_value = color_min
                    inter_community = True

                d = edgeRecord.setdefault("dictionary", {})

                if key is not None:
                    d[key] = edge_community

                if colorKey is not None:
                    color = _community_color(edge_color_value)
                    if color is not None:
                        d[colorKey] = color

                d["inter_community"] = inter_community

            return values

        # ---------------------------------------------------------------------
        # Louvain
        # Uses python-igraph community_multilevel, adapted from Graph.py.
        # ---------------------------------------------------------------------

        def _louvain_partition():
            import os
            import warnings

            try:
                import igraph as ig
            except Exception:
                if not silent:
                    print("TGraph.CommunityPartition - Installing required python-igraph library.")
                try:
                    os.system("pip install python-igraph")
                except Exception:
                    os.system("pip install python-igraph --user")
                try:
                    import igraph as ig
                    if not silent:
                        print("TGraph.CommunityPartition - python-igraph library installed correctly.")
                except Exception:
                    warnings.warn("TGraph.CommunityPartition - Error: Could not import python-igraph. Please install it manually.")
                    return None

            n_vertices = len(vertices)

            if n_vertices == 0:
                return {}

            vertex_to_local = {v: i for i, v in enumerate(vertices)}
            local_to_vertex = {i: v for v, i in vertex_to_local.items()}

            ig_edge_weight_map = {}

            for edgeRecord in graph._edges:
                if not edgeRecord.get("active", True):
                    continue

                src = edgeRecord.get("src", None)
                dst = edgeRecord.get("dst", None)

                if src not in vertex_to_local or dst not in vertex_to_local:
                    continue

                a = vertex_to_local[src]
                b = vertex_to_local[dst]

                if a == b:
                    continue

                weight = _edge_weight(edgeRecord)

                try:
                    weight = float(weight)
                except Exception:
                    weight = 1.0

                if weight <= 0:
                    continue

                u, v = (a, b) if a <= b else (b, a)
                ig_edge_weight_map[(u, v)] = ig_edge_weight_map.get((u, v), 0.0) + weight

            ig_edges = list(ig_edge_weight_map.keys())
            weights = [ig_edge_weight_map[e] for e in ig_edges]

            if not ig_edges:
                partition_list = [0 for _ in range(n_vertices)]
            else:
                try:
                    ig_graph = ig.Graph(n=n_vertices, edges=ig_edges, directed=False)
                    communities = ig_graph.community_multilevel(weights=weights)
                    partition_list = list(communities.membership)
                except Exception as err:
                    if not silent:
                        print(f"TGraph.CommunityPartition - Error: igraph Louvain failed: {err}. Returning None.")
                    return None

            if len(partition_list) != n_vertices:
                if not silent:
                    print("TGraph.CommunityPartition - Error: Community detection returned an invalid partition list. Returning None.")
                return None

            return {local_to_vertex[i]: int(partition_list[i]) for i in range(n_vertices)}

        # ---------------------------------------------------------------------
        # Propagation
        # Uses maxIterations.
        # ---------------------------------------------------------------------

        def _propagation_partition(flow_weighted=False):
            labels = {v: i for i, v in enumerate(vertices)}
            rng = random.Random(seed)
            order = list(vertices)

            for _ in range(maxIterations):
                old = dict(labels)
                rng.shuffle(order)

                changed = False

                for v in order:
                    if not adj_original.get(v):
                        continue

                    scores = {}

                    for u, w in adj_original[v].items():
                        lbl = labels.get(u, u)

                        if flow_weighted:
                            deg_u = sum(adj_original.get(u, {}).values())
                            score = 0.0 if deg_u <= 0 else float(w) / float(deg_u)
                        else:
                            score = float(w)

                        scores[lbl] = scores.get(lbl, 0.0) + score

                    if not scores:
                        continue

                    best_label = min(scores.keys(), key=lambda lbl: (-scores[lbl], str(lbl)))

                    if best_label != labels[v]:
                        labels[v] = best_label
                        changed = True

                if not changed or labels == old:
                    break

            return labels

        # ---------------------------------------------------------------------
        # Spectral clustering
        # Uses maxIterations for k-means.
        # ---------------------------------------------------------------------

        def _spectral_partition():
            try:
                import numpy as np
            except Exception:
                if not silent:
                    print("TGraph.CommunityPartition - Warning: NumPy is required for spectral clustering. Falling back to propagation.")
                return _propagation_partition()

            n = len(vertices)

            if n <= 1:
                return {v: 0 for v in vertices}

            pos = {v: i for i, v in enumerate(vertices)}
            W = np.zeros((n, n), dtype=float)

            for (u, v), w in edge_weight_map.items():
                if u in pos and v in pos:
                    W[pos[u], pos[v]] += float(w)
                    W[pos[v], pos[u]] += float(w)

            degree = W.sum(axis=1)

            if float(degree.sum()) <= 0.0:
                return {v: i for i, v in enumerate(vertices)}

            D_inv_sqrt = np.zeros((n, n), dtype=float)

            for i, d in enumerate(degree):
                if d > 0:
                    D_inv_sqrt[i, i] = 1.0 / math.sqrt(float(d))

            L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(L)
            except Exception:
                if not silent:
                    print("TGraph.CommunityPartition - Warning: Spectral decomposition failed. Falling back to propagation.")
                return _propagation_partition()

            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            if nClusters is None:
                zero_count = int(np.sum(eigenvalues <= max(eps, 1e-9)))

                if zero_count > 1:
                    k = zero_count
                else:
                    max_k = min(10, n - 1)
                    if max_k <= 1:
                        k = 2
                    else:
                        gaps = [float(eigenvalues[i + 1] - eigenvalues[i]) for i in range(0, max_k)]
                        k = int(np.argmax(gaps) + 1)
                        k = max(2, k)
            else:
                try:
                    k = int(nClusters)
                except Exception:
                    k = 2

            k = max(1, min(k, n))
            X = eigenvectors[:, :k]

            row_norm = np.linalg.norm(X, axis=1)
            row_norm[row_norm == 0] = 1.0
            X = X / row_norm[:, None]

            def kmeans(data, k_value):
                rng = random.Random(seed)
                n_points = data.shape[0]

                first = rng.randrange(n_points)
                centres = [data[first].copy()]

                while len(centres) < k_value:
                    distances = []
                    for p in data:
                        d2 = min(float(np.sum((p - c) ** 2)) for c in centres)
                        distances.append(d2)
                    centres.append(data[int(np.argmax(distances))].copy())

                centres = np.array(centres, dtype=float)
                labels_np = np.zeros(n_points, dtype=int)

                for _ in range(maxIterations):
                    old_labels = labels_np.copy()

                    for i in range(n_points):
                        dists = np.sum((centres - data[i]) ** 2, axis=1)
                        labels_np[i] = int(np.argmin(dists))

                    for c in range(k_value):
                        members = data[labels_np == c]
                        if len(members) > 0:
                            centres[c] = members.mean(axis=0)

                    if np.array_equal(old_labels, labels_np):
                        break

                return labels_np.tolist()

            labels_np = kmeans(X, k)
            return {v: int(labels_np[pos[v]]) for v in vertices}

        # ---------------------------------------------------------------------
        # Infomap
        # Uses real Infomap when available; otherwise falls back to propagation.
        # ---------------------------------------------------------------------

        def _infomap_partition():
            try:
                from infomap import Infomap
            except Exception:
                if not silent:
                    print("TGraph.CommunityPartition - Warning: The 'infomap' package is not installed. Falling back to flow-weighted propagation.")
                return _propagation_partition(flow_weighted=True)

            pos = {v: i for i, v in enumerate(vertices)}
            reverse_pos = {i: v for v, i in pos.items()}

            try:
                im = Infomap("--two-level --silent")
            except Exception:
                try:
                    im = Infomap()
                except Exception:
                    if not silent:
                        print("TGraph.CommunityPartition - Warning: Could not initialise Infomap. Falling back to flow-weighted propagation.")
                    return _propagation_partition(flow_weighted=True)

            try:
                for (u, v), w in edge_weight_map.items():
                    if u == v:
                        continue
                    im.add_link(pos[u], pos[v], float(w))
                im.run()
            except Exception:
                if not silent:
                    print("TGraph.CommunityPartition - Warning: Infomap failed during execution. Falling back to flow-weighted propagation.")
                return _propagation_partition(flow_weighted=True)

            labels = {}

            try:
                for node in im.tree:
                    if getattr(node, "is_leaf", False):
                        original = reverse_pos.get(int(node.node_id), None)
                        if original is not None:
                            labels[original] = int(node.module_id)
            except Exception:
                labels = {}

            if not labels:
                try:
                    modules = im.get_modules()
                    for node_id, module_id in modules.items():
                        original = reverse_pos.get(int(node_id), None)
                        if original is not None:
                            labels[original] = int(module_id)
                except Exception:
                    labels = {}

            if not labels:
                if not silent:
                    print("TGraph.CommunityPartition - Warning: Could not read Infomap modules. Falling back to flow-weighted propagation.")
                return _propagation_partition(flow_weighted=True)

            for v in vertices:
                labels.setdefault(v, v)

            return labels

        # ---------------------------------------------------------------------
        # Dispatch
        # ---------------------------------------------------------------------

        if algorithm == "louvain":
            labels = _louvain_partition()

            if labels is None:
                return []

        elif algorithm == "propagation":
            labels = _propagation_partition()

        elif algorithm == "spectral":
            labels = _spectral_partition()

        elif algorithm == "infomap":
            labels = _infomap_partition()

        else:
            labels = _louvain_partition()

            if labels is None:
                return []

        return _write_values(labels)

    @staticmethod
    def Compare(graphA: "TGraph", graphB: "TGraph",
                weightAttributes: float = 0,
                weightStructure: float = 1,
                weightWeisfeilerLehman: float = 1,
                weightJaccard: float = 1,
                weightHopper: float = 0,
                wlKey: str = None,
                hopperKey: str = None,
                edgeWeightKey: str = None,
                iterations: int = 2,
                maxHops: int = 2,
                decay: float = 1.0,
                mantissa: int = 6,
                silent: bool = False,
                **kwargs) -> Optional[Dict[str, Any]]:
        """
        Compares two TGraphs and returns a similarity or difference report.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        weightAttributes : float , optional
            The input weight attributes value. Default is 0.
        weightStructure : float , optional
            The input weight structure value. Default is 1.
        weightWeisfeilerLehman : float , optional
            The input weight weisfeiler lehman value. Default is 1.
        weightJaccard : float , optional
            The input weight jaccard value. Default is 1.
        weightHopper : float , optional
            The input weight hopper value. Default is 0.
        wlKey : str , optional
            The dictionary key to use. Default is None.
        hopperKey : str , optional
            The dictionary key to use. Default is None.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is None.
        iterations : int , optional
            The input iterations value. Default is 2.
        maxHops : int , optional
            The input max hops value. Default is 2.
        decay : float , optional
            The input decay value. Default is 1.0.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[Dict[str, Any]]
            The resulting compare dictionary.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            if not silent:
                print("TGraph.Compare - Error: One or both inputs are not valid TGraphs. Returning None.")
            return None

        nA, nB = TGraph.Order(graphA), TGraph.Order(graphB)
        eA, eB = TGraph.Size(graphA), TGraph.Size(graphB)
        order_score = 1.0 if max(nA, nB) == 0 else 1.0 - abs(nA - nB) / float(max(nA, nB))
        size_score = 1.0 if max(eA, eB) == 0 else 1.0 - abs(eA - eB) / float(max(eA, eB))
        structure_score = (order_score + size_score) / 2.0

        scores = {}
        weights = {}

        def add_score(name, value, weight):
            try:
                weight = float(weight)
            except Exception:
                weight = 0.0
            if weight <= 0:
                return
            try:
                value = float(value)
            except Exception:
                value = 0.0
            scores[name] = max(0.0, min(1.0, value))
            weights[name] = weight

        add_score("structure", structure_score, weightStructure)

        if float(weightAttributes or 0) > 0:
            dA = TGraph.Dictionary(graphA)
            dB = TGraph.Dictionary(graphB)
            keys = set(dA) | set(dB)
            attr_score = 1.0 if not keys else sum(1 for k in keys if dA.get(k, None) == dB.get(k, None)) / float(len(keys))
            add_score("attributes", attr_score, weightAttributes)

        wl_score = TGraph.WLKernel(graphA, graphB, key=wlKey, iterations=iterations, normalize=True, mantissa=mantissa, silent=silent)
        jaccard_score = TGraph.WeightedJaccardSimilarity(graphA, graphB, edgeWeightKey=edgeWeightKey, mantissa=mantissa, silent=True)

        add_score("weisfeiler_lehman", wl_score, weightWeisfeilerLehman)
        add_score("jaccard", jaccard_score, weightJaccard)

        hopper_score = None
        if float(weightHopper or 0) > 0:
            hopper_score = TGraph.HopperKernel(graphA, graphB, key=hopperKey, maxHops=maxHops,
                                               decay=decay, normalize=True, mantissa=mantissa, silent=silent)
            add_score("hopper", hopper_score, weightHopper)

        if not scores:
            score = 0.0
        else:
            total_weight = sum(weights.values())
            score = sum(scores[k] * weights[k] for k in scores) / total_weight if total_weight > 0 else 0.0

        return {
            "orderA": nA,
            "orderB": nB,
            "sizeA": eA,
            "sizeB": eB,
            "same_order": nA == nB,
            "same_size": eA == eB,
            "order_score": round(order_score, mantissa),
            "size_score": round(size_score, mantissa),
            "structure": round(structure_score, mantissa),
            "weisfeiler_lehman": round(float(wl_score or 0.0), mantissa),
            "jaccard": round(float(jaccard_score or 0.0), mantissa),
            "hopper": None if hopper_score is None else round(float(hopper_score), mantissa),
            "score": round(score, mantissa),
            "scores": {k: round(v, mantissa) for k, v in scores.items()},
            "weights": weights,
        }

    @staticmethod
    def Compile(graph: "TGraph", weightKey: str = "weight", force: bool = False,
                useNumpy: bool = True, useSciPy: bool = True, useNumba: bool = False) -> Optional[Dict[str, Any]]:
        """
        Compiles the active graph records into compact arrays used by traversal and analysis methods.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        weightKey : str , optional
            The edge dictionary key to use as a weight. Default is 'weight'.
        force : bool , optional
            The input force value. Default is False.
        useNumpy : bool , optional
            If set to True, NumPy acceleration is used when available. Default is True.
        useSciPy : bool , optional
            If set to True, SciPy acceleration is used when available. Default is True.
        useNumba : bool , optional
            If set to True, Numba acceleration is used when available. Default is False.

        Returns
        -------
        Optional[Dict[str, Any]]
            The resulting compile dictionary.
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
        """
        Returns a compiled adjacency array for the requested adjacency mode.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        weightKey : str , optional
            The edge dictionary key to use as a weight. Default is 'weight'.

        Returns
        -------
        List[List[int]]
            The resulting compiled adjacency list.
        """
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
    def _CompiledAdjacencyKeys(mode: str) -> Tuple[str, str, str]:
        """
        Returns the compiled adjacency key names associated with the requested adjacency mode.

        Parameters
        ----------
        mode : str
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".

        Returns
        -------
        Tuple[str, str, str]
            The resulting compiled adjacency keys string.
        """
        mode_l = str(mode).lower()
        if mode_l == "in":
            return "adj_in", "indptr_in", "indices_in"
        if mode_l == "all":
            return "adj_all", "indptr_all", "indices_all"
        return "adj_out", "indptr_out", "indices_out"

    @staticmethod
    def Complement(graph: "TGraph",
                ontology: bool = True,
                tolerance: float = 0.0001,
                silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the complement graph of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """

        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.Complement - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        # Active vertex indices in the original graph.
        vertices = TGraph.ActiveVertexIndices(graph)
        n = len(vertices)

        # Preserve the actual active vertex objects in the same order as the
        # new compact index space used by the complement graph.
        original_vertices = [graph._vertices[i] for i in vertices]

        # Map old vertex indices to compact indices 0..n-1.
        mapping = {old: i for i, old in enumerate(vertices)}

        existing = set()

        for e in graph._edges:
            if not e.get("active", True):
                continue

            u = e.get("src")
            v = e.get("dst")

            if u == v or u not in mapping or v not in mapping:
                continue

            a = mapping[u]
            b = mapping[v]

            if graph._directed:
                existing.add((a, b))
            else:
                existing.add((a, b) if a <= b else (b, a))

        if graph._directed:
            pairs = (
                (i, j)
                for i in range(n)
                for j in range(n)
                if i != j and (i, j) not in existing
            )
        else:
            pairs = (
                (i, j)
                for i in range(n)
                for j in range(i + 1, n)
                if (i, j) not in existing
            )

        g = TGraph._ByEdgeIndexPairsLeanFast(
            order=n,
            edgeIndexPairs=pairs,
            directed=graph._directed,
            allowSelfLoops=False,
            allowParallelEdges=False,
            dictionary=dict(graph._dictionary),
            silent=silent,
            buildEdgeLookup=False,
            inputUnique=True,
        )

        if g is None:
            return None

        # Restore the original active vertices. This preserves x, y, z coordinates
        # and vertex dictionaries.
        if hasattr(g, "_vertices") and len(original_vertices) == n:
            g._vertices = original_vertices
        else:
            if not silent:
                print("TGraph.Complement - Warning: Could not restore original vertices.")

        return TGraph._OntologyAnnotateGraph(
            g,
            graphClass="top:Graph",
            vertexClass="top:Node",
            edgeClass="top:Relationship",
            generatedBy="TGraph.Complement",
            ontology=ontology,
            silent=silent,
        )

    @staticmethod
    def Complete(graph: "TGraph",
                ontology: bool = True,
                tolerance: float = 0.0001,
                silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the complete graph on the active vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """

        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.Complete - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        n = TGraph.Order(graph)

        # Preserve the original vertex objects before creating the complete graph.
        # This is essential because _ByEdgeIndexPairsLeanFast(order=...) creates
        # replacement vertices from indices only, and therefore loses x, y, z.
        original_vertices = list(graph._vertices)

        if graph._directed:
            pairs = ((i, j) for i in range(n) for j in range(n) if i != j)
        else:
            pairs = ((i, j) for i in range(n) for j in range(i + 1, n))

        g = TGraph._ByEdgeIndexPairsLeanFast(
            order=n,
            edgeIndexPairs=pairs,
            directed=graph._directed,
            allowSelfLoops=False,
            allowParallelEdges=False,
            dictionary=dict(graph._dictionary),
            silent=silent,
            buildEdgeLookup=False,
            inputUnique=True,
        )

        if g is None:
            return None

        # Restore the original vertices so their x, y, z coordinates and dictionaries
        # are preserved. The edge index pairs remain valid because the vertex order is
        # unchanged.
        if hasattr(g, "_vertices") and len(original_vertices) == n:
            g._vertices = original_vertices
        else:
            if not silent:
                print("TGraph.Complete - Warning: Could not restore original vertices.")

        return TGraph._OntologyAnnotateGraph(
            g,
            graphClass="top:Graph",
            vertexClass="top:Node",
            edgeClass="top:Relationship",
            generatedBy="TGraph.Complete",
            ontology=ontology,
            silent=silent,
        )

    @staticmethod
    def ConnectedComponents(graph: "TGraph", mode: str = "all") -> List[List[int]]:
        """
        Returns the connected components of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        List[List[int]]
            The resulting connected components list.
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
    def Connectivity(graph: "TGraph", key: str = "connectivity", colorKey: str = None,
                     mode: str = "all", normalize: bool = False, mantissa: int = 6,
                     silent: bool = False) -> List[float]:
        """
        Returns graph connectivity information for the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str , optional
            The dictionary key to use. Default is 'connectivity'.
        colorKey : str , optional
            The dictionary key under which computed color values are stored. Default is None.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[float]
            The resulting connectivity list.
        """
        if not isinstance(graph, TGraph):
            return []
        vals = [float(TGraph.Degree(graph, i, mode=mode)) for i in TGraph._ActiveVertexIndices(graph)]
        if normalize and vals:
            mx = max(vals) or 1.0
            vals = [v/mx for v in vals]
        vals = [round(v, mantissa) for v in vals]
        for idx, value in zip(TGraph._ActiveVertexIndices(graph), vals):
            graph._vertices[idx]["dictionary"][key] = value
        return vals

    @staticmethod
    def ContainsEdge(graph: "TGraph", edge: Any, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input TGraph contains the input edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edge : Any
            The input edge, edge index, or edge record.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not isinstance(graph, TGraph):
            return False
        idx = TGraph.EdgeIndex(graph, edge)
        if isinstance(idx, int) and graph._validate_edge_index(idx):
            return True
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src = TGraph.VertexIndex(graph, edge[0])
            dst = TGraph.VertexIndex(graph, edge[1])
            if isinstance(src, int) and isinstance(dst, int):
                return TGraph.HasEdge(graph, src, dst)
        try:
            from topologicpy.Topology import Topology
            if Topology.IsInstance(edge, "Edge"):
                for rec in graph._edges:
                    if not rec.get("active", True):
                        continue
                    rep = rec.get("representation", None)
                    if rep is edge:
                        return True
                    if rep is not None:
                        try:
                            if Topology.IsSame(rep, edge):
                                return True
                        except Exception:
                            pass
        except Exception:
            pass
        return False

    @staticmethod
    def ContainsVertex(graph: "TGraph", vertex: Any, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input TGraph contains the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Any
            The input vertex, vertex index, or vertex record.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not isinstance(graph, TGraph):
            return False
        idx = TGraph.VertexIndex(graph, vertex)
        if isinstance(idx, int) and graph._validate_vertex_index(idx):
            return True
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Vertex import Vertex
            if Topology.IsInstance(vertex, "Vertex"):
                for rec in graph._vertices:
                    if not rec.get("active", True):
                        continue
                    rep = rec.get("representation", None)
                    if rep is vertex:
                        return True
                    if rep is not None:
                        try:
                            if Topology.IsSame(rep, vertex):
                                return True
                        except Exception:
                            pass
                    coords = TGraph.Coordinates(graph, rec.get("index"), default=None)
                    if coords is not None:
                        try:
                            if Vertex.Distance(vertex, Vertex.ByCoordinates(coords[0], coords[1], coords[2])) <= tolerance:
                                return True
                        except Exception:
                            pass
        except Exception:
            pass
        return False

    @staticmethod
    def _ControlPointsToWire(controlPoints: List[Any], dictionary: Optional[Dict[str, Any]] = None,
                             tolerance: float = 0.0001, silent: bool = False) -> Optional[Any]:
        """
        Converts control-point data to a Topologic wire when possible.

        Parameters
        ----------
        controlPoints : List[Any]
            The input control points value.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Any]
            The resulting control points to wire object or value.
        """
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
    def Coordinates(graph: "TGraph", vertex: Union[int, Dict[str, Any]], default: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        Returns the coordinates associated with a vertex of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.
        default : Optional[List[float]] , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        list or None
            The [x, y, z] coordinates of the input vertex, or the default value if unavailable.
        """
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
    def Copy(graph: "TGraph") -> Optional["TGraph"]:
        """
        Returns a copy of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        return TGraph.FromPython(TGraph.ToPython(graph, includeRepresentations=True)) if isinstance(graph, TGraph) else None

    @staticmethod
    def _CopyGraph(graph: "TGraph") -> Optional["TGraph"]:
        """
        Returns an internal copy of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None
        return TGraph.FromPython(TGraph.ToPython(graph, includeRepresentations=False), ontology=False)

    @staticmethod
    def CutVertices(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Returns the articulation vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting cut vertices list.
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

    @staticmethod
    def Degree(graph: "TGraph", index: int, mode: str = "all") -> int:
        """
        Returns the degree of the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        index : int
            The input index.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        int
            The resulting degree index or count.
        """
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
    def DegreeCentrality(
        graph: "TGraph",
        weightKey: str = None,
        normalize: bool = False,
        nxCompatible: bool = True,
        useEdges: bool = False,
        edgeKey: str = None,
        key: str = "degree_centrality",
        colorKey: str = "dc_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 0.001,
        silent: bool = False,
    ) -> Optional[List[float]]:
        """
        Computes the degree centrality of the input TGraph and stores the result
        in the dictionary of each vertex or edge.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        weightKey : str , optional
            If set to None, each edge is assumed to have a weight of 1. If set to
            "length" or "distance", the geometric length of each edge is used as its
            weight. If set to any other value, the value associated with that key in
            each edge dictionary is used as the edge weight. Default is None.
        normalize : bool , optional
            If set to True, the values are normalized between 0 and 1. Default is False.
        nxCompatible : bool , optional
            Kept for consistency with other centrality functions. If set to True,
            NetworkX-style degree centrality scaling is applied. Default is True.
        useEdges : bool , optional
            If set to True, the calculation uses the edges rather than the vertices.
            Default is False.
        edgeKey : str , optional
            If not None, the value associated with that key in each edge dictionary is
            used to bundle the edges into one entity for the calculation. Otherwise,
            each edge segment is assumed to be an independent entity. Default is None.
        key : str , optional
            The desired dictionary key name under which to store the calculated value.
            Default is "degree_centrality".
        colorKey : str , optional
            The desired dictionary key name under which to store the calculated color.
            Default is "dc_color".
        colorScale : str , optional
            The desired color scale name to use for colors. Default is "viridis".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The list of centralities in the order matching the vertices or edges as requested.
        """

        import math
        import numbers

        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.DegreeCentrality - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        # ---------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------

        def _active_vertex_indices(g):
            return [v.get("index") for v in g._vertices if v.get("active", True)]

        def _active_edge_records(g):
            return [e for e in g._edges if e.get("active", True)]

        def _unwrap(x):
            if isinstance(x, list) and len(x) == 1:
                return x[0]
            return x

        def _as_float(x, default=0.0):
            x = _unwrap(x)
            if isinstance(x, numbers.Number):
                return float(x)
            try:
                return float(x)
            except Exception:
                return float(default)

        def _round(x):
            if mantissa is None or mantissa < 0:
                return float(x)
            return round(float(x), mantissa)

        def _normalize_flat(vals):
            if not vals:
                return []
            xs = [float(v) for v in vals]
            mn = min(xs)
            mx = max(xs)
            eps = tolerance if tolerance and tolerance > 0 else 1e-12
            if abs(mx - mn) < eps:
                return [0.0 for _ in xs]
            return [(x - mn) / (mx - mn) for x in xs]

        def _color(value, minValue, maxValue):
            try:
                from topologicpy.Color import Color
                return Color.AnyToHex(
                    Color.ByValueInRange(
                        value,
                        minValue=minValue,
                        maxValue=maxValue,
                        colorScale=colorScale,
                    )
                )
            except Exception:
                return None

        def _color_range(vals, unit_range=False):
            if not vals:
                return 0.0, 1.0
            if unit_range:
                return 0.0, 1.0

            mn = min(float(v) for v in vals)
            mx = max(float(v) for v in vals)

            eps = tolerance if tolerance and tolerance > 0 else 1e-12
            if abs(mx - mn) < eps:
                mx = mn + eps

            return mn, mx

        def _apply_values_to_vertices(g, vertexIndices, values, unit_range=False):
            if not vertexIndices or not values:
                return

            mn, mx = _color_range(values, unit_range=unit_range)

            for vertexIndex, value in zip(vertexIndices, values):
                if not g._validate_vertex_index(vertexIndex, active=False):
                    continue

                d = g._vertices[vertexIndex].setdefault("dictionary", {})
                v = float(value)

                if key is not None:
                    d[key] = v

                if colorKey is not None:
                    c = _color(v, mn, mx)
                    if c is not None:
                        d[colorKey] = c

        def _apply_values_to_edges(g, edgeRecords, values, unit_range=False):
            if not edgeRecords or not values:
                return

            mn, mx = _color_range(values, unit_range=unit_range)

            for edgeRecord, value in zip(edgeRecords, values):
                edgeIndex = edgeRecord.get("index", None)

                if not g._validate_edge_index(edgeIndex, active=False):
                    continue

                d = g._edges[edgeIndex].setdefault("dictionary", {})
                v = float(value)

                if key is not None:
                    d[key] = v

                if colorKey is not None:
                    c = _color(v, mn, mx)
                    if c is not None:
                        d[colorKey] = c

        def _edge_length(g, edgeRecord):
            srcIndex = edgeRecord.get("src", None)
            dstIndex = edgeRecord.get("dst", None)

            c1 = TGraph.Coordinates(g, srcIndex, default=None)
            c2 = TGraph.Coordinates(g, dstIndex, default=None)

            if c1 is None or c2 is None:
                return 1.0

            try:
                return float(math.dist(c1, c2))
            except Exception:
                return 1.0

        def _edge_weight(g, edgeRecord):
            if weightKey is None:
                return 1.0

            if isinstance(weightKey, str):
                wl = weightKey.lower()
                if ("len" in wl) or ("dis" in wl):
                    return _edge_length(g, edgeRecord)

            d = edgeRecord.get("dictionary", {})
            if isinstance(d, dict):
                return _as_float(d.get(weightKey, None), default=1.0)

            return 1.0

        # ---------------------------------------------------------------------
        # Edge mode.
        # This computes degree centrality on the implicit line graph:
        # each original edge becomes a node, and two line-graph nodes are adjacent
        # when their original edges share a vertex.
        # ---------------------------------------------------------------------

        if useEdges:
            edgeRecords = _active_edge_records(graph)

            if not edgeRecords:
                return []

            incident = {}

            for localEdgeIndex, edgeRecord in enumerate(edgeRecords):
                srcIndex = edgeRecord.get("src", None)
                dstIndex = edgeRecord.get("dst", None)

                if srcIndex is None or dstIndex is None:
                    continue

                if not graph._validate_vertex_index(srcIndex, active=False):
                    continue

                if not graph._validate_vertex_index(dstIndex, active=False):
                    continue

                incident.setdefault(srcIndex, set()).add(localEdgeIndex)
                incident.setdefault(dstIndex, set()).add(localEdgeIndex)

            per_edge = [0.0] * len(edgeRecords)

            for incidentEdges in incident.values():
                m = len(incidentEdges)

                if m <= 1:
                    continue

                add = float(m - 1)

                for localEdgeIndex in incidentEdges:
                    per_edge[localEdgeIndex] += add

            if edgeKey is not None:
                group_id = []
                group_sum = {}

                for localEdgeIndex, edgeRecord in enumerate(edgeRecords):
                    d = edgeRecord.get("dictionary", {})
                    gid = d.get(edgeKey, None) if isinstance(d, dict) else None
                    gid = _unwrap(gid)

                    if gid is None:
                        gid = "__edge_" + str(localEdgeIndex)

                    group_id.append(gid)
                    group_sum[gid] = group_sum.get(gid, 0.0) + float(per_edge[localEdgeIndex])

                values = [float(group_sum[group_id[i]]) for i in range(len(edgeRecords))]
                entity_count = len(set(group_id))
            else:
                values = [float(x) for x in per_edge]
                entity_count = len(edgeRecords)

            if nxCompatible:
                if entity_count <= 1:
                    values = [0.0 for _ in values]
                else:
                    denom = float(entity_count - 1)
                    values = [v / denom for v in values]

                unit_range_for_color = True
            else:
                if normalize:
                    values = _normalize_flat(values)
                    unit_range_for_color = True
                else:
                    unit_range_for_color = False

            out_vals = [_round(v) for v in values]

            _apply_values_to_edges(
                graph,
                edgeRecords,
                out_vals,
                unit_range=unit_range_for_color,
            )

            return out_vals

        # ---------------------------------------------------------------------
        # Vertex mode.
        # ---------------------------------------------------------------------

        vertexIndices = _active_vertex_indices(graph)
        edgeRecords = _active_edge_records(graph)

        n = len(vertexIndices)

        if n == 0:
            if not silent:
                print("TGraph.DegreeCentrality - Warning: TGraph has no active vertices. Returning [].")
            return []

        vertexIndexToPosition = {vertexIndex: i for i, vertexIndex in enumerate(vertexIndices)}

        degree = [0.0] * n

        for edgeRecord in edgeRecords:
            srcIndex = edgeRecord.get("src", None)
            dstIndex = edgeRecord.get("dst", None)

            if srcIndex not in vertexIndexToPosition or dstIndex not in vertexIndexToPosition:
                continue

            w = _edge_weight(graph, edgeRecord)

            try:
                w = float(w)
            except Exception:
                w = 1.0

            if srcIndex == dstIndex:
                # NetworkX-style degree: a self-loop contributes 2 to degree.
                degree[vertexIndexToPosition[srcIndex]] += 2.0 * w
            else:
                degree[vertexIndexToPosition[srcIndex]] += w
                degree[vertexIndexToPosition[dstIndex]] += w

        values = [float(x) for x in degree]

        if nxCompatible:
            if n <= 1:
                values = [0.0 for _ in values]
            else:
                denom = float(n - 1)
                values = [v / denom for v in values]

            unit_range_for_color = True
        else:
            if normalize:
                values = _normalize_flat(values)
                unit_range_for_color = True
            else:
                unit_range_for_color = False

        out_vals = [_round(v) for v in values]

        _apply_values_to_vertices(
            graph,
            vertexIndices,
            out_vals,
            unit_range=unit_range_for_color,
        )

        return out_vals
    
    @staticmethod
    def DegreeCentrality_old(graph: "TGraph", mode: str = "all", normalize: bool = True,
                         key: str = "degree_centrality", mantissa: int = 6) -> List[float]:
        """
        Computes degree centrality values for the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.
        key : str , optional
            The dictionary key to use. Default is 'degree_centrality'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.

        Returns
        -------
        List[float]
            The resulting degree centrality list.
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
    def DegreeMatrix(graph: "TGraph", mode: str = "all") -> List[List[int]]:
        """
        Returns the degree matrix of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        List[List[int]]
            The resulting degree matrix list.
        """
        if not isinstance(graph, TGraph):
            return []
        degrees = TGraph.DegreeSequence(graph, mode=mode)
        n = len(degrees)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i, d in enumerate(degrees):
            matrix[i][i] = d
        return matrix

    @staticmethod
    def DegreeSequence(graph: "TGraph", mode: str = "all") -> List[int]:
        """
        Returns the degree sequence of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        List[int]
            The resulting degree sequence list.
        """
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
    def Density(graph: "TGraph", includeSelfLoops: bool = False) -> float:
        """
        Returns the density of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeSelfLoops : bool , optional
            If set to True, include self loops are included. Default is False.

        Returns
        -------
        float
            The resulting density value.
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
    def Depth(graph: "TGraph", vertex: Any, source: Any = None, mode: str = "out", silent: bool = False) -> Optional[int]:
        """
        Returns the traversal depth of vertices from a source vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Any
            The input vertex, vertex index, or vertex record.
        source : Any , optional
            The input source vertex, vertex index, or source identifier. Default is None.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[int]
            The resulting depth index or count.
        """
        if not isinstance(graph, TGraph):
            return None
        target = TGraph.VertexIndex(graph, vertex)
        if target is None:
            return None
        if source is None:
            active = TGraph._ActiveVertexIndices(graph)
            if not active:
                return None
            src = active[0]
        else:
            src = TGraph.VertexIndex(graph, source)
        if src is None:
            return None
        return TGraph.TopologicalDistance(graph, src, target, mode=mode, silent=silent)

    @staticmethod
    def DepthFirstSearch(graph: "TGraph", source: int, mode: str = "out") -> List[int]:
        """
        Returns the depth-first traversal order from the input source vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        source : int
            The input source vertex, vertex index, or source identifier.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.

        Returns
        -------
        List[int]
            The resulting depth first search list.
        """
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
    def DepthMap(graph: "TGraph", source: Any = None, mode: str = "out", key: str = "depth",
                 silent: bool = False) -> Dict[int, int]:
        """
        Returns a dictionary mapping vertices to traversal depths.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        source : Any , optional
            The input source vertex, vertex index, or source identifier. Default is None.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        key : str , optional
            The dictionary key to use. Default is 'depth'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Dict[int, int]
            The resulting depth map dictionary.
        """
        if not isinstance(graph, TGraph):
            return {}
        active = TGraph._ActiveVertexIndices(graph)
        if not active:
            return {}
        src = active[0] if source is None else TGraph.VertexIndex(graph, source)
        if src is None:
            return {}
        from collections import deque as _deque
        depths = {src:0}
        q = _deque([src])
        while q:
            v=q.popleft()
            for u in TGraph.AdjacentIndices(graph,v,mode=mode):
                if u not in depths and graph._validate_vertex_index(u):
                    depths[u]=depths[v]+1
                    q.append(u)
        for idx, depth in depths.items():
            graph._vertices[idx]["dictionary"][key]=depth
        return depths

    @staticmethod
    def Diameter(graph: "TGraph", mode: str = "all") -> Optional[int]:
        """
        Returns the graph diameter of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        Optional[int]
            The resulting diameter index or count.
        """
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
    def Dictionary(graph: "TGraph") -> Dict[str, Any]:
        """
        Returns the dictionary of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        dict
            The graph dictionary.
        """
        return dict(graph._dictionary) if isinstance(graph, TGraph) else {}

    @staticmethod
    def _DictionaryToPython(dictionary: Any) -> Dict[str, Any]:
        """
        Converts a Topologic dictionary or Python dictionary to a Python dictionary.

        Parameters
        ----------
        dictionary : Any
            The input dictionary.

        Returns
        -------
        Dict[str, Any]
            The resulting dictionary to python dictionary.
        """
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
    def Difference(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the graph difference between the first and second input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            return None
        g = TGraph._CopyGraph(graphA)
        for e in list(g._edges):
            if not e.get("active", True):
                continue
            if TGraph.HasEdge(graphB, e.get("src"), e.get("dst"), directed=e.get("directed", g._directed)):
                g.RemoveEdge(e.get("index"), silent=True)
        return g

    @staticmethod
    def _DijkstraStateetVertexValue(graph: "TGraph", stable_index: int, key: Optional[str], value: Any) -> None:
        """
        Sets a vertex dictionary value using Dijkstra-state indexing.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        stable_index : int
            The input stable index value.
        key : Optional[str]
            The dictionary key to use.
        value : Any
            The input value value.

        Returns
        -------
        None
            None.
        """
        if key is None or not isinstance(graph, TGraph):
            return
        if graph._validate_vertex_index(stable_index, active=False):
            graph._vertices[stable_index].setdefault("dictionary", {})[key] = value

    @staticmethod
    def Distance(graph: "TGraph", vertexA: Any, vertexB: Any, distanceType: str = "topological",
                 mode: str = "out", mantissa: int = 6, silent: bool = False) -> Optional[float]:
        """
        Returns the shortest-path distance between two vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexA : Any
            The first input vertex or vertex index.
        vertexB : Any
            The second input vertex or vertex index.
        distanceType : str , optional
            The input distance type value. Default is 'topological'.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[float]
            The resulting distance value.
        """
        if str(distanceType or "topological").lower().startswith("metric"):
            return TGraph.MetricDistance(graph, vertexA, vertexB, mantissa=mantissa, silent=silent)
        return TGraph.TopologicalDistance(graph, vertexA, vertexB, mode=mode, silent=silent)

    @staticmethod
    def Edge(graph: "TGraph", index: Any = None, vertexA: Any = None, vertexB: Any = None,
             directed: Optional[bool] = None, silent: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns an edge record or Topologic edge from the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        index : Any , optional
            The input index. Default is None.
        vertexA : Any , optional
            The first input vertex or vertex index. Default is None.
        vertexB : Any , optional
            The second input vertex or vertex index. Default is None.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict or topologic_core.Edge or None
            The requested edge record or Topologic edge.
        """
        if not isinstance(graph, TGraph):
            return None
        if vertexA is not None and vertexB is not None:
            a = TGraph.VertexIndex(graph, vertexA); b = TGraph.VertexIndex(graph, vertexB)
            if a is None or b is None:
                return None
            return TGraph.EdgeBetween(graph, a, b, directed=directed)
        idx = TGraph.EdgeIndex(graph, index)
        if idx is None or not graph._validate_edge_index(idx):
            return None
        e = graph._edges[idx]
        return dict(e, dictionary=dict(e.get("dictionary", {})))

    def _edge_key(self, src: int, dst: int, directed: bool) -> Tuple[int, int, bool]:
        """
        Returns the canonical edge lookup key for a source, destination, and direction flag.

        Parameters
        ----------
        src : int
            The source vertex index.
        dst : int
            The destination vertex index.
        directed : bool
            If set to True, graph edges are treated as directed.

        Returns
        -------
        Tuple[int, int, bool]
            The resulting edge key index or count.
        """
        if directed:
            return (src, dst, True)
        a, b = (src, dst) if src <= dst else (dst, src)
        return (a, b, False)

    @staticmethod
    def EdgeBetween(graph: "TGraph", src: int, dst: int, directed: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        Returns the first active edge between two input vertex indices.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        src : int
            The source vertex index.
        dst : int
            The destination vertex index.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.

        Returns
        -------
        Optional[Dict[str, Any]]
            The resulting edge between dictionary.
        """
        edges = TGraph.EdgesBetween(graph, src, dst, directed=directed)
        return edges[0] if edges else None

    def EdgeDictionary(self, index: int) -> Dict[str, Any]:
        """
        Returns the dictionary of the input edge index.

        Parameters
        ----------
        index : int
            The input index.

        Returns
        -------
        Dict[str, Any]
            The resulting edge dictionary dictionary.
        """
        if not self._validate_edge_index(index, active=False):
            return {}
        return dict(self._edges[index].get("dictionary", {}))

    @staticmethod
    def EdgeIndex(graph: "TGraph", edge: Union[int, Dict[str, Any]]) -> Optional[int]:
        """
        Returns the index of the input edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edge : Union[int, Dict[str, Any]]
            The input edge, edge index, or edge record.

        Returns
        -------
        Optional[int]
            The resulting edge index index or count.
        """
        idx = TGraph._as_index(edge)
        if idx is None and isinstance(edge, dict):
            idx = edge.get("index")
        return idx if isinstance(graph, TGraph) and graph._validate_edge_index(idx) else None

    @staticmethod
    def _EdgeLength(graph: "TGraph", edge: Dict[str, Any], mantissa: int = 6, tolerance: float = 0.0001) -> float:
        """
        Returns the geometric length of an edge record.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edge : Dict[str, Any]
            The input edge, edge index, or edge record.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        float
            The resulting edge length value.
        """
        c1 = TGraph.Coordinates(graph, edge.get("src"))
        c2 = TGraph.Coordinates(graph, edge.get("dst"))
        if c1 is None or c2 is None:
            return 1.0
        return round(math.dist(c1, c2), mantissa)

    @staticmethod
    def _EdgeRepresentationToTopology(representation: Any, dictionary: Optional[Dict[str, Any]] = None,
                                      segmentCurves: bool = True, tolerance: float = 0.0001,
                                      silent: bool = False) -> Optional[Any]:
        """
        Converts an edge representation to a Topologic topology when possible.

        Parameters
        ----------
        representation : Any
            The optional representation object to store with the graph record.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        segmentCurves : bool , optional
            The input segment curves value. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Any]
            The resulting edge representation to topology object or value.
        """
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
    def Edges(graph: "TGraph", asTopologic: bool = False, useRepresentation: bool = True,
              activeOnly: bool = True, segmentCurves: bool = True, mantissa: int = 6,
              tolerance: float = 0.0001, silent: bool = False,
              selfLoopMode: str = "circle", selfLoopRadius: float = 0.25,
              selfLoopMajorRadius: Optional[float] = None, selfLoopMinorRadius: Optional[float] = None,
              selfLoopSides: int = 32, selfLoopNormal: Optional[List[float]] = None,
              sagittaKey: str = "sagitta") -> List[Any]:
        """
        Returns the edges of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        asTopologic : bool , optional
            If set to True, records are returned as Topologic objects when possible. Default is
            False.
        useRepresentation : bool , optional
            If set to True, stored representation objects are used when possible. Default is
            True.
        activeOnly : bool , optional
            If set to True, only active records are considered. Default is True.
        segmentCurves : bool , optional
            The input segment curves value. Default is True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        selfLoopMode : str , optional
            The input self loop mode value. Default is 'circle'.
        selfLoopRadius : float , optional
            The radius value to use. Default is 0.25.
        selfLoopMajorRadius : Optional[float] , optional
            The radius value to use. Default is None.
        selfLoopMinorRadius : Optional[float] , optional
            The radius value to use. Default is None.
        selfLoopSides : int , optional
            The input self loop sides value. Default is 32.
        selfLoopNormal : Optional[List[float]] , optional
            The input self loop normal value. Default is None.
        sagittaKey : str , optional
            The dictionary key to use. Default is 'sagitta'.

        Returns
        -------
        list
            The requested edge records or Topologic edge representations.
        """
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
    def EdgesBetween(graph: "TGraph", src: int, dst: int, directed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Returns all active edges between two input vertex indices.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        src : int
            The source vertex index.
        dst : int
            The destination vertex index.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting edges between list.
        """
        if not isinstance(graph, TGraph):
            return []
        TGraph._EnsureEdgeLookup(graph)
        edge_directed = graph._directed if directed is None else bool(directed)
        ids = graph._edge_lookup.get(graph._edge_key(src, dst, edge_directed), set())
        return [TGraph.Edge(graph, i) for i in sorted(ids) if graph._validate_edge_index(i)]

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
        Computes eigenvector centrality values for the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        iterations : int , optional
            The input iterations value. Default is 100.
        tolerance : float , optional
            The desired tolerance. Default is 1e-09.
        key : str , optional
            The dictionary key to use. Default is 'eigenvector_centrality'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.

        Returns
        -------
        List[float]
            The resulting eigenvector centrality list.
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
        """
        Computes eigenvector centrality values for the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is False.
        key : str , optional
            The dictionary key to use. Default is 'eigen_vector_centrality'.
        colorKey : str , optional
            The dictionary key under which computed color values are stored. Default is
            'evc_color'.
        colorScale : str , optional
            The Plotly color scale to use. Default is 'viridis'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[float]
            The resulting eigen vector centrality list.
        """
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
    def _EnsureEdgeLookup(graph: "TGraph") -> None:
        """
        Ensures that the edge lookup dictionary is populated for the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        None
            None.
        """
        if not isinstance(graph, TGraph):
            return
        if graph._dictionary.get("__edge_lookup_valid__", True) is True and graph._edge_lookup:
            return
        edge_lookup = {}
        for e in graph._edges:
            if not e.get("active", True):
                continue
            key = graph._edge_key(e.get("src"), e.get("dst"), bool(e.get("directed", graph._directed)))
            edge_lookup.setdefault(key, set()).add(e.get("index"))
        graph._edge_lookup = edge_lookup
        graph._dictionary["__edge_lookup_valid__"] = True

    @staticmethod
    def _ExportDictionary(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns an export-safe copy of a dictionary.

        Parameters
        ----------
        data : Optional[Dict[str, Any]]
            The input data dictionary.

        Returns
        -------
        Dict[str, Any]
            The resulting export dictionary dictionary.
        """
        if not isinstance(data, dict):
            return {}
        return {str(k): TGraph._ExportScalar(v) for k, v in data.items()}

    @staticmethod
    def ExportJSONLD(graph: "TGraph", path: str, indent: Optional[int] = 2, silent: bool = False, **kwargs) -> Optional[str]:
        """
        Exports the input TGraph to a JSON-LD file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        indent : Optional[int] , optional
            The input indent value. Default is 2.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting export jsonld string.
        """
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
    def _ExportScalar(value: Any) -> Any:
        """
        Returns an export-safe scalar value.

        Parameters
        ----------
        value : Any
            The input value value.

        Returns
        -------
        Any
            The resulting export scalar object or value.
        """
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    @staticmethod
    def ExportToBOT(graph: "TGraph", path: str, overwrite: bool = False, silent: bool = False, **kwargs) -> Optional[str]:
        """
        Exports the input TGraph to a BOT-compatible TTL file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        overwrite : bool , optional
            The input overwrite value. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting export to bot string.
        """
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
    def ExportToGEXF(graph: "TGraph", path: str = None, graphWidth: float = 20, graphLength: float = 20,
                     graphHeight: float = 20, defaultVertexColor: str = "black", defaultVertexSize: float = 3,
                     vertexLabelKey: str = None, vertexColorKey: str = None, vertexSizeKey: str = None,
                     defaultEdgeColor: str = "black", defaultEdgeWeight: float = 1,
                     defaultEdgeType: str = "undirected", edgeLabelKey: str = None,
                     edgeColorKey: str = None, edgeWeightKey: str = None, overwrite: bool = False,
                     mantissa: int = 6, tolerance: float = 0.0001) -> Optional[str]:
        """
        Exports the input TGraph to a GEXF file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str , optional
            The input file path, folder path, or sequence of vertex indices, depending on
            context. Default is None.
        graphWidth : float , optional
            The input graph width value. Default is 20.
        graphLength : float , optional
            The input graph length value. Default is 20.
        graphHeight : float , optional
            The input graph height value. Default is 20.
        defaultVertexColor : str , optional
            The color value to use. Default is 'black'.
        defaultVertexSize : float , optional
            The input default vertex size value. Default is 3.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is None.
        vertexColorKey : str , optional
            The dictionary key to use. Default is None.
        vertexSizeKey : str , optional
            The dictionary key to use. Default is None.
        defaultEdgeColor : str , optional
            The color value to use. Default is 'black'.
        defaultEdgeWeight : float , optional
            The input default edge weight value. Default is 1.
        defaultEdgeType : str , optional
            The input default edge type value. Default is 'undirected'.
        edgeLabelKey : str , optional
            The dictionary key to use. Default is None.
        edgeColorKey : str , optional
            The dictionary key to use. Default is None.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is None.
        overwrite : bool , optional
            The input overwrite value. Default is False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        Optional[str]
            The resulting export to gexf string.
        """

        import os
        if not isinstance(graph, TGraph) or path is None:
            return None
        if os.path.exists(path) and not overwrite:
            return None
        try:
            nx_graph = TGraph.NetworkXGraph(graph, scalarAttributes=True)
            if nx_graph is None:
                return None
            for node, attrs in nx_graph.nodes(data=True):
                if vertexLabelKey and vertexLabelKey in attrs:
                    attrs["label"] = attrs.get(vertexLabelKey)
                if vertexColorKey and vertexColorKey in attrs:
                    attrs["viz_color"] = attrs.get(vertexColorKey)
                if vertexSizeKey and vertexSizeKey in attrs:
                    attrs["viz_size"] = attrs.get(vertexSizeKey)
                else:
                    attrs.setdefault("size", defaultVertexSize)
            for u, v, attrs in nx_graph.edges(data=True):
                if edgeLabelKey and edgeLabelKey in attrs:
                    attrs["label"] = attrs.get(edgeLabelKey)
                if edgeWeightKey and edgeWeightKey in attrs:
                    attrs["weight"] = attrs.get(edgeWeightKey)
                else:
                    attrs.setdefault("weight", defaultEdgeWeight)
            import networkx as nx
            nx.write_gexf(nx_graph, path)
            return path
        except Exception:
            return None

    @staticmethod
    def ExportToGraphVizGraph(graph: "TGraph", path: str, overwrite: bool = False, silent: bool = False, **kwargs) -> Optional[str]:
        """
        Exports the input TGraph to a GraphViz graph object or file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        overwrite : bool , optional
            The input overwrite value. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting export to graph viz graph string.
        """

        import os
        if path is None:
            return None
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("TGraph.ExportToGraphVizGraph - Error: File exists and overwrite is False. Returning None.")
            return None
        try:
            dot = TGraph.GraphVizGraph(graph, **kwargs)
            with open(path, "w", encoding="utf-8") as f:
                f.write(dot)
            return path
        except Exception as exc:
            if not silent:
                print("TGraph.ExportToGraphVizGraph - Error:", exc)
            return None

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
        """
        Exports the input TGraph to a JSON file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        propertiesKey : str , optional
            The dictionary key to use. Default is 'properties'.
        verticesKey : str , optional
            The dictionary key to use. Default is 'vertices'.
        edgesKey : str , optional
            The dictionary key to use. Default is 'edges'.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is ''.
        edgeLabelKey : str , optional
            The dictionary key to use. Default is ''.
        xKey : str , optional
            The dictionary key to use. Default is 'x'.
        yKey : str , optional
            The dictionary key to use. Default is 'y'.
        zKey : str , optional
            The dictionary key to use. Default is 'z'.
        indent : int , optional
            The input indent value. Default is 4.
        sortKeys : bool , optional
            The input sort keys value. Default is False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        overwrite : bool , optional
            The input overwrite value. Default is False.

        Returns
        -------
        Optional[str]
            The resulting export to json string.
        """

        import os
        if not isinstance(graph, TGraph) or path is None:
            return None
        if os.path.exists(path) and not overwrite:
            return None
        try:
            data = TGraph.JSONDataGraphCompatible(
                graph,
                propertiesKey=propertiesKey,
                verticesKey=verticesKey,
                edgesKey=edgesKey,
                xKey=xKey,
                yKey=yKey,
                zKey=zKey,
                mantissa=mantissa,
            )
            if vertexLabelKey:
                for rec in data.get(verticesKey, []):
                    if vertexLabelKey in rec:
                        rec.setdefault("label", rec.get(vertexLabelKey))
            if edgeLabelKey:
                for rec in data.get(edgesKey, []):
                    if edgeLabelKey in rec:
                        rec.setdefault("label", rec.get(edgeLabelKey))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, sort_keys=sortKeys)
            return path
        except Exception:
            return None

    @staticmethod
    def ExportToJSONLD(graph: "TGraph", path: str, indent: int = 2, sortKeys: bool = False, overwrite: bool = False, silent: bool = False, **kwargs) -> Optional[str]:
        """
        Exports the input TGraph to a JSON-LD file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        indent : int , optional
            The input indent value. Default is 2.
        sortKeys : bool , optional
            The input sort keys value. Default is False.
        overwrite : bool , optional
            The input overwrite value. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting export to jsonld string.
        """
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
    def ExportToRDF(graph: "TGraph", path: str, overwrite: bool = True, silent: bool = False, **kwargs) -> Optional[str]:
        """
        Exports the input TGraph to an RDF/Turtle file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        overwrite : bool , optional
            The input overwrite value. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting export to rdf string.
        """
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
    def ExportToWiki(graph: "TGraph", path: str, overwrite: bool = True, silent: bool = False, **kwargs) -> Optional[str]:
        """
        Exports the input TGraph to a Wiki-compatible text file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        overwrite : bool , optional
            The input overwrite value. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting export to wiki string.
        """
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

    @staticmethod
    def ExportTTL(graph: "TGraph", path: str, silent: bool = False, **kwargs) -> Optional[str]:
        """
        Exports the input TGraph to a TTL file.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : str
            The input file path, folder path, or sequence of vertex indices, depending on
            context.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting export ttl string.
        """
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
    def FiedlerVector(graph: "TGraph", mantissa: int = 6, silent: bool = False) -> List[float]:
        """
        Returns the Fiedler vector of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[float]
            The resulting fiedler vector list.
        """
        if not isinstance(graph, TGraph):
            return []

        vertices = TGraph.ActiveVertexIndices(graph)
        n = len(vertices)
        if n == 0:
            return []
        if n == 1:
            return [0.0]

        pos = {v: i for i, v in enumerate(vertices)}

        try:
            import numpy as np
        except Exception:
            return [0.0 for _ in vertices]

        def _fix_sign(vector):
            for val in vector:
                if abs(float(val)) > 1e-12:
                    if val < 0:
                        vector = -vector
                    break
            return vector

        # Sparse path for medium and larger graphs.
        if n > 250:
            try:
                from scipy import sparse
                from scipy.sparse.linalg import eigsh

                rows = []
                cols = []
                data = []
                degree = np.zeros(n, dtype=float)
                seen = set()

                for e in graph._edges:
                    if not e.get("active", True):
                        continue
                    u = e.get("src")
                    v = e.get("dst")
                    if u == v or u not in pos or v not in pos:
                        continue
                    i = pos[u]
                    j = pos[v]
                    a, b = (i, j) if i <= j else (j, i)
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([1.0, 1.0])
                    degree[i] += 1.0
                    degree[j] += 1.0

                A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
                D = sparse.diags(degree, offsets=0, shape=(n, n), dtype=float, format="csr")
                L = D - A

                vals, vecs = eigsh(L, k=2, which="SM", tol=1e-8)
                order = np.argsort(vals)
                vector = vecs[:, order[1]] if len(order) > 1 else vecs[:, order[0]]
                vector = _fix_sign(vector)
                return [round(float(x), mantissa) for x in vector]
            except Exception:
                # Fall through to dense path.
                pass

        try:
            A = np.zeros((n, n), dtype=float)
            for e in graph._edges:
                if not e.get("active", True):
                    continue
                u = e.get("src")
                v = e.get("dst")
                if u == v or u not in pos or v not in pos:
                    continue
                i, j = pos[u], pos[v]
                A[i, j] = 1.0
                A[j, i] = 1.0
            D = np.diag(A.sum(axis=1))
            L = D - A
            vals, vecs = np.linalg.eigh(L)
            order = np.argsort(vals)
            vector = vecs[:, order[1]] if len(order) > 1 else vecs[:, order[0]]
            vector = _fix_sign(vector)
            return [round(float(x), mantissa) for x in vector]
        except Exception:
            return [0.0 for _ in vertices]

    @staticmethod
    def FiedlerVectorPartition(graph: "TGraph", key: str = "partition", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> List[int]:
        """
        Partitions the input TGraph using the Fiedler vector.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str , optional
            The dictionary key to use. Default is 'partition'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[int]
            The resulting fiedler vector partition list.
        """
        if not isinstance(graph, TGraph):
            return []
        vertices = TGraph.ActiveVertexIndices(graph)
        vector = TGraph.FiedlerVector(graph, mantissa=mantissa, silent=silent)
        if len(vector) != len(vertices):
            return []
        if not vector:
            return []
        has_negative = any(value < -float(tolerance) for value in vector)
        has_positive = any(value > float(tolerance) for value in vector)
        if has_negative and has_positive:
            threshold = 0.0
        else:
            sorted_values = sorted(vector)
            threshold = 0.5 * (sorted_values[(len(sorted_values) - 1) // 2] + sorted_values[len(sorted_values) // 2])
        parts = []
        for v, value in zip(vertices, vector):
            part = 0 if value <= threshold else 1
            parts.append(part)
            if key is not None:
                TGraph._SetVertexValue(graph, v, key, part)
        return parts

    @staticmethod
    def Figure(
        graph: "TGraph",
        title: str = None,
        width: int = None,
        height: int = None,
        showVertexLabels: bool = False,
        showEdgeLabels: bool = False,
        backgroundColor: str = "white",
        silent: bool = False,
        **kwargs,
    ):
        """
        Returns a Plotly figure representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        title : str , optional
            The optional figure title. Default is None.
        width : int , optional
            The figure or output width. Default is None.
        height : int , optional
            The figure or output height. Default is None.
        showVertexLabels : bool , optional
            If set to True, show vertex labels are shown. Default is False.
        showEdgeLabels : bool , optional
            If set to True, show edge labels are shown. Default is False.
        backgroundColor : str , optional
            The color value to use. Default is 'white'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Any
            The resulting figure object or value.
        """
        if not isinstance(graph, TGraph):
            return None
        try:
            import plotly.graph_objects as go
        except Exception:
            if not silent:
                print("TGraph.Figure - Error: Plotly is not installed. Returning None.")
            return None
        traces = TGraph.PlotlyData(
            graph,
            showVertexLabels=showVertexLabels,
            showEdgeLabels=showEdgeLabels,
            silent=silent,
            **kwargs,
        )
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            paper_bgcolor=backgroundColor,
            plot_bgcolor=backgroundColor,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        )
        return fig

    @staticmethod
    def _FrameFromNormal(normal: Optional[List[float]] = None) -> Tuple[List[float], List[float], List[float]]:
        """
        Returns an orthonormal frame from an input normal vector.

        Parameters
        ----------
        normal : Optional[List[float]] , optional
            The input normal value. Default is None.

        Returns
        -------
        Tuple[List[float], List[float], List[float]]
            The resulting frame from normal list.
        """
        n = TGraph._VectorNormalised(normal, default=[0.0, 0.0, 1.0])
        ref = [1.0, 0.0, 0.0]
        if abs(TGraph._VectorDot(n, ref)) > 0.9:
            ref = [0.0, 1.0, 0.0]
        u = TGraph._VectorNormalised(TGraph._VectorCross(n, ref), default=[1.0, 0.0, 0.0])
        v = TGraph._VectorNormalised(TGraph._VectorCross(n, u), default=[0.0, 1.0, 0.0])
        return u, v, n

    @staticmethod
    def FromPython(data: Dict[str, Any], ontology: bool = True) -> Optional["TGraph"]:
        """
        Creates a TGraph from a Python data dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data dictionary.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass=g._dictionary.get("ontology_class", "top:Graph"), vertexClass="top:Node",
            edgeClass="top:Relationship", generatedBy=g._dictionary.get("generated_by", "TGraph.FromPython"),
            ontology=ontology, silent=True)

    @staticmethod
    def GlobalClusteringCoefficient(graph: "TGraph", mantissa: int = 6, silent: bool = False) -> float:
        """
        Returns the global clustering coefficient of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The resulting global clustering coefficient value.
        """
        if not isinstance(graph, TGraph):
            return 0.0
        adjacency = TGraph._SimpleUndirectedNeighborSets(graph, includeSelfLoops=False)
        triangles_times_3 = 0
        connected_triples = 0
        for v, nbrs_set in adjacency.items():
            nbrs = sorted(nbrs_set)
            k = len(nbrs)
            if k < 2:
                continue
            connected_triples += k * (k - 1)
            for a in nbrs:
                a_nbrs = adjacency.get(a, set())
                for b in nbrs:
                    if a != b and b in a_nbrs:
                        triangles_times_3 += 1
        if connected_triples == 0:
            return 0.0
        return round(float(triangles_times_3) / float(connected_triples), mantissa)

    @staticmethod
    def GraphVizGraph(graph: "TGraph", directed: Optional[bool] = None, rankDir: str = "TB",
                      showVertexLabel: bool = True, vertexLabelKey: str = "label",
                      showEdgeLabel: bool = False, edgeLabelKey: str = "label") -> str:
        """
        Returns a GraphViz graph representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.
        rankDir : str , optional
            The input rank dir value. Default is 'TB'.
        showVertexLabel : bool , optional
            If set to True, show vertex label are shown. Default is True.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        showEdgeLabel : bool , optional
            If set to True, show edge label are shown. Default is False.
        edgeLabelKey : str , optional
            The dictionary key to use. Default is 'label'.

        Returns
        -------
        str
            The resulting graph viz graph string.
        """

        if not isinstance(graph, TGraph):
            return ""
        is_directed = graph._directed if directed is None else bool(directed)
        connector = "->" if is_directed else "--"
        graph_type = "digraph" if is_directed else "graph"

        def esc(value: Any) -> str:
            return str(value).replace('\\', '\\\\').replace('"', '\\"')

        lines = [f"{graph_type} TGraph {{", f"  rankdir={rankDir};"]
        for vertex in graph._vertices:
            if not vertex.get("active", True):
                continue
            d = vertex.get("dictionary", {})
            label = d.get(vertexLabelKey, vertex.get("index")) if showVertexLabel else vertex.get("index")
            lines.append(f"  {vertex.get('index')} [label=\"{esc(label)}\"];")
        emitted = set()
        for edge in graph._edges:
            if not edge.get("active", True):
                continue
            src = edge.get("src")
            dst = edge.get("dst")
            if not is_directed:
                key = tuple(sorted((src, dst))) + (edge.get("index"),)
                if key in emitted:
                    continue
                emitted.add(key)
            d = edge.get("dictionary", {})
            attrs = ""
            if showEdgeLabel:
                attrs = f" [label=\"{esc(d.get(edgeLabelKey, ''))}\"]"
            lines.append(f"  {src} {connector} {dst}{attrs};")
        lines.append("}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def HasEdge(graph: "TGraph", srcIndex: int, dstIndex: int, directed: Optional[bool] = None) -> bool:
        """
        Returns True if an edge exists between two input vertex indices.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        srcIndex : int
            The source vertex index.
        dstIndex : int
            The destination vertex index.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not isinstance(graph, TGraph) or not graph._validate_vertex_index(srcIndex) or not graph._validate_vertex_index(dstIndex):
            return False
        TGraph._EnsureEdgeLookup(graph)
        edge_directed = graph._directed if directed is None else bool(directed)
        key = graph._edge_key(srcIndex, dstIndex, edge_directed)
        return any(graph._validate_edge_index(i) for i in graph._edge_lookup.get(key, set()))

    @staticmethod
    def HopperKernel(graphA: "TGraph", graphB: "TGraph", key: str = None,
                     maxHops: int = 2, hop: int = None, labelKey: str = None,
                     decay: float = 1.0, normalize: bool = True,
                     mantissa: int = 6, silent: bool = False) -> Optional[float]:
        """
        Returns a graph kernel similarity based on hop features.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        key : str , optional
            The dictionary key to use. Default is None.
        maxHops : int , optional
            The input max hops value. Default is 2.
        hop : int , optional
            The input hop value. Default is None.
        labelKey : str , optional
            The dictionary key to use. Default is None.
        decay : float , optional
            The input decay value. Default is 1.0.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[float]
            The resulting hopper kernel value.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            if not silent:
                print("TGraph.HopperKernel - Error: One or both inputs are not valid TGraphs. Returning None.")
            return None
        if hop is not None:
            maxHops = hop
        label_key = labelKey if labelKey is not None else key
        fA = TGraph._P81HopFeatures(graphA, key=label_key, maxHops=maxHops, decay=decay)
        fB = TGraph._P81HopFeatures(graphB, key=label_key, maxHops=maxHops, decay=decay)
        return round(TGraph._P8Cosine(fA, fB, normalize=normalize), mantissa)

    @staticmethod
    def Impose(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the imposed graph relationship between two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        return TGraph.Union(graphA, graphB, silent=silent)

    @staticmethod
    def Imprint(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the imprinted graph relationship between two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        return TGraph.Union(graphA, graphB, silent=silent)

    @staticmethod
    def IncidentEdges(graph: "TGraph", index: int, mode: str = "all") -> List[Dict[str, Any]]:
        """
        Returns the active edges incident to the input vertex index.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        index : int
            The input index.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting incident edges list.
        """
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
    def IncomingEdges(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns the incoming edges of the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting incoming edges list.
        """
        return TGraph.IncidentEdges(graph, TGraph._as_index(vertex), mode="in")

    @staticmethod
    def IncomingVertices(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns the incoming adjacent vertices of the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting incoming vertices list.
        """
        return TGraph.AdjacentVertices(graph, vertex, mode="in")

    @staticmethod
    def InducedSubgraph(graph: "TGraph", vertices: Iterable[Any], silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the induced subgraph on the input vertex indices.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertices : Iterable[Any]
            The input vertices or vertex indices.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None
        indices = []
        for v in vertices or []:
            idx = TGraph.VertexIndex(graph, v)
            if isinstance(idx, int) and graph._validate_vertex_index(idx):
                indices.append(idx)
        return TGraph.Subgraph(graph, indices, induced=True)

    @staticmethod
    def Integration(graph: "TGraph", normalize: bool = True, key: str = "integration", mantissa: int = 6,
                    silent: bool = False) -> List[float]:
        """
        Computes integration values for the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.
        key : str , optional
            The dictionary key to use. Default is 'integration'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[float]
            The resulting integration list.
        """
        return TGraph.ClosenessCentrality(graph, mode="all", normalize=normalize, key=key, mantissa=mantissa)

    @staticmethod
    def Intersect(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the graph intersection of two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            return None
        common_vertices = [i for i in TGraph._ActiveVertexIndices(graphA) if i in set(TGraph._ActiveVertexIndices(graphB))]
        g = TGraph.Subgraph(graphA, common_vertices, induced=False)
        index_map = {old: new for new, old in enumerate(common_vertices)}
        for e in graphA._edges:
            if not e.get("active", True):
                continue
            srcIndex, dstIndex, edirected = e.get("src"), e.get("dst"), e.get("directed", graphA._directed)
            if srcIndex in index_map and dstIndex in index_map and TGraph.HasEdge(graphB, srcIndex, dstIndex, directed=edirected):
                d = dict(e.get("dictionary", {})); d.pop("index", None)
                g.AddEdge(index_map[srcIndex], index_map[dstIndex], directed=edirected, dictionary=d, representation=e.get("representation", None))
        return g

    def _invalidate_cache(self) -> None:
        """
        Invalidates the compiled cache of this TGraph.

        Parameters
        ----------
        None

        Returns
        -------
        None
            None.
        """
        self._version += 1
        self._compiled = None

    @staticmethod
    def InvalidateCache(graph: "TGraph") -> Optional["TGraph"]:
        """
        Invalidates the compiled cache of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None
        graph._invalidate_cache()
        return graph

    @staticmethod
    def IsBipartite(graph: "TGraph") -> bool:
        """
        Returns True if the input TGraph is bipartite.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
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
    def IsComplete(graph: "TGraph", includeSelfLoops: bool = False) -> bool:
        """
        Returns True if the input TGraph is complete.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeSelfLoops : bool , optional
            If set to True, include self loops are included. Default is False.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
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
    def IsConnected(graph: "TGraph", mode: str = "all") -> bool:
        """
        Returns True if the input TGraph is connected.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        c = TGraph.Compile(graph)
        if not isinstance(c, dict):
            return False
        if c["n"] <= 1:
            return True
        return len(TGraph.ConnectedComponents(graph, mode=mode)) == 1

    @staticmethod
    def IsDirected(graph: "TGraph") -> bool:
        """
        Returns True if the input TGraph is directed.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        bool
            True if the input TGraph is directed. Otherwise, False.
        """
        return bool(graph._directed) if isinstance(graph, TGraph) else False

    @staticmethod
    def IsEmpty(graph: "TGraph") -> bool:
        """
        Returns True if the input TGraph has no active vertices or edges.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        return not isinstance(graph, TGraph) or TGraph.Order(graph) == 0

    @staticmethod
    def IsErdoesGallai(sequence: Any, silent: bool = False) -> bool:
        """
        Returns True if the input degree sequence satisfies the Erdős-Gallai theorem.

        Parameters
        ----------
        sequence : Any
            The input sequence value.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if isinstance(sequence, TGraph):
            degrees = TGraph.DegreeSequence(sequence, mode="all")
        else:
            degrees = list(sequence or []) if isinstance(sequence, (list, tuple)) else []

        try:
            degrees = [int(d) for d in degrees]
        except Exception:
            return False

        n = len(degrees)
        if n == 0:
            return True
        if any(d < 0 or d >= n for d in degrees):
            return False

        total = sum(degrees)
        if total % 2 != 0:
            return False

        degrees.sort(reverse=True)

        # 1-based sequence and prefix sums: prefix[k] = sum(d_1..d_k).
        d = [0] + degrees
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + d[i]

        # w(k) = largest index i such that d_i >= k. Since k increases, w only
        # moves left. This gives each EG inequality in O(1) amortised time.
        w = n
        for k in range(1, n + 1):
            while w > 0 and d[w] < k:
                w -= 1

            if w > k:
                right = k * (k - 1) + k * (w - k) + (prefix[n] - prefix[w])
            else:
                right = k * (k - 1) + (prefix[n] - prefix[k])

            if prefix[k] > right:
                return False

        return True

    @staticmethod
    def IsInstance(graph: Any) -> bool:
        """
        Returns True if the input object is a TGraph.

        Parameters
        ----------
        graph : Any
            The input TGraph.

        Returns
        -------
        bool
            True if the input object is a TGraph. Otherwise, False.
        """
        return isinstance(graph, TGraph)

    @staticmethod
    def IsIsomorphic(graphA: "TGraph", graphB: "TGraph", vertexIDKey: str = None, edgeWeightKey: str = None, wlKey: str = None, iterations: int = 2, silent: bool = False) -> bool:
        """
        Returns True if two input TGraphs are isomorphic under the requested constraints.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        vertexIDKey : str , optional
            The dictionary key to use. Default is None.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is None.
        wlKey : str , optional
            The dictionary key to use. Default is None.
        iterations : int , optional
            The input iterations value. Default is 2.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            return False
        if graphA._directed != graphB._directed:
            return False
        if TGraph.Order(graphA) != TGraph.Order(graphB) or TGraph.Size(graphA) != TGraph.Size(graphB):
            return False

        vertices_a = TGraph.ActiveVertexIndices(graphA)
        vertices_b = TGraph.ActiveVertexIndices(graphB)
        n = len(vertices_a)
        if n == 0:
            return True

        sig_a = {v: TGraph._IsomorphismVertexSignature(graphA, v, vertexIDKey=vertexIDKey) for v in vertices_a}
        sig_b = {v: TGraph._IsomorphismVertexSignature(graphB, v, vertexIDKey=vertexIDKey) for v in vertices_b}
        if sorted(sig_a.values()) != sorted(sig_b.values()):
            return False

        # Candidate domains by invariant signature.
        candidates = {v: [w for w in vertices_b if sig_b[w] == sig_a[v]] for v in vertices_a}
        if any(len(candidates[v]) == 0 for v in vertices_a):
            return False

        # Search most constrained vertices first.
        order = sorted(vertices_a, key=lambda v: (len(candidates[v]), -TGraph.Degree(graphA, v, mode="all"), v))
        mapping: Dict[int, int] = {}
        used: Set[int] = set()

        def compatible_pair(u1: int, v1: int, u2: int, v2: int) -> bool:
            # Compare edge multisets in the relevant direction(s).
            if TGraph._IsomorphismEdgeValues(graphA, u1, u2, edgeWeightKey=edgeWeightKey) != TGraph._IsomorphismEdgeValues(graphB, v1, v2, edgeWeightKey=edgeWeightKey):
                return False
            if graphA._directed:
                if TGraph._IsomorphismEdgeValues(graphA, u2, u1, edgeWeightKey=edgeWeightKey) != TGraph._IsomorphismEdgeValues(graphB, v2, v1, edgeWeightKey=edgeWeightKey):
                    return False
            return True

        def backtrack(pos: int) -> bool:
            if pos >= len(order):
                return True
            u = order[pos]
            for v in sorted(candidates[u]):
                if v in used:
                    continue
                ok = True
                for mapped_u, mapped_v in mapping.items():
                    if not compatible_pair(u, v, mapped_u, mapped_v):
                        ok = False
                        break
                if not ok:
                    continue
                mapping[u] = v
                used.add(v)
                if backtrack(pos + 1):
                    return True
                used.remove(v)
                del mapping[u]
            return False

        return bool(backtrack(0))

    @staticmethod
    def IsolatedVertices(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Returns the isolated vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting isolated vertices list.
        """
        if not isinstance(graph, TGraph):
            return []
        return [TGraph.Vertex(graph, v["index"]) for v in graph._vertices if v.get("active", True) and TGraph.Degree(graph, v["index"]) == 0]

    @staticmethod
    def _IsomorphismEdgeValues(graph: "TGraph", u: int, v: int, edgeWeightKey: str = None) -> List[Any]:
        """
        Returns comparable edge values used by isomorphism checks.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        u : int
            The input u value.
        v : int
            The input v value.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is None.

        Returns
        -------
        List[Any]
            The resulting isomorphism edge values list.
        """
        if not isinstance(graph, TGraph):
            return []
        if graph._directed:
            records = TGraph.EdgesBetween(graph, u, v, directed=True)
        else:
            records = TGraph.EdgesBetween(graph, u, v, directed=False)
        values = []
        for e in records:
            if not isinstance(e, dict):
                continue
            if edgeWeightKey is None:
                values.append(1)
            else:
                values.append(e.get("dictionary", {}).get(edgeWeightKey, None))
        return sorted(values, key=lambda x: str(x))

    @staticmethod
    def _IsomorphismVertexSignature(graph: "TGraph", v: int, vertexIDKey: str = None) -> Tuple[Any, ...]:
        """
        Returns a comparable vertex signature used by isomorphism checks.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        v : int
            The input v value.
        vertexIDKey : str , optional
            The dictionary key to use. Default is None.

        Returns
        -------
        Tuple[Any, ...]
            The resulting isomorphism vertex signature object or value.
        """
        d = graph._vertices[v].get("dictionary", {}) if isinstance(graph, TGraph) else {}
        label = d.get(vertexIDKey, None) if vertexIDKey is not None else None
        if graph._directed:
            sig = (
                TGraph.Degree(graph, v, mode="in"),
                TGraph.Degree(graph, v, mode="out"),
                TGraph.HasEdge(graph, v, v, directed=True),
                label,
            )
        else:
            sig = (
                TGraph.Degree(graph, v, mode="all"),
                TGraph.HasEdge(graph, v, v, directed=False),
                label,
            )
        return sig

    @staticmethod
    def IsTree(graph: "TGraph") -> bool:
        """
        Returns True if the input TGraph is a tree.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
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
    def JSONData(graph: "TGraph", includeRepresentations: bool = False) -> Dict[str, Any]:
        """
        Returns a JSON-compatible data dictionary representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeRepresentations : bool , optional
            If set to True, stored representation objects are included in the output. Default is
            False.

        Returns
        -------
        Dict[str, Any]
            The resulting jsondata dictionary.
        """
        return TGraph.ToPython(graph, includeRepresentations=includeRepresentations)

    @staticmethod
    def JSONDataGraphCompatible(graph: "TGraph", propertiesKey: str = "properties",
                                verticesKey: str = "vertices", edgesKey: str = "edges",
                                xKey: str = "x", yKey: str = "y", zKey: str = "z",
                                mantissa: int = 6) -> Dict[str, Any]:
        """
        Returns graph-compatible JSON data representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        propertiesKey : str , optional
            The dictionary key to use. Default is 'properties'.
        verticesKey : str , optional
            The dictionary key to use. Default is 'vertices'.
        edgesKey : str , optional
            The dictionary key to use. Default is 'edges'.
        xKey : str , optional
            The dictionary key to use. Default is 'x'.
        yKey : str , optional
            The dictionary key to use. Default is 'y'.
        zKey : str , optional
            The dictionary key to use. Default is 'z'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.

        Returns
        -------
        Dict[str, Any]
            The resulting jsondata graph compatible dictionary.
        """

        if not isinstance(graph, TGraph):
            return {}
        data = {
            propertiesKey: dict(graph._dictionary),
            verticesKey: [],
            edgesKey: [],
        }
        data[propertiesKey].setdefault("directed", graph._directed)
        data[propertiesKey].setdefault("allowSelfLoops", graph._allow_self_loops)
        data[propertiesKey].setdefault("allowParallelEdges", graph._allow_parallel_edges)

        for vertex in graph._vertices:
            if not vertex.get("active", True):
                continue
            d = dict(vertex.get("dictionary", {}))
            d.setdefault("id", vertex.get("index"))
            coords = TGraph.Coordinates(graph, vertex.get("index"), default=None)
            if coords is not None:
                d[xKey] = round(float(coords[0]), mantissa)
                d[yKey] = round(float(coords[1]), mantissa)
                d[zKey] = round(float(coords[2]), mantissa)
            data[verticesKey].append(d)

        for edge in graph._edges:
            if not edge.get("active", True):
                continue
            d = dict(edge.get("dictionary", {}))
            d.setdefault("id", edge.get("index"))
            d.setdefault("source", edge.get("src"))
            d.setdefault("target", edge.get("dst"))
            d.setdefault("directed", edge.get("directed"))
            data[edgesKey].append(d)
        return data

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
        """
        Returns JSON-LD data representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeVertices : bool , optional
            If set to True, include vertices are included. Default is True.
        includeEdges : bool , optional
            If set to True, include edges are included. Default is True.
        includeDictionaries : bool , optional
            If set to True, dictionaries are included in the output. Default is True.
        includeBOT : bool , optional
            If set to True, include bot are included. Default is True.
        namespacePrefix : str , optional
            The input namespace prefix value. Default is 'inst'.
        instanceNamespace : str , optional
            The input instance namespace value. Default is
            'http://w3id.org/topologicpy/instance#'.

        Returns
        -------
        Dict[str, Any]
            The resulting jsonlddata dictionary.
        """
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
        """
        Returns a JSON-LD string representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        indent : Optional[int] , optional
            The input indent value. Default is 2.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        str
            The resulting jsonldstring string.
        """
        try:
            return json.dumps(TGraph.JSONLDData(graph, **kwargs), indent=indent)
        except Exception:
            return "{}"

    @staticmethod
    def JSONString(graph: "TGraph", indent: Optional[int] = None, includeRepresentations: bool = False) -> str:
        """
        Returns a JSON string representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        indent : Optional[int] , optional
            The input indent value. Default is None.
        includeRepresentations : bool , optional
            If set to True, stored representation objects are included in the output. Default is
            False.

        Returns
        -------
        str
            The resulting jsonstring string.
        """
        try:
            return json.dumps(TGraph.JSONData(graph, includeRepresentations=includeRepresentations), indent=indent)
        except Exception:
            return "{}"

    @staticmethod
    def Kernel(graphA: "TGraph", graphB: "TGraph", method: str = "WL", key: str = None,
               labelKey: str = None, iterations: int = 2, maxHops: int = 2,
               hop: int = None, decay: float = 1.0, normalize: bool = True,
               mantissa: int = 6, silent: bool = False, **kwargs) -> Optional[float]:
        """
        Returns a graph kernel similarity score between two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        method : str , optional
            The input method value. Default is 'WL'.
        key : str , optional
            The dictionary key to use. Default is None.
        labelKey : str , optional
            The dictionary key to use. Default is None.
        iterations : int , optional
            The input iterations value. Default is 2.
        maxHops : int , optional
            The input max hops value. Default is 2.
        hop : int , optional
            The input hop value. Default is None.
        decay : float , optional
            The input decay value. Default is 1.0.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[float]
            The resulting kernel value.
        """
        m = str(method or "WL").strip().lower()
        label_key = labelKey if labelKey is not None else key
        if m in ("wl", "weisfeiler", "weisfeiler-lehman", "weisfeiler_lehman"):
            return TGraph.WLKernel(graphA, graphB, key=label_key, iterations=iterations,
                                   normalize=normalize, mantissa=mantissa, silent=silent)
        if m in ("hopper", "graphhopper", "graph_hopper"):
            return TGraph.HopperKernel(graphA, graphB, key=label_key, maxHops=maxHops,
                                       hop=hop, decay=decay, normalize=normalize,
                                       mantissa=mantissa, silent=silent)
        if not silent:
            print(f'TGraph.Kernel - Error: Unsupported method "{method}". Supported methods are "WL" and "Hopper". Returning None.')
        return None

    @staticmethod
    def Laplacian(graph: "TGraph", mode: str = "all", silent: bool = False) -> List[List[int]]:
        """
        Returns the Laplacian matrix of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[List[int]]
            The resulting laplacian list.
        """
        if not isinstance(graph, TGraph):
            return []
        A = TGraph.AdjacencyMatrix(graph)
        n = len(A)
        L = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            deg = sum(1 for x in A[i] if x != 0)
            for j in range(n):
                L[i][j] = deg if i == j else (-1 if A[i][j] != 0 else 0)
        return L

    @staticmethod
    def Leaves(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Returns the leaf vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting leaves list.
        """
        if not isinstance(graph, TGraph):
            return []
        return [TGraph.Vertex(graph, v["index"]) for v in graph._vertices if v.get("active", True) and TGraph.Degree(graph, v["index"]) == 1]

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
        Returns the line graph of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.
        transferDictionaries : bool , optional
            The input transfer dictionaries value. Default is True.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        originalEdgeKey : str , optional
            The dictionary key to use. Default is 'original_edge_index'.
        sharedVertexKey : str , optional
            The dictionary key to use. Default is 'shared_vertex_index'.
        relationshipKey : str , optional
            The dictionary key to use. Default is 'relationship'.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
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

        return TGraph._OntologyAnnotateGraph(
            lg, graphClass="top:LineGraph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.LineGraph", ontology=True, silent=True)

    @staticmethod
    def LocalClusteringCoefficient(graph: "TGraph", vertices: list = None, key: str = "local_clustering_coefficient", mantissa: int = 6, silent: bool = False) -> List[float]:
        """
        Computes local clustering coefficient values for the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertices : list , optional
            The input vertices or vertex indices. Default is None.
        key : str , optional
            The dictionary key to use. Default is 'local_clustering_coefficient'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[float]
            The resulting local clustering coefficient list.
        """
        if not isinstance(graph, TGraph):
            return []
        adjacency = TGraph._SimpleUndirectedNeighborSets(graph, includeSelfLoops=False)
        if vertices is None:
            selected = TGraph.ActiveVertexIndices(graph)
        else:
            selected = []
            for v in vertices:
                idx = TGraph._as_index(v)
                if graph._validate_vertex_index(idx):
                    selected.append(idx)
        values = []
        for v in selected:
            nbrs = sorted(adjacency.get(v, set()))
            k = len(nbrs)
            if k < 2:
                coeff = 0.0
            else:
                links = 0
                for i, a in enumerate(nbrs):
                    a_nbrs = adjacency.get(a, set())
                    for b in nbrs[i + 1:]:
                        if b in a_nbrs:
                            links += 1
                coeff = (2.0 * float(links)) / float(k * (k - 1))
            coeff = round(float(coeff), mantissa)
            values.append(coeff)
            if key is not None:
                TGraph._SetVertexValue(graph, v, key, coeff)
        return values

    @staticmethod
    def LongestPath(graph: "TGraph", mode: str = "all", silent: bool = False) -> List[int]:
        """
        Returns a longest path approximation for the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[int]
            The resulting longest path list.
        """
        if not isinstance(graph, TGraph):
            return []
        best=[]
        active=TGraph._ActiveVertexIndices(graph)
        for i,a in enumerate(active):
            for b in active[i+1:]:
                p=TGraph.ShortestPath(graph,a,b,mode=mode)
                if p and len(p)>len(best):
                    best=p
        return best

    @staticmethod
    def Match(graphA: "TGraph", graphB: "TGraph", vertexKeys: Any = None, edgeKeys: Any = None,
              maxMatches: int = 25, timeLimit: float = 5, exact: bool = False,
              silent: bool = False) -> List[Dict[int, int]]:
        """
        Returns subgraph matches of a pattern graph in a target graph.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.
        maxMatches : int , optional
            The input max matches value. Default is 25.
        timeLimit : float , optional
            The maximum time, in seconds, allowed for the search. Default is 5.
        exact : bool , optional
            The input exact value. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[Dict[int, int]]
            The resulting match list.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            if not silent:
                print("TGraph.Match - Error: One or both inputs are not valid TGraphs. Returning None.")
            return None

        if exact:
            return TGraph._P8SubgraphIsomorphisms(
                graphA, graphB,
                vertexKeys=vertexKeys,
                edgeKeys=edgeKeys,
                maxMatches=maxMatches,
                timeLimit=timeLimit,
                exact=True,
            )

        if TGraph.Order(graphA) <= TGraph.Order(graphB):
            pattern, superGraph = graphA, graphB
        else:
            pattern, superGraph = graphB, graphA

        return TGraph._P8SubgraphIsomorphisms(
            pattern, superGraph,
            vertexKeys=vertexKeys,
            edgeKeys=edgeKeys,
            maxMatches=maxMatches,
            timeLimit=timeLimit,
            exact=False,
        )

    @staticmethod
    def MaximumDelta(graph: "TGraph", mode: str = "all", silent: bool = False) -> Optional[int]:
        """
        Returns the maximum vertex degree of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[int]
            The resulting maximum delta index or count.
        """
        if not isinstance(graph, TGraph):
            return None
        seq = TGraph.DegreeSequence(graph, mode=mode)
        return max(seq) if seq else 0

    @staticmethod
    def MaximumFlow(graph: "TGraph", source: Any, sink: Any, capacityKey: str = "capacity",
                    defaultCapacity: float = 1.0, silent: bool = False) -> float:
        """
        Returns the maximum flow between two vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        source : Any
            The input source vertex, vertex index, or source identifier.
        sink : Any
            The input sink value.
        capacityKey : str , optional
            The dictionary key to use. Default is 'capacity'.
        defaultCapacity : float , optional
            The input default capacity value. Default is 1.0.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The resulting maximum flow value.
        """
        if not isinstance(graph, TGraph):
            return 0.0
        s=TGraph.VertexIndex(graph,source); t=TGraph.VertexIndex(graph,sink)
        if s is None or t is None:
            return 0.0
        cap={}
        for e in graph._edges:
            if not e.get("active",True): continue
            u,v=e.get("src"),e.get("dst")
            d=e.get("dictionary",{})
            c=float(d.get(capacityKey, defaultCapacity) or defaultCapacity)
            cap[(u,v)]=cap.get((u,v),0.0)+c
            cap.setdefault((v,u),0.0)
            if not e.get("directed", graph._directed):
                cap[(v,u)]=cap.get((v,u),0.0)+c
                cap.setdefault((u,v),cap.get((u,v),0.0))
        from collections import deque as _deque
        flow=0.0
        while True:
            parent={s:None}
            q=_deque([s])
            while q and t not in parent:
                u=q.popleft()
                for (a,b),c in list(cap.items()):
                    if a==u and c>1e-12 and b not in parent:
                        parent[b]=u; q.append(b)
            if t not in parent: break
            inc=float('inf'); v=t
            while v!=s:
                u=parent[v]; inc=min(inc,cap[(u,v)]); v=u
            v=t
            while v!=s:
                u=parent[v]; cap[(u,v)]-=inc; cap[(v,u)]=cap.get((v,u),0.0)+inc; v=u
            flow+=inc
        return flow

    @staticmethod
    def Merge(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns a merged graph from two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            return None
        g = TGraph._CopyGraph(graphA)
        offset = len(g._vertices)
        for v in graphB._vertices:
            if v.get("active", True):
                d = dict(v.get("dictionary", {})); d.pop("index", None)
                g.AddVertex(dictionary=d, representation=v.get("representation", None))
        for e in graphB._edges:
            if e.get("active", True):
                d = dict(e.get("dictionary", {})); d.pop("index", None)
                d.pop("src", None); d.pop("dst", None)
                g.AddEdge(e.get("src") + offset, e.get("dst") + offset,
                          directed=e.get("directed", graphB._directed), dictionary=d,
                          representation=e.get("representation", None))
        return g

    @staticmethod
    def MeshData(graph: "TGraph", activeOnly: bool = True) -> Dict[str, Any]:
        """
        Returns mesh data representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        activeOnly : bool , optional
            If set to True, only active records are considered. Default is True.

        Returns
        -------
        Dict[str, Any]
            The resulting mesh data dictionary.
        """
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
            srcIndex = e.get("src")
            dstIndex = e.get("dst")
            if srcIndex in index_map and dstIndex in index_map:
                edges.append([index_map[srcIndex], index_map[dstIndex]])
        return {"vertices": coords, "edges": edges}

    @staticmethod
    def MetricDistance(graph: "TGraph", vertexA: Any, vertexB: Any, mantissa: int = 6,
                       silent: bool = False) -> Optional[float]:
        """
        Returns a metric distance between two input TGraphs.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexA : Any
            The first input vertex or vertex index.
        vertexB : Any
            The second input vertex or vertex index.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[float]
            The resulting metric distance value.
        """
        if not isinstance(graph, TGraph):
            return None
        a = TGraph.VertexIndex(graph, vertexA)
        b = TGraph.VertexIndex(graph, vertexB)
        ca = TGraph.Coordinates(graph, a, default=None)
        cb = TGraph.Coordinates(graph, b, default=None)
        if ca is None or cb is None:
            return None
        import math as _math
        return round(_math.sqrt((cb[0]-ca[0])**2 + (cb[1]-ca[1])**2 + (cb[2]-ca[2])**2), mantissa)

    @staticmethod
    def MinimumDelta(graph: "TGraph", mode: str = "all", silent: bool = False) -> Optional[int]:
        """
        Returns the minimum vertex degree of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[int]
            The resulting minimum delta index or count.
        """
        if not isinstance(graph, TGraph):
            return None
        seq = TGraph.DegreeSequence(graph, mode=mode)
        return min(seq) if seq else 0

    @staticmethod
    def MinimumSpanningTree(graph: "TGraph", edgeKey: str = "weight") -> Optional["TGraph"]:
        """
        Returns a minimum spanning tree of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edgeKey : str , optional
            The edge dictionary key to use. Default is 'weight'.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
        return TGraph._OntologyAnnotateGraph(
            mst, graphClass=graph._dictionary.get("ontology_class", "top:Graph"), vertexClass="top:Node",
            edgeClass="top:Relationship", generatedBy="TGraph.MinimumSpanningTree", ontology=True, silent=True)

    @staticmethod
    def _NativeEdgeBetweenness(graph: "TGraph") -> Dict[Tuple[int, int], float]:
        """
        Returns native edge betweenness scores for the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        Dict[Tuple[int, int], float]
            The resulting native edge betweenness dictionary.
        """
        if not isinstance(graph, TGraph):
            return {}
        vertices = TGraph.ActiveVertexIndices(graph)
        adjacency = TGraph._SimpleUndirectedNeighborSets(graph, includeSelfLoops=False)
        edge_bc: Dict[Tuple[int, int], float] = {}

        for s in vertices:
            stack = []
            pred = {w: [] for w in vertices}
            sigma = {w: 0.0 for w in vertices}
            dist = {w: -1 for w in vertices}
            sigma[s] = 1.0
            dist[s] = 0
            q = deque([s])

            while q:
                v = q.popleft()
                stack.append(v)
                for w in sorted(adjacency.get(v, set())):
                    if dist[w] < 0:
                        q.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta = {w: 0.0 for w in vertices}
            while stack:
                w = stack.pop()
                if sigma[w] == 0:
                    continue
                for v in pred[w]:
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    a, b = (v, w) if v <= w else (w, v)
                    edge_bc[(a, b)] = edge_bc.get((a, b), 0.0) + c
                    delta[v] += c

        # Undirected paths were counted twice.
        for edge in list(edge_bc.keys()):
            edge_bc[edge] *= 0.5
        return edge_bc

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
        Returns a navigation graph derived from the input topology.

        Parameters
        ----------
        face : Any
            The input face value.
        vertices : Optional[List[Any]] , optional
            The input vertices or vertex indices. Default is None.
        obstacles : Optional[List[Any]] , optional
            The input obstacles value. Default is None.
        bidirectional : bool , optional
            The input bidirectional value. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """

        graph = TGraph.VisibilityGraph(face, vertices=vertices, obstacles=obstacles,
                                       bidirectional=bidirectional, tolerance=tolerance, silent=silent)
        if isinstance(graph, TGraph):
            graph._dictionary["generated_by"] = "TGraph.NavigationGraph"
        return graph

    @staticmethod
    def NearestVertex(graph: "TGraph",
                      vertex: Any = None,
                      x: float = 0,
                      y: float = 0,
                      z: float = 0,
                      silent: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns the nearest active TGraph vertex record to the input coordinates, TGraph vertex,
        or Topologic vertex.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        vertex : Any , optional
            A TGraph vertex index, TGraph vertex record, or Topologic vertex. If specified,
            its coordinates override x, y, and z. Default is None.
        x : float , optional
            The X coordinate. Default is 0.
        y : float , optional
            The Y coordinate. Default is 0.
        z : float , optional
            The Z coordinate. Default is 0.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict or None
            The nearest active TGraph vertex record, or None if no valid active vertex is found.
        """

        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.NearestVertex - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        query = None

        # First try resolving as a TGraph vertex index or TGraph vertex record.
        if vertex is not None:
            query = TGraph.Coordinates(graph, vertex, default=None)

            # Then try resolving as a Topologic vertex.
            if query is None:
                try:
                    from topologicpy.Vertex import Vertex
                    coords = Vertex.Coordinates(vertex)
                    if coords and len(coords) >= 3:
                        query = [float(coords[0]), float(coords[1]), float(coords[2])]
                except Exception:
                    query = None

            if query is None:
                if not silent:
                    print("TGraph.NearestVertex - Error: Could not resolve the input vertex coordinates. Returning None.")
                return None
        else:
            try:
                query = [float(x), float(y), float(z)]
            except Exception:
                if not silent:
                    print("TGraph.NearestVertex - Error: The input coordinates are invalid. Returning None.")
                return None

        qx, qy, qz = query

        bestIndex = None
        bestDistanceSquared = None

        for vertexRecord in graph._vertices:
            if not vertexRecord.get("active", True):
                continue

            vertexIndex = vertexRecord.get("index", None)
            coordinates = TGraph.Coordinates(graph, vertexIndex, default=None)

            if coordinates is None or len(coordinates) < 3:
                continue

            dx = coordinates[0] - qx
            dy = coordinates[1] - qy
            dz = coordinates[2] - qz
            distanceSquared = dx * dx + dy * dy + dz * dz

            if bestDistanceSquared is None or distanceSquared < bestDistanceSquared:
                bestIndex = vertexIndex
                bestDistanceSquared = distanceSquared

        return TGraph.Vertex(graph, bestIndex) if bestIndex is not None else None

    @staticmethod
    def NetworkXGraph(graph: "TGraph", nodeIDKey: str = None, edgeIDKey: str = None,
                      includeInactive: bool = False, scalarAttributes: bool = False) -> Optional[Any]:
        """
        Returns a NetworkX graph representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        nodeIDKey : str , optional
            The dictionary key to use. Default is None.
        edgeIDKey : str , optional
            The dictionary key to use. Default is None.
        includeInactive : bool , optional
            If set to True, include inactive are included. Default is False.
        scalarAttributes : bool , optional
            The input scalar attributes value. Default is False.

        Returns
        -------
        Optional[Any]
            The resulting network xgraph object or value.
        """

        if not isinstance(graph, TGraph):
            return None
        try:
            import networkx as nx
        except Exception:
            return None

        if graph._allow_parallel_edges:
            nx_graph = nx.MultiDiGraph() if graph._directed else nx.MultiGraph()
        else:
            nx_graph = nx.DiGraph() if graph._directed else nx.Graph()

        index_to_node = {}
        for vertex in graph._vertices:
            if not includeInactive and not vertex.get("active", True):
                continue
            d = dict(vertex.get("dictionary", {}))
            node_id = d.get(nodeIDKey, vertex.get("index")) if nodeIDKey else vertex.get("index")
            index_to_node[vertex.get("index")] = node_id
            attrs = TGraph._ExportDictionary(d) if scalarAttributes else d
            nx_graph.add_node(node_id, **attrs)

        for edge in graph._edges:
            if not includeInactive and not edge.get("active", True):
                continue
            srcIndex = edge.get("src")
            dstIndex = edge.get("dst")
            if srcIndex not in index_to_node or dstIndex not in index_to_node:
                continue
            d = dict(edge.get("dictionary", {}))
            if edgeIDKey:
                d.setdefault(edgeIDKey, edge.get("index"))
            attrs = TGraph._ExportDictionary(d) if scalarAttributes else d
            if graph._allow_parallel_edges:
                nx_graph.add_edge(index_to_node[srcIndex], index_to_node[dstIndex], key=edge.get("index"), **attrs)
            else:
                nx_graph.add_edge(index_to_node[srcIndex], index_to_node[dstIndex], **attrs)
        return nx_graph

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
        """
        Normalizes ontology-related dictionary values in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        labelKeys : Optional[List[str]] , optional
            The input label keys value. Default is None.
        categoryKeys : Optional[List[str]] , optional
            The input category keys value. Default is None.
        ifcClassKeys : Optional[List[str]] , optional
            The input ifc class keys value. Default is None.
        ifcGUIDKeys : Optional[List[str]] , optional
            The input ifc guidkeys value. Default is None.
        includeGraph : bool , optional
            If set to True, include graph are included. Default is True.
        includeVertices : bool , optional
            If set to True, include vertices are included. Default is True.
        includeEdges : bool , optional
            If set to True, include edges are included. Default is True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
    def _NumbaBFSTreeKernel():
        """
        Runs a Numba-accelerated breadth-first-search tree kernel when available.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            The resulting numba bfstree kernel object or value.
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
    def _NumbaShortestPathKernel():
        """
        Runs a Numba-accelerated shortest path kernel when available.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            The resulting numba shortest path kernel object or value.
        """
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
    def _OntologyAnnotateDictionary(
        dictionary: Dict[str, Any],
        ontologyClass: Optional[str] = None,
        category: Optional[str] = None,
        label: Any = None,
        generatedBy: Any = None,
        source: Any = None,
        preserveExisting: bool = True,
    ) -> Dict[str, Any]:
        """
        Annotates a dictionary with ontology metadata.

        Parameters
        ----------
        dictionary : Dict[str, Any]
            The input dictionary.
        ontologyClass : Optional[str] , optional
            The ontology class value. Default is None.
        category : Optional[str] , optional
            The ontology category value. Default is None.
        label : Any , optional
            The label value. Default is None.
        generatedBy : Any , optional
            The provenance value identifying the generating method. Default is None.
        source : Any , optional
            The input source vertex, vertex index, or source identifier. Default is None.
        preserveExisting : bool , optional
            The input preserve existing value. Default is True.

        Returns
        -------
        Dict[str, Any]
            The resulting ontology annotate dictionary dictionary.
        """
        d = dictionary if isinstance(dictionary, dict) else {}
        if ontologyClass is not None and (not preserveExisting or d.get("ontology_class") in (None, "")):
            d["ontology_class"] = ontologyClass
        if category is None:
            category = TGraph._OntologyDefaultCategory(d.get("ontology_class"), fallback=d.get("category", "topology"))
        if category is not None and (not preserveExisting or d.get("category") in (None, "")):
            d["category"] = category
        if label is not None and (not preserveExisting or d.get("label") in (None, "")):
            d["label"] = label
        if generatedBy is not None and (not preserveExisting or d.get("generated_by") in (None, "")):
            d["generated_by"] = generatedBy
        if source is not None and (not preserveExisting or d.get("source") in (None, "")):
            d["source"] = source
        if d.get("ontology_class") not in (None, "") and d.get("ontology_uri") in (None, ""):
            uri = TGraph._OntologyExpandQName(str(d.get("ontology_class")), defaultValue=None)
            if uri is not None:
                d["ontology_uri"] = uri
        return d

    @staticmethod
    def _OntologyAnnotateGraph(
        graph: "TGraph",
        graphClass: str = "top:Graph",
        vertexClass: str = "top:Node",
        edgeClass: str = "top:Relationship",
        generatedBy: Optional[str] = None,
        ontology: bool = True,
        includeVertices: bool = True,
        includeEdges: bool = True,
        preserveExisting: bool = True,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Annotates graph, vertex, and edge dictionaries with ontology metadata.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        graphClass : str , optional
            The input graph class value. Default is 'top:Graph'.
        vertexClass : str , optional
            The input vertex class value. Default is 'top:Node'.
        edgeClass : str , optional
            The input edge class value. Default is 'top:Relationship'.
        generatedBy : Optional[str] , optional
            The provenance value identifying the generating method. Default is None.
        ontology : bool , optional
            If set to True, ontology metadata is added or preserved where applicable. Default is
            True.
        includeVertices : bool , optional
            If set to True, include vertices are included. Default is True.
        includeEdges : bool , optional
            If set to True, include edges are included. Default is True.
        preserveExisting : bool , optional
            The input preserve existing value. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None
        if not ontology:
            return graph

        TGraph._OntologyAnnotateDictionary(
            graph._dictionary,
            ontologyClass=graphClass,
            category="graph",
            generatedBy=generatedBy,
            preserveExisting=preserveExisting,
        )

        if includeVertices:
            for v in graph._vertices:
                d = v.setdefault("dictionary", {})
                rep = v.get("representation", None)
                inferred = d.get("ontology_class", None)
                if inferred in (None, ""):
                    ifc_class = d.get("ifc_class", d.get("IfcClass", d.get("class", None)))
                    if ifc_class not in (None, ""):
                        inferred = TGraph.OntologyClassByIFCClass(str(ifc_class), defaultValue=None)
                    if inferred in (None, ""):
                        inferred = TGraph._OntologyClassFromRepresentation(rep, defaultValue=vertexClass)
                TGraph._OntologyAnnotateDictionary(
                    d,
                    ontologyClass=inferred,
                    category=TGraph._OntologyDefaultCategory(inferred, fallback="topology"),
                    label=d.get("label", d.get("name", d.get("Name", d.get("id", d.get("index", None))))),
                    generatedBy=generatedBy,
                    preserveExisting=preserveExisting,
                )

        if includeEdges:
            for e in graph._edges:
                d = e.setdefault("dictionary", {})
                relationship = d.get("relationship", d.get("label", None))
                inferred = d.get("ontology_class", edgeClass)
                TGraph._OntologyAnnotateDictionary(
                    d,
                    ontologyClass=inferred,
                    category=TGraph._OntologyDefaultCategory(inferred, fallback="topology"),
                    label=d.get("label", relationship if relationship is not None else d.get("index", None)),
                    generatedBy=generatedBy,
                    preserveExisting=preserveExisting,
                )
                d.setdefault("src", e.get("src"))
                d.setdefault("dst", e.get("dst"))

        try:
            TGraph.NormalizeOntologyDictionaries(graph, includeGraph=True, includeVertices=includeVertices, includeEdges=includeEdges)
        except Exception:
            pass
        return graph

    @staticmethod
    def OntologyAnnotateGraph(*args, **kwargs) -> Optional["TGraph"]:
        """
        Annotates graph, vertex, and edge dictionaries with ontology metadata.

        Parameters
        ----------
        *args : Any , optional
            Additional positional arguments.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        return TGraph._OntologyAnnotateGraph(*args, **kwargs)

    @staticmethod
    def OntologyCategory(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        """
        Returns the ontology category of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        Any
            The resulting ontology category object or value.
        """
        return TGraph._OntologyGet(graph, "category", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def OntologyClass(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        """
        Returns the ontology class of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        Any
            The resulting ontology class object or value.
        """
        return TGraph._OntologyGet(graph, "ontology_class", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def OntologyClassByIFCClass(ifcClass: str, defaultValue: Any = "top:Element") -> Any:
        """
        Returns the ontology class corresponding to the input IFC class.

        Parameters
        ----------
        ifcClass : str
            The IFC class value.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is 'top:Element'.

        Returns
        -------
        Any
            The resulting ontology class by ifcclass object or value.
        """
        if ifcClass is None:
            return defaultValue
        return TGraph._OntologyConfig()["ifc_to_top"].get(str(ifcClass).strip(), defaultValue)

    @staticmethod
    def _OntologyClassFromRepresentation(representation: Any, defaultValue: str = "top:Node") -> str:
        """
        Returns an ontology class inferred from a Topologic representation.

        Parameters
        ----------
        representation : Any
            The optional representation object to store with the graph record.
        defaultValue : str , optional
            The default value to return when no valid value is found. Default is 'top:Node'.

        Returns
        -------
        str
            The resulting ontology class from representation string.
        """
        if representation is None:
            return defaultValue
        try:
            from topologicpy.Topology import Topology
            type_name = None
            try:
                type_name = Topology.TypeAsString(representation)
            except Exception:
                type_name = None
            if type_name is None:
                for candidate in ["CellComplex", "Cell", "Shell", "Face", "Wire", "Edge", "Vertex", "Cluster", "Aperture"]:
                    try:
                        if Topology.IsInstance(representation, candidate):
                            type_name = candidate
                            break
                    except Exception:
                        pass
            mapping = {
                "Vertex": "top:Vertex",
                "Edge": "top:Edge",
                "Wire": "top:Wire",
                "Face": "top:Face",
                "Shell": "top:Shell",
                "Cell": "top:Cell",
                "CellComplex": "top:CellComplex",
                "Cluster": "top:Cluster",
                "Aperture": "top:Aperture",
            }
            if type_name in mapping:
                return mapping[type_name]
        except Exception:
            pass
        return defaultValue

    @staticmethod
    def _OntologyConfig() -> Dict[str, Any]:
        """
        Returns the internal ontology configuration dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            The resulting ontology config dictionary.
        """
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
    def _OntologyDefaultCategory(ontologyClass: Optional[str], fallback: str = "topology") -> str:
        """
        Returns the default ontology category for an ontology class.

        Parameters
        ----------
        ontologyClass : Optional[str]
            The ontology class value.
        fallback : str , optional
            The input fallback value. Default is 'topology'.

        Returns
        -------
        str
            The resulting ontology default category string.
        """
        if ontologyClass is None:
            return fallback
        category = TGraph.CategoryByOntologyClass(ontologyClass, defaultValue=None)
        return category if category is not None else fallback

    @staticmethod
    def _OntologyDictionary(graph: "TGraph", element: str = "graph", index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Returns the ontology dictionary for a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Optional[Dict[str, Any]]
            The resulting ontology dictionary dictionary.
        """
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
    def _OntologyExpandQName(qname: str, defaultValue: Any = None) -> Any:
        """
        Expands a QName to a full ontology URI.

        Parameters
        ----------
        qname : str
            The input qname value.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        Any
            The resulting ontology expand qname object or value.
        """
        if not isinstance(qname, str) or ":" not in qname:
            return defaultValue
        prefix, local = qname.split(":", 1)
        ns = TGraph._OntologyConfig()["namespaces"].get(prefix)
        if ns is None:
            return defaultValue
        return ns + local

    @staticmethod
    def _OntologyGet(graph: "TGraph", key: str, defaultValue: Any = None, element: str = "graph", index: Optional[int] = None) -> Any:
        """
        Returns an ontology value from a graph, vertex, or edge dictionary.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str
            The dictionary key to use.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Any
            The resulting ontology get object or value.
        """
        d = TGraph._OntologyDictionary(graph, element=element, index=index)
        if not isinstance(d, dict):
            return defaultValue
        value = d.get(key, defaultValue)
        return defaultValue if value is None else value

    @staticmethod
    def OntologyLabel(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        """
        Returns the ontology label of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        Any
            The resulting ontology label object or value.
        """
        return TGraph._OntologyGet(graph, "label", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def _OntologyPropertyQName(key: str, defaultPrefix: str = "top") -> Optional[str]:
        """
        Returns the RDF property QName for an ontology dictionary key.

        Parameters
        ----------
        key : str
            The dictionary key to use.
        defaultPrefix : str , optional
            The input default prefix value. Default is 'top'.

        Returns
        -------
        Optional[str]
            The resulting ontology property qname string.
        """
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
    def _OntologyRDFLiteral(value: Any) -> str:
        """
        Returns an RDF literal string for a Python value.

        Parameters
        ----------
        value : Any
            The input value value.

        Returns
        -------
        str
            The resulting ontology rdfliteral string.
        """
        if isinstance(value, bool):
            return '"' + str(value).lower() + '"^^xsd:boolean'
        if isinstance(value, int) and not isinstance(value, bool):
            return '"' + str(value) + '"^^xsd:integer'
        if isinstance(value, float):
            return '"' + repr(float(value)) + '"^^xsd:double'
        return '"' + TGraph._OntologySafeString(value) + '"'

    @staticmethod
    def _OntologySafeLocalName(value: Any) -> str:
        """
        Returns a safe local name for ontology serialization.

        Parameters
        ----------
        value : Any
            The input value value.

        Returns
        -------
        str
            The resulting ontology safe local name string.
        """
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
    def _OntologySafeString(value: Any) -> str:
        """
        Returns a safe string for ontology serialization.

        Parameters
        ----------
        value : Any
            The input value value.

        Returns
        -------
        str
            The resulting ontology safe string string.
        """
        if value is None:
            return ""
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    @staticmethod
    def _OntologySet(graph: "TGraph", key: str, value: Any, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        """
        Sets an ontology value on a graph, vertex, or edge dictionary.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str
            The dictionary key to use.
        value : Any
            The input value value.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
    def _OntologySubjectFromDictionary(dictionary: Dict[str, Any], fallback: str, namespacePrefix: str = "inst") -> str:
        """
        Returns an RDF subject URI from an ontology dictionary.

        Parameters
        ----------
        dictionary : Dict[str, Any]
            The input dictionary.
        fallback : str
            The input fallback value.
        namespacePrefix : str , optional
            The input namespace prefix value. Default is 'inst'.

        Returns
        -------
        str
            The resulting ontology subject from dictionary string.
        """
        d = dictionary if isinstance(dictionary, dict) else {}
        for key in ("uri", "ifc_guid", "global_id", "guid", "label", "name"):
            value = d.get(key, None)
            if value not in (None, ""):
                if key == "uri" and ":" in str(value):
                    return str(value)
                return namespacePrefix + ":" + TGraph._OntologySafeLocalName(value)
        return namespacePrefix + ":" + TGraph._OntologySafeLocalName(fallback)

    @staticmethod
    def OntologyTriples(
        graph: "TGraph",
        includeVertices: bool = True,
        includeEdges: bool = True,
        includeDictionaries: bool = True,
        includeBOT: bool = True,
        namespacePrefix: str = "inst",
    ) -> List[Tuple[str, str, str]]:
        """
        Returns ontology triples representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeVertices : bool , optional
            If set to True, include vertices are included. Default is True.
        includeEdges : bool , optional
            If set to True, include edges are included. Default is True.
        includeDictionaries : bool , optional
            If set to True, dictionaries are included in the output. Default is True.
        includeBOT : bool , optional
            If set to True, include bot are included. Default is True.
        namespacePrefix : str , optional
            The input namespace prefix value. Default is 'inst'.

        Returns
        -------
        List[Tuple[str, str, str]]
            The resulting ontology triples list.
        """
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
                srcIndex = e.get("src")
                dstIndex = e.get("dst")
                d = dict(e.get("dictionary", {}))
                subject = TGraph._OntologySubjectFromDictionary(d, f"edge_{idx}", namespacePrefix=namespacePrefix)
                triples.append((graph_subject, "top:hasRelationship", subject))
                triples.extend(_dictionary_triples(subject, d, default_class="top:Relationship"))
                sv = vertex_subjects.get(srcIndex, namespacePrefix + ":" + TGraph._OntologySafeLocalName(f"vertex_{srcIndex}"))
                tv = vertex_subjects.get(dstIndex, namespacePrefix + ":" + TGraph._OntologySafeLocalName(f"vertex_{dstIndex}"))
                triples.append((subject, "top:hasStartVertex", sv))
                triples.append((subject, "top:hasEndVertex", tv))
                triples.append((sv, "top:connectsTo", tv))
                if not e.get("directed", graph._directed):
                    triples.append((tv, "top:connectsTo", sv))
        return triples

    @staticmethod
    def OntologyURI(graph: "TGraph", element: str = "graph", index: Optional[int] = None, defaultValue: Any = None) -> Any:
        """
        Returns the ontology URI of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.
        defaultValue : Any , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        Any
            The resulting ontology uri object or value.
        """
        return TGraph._OntologyGet(graph, "uri", defaultValue=defaultValue, element=element, index=index)

    @staticmethod
    def Order(graph: "TGraph") -> int:
        """
        Returns the number of active vertices in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        int
            The number of active vertices.
        """
        if not isinstance(graph, TGraph):
            return 0
        return sum(1 for v in graph._vertices if v.get("active", True))

    @staticmethod
    def OutgoingEdges(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns the outgoing edges of the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting outgoing edges list.
        """
        return TGraph.IncidentEdges(graph, TGraph._as_index(vertex), mode="out")

    @staticmethod
    def OutgoingVertices(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns the outgoing adjacent vertices of the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting outgoing vertices list.
        """
        return TGraph.AdjacentVertices(graph, vertex, mode="out")

    @staticmethod
    def _P6ActiveEdgeRecords(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Internal helper that returns p6 active edge records data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting p6 active edge records list.
        """
        return [e for e in graph._edges if e.get("active", True)] if isinstance(graph, TGraph) else []

    @staticmethod
    def _P6ActiveVertexIndices(graph: "TGraph") -> List[int]:
        """
        Internal helper that returns p6 active vertex indices data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[int]
            The resulting p6 active vertex indices list.
        """
        return [v.get("index") for v in graph._vertices if v.get("active", True)] if isinstance(graph, TGraph) else []

    @staticmethod
    def _P6CandidateMap(pattern: "TGraph", superGraph: "TGraph", vertexKeys: Any = None,
                        tolerance: float = 0.0) -> Dict[int, List[int]]:
        """
        Internal helper that returns p6 candidate map data.

        Parameters
        ----------
        pattern : 'TGraph'
            The input pattern TGraph.
        superGraph : 'TGraph'
            The input super graph value.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.

        Returns
        -------
        Dict[int, List[int]]
            The resulting p6 candidate map list.
        """
        p_vertices = TGraph._P6ActiveVertexIndices(pattern)
        s_vertices = TGraph._P6ActiveVertexIndices(superGraph)
        candidate_map: Dict[int, List[int]] = {}
        for pv in p_vertices:
            candidates = []
            p_degree = TGraph.Degree(pattern, pv, mode="all")
            for sv in s_vertices:
                if TGraph.Degree(superGraph, sv, mode="all") < p_degree:
                    continue
                if TGraph._P6VertexCompatible(pattern, pv, superGraph, sv, vertexKeys=vertexKeys, tolerance=tolerance):
                    candidates.append(sv)
            candidate_map[pv] = candidates
        return candidate_map

    @staticmethod
    def _P6DictionaryMatch(dictA: Dict[str, Any], dictB: Dict[str, Any], keys: Any = None, tolerance: float = 0.0) -> bool:
        """
        Internal helper that returns p6 dictionary match data.

        Parameters
        ----------
        dictA : Dict[str, Any]
            The input dict a value.
        dictB : Dict[str, Any]
            The input dict b value.
        keys : Any , optional
            The input keys value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if keys is None:
            return True
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, (list, tuple)):
            return True
        dictA = dictA if isinstance(dictA, dict) else {}
        dictB = dictB if isinstance(dictB, dict) else {}
        threshold = 1.0 - float(tolerance or 0.0)
        for key in keys:
            if key not in dictA or key not in dictB:
                return False
            if TGraph._P6ValueSimilarity(dictA.get(key), dictB.get(key), tolerance=tolerance) < threshold:
                return False
        return True

    @staticmethod
    def _P6EdgesBetween(graph: "TGraph", src: int, dst: int, directed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Internal helper that returns p6 edges between data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        src : int
            The source vertex index.
        dst : int
            The destination vertex index.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting p6 edges between list.
        """
        if not isinstance(graph, TGraph):
            return []
        try:
            edges = TGraph.EdgesBetween(graph, src, dst, directed=directed)
            return [e for e in edges if isinstance(e, dict) and e.get("active", True)]
        except Exception:
            result = []
            for e in TGraph._P6ActiveEdgeRecords(graph):
                if directed is True or e.get("directed", graph._directed):
                    if e.get("src") == src and e.get("dst") == dst:
                        result.append(e)
                else:
                    a, b = e.get("src"), e.get("dst")
                    if (a == src and b == dst) or (a == dst and b == src):
                        result.append(e)
            return result

    @staticmethod
    def _P6EdgesCompatible(graphA: "TGraph", edgeA: Dict[str, Any], graphB: "TGraph", edgeB: Dict[str, Any],
                           edgeKeys: Any = None, tolerance: float = 0.0) -> bool:
        """
        Internal helper that returns p6 edges compatible data.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        edgeA : Dict[str, Any]
            The input edge a value.
        graphB : 'TGraph'
            The second input TGraph.
        edgeB : Dict[str, Any]
            The input edge b value.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not isinstance(edgeA, dict) or not isinstance(edgeB, dict):
            return False
        dA = edgeA.get("dictionary", {})
        dB = edgeB.get("dictionary", {})
        return TGraph._P6DictionaryMatch(dA, dB, keys=edgeKeys, tolerance=tolerance)

    @staticmethod
    def _P6FeatureValue(graph: "TGraph", vertex: int, key: str = None) -> Any:
        """
        Internal helper that returns p6 feature value data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : int
            The input vertex, vertex index, or vertex record.
        key : str , optional
            The dictionary key to use. Default is None.

        Returns
        -------
        Any
            The resulting p6 feature value object or value.
        """
        d = graph._vertices[vertex].get("dictionary", {})
        if key is not None and d.get(key, None) is not None:
            return d.get(key)
        return TGraph.Degree(graph, vertex, mode="all")

    @staticmethod
    def _P6MappingScore(pattern: "TGraph", superGraph: "TGraph", mapping: Dict[int, int],
                        vertexKeys: Any = None, edgeKeys: Any = None, tolerance: float = 0.0) -> float:
        """
        Internal helper that returns p6 mapping score data.

        Parameters
        ----------
        pattern : 'TGraph'
            The input pattern TGraph.
        superGraph : 'TGraph'
            The input super graph value.
        mapping : Dict[int, int]
            The input mapping value.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.

        Returns
        -------
        float
            The resulting p6 mapping score value.
        """
        scores = []
        if isinstance(vertexKeys, str):
            vertexKeys = [vertexKeys]
        if isinstance(edgeKeys, str):
            edgeKeys = [edgeKeys]
        if vertexKeys:
            for pv, sv in mapping.items():
                dA = pattern._vertices[pv].get("dictionary", {})
                dB = superGraph._vertices[sv].get("dictionary", {})
                for k in vertexKeys:
                    if k in dA and k in dB:
                        scores.append(TGraph._P6ValueSimilarity(dA[k], dB[k], tolerance=tolerance))
        if edgeKeys:
            for pe in TGraph._P6ActiveEdgeRecords(pattern):
                ps, pd = pe.get("src"), pe.get("dst")
                if ps not in mapping or pd not in mapping:
                    continue
                candidates = TGraph._P6EdgesBetween(superGraph, mapping[ps], mapping[pd], directed=True if pe.get("directed", pattern._directed) else None)
                if not candidates:
                    continue
                best = 0.0
                for se in candidates:
                    vals = []
                    dA = pe.get("dictionary", {})
                    dB = se.get("dictionary", {})
                    for k in edgeKeys:
                        if k in dA and k in dB:
                            vals.append(TGraph._P6ValueSimilarity(dA[k], dB[k], tolerance=tolerance))
                    if vals:
                        best = max(best, sum(vals) / float(len(vals)))
                if best > 0:
                    scores.append(best)
        return sum(scores) / float(len(scores)) if scores else 1.0

    @staticmethod
    def _P6RequiredEdgesSatisfied(pattern: "TGraph", superGraph: "TGraph", mapping: Dict[int, int],
                                  edgeKeys: Any = None, tolerance: float = 0.0,
                                  strictPath: bool = True) -> bool:
        """
        Internal helper that returns p6 required edges satisfied data.

        Parameters
        ----------
        pattern : 'TGraph'
            The input pattern TGraph.
        superGraph : 'TGraph'
            The input super graph value.
        mapping : Dict[int, int]
            The input mapping value.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.
        strictPath : bool , optional
            The file or folder path to use. Default is True.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        for pe in TGraph._P6ActiveEdgeRecords(pattern):
            ps = pe.get("src")
            pd = pe.get("dst")
            if ps not in mapping or pd not in mapping:
                return False
            ss = mapping[ps]
            sd = mapping[pd]
            pdir = bool(pe.get("directed", pattern._directed))
            candidates = TGraph._P6EdgesBetween(superGraph, ss, sd, directed=True if pdir else None)
            if candidates:
                if any(TGraph._P6EdgesCompatible(pattern, pe, superGraph, se, edgeKeys=edgeKeys, tolerance=tolerance) for se in candidates):
                    continue
                return False
            if not strictPath:
                path = TGraph.ShortestPath(superGraph, ss, sd, mode="out" if pdir else "all")
                if path and len(path) >= 2:
                    continue
            return False
        return True

    @staticmethod
    def _P6ShortestPathShells(graph: "TGraph", maxHops: int = 3) -> Dict[int, Dict[int, List[int]]]:
        """
        Internal helper that returns p6 shortest path shells data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        maxHops : int , optional
            The input max hops value. Default is 3.

        Returns
        -------
        Dict[int, Dict[int, List[int]]]
            The resulting p6 shortest path shells list.
        """
        from collections import deque
        shells = {}
        vertices = TGraph._P6ActiveVertexIndices(graph)
        for source in vertices:
            dist = {source: 0}
            q = deque([source])
            while q:
                u = q.popleft()
                if dist[u] >= maxHops:
                    continue
                for v in TGraph.AdjacentIndices(graph, u, mode="all"):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            by_hop = {h: [] for h in range(maxHops + 1)}
            for v, d in dist.items():
                if d <= maxHops:
                    by_hop.setdefault(d, []).append(v)
            shells[source] = by_hop
        return shells

    @staticmethod
    def _P6SubgraphIsomorphisms(pattern: "TGraph", superGraph: "TGraph", vertexKeys: Any = None,
                                edgeKeys: Any = None, maxMatches: int = 10, timeLimit: int = 10,
                                tolerance: float = 0.0, strictPath: bool = True) -> List[Dict[int, int]]:
        """
        Internal helper that returns p6 subgraph isomorphisms data.

        Parameters
        ----------
        pattern : 'TGraph'
            The input pattern TGraph.
        superGraph : 'TGraph'
            The input super graph value.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.
        maxMatches : int , optional
            The input max matches value. Default is 10.
        timeLimit : int , optional
            The maximum time, in seconds, allowed for the search. Default is 10.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.
        strictPath : bool , optional
            The file or folder path to use. Default is True.

        Returns
        -------
        List[Dict[int, int]]
            The resulting p6 subgraph isomorphisms list.
        """
        import time as _time
        if not isinstance(pattern, TGraph) or not isinstance(superGraph, TGraph):
            return []
        if TGraph.Order(pattern) > TGraph.Order(superGraph):
            return []
        if strictPath and TGraph.Size(pattern) > TGraph.Size(superGraph):
            return []

        start = _time.time()
        p_vertices = TGraph._P6ActiveVertexIndices(pattern)
        candidate_map = TGraph._P6CandidateMap(pattern, superGraph, vertexKeys=vertexKeys, tolerance=tolerance)
        if any(len(candidate_map.get(pv, [])) == 0 for pv in p_vertices):
            return []

        # Search most constrained pattern vertices first.
        p_order = sorted(p_vertices, key=lambda v: (len(candidate_map.get(v, [])), -TGraph.Degree(pattern, v, mode="all")))
        maxMatches = max(1, int(maxMatches or 1))
        timeLimit = max(0.001, float(timeLimit or 10))
        matches: List[Dict[int, int]] = []
        mapping: Dict[int, int] = {}
        used_super: Set[int] = set()

        def partial_feasible(pv: int, sv: int) -> bool:
            # Check all already mapped edges incident to pv immediately.
            for pe in TGraph._P6ActiveEdgeRecords(pattern):
                ps = pe.get("src")
                pd = pe.get("dst")
                other = None
                forward = True
                if ps == pv and pd in mapping:
                    other = pd
                    forward = True
                elif pd == pv and ps in mapping:
                    other = ps
                    forward = False
                else:
                    continue
                ss = sv if forward else mapping[other]
                sd = mapping[other] if forward else sv
                pdir = bool(pe.get("directed", pattern._directed))
                candidates = TGraph._P6EdgesBetween(superGraph, ss, sd, directed=True if pdir else None)
                if candidates:
                    if any(TGraph._P6EdgesCompatible(pattern, pe, superGraph, se, edgeKeys=edgeKeys, tolerance=tolerance) for se in candidates):
                        continue
                    return False
                if not strictPath:
                    path = TGraph.ShortestPath(superGraph, ss, sd, mode="out" if pdir else "all")
                    if path and len(path) >= 2:
                        continue
                return False
            return True

        def backtrack(depth: int) -> None:
            if len(matches) >= maxMatches:
                return
            if (_time.time() - start) >= timeLimit:
                return
            if depth >= len(p_order):
                if TGraph._P6RequiredEdgesSatisfied(pattern, superGraph, mapping, edgeKeys=edgeKeys,
                                                     tolerance=tolerance, strictPath=strictPath):
                    matches.append(dict(mapping))
                return
            pv = p_order[depth]
            for sv in candidate_map[pv]:
                if sv in used_super:
                    continue
                if not partial_feasible(pv, sv):
                    continue
                mapping[pv] = sv
                used_super.add(sv)
                backtrack(depth + 1)
                used_super.remove(sv)
                del mapping[pv]
                if len(matches) >= maxMatches:
                    return
                if (_time.time() - start) >= timeLimit:
                    return

        backtrack(0)
        return matches

    @staticmethod
    def _P6ValueSimilarity(valueA: Any, valueB: Any, tolerance: float = 0.0) -> float:
        """
        Internal helper that returns p6 value similarity data.

        Parameters
        ----------
        valueA : Any
            The input value a value.
        valueB : Any
            The input value b value.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.

        Returns
        -------
        float
            The resulting p6 value similarity value.
        """
        from difflib import SequenceMatcher
        try:
            a = float(valueA)
            b = float(valueB)
            if abs(a) <= 1e-12:
                return 1.0 if abs(b) <= float(tolerance or 0.0) else 0.0
            diff = abs(a - b) / abs(a)
            return max(0.0, 1.0 - diff)
        except Exception:
            return SequenceMatcher(None, str(valueA).lower(), str(valueB).lower()).ratio()

    @staticmethod
    def _P6VertexCompatible(graphA: "TGraph", vertexA: int, graphB: "TGraph", vertexB: int,
                            vertexKeys: Any = None, tolerance: float = 0.0) -> bool:
        """
        Internal helper that returns p6 vertex compatible data.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        vertexA : int
            The first input vertex or vertex index.
        graphB : 'TGraph'
            The second input TGraph.
        vertexB : int
            The second input vertex or vertex index.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not graphA._validate_vertex_index(vertexA) or not graphB._validate_vertex_index(vertexB):
            return False
        dA = graphA._vertices[vertexA].get("dictionary", {})
        dB = graphB._vertices[vertexB].get("dictionary", {})
        return TGraph._P6DictionaryMatch(dA, dB, keys=vertexKeys, tolerance=tolerance)

    @staticmethod
    def _P6VertexID(graph: "TGraph", vertex: Any, vertexIDKey: str = "id", mantissa: int = 6) -> Any:
        """
        Internal helper that returns p6 vertex id data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Any
            The input vertex, vertex index, or vertex record.
        vertexIDKey : str , optional
            The dictionary key to use. Default is 'id'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.

        Returns
        -------
        Any
            The resulting p6 vertex id object or value.
        """
        idx = TGraph.VertexIndex(graph, vertex)
        if idx is None:
            idx = TGraph._as_index(vertex)
        if idx is None or not graph._validate_vertex_index(idx):
            return None
        d = graph._vertices[idx].get("dictionary", {})
        value = d.get(vertexIDKey, None)
        if value is not None:
            return value
        coords = TGraph.Coordinates(graph, idx, default=None)
        if coords is not None:
            return str([round(float(x), mantissa) for x in coords])
        return idx

    @staticmethod
    def _P7ActiveEdgeRecords(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Internal helper that returns p7 active edge records data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting p7 active edge records list.
        """
        return [e for e in graph._edges if e.get("active", True)] if isinstance(graph, TGraph) else []

    @staticmethod
    def _P7ActiveVertexIndices(graph: "TGraph") -> List[int]:
        """
        Internal helper that returns p7 active vertex indices data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[int]
            The resulting p7 active vertex indices list.
        """
        return [i for i, v in enumerate(graph._vertices) if v.get("active", True)] if isinstance(graph, TGraph) else []

    @staticmethod
    def _P7Cosine(featuresA: Dict[Any, float], featuresB: Dict[Any, float], normalize: bool = True) -> float:
        """
        Internal helper that returns p7 cosine data.

        Parameters
        ----------
        featuresA : Dict[Any, float]
            The input features a value.
        featuresB : Dict[Any, float]
            The input features b value.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.

        Returns
        -------
        float
            The resulting p7 cosine value.
        """
        keys = set(featuresA.keys()) | set(featuresB.keys())
        dot = sum(float(featuresA.get(k, 0.0)) * float(featuresB.get(k, 0.0)) for k in keys)
        if not normalize:
            return float(dot)
        normA = math.sqrt(sum(float(v) * float(v) for v in featuresA.values()))
        normB = math.sqrt(sum(float(v) * float(v) for v in featuresB.values()))
        return float(dot) / (normA * normB) if normA > 0 and normB > 0 else 0.0

    @staticmethod
    def _P7GraphEdgeWeights(graph: "TGraph", edgeWeightKey: str = None) -> Dict[Tuple[Any, Any], float]:
        """
        Internal helper that returns p7 graph edge weights data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is None.

        Returns
        -------
        Dict[Tuple[Any, Any], float]
            The resulting p7 graph edge weights dictionary.
        """
        weights = {}
        for e in TGraph._P7ActiveEdgeRecords(graph):
            srcIndex = e.get("src")
            dstIndex = e.get("dst")
            if srcIndex is None or dstIndex is None:
                continue
            a, b = (srcIndex, dstIndex) if srcIndex <= dstIndex else (dstIndex, srcIndex)
            if edgeWeightKey is None:
                weight = 1.0
            else:
                try:
                    weight = float(e.get("dictionary", {}).get(edgeWeightKey, 1.0))
                except Exception:
                    weight = 1.0
            weights[(a, b)] = weights.get((a, b), 0.0) + float(weight)
        return weights

    @staticmethod
    def _P7HopFeatures(graph: "TGraph", key: str = None, maxHops: int = 2,
                       decay: float = 1.0) -> Dict[Any, float]:
        """
        Internal helper that returns p7 hop features data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str , optional
            The dictionary key to use. Default is None.
        maxHops : int , optional
            The input max hops value. Default is 2.
        decay : float , optional
            The input decay value. Default is 1.0.

        Returns
        -------
        Dict[Any, float]
            The resulting p7 hop features dictionary.
        """
        features = {}
        vertices = TGraph._P7ActiveVertexIndices(graph)
        labels = {v: TGraph._P7Label(graph, v, key=key, defaultToDegree=True) for v in vertices}
        nbrs = {v: TGraph._P7Neighbors(graph, v, mode="all") for v in vertices}
        maxHops = max(0, int(maxHops or 0))
        decay = float(decay if decay is not None else 1.0)
        from collections import deque as _deque
        for source in vertices:
            source_label = labels[source]
            visited = {source}
            queue = _deque([(source, 0)])
            while queue:
                vertex, depth = queue.popleft()
                if depth > maxHops:
                    continue
                feature_key = (depth, source_label, labels.get(vertex, "0"))
                features[feature_key] = features.get(feature_key, 0.0) + (decay ** depth)
                if depth == maxHops:
                    continue
                for nbr in nbrs.get(vertex, []):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, depth + 1))
        return features

    @staticmethod
    def _P7Label(graph: "TGraph", vertex: int, key: str = None, defaultToDegree: bool = True) -> str:
        """
        Internal helper that returns p7 label data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : int
            The input vertex, vertex index, or vertex record.
        key : str , optional
            The dictionary key to use. Default is None.
        defaultToDegree : bool , optional
            The input default to degree value. Default is True.

        Returns
        -------
        str
            The resulting p7 label string.
        """
        try:
            d = graph._vertices[vertex].get("dictionary", {})
            if key is not None and isinstance(d, dict) and key in d:
                return str(d.get(key))
            if defaultToDegree:
                return str(len(TGraph._P7Neighbors(graph, vertex, mode="all")))
        except Exception:
            pass
        return "0"

    @staticmethod
    def _P7Neighbors(graph: "TGraph", vertex: int, mode: str = "all") -> List[int]:
        """
        Internal helper that returns p7 neighbors data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : int
            The input vertex, vertex index, or vertex record.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.

        Returns
        -------
        List[int]
            The resulting p7 neighbors list.
        """
        if not isinstance(graph, TGraph) or not graph._validate_vertex_index(vertex):
            return []
        mode = str(mode or "all").lower()
        result = []
        seen = set()
        if mode in ("out", "all"):
            for eid in graph._out_edges.get(vertex, set()):
                if not graph._validate_edge_index(eid):
                    continue
                e = graph._edges[eid]
                if e.get("src") == vertex:
                    nbr = e.get("dst")
                else:
                    nbr = e.get("src")
                if graph._validate_vertex_index(nbr) and nbr not in seen:
                    seen.add(nbr)
                    result.append(nbr)
        if mode in ("in", "all"):
            for eid in graph._in_edges.get(vertex, set()):
                if not graph._validate_edge_index(eid):
                    continue
                e = graph._edges[eid]
                if e.get("dst") == vertex:
                    nbr = e.get("src")
                else:
                    nbr = e.get("dst")
                if graph._validate_vertex_index(nbr) and nbr not in seen:
                    seen.add(nbr)
                    result.append(nbr)
        return result

    @staticmethod
    def _P81GraphEdgeWeights(graph: "TGraph", edgeWeightKey: str = None) -> Dict[Tuple[Any, Any], float]:
        """
        Internal helper that returns p81 graph edge weights data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is None.

        Returns
        -------
        Dict[Tuple[Any, Any], float]
            The resulting p81 graph edge weights dictionary.
        """
        weights = {}
        for e in graph._edges:
            if not e.get("active", True):
                continue
            srcIndex = e.get("src")
            dstIndex = e.get("dst")
            if srcIndex is None or dstIndex is None:
                continue
            a, b = (srcIndex, dstIndex) if srcIndex <= dstIndex else (dstIndex, srcIndex)
            if edgeWeightKey is None:
                weight = 1.0
            else:
                try:
                    weight = float(e.get("dictionary", {}).get(edgeWeightKey, 1.0))
                except Exception:
                    weight = 1.0
            weights[(a, b)] = weights.get((a, b), 0.0) + float(weight)
        return weights

    @staticmethod
    def _P81HopFeatures(graph: "TGraph", key: str = None, maxHops: int = 2,
                        decay: float = 1.0) -> Dict[Any, float]:
        """
        Internal helper that returns p81 hop features data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str , optional
            The dictionary key to use. Default is None.
        maxHops : int , optional
            The input max hops value. Default is 2.
        decay : float , optional
            The input decay value. Default is 1.0.

        Returns
        -------
        Dict[Any, float]
            The resulting p81 hop features dictionary.
        """
        from collections import deque as _deque

        vertices = [i for i, v in enumerate(graph._vertices) if v.get("active", True)]
        vertex_set = set(vertices)
        labels = {}
        adj = {v: [] for v in vertices}

        for e in graph._edges:
            if not e.get("active", True):
                continue
            srcIndex = e.get("src")
            dstIndex = e.get("dst")
            if srcIndex not in vertex_set or dstIndex not in vertex_set:
                continue
            adj[srcIndex].append(dstIndex)
            adj[dstIndex].append(srcIndex)

        for v in vertices:
            d = graph._vertices[v].get("dictionary", {})
            if key is not None and isinstance(d, dict) and key in d:
                labels[v] = str(d.get(key))
            else:
                labels[v] = str(len(adj[v]))

        features = {}
        maxHops = max(0, int(maxHops or 0))
        decay = float(decay if decay is not None else 1.0)

        for source in vertices:
            source_label = labels[source]
            visited = {source}
            queue = _deque([(source, 0)])
            while queue:
                vertex, depth = queue.popleft()
                if depth > maxHops:
                    continue
                feature_key = (depth, source_label, labels.get(vertex, "0"))
                features[feature_key] = features.get(feature_key, 0.0) + (decay ** depth)
                if depth == maxHops:
                    continue
                for nbr in adj.get(vertex, []):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, depth + 1))
        return features

    @staticmethod
    def _P8ActiveEdgeRecords(graph: "TGraph") -> List[Dict[str, Any]]:
        """
        Internal helper that returns p8 active edge records data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[Dict[str, Any]]
            The resulting p8 active edge records list.
        """
        return [e for e in graph._edges if e.get("active", True)] if isinstance(graph, TGraph) else []

    @staticmethod
    def _P8ActiveVertexIndices(graph: "TGraph") -> List[int]:
        """
        Internal helper that returns p8 active vertex indices data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        List[int]
            The resulting p8 active vertex indices list.
        """
        return [i for i, v in enumerate(graph._vertices) if v.get("active", True)] if isinstance(graph, TGraph) else []

    @staticmethod
    def _P8Cosine(featuresA: Dict[Any, float], featuresB: Dict[Any, float], normalize: bool = True) -> float:
        """
        Internal helper that returns p8 cosine data.

        Parameters
        ----------
        featuresA : Dict[Any, float]
            The input features a value.
        featuresB : Dict[Any, float]
            The input features b value.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.

        Returns
        -------
        float
            The resulting p8 cosine value.
        """
        if not featuresA and not featuresB:
            return 1.0 if normalize else 0.0
        if len(featuresA) > len(featuresB):
            featuresA, featuresB = featuresB, featuresA
        dot = 0.0
        for k, v in featuresA.items():
            dot += float(v) * float(featuresB.get(k, 0.0))
        if not normalize:
            return float(dot)
        normA = math.sqrt(sum(float(v) * float(v) for v in featuresA.values()))
        normB = math.sqrt(sum(float(v) * float(v) for v in featuresB.values()))
        return float(dot) / (normA * normB) if normA > 0 and normB > 0 else 0.0

    @staticmethod
    def _P8EdgeCompatible(data: Dict[str, Any], superA: int, superB: int, patternA: int, patternB: int, edgeKeys: Any = None) -> bool:
        """
        Internal helper that returns p8 edge compatible data.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data dictionary.
        superA : int
            The input super a value.
        superB : int
            The input super b value.
        patternA : int
            The input pattern a value.
        patternB : int
            The input pattern b value.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if isinstance(edgeKeys, str):
            edgeKeys = [edgeKeys]
        edgeKeys = edgeKeys or []

        skey = (superA, superB) if superA <= superB else (superB, superA)
        pkey = (patternA, patternB) if patternA <= patternB else (patternB, patternA)

        super_edges = data["super"].get("undirected_edge_map", {}).get(skey, [])
        pattern_edges = data["pattern"].get("undirected_edge_map", {}).get(pkey, [])

        if not super_edges or not pattern_edges:
            return False

        if not edgeKeys:
            return True

        for ped in pattern_edges:
            if any(k not in ped for k in edgeKeys):
                continue
            pvals = tuple(ped.get(k, None) for k in edgeKeys)
            for sed in super_edges:
                if any(k not in sed for k in edgeKeys):
                    continue
                if tuple(sed.get(k, None) for k in edgeKeys) == pvals:
                    return True
        return False

    @staticmethod
    def _P8GraphArrays(graph: "TGraph", vertexKeys: Any = None, edgeKeys: Any = None) -> Optional[Dict[str, Any]]:
        """
        Internal helper that returns p8 graph arrays data.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.

        Returns
        -------
        Optional[Dict[str, Any]]
            The resulting p8 graph arrays dictionary.
        """
        if not isinstance(graph, TGraph):
            return None
        if isinstance(vertexKeys, str):
            vertexKeys = [vertexKeys]
        if isinstance(edgeKeys, str):
            edgeKeys = [edgeKeys]
        vertexKeys = vertexKeys or []
        edgeKeys = edgeKeys or []

        vertices = [i for i, v in enumerate(graph._vertices) if v.get("active", True)]
        vertex_set = set(vertices)
        vlabels = {}
        for v in vertices:
            d = graph._vertices[v].get("dictionary", {})
            if not isinstance(d, dict):
                d = {}
            vlabels[v] = tuple(d.get(k, None) for k in vertexKeys) if vertexKeys else None

        adj = {v: set() for v in vertices}
        edge_map = {}
        undirected_edge_map = {}

        for e in graph._edges:
            if not e.get("active", True):
                continue
            srcIndex = e.get("src")
            dstIndex = e.get("dst")
            if srcIndex not in vertex_set or dstIndex not in vertex_set:
                continue
            directed = bool(e.get("directed", graph._directed))
            ed = e.get("dictionary", {})
            if not isinstance(ed, dict):
                ed = {}

            adj[srcIndex].add(dstIndex)
            adj[dstIndex].add(srcIndex)

            key = (srcIndex, dstIndex, directed)
            edge_map.setdefault(key, []).append(ed)
            if not directed:
                edge_map.setdefault((dstIndex, srcIndex, directed), []).append(ed)

            ukey = (srcIndex, dstIndex) if srcIndex <= dstIndex else (dstIndex, srcIndex)
            undirected_edge_map.setdefault(ukey, []).append(ed)

        degrees = {v: len(adj[v]) for v in vertices}
        return {
            "vertices": vertices,
            "vertex_set": vertex_set,
            "vlabels": vlabels,
            "adj": adj,
            "edge_map": edge_map,
            "undirected_edge_map": undirected_edge_map,
            "degrees": degrees,
            "directed": bool(graph._directed),
            "vertexKeys": vertexKeys,
            "edgeKeys": edgeKeys,
        }

    @staticmethod
    def _P8SubgraphIsomorphisms(pattern: "TGraph", superGraph: "TGraph",
                                vertexKeys: Any = None, edgeKeys: Any = None,
                                maxMatches: int = 25, timeLimit: float = 5,
                                exact: bool = False) -> List[Dict[int, int]]:
        """
        Internal helper that returns p8 subgraph isomorphisms data.

        Parameters
        ----------
        pattern : 'TGraph'
            The input pattern TGraph.
        superGraph : 'TGraph'
            The input super graph value.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.
        maxMatches : int , optional
            The input max matches value. Default is 25.
        timeLimit : float , optional
            The maximum time, in seconds, allowed for the search. Default is 5.
        exact : bool , optional
            The input exact value. Default is False.

        Returns
        -------
        List[Dict[int, int]]
            The resulting p8 subgraph isomorphisms list.
        """
        import time as _time

        if not isinstance(pattern, TGraph) or not isinstance(superGraph, TGraph):
            return []

        if exact:
            if TGraph.Order(pattern) != TGraph.Order(superGraph) or TGraph.Size(pattern) != TGraph.Size(superGraph):
                return []

        if TGraph.Order(pattern) == 0:
            return [{}]

        if TGraph.Order(pattern) > TGraph.Order(superGraph):
            return []

        if exact is True and TGraph.Size(pattern) > TGraph.Size(superGraph):
            return []

        if isinstance(vertexKeys, str):
            vertexKeys = [vertexKeys]
        if isinstance(edgeKeys, str):
            edgeKeys = [edgeKeys]
        vertexKeys = vertexKeys or []
        edgeKeys = edgeKeys or []

        P = TGraph._P8GraphArrays(pattern, vertexKeys=vertexKeys, edgeKeys=edgeKeys)
        S = TGraph._P8GraphArrays(superGraph, vertexKeys=vertexKeys, edgeKeys=edgeKeys)
        if P is None or S is None:
            return []

        data = {"pattern": P, "super": S}

        base_candidates = {}
        for pv in P["vertices"]:
            plabel = P["vlabels"].get(pv, None)
            pdeg = P["degrees"].get(pv, 0)
            candidates = []
            for sv in S["vertices"]:
                if vertexKeys:
                    slabel = S["vlabels"].get(sv, None)
                    if plabel is None or slabel is None:
                        continue
                    if any(value is None for value in plabel) or any(value is None for value in slabel):
                        continue
                    if slabel != plabel:
                        continue
                if S["degrees"].get(sv, 0) < pdeg:
                    continue
                candidates.append(sv)
            if not candidates:
                return []
            base_candidates[pv] = set(candidates)

        order = sorted(P["vertices"], key=lambda v: (len(base_candidates[v]), -P["degrees"].get(v, 0), v))

        maxMatches = max(1, int(maxMatches or 1))
        timeLimit = float(timeLimit if timeLimit is not None else 5.0)
        start_time = _time.perf_counter()

        matches: List[Dict[int, int]] = []
        mapping: Dict[int, int] = {}
        used_super: Set[int] = set()

        def compatible(pv, sv):
            for pn in P["adj"].get(pv, set()):
                if pn not in mapping:
                    continue
                sn = mapping[pn]
                if sn not in S["adj"].get(sv, set()):
                    return False
                if not TGraph._P8EdgeCompatible(data, sv, sn, pv, pn, edgeKeys=edgeKeys):
                    return False

            unmapped_pattern_neighbours = [pn for pn in P["adj"].get(pv, set()) if pn not in mapping]
            if unmapped_pattern_neighbours:
                available_super_neighbours = S["adj"].get(sv, set()) - used_super
                if len(available_super_neighbours) < len(unmapped_pattern_neighbours):
                    return False
            return True

        def recurse(depth):
            if len(matches) >= maxMatches:
                return
            if _time.perf_counter() - start_time > timeLimit:
                return
            if depth >= len(order):
                matches.append(dict(mapping))
                return

            pv = order[depth]
            mapped_neighbours = [pn for pn in P["adj"].get(pv, set()) if pn in mapping]
            if mapped_neighbours:
                dynamic = None
                for pn in mapped_neighbours:
                    sn = mapping[pn]
                    neighbours = S["adj"].get(sn, set())
                    dynamic = set(neighbours) if dynamic is None else dynamic & neighbours
                candidates = base_candidates[pv] & (dynamic if dynamic is not None else set())
            else:
                candidates = base_candidates[pv]

            for sv in sorted(candidates, key=lambda x: (-S["degrees"].get(x, 0), x)):
                if sv in used_super:
                    continue
                if not compatible(pv, sv):
                    continue
                mapping[pv] = sv
                used_super.add(sv)
                recurse(depth + 1)
                used_super.remove(sv)
                del mapping[pv]
                if len(matches) >= maxMatches:
                    return
                if _time.perf_counter() - start_time > timeLimit:
                    return

        recurse(0)
        return matches

    @staticmethod
    def PageRank(graph: "TGraph", damping: float = 0.85, iterations: int = 100,
                 tolerance: float = 1e-9, key: str = "pagerank", mantissa: int = 6,
                 useNumpy: bool = True) -> List[float]:
        """
        Computes PageRank values for the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        damping : float , optional
            The input damping value. Default is 0.85.
        iterations : int , optional
            The input iterations value. Default is 100.
        tolerance : float , optional
            The desired tolerance. Default is 1e-09.
        key : str , optional
            The dictionary key to use. Default is 'pagerank'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        useNumpy : bool , optional
            If set to True, NumPy acceleration is used when available. Default is True.

        Returns
        -------
        List[float]
            The resulting page rank list.
        """
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

    @staticmethod
    def _PairGroupingStats(pairs: Iterable[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Returns grouping statistics for pairs of dictionary values.

        Parameters
        ----------
        pairs : Iterable[Tuple[int, int]]
            The input pairs value.

        Returns
        -------
        Dict[str, Any]
            The resulting pair grouping stats dictionary.
        """
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
    def Path(graph: "TGraph", vertexA, vertexB, tolerance: float = 0.0001, silent: bool = False) -> List[int]:
        """
        Returns a path between two vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexA : Any
            The first input vertex or vertex index.
        vertexB : Any
            The second input vertex or vertex index.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[int]
            The resulting path list.
        """
        return TGraph.ShortestPath(graph, TGraph._as_index(vertexA), TGraph._as_index(vertexB), mode="out" if TGraph.IsDirected(graph) else "all")

    @staticmethod
    def PlotlyData(
        graph: "TGraph",
        vertexLabelKey: str = "label",
        vertexColorKey: str = "color",
        vertexSizeKey: str = "size",
        edgeColorKey: str = "color",
        edgeWidthKey: str = "width",
        defaultVertexSize: float = 6,
        defaultEdgeWidth: float = 2,
        showVertexLabels: bool = False,
        showEdgeLabels: bool = False,
        edgeLabelKey: str = "label",
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> List[Any]:
        """
        Returns Plotly trace data representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        vertexColorKey : str , optional
            The dictionary key to use. Default is 'color'.
        vertexSizeKey : str , optional
            The dictionary key to use. Default is 'size'.
        edgeColorKey : str , optional
            The dictionary key to use. Default is 'color'.
        edgeWidthKey : str , optional
            The dictionary key to use. Default is 'width'.
        defaultVertexSize : float , optional
            The input default vertex size value. Default is 6.
        defaultEdgeWidth : float , optional
            The input default edge width value. Default is 2.
        showVertexLabels : bool , optional
            If set to True, show vertex labels are shown. Default is False.
        showEdgeLabels : bool , optional
            If set to True, show edge labels are shown. Default is False.
        edgeLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[Any]
            The resulting plotly data list.
        """
        if not isinstance(graph, TGraph):
            return []
        try:
            import plotly.graph_objects as go
        except Exception:
            if not silent:
                print("TGraph.PlotlyData - Error: Plotly is not installed. Returning an empty list.")
            return []

        active = TGraph.ActiveVertexIndices(graph)
        n = max(1, len(active))
        coords: Dict[int, List[float]] = {}
        for pos, idx in enumerate(active):
            c = TGraph.Coordinates(graph, idx)
            if c is None:
                angle = 2.0 * math.pi * float(pos) / float(n)
                c = [math.cos(angle), math.sin(angle), 0.0]
            coords[idx] = [round(float(c[0]), mantissa), round(float(c[1]), mantissa), round(float(c[2]), mantissa)]

        edge_traces = []
        # Group edges by style to avoid one trace per edge where possible.
        grouped: Dict[Tuple[Any, Any], Dict[str, List[Any]]] = {}
        edge_text_grouped: Dict[Tuple[Any, Any], List[str]] = {}
        for e in graph._edges:
            if not e.get("active", True):
                continue
            srcIndex = e.get("src")
            dstIndex = e.get("dst")
            if srcIndex not in coords or dstIndex not in coords:
                continue
            d = e.get("dictionary", {}) if isinstance(e.get("dictionary", {}), dict) else {}
            color = d.get(edgeColorKey, "black")
            width = d.get(edgeWidthKey, defaultEdgeWidth)
            key = (str(color), float(width) if isinstance(width, (int, float)) else defaultEdgeWidth)
            grouped.setdefault(key, {"x": [], "y": [], "z": []})
            edge_text_grouped.setdefault(key, [])
            a = coords[srcIndex]
            b = coords[dstIndex]
            grouped[key]["x"].extend([a[0], b[0], None])
            grouped[key]["y"].extend([a[1], b[1], None])
            grouped[key]["z"].extend([a[2], b[2], None])
            edge_text_grouped[key].append(str(d.get(edgeLabelKey, "")))

        for (color, width), data in grouped.items():
            edge_traces.append(
                go.Scatter3d(
                    x=data["x"],
                    y=data["y"],
                    z=data["z"],
                    mode="lines+text" if showEdgeLabels else "lines",
                    line=dict(color=color, width=width),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        xs, ys, zs, labels, colors, sizes = [], [], [], [], [], []
        for idx in active:
            c = coords[idx]
            d = graph._vertices[idx].get("dictionary", {})
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])
            labels.append(str(d.get(vertexLabelKey, d.get("index", idx))))
            colors.append(d.get(vertexColorKey, "black"))
            sizes.append(d.get(vertexSizeKey, defaultVertexSize))

        vertex_trace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text" if showVertexLabels else "markers",
            marker=dict(size=sizes, color=colors),
            text=labels if showVertexLabels else None,
            hovertext=labels,
            hoverinfo="text",
            showlegend=False,
        )
        return edge_traces + [vertex_trace]

    @staticmethod
    def _PythonToDictionary(data: Optional[Dict[str, Any]]) -> Any:
        """
        Converts a Python dictionary to a Topologic dictionary.

        Parameters
        ----------
        data : Optional[Dict[str, Any]]
            The input data dictionary.

        Returns
        -------
        Any
            The resulting python to dictionary object or value.
        """
        if not isinstance(data, dict) or len(data) == 0:
            return None
        try:
            from topologicpy.Dictionary import Dictionary
            keys = list(data.keys())
            values = [data[k] for k in keys]
            return Dictionary.ByKeysValues(keys, values)
        except Exception:
            return None

    @staticmethod
    def PyvisGraph(
        graph: "TGraph",
        height: str = "750px",
        width: str = "100%",
        directed: Optional[bool] = None,
        vertexLabelKey: str = "label",
        edgeLabelKey: str = "label",
        notebook: bool = False,
        bgcolor: str = "#ffffff",
        fontColor: str = "#000000",
        silent: bool = False,
    ) -> Optional[Any]:
        """
        Returns a PyVis graph representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        height : str , optional
            The figure or output height. Default is '750px'.
        width : str , optional
            The figure or output width. Default is '100%'.
        directed : Optional[bool] , optional
            If set to True, graph edges are treated as directed. Default is None.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        edgeLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        notebook : bool , optional
            The input notebook value. Default is False.
        bgcolor : str , optional
            The input bgcolor value. Default is '#ffffff'.
        fontColor : str , optional
            The color value to use. Default is '#000000'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Any]
            The resulting pyvis graph object or value.
        """
        if not isinstance(graph, TGraph):
            return None
        try:
            from pyvis.network import Network
        except Exception:
            if not silent:
                print("TGraph.PyvisGraph - Error: pyvis is not installed. Returning None.")
            return None
        is_directed = graph._directed if directed is None else bool(directed)
        net = Network(height=height, width=width, directed=is_directed, notebook=notebook, bgcolor=bgcolor, font_color=fontColor)
        for v in graph._vertices:
            if not v.get("active", True):
                continue
            d = v.get("dictionary", {}) if isinstance(v.get("dictionary", {}), dict) else {}
            idx = v.get("index")
            label = d.get(vertexLabelKey, idx)
            title = "<br>".join([f"{k}: {d[k]}" for k in sorted(d.keys())])
            net.add_node(idx, label=str(label), title=title)
        for e in graph._edges:
            if not e.get("active", True):
                continue
            d = e.get("dictionary", {}) if isinstance(e.get("dictionary", {}), dict) else {}
            label = d.get(edgeLabelKey, "")
            title = "<br>".join([f"{k}: {d[k]}" for k in sorted(d.keys())])
            net.add_edge(e.get("src"), e.get("dst"), label=str(label) if label is not None else "", title=title)
        return net


    @staticmethod
    def Quotient(
        topology,
        topologyType: str = "vertex",
        key: str = None,
        groupLabelKey: str = None,
        groupCountKey: str = "count",
        weighted: bool = False,
        edgeWeightKey: str = "weight",
        idKey: str = None,
        transferDictionaries: bool = False,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Constructs the quotient graph induced by grouping TGraph vertices by a
        dictionary value.

        Two groups are connected if any member vertex of one group is adjacent to
        any member vertex of the other group. If weighted is True, edge weights count
        the number of distinct member-level adjacencies across groups.

        Parameters
        ----------
        topology : TGraph
            The input TGraph.
        topologyType : str , optional
            The type of item to quotient. Supported values are "vertex" and "edge".
            If set to "edge", the line graph of the input TGraph is constructed first,
            then quotienting is performed on the line-graph vertices. Default is
            "vertex".
        key : str
            The vertex dictionary key used to form groups.
        groupLabelKey : str , optional
            Vertex dictionary key under which to store the group label. If set to None,
            the input key is used. Default is None.
        groupCountKey : str , optional
            Vertex dictionary key under which to store the group size. Default is
            "count".
        weighted : bool , optional
            If set to True, stores counts of cross-group adjacencies on quotient edges
            under edgeWeightKey. Default is False.
        edgeWeightKey : str , optional
            Edge dictionary key under which to store the weight when weighted is True.
            Default is "weight".
        idKey : str , optional
            Kept for API compatibility with Graph.Quotient. Not required for TGraph,
            because TGraph vertices already have stable integer indices. Default is
            None.
        transferDictionaries : bool , optional
            If set to True, dictionaries of grouped member vertices are merged into the
            representative quotient vertex. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The resulting quotient TGraph, or None if the operation fails.
        """

        if not isinstance(topology, TGraph):
            if not silent:
                print("TGraph.Quotient - Error: The input topology parameter is not a valid TGraph. Returning None.")
            return None

        topologyType = str(topologyType or "vertex").strip().lower()

        if topologyType not in ["vertex", "edge"]:
            if not silent:
                print("TGraph.Quotient - Error: topologyType must be either 'vertex' or 'edge'. Returning None.")
            return None

        if not isinstance(key, str):
            if not silent:
                print("TGraph.Quotient - Error: The input key parameter is not a valid string. Returning None.")
            return None

        if groupLabelKey is None:
            groupLabelKey = key

        if not isinstance(groupLabelKey, str):
            if not silent:
                print("TGraph.Quotient - Error: The input groupLabelKey parameter is not a valid string. Returning None.")
            return None

        if not isinstance(groupCountKey, str):
            if not silent:
                print("TGraph.Quotient - Error: The input groupCountKey parameter is not a valid string. Returning None.")
            return None

        if not isinstance(weighted, bool):
            if not silent:
                print("TGraph.Quotient - Error: The input weighted parameter is not a valid boolean. Returning None.")
            return None

        if not isinstance(edgeWeightKey, str):
            if not silent:
                print("TGraph.Quotient - Error: The input edgeWeightKey parameter is not a valid string. Returning None.")
            return None

        # ---------------------------------------------------------------------
        # Edge quotient mode.
        # Original edges become line-graph vertices. Then the normal vertex
        # quotient is applied to the line graph.
        # ---------------------------------------------------------------------

        if topologyType == "edge":
            try:
                lineGraph = TGraph.LineGraph(
                    topology,
                    directed=topology._directed,
                    transferDictionaries=True,
                )
            except TypeError:
                try:
                    lineGraph = TGraph.LineGraph(topology)
                except Exception:
                    lineGraph = None
            except Exception:
                lineGraph = None

            if not isinstance(lineGraph, TGraph):
                if not silent:
                    print("TGraph.Quotient - Error: Could not create a line graph for edge quotienting. Returning None.")
                return None

            return TGraph.Quotient(
                lineGraph,
                topologyType="vertex",
                key=key,
                groupLabelKey=groupLabelKey,
                groupCountKey=groupCountKey,
                weighted=weighted,
                edgeWeightKey=edgeWeightKey,
                idKey=idKey,
                transferDictionaries=transferDictionaries,
                silent=silent,
            )

        graph = topology

        # ---------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------

        def _normalise_label(value):
            """
            Converts potentially unhashable dictionary values into stable hashable
            group labels while preserving ordinary scalar values.
            """
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            if isinstance(value, tuple):
                return tuple(_normalise_label(v) for v in value)
            if isinstance(value, list):
                return tuple(_normalise_label(v) for v in value)
            if isinstance(value, dict):
                return tuple(sorted((str(k), _normalise_label(v)) for k, v in value.items()))
            try:
                hash(value)
                return value
            except Exception:
                return str(value)

        def _value_from_record(record, valueKey, default=None):
            if not isinstance(record, dict):
                return default
            d = record.get("dictionary", {})
            if not isinstance(d, dict):
                return default
            return d.get(valueKey, default)

        def _copy_dictionary(record):
            if not isinstance(record, dict):
                return {}
            d = record.get("dictionary", {})
            return dict(d) if isinstance(d, dict) else {}

        def _merge_into(target, source, overwrite=False):
            if not isinstance(source, dict):
                return target
            for k, v in source.items():
                if overwrite or k not in target:
                    target[k] = v
            return target

        def _average_coordinates(vertexRecords):
            coords = []
            for record in vertexRecords:
                vertexIndex = record.get("index", None)
                c = TGraph.Coordinates(graph, vertexIndex, default=None)
                if c is None or len(c) < 3:
                    continue
                try:
                    coords.append([float(c[0]), float(c[1]), float(c[2])])
                except Exception:
                    continue

            if not coords:
                return None

            n = float(len(coords))
            return [
                sum(c[0] for c in coords) / n,
                sum(c[1] for c in coords) / n,
                sum(c[2] for c in coords) / n,
            ]

        def _edge_pair(a, b, directed):
            if directed:
                return (a, b)
            return (a, b) if a <= b else (b, a)

        # ---------------------------------------------------------------------
        # Collect active vertices and group them by dictionary key.
        # ---------------------------------------------------------------------

        activeVertices = [v for v in graph._vertices if v.get("active", True)]

        if not activeVertices:
            if not silent:
                print("TGraph.Quotient - Error: The input TGraph has no active vertices. Returning None.")
            return None

        groups = {}

        for vertexRecord in activeVertices:
            label = _value_from_record(vertexRecord, key, None)
            label = _normalise_label(label)
            groups.setdefault(label, []).append(vertexRecord)

        if not groups:
            if not silent:
                print("TGraph.Quotient - Error: Could not form any groups. Returning None.")
            return None

        # ---------------------------------------------------------------------
        # Create quotient vertices.
        # ---------------------------------------------------------------------

        quotient = TGraph(
            directed=graph._directed,
            allowSelfLoops=False,
            allowParallelEdges=False,
            dictionary={
                "source": "TGraph.Quotient",
                "source_order": TGraph.Order(graph),
                "source_size": TGraph.Size(graph),
                "source_directed": graph._directed,
                "quotient_key": key,
                "quotient_topology_type": topologyType,
            },
        )

        labelToQuotientIndex = {}
        memberVertexToGroup = {}

        for groupIndex, label in enumerate(groups.keys()):
            members = groups[label]
            groupDictionary = {}

            if transferDictionaries:
                for member in members:
                    groupDictionary = _merge_into(
                        groupDictionary,
                        _copy_dictionary(member),
                        overwrite=False,
                    )

            groupDictionary[groupLabelKey] = label
            groupDictionary[key] = label
            groupDictionary[groupCountKey] = len(members)
            groupDictionary["index"] = groupIndex
            groupDictionary["member_indices"] = [m.get("index") for m in members]

            coordinates = _average_coordinates(members)

            if coordinates is not None:
                groupDictionary["x"] = coordinates[0]
                groupDictionary["y"] = coordinates[1]
                groupDictionary["z"] = coordinates[2]

            quotientIndex = quotient.AddVertex(dictionary=groupDictionary)
            labelToQuotientIndex[label] = quotientIndex

            for member in members:
                memberIndex = member.get("index", None)
                if memberIndex is not None:
                    memberVertexToGroup[memberIndex] = quotientIndex

        # ---------------------------------------------------------------------
        # Create quotient edges.
        # ---------------------------------------------------------------------

        pairToCount = {}
        pairToFirstEdgeDictionary = {}

        for edgeRecord in graph._edges:
            if not edgeRecord.get("active", True):
                continue

            srcIndex = edgeRecord.get("src", None)
            dstIndex = edgeRecord.get("dst", None)

            if srcIndex not in memberVertexToGroup or dstIndex not in memberVertexToGroup:
                continue

            srcGroup = memberVertexToGroup[srcIndex]
            dstGroup = memberVertexToGroup[dstIndex]

            # Quotient graph convention: ignore intra-group member adjacencies.
            if srcGroup == dstGroup:
                continue

            directedEdge = bool(edgeRecord.get("directed", graph._directed))
            pair = _edge_pair(srcGroup, dstGroup, graph._directed or directedEdge)

            pairToCount[pair] = pairToCount.get(pair, 0) + 1

            if pair not in pairToFirstEdgeDictionary:
                d = _copy_dictionary(edgeRecord)
                pairToFirstEdgeDictionary[pair] = d

        for pair, count in pairToCount.items():
            srcGroup, dstGroup = pair

            edgeDictionary = {}

            if transferDictionaries:
                edgeDictionary = dict(pairToFirstEdgeDictionary.get(pair, {}))

            edgeDictionary["src"] = srcGroup
            edgeDictionary["dst"] = dstGroup
            edgeDictionary["relationship"] = "quotient"
            edgeDictionary["source_adjacency_count"] = count

            if weighted:
                edgeDictionary[edgeWeightKey] = count

            quotient.AddEdge(
                srcGroup,
                dstGroup,
                directed=graph._directed,
                dictionary=edgeDictionary,
                representation=None,
            )

        return TGraph._OntologyAnnotateGraph(
            quotient,
            graphClass="top:QuotientGraph",
            vertexClass="top:Node",
            edgeClass="top:Relationship",
            generatedBy="TGraph.Quotient",
            ontology=True,
            silent=True,
        )


    @staticmethod
    def RDFString(*args, **kwargs) -> Optional[str]:
        """
        Returns an RDF/Turtle string representation of the input TGraph.

        Parameters
        ----------
        *args : Any , optional
            Additional positional arguments.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        Optional[str]
            The resulting rdfstring string.
        """
        return TGraph.TTLString(*args, **kwargs)

    def _register_edge_adjacency(self, edge_index: int, src: int, dst: int, directed: bool) -> None:
        """
        Registers an edge in this TGraph adjacency lookup tables.

        Parameters
        ----------
        edge_index : int
            The input edge index value.
        src : int
            The source vertex index.
        dst : int
            The destination vertex index.
        directed : bool
            If set to True, graph edges are treated as directed.

        Returns
        -------
        None
            None.
        """
        self._out_edges.setdefault(src, set()).add(edge_index)
        self._in_edges.setdefault(dst, set()).add(edge_index)
        self._incident_edges.setdefault(src, set()).add(edge_index)
        self._incident_edges.setdefault(dst, set()).add(edge_index)
        if not directed:
            self._out_edges.setdefault(dst, set()).add(edge_index)
            self._in_edges.setdefault(src, set()).add(edge_index)
        key = self._edge_key(src, dst, directed)
        self._edge_lookup.setdefault(key, set()).add(edge_index)

    def RemoveEdge(self, edge: Union[int, Dict[str, Any]], silent: bool = False) -> "TGraph":
        """
        Removes an edge from this TGraph.

        Parameters
        ----------
        edge : Union[int, Dict[str, Any]]
            The input edge, edge index, or edge record.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
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

    @staticmethod
    def RemoveIsolatedEdges(graph: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Removes isolated or invalid edge records from the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None
        for rec in list(graph._edges):
            if not rec.get("active", True):
                continue
            srcIndex = rec.get("src")
            dstIndex = rec.get("dst")
            if not graph._validate_vertex_index(srcIndex) or not graph._validate_vertex_index(dstIndex):
                graph.RemoveEdge(rec.get("index"), silent=True)
        return graph

    @staticmethod
    def RemoveIsolatedVertices(graph: "TGraph", mode: str = "all", silent: bool = False) -> Optional["TGraph"]:
        """
        Removes isolated vertices from the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graph, TGraph):
            return None
        for rec in list(graph._vertices):
            idx = rec.get("index")
            if rec.get("active", True) and TGraph.Degree(graph, idx, mode=mode) == 0:
                graph.RemoveVertex(idx, silent=True)
        return graph

    def RemoveVertex(self, vertex: Union[int, Dict[str, Any]], silent: bool = False) -> "TGraph":
        """
        Removes a vertex and its incident edges from this TGraph.

        Parameters
        ----------
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
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

    @staticmethod
    def _SagittaArcToWire(vertexA: Any, vertexB: Any, dictionary: Optional[Dict[str, Any]] = None,
                          representation: Any = None, sagittaKey: str = "sagitta",
                          tolerance: float = 0.0001, silent: bool = False) -> Optional[Any]:
        """
        Returns a wire arc between two vertices using sagitta metadata when available.

        Parameters
        ----------
        vertexA : Any
            The first input vertex or vertex index.
        vertexB : Any
            The second input vertex or vertex index.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        representation : Any , optional
            The optional representation object to store with the graph record. Default is None.
        sagittaKey : str , optional
            The dictionary key to use. Default is 'sagitta'.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Any]
            The resulting sagitta arc to wire object or value.
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
    def _SelfLoopToWire(
        vertex: Any,
        dictionary: Optional[Dict[str, Any]] = None,
        representation: Any = None,
        mode: str = "circle",
        radius: float = 0.25,
        majorRadius: Optional[float] = None,
        minorRadius: Optional[float] = None,
        sides: int = 32,
        normal: Optional[List[float]] = None,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional[Any]:
        """
        Returns a wire representation for a self-loop edge.

        Parameters
        ----------
        vertex : Any
            The input vertex, vertex index, or vertex record.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.
        representation : Any , optional
            The optional representation object to store with the graph record. Default is None.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'circle'.
        radius : float , optional
            The input radius. Default is 0.25.
        majorRadius : Optional[float] , optional
            The radius value to use. Default is None.
        minorRadius : Optional[float] , optional
            The radius value to use. Default is None.
        sides : int , optional
            The input sides value. Default is 32.
        normal : Optional[List[float]] , optional
            The input normal value. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Any]
            The resulting self loop to wire object or value.
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

        radius = _number(
            rep.get("radius"),
            d.get("self_loop_radius"),
            radius,
            default=0.25,
        )

        major = _number(
            rep.get("major_radius"),
            rep.get("majorRadius"),
            d.get("self_loop_major_radius"),
            majorRadius,
            radius,
            default=radius,
        )

        minor = _number(
            rep.get("minor_radius"),
            rep.get("minorRadius"),
            d.get("self_loop_minor_radius"),
            minorRadius,
            radius * 0.65,
            default=radius * 0.65,
        )

        if mode == "circle":
            major = radius
            minor = radius

        try:
            sides = int(rep.get("sides", d.get("self_loop_sides", sides)))
        except Exception:
            sides = 32
        sides = max(8, sides)

        normal = rep.get("normal", d.get("self_loop_normal", normal))

        anchor = TGraph._VertexCoordinates(vertex)
        if anchor is None:
            return None

        u, v, _ = TGraph._FrameFromNormal(normal)

        # The graph vertex is the anchor point on the loop perimeter.
        # Move the loop centre along +U so that the local -U point of the loop
        # coincides with the graph vertex.
        centre = [
            anchor[0] + major * u[0],
            anchor[1] + major * u[1],
            anchor[2] + major * u[2],
        ]

        points = []

        # Start at angle pi so that the first point is exactly the graph vertex.
        for i in range(sides):
            angle = math.pi + (2.0 * math.pi * float(i) / float(sides))
            ca = math.cos(angle)
            sa = math.sin(angle)

            points.append([
                centre[0] + major * ca * u[0] + minor * sa * v[0],
                centre[1] + major * ca * u[1] + minor * sa * v[1],
                centre[2] + major * ca * u[2] + minor * sa * v[2],
            ])

        points.append(points[0])

        return TGraph._ControlPointsToWire(
            points,
            dictionary=d,
            tolerance=tolerance,
            silent=silent,
        )

    def SetDictionary(self, dictionary: Optional[Dict[str, Any]] = None) -> "TGraph":
        """
        Sets the dictionary of this TGraph.

        Parameters
        ----------
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
        self._dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
        self._invalidate_cache()
        return self

    def SetEdgeDictionary(self, index: int, dictionary: Optional[Dict[str, Any]] = None) -> "TGraph":
        """
        Sets the dictionary of an edge in this TGraph.

        Parameters
        ----------
        index : int
            The input index.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
        if not self._validate_edge_index(index, active=False):
            return self
        e = self._edges[index]
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d.update({"index": index, "src": e["src"], "dst": e["dst"], "directed": e["directed"], "active": e.get("active", True)})
        self._edges[index]["dictionary"] = d
        self._invalidate_cache()
        return self

    @staticmethod
    def SetOntologyCategory(graph: "TGraph", category: str, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        """
        Sets the ontology category of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        category : str
            The ontology category value.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(category, str) or category.strip() == "":
            return None
        return TGraph._OntologySet(graph, "category", category.strip(), element=element, index=index)

    @staticmethod
    def SetOntologyClass(graph: "TGraph", ontologyClass: str, element: str = "graph", index: Optional[int] = None,
                         setCategory: bool = True, setURI: bool = True) -> Optional["TGraph"]:
        """
        Sets the ontology class of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        ontologyClass : str
            The ontology class value.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.
        setCategory : bool , optional
            The input set category value. Default is True.
        setURI : bool , optional
            The input set uri value. Default is True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
    def SetOntologyLabel(graph: "TGraph", label: Any, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        """
        Sets the ontology label of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        label : Any
            The label value.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        return TGraph._OntologySet(graph, "label", label, element=element, index=index)

    @staticmethod
    def SetOntologyURI(graph: "TGraph", uri: str, element: str = "graph", index: Optional[int] = None) -> Optional["TGraph"]:
        """
        Sets the ontology URI of a graph, vertex, or edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        uri : str
            The URI value.
        element : str , optional
            The graph element to annotate or query. Valid values are typically "graph",
            "vertex", or "edge". Default is 'graph'.
        index : Optional[int] , optional
            The input index. Default is None.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(uri, str) or uri.strip() == "":
            return None
        return TGraph._OntologySet(graph, "uri", uri.strip(), element=element, index=index)

    def SetVertexDictionary(self, index: int, dictionary: Optional[Dict[str, Any]] = None) -> "TGraph":
        """
        Sets the dictionary of a vertex in this TGraph.

        Parameters
        ----------
        index : int
            The input index.
        dictionary : Optional[Dict[str, Any]] , optional
            The input dictionary. Default is None.

        Returns
        -------
        TGraph
            The resulting TGraph, or None if the operation fails.
        """
        if not self._validate_vertex_index(index, active=False):
            return self
        d = dict(dictionary) if isinstance(dictionary, dict) else {}
        d["index"] = index
        d.setdefault("active", self._vertices[index].get("active", True))
        self._vertices[index]["dictionary"] = d
        self._invalidate_cache()
        return self

    @staticmethod
    def _SetVertexValue(graph: "TGraph", index: int, key: Optional[str], value: Any) -> None:
        """
        Sets a dictionary value on a vertex record.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        index : int
            The input index.
        key : Optional[str]
            The dictionary key to use.
        value : Any
            The input value value.

        Returns
        -------
        None
            None.
        """
        if not isinstance(graph, TGraph) or key is None:
            return
        if not graph._validate_vertex_index(index):
            return
        graph._vertices[index].setdefault("dictionary", {})[key] = value

    @staticmethod
    def ShortestPath(graph: "TGraph", source: int, target: int, mode: str = "out", useNumba: bool = False) -> Optional[List[int]]:
        """
        Returns a shortest path between two vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        source : int
            The input source vertex, vertex index, or source identifier.
        target : int
            The input target vertex, vertex index, graph, or comparison value, depending on
            context.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        useNumba : bool , optional
            If set to True, Numba acceleration is used when available. Default is False.

        Returns
        -------
        Optional[List[int]]
            The resulting shortest path list.
        """
        use_np = bool(useNumba)
        c = TGraph.Compile(graph, useNumpy=use_np, useSciPy=False, useNumba=useNumba)
        if not isinstance(c, dict):
            return None
        pos = c["position"]

        def _resolve_vertex_index(graph, v):
            """
            Resolves a TGraph vertex input to an active vertex index.

            Accepts:
            - integer vertex index
            - internal vertex dictionary/object returned by TGraph.NearestVertex
            """

            if isinstance(v, int):
                return v

            # Case 1: vertex is exactly one of graph._vertices entries.
            try:
                for i, gv in enumerate(graph._vertices):
                    if gv is v:
                        return i
            except Exception:
                pass

            # Case 2: vertex is equal to one of graph._vertices entries.
            try:
                for i, gv in enumerate(graph._vertices):
                    if gv == v:
                        return i
            except Exception:
                pass

            # Case 3: vertex dictionary contains an explicit index/id.
            if isinstance(v, dict):
                for key in ("index", "id", "vertex_index", "vertexIndex"):
                    value = v.get(key, None)
                    if isinstance(value, int):
                        return value

            return None


        source_index = _resolve_vertex_index(graph, source)
        target_index = _resolve_vertex_index(graph, target)

        if source_index is None or target_index is None:
            return None

        if source_index not in pos or target_index not in pos:
            return None

        s = pos[source_index]
        t = pos[target_index]
        if s == t:
            return [source]

        adj_key, indptr_key, indices_key = TGraph._CompiledAdjacencyKeys(mode)

        if useNumba and c.get("numpy_available", False):
            bfs = TGraph._NumbaShortestPathKernel()
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

    @staticmethod
    def ShortestPaths(graph: "TGraph", pairs: Iterable[Tuple[int, int]], mode: str = "out", useNumba: bool = False,
                      grouped: Any = "auto", groupThreshold: float = 1.5) -> List[Optional[List[int]]]:
        """
        Returns shortest paths between all requested vertex pairs of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        pairs : Iterable[Tuple[int, int]]
            The input pairs value.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        useNumba : bool , optional
            If set to True, Numba acceleration is used when available. Default is False.
        grouped : Any , optional
            The input grouped value. Default is 'auto'.
        groupThreshold : float , optional
            The input group threshold value. Default is 1.5.

        Returns
        -------
        List[Optional[List[int]]]
            The resulting shortest paths list.
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
    def ShortestPathsFromSource(graph: "TGraph", source: int, targets: Optional[Iterable[int]] = None,
                                mode: str = "out", useNumba: bool = False,
                                returnTree: bool = False) -> Dict[int, Optional[List[int]]]:
        """
        Returns shortest paths from one source vertex to reachable vertices.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        source : int
            The input source vertex, vertex index, or source identifier.
        targets : Optional[Iterable[int]] , optional
            The input targets value. Default is None.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        useNumba : bool , optional
            If set to True, Numba acceleration is used when available. Default is False.
        returnTree : bool , optional
            The input return tree value. Default is False.

        Returns
        -------
        Dict[int, Optional[List[int]]]
            The resulting shortest paths from source list.
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
    def ShortestPathTree(graph: "TGraph", source: int, mode: str = "out", useNumba: bool = False,
                         includePaths: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns a shortest path tree rooted at the input source vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        source : int
            The input source vertex, vertex index, or source identifier.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        useNumba : bool , optional
            If set to True, Numba acceleration is used when available. Default is False.
        includePaths : bool , optional
            If set to True, include paths are included. Default is False.

        Returns
        -------
        Optional[Dict[str, Any]]
            The resulting shortest path tree dictionary.
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
            bfs_tree = TGraph._NumbaBFSTreeKernel()
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
    def ShortestPathViaVertices(graph: "TGraph", startVertex, endVertex, vertices: list = None, tolerance: float = 0.0001, silent: bool = False) -> Optional[List[int]]:
        """
        Returns a shortest path that passes through a sequence of required vertices.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        startVertex : Any
            The input start vertex value.
        endVertex : Any
            The input end vertex value.
        vertices : list , optional
            The input vertices or vertex indices. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[List[int]]
            The resulting shortest path via vertices list.
        """
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
    def Show(*graphs,
             sagitta = 0,
             absolute = False,
             sides = 8,
             angle = 0,
             vertexColor="black",
             vertexColorKey=None,
             vertexSize=10,
             vertexSizeKey=None,
             vertexLabelKey=None,
             vertexGroupKey=None,
             vertexGroups=[],
             vertexMinGroup=None,
             vertexMaxGroup=None,
             showVertices=True,
             showVertexLabel=False,
             showVertexLegend=False,
             edgeColor="red",
             edgeColorKey=None,
             edgeWidth=1,
             edgeWidthKey=None,
             edgeLabelKey=None,
             edgeGroupKey=None,
             edgeGroups=[],
             edgeMinGroup=None,
             edgeMaxGroup=None,
             showEdges=True,
             showEdgeLabel=False,
             showEdgeLegend=False,
             colorScale='viridis',
             renderer=None,
             width=950,
             height=500,
             xAxis=False,
             yAxis=False,
             zAxis=False,
             axisSize=1,
             backgroundColor='rgba(0,0,0,0)',
             marginLeft=0,
             marginRight=0,
             marginTop=20,
             marginBottom=0,
             camera=[-1.25, -1.25, 1.25],
             center=[0, 0, 0], up=[0, 0, 1],
             projection="perspective",
             tolerance=0.0001,
             silent=False):
        """
        Displays a representation of the input TGraph.

        Parameters
        ----------
        *graphs : Any , optional
            The input list of TGraphs.
        sagitta : Any , optional
            The input sagitta value. Default is 0.
        absolute : Any , optional
            The input absolute value. Default is False.
        sides : Any , optional
            The input sides value. Default is 8.
        angle : Any , optional
            The input angle value. Default is 0.
        vertexColor : Any , optional
            The color value to use. Default is 'black'.
        vertexColorKey : Any , optional
            The dictionary key to use. Default is None.
        vertexSize : Any , optional
            The input vertex size value. Default is 10.
        vertexSizeKey : Any , optional
            The dictionary key to use. Default is None.
        vertexLabelKey : Any , optional
            The dictionary key to use. Default is None.
        vertexGroupKey : Any , optional
            The dictionary key to use. Default is None.
        vertexGroups : Any , optional
            The input vertex groups value. Default is [].
        vertexMinGroup : Any , optional
            The input vertex min group value. Default is None.
        vertexMaxGroup : Any , optional
            The input vertex max group value. Default is None.
        showVertices : Any , optional
            If set to True, show vertices are shown. Default is True.
        showVertexLabel : Any , optional
            If set to True, show vertex label are shown. Default is False.
        showVertexLegend : Any , optional
            If set to True, show vertex legend are shown. Default is False.
        edgeColor : Any , optional
            The color value to use. Default is 'red'.
        edgeColorKey : Any , optional
            The dictionary key to use. Default is None.
        edgeWidth : Any , optional
            The input edge width value. Default is 1.
        edgeWidthKey : Any , optional
            The dictionary key to use. Default is None.
        edgeLabelKey : Any , optional
            The dictionary key to use. Default is None.
        edgeGroupKey : Any , optional
            The dictionary key to use. Default is None.
        edgeGroups : Any , optional
            The input edge groups value. Default is [].
        edgeMinGroup : Any , optional
            The input edge min group value. Default is None.
        edgeMaxGroup : Any , optional
            The input edge max group value. Default is None.
        showEdges : Any , optional
            If set to True, show edges are shown. Default is True.
        showEdgeLabel : Any , optional
            If set to True, show edge label are shown. Default is False.
        showEdgeLegend : Any , optional
            If set to True, show edge legend are shown. Default is False.
        colorScale : Any , optional
            The Plotly color scale to use. Default is 'viridis'.
        renderer : Any , optional
            The input renderer value. Default is None.
        width : Any , optional
            The figure or output width. Default is 950.
        height : Any , optional
            The figure or output height. Default is 500.
        xAxis : Any , optional
            The input x axis value. Default is False.
        yAxis : Any , optional
            The input y axis value. Default is False.
        zAxis : Any , optional
            The input z axis value. Default is False.
        axisSize : Any , optional
            The input axis size value. Default is 1.
        backgroundColor : Any , optional
            The color value to use. Default is 'rgba(0,0,0,0)'.
        marginLeft : Any , optional
            The input margin left value. Default is 0.
        marginRight : Any , optional
            The input margin right value. Default is 0.
        marginTop : Any , optional
            The input margin top value. Default is 20.
        marginBottom : Any , optional
            The input margin bottom value. Default is 0.
        camera : Any , optional
            The input camera value. Default is [-1.25, -1.25, 1.25].
        center : Any , optional
            The input center value. Default is [0, 0, 0].
        up : Any , optional
            The input up value. Default is [0, 0, 1].
        projection : Any , optional
            The input projection value. Default is 'perspective'.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Any
            The resulting show object or value.
        """
        from topologicpy.Plotly import Plotly
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if isinstance(graphs, tuple):
            graphs = Helper.Flatten(list(graphs))
        if isinstance(graphs, list):
            new_graphs = [t for t in graphs if isinstance(t, TGraph)]
        if len(new_graphs) == 0:
            if not silent:
                print("Graph.Show - Error: the input graphs parameter does not contain any valid graphs. Returning None.")
            return None
        data = []
        for graph in new_graphs:
            data += Plotly.DataByTGraph(graph,
                                    sagitta=sagitta,
                                    absolute=absolute,
                                    sides=sides,
                                    angle=angle,
                                    vertexColor=vertexColor,
                                    vertexColorKey=vertexColorKey,
                                    vertexSize=vertexSize,
                                    vertexSizeKey=vertexSizeKey,
                                    vertexLabelKey=vertexLabelKey,
                                    vertexGroupKey=vertexGroupKey,
                                    vertexGroups=vertexGroups,
                                    vertexMinGroup=vertexMinGroup,
                                    vertexMaxGroup=vertexMaxGroup,
                                    showVertices=showVertices,
                                    showVertexLabel=showVertexLabel,
                                    showVertexLegend=showVertexLegend,
                                    edgeColor=edgeColor,
                                    edgeColorKey=edgeColorKey,
                                    edgeWidth=edgeWidth,
                                    edgeWidthKey=edgeWidthKey,
                                    edgeLabelKey=edgeLabelKey,
                                    edgeGroupKey=edgeGroupKey,
                                    edgeGroups=edgeGroups,
                                    edgeMinGroup=edgeMinGroup,
                                    edgeMaxGroup=edgeMaxGroup,
                                    showEdges=showEdges,
                                    showEdgeLabel=showEdgeLabel,
                                    showEdgeLegend=showEdgeLegend,
                                    colorScale=colorScale,
                                    silent=silent)
        fig = Plotly.FigureByData(data, width=width, height=height, xAxis=xAxis, yAxis=yAxis, zAxis=zAxis, axisSize=axisSize, backgroundColor=backgroundColor,
                                  marginLeft=marginLeft, marginRight=marginRight, marginTop=marginTop, marginBottom=marginBottom, tolerance=tolerance)
        Plotly.Show(fig, renderer=renderer, camera=camera, center=center, up=up, projection=projection)

    @staticmethod
    def _SimpleUndirectedNeighborSets(graph: "TGraph", includeSelfLoops: bool = False) -> Dict[int, Set[int]]:
        """
        Returns simple undirected neighbor sets for the active vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeSelfLoops : bool , optional
            If set to True, include self loops are included. Default is False.

        Returns
        -------
        Dict[int, Set[int]]
            The resulting simple undirected neighbor sets dictionary.
        """
        if not isinstance(graph, TGraph):
            return {}
        adjacency = {v: set() for v in TGraph.ActiveVertexIndices(graph)}
        for e in graph._edges:
            if not e.get("active", True):
                continue
            u = e.get("src")
            v = e.get("dst")
            if u not in adjacency or v not in adjacency:
                continue
            if u == v and not includeSelfLoops:
                continue
            adjacency[u].add(v)
            adjacency[v].add(u)
        return adjacency

    @staticmethod
    def Size(graph: "TGraph") -> int:
        """
        Returns the number of active edges in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        int
            The number of active edges.
        """
        if not isinstance(graph, TGraph):
            return 0
        return sum(1 for e in graph._edges if e.get("active", True))

    @staticmethod
    def Subgraph(graph: "TGraph", vertices: Iterable[int], induced: bool = True) -> Optional["TGraph"]:
        """
        Returns a subgraph containing the requested vertices.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertices : Iterable[int]
            The input vertices or vertex indices.
        induced : bool , optional
            The input induced value. Default is True.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass=graph._dictionary.get("ontology_class", "top:Graph"), vertexClass="top:Node",
            edgeClass="top:Relationship", generatedBy="TGraph.Subgraph", ontology=True, silent=True)

    @staticmethod
    def SubGraphMatches(subGraph: "TGraph", superGraph: "TGraph", strict: bool = False,
                        vertexKeys: Any = None, edgeKeys: Any = None,
                        vertexKey: str = None,
                        maxMatches: int = 25, timeLimit: float = 5,
                        silent: bool = False) -> List["TGraph"]:
        """
        Returns subgraph matches of a pattern graph in a target graph.

        Parameters
        ----------
        subGraph : 'TGraph'
            The input sub graph value.
        superGraph : 'TGraph'
            The input super graph value.
        strict : bool , optional
            The input strict value. Default is False.
        vertexKeys : Any , optional
            The input vertex keys value. Default is None.
        edgeKeys : Any , optional
            The input edge keys value. Default is None.
        vertexKey : str , optional
            The vertex dictionary key to use. Default is None.
        maxMatches : int , optional
            The input max matches value. Default is 25.
        timeLimit : float , optional
            The maximum time, in seconds, allowed for the search. Default is 5.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        List[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(subGraph, TGraph) or not isinstance(superGraph, TGraph):
            return []

        if vertexKeys is None and vertexKey is not None:
            vertexKeys = vertexKey

        # Use the existing P6 matcher here because it supports the intended
        # strictPath=False semantics, where a pattern edge may map to a path.
        if hasattr(TGraph, "_P6SubgraphIsomorphisms"):
            mappings = TGraph._P6SubgraphIsomorphisms(
                subGraph,
                superGraph,
                vertexKeys=vertexKeys,
                edgeKeys=edgeKeys,
                maxMatches=maxMatches,
                timeLimit=timeLimit,
                tolerance=0.0,
                strictPath=bool(strict),
            )
        else:
            mappings = TGraph._P8SubgraphIsomorphisms(
                subGraph,
                superGraph,
                vertexKeys=vertexKeys,
                edgeKeys=edgeKeys,
                maxMatches=maxMatches,
                timeLimit=timeLimit,
                exact=bool(strict),
            )

        result = []
        pattern_vertices = TGraph._P8ActiveVertexIndices(subGraph)

        for mapping in mappings:
            new_g = TGraph(
                directed=subGraph._directed,
                allowSelfLoops=subGraph._allow_self_loops,
                allowParallelEdges=subGraph._allow_parallel_edges,
                dictionary=dict(subGraph._dictionary),
            )

            old_to_new = {}

            for pv in pattern_vertices:
                if pv not in mapping:
                    continue
                sv = mapping[pv]
                d = dict(subGraph._vertices[pv].get("dictionary", {}))
                try:
                    d.update(dict(superGraph._vertices[sv].get("dictionary", {})))
                except Exception:
                    pass
                old_to_new[pv] = new_g.AddVertex(
                    dictionary=d,
                    representation=superGraph._vertices[sv].get("representation", None)
                    if superGraph._validate_vertex_index(sv) else None,
                )

            for pe in TGraph._P8ActiveEdgeRecords(subGraph):
                ps = pe.get("src")
                pd = pe.get("dst")
                if ps in old_to_new and pd in old_to_new:
                    new_g.AddEdge(
                        old_to_new[ps],
                        old_to_new[pd],
                        directed=pe.get("directed", subGraph._directed),
                        dictionary=dict(pe.get("dictionary", {})),
                        representation=pe.get("representation", None),
                    )

            result.append(new_g)

        return result

    @staticmethod
    def SymmetricDifference(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the symmetric difference of two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            return None
        return TGraph.Union(TGraph.Difference(graphA, graphB), TGraph.Difference(graphB, graphA))

    @staticmethod
    def TopologicalDistance(graph: "TGraph", vertexA: Any, vertexB: Any, mode: str = "out",
                            silent: bool = False) -> Optional[int]:
        """
        Returns the topological distance between two vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexA : Any
            The first input vertex or vertex index.
        vertexB : Any
            The second input vertex or vertex index.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'out'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[int]
            The resulting topological distance index or count.
        """
        if not isinstance(graph, TGraph):
            return None
        a = TGraph.VertexIndex(graph, vertexA)
        b = TGraph.VertexIndex(graph, vertexB)
        if a is None or b is None:
            return None
        path = TGraph.ShortestPath(graph, a, b, mode=mode)
        return None if path is None else max(0, len(path) - 1)

    @staticmethod
    def TopologicEdge(graph: "TGraph",
                      edge: Union[int, Dict[str, Any]],
                      transferVertexDictionaries: bool = True,
                      transferEdgeDictionary: bool = True,
                      useRepresentation: bool = True,
                      mantissa: int = 6,
                      tolerance: float = 0.0001,
                      silent: bool = False):
        """
        Converts a TGraph edge to a Topologic edge.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        edge : Union[int, Dict[str, Any]]
            The input edge, edge index, or edge record.
        transferVertexDictionaries : bool , optional
            If set to True, vertex dictionaries are transferred to created Topologic vertices.
            Default is True.
        transferEdgeDictionary : bool , optional
            If set to True, the edge dictionary is transferred to the created Topologic edge.
            Default is True.
        useRepresentation : bool , optional
            If set to True, stored representation objects are used when possible. Default is
            True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge or None
            The converted Topologic edge.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.TopologicEdge - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        idx = TGraph.EdgeIndex(graph, edge)

        if idx is None:
            if not silent:
                print("TGraph.TopologicEdge - Error: Could not resolve the input edge. Returning None.")
            return None

        if not graph._validate_edge_index(idx, active=False):
            if not silent:
                print("TGraph.TopologicEdge - Error: The input edge index is invalid. Returning None.")
            return None

        record = graph._edges[idx]
        d = dict(record.get("dictionary", {}))
        rep = record.get("representation", None)
        topology = None

        if useRepresentation and rep is not None:
            try:
                from topologicpy.Topology import Topology
                if Topology.IsInstance(rep, "Edge"):
                    topology = rep
            except Exception:
                topology = None

        if topology is None:
            src = record.get("src", None)
            dst = record.get("dst", None)

            if not isinstance(src, int) or not isinstance(dst, int):
                if not silent:
                    print("TGraph.TopologicEdge - Error: The edge does not contain valid src and dst indices. Returning None.")
                return None

            if src == dst:
                if not silent:
                    print("TGraph.TopologicEdge - Error: A self-loop cannot be converted to a straight Topologic Edge. Returning None.")
                return None

            sv = TGraph.TopologicVertex(
                graph,
                src,
                transferDictionary=transferVertexDictionaries,
                useRepresentation=useRepresentation,
                mantissa=mantissa,
                silent=silent,
            )

            tv = TGraph.TopologicVertex(
                graph,
                dst,
                transferDictionary=transferVertexDictionaries,
                useRepresentation=useRepresentation,
                mantissa=mantissa,
                silent=silent,
            )

            if sv is None or tv is None:
                if not silent:
                    print("TGraph.TopologicEdge - Error: Could not create source or destination vertex. Returning None.")
                return None

            try:
                from topologicpy.Edge import Edge
                topology = Edge.ByStartVertexEndVertex(sv, tv, tolerance=tolerance)
            except Exception:
                try:
                    topology = Edge.ByVertices([sv, tv])
                except Exception:
                    if not silent:
                        print("TGraph.TopologicEdge - Error: Could not create Topologic edge. Returning None.")
                    return None

        if topology is not None and transferEdgeDictionary:
            try:
                from topologicpy.Topology import Topology
                td = TGraph._PythonToDictionary(d)
                if td is not None:
                    topology = Topology.SetDictionary(topology, td, silent=True)
            except Exception:
                pass

        return topology

    @staticmethod
    def TopologicVertex(graph: "TGraph",
                        vertex: Union[int, Dict[str, Any]],
                        transferDictionary: bool = True,
                        useRepresentation: bool = True,
                        mantissa: int = 6,
                        silent: bool = False):
        """
        Converts a TGraph vertex to a Topologic vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.
        transferDictionary : bool , optional
            If set to True, the source dictionary is transferred to the created topology.
            Default is True.
        useRepresentation : bool , optional
            If set to True, stored representation objects are used when possible. Default is
            True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex or None
            The converted Topologic vertex.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.TopologicVertex - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        idx = TGraph.VertexIndex(graph, vertex)

        if idx is None:
            if not silent:
                print("TGraph.TopologicVertex - Error: Could not resolve the input vertex. Returning None.")
            return None

        if not graph._validate_vertex_index(idx, active=False):
            if not silent:
                print("TGraph.TopologicVertex - Error: The input vertex index is invalid. Returning None.")
            return None

        record = graph._vertices[idx]
        d = dict(record.get("dictionary", {}))
        rep = record.get("representation", None)
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
            x = d.get("x", None)
            y = d.get("y", None)
            z = d.get("z", None)

            if not all(isinstance(c, (int, float)) for c in [x, y, z]):
                coords = TGraph.Coordinates(graph, idx, default=None)
                if coords is not None and len(coords) >= 3:
                    x, y, z = coords[0], coords[1], coords[2]

            if not all(isinstance(c, (int, float)) for c in [x, y, z]):
                if not silent:
                    print("TGraph.TopologicVertex - Error: Could not find valid x, y, z coordinates. Returning None.")
                return None

            try:
                from topologicpy.Vertex import Vertex
                v = Vertex.ByCoordinates(
                    round(float(x), mantissa),
                    round(float(y), mantissa),
                    round(float(z), mantissa)
                )
            except Exception:
                if not silent:
                    print("TGraph.TopologicVertex - Error: Could not create Topologic vertex. Returning None.")
                return None

        if transferDictionary:
            try:
                from topologicpy.Topology import Topology
                td = TGraph._PythonToDictionary(d)
                if td is not None:
                    v = Topology.SetDictionary(v, td, silent=True)
            except Exception:
                pass

        return v

    @staticmethod
    def Topology(graph: "TGraph", includeVertices: bool = True, includeEdges: bool = True,
                 useRepresentations: bool = True, segmentCurves: bool = True,
                 tolerance: float = 0.0001, silent: bool = False) -> Optional[Any]:
        """
        Returns a Topologic topology representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeVertices : bool , optional
            If set to True, include vertices are included. Default is True.
        includeEdges : bool , optional
            If set to True, include edges are included. Default is True.
        useRepresentations : bool , optional
            If set to True, stored representation objects are used when possible. Default is
            True.
        segmentCurves : bool , optional
            The input segment curves value. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Any]
            The resulting topology object or value.
        """
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

    @staticmethod
    def _TopologyCoordinates(topology: Any, useInternalVertex: bool = False, mantissa: int = 6, tolerance: float = 0.0001) -> Optional[List[float]]:
        """
        Returns representative coordinates for the input Topologic topology.

        Parameters
        ----------
        topology : Any
            The input Topologic topology.
        useInternalVertex : bool , optional
            If set to True, an internal vertex is used when deriving topology coordinates.
            Default is False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        Optional[List[float]]
            The resulting topology coordinates list.
        """
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
    def _TopologyDictionary(topology: Any, storeBREP: bool = False, mantissa: int = 6, tolerance: float = 0.0001, useInternalVertex: bool = False) -> Dict[str, Any]:
        """
        Returns a Python dictionary extracted from a Topologic topology and its geometry.

        Parameters
        ----------
        topology : Any
            The input Topologic topology.
        storeBREP : bool , optional
            If set to True, BREP strings are stored in dictionaries where possible. Default is
            False.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        useInternalVertex : bool , optional
            If set to True, an internal vertex is used when deriving topology coordinates.
            Default is False.

        Returns
        -------
        Dict[str, Any]
            The resulting topology dictionary dictionary.
        """
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
    def _TopologyDictionaryToPython(topology: Any) -> Dict[str, Any]:
        """
        Returns the dictionary of a Topologic topology as a Python dictionary.

        Parameters
        ----------
        topology : Any
            The input Topologic topology.

        Returns
        -------
        Dict[str, Any]
            The resulting topology dictionary to python dictionary.
        """
        if topology is None:
            return {}
        try:
            from topologicpy.Topology import Topology
            d = Topology.Dictionary(topology)
        except Exception:
            d = None
        return TGraph._DictionaryToPython(d)

    @staticmethod
    def _TopologyFromAperture(topology: Any) -> Any:
        """
        Returns the topology associated with an aperture when possible.

        Parameters
        ----------
        topology : Any
            The input Topologic topology.

        Returns
        -------
        Any
            The resulting topology from aperture object or value.
        """
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Aperture import Aperture
            if Topology.IsInstance(topology, "Aperture"):
                return Aperture.Topology(topology)
        except Exception:
            pass
        return topology

    @staticmethod
    def _TopologyType(topology: Any) -> str:
        """
        Returns the Topologic type name of the input topology.

        Parameters
        ----------
        topology : Any
            The input Topologic topology.

        Returns
        -------
        str
            The resulting topology type string.
        """
        if topology is None:
            return "None"
        try:
            from topologicpy.Topology import Topology
            return str(Topology.TypeAsString(topology))
        except Exception:
            return type(topology).__name__

    @staticmethod
    def ToPython(graph: "TGraph", includeRepresentations: bool = False) -> Dict[str, Any]:
        """
        Returns a Python dictionary representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeRepresentations : bool , optional
            If set to True, stored representation objects are included in the output. Default is
            False.

        Returns
        -------
        Dict[str, Any]
            The resulting to python dictionary.
        """
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
    def Tree(graph: "TGraph", vertex=None, mode: str = "all", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns a tree derived from the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : int or dict , optional
            The input vertex, vertex index, or vertex record. Default is None.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:Tree", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.Tree", ontology=True, silent=silent)

    @staticmethod
    def Triples(graph: "TGraph", **kwargs) -> List[Tuple[str, str, str]]:
        """
        Returns relationship triples representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        **kwargs : Any , optional
            Additional keyword arguments.

        Returns
        -------
        List[Tuple[str, str, str]]
            The resulting triples list.
        """
        return TGraph.OntologyTriples(graph, **kwargs)

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
        """
        Returns a Turtle TTL string representing the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        includeVertices : bool , optional
            If set to True, include vertices are included. Default is True.
        includeEdges : bool , optional
            If set to True, include edges are included. Default is True.
        includeDictionaries : bool , optional
            If set to True, dictionaries are included in the output. Default is True.
        includeBOT : bool , optional
            If set to True, include bot are included. Default is True.
        namespacePrefix : str , optional
            The input namespace prefix value. Default is 'inst'.
        instanceNamespace : str , optional
            The input instance namespace value. Default is
            'http://w3id.org/topologicpy/instance#'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[str]
            The resulting ttlstring string.
        """
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
    def TurtleFromTriples(
        triples: List[Tuple[str, str, str]],
        namespaces: Optional[Dict[str, str]] = None,
        instanceNamespace: str = "http://w3id.org/topologicpy/instance#",
        includeHeader: bool = True,
    ) -> str:
        """
        Returns a Turtle string generated from triples.

        Parameters
        ----------
        triples : List[Tuple[str, str, str]]
            The input triples value.
        namespaces : Optional[Dict[str, str]] , optional
            The input namespaces value. Default is None.
        instanceNamespace : str , optional
            The input instance namespace value. Default is
            'http://w3id.org/topologicpy/instance#'.
        includeHeader : bool , optional
            If set to True, include header are included. Default is True.

        Returns
        -------
        str
            The resulting turtle from triples string.
        """
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
    def _UndirectedAdjacency(graph: "TGraph") -> Dict[int, Set[int]]:
        """
        Returns an undirected adjacency dictionary for the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.

        Returns
        -------
        Dict[int, Set[int]]
            The resulting undirected adjacency dictionary.
        """
        adjacency = {i: set() for i in TGraph._ActiveVertexIndices(graph)}
        for e in TGraph._ActiveEdges(graph):
            src = e.get("src")
            dst = e.get("dst")
            if src in adjacency and dst in adjacency:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
        return adjacency

    @staticmethod
    def Union(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the union of two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            return None
        g = TGraph._CopyGraph(graphA)
        n = max(len(graphA._vertices), len(graphB._vertices))
        while len(g._vertices) < n:
            idx = len(g._vertices)
            bv = graphB._vertices[idx] if idx < len(graphB._vertices) else {}
            d = dict(bv.get("dictionary", {})) if isinstance(bv, dict) else {}
            d.pop("index", None)
            g.AddVertex(dictionary=d, representation=bv.get("representation", None) if isinstance(bv, dict) else None)
        for e in graphB._edges:
            if not e.get("active", True):
                continue
            src, dst, edirected = e.get("src"), e.get("dst"), e.get("directed", graphB._directed)
            if src >= len(g._vertices) or dst >= len(g._vertices):
                continue
            if not TGraph.HasEdge(g, src, dst, directed=edirected):
                d = dict(e.get("dictionary", {})); d.pop("index", None)
                g.AddEdge(src, dst, directed=edirected, dictionary=d, representation=e.get("representation", None))
        return g

    def _unregister_edge_adjacency(self, edge_index: int) -> None:
        """
        Unregisters an edge from this TGraph adjacency lookup tables.

        Parameters
        ----------
        edge_index : int
            The input edge index value.

        Returns
        -------
        None
            None.
        """
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

    def _validate_edge_index(self, index: int, active: bool = True) -> bool:
        """
        Returns True if the input edge index is valid.

        Parameters
        ----------
        index : int
            The input index.
        active : bool , optional
            The input active value. Default is True.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not isinstance(index, int) or index < 0 or index >= len(self._edges):
            return False
        return True if not active else bool(self._edges[index].get("active", True))

    def _validate_vertex_index(self, index: int, active: bool = True) -> bool:
        """
        Returns True if the input vertex index is valid.

        Parameters
        ----------
        index : int
            The input index.
        active : bool , optional
            The input active value. Default is True.

        Returns
        -------
        bool
            True if the requested condition is satisfied. Otherwise, False.
        """
        if not isinstance(index, int) or index < 0 or index >= len(self._vertices):
            return False
        return True if not active else bool(self._vertices[index].get("active", True))

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
        """
        Validates ontology metadata in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        requireClass : bool , optional
            The input require class value. Default is True.
        requireVertexClasses : bool , optional
            The input require vertex classes value. Default is False.
        requireEdgeClasses : bool , optional
            The input require edge classes value. Default is False.
        requireLabels : bool , optional
            The input require labels value. Default is False.
        checkClassKnown : bool , optional
            The input check class known value. Default is True.
        checkCategory : bool , optional
            The input check category value. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Dict[str, Any]
            The resulting validate ontology dictionary.
        """
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
        """
        Validates a Turtle TTL string.

        Parameters
        ----------
        ttlString : str
            The input ttl string value.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Dict[str, Any]
            The resulting validate ttlstring dictionary.
        """
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

    @staticmethod
    def _VectorCross(a: List[float], b: List[float]) -> List[float]:
        """
        Returns the cross product of two vectors.

        Parameters
        ----------
        a : List[float]
            The input a value.
        b : List[float]
            The input b value.

        Returns
        -------
        List[float]
            The resulting vector cross list.
        """
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        ]

    @staticmethod
    def _VectorDot(a: List[float], b: List[float]) -> float:
        """
        Returns the dot product of two vectors.

        Parameters
        ----------
        a : List[float]
            The input a value.
        b : List[float]
            The input b value.

        Returns
        -------
        float
            The resulting vector dot value.
        """
        return float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

    @staticmethod
    def _VectorNormalised(vector: Optional[List[float]], default: Optional[List[float]] = None) -> List[float]:
        """
        Returns a normalized vector.

        Parameters
        ----------
        vector : Optional[List[float]]
            The input vector value.
        default : Optional[List[float]] , optional
            The default value to return when no valid value is found. Default is None.

        Returns
        -------
        List[float]
            The resulting vector normalised list.
        """
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
    def Vertex(graph: "TGraph", index: int, asTopologic: bool = False) -> Optional[Any]:
        """
        Returns a vertex record or Topologic vertex from the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        index : int
            The input index.
        asTopologic : bool , optional
            If set to True, records are returned as Topologic objects when possible. Default is
            False.

        Returns
        -------
        dict or topologic_core.Vertex or None
            The requested vertex record or Topologic vertex.
        """
        if not isinstance(graph, TGraph) or not graph._validate_vertex_index(index):
            return None
        if not asTopologic:
            v = graph._vertices[index]
            return dict(v, dictionary=dict(v.get("dictionary", {})))
        verts = TGraph.Vertices(graph, asTopologic=True, activeOnly=False)
        return verts[index] if index < len(verts) else None

    @staticmethod
    def VertexByKeyValue(graph: "TGraph", key: str = None, value: Any = None, silent: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns the first active vertex whose dictionary contains the requested key-value pair.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str , optional
            The dictionary key to use. Default is None.
        value : Any , optional
            The input value value. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Dict[str, Any]]
            The resulting vertex by key value dictionary.
        """
        if not isinstance(graph, TGraph) or key is None:
            return None
        for v in graph._vertices:
            if v.get("active", True) and v.get("dictionary", {}).get(key) == value:
                return TGraph.Vertex(graph, v["index"])
        return None

    @staticmethod
    def _VertexCoordinates(vertex: Any) -> Optional[List[float]]:
        """
        Returns the coordinates of a Topologic vertex.

        Parameters
        ----------
        vertex : Any
            The input vertex, vertex index, or vertex record.

        Returns
        -------
        Optional[List[float]]
            The resulting vertex coordinates list.
        """
        try:
            from topologicpy.Vertex import Vertex
            coords = Vertex.Coordinates(vertex)
            if coords and len(coords) >= 3:
                return [float(coords[0]), float(coords[1]), float(coords[2])]
        except Exception:
            return None
        return None

    @staticmethod
    def VertexDegree(graph: "TGraph", vertex: Any, mode: str = "all", silent: bool = False) -> Optional[int]:
        """
        Returns the degree of the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Any
            The input vertex, vertex index, or vertex record.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[int]
            The resulting vertex degree index or count.
        """
        if not isinstance(graph, TGraph):
            return None
        idx = TGraph.VertexIndex(graph, vertex)
        if idx is None:
            return None
        return TGraph.Degree(graph, idx, mode=mode)

    def VertexDictionary(self, index: int) -> Dict[str, Any]:
        """
        Returns the dictionary of the input vertex index.

        Parameters
        ----------
        index : int
            The input index.

        Returns
        -------
        Dict[str, Any]
            The resulting vertex dictionary dictionary.
        """
        if not self._validate_vertex_index(index, active=False):
            return {}
        return dict(self._vertices[index].get("dictionary", {}))

    @staticmethod
    def VertexDictionaryStatic(graph: "TGraph", index: int) -> Dict[str, Any]:
        """
        Returns the dictionary of a vertex in the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        index : int
            The input index.

        Returns
        -------
        Dict[str, Any]
            The resulting vertex dictionary static dictionary.
        """
        return graph.VertexDictionary(index) if isinstance(graph, TGraph) else {}

    @staticmethod
    def VertexIndex(graph: "TGraph", vertex: Union[int, Dict[str, Any]]) -> Optional[int]:
        """
        Returns the index of the input vertex.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertex : Union[int, Dict[str, Any]]
            The input vertex, vertex index, or vertex record.

        Returns
        -------
        Optional[int]
            The resulting vertex index index or count.
        """
        idx = TGraph._as_index(vertex)
        return idx if isinstance(graph, TGraph) and graph._validate_vertex_index(idx) else None

    @staticmethod
    def Vertices(graph: "TGraph", asTopologic: bool = False, useRepresentation: bool = True,
                 activeOnly: bool = True, mantissa: int = 6, tolerance: float = 0.0001,
                 silent: bool = False) -> List[Any]:
        """
        Returns the vertices of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        asTopologic : bool , optional
            If set to True, records are returned as Topologic objects when possible. Default is
            False.
        useRepresentation : bool , optional
            If set to True, stored representation objects are used when possible. Default is
            True.
        activeOnly : bool , optional
            If set to True, only active records are considered. Default is True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The requested vertex records or Topologic vertices.
        """
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
    def VisibilityGraph(
        face: Any,
        vertices: Optional[List[Any]] = None,
        obstacles: Optional[List[Any]] = None,
        bidirectional: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["TGraph"]:
        """
        Returns a visibility graph derived from the input topology.

        Parameters
        ----------
        face : Any
            The input face value.
        vertices : Optional[List[Any]] , optional
            The input vertices or vertex indices. Default is None.
        obstacles : Optional[List[Any]] , optional
            The input obstacles value. Default is None.
        bidirectional : bool , optional
            The input bidirectional value. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
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
        return TGraph._OntologyAnnotateGraph(
            g, graphClass="top:VisibilityGraph", vertexClass="top:Node", edgeClass="top:Relationship",
            generatedBy="TGraph.VisibilityGraph", ontology=True, silent=silent)

    @staticmethod
    def WarmUpAcceleration(graph: "TGraph", mode: str = "all", useNumba: bool = True) -> Dict[str, Any]:
        """
        Warms up optional acceleration paths used by TGraph algorithms.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        mode : str , optional
            The traversal or adjacency mode. Valid values are typically "out", "in", or "all".
            Default is 'all'.
        useNumba : bool , optional
            If set to True, Numba acceleration is used when available. Default is True.

        Returns
        -------
        Dict[str, Any]
            The resulting warm up acceleration dictionary.
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

    @staticmethod
    def WeightedJaccardSimilarity(graphA: "TGraph", graphB: "TGraph" = None,
                                  vertexA: Any = None, vertexB: Any = None,
                                  vertexIDKey: str = "id", edgeWeightKey: str = None,
                                  weightKey: str = None, key: str = None,
                                  mantissa: int = 6, silent: bool = False) -> Optional[float]:
        """
        Returns a weighted Jaccard similarity score between two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph' , optional
            The second input TGraph. Default is None.
        vertexA : Any , optional
            The first input vertex or vertex index. Default is None.
        vertexB : Any , optional
            The second input vertex or vertex index. Default is None.
        vertexIDKey : str , optional
            The dictionary key to use. Default is 'id'.
        edgeWeightKey : str , optional
            The dictionary key to use. Default is None.
        weightKey : str , optional
            The edge dictionary key to use as a weight. Default is None.
        key : str , optional
            The dictionary key to use. Default is None.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[float]
            The resulting weighted jaccard similarity value.
        """
        if edgeWeightKey is None:
            edgeWeightKey = weightKey if weightKey is not None else key
        if graphB is None:
            graphB = graphA
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            if not silent:
                print("TGraph.WeightedJaccardSimilarity - Error: One or both graph inputs are not valid TGraphs. Returning None.")
            return None

        if vertexA is None and vertexB is None:
            weightsA = TGraph._P81GraphEdgeWeights(graphA, edgeWeightKey=edgeWeightKey)
            weightsB = TGraph._P81GraphEdgeWeights(graphB, edgeWeightKey=edgeWeightKey)
            keys = set(weightsA) | set(weightsB)
            if not keys:
                return 1.0
            numerator = sum(min(weightsA.get(k, 0.0), weightsB.get(k, 0.0)) for k in keys)
            denominator = sum(max(weightsA.get(k, 0.0), weightsB.get(k, 0.0)) for k in keys)
            return round(numerator / denominator, mantissa) if denominator > 0 else 0.0

        if vertexA is None or vertexB is None:
            return None

        ia = TGraph._as_index(vertexA)
        ib = TGraph._as_index(vertexB)
        if ia is None:
            ia = TGraph.VertexIndex(graphA, vertexIDKey, vertexA)
        if ib is None:
            ib = TGraph.VertexIndex(graphB, vertexIDKey, vertexB)
        if not graphA._validate_vertex_index(ia) or not graphB._validate_vertex_index(ib):
            return None

        def incident_weights(graph, vertex):
            weights = {}
            for eid in graph._incident_edges.get(vertex, set()):
                if not graph._validate_edge_index(eid):
                    continue
                e = graph._edges[eid]
                other = e.get("dst") if e.get("src") == vertex else e.get("src")
                if edgeWeightKey is None:
                    weight = 1.0
                else:
                    try:
                        weight = float(e.get("dictionary", {}).get(edgeWeightKey, 1.0))
                    except Exception:
                        weight = 1.0
                weights[other] = weights.get(other, 0.0) + weight
            keys = set(weights)
            return weights, keys

        weightsA, keysA = incident_weights(graphA, ia)
        weightsB, keysB = incident_weights(graphB, ib)
        keys = keysA | keysB
        if not keys:
            return 0.0
        numerator = sum(min(weightsA.get(k, 0.0), weightsB.get(k, 0.0)) for k in keys)
        denominator = sum(max(weightsA.get(k, 0.0), weightsB.get(k, 0.0)) for k in keys)
        return round(numerator / denominator, mantissa) if denominator > 0 else 0.0

    @staticmethod
    def WikiString(graph: "TGraph", vertexKey: str = "id", vertexLabelKey: str = "label", vertexTypeKey: str = "type", edgeKey: str = "predicate", titleKey: str = None, includeDictionaries: bool = True, includeBacklinks: bool = True, tolerance: float = 0.0001, silent: bool = False) -> str:
        """
        Returns a Wiki-compatible string representation of the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        vertexKey : str , optional
            The vertex dictionary key to use. Default is 'id'.
        vertexLabelKey : str , optional
            The dictionary key to use. Default is 'label'.
        vertexTypeKey : str , optional
            The dictionary key to use. Default is 'type'.
        edgeKey : str , optional
            The edge dictionary key to use. Default is 'predicate'.
        titleKey : str , optional
            The dictionary key to use. Default is None.
        includeDictionaries : bool , optional
            If set to True, dictionaries are included in the output. Default is True.
        includeBacklinks : bool , optional
            If set to True, include backlinks are included. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The resulting wiki string string.
        """
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
    def WireByPath(graph: "TGraph",
                   path: List[Union[int, Dict[str, Any]]],
                   transferVertexDictionaries: bool = True,
                   transferEdgeDictionaries: bool = True,
                   useRepresentations: bool = True,
                   mantissa: int = 6,
                   tolerance: float = 0.0001,
                   silent: bool = False):
        """
        Converts a path of vertex indices to a Topologic wire.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        path : List[Union[int, Dict[str, Any]]]
            The ordered path of vertex indices or vertex records to convert to a wire.
        transferVertexDictionaries : bool , optional
            If set to True, vertex dictionaries are transferred to created Topologic vertices.
            Default is True.
        transferEdgeDictionaries : bool , optional
            If set to True, edge dictionaries are transferred to created Topologic edges.
            Default is True.
        useRepresentations : bool , optional
            If set to True, stored representation objects are used when possible. Default is
            True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire or None
            The wire created from the input path.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.WireByPath - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        if not isinstance(path, list) or len(path) < 2:
            if not silent:
                print("TGraph.WireByPath - Error: The input path must be a list with at least two vertices. Returning None.")
            return None

        try:
            from topologicpy.Edge import Edge
            from topologicpy.Wire import Wire
        except Exception:
            if not silent:
                print("TGraph.WireByPath - Error: Could not import Edge or Wire. Returning None.")
            return None

        edges = []

        for i in range(len(path) - 1):
            src = TGraph.VertexIndex(graph, path[i])
            dst = TGraph.VertexIndex(graph, path[i + 1])

            if src is None or dst is None:
                if not silent:
                    print("TGraph.WireByPath - Error: Could not resolve one of the path vertices. Returning None.")
                return None

            edge_record = TGraph.EdgeBetween(graph, src, dst, directed=graph._directed)

            if edge_record is not None:
                e = TGraph.TopologicEdge(
                    graph,
                    edge_record,
                    transferVertexDictionaries=transferVertexDictionaries,
                    transferEdgeDictionary=transferEdgeDictionaries,
                    useRepresentation=useRepresentations,
                    mantissa=mantissa,
                    tolerance=tolerance,
                    silent=silent,
                )
            else:
                sv = TGraph.TopologicVertex(
                    graph,
                    src,
                    transferDictionary=transferVertexDictionaries,
                    useRepresentation=useRepresentations,
                    mantissa=mantissa,
                    silent=silent,
                )

                tv = TGraph.TopologicVertex(
                    graph,
                    dst,
                    transferDictionary=transferVertexDictionaries,
                    useRepresentation=useRepresentations,
                    mantissa=mantissa,
                    silent=silent,
                )

                if sv is None or tv is None:
                    if not silent:
                        print("TGraph.WireByPath - Error: Could not create fallback edge vertices. Returning None.")
                    return None

                try:
                    e = Edge.ByStartVertexEndVertex(sv, tv, tolerance=tolerance)
                except Exception:
                    try:
                        e = Edge.ByVertices([sv, tv])
                    except Exception:
                        if not silent:
                            print("TGraph.WireByPath - Error: Could not create fallback edge. Returning None.")
                        return None

            if e is None:
                if not silent:
                    print("TGraph.WireByPath - Error: Could not create one of the path edges. Returning None.")
                return None

            edges.append(e)

        try:
            return Wire.ByEdges(edges, tolerance=tolerance)
        except Exception:
            try:
                return Wire.ByEdges(edges)
            except Exception:
                if not silent:
                    print("TGraph.WireByPath - Error: Could not create wire from path edges. Returning None.")
                return None

    @staticmethod
    def WLFeatures(graph: "TGraph", key: str = None, iterations: int = 2, labelKey: str = None,
                   silent: bool = False) -> Optional[Dict[Any, int]]:
        """
        Returns Weisfeiler-Lehman feature counts for the input TGraph.

        Parameters
        ----------
        graph : 'TGraph'
            The input TGraph.
        key : str , optional
            The dictionary key to use. Default is None.
        iterations : int , optional
            The input iterations value. Default is 2.
        labelKey : str , optional
            The dictionary key to use. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[Dict[Any, int]]
            The resulting wlfeatures dictionary.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.WLFeatures - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        label_key = labelKey if labelKey is not None else key
        vertices = [i for i, v in enumerate(graph._vertices) if v.get("active", True)]
        if not vertices:
            return {}

        vertex_set = set(vertices)
        adj = {v: [] for v in vertices}
        for e in graph._edges:
            if not e.get("active", True):
                continue
            src = e.get("src")
            dst = e.get("dst")
            if src not in vertex_set or dst not in vertex_set:
                continue
            adj[src].append(dst)
            adj[dst].append(src)

        labels = {}
        for v in vertices:
            d = graph._vertices[v].get("dictionary", {})
            if not isinstance(d, dict):
                d = {}
            if label_key is not None and label_key in d:
                labels[v] = d.get(label_key)
            else:
                labels[v] = len(adj[v])

        features = {}
        for label in labels.values():
            features[label] = features.get(label, 0) + 1

        iterations = max(0, int(iterations or 0))
        for _ in range(iterations):
            new_labels = {}
            for v in vertices:
                neighbour_labels = tuple(sorted(labels[n] for n in adj[v] if n in labels))
                new_labels[v] = (labels[v], neighbour_labels)
            labels = new_labels
            for label in labels.values():
                features[label] = features.get(label, 0) + 1

        return features

    @staticmethod
    def WLKernel(graphA: "TGraph", graphB: "TGraph", key: str = None, iterations: int = 2,
                 labelKey: str = None, normalize: bool = True, mantissa: int = 6,
                 silent: bool = False) -> Optional[float]:
        """
        Returns a Weisfeiler-Lehman kernel similarity score between two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        key : str , optional
            The dictionary key to use. Default is None.
        iterations : int , optional
            The input iterations value. Default is 2.
        labelKey : str , optional
            The dictionary key to use. Default is None.
        normalize : bool , optional
            If set to True, returned values are normalized. Default is True.
        mantissa : int , optional
            The number of decimal places to round numeric results to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[float]
            The resulting wlkernel value.
        """
        if not isinstance(graphA, TGraph) or not isinstance(graphB, TGraph):
            if not silent:
                print("TGraph.WLKernel - Error: One or both inputs are not valid TGraphs. Returning None.")
            return None
        label_key = labelKey if labelKey is not None else key
        fA = TGraph.WLFeatures(graphA, key=label_key, iterations=iterations, silent=silent) or {}
        fB = TGraph.WLFeatures(graphB, key=label_key, iterations=iterations, silent=silent) or {}
        return round(TGraph._P8Cosine(fA, fB, normalize=normalize), mantissa)

    @staticmethod
    def XOR(graphA: "TGraph", graphB: "TGraph", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the symmetric difference of two input TGraphs.

        Parameters
        ----------
        graphA : 'TGraph'
            The first input TGraph.
        graphB : 'TGraph'
            The second input TGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Optional[TGraph]
            The resulting TGraph, or None if the operation fails.
        """
        return TGraph.SymmetricDifference(graphA, graphB, silent=silent)


    @staticmethod
    def IsCompiled(graph: "TGraph", weightKey: str = None) -> bool:
        """
        Returns True if the input TGraph has a valid compiled cache.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        weightKey : str , optional
            If specified, the compiled cache must also match this edge weight key.
            If set to None, only the graph version is checked. Default is None.

        Returns
        -------
        bool
            True if the graph has a valid compiled cache; otherwise False.
        """
        if not isinstance(graph, TGraph):
            return False
        c = graph._compiled
        if not isinstance(c, dict):
            return False
        if c.get("version", None) != graph._version:
            return False
        if weightKey is not None and c.get("weightKey", None) != weightKey:
            return False
        return True

    @staticmethod
    def ClearCompiled(graph: "TGraph") -> Optional["TGraph"]:
        """
        Clears the compiled cache of the input TGraph.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.

        Returns
        -------
        TGraph or None
            The input TGraph with its compiled cache cleared, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        graph._compiled = None
        return graph

    @staticmethod
    def EnsureCompiled(graph: "TGraph", weightKey: str = "weight", force: bool = False,
                       useNumpy: bool = True, useSciPy: bool = True,
                       useNumba: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns a valid compiled cache for the input TGraph, compiling it if needed.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        weightKey : str , optional
            The edge dictionary key to use as a weight. Default is "weight".
        force : bool , optional
            If set to True, the cache is rebuilt even if it appears valid. Default is False.
        useNumpy : bool , optional
            If set to True, NumPy acceleration arrays are created when available. Default is True.
        useSciPy : bool , optional
            If set to True, SciPy sparse matrices are created when available. Default is True.
        useNumba : bool , optional
            If set to True, the compiled cache records that Numba acceleration was requested.
            Default is False.

        Returns
        -------
        dict or None
            The compiled cache dictionary, or None if the input is invalid.
        """
        return TGraph.Compile(graph, weightKey=weightKey, force=force,
                              useNumpy=useNumpy, useSciPy=useSciPy,
                              useNumba=useNumba)

    @staticmethod
    def CompileInfo(graph: "TGraph") -> Dict[str, Any]:
        """
        Returns a compact diagnostic report about the compiled cache.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.

        Returns
        -------
        dict
            A dictionary reporting cache validity, version, size, and optional acceleration state.
        """
        if not isinstance(graph, TGraph):
            return {"valid": False}
        c = graph._compiled
        valid = TGraph.IsCompiled(graph)
        if not isinstance(c, dict):
            return {"valid": False, "version": graph._version, "compiled": False}
        return {
            "valid": valid,
            "compiled": True,
            "graph_version": graph._version,
            "compiled_version": c.get("version", None),
            "weightKey": c.get("weightKey", None),
            "order": c.get("n", 0),
            "size": len(c.get("edges", [])),
            "numpy_available": bool(c.get("numpy_available", False)),
            "scipy_available": bool(c.get("scipy_available", False)),
            "numba_requested": bool(c.get("numba_requested", False)),
        }

    @staticmethod
    def Guid(graph: "TGraph") -> Optional[str]:
        """
        Returns a persistent GUID for the input TGraph, creating one if needed.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.

        Returns
        -------
        str or None
            The graph GUID, or None if the input is invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        import uuid
        value = graph._dictionary.get("guid", None)
        if value in [None, ""]:
            value = str(uuid.uuid4())
            graph._dictionary["guid"] = value
        return value

    @staticmethod
    def CardinalityReport(graph: "TGraph", vertexKey: str = "id", edgeKey: str = "predicate",
                          predicates: list = None, direction: str = "both",
                          includeZero: bool = True, tolerance: float = 0.0001,
                          silent: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Returns a cardinality report for vertices in a TGraph.

        This method counts how many incident edges of each selected predicate are
        connected to each active vertex.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        vertexKey : str , optional
            Vertex dictionary key used to identify each vertex. Default is "id".
        edgeKey : str , optional
            Edge dictionary key used to identify the relationship/predicate. Default is
            "predicate".
        predicates : list , optional
            If provided, only edges whose predicate is in this list are counted.
            Matching is case-insensitive. Default is None.
        direction : str , optional
            Edge direction to count. Valid values are "in", "out", and "both".
            Default is "both".
        includeZero : bool , optional
            If set to True, include vertices with zero matching edges. Default is True.
        tolerance : float , optional
            Included for API compatibility. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            A list of dictionaries, one row per reported vertex.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.CardinalityReport - Error: The input graph is not a valid TGraph. Returning None.")
            return None
        direction = str(direction or "both").lower()
        if direction not in ["in", "out", "both", "all"]:
            if not silent:
                print("TGraph.CardinalityReport - Error: direction must be 'in', 'out', or 'both'. Returning None.")
            return None
        if direction == "all":
            direction = "both"
        if predicates is None:
            predicate_set = None
        else:
            if not isinstance(predicates, (list, tuple, set)):
                predicates = [predicates]
            predicate_set = {str(p).strip().lower() for p in predicates if p not in [None, ""]}
        rows = []
        for v in graph._vertices:
            if not v.get("active", True):
                continue
            idx = v.get("index")
            vd = v.get("dictionary", {}) if isinstance(v.get("dictionary", {}), dict) else {}
            counts = {}
            edge_ids = set()
            if direction in ["out", "both"]:
                edge_ids |= set(graph._out_edges.get(idx, set()))
            if direction in ["in", "both"]:
                edge_ids |= set(graph._in_edges.get(idx, set()))
            for eid in edge_ids:
                if not graph._validate_edge_index(eid):
                    continue
                e = graph._edges[eid]
                if direction == "out" and e.get("src") != idx and bool(e.get("directed", graph._directed)):
                    continue
                if direction == "in" and e.get("dst") != idx and bool(e.get("directed", graph._directed)):
                    continue
                ed = e.get("dictionary", {}) if isinstance(e.get("dictionary", {}), dict) else {}
                pred = ed.get(edgeKey, ed.get("predicate", ed.get("relationship", ed.get("label", ""))))
                pred_key = str(pred).strip()
                pred_l = pred_key.lower()
                if predicate_set is not None and pred_l not in predicate_set:
                    continue
                counts[pred_key] = counts.get(pred_key, 0) + 1
            total = sum(counts.values())
            if total > 0 or includeZero:
                row = {
                    "vertex_index": idx,
                    "vertex": vd.get(vertexKey, vd.get("label", idx)),
                    "total": total,
                }
                row.update(counts)
                rows.append(row)
        return rows

    @staticmethod
    def AdjacentVerticesByVector(graph: "TGraph", vertex: Any, vector: list = [0, 0, 1],
                                 tolerance: float = 0.0001, silent: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Returns adjacent vertices that lie in the input vector direction from the input vertex.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        vertex : int or dict
            The input vertex, vertex index, or vertex record.
        vector : list , optional
            The target vector direction. Default is [0, 0, 1].
        tolerance : float , optional
            Angular comparison tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The list of adjacent vertex records in the requested direction.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.AdjacentVerticesByVector - Error: The input graph is not a valid TGraph. Returning None.")
            return None
        idx = TGraph.VertexIndex(graph, vertex)
        if not graph._validate_vertex_index(idx):
            if not silent:
                print("TGraph.AdjacentVerticesByVector - Error: The input vertex is not valid. Returning None.")
            return None
        try:
            vx, vy, vz = float(vector[0]), float(vector[1]), float(vector[2])
            vlen = math.sqrt(vx*vx + vy*vy + vz*vz)
            if vlen <= 0:
                return []
            vx, vy, vz = vx/vlen, vy/vlen, vz/vlen
        except Exception:
            if not silent:
                print("TGraph.AdjacentVerticesByVector - Error: The input vector is not valid. Returning None.")
            return None
        c0 = TGraph.Coordinates(graph, idx, default=None)
        if c0 is None:
            return []
        eps = max(float(tolerance or 0.0), 1e-9)
        result = []
        for nb in TGraph.AdjacentIndices(graph, idx, mode="all"):
            c1 = TGraph.Coordinates(graph, nb, default=None)
            if c1 is None:
                continue
            dx, dy, dz = float(c1[0])-float(c0[0]), float(c1[1])-float(c0[1]), float(c1[2])-float(c0[2])
            dlen = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dlen <= eps:
                continue
            dx, dy, dz = dx/dlen, dy/dlen, dz/dlen
            dot = dx*vx + dy*vy + dz*vz
            if dot >= 1.0 - eps:
                result.append(TGraph.Vertex(graph, nb))
        return result

    @staticmethod
    def AdjacentVerticesByCompassDirection(graph: "TGraph", vertex: Any,
                                           compassDirection: str = "Up",
                                           tolerance: float = 0.0001,
                                           silent: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Returns adjacent vertices that lie in the requested compass direction from the input vertex.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        vertex : int or dict
            The input vertex, vertex index, or vertex record.
        compassDirection : str , optional
            The compass direction. Common values include "Up", "Down", "North",
            "South", "East", and "West". Default is "Up".
        tolerance : float , optional
            The direction tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list or None
            The list of adjacent vertex records in the requested compass direction.
        """
        if not isinstance(compassDirection, str):
            if not silent:
                print("TGraph.AdjacentVerticesByCompassDirection - Error: compassDirection must be a string. Returning None.")
            return None
        direction = compassDirection.strip().lower()
        mapping = {
            "up": [0, 0, 1], "u": [0, 0, 1], "+z": [0, 0, 1],
            "down": [0, 0, -1], "d": [0, 0, -1], "-z": [0, 0, -1],
            "north": [0, 1, 0], "n": [0, 1, 0], "+y": [0, 1, 0],
            "south": [0, -1, 0], "s": [0, -1, 0], "-y": [0, -1, 0],
            "east": [1, 0, 0], "e": [1, 0, 0], "+x": [1, 0, 0],
            "west": [-1, 0, 0], "w": [-1, 0, 0], "-x": [-1, 0, 0],
            "northeast": [1, 1, 0], "ne": [1, 1, 0],
            "northwest": [-1, 1, 0], "nw": [-1, 1, 0],
            "southeast": [1, -1, 0], "se": [1, -1, 0],
            "southwest": [-1, -1, 0], "sw": [-1, -1, 0],
        }
        if direction not in mapping:
            try:
                from topologicpy.Vector import Vector
                all_dirs = [d.lower() for d in Vector.CompassDirections()]
                if direction not in all_dirs:
                    if not silent:
                        print("TGraph.AdjacentVerticesByCompassDirection - Error: Invalid compass direction. Returning None.")
                    return None
            except Exception:
                if not silent:
                    print("TGraph.AdjacentVerticesByCompassDirection - Error: Invalid compass direction. Returning None.")
                return None
        return TGraph.AdjacentVerticesByVector(graph, vertex, mapping.get(direction, [0, 0, 1]), tolerance=tolerance, silent=silent)

    @staticmethod
    def Connect(graph: "TGraph", verticesA, verticesB, tolerance: float = 0.0001) -> Optional["TGraph"]:
        """
        Connects every vertex in verticesA to every vertex in verticesB with an edge.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        verticesA : list or single vertex
            The first vertex set.
        verticesB : list or single vertex
            The second vertex set.
        tolerance : float , optional
            Included for API compatibility. Default is 0.0001.

        Returns
        -------
        TGraph or None
            The modified input TGraph, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        if not isinstance(verticesA, (list, tuple, set)):
            verticesA = [verticesA]
        if not isinstance(verticesB, (list, tuple, set)):
            verticesB = [verticesB]
        a_indices = [TGraph.VertexIndex(graph, v) for v in verticesA]
        b_indices = [TGraph.VertexIndex(graph, v) for v in verticesB]
        for a in a_indices:
            if not graph._validate_vertex_index(a):
                continue
            for b in b_indices:
                if not graph._validate_vertex_index(b) or a == b:
                    continue
                if TGraph.EdgeBetween(graph, a, b) is None:
                    graph.AddEdge(a, b, silent=True)
        return graph

    @staticmethod
    def DetachVertex(graph: "TGraph", *vertices, silent: bool = False) -> Optional["TGraph"]:
        """
        Removes all incident edges from the specified vertices while keeping the vertices active.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        *vertices : int, dict, or list
            Vertices, vertex indices, or vertex records to detach.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The modified input TGraph, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            if not silent:
                print("TGraph.DetachVertex - Error: The input graph is not a valid TGraph. Returning None.")
            return None
        items = []
        for item in vertices:
            if isinstance(item, (list, tuple, set)):
                items.extend(list(item))
            else:
                items.append(item)
        for item in items:
            idx = TGraph.VertexIndex(graph, item)
            if not graph._validate_vertex_index(idx):
                continue
            for eid in list(graph._incident_edges.get(idx, set())):
                if graph._validate_edge_index(eid):
                    graph.RemoveEdge(eid, silent=True)
        graph._invalidate_cache()
        return graph

    @staticmethod
    def ContractEdge(graph: "TGraph", edge: Any, vertex: Any = None,
                     tolerance: float = 0.0001, silent: bool = False) -> Optional["TGraph"]:
        """
        Contracts an edge by merging its endpoints into one replacement vertex.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        edge : int or dict
            The edge, edge index, or edge record to contract.
        vertex : Any , optional
            Optional replacement vertex or vertex index. If omitted, a new midpoint
            vertex is created. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The modified input TGraph, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        eid = TGraph.EdgeIndex(graph, edge)
        if not graph._validate_edge_index(eid):
            return graph
        e = graph._edges[eid]
        a, b = e.get("src"), e.get("dst")
        if not graph._validate_vertex_index(a) or not graph._validate_vertex_index(b):
            return graph
        if vertex is None:
            ca = TGraph.Coordinates(graph, a, default=None)
            cb = TGraph.Coordinates(graph, b, default=None)
            d = {}
            d.update(graph._vertices[a].get("dictionary", {}))
            d.update(graph._vertices[b].get("dictionary", {}))
            d.update(e.get("dictionary", {}))
            if ca is not None and cb is not None:
                d["x"] = (float(ca[0]) + float(cb[0])) / 2.0
                d["y"] = (float(ca[1]) + float(cb[1])) / 2.0
                d["z"] = (float(ca[2]) + float(cb[2])) / 2.0
            target = graph.AddVertex(dictionary=d)
        else:
            target = TGraph.VertexIndex(graph, vertex)
            if target is None:
                target = graph.AddVertex(dictionary=vertex)
        old_vertices = {a, b}
        incident = sorted(set(graph._incident_edges.get(a, set())) | set(graph._incident_edges.get(b, set())))
        for old_eid in incident:
            if not graph._validate_edge_index(old_eid):
                continue
            if old_eid == eid:
                continue
            old_e = graph._edges[old_eid]
            src, dst = old_e.get("src"), old_e.get("dst")
            if src in old_vertices and dst in old_vertices:
                continue
            new_src = target if src in old_vertices else src
            new_dst = target if dst in old_vertices else dst
            if new_src == new_dst and not graph._allow_self_loops:
                continue
            graph.AddEdge(new_src, new_dst, directed=old_e.get("directed", graph._directed),
                          dictionary=dict(old_e.get("dictionary", {})),
                          representation=old_e.get("representation"), silent=True)
        graph.RemoveVertex(a, silent=True)
        graph.RemoveVertex(b, silent=True)
        graph._invalidate_cache()
        return graph

    @staticmethod
    def MergeVertices(graph: "TGraph", *vertices, targetVertex=None,
                      transferDictionaries: bool = True, tolerance: float = 0.0001,
                      silent: bool = False) -> Optional["TGraph"]:
        """
        Merges several vertices into one target vertex and reconnects incident edges.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        *vertices : int, dict, or list
            Vertices to merge.
        targetVertex : int or dict , optional
            Optional target vertex. If omitted, the first valid input vertex is used.
            Default is None.
        transferDictionaries : bool , optional
            If set to True, missing target dictionary values are filled from merged
            vertices. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The modified input TGraph, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        items = []
        for item in vertices:
            if isinstance(item, (list, tuple, set)):
                items.extend(list(item))
            else:
                items.append(item)
        ids = []
        for item in items:
            idx = TGraph.VertexIndex(graph, item)
            if graph._validate_vertex_index(idx) and idx not in ids:
                ids.append(idx)
        if not ids:
            return graph
        target = TGraph.VertexIndex(graph, targetVertex) if targetVertex is not None else ids[0]
        if not graph._validate_vertex_index(target):
            target = ids[0]
        merge_set = set(ids)
        target_dict = graph._vertices[target].setdefault("dictionary", {})
        if transferDictionaries:
            for idx in ids:
                for k, v in graph._vertices[idx].get("dictionary", {}).items():
                    target_dict.setdefault(k, v)
        incident = set()
        for idx in ids:
            incident |= set(graph._incident_edges.get(idx, set()))
        for eid in sorted(incident):
            if not graph._validate_edge_index(eid):
                continue
            e = graph._edges[eid]
            src, dst = e.get("src"), e.get("dst")
            new_src = target if src in merge_set else src
            new_dst = target if dst in merge_set else dst
            if src in merge_set and dst in merge_set:
                continue
            if new_src == new_dst and not graph._allow_self_loops:
                continue
            graph.AddEdge(new_src, new_dst, directed=e.get("directed", graph._directed),
                          dictionary=dict(e.get("dictionary", {})), representation=e.get("representation"), silent=True)
        for idx in ids:
            if idx != target:
                graph.RemoveVertex(idx, silent=True)
        graph._invalidate_cache()
        return graph

    @staticmethod
    def KHopsSubgraph(graph: "TGraph", vertices: list, k: int = 1,
                      direction: str = "both", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns the induced subgraph of vertices within k hops of the input vertices.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        vertices : list
            Starting vertices, vertex indices, or vertex records.
        k : int , optional
            Maximum hop distance from the starting vertices. Default is 1.
        direction : str , optional
            Traversal direction: "in", "out", or "both". Default is "both".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The resulting induced TGraph subgraph.
        """
        if not isinstance(graph, TGraph):
            return None
        if vertices is None:
            vertices = []
        if not isinstance(vertices, (list, tuple, set)):
            vertices = [vertices]
        starts = [TGraph.VertexIndex(graph, v) for v in vertices]
        starts = [v for v in starts if graph._validate_vertex_index(v)]
        if not starts:
            return TGraph.Subgraph(graph, [], induced=True)
        try:
            k = max(0, int(k))
        except Exception:
            k = 1
        mode = str(direction or "both").lower()
        if mode == "both":
            mode = "all"
        if mode not in ["in", "out", "all"]:
            mode = "all"
        visited = set(starts)
        frontier = set(starts)
        for _ in range(k):
            nxt = set()
            for v in frontier:
                nxt.update(TGraph.AdjacentIndices(graph, v, mode=mode))
            nxt = {v for v in nxt if graph._validate_vertex_index(v)} - visited
            if not nxt:
                break
            visited |= nxt
            frontier = nxt
        return TGraph.Subgraph(graph, sorted(visited), induced=True)

    @staticmethod
    def Neigborhood(graph: "TGraph", vertices: list = None, k: int = 1,
                    searchType: str = "equal to", key: str = None, value: Any = None,
                    direction: str = "both", silent: bool = False) -> Optional["TGraph"]:
        """
        Returns a k-hop neighbourhood subgraph.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        vertices : list , optional
            Seed vertices. If None, seed vertices are selected using key/value or all
            active vertices. Default is None.
        k : int , optional
            Hop distance. Default is 1.
        searchType : str , optional
            One of "equal to", "less than", "greater than", "contains", or "not equal to".
            Used only when key is specified. Default is "equal to".
        key : str , optional
            Vertex dictionary key used to select seed vertices. Default is None.
        value : Any , optional
            Value used with key/searchType. Default is None.
        direction : str , optional
            Traversal direction: "in", "out", or "both". Default is "both".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The resulting neighbourhood subgraph.
        """
        if not isinstance(graph, TGraph):
            return None
        def _match(x):
            st = str(searchType or "equal to").lower()
            if st in ["equal", "equal to", "=="]:
                return x == value
            if st in ["not equal", "not equal to", "!="]:
                return x != value
            if st in ["contains", "in"]:
                try:
                    return str(value) in str(x)
                except Exception:
                    return False
            try:
                xf = float(x); vf = float(value)
                if st in ["less", "less than", "<"]:
                    return xf < vf
                if st in ["less than or equal to", "<="]:
                    return xf <= vf
                if st in ["greater", "greater than", ">"]:
                    return xf > vf
                if st in ["greater than or equal to", ">="]:
                    return xf >= vf
            except Exception:
                return False
            return False
        if vertices is None:
            if key is None:
                vertices = TGraph.ActiveVertexIndices(graph)
            else:
                vertices = []
                for rec in graph._vertices:
                    if not rec.get("active", True):
                        continue
                    d = rec.get("dictionary", {}) if isinstance(rec.get("dictionary", {}), dict) else {}
                    if _match(d.get(key, None)):
                        vertices.append(rec.get("index"))
        return TGraph.KHopsSubgraph(graph, vertices, k=k, direction=direction, silent=silent)

    @staticmethod
    def Neighborhood(graph: "TGraph", vertices: list = None, k: int = 1,
                     searchType: str = "equal to", key: str = None, value: Any = None,
                     direction: str = "both", silent: bool = False) -> Optional["TGraph"]:
        """
        Correctly spelled alias for TGraph.Neigborhood.
        """
        return TGraph.Neigborhood(graph, vertices=vertices, k=k, searchType=searchType,
                                  key=key, value=value, direction=direction, silent=silent)

    @staticmethod
    def Partition(graph: "TGraph", method: str = "Betweenness", n: int = 2,
                  m: int = 10, key: str = "partition", mantissa: int = 6,
                  tolerance: float = 0.0001, silent: bool = False) -> Optional["TGraph"]:
        """
        Partitions the input graph and stores partition ids in dictionaries.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        method : str , optional
            Partition method: "Betweenness", "Community"/"Louvain", or
            "Fiedler"/"Eigen". Default is "Betweenness".
        n : int , optional
            Desired number of partitions for betweenness partitioning. Default is 2.
        m : int , optional
            Maximum number of tries for betweenness partitioning. Default is 10.
        key : str , optional
            Dictionary key under which to store partition ids. Default is "partition".
        mantissa : int , optional
            Number of decimal places for numeric calculations. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The input TGraph after annotation, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        ml = str(method or "").lower()
        if "between" in ml:
            TGraph.BetweennessPartition(graph, n=n, m=m, key=key, tolerance=tolerance, silent=silent)
        elif "community" in ml or "louvain" in ml:
            TGraph.CommunityPartition(graph, key=key, mantissa=mantissa, tolerance=tolerance, silent=silent)
        elif "fied" in ml or "eig" in ml:
            TGraph.FiedlerVectorPartition(graph, key=key, mantissa=mantissa, tolerance=tolerance, silent=silent)
        else:
            if not silent:
                print("TGraph.Partition - Error: The chosen method is not supported. Returning None.")
            return None
        return graph

    @staticmethod
    def PropagateValues(graph: "TGraph", sourceVertexKey: str = "id",
                        targetVertexKey: str = "id", edgeKey: str = "predicate",
                        predicates: list = None, sourceKeys: list = None,
                        targetKeys: list = None, direction: str = "out",
                        overwrite: bool = False, prefix: str = "", suffix: str = "",
                        tolerance: float = 0.0001, silent: bool = False) -> Optional["TGraph"]:
        """
        Propagates dictionary values from source vertices to target vertices along selected edges.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        sourceVertexKey : str , optional
            Vertex dictionary key used to identify source vertices. Default is "id".
        targetVertexKey : str , optional
            Vertex dictionary key used to identify target vertices. Default is "id".
        edgeKey : str , optional
            Edge dictionary key used to identify the relationship/predicate. Default is
            "predicate".
        predicates : list , optional
            If provided, values are propagated only along edges whose predicate is in
            this list. Matching is case-insensitive. Default is None.
        sourceKeys : list , optional
            Source dictionary keys to copy. If None, all non-reserved source keys are
            considered. Default is None.
        targetKeys : list , optional
            Target dictionary keys. If None, sourceKeys are used with prefix/suffix.
            Default is None.
        direction : str , optional
            "out", "in", or "both". Default is "out".
        overwrite : bool , optional
            If True, existing target values are overwritten. Default is False.
        prefix : str , optional
            Prefix added to target keys when targetKeys is None. Default is "".
        suffix : str , optional
            Suffix added to target keys when targetKeys is None. Default is "".
        tolerance : float , optional
            Included for API compatibility. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The input graph after propagation, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        direction = str(direction or "out").lower()
        if direction not in ["out", "in", "both"]:
            return None
        if predicates is None:
            predicate_set = None
        else:
            if not isinstance(predicates, (list, tuple, set)):
                predicates = [predicates]
            predicate_set = {str(p).strip().lower() for p in predicates if p not in [None, ""]}
        if sourceKeys is not None and not isinstance(sourceKeys, (list, tuple)):
            sourceKeys = [sourceKeys]
        if targetKeys is not None and not isinstance(targetKeys, (list, tuple)):
            targetKeys = [targetKeys]
        if sourceKeys is not None:
            sourceKeys = list(sourceKeys)
        if targetKeys is not None:
            targetKeys = list(targetKeys)
        if sourceKeys is not None and targetKeys is not None and len(sourceKeys) != len(targetKeys):
            return None
        reserved = {"index", "src", "dst", "active", "id", "key", "label", "type", "category"}
        def _is_empty(x):
            return x in [None, "", [], {}, ()]
        def _propagate(src_idx, dst_idx):
            if not graph._validate_vertex_index(src_idx) or not graph._validate_vertex_index(dst_idx):
                return
            sd = graph._vertices[src_idx].get("dictionary", {})
            td = graph._vertices[dst_idx].setdefault("dictionary", {})
            keys = sourceKeys if sourceKeys is not None else [k for k in sd.keys() if k not in reserved]
            if targetKeys is not None:
                pairs = zip(keys, targetKeys)
            else:
                pairs = [(k, f"{prefix}{k}{suffix}") for k in keys]
            for sk, tk in pairs:
                if sk not in sd:
                    continue
                if overwrite or _is_empty(td.get(tk, None)):
                    td[tk] = sd.get(sk)
        for e in graph._edges:
            if not e.get("active", True):
                continue
            ed = e.get("dictionary", {}) if isinstance(e.get("dictionary", {}), dict) else {}
            pred = ed.get(edgeKey, ed.get("predicate", ed.get("relationship", ed.get("label", ""))))
            if predicate_set is not None and str(pred).strip().lower() not in predicate_set:
                continue
            if direction in ["out", "both"]:
                _propagate(e.get("src"), e.get("dst"))
            if direction in ["in", "both"]:
                _propagate(e.get("dst"), e.get("src"))
        return graph

    @staticmethod
    def BOTGraph(graph: "TGraph", *args, **kwargs):
        """
        Returns an RDFLib graph containing the BOT-compatible TTL representation of the TGraph.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        *args, **kwargs
            Additional arguments are passed to TGraph.BOTString.

        Returns
        -------
        rdflib.Graph or None
            The BOT RDF graph, or None if RDFLib is unavailable or the graph is invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        try:
            from rdflib import Graph as RDFGraph
            ttl = TGraph.BOTString(graph, *args, **kwargs)
            if ttl is None:
                return None
            g = RDFGraph()
            g.parse(data=ttl, format="turtle")
            return g
        except Exception:
            return None

    @staticmethod
    def ByBOTGraph(botGraph, includeContext: bool = False, xMin: float = -0.5,
                   xMax: float = 0.5, yMin: float = -0.5, yMax: float = 0.5,
                   zMin: float = -0.5, zMax: float = 0.5, ontology: bool = True,
                   tolerance: float = 0.0001, silent: bool = False) -> Optional["TGraph"]:
        """
        Creates a TGraph from an RDFLib BOT graph or compatible RDF graph.

        Parameters
        ----------
        botGraph : rdflib.Graph
            The input RDF graph.
        includeContext : bool , optional
            Included for API compatibility. Default is False.
        xMin, xMax, yMin, yMax, zMin, zMax : float , optional
            Coordinate bounds used only when synthetic coordinates are needed.
        ontology : bool , optional
            If set to True, ontology metadata is added. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The created TGraph.
        """
        try:
            triples = list(botGraph.triples((None, None, None)))
        except Exception:
            if not silent:
                print("TGraph.ByBOTGraph - Error: The input is not a valid RDF graph. Returning None.")
            return None
        g = TGraph(directed=True, allowSelfLoops=True, allowParallelEdges=True)
        node_index = {}
        def _label(term):
            s = str(term)
            if "#" in s:
                return s.rsplit("#", 1)[-1]
            if "/" in s:
                return s.rstrip("/").rsplit("/", 1)[-1]
            return s
        def _ensure(term):
            if term not in node_index:
                i = len(node_index)
                x = xMin + (xMax-xMin) * ((i % 10) / 9.0 if 9 else 0.0)
                y = yMin + (yMax-yMin) * (((i // 10) % 10) / 9.0 if 9 else 0.0)
                z = zMin + (zMax-zMin) * (((i // 100) % 10) / 9.0 if 9 else 0.0)
                node_index[term] = g.AddVertex(dictionary={"uri": str(term), "label": _label(term), "x": x, "y": y, "z": z})
            return node_index[term]
        for s, p, o in triples:
            si = _ensure(s)
            oi = _ensure(o)
            g.AddEdge(si, oi, directed=True, dictionary={"uri": str(p), "label": _label(p), "predicate": _label(p), "relationship": _label(p)})
        return TGraph._OntologyAnnotateGraph(g, graphClass="top:BOTGraph", vertexClass="top:Node", edgeClass="top:Relationship", generatedBy="TGraph.ByBOTGraph", ontology=ontology, silent=True)

    @staticmethod
    def ByBOTPath(path, includeContext: bool = False, xMin: float = -0.5,
                  xMax: float = 0.5, yMin: float = -0.5, yMax: float = 0.5,
                  zMin: float = -0.5, zMax: float = 0.5, ontology: bool = True,
                  tolerance: float = 0.0001, silent: bool = False) -> Optional["TGraph"]:
        """
        Creates a TGraph from a BOT/RDF file path.

        Parameters
        ----------
        path : str
            Path to a Turtle, RDF/XML, JSON-LD, or N-Triples file.
        includeContext : bool , optional
            Included for API compatibility. Default is False.
        xMin, xMax, yMin, yMax, zMin, zMax : float , optional
            Coordinate bounds used only when synthetic coordinates are needed.
        ontology : bool , optional
            If set to True, ontology metadata is added. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The created TGraph.
        """
        try:
            from rdflib import Graph as RDFGraph
            rdf = RDFGraph()
            rdf.parse(path)
        except Exception as exc:
            if not silent:
                print(f"TGraph.ByBOTPath - Error: {exc}. Returning None.")
            return None
        return TGraph.ByBOTGraph(rdf, includeContext=includeContext, xMin=xMin, xMax=xMax,
                                 yMin=yMin, yMax=yMax, zMin=zMin, zMax=zMax,
                                 ontology=ontology, tolerance=tolerance, silent=silent)

    @staticmethod
    def HasseDiagram(topology, types=["vertex", "edge", "wire", "face", "shell", "cell", "cellComplex"],
                     topDown: bool = False, minDistance: float = 0.1,
                     vertexLabelKey: str = "label", vertexTypeKey: str = "type",
                     vertexColorKey: str = "color", colorScale: str = "viridis",
                     storeBREP: bool = False, tolerance: float = 0.0001,
                     silent: bool = False) -> Optional["TGraph"]:
        """
        Creates a Hasse diagram TGraph for the subtopologies of an input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        types : list , optional
            Subtopology types to include. Default is vertex, edge, wire, face, shell,
            cell, and cellComplex.
        topDown : bool , optional
            If set to True, edges point from higher-dimensional topologies to lower-
            dimensional topologies. Default is False.
        minDistance : float , optional
            Used as the vertical spacing between ranks in generated coordinates.
            Default is 0.1.
        vertexLabelKey : str , optional
            Dictionary key for node labels. Default is "label".
        vertexTypeKey : str , optional
            Dictionary key for topology type. Default is "type".
        vertexColorKey : str , optional
            Dictionary key for colour. Default is "color".
        colorScale : str , optional
            Colour scale name. Default is "viridis".
        storeBREP : bool , optional
            If set to True, stores BREP strings where available. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The Hasse diagram as a TGraph.
        """
        try:
            from topologicpy.Topology import Topology
        except Exception:
            if not silent:
                print("TGraph.HasseDiagram - Error: TopologicPy Topology is unavailable. Returning None.")
            return None
        rank = {"vertex": 0, "edge": 1, "wire": 2, "face": 3, "shell": 4, "cell": 5, "cellcomplex": 6, "cellComplex": 6}
        type_list = list(types or [])
        type_list = [str(t) for t in type_list]
        def _subtopologies(tname):
            candidates = [tname, tname.lower(), tname.capitalize()]
            for cand in candidates:
                try:
                    vals = Topology.SubTopologies(topology, subTopologyType=cand)
                    if isinstance(vals, list):
                        return vals
                except Exception:
                    pass
            method_names = {
                "vertex": "Vertices", "edge": "Edges", "wire": "Wires", "face": "Faces",
                "shell": "Shells", "cell": "Cells", "cellComplex": "CellComplexes", "cellcomplex": "CellComplexes",
            }
            m = method_names.get(tname, method_names.get(tname.lower(), None))
            if m is not None:
                try:
                    vals = getattr(Topology, m)(topology)
                    if isinstance(vals, list):
                        return vals
                except Exception:
                    pass
            return []
        def _brep(obj):
            try:
                return Topology.BREPString(obj)
            except Exception:
                return str(id(obj))
        def _vertex_key_set(obj):
            try:
                verts = Topology.Vertices(obj)
                return {_brep(v) for v in verts}
            except Exception:
                return {_brep(obj)}
        by_type = {}
        for tname in type_list:
            by_type[tname] = _subtopologies(tname)
        g = TGraph(directed=True, allowSelfLoops=False, allowParallelEdges=False)
        index_by_key = {}
        ordered = []
        for tname in type_list:
            r = rank.get(tname, rank.get(tname.lower(), 0))
            objs = by_type.get(tname, [])
            for i, obj in enumerate(objs):
                key_obj = _brep(obj)
                if key_obj in index_by_key:
                    continue
                d = {vertexLabelKey: f"{tname}_{i}", vertexTypeKey: tname, "rank": r,
                     "x": float(i), "y": float(r) * float(minDistance), "z": 0.0}
                if storeBREP:
                    d["brep"] = key_obj
                try:
                    from topologicpy.Color import Color
                    d[vertexColorKey] = Color.AnyToHex(Color.ByValueInRange(r, minValue=0, maxValue=max(1, len(type_list)-1), colorScale=colorScale))
                except Exception:
                    pass
                idx = g.AddVertex(dictionary=d, representation=obj)
                index_by_key[key_obj] = idx
                ordered.append((idx, obj, tname, r, _vertex_key_set(obj)))
        for child_idx, child_obj, child_type, child_rank, child_vs in ordered:
            for parent_idx, parent_obj, parent_type, parent_rank, parent_vs in ordered:
                if parent_rank != child_rank + 1:
                    continue
                if child_vs and child_vs.issubset(parent_vs):
                    src, dst = (parent_idx, child_idx) if topDown else (child_idx, parent_idx)
                    g.AddEdge(src, dst, directed=True, dictionary={"relationship": "contains"})
        return TGraph._OntologyAnnotateGraph(g, graphClass="top:HasseDiagram", vertexClass="top:Node", edgeClass="top:Relationship", generatedBy="TGraph.HasseDiagram", ontology=True, silent=True)

    @staticmethod
    def Reshape(graph: "TGraph", shape="spring 2D", k=0.8, seed=None, iterations=50,
                rootVertex=None, size=1, factor=1, sides=16, key="",
                tolerance=0.0001, silent=False) -> Optional["TGraph"]:
        """
        Repositions TGraph vertex coordinates using a simple layout algorithm.

        Parameters
        ----------
        graph : TGraph
            The input TGraph.
        shape : str , optional
            Layout name. Supported values include "spring 2D", "circle", "circular",
            "random", and "line". Default is "spring 2D".
        k : float , optional
            Spring-layout ideal distance factor. Default is 0.8.
        seed : int , optional
            Random seed. Default is None.
        iterations : int , optional
            Number of spring iterations. Default is 50.
        rootVertex : int or dict , optional
            Optional root vertex for line/tree-style layouts. Default is None.
        size : float , optional
            Overall layout size. Default is 1.
        factor : float , optional
            Additional coordinate scale factor. Default is 1.
        sides : int , optional
            Included for API compatibility. Default is 16.
        key : str , optional
            Optional dictionary key whose values may influence ordering. Default is "".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        TGraph or None
            The reshaped graph, or None if invalid.
        """
        if not isinstance(graph, TGraph):
            return None
        import random as _random
        rng = _random.Random(seed)
        verts = TGraph.ActiveVertexIndices(graph)
        n = len(verts)
        if n == 0:
            return graph
        order = list(verts)
        if key:
            order.sort(key=lambda i: str(graph._vertices[i].get("dictionary", {}).get(key, i)))
        pos = {}
        shape_l = str(shape or "spring 2D").lower()
        scale = float(size or 1) * float(factor or 1)
        if "circle" in shape_l or "circular" in shape_l:
            for i, v in enumerate(order):
                a = 2.0 * math.pi * i / max(1, n)
                pos[v] = [scale * math.cos(a), scale * math.sin(a), 0.0]
        elif "line" in shape_l:
            for i, v in enumerate(order):
                x = 0.0 if n <= 1 else scale * ((2.0 * i / (n - 1)) - 1.0)
                pos[v] = [x, 0.0, 0.0]
        else:
            for v in order:
                pos[v] = [rng.uniform(-scale, scale), rng.uniform(-scale, scale), 0.0]
            if "spring" in shape_l and n > 1:
                area = max((2.0 * scale) ** 2, 1e-9)
                ideal = float(k or 0.8) * math.sqrt(area / n)
                adj = {v: set(TGraph.AdjacentIndices(graph, v, mode="all")) for v in order}
                for it in range(max(1, int(iterations))):
                    disp = {v: [0.0, 0.0] for v in order}
                    for i, v in enumerate(order):
                        for u in order[i+1:]:
                            dx = pos[v][0] - pos[u][0]
                            dy = pos[v][1] - pos[u][1]
                            dist = math.sqrt(dx*dx + dy*dy) + 1e-9
                            force = (ideal * ideal) / dist
                            fx, fy = dx / dist * force, dy / dist * force
                            disp[v][0] += fx; disp[v][1] += fy
                            disp[u][0] -= fx; disp[u][1] -= fy
                    for v in order:
                        for u in adj.get(v, set()):
                            if u not in pos:
                                continue
                            dx = pos[v][0] - pos[u][0]
                            dy = pos[v][1] - pos[u][1]
                            dist = math.sqrt(dx*dx + dy*dy) + 1e-9
                            force = (dist * dist) / ideal
                            fx, fy = dx / dist * force, dy / dist * force
                            disp[v][0] -= fx; disp[v][1] -= fy
                    temp = scale * (1.0 - (it / max(1, int(iterations))))
                    for v in order:
                        dx, dy = disp[v]
                        length = math.sqrt(dx*dx + dy*dy) + 1e-9
                        pos[v][0] += dx / length * min(length, temp)
                        pos[v][1] += dy / length * min(length, temp)
        for v, c in pos.items():
            d = graph._vertices[v].setdefault("dictionary", {})
            d["x"], d["y"], d["z"] = float(c[0]), float(c[1]), float(c[2])
        graph._invalidate_cache()
        return graph

    @staticmethod
    def Tietze(radius: float = 0.5, height: float = 1) -> "TGraph":
        """
        Creates Tietze's graph as a TGraph.

        Parameters
        ----------
        radius : float , optional
            Radius used for the generated circular embedding. Default is 0.5.
        height : float , optional
            Vertical amplitude used for the generated embedding. Default is 1.

        Returns
        -------
        TGraph
            The created Tietze graph.
        """
        g = TGraph(directed=False, allowSelfLoops=False, allowParallelEdges=False)
        try:
            from topologicpy.Shell import Shell
            from topologicpy.Topology import Topology
            from topologicpy.Edge import Edge
            m = Shell.MobiusStrip(radius=radius, height=height, uSides=12, vSides=3)
            eb = Shell.ExternalBoundary(m)
            verts = Topology.Vertices(eb)
            new_verts = [verts[i] for i in range(0, len(verts), 2)]
            graph_vertices = []
            graph_edges = []
            for r in range(0, 6):
                s = r + 6
                e = Edge.ByVertices(new_verts[r], new_verts[s])
                if r == 0:
                    v1 = Edge.VertexByParameter(e, 2/3); v2 = Edge.EndVertex(e); e = Edge.ByVertices(v1, v2)
                elif r == 1:
                    v3 = Edge.VertexByParameter(e, 1/3); v4 = Edge.VertexByParameter(e, 2/3); e = Edge.ByVertices(v3, v4)
                elif r == 2:
                    v5 = Edge.StartVertex(e); v6 = Edge.VertexByParameter(e, 1/3); e = Edge.ByVertices(v5, v6)
                elif r == 3:
                    v7 = Edge.VertexByParameter(e, 1/3); v8 = Edge.VertexByParameter(e, 2/3); e = Edge.ByVertices(v7, v8)
                elif r == 4:
                    v9 = Edge.VertexByParameter(e, 2/3); v10 = Edge.EndVertex(e); e = Edge.ByVertices(v9, v10)
                elif r == 5:
                    v11 = Edge.VertexByParameter(e, 1/3); v12 = Edge.VertexByParameter(e, 2/3); e = Edge.ByVertices(v11, v12)
                graph_edges.append(e)
            graph_vertices = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]
            extra_edges = [(9,1),(4,9),(1,4),(0,3),(3,7),(7,8),(8,11),(11,2),(2,5),(5,6),(6,10),(10,0)]
            for a,b in extra_edges:
                graph_edges.append(Edge.ByVertices(graph_vertices[a], graph_vertices[b]))
            return TGraph.ByVerticesEdges(graph_vertices, graph_edges, directed=False, allowSelfLoops=False, allowParallelEdges=False)
        except Exception:
            pass
        for i in range(12):
            a = 2.0 * math.pi * i / 12.0
            z = (float(height) * 0.25) * math.sin(3.0 * a)
            g.AddVertex(dictionary={"label": f"v{i+1}", "x": float(radius) * math.cos(a), "y": float(radius) * math.sin(a), "z": z})
        edges = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(9,1),(4,9),(1,4),(0,3),(3,7),(7,8),(8,11),(11,2),(2,5),(5,6),(6,10),(10,0)]
        for a,b in edges:
            g.AddEdge(a, b)
        return g

    # ---------------------------------------------------------------------
    # Class-level aliases and cached kernels
    # ---------------------------------------------------------------------

    _NUMBA_BFS_PARENT = None
    _NUMBA_BFS_TREE = None
    AngularConnectivity = Connectivity
    AngularChoice = Choice
    AngularIntegration = Integration
    AngularBetweenness = Choice

