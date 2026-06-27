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


class CSG():
    # -------------------------------------------------------------------------
    # Internal Graph / TGraph compatibility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_tgraph(graph) -> bool:
        """Returns True if the input object is a topologicpy.TGraph object."""
        try:
            from topologicpy.TGraph import TGraph
            return isinstance(graph, TGraph)
        except Exception:
            return False

    @staticmethod
    def _is_legacy_graph(graph) -> bool:
        """Returns True if the input object is a legacy topologic_core.Graph."""
        if graph is None or CSG._is_tgraph(graph):
            return False
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsInstance(graph, "Graph"))
        except Exception:
            return False

    @staticmethod
    def _is_graph_like(graph) -> bool:
        """Returns True if the input object is a legacy Graph or TGraph."""
        return CSG._is_tgraph(graph) or CSG._is_legacy_graph(graph)

    @staticmethod
    def _python_dictionary(obj) -> dict:
        """Returns a Python dictionary from a Topologic object or TGraph record."""
        if obj is None:
            return {}

        if isinstance(obj, dict):
            d = obj.get("dictionary", None)
            if isinstance(d, dict):
                return dict(d)
            # TGraph records often store useful fields at top level as well.
            out = {}
            for key, value in obj.items():
                if key not in ["dictionary", "representation", "topology", "object"]:
                    out[key] = value
            if isinstance(d, dict):
                out.update(d)
            return out

        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            d = Topology.Dictionary(obj)
            pd = Dictionary.PythonDictionary(d)
            return dict(pd or {})
        except Exception:
            return {}

    @staticmethod
    def _dictionary_value(obj, key: str, default=None):
        """Returns a dictionary value from a Topologic object or TGraph record."""
        if key is None:
            return default
        try:
            return CSG._python_dictionary(obj).get(key, default)
        except Exception:
            return default

    @staticmethod
    def _set_dictionary_values(obj, values: dict):
        """Sets dictionary values on a Topologic object or TGraph record."""
        values = dict(values or {})
        if obj is None:
            return None

        if isinstance(obj, dict):
            d = obj.get("dictionary", {})
            if not isinstance(d, dict):
                d = {}
            d.update(values)
            obj["dictionary"] = d
            for key, value in values.items():
                obj.setdefault(key, value)
            return obj

        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            d = Topology.Dictionary(obj)
            try:
                d = Dictionary.SetValuesAtKeys(d, list(values.keys()), list(values.values()))
            except TypeError:
                d = Dictionary.SetValuesAtKeys(d, keys=list(values.keys()), values=list(values.values()))
            return Topology.SetDictionary(obj, d, silent=True)
        except Exception:
            try:
                from topologicpy.Topology import Topology
                from topologicpy.Dictionary import Dictionary
                d = Dictionary.ByKeysValues(list(values.keys()), list(values.values()))
                return Topology.SetDictionary(obj, d, silent=True)
            except Exception:
                return obj

    @staticmethod
    def _vertex_representation(vertex):
        """Returns a Topologic vertex representation from a vertex or TGraph record."""
        if vertex is None:
            return None
        if isinstance(vertex, dict):
            for key in ["representation", "topology", "object", "vertex"]:
                rep = vertex.get(key, None)
                if rep is not None:
                    return rep
            try:
                from topologicpy.Vertex import Vertex
                coords = CSG._coordinates(vertex)
                if coords is not None:
                    return Vertex.ByCoordinates(coords[0], coords[1], coords[2])
            except Exception:
                return None
        return vertex

    @staticmethod
    def _coordinates(vertex, mantissa: int = 6):
        """Returns XYZ coordinates from a Topologic vertex or TGraph vertex record."""
        if vertex is None:
            return None
        if isinstance(vertex, dict):
            d = CSG._python_dictionary(vertex)
            for keys in [("x", "y", "z"), ("X", "Y", "Z")]:
                if all(k in d for k in keys):
                    try:
                        return [round(float(d[keys[0]]), mantissa), round(float(d[keys[1]]), mantissa), round(float(d[keys[2]]), mantissa)]
                    except Exception:
                        pass
            coords = vertex.get("coordinates", None) or vertex.get("coords", None)
            if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                try:
                    return [round(float(coords[0]), mantissa), round(float(coords[1]), mantissa), round(float(coords[2]), mantissa)]
                except Exception:
                    return None
            rep = CSG._vertex_representation(vertex)
            if rep is not None and rep is not vertex:
                return CSG._coordinates(rep, mantissa=mantissa)
            return None
        try:
            from topologicpy.Vertex import Vertex
            return Vertex.Coordinates(vertex, mantissa=mantissa)
        except Exception:
            return None

    @staticmethod
    def _vertices(graph, asTopologic: bool = True):
        """Returns graph vertices from a legacy Graph or TGraph."""
        if graph is None:
            return []
        if CSG._is_tgraph(graph):
            try:
                from topologicpy.TGraph import TGraph
                try:
                    return TGraph.Vertices(graph, asTopologic=asTopologic, active=True) or []
                except TypeError:
                    try:
                        return TGraph.Vertices(graph, asTopologic=asTopologic) or []
                    except TypeError:
                        return TGraph.Vertices(graph) or []
            except Exception:
                return []
        try:
            from topologicpy.Graph import Graph
            return Graph.Vertices(graph) or []
        except Exception:
            return []

    @staticmethod
    def _edges(graph, asTopologic: bool = True):
        """Returns graph edges from a legacy Graph or TGraph."""
        if graph is None:
            return []
        if CSG._is_tgraph(graph):
            try:
                from topologicpy.TGraph import TGraph
                try:
                    return TGraph.Edges(graph, asTopologic=asTopologic, active=True) or []
                except TypeError:
                    try:
                        return TGraph.Edges(graph, asTopologic=asTopologic) or []
                    except TypeError:
                        return TGraph.Edges(graph) or []
            except Exception:
                return []
        try:
            from topologicpy.Graph import Graph
            return Graph.Edges(graph) or []
        except Exception:
            return []

    @staticmethod
    def _vertex_index(vertex, fallback=None):
        """Returns a stable graph index/id for a vertex or TGraph record."""
        if isinstance(vertex, dict):
            for key in ["index", "id", "node_id", "vertex_id"]:
                value = vertex.get(key, None)
                if value is not None:
                    return value
            d = CSG._python_dictionary(vertex)
            for key in ["index", "id", "node_id", "vertex_id"]:
                value = d.get(key, None)
                if value is not None:
                    return value
            return fallback
        for key in ["index", "id", "node_id", "vertex_id"]:
            value = CSG._dictionary_value(vertex, key, None)
            if value is not None:
                return value
        return fallback

    @staticmethod
    def _normalise_index(value):
        """Normalises integer-like indices to integers."""
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except Exception:
                pass
            return s
        return value

    @staticmethod
    def _same_vertex(a, b) -> bool:
        """Returns True if two vertex objects/records represent the same graph node."""
        if a is b:
            return True
        aid = CSG._vertex_index(a, None)
        bid = CSG._vertex_index(b, None)
        if aid is not None and bid is not None:
            return CSG._normalise_index(aid) == CSG._normalise_index(bid)
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsSame(a, b))
        except Exception:
            return False

    @staticmethod
    def _vertex_lookup(vertices):
        """Returns a lookup dictionary for vertex ids/indices."""
        lookup = {}
        for i, vertex in enumerate(vertices or []):
            for value in [i, str(i), CSG._vertex_index(vertex, i)]:
                if value is None:
                    continue
                value = CSG._normalise_index(value)
                lookup[value] = vertex
                lookup[str(value)] = vertex
        return lookup

    @staticmethod
    def _edge_endpoints(edge, vertices=None, lookup=None):
        """Returns start and end vertices for an edge or TGraph edge record."""
        vertices = vertices or []
        lookup = lookup or CSG._vertex_lookup(vertices)

        if isinstance(edge, dict):
            src = edge.get("src", edge.get("source", edge.get("start", edge.get("start_index", None))))
            dst = edge.get("dst", edge.get("target", edge.get("end", edge.get("end_index", None))))
            d = CSG._python_dictionary(edge)
            if src is None:
                src = d.get("src", d.get("source", d.get("start", d.get("start_index", None))))
            if dst is None:
                dst = d.get("dst", d.get("target", d.get("end", d.get("end_index", None))))
            src = CSG._normalise_index(src)
            dst = CSG._normalise_index(dst)
            return lookup.get(src, lookup.get(str(src))), lookup.get(dst, lookup.get(str(dst)))

        try:
            from topologicpy.Edge import Edge
            return Edge.StartVertex(edge), Edge.EndVertex(edge)
        except Exception:
            try:
                from topologicpy.Topology import Topology
                return Topology.StartVertex(edge), Topology.EndVertex(edge)
            except Exception:
                return None, None

    @staticmethod
    def _incoming_vertices(graph, vertex, directed: bool = True):
        """Returns incoming vertices for a vertex in a legacy Graph or TGraph."""
        if graph is None or vertex is None:
            return []
        if CSG._is_tgraph(graph):
            vertices = CSG._vertices(graph, asTopologic=False)
            edges = CSG._edges(graph, asTopologic=False)
            lookup = CSG._vertex_lookup(vertices)
            result = []
            for edge in edges:
                sv, ev = CSG._edge_endpoints(edge, vertices=vertices, lookup=lookup)
                if CSG._same_vertex(ev, vertex):
                    result.append(sv)
                elif not directed and CSG._same_vertex(sv, vertex):
                    result.append(ev)
            return [v for v in result if v is not None]
        try:
            from topologicpy.Graph import Graph
            return Graph.IncomingVertices(graph, vertex, directed=directed) or []
        except Exception:
            return []

    @staticmethod
    def _outgoing_vertices(graph, vertex, directed: bool = True):
        """Returns outgoing vertices for a vertex in a legacy Graph or TGraph."""
        if graph is None or vertex is None:
            return []
        if CSG._is_tgraph(graph):
            vertices = CSG._vertices(graph, asTopologic=False)
            edges = CSG._edges(graph, asTopologic=False)
            lookup = CSG._vertex_lookup(vertices)
            result = []
            for edge in edges:
                sv, ev = CSG._edge_endpoints(edge, vertices=vertices, lookup=lookup)
                if CSG._same_vertex(sv, vertex):
                    result.append(ev)
                elif not directed and CSG._same_vertex(ev, vertex):
                    result.append(sv)
            return [v for v in result if v is not None]
        try:
            from topologicpy.Graph import Graph
            return Graph.OutgoingVertices(graph, vertex, directed=directed) or []
        except Exception:
            return []

    @staticmethod
    def _graph_by_vertices_edges(vertices, edges, asTGraph: bool = False, ontology: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Constructs either a legacy Graph or a TGraph from vertices and edges."""
        if asTGraph:
            try:
                from topologicpy.TGraph import TGraph
                try:
                    return TGraph.ByVerticesEdges(vertices=vertices, edges=edges, directed=True, allowSelfLoops=True, allowParallelEdges=True, ontology=ontology, silent=silent)
                except TypeError:
                    try:
                        return TGraph.ByVerticesEdges(vertices, edges, directed=True, ontology=ontology, silent=silent)
                    except TypeError:
                        return TGraph.ByVerticesEdges(vertices, edges)
            except Exception as e:
                if not silent:
                    print("CSG._graph_by_vertices_edges - Warning: Could not create TGraph. Falling back to Graph.")
                    print("Error:", e)
        try:
            from topologicpy.Graph import Graph
            try:
                return Graph.ByVerticesEdges(vertices, edges, ontology=ontology, silent=silent)
            except TypeError:
                try:
                    return Graph.ByVerticesEdges(vertices, edges, silent=silent)
                except TypeError:
                    return Graph.ByVerticesEdges(vertices, edges)
        except Exception as e:
            if not silent:
                print("CSG._graph_by_vertices_edges - Error: Could not create graph. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def _add_vertex(graph, vertex, tolerance: float = 0.0001, silent: bool = False):
        """Adds a vertex to a legacy Graph or TGraph and returns the updated graph when available."""
        if graph is None:
            return None
        if CSG._is_tgraph(graph):
            try:
                from topologicpy.TGraph import TGraph
                d = CSG._python_dictionary(vertex)
                try:
                    return TGraph.AddVertex(graph, vertex=vertex, dictionary=d, tolerance=tolerance, silent=silent)
                except TypeError:
                    try:
                        return TGraph.AddVertex(graph, vertex, tolerance=tolerance, silent=silent)
                    except TypeError:
                        return TGraph.AddVertex(graph, vertex)
            except Exception:
                return graph
        try:
            from topologicpy.Graph import Graph
            return Graph.AddVertex(graph, vertex, tolerance=tolerance, silent=silent)
        except Exception:
            return graph

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @staticmethod
    def _unique_coords(used_coords=[], width=10, length=10, height=10, max_attempts=1000, mantissa=6, tolerance=0.0001):
        import math
        import random

        def is_too_close(p1, p2):
            return math.dist(p1, p2) < tolerance

        if used_coords == []:
            return [0, 0, 0]
        
        attempts = 0
        while attempts < max_attempts:
            x = round(random.uniform(0, width), mantissa)
            y = round(random.uniform(0, length), mantissa)
            z = round(random.uniform(0, height), mantissa)
            candidate = [x, y, z]
            if all(not is_too_close(candidate, used) for used in used_coords):
                return candidate
            attempts += 1

        raise RuntimeError("Could not find a unique coordinate within the attempt limit.")
    
    @staticmethod
    def AddTopologyVertex(graph, topology, matrix: list = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Adds a topology vertex to the input CSG graph.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input CSG graph. If set to None, the vertex is created and returned
            without requiring an existing graph.
        topology : topologic_core.Topology
            The input topology represented by the created graph vertex.
        matrix : list , optional
            The desired 4X4 transformation matrix to apply to the topology before
            any further operations. Default is None.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Vertex
            The added vertex.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Matrix import Matrix
        from topologicpy.Topology import Topology

        if matrix is None:
            matrix = Matrix.Identity()
        used_coords = []
        if graph is not None:
            used_coords = [c for c in [CSG._coordinates(v, mantissa=mantissa) for v in CSG._vertices(graph, asTopologic=False)] if c is not None]
        loc = CSG._unique_coords(used_coords=used_coords, width=10, length=10, height=10, max_attempts=1000, mantissa=mantissa, tolerance=tolerance)
        v = Vertex.ByCoordinates(loc)
        keys = ["brep", "brepType", "brepTypeString", "matrix", "type", "id"]
        values = [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology), matrix, "topology", Topology.UUID(v)]
        d = Dictionary.ByKeysValues(keys, values)
        v = Topology.SetDictionary(v, d)

        if graph is not None:
            _ = CSG._add_vertex(graph, v, tolerance=tolerance, silent=silent)
        return v

    @staticmethod
    def AddOperationVertex(graph, operation, a, b, matrix=None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Adds an operation vertex to the input CSG graph.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input CSG graph.
        operation : str
            The operation to perform. Options include "union", "difference",
            "intersection", "xor", "merge", "impose", "imprint", and "slice".
        a : topologic_core.Vertex or dict
            The first input vertex. For ordered operations, this is operand A.
        b : topologic_core.Vertex or dict
            The second input vertex. For ordered operations, this is operand B.
        matrix : list , optional
            The desired 4X4 transformation matrix to apply to the result before any
            further operations. Default is None.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Vertex
            The added operation vertex.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        used_coords = []
        if graph is not None:
            used_coords = [c for c in [CSG._coordinates(v, mantissa=mantissa) for v in CSG._vertices(graph, asTopologic=False)] if c is not None]
        loc = CSG._unique_coords(used_coords=used_coords, width=10, length=10, height=10, max_attempts=1000, mantissa=mantissa, tolerance=tolerance)
        v = Vertex.ByCoordinates(loc)
        a_id = CSG._dictionary_value(a, "id", None)
        b_id = CSG._dictionary_value(b, "id", None)

        keys = ["brep", "brepType", "brepTypeString", "matrix", "type", "id", "operation", "a_id", "b_id"]
        values = [None, None, None, matrix, "operation", Topology.UUID(v), operation, a_id, b_id]
        d = Dictionary.ByKeysValues(keys, values)
        v = Topology.SetDictionary(v, d)

        if graph is not None:
            _ = CSG._add_vertex(graph, v, tolerance=tolerance, silent=silent)
        return v
    
    @staticmethod
    def Connect(graph, vertexA, vertexB, asTGraph: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Connects two vertices in the CSG graph with a directional edge from vertexA to vertexB.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input graph. If set to None, a new graph is created.
        vertexA : topologic_core.Vertex or dict
            The start vertex.
        vertexB : topologic_core.Vertex or dict
            The end vertex.
        asTGraph : bool , optional
            If graph is None and asTGraph is True, a TGraph is created. If graph is
            already a TGraph, this option is ignored and a TGraph is returned.
            Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The updated graph.
        """
        
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        va = CSG._vertex_representation(vertexA)
        vb = CSG._vertex_representation(vertexB)
        if not Topology.IsInstance(va, "Vertex"):
            if not silent:
                print("CSG.Connect - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vb, "Vertex"):
            if not silent:
                print("CSG.Connect - Error: The input vertexB parameter is not a valid vertex. Returning None.")
            return None

        edge = Edge.ByVertices(va, vb, tolerance=tolerance, silent=silent)

        # Store src/dst ids for TGraph and for robust endpoint recovery.
        a_id = CSG._dictionary_value(vertexA, "id", CSG._dictionary_value(va, "id", None))
        b_id = CSG._dictionary_value(vertexB, "id", CSG._dictionary_value(vb, "id", None))
        CSG._set_dictionary_values(edge, {"src": a_id, "dst": b_id})

        if graph is None:
            vertices = [va, vb]
            edges = [edge]
            return CSG._graph_by_vertices_edges(vertices, edges, asTGraph=asTGraph, ontology=False, tolerance=tolerance, silent=silent)

        if not CSG._is_graph_like(graph):
            if not silent:
                print("CSG.Connect - Error: The input graph parameter is not a valid Graph or TGraph. Returning None.")
            return None

        graph_is_tgraph = CSG._is_tgraph(graph)
        vertices = CSG._vertices(graph, asTopologic=True)
        edges = CSG._edges(graph, asTopologic=True)

        # Ensure both endpoints are present in the rebuilt graph.
        if not any(CSG._same_vertex(v, va) for v in vertices):
            vertices.append(va)
        if not any(CSG._same_vertex(v, vb) for v in vertices):
            vertices.append(vb)
        edges.append(edge)

        return CSG._graph_by_vertices_edges(vertices, edges, asTGraph=graph_is_tgraph or asTGraph, ontology=False, tolerance=tolerance, silent=silent)

    @staticmethod
    def Init(asTGraph: bool = False, silent: bool = False):
        """
        Returns an initial empty CSG graph.

        Parameters
        ----------
        asTGraph : bool , optional
            If set to True, returns a TGraph. Otherwise returns a legacy Graph.
            Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The initialized empty graph.
        """

        return CSG._graph_by_vertices_edges([], [], asTGraph=asTGraph, ontology=False, silent=silent)
    
    @staticmethod
    def Invoke(graph, silent: bool = False):
        """
        Traverses the CSG graph and evaluates operations from leaves to root.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input CSG graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Topology
            The final topology.
        """

        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        
        if not CSG._is_graph_like(graph):
            if not silent:
                print("CSG.Invoke - Error: The input graph parameter is not a valid Graph or TGraph. Returning None.")
            return None
        
        def traverse(vertex):
            d_py = CSG._python_dictionary(vertex)
            node_type = d_py.get("type", None)

            if node_type == "topology":
                topology = Topology.ByBREPString(d_py.get("brep", None))
                matrix = d_py.get("matrix", None)
                if matrix is not None:
                    topology = Topology.Transform(topology, matrix)
                try:
                    d = Dictionary.ByPythonDictionary(d_py)
                    topology = Topology.SetDictionary(topology, d, silent=True)
                except Exception:
                    pass
                return topology

            elif node_type == "operation":
                op = d_py.get("operation", None)
                a_id = d_py.get("a_id", None)

                children = CSG._incoming_vertices(graph, vertex, directed=True)
                if len(children) != 2:
                    if not silent:
                        print(f"CSG.Invoke - Error: Operation '{op}' must have exactly 2 children. Returning None.")
                    return None
                
                child_0 = children[0]
                child_1 = children[1]
                child_0_id = CSG._dictionary_value(child_0, "id", None)
                if child_0_id != a_id:
                    child_0 = children[1]
                    child_1 = children[0]

                a = traverse(child_0)
                b = traverse(child_1)
                if a is None or b is None:
                    return None

                op_l = str(op or "").lower()
                if op_l == "union":
                    result = Topology.Union(a, b, silent=silent)
                elif op_l == "intersection":
                    result = Topology.Intersect(a, b, silent=silent)
                elif op_l == "difference":
                    result = Topology.Difference(a, b, silent=silent)
                elif op_l == "xor" or "sym" in op_l:
                    try:
                        result = Topology.SymmetricDifference(a, b, silent=silent)
                    except AttributeError:
                        result = Topology.SymmeticDifference(a, b, silent=silent)
                elif op_l == "merge":
                    result = Topology.Merge(a, b, silent=silent)
                elif op_l == "impose":
                    result = Topology.Impose(a, b, silent=silent)
                elif op_l == "imprint":
                    result = Topology.Imprint(a, b, silent=silent)
                elif op_l == "slice":
                    result = Topology.Slice(a, b, silent=silent)
                else:
                    if not silent:
                        print(f"CSG.Invoke - Error: Unknown operation '{op}'. Returning None.")
                    return None

                if result is None:
                    return None

                update = {
                    "brep": Topology.BREPString(result),
                    "brepType": Topology.Type(result),
                    "brepTypeString": Topology.TypeAsString(result),
                }
                CSG._set_dictionary_values(vertex, update)
                d_py.update(update)
                matrix = d_py.get("matrix", None)
                if matrix is not None:
                    result = Topology.Transform(result, matrix)
                try:
                    d = Dictionary.ByPythonDictionary(d_py)
                    result = Topology.SetDictionary(result, d, silent=True)
                except Exception:
                    pass
                return result
            else:
                if not silent:
                    print(f"CSG.Invoke - Error: Unknown node type '{node_type}'. Returning None.")
                return None

        roots = [v for v in CSG._vertices(graph, asTopologic=False) if not CSG._outgoing_vertices(graph, v, directed=True)]
        if len(roots) != 1:
            if not silent:
                print("CSG.Invoke - Error: Graph must have exactly one root node. Returning None.")
            return None

        return traverse(roots[0])

    @staticmethod
    def Topologies(graph, xOffset: float = 0, yOffset: float = 0, zOffset: float = 0, scale: float = 1, silent: bool = False):
        """
        Places each stored topology at its corresponding graph vertex location.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input CSG graph.
        xOffset : float , optional
            An additional x offset. Default is 0.
        yOffset : float , optional
            An additional y offset. Default is 0.
        zOffset : float , optional
            An additional z offset. Default is 0.
        scale : float , optional
            A desired scale to resize the placed topologies. Default is 1.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The list of topologies placed at their corresponding location in the graph.
        """
        
        from topologicpy.Topology import Topology

        if not CSG._is_graph_like(graph):
            if not silent:
                print("CSG.Topologies - Error: The input graph parameter is not a valid Graph or TGraph. Returning None.")
            return None

        placed_topologies = []
        for vertex in CSG._vertices(graph, asTopologic=False):
            brep = CSG._dictionary_value(vertex, "brep", None)
            if brep in [None, ""]:
                continue
            geom = Topology.ByBREPString(brep)
            if geom is None:
                continue
            originA = Topology.Centroid(geom)
            geom = Topology.Scale(geom, origin=originA, x=scale, y=scale, z=scale)
            originB = CSG._vertex_representation(vertex)
            if originB is None:
                continue
            placed = Topology.Place(geom, originA, originB)
            placed = Topology.Translate(placed, x=xOffset, y=yOffset, z=zOffset)
            placed_topologies.append(placed)

        return placed_topologies
