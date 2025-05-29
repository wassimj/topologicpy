# Copyright (C) 2025
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

class CSG():
    @staticmethod
    def _unique_coords(used_coords=[], width=10, length=10, height=10, max_attempts=1000, mantissa=6, tolerance=0.0001):
        import math
        import random

        def is_too_close(p1, p2):
            return math.dist(p1, p2) < tolerance

        if used_coords == []:
            return [0,0,0]
        
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
        Adds a topology vertex to the graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        topology : topologic_core.Topology
            The input topology..
        matrix : list , optional
            The desired 4X4 transformation matrix to apply to the result before any further operations. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
            topologic_core.Vertex
                The added vertex.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Matrix import Matrix
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph

        if matrix == None:
            matrix = Matrix.Identity()
        if graph == None:
            used_coords = []
        else:
            used_coords = [Vertex.Coordinates(v, mantissa=mantissa) for v in Graph.Vertices(graph)]
        loc = CSG._unique_coords(used_coords=used_coords, width=10, length=10, height=10, max_attempts=1000, mantissa=mantissa, tolerance=tolerance)
        v = Vertex.ByCoordinates(loc)
        keys = ["brep",
                "brepType",
                "brepTypeString",
                "matrix",
                "type",
                "id"]
        values = [Topology.BREPString(topology),
                Topology.Type(topology),
                Topology.TypeAsString(topology),
                matrix,
                "topology",
                Topology.UUID(v)]
        
        d = Dictionary.ByKeysValues(keys, values)
        v = Topology.SetDictionary(v, d)
        if graph == None:
            graph = Graph.ByVerticesEdges([v], [])
        else:
            graph = Graph.AddVertex(graph, v, tolerance=tolerance, silent=silent)
        return v

    @staticmethod
    def AddOperationVertex(graph, operation, a, b, matrix = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Adds an operation vertex to the graph that performs a CSG operation on two child vertices. For ordered operations, the order of a and b inputs is important.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        operation : str
            The operation to perform. One of "union", "difference", "intersection", "xor", "merge", "impose", "imprint", "slice".
        a : topologic_core.Vertex
            The first input vertex.
        b : topologic_core.Vertex
            The second input vertex.
        matrix : list , optional
            The desired 4X4 transformation matrix to apply to the result before any further operations. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
            topologic_core.Vertex
                The added vertex.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph

        if graph == None:
            used_coords = []
        else:
            used_coords = [Vertex.Coordinates(v, mantissa=mantissa) for v in Graph.Vertices(graph)]
        loc = CSG._unique_coords(used_coords=used_coords, width=10, length=10, height=10, max_attempts=1000, mantissa=mantissa, tolerance=tolerance)
        v = Vertex.ByCoordinates(loc)
        a_id = Dictionary.ValueAtKey(Topology.Dictionary(a), "id")
        b_id = Dictionary.ValueAtKey(Topology.Dictionary(b), "id")

        keys = ["brep",
                "brepType",
                "brepTypeString",
                "matrix",
                "type",
                "id",
                "operation",
                "a_id",
                "b_id"]
        values = [None,
                None,
                None,
                matrix,
                "operation",
                Topology.UUID(v),
                operation,
                a_id,
                b_id]
        
        d = Dictionary.ByKeysValues(keys, values)
        v = Topology.SetDictionary(v, d)
        return v
    
    @staticmethod
    def Connect(graph, vertexA, vertexB, tolerance: float = 0.0001, silent: bool = False):
        """
        Connects two vertices in the graph with a directional edge from vertexA to vertexB.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        matrix : list , optional
            The desired 4X4 transformation matrix to apply to the result before any further operations. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
            topologic_core.Vertex
                The added vertex.
        """
        from topologicpy.Graph import Graph
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("CSG.Connect - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("CSG.Connect - Error: The input vertexB parameter is not a valid vertex. Returning None.")
            return None
        edge = Edge.ByVertices(vertexA, vertexB, tolerance=tolerance, silent=silent)
        if graph == None:
            vertices = [vertexA, vertexB]
            edges = [edge]
        else:
            if not Topology.IsInstance(graph, "Graph"):
                if not silent:
                    print("CSG.Connect - Error: The input graph parameter is not a valid graph. Returning None.")
                return None
            vertices = Graph.Vertices(graph)
            edges = Graph.Edges(graph)
            if len(edges) > 0:
                edges.append(edge)
            else:
                edges = [edge]
        graph = Graph.ByVerticesEdges(vertices, edges)
        return graph

    @staticmethod
    def Init():
        """
        Returns an initial empty graph.

        Parameters
        ----------

        Returns
        -------
            topologic_core.Graph
                The initialized empty graph.
        """

        from topologicpy.Graph import Graph

        return Graph.ByVerticesEdges([], [])
    
    @staticmethod
    def Invoke(graph, silent: bool = False):
        """
        Traverses the graph and evaluates all CSG operations from leaves to root, returning the final result.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
            topologic_core.Topology
                The final topology.
        """

        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Graph import Graph
        
        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("CSG.Connect - Error: The input graphA parameter is not a valid graph. Returning None.")
            return None
        
        def traverse(vertex):
            d = Topology.Dictionary(vertex)
            node_type = Dictionary.ValueAtKey(d, "type")

            if node_type == "topology":
                topology = Topology.ByBREPString(Dictionary.ValueAtKey(d, "brep"))
                matrix = Dictionary.ValueAtKey(d, "matrix")
                topology = Topology.Transform(topology, matrix)
                topology = Topology.SetDictionary(topology, d)
                return topology

            elif node_type == "operation":
                op = Dictionary.ValueAtKey(d, "operation")
                a_id = Dictionary.ValueAtKey(d, "a_id")

                children = Graph.IncomingVertices(graph, vertex, directed=True)
                if len(children) != 2:
                    if not silent:
                        print(f"CSG.Invoke - Error: Operation '{op}' must have exactly 2 children. Returning None.")
                    return None
                
                child_0 = children[0]
                child_1 = children[1]
                child_0_id = Dictionary.ValueAtKey(Topology.Dictionary(child_0), "id")
                if child_0_id != a_id:
                    child_0 = children[1]
                    child_1 = children[0]
                a = traverse(child_0)
                b = traverse(child_1)
                if op.lower() == "union":
                    result = Topology.Union(a, b, silent=silent)
                elif op.lower() == "intersection":
                    result = Topology.Intersect(a, b, silent=silent)
                elif op.lower() == "difference":
                    result = Topology.Difference(a, b, silent=silent)
                elif op.lower() == "xor" or "sym" in op.lower():
                    result = Topology.SymmeticDifference(a, b, silent=silent)
                elif op.lower() == "merge":
                    result = Topology.Merge(a, b, silent=silent)
                elif op.lower() == "impose":
                    result = Topology.Impose(a, b, silent=silent)
                elif op.lower() == "imprint":
                    result = Topology.Imprint(a, b, silent=silent)
                elif op.lower() == "slice":
                    result = Topology.Slice(a, b, silent=silent)
                else:
                    if not silent:
                        print(f"CSG.Invoke - Error: Unknown operation '{op}'. Returning None.")
                    return None
                keys = ["brep",
                        "brepType",
                        "brepTypeString"]
                values = [Topology.BREPString(result),
                        Topology.Type(result),
                        Topology.TypeAsString(result)]
                d = Dictionary.SetValuesAtKeys(d, keys=keys, values=values)
                vertex = Topology.SetDictionary(vertex, d)
                matrix = Dictionary.ValueAtKey(d, "matrix")
                if not matrix == None:
                    result = Topology.Transform(result, matrix)
                result = Topology.SetDictionary(result, d)
                return result
            else:
                if not silent:
                    print(f"CSG.Invoke - Error: Unknown node type '{node_type}'. Returning None.")
                return None

        # Assume root is the vertex with no outgoing edges
        roots = [v for v in Graph.Vertices(graph) if not Graph.OutgoingVertices(graph, v, directed=True)]
        if len(roots) != 1:
            if not silent:
                print("CSG.Invoke - Error: Graph must have exactly one root node. Returning None.")
            return None

        return traverse(roots[0])

    def Topologies(graph, xOffset: float = 0, yOffset: float = 0, zOffset: float = 0, scale: float = 1, silent: bool = False):
        """
        Places each geometry (using its centroid) at its corresponding vertex location in the graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        xOffset : float , optional
            An additional x offset. The default is 0.
        yOffset : float , optional
            An additional y offset. The default is 0.
        zOffset : float , optional
            An additional z offset. The default is 0.
        scale : float , optional
            A desired scale to resize the placed topologies. The default is 1.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
           list
                The list of topologies placed at their corresponding location in the graph.
        """
        
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Graph import Graph

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("CSG.Topologies - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        placed_topologies= []

        for vertex in Graph.Vertices(graph):
            geom = Topology.ByBREPString(Dictionary.ValueAtKey(Topology.Dictionary(vertex), "brep"))
            originA = Topology.Centroid(geom)
            geom = Topology.Scale(geom, origin=originA, x=scale, y=scale, z=scale)
            originB = vertex
            placed = Topology.Place(geom, originA, originB)
            placed = Topology.Translate(placed, x=xOffset, y=yOffset, z=zOffset)
            placed_topologies.append(placed)

        return placed_topologies

