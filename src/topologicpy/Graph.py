# Copyright (C) 2024
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

import topologic_core as topologic
import random
import time
import os
import warnings

from collections import namedtuple
from multiprocessing import Process, Queue

try:
    import numpy as np
except:
    print("Graph - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("Graph - numpy library installed correctly.")
    except:
        warnings.warn("Graph - Error: Could not import numpy.")

try:
    import pandas as pd
except:
    print("Graph - Installing required pandas library.")
    try:
        os.system("pip install pandas")
    except:
        os.system("pip install pandas --user")
    try:
        import pandas as pd
        print("Graph - pandas library installed correctly.")
    except:
        warnings.warn("Graph - Error: Could not import pandas.")

try:
    from tqdm.auto import tqdm
except:
    print("Graph - Installing required tqdm library.")
    try:
        os.system("pip install tqdm")
    except:
        os.system("pip install tqdm --user")
    try:
        from tqdm.auto import tqdm
        print("Graph - tqdm library installed correctly.")
    except:
        warnings.warn("Graph - Error: Could not import tqdm.")

GraphQueueItem = namedtuple('GraphQueueItem', ['edges'])

class WorkerProcessPool(object):
    """
    Create and manage a list of Worker processes. Each worker process
    creates a 2D navigation graph.
    """
    def __init__(self,
                 num_workers,
                 message_queue,
                 used,
                 face,
                 sources,
                 destinations,
                 tolerance=0.0001):
        self.num_workers = num_workers
        self.message_queue = message_queue
        self.used = used
        self.face = face
        self.sources = sources
        self.destinations = destinations
        self.tolerance = tolerance
        self.process_list = []

    def startProcesses(self):
        num_item_per_worker = len(self.sources) // self.num_workers
        for i in range(self.num_workers):
            if i == self.num_workers - 1:
                begin = i * num_item_per_worker
                sub_sources = self.sources[begin:]
            else:
                begin = i * num_item_per_worker
                end = begin + num_item_per_worker
                sub_sources = self.sources[begin : end]
            wp = WorkerProcess(begin,
                               self.message_queue,
                               self.used,
                               self.face,
                               sub_sources,
                               self.destinations,
                               self.tolerance)
            wp.start()
            self.process_list.append(wp)

    def stopProcesses(self):
        for p in self.process_list:
            p.join()
        self.process_list = []

    def join(self):
        for p in self.process_list:
            p.join()

class WorkerProcess(Process):
    """
    Creates a 2D navigation graph from a subset of sources and the list of destinations.
    """
    def __init__(self,
                 start_index,
                 message_queue,
                 used,
                 face,
                 sources,
                 destinations,
                 tolerance=0.0001):
        Process.__init__(self, target=self.run)
        self.start_index = start_index
        self.message_queue = message_queue
        self.used = used
        self.face = face
        self.sources =  sources
        self.destinations = destinations
        self.tolerance = tolerance

    def run(self):
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge

        edges = []
        face = Topology.ByBREPString(self.face)
        sources = [Topology.ByBREPString(s) for s in self.sources]
        destinations = [Topology.ByBREPString(s) for s in self.destinations]
        for i in range(len(sources)):
            source = sources[i]
            index_b = Vertex.Index(source, destinations, tolerance=self.tolerance)
            for j in range(len(destinations)):
                destination = destinations[j]
                index_a = Vertex.Index(destination, sources, tolerance=self.tolerance)
                if self.used[i + self.start_index][j] == 1 or self.used[j][i + self.start_index]:
                    continue
                if Vertex.Distance(source, destination) > self.tolerance:
                    edge = Edge.ByVertices([source, destination])
                    e = Topology.Intersect(edge, face)
                    if Topology.IsInstance(e, "Edge"):
                        edges.append(edge)
                self.used[i + self.start_index][j] = 1
                if not index_a == None and not index_b == None:
                    self.used[j][i + self.start_index] = 1
        if len(edges) > 0:
            edges_str = [Topology.BREPString(s) for s in edges]
            self.message_queue.put(GraphQueueItem(edges_str))

class MergingProcess(Process):
    """
    Receive message from other processes and merging the result
    """
    def __init__(self, message_queue):
        Process.__init__(self, target=self.wait_message)
        self.message_queue = message_queue
        self.final_edges = []

    def wait_message(self):
        while True:
            try:
                item = self.message_queue.get()
                if item is None:
                    self.message_queue.put(GraphQueueItem(self.final_edges))
                    break
                self.final_edges.extend(item.edges)
            except Exception as e:
                print(str(e))

class _Tree:
    def __init__(self, node="", *children):
        self.node = node
        self.width = len(node)
        if children:
            self.children = children
        else:
            self.children = []

    def __str__(self):
        return "%s" % (self.node)

    def __repr__(self):
        return "%s" % (self.node)

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.children[key]
        if isinstance(key, str):
            for child in self.children:
                if child.node == key:
                    return child

    def __iter__(self):
        return self.children.__iter__()

    def __len__(self):
        return len(self.children)

class _DrawTree(object):
    def __init__(self, tree, parent=None, depth=0, number=1):
        self.x = -1.0
        self.y = depth
        self.tree = tree
        self.children = [
            _DrawTree(c, self, depth + 1, i + 1) for i, c in enumerate(tree.children)
        ]
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        # this is the number of the node in its group of siblings 1..n
        self.number = number

    def left(self):
        return self.thread or len(self.children) and self.children[0]

    def right(self):
        return self.thread or len(self.children) and self.children[-1]

    def lbrother(self):
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return n
                else:
                    n = node
        return n

    def get_lmost_sibling(self):
        if not self._lmost_sibling and self.parent and self != self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    lmost_sibling = property(get_lmost_sibling)

    def __str__(self):
        return "%s: x=%s mod=%s" % (self.tree, self.x, self.mod)

    def __repr__(self):
        return self.__str__()

class Graph:
    def AdjacencyDictionary(graph, vertexLabelKey: str = "label", edgeKey: str = "Length", reverse: bool = False, mantissa: int = 6):
        """
        Returns the adjacency dictionary of the input Graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexLabelKey : str , optional
            The returned vertices are labelled according to the dictionary values stored under this key.
            If the vertexLabelKey does not exist, it will be created and the vertices are labelled numerically and stored in the vertex dictionary under this key. The default is "label".
        edgeKey : str , optional
            If set, the edges' dictionaries will be searched for this key to set their weight. If the key is set to "length" (case insensitive), the length of the edge will be used as its weight. If set to None, a weight of 1 will be used. The default is "Length".
        reverse : bool , optional
                If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. The default is False.
        mantissa : int , optional
                The desired length of the mantissa. The default is 6.

        Returns
        -------
        dict
            The adjacency dictionary.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.AdjacencyDictionary - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexLabelKey, str):
            print("Graph.AdjacencyDictionary - Error: The input vertexLabelKey is not a valid string. Returning None.")
            return None
        vertices = Graph.Vertices(graph)
        labels = []
        n = max(len(str(len(vertices))), 3)
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            value = Dictionary.ValueAtKey(d, vertexLabelKey)
            if value == None:
                value = str(i).zfill(n)
            if d == None:
                d = Dictionary.ByKeyValue(vertexLabelKey, value)
            else:
                d = Dictionary.SetValueAtKey(d, vertexLabelKey, value)
            v = Topology.SetDictionary(v, d)
            labels.append(value)
        vertices = Helper.Sort(vertices, labels)
        labels.sort()
        if reverse == True:
            vertices.reverse()
            labels.reverse()
        order = len(vertices)
        adjDict = {}
        for i in range(order):
            v = Graph.NearestVertex(graph, vertices[i])
            vertex_label = labels[i]
            adjVertices = Graph.AdjacentVertices(graph, v)
            temp_list = []
            for adjVertex in adjVertices:
                if edgeKey == None:
                    weight = 1
                elif "length" in edgeKey.lower():
                    edge = Graph.Edge(graph, v, adjVertex)
                    weight = Edge.Length(edge, mantissa=mantissa)
                else:
                    edge = Graph.Edge(graph, v, adjVertex)
                    weight = Dictionary.ValueAtKey(Topology.Dictionary(edge), edgeKey)
                    if weight == None:
                        weight = Edge.Length(edge, mantissa=mantissa)
                    else:
                        weight = round(weight, mantissa)
                adjIndex = Vertex.Index(adjVertex, vertices)
                adjLabel = labels[adjIndex]
                if not adjIndex == None:
                    temp_list.append((adjLabel, weight))
            temp_list.sort()
            adjDict[vertex_label] = temp_list
        return adjDict
    
    @staticmethod
    def AdjacencyMatrix(graph, vertexKey=None, reverse=False, edgeKeyFwd=None, edgeKeyBwd=None, bidirKey=None, bidirectional=True, useEdgeIndex=False, useEdgeLength=False, tolerance=0.0001):
        """
        Returns the adjacency matrix of the input Graph. See https://en.wikipedia.org/wiki/Adjacency_matrix.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            If set, the returned list of vertices is sorted according to the dictionary values stored under this key. The default is None.
        reverse : bool , optional
            If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. The default is False.
        edgeKeyFwd : str , optional
            If set, the value at this key in the connecting edge from start vertex to end vertex (forward) will be used instead of the value 1. The default is None. useEdgeIndex and useEdgeLength override this setting.
        edgeKeyBwd : str , optional
            If set, the value at this key in the connecting edge from end vertex to start vertex (backward) will be used instead of the value 1. The default is None. useEdgeIndex and useEdgeLength override this setting.
        bidirKey : bool , optional
            If set to True or False, this key in the connecting edge will be used to determine is the edge is supposed to be bidirectional or not. If set to None, the input variable bidrectional will be used instead. The default is None
        bidirectional : bool , optional
            If set to True, the edges in the graph that do not have a bidireKey in their dictionaries will be treated as being bidirectional. Otherwise, the start vertex and end vertex of the connecting edge will determine the direction. The default is True.
        useEdgeIndex : bool , False
            If set to True, the adjacency matrix values will the index of the edge in Graph.Edges(graph). The default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        useEdgeLength : bool , False
            If set to True, the adjacency matrix values will the length of the edge in Graph.Edges(graph). The default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The adjacency matrix.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.AdjacencyMatrix - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        vertices = Graph.Vertices(graph)
        if not vertexKey == None:
            sorting_values = []
            for v in vertices:
                d = Topology.Dictionary(v)
                value = Dictionary.ValueAtKey(d, vertexKey)
                sorting_values.append(value)
            vertices = Helper.Sort(vertices, sorting_values)
            if reverse == True:
                vertices.reverse()

        edges = Graph.Edges(graph)
        order = len(vertices)
        matrix = []
        # Initialize the matrix with zeroes
        for i in range(order):
            tempRow = []
            for j in range(order):
                tempRow.append(0)
            matrix.append(tempRow)
        
        for i, edge in enumerate(edges):
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            svi = Vertex.Index(sv, vertices, tolerance=tolerance)
            evi = Vertex.Index(ev, vertices, tolerance=tolerance)
            if bidirKey == None:
                bidir = bidirectional
            else:
                bidir = Dictionary.ValueAtKey(Topology.Dictionary(edge), bidirKey)
                if bidir == None:
                    bidir = bidirectional
            if edgeKeyFwd == None:
                valueFwd = 1
            else:
                valueFwd = Dictionary.ValueAtKey(Topology.Dictionary(edge), edgeKeyFwd)
                if valueFwd == None:
                    valueFwd = 1
            if edgeKeyBwd == None:
                valueBwd = 1
            else:
                valueBwd = Dictionary.ValueAtKey(Topology.Dictionary(edge), edgeKeyBwd)
                if valueBwd == None:
                    valueBwd = 1
            if useEdgeIndex:
                valueFwd = i+1
                valueBwd = i+1
            if useEdgeLength:
                valueFwd = Edge.Length(edge)
                valueBwd = Edge.Length(edge)
            if not svi == None and not evi == None:
                matrix[svi][evi] = valueFwd
                if bidir:
                    matrix[evi][svi] = valueBwd
        return matrix
    
    @staticmethod
    def AdjacencyList(graph, vertexKey=None, reverse=True, tolerance=0.0001):
        """
        Returns the adjacency list of the input Graph. See https://en.wikipedia.org/wiki/Adjacency_list.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            If set, the returned list of vertices is sorted according to the dictionary values stored under this key. The default is None.
        reverse : bool , optional
            If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The adjacency list.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.AdjacencyList - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertices = Graph.Vertices(graph)
        if not vertexKey == None:
            sorting_values = []
            for v in vertices:
                d = Topology.Dictionary(v)
                value = Dictionary.ValueAtKey(d, vertexKey)
                sorting_values.append(value)
            vertices = Helper.Sort(vertices, sorting_values)
            if reverse == True:
                vertices.reverse()
        order = len(vertices)
        adjList = []
        for i in range(order):
            tempRow = []
            v = Graph.NearestVertex(graph, vertices[i])
            adjVertices = Graph.AdjacentVertices(graph, v)
            for adjVertex in adjVertices:
                adjIndex = Vertex.Index(vertex=adjVertex, vertices=vertices, strict=True, tolerance=tolerance)
                if not adjIndex == None:
                    tempRow.append(adjIndex)
            tempRow.sort()
            adjList.append(tempRow)
        return adjList

    @staticmethod
    def AddEdge(graph, edge, transferVertexDictionaries: bool = False, transferEdgeDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Adds the input edge to the input Graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        edge : topologic_core.Edge
            The input edge.
        transferVertexDictionaries : bool, optional
            If set to True, the dictionaries of the vertices are transferred to the graph.
        transferEdgeDictionaries : bool, optional
            If set to True, the dictionaries of the edges are transferred to the graph.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input edge added to it.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        def addIfUnique(graph_vertices, vertex, tolerance):
            unique = True
            returnVertex = vertex
            for gv in graph_vertices:
                if (Vertex.Distance(vertex, gv) < tolerance):
                    if transferVertexDictionaries == True:
                        gd = Topology.Dictionary(gv)
                        vd = Topology.Dictionary(vertex)
                        gk = gd.Keys()
                        vk = vd.Keys()
                        d = None
                        if (len(gk) > 0) and (len(vk) > 0):
                            d = Dictionary.ByMergedDictionaries([gd, vd])
                        elif (len(gk) > 0) and (len(vk) < 1):
                            d = gd
                        elif (len(gk) < 1) and (len(vk) > 0):
                            d = vd
                        if d:
                            _ = Topology.SetDictionary(gv, d)
                    unique = False
                    returnVertex = gv
                    break
            if unique:
                graph_vertices.append(vertex)
            return [graph_vertices, returnVertex]

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.AddEdge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Graph.AddEdge - Error: The input edge is not a valid edge. Returning the input graph.")
            return graph
        graph_vertices = Graph.Vertices(graph)
        graph_edges = Graph.Edges(graph, graph_vertices, tolerance)
        vertices = Edge.Vertices(edge)
        new_vertices = []
        for vertex in vertices:
            graph_vertices, nv = addIfUnique(graph_vertices, vertex, tolerance)
            new_vertices.append(nv)
        new_edge = Edge.ByVertices([new_vertices[0], new_vertices[1]], tolerance=tolerance)
        if transferEdgeDictionaries == True:
            d = Topology.Dictionary(edge)
            keys = Dictionary.Keys(d)
            if isinstance(keys, list):
                if len(keys) > 0:
                    _ = Topology.SetDictionary(new_edge, d)
        graph_edges.append(new_edge)
        new_graph = Graph.ByVerticesEdges(graph_vertices, graph_edges)
        return new_graph
    
    @staticmethod
    def AddVertex(graph, vertex, tolerance: float = 0.0001, silent: bool = False):
        """
        Adds the input vertex to the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input vertex added to it.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.AddVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Graph.AddVertex - Error: The input vertex is not a valid vertex. Returning the input graph.")
            return graph
        _ = graph.AddVertices([vertex], tolerance)
        return graph

    @staticmethod
    def AddVertices(graph, vertices, tolerance: float = 0.0001, silent: bool = False):
        """
        Adds the input vertex to the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list
            The input list of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input vertex added to it.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.AddVertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertices, list):
            if not silent:
                print("Graph.AddVertices - Error: The input list of vertices is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            if not silent:
                print("Graph.AddVertices - Error: Could not find any valid vertices in the input list of vertices. Returning None.")
            return None
        _ = graph.AddVertices(vertices, tolerance)
        return graph
    
    @staticmethod
    def AdjacentVertices(graph, vertex, silent: bool = False):
        """
        Returns the list of vertices connected to the input vertex.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            the input vertex.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The list of adjacent vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.AdjacentVertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Graph.AdjacentVertices - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        vertices = []
        _ = graph.AdjacentVertices(vertex, vertices)
        return list(vertices)
    
    @staticmethod
    def AllPaths(graph, vertexA, vertexB, timeLimit=10, silent: bool = False):
        """
        Returns all the paths that connect the input vertices within the allowed time limit in seconds.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        timeLimit : int , optional
            The time limit in second. The default is 10 seconds.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The list of all paths (wires) found within the time limit.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.AllPaths - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Graph.AllPaths - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Graph.AllPaths - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        paths = []
        _ = graph.AllPaths(vertexA, vertexB, True, timeLimit, paths)
        return paths

    @staticmethod
    def AverageClusteringCoefficient(graph, mantissa: int = 6, silent: bool = False):
        """
        Returns the average clustering coefficient of the input graph. See https://en.wikipedia.org/wiki/Clustering_coefficient.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The average clustering coefficient of the input graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.LocalClusteringCoefficient - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        vertices = Graph.Vertices(graph)
        if len(vertices) < 1:
            if not silent:
                print("Graph.LocalClusteringCoefficient - Error: The input graph parameter is a NULL graph. Returning None.")
            return None
        if len(vertices) == 1:
            return 0.0
        lcc = Graph.LocalClusteringCoefficient(graph, vertices)
        acc = round(sum(lcc)/float(len(lcc)), mantissa)
        return acc
    
    @staticmethod
    def BOTGraph(graph,
                bidirectional: bool = False,
                includeAttributes: bool = False,
                includeLabel: bool = False,
                includeGeometry: bool = False,
                siteLabel: str = "Site_0001",
                siteDictionary: dict = None,
                buildingLabel: str = "Building_0001",
                buildingDictionary: dict = None , 
                storeyPrefix: str = "Storey",
                floorLevels: list = [],
                vertexLabelKey: str = "label",
                typeKey: str = "type",
                verticesKey: str = "vertices",
                edgesKey: str = "edges",
                edgeLabelKey: str = "",
                sourceKey: str = "source",
                targetKey: str = "target",
                xKey: str = "hasX",
                yKey: str = "hasY",
                zKey: str = "hasZ",
                geometryKey: str = "brep",
                spaceType: str = "space",
                wallType: str = "wall",
                slabType: str = "slab",
                doorType: str = "door",
                windowType: str = "window",
                contentType: str = "content",
                namespace: str = "http://github.com/wassimj/topologicpy/resources",
                mantissa: int = 6
                ):
        """
        Creates an RDF graph according to the BOT ontology. See https://w3c-lbd-cg.github.io/bot/.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        bidirectional : bool , optional
            If set to True, reverse relationships are created wherever possible. Otherwise, they are not. The default is False.
        includeAttributes : bool , optional
            If set to True, the attributes associated with vertices in the graph are written out. Otherwise, they are not. The default is False.
        includeLabel : bool , optional
            If set to True, a label is attached to each node. Otherwise, it is not. The default is False.
        includeGeometry : bool , optional
            If set to True, the geometry associated with vertices in the graph are written out. Otherwise, they are not. The default is False.
        siteLabel : str , optional
            The desired site label. The default is "Site_0001".
        siteDictionary : dict , optional
            The dictionary of site attributes to include in the output. The default is None.
        buildingLabel : str , optional
            The desired building label. The default is "Building_0001".
        buildingDictionary : dict , optional
            The dictionary of building attributes to include in the output. The default is None.
        storeyPrefix : str , optional
            The desired prefixed to use for each building storey. The default is "Storey".
        floorLevels : list , optional
            The list of floor levels. This should be a numeric list, sorted from lowest to highest.
            If not provided, floorLevels will be computed automatically based on the vertices' (zKey)) attribute. See below.
        verticesKey : str , optional
            The desired key name to call vertices. The default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. The default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. The default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. The default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. The default is "hasX".
        yKey : str , optional
            The desired key name to use for y-coordinates. The default is "hasY".
        zKey : str , optional
            The desired key name to use for z-coordinates. The default is "hasZ".
        geometryKey : str , optional
            The desired key name to use for geometry. The default is "brep".
        typeKey : str , optional
            The dictionary key to use to look up the type of the node. The default is "type".
        geometryKey : str , optional
            The dictionary key to use to look up the geometry of the node. The default is "brep".
        spaceType : str , optional
            The dictionary string value to use to look up vertices of type "space". The default is "space".
        wallType : str , optional
            The dictionary string value to use to look up vertices of type "wall". The default is "wall".
        slabType : str , optional
            The dictionary string value to use to look up vertices of type "slab". The default is "slab".
        doorType : str , optional
            The dictionary string value to use to look up vertices of type "door". The default is "door".
        windowType : str , optional
            The dictionary string value to use to look up vertices of type "window". The default is "window".
        contentType : str , optional
            The dictionary string value to use to look up vertices of type "content". The default is "contents".
        namespace : str , optional
            The desired namespace to use in the BOT graph. The default is "http://github.com/wassimj/topologicpy/resources".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

            
        Returns
        -------
        rdflib.graph.Graph
            The rdf graph using the BOT ontology.
        """

        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import os
        import warnings
        
        try:
            from rdflib import Graph as RDFGraph
            from rdflib import URIRef, Literal, Namespace
            from rdflib.namespace import RDF, RDFS, XSD
        except:
            print("Graph.BOTGraph - Information: Installing required rdflib library.")
            try:
                os.system("pip install rdflib")
            except:
                os.system("pip install rdflib --user")
            try:
                from rdflib import Graph as RDFGraph
                from rdflib import URIRef, Literal, Namespace
                from rdflib.namespace import RDF, RDFS
                print("Graph.BOTGraph - Information: rdflib library installed correctly.")
            except:
                warnings.warn("Graph.BOTGraph - Error: Could not import rdflib. Please try to install rdflib manually. Returning None.")
                return None
        
        if floorLevels == None:
            floorLevels = []
        
        json_data = Graph.JSONData(graph,
                                   verticesKey=verticesKey,
                                   edgesKey=edgesKey,
                                   vertexLabelKey=vertexLabelKey,
                                   edgeLabelKey=edgeLabelKey,
                                   sourceKey=sourceKey,
                                   targetKey=targetKey,
                                   xKey=xKey,
                                   yKey=yKey,
                                   zKey=zKey,
                                   geometryKey=geometryKey,
                                   mantissa=mantissa)
        
        # Create an empty RDF graph
        bot_graph = RDFGraph()
        
        # Define namespaces
        rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        bot = Namespace("https://w3id.org/bot#")
        top = Namespace(namespace)
        
        # Define a custom prefix mapping
        bot_graph.namespace_manager.bind("bot", bot)
        bot_graph.namespace_manager.bind("xsd", XSD)
        bot_graph.namespace_manager.bind("top", top)
        
        # Add site
        site_uri = URIRef(siteLabel)
        bot_graph.add((site_uri, rdf.type, bot.Site))
        if includeLabel:
            bot_graph.add((site_uri, RDFS.label, Literal(siteLabel)))
        if Topology.IsInstance(siteDictionary, "Dictionary"):
            keys = Dictionary.Keys(siteDictionary)
            for key in keys:
                value = Dictionary.ValueAtKey(siteDictionary, key)
                if not (key == vertexLabelKey) and not (key == typeKey):
                    if isinstance(value, float):
                        datatype = XSD.float
                    elif isinstance(value, bool):
                        datatype = XSD.boolean
                    elif isinstance(value, int):
                        datatype = XSD.integer
                    elif isinstance(value, str):
                        datatype = XSD.string
                    bot_graph.add((site_uri, top[key], Literal(value, datatype=datatype)))
        # Add building
        building_uri = URIRef(buildingLabel)
        bot_graph.add((building_uri, rdf.type, bot.Building))
        if includeLabel:
            bot_graph.add((building_uri, RDFS.label, Literal(buildingLabel)))
        if Topology.IsInstance(buildingDictionary, "Dictionary"):
            keys = Dictionary.Keys(siteDictionary)
            for key in keys:
                value = Dictionary.ValueAtKey(siteDictionary, key)
                if not (key == vertexLabelKey) and not (key == typeKey):
                    if isinstance(value, float):
                        datatype = XSD.float
                    elif isinstance(value, bool):
                        datatype = XSD.boolean
                    elif isinstance(value, int):
                        datatype = XSD.integer
                    elif isinstance(value, str):
                        datatype = XSD.string
                    bot_graph.add((building_uri, top[key], Literal(value, datatype=datatype)))
        # Add stories
        # if floor levels are not given, then need to be computed
        if len(floorLevels) == 0:
            for node, attributes in json_data[verticesKey].items():
                if slabType.lower() in attributes[typeKey].lower():
                    floorLevels.append(attributes[zKey])
            floorLevels = list(set(floorLevels))
            floorLevels.sort()
            floorLevels = floorLevels[:-1]
        storey_uris = []
        n = max(len(str(len(floorLevels))),4)
        for i, floor_level in enumerate(floorLevels):
            storey_uri = URIRef(storeyPrefix+"_"+str(i+1).zfill(n))
            bot_graph.add((storey_uri, rdf.type, bot.Storey))
            if includeLabel:
                bot_graph.add((storey_uri, RDFS.label, Literal(storeyPrefix+"_"+str(i+1).zfill(n))))
            storey_uris.append(storey_uri)

        # Add triples to relate building to site and stories to building
        bot_graph.add((site_uri, bot.hasBuilding, building_uri))
        if bidirectional:
            bot_graph.add((building_uri, bot.isPartOf, site_uri)) # might not be needed

        for storey_uri in storey_uris:
            bot_graph.add((building_uri, bot.hasStorey, storey_uri))
            if bidirectional:
                bot_graph.add((storey_uri, bot.isPartOf, building_uri)) # might not be needed
        
        # Add vertices as RDF resources
        for node, attributes in json_data[verticesKey].items():
            node_uri = URIRef(top[node])
            if spaceType.lower() in attributes[typeKey].lower():
                bot_graph.add((node_uri, rdf.type, bot.Space))
                # Find the storey it is on
                z = attributes[zKey]
                level = Helper.Position(z, floorLevels)
                if level > len(storey_uris):
                    level = len(storey_uris)
                storey_uri = storey_uris[level-1]
                bot_graph.add((storey_uri, bot.hasSpace, node_uri))
                if bidirectional:
                    bot_graph.add((node_uri, bot.isPartOf, storey_uri)) # might not be needed
            elif windowType.lower() in attributes[typeKey].lower():
                bot_graph.add((node_uri, rdf.type, bot.Window))
            elif doorType.lower() in attributes[typeKey].lower():
                bot_graph.add((node_uri, rdf.type, bot.Door))
            elif wallType.lower() in attributes[typeKey].lower():
                bot_graph.add((node_uri, rdf.type, bot.Wall))
            elif slabType.lower() in attributes[typeKey].lower():
                bot_graph.add((node_uri, rdf.type, bot.Slab))
            else:
                bot_graph.add((node_uri, rdf.type, bot.Element))
            
            if includeAttributes:
                for key, value in attributes.items():
                    if key == geometryKey:
                        if includeGeometry:
                            bot_graph.add((node_uri, bot.hasSimpleGeometry, Literal(value)))
                    if key == vertexLabelKey:
                        if includeLabel:
                            bot_graph.add((node_uri, RDFS.label, Literal(value)))
                    elif key != typeKey and key != geometryKey:
                        if isinstance(value, float):
                            datatype = XSD.float
                        elif isinstance(value, bool):
                            datatype = XSD.boolean
                        elif isinstance(value, int):
                            datatype = XSD.integer
                        elif isinstance(value, str):
                            datatype = XSD.string
                        bot_graph.add((node_uri, top[key], Literal(value, datatype=datatype)))
            if includeLabel:
                for key, value in attributes.items():
                    if key == vertexLabelKey:
                        bot_graph.add((node_uri, RDFS.label, Literal(value)))
        
        # Add edges as RDF triples
        for edge, attributes in json_data[edgesKey].items():
            source = attributes[sourceKey]
            target = attributes[targetKey]
            source_uri = URIRef(top[source])
            target_uri = URIRef(top[target])
            if spaceType.lower() in json_data[verticesKey][source][typeKey].lower() and spaceType.lower() in json_data[verticesKey][target][typeKey].lower():
                bot_graph.add((source_uri, bot.adjacentTo, target_uri))
                if bidirectional:
                    bot_graph.add((target_uri, bot.adjacentTo, source_uri))
            elif spaceType.lower() in json_data[verticesKey][source][typeKey].lower() and wallType.lower() in json_data[verticesKey][target][typeKey].lower():
                bot_graph.add((target_uri, bot.interfaceOf, source_uri))
            elif spaceType.lower() in json_data[verticesKey][source][typeKey].lower() and slabType.lower() in json_data['vertices'][target][typeKey].lower():
                bot_graph.add((target_uri, bot.interfaceOf, source_uri))
            elif spaceType.lower() in json_data[verticesKey][source][typeKey].lower() and contentType.lower() in json_data[verticesKey][target][typeKey].lower():
                bot_graph.add((source_uri, bot.containsElement, target_uri))
                if bidirectional:
                    bot_graph.add((target_uri, bot.isPartOf, source_uri))
            else:
                bot_graph.add((source_uri, bot.connectsTo, target_uri))
                if bidirectional:
                    bot_graph.add((target_uri, bot.connectsTo, source_uri))
        return bot_graph

    @staticmethod
    def BOTString(graph,
                  format="turtle",
                  bidirectional: bool = False,
                  includeAttributes: bool = False,
                  includeLabel: bool = False,
                  includeGeometry: bool = False,
                  siteLabel: str = "Site_0001",
                  siteDictionary: dict = None,
                  buildingLabel: str = "Building_0001",
                  buildingDictionary: dict = None , 
                  storeyPrefix: str = "Storey",
                  floorLevels: list = [],
                  vertexLabelKey: str = "label",
                  typeKey: str = "type",
                  verticesKey: str = "vertices",
                  edgesKey: str = "edges",
                  edgeLabelKey: str = "",
                  sourceKey: str = "source",
                  targetKey: str = "target",
                  xKey: str = "hasX",
                  yKey: str = "hasY",
                  zKey: str = "hasZ",
                  geometryKey: str = "brep",
                  spaceType: str = "space",
                  wallType: str = "wall",
                  slabType: str = "slab",
                  doorType: str = "door",
                  windowType: str = "window",
                  contentType: str = "content",
                  namespace: str = "http://github.com/wassimj/topologicpy/resources",
                  mantissa: int = 6
                ):
        
        """
        Returns an RDF graph serialized string according to the BOT ontology. See https://w3c-lbd-cg.github.io/bot/.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        format : str , optional
            The desired output format, the options are listed below. Thde default is "turtle".
            turtle, ttl or turtle2 : Turtle, turtle2 is just turtle with more spacing & linebreaks
            xml or pretty-xml : RDF/XML, Was the default format, rdflib < 6.0.0
            json-ld : JSON-LD , There are further options for compact syntax and other JSON-LD variants
            ntriples, nt or nt11 : N-Triples , nt11 is exactly like nt, only utf8 encoded
            n3 : Notation-3 , N3 is a superset of Turtle that also caters for rules and a few other things
            trig : Trig , Turtle-like format for RDF triples + context (RDF quads) and thus multiple graphs
            trix : Trix , RDF/XML-like format for RDF quads
            nquads : N-Quads , N-Triples-like format for RDF quads
        bidirectional : bool , optional
            If set to True, reverse relationships are created wherever possible. Otherwise, they are not. The default is False.
        includeAttributes : bool , optional
            If set to True, the attributes associated with vertices in the graph are written out. Otherwise, they are not. The default is False.
        includeLabel : bool , optional
            If set to True, a label is attached to each node. Otherwise, it is not. The default is False.
        includeGeometry : bool , optional
            If set to True, the geometry associated with vertices in the graph are written out. Otherwise, they are not. The default is False.
        siteLabel : str , optional
            The desired site label. The default is "Site_0001".
        siteDictionary : dict , optional
            The dictionary of site attributes to include in the output. The default is None.
        buildingLabel : str , optional
            The desired building label. The default is "Building_0001".
        buildingDictionary : dict , optional
            The dictionary of building attributes to include in the output. The default is None.
        storeyPrefix : str , optional
            The desired prefixed to use for each building storey. The default is "Storey".
        floorLevels : list , optional
            The list of floor levels. This should be a numeric list, sorted from lowest to highest.
            If not provided, floorLevels will be computed automatically based on the vertices' (zKey)) attribute. See below.
        verticesKey : str , optional
            The desired key name to call vertices. The default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. The default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. The default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. The default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. The default is "hasX".
        yKey : str , optional
            The desired key name to use for y-coordinates. The default is "hasY".
        zKey : str , optional
            The desired key name to use for z-coordinates. The default is "hasZ".
        geometryKey : str , optional
            The desired key name to use for geometry. The default is "brep".
        typeKey : str , optional
            The dictionary key to use to look up the type of the node. The default is "type".
        spaceType : str , optional
            The dictionary string value to use to look up vertices of type "space". The default is "space".
        wallType : str , optional
            The dictionary string value to use to look up vertices of type "wall". The default is "wall".
        slabType : str , optional
            The dictionary string value to use to look up vertices of type "slab". The default is "slab".
        doorType : str , optional
            The dictionary string value to use to look up vertices of type "door". The default is "door".
        windowType : str , optional
            The dictionary string value to use to look up vertices of type "window". The default is "window".
        contentType : str , optional
            The dictionary string value to use to look up vertices of type "content". The default is "contents".
        namespace : str , optional
            The desired namespace to use in the BOT graph. The default is "http://github.com/wassimj/topologicpy/resources".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        
        Returns
        -------
        str
            The rdf graph serialized string using the BOT ontology.
        """
        
        bot_graph = Graph.BOTGraph(graph= graph,
                                   bidirectional= bidirectional,
                                   includeAttributes= includeAttributes,
                                   includeLabel= includeLabel,
                                   includeGeometry= includeGeometry,
                                   siteLabel= siteLabel,
                                   siteDictionary= siteDictionary,
                                   buildingLabel= buildingLabel,
                                   buildingDictionary=  buildingDictionary,
                                   storeyPrefix= storeyPrefix,
                                   floorLevels= floorLevels,
                                   vertexLabelKey= vertexLabelKey,
                                   typeKey= typeKey,
                                   verticesKey= verticesKey,
                                   edgesKey= edgesKey,
                                   edgeLabelKey= edgeLabelKey,
                                   sourceKey= sourceKey,
                                   targetKey= targetKey,
                                   xKey= xKey,
                                   yKey= yKey,
                                   zKey= zKey,
                                   geometryKey= geometryKey,
                                   spaceType= spaceType,
                                   wallType= wallType,
                                   slabType= slabType,
                                   doorType= doorType,
                                   windowType= windowType,
                                   contentType= contentType,
                                   namespace= namespace,
                                   mantissa= mantissa)
        return bot_graph.serialize(format=format)

    @staticmethod
    def BetweenessCentrality(graph, vertices=None, sources=None, destinations=None, key: str = "betweeness_centrality", mantissa: int = 6, tolerance: float = 0.001):
        """
            Returns the betweeness centrality measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the betweeness centrality of all the vertices in the input graph is computed. See https://en.wikipedia.org/wiki/Betweenness_centrality.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. The default is None which means all vertices in the input graph are considered.
        sources : list , optional
            The input list of source vertices. The default is None which means all vertices in the input graph are considered.
        destinations : list , optional
            The input list of destination vertices. The default is None which means all vertices in the input graph are considered.
        key : str , optional
            The dictionary key under which to save the betweeness centrality score. The default is "betweneess_centrality".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The betweeness centrality of the input list of vertices within the input graph. The values are in the range 0 to 1.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def betweeness(vertices, topologies, tolerance=0.001):
            returnList = [0] * len(vertices)
            for topology in topologies:
                t_vertices = Topology.Vertices(topology)
                for t_v in t_vertices:
                    index = Vertex.Index(t_v, vertices, strict=False, tolerance=tolerance)
                    if not index == None:
                        returnList[index] = returnList[index]+1
            return returnList

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.BetweenessCentrality - Error: The input graph is not a valid graph. Returning None.")
            return None
        graphVertices = Graph.Vertices(graph)
        if not isinstance(vertices, list):
            vertices = graphVertices
        else:
            vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Graph.BetweenessCentrality - Error: The input list of vertices does not contain valid vertices. Returning None.")
            return None
        if not isinstance(sources, list):
            sources = graphVertices
        else:
            sources = [v for v in sources if Topology.IsInstance(v, "Vertex")]
        if len(sources) < 1:
            print("Graph.BetweenessCentrality - Error: The input list of sources does not contain valid vertices. Returning None.")
            return None
        if not isinstance(destinations, list):
            destinations = graphVertices
        else:
            destinations = [v for v in destinations if Topology.IsInstance(v, "Vertex")]
        if len(destinations) < 1:
            print("Graph.BetweenessCentrality - Error: The input list of destinations does not contain valid vertices. Returning None.")
            return None
        
        paths = []
        try:
            for so in tqdm(sources, desc="Computing Shortest Paths", leave=False):
                v1 = Graph.NearestVertex(graph, so)
                for si in destinations:
                    v2 = Graph.NearestVertex(graph, si)
                    if not v1 == v2:
                        path = Graph.ShortestPath(graph, v1, v2)
                        if path:
                            paths.append(path)
        except:
            for so in sources:
                v1 = Graph.NearestVertex(graph, so)
                for si in destinations:
                    v2 = Graph.NearestVertex(graph, si)
                    if not v1 == v2:
                        path = Graph.ShortestPath(graph, v1, v2)
                        if path:
                            paths.append(path)

        scores = betweeness(vertices, paths, tolerance=tolerance)
        minValue = min(scores)
        maxValue = max(scores)
        size = maxValue - minValue
        scores = [round((v-minValue)/size, mantissa) for v in scores]
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, key, scores[i])
            v = Topology.SetDictionary(v, d)
        return scores
    
    @staticmethod
    def ByAdjacencyMatrixCSVPath(path: str, dictionaries: list = None, silent: bool = False):
        """
        Returns graphs according to the input path. This method assumes the CSV files follow an adjacency matrix schema.

        Parameters
        ----------
        path : str
            The file path to the adjacency matrix CSV file.
        dictionaries : list , optional
            A list of dictionaries to assign to the vertices of the graph. This list should be in
            the same order and of the same length as the rows in the adjacency matrix.
        silent : bool , optional
            If set to True, no warnings or error messages are displayed. The default is False.
        
        Returns
        -------
        topologic_core.Graph
            The created graph.
        
        """

        # Read the adjacency matrix from CSV file using pandas
        adjacency_matrix_df = pd.read_csv(path, header=None)
        
        # Convert DataFrame to a nested list
        adjacency_matrix = adjacency_matrix_df.values.tolist()
        return Graph.ByAdjacencyMatrix(adjacencyMatrix=adjacency_matrix, dictionaries=dictionaries, silent=silent)

    @staticmethod
    def ByAdjacencyMatrix(adjacencyMatrix, dictionaries = None, xMin=-0.5, yMin=-0.5, zMin=-0.5, xMax=0.5, yMax=0.5, zMax=0.5, silent=False):
        """
        Returns graphs according to the input folder path. This method assumes the CSV files follow DGL's schema.

        Parameters
        ----------
        adjacencyMatrix : list
            The adjacency matrix expressed as a nested list of 0s and 1s.
        dictionaries : list , optional
            A list of dictionaries to assign to the vertices of the graph. This list should be in
            the same order and of the same length as the rows in the adjacency matrix.
        xMin : float , optional
            The desired minimum value to assign for a vertex's X coordinate. The default is -0.5.
        yMin : float , optional
            The desired minimum value to assign for a vertex's Y coordinate. The default is -0.5.
        zMin : float , optional
            The desired minimum value to assign for a vertex's Z coordinate. The default is -0.5.
        xMax : float , optional
            The desired maximum value to assign for a vertex's X coordinate. The default is 0.5.
        yMax : float , optional
            The desired maximum value to assign for a vertex's Y coordinate. The default is 0.5.
        zMax : float , optional
            The desired maximum value to assign for a vertex's Z coordinate. The default is 0.5.
        silent : bool , optional
            If set to True, no warnings or error messages are displayed. The default is False.
        
        Returns
        -------
        topologic_core.Graph
            The created graph.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        import  random

        if not isinstance(adjacencyMatrix, list):
            print("Graph.ByAdjacencyMatrix - Error: The input adjacencyMatrix parameter is not a valid list. Returning None.")
            return None
        if isinstance(dictionaries, list):
            if not len(dictionaries) == len(adjacencyMatrix):
                if not silent:
                    print("Graph.ByAdjacencyMatrix - Error: The length of the dictionaries list and the adjacency matrix are different. Returning None.")
                return None

        # Add vertices with random coordinates
        vertices = []
        for i in range(len(adjacencyMatrix)):
            x, y, z = random.uniform(xMin,xMax), random.uniform(yMin,yMax), random.uniform(zMin,zMax)
            v = Vertex.ByCoordinates(x, y, z)
            v = Topology.SetDictionary(v, dictionaries[i])
            vertices.append(v)

        # Create the graph using vertices and edges
        if len(vertices) == 0:
            print("Graph.ByAdjacencyMatrix - Error: The graph does not contain any vertices. Returning None.")
            return None
        
        # Add edges based on the adjacency matrix
        edges = []
        for i in range(len(adjacencyMatrix)):
            for j in range(len(adjacencyMatrix)):
                if not i == j:
                    if not adjacencyMatrix[i][j] == 0:
                        edges.append(Edge.ByVertices([vertices[i], vertices[j]]))
        
        return Graph.ByVerticesEdges(vertices, edges)

    @staticmethod
    def ByBOTGraph(botGraph,
                   includeContext = False,
                   xMin = -0.5,
                   xMax = 0.5,
                   yMin = -0.5,
                   yMax = 0.5,
                   zMin = -0.5,
                   zMax = 0.5,
                   tolerance = 0.0001
                ):

        def value_by_string(s):
            if s.lower() == "true":
                return True
            if s.lower() == "false":
                return False
            vt = "str"
            s2 = s.strip("-")
            if s2.isnumeric():
                vt = "int"
            else:
                try:
                    s3 = s2.split(".")[0]
                    s4 = s2.split(".")[1]
                    if (s3.isnumeric() or s4.isnumeric()):
                        vt = "float"
                except:
                    vt = "str"
            if vt == "str":
                return s
            elif vt == "int":
                return int(s)
            elif vt == "float":
                return float(s)

        def collect_nodes_by_type(rdf_graph, node_type=None):
            results = set()

            if node_type is not None:
                for subj, pred, obj in rdf_graph.triples((None, None, None)):
                    if "type" in pred.lower():
                        if node_type.lower() in obj.lower():
                            results.add(subj)
            return list(results)

        def collect_attributes_for_subject(rdf_graph, subject):
            attributes = {}

            for subj, pred, obj in rdf_graph.triples((subject, None, None)):
                predicate_str = str(pred)
                object_str = str(obj)
                attributes[predicate_str] = object_str

            return attributes

        def get_triples_by_predicate_type(rdf_graph, predicate_type):
            triples = []

            for subj, pred, obj in rdf_graph:
                if pred.split('#')[-1].lower() == predicate_type.lower():
                    triples.append((str(subj), str(pred), str(obj)))

            return triples

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import random

        try:
            import rdflib
        except:
            print("Graph.BOTGraph - Information: Installing required rdflib library.")
            try:
                os.system("pip install rdflib")
            except:
                os.system("pip install rdflib --user")
            try:
                import rdflib
                print("Graph.BOTGraph - Information: rdflib library installed correctly.")
            except:
                warnings.warn("Graph.BOTGraph - Error: Could not import rdflib. Please try to install rdflib manually. Returning None.")
                return None
        
        predicates = ['adjacentto', 'interfaceof', 'containselement', 'connectsto']
        bot_types = ['Space', 'Wall', 'Slab', 'Door', 'Window', 'Element']

        if includeContext:
            predicates += ['hasspace', 'hasbuilding', 'hasstorey']
            bot_types += ['Site', 'Building', 'Storey']

        namespaces = botGraph.namespaces()

        for ns in namespaces:
            if 'bot' in ns[0].lower():
                bot_namespace = ns
                break

        ref = bot_namespace[1]

        nodes = []
        for bot_type in bot_types:
            node_type = rdflib.term.URIRef(ref+bot_type)
            nodes +=collect_nodes_by_type(botGraph, node_type=node_type)

        vertices = []
        dic = {}
        for node in nodes:
            x, y, z = random.uniform(xMin,xMax), random.uniform(yMin,yMax), random.uniform(zMin,zMax)
            d_keys = ["bot_id"]
            d_values = [str(node)]
            attributes = collect_attributes_for_subject(botGraph, node)
            keys = attributes.keys()
            for key in keys:
                key_type = key.split('#')[-1]
                if key_type.lower() not in predicates:
                    if 'x' == key_type.lower():
                        x = value_by_string(attributes[key])
                        d_keys.append('x')
                        d_values.append(x)
                    elif 'y' == key_type.lower():
                        y = value_by_string(attributes[key])
                        d_keys.append('y')
                        d_values.append(y)
                    elif 'z' == key_type.lower():
                        z = value_by_string(attributes[key])
                        d_keys.append('z')
                        d_values.append(z)
                    else:
                        d_keys.append(key_type.lower())
                        d_values.append(value_by_string(attributes[key].split("#")[-1]))

            d = Dictionary.ByKeysValues(d_keys, d_values)
            v = Vertex.ByCoordinates(x,y,z)
            v = Topology.SetDictionary(v, d)
            dic[str(node)] = v
            vertices.append(v)

        edges = []
        for predicate in predicates:
            triples = get_triples_by_predicate_type(botGraph, predicate)
            for triple in triples:
                subj = triple[0]
                obj = triple[2]
                sv = dic[subj]
                ev = dic[obj]
                e = Edge.ByVertices([sv,ev], tolerance=tolerance)
                d = Dictionary.ByKeyValue("type", predicate)
                e = Topology.SetDictionary(e, d)
                edges.append(e)

        return Graph.ByVerticesEdges(vertices, edges)

    @staticmethod
    def ByBOTPath(path,
                  includeContext = False,
                  xMin = -0.5,
                  xMax = 0.5,
                  yMin = -0.5,
                  yMax = 0.5,
                  zMin = -0.5,
                  zMax = 0.5,
                  tolerance = 0.0001
                  ):
        
        try:
            from rdflib import Graph as RDFGraph
        except:
            print("Graph.ByBOTPath - Information: Installing required rdflib library.")
            try:
                os.system("pip install rdflib")
            except:
                os.system("pip install rdflib --user")
            try:
                from rdflib import Graph as RDFGraph
                print("Graph.ByBOTPath - Information: rdflib library installed correctly.")
            except:
                warnings.warn("Graph.ByBOTPath - Error: Could not import rdflib. Please try to install rdflib manually. Returning None.")
                return None
        
        bot_graph = RDFGraph()
        bot_graph.parse(path)
        return Graph.ByBOTGraph(bot_graph,
                                includeContext = includeContext,
                                xMin = xMin,
                                xMax = xMax,
                                yMin = yMin,
                                yMax = yMax,
                                zMin = zMin,
                                zMax = zMax,
                                tolerance = tolerance
                                )
    @staticmethod
    def ByCSVPath(path,
                  graphIDHeader="graph_id", graphLabelHeader="label", graphFeaturesHeader="feat", graphFeaturesKeys=[],
                  edgeSRCHeader="src_id", edgeDSTHeader="dst_id", edgeLabelHeader="label", edgeTrainMaskHeader="train_mask", 
                  edgeValidateMaskHeader="val_mask", edgeTestMaskHeader="test_mask", edgeFeaturesHeader="feat", edgeFeaturesKeys=[],
                  nodeIDHeader="node_id", nodeLabelHeader="label", nodeTrainMaskHeader="train_mask", 
                  nodeValidateMaskHeader="val_mask", nodeTestMaskHeader="test_mask", nodeFeaturesHeader="feat", nodeXHeader="X", nodeYHeader="Y", nodeZHeader="Z",
                  nodeFeaturesKeys=[], tolerance=0.0001):
        """
        Returns graphs according to the input folder path. This method assumes the CSV files follow DGL's schema.

        Parameters
        ----------
        path : str
            The path to the folder containing the .yaml and .csv files for graphs, edges, and nodes.
        graphIDHeader : str , optional
            The column header string used to specify the graph id. The default is "graph_id".
        graphLabelHeader : str , optional
            The column header string used to specify the graph label. The default is "label".
        graphFeaturesHeader : str , optional
            The column header string used to specify the graph features. The default is "feat".
        edgeSRCHeader : str , optional
            The column header string used to specify the source vertex id of edges. The default is "src_id".
        edgeDSTHeader : str , optional
            The column header string used to specify the destination vertex id of edges. The default is "dst_id".
        edgeLabelHeader : str , optional
            The column header string used to specify the label of edges. The default is "label".
        edgeTrainMaskHeader : str , optional
            The column header string used to specify the train mask of edges. The default is "train_mask".
        edgeValidateMaskHeader : str , optional
            The column header string used to specify the validate mask of edges. The default is "val_mask".
        edgeTestMaskHeader : str , optional
            The column header string used to specify the test mask of edges. The default is "test_mask".
        edgeFeaturesHeader : str , optional
            The column header string used to specify the features of edges. The default is "feat".
        edgeFeaturesKeys : list , optional
            The list of dictionary keys to use to index the edge features. The length of this list must match the length of edge features. The default is [].
        nodeIDHeader : str , optional
            The column header string used to specify the id of nodes. The default is "node_id".
        nodeLabelHeader : str , optional
            The column header string used to specify the label of nodes. The default is "label".
        nodeTrainMaskHeader : str , optional
            The column header string used to specify the train mask of nodes. The default is "train_mask".
        nodeValidateMaskHeader : str , optional
            The column header string used to specify the validate mask of nodes. The default is "val_mask".
        nodeTestMaskHeader : str , optional
            The column header string used to specify the test mask of nodes. The default is "test_mask".
        nodeFeaturesHeader : str , optional
            The column header string used to specify the features of nodes. The default is "feat".
        nodeXHeader : str , optional
            The column header string used to specify the X coordinate of nodes. The default is "X".
        nodeYHeader : str , optional
            The column header string used to specify the Y coordinate of nodes. The default is "Y".
        nodeZHeader : str , optional
            The column header string used to specify the Z coordinate of nodes. The default is "Z".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        dict
            The dictionary of DGL graphs and labels found in the input CSV files. The keys in the dictionary are "graphs", "labels", "features"

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import os
        from os.path import exists, isdir
        import yaml
        import glob
        import random
        import numbers
    
        def find_yaml_files(folder_path):
            yaml_files = glob.glob(f"{folder_path}/*.yaml")
            return yaml_files

        def read_yaml(file_path):
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                edge_data = data.get('edge_data', [])
                node_data = data.get('node_data', [])
                graph_data = data.get('graph_data', {})

                edges_path = edge_data[0].get('file_name') if edge_data else None
                nodes_path = node_data[0].get('file_name') if node_data else None
                graphs_path = graph_data.get('file_name')

            return graphs_path, edges_path, nodes_path

        if not exists(path):
            print("Graph.ByCSVPath - Error: the input path parameter does not exists. Returning None.")
            return None
        if not isdir(path):
            print("Graph.ByCSVPath - Error: the input path parameter is not a folder. Returning None.")
            return None
        
        yaml_files = find_yaml_files(path)
        if len(yaml_files) < 1:
            print("Graph.ByCSVPath - Error: the input path parameter does not contain any valid YAML files. Returning None.")
            return None
        yaml_file = yaml_files[0]
        yaml_file_path = os.path.join(path, yaml_file)

        graphs_path, edges_path, nodes_path = read_yaml(yaml_file_path)
        if not graphs_path == None:
            graphs_path = os.path.join(path, graphs_path)
        if graphs_path == None:
            print("Graph.ByCSVPath - Warning: a graphs.csv file does not exist inside the folder specified by the input path parameter. Will assume the dataset includes only one graph.")
            graphs_df = pd.DataFrame()
            graph_ids=[0]
            graph_labels=[0]
            graph_features=[None]
        else:
            graphs_df = pd.read_csv(graphs_path)
            graph_ids = []
            graph_labels = []
            graph_features = []

        if not edges_path == None:
            edges_path = os.path.join(path, edges_path)
        if not exists(edges_path):
            print("Graph.ByCSVPath - Error: an edges.csv file does not exist inside the folder specified by the input path parameter. Returning None.")
            return None
        edges_path = os.path.join(path, edges_path)
        edges_df = pd.read_csv(edges_path)
        grouped_edges = edges_df.groupby(graphIDHeader)
        if not nodes_path == None:
            nodes_path = os.path.join(path, nodes_path)
        if not exists(nodes_path):
            print("Graph.ByCSVPath - Error: a nodes.csv file does not exist inside the folder specified by the input path parameter. Returning None.")
            return None
        nodes_df = pd.read_csv(nodes_path)
        # Group nodes and nodes by their 'graph_id'
        grouped_nodes = nodes_df.groupby(graphIDHeader)

        if len(nodeFeaturesKeys) == 0:
            node_keys = [nodeIDHeader, nodeLabelHeader, "mask", nodeFeaturesHeader]
        else:
            node_keys = [nodeIDHeader, nodeLabelHeader, "mask"]+nodeFeaturesKeys
        if len(edgeFeaturesKeys) == 0:
            edge_keys = [edgeLabelHeader, "mask", edgeFeaturesHeader]
        else:
            edge_keys = [edgeLabelHeader, "mask"]+edgeFeaturesKeys
        if len(graphFeaturesKeys) == 0:
            graph_keys = [graphIDHeader, graphLabelHeader, graphFeaturesHeader]
        else:
            graph_keys = [graphIDHeader, graphLabelHeader]+graphFeaturesKeys
        # Iterate through the graphs DataFrame
        for index, row in graphs_df.iterrows():
            graph_ids.append(row[graphIDHeader])
            graph_labels.append(row[graphLabelHeader])
            graph_features.append(row[graphFeaturesHeader])

        vertices_ds = [] # A list to hold the vertices data structures until we can build the actual graphs
        # Iterate through the grouped nodes DataFrames
        for graph_id, group_node_df in grouped_nodes:
            vertices = []
            verts = [] #This is a list of x, y, z tuples to make sure the vertices have unique locations.
            n_verts = 0
            for index, row in group_node_df.iterrows():
                n_verts += 1
                node_id = row[nodeIDHeader]
                label = row[nodeLabelHeader]
                train_mask = row[nodeTrainMaskHeader]
                val_mask = row[nodeValidateMaskHeader]
                test_mask = row[nodeTestMaskHeader]
                mask = 0
                if [train_mask, val_mask, test_mask] == [True, False, False]:
                    mask = 0
                elif [train_mask, val_mask, test_mask] == [False, True, False]:
                    mask = 1
                elif [train_mask, val_mask, test_mask] == [False, False, True]:
                    mask = 2
                else:
                    mask = 0
                features = row[nodeFeaturesHeader]
                x = row[nodeXHeader]
                y = row[nodeYHeader]
                z = row[nodeZHeader]
                if not isinstance(x, numbers.Number):
                    x = random.randrange(0,1000)
                if not isinstance(y, numbers.Number):
                    y = random.randrange(0,1000)
                if not isinstance(z, numbers.Number):
                    z = random.randrange(0,1000)
                while [x, y, z] in verts:
                    x = x + random.randrange(10000,30000,1000)
                    y = y + random.randrange(4000,6000, 100)
                    z = z + random.randrange(70000,90000, 1000)
                verts.append([x, y, z])
                v = Vertex.ByCoordinates(x, y, z)
                if Topology.IsInstance(v, "Vertex"):
                    if len(nodeFeaturesKeys) == 0:
                        values = [node_id, label, mask, features]
                    else:
                        values = [node_id, label, mask]
                        featureList = features.split(",")
                        featureList = [float(s) for s in featureList]
                        values = [node_id, label, mask]+featureList
                    d = Dictionary.ByKeysValues(node_keys, values)
                    if Topology.IsInstance(d, "Dictionary"):
                        v = Topology.SetDictionary(v, d)
                    else:
                        print("Graph.ByCSVPath - Warning: Failed to create and add a dictionary to the created vertex.")
                    vertices.append(v)
                else:
                    print("Graph.ByCSVPath - Warning: Failed to create and add a vertex to the list of vertices.")
            vertices_ds.append(vertices)
        edges_ds = [] # A list to hold the vertices data structures until we can build the actual graphs
        # Access specific columns within the grouped DataFrame
        for graph_id, group_edge_df in grouped_edges:
            vertices = vertices_ds[graph_id]
            edges = []
            es = [] # a list to check for duplicate edges
            duplicate_edges = 0
            for index, row in group_edge_df.iterrows():
                src_id = int(row[edgeSRCHeader])
                dst_id = int(row[edgeDSTHeader])
                label = row[nodeLabelHeader]
                train_mask = row[edgeTrainMaskHeader]
                val_mask = row[edgeValidateMaskHeader]
                test_mask = row[edgeTestMaskHeader]
                mask = 0
                if [train_mask, val_mask, test_mask] == [True, False, False]:
                    mask = 0
                elif [train_mask, val_mask, test_mask] == [False, True, False]:
                    mask = 1
                elif [train_mask, val_mask, test_mask] == [False, False, True]:
                    mask = 2
                else:
                    mask = 0
                features = row[edgeFeaturesHeader]
                if len(edgeFeaturesKeys) == 0:
                    values = [label, mask, features]
                else:
                    featureList = features.split(",")
                    featureList = [float(s) for s in featureList]
                    values = [label, mask]+featureList
                if not (src_id == dst_id) and not [src_id, dst_id] in es and not [dst_id, src_id] in es:
                    es.append([src_id, dst_id])
                    try:
                        edge = Edge.ByVertices([vertices[src_id], vertices[dst_id]], tolerance=tolerance)
                    except:
                        print("Graph.ByCSVPath - Warning: Failed to create and add a edge to the list of edges.")
                        edge = None
                    if Topology.IsInstance(edge, "Edge"):
                        d = Dictionary.ByKeysValues(edge_keys, values)
                        if Topology.IsInstance(d, "Dictionary"):
                            edge = Topology.SetDictionary(edge, d)
                        else:
                            print("Graph.ByCSVPath - Warning: Failed to create and add a dictionary to the created edge.")
                        edges.append(edge)
                    else:
                        print("Graph.ByCSVPath - Warning: Failed to create and add an edge to the list of edges.")
                else:
                    duplicate_edges += 1
            if duplicate_edges > 0:
                print("Graph.ByCSVPath - Warning: Found", duplicate_edges, "duplicate edges in graph id:", graph_id)
            edges_ds.append(edges)
        
        # Build the graphs
        graphs = []
        for i, vertices, in enumerate(vertices_ds):
            edges = edges_ds[i]
            g = Graph.ByVerticesEdges(vertices, edges)
            temp_v = Graph.Vertices(g)
            temp_e = Graph.Edges(g)
            if Topology.IsInstance(g, "Graph"):
                if len(graphFeaturesKeys) == 0:
                    values = [graph_ids[i], graph_labels[i], graph_features[i]]
                else:
                    values = [graph_ids[i], graph_labels[i]]
                    featureList = graph_features[i].split(",")
                    featureList = [float(s) for s in featureList]
                    values = [graph_ids[i], graph_labels[i]]+featureList
                l1 = len(graph_keys)
                l2 = len(values)
                if not l1 == l2:
                    print("Graph.ByCSVPath - Error: The length of the keys and values lists do not match. Returning None.")
                    return None
                d = Dictionary.ByKeysValues(graph_keys, values)
                if Topology.IsInstance(d, "Dictionary"):
                    g = Graph.SetDictionary(g, d)
                else:
                    print("Graph.ByCSVPath - Warning: Failed to create and add a dictionary to the created graph.")
                graphs.append(g)
            else:
                print("Graph.ByCSVPath - Error: Failed to create and add a graph to the list of graphs.")
        return {"graphs": graphs, "labels": graph_labels, "features": graph_features}
    
    @staticmethod
    def ByDGCNNFile(file, key: str = "label", tolerance: float = 0.0001):
        """
        Creates a graph from a DGCNN File.

        Parameters
        ----------
        file : file object
            The input file.
        key : str , optional
            The desired key for storing the node label. The default is "label".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        dict
            A dictionary with the graphs and labels. The keys are 'graphs' and 'labels'.

        """
        
        if not file:
            print("Graph.ByDGCNNFile - Error: The input file is not a valid file. Returning None.")
            return None
        dgcnn_string = file.read()
        file.close()
        return Graph.ByDGCNNString(dgcnn_string, key=key, tolerance=tolerance)
    
    @staticmethod
    def ByDGCNNPath(path, key: str = "label", tolerance: float = 0.0001):
        """
        Creates a graph from a DGCNN path.

        Parameters
        ----------
        path : str
            The input file path.
        key : str , optional
            The desired key for storing the node label. The default is "label".
        tolerance : str , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        dict
            A dictionary with the graphs and labels. The keys are 'graphs' and 'labels'.

        """
        if not path:
            print("Graph.ByDGCNNPath - Error: the input path is not a valid path. Returning None.")
            return None
        try:
            file = open(path)
        except:
            print("Graph.ByDGCNNPath - Error: the DGCNN file is not a valid file. Returning None.")
            return None
        return Graph.ByDGCNNFile(file, key=key, tolerance=tolerance)
    
    @staticmethod
    def ByDGCNNString(string, key: str ="label", tolerance: float = 0.0001):
        """
        Creates a graph from a DGCNN string.

        Parameters
        ----------
        string : str
            The input string.
        key : str , optional
            The desired key for storing the node label. The default is "label".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dict
            A dictionary with the graphs and labels. The keys are 'graphs' and 'labels'.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import random

        def verticesByCoordinates(x_coords, y_coords):
            vertices = []
            for i in range(len(x_coords)):
                vertices.append(Vertex.ByCoordinates(x_coords[i], y_coords[i], 0))
            return vertices

        graphs = []
        labels = []
        lines = string.split("\n")
        n_graphs = int(lines[0])
        index = 1
        for i in range(n_graphs):
            edges = []
            line = lines[index].split()
            n_nodes = int(line[0])
            graph_label = int(line[1])
            labels.append(graph_label)
            index+=1
            x_coordinates = random.sample(range(0, n_nodes), n_nodes)
            y_coordinates = random.sample(range(0, n_nodes), n_nodes)
            vertices = verticesByCoordinates(x_coordinates, y_coordinates)
            for j in range(n_nodes):
                line = lines[index+j].split()
                node_label = int(line[0])
                node_dict = Dictionary.ByKeysValues([key], [node_label])
                Topology.SetDictionary(vertices[j], node_dict)
            for j in range(n_nodes):
                line = lines[index+j].split()
                sv = vertices[j]
                adj_vertices = line[2:]
                for adj_vertex in adj_vertices:
                    ev = vertices[int(adj_vertex)]
                    e = Edge.ByStartVertexEndVertex(sv, ev, tolerance=tolerance)
                    edges.append(e)
            index+=n_nodes
            graphs.append(Graph.ByVerticesEdges(vertices, edges))
        return {'graphs':graphs, 'labels':labels}

    @staticmethod
    def ByIFCFile(file, includeTypes: list = [], excludeTypes: list = [],
                  includeRels: list = [], excludeRels: list = [],
                  xMin: float = -0.5, yMin: float = -0.5, zMin: float = -0.5,
                  xMax: float = 0.5, yMax: float = 0.5, zMax: float = 0.5,
                  tolerance: float = 0.0001):
        """
        Create a Graph from an IFC file. This code is partially based on code from Bruno Postle.

        Parameters
        ----------
        file : file
            The input IFC file
        includeTypes : list , optional
            A list of IFC object types to include in the graph. The default is [] which means all object types are included.
        excludeTypes : list , optional
            A list of IFC object types to exclude from the graph. The default is [] which mean no object type is excluded.
        includeRels : list , optional
            A list of IFC relationship types to include in the graph. The default is [] which means all relationship types are included.
        excludeRels : list , optional
            A list of IFC relationship types to exclude from the graph. The default is [] which mean no relationship type is excluded.
        xMin : float, optional
            The desired minimum value to assign for a vertex's X coordinate. The default is -0.5.
        yMin : float, optional
            The desired minimum value to assign for a vertex's Y coordinate. The default is -0.5.
        zMin : float, optional
            The desired minimum value to assign for a vertex's Z coordinate. The default is -0.5.
        xMax : float, optional
            The desired maximum value to assign for a vertex's X coordinate. The default is 0.5.
        yMax : float, optional
            The desired maximum value to assign for a vertex's Y coordinate. The default is 0.5.
        zMax : float, optional
            The desired maximum value to assign for a vertex's Z coordinate. The default is 0.5.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Graph
            The created graph.
        
        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        try:
            import ifcopenshell
            import ifcopenshell.util.placement
            import ifcopenshell.util.element
            import ifcopenshell.util.shape
            import ifcopenshell.geom
        except:
            print("Graph.ByIFCFile - Warning: Installing required ifcopenshell library.")
            try:
                os.system("pip install ifcopenshell")
            except:
                os.system("pip install ifcopenshell --user")
            try:
                import ifcopenshell
                import ifcopenshell.util.placement
                import ifcopenshell.util.element
                import ifcopenshell.util.shape
                import ifcopenshell.geom
                print("Graph.ByIFCFile - Warning: ifcopenshell library installed correctly.")
            except:
                warnings.warn("Graph.ByIFCFile - Error: Could not import ifcopenshell. Please try to install ifcopenshell manually. Returning None.")
                return None
        
        import random

        def vertexAtKeyValue(vertices, key, value):
            for v in vertices:
                d = Topology.Dictionary(v)
                d_value = Dictionary.ValueAtKey(d, key)
                if value == d_value:
                    return v
            return None

        def IFCObjects(ifc_file, include=[], exclude=[]):
            include = [s.lower() for s in include]
            exclude = [s.lower() for s in exclude]
            all_objects = ifc_file.by_type('IfcProduct')
            return_objects = []
            for obj in all_objects:
                is_a = obj.is_a().lower()
                if is_a in exclude:
                    continue
                if is_a in include or len(include) == 0:
                    return_objects.append(obj)
            return return_objects

        def IFCObjectTypes(ifc_file):
            products = IFCObjects(ifc_file)
            obj_types = []
            for product in products:
                obj_types.append(product.is_a())  
            obj_types = list(set(obj_types))
            obj_types.sort()
            return obj_types

        def IFCRelationshipTypes(ifc_file):
            rel_types = [ifc_rel.is_a() for ifc_rel in ifc_file.by_type("IfcRelationship")]
            rel_types = list(set(rel_types))
            rel_types.sort()
            return rel_types

        def IFCRelationships(ifc_file, include=[], exclude=[]):
            include = [s.lower() for s in include]
            exclude = [s.lower() for s in exclude]
            rel_types = [ifc_rel.is_a() for ifc_rel in ifc_file.by_type("IfcRelationship")]
            rel_types = list(set(rel_types))
            relationships = []
            for ifc_rel in ifc_file.by_type("IfcRelationship"):
                rel_type = ifc_rel.is_a().lower()
                if rel_type in exclude:
                    continue
                if rel_type in include or len(include) == 0:
                    relationships.append(ifc_rel)
            return relationships

        def vertexByIFCObject(ifc_object, object_types, restrict=False):
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS,True)
            try:
                shape = ifcopenshell.geom.create_shape(settings, ifc_object)
            except:
                shape = None
            if shape or restrict == False: #Only add vertices of entities that have 3D geometries.
                obj_id = ifc_object.id()
                psets = ifcopenshell.util.element.get_psets(ifc_object)
                obj_type = ifc_object.is_a()
                obj_type_id = object_types.index(obj_type)
                name = "Untitled"
                LongName = "Untitled"
                try:
                    name = ifc_object.Name
                except:
                    name = "Untitled"
                try:
                    LongName = ifc_object.LongName
                except:
                    LongName = name

                if name == None:
                    name = "Untitled"
                if LongName == None:
                    LongName = "Untitled"
                label = str(obj_id)+" "+LongName+" ("+obj_type+" "+str(obj_type_id)+")"
                try:
                    grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
                    vertices = [Vertex.ByCoordinates(list(coords)) for coords in grouped_verts]
                    centroid = Vertex.Centroid(vertices)
                except:
                    x = random.uniform(xMin,xMax)
                    y = random.uniform(yMin,yMax)
                    z = random.uniform(zMin,zMax)
                    centroid = Vertex.ByCoordinates(x, y, z)
                d = Dictionary.ByKeysValues(["id","psets", "type", "type_id", "name", "label"], [obj_id, psets, obj_type, obj_type_id, name, label])
                centroid = Topology.SetDictionary(centroid, d)
                return centroid
            return None

        def edgesByIFCRelationships(ifc_relationships, ifc_types, vertices):
            tuples = []
            edges = []

            for ifc_rel in ifc_relationships:
                source = None
                destinations = []
                if ifc_rel.is_a("IfcRelConnectsPorts"):
                    source = ifc_rel.RelatingPort
                    destinations = ifc_rel.RelatedPorts
                if ifc_rel.is_a("IfcRelConnectsPortToElement"):
                    source = ifc_rel.RelatingPort
                    destinations = [ifc_rel.RelatedElement]
                if ifc_rel.is_a("IfcRelAggregates"):
                    source = ifc_rel.RelatingObject
                    destinations = ifc_rel.RelatedObjects
                if ifc_rel.is_a("IfcRelNests"):
                    source = ifc_rel.RelatingObject
                    destinations = ifc_rel.RelatedObjects
                if ifc_rel.is_a("IfcRelAssignsToGroup"):
                    source = ifc_rel.RelatingGroup
                    destinations = ifc_rel.RelatedObjects
                if ifc_rel.is_a("IfcRelConnectsPathElements"):
                    source = ifc_rel.RelatingElement
                    destinations = [ifc_rel.RelatedElement]
                if ifc_rel.is_a("IfcRelConnectsStructuralMember"):
                    source = ifc_rel.RelatingStructuralMember
                    destinations = [ifc_rel.RelatedStructuralConnection]
                if ifc_rel.is_a("IfcRelContainedInSpatialStructure"):
                    source = ifc_rel.RelatingStructure
                    destinations = ifc_rel.RelatedElements
                if ifc_rel.is_a("IfcRelFillsElement"):
                    source = ifc_rel.RelatingOpeningElement
                    destinations = [ifc_rel.RelatedBuildingElement]
                if ifc_rel.is_a("IfcRelSpaceBoundary"):
                    source = ifc_rel.RelatingSpace
                    destinations = [ifc_rel.RelatedBuildingElement]
                if ifc_rel.is_a("IfcRelVoidsElement"):
                    source = ifc_rel.RelatingBuildingElement
                    destinations = [ifc_rel.RelatedOpeningElement]
                if source:
                    sv = vertexAtKeyValue(vertices, key="id", value=source.id())
                    if sv:
                        si = Vertex.Index(sv, vertices, tolerance=tolerance)
                        if not si == None:
                            for destination in destinations:
                                if destination == None:
                                    continue
                                ev = vertexAtKeyValue(vertices, key="id", value=destination.id())
                                if ev:
                                    ei = Vertex.Index(ev, vertices, tolerance=tolerance)
                                    if not ei == None:
                                        if not([si,ei] in tuples or [ei,si] in tuples):
                                            tuples.append([si,ei])
                                            e = Edge.ByVertices([sv,ev])
                                            d = Dictionary.ByKeysValues(["id", "name", "type"], [ifc_rel.id(), ifc_rel.Name, ifc_rel.is_a()])
                                            e = Topology.SetDictionary(e, d)
                                            edges.append(e)
            return edges
        
        ifc_types = IFCObjectTypes(file)
        ifc_objects = IFCObjects(file, include=includeTypes, exclude=excludeTypes)
        vertices = []
        for ifc_object in ifc_objects:
            v = vertexByIFCObject(ifc_object, ifc_types)
            if v:
                vertices.append(v)
        if len(vertices) > 0:
            ifc_relationships = IFCRelationships(file, include=includeRels, exclude=excludeRels)
            edges = edgesByIFCRelationships(ifc_relationships, ifc_types, vertices)
            g = Graph.ByVerticesEdges(vertices, edges)
        else:
            g = None
        return g

    @staticmethod
    def ByIFCPath(path, includeTypes=[], excludeTypes=[], includeRels=[], excludeRels=[], xMin=-0.5, yMin=-0.5, zMin=-0.5, xMax=0.5, yMax=0.5, zMax=0.5):
        """
        Create a Graph from an IFC path. This code is partially based on code from Bruno Postle.

        Parameters
        ----------
        path : str
            The input IFC file path.
        includeTypes : list , optional
            A list of IFC object types to include in the graph. The default is [] which means all object types are included.
        excludeTypes : list , optional
            A list of IFC object types to exclude from the graph. The default is [] which mean no object type is excluded.
        includeRels : list , optional
            A list of IFC relationship types to include in the graph. The default is [] which means all relationship types are included.
        excludeRels : list , optional
            A list of IFC relationship types to exclude from the graph. The default is [] which mean no relationship type is excluded.
        xMin : float, optional
            The desired minimum value to assign for a vertex's X coordinate. The default is -0.5.
        yMin : float, optional
            The desired minimum value to assign for a vertex's Y coordinate. The default is -0.5.
        zMin : float, optional
            The desired minimum value to assign for a vertex's Z coordinate. The default is -0.5.
        xMax : float, optional
            The desired maximum value to assign for a vertex's X coordinate. The default is 0.5.
        yMax : float, optional
            The desired maximum value to assign for a vertex's Y coordinate. The default is 0.5.
        zMax : float, optional
            The desired maximum value to assign for a vertex's Z coordinate. The default is 0.5.
        
        Returns
        -------
        topologic_core.Graph
            The created graph.
        
        """
        try:
            import ifcopenshell
            import ifcopenshell.util.placement
            import ifcopenshell.util.element
            import ifcopenshell.util.shape
            import ifcopenshell.geom
        except:
            print("Graph.ByIFCPath - Warning: Installing required ifcopenshell library.")
            try:
                os.system("pip install ifcopenshell")
            except:
                os.system("pip install ifcopenshell --user")
            try:
                import ifcopenshell
                import ifcopenshell.util.placement
                import ifcopenshell.util.element
                import ifcopenshell.util.shape
                import ifcopenshell.geom
                print("Graph.ByIFCPath - Warning: ifcopenshell library installed correctly.")
            except:
                warnings.warn("Graph.ByIFCPath - Error: Could not import ifcopenshell. Please try to install ifcopenshell manually. Returning None.")
                return None
        if not path:
            print("Graph.ByIFCPath - Error: the input path is not a valid path. Returning None.")
            return None
        ifc_file = ifcopenshell.open(path)
        if not ifc_file:
            print("Graph.ByIFCPath - Error: Could not open the IFC file. Returning None.")
            return None
        return Graph.ByIFCFile(ifc_file, includeTypes=includeTypes, excludeTypes=excludeTypes, includeRels=includeRels, excludeRels=excludeRels, xMin=xMin, yMin=yMin, zMin=zMin, xMax=xMax, yMax=yMax, zMax=zMax)

    @staticmethod
    def ByMeshData(vertices, edges, vertexDictionaries=None, edgeDictionaries=None, tolerance=0.0001):
        """
        Creates a graph from the input mesh data

        Parameters
        ----------
        vertices : list
            The list of [x, y, z] coordinates of the vertices/
        edges : list
            the list of [i, j] indices into the vertices list to signify and edge that connects vertices[i] to vertices[j].
        vertexDictionaries : list , optional
            The python dictionaries of the vertices (in the same order as the list of vertices).
        edgeDictionaries : list , optional
            The python dictionaries of the edges (in the same order as the list of edges).
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The created graph

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        g_vertices = []
        for i, v in enumerate(vertices):
            g_v = Vertex.ByCoordinates(v[0], v[1], v[2])
            if not vertexDictionaries == None:
                if isinstance(vertexDictionaries[i], dict):
                    d = Dictionary.ByPythonDictionary(vertexDictionaries[i])
                else:
                    d = vertexDictionaries[i]
                if not d == None:
                    if len(Dictionary.Keys(d)) > 0:
                        g_v = Topology.SetDictionary(g_v, d)
            g_vertices.append(g_v)
            
        g_edges = []
        for i, e in enumerate(edges):
            sv = g_vertices[e[0]]
            ev = g_vertices[e[1]]
            g_e = Edge.ByVertices([sv, ev], tolerance=tolerance)
            if not edgeDictionaries == None:
                if isinstance(edgeDictionaries[i], dict):
                    d = Dictionary.ByPythonDictionary(edgeDictionaries[i])
                else:
                    d = edgeDictionaries[i]
                if not d == None:
                    if len(Dictionary.Keys(d)) > 0:
                        g_e = Topology.SetDictionary(g_e, d)
            g_edges.append(g_e)
        return Graph.ByVerticesEdges(g_vertices, g_edges)
    
    @staticmethod
    def ByTopology(topology,
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
                   useInternalVertex: bool =True,
                   storeBREP: bool =False,
                   mantissa: int = 6,
                   tolerance: float = 0.0001):
        """
        Creates a graph.See https://en.wikipedia.org/wiki/Graph_(discrete_mathematics).

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        direct : bool , optional
            If set to True, connect the subtopologies directly with a single edge. The default is True.
        directApertures : bool , optional
            If set to True, connect the subtopologies directly with a single edge if they share one or more apertures. The default is False.
        viaSharedTopologies : bool , optional
            If set to True, connect the subtopologies via their shared topologies. The default is False.
        viaSharedApertures : bool , optional
            If set to True, connect the subtopologies via their shared apertures. The default is False.
        toExteriorTopologies : bool , optional
            If set to True, connect the subtopologies to their exterior topologies. The default is False.
        toExteriorApertures : bool , optional
            If set to True, connect the subtopologies to their exterior apertures. The default is False.
        toContents : bool , optional
            If set to True, connect the subtopologies to their contents. The default is False.
        toOutposts : bool , optional
            If set to True, connect the topology to the list specified in its outposts. The default is False.
        idKey : str , optional
            The key to use to find outpost by ID. It is case insensitive. The default is "TOPOLOGIC_ID".
        outpostsKey : str , optional
            The key to use to find the list of outposts. It is case insensitive. The default is "outposts".
        vertexCategoryKey : str , optional
            The key under which to store the node type. Node types are:
            0 : main topology
            1 : shared topology
            2 : shared aperture
            3 : exterior topology
            4 : exterior aperture
            5 : content
            6 : outpost
            The default is "node_type".
        useInternalVertex : bool , optional
            If set to True, use an internal vertex to represent the subtopology. Otherwise, use its centroid. The default is False.
        storeBREP : bool , optional
            If set to True, store the BRep of the subtopology in its representative vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The created graph.

        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Aperture import Aperture

        def mergeDictionaries(sources):
            if isinstance(sources, list) == False:
                sources = [sources]
            sinkKeys = []
            sinkValues = []
            d = sources[0].GetDictionary()
            if d != None:
                stlKeys = d.Keys()
                if len(stlKeys) > 0:
                    sinkKeys = d.Keys()
                    sinkValues = Dictionary.Values(d)
            for i in range(1,len(sources)):
                d = sources[i].GetDictionary()
                if d == None:
                    continue
                stlKeys = d.Keys()
                if len(stlKeys) > 0:
                    sourceKeys = d.Keys()
                    for aSourceKey in sourceKeys:
                        if aSourceKey not in sinkKeys:
                            sinkKeys.append(aSourceKey)
                            sinkValues.append("")
                    for i in range(len(sourceKeys)):
                        index = sinkKeys.index(sourceKeys[i])
                        sourceValue = Dictionary.ValueAtKey(d, sourceKeys[i])
                        if sourceValue != None:
                            if sinkValues[index] != "":
                                if isinstance(sinkValues[index], list):
                                    sinkValues[index].append(sourceValue)
                                else:
                                    sinkValues[index] = [sinkValues[index], sourceValue]
                            else:
                                sinkValues[index] = sourceValue
            if len(sinkKeys) > 0 and len(sinkValues) > 0:
                return Dictionary.ByKeysValues(sinkKeys, sinkValues)
            return None

        def mergeDictionaries2(sources):
            if isinstance(sources, list) == False:
                sources = [sources]
            sinkKeys = []
            sinkValues = []
            d = sources[0]
            if d != None:
                stlKeys = d.Keys()
                if len(stlKeys) > 0:
                    sinkKeys = d.Keys()
                    sinkValues = Dictionary.Values(d)
            for i in range(1,len(sources)):
                d = sources[i]
                if d == None:
                    continue
                stlKeys = d.Keys()
                if len(stlKeys) > 0:
                    sourceKeys = d.Keys()
                    for aSourceKey in sourceKeys:
                        if aSourceKey not in sinkKeys:
                            sinkKeys.append(aSourceKey)
                            sinkValues.append("")
                    for i in range(len(sourceKeys)):
                        index = sinkKeys.index(sourceKeys[i])
                        sourceValue = Dictionary.ValueAtKey(d, sourceKeys[i])
                        if sourceValue != None:
                            if sinkValues[index] != "":
                                if isinstance(sinkValues[index], list):
                                    sinkValues[index].append(sourceValue)
                                else:
                                    sinkValues[index] = [sinkValues[index], sourceValue]
                            else:
                                sinkValues[index] = sourceValue
            if len(sinkKeys) > 0 and len(sinkValues) > 0:
                return Dictionary.ByKeysValues(sinkKeys, sinkValues)
            return None
        
        def outpostsByID(topologies, ids, idKey="TOPOLOGIC_ID"):
            returnList = []
            idList = []
            for t in topologies:
                d = Topology.Dictionary(t)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == idKey.lower():
                        k = key
                if k:
                    id = Dictionary.ValueAtKey(d, k)
                else:
                    id = ""
                idList.append(id)
            for id in ids:
                try:
                    index = idList.index(id)
                except:
                    index = None
                if index:
                    returnList.append(topologies[index])
            return returnList
                
        def processCellComplex(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            edges = []
            vertices = []
            cellmat = []
            if direct == True:
                cells = []
                _ = topology.Cells(None, cells)
                # Create a matrix of zeroes
                for i in range(len(cells)):
                    cellRow = []
                    for j in range(len(cells)):
                        cellRow.append(0)
                    cellmat.append(cellRow)
                for i in range(len(cells)):
                    for j in range(len(cells)):
                        if (i != j) and cellmat[i][j] == 0:
                            cellmat[i][j] = 1
                            cellmat[j][i] = 1
                            sharedt = Topology.SharedFaces(cells[i], cells[j])
                            if len(sharedt) > 0:
                                if useInternalVertex == True:
                                    v1 = Topology.InternalVertex(cells[i], tolerance=tolerance)
                                    v2 = Topology.InternalVertex(cells[j], tolerance=tolerance)
                                else:
                                    v1 = cells[i].CenterOfMass()
                                    v2 = cells[j].CenterOfMass()
                                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                mDict = mergeDictionaries(sharedt)
                                if not mDict == None:
                                    keys = (Dictionary.Keys(mDict) or [])+["relationship"]
                                    values = (Dictionary.Values(mDict) or [])+["Direct"]
                                else:
                                    keys = ["relationship"]
                                    values = ["Direct"]
                                mDict = Dictionary.ByKeysValues(keys, values)
                                if mDict:
                                    e.SetDictionary(mDict)
                                edges.append(e)
            if directApertures == True:
                cellmat = []
                cells = []
                _ = topology.Cells(None, cells)
                # Create a matrix of zeroes
                for i in range(len(cells)):
                    cellRow = []
                    for j in range(len(cells)):
                        cellRow.append(0)
                    cellmat.append(cellRow)
                for i in range(len(cells)):
                    for j in range(len(cells)):
                        if (i != j) and cellmat[i][j] == 0:
                            cellmat[i][j] = 1
                            cellmat[j][i] = 1
                            sharedt = Topology.SharedFaces(cells[i], cells[j])
                            if len(sharedt) > 0:
                                apertureExists = False
                                for x in sharedt:
                                    apList = Topology.Apertures(x)
                                    if len(apList) > 0:
                                        apTopList = []
                                        for ap in apList:
                                            apTopList.append(ap)
                                        apertureExists = True
                                        break
                                if apertureExists:
                                    if useInternalVertex == True:
                                        v1 = Topology.InternalVertex(cells[i], tolerance=tolerance)
                                        v2 = Topology.InternalVertex(cells[j], tolerance=tolerance)
                                    else:
                                        v1 = cells[i].CenterOfMass()
                                        v2 = cells[j].CenterOfMass()
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    mDict = mergeDictionaries(apTopList)
                                    if mDict:
                                        e.SetDictionary(mDict)
                                    edges.append(e)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == outpostsKey.lower():
                        k = key
                if k:
                    ids = Dictionary.ValueAtKey(k)
                    outposts = outpostsByID(others, ids, idKey)
                else:
                    outposts = []
                for outpost in outposts:
                    if useInternalVertex == True:
                        vop = Topology.InternalVertex(outpost, tolerance=tolerance)
                        vcc = Topology.InternalVertex(topology, tolerance=tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                        vcc = Topology.CenterOfMass(topology)
                    d1 = Topology.Dictionary(vcc)
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    d1 = Topology.Dictionary(vop)
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 6) # outpost
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(outpost), Topology.Type(outpost), Topology.TypeAsString(outpost)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vop.SetDictionary(d3)
                    else:
                        _ = vop.SetDictionary(d1)
                    vertices.append(vcc)
                    vertices.append(vop)
                    tempe = Edge.ByStartVertexEndVertex(vcc, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)

            cells = []
            _ = topology.Cells(None, cells)
            if any([viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents]):
                for aCell in cells:
                    if useInternalVertex == True:
                        vCell = Topology.InternalVertex(aCell, tolerance=tolerance)
                    else:
                        vCell = aCell.CenterOfMass()
                    faces = []
                    _ = aCell.Faces(None, faces)
                    sharedTopologies = []
                    exteriorTopologies = []
                    sharedApertures = []
                    exteriorApertures = []
                    contents = []
                    _ = aCell.Contents(contents)
                    for aFace in faces:
                        cells1 = []
                        _ = aFace.Cells(topology, cells1)
                        if len(cells1) > 1:
                            sharedTopologies.append(aFace)
                            apertures = Topology.Apertures(aFace)
                            for anAperture in apertures:
                                sharedApertures.append(anAperture)
                        else:
                            exteriorTopologies.append(aFace)
                            apertures = Topology.Apertures(aFace)
                            for anAperture in apertures:
                                exteriorApertures.append(anAperture)

                    if viaSharedTopologies:
                        for sharedTopology in sharedTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedTopology, tolerance)
                            else:
                                vst = sharedTopology.CenterOfMass()
                            d1 = sharedTopology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 1) # shared topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via_Shared_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = sharedTopology.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa), Vertex.Y(vst2, mantissa=mantissa), Vertex.Z(vst2, mantissa=mantissa))
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if viaSharedApertures:
                        for sharedAp in sharedApertures:
                            if useInternalVertex == True:
                                vsa = Topology.InternalVertex(sharedAp, tolerance)
                            else:
                                vsa = sharedAp.CenterOfMass()
                            d1 = sharedAp.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 2) # shared aperture
                            vsa = Vertex.ByCoordinates(Vertex.X(vsa, mantissa=mantissa)+(tolerance*100), Vertex.Y(vsa, mantissa=mantissa)+(tolerance*100), Vertex.Z(vsa, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedAp), Topology.Type(sharedAp), Topology.TypeAsString(sharedAp)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vsa.SetDictionary(d3)
                            else:
                                _ = vsa.SetDictionary(d1)
                            vertices.append(vsa)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vsa, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via_Shared_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vet = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vet = exteriorTopology.CenterOfMass()
                            _ = vet.SetDictionary(exteriorTopology.GetDictionary())
                            d1 = exteriorTopology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 3) # exterior topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vet.SetDictionary(d3)
                            else:
                                _ = vet.SetDictionary(d1)
                            vertices.append(vet)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vet, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vet, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exTop in exteriorApertures:
                            if useInternalVertex == True:
                                vea = Topology.InternalVertex(exTop, tolerance)
                            else:
                                vea = exTop.CenterOfMass()
                            d1 = exTop.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 4) # exterior aperture
                            vea = Vertex.ByCoordinates(Vertex.X(vea, mantissa=mantissa)+(tolerance*100), Vertex.Y(vea, mantissa=mantissa)+(tolerance*100), Vertex.Z(vea, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exTop), Topology.Type(exTop), Topology.TypeAsString(exTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vea.SetDictionary(d3)
                            else:
                                _ = vea.SetDictionary(d1)
                            vertices.append(vea)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vea, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = aCell.Contents(contents)
                        for content in contents:
                            if Topology.IsInstance(content, "Aperture"):
                                content = Aperture.Topology(content)
                            if useInternalVertex == True:
                                vcn = Topology.InternalVertex(content, tolerance)
                            else:
                                vcn = content.CenterOfMass()
                            vcn = Vertex.ByCoordinates(Vertex.X(vcn, mantissa=mantissa)+(tolerance*100), Vertex.Y(vcn, mantissa=mantissa)+(tolerance*100), Vertex.Z(vcn, mantissa=mantissa)+(tolerance*100))
                            d1 = content.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vcn.SetDictionary(d3)
                            else:
                                _ = vcn.SetDictionary(d1)
                            vertices.append(vcn)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vcn, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)

            for aCell in cells:
                if useInternalVertex == True:
                    vCell = Topology.InternalVertex(aCell, tolerance=tolerance)
                else:
                    vCell = aCell.CenterOfMass()
                d1 = aCell.GetDictionary()
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                if storeBREP:
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(aCell), Topology.Type(aCell), Topology.TypeAsString(aCell)])
                    d3 = mergeDictionaries2([d1, d2])
                    _ = vCell.SetDictionary(d3)
                else:
                    _ = vCell.SetDictionary(d1)
                vertices.append(vCell)
            return [vertices,edges]

        def processCell(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            vertices = []
            edges = []
            if useInternalVertex == True:
                vCell = Topology.InternalVertex(topology, tolerance=tolerance)
            else:
                vCell = topology.CenterOfMass()
            d1 = topology.GetDictionary()
            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
            if storeBREP:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                _ = vCell.SetDictionary(d3)
            else:
                _ = vCell.SetDictionary(d1)
            vertices.append(vCell)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == outpostsKey.lower():
                        k = key
                if k:
                    ids = Dictionary.ValueAtKey(d, k)
                    outposts = outpostsByID(others, ids, idKey)
                else:
                    outposts = []
                for outpost in outposts:
                    if useInternalVertex == True:
                        vop = Topology.InternalVertex(outpost, tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                    tempe = Edge.ByStartVertexEndVertex(vCell, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
                    d1 = outpost.GetDictionary()
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 6) # outpost
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(outpost), Topology.Type(outpost), Topology.TypeAsString(outpost)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vop.SetDictionary(d3)
                    else:
                        _ = vop.SetDictionary(d1)
                    vertices.append(vop)
            if any([toExteriorTopologies, toExteriorApertures, toContents]):
                faces = Topology.Faces(topology)
                exteriorTopologies = []
                exteriorApertures = []
                for aFace in faces:
                    exteriorTopologies.append(aFace)
                    apertures = Topology.Apertures(aFace)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            d1 = exteriorTopology.GetDictionary()
                            d1 = topology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 3) # exterior topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exTop in exteriorApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exTop, tolerance)
                            else:
                                vst = exTop.CenterOfMass()
                            d1 = exTop.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 4) # exterior aperture
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exTop), Topology.Type(exTop), Topology.TypeAsString(exTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = topology.Contents(contents)
                        for content in contents:
                            if Topology.IsInstance(content, "Aperture"):
                                content = Aperture.Topology(content)
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            d1 = content.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
            return [vertices, edges]

        def processShell(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            edges = []
            vertices = []
            facemat = []
            if direct == True:
                topFaces = []
                _ = topology.Faces(None, topFaces)
                # Create a matrix of zeroes
                for i in range(len(topFaces)):
                    faceRow = []
                    for j in range(len(topFaces)):
                        faceRow.append(0)
                    facemat.append(faceRow)
                for i in range(len(topFaces)):
                    for j in range(len(topFaces)):
                        if (i != j) and facemat[i][j] == 0:
                            facemat[i][j] = 1
                            facemat[j][i] = 1
                            sharedt = Topology.SharedEdges(topFaces[i], topFaces[j])
                            if len(sharedt) > 0:
                                if useInternalVertex == True:
                                    v1 = Topology.InternalVertex(topFaces[i], tolerance=tolerance)
                                    v2 = Topology.InternalVertex(topFaces[j], tolerance=tolerance)
                                else:
                                    v1 = topFaces[i].CenterOfMass()
                                    v2 = topFaces[j].CenterOfMass()
                                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                mDict = mergeDictionaries(sharedt)
                                if not mDict == None:
                                    keys = (Dictionary.Keys(mDict) or [])+["relationship"]
                                    values = (Dictionary.Values(mDict) or [])+["Direct"]
                                else:
                                    keys = ["relationship"]
                                    values = ["Direct"]
                                mDict = Dictionary.ByKeysValues(keys, values)
                                if mDict:
                                    e.SetDictionary(mDict)
                                edges.append(e)
            if directApertures == True:
                facemat = []
                topFaces = []
                _ = topology.Faces(None, topFaces)
                # Create a matrix of zeroes
                for i in range(len(topFaces)):
                    faceRow = []
                    for j in range(len(topFaces)):
                        faceRow.append(0)
                    facemat.append(faceRow)
                for i in range(len(topFaces)):
                    for j in range(len(topFaces)):
                        if (i != j) and facemat[i][j] == 0:
                            facemat[i][j] = 1
                            facemat[j][i] = 1
                            sharedt = Topology.SharedEdges(topFaces[i], topFaces[j])
                            if len(sharedt) > 0:
                                apertureExists = False
                                for x in sharedt:
                                    apList = Topology.Apertures(x)
                                    if len(apList) > 0:
                                        apertureExists = True
                                        break
                                if apertureExists:
                                    if useInternalVertex == True:
                                        v1 = Topology.InternalVertex(topFaces[i], tolerance=tolerance)
                                        v2 = Topology.InternalVertex(topFaces[j], tolerance=tolerance)
                                    else:
                                        v1 = topFaces[i].CenterOfMass()
                                        v2 = topFaces[j].CenterOfMass()
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    mDict = mergeDictionaries(apList)
                                    if mDict:
                                        e.SetDictionary(mDict)
                                    edges.append(e)

            topFaces = []
            _ = topology.Faces(None, topFaces)
            if any([viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents == True]):
                for aFace in topFaces:
                    if useInternalVertex == True:
                        vFace = Topology.InternalVertex(aFace, tolerance=tolerance)
                    else:
                        vFace = aFace.CenterOfMass()
                    _ = vFace.SetDictionary(aFace.GetDictionary())
                    fEdges = []
                    _ = aFace.Edges(None, fEdges)
                    sharedTopologies = []
                    exteriorTopologies = []
                    sharedApertures = []
                    exteriorApertures = []
                    for anEdge in fEdges:
                        faces = []
                        _ = anEdge.Faces(topology, faces)
                        if len(faces) > 1:
                            sharedTopologies.append(anEdge)
                            apertures = Topology.Apertures(anEdge)
                            for anAperture in apertures:
                                sharedApertures.append(anAperture)
                        else:
                            exteriorTopologies.append(anEdge)
                            apertures = Topology.Apertures(anEdge)
                            for anAperture in apertures:
                                exteriorApertures.append(anAperture)
                    if viaSharedTopologies:
                        for sharedTopology in sharedTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedTopology, tolerance)
                            else:
                                vst = sharedTopology.CenterOfMass()
                            d1 = sharedTopology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 1) # shared topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via_Shared_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = sharedTopology.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if viaSharedApertures:
                        for sharedAp in sharedApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedAp, tolerance)
                            else:
                                vst = sharedAp.CenterOfMass()
                            d1 = sharedAp.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 2) # shared aperture
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedAp), Topology.Type(sharedAp), Topology.TypeAsString(sharedAp)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via_Shared_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            d1 = exteriorTopology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 3) # exterior topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) #  content
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exTop in exteriorApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exTop, tolerance)
                            else:
                                vst = exTop.CenterOfMass()
                            d1 = exTop.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 4) # exterior aperture
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exTop), Topology.Type(exTop), Topology.TypeAsString(exTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = aFace.Contents(contents)
                        for content in contents:
                            if Topology.IsInstance(content, "Aperture"):
                                content = Aperture.Topology(content)
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            d1 = content.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)

            for aFace in topFaces:
                if useInternalVertex == True:
                    vFace = Topology.InternalVertex(aFace, tolerance)
                else:
                    vFace = aFace.CenterOfMass()
                d1 = aFace.GetDictionary()
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                if storeBREP:
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(aFace), Topology.Type(aFace), Topology.TypeAsString(aFace)])
                    d3 = mergeDictionaries2([d1, d2])
                    _ = vFace.SetDictionary(d3)
                else:
                    _ = vFace.SetDictionary(d1)
                vertices.append(vFace)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == outpostsKey.lower():
                        k = key
                if k:
                    ids = Dictionary.ValueAtKey(k)
                    outposts = outpostsByID(others, ids, idKey)
                else:
                    outposts = []
                for outpost in outposts:
                    if useInternalVertex == True:
                        vop = Topology.InternalVertex(outpost, tolerance)
                        vcc = Topology.InternalVertex(topology, tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                        vcc = Topology.CenterOfMass(topology)
                    d1 = Topology.Dictionary(vcc)
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    d1 = Topology.Dictionary(vop)
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 6) # outpost
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vop.SetDictionary(d3)
                    else:
                        _ = vop.SetDictionary(d1)
                    vertices.append(vcc)
                    tempe = Edge.ByStartVertexEndVertex(vcc, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            return [vertices, edges]

        def processFace(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            vertices = []
            edges = []

            if useInternalVertex == True:
                vFace = Topology.InternalVertex(topology, tolerance=tolerance)
            else:
                vFace = topology.CenterOfMass()
            d1 = topology.GetDictionary()
            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
            if storeBREP:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                _ = vFace.SetDictionary(d3)
            else:
                _ = vFace.SetDictionary(d1)
            vertices.append(vFace)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == outpostsKey.lower():
                        k = key
                if k:
                    ids = Dictionary.ValueAtKey(d, k)
                    outposts = outpostsByID(others, ids, idKey)
                else:
                    outposts = []
                for outpost in outposts:
                    if useInternalVertex == True:
                        vop = Topology.InternalVertex(outpost, tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                    tempe = Edge.ByStartVertexEndVertex(vFace, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                fEdges = []
                _ = topology.Edges(None, fEdges)
                exteriorTopologies = []
                exteriorApertures = []

                for anEdge in fEdges:
                    exteriorTopologies.append(anEdge)
                    apertures = Topology.Apertures(anEdge)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            d1 = exteriorTopology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 3) # exterior topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exTop in exteriorApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exTop, tolerance)
                            else:
                                vst = exTop.CenterOfMass()
                            d1 = exTop.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 4) # exterior aperture
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exTop), Topology.Type(exTop), Topology.TypeAsString(exTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = topology.Contents(contents)
                        for content in contents:
                            if Topology.IsInstance(content, "Aperture"):
                                content = Aperture.Topology(content)
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            d1 = content.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
            return [vertices, edges]

        def processWire(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            edges = []
            vertices = []
            edgemat = []
            if direct == True:
                topEdges = []
                _ = topology.Edges(None, topEdges)
                # Create a matrix of zeroes
                for i in range(len(topEdges)):
                    edgeRow = []
                    for j in range(len(topEdges)):
                        edgeRow.append(0)
                    edgemat.append(edgeRow)
                for i in range(len(topEdges)):
                    for j in range(len(topEdges)):
                        if (i != j) and edgemat[i][j] == 0:
                            edgemat[i][j] = 1
                            edgemat[j][i] = 1
                            sharedt = Topology.SharedVertices(topEdges[i], topEdges[j])
                            if len(sharedt) > 0:
                                try:
                                    v1 = Edge.VertexByParameter(topEdges[i], 0.5)
                                except:
                                    v1 = topEdges[j].CenterOfMass()
                                try:
                                    v2 = Edge.VertexByParameter(topEdges[j], 0.5)
                                except:
                                    v2 = topEdges[j].CenterOfMass()
                                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                mDict = mergeDictionaries(sharedt)
                                if not mDict == None:
                                    keys = (Dictionary.Keys(mDict) or [])+["relationship"]
                                    values = (Dictionary.Values(mDict) or [])+["Direct"]
                                else:
                                    keys = ["relationship"]
                                    values = ["Direct"]
                                mDict = Dictionary.ByKeysValues(keys, values)
                                if mDict:
                                    e.SetDictionary(mDict)
                                edges.append(e)
            if directApertures == True:
                edgemat = []
                topEdges = []
                _ = topology.Edges(None, topEdges)
                # Create a matrix of zeroes
                for i in range(len(topEdges)):
                    edgeRow = []
                    for j in range(len(topEdges)):
                        edgeRow.append(0)
                    edgemat.append(edgeRow)
                for i in range(len(topEdges)):
                    for j in range(len(topEdges)):
                        if (i != j) and edgemat[i][j] == 0:
                            edgemat[i][j] = 1
                            edgemat[j][i] = 1
                            sharedt = Topology.SharedVertices(topEdges[i], topEdges[j])
                            if len(sharedt) > 0:
                                apertureExists = False
                                for x in sharedt:
                                    apList = Topology.Apertures(x)
                                    if len(apList) > 0:
                                        apertureExists = True
                                        break
                                if apertureExists:
                                    try:
                                        v1 = Edge.VertexByParameter(topEdges[i], 0.5)
                                    except:
                                        v1 = topEdges[j].CenterOfMass()
                                    try:
                                        v2 = Edge.VertexByParameter(topEdges[j], 0.5)
                                    except:
                                        v2 = topEdges[j].CenterOfMass()
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    mDict = mergeDictionaries(apList)
                                    if mDict:
                                        e.SetDictionary(mDict)
                                    edges.append(e)

            topEdges = []
            _ = topology.Edges(None, topEdges)
            if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                for anEdge in topEdges:
                    try:
                        vEdge = Edge.VertexByParameter(anEdge, 0.5)
                    except:
                        vEdge = anEdge.CenterOfMass()
                    # d1 = anEdge.GetDictionary()
                    # d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                    # if storeBREP:
                    #     d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(anEdge), Topology.Type(anEdge), Topology.TypeAsString(anEdge)])
                    #     d3 = mergeDictionaries2([d1, d2])
                    #     _ = vEdge.SetDictionary(d3)
                    # else:
                    #     _ = vEdge.SetDictionary(d1)
                    # vertices.append(vEdge)
                    eVertices = []
                    _ = anEdge.Vertices(None, eVertices)
                    sharedTopologies = []
                    exteriorTopologies = []
                    sharedApertures = []
                    exteriorApertures = []
                    contents = []
                    _ = anEdge.Contents(contents)
                    for aVertex in eVertices:
                        tempEdges = []
                        _ = aVertex.Edges(topology, tempEdges)
                        if len(tempEdges) > 1:
                            sharedTopologies.append(aVertex)
                            apertures = Topology.Apertures(aVertex)
                            for anAperture in apertures:
                                sharedApertures.append(anAperture)
                        else:
                            exteriorTopologies.append(aVertex)
                            apertures = Topology.Apertures(aVertex)
                            for anAperture in apertures:
                                exteriorApertures.append(anAperture)
                    if viaSharedTopologies:
                        for sharedTopology in sharedTopologies:
                            vst = sharedTopology.CenterOfMass()
                            d1 = sharedTopology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 1) # shared topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via_Shared_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = sharedTopology.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if viaSharedApertures:
                        for sharedAp in sharedApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedAp, tolerance)
                            else:
                                vst = sharedAp.CenterOfMass()
                            d1 = sharedAp.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 2) # shared aperture
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedAp), Topology.Type(sharedAp), Topology.TypeAsString(sharedAp)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via_Shared_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            vst = exteriorTopology
                            vertices.append(exteriorTopology)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = vst.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exTop in exteriorApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exTop, tolerance)
                            else:
                                vst = exTop.CenterOfMass()
                            d1 = exTop.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 4) # exterior aperture
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exTop), Topology.Type(exTop), Topology.TypeAsString(exTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = anEdge.Contents(contents)
                        for content in contents:
                            if Topology.IsInstance(content, "Aperture"):
                                content = Aperture.Topology(content)
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            d1 = content.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
            for anEdge in topEdges:
                try:
                    vEdge = Edge.VertexByParameter(anEdge, 0.5)
                except:
                    vEdge = anEdge.CenterOfMass()
                d1 = anEdge.GetDictionary()
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topologuy
                if storeBREP:
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(anEdge), Topology.Type(anEdge), Topology.TypeAsString(anEdge)])
                    d3 = mergeDictionaries2([d1, d2])
                    _ = vEdge.SetDictionary(d3)
                else:
                    _ = vEdge.SetDictionary(d1)
                vertices.append(vEdge)
            
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == outpostsKey.lower():
                        k = key
                if k:
                    ids = Dictionary.ValueAtKey(k)
                    outposts = outpostsByID(others, ids, idKey)
                else:
                    outposts = []
                for outpost in outposts:
                    if useInternalVertex == True:
                        vop = Topology.InternalVertex(outpost, tolerance)
                        vcc = Topology.InternalVertex(topology, tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                        vcc = Topology.CenterOfMass(topology)
                    d1 = Topology.Dictionary(vcc)
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    d1 = Topology.Dictionary(vop)
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 6) # outpost
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vop.SetDictionary(d3)
                    else:
                        _ = vop.SetDictionary(d1)
                    vertices.append(vop)
                    tempe = Edge.ByStartVertexEndVertex(vcc, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            return [vertices, edges]

        def processEdge(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            vertices = []
            edges = []

            if useInternalVertex == True:
                try:
                    vEdge = Edge.VertexByParameter(topology, 0.5)
                except:
                    vEdge = topology.CenterOfMass()
            else:
                vEdge = topology.CenterOfMass()

            d1 = vEdge.GetDictionary()
            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
            if storeBREP:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                _ = vEdge.SetDictionary(d3)
            else:
                _ = vEdge.SetDictionary(topology.GetDictionary())

            vertices.append(vEdge)

            if toOutposts and others:
                d = Topology.Dictionary(topology)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == outpostsKey.lower():
                        k = key
                if k:
                    ids = Dictionary.ValueAtKey(d, k)
                    outposts = outpostsByID(others, ids, idKey)
                else:
                    outposts = []
                for outpost in outposts:
                    if useInternalVertex == True:
                        vop = Topology.InternalVertex(outpost, tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                    tempe = Edge.ByStartVertexEndVertex(vEdge, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                eVertices = []
                _ = topology.Vertices(None, eVertices)
                exteriorTopologies = []
                exteriorApertures = []
                for aVertex in eVertices:
                    exteriorTopologies.append(aVertex)
                    apertures = Topology.Apertures(aVertex)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            d1 = exteriorTopology.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 3) # exterior topology
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = vst.Contents(contents)
                                for content in contents:
                                    if Topology.IsInstance(content, "Aperture"):
                                        content = Aperture.Topology(content)
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = Vertex.ByCoordinates(Vertex.X(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst2, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst2, mantissa=mantissa)+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                                    if storeBREP:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exTop in exteriorApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exTop, tolerance)
                            else:
                                vst = exTop.CenterOfMass()
                            d1 = exTop.GetDictionary()
                            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 4) # exterior aperture
                            vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                            if storeBREP:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exTop), Topology.Type(exTop), Topology.TypeAsString(exTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            _ = vst.SetDictionary(exTop.GetDictionary())
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To_Exterior_Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    
            return [vertices, edges]

        def processVertex(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            vertices = [topology]
            edges = []

            if toContents:
                contents = []
                _ = topology.Contents(contents)
                for content in contents:
                    if Topology.IsInstance(content, "Aperture"):
                        content = Aperture.Topology(content)
                    if useInternalVertex == True:
                        vst = Topology.InternalVertex(content, tolerance)
                    else:
                        vst = content.CenterOfMass()
                    d1 = content.GetDictionary()
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                    vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vst.SetDictionary(d3)
                    else:
                        _ = vst.SetDictionary(d1)
                    vertices.append(vst)
                    tempe = Edge.ByStartVertexEndVertex(topology, vst, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Contents"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                if not d == None:
                    keys = Dictionary.Keys(d)
                else:
                    keys = []
                k = None
                for key in keys:
                    if key.lower() == outpostsKey.lower():
                        k = key
                if k:
                    ids = Dictionary.ValueAtKey(d, k)
                    outposts = outpostsByID(others, ids, idKey)
                else:
                    outposts = []
                for outpost in outposts:
                    if useInternalVertex == True:
                        vop = Topology.InternalVertex(outpost, tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                    tempe = Edge.ByStartVertexEndVertex(topology, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To_Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            return [vertices, edges]

        
        if not Topology.IsInstance(topology, "Topology"):
            print("Graph.ByTopology - Error: The input topology is not a valid topology. Returning None.")
            return None
        item = [topology, None, None, None, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, None, useInternalVertex, storeBREP, tolerance]
        vertices = []
        edges = []
        if Topology.IsInstance(topology, "CellComplex"):
            vertices, edges = processCellComplex(item)
        elif Topology.IsInstance(topology, "Cell"):
            vertices, edges = processCell(item)
        elif Topology.IsInstance(topology, "Shell"):
            vertices, edges = processShell(item)
        elif Topology.IsInstance(topology, "Face"):
            vertices, edges = processFace(item)
        elif Topology.IsInstance(topology, "Wire"):
            vertices, edges = processWire(item)
        elif Topology.IsInstance(topology, "Edge"):
            vertices, edges = processEdge(item)
        elif Topology.IsInstance(topology, "Vertex"):
            vertices, edges = processVertex(item)
        elif Topology.IsInstance(topology, "Cluster"):
            c_cellComplexes = Topology.CellComplexes(topology)
            c_cells = Cluster.FreeCells(topology, tolerance=tolerance)
            c_shells = Cluster.FreeShells(topology, tolerance=tolerance)
            c_faces = Cluster.FreeFaces(topology, tolerance=tolerance)
            c_wires = Cluster.FreeWires(topology, tolerance=tolerance)
            c_edges = Cluster.FreeEdges(topology, tolerance=tolerance)
            c_vertices = Cluster.FreeVertices(topology, tolerance=tolerance)
            others = c_cellComplexes+c_cells+c_shells+c_faces+c_wires+c_edges+c_vertices
            parameters = [others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance]

            for t in c_cellComplexes:
                v, e = processCellComplex([t]+parameters)
                vertices += v
                edges += e
            for t in c_cells:
                v, e = processCell([t]+parameters)
                vertices += v
                edges += e
            for t in c_shells:
                v, e = processShell([t]+parameters)
                vertices += v
                edges += e
            for t in c_faces:
                v, e = processFace([t]+parameters)
                vertices += v
                edges += e
            for t in c_wires:
                v, e = processWire([t]+parameters)
                vertices += v
                edges += e
            for t in c_edges:
                v, e = processEdge([t]+parameters)
                vertices += v
                edges += e
            for t in c_vertices:
                v, e = processVertex([t]+parameters)
                vertices += v
                edges += e
        else:
            return None
        return Graph.ByVerticesEdges(vertices, edges)
    
    @staticmethod
    def ByVerticesEdges(vertices, edges):
        """
        Creates a graph from the input list of vertices and edges.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        edges : list
            The input list of edges.

        Returns
        -------
        topologic_core.Graph
            The created graph.

        """
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            print("Graph.ByVerticesEdges - Error: The input list of vertices is not a valid list. Returning None.")
            return None
        if not isinstance(edges, list):
            print("Graph.ByVerticesEdges - Error: The input list of edges is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        edges = [e for e in edges if Topology.IsInstance(e, "Edge")]
        return topologic.Graph.ByVerticesEdges(vertices, edges) # Hook to Core
    
    @staticmethod
    def ChromaticNumber(graph, maxColors: int = 3, silent: bool = False):
        """
        Returns the chromatic number of the input graph. See https://en.wikipedia.org/wiki/Graph_coloring.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        maxColors : int , optional
            The desired maximum number of colors to test against. The default is 3.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
       
        Returns
        -------
        int
            The chromatic number of the input graph.

        """
        # This is based on code from https://www.geeksforgeeks.org/graph-coloring-applications/
        
        from topologicpy.Topology import Topology

        def is_safe(graph, v, color, c):
            for i in range(len(graph)):
                if graph[v][i] == 1 and color[i] == c:
                    return False
            return True

        def graph_coloring(graph, m, color, v):
            V = len(graph)
            if v == V:
                return True

            for c in range(1, m + 1):
                if is_safe(graph, v, color, c):
                    color[v] = c
                    if graph_coloring(graph, m, color, v + 1):
                        return True
                    color[v] = 0

            return False

        def chromatic_number(graph):
            V = len(graph)
            color = [0] * V
            m = 1

            while True:
                if graph_coloring(graph, m, color, 0):
                    return m
                m += 1
        
        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.ChromaticNumber - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if maxColors < 1:
            if not silent:
                print("Graph.ChromaticNumber - Error: The input maxColors parameter is not a valid positive number. Returning None.")
            return None
        adj_matrix = Graph.AdjacencyMatrix(graph)
        return chromatic_number(adj_matrix)

    @staticmethod
    def Color(graph, oldKey: str = "color", key: str = "color", maxColors: int = None, tolerance: float = 0.0001):
        """
        Colors the input vertices within the input graph. The saved value is an integer rather than an actual color. See Color.ByValueInRange to convert to an actual color.
        Any vertices that have been pre-colored will not be affected. See https://en.wikipedia.org/wiki/Graph_coloring.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        oldKey : str , optional
            The existing dictionary key to use to read any pre-existing color information. The default is "color".
        key : str , optional
            The new dictionary key to use to write out new color information. The default is "color".
        maxColors : int , optional
            The desired maximum number of colors to use. If set to None, the chromatic number of the graph is used. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The input graph, but with its vertices colored.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import math

        def is_safe(v, graph, colors, c):
            # Check if the color 'c' is safe for the vertex 'v'
            for i in range(len(graph)):
                if graph[v][i] and c == colors[i]:
                    return False
            return True

        def graph_coloring_util(graph, m, colors, v):
            # Base case: If all vertices are assigned a color, return true
            if v == len(graph):
                return True

            # Try different colors for the current vertex 'v'
            for c in range(1, m + 1):
                # Check if assignment of color 'c' to 'v' is fine
                if is_safe(v, graph, colors, c):
                    colors[v] = c

                    # Recur to assign colors to the rest of the vertices
                    if graph_coloring_util(graph, m, colors, v + 1):
                        return True

                    # If assigning color 'c' doesn't lead to a solution, remove it
                    colors[v] = 0

            # If no color can be assigned to this vertex, return false
            return False

        def graph_coloring(graph, m, colors):

            # Call graph_coloring_util() for vertex 0
            if not graph_coloring_util(graph, m, colors, 0):
                return None
            return [x-1 for x in colors]

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Color - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        vertices = Graph.Vertices(graph)
        adj_mat = Graph.AdjacencyMatrix(graph)
        # Isolate vertices that have pre-existing colors as they shouldn't affect graph coloring.
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            c = Dictionary.ValueAtKey(d, oldKey)
            if not c == None:
                adj_mat[i] = [0] * len(vertices)
                for j in range(len(adj_mat)):
                    row = adj_mat[j]
                    row[i] = 0
        temp_graph = Graph.ByAdjacencyMatrix(adj_mat)
        # If the maximum number of colors are not provided, compute it using the graph's chromatic number.
        if maxColors == None:
            maxColors = Graph.ChromaticNumber(temp_graph)
        colors = [0] * len(vertices)
        colors = graph_coloring(adj_mat, maxColors, colors)
        for i, v in enumerate(vertices):
                d = Topology.Dictionary(v)
                d = Dictionary.SetValueAtKey(d, key, colors[i])
                v = Topology.SetDictionary(v, d)
        return graph
    
    @staticmethod
    def ContractEdge(graph, edge, vertex=None, tolerance=0.0001):
        """
        Contracts the input edge in the input graph into a single vertex. Please note that the dictionary of the edge is transferred to the
        vertex that replaces it. See https://en.wikipedia.org/wiki/Edge_contraction

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        edge : topologic_core.Edge
            The input graph edge that needs to be contracted.
        vertex : topollogic.Vertex , optional
            The vertex to replace the contracted edge. If set to None, the centroid of the edge is chosen. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The input graph, but with input edge contracted into a single vertex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def OppositeVertex(edge, vertex, tolerance=0.0001):
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            d1 = Vertex.Distance(vertex, sv)
            d2 = Vertex.Distance(vertex, ev)
            if d1 < d2:
                return [ev, 1]
            return [sv, 0]
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ContractEdge - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(edge, "Edge"):
            print("Graph.ContractEdge - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        if vertex == None:
            vertex = Topology.Centroid(edge)
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        vd = Topology.Dictionary(vertex)
        sd = Topology.Dictionary(sv)
        dictionaries = []
        keys = Dictionary.Keys(vd)
        if isinstance(keys, list):
            if len(keys) > 0:
                dictionaries.append(vd)
        keys = Dictionary.Keys(sd)
        if isinstance(keys, list):
            if len(keys) > 0:
                dictionaries.append(sd)
        ed = Topology.Dictionary(ev)
        keys = Dictionary.Keys(ed)
        if isinstance(keys, list):
            if len(keys) > 0:
                dictionaries.append(ed)
        if len(dictionaries) == 1:
            vertex = Topology.SetDictionary(vertex, dictionaries[0])
        elif len(dictionaries) > 1:
            cd = Dictionary.ByMergedDictionaries(dictionaries)
            vertex = Topology.SetDictionary(vertex, cd)
        graph = Graph.RemoveEdge(graph, edge, tolerance=tolerance)
        graph = Graph.AddVertex(graph, vertex, tolerance=tolerance)
        adj_edges_sv = Graph.Edges(graph, [sv])
        adj_edges_ev = Graph.Edges(graph, [ev])
        new_edges = []
        for adj_edge_sv in adj_edges_sv:
            ov, flag = OppositeVertex(adj_edge_sv, sv)
            if flag == 0:
                new_edge = Edge.ByVertices([ov, vertex])
            else:
                new_edge = Edge.ByVertices([vertex, ov])
            d = Topology.Dictionary(adj_edge_sv)
            keys = Dictionary.Keys(d)
            if isinstance(keys, list):
                if len(keys) > 0:
                    new_edge = Topology.SetDictionary(new_edge, d)
            new_edges.append(new_edge)
        for adj_edge_ev in adj_edges_ev:
            ov, flag = OppositeVertex(adj_edge_ev, ev)
            if flag == 0:
                new_edge = Edge.ByVertices([ov, vertex])
            else:
                new_edge = Edge.ByVertices([vertex, ov])
            d = Topology.Dictionary(adj_edge_ev)
            keys = Dictionary.Keys(d)
            if isinstance(keys, list):
                if len(keys) > 0:
                    new_edge = Topology.SetDictionary(new_edge, d)
            new_edges.append(new_edge)
        for new_edge in new_edges:
            graph = Graph.AddEdge(graph, new_edge, transferVertexDictionaries=True, transferEdgeDictionaries=True, tolerance=tolerance)
        graph = Graph.RemoveVertex(graph,sv, tolerance=tolerance)
        graph = Graph.RemoveVertex(graph,ev, tolerance=tolerance)
        return graph
    
    @staticmethod
    def ClosenessCentrality(graph, vertices=None, key: str = "closeness_centrality", mantissa: int = 6, tolerance = 0.0001):
        """
        Return the closeness centrality measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the closeness centrality of all the vertices in the input graph is computed. See https://en.wikipedia.org/wiki/Closeness_centrality.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. The default is None.
        key : str , optional
            The dictionary key under which to save the closeness centrality score. The default is "closeness_centrality".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The closeness centrality of the input list of vertices within the input graph. The values are in the range 0 to 1.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ClosenessCentrality - Error: The input graph is not a valid graph. Returning None.")
            return None
        graphVertices = Graph.Vertices(graph)
        if not isinstance(vertices, list):
            vertices = graphVertices
        else:
            vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Graph.ClosenessCentrality - Error: The input list of vertices does not contain any valid vertices. Returning None.")
            return None
        n = len(graphVertices)

        scores = []
        try:
            for va in tqdm(vertices, desc="Computing Closeness Centrality", leave=False):
                top_dist = 0
                for vb in graphVertices:
                    if Topology.IsSame(va, vb):
                        d = 0
                    else:
                        d = Graph.TopologicalDistance(graph, va, vb, tolerance)
                    top_dist += d
                if top_dist == 0:
                    print("Topological Distance is Zero!!")
                    scores.append(0)
                else:
                    scores.append(round((n-1)/top_dist, mantissa))
        except:
            print("Graph.ClosenessCentrality - Warning: Could not use tqdm.")
            for va in vertices:
                top_dist = 0
                for vb in graphVertices:
                    if Topology.IsSame(va, vb):
                        d = 0
                    else:
                        d = Graph.TopologicalDistance(graph, va, vb, tolerance)
                    top_dist += d
                if top_dist == 0:
                    scores.append(0)
                else:
                    scores.append(round((n-1)/top_dist, mantissa))
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, key, scores[i])
            v = Topology.SetDictionary(v, d)
        return scores

    @staticmethod
    def Connect(graph, verticesA, verticesB, tolerance=0.0001):
        """
        Connects the two lists of input vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        verticesA : list
            The first list of input vertices.
        verticesB : topologic_core.Vertex
            The second list of input vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The input graph with the connected input vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Connect - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(verticesA, list):
            print("Graph.Connect - Error: The input list of verticesA is not a valid list. Returning None.")
            return None
        if not isinstance(verticesB, list):
            print("Graph.Connect - Error: The input list of verticesB is not a valid list. Returning None.")
            return None
        verticesA = [v for v in verticesA if Topology.IsInstance(v, "Vertex")]
        verticesB = [v for v in verticesB if Topology.IsInstance(v, "Vertex")]
        if len(verticesA) < 1:
            print("Graph.Connect - Error: The input list of verticesA does not contain any valid vertices. Returning None.")
            return None
        if len(verticesB) < 1:
            print("Graph.Connect - Error: The input list of verticesB does not contain any valid vertices. Returning None.")
            return None
        if not len(verticesA) == len(verticesB):
            print("Graph.Connect - Error: The input lists verticesA and verticesB have different lengths. Returning None.")
            return None
        _ = graph.Connect(verticesA, verticesB, tolerance)
        return graph
    
    @staticmethod
    def ContainsEdge(graph, edge, tolerance=0.0001):
        """
        Returns True if the input graph contains the input edge. Returns False otherwise.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input graph contains the input edge. False otherwise.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ContainsEdge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(edge, "Edge"):
            print("Graph.ContainsEdge - Error: The input edge is not a valid edge. Returning None.")
            return None
        return graph.ContainsEdge(edge, tolerance)
    
    @staticmethod
    def ContainsVertex(graph, vertex, tolerance=0.0001):
        """
        Returns True if the input graph contains the input Vertex. Returns False otherwise.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input Vertex.
        tolerance : float , optional
            Ther desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input graph contains the input vertex. False otherwise.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ContainsVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.ContainsVertex - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        return graph.ContainsVertex(vertex, tolerance)


    @staticmethod
    def Degree(graph, vertices=None, key: str = "degree", edgeKey: str = None, mantissa: int = 6, tolerance = 0.0001):
        """
        Return the degree measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the closeness centrality of all the vertices in the input graph is computed. See https://en.wikipedia.org/wiki/Degree_(graph_theory).

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. The default is None.
        key : str , optional
            The dictionary key under which to save the closeness centrality score. The default is "degree".
        edgeKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the edgeKey string will be aggregated to calculate
            the vertex degree. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead. This is used in weighted graphs.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The degree of the input list of vertices within the input graph.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ClosenessCentrality - Error: The input graph is not a valid graph. Returning None.")
            return None
        graphVertices = Graph.Vertices(graph)
        if not isinstance(vertices, list):
            vertices = graphVertices
        else:
            vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Graph.Degree - Error: The input list of vertices does not contain any valid vertices. Returning None.")
            return None
        n = len(graphVertices)

        scores = []
        for i, v in enumerate(vertices):
            degree = Graph.VertexDegree(graph, v, edgeKey= edgeKey, tolerance = tolerance)
            if isinstance(degree, float):
                degree = round(degree, mantissa)
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, key, degree)
            v = Topology.SetDictionary(v, d)
            scores.append(degree)
        return scores
    
    @staticmethod
    def DegreeSequence(graph):
        """
        Returns the degree sequence of the input graph. See https://mathworld.wolfram.com/DegreeSequence.html.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        list
            The degree sequence of the input graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.DegreeSequence - Error: The input graph is not a valid graph. Returning None.")
            return None
        sequence = []
        _ = graph.DegreeSequence(sequence)
        return sequence
    
    @staticmethod
    def Density(graph):
        """
        Returns the density of the input graph. See https://en.wikipedia.org/wiki/Dense_graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        float
            The density of the input graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Density - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.Density()
    
    @staticmethod
    def DepthMap(graph, vertices=None, key: str = "depth", type: str = "topological", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Return the depth map of the input list of vertices within the input graph. The returned list contains the total of the topological distances of each vertex to every other vertex in the input graph. The order of the depth map list is the same as the order of the input list of vertices. If no vertices are specified, the depth map of all the vertices in the input graph is computed.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. The default is None.
        key : str , optional
            The dictionary key under which to save the depth score. The default is "depth".
        type : str , optional
            The type of depth distance to calculate. The options are "topological" or "metric". The default is "topological". See https://www.spacesyntax.online/overview-2/analysis-of-spatial-relations/.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The depth map of the input list of vertices within the input graph.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.DepthMap - Error: The input graph is not a valid graph. Returning None.")
            return None
        graphVertices = Graph.Vertices(graph)
        if not isinstance(vertices, list):
            vertices = graphVertices
        else:
            vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Graph.DepthMap - Error: The input list of vertices does not contain any valid vertices. Returning None.")
            return None
        scores = []
        for va in vertices:
            depth = 0
            for vb in graphVertices:
                if Topology.IsSame(va, vb):
                    dist = 0
                else:
                    dist = Graph.Distance(graph, va, vb, type=type, mantissa=mantissa, tolerance=tolerance)
                depth = depth + dist
            depth = round(depth, mantissa)
            d = Topology.Dictionary(va)
            d = Dictionary.SetValueAtKey(d, key, depth)
            va = Topology.SetDictionary(va, d)
            scores.append(depth)
        return scores
    
    @staticmethod
    def Diameter(graph):
        """
        Returns the diameter of the input graph. See https://mathworld.wolfram.com/GraphDiameter.html.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        int
            The diameter of the input graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Diameter - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.Diameter()
    
    @staticmethod
    def Dictionary(graph):
        """
        Returns the dictionary of the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        topologic_core.Dictionary
            The dictionary of the input graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Dictionary - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        return graph.GetDictionary()
    
    @staticmethod
    def Distance(graph, vertexA, vertexB, type: str = "topological", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the shortest-path distance between the input vertices. See https://en.wikipedia.org/wiki/Distance_(graph_theory).

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        type : str , optional
            The type of depth distance to calculate. The options are "topological" or "metric". The default is "topological". See https://www.spacesyntax.online/overview-2/analysis-of-spatial-relations/.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        float
            The shortest-path metric distance between the input vertices.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Distance - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.Distance - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.Distance - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        
        if "topo" in type.lower():
            return Graph.TopologicalDistance(graph, vertexA, vertexB, tolerance=tolerance)
        return Graph.MetricDistance(graph, vertexA, vertexB, mantissa=mantissa, tolerance=tolerance)
    
    @staticmethod
    def Edge(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Returns the edge in the input graph that connects in the input vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input Vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Edge
            The edge in the input graph that connects the input vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Edge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.Edge - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.Edge - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        return graph.Edge(vertexA, vertexB, tolerance)
    
    @staticmethod
    def Edges(graph, vertices=None, tolerance=0.0001):
        """
        Returns the edges found in the input graph. If the input list of vertices is specified, this method returns the edges connected to this list of vertices. Otherwise, it returns all graph edges.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            An optional list of vertices to restrict the returned list of edges only to those connected to this list.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of edges in the graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Edges - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not vertices:
            edges = []
            _ = graph.Edges(edges, tolerance)
            return edges
        else:
            vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Graph.Edges - Error: The input list of vertices does not contain any valid vertices. Returning None.")
            return None
        edges = []
        _ = graph.Edges(vertices, tolerance, edges)
        return list(dict.fromkeys(edges)) # remove duplicates
    
    @staticmethod
    def ExportToAdjacencyMatrixCSV(adjacencyMatrix, path):
        """
        Exports the input graph into a set of CSV files compatible with DGL.

        Parameters
        ----------
        adjacencyMatrix: list
            The input adjacency matrix.
        path : str
            The desired path to the output folder where the graphs, edges, and nodes CSV files will be saved.

        Returns
        -------
        bool
            True if the graph has been successfully exported. False otherwise.

        """
        
        # Convert the adjacency matrix (nested list) to a DataFrame
        adjacency_matrix_df = pd.DataFrame(adjacencyMatrix)

        # Export the DataFrame to a CSV file
        try:
            adjacency_matrix_df.to_csv(path, index=False, header=False)
            return True
        except:
            return False

    @staticmethod
    def ExportToBOT(graph,
                    path: str,
                    format: str = "turtle",
                    overwrite: bool = False,
                    bidirectional: bool = False,
                    includeAttributes: bool = False,
                    includeLabel: bool = False,
                    includeGeometry: bool = False,
                    siteLabel: str = "Site_0001",
                    siteDictionary: dict = None,
                    buildingLabel: str = "Building_0001",
                    buildingDictionary: dict = None , 
                    storeyPrefix: str = "Storey",
                    floorLevels: list = [],
                    vertexLabelKey: str = "label",
                    typeKey: str = "type",
                    verticesKey: str = "vertices",
                    edgesKey: str = "edges",
                    edgeLabelKey: str = "",
                    sourceKey: str = "source",
                    targetKey: str = "target",
                    xKey: str = "hasX",
                    yKey: str = "hasY",
                    zKey: str = "hasZ",
                    geometryKey: str = "brep",
                    spaceType: str = "space",
                    wallType: str = "wall",
                    slabType: str = "slab",
                    doorType: str = "door",
                    windowType: str = "window",
                    contentType: str = "content",
                    namespace: str = "http://github.com/wassimj/topologicpy/resources",
                    mantissa: int = 6
                    ):
        
        """
        Exports the input graph to an RDF graph serialized according to the BOT ontology. See https://w3c-lbd-cg.github.io/bot/.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        path : str
            The desired path to where the RDF/BOT file will be saved.
        format : str , optional
            The desired output format, the options are listed below. Thde default is "turtle".
            turtle, ttl or turtle2 : Turtle, turtle2 is just turtle with more spacing & linebreaks
            xml or pretty-xml : RDF/XML, Was the default format, rdflib < 6.0.0
            json-ld : JSON-LD , There are further options for compact syntax and other JSON-LD variants
            ntriples, nt or nt11 : N-Triples , nt11 is exactly like nt, only utf8 encoded
            n3 : Notation-3 , N3 is a superset of Turtle that also caters for rules and a few other things
            trig : Trig , Turtle-like format for RDF triples + context (RDF quads) and thus multiple graphs
            trix : Trix , RDF/XML-like format for RDF quads
            nquads : N-Quads , N-Triples-like format for RDF quads
        overwrite : bool , optional
            If set to True, any existing file is overwritten. Otherwise, it is not. The default is False.
        bidirectional : bool , optional
            If set to True, reverse relationships are created wherever possible. Otherwise, they are not. The default is False.
        includeAttributes : bool , optional
            If set to True, the attributes associated with vertices in the graph are written out. Otherwise, they are not. The default is False.
        includeLabel : bool , optional
            If set to True, a label is attached to each node. Otherwise, it is not. The default is False.
        includeGeometry : bool , optional
            If set to True, the geometry associated with vertices in the graph are written out. Otherwise, they are not. The default is False.
        siteLabel : str , optional
            The desired site label. The default is "Site_0001".
        siteDictionary : dict , optional
            The dictionary of site attributes to include in the output. The default is None.
        buildingLabel : str , optional
            The desired building label. The default is "Building_0001".
        buildingDictionary : dict , optional
            The dictionary of building attributes to include in the output. The default is None.
        storeyPrefix : str , optional
            The desired prefixed to use for each building storey. The default is "Storey".
        floorLevels : list , optional
            The list of floor levels. This should be a numeric list, sorted from lowest to highest.
            If not provided, floorLevels will be computed automatically based on the vertices' (zKey)) attribute. See below.
        verticesKey : str , optional
            The desired key name to call vertices. The default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. The default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. The default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. The default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. The default is "hasX".
        yKey : str , optional
            The desired key name to use for y-coordinates. The default is "hasY".
        zKey : str , optional
            The desired key name to use for z-coordinates. The default is "hasZ".
        geometryKey : str , optional
            The desired key name to use for geometry. The default is "brep".
        typeKey : str , optional
            The dictionary key to use to look up the type of the node. The default is "type".
        geometryKey : str , optional
            The dictionary key to use to look up the geometry of the node. The default is "brep".
        spaceType : str , optional
            The dictionary string value to use to look up vertices of type "space". The default is "space".
        wallType : str , optional
            The dictionary string value to use to look up vertices of type "wall". The default is "wall".
        slabType : str , optional
            The dictionary string value to use to look up vertices of type "slab". The default is "slab".
        doorType : str , optional
            The dictionary string value to use to look up vertices of type "door". The default is "door".
        windowType : str , optional
            The dictionary string value to use to look up vertices of type "window". The default is "window".
        contentType : str , optional
            The dictionary string value to use to look up vertices of type "content". The default is "contents".
        namespace : str , optional
            The desired namespace to use in the BOT graph. The default is "http://github.com/wassimj/topologicpy/resources".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        str
            The rdf graph serialized string using the BOT ontology.
        """
        from os.path import exists
        bot_graph = Graph.BOTGraph(graph= graph,
                                   bidirectional= bidirectional,
                                   includeAttributes= includeAttributes,
                                   includeLabel= includeLabel,
                                   includeGeometry= includeGeometry,
                                   siteLabel= siteLabel,
                                   siteDictionary= siteDictionary,
                                   buildingLabel= buildingLabel,
                                   buildingDictionary=  buildingDictionary,
                                   storeyPrefix= storeyPrefix,
                                   floorLevels= floorLevels,
                                   vertexLabelKey= vertexLabelKey,
                                   typeKey= typeKey,
                                   verticesKey= verticesKey,
                                   edgesKey= edgesKey,
                                   edgeLabelKey= edgeLabelKey,
                                   sourceKey= sourceKey,
                                   targetKey= targetKey,
                                   xKey= xKey,
                                   yKey= yKey,
                                   zKey= zKey,
                                   geometryKey= geometryKey,
                                   spaceType= spaceType,
                                   wallType= wallType,
                                   slabType= slabType,
                                   doorType= doorType,
                                   windowType= windowType,
                                   contentType= contentType,
                                   namespace= namespace,
                                   mantissa= mantissa)
        
        if "turtle" in format.lower() or "ttl" in format.lower() or "turtle2" in format.lower():
            ext = ".ttl"
        elif "xml" in format.lower() or "pretty=xml" in format.lower() or "rdf/xml" in format.lower():
            ext = ".xml"
        elif "json" in format.lower():
            ext = ".json"
        elif "ntriples" in format.lower() or "nt" in format.lower() or "nt11" in format.lower():
            ext = ".nt"
        elif "n3" in format.lower() or "notation" in format.lower():
            ext = ".n3"
        elif "trig" in format.lower():
            ext = ".trig"
        elif "trix" in format.lower():
            ext = ".trix"
        elif "nquads" in format.lower():
            ext = ".nquads"
        else:
            format = "turtle"
            ext = ".ttl"
        n = len(ext)
        # Make sure the file extension is .brep
        ext = path[len(path)-n:len(path)]
        if ext.lower() != ext:
            path = path+ext
        if not overwrite and exists(path):
            print("Graph.ExportToBOT - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        status = False
        try:
            bot_graph.serialize(destination=path, format=format)
            status = True
        except:
            status = False
        return status
    
    @staticmethod
    def ExportToCSV(graph, path, graphLabel, graphFeatures="",  
                       graphIDHeader="graph_id", graphLabelHeader="label", graphFeaturesHeader="feat",
                       
                       edgeLabelKey="label", defaultEdgeLabel=0, edgeFeaturesKeys=[],
                       edgeSRCHeader="src_id", edgeDSTHeader="dst_id",
                       edgeLabelHeader="label", edgeFeaturesHeader="feat",
                       edgeTrainMaskHeader="train_mask", edgeValidateMaskHeader="val_mask", edgeTestMaskHeader="test_mask",
                       edgeMaskKey="mask",
                       edgeTrainRatio=0.8, edgeValidateRatio=0.1, edgeTestRatio=0.1,
                       bidirectional=True,

                       nodeLabelKey="label", defaultNodeLabel=0, nodeFeaturesKeys=[],
                       nodeIDHeader="node_id", nodeLabelHeader="label", nodeFeaturesHeader="feat",
                       nodeTrainMaskHeader="train_mask", nodeValidateMaskHeader="val_mask", nodeTestMaskHeader="test_mask",
                       nodeMaskKey="mask",
                       nodeTrainRatio=0.8, nodeValidateRatio=0.1, nodeTestRatio=0.1,
                       mantissa=6, tolerance=0.0001, overwrite=False):
        """
        Exports the input graph into a set of CSV files compatible with DGL.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph
        path : str
            The desired path to the output folder where the graphs, edges, and nodes CSV files will be saved.
        graphLabel : float or int
            The input graph label. This can be an int (categorical) or a float (continous)
        graphFeatures : str , optional
            The input graph features. This is a single string of numeric features separated by commas. Example: "3.456, 2.011, 56.4". The defauly is "".
        graphIDHeader : str , optional
            The desired graph ID column header. The default is "graph_id".
        graphLabelHeader : str , optional
            The desired graph label column header. The default is "label".
        graphFeaturesHeader : str , optional
            The desired graph features column header. The default is "feat".
        edgeLabelKey : str , optional
            The edge label dictionary key saved in each graph edge. The default is "label".
        defaultEdgeLabel : int , optional
            The default edge label to use if no edge label is found. The default is 0.
        edgeLabelHeader : str , optional
            The desired edge label column header. The default is "label".
        edgeSRCHeader : str , optional
            The desired edge source column header. The default is "src_id".
        edgeDSTHeader : str , optional
            The desired edge destination column header. The default is "dst_id".
        edgeFeaturesHeader : str , optional
            The desired edge features column header. The default is "feat".
        edgeFeaturesKeys : list , optional
            The list of feature dictionary keys saved in the dicitonaries of edges. The default is [].
        edgeTrainMaskHeader : str , optional
            The desired edge train mask column header. The default is "train_mask".
        edgeValidateMaskHeader : str , optional
            The desired edge validate mask column header. The default is "val_mask".
        edgeTestMaskHeader : str , optional
            The desired edge test mask column header. The default is "test_mask".
        edgeMaskKey : str , optional
            The dictionary key where the edge train, validate, test category is to be found. The value should be 0 for train
            1 for validate, and 2 for test. If no key is found, the ratio of train/validate/test will be used. The default is "mask".
        edgeTrainRatio : float , optional
            The desired ratio of the edge data to use for training. The number must be between 0 and 1. The default is 0.8 which means 80% of the data will be used for training.
            This value is ignored if an edgeMaskKey is foud.
        edgeValidateRatio : float , optional
            The desired ratio of the edge data to use for validation. The number must be between 0 and 1. The default is 0.1 which means 10% of the data will be used for validation.
            This value is ignored if an edgeMaskKey is foud.
        edgeTestRatio : float , optional
            The desired ratio of the edge data to use for testing. The number must be between 0 and 1. The default is 0.1 which means 10% of the data will be used for testing.
            This value is ignored if an edgeMaskKey is foud.
        bidirectional : bool , optional
            If set to True, a reversed edge will also be saved for each edge in the graph. Otherwise, it will not. The default is True.
        nodeFeaturesKeys : list , optional
            The list of features keys saved in the dicitonaries of nodes. The default is [].
        nodeLabelKey : str , optional
            The node label dictionary key saved in each graph vertex. The default is "label".
        defaultNodeLabel : int , optional
            The default node label to use if no node label is found. The default is 0.
        nodeIDHeader : str , optional
            The desired node ID column header. The default is "node_id".
        nodeLabelHeader : str , optional
            The desired node label column header. The default is "label".
        nodeFeaturesHeader : str , optional
            The desired node features column header. The default is "feat".
        nodeTrainMaskHeader : str , optional
            The desired node train mask column header. The default is "train_mask".
        nodeValidateMaskHeader : str , optional
            The desired node validate mask column header. The default is "val_mask".
        nodeTestMaskHeader : str , optional
            The desired node test mask column header. The default is "test_mask".
        nodeMaskKey : str , optional
            The dictionary key where the node train, validate, test category is to be found. The value should be 0 for train
            1 for validate, and 2 for test. If no key is found, the ratio of train/validate/test will be used. The default is "mask".
        nodeTrainRatio : float , optional
            The desired ratio of the node data to use for training. The number must be between 0 and 1. The default is 0.8 which means 80% of the data will be used for training.
            This value is ignored if an nodeMaskKey is foud.
        nodeValidateRatio : float , optional
            The desired ratio of the node data to use for validation. The number must be between 0 and 1. The default is 0.1 which means 10% of the data will be used for validation.
            This value is ignored if an nodeMaskKey is foud.
        nodeTestRatio : float , optional
            The desired ratio of the node data to use for testing. The number must be between 0 and 1. The default is 0.1 which means 10% of the data will be used for testing.
            This value is ignored if an nodeMaskKey is foud.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        overwrite : bool , optional
            If set to True, any existing files are overwritten. Otherwise, the input list of graphs is appended to the end of each file. The default is False.

        Returns
        -------
        bool
            True if the graph has been successfully exported. False otherwise.
        
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import os
        import math
        import random
        from os.path import exists
        
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ExportToCSV - Error: The input graph parameter is not a valid topologic graph. Returning None.")
            return None
        
        if not exists(path):
            try:
                os.mkdir(path)
            except:
                print("Graph.ExportToCSV - Error: Could not create a folder at the specified path parameter. Returning None.")
                return None
        if overwrite == False:
            if not exists(os.path.join(path, "graphs.csv")):
                print("Graph.ExportToCSV - Error: Overwrite is set to False, but could not find a graphs.csv file at specified path parameter. Returning None.")
                return None
            if not exists(os.path.join(path, "edges.csv")):
                print("Graph.ExportToCSV - Error: Overwrite is set to False, but could not find a graphs.csv file at specified path parameter. Returning None.")
                return None
            if not exists(os.path.join(path, "nodes.csv")):
                print("Graph.ExportToCSV - Error: Overwrite is set to False, but could not find a graphs.csv file at specified path parameter. Returning None.")
                return None
        if abs(nodeTrainRatio  + nodeValidateRatio + nodeTestRatio - 1) > 0.001:
            print("Graph.ExportToCSV - Error: The node train, validate, test ratios do not add up to 1. Returning None")
            return None
        if abs(edgeTrainRatio  + edgeValidateRatio + edgeTestRatio - 1) > 0.001:
            print("Graph.ExportToCSV - Error: The edge train, validate, test ratios do not add up to 1. Returning None")
            return None
        
        # Step 1: Export Graphs
        if overwrite == False:
            graphs = pd.read_csv(os.path.join(path,"graphs.csv"))
            max_id = max(list(graphs[graphIDHeader]))
            graph_id = max_id + 1
        else:
            graph_id = 0
        data = [[graph_id], [graphLabel], [graphFeatures]]
        columns = [graphIDHeader, graphLabelHeader, graphFeaturesHeader]
        
        # Write Graph Data to CSV file
        data = Helper.Iterate(data)
        data = Helper.Transpose(data)
        df = pd.DataFrame(data, columns=columns)
        if overwrite == False:
            df.to_csv(os.path.join(path, "graphs.csv"), mode='a', index = False, header=False)
        else:
            df.to_csv(os.path.join(path, "graphs.csv"), mode='w+', index = False, header=True)

        # Step 2: Export Nodes
        vertices = Graph.Vertices(graph)
        if len(vertices) < 3:
            print("Graph.ExportToCSV - Error: The graph is too small to be used. Returning None")
            return None
        # Shuffle the vertices
        vertices = random.sample(vertices, len(vertices))
        node_train_max = math.floor(float(len(vertices))*nodeTrainRatio)
        if node_train_max == 0:
            node_train_max = 1
        node_validate_max = math.floor(float(len(vertices))*nodeValidateRatio)
        if node_validate_max == 0:
            node_validate_max = 1
        node_test_max = len(vertices) - node_train_max - node_validate_max
        if node_test_max == 0:
            node_test_max = 1
        node_data = []
        node_columns = [graphIDHeader, nodeIDHeader, nodeLabelHeader, nodeTrainMaskHeader, nodeValidateMaskHeader, nodeTestMaskHeader, nodeFeaturesHeader, "X", "Y", "Z"]
        train = 0
        test = 0
        validate = 0
        for i, v in enumerate(vertices):
            # Get the node label
            nd = Topology.Dictionary(v)
            vLabel = Dictionary.ValueAtKey(nd, nodeLabelKey)
            if vLabel == None:
                vLabel = defaultNodeLabel
            
            # Get the train/validate/test mask value
            flag = False
            if not nodeMaskKey == None:
                if not nd == None:
                    keys = Dictionary.Keys(nd)
                else:
                    keys = []
                    flag = True
                if nodeMaskKey in keys:
                    value = Dictionary.ValueAtKey(nd, nodeMaskKey)
                    if not value in [0, 1, 2]:
                        flag = True
                    elif value == 0:
                        train_mask = True
                        validate_mask = False
                        test_mask = False
                        train = train + 1
                    elif value == 1:
                        train_mask = False
                        validate_mask = True
                        test_mask = False
                        validate = validate + 1
                    else:
                        train_mask = False
                        validate_mask = False
                        test_mask = True
                        test = test + 1
                else:
                    flag = True
            else:
                flag = True
            if flag:
                if train < node_train_max:
                    train_mask = True
                    validate_mask = False
                    test_mask = False
                    train = train + 1
                elif validate < node_validate_max:
                    train_mask = False
                    validate_mask = True
                    test_mask = False
                    validate = validate + 1
                elif test < node_test_max:
                    train_mask = False
                    validate_mask = False
                    test_mask = True
                    test = test + 1
                else:
                    train_mask = True
                    validate_mask = False
                    test_mask = False
                    train = train + 1
            
            # Get the features of the vertex
            node_features = ""
            node_features_keys = Helper.Flatten(nodeFeaturesKeys)
            for node_feature_key in node_features_keys:
                if len(node_features) > 0:
                    node_features = node_features + ","+ str(round(float(Dictionary.ValueAtKey(nd, node_feature_key)),mantissa))
                else:
                    node_features = str(round(float(Dictionary.ValueAtKey(nd, node_feature_key)),mantissa))
            single_node_data = [graph_id, i, vLabel, train_mask, validate_mask, test_mask, node_features, float(Vertex.X(v, mantissa=mantissa)), float(Vertex.Y(v,mantissa)), float(Vertex.Z(v,mantissa))]
            node_data.append(single_node_data)

        # Write Node Data to CSV file
        df = pd.DataFrame(node_data, columns= node_columns)
        if graph_id == 0:
            df.to_csv(os.path.join(path, "nodes.csv"), mode='w+', index = False, header=True)
        else:
            df.to_csv(os.path.join(path, "nodes.csv"), mode='a', index = False, header=False)
        
        # Step 3: Export Edges
        edge_data = []
        edge_columns = [graphIDHeader, edgeSRCHeader, edgeDSTHeader, edgeLabelHeader, edgeTrainMaskHeader, edgeValidateMaskHeader, edgeTestMaskHeader, edgeFeaturesHeader]
        train = 0
        test = 0
        validate = 0
        edges = Graph.Edges(graph)
        edge_train_max = math.floor(float(len(edges))*edgeTrainRatio)
        edge_validate_max = math.floor(float(len(edges))*edgeValidateRatio)
        edge_test_max = len(edges) - edge_train_max - edge_validate_max
        for edge in edges:
            # Get the edge label
            ed = Topology.Dictionary(edge)
            edge_label = Dictionary.ValueAtKey(ed, edgeLabelKey)
            if edge_label == None:
                edge_label = defaultEdgeLabel
            # Get the train/validate/test mask value
            flag = False
            if not edgeMaskKey == None:
                if not ed == None:
                    keys = Dictionary.Keys(ed)
                else:
                    keys = []
                    flag = True
                if edgeMaskKey in keys:
                    value = Dictionary.ValueAtKey(ed, edgeMaskKey)
                    if not value in [0, 1, 2]:
                        flag = True
                    elif value == 0:
                        train_mask = True
                        validate_mask = False
                        test_mask = False
                        train = train + 1
                    elif value == 1:
                        train_mask = False
                        validate_mask = True
                        test_mask = False
                        validate = validate + 1
                    else:
                        train_mask = False
                        validate_mask = False
                        test_mask = True
                        test = test + 1
                else:
                    flag = True
            else:
                flag = True
            if flag:
                if train < edge_train_max:
                    train_mask = True
                    validate_mask = False
                    test_mask = False
                    train = train + 1
                elif validate < edge_validate_max:
                    train_mask = False
                    validate_mask = True
                    test_mask = False
                    validate = validate + 1
                elif test < edge_test_max:
                    train_mask = False
                    validate_mask = False
                    test_mask = True
                    test = test + 1
                else:
                    train_mask = True
                    validate_mask = False
                    test_mask = False
                    train = train + 1
            # Get the edge features
            edge_features = ""
            edge_features_keys = Helper.Flatten(edgeFeaturesKeys)
            for edge_feature_key in edge_features_keys:
                if len(edge_features) > 0:
                    edge_features = edge_features + ","+ str(round(float(Dictionary.ValueAtKey(ed, edge_feature_key)),mantissa))
                else:
                    edge_features = str(round(float(Dictionary.ValueAtKey(ed, edge_feature_key)), mantissa))
            # Get the Source and Destination vertex indices
            src = Vertex.Index(Edge.StartVertex(edge), vertices, tolerance=tolerance)
            dst = Vertex.Index(Edge.EndVertex(edge), vertices, tolerance=tolerance)
            if not src == None and not dst == None:
                single_edge_data = [graph_id, src, dst, edge_label, train_mask, validate_mask, test_mask, edge_features]
                edge_data.append(single_edge_data)

                if bidirectional == True:
                    single_edge_data = [graph_id, src, dst, edge_label, train_mask, validate_mask, test_mask, edge_features]
                    edge_data.append(single_edge_data)
        df = pd.DataFrame(edge_data, columns=edge_columns)

        if graph_id == 0:
            df.to_csv(os.path.join(path, "edges.csv"), mode='w+', index = False, header=True)
        else:
            df.to_csv(os.path.join(path, "edges.csv"), mode='a', index = False, header=False)
        
        # Write out the meta.yaml file
        yaml_file = open(os.path.join(path,"meta.yaml"), "w")
        if graph_id > 0:
            yaml_file.write('dataset_name: topologic_dataset\nedge_data:\n- file_name: edges.csv\nnode_data:\n- file_name: nodes.csv\ngraph_data:\n  file_name: graphs.csv')
        else:
            yaml_file.write('dataset_name: topologic_dataset\nedge_data:\n- file_name: edges.csv\nnode_data:\n- file_name: nodes.csv')
        yaml_file.close()
        return True
    
    @staticmethod
    def ExportToGEXF(graph, path: str = None, graphWidth: float = 20, graphLength: float = 20, graphHeight: float = 20,
                    defaultVertexColor: str = "black", defaultVertexSize: float = 3,
                    vertexLabelKey: str = None, vertexColorKey: str = None, vertexSizeKey: str = None, 
                    defaultEdgeColor: str = "black", defaultEdgeWeight: float = 1, defaultEdgeType: str = "undirected",
                    edgeLabelKey: str = None, edgeColorKey: str = None, edgeWeightKey: str = None,
                    overwrite: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Exports the input graph to a Graph Exchange XML (GEXF) file format. See https://gexf.net/

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph
        path : str
            The desired path to the output folder where the graphs, edges, and nodes CSV files will be saved.
        graphWidth : float or int , optional
            The desired graph width. The default is 20.
        graphLength : float or int , optional
            The desired graph length. The default is 20.
        graphHeight : float or int , optional
            The desired graph height. The default is 20.
        defaultVertexColor : str , optional
            The desired default vertex color. The default is "black".
        defaultVertexSize : float or int , optional
            The desired default vertex size. The default is 3.
        defaultEdgeColor : str , optional
            The desired default edge color. The default is "black".
        defaultEdgeWeight : float or int , optional
            The desired default edge weight. The edge weight determines the width of the displayed edge. The default is 3.
        defaultEdgeType : str , optional
            The desired default edge type. This can be one of "directed" or "undirected". The default is "undirected".
        vertexLabelKey : str , optional
            If specified, the vertex dictionary is searched for this key to determine the vertex label. If not specified
            the vertex label being is set to "Node X" where is X is a unique number. The default is None.
        vertexColorKey : str , optional
            If specified, the vertex dictionary is searched for this key to determine the vertex color. If not specified
            the vertex color is set to the value defined by defaultVertexColor parameter. The default is None.
        vertexSizeKey : str , optional
            If specified, the vertex dictionary is searched for this key to determine the vertex size. If not specified
            the vertex size is set to the value defined by defaultVertexSize parameter. The default is None.
        edgeLabelKey : str , optional
            If specified, the edge dictionary is searched for this key to determine the edge label. If not specified
            the edge label being is set to "Edge X" where is X is a unique number. The default is None.
        edgeColorKey : str , optional
            If specified, the edge dictionary is searched for this key to determine the edge color. If not specified
            the edge color is set to the value defined by defaultEdgeColor parameter. The default is None.
        edgeWeightKey : str , optional
            If specified, the edge dictionary is searched for this key to determine the edge weight. If not specified
            the edge weight is set to the value defined by defaultEdgeWeight parameter. The default is None.
        overwrite : bool , optional
            If set to True, any existing file is overwritten. Otherwise, it is not. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the graph has been successfully exported. False otherwise.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        import numbers
        from datetime import datetime
        import os
        from os.path import exists
        
        def create_gexf_file(nodes, edges, default_edge_type, node_attributes, edge_attributes, path):

            with open(path, 'w') as file:
                # Write the GEXF header
                formatted_date = datetime.now().strftime("%Y-%m-%d")
                if not isinstance(default_edge_type, str):
                    default_edge_type = "undirected"
                if default_edge_type.lower() == "directed":
                    defaultedge_type = "directed"
                else:
                    default_edge_type = "undirected"
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<gexf version="1.3" xmlns="http://www.gephi.org/gexf" xmlns:viz="http://www.gephi.org/gexf/viz">\n')
                file.write(f'<meta lastmodifieddate="{formatted_date}">\n')
                file.write('<creator>Topologic GEXF Generator</creator>\n')
                file.write('<title>"Topologic Graph"</title>\n')
                file.write('<description>"This is a Topologic Graph"</description>\n')
                file.write('</meta>\n')
                file.write(f'<graph type="static" defaultedgetype="{defaultEdgeType}">\n')

                # Write attribute definitions
                file.write('<attributes class="node" mode="static">\n')
                for attr_name, attr_type in node_attributes.items():
                    file.write(f'<attribute id="{attr_name}" title="{attr_name}" type="{attr_type}"/>\n')
                file.write('</attributes>\n')
                # Write attribute definitions
                file.write('<attributes class="edge" mode="static">\n')
                for attr_name, attr_type in edge_attributes.items():
                    file.write(f'<attribute id="{attr_name}" title="{attr_name}" type="{attr_type}"/>\n')
                file.write('</attributes>\n')

                # Write nodes with attributes
                file.write('<nodes>\n')
                for node_id, node_attrs in nodes.items():
                    file.write(f'<node id="{node_id}" label="{node_attrs["label"]}">\n')
                    if "r" in node_attrs and "g" in node_attrs and "b" in node_attrs:
                        r = node_attrs['r']
                        g = node_attrs['g']
                        b = node_attrs['b']
                        file.write(f'<viz:color r="{r}" g="{g}" b="{b}"/>\n')
                    if "size" in node_attrs:
                        file.write(f'<viz:size value="{node_attrs["size"]}"/>\n')
                    if "x" in node_attrs and "y" in node_attrs and "z" in node_attrs:
                        file.write(f'<viz:position x="{node_attrs["x"]}" y="{node_attrs["y"]}" z="{node_attrs["z"]}"/>\n')
                    file.write('<attvalues>\n')
                    keys = node_attrs.keys()
                    for key in keys:
                        file.write(f'<attvalue id="{key}" value="{node_attrs[key]}"/>\n')
                    file.write('</attvalues>\n')
                    file.write('</node>\n')
                file.write('</nodes>\n')

                # Write edges with attributes
                file.write('<edges>\n')
                for edge_id, edge_attrs in edges.items():
                    source, target = edge_id
                    file.write(f'<edge id="{edge_id}" source="{source}" target="{target}" label="{edge_attrs["label"]}">\n')
                    if "color" in edge_attrs:
                        r, g, b = Color.ByCSSNamedColor(edge_attrs["color"])
                        file.write(f'<viz:color r="{r}" g="{g}" b="{b}"/>\n')
                    file.write('<attvalues>\n')
                    keys = edge_attrs.keys()
                    for key in keys:
                        file.write(f'<attvalue id="{key}" value="{edge_attrs[key]}"/>\n')
                    file.write('</attvalues>\n')
                    file.write('</edge>\n')
                file.write('</edges>\n')

                # Write the GEXF footer
                file.write('</graph>\n')
                file.write('</gexf>\n')

        def valueType(value):
            if isinstance(value, str):
                return 'string'
            elif isinstance(value, float):
                return 'double'
            elif isinstance(value, int):
                return 'integer'
            else:
                return 'string'
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ExportToGEXF - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        if not isinstance(path, str):
            print("Graph.ExportToGEXF - Error: the input path parameter is not a valid string. Returning None.")
            return None
        # Make sure the file extension is .brep
        ext = path[len(path)-5:len(path)]
        if ext.lower() != ".gexf":
            path = path+".gexf"
        if not overwrite and exists(path):
            print("Graph.ExportToGEXF - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        
        g_vertices = Graph.Vertices(graph)
        g_edges = Graph.Edges(graph)
        
        node_attributes = {'id': 'integer',
                        'label': 'string',
                        'x': 'double',
                        'y': 'double',
                        'z': 'double',
                        'r': 'integer',
                        'g': 'integer',
                        'b': 'integer',
                        'color': 'string',
                        'size': 'double'}
        nodes = {}
        # Resize the graph
        xList = [Vertex.X(v, mantissa=mantissa) for v in g_vertices]
        yList = [Vertex.Y(v, mantissa=mantissa) for v in g_vertices]
        zList = [Vertex.Z(v, mantissa=mantissa) for v in g_vertices]
        xMin = min(xList)
        xMax = max(xList)
        yMin = min(yList)
        yMax = max(yList)
        zMin = min(zList)
        zMax = max(zList)
        width = max(abs(xMax - xMin), 0.01)
        length = max(abs(yMax - yMin), 0.01)
        height = max(abs(zMax - zMin), 0.01)
        x_sf = graphWidth/width
        y_sf = graphLength/length
        z_sf = graphHeight/height
        x_avg = sum(xList)/float(len(xList))
        y_avg = sum(yList)/float(len(yList))
        z_avg = sum(zList)/float(len(zList))
        
        for i, v in enumerate(g_vertices):
            node_dict = {}
            d = Topology.Dictionary(v)
            keys = Dictionary.Keys(d)
            values = Dictionary.Values(d)
            x = (Vertex.X(v, mantissa=mantissa) - x_avg)*x_sf + x_avg
            y = (Vertex.Y(v, mantissa=mantissa) - y_avg)*y_sf + y_avg
            z = (Vertex.Z(v, mantissa=mantissa) - z_avg)*z_sf + z_avg
            node_dict['x'] = x
            node_dict['y'] = y
            node_dict['z'] = z
            node_dict['id'] = i
            for m, key in enumerate(keys):
                if key == "psets": #We cannot handle IFC psets at this point.
                    continue
                if key == "id":
                    key = "TOPOLOGIC_ID"
                if not key in node_attributes.keys():
                    node_attributes[key] = valueType(values[m])
                if isinstance(values[m], str):
                    values[m] = values[m].replace('&','&amp;')
                    values[m] = values[m].replace('<','&lt;')
                    values[m] = values[m].replace('>','&gt;')
                    values[m] = values[m].replace('"','&quot;')
                    values[m] = values[m].replace('\'','&apos;')
                node_dict[key] = values[m]
            dict_color = None
            if not defaultVertexColor in Color.CSSNamedColors():
                defaultVertexColor = "black"
            vertex_color = defaultVertexColor
            if isinstance(vertexColorKey, str):
                dict_color = Dictionary.ValueAtKey(d, vertexColorKey)
            if not dict_color == None:
                vertex_color = dict_color
            if not vertex_color in Color.CSSNamedColors():
                vertex_color = defaultVertexColor
            node_dict['color'] = vertex_color
            r, g, b = Color.ByCSSNamedColor(vertex_color)
            node_dict['r'] = r
            node_dict['g'] = g
            node_dict['b'] = b
        
            dict_size = None
            if isinstance(vertexSizeKey, str):
                dict_size = Dictionary.ValueAtKey(d, vertexSizeKey)
            
            vertex_size = defaultVertexSize
            if not dict_size == None:
                if isinstance(dict_size, numbers.Real):
                    vertex_size = dict_size
            if not isinstance(vertex_size, numbers.Real):
                vertex_size = defaultVertexSize
            
            node_dict['size'] = vertex_size

            vertex_label = "Node "+str(i)
            if isinstance(vertexLabelKey, str):
                vertex_label = Dictionary.ValueAtKey(d, vertexLabelKey)
            if not isinstance(vertex_label, str):
                vertex_label = "Node "+str(i)
            if isinstance(vertex_label, str):
                vertex_label = vertex_label.replace('&','&amp;')
                vertex_label = vertex_label.replace('<','&lt;')
                vertex_label = vertex_label.replace('>','&gt;')
                vertex_label = vertex_label.replace('"','&quot;')
                vertex_label = vertex_label.replace('\'','&apos;')
            node_dict['label'] = vertex_label

            nodes[i] = node_dict
            
        edge_attributes = {'id': 'integer',
                        'label': 'string',
                        'source': 'integer',
                        'target': 'integer',
                        'r': 'integer',
                        'g': 'integer',
                        'b': 'integer',
                        'color': 'string',
                        'weight': 'double'}
        edges = {}
        for i, edge in enumerate(g_edges):
            edge_dict = {}
            d = Topology.Dictionary(edge)
            keys = Dictionary.Keys(d)
            values = Dictionary.Values(d)
            edge_dict['id'] = i
            for m, key in enumerate(keys):
                if key == "id":
                    key = "TOPOLOGIC_ID"
                if not key in edge_attributes.keys():
                    edge_attributes[key] = valueType(values[m])
                edge_dict[key] = values[m]
                
            dict_color = None
            if not defaultEdgeColor in Color.CSSNamedColors():
                defaultEdgeColor = "black"
            edge_color = defaultEdgeColor
            if isinstance(edgeColorKey, str):
                dict_color = Dictionary.ValueAtKey(d, edgeColorKey)
            if not dict_color == None:
                edge_color = dict_color
            if not vertex_color in Color.CSSNamedColors():
                edge_color = defaultVertexColor
            edge_dict['color'] = edge_color
            
            r, g, b = Color.ByCSSNamedColor(edge_color)
            edge_dict['r'] = r
            edge_dict['g'] = g
            edge_dict['b'] = b
        
            dict_weight = None
            if not isinstance(defaultEdgeWeight, numbers.Real):
                defaultEdgeWeight = 1
            edge_weight = defaultEdgeWeight
            if isinstance(edgeWeightKey, str):
                dict_weight = Dictionary.ValueAtKey(d, edgeWeightKey)
            if not dict_weight == None:
                if isinstance(dict_weight, numbers.Real):
                    edge_weight = dict_weight
            if not isinstance(edge_weight, numbers.Real):
                edge_weight = defaultEdgeWeight
            
            edge_dict['weight'] = edge_weight

            
            sv = g_vertices[Vertex.Index(Edge.StartVertex(edge), g_vertices, tolerance=tolerance)]
            ev = g_vertices[Vertex.Index(Edge.EndVertex(edge), g_vertices, tolerance=tolerance)]
            svid = Vertex.Index(sv, g_vertices, tolerance=tolerance)
            evid = Vertex.Index(ev, g_vertices, tolerance=tolerance)
            if not svid == None and not evid == None:
                edge_dict['source'] = svid
                edge_dict['target'] = evid
                edge_label = "Edge "+str(svid)+"-"+str(evid)
                if isinstance(edgeLabelKey, str):
                    edge_label = Dictionary.ValueAtKey(d, edgeLabelKey)
                if not isinstance(edge_label, str):
                    edge_label = "Edge "+str(svid)+"-"+str(evid)
                edge_dict['label'] = edge_label
                edges[(str(svid), str(evid))] = edge_dict

        create_gexf_file(nodes, edges, defaultEdgeType, node_attributes, edge_attributes, path)
        return True

    @staticmethod
    def ExportToJSON(graph, path, verticesKey="vertices", edgesKey="edges", vertexLabelKey="", edgeLabelKey="", xKey="x", yKey="y", zKey="z", indent=4, sortKeys=False, mantissa=6, overwrite=False):
        """
        Exports the input graph to a JSON file.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        path : str
            The path to the JSON file.
        verticesKey : str , optional
            The desired key name to call vertices. The default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. The default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        xKey : str , optional
            The desired key name to use for x-coordinates. The default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. The default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. The default is "z".
        indent : int , optional
            The desired amount of indent spaces to use. The default is 4.
        sortKeys : bool , optional
            If set to True, the keys will be sorted. Otherwise, they won't be. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.

        Returns
        -------
        bool
            The status of exporting the JSON file. If True, the operation was successful. Otherwise, it was unsuccesful.

        """
        import json
        from os.path import exists
        # Make sure the file extension is .json
        ext = path[len(path)-5:len(path)]
        if ext.lower() != ".json":
            path = path+".json"
        if not overwrite and exists(path):
            print("Graph.ExportToJSON - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        f = None
        try:
            if overwrite == True:
                f = open(path, "w")
            else:
                f = open(path, "x") # Try to create a new File
        except:
            raise Exception("Graph.ExportToJSON - Error: Could not create a new file at the following location: "+path)
        if (f):
            jsondata = Graph.JSONData(graph, verticesKey=verticesKey, edgesKey=edgesKey, vertexLabelKey=vertexLabelKey, edgeLabelKey=edgeLabelKey, xKey=xKey, yKey=yKey, zKey=zKey, mantissa=mantissa)
            if jsondata != None:
                json.dump(jsondata, f, indent=indent, sort_keys=sortKeys)
                f.close()
                return True
            else:
                f.close()
                return False
        return False
    
    @staticmethod
    def Flatten(graph, layout="spring", k=0.8, seed=None, iterations=50, rootVertex=None, tolerance=0.0001):
        """
        Flattens the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        layout : str , optional
            The desired mode for flattening. If set to 'spring', the algorithm uses a simplified version of the Fruchterman-Reingold force-directed algorithm to flatten and distribute the vertices.
            If set to 'radial', the nodes will be distributed along a circle.
            If set to 'tree', the nodes will be distributed using the Reingold-Tillford layout. The default is 'spring'.
        k : float, optional
            The desired spring constant to use for the attractive and repulsive forces. The default is 0.8.
        seed : int , optional
            The desired random seed to use. The default is None.
        iterations : int , optional
            The desired maximum number of iterations to solve the forces in the 'spring' mode. The default is 50.
        rootVertex : topologic_core.Vertex , optional
            The desired vertex to use as the root of the tree and radial layouts.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The flattened graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import numpy as np

        def buchheim(tree):
            dt = firstwalk(_DrawTree(tree))
            min = second_walk(dt)
            if min < 0:
                third_walk(dt, -min)
            return dt

        def third_walk(tree, n):
            tree.x += n
            for c in tree.children:
                third_walk(c, n)

        def firstwalk(v, distance=1.0):
            if len(v.children) == 0:
                if v.lmost_sibling:
                    v.x = v.lbrother().x + distance
                else:
                    v.x = 0.0
            else:
                default_ancestor = v.children[0]
                for w in v.children:
                    firstwalk(w)
                    default_ancestor = apportion(w, default_ancestor, distance)
                execute_shifts(v)

                midpoint = (v.children[0].x + v.children[-1].x) / 2

                w = v.lbrother()
                if w:
                    v.x = w.x + distance
                    v.mod = v.x - midpoint
                else:
                    v.x = midpoint
            return v

        def apportion(v, default_ancestor, distance):
            w = v.lbrother()
            if w is not None:
                vir = vor = v
                vil = w
                vol = v.lmost_sibling
                sir = sor = v.mod
                sil = vil.mod
                sol = vol.mod
                while vil.right() and vir.left():
                    vil = vil.right()
                    vir = vir.left()
                    vol = vol.left()
                    vor = vor.right()
                    vor.ancestor = v
                    shift = (vil.x + sil) - (vir.x + sir) + distance
                    if shift > 0:
                        move_subtree(ancestor(vil, v, default_ancestor), v, shift)
                        sir = sir + shift
                        sor = sor + shift
                    sil += vil.mod
                    sir += vir.mod
                    sol += vol.mod
                    sor += vor.mod
                if vil.right() and not vor.right():
                    vor.thread = vil.right()
                    vor.mod += sil - sor
                else:
                    if vir.left() and not vol.left():
                        vol.thread = vir.left()
                        vol.mod += sir - sol
                    default_ancestor = v
            return default_ancestor

        def move_subtree(wl, wr, shift):
            subtrees = wr.number - wl.number
            wr.change -= shift / subtrees
            wr.shift += shift
            wl.change += shift / subtrees
            wr.x += shift
            wr.mod += shift

        def execute_shifts(v):
            shift = change = 0
            for w in v.children[::-1]:
                w.x += shift
                w.mod += shift
                change += w.change
                shift += w.shift + change

        def ancestor(vil, v, default_ancestor):
            if vil.ancestor in v.parent.children:
                return vil.ancestor
            else:
                return default_ancestor

        def second_walk(v, m=0, depth=0, min=None):
            v.x += m
            v.y = depth

            if min is None or v.x < min:
                min = v.x

            for w in v.children:
                min = second_walk(w, m + v.mod, depth + 1, min)

            return min

        def edge_list_to_adjacency_matrix(edge_list):
            """Converts an edge list to an adjacency matrix.

            Args:
                edge_list: A list of tuples, where each tuple is an edge.

            Returns:
                A numpy array representing the adjacency matrix.
            """
            from topologicpy.Helper import Helper
            # Get the number of nodes from the edge list.
            flat_list = Helper.Flatten(edge_list)
            flat_list = [x for x in flat_list if not x == None]
            num_nodes = max(flat_list) + 1

            # Create an adjacency matrix.
            adjacency_matrix = np.zeros((num_nodes, num_nodes))

            # Fill in the adjacency matrix.
            for edge in edge_list:
                adjacency_matrix[edge[0], edge[1]] = 1
                adjacency_matrix[edge[1], edge[0]] = 1

            return adjacency_matrix

        def tree_from_edge_list(edge_list, root_index=0):
            
            adj_matrix = edge_list_to_adjacency_matrix(edge_list)
            num_nodes = adj_matrix.shape[0]
            root = _Tree(str(root_index))
            is_visited = np.zeros(num_nodes)
            is_visited[root_index] = 1
            old_roots = [root]

            new_roots = []
            while(np.sum(is_visited) < num_nodes):
                new_roots = []
                for temp_root in old_roots:
                    children = []
                    for i in range(num_nodes):
                        if adj_matrix[int(temp_root.node), i] == 1  and is_visited[i] == 0:
                            is_visited[i] = 1
                            child = _Tree(str(i))
                            temp_root.children.append(child)
                            children.append(child)

                    new_roots.extend(children)
                old_roots = new_roots
            return root, num_nodes

        def spring_layout(edge_list, iterations=500, k=None, seed=None):
            # Compute the layout of a graph using the Fruchterman-Reingold algorithm
            # with a force-directed layout approach.

            adj_matrix = edge_list_to_adjacency_matrix(edge_list)
            # Set the random seed
            if seed is not None:
                np.random.seed(seed)

            # Set the optimal distance between nodes
            if k is None or k <= 0:
                k = np.sqrt(1.0 / adj_matrix.shape[0])

            # Initialize the positions of the nodes randomly
            pos = np.random.rand(adj_matrix.shape[0], 2)

            # Compute the initial temperature
            t = 0.1 * np.max(pos)

            # Compute the cooling factor
            cooling_factor = t / iterations

            # Iterate over the specified number of iterations
            for i in range(iterations):
                # Compute the distance between each pair of nodes
                delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                distance = np.linalg.norm(delta, axis=-1)

                # Avoid division by zero
                distance = np.where(distance == 0, 0.1, distance)

                # Compute the repulsive force between each pair of nodes
                repulsive_force = k ** 2 / distance ** 2

                # Compute the attractive force between each pair of adjacent nodes
                attractive_force = adj_matrix * distance / k

                # Compute the total force acting on each node
                force = np.sum((repulsive_force - attractive_force)[:, :, np.newaxis] * delta, axis=1)

                # Compute the displacement of each node
                displacement = t * force / np.linalg.norm(force, axis=1)[:, np.newaxis]

                # Update the positions of the nodes
                pos += displacement

                # Cool the temperature
                t -= cooling_factor

            return pos

        def tree_layout(edge_list,  root_index=0):

            root, num_nodes = tree_from_edge_list(edge_list, root_index)
            dt = buchheim(root)
            pos = np.zeros((num_nodes, 2))

            pos[int(dt.tree.node), 0] = dt.x
            pos[int(dt.tree.node), 1] = dt.y

            old_roots = [dt]
            new_roots = []

            while(len(old_roots) > 0):
                new_roots = []
                for temp_root in old_roots:
                    children = temp_root.children
                    for child in children:
                        pos[int(child.tree.node), 0] = child.x
                        pos[int(child.tree.node), 1] = child.y
                    new_roots.extend(children)
                    
                old_roots = new_roots

            pos[:, 1] = np.max(pos[:, 1]) - pos[:, 1]
            
            return pos

        def radial_layout(edge_list, root_index=0):
            root, num_nodes = tree_from_edge_list(edge_list, root_index)
            dt = buchheim(root)
            pos = np.zeros((num_nodes, 2))

            pos[int(dt.tree.node), 0] = dt.x
            pos[int(dt.tree.node), 1] = dt.y

            old_roots = [dt]
            new_roots = []

            while(len(old_roots) > 0):
                new_roots = []
                for temp_root in old_roots:
                    children = temp_root.children
                    for child in children:
                        pos[int(child.tree.node), 0] = child.x
                        pos[int(child.tree.node), 1] = child.y
                    new_roots.extend(children)
                    
                old_roots = new_roots

            # pos[:, 1] = np.max(pos[:, 1]) - pos[:, 1]
            pos[:, 0] = pos[:, 0] - np.min(pos[:, 0])
            pos[:, 1] = pos[:, 1] - np.min(pos[:, 1])

            pos[:, 0] = pos[:, 0] / np.max(pos[:, 0])
            pos[:, 0] = pos[:, 0] - pos[:, 0][root_index]
            
            range_ = np.max(pos[:, 0]) - np.min(pos[:, 0])
            pos[:, 0] = pos[:, 0] / range_

            pos[:, 0] = pos[:, 0] * np.pi * 1.98
            pos[:, 1] = pos[:, 1] / np.max(pos[:, 1]) 

            new_pos = np.zeros((num_nodes, 2))
            new_pos[:, 0] = pos[:, 1] * np.cos(pos[:, 0])
            new_pos[:, 1] = pos[:, 1] * np.sin(pos[:, 0])
            
            return new_pos

        def graph_layout(edge_list, layout='tree', root_index=0, k=None, seed=None, iterations=500):

            if layout == 'tree':
                return tree_layout(edge_list, root_index=root_index)
            elif layout == 'spring':
                return spring_layout(edge_list, k=k, seed=seed, iterations=iterations)
            elif layout == 'radial':
                return radial_layout(edge_list, root_index=root_index)
            else:
                raise NotImplementedError(f"{layout} is not implemented yet. Please choose from ['radial', 'spring', 'tree']")

        def vertex_max_degree(graph, vertices):
            degrees = [Graph.VertexDegree(graph, vertex) for vertex in vertices]
            i = degrees.index(max(degrees))
            return vertices[i], i

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Flatten - Error: The input graph is not a valid topologic graph. Returning None.")
            return None
        d = Graph.MeshData(graph)
        vertices = d['vertices']
        edges = d['edges']
        v_dicts = d['vertexDictionaries']
        e_dicts = d['edgeDictionaries']
        vertices = Graph.Vertices(graph)
        if rootVertex == None:
            rootVertex, root_index = vertex_max_degree(graph, vertices)
        else:
            root_index = Vertex.Index(rootVertex, vertices, tolerance=tolerance)
        if root_index == None:
            root_index = 0
        if 'rad' in layout.lower():
            positions = radial_layout(edges, root_index=root_index)
        elif 'spring' in layout.lower():
            positions = spring_layout(edges, k=k, seed=seed, iterations=iterations)
        elif 'tree' in layout.lower():
            positions = tree_layout(edges, root_index=root_index)
        else:
            raise NotImplementedError(f"{layout} is not implemented yet. Please choose from ['radial', 'spring', 'tree']")
        positions = positions.tolist()
        positions = [[p[0], p[1], 0] for p in positions]
        flat_graph = Graph.ByMeshData(positions, edges, v_dicts, e_dicts, tolerance=tolerance)
        return flat_graph

    @staticmethod
    def GlobalClusteringCoefficient(graph):
        """
        Returns the global clustering coefficient of the input graph. See https://en.wikipedia.org/wiki/Clustering_coefficient.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        
        Returns
        -------
        int
            The computed global clustering coefficient.

        """
        from topologicpy.Topology import Topology

        def global_clustering_coefficient(adjacency_matrix):
            total_triangles = 0
            total_possible_triangles = 0

            num_nodes = len(adjacency_matrix)

            for i in range(num_nodes):
                neighbors = [j for j, value in enumerate(adjacency_matrix[i]) if value == 1]
                num_neighbors = len(neighbors)
                num_triangles = 0
                if num_neighbors >= 2:
                    # Count the number of connections between the neighbors
                    for i in range(num_neighbors):
                        for j in range(i + 1, num_neighbors):
                            if adjacency_matrix[neighbors[i]][neighbors[j]] == 1:
                                num_triangles += 1
                    
                    # Update total triangles and possible triangles
                    total_triangles += num_triangles
                    total_possible_triangles += num_neighbors * (num_neighbors - 1) // 2
                    

            # Calculate the global clustering coefficient
            global_clustering_coeff = 3.0 * total_triangles / total_possible_triangles if total_possible_triangles > 0 else 0.0

            return global_clustering_coeff

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.LocalClusteringCoefficient - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        adjacency_matrix = Graph.AdjacencyMatrix(graph)
        return global_clustering_coefficient(adjacency_matrix)
    
    @staticmethod
    def Guid(graph):
        """
        Returns the guid of the input graph

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Guid - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        return graph.GetGUID()

    @staticmethod
    def IncomingEdges(graph, vertex, directed: bool = False, tolerance: float = 0.0001) -> list:
        """
        Returns the incoming edges connected to a vertex. An edge is considered incoming if its end vertex is
        coincident with the input vertex.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        directed : bool , optional
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of incoming edges

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.IncomingEdges - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.IncomingEdges - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        
        edges = Graph.Edges(graph, [vertex])
        if directed == False:
            return edges
        incoming_edges = []
        for edge in edges:
            ev = Edge.EndVertex(edge)
            if Vertex.Distance(vertex, ev) < tolerance:
                incoming_edges.append(edge)
        return incoming_edges
    
    @staticmethod
    def IncomingVertices(graph, vertex, directed: bool = False, tolerance: float = 0.0001) -> list:
        """
        Returns the incoming vertices connected to a vertex. A vertex is considered incoming if it is an adjacent vertex to the input vertex
        and the the edge connecting it to the input vertex is an incoming edge.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        directed : bool , optional
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of incoming vertices

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.IncomingVertices - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.IncomingVertices - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        
        if directed == False:
            return Graph.AdjacentVertices(graph, vertex)
        incoming_edges = Graph.IncomingEdges(graph, vertex, directed=directed, tolerance=tolerance)
        incoming_vertices = []
        for edge in incoming_edges:
            sv = Edge.StartVertex(edge)
            incoming_vertices.append(Graph.NearestVertex(graph, sv))
        return incoming_vertices
    
    @staticmethod
    def IsBipartite(graph, tolerance=0.0001):
        """
        Returns True if the input graph is bipartite. Returns False otherwise. See https://en.wikipedia.org/wiki/Bipartite_graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input graph is complete. False otherwise

        """
        # From https://www.geeksforgeeks.org/bipartite-graph/
        # This code is contributed by divyesh072019.

        from topologicpy.Topology import Topology

        def isBipartite(V, adj):
            # vector to store colour of vertex
            # assigning all to -1 i.e. uncoloured
            # colours are either 0 or 1
            # for understanding take 0 as red and 1 as blue
            col = [-1]*(V)

            # queue for BFS storing {vertex , colour}
            q = []

            #loop incase graph is not connected
            for i in range(V):
    
                # if not coloured
                if (col[i] == -1):
        
                    # colouring with 0 i.e. red
                    q.append([i, 0])
                    col[i] = 0
        
                    while len(q) != 0:
                        p = q[0]
                        q.pop(0)
            
                        # current vertex
                        v = p[0]
                
                        # colour of current vertex
                        c = p[1]
                
                        # traversing vertexes connected to current vertex
                        for j in adj[v]:
                
                            # if already coloured with parent vertex color
                            # then bipartite graph is not possible
                            if (col[j] == c):
                                return False
                
                            # if uncoloured
                            if (col[j] == -1):
                    
                                # colouring with opposite color to that of parent
                                if c == 1:
                                    col[j] = 0
                                else:
                                    col[j] = 1
                                q.append([j, col[j]])
    
            # if all vertexes are coloured such that
            # no two connected vertex have same colours
            return True
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.IsBipartite - Error: The input graph is not a valid graph. Returning None.")
            return None
        order = Graph.Order(graph)
        adjList = Graph.AdjacencyList(graph, tolerance=tolerance)
        return isBipartite(order, adjList)

    @staticmethod
    def IsComplete(graph):
        """
        Returns True if the input graph is complete. Returns False otherwise. See https://en.wikipedia.org/wiki/Complete_graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        bool
            True if the input graph is complete. False otherwise

        """
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.IsComplete - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.IsComplete()
    
    @staticmethod
    def IsErdoesGallai(graph, sequence):
        """
        Returns True if the input sequence satisfies the Erdős–Gallai theorem. Returns False otherwise. See https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93Gallai_theorem.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        sequence : list
            The input sequence.

        Returns
        -------
        bool
            True if the input sequence satisfies the Erdős–Gallai theorem. False otherwise.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.IsErdoesGallai - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.IsErdoesGallai(sequence)
    
    @staticmethod
    def IsolatedVertices(graph):
        """
        Returns the list of isolated vertices in the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        list
            The list of isolated vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.IsolatedVertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertices = []
        _ = graph.IsolatedVertices(vertices)
        return vertices
    
    @staticmethod
    def JSONData(graph,
                 verticesKey: str = "vertices",
                 edgesKey: str = "edges",
                 vertexLabelKey: str = "",
                 edgeLabelKey: str = "",
                 sourceKey: str = "source",
                 targetKey: str = "target",
                 xKey: str = "x",
                 yKey: str = "y",
                 zKey: str = "z",
                 geometryKey: str = "brep",
                 mantissa: int = 6,
                 tolerance: float = 0.0001):
        """
        Converts the input graph into JSON data.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        verticesKey : str , optional
            The desired key name to call vertices. The default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. The default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. The default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. The default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. The default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. The default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. The default is "z".
        geometryKey : str , optional
            The desired key name to use for geometry. The default is "brep".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dict
            The JSON data

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        vertices = Graph.Vertices(graph)
        j_data = {}
        j_data[verticesKey] = {}
        j_data[edgesKey] = {}
        n = max(len(str(len(vertices))), 4)
        v_labels = []
        v_dicts = []
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, xKey, Vertex.X(v, mantissa=mantissa))
            d = Dictionary.SetValueAtKey(d, yKey, Vertex.Y(v, mantissa=mantissa))
            d = Dictionary.SetValueAtKey(d, zKey, Vertex.Z(v, mantissa=mantissa))
            if geometryKey:
                v_d = Topology.Dictionary(v)
                brep = Dictionary.ValueAtKey(v_d, geometryKey)
                if brep:
                    d = Dictionary.SetValueAtKey(d, geometryKey, brep)
            v_dict = Dictionary.PythonDictionary(d)
            v_label = Dictionary.ValueAtKey(d, vertexLabelKey)
            if isinstance(v_label, str):
                v_label = Dictionary.ValueAtKey(d, vertexLabelKey)
            else:
                v_label = "Vertex_"+str(i).zfill(n)
            v_labels.append(v_label)
            v_dicts.append(v_dict)
        v_labels = Helper.MakeUnique(v_labels)
        for i, v_label in enumerate(v_labels):
            j_data[verticesKey][v_label] = v_dicts[i]

        edges = Graph.Edges(graph)
        n = len(str(len(edges)))    
        e_labels = []
        e_dicts = []
        for i, e in enumerate(edges):
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)
            svi = Vertex.Index(sv, vertices, tolerance=tolerance)
            evi = Vertex.Index(ev, vertices, tolerance=tolerance)
            if not svi == None and not evi == None:
                sv_label = v_labels[svi]
                ev_label = v_labels[evi]
                d = Topology.Dictionary(e)
                
                d = Dictionary.SetValueAtKey(d, sourceKey, sv_label)
                d = Dictionary.SetValueAtKey(d, targetKey, ev_label)
                e_dict = Dictionary.PythonDictionary(d)
                e_label = Dictionary.ValueAtKey(d, edgeLabelKey)
                if isinstance(e_label, str):
                    e_label = Dictionary.ValueAtKey(d, edgeLabelKey)
                else:
                    e_label = "Edge_"+str(i).zfill(n)
                e_labels.append(e_label)
                e_dicts.append(e_dict)
        e_labels = Helper.MakeUnique(e_labels)
        for i, e_label in enumerate(e_labels):
            j_data[edgesKey][e_label] = e_dicts[i]

        return j_data
    
    @staticmethod
    def JSONString(graph,
                   verticesKey="vertices",
                   edgesKey="edges",
                   vertexLabelKey="",
                   edgeLabelKey="",
                   xKey = "x",
                   yKey = "y",
                   zKey = "z",
                   indent=4,
                   sortKeys=False,
                   mantissa=6):
        """
        Converts the input graph into JSON data.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        verticesKey : str , optional
            The desired key name to call vertices. The default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. The default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        xKey : str , optional
            The desired key name to use for x-coordinates. The default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. The default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. The default is "z".
        indent : int , optional
            The desired amount of indent spaces to use. The default is 4.
        sortKeys : bool , optional
            If set to True, the keys will be sorted. Otherwise, they won't be. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        str
            The JSON str

        """
        import json
        json_data = Graph.JSONData(graph, verticesKey=verticesKey, edgesKey=edgesKey, vertexLabelKey=vertexLabelKey, edgeLabelKey=edgeLabelKey, xKey=xKey, yKey=yKey, zKey=zKey, mantissa=mantissa)
        json_string = json.dumps(json_data, indent=indent, sort_keys=sortKeys)
        return json_string
    
    @staticmethod
    def LocalClusteringCoefficient(graph, vertices: list = None, key: str = "lcc", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the local clustering coefficient of the input list of vertices within the input graph. See https://en.wikipedia.org/wiki/Clustering_coefficient.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. If set to None, the local clustering coefficient of all vertices will be computed. The default is None.
        key : str , optional
            The dictionary key under which to save the local clustering coefficient score. The default is "lcc".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        list
            The list of local clustering coefficient. The order of the list matches the order of the list of input vertices.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def local_clustering_coefficient(adjacency_matrix, node):
            """
            Compute the local clustering coefficient for a given node in a graph represented by an adjacency matrix.

            Parameters:
            - adjacency_matrix: 2D list representing the adjacency matrix of the graph
            - node: Node for which the local clustering coefficient is computed

            Returns:
            - Local clustering coefficient for the given node
            """
            neighbors = [i for i, value in enumerate(adjacency_matrix[node]) if value == 1]
            num_neighbors = len(neighbors)

            if num_neighbors < 2:
                # If the node has less than 2 neighbors, the clustering coefficient is undefined
                return 0.0

            # Count the number of connections between the neighbors
            num_connections = 0
            for i in range(num_neighbors):
                for j in range(i + 1, num_neighbors):
                    if adjacency_matrix[neighbors[i]][neighbors[j]] == 1:
                        num_connections += 1
            # Calculate the local clustering coefficient
            local_clustering_coeff = 2.0 * num_connections / (num_neighbors * (num_neighbors - 1))

            return local_clustering_coeff
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.LocalClusteringCoefficient - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if vertices == None:
            vertices = Graph.Vertices(graph)
        if Topology.IsInstance(vertices, "Vertex"):
            vertices = [vertices]
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Graph.LocalClusteringCoefficient - Error: The input vertices parameter does not contain valid vertices. Returning None.")
            return None
        g_vertices = Graph.Vertices(graph)
        adjacency_matrix = Graph.AdjacencyMatrix(graph)
        scores = []
        for v in vertices:
            i = Vertex.Index(v, g_vertices, tolerance=tolerance)
            if not i == None:
                lcc_score = round(local_clustering_coefficient(adjacency_matrix, i), mantissa)
                d = Topology.Dictionary(v)
                d = Dictionary.SetValueAtKey(d, key, lcc_score)
                v = Topology.SetDictionary(v, d)
                scores.append(lcc_score)
            else:
                scores.append(None)
        return scores
    
    @staticmethod
    def LongestPath(graph, vertexA, vertexB, vertexKey=None, edgeKey=None, costKey=None, timeLimit=10, tolerance=0.0001):
        """
        Returns the longest path that connects the input vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        vertexKey : str , optional
            The vertex key to maximize. If set the vertices dictionaries will be searched for this key and the associated value will be used to compute the longest path that maximizes the total value. The value must be numeric. The default is None.
        edgeKey : str , optional
            The edge key to maximize. If set the edges dictionaries will be searched for this key and the associated value will be used to compute the longest path that maximizes the total value. The value of the key must be numeric. If set to "length" (case insensitive), the shortest path by length is computed. The default is "length".
        costKey : str , optional
            If not None, the total cost of the longest_path will be stored in its dictionary under this key. The default is None. 
        timeLimit : int , optional
            The time limit in second. The default is 10 seconds.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The longest path between the input vertices.

        """
        from topologicpy. Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
    
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.LongestPath - Error: the input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.LongestPath - Error: the input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.LongestPath - Error: the input vertexB is not a valid vertex. Returning None.")
            return None
        
        g_edges = Graph.Edges(graph)

        paths = Graph.AllPaths(graph, vertexA, vertexB, timeLimit=timeLimit)
        if not paths:
            print("Graph.LongestPath - Error: Could not find any paths within the specified time limit. Returning None.")
            return None
        if len(paths) < 1:
            print("Graph.LongestPath - Error: Could not find any paths within the specified time limit. Returning None.")
            return None
        if edgeKey == None:
            lengths = [len(Topology.Edges(path)) for path in paths]
        elif edgeKey.lower() == "length":
            lengths = [Wire.Length(path) for path in paths]
        else:
            lengths = []
            for path in paths:
                edges = Topology.Edges(path)
                pathCost = 0
                for edge in edges:
                    index = Edge.Index(edge, g_edges)
                    d = Topology.Dictionary(g_edges[index])
                    value = Dictionary.ValueAtKey(d, edgeKey)
                    if not value == None:
                        pathCost += value
                lengths.append(pathCost)
        if not vertexKey == None:
            g_vertices = Graph.Vertices(graph)
            for i, path in enumerate(paths):
                vertices = Topology.Vertices(path)
                pathCost = 0
                for vertex in vertices:
                    index = Vertex.Index(vertex, g_vertices, tolerance=tolerance)
                    if not index == None:
                        d = Topology.Dictionary(g_vertices[index])
                        value = Dictionary.ValueAtKey(d, vertexKey)
                    if not value == None:
                        pathCost += value
                lengths[i] += pathCost
        new_paths = Helper.Sort(paths, lengths)
        temp_path = new_paths[-1]
        cost = lengths[-1]
        new_edges = []
        for edge in Topology.Edges(temp_path):
            new_edges.append(g_edges[Edge.Index(edge, g_edges)])
        longest_path = Topology.SelfMerge(Cluster.ByTopologies(new_edges), tolerance=tolerance)

        sv = Topology.Vertices(longest_path)[0]
        if Vertex.Distance(sv, vertexB) < tolerance: # Wire is reversed. Re-reverse it
            if Topology.IsInstance(longest_path, "Edge"):
                longest_path = Edge.Reverse(longest_path)
            elif Topology.IsInstance(longest_path, "Wire"):
                longest_path = Wire.Reverse(longest_path)
                longest_path = Wire.OrientEdges(longest_path, Wire.StartVertex(longest_path), tolerance=tolerance)
        if not costKey == None:
            lengths.sort()
            d = Dictionary.ByKeysValues([costKey], [cost])
            longest_path = Topology.SetDictionary(longest_path, d)
        return longest_path

    @staticmethod
    def MaximumDelta(graph):
        """
        Returns the maximum delta of the input graph. The maximum delta of a graph is the maximum degree of a vertex in the graph. 

        Parameters
        ----------
        graph : topologic_core.Graph
            the input graph.

        Returns
        -------
        int
            The maximum delta.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.MaximumDelta - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.MaximumDelta()
    
    @staticmethod
    def MaximumFlow(graph, source, sink, edgeKeyFwd=None, edgeKeyBwd=None, bidirKey=None, bidirectional=False, residualKey="residual", tolerance=0.0001):
        """
        Returns the maximum flow of the input graph. See https://en.wikipedia.org/wiki/Maximum_flow_problem 

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph. This is assumed to be a directed graph
        source : topologic_core.Vertex
            The input source vertex.
        sink : topologic_core.Vertex
            The input sink/target vertex.
        edgeKeyFwd : str , optional
            The edge dictionary key to use to find the value of the forward capacity of the edge. If not set, the length of the edge is used as its capacity. The default is None.
        edgeKeyBwd : str , optional
            The edge dictionary key to use to find the value of the backward capacity of the edge. This is only considered if the edge is set to be bidrectional. The default is None.
        bidirKey : str , optional
            The edge dictionary key to use to determine if the edge is bidrectional. The default is None.
        bidrectional : bool , optional
            If set to True, the whole graph is considered to be bidirectional. The default is False.
        residualKey : str , optional
            The name of the key to use to store the residual value of each edge capacity in the input graph. The default is "residual".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        float
            The maximum flow.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        # Using BFS as a searching algorithm 
        def searching_algo_BFS(adjMatrix, s, t, parent):

            visited = [False] * (len(adjMatrix))
            queue = []

            queue.append(s)
            visited[s] = True

            while queue:

                u = queue.pop(0)

                for ind, val in enumerate(adjMatrix[u]):
                    if visited[ind] == False and val > 0:
                        queue.append(ind)
                        visited[ind] = True
                        parent[ind] = u

            return True if visited[t] else False

        # Applying fordfulkerson algorithm
        def ford_fulkerson(adjMatrix, source, sink):
            am = adjMatrix.copy()
            row = len(am)
            parent = [-1] * (row)
            max_flow = 0

            while searching_algo_BFS(am, source, sink, parent):

                path_flow = float("Inf")
                s = sink
                while(s != source):
                    path_flow = min(path_flow, am[parent[s]][s])
                    s = parent[s]

                # Adding the path flows
                max_flow += path_flow

                # Updating the residual values of edges
                v = sink
                while(v != source):
                    u = parent[v]
                    am[u][v] -= path_flow
                    am[v][u] += path_flow
                    v = parent[v]
            return [max_flow, am]
        if edgeKeyFwd == None:
            useEdgeLength = True
        else:
            useEdgeLength = False
        adjMatrix = Graph.AdjacencyMatrix(graph, edgeKeyFwd=edgeKeyFwd, edgeKeyBwd=edgeKeyBwd, bidirKey=bidirKey, bidirectional=bidirectional, useEdgeIndex = False, useEdgeLength=useEdgeLength, tolerance=tolerance)
        edgeMatrix = Graph.AdjacencyMatrix(graph, edgeKeyFwd=edgeKeyFwd, edgeKeyBwd=edgeKeyBwd, bidirKey=bidirKey, bidirectional=bidirectional, useEdgeIndex = True, useEdgeLength=False, tolerance=tolerance)
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)
        sourceIndex = Vertex.Index(source, vertices, tolerance=tolerance)
        sinkIndex = Vertex.Index(sink, vertices, tolerance=tolerance)
        max_flow = None
        if not sourceIndex == None and not sinkIndex == None:
            max_flow, am = ford_fulkerson(adjMatrix=adjMatrix, source=sourceIndex, sink=sinkIndex)
            for i in range(len(am)):
                row = am[i]
                for j in range(len(row)):
                    residual = am[i][j]
                    edge = edges[edgeMatrix[i][j]-1]
                    d = Topology.Dictionary(edge)
                    if not d == None:
                        keys = Dictionary.Keys(d)
                        values = Dictionary.Values(d)
                    else:
                        keys = []
                        values = []
                    keys.append(residualKey)
                    values.append(residual)
                    d = Dictionary.ByKeysValues(keys, values)
                    edge = Topology.SetDictionary(edge, d)
        return max_flow

    @staticmethod
    def MeshData(g, tolerance: float = 0.0001):
        """
        Returns the mesh data of the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dict
            The python dictionary of the mesh data of the input graph. The keys in the dictionary are:
            'vertices' : The list of [x, y, z] coordinates of the vertices.
            'edges' : the list of [i, j] indices into the vertices list to signify and edge that connects vertices[i] to vertices[j].
            'vertexDictionaries' : The python dictionaries of the vertices (in the same order as the list of vertices).
            'edgeDictionaries' : The python dictionaries of the edges (in the same order as the list of edges).

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        g_vertices = Graph.Vertices(g)
        m_vertices = []
        v_dicts = []
        for g_vertex in g_vertices:
            m_vertices.append(Vertex.Coordinates(g_vertex))
            d = Dictionary.PythonDictionary(Topology.Dictionary(g_vertex))
            v_dicts.append(d)
        g_edges = Graph.Edges(g)
        m_edges = []
        e_dicts = []
        for g_edge in g_edges:
            sv = g_edge.StartVertex()
            ev = g_edge.EndVertex()
            si = Vertex.Index(sv, g_vertices, tolerance=tolerance)
            ei = Vertex.Index(ev, g_vertices, tolerance=tolerance)
            if (not si == None) and (not ei == None):
                m_edges.append([si, ei])
                d = Dictionary.PythonDictionary(Topology.Dictionary(g_edge))
                e_dicts.append(d)
        return {'vertices':m_vertices,
                'edges': m_edges,
                'vertexDictionaries': v_dicts,
                'edgeDictionaries': e_dicts
                }
    
    @staticmethod
    def MetricDistance(graph, vertexA, vertexB, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the shortest-path distance between the input vertices. See https://en.wikipedia.org/wiki/Distance_(graph_theory).

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        float
            The shortest-path metric distance between the input vertices.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.MetricDistance - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.MetricDistance - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.MetricDistance - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        sp = Graph.ShortestPath(graph, vertexA, vertexB, vertexKey="", edgeKey="Length", tolerance=tolerance)
        if Topology.IsInstance(sp, "Wire"):
            dist = round(Wire.Length(sp), mantissa)
        elif Topology.IsInstance(sp, "Edge"):
            dist = round(Edge.Length(sp), mantissa)
        else:
            dist = float('inf')
        return dist
    
    @staticmethod
    def MinimumDelta(graph):
        """
        Returns the minimum delta of the input graph. The minimum delta of a graph is the minimum degree of a vertex in the graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        int
            The minimum delta.

        """
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.MinimumDelta - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.MinimumDelta()
    
    @staticmethod
    def MinimumSpanningTree(graph, edgeKey=None, tolerance=0.0001):
        """
        Returns the minimum spanning tree of the input graph. See https://en.wikipedia.org/wiki/Minimum_spanning_tree.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        edgeKey : string , optional
            If set, the value of the edgeKey will be used as the weight and the tree will minimize the weight. The value associated with the edgeKey must be numerical. If the key is not set, the edges will be sorted by their length. The default is None
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The minimum spanning tree.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        def vertexInList(vertex, vertexList, tolerance=0.0001):
            for v in vertexList:
                if Vertex.Distance(v, vertex) < tolerance:
                    return True
            return False
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.MinimumSpanningTree - Error: The input graph is not a valid graph. Returning None.")
            return None
        edges = Graph.Edges(graph)
        vertices = Graph.Vertices(graph)
        values = []
        if isinstance(edgeKey, str):
            for edge in edges:
                d = Dictionary.Dictionary(edge)
                value = Dictionary.ValueAtKey(d, edgeKey)
                if value == None or not isinstance(value, int) or not isinstance(value, float):
                    return None
                values.append(value)
        else:
            for edge in edges:
                value = Edge.Length(edge)
                values.append(value)
        keydict = dict(zip(edges, values))
        edges.sort(key=keydict.get)
        mst = Graph.ByVerticesEdges(vertices,[])
        for edge in edges:
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            if len(Graph.Vertices(mst)) > 0:
                if not Graph.Path(mst, Graph.NearestVertex(mst, sv), Graph.NearestVertex(mst, ev)):
                    d = Topology.Dictionary(edge)
                    if len(Dictionary.Keys(d)) > 0:
                        tranEdgeDicts = True
                    else:
                        tranEdgeDicts = False
                    mst = Graph.AddEdge(mst, edge, transferVertexDictionaries=False, transferEdgeDictionaries=tranEdgeDicts)
        return mst

    @staticmethod
    def NavigationGraph(face, sources=None, destinations=None, tolerance=0.0001, numWorkers=None):
        """
        Creates a 2D navigation graph.

        Parameters
        ----------
        face : topologic_core.Face
            The input boundary. View edges will be clipped to this face. The holes in the face are used as the obstacles
        sources : list
            The first input list of sources (vertices). Navigation edges will connect these veritces to destinations.
        destinations : list
            The input list of destinations (vertices). Navigation edges will connect these vertices to sources.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        numWorkers : int, optional
            Number of workers run in parallel to process. The default is None which sets the number to twice the number of CPU cores.

        Returns
        -------
        topologic_core.Graph
            The navigation graph.

        """

        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster

        if not numWorkers:
            import multiprocessing
            numWorkers = multiprocessing.cpu_count()*2

        if not Topology.IsInstance(face, "Face"):
            print("Graph.NavigationGraph - Error: The input face parameter is not a valid face. Returning None")
            return None
        if sources == None:
            sources = Topology.Vertices(face)
        if destinations == None:
            destinations = Topology.Vertices(face)

        if not isinstance(sources, list):
            print("Graph.NavigationGraph - Error: The input sources parameter is not a valid list. Returning None")
            return None
        if not isinstance(destinations, list):
            print("Graph.NavigationGraph - Error: The input destinations parameter is not a valid list. Returning None")
            return None
        sources = [v for v in sources if Topology.IsInstance(v, "Vertex")]
        if len(sources) < 1:
            print("Graph.NavigationGraph - Error: The input sources parameter does not contain any vertices. Returning None")
            return None
        destinations = [v for v in destinations if Topology.IsInstance(v, "Vertex")]

        # Add obstuse angles of external boundary to viewpoints
        e_boundary = Face.ExternalBoundary(face)
        if Topology.IsInstance(e_boundary, "Wire"):
            vertices = Topology.Vertices(e_boundary)
            interior_angles = Wire.InteriorAngles(e_boundary)
            for i, ang in enumerate(interior_angles):
                if ang > 180:
                    sources.append(vertices[i])
                    destinations.append(vertices[i])
        i_boundaries = Face.InternalBoundaries(face)
        for i_boundary in i_boundaries:
            if Topology.IsInstance(i_boundary, "Wire"):
                vertices = Topology.Vertices(i_boundary)
                interior_angles = Wire.InteriorAngles(i_boundary)
                for i, ang in enumerate(interior_angles):
                    if ang < 180:
                        sources.append(vertices[i])
                        destinations.append(vertices[i])
        used = []
        for i in range(max(len(sources), len(destinations))):
            temp_row = []
            for j in  range(max(len(sources), len(destinations))):
                temp_row.append(0)
            used.append(temp_row)

        queue = Queue()
        mergingProcess = MergingProcess(queue)
        mergingProcess.start()

        sources_str = [Topology.BREPString(s) for s in sources]
        destinations_str = [Topology.BREPString(s) for s in destinations]
        face_str = Topology.BREPString(face)
        workerProcessPool = WorkerProcessPool(numWorkers,
                                              queue,
                                              used,
                                              face_str,
                                              sources_str,
                                              destinations_str,
                                              tolerance)
        workerProcessPool.startProcesses()
        workerProcessPool.join()

        queue.put_nowait(None)
        it = queue.get()
        final_edges = [Topology.ByBREPString(edge_str) for edge_str in it.edges]
        mergingProcess.join()

        if len(i_boundaries) > 0:
            holes_edges = Topology.Edges(Cluster.ByTopologies(i_boundaries))
            final_edges += holes_edges
        if len(final_edges) > 0:
            final_vertices = Topology.Vertices(Cluster.ByTopologies(final_edges))
            g = Graph.ByVerticesEdges(final_vertices, final_edges)
            return g
        return None

    @staticmethod
    def NearestVertex(graph, vertex):
        """
        Returns the vertex in the input graph that is the nearest to the input vertex.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.

        Returns
        -------
        topologic_core.Vertex
            The vertex in the input graph that is the nearest to the input vertex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.NearestVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.NearestVertex - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        vertices = Graph.Vertices(graph)

        nearestVertex = vertices[0]
        nearestDistance = Vertex.Distance(vertex, nearestVertex)
        for aGraphVertex in vertices:
            newDistance = Vertex.Distance(vertex, aGraphVertex)
            if newDistance < nearestDistance:
                nearestDistance = newDistance
                nearestVertex = aGraphVertex
        return nearestVertex

    @staticmethod
    def NetworkXGraph(graph, mantissa: int = 6, tolerance: float = 0.0001):
        """
        converts the input graph into a NetworkX Graph. See http://networkx.org

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        networkX Graph
            The created networkX Graph

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        try:
            import networkx as nx
        except:
            print("Graph.NetworkXGraph - Information: Installing required networkx library.")
            try:
                os.system("pip install networkx")
            except:
                os.system("pip install networkx --user")
            try:
                import networkx as nx
                print("Graph.NetworkXGraph - Infromation: networkx library installed correctly.")
            except:
                warnings.warn("Graph - Error: Could not import networkx. Please try to install networkx manually. Returning None.")
                return None
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.NetworkXGraph - Error: The input graph is not a valid graph. Returning None.")
            return None

        nxGraph = nx.Graph()
        vertices = Graph.Vertices(graph)
        order = len(vertices)
        nodes = []
        for i in range(order):
            v = vertices[i]
            d = Topology.Dictionary(vertices[i])
            if d:
                keys = Dictionary.Keys(d)
                if not keys:
                    keys = []
                values = Dictionary.Values(d)
                if not values:
                    values = []
                keys += ["x","y","z"]
                import random
                values += [Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa), Vertex.Z(v, mantissa=mantissa)]
                d = Dictionary.ByKeysValues(keys,values)
                pythonD = Dictionary.PythonDictionary(d)
                nodes.append((i, pythonD))
            else:
                nodes.append((i, {"name": str(i)}))
        nxGraph.add_nodes_from(nodes)
        for i in range(order):
            v = vertices[i]
            adjVertices = Graph.AdjacentVertices(graph, vertices[i])
            for adjVertex in adjVertices:
                adjIndex = Vertex.Index(vertex=adjVertex, vertices=vertices, strict=True, tolerance=tolerance)
                if not adjIndex == None:
                    nxGraph.add_edge(i,adjIndex, length=(Vertex.Distance(v, adjVertex, mantissa=mantissa)))

        pos=nx.spring_layout(nxGraph, k=0.2)
        nx.set_node_attributes(nxGraph, pos, "pos")
        return nxGraph

    @staticmethod
    def Order(graph):
        """
        Returns the graph order of the input graph. The graph order is its number of vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        int
            The number of vertices in the input graph

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Order - Error: The input graph is not a valid graph. Returning None.")
            return None
        return len(Graph.Vertices(graph))
    
    @staticmethod
    def OutgoingEdges(graph, vertex, directed: bool = False, tolerance: float = 0.0001) -> list:
        """
        Returns the outgoing edges connected to a vertex. An edge is considered outgoing if its start vertex is
        coincident with the input vertex.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        directed : bool , optional
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of outgoing edges

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.IncomingEdges - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.IncomingEdges - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        
        edges = Graph.Edges(graph, [vertex])
        if directed == False:
            return edges
        outgoing_edges = []
        for edge in edges:
            sv = Edge.StartVertex(edge)
            if Vertex.Distance(vertex, sv) < tolerance:
                outgoing_edges.append(edge)
        return outgoing_edges
    
    @staticmethod
    def OutgoingVertices(graph, vertex, directed: bool = False, tolerance: float = 0.0001) -> list:
        """
        Returns the list of outgoing vertices connected to a vertex. A vertex is considered outgoing if it is an adjacent vertex to the input vertex
        and the the edge connecting it to the input vertex is an outgoing edge.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        directed : bool , optional
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of incoming vertices

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.OutgoingVertices - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.OutgoingVertices - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        
        if directed == False:
            return Graph.AdjacentVertices(graph, vertex)
        outgoing_edges = Graph.OutgoingEdges(graph, vertex, directed=directed, tolerance=tolerance)
        outgoing_vertices = []
        for edge in outgoing_edges:
            ev = Edge.EndVertex(edge)
            outgoing_vertices.append(Graph.NearestVertex(graph, ev))
        return outgoing_vertices
    
    @staticmethod
    def PageRank(graph, alpha: float = 0.85, maxIterations: int = 100, normalize: bool = True, directed: bool = False, key: str = "page_rank", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Calculates PageRank scores for nodes in a directed graph. see https://en.wikipedia.org/wiki/PageRank.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        alpha : float , optional
            The damping (dampening) factor. The default is 0.85. See https://en.wikipedia.org/wiki/PageRank.
        maxIterations : int , optional
            The maximum number of iterations to calculate the page rank. The default is 100.
        normalize : bool , optional
            If set to True, the results will be normalized from 0 to 1. Otherwise, they won't be. The default is True.
        directed : bool , optional
            If set to True, the graph is considered as a directed graph. Otherwise, it will be considered as an undirected graph. The default is False.
        key : str , optional
            The dictionary key under which to save the page_rank score. The default is "page_rank"
        mantissa : int , optional
            The desired length of the mantissa.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of page ranks for the vertices in the graph.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        vertices = Graph.Vertices(graph)
        num_vertices = len(vertices)
        if num_vertices < 1:
            print("Graph.PageRank - Error: The input graph parameter has no vertices. Returning None")
            return None
        initial_score = 1.0 / num_vertices
        scores = [initial_score for vertex in vertices]
        for _ in range(maxIterations):
            new_scores = [0 for vertex in vertices]
            for i, vertex in enumerate(vertices):
                incoming_score = 0
                for incoming_vertex in Graph.IncomingVertices(graph, vertex, directed=directed):
                    if len(Graph.IncomingVertices(graph, incoming_vertex, directed=directed)) > 0:
                        vi = Vertex.Index(incoming_vertex, vertices, tolerance=tolerance)
                        if not vi == None:
                            incoming_score += scores[vi] / len(Graph.IncomingVertices(graph, incoming_vertex, directed=directed))
                new_scores[i] = alpha * incoming_score + (1 - alpha) / num_vertices

            # Check for convergence
            if all(abs(new_scores[i] - scores[i]) < tolerance for i in range(len(vertices))):
                break

            scores = new_scores
        if normalize == True:
            scores = Helper.Normalize(scores, mantissa=mantissa)
        else:
            scores = [round(x, mantissa) for x in scores]
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, key, scores[i])
            v = Topology.SetDictionary(v, d)
        return scores
    
    @staticmethod
    def Path(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Returns a path (wire) in the input graph that connects the input vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The path (wire) in the input graph that connects the input vertices.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Path - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.Path - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.Path - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        path = graph.Path(vertexA, vertexB)
        if Topology.IsInstance(path, "Wire"):
            path = Wire.OrientEdges(path, Wire.StartVertex(path), tolerance=tolerance)
        return path

    
    @staticmethod
    def PyvisGraph(graph, path, overwrite: bool = True, height: int = 900, backgroundColor: str = "white",
                   fontColor: str = "black", notebook: bool = False,
                   vertexSize: int = 6, vertexSizeKey: str = None, vertexColor: str = "black",
                   vertexColorKey: str = None, vertexLabelKey: str = None, vertexGroupKey: str = None,
                   vertexGroups: list = None, minVertexGroup: float = None, maxVertexGroup: float = None, 
                   edgeLabelKey: str = None, edgeWeight: int = 0, edgeWeightKey: str = None,
                   showNeighbours: bool = True, selectMenu: bool = True,
                   filterMenu: bool = True, colorScale: str = "viridis", tolerance: float = 0.0001):
        """
        Displays a pyvis graph. See https://pyvis.readthedocs.io/.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        path : str
            The desired file path to the HTML file into which to save the pyvis graph.
        overwrite : bool , optional
            If set to True, the HTML file is overwritten.
        height : int , optional
            The desired figure height in pixels. The default is 900 pixels.
        backgroundColor : str, optional
            The desired background color for the figure. This can be a named color or a hexadecimal value. The default is 'white'.
        fontColor : str , optional
            The desired font color for the figure. This can be a named color or a hexadecimal value. The default is 'black'.
        notebook : bool , optional
            If set to True, the figure will be targeted at a Jupyter Notebook. Note that this is not working well. Pyvis has bugs. The default is False.
        vertexSize : int , optional
            The desired default vertex size. The default is 6.
        vertexSizeKey : str , optional
            If not set to None, the vertex size will be derived from the dictionary value set at this key. If set to "degree", the size of the vertex will be determined by its degree (number of neighbors). The default is None.
        vertexColor : str , optional
            The desired default vertex color. his can be a named color or a hexadecimal value. The default is 'black'.
        vertexColorKey : str , optional
            If not set to None, the vertex color will be derived from the dictionary value set at this key. The default is None.
        vertexLabelKey : str , optional
            If not set to None, the vertex label will be derived from the dictionary value set at this key. The default is None.
        vertexGroupKey : str , optional
            If not set to None, the vertex color will be determined by the group the vertex belongs to as derived from the value set at this key. The default is None.
        vertexGroups : list , optional
            The list of all possible vertex groups. This will help in vertex coloring. The default is None.
        minVertexGroup : int or float , optional
            If the vertex groups are numeric, specify the minimum value you wish to consider for vertex coloring. The default is None.
        maxVertexGroup : int or float , optional
            If the vertex groups are numeric, specify the maximum value you wish to consider for vertex coloring. The default is None.
        
        edgeWeight : int , optional
            The desired default weight of the edge. This determines its thickness. The default is 0.
        edgeWeightKey : str, optional
            If not set to None, the edge weight will be derived from the dictionary value set at this key. If set to "length" or "distance", the weight of the edge will be determined by its geometric length. The default is None.
        edgeLabelKey : str , optional
            If not set to None, the edge label will be derived from the dictionary value set at this key. The default is None.
        showNeighbors : bool , optional
            If set to True, a list of neighbors is shown when you hover over a vertex. The default is True.
        selectMenu : bool , optional
            If set to True, a selection menu will be displayed. The default is True
        filterMenu : bool , optional
            If set to True, a filtering menu will be displayed. The default is True.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). The default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        Returns
        -------
        None
            The pyvis graph is displayed either inline (notebook mode) or in a new browser window or tab.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        from os.path import exists

        try:
            from pyvis.network import Network
        except:
            print("Graph.PyvisGraph - Information: Installing required pyvis library.")
            try:
                os.system("pip install pyvis")
            except:
                os.system("pip install pyvis --user")
            try:
                from pyvis.network import Network
                print("Graph.PyvisGraph - Information: pyvis library installed correctly.")
            except:
                warnings.warn("Graph - Error: Could not import pyvis. Please try to install pyvis manually. Returning None.")
                return None
        
        net = Network(height=str(height)+"px", width="100%", bgcolor=backgroundColor, font_color=fontColor, select_menu=selectMenu, filter_menu=filterMenu, cdn_resources="remote", notebook=notebook)
        if notebook == True:
            net.prep_notebook()
        
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)

        nodes = [i for i in range(len(vertices))]
        if not vertexLabelKey == None:
            node_labels = [Dictionary.ValueAtKey(Topology.Dictionary(v), vertexLabelKey) for v in vertices]
        else:
            node_labels = list(range(len(vertices)))
        if not vertexColorKey == None:
            colors = [Dictionary.ValueAtKey(Topology.Dictionary(v), vertexColorKey) for v in vertices]
        else:
            colors = [vertexColor for v in vertices]
        node_titles = [str(n) for n in node_labels]
        group = ""
        if not vertexGroupKey == None:
            colors = []
            if vertexGroups:
                if len(vertexGroups) > 0:
                    if type(vertexGroups[0]) == int or type(vertexGroups[0]) == float:
                        if not minVertexGroup:
                            minVertexGroup = min(vertexGroups)
                        if not maxVertexGroup:
                            maxVertexGroup = max(vertexGroups)
                    else:
                        minVertexGroup = 0
                        maxVertexGroup = len(vertexGroups) - 1
            else:
                minVertexGroup = 0
                maxVertexGroup = 1
            for m, v in enumerate(vertices):
                group = ""
                d = Topology.Dictionary(v)
                if d:
                    try:
                        group = Dictionary.ValueAtKey(d, key=vertexGroupKey) or None
                    except:
                        group = ""
                try:
                    if type(group) == int or type(group) == float:
                        if group < minVertexGroup:
                            group = minVertexGroup
                        if group > maxVertexGroup:
                            group = maxVertexGroup
                        color = Color.RGBToHex(Color.ByValueInRange(group, minValue=minVertexGroup, maxValue=maxVertexGroup, colorScale=colorScale))
                    else:
                        color = Color.RGBToHex(Color.ByValueInRange(vertexGroups.index(group), minValue=minVertexGroup, maxValue=maxVertexGroup, colorScale=colorScale))
                    colors.append(color)
                except:
                    colors.append(vertexColor)
        net.add_nodes(nodes, label=node_labels, title=node_titles, color=colors)

        for e in edges:
            edge_label = ""
            if not edgeLabelKey == None:
                d = Topology.Dictionary(e)
                edge_label = Dictionary.ValueAtKey(d, edgeLabelKey)
                if edge_label == None:
                    edge_label = ""
            w = edgeWeight
            if not edgeWeightKey == None:
                d = Topology.Dictionary(e)
                if edgeWeightKey.lower() == "length" or edgeWeightKey.lower() == "distance":
                    w = Edge.Length(e)
                else:
                    weightValue = Dictionary.ValueAtKey(d, edgeWeightKey)
                if weightValue:
                    w = weightValue
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)
            svi = Vertex.Index(sv, vertices, tolerance=tolerance)
            evi = Vertex.Index(ev, vertices, tolerance=tolerance)
            if (not svi == None) and (not evi == None):
                net.add_edge(svi, evi, weight=w, label=edge_label)
        net.inherit_edge_colors(False)
        
        # add neighbor data to node hover data and compute vertexSize
        if showNeighbours == True or not vertexSizeKey == None:
            for i, node in enumerate(net.nodes):
                if showNeighbours == True:
                    neighbors = list(net.neighbors(node["id"]))
                    neighbor_labels = [str(net.nodes[n]["id"])+": "+str(net.nodes[n]["label"]) for n in neighbors]
                    node["title"] = str(node["id"])+": "+node["title"]+"\n"
                    node["title"] += "Neighbors:\n" + "\n".join(neighbor_labels)
                vs = vertexSize
                if not vertexSizeKey == None:
                    d = Topology.Dictionary(vertices[i])
                    if vertexSizeKey.lower() == "neighbours" or vertexSizeKey.lower() == "degree":
                        temp_vs = Graph.VertexDegree(graph, vertices[i])
                    else:
                        temp_vs = Dictionary.ValueAtKey(vertices[i], vertexSizeKey)
                    if temp_vs:
                        vs = temp_vs
                node["value"] = vs
        
        # Make sure the file extension is .html
        ext = path[len(path)-5:len(path)]
        if ext.lower() != ".html":
            path = path+".html"
        if not overwrite and exists(path):
            print("Graph.PyvisGraph - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        if overwrite == True:
            net.save_graph(path)
        net.show_buttons()
        net.show(path, notebook=notebook)
        return None
        
    @staticmethod
    def RemoveEdge(graph, edge, tolerance=0.0001):
        """
        Removes the input edge from the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input edge removed.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.RemoveEdge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(edge, "Edge"):
            print("Graph.RemoveEdge - Error: The input edge is not a valid edge. Returning None.")
            return None
        _ = graph.RemoveEdges([edge], tolerance)
        return graph
    
    @staticmethod
    def RemoveVertex(graph, vertex, tolerance=0.0001):
        """
        Removes the input vertex from the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input vertex removed.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.RemoveVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.RemoveVertex - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        graphVertex = Graph.NearestVertex(graph, vertex)
        _ = graph.RemoveVertices([graphVertex])
        return graph

    @staticmethod
    def SetDictionary(graph, dictionary):
        """
        Sets the input graph's dictionary to the input dictionary

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        dictionary : topologic_core.Dictionary or dict
            The input dictionary.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input dictionary set in it.

        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.SetDictionary - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        if isinstance(dictionary, dict):
            dictionary = Dictionary.ByPythonDictionary(dictionary)
        if not Topology.IsInstance(dictionary, "Dictionary"):
            print("Graph.SetDictionary - Warning: the input dictionary parameter is not a valid dictionary. Returning original input.")
            return graph
        if len(dictionary.Keys()) < 1:
            print("Graph.SetDictionary - Warning: the input dictionary parameter is empty. Returning original input.")
            return graph
        _ = graph.SetDictionary(dictionary)
        return graph

    @staticmethod
    def ShortestPath(graph, vertexA, vertexB, vertexKey="", edgeKey="Length", tolerance=0.0001):
        """
        Returns the shortest path that connects the input vertices. The shortest path will take into consideration both the vertexKey and the edgeKey if both are specified and will minimize the total "cost" of the path. Otherwise, it will take into consideration only whatever key is specified.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        vertexKey : string , optional
            The vertex key to minimise. If set the vertices dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value must be numeric. The default is None.
        edgeKey : string , optional
            The edge key to minimise. If set the edges dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value of the key must be numeric. If set to "length" (case insensitive), the shortest path by length is computed. The default is "length".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Wire
            The shortest path between the input vertices.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ShortestPath - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.ShortestPath - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.ShortestPath - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        if edgeKey:
            if edgeKey.lower() == "length":
                edgeKey = "Length"
        try:
            gsv = Graph.NearestVertex(graph, vertexA)
            gev = Graph.NearestVertex(graph, vertexB)
            shortest_path = graph.ShortestPath(gsv, gev, vertexKey, edgeKey)
            if not shortest_path == None:
                if Topology.IsInstance(shortest_path, "Edge"):
                        shortest_path = Wire.ByEdges([shortest_path])
                sv = Topology.Vertices(shortest_path)[0]
                if Vertex.Distance(sv, gev) < tolerance: # Path is reversed. Correct it.
                    if Topology.IsInstance(shortest_path, "Wire"):
                        shortest_path = Wire.Reverse(shortest_path)
                shortest_path = Wire.OrientEdges(shortest_path, Wire.StartVertex(shortest_path), tolerance=tolerance)
            return shortest_path
        except:
            return None

    @staticmethod
    def ShortestPaths(graph, vertexA, vertexB, vertexKey="", edgeKey="length", timeLimit=10,
                           pathLimit=10, tolerance=0.0001):
        """
        Returns the shortest path that connects the input vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        vertexKey : string , optional
            The vertex key to minimise. If set the vertices dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value must be numeric. The default is None.
        edgeKey : string , optional
            The edge key to minimise. If set the edges dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value of the key must be numeric. If set to "length" (case insensitive), the shortest path by length is computed. The default is "length".
        timeLimit : int , optional
            The search time limit in seconds. The default is 10 seconds
        pathLimit: int , optional
            The number of found paths limit. The default is 10 paths.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of shortest paths between the input vertices.

        """
        from topologicpy.Topology import Topology
        
        def isUnique(paths, path):
            if path == None:
                return False
            if len(paths) < 1:
                return True
            for aPath in paths:
                copyPath = topologic.Topology.DeepCopy(aPath) # Hook to Core
                dif = copyPath.Difference(path, False)
                if dif == None:
                    return False
            return True
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.ShortestPaths - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.ShortestPaths - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.ShortestPaths - Error: The input vertexB parameter is not a valid vertex. Returning None.")
            return None
        shortestPaths = []
        end = time.time() + timeLimit
        while time.time() < end and len(shortestPaths) < pathLimit:
            if (graph != None):
                if edgeKey:
                    if edgeKey.lower() == "length":
                        edgeKey = "Length"
                shortest_path = Graph.ShortestPath(graph, vertexA, vertexB, vertexKey=vertexKey, edgeKey=edgeKey, tolerance=tolerance) # Find the first shortest path
                if isUnique(shortestPaths, shortest_path):
                    shortestPaths.append(shortest_path)
                vertices = Graph.Vertices(graph)
                random.shuffle(vertices)
                edges = Graph.Edges(graph)
                graph = Graph.ByVerticesEdges(vertices, edges)
        return shortestPaths

    @staticmethod
    def Show(graph, vertexColor="black",
             vertexSize=6,
             vertexLabelKey=None,
             vertexGroupKey=None,
             vertexGroups=[],
             showVertices=True,
             showVertexLabels=False,
             showVertexLegend=False,
             edgeColor="black",
             edgeWidth=1,
             edgeLabelKey=None,
             edgeGroupKey=None,
             edgeGroups=[],
             showEdges=True,
             showEdgeLabels=False,
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
             tolerance=0.0001):
        """
        Shows the graph using Plotly.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexSize : float , optional
            The desired size of the vertices. The default is 1.1.
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. The default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. The default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. The default is [].
        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.
        showVertexLabels : bool , optional
            If set to True, the vertex labels are shown permenantely on screen. Otherwise, they are not. The default is False.
        showVertexLegend : bool , optional
            If set to True the vertex legend will be drawn. Otherwise, it will not be drawn. The default is False.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeWidth : float , optional
            The desired thickness of the output edges. The default is 1.
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. The default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. The default is None.
        edgeGroups : list , optional
            The list of edge groups against which to index the color of the edge. The default is [].
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. The default is True.
        showEdgeLabels : bool , optional
            If set to True, the edge labels are shown permenantely on screen. Otherwise, they are not. The default is False.
        showEdgeLegend : bool , optional
            If set to True the edge legend will be drawn. Otherwise, it will not be drawn. The default is False.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). The default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). If set to None, the code will attempt to discover the most suitable renderer. The default is None.
        width : int , optional
            The width in pixels of the figure. The default value is 950.
        height : int , optional
            The height in pixels of the figure. The default value is 950.
        xAxis : bool , optional
            If set to True the x axis is drawn. Otherwise it is not drawn. The default is False.
        yAxis : bool , optional
            If set to True the y axis is drawn. Otherwise it is not drawn. The default is False.
        zAxis : bool , optional
            If set to True the z axis is drawn. Otherwise it is not drawn. The default is False.
        axisSize : float , optional
            The size of the X, Y, Z, axes. The default is 1.
        backgroundColor : str , optional
            The desired color of the background. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "rgba(0,0,0,0)".
        marginLeft : int , optional
            The size in pixels of the left margin. The default value is 0.
        marginRight : int , optional
            The size in pixels of the right margin. The default value is 0.
        marginTop : int , optional
            The size in pixels of the top margin. The default value is 20.
        marginBottom : int , optional
            The size in pixels of the bottom margin. The default value is 0.
        camera : list , optional
            The desired location of the camera). The default is [-1.25, -1.25, 1.25].
        center : list , optional
            The desired center (camera target). The default is [0, 0, 0].
        up : list , optional
            The desired up vector. The default is [0, 0, 1].
        projection : str , optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. The default is "perspective"

        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        None

        """
        from topologicpy.Plotly import Plotly
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Show - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        data= Plotly.DataByGraph(graph, vertexColor=vertexColor, vertexSize=vertexSize, vertexLabelKey=vertexLabelKey, vertexGroupKey=vertexGroupKey, vertexGroups=vertexGroups, showVertices=showVertices, showVertexLabels=showVertexLabels, showVertexLegend=showVertexLegend, edgeColor=edgeColor, edgeWidth=edgeWidth, edgeLabelKey=edgeLabelKey, edgeGroupKey=edgeGroupKey, edgeGroups=edgeGroups, showEdges=showEdges, showEdgeLabels=showEdgeLabels, showEdgeLegend=showEdgeLegend, colorScale=colorScale)
        fig = Plotly.FigureByData(data, width=width, height=height, xAxis=xAxis, yAxis=yAxis, zAxis=zAxis, axisSize=axisSize, backgroundColor=backgroundColor,
                                  marginLeft=marginLeft, marginRight=marginRight, marginTop=marginTop, marginBottom=marginBottom, tolerance=tolerance)
        Plotly.Show(fig, renderer=renderer, camera=camera, center=center, up=up, projection=projection)

    @staticmethod
    def Size(graph):
        """
        Returns the graph size of the input graph. The graph size is its number of edges.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        int
            The number of edges in the input graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Size - Error: The input graph is not a valid graph. Returning None.")
            return None
        return len(Graph.Edges(graph))

    @staticmethod
    def TopologicalDistance(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Returns the topological distance between the input vertices. See https://en.wikipedia.org/wiki/Distance_(graph_theory).

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        int
            The topological distance between the input vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.TopologicalDistance - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Graph.TopologicalDistance - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            print("Graph.TopologicalDistance - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        return graph.TopologicalDistance(vertexA, vertexB, tolerance)
    
    @staticmethod
    def Topology(graph):
        """
        Returns the topology (cluster) of the input graph

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        topologic_core.Cluster
            The topology of the input graph.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Topology - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.Topology()
    
    @staticmethod
    def Tree(graph, vertex=None, tolerance=0.0001):
        """
        Creates a tree graph version of the input graph rooted at the input vertex.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex , optional
            The input root vertex. If not set, the first vertex in the graph is set as the root vertex. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The tree graph version of the input graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        
        def vertexInList(vertex, vertexList):
            if vertex and vertexList:
                if Topology.IsInstance(vertex, "Vertex") and isinstance(vertexList, list):
                    for i in range(len(vertexList)):
                        if vertexList[i]:
                            if Topology.IsInstance(vertexList[i], "Vertex"):
                                if Topology.IsSame(vertex, vertexList[i]):
                                    return True
            return False

        def getChildren(vertex, parent, graph, vertices):
            children = []
            adjVertices = []
            if vertex:
                adjVertices = Graph.AdjacentVertices(graph, vertex)
            if parent == None:
                return adjVertices
            else:
                for aVertex in adjVertices:
                    if (not vertexInList(aVertex, [parent])) and (not vertexInList(aVertex, vertices)):
                        children.append(aVertex)
            return children
        
        def buildTree(graph, dictionary, vertex, parent, tolerance=0.0001):
            vertices = dictionary['vertices']
            edges = dictionary['edges']
            if not vertexInList(vertex, vertices):
                vertices.append(vertex)
                if parent:
                    edge = Graph.Edge(graph, parent, vertex, tolerance)
                    ev = Edge.EndVertex(edge)
                    if Vertex.Distance(parent, ev) < tolerance:
                        edge = Edge.Reverse(edge)
                    edges.append(edge)
            if parent == None:
                parent = vertex
            children = getChildren(vertex, parent, graph, vertices)
            dictionary['vertices'] = vertices
            dictionary['edges'] = edges
            for child in children:
                dictionary = buildTree(graph, dictionary, child, vertex, tolerance)
            return dictionary
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Tree - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            vertex = Graph.Vertices(graph)[0]
        else:
            vertex = Graph.NearestVertex(graph, vertex)
        dictionary = {'vertices':[], 'edges':[]}
        dictionary = buildTree(graph, dictionary, vertex, None, tolerance)
        return Graph.ByVerticesEdges(dictionary['vertices'], dictionary['edges'])
    
    @staticmethod
    def VertexDegree(graph, vertex, edgeKey: str = None, tolerance: float = 0.0001):
        """
        Returns the degree of the input vertex. See https://en.wikipedia.org/wiki/Degree_(graph_theory).

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        edgeKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the edgeKey string will be aggregated to calculate
            the vertex degree. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead. This is used in weighted graphs.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        int
            The degree of the input vertex.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import numbers

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.VertexDegree - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Graph.VertexDegree - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        if not isinstance(edgeKey, str):
            edgeKey = ""
        edges = Graph.Edges(graph, [vertex], tolerance=tolerance)
        degree = 0
        for edge in edges:
            d = Topology.Dictionary(edge)
            value = Dictionary.ValueAtKey(d, edgeKey)
            if isinstance(value, numbers.Number):
                degree += value
            else:
                degree += 1
        return degree
    
    @staticmethod
    def Vertices(graph, vertexKey=None, reverse=False):
        """
        Returns the list of vertices in the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            If set, the returned list of vertices is sorted according to the dicitonary values stored under this key. The default is None.
        reverse : bool , optional
            If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. The default is False.
        
        Returns
        -------
        list
            The list of vertices in the input graph.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Vertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertices = []
        if graph:
            try:
                _ = graph.Vertices(vertices)
            except:
                vertices = []
        if not vertexKey == None:
            sorting_values = []
            for v in vertices:
                d = Topology.Dictionary(v)
                value = Dictionary.ValueAtKey(d, vertexKey)
                sorting_values.append(value)
            vertices = Helper.Sort(vertices, sorting_values)
            if reverse == True:
                vertices.reverse()
        return vertices

    @staticmethod
    def VisibilityGraph(face, viewpointsA=None, viewpointsB=None, tolerance=0.0001):
        """
        Creates a 2D visibility graph.

        Parameters
        ----------
        face : topologic_core.Face
            The input boundary. View edges will be clipped to this face. The holes in the face are used as the obstacles
        viewpointsA : list , optional
            The first input list of viewpoints (vertices). Visibility edges will connect these veritces to viewpointsB. If set to None, this parameters will be set to all vertices of the input face. The default is None.
        viewpointsB : list , optional
            The input list of viewpoints (vertices). Visibility edges will connect these vertices to viewpointsA. If set to None, this parameters will be set to all vertices of the input face. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The visibility graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Graph import Graph
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Graph.VisibilityGraph - Error: The input face parameter is not a valid face. Returning None")
            return None
        if viewpointsA == None:
            viewpointsA = Topology.Vertices(face)
        if viewpointsB == None:
            viewpointsB = Topology.Vertices(face)
        
        if not isinstance(viewpointsA, list):
            print("Graph.VisibilityGraph - Error: The input viewpointsA parameter is not a valid list. Returning None")
            return None
        if not isinstance(viewpointsB, list):
            print("Graph.VisibilityGraph - Error: The input viewpointsB parameter is not a valid list. Returning None")
            return None
        viewpointsA = [v for v in viewpointsA if Topology.IsInstance(v, "Vertex")]
        if len(viewpointsA) < 1:
            print("Graph.VisibilityGraph - Error: The input viewpointsA parameter does not contain any vertices. Returning None")
            return None
        viewpointsB = [v for v in viewpointsB if Topology.IsInstance(v, "Vertex")]
        if len(viewpointsB) < 1: #Nothing to look at, so return a graph made of viewpointsA
            return Graph.ByVerticesEdges(viewpointsA, [])
        
        i_boundaries = Face.InternalBoundaries(face)
        obstacles = []
        for i_boundary in i_boundaries:
            if Topology.IsInstance(i_boundary, "Wire"):
                obstacles.append(Face.ByWire(i_boundary))
        if len(obstacles) > 0:
            obstacle_cluster = Cluster.ByTopologies(obstacles)
        else:
            obstacle_cluster = None

        def intersects_obstacles(edge, obstacle_cluster, tolerance=0.0001):
            result = Topology.Difference(edge, obstacle_cluster)
            if result == None:
                return True
            if Topology.IsInstance(result, "Cluster"):
                return True
            if Topology.IsInstance(result, "Edge"):
                if abs(Edge.Length(edge) - Edge.Length(result)) > tolerance:
                    return True
            return False
            
        
        final_edges = []
        for i in tqdm(range(len(viewpointsA))):
            va = viewpointsA[i]
            for j in range(len(viewpointsB)):
                vb = viewpointsB[j]
                if Vertex.Distance(va, vb) > tolerance:
                    edge = Edge.ByVertices([va,vb])
                    if not intersects_obstacles(edge, obstacle_cluster):
                        final_edges.append(edge)
        if len(final_edges) > 0:
            final_vertices = Topology.Vertices(Cluster.ByTopologies(final_edges))
            g = Graph.ByVerticesEdges(final_vertices, final_edges)
            return g
        return None
