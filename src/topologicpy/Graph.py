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

import topologic_core as topologic
import random
import time
import os
import warnings

from collections import namedtuple
from multiprocessing import Process, Queue
from typing import Any

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
                    edge = Edge.ByVertices([source, destination], tolerance=self.tolerance, silent=True)
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

    @staticmethod
    def AccessibilityCentrality(graph, step: int = 2, normalize: bool = False, key: str = "accessibility_centrality", colorKey: str = "ac_color", colorScale: str = "viridis", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Computes the accessibility centrality of each vertex in the graph using random walks of fixed step length.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        step : int, optional
            The length of the random walk (number of steps). Default is 2.
        normalize : bool, optional
            If True, normalize the output to the range 0 to 1. Default is False.
        key : str, optional
            Dictionary key to store the accessibility centrality value. Default is "accessibility_centrality".
        colorKey : str, optional
            Dictionary key to store the color value. Default is "ac_color".
        colorScale : str, optional
            Name of the Plotly color scale to use. Default is "viridis".
        mantissa : int, optional
            Decimal precision. Default is 6.
        tolerance : float, optional
            The desired Tolerance. Not used here but included for API compatibility. Default is 0.0001.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of accessibility centrality values for each vertex in the graph.
        """
        import numpy as np
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.AccessibilityCentrality - Error: The input graph paramter is not a valid Topologic Graph. Returning None.")
            return None
        vertices = Graph.Vertices(graph)
        n = len(vertices)
        if n == 0:
            return []

        # Step 1: get transition matrix (row-normalized adjacency matrix)
        A = np.array(Graph.AdjacencyMatrix(graph), dtype=float)
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # prevent division by zero
        P = A / row_sums

        # Step 2: walk matrix of length `step`
        P_h = np.linalg.matrix_power(P, step)

        # Step 3: compute entropy-based accessibility for each vertex
        values = []
        for i in range(n):
            probs = P_h[i]
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            acc = np.exp(entropy)
            values.append(float(acc))

        # Optional normalization
        if normalize == True:
            if mantissa > 0: # We cannot round numbers from 0 to 1 with a mantissa = 0.
                values = [round(v, mantissa) for v in Helper.Normalize(values)]
            else:
                values = Helper.Normalize(values)
            min_value = 0
            max_value = 1
        else:
            min_value = min(values)
            max_value = max(values)
        
        # Assign Values and Colors to Dictionaries
        for i, value in enumerate(values):
            d = Topology.Dictionary(vertices[i])
            color = Color.AnyToHex(Color.ByValueInRange(value, minValue=min_value, maxValue=max_value, colorScale=colorScale))
            d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color])
            vertices[i] = Topology.SetDictionary(vertices[i], d)

        return values

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
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input edge added to it.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        def addIfUnique(graph_vertices, vertex, tolerance=0.0001):
            unique = True
            returnVertex = vertex
            for gv in graph_vertices:
                if (Vertex.Distance(vertex, gv) <= tolerance):
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
                            _ = Topology.SetDictionary(gv, d, silent=True)
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
        graph_edges = Graph.Edges(graph, graph_vertices, tolerance=tolerance)
        vertices = Topology.Vertices(edge)
        new_vertices = []
        for vertex in vertices:
            graph_vertices, nv = addIfUnique(graph_vertices, vertex, tolerance=tolerance)
            new_vertices.append(nv)
        new_edge = Edge.ByVertices([new_vertices[0], new_vertices[1]], tolerance=tolerance, silent=silent)
        if transferEdgeDictionaries == True:
            d = Topology.Dictionary(edge)
            keys = Dictionary.Keys(d)
            if isinstance(keys, list):
                if len(keys) > 0:
                    _ = Topology.SetDictionary(new_edge, d, silent=True)
        graph_edges.append(new_edge)
        new_graph = Graph.ByVerticesEdges(graph_vertices, graph_edges)
        return new_graph
    
    @staticmethod
    def AddEdgeByIndex(graph, index: list = None, dictionary = None, silent: bool = False):
        """
        Creates an edge in the input Graph by connecting the two vertices specified by their indices (e.g., [5, 6] connects the 4th and 6th vertices).

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        index : list or tuple
            The input list of vertex indices (e.g. [4, 6]).
        dictionary : topologic_core.Dictionary , optional
            The input edge dictionary.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input edge added to it.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.AddEdgeIndex - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if dictionary:
            if not Topology.IsInstance(dictionary, "Dictionary"):
                if not silent:
                    print("Graph.AddEdgeIndex - Error: The input dictionary parameter is not a valid dictionary. Returning None.")
                return None
        if not isinstance(index, list):
            if not silent:
                print("Graph.AddEdgeIndex - Error: The input index parameter is not a valid list. Returning None.")
            return None
        index = [x for x in index if isinstance(x, int)]
        if not len(index) == 2:
            if not silent:
                print("Graph.AddEdgeIndex - Error: The input index parameter should only contain two integer numbers. Returning None.")
            return None
        vertices = Graph.Vertices(graph)
        n = len(vertices)
        if index[0] < 0 or index[0] > n-1:
            if not silent:
                print("Graph.AddEdgeIndex - Error: The first integer in the input index parameter does not exist in the input graph. Returning None.")
            return None
        if index[1] < 0 or index[1] > n-1:
            if not silent:
                print("Graph.AddEdgeIndex - Error: The second integer in the input index parameter does not exist in the input graph. Returning None.")
            return None
        sv = vertices[index[0]]
        ev = vertices[index[1]]
        edge = Edge.ByVertices(sv, ev)
        if dictionary:
            edge = Topology.SetDictionary(edge, dictionary)
        graph = Graph.AddEdge(graph,edge)
        return graph

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
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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

        _ = graph.AddVertices([vertex], tolerance) # Hook to Core
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
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        _ = graph.AddVertices(vertices, tolerance) # Hook to Core
        return graph
    
    def AdjacencyDictionary(graph, vertexLabelKey: str = None, edgeKey: str = "Length", includeWeights: bool = False, mantissa: int = 6):
        """
        Returns the adjacency dictionary of the input Graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexLabelKey : str , optional
            The returned vertices are labelled according to the dictionary values stored under this key.
            If the vertexLabelKey does not exist, it will be created and the vertices are labelled numerically and stored in the vertex dictionary under this key. Default is None.
        edgeKey : str , optional
            If set, the edges' dictionaries will be searched for this key to set their weight. If the key is set to "length" (case insensitive), the length of the edge will be used as its weight. If set to None, a weight of 1 will be used. Default is "Length".
        includeWeights : bool , optional
            If set to True, edge weights are included. Otherwise, they are not. Default is False.        
        mantissa : int , optional
                The number of decimal places to round the result to. Default is 6.

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
        if vertexLabelKey == None:
            vertexLabelKey = "__label__"
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
        order = len(vertices)
        adjDict = {}
        for i in range(order):
            v = Graph.NearestVertex(graph, vertices[i])
            vertex_label = labels[i]
            adjVertices = Graph.AdjacentVertices(graph, v)
            temp_list = []
            for adjVertex in adjVertices:
                adjIndex = Vertex.Index(adjVertex, vertices)
                if not adjIndex == None:
                    adjLabel = labels[adjIndex]
                else:
                    adjLabel = None
                if includeWeights == True:
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
                    if not adjIndex == None:
                        temp_list.append((adjLabel, weight))
                else:
                    if not adjIndex == None:
                        temp_list.append(adjLabel)
            temp_list.sort()
            adjDict[vertex_label] = temp_list
        if vertexLabelKey == "__label__": # This is label we added, so remove it
            vertices = Graph.Vertices(graph)
            for v in vertices:
                d = Topology.Dictionary(v)
                d = Dictionary.RemoveKey(d, vertexLabelKey)
                v = Topology.SetDictionary(v, d)
        return adjDict
    
    @staticmethod
    def AdjacencyMatrix(graph, vertexKey=None, reverse=False, edgeKeyFwd=None, edgeKeyBwd=None, bidirKey=None, bidirectional=True, useEdgeIndex=False, useEdgeLength=False, mantissa: int = 6, tolerance=0.0001):
        """
        Returns the adjacency matrix of the input Graph. See https://en.wikipedia.org/wiki/Adjacency_matrix.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            If set, the returned list of vertices is sorted according to the dictionary values stored under this key. Default is None.
        reverse : bool , optional
            If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. Default is False.
        edgeKeyFwd : str , optional
            If set, the value at this key in the connecting edge from start vertex to end vertex (forward) will be used instead of the value 1. Default is None. useEdgeIndex and useEdgeLength override this setting.
        edgeKeyBwd : str , optional
            If set, the value at this key in the connecting edge from end vertex to start vertex (backward) will be used instead of the value 1. Default is None. useEdgeIndex and useEdgeLength override this setting.
        bidirKey : bool , optional
            If set to True or False, this key in the connecting edge will be used to determine is the edge is supposed to be bidirectional or not. If set to None, the input variable bidrectional will be used instead. Default is None
        bidirectional : bool , optional
            If set to True, the edges in the graph that do not have a bidirKey in their dictionaries will be treated as being bidirectional. Otherwise, the start vertex and end vertex of the connecting edge will determine the direction. Default is True.
        useEdgeIndex : bool , optional
            If set to True, the adjacency matrix values will the index of the edge in Graph.Edges(graph). Default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        useEdgeLength : bool , optional
            If set to True, the adjacency matrix values will the length of the edge in Graph.Edges(graph). Default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            elif "len" in edgeKeyFwd.lower() or "dis" in edgeKeyFwd.lower():
                valueFwd = Edge.Length(edge, mantissa=mantissa)
            else:
                valueFwd = Dictionary.ValueAtKey(Topology.Dictionary(edge), edgeKeyFwd)
                if valueFwd == None:
                    valueFwd = 1
            if edgeKeyBwd == None:
                valueBwd = 1
            elif "len" in edgeKeyBwd.lower() or "dis" in edgeKeyBwd.lower():
                valueBwd = Edge.Length(edge, mantissa=mantissa)
            else:
                valueBwd = Dictionary.ValueAtKey(Topology.Dictionary(edge), edgeKeyBwd)
                if valueBwd == None:
                    valueBwd = 1
            if useEdgeIndex:
                valueFwd = i+1
                valueBwd = i+1
            if useEdgeLength:
                valueFwd = Edge.Length(edge, mantissa=mantissa)
                valueBwd = Edge.Length(edge, mantissa=mantissa)
            if not svi == None and not evi == None:
                matrix[svi][evi] = valueFwd
                if bidir:
                    matrix[evi][svi] = valueBwd
        return matrix
    

    @staticmethod
    def AdjacencyMatrixCSVString(graph, vertexKey=None, reverse=False, edgeKeyFwd=None, edgeKeyBwd=None, bidirKey=None, bidirectional=True, useEdgeIndex=False, useEdgeLength=False, mantissa: int = 6, tolerance=0.0001):
        """
        Returns the adjacency matrix CSV string of the input Graph. See https://en.wikipedia.org/wiki/Adjacency_matrix.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            If set, the returned list of vertices is sorted according to the dictionary values stored under this key. Default is None.
        reverse : bool , optional
            If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. Default is False.
        edgeKeyFwd : str , optional
            If set, the value at this key in the connecting edge from start vertex to end vertex (forward) will be used instead of the value 1. Default is None. useEdgeIndex and useEdgeLength override this setting.
        edgeKeyBwd : str , optional
            If set, the value at this key in the connecting edge from end vertex to start vertex (backward) will be used instead of the value 1. Default is None. useEdgeIndex and useEdgeLength override this setting.
        bidirKey : bool , optional
            If set to True or False, this key in the connecting edge will be used to determine is the edge is supposed to be bidirectional or not. If set to None, the input variable bidrectional will be used instead. Default is None
        bidirectional : bool , optional
            If set to True, the edges in the graph that do not have a bidireKey in their dictionaries will be treated as being bidirectional. Otherwise, the start vertex and end vertex of the connecting edge will determine the direction. Default is True.
        useEdgeIndex : bool , optional
            If set to True, the adjacency matrix values will the index of the edge in Graph.Edges(graph). Default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        useEdgeLength : bool , optional
            If set to True, the adjacency matrix values will the length of the edge in Graph.Edges(graph). Default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        str
            A string in CSV format representing the adjacency matrix.
            Returns an empty string if conversion fails.
        """
        import io

        adj_matrix = Graph.AdjacencyMatrix(graph,
                                           vertexKey=vertexKey,
                                           reverse=reverse,
                                           edgeKeyFwd=edgeKeyFwd,
                                           edgeKeyBwd=edgeKeyBwd,
                                           bidirKey=bidirKey,
                                           bidirectional=bidirectional,
                                           useEdgeIndex=useEdgeIndex,
                                           useEdgeLength=useEdgeLength,
                                           mantissa=mantissa,
                                           tolerance=tolerance)

        try:
            # Convert the adjacency matrix (nested list) to a DataFrame
            adjacency_matrix_df = pd.DataFrame(adj_matrix)

            # Use a buffer to get the CSV output as a string
            csv_buffer = io.StringIO()
            adjacency_matrix_df.to_csv(csv_buffer, index=False, header=False)
            return csv_buffer.getvalue()
        except Exception as e:
            return ""

    @staticmethod
    def AdjacencyMatrixFigure(graph,
                            vertexKey: str = None,
                            showZero: bool = False,
                            zeroChar: str = "·",
                            zeroColor: str = 'rgba(0,0,0,0)',
                            valueColor: str = 'rgba(0,0,0,0.05)',
                            diagonalHighlight: bool = True,
                            diagonalColor: str = 'rgba(0,0,0,0)',
                            title: str = None,
                            cellSize: int = 24,
                            fontFamily: str = "Arial",
                            fontSize: int = 12,
                            fontColor: str = 'rgba(0,0,0,0)',
                            backgroundColor: str = 'rgba(0,0,0,0)',
                            headerColor: str = 'rgba(0,0,0,0)',
                            reverse=False,
                            edgeKeyFwd=None,
                            edgeKeyBwd=None,
                            bidirKey=None,
                            bidirectional=True,
                            useEdgeIndex=False,
                            useEdgeLength=False,
                            mantissa: int = 6,
                            tolerance=0.0001,
                            silent: bool = False):
        """
        Returns a Plotly table figure visualizing the adjacency matrix of a Graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            If set, the returned list of vertices is sorted according to the dictionary values stored under this key. Default is None.
        showZero : bool, optional
            If True, show zeros as "0"; if False, show a subtle glyph (zero_char) or blank. Default is False.
        zeroChar : str, optional
            Character to display for zero entries when show_zero is False. Default is "·".
        zeroColor : list or str , optional
            The desired color to display for zero-valued cells. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        valueColor : list or str , optional
            The desired color to display for non-zero-valued cells. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0.35)' (slight highlight).
        diagonalHighlight : bool, optional
            If True, lightly highlight diagonal cells. Default is True.
        diagonalColor : list or str , optional
            The desired diagonal highlight color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        title : str, optional
            Optional figure title.
        cellSize : int, optional
            Approximate pixel height of each table row. Default is 24.
        fontFamily : str, optional
            Font family for table text. Default is "Arial".
        fontSize : int, optional
            Font size for table text. Default is 12.
        fontColor : list or str , optional
            The desired font color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        backgroundColor : list or str , optional
            The desired background color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        headerColor : list or str , optional
            The desired header color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        reverse : bool , optional
            If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. Default is False.
        edgeKeyFwd : str , optional
            If set, the value at this key in the connecting edge from start vertex to end vertex (forward) will be used instead of the value 1. Default is None. useEdgeIndex and useEdgeLength override this setting.
        edgeKeyBwd : str , optional
            If set, the value at this key in the connecting edge from end vertex to start vertex (backward) will be used instead of the value 1. Default is None. useEdgeIndex and useEdgeLength override this setting.
        bidirKey : bool , optional
            If set to True or False, this key in the connecting edge will be used to determine is the edge is supposed to be bidirectional or not. If set to None, the input variable bidrectional will be used instead. Default is None
        bidirectional : bool , optional
            If set to True, the edges in the graph that do not have a bidireKey in their dictionaries will be treated as being bidirectional. Otherwise, the start vertex and end vertex of the connecting edge will determine the direction. Default is True.
        useEdgeIndex : bool , optional
            If set to True, the adjacency matrix values will the index of the edge in Graph.Edges(graph). Default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        useEdgeLength : bool , optional
            If set to True, the adjacency matrix values will the length of the edge in Graph.Edges(graph). Default is False. Both useEdgeIndex, useEdgeLength should not be True at the same time. If they are, useEdgeLength will be used.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, suppresses warning messages. Default is False.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            A Plotly table figure containing the adjacency matrix table.
        """

        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        import plotly.graph_objects as go

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Plotly.AdjacencyMatrixTable - Error: The input is not a valid Graph. Returning None.")
            return None

        # Build adjacency matrix
        adj = Graph.AdjacencyMatrix(graph,
                                    vertexKey=vertexKey,
                                    reverse=reverse,
                                    edgeKeyFwd=edgeKeyFwd,
                                    edgeKeyBwd=edgeKeyBwd,
                                    bidirKey=bidirKey,
                                    bidirectional=bidirectional,
                                    useEdgeIndex=useEdgeIndex,
                                    useEdgeLength=useEdgeLength,
                                    mantissa=mantissa,
                                    tolerance=tolerance)

        if adj is None or not isinstance(adj, list) or len(adj) == 0:
            if not silent:
                print("Plotly.AdjacencyMatrixTable - Warning: Empty adjacency matrix. Returning None.")
            return None

        n = len(adj)
        # Validate squareness
        if any((not isinstance(row, list) or len(row) != n) for row in adj):
            if not silent:
                print("Plotly.AdjacencyMatrixTable - Error: Adjacency matrix must be square. Returning None.")
            return None

        # Derive labels
        verts = Graph.Vertices(graph)
        labels = [Dictionary.ValueAtKey(Topology.Dictionary(v), vertexKey, str(i)) for i, v in enumerate(verts)]
        if len(labels) > 0 and not vertexKey == None:
            labels.sort()
            if reverse == True:
                labels.reverse()
        # Build display matrix (strings) while keeping a parallel style mask for diagonal
        display_matrix = []
        diag_mask = []
        for r in range(n):
            row_vals = []
            row_diag = []
            for c in range(n):
                v = adj[r][c]
                if v == 0:
                    row_vals.append("0" if showZero else (zeroChar if zeroChar else ""))
                else:
                    # Keep integers unpadded for clarity; cast others nicely
                    row_vals.append(str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v))
                row_diag.append(r == c)
            display_matrix.append(row_vals)
            diag_mask.append(row_diag)

        # Construct header and cells for Plotly Table
        # Header: blank corner + column labels
        header_values = [""] + labels

        # Body: first column is row labels, then matrix cells as strings
        # Plotly Table expects columns as lists; we need to transpose
        columns = []
        # Column 0: row labels
        columns.append(labels)
        # Subsequent columns: for each c, collect display_matrix[r][c]
        for c in range(n):
            columns.append([display_matrix[r][c] for r in range(n)])

        # Flatten cell fill_colors to highlight diagonal subtly.
        # Plotly Table allows per-cell fillcolor via a 2D list matching the table shape for 'cells'.
        # Our cells shape is n rows x (n+1) cols (including row label column).

        fill_colors = []
        # Column 0: row labels (no highlight)
        fill_colors.append([headerColor] * n)

        # Columns 1..n: highlight diagonal where row index r == (column_index-1)
        for c in range(1, n + 1):
            col_colors = []
            for r in range(n):
                if diagonalHighlight and r == (c - 1):
                    col_colors.append(diagonalColor)
                elif columns[c][r] == "0" or columns[c][r] == zeroChar:
                    col_colors.append(zeroColor)
                else:
                    col_colors.append(valueColor)
            fill_colors.append(col_colors)

        # Minimal line style
        line_color = "rgba(0,0,0,0.12)"
        # --- Sizing to prevent cropped text ---
        # Heuristic widths (pixels)
        max_label_len = max(len(str(x)) for x in labels) if labels else 1
        row_label_px = max(120, min(320, 8 * max_label_len))  # scale with label length
        cell_px = 36 if n <= 30 else (30 if n <= 50 else 24)   # narrower cells for very wide matrices

        # Adaptive cell font size for many columns
        fontSize = max(fontSize, 3)
        cell_font_size = fontSize if n <= 35 else (fontSize-1 if n <= 60 else fontSize-2)

        # Figure width: row label column + all matrix columns
        fig_width = row_label_px + n * cell_px
        fig_width = max(600, min(2400, fig_width))  # clamp to reasonable bounds

        # Increase row height a bit for readability
        cellSize = max(cellSize, 26)

        # Column widths in px (Plotly Table accepts pixel widths)
        columnwidth = [row_label_px] + [cell_px] * n
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_values,
                        align="center",
                        font=dict(family=fontFamily, size=cell_font_size, color=fontColor),
                        fill_color=headerColor,
                        line_color=line_color,
                        height=cellSize + 4  # a touch taller for the header
                    ),
                    cells=dict(
                        values=columns,
                        align=["right"] + ["center"] * n,
                        font=dict(family=fontFamily, size=cell_font_size, color=fontColor),
                        fill_color=fill_colors,
                        line_color=line_color,
                        height=cellSize
                    ),
                    columnorder=list(range(n + 1)),
                    columnwidth=columnwidth
                )
            ]
        )

        # Layout: generous margins, white background, optional title
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center") if title else None,
            paper_bgcolor="white",
            plot_bgcolor=backgroundColor,
            margin=dict(l=20, r=20, t=40 if title else 10, b=20)
        )

        return fig

    @staticmethod
    def AdjacencyList(graph, vertexKey=None, reverse=True, tolerance=0.0001):
        """
        Returns the adjacency list of the input Graph. See https://en.wikipedia.org/wiki/Adjacency_list.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            If set, the returned list of vertices is sorted according to the dictionary values stored under this key. Default is None.
        reverse : bool , optional
            If set to True, the vertices are sorted in reverse order (only if vertexKey is set). Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def AdjacentEdges(graph, edge, silent: bool = False):
        """
        Returns the list of edges connected to the input edge.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        edge : topologic_core.Edge
            the input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of adjacent edges.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.AdjacentEdges - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Graph.AdjacentEdges - Error: The input edge is not a valid edge. Returning None.")
            return None
        edges = []
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        edges.extend(Graph.Edges(graph, [sv]))
        edges.extend(Graph.Edges(graph, [ev]))
        print(edges)
        edges = [e for e in edges if not Topology.IsSame(e, edge)]
        # Complete the algorithm here
        return edges
    
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
            If set to True, error and warning messages are suppressed. Default is False.

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
        _ = graph.AdjacentVertices(vertex, vertices) # Hook to Core
        return list(vertices)
    
    @staticmethod
    def AdjacentVerticesByCompassDirection(graph, vertex, compassDirection: str = "Up", tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the list of vertices connected to the input vertex that are in the input compass direction.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            the input vertex.
        compassDirection : str , optional
            The compass direction. See Vector.CompassDirections(). Default is "Up".
        tolerance : float , optional
                The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of adjacent vertices that are in the compass direction.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.AdjacentVerticesByCompassDirection - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "vertex"):
            if not silent:
                print("Graph.AdjacentVerticesByCompassDirection - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not isinstance(compassDirection, str):
            if not silent:
                print("Graph.AdjacentVerticesByCompassDirection - Error: The input compassDirection parameter is not a valid string. Returning None.")
            return None
        
        directions = [v.lower() for v in Vector.CompassDirections()]

        if not compassDirection.lower() in directions:
            if not silent:
                print("Graph.AdjacentVerticesByCompassDirection - Error: The input compassDirection parameter is not a valid compass direction. Returning None.")
            return None

        adjacent_vertices = Graph.AdjacentVertices(graph, vertex)
        return_vertices = []
        for v in adjacent_vertices:
            e = Edge.ByVertices(vertex, v)
            vector = Edge.Direction(e)
            compass_direction = Vector.CompassDirection(vector, tolerance=tolerance)
            if compass_direction.lower() == compassDirection.lower():
                return_vertices.append(v)
        return return_vertices

    @staticmethod
    def AdjacentVerticesByVector(graph, vertex, vector: list = [0,0,1], tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the list of vertices connected to the input vertex that are in the input vector direction.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            the input vertex.
        vector : list , optional
            The vector direction. Default is [0,0,1].
        tolerance : float , optional
                The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of adjacent vertices that are in the vector direction.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.AdjacentVerticesByVector- Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "vertex"):
            if not silent:
                print("Graph.AdjacentVerticesByVector- Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not isinstance(vector, list):
            if not silent:
                print("Graph.AdjacentVerticesByVector- Error: The input vector parameter is not a valid vector. Returning None.")
            return None
        if len(vector) != 3:
            if not silent:
                print("Graph.AdjacentVerticesByVector- Error: The input vector parameter is not a valid vector. Returning None.")
            return None

        adjacent_vertices = Graph.AdjacentVertices(graph, vertex)
        return_vertices = []
        for v in adjacent_vertices:
            e = Edge.ByVertices(vertex, v)
            edge_vector = Edge.Direction(e)
            if Vector.CompassDirection(vector, tolerance=tolerance).lower() == Vector.CompassDirection(edge_vector, tolerance=tolerance).lower():
                return_vertices.append(v)
        return return_vertices
    
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
            The time limit in second. Default is 10 seconds.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        _ = graph.AllPaths(vertexA, vertexB, True, timeLimit, paths) # Hook to Core
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
            The number of decimal places to round the result to. Default is 6.

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
            If set to True, reverse relationships are created wherever possible. Otherwise, they are not. Default is False.
        includeAttributes : bool , optional
            If set to True, the attributes associated with vertices in the graph are written out. Otherwise, they are not. Default is False.
        includeLabel : bool , optional
            If set to True, a label is attached to each node. Otherwise, it is not. Default is False.
        includeGeometry : bool , optional
            If set to True, the geometry associated with vertices in the graph are written out. Otherwise, they are not. Default is False.
        siteLabel : str , optional
            The desired site label. Default is "Site_0001".
        siteDictionary : dict , optional
            The dictionary of site attributes to include in the output. Default is None.
        buildingLabel : str , optional
            The desired building label. Default is "Building_0001".
        buildingDictionary : dict , optional
            The dictionary of building attributes to include in the output. Default is None.
        storeyPrefix : str , optional
            The desired prefixed to use for each building storey. Default is "Storey".
        floorLevels : list , optional
            The list of floor levels. This should be a numeric list, sorted from lowest to highest.
            If not provided, floorLevels will be computed automatically based on the vertices' (zKey)) attribute. See below.
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. Default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. Default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "hasX".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "hasY".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "hasZ".
        geometryKey : str , optional
            The desired key name to use for geometry. Default is "brep".
        typeKey : str , optional
            The dictionary key to use to look up the type of the node. Default is "type".
        geometryKey : str , optional
            The dictionary key to use to look up the geometry of the node. Default is "brep".
        spaceType : str , optional
            The dictionary string value to use to look up vertices of type "space". Default is "space".
        wallType : str , optional
            The dictionary string value to use to look up vertices of type "wall". Default is "wall".
        slabType : str , optional
            The dictionary string value to use to look up vertices of type "slab". Default is "slab".
        doorType : str , optional
            The dictionary string value to use to look up vertices of type "door". Default is "door".
        windowType : str , optional
            The dictionary string value to use to look up vertices of type "window". Default is "window".
        contentType : str , optional
            The dictionary string value to use to look up vertices of type "content". Default is "contents".
        namespace : str , optional
            The desired namespace to use in the BOT graph. Default is "http://github.com/wassimj/topologicpy/resources".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

            
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
            If set to True, reverse relationships are created wherever possible. Otherwise, they are not. Default is False.
        includeAttributes : bool , optional
            If set to True, the attributes associated with vertices in the graph are written out. Otherwise, they are not. Default is False.
        includeLabel : bool , optional
            If set to True, a label is attached to each node. Otherwise, it is not. Default is False.
        includeGeometry : bool , optional
            If set to True, the geometry associated with vertices in the graph are written out. Otherwise, they are not. Default is False.
        siteLabel : str , optional
            The desired site label. Default is "Site_0001".
        siteDictionary : dict , optional
            The dictionary of site attributes to include in the output. Default is None.
        buildingLabel : str , optional
            The desired building label. Default is "Building_0001".
        buildingDictionary : dict , optional
            The dictionary of building attributes to include in the output. Default is None.
        storeyPrefix : str , optional
            The desired prefixed to use for each building storey. Default is "Storey".
        floorLevels : list , optional
            The list of floor levels. This should be a numeric list, sorted from lowest to highest.
            If not provided, floorLevels will be computed automatically based on the vertices' (zKey)) attribute. See below.
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. Default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. Default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "hasX".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "hasY".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "hasZ".
        geometryKey : str , optional
            The desired key name to use for geometry. Default is "brep".
        typeKey : str , optional
            The dictionary key to use to look up the type of the node. Default is "type".
        spaceType : str , optional
            The dictionary string value to use to look up vertices of type "space". Default is "space".
        wallType : str , optional
            The dictionary string value to use to look up vertices of type "wall". Default is "wall".
        slabType : str , optional
            The dictionary string value to use to look up vertices of type "slab". Default is "slab".
        doorType : str , optional
            The dictionary string value to use to look up vertices of type "door". Default is "door".
        windowType : str , optional
            The dictionary string value to use to look up vertices of type "window". Default is "window".
        contentType : str , optional
            The dictionary string value to use to look up vertices of type "content". Default is "contents".
        namespace : str , optional
            The desired namespace to use in the BOT graph. Default is "http://github.com/wassimj/topologicpy/resources".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        
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
    def BetweennessCentrality(
        graph,
        method: str = "vertex",
        weightKey: str = "length",
        normalize: bool = False,
        nxCompatible: bool = False,
        key: str = "betweenness_centrality",
        colorKey: str = "bc_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 0.001,
        silent: bool = False
    ):
        """
        Returns the betweenness centrality of the input graph. The order of the returned list is the same as the order of vertices/edges. See https://en.wikipedia.org/wiki/Betweenness_centrality.
        Optimized betweenness centrality (undirected) using Brandes:
        - Unweighted: O(VE) BFS per source
        - Weighted: Dijkstra-Brandes with binary heap
        - Vertex or Edge mode
        - Optional NetworkX-compatible normalization or 0..1 rescale
        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        method : str , optional
            The method of computing the betweenness centrality. The options are "vertex" or "edge". Default is "vertex".
        weightKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the weightKey string will be aggregated to calculate
            the shortest path. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead.
            This is used in weighted graphs. if weightKey is set to "Length" or "Distance", the length of the edge will be used as its weight.
        normalize : bool , optional
            If set to True, the values are normalized to be in the range 0 to 1. Otherwise they are not. Default is False.
        nxCompatible : bool , optional
            If set to True, and normalize input parameter is also set to True, the values are set to be identical to NetworkX values. Otherwise, they are normalized between 0 and 1. Default is False.
        key : str , optional
            The desired dictionary key under which to store the betweenness centrality score. Default is "betweenness_centrality".
        colorKey : str , optional
            The desired dictionary key under which to store the betweenness centrality color. Default is "betweenness_centrality".
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
            In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            The betweenness centrality of the input list of vertices within the input graph. The values are in the range 0 to 1.
        """
        from collections import deque
        import math

        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        from topologicpy.Helper import Helper
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        # We are inside Graph.* context; Graph.<...> methods available.

        # ---------- validate ----------
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.BetweennessCentrality - Error: The input is not a valid Graph. Returning None.")
            return None

        vertices = Graph.Vertices(graph)
        n = len(vertices)
        if n == 0:
            if not silent:
                print("Graph.BetweennessCentrality - Warning: Graph has no vertices. Returning [].")
            return []

        method_l = (method or "vertex").lower()
        compute_edges = "edge" in method_l

        # ---------- stable vertex indexing ----------
        def vkey(v, r=9):
            d = Topology.Dictionary(v)
            vid = Dictionary.ValueAtKey(d, "id")
            if vid is not None:
                return ("id", vid)
            return ("xyz", round(Vertex.X(v), r), round(Vertex.Y(v), r), round(Vertex.Z(v), r))

        idx_of = {vkey(v): i for i, v in enumerate(vertices)}

        # ---------- weight handling ----------
        dist_attr = None
        if isinstance(weightKey, str) and weightKey:
            wl = weightKey.lower()
            if ("len" in wl) or ("dis" in wl):
                weightKey = "length"
            dist_attr = weightKey

        def edge_weight(e):
            if dist_attr == "length":
                try:
                    return float(Edge.Length(e))
                except Exception:
                    return 1.0
            elif dist_attr:
                try:
                    d = Topology.Dictionary(e)
                    w = Dictionary.ValueAtKey(d, dist_attr)
                    return float(w) if (w is not None) else 1.0
                except Exception:
                    return 1.0
            else:
                return 1.0

        # ---------- build undirected adjacency (min weight on multi-edges) ----------
        edges = Graph.Edges(graph)
        # For per-edge outputs in input order:
        edge_end_idx = []  # [(iu, iv)] aligned with edges list (undirected as sorted pair)
        tmp_adj = [dict() for _ in range(n)]  # temporary: dedup by neighbor with min weight

        for e in edges:
            try:
                u = Edge.StartVertex(e)
                v = Edge.EndVertex(e)
            except Exception:
                continue
            iu = idx_of.get(vkey(u))
            iv = idx_of.get(vkey(v))
            if iu is None or iv is None or iu == iv:
                # still store mapping for return list to avoid index error
                pair = None
            else:
                w = edge_weight(e)
                # keep minimal weight for duplicates
                pu = tmp_adj[iu].get(iv)
                if (pu is None) or (w < pu):
                    tmp_adj[iu][iv] = w
                    tmp_adj[iv][iu] = w
                pair = (iu, iv) if iu < iv else (iv, iu)
            edge_end_idx.append(pair)

        # finalize adjacency as list-of-tuples for fast loops
        adj = [list(neigh.items()) for neigh in tmp_adj]  # adj[i] = [(j, w), ...]
        del tmp_adj

        # detect weightedness
        weighted = False
        for i in range(n):
            if any(abs(w - 1.0) > 1e-12 for _, w in adj[i]):
                weighted = True
                break

        # ---------- Brandes ----------
        CB_v = [0.0] * n
        CB_e = {}  # key: (min_i, max_j) -> score (only if compute_edges)

        if n > 1:
            if not weighted:
                # Unweighted BFS Brandes
                for s in range(n):
                    S = []
                    P = [[] for _ in range(n)]
                    sigma = [0.0] * n
                    sigma[s] = 1.0
                    dist = [-1] * n
                    dist[s] = 0
                    Q = deque([s])
                    pushQ, popQ = Q.append, Q.popleft

                    while Q:
                        v = popQ()
                        S.append(v)
                        dv = dist[v]
                        sv = sigma[v]
                        for w, _ in adj[v]:
                            if dist[w] < 0:
                                dist[w] = dv + 1
                                pushQ(w)
                            if dist[w] == dv + 1:
                                sigma[w] += sv
                                P[w].append(v)

                    delta = [0.0] * n
                    while S:
                        w = S.pop()
                        sw = sigma[w]
                        dw = 1.0 + delta[w]
                        for v in P[w]:
                            c = (sigma[v] / sw) * dw
                            delta[v] += c
                            if compute_edges:
                                a, b = (v, w) if v < w else (w, v)
                                CB_e[a, b] = CB_e.get((a, b), 0.0) + c
                        if w != s:
                            CB_v[w] += delta[w]
            else:
                # Weighted Dijkstra-Brandes
                import heapq
                EPS = 1e-12
                for s in range(n):
                    S = []
                    P = [[] for _ in range(n)]
                    sigma = [0.0] * n
                    sigma[s] = 1.0
                    dist = [math.inf] * n
                    dist[s] = 0.0
                    H = [(0.0, s)]
                    pushH, popH = heapq.heappush, heapq.heappop

                    while H:
                        dv, v = popH(H)
                        if dv > dist[v] + EPS:
                            continue
                        S.append(v)
                        sv = sigma[v]
                        for w, wgt in adj[v]:
                            nd = dv + wgt
                            dw = dist[w]
                            if nd + EPS < dw:
                                dist[w] = nd
                                sigma[w] = sv
                                P[w] = [v]
                                pushH(H, (nd, w))
                            elif abs(nd - dw) <= EPS:
                                sigma[w] += sv
                                P[w].append(v)

                    delta = [0.0] * n
                    while S:
                        w = S.pop()
                        sw = sigma[w]
                        if sw == 0.0:
                            continue
                        dw = 1.0 + delta[w]
                        for v in P[w]:
                            c = (sigma[v] / sw) * dw
                            delta[v] += c
                            if compute_edges:
                                a, b = (v, w) if v < w else (w, v)
                                CB_e[a, b] = CB_e.get((a, b), 0.0) + c
                        if w != s:
                            CB_v[w] += delta[w]

        # ---------- normalization ----------
        # NetworkX-compatible normalization (undirected):
        # vertices/edges factor = 2/((n-1)(n-2)) for n > 2 when normalized=True
        if nxCompatible:
            if normalize and n > 2:
                scale = 2.0 / ((n - 1) * (n - 2))
                CB_v = [v * scale for v in CB_v]
                if compute_edges:
                    for k in list(CB_e.keys()):
                        CB_e[k] *= scale
            # else: leave raw Brandes scores (normalized=False behavior)
            values_raw = CB_v if not compute_edges else [
                CB_e.get(tuple(sorted(pair)) if pair else None, 0.0) if pair else 0.0
                for pair in edge_end_idx
            ]
            values_for_return = values_raw
        else:
            # Rescale to [0,1] regardless of theoretical normalization
            values_raw = CB_v if not compute_edges else [
                CB_e.get(tuple(sorted(pair)) if pair else None, 0.0) if pair else 0.0
                for pair in edge_end_idx
            ]
            values_for_return = Helper.Normalize(values_raw)

        # rounding once
        if mantissa is not None and mantissa >= 0:
            values_for_return = [round(v, mantissa) for v in values_for_return]

        # ---------- color mapping ----------
        if values_for_return:
            min_v, max_v = min(values_for_return), max(values_for_return)
        else:
            min_v, max_v = 0.0, 1.0
        if abs(max_v - min_v) < tolerance:
            max_v = min_v + tolerance

        # annotate (vertices or edges) in input order
        if compute_edges:
            elems = edges
        else:
            elems = vertices
        for i, value in enumerate(values_for_return):
            d = Topology.Dictionary(elems[i])
            color_hex = Color.AnyToHex(
                Color.ByValueInRange(value, minValue=min_v, maxValue=max_v, colorScale=colorScale)
            )
            d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color_hex])
            elems[i] = Topology.SetDictionary(elems[i], d)

        return values_for_return

    # @staticmethod
    # def BetweennessCentrality_old(graph, method: str = "vertex", weightKey="length", normalize: bool = False, nxCompatible: bool = False, key: str = "betweenness_centrality", colorKey="bc_color", colorScale="viridis", mantissa: int = 6, tolerance: float = 0.001, silent: bool = False):
    #     """
    #     Returns the betweenness centrality of the input graph. The order of the returned list is the same as the order of vertices/edges. See https://en.wikipedia.org/wiki/Betweenness_centrality.

    #     Parameters
    #     ----------
    #     graph : topologic_core.Graph
    #         The input graph.
    #     method : str , optional
    #         The method of computing the betweenness centrality. The options are "vertex" or "edge". Default is "vertex".
    #     weightKey : str , optional
    #         If specified, the value in the connected edges' dictionary specified by the weightKey string will be aggregated to calculate
    #         the shortest path. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead.
    #         This is used in weighted graphs. if weightKey is set to "Length" or "Distance", the length of the edge will be used as its weight.
    #     normalize : bool , optional
    #         If set to True, the values are normalized to be in the range 0 to 1. Otherwise they are not. Default is False.
    #     nxCompatible : bool , optional
    #         If set to True, and normalize input parameter is also set to True, the values are set to be identical to NetworkX values. Otherwise, they are normalized between 0 and 1. Default is False.
    #     key : str , optional
    #         The desired dictionary key under which to store the betweenness centrality score. Default is "betweenness_centrality".
    #     colorKey : str , optional
    #         The desired dictionary key under which to store the betweenness centrality color. Default is "betweenness_centrality".
    #     colorScale : str , optional
    #         The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
    #         In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
    #     mantissa : int , optional
    #         The number of decimal places to round the result to. Default is 6.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.

    #     Returns
    #     -------
    #     list
    #         The betweenness centrality of the input list of vertices within the input graph. The values are in the range 0 to 1.

    #     """
    #     import warnings

    #     try:
    #         import networkx as nx
    #     except:
    #         print("Graph.BetwennessCentrality - Information: Installing required networkx library.")
    #         try:
    #             os.system("pip install networkx")
    #         except:
    #             os.system("pip install networkx --user")
    #         try:
    #             import networkx as nx
    #             print("Graph.BetwennessCentrality - Infromation: networkx library installed correctly.")
    #         except:
    #             warnings.warn("Graph.BetwennessCentrality - Error: Could not import networkx. Please try to install networkx manually. Returning None.")
    #             return None
        
    #     from topologicpy.Dictionary import Dictionary
    #     from topologicpy.Color import Color
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Helper import Helper

    #     if weightKey:
    #         if "len" in weightKey.lower() or "dis" in weightKey.lower():
    #             weightKey = "length"
    #     nx_graph = Graph.NetworkXGraph(graph)
    #     if "vert" in method.lower():
    #         elements = Graph.Vertices(graph)
    #         elements_dict = nx.betweenness_centrality(nx_graph, normalized=normalize, weight=weightKey)
    #         values = [round(value, mantissa) for value in list(elements_dict.values())]
    #     else:
    #         elements = Graph.Edges(graph)
    #         elements_dict = nx.edge_betweenness_centrality(nx_graph, normalized=normalize, weight=weightKey)
    #         values = [round(value, mantissa) for value in list(elements_dict.values())]
    #     if nxCompatible == False:
    #         if mantissa > 0: # We cannot have values in the range 0 to 1 with a mantissa < 1
    #             values = [round(v, mantissa) for v in Helper.Normalize(values)]
    #         else:
    #             values = Helper.Normalize(values)
    #         min_value = 0
    #         max_value = 1
    #     else:
    #         min_value = min(values)
    #         max_value = max(values)

    #     for i, value in enumerate(values):
    #         d = Topology.Dictionary(elements[i])
    #         color = Color.AnyToHex(Color.ByValueInRange(value, minValue=min_value, maxValue=max_value, colorScale=colorScale))
    #         d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color])
    #         elements[i] = Topology.SetDictionary(elements[i], d)

    #     return values

    @staticmethod
    def BetweennessPartition(graph, n=2, m=10, key="partition", tolerance=0.0001, silent=False):
        """
        Computes a partition of the input graph based on the edge betweenness method. See https://en.wikipedia.org/wiki/Graph_partition.

        Parameters
        ----------
        graph : topologicp.Graph
            The input topologic graph.
        n : int , optional
            The desired number of partitions when selecting the "Betweenness" method. This parameter is ignored for other methods. Default is 2.
        m : int , optional
            The desired maximum number of tries to partition the graph when selecting the "Betweenness" method. This parameter is ignored for other methods. Default is 10.
        key : str , optional
            The vertex and edge dictionary key under which to store the parition number. Default is "partition".
            Valid partition numbers start from 1. Cut edges receive a partition number of 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Graph
            The partitioned topologic graph.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary

        edge_scores = Graph.BetweennessCentrality(graph, method="edge")
        graph_edges = Graph.Edges(graph)

        graph_edges = Helper.Sort(graph_edges, edge_scores)
        graph_edges.reverse()
        cut_edges = []
        components = 1
        tries = 0
        while components < n and tries < m:
            components = len(Graph.ConnectedComponents(graph, key=key, tolerance=tolerance, silent=silent))
            if components == n:
                if not silent:
                    print("Graph.BetweennessPartition - Warning: The input graph is already partitioned into partitions that are equal in number to the desired number of partitions.")
                return graph
            elif components > n:
                if not silent:
                    print("Graph.BetweennessPartition - Warning: The input graph is already partitioned into partitions that are greater in number than the desired number of partitions.")
                return graph
            elif len(graph_edges) < 1:
                components = n
            else:
                edge = graph_edges[0]
                d = Topology.Dictionary(edge)
                d = Dictionary.SetValueAtKey(d, key, 0) # 0 indicates a cut edge
                edge = Topology.SetDictionary(edge, d)
                cut_edges.append(edge)
                graph = Graph.RemoveEdge(graph, edge, tolerance=tolerance, silent=silent)
                graph_edges = graph_edges[1:]
                components = len(Graph.ConnectedComponents(graph, key=key, tolerance=tolerance, silent=silent))
            tries += 1
            if tries == m:
                if not silent:
                    print("Graph.Partition - Warning: Reached the maximum number of tries.")
        return_vertices = Graph.Vertices(graph)
        return_edges = Graph.Edges(graph) + cut_edges
        graph = Graph.ByVerticesEdges(return_vertices, return_edges)
        return graph
    
    @staticmethod
    def Bridges(graph, key: str = "bridge", silent: bool = False):
        """
        Returns the list of bridge edges in the input graph. See: https://en.wikipedia.org/wiki/Bridge_(graph_theory)

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        key : str , optional
            The edge dictionary key under which to store the bridge status. 0 means the edge is NOT a bridge. 1 means that the edge IS a bridge. Default is "bridge".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of bridge edges in the input graph.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Graph import Graph

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.Bridges - Error: The input graph parameter is not a valid topologic graph. Returning None")
            return None

        graph_edges = Graph.Edges(graph)
        for edge in graph_edges:
            d = Topology.Dictionary(edge)
            d = Dictionary.SetValueAtKey(d, key, 0)
            edge = Topology.SetDictionary(edge, d)
        mesh_data = Graph.MeshData(graph)
        mesh_edges = mesh_data['edges']

        # Get adjacency dictionary
        adjacency_dict = Graph.AdjacencyDictionary(graph)
        if not adjacency_dict:
            if not silent:
                print("Graph.Bridges - Error: Failed to retrieve adjacency dictionary. Returning None")
            return None

        # Helper function to perform DFS and find bridges
        def dfs(vertex, parent, time, low, disc, visited, adjacency_dict, bridges, edge_map):
            visited[int(vertex)] = True
            disc[int(vertex)] = low[int(vertex)] = time[0]
            time[0] += 1
            for neighbor in adjacency_dict[vertex]:
                if not visited[int(neighbor)]:
                    dfs(neighbor, vertex, time, low, disc, visited, adjacency_dict, bridges, edge_map)
                    low[int(vertex)] = min(low[int(vertex)], low[int(neighbor)])
                    
                    # Check if edge is a bridge
                    if low[int(neighbor)] > disc[int(vertex)]:
                        bridges.add((vertex, neighbor))
                elif neighbor != parent:
                    low[int(vertex)] = min(low[int(vertex)], disc[int(neighbor)])

        # Prepare adjacency list and edge mapping
        vertices = list(adjacency_dict.keys())
        num_vertices = len(vertices)
        visited = [False] * num_vertices
        disc = [-1] * num_vertices
        low = [-1] * num_vertices
        time = [0]
        bridges = set()

        # Map edges to indices
        edge_map = {}
        index = 0
        for vertex, neighbors in adjacency_dict.items():
            for neighbor in neighbors:
                if (neighbor, vertex) not in edge_map:  # Avoid duplicating edges in undirected graphs
                    edge_map[(vertex, neighbor)] = index
                    index += 1

        # Run DFS from all unvisited vertices
        for i, vertex in enumerate(vertices):
            if not visited[i]:
                dfs(vertex, -1, time, low, disc, visited, adjacency_dict, bridges, edge_map)

        # Mark bridges in the edges' dictionaries
        bridge_edges = []
        for edge in bridges:
            i, j = edge
            i = int(i)
            j = int(j)
            try:
                edge_index = mesh_edges.index([i,j])
            except:
                edge_index = mesh_edges.index([j,i])
            bridge_edges.append(graph_edges[edge_index])
        for edge in bridge_edges:
            d = Topology.Dictionary(edge)
            d = Dictionary.SetValueAtKey(d, key, 1)
            edge = Topology.SetDictionary(edge, d)

        return bridge_edges

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
            If set to True, no warnings or error messages are displayed. Default is False.
        
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
    def ByAdjacencyMatrix(adjacencyMatrix, dictionaries = None, edgeKeyFwd="weightFwd", edgeKeyBwd="weightBwd", xMin=-0.5, yMin=-0.5, zMin=-0.5, xMax=0.5, yMax=0.5, zMax=0.5, silent=False):
        """
        Returns graphs according to the input folder path. This method assumes the CSV files follow DGL's schema.

        Parameters
        ----------
        adjacencyMatrix : list
            The adjacency matrix expressed as a nested list of 0s and a number not 0 which represents the edge weight.
        dictionaries : list , optional
            A list of dictionaries to assign to the vertices of the graph. This list should be in
            the same order and of the same length as the rows in the adjacency matrix.
        edgeKeyFwd : str , optional
            The dictionary key under which to store the edge weight value for forward edge. Default is "weight".
        edgeKeyBwd : str , optional
            The dictionary key under which to store the edge weight value for backward edge. Default is "weight".
        xMin : float , optional
            The desired minimum value to assign for a vertex's X coordinate. Default is -0.5.
        yMin : float , optional
            The desired minimum value to assign for a vertex's Y coordinate. Default is -0.5.
        zMin : float , optional
            The desired minimum value to assign for a vertex's Z coordinate. Default is -0.5.
        xMax : float , optional
            The desired maximum value to assign for a vertex's X coordinate. Default is 0.5.
        yMax : float , optional
            The desired maximum value to assign for a vertex's Y coordinate. Default is 0.5.
        zMax : float , optional
            The desired maximum value to assign for a vertex's Z coordinate. Default is 0.5.
        silent : bool , optional
            If set to True, no warnings or error messages are displayed. Default is False.
        
        Returns
        -------
        topologic_core.Graph
            The created graph.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
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
            if isinstance(dictionaries, list):
                v = Topology.SetDictionary(v, dictionaries[i])
            vertices.append(v)

        # Create the graph using vertices and edges
        if len(vertices) == 0:
            print("Graph.ByAdjacencyMatrix - Error: The graph does not contain any vertices. Returning None.")
            return None
        
        # Add edges based on the adjacency matrix
        edges = []
        visited = []
        for i in range(len(adjacencyMatrix)):
            for j in range(len(adjacencyMatrix)):
                if not i == j:
                    if (adjacencyMatrix[i][j] != 0 or adjacencyMatrix[j][i] != 0) and not (i,j) in visited:
                        edge = Edge.ByVertices([vertices[i], vertices[j]]) # Create only one edge
                        d = Dictionary.ByKeysValues([edgeKeyFwd, edgeKeyBwd], [adjacencyMatrix[i][j], adjacencyMatrix[j][i]])
                        edge = Topology.SetDictionary(edge, d)
                        edges.append(edge)
                        visited.append((i,j))
                        visited.append((j,i))
        
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
                  nodeFeaturesKeys=[], tolerance=0.0001, silent=False):
        """
        Returns graphs according to the input folder path. This method assumes the CSV files follow DGL's schema.

        Parameters
        ----------
        path : str
            The path to the folder containing the .yaml and .csv files for graphs, edges, and nodes.
        graphIDHeader : str , optional
            The column header string used to specify the graph id. Default is "graph_id".
        graphLabelHeader : str , optional
            The column header string used to specify the graph label. Default is "label".
        graphFeaturesHeader : str , optional
            The column header string used to specify the graph features. Default is "feat".
        edgeSRCHeader : str , optional
            The column header string used to specify the source vertex id of edges. Default is "src_id".
        edgeDSTHeader : str , optional
            The column header string used to specify the destination vertex id of edges. Default is "dst_id".
        edgeLabelHeader : str , optional
            The column header string used to specify the label of edges. Default is "label".
        edgeTrainMaskHeader : str , optional
            The column header string used to specify the train mask of edges. Default is "train_mask".
        edgeValidateMaskHeader : str , optional
            The column header string used to specify the validate mask of edges. Default is "val_mask".
        edgeTestMaskHeader : str , optional
            The column header string used to specify the test mask of edges. Default is "test_mask".
        edgeFeaturesHeader : str , optional
            The column header string used to specify the features of edges. Default is "feat".
        edgeFeaturesKeys : list , optional
            The list of dictionary keys to use to index the edge features. The length of this list must match the length of edge features. Default is [].
        nodeIDHeader : str , optional
            The column header string used to specify the id of nodes. Default is "node_id".
        nodeLabelHeader : str , optional
            The column header string used to specify the label of nodes. Default is "label".
        nodeTrainMaskHeader : str , optional
            The column header string used to specify the train mask of nodes. Default is "train_mask".
        nodeValidateMaskHeader : str , optional
            The column header string used to specify the validate mask of nodes. Default is "val_mask".
        nodeTestMaskHeader : str , optional
            The column header string used to specify the test mask of nodes. Default is "test_mask".
        nodeFeaturesHeader : str , optional
            The column header string used to specify the features of nodes. Default is "feat".
        nodeXHeader : str , optional
            The column header string used to specify the X coordinate of nodes. Default is "X".
        nodeYHeader : str , optional
            The column header string used to specify the Y coordinate of nodes. Default is "Y".
        nodeZHeader : str , optional
            The column header string used to specify the Z coordinate of nodes. Default is "Z".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
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
            if not silent:
                print("Graph.ByCSVPath - Error: the input path parameter does not exists. Returning None.")
            return None
        if not isdir(path):
            if not silent:
                print("Graph.ByCSVPath - Error: the input path parameter is not a folder. Returning None.")
            return None
        
        yaml_files = find_yaml_files(path)
        if len(yaml_files) < 1:
            if not silent:
                print("Graph.ByCSVPath - Error: the input path parameter does not contain any valid YAML files. Returning None.")
            return None
        yaml_file = yaml_files[0]
        yaml_file_path = os.path.join(path, yaml_file)

        graphs_path, edges_path, nodes_path = read_yaml(yaml_file_path)
        if not graphs_path == None:
            graphs_path = os.path.join(path, graphs_path)
        if graphs_path == None:
            if not silent:
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
            if not silent:
                print("Graph.ByCSVPath - Error: an edges.csv file does not exist inside the folder specified by the input path parameter. Returning None.")
            return None
        edges_path = os.path.join(path, edges_path)
        edges_df = pd.read_csv(edges_path)
        grouped_edges = edges_df.groupby(graphIDHeader)
        if not nodes_path == None:
            nodes_path = os.path.join(path, nodes_path)
        if not exists(nodes_path):
            if not silent:
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
            edge_keys = [edgeSRCHeader, edgeDSTHeader, edgeLabelHeader, "mask", edgeFeaturesHeader]
        else:
            edge_keys = [edgeSRCHeader, edgeDSTHeader, edgeLabelHeader, "mask"]+edgeFeaturesKeys
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
                        if not silent:
                            print("Graph.ByCSVPath - Warning: Failed to create and add a dictionary to the created vertex.")
                    vertices.append(v)
                else:
                    if not silent:
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
                    values = [src_id, dst_id, label, mask, features]
                else:
                    featureList = features.split(",")
                    featureList = [float(s) for s in featureList]
                    values = [src_id, dst_id, label, mask]+featureList
                if not (src_id == dst_id) and not [src_id, dst_id] in es and not [dst_id, src_id] in es:
                    es.append([src_id, dst_id])
                    try:
                        edge = Edge.ByVertices([vertices[src_id], vertices[dst_id]], tolerance=tolerance)
                    except:
                        if not silent:
                            print("Graph.ByCSVPath - Warning: Failed to create and add a edge to the list of edges.")
                        edge = None
                    if Topology.IsInstance(edge, "Edge"):
                        d = Dictionary.ByKeysValues(edge_keys, values)
                        if Topology.IsInstance(d, "Dictionary"):
                            edge = Topology.SetDictionary(edge, d)
                        else:
                            if not silent:
                                print("Graph.ByCSVPath - Warning: Failed to create and add a dictionary to the created edge.")
                        edges.append(edge)
                    else:
                        if not silent:
                            print("Graph.ByCSVPath - Warning: Failed to create and add an edge to the list of edges.")
                else:
                    duplicate_edges += 1
            if duplicate_edges > 0:
                if not silent:
                    print("Graph.ByCSVPath - Warning: Found", duplicate_edges, "duplicate edges in graph id:", graph_id)
            edges_ds.append(edges)
        
        # Build the graphs
        graphs = []
        for i, vertices, in enumerate(vertices_ds):
            edges = edges_ds[i]
            g = Graph.ByVerticesEdges(vertices, edges)
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
                    if not silent:
                        print("Graph.ByCSVPath - Error: The length of the keys and values lists do not match. Returning None.")
                    return None
                d = Dictionary.ByKeysValues(graph_keys, values)
                if Topology.IsInstance(d, "Dictionary"):
                    g = Graph.SetDictionary(g, d)
                else:
                    if not silent:
                        print("Graph.ByCSVPath - Warning: Failed to create and add a dictionary to the created graph.")
                graphs.append(g)
            else:
                if not silent:
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
            The desired key for storing the node label. Default is "label".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
            The desired key for storing the node label. Default is "label".
        tolerance : str , optional
            The desired tolerance. Default is 0.0001.
        
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
            The desired key for storing the node label. Default is "label".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def ByDictionaries(graphDictionary, vertexDictionaries, edgeDictionaries, vertexKey: str = None, edgeKey: str = None, silent: bool = False, tolerance: float = 0.0001):
        """
        Creates a graph from input python dictionaries.

        Rules:
        All vertex dictionaries must contain at least the vertexKey.
        All edge dictionaries must contain at least the edgeKey.
        The edgeKey must be a tuple or list of two str values.
        x,y,z coordinates are optional. However, if a vertex dictionary contains x,y,z coordinates then all vertex dictionaries must contain x,y,z coordinates.
        If vertex dictionaries contain x,y,z coordinates they must not overlap and be separated by a distance greater than tolerance.
        Keys and values are case sensitive.
        x,y,z keys, if present  must be lowercase.

        Example:
        graphDictionary = {"name": "Small Apartment", "location": "123 Main Street"}
        vertexDictionaries = [
            {"name":"Entry", "type":"Circulation", "x":1, "y":4, "z":0, "area":5},
            {"name":"Living Room", "type":"Living Room", "x":3,  "y":4 , "z":0, "area":24},
            {"name":"Dining Room", "type":"Dining Room", "x":5,  "y":2, "z":0,  "area":18},
            {"name":"Kitchen", "type":"Kitchen", "x":1, "y":2, "z":0,  "area":15},
            {"name":"Bathroom", "type":"Bathroom", "x":3, "y":6, "z":0, "area":9},
            {"name":"Bedroom", "type":"Bedroom", "x":5, "y":4, "z":0, "area":16}
        ]
        edgeDictionaries = [
            {"connects": ("Entry","Living Room"), "relationship": "adjacent_to"},
            {"connects": ("Living Room","Kitchen"), "relationship": "adjacent_to"},
            {"connects": ("Dining Room","Kitchen"), "relationship": "adjacent_to"},
            {"connects": ("Living Room","Dining Room"), "relationship": "adjacent_to"},
            {"connects": ("Living Room","Bedroom"), "relationship": "adjacent_to"},
            {"connects": ("Living Room","Bathroom"), "relationship": "adjacent_to"}
        ]
        vertexKey = "name"
        edgeKey = "connects"

        Parameters
        ----------
        graphDictionary : dict
            The python dictionary to associate with the resulting graph
        vertexDictionaries : list
            The input list of vertex dictionaries. These must contain the vertexKey. X,Y,Z coordinates are optional.
        edgeDictionaries : list
            The input list of edge dictionaries. These must have the edgeKey to specify the two vertices they connect (by using the vertexKey)
        vertexKey: str
            The vertex key used to identify which vertices and edge connects.
        edgeKey: str
            The edge key under which the pair of vertex keys are listed as a tuple or list.
        tolerance: float , optional
            The desired tolerance. The default is 0.0001
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The resulting graph

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def _set_dict(obj, kv: dict):
            keys = list(kv.keys())
            vals = list(kv.values())
            d = Dictionary.ByKeysValues(keys, vals)
            Topology.SetDictionary(obj, d)
            return obj

        def _vertex(vertexDictionary, vertices, vertexKey, tolerance=0.0001, silent=False):
            x = vertexDictionary.get("x", 0)
            y = vertexDictionary.get("y", 0)
            z = vertexDictionary.get("z", 0)
            v = Vertex.ByCoordinates(x, y, z)
            v = _set_dict(v, vertexDictionary)
            if "x" in vertexDictionary.keys(): # Check for overlap only if coords are given.
                if len(vertices) > 0:
                    nv = Vertex.NearestVertex(v, Cluster.ByTopologies(vertices))
                    d = Topology.Dictionary(nv)
                    nv_name = Dictionary.ValueAtKey(d, vertexKey, "Unknown")
                    if Vertex.Distance(v, nv) < tolerance:
                        if not silent:
                            v_name = vertexDictionary[vertexKey]
                            print(f"Graph.ByDictionaries - Warning: Vertices {v_name} and {nv_name} overlap.")
            return v


        if graphDictionary:
            if not isinstance(graphDictionary, dict):
                if not silent:
                    print("Graph.ByDictionaries - Error: The input graphDictionary parameter is not a valid python dictionary. Returning None.")
                return None
        
        if not isinstance(vertexDictionaries, list):
            if not silent:    
                print("Graph.ByDictionaries - Error: The input vertexDictionaries parameter is not a valid list. Returning None.")
            return None
        
        if not isinstance(edgeDictionaries, list):
            if not silent:    
                print("Graph.ByDictionaries - Error: The input edgeDictionaries parameter is not a valid list. Returning None.")
            return None
        
        name_to_vertex = {}
        vertices = []
        for vd in vertexDictionaries:
            v = _vertex(vd, vertices, vertexKey=vertexKey, tolerance=tolerance, silent=silent)
            if v:
                vertices.append(v)

        # If coordinates are not present, make sure you separate the vertices to allow edges to be created.
        if "x" not in vertexDictionaries[0].keys():
            vertices = Vertex.Separate(vertices, minDistance=max(1, tolerance))

        for i, v in enumerate(vertices):
            vd = vertexDictionaries[i]
            name_to_vertex[vd[vertexKey]] = v

        # Create adjacency edges (undirected: one edge per pair)
        edges = []
        for d in edgeDictionaries:
            a, b = d[edgeKey]
            va = name_to_vertex.get(a, None)
            vb = name_to_vertex.get(b, None)
            if not va and not vb:
                if not silent:
                    print(f"Graph.ByDictionaries - Warning: vertices '{a}' and '{b}' are missing. Could not create an edge between them.")
                continue
            if not va:
                if not silent:
                    print(f"Graph.ByDictionaries - Warning: vertex '{a}' is missing. Could not create an edge between '{a}' and '{b}'.")
                continue
            if not vb:
                if not silent:
                    print(f"Graph.ByDictionaries - Warning: vertex '{b}' is missing. Could not create an edge between '{a}' and '{b}'.")
                continue
            e = Edge.ByStartVertexEndVertex(va, vb, silent=True)
            if not e:
                if not silent:
                    print(f"Graph.ByDictionaries - Warning: Could not create an edge between '{a}' and '{b}'. Check if the distance betwen '{a}' and '{b}' is kess than the input tolerance.")
                continue
            edges.append(e)
        # Build graph
        g = Graph.ByVerticesEdges(vertices, edges)

        # Attach graph-level metadata
        if graphDictionary:
            _set_dict(g, graphDictionary)
        return g

    @staticmethod
    def ByIFCFile(file,
                  includeTypes: list = [],
                  excludeTypes: list = [],
                  includeRels: list = [],
                  excludeRels: list = [],
                  transferDictionaries: bool = False,
                  useInternalVertex: bool = False,
                  storeBREP: bool = False,
                  removeCoplanarFaces: bool = False,
                  xMin: float = -0.5, yMin: float = -0.5, zMin: float = -0.5,
                  xMax: float = 0.5, yMax: float = 0.5, zMax: float = 0.5,
                  epsilon: float = 0.0001,
                  tolerance: float = 0.0001,
                  silent: bool = False):
        
        """
        Create a Graph from an IFC file. This code is partially based on code from Bruno Postle.

        Parameters
        ----------
        file : file
            The input IFC file
        includeTypes : list , optional
            A list of IFC object types to include in the graph. Default is [] which means all object types are included.
        excludeTypes : list , optional
            A list of IFC object types to exclude from the graph. Default is [] which mean no object type is excluded.
        includeRels : list , optional
            A list of IFC relationship types to include in the graph. Default is [] which means all relationship types are included.
        excludeRels : list , optional
            A list of IFC relationship types to exclude from the graph. Default is [] which mean no relationship type is excluded.
        transferDictionaries : bool , optional
            NOT USED. If set to True, the dictionaries from the IFC file will be transferred to the topology. Otherwise, they won't. Default is False.
        useInternalVertex : bool , optional
            If set to True, use an internal vertex to represent the subtopology. Otherwise, use its centroid. Default is False.
        storeBREP : bool , optional
            If set to True, store the BRep of the subtopology in its representative vertex. Default is False.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are removed. Otherwise they are not. Default is False.
        xMin : float, optional
            The desired minimum value to assign for a vertex's X coordinate. Default is -0.5.
        yMin : float, optional
            The desired minimum value to assign for a vertex's Y coordinate. Default is -0.5.
        zMin : float, optional
            The desired minimum value to assign for a vertex's Z coordinate. Default is -0.5.
        xMax : float, optional
            The desired maximum value to assign for a vertex's X coordinate. Default is 0.5.
        yMax : float, optional
            The desired maximum value to assign for a vertex's Y coordinate. Default is 0.5.
        zMax : float, optional
            The desired maximum value to assign for a vertex's Z coordinate. Default is 0.5.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.Graph
            The created graph.
        
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        
        def vertex_at_key_value(vertices, key, value):
            for v in vertices:
                d = Topology.Dictionary(v)
                d_value = Dictionary.ValueAtKey(d, key)
                if value == d_value:
                    return v
            return None
        
        def get_vertices(includeTypes=[], excludeTypes=[], removeCoplanarFaces=False, storeBREP=False, useInternalVertex=useInternalVertex, epsilon=0.0001, tolerance=0.0001):
            # Get the topologies
            topologies = Topology.ByIFCFile(file,
                                            includeTypes=includeTypes,
                                            excludeTypes=excludeTypes,
                                        transferDictionaries=True,
                                        removeCoplanarFaces=removeCoplanarFaces,
                                        epsilon=epsilon,
                                        tolerance=tolerance)
            vertices = []
            for topology in topologies:
                if Topology.IsInstance(topology, "Topology"):
                    if useInternalVertex == True:
                        v = Topology.InternalVertex(topology)
                    else:
                        v = Topology.Centroid(topology)
                    d = Topology.Dictionary(topology)
                    if storeBREP:
                        d = Dictionary.SetValueAtKey(d, "BREP", Topology.BREPString(topology))
                    if Topology.IsInstance(v, "vertex"):
                        v = Topology.SetDictionary(v, Topology.Dictionary(topology))
                        vertices.append(v)
                    else:
                        if not silent:
                            ifc_id = Dictionary.ValueAtKey(Topology.Dictionary(topology), "IFC_global_id", 0)
                            print(f"Graph.ByIFCFile - Warning: Could not create a vertex for entity {ifc_id}. Skipping")
            return vertices
        
        # Get the relationships
        def get_relationships(ifc_file, includeRels=[], excludeRels=[]):
            include_set = set(s.lower() for s in includeRels)
            exclude_set = set(s.lower() for s in excludeRels)

            relationships = [
                rel for rel in ifc_file.by_type("IfcRelationship")
                if (rel.is_a().lower() not in exclude_set) and
                (not include_set or rel.is_a().lower() in include_set)
            ]

            return relationships
        def get_edges(ifc_relationships, vertices):
            tuples = []
            edges = []

            for ifc_rel in ifc_relationships:
                source = None
                destinations = []
                if ifc_rel.is_a("IfcRelConnectsPorts"):
                    source = ifc_rel.RelatingPort
                    destinations = ifc_rel.RelatedPorts
                elif ifc_rel.is_a("IfcRelConnectsPortToElement"):
                    source = ifc_rel.RelatingPort
                    destinations = [ifc_rel.RelatedElement]
                elif ifc_rel.is_a("IfcRelAggregates"):
                    source = ifc_rel.RelatingObject
                    destinations = ifc_rel.RelatedObjects
                elif ifc_rel.is_a("IfcRelNests"):
                    source = ifc_rel.RelatingObject
                    destinations = ifc_rel.RelatedObjects
                elif ifc_rel.is_a("IfcRelAssignsToGroup"):
                    source = ifc_rel.RelatingGroup
                    destinations = ifc_rel.RelatedObjects
                elif ifc_rel.is_a("IfcRelConnectsPathElements"):
                    source = ifc_rel.RelatingElement
                    destinations = [ifc_rel.RelatedElement]
                elif ifc_rel.is_a("IfcRelConnectsStructuralMember"):
                    source = ifc_rel.RelatingStructuralMember
                    destinations = [ifc_rel.RelatedStructuralConnection]
                elif ifc_rel.is_a("IfcRelContainedInSpatialStructure"):
                    source = ifc_rel.RelatingStructure
                    destinations = ifc_rel.RelatedElements
                elif ifc_rel.is_a("IfcRelFillsElement"):
                    source = ifc_rel.RelatingOpeningElement
                    destinations = [ifc_rel.RelatedBuildingElement]
                elif ifc_rel.is_a("IfcRelSpaceBoundary"):
                    source = ifc_rel.RelatingSpace
                    destinations = [ifc_rel.RelatedBuildingElement]
                elif ifc_rel.is_a("IfcRelVoidsElement"):
                    source = ifc_rel.RelatingBuildingElement
                    destinations = [ifc_rel.RelatedOpeningElement]
                elif ifc_rel.is_a("IfcRelDefinesByProperties") or ifc_rel.is_a("IfcRelAssociatesMaterial") or ifc_rel.is_a("IfcRelDefinesByType"):
                    source = None
                    destinations = None
                else:
                    print("Graph.ByIFCFile - Warning: The relationship", ifc_rel, "is not supported. Skipping.")
                if source:
                    sv = vertex_at_key_value(vertices, key="IFC_global_id", value=getattr(source, 'GlobalId', 0))
                    if sv:
                        si = Vertex.Index(sv, vertices, tolerance=tolerance)
                        if not si == None:
                            for destination in destinations:
                                if destination == None:
                                    continue
                                ev = vertex_at_key_value(vertices, key="IFC_global_id", value=getattr(destination, 'GlobalId', 0))
                                if ev:
                                    ei = Vertex.Index(ev, vertices, tolerance=tolerance)
                                    if not ei == None:
                                        if not si == ei:
                                            if not [si,ei] in tuples:
                                                tuples.append([si,ei])
                                                tuples.append([ei,si])
                                                e = Edge.ByVertices([sv,ev])
                                                if Topology.IsInstance(e, "edge"):
                                                    d = Dictionary.ByKeysValues(["IFC_global_id", "IFC_name", "IFC_type"], [ifc_rel.id(), ifc_rel.Name, ifc_rel.is_a()])
                                                    e = Topology.SetDictionary(e, d)
                                                    edges.append(e)
                                                else:
                                                    if not silent:
                                                        if not silent:
                                                            print(f"Graph.ByIFCFile - Warning: Could not create an edge for relationship {ifc_rel.id()}. Skipping")

            return edges
        
        vertices = get_vertices(includeTypes=includeTypes,
                                excludeTypes=excludeTypes,
                                removeCoplanarFaces=removeCoplanarFaces,
                                storeBREP=storeBREP,
                                useInternalVertex=useInternalVertex,
                                epsilon=epsilon,
                                tolerance=0.0001)
        relationships = get_relationships(file, includeRels=includeRels, excludeRels=excludeRels)
        edges = get_edges(relationships, vertices)
        return Graph.ByVerticesEdges(vertices, edges)
        
    # @staticmethod
    # def ByIFCFile_old(file,
    #               includeTypes: list = [],
    #               excludeTypes: list = [],
    #               includeRels: list = [],
    #               excludeRels: list = [],
    #               transferDictionaries: bool = False,
    #               useInternalVertex: bool = False,
    #               storeBREP: bool = False,
    #               removeCoplanarFaces: bool = False,
    #               xMin: float = -0.5, yMin: float = -0.5, zMin: float = -0.5,
    #               xMax: float = 0.5, yMax: float = 0.5, zMax: float = 0.5,
    #               tolerance: float = 0.0001):
    #     """
    #     Create a Graph from an IFC file. This code is partially based on code from Bruno Postle.

    #     Parameters
    #     ----------
    #     file : file
    #         The input IFC file
    #     includeTypes : list , optional
    #         A list of IFC object types to include in the graph. Default is [] which means all object types are included.
    #     excludeTypes : list , optional
    #         A list of IFC object types to exclude from the graph. Default is [] which mean no object type is excluded.
    #     includeRels : list , optional
    #         A list of IFC relationship types to include in the graph. Default is [] which means all relationship types are included.
    #     excludeRels : list , optional
    #         A list of IFC relationship types to exclude from the graph. Default is [] which mean no relationship type is excluded.
    #     transferDictionaries : bool , optional
    #         If set to True, the dictionaries from the IFC file will be transferred to the topology. Otherwise, they won't. Default is False.
    #     useInternalVertex : bool , optional
    #         If set to True, use an internal vertex to represent the subtopology. Otherwise, use its centroid. Default is False.
    #     storeBREP : bool , optional
    #         If set to True, store the BRep of the subtopology in its representative vertex. Default is False.
    #     removeCoplanarFaces : bool , optional
    #         If set to True, coplanar faces are removed. Otherwise they are not. Default is False.
    #     xMin : float, optional
    #         The desired minimum value to assign for a vertex's X coordinate. Default is -0.5.
    #     yMin : float, optional
    #         The desired minimum value to assign for a vertex's Y coordinate. Default is -0.5.
    #     zMin : float, optional
    #         The desired minimum value to assign for a vertex's Z coordinate. Default is -0.5.
    #     xMax : float, optional
    #         The desired maximum value to assign for a vertex's X coordinate. Default is 0.5.
    #     yMax : float, optional
    #         The desired maximum value to assign for a vertex's Y coordinate. Default is 0.5.
    #     zMax : float, optional
    #         The desired maximum value to assign for a vertex's Z coordinate. Default is 0.5.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.
        
    #     Returns
    #     -------
    #     topologic_core.Graph
    #         The created graph.
        
    #     """
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Vertex import Vertex
    #     from topologicpy.Edge import Edge
    #     from topologicpy.Graph import Graph
    #     from topologicpy.Dictionary import Dictionary
    #     try:
    #         import ifcopenshell
    #         import ifcopenshell.util.placement
    #         import ifcopenshell.util.element
    #         import ifcopenshell.util.shape
    #         import ifcopenshell.geom
    #     except:
    #         print("Graph.ByIFCFile - Warning: Installing required ifcopenshell library.")
    #         try:
    #             os.system("pip install ifcopenshell")
    #         except:
    #             os.system("pip install ifcopenshell --user")
    #         try:
    #             import ifcopenshell
    #             import ifcopenshell.util.placement
    #             import ifcopenshell.util.element
    #             import ifcopenshell.util.shape
    #             import ifcopenshell.geom
    #             print("Graph.ByIFCFile - Warning: ifcopenshell library installed correctly.")
    #         except:
    #             warnings.warn("Graph.ByIFCFile - Error: Could not import ifcopenshell. Please try to install ifcopenshell manually. Returning None.")
    #             return None
        
    #     import random

    #     def vertexAtKeyValue(vertices, key, value):
    #         for v in vertices:
    #             d = Topology.Dictionary(v)
    #             d_value = Dictionary.ValueAtKey(d, key)
    #             if value == d_value:
    #                 return v
    #         return None

    #     def IFCObjects(ifc_file, include=[], exclude=[]):
    #         include = [s.lower() for s in include]
    #         exclude = [s.lower() for s in exclude]
    #         all_objects = ifc_file.by_type('IfcProduct')
    #         return_objects = []
    #         for obj in all_objects:
    #             is_a = obj.is_a().lower()
    #             if is_a in exclude:
    #                 continue
    #             if is_a in include or len(include) == 0:
    #                 return_objects.append(obj)
    #         return return_objects

    #     def IFCObjectTypes(ifc_file):
    #         products = IFCObjects(ifc_file)
    #         obj_types = []
    #         for product in products:
    #             obj_types.append(product.is_a())  
    #         obj_types = list(set(obj_types))
    #         obj_types.sort()
    #         return obj_types

    #     def IFCRelationshipTypes(ifc_file):
    #         rel_types = [ifc_rel.is_a() for ifc_rel in ifc_file.by_type("IfcRelationship")]
    #         rel_types = list(set(rel_types))
    #         rel_types.sort()
    #         return rel_types

    #     def IFCRelationships(ifc_file, include=[], exclude=[]):
    #         include = [s.lower() for s in include]
    #         exclude = [s.lower() for s in exclude]
    #         rel_types = [ifc_rel.is_a() for ifc_rel in ifc_file.by_type("IfcRelationship")]
    #         rel_types = list(set(rel_types))
    #         relationships = []
    #         for ifc_rel in ifc_file.by_type("IfcRelationship"):
    #             rel_type = ifc_rel.is_a().lower()
    #             if rel_type in exclude:
    #                 continue
    #             if rel_type in include or len(include) == 0:
    #                 relationships.append(ifc_rel)
    #         return relationships

    #     def get_psets(entity):
    #         # Initialize the PSET dictionary for this entity
    #         psets = {}
            
    #         # Check if the entity has a GlobalId
    #         if not hasattr(entity, 'GlobalId'):
    #             raise ValueError("The provided entity does not have a GlobalId.")
            
    #         # Get the property sets related to this entity
    #         for definition in entity.IsDefinedBy:
    #             if definition.is_a('IfcRelDefinesByProperties'):
    #                 property_set = definition.RelatingPropertyDefinition
                    
    #                 # Check if it is a property set
    #                 if not property_set == None:
    #                     if property_set.is_a('IfcPropertySet'):
    #                         pset_name = "IFC_"+property_set.Name
                            
    #                         # Dictionary to hold individual properties
    #                         properties = {}
                            
    #                         # Iterate over the properties in the PSET
    #                         for prop in property_set.HasProperties:
    #                             if prop.is_a('IfcPropertySingleValue'):
    #                                 # Get the property name and value
    #                                 prop_name = "IFC_"+prop.Name
    #                                 prop_value = prop.NominalValue.wrappedValue if prop.NominalValue else None
    #                                 properties[prop_name] = prop_value
                            
    #                         # Add this PSET to the dictionary for this entity
    #                         psets[pset_name] = properties
    #         return psets
        
    #     def get_color_transparency_material(entity):
    #         import random

    #         # Set default Material Name and ID
    #         material_list = []
    #         # Set default transparency based on entity type or material
    #         default_transparency = 0.0
            
    #         # Check if the entity is an opening or made of glass
    #         is_a = entity.is_a().lower()
    #         if "opening" in is_a or "window" in is_a or "door" in is_a or "space" in is_a:
    #             default_transparency = 0.7
    #         elif "space" in is_a:
    #             default_transparency = 0.8
            
    #         # Check if the entity has constituent materials (e.g., glass)
    #         else:
    #             # Check for associated materials (ConstituentMaterial or direct material assignment)
    #             materials_checked = False
    #             if hasattr(entity, 'HasAssociations'):
    #                 for rel in entity.HasAssociations:
    #                     if rel.is_a('IfcRelAssociatesMaterial'):
    #                         material = rel.RelatingMaterial
    #                         if material.is_a('IfcMaterial') and 'glass' in material.Name.lower():
    #                             default_transparency = 0.5
    #                             materials_checked = True
    #                         elif material.is_a('IfcMaterialLayerSetUsage'):
    #                             material_layers = material.ForLayerSet.MaterialLayers
    #                             for layer in material_layers:
    #                                 material_list.append(layer.Material.Name)
    #                                 if 'glass' in layer.Material.Name.lower():
    #                                     default_transparency = 0.5
    #                                     materials_checked = True
                                        
    #             # Check for ConstituentMaterial if available
    #             if hasattr(entity, 'HasAssociations') and not materials_checked:
    #                 for rel in entity.HasAssociations:
    #                     if rel.is_a('IfcRelAssociatesMaterial'):
    #                         material = rel.RelatingMaterial
    #                         if material.is_a('IfcMaterialConstituentSet'):
    #                             for constituent in material.MaterialConstituents:
    #                                 material_list.append(constituent.Material.Name)
    #                                 if 'glass' in constituent.Material.Name.lower():
    #                                     default_transparency = 0.5
    #                                     materials_checked = True

    #             # Check if the entity has ShapeAspects with associated materials or styles
    #             if hasattr(entity, 'HasShapeAspects') and not materials_checked:
    #                 for shape_aspect in entity.HasShapeAspects:
    #                     if hasattr(shape_aspect, 'StyledByItem') and shape_aspect.StyledByItem:
    #                         for styled_item in shape_aspect.StyledByItem:
    #                             for style in styled_item.Styles:
    #                                 if style.is_a('IfcSurfaceStyle'):
    #                                     for surface_style in style.Styles:
    #                                         if surface_style.is_a('IfcSurfaceStyleRendering'):
    #                                             transparency = getattr(surface_style, 'Transparency', default_transparency)
    #                                             if transparency > 0:
    #                                                 default_transparency = transparency

    #         # Try to get the actual color and transparency if defined
    #         if hasattr(entity, 'Representation') and entity.Representation:
    #             for rep in entity.Representation.Representations:
    #                 for item in rep.Items:
    #                     if hasattr(item, 'StyledByItem') and item.StyledByItem:
    #                         for styled_item in item.StyledByItem:
    #                             if hasattr(styled_item, 'Styles'):
    #                                 for style in styled_item.Styles:
    #                                     if style.is_a('IfcSurfaceStyle'):
    #                                         for surface_style in style.Styles:
    #                                             if surface_style.is_a('IfcSurfaceStyleRendering'):
    #                                                 color = surface_style.SurfaceColour
    #                                                 transparency = getattr(surface_style, 'Transparency', default_transparency)
    #                                                 return (color.Red*255, color.Green*255, color.Blue*255), transparency, material_list
            
    #         # If no color is defined, return a consistent random color based on the entity type
    #         if "wall" in is_a:
    #             color = (175, 175, 175)
    #         elif "slab" in is_a:
    #             color = (200, 200, 200)
    #         elif "space" in is_a:
    #             color = (250, 250, 250)
    #         else:
    #             random.seed(hash(is_a))
    #             color = (random.random(), random.random(), random.random())
            
    #         return color, default_transparency, material_list

    #     def vertexByIFCObject(ifc_object, object_types, restrict=False):
    #         settings = ifcopenshell.geom.settings()
    #         settings.set(settings.USE_WORLD_COORDS,True)
    #         try:
    #             shape = ifcopenshell.geom.create_shape(settings, ifc_object)
    #         except:
    #             shape = None
    #         if shape or restrict == False: #Only add vertices of entities that have 3D geometries.
    #             obj_id = ifc_object.id()
    #             psets = ifcopenshell.util.element.get_psets(ifc_object)
    #             obj_type = ifc_object.is_a()
    #             obj_type_id = object_types.index(obj_type)
    #             name = "Untitled"
    #             LongName = "Untitled"
    #             try:
    #                 name = ifc_object.Name
    #             except:
    #                 name = "Untitled"
    #             try:
    #                 LongName = ifc_object.LongName
    #             except:
    #                 LongName = name

    #             if name == None:
    #                 name = "Untitled"
    #             if LongName == None:
    #                 LongName = "Untitled"
    #             label = str(obj_id)+" "+LongName+" ("+obj_type+" "+str(obj_type_id)+")"
    #             try:
    #                 grouped_verts = ifcopenshell.util.shape.get_vertices(shape.geometry)
    #                 vertices = [Vertex.ByCoordinates(list(coords)) for coords in grouped_verts]
    #                 centroid = Vertex.Centroid(vertices)
    #             except:
    #                 x = random.uniform(xMin,xMax)
    #                 y = random.uniform(yMin,yMax)
    #                 z = random.uniform(zMin,zMax)
    #                 centroid = Vertex.ByCoordinates(x, y, z)
                
    #             # Store relevant information
    #             if transferDictionaries == True:
    #                 color, transparency, material_list = get_color_transparency_material(ifc_object)
    #                 if color == None:
    #                     color = "white"
    #                 if transparency == None:
    #                     transparency = 0
    #                 entity_dict = {
    #                     "TOPOLOGIC_id": str(Topology.UUID(centroid)),
    #                     "TOPOLOGIC_name": getattr(ifc_object, 'Name', "Untitled"),
    #                     "TOPOLOGIC_type": Topology.TypeAsString(centroid),
    #                     "TOPOLOGIC_color": color,
    #                     "TOPOLOGIC_opacity": 1.0 - transparency,
    #                     "IFC_global_id": getattr(ifc_object, 'GlobalId', 0),
    #                     "IFC_name": getattr(ifc_object, 'Name', "Untitled"),
    #                     "IFC_type": ifc_object.is_a(),
    #                     "IFC_material_list": material_list,
    #                 }
    #                 topology_dict = Dictionary.ByPythonDictionary(entity_dict)
    #                 # Get PSETs dictionary
    #                 pset_python_dict = get_psets(ifc_object)
    #                 pset_dict = Dictionary.ByPythonDictionary(pset_python_dict)
    #                 topology_dict = Dictionary.ByMergedDictionaries([topology_dict, pset_dict])
    #                 if storeBREP == True or useInternalVertex == True:
    #                     shape_topology = None
    #                     if hasattr(ifc_object, "Representation") and ifc_object.Representation:
    #                         for rep in ifc_object.Representation.Representations:
    #                             if rep.is_a("IfcShapeRepresentation"):
    #                                 try:
    #                                     # Generate the geometry for this entity
    #                                     shape = ifcopenshell.geom.create_shape(settings, ifc_object)
    #                                     # Get grouped vertices and grouped faces     
    #                                     grouped_verts = shape.geometry.verts
    #                                     verts = [ [grouped_verts[i], grouped_verts[i + 1], grouped_verts[i + 2]] for i in range(0, len(grouped_verts), 3)]
    #                                     grouped_edges = shape.geometry.edges
    #                                     edges = [[grouped_edges[i], grouped_edges[i + 1]] for i in range(0, len(grouped_edges), 2)]
    #                                     grouped_faces = shape.geometry.faces
    #                                     faces = [ [grouped_faces[i], grouped_faces[i + 1], grouped_faces[i + 2]] for i in range(0, len(grouped_faces), 3)]
    #                                     shape_topology = Topology.ByGeometry(verts, edges, faces, silent=True)
    #                                     if not shape_topology == None:
    #                                         if removeCoplanarFaces == True:
    #                                             shape_topology = Topology.RemoveCoplanarFaces(shape_topology, epsilon=0.0001)
    #                                 except:
    #                                     pass
    #                     if not shape_topology == None and storeBREP:
    #                         topology_dict = Dictionary.SetValuesAtKeys(topology_dict, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(shape_topology), Topology.Type(shape_topology), Topology.TypeAsString(shape_topology)])
    #                     if not shape_topology == None and useInternalVertex == True:
    #                         centroid = Topology.InternalVertex(shape_topology)
    #                 centroid = Topology.SetDictionary(centroid, topology_dict)
    #             return centroid
    #         return None

    #     def edgesByIFCRelationships(ifc_relationships, ifc_types, vertices):
    #         tuples = []
    #         edges = []

    #         for ifc_rel in ifc_relationships:
    #             source = None
    #             destinations = []
    #             if ifc_rel.is_a("IfcRelConnectsPorts"):
    #                 source = ifc_rel.RelatingPort
    #                 destinations = ifc_rel.RelatedPorts
    #             elif ifc_rel.is_a("IfcRelConnectsPortToElement"):
    #                 source = ifc_rel.RelatingPort
    #                 destinations = [ifc_rel.RelatedElement]
    #             elif ifc_rel.is_a("IfcRelAggregates"):
    #                 source = ifc_rel.RelatingObject
    #                 destinations = ifc_rel.RelatedObjects
    #             elif ifc_rel.is_a("IfcRelNests"):
    #                 source = ifc_rel.RelatingObject
    #                 destinations = ifc_rel.RelatedObjects
    #             elif ifc_rel.is_a("IfcRelAssignsToGroup"):
    #                 source = ifc_rel.RelatingGroup
    #                 destinations = ifc_rel.RelatedObjects
    #             elif ifc_rel.is_a("IfcRelConnectsPathElements"):
    #                 source = ifc_rel.RelatingElement
    #                 destinations = [ifc_rel.RelatedElement]
    #             elif ifc_rel.is_a("IfcRelConnectsStructuralMember"):
    #                 source = ifc_rel.RelatingStructuralMember
    #                 destinations = [ifc_rel.RelatedStructuralConnection]
    #             elif ifc_rel.is_a("IfcRelContainedInSpatialStructure"):
    #                 source = ifc_rel.RelatingStructure
    #                 destinations = ifc_rel.RelatedElements
    #             elif ifc_rel.is_a("IfcRelFillsElement"):
    #                 source = ifc_rel.RelatingOpeningElement
    #                 destinations = [ifc_rel.RelatedBuildingElement]
    #             elif ifc_rel.is_a("IfcRelSpaceBoundary"):
    #                 source = ifc_rel.RelatingSpace
    #                 destinations = [ifc_rel.RelatedBuildingElement]
    #             elif ifc_rel.is_a("IfcRelVoidsElement"):
    #                 source = ifc_rel.RelatingBuildingElement
    #                 destinations = [ifc_rel.RelatedOpeningElement]
    #             elif ifc_rel.is_a("IfcRelDefinesByProperties") or ifc_rel.is_a("IfcRelAssociatesMaterial") or ifc_rel.is_a("IfcRelDefinesByType"):
    #                 source = None
    #                 destinations = None
    #             else:
    #                 print("Graph.ByIFCFile - Warning: The relationship", ifc_rel, "is not supported. Skipping.")
    #             if source:
    #                 sv = vertexAtKeyValue(vertices, key="IFC_global_id", value=getattr(source, 'GlobalId', 0))
    #                 if sv:
    #                     si = Vertex.Index(sv, vertices, tolerance=tolerance)
    #                     if not si == None:
    #                         for destination in destinations:
    #                             if destination == None:
    #                                 continue
    #                             ev = vertexAtKeyValue(vertices, key="IFC_global_id", value=getattr(destination, 'GlobalId', 0),)
    #                             if ev:
    #                                 ei = Vertex.Index(ev, vertices, tolerance=tolerance)
    #                                 if not ei == None:
    #                                     if not([si,ei] in tuples or [ei,si] in tuples):
    #                                         tuples.append([si,ei])
    #                                         e = Edge.ByVertices([sv,ev])
    #                                         d = Dictionary.ByKeysValues(["IFC_global_id", "IFC_name", "IFC_type"], [ifc_rel.id(), ifc_rel.Name, ifc_rel.is_a()])
    #                                         e = Topology.SetDictionary(e, d)
    #                                         edges.append(e)
    #         return edges
        
    #     ifc_types = IFCObjectTypes(file)
    #     ifc_objects = IFCObjects(file, include=includeTypes, exclude=excludeTypes)
    #     vertices = []
    #     for ifc_object in ifc_objects:
    #         v = vertexByIFCObject(ifc_object, ifc_types)
    #         if v:
    #             vertices.append(v)
    #     if len(vertices) > 0:
    #         ifc_relationships = IFCRelationships(file, include=includeRels, exclude=excludeRels)
    #         edges = edgesByIFCRelationships(ifc_relationships, ifc_types, vertices)
    #         g = Graph.ByVerticesEdges(vertices, edges)
    #     else:
    #         g = None
    #     return g

    @staticmethod
    def ByIFCPath(path,
                  includeTypes=[],
                  excludeTypes=[],
                  includeRels=[],
                  excludeRels=[],
                  transferDictionaries=False,
                  useInternalVertex=False,
                  storeBREP=False,
                  removeCoplanarFaces=False,
                  xMin=-0.5, yMin=-0.5, zMin=-0.5, xMax=0.5, yMax=0.5, zMax=0.5):
        """
        Create a Graph from an IFC path. This code is partially based on code from Bruno Postle.

        Parameters
        ----------
        path : str
            The input IFC file path.
        includeTypes : list , optional
            A list of IFC object types to include in the graph. Default is [] which means all object types are included.
        excludeTypes : list , optional
            A list of IFC object types to exclude from the graph. Default is [] which mean no object type is excluded.
        includeRels : list , optional
            A list of IFC relationship types to include in the graph. Default is [] which means all relationship types are included.
        excludeRels : list , optional
            A list of IFC relationship types to exclude from the graph. Default is [] which mean no relationship type is excluded.
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the IFC file will be transferred to the topology. Otherwise, they won't. Default is False.
        useInternalVertex : bool , optional
            If set to True, use an internal vertex to represent the subtopology. Otherwise, use its centroid. Default is False.
        storeBREP : bool , optional
            If set to True, store the BRep of the subtopology in its representative vertex. Default is False.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are removed. Otherwise they are not. Default is False.
        xMin : float, optional
            The desired minimum value to assign for a vertex's X coordinate. Default is -0.5.
        yMin : float, optional
            The desired minimum value to assign for a vertex's Y coordinate. Default is -0.5.
        zMin : float, optional
            The desired minimum value to assign for a vertex's Z coordinate. Default is -0.5.
        xMax : float, optional
            The desired maximum value to assign for a vertex's X coordinate. Default is 0.5.
        yMax : float, optional
            The desired maximum value to assign for a vertex's Y coordinate. Default is 0.5.
        zMax : float, optional
            The desired maximum value to assign for a vertex's Z coordinate. Default is 0.5.
        
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
        return Graph.ByIFCFile(ifc_file,
                               includeTypes=includeTypes,
                               excludeTypes=excludeTypes,
                               includeRels=includeRels,
                               excludeRels=excludeRels,
                               transferDictionaries=transferDictionaries,
                               useInternalVertex=useInternalVertex,
                               storeBREP=storeBREP,
                               removeCoplanarFaces=removeCoplanarFaces,
                               xMin=xMin, yMin=yMin, zMin=zMin, xMax=xMax, yMax=yMax, zMax=zMax)

    @staticmethod
    def ByJSONDictionary(
        jsonDictionary: dict,
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
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Loads a Graph from a JSON file and attaches graph-, vertex-, and edge-level dictionaries.

        Parameters
        ----------
        path : str
            Path to a JSON file containing:
            - graph-level properties under `graphPropsKey` (default "properties"),
            - a vertex dict under `verticesKey` (default "vertices") keyed by vertex IDs,
            - an edge dict under `edgesKey` (default "edges") keyed by edge IDs.
        xKey: str , optional
            JSON key used to read vertex's x coordinate. Default is "x".
        yKey: str , optional
            JSON key used to read vertex's y coordinate. Default is "y".
        zKey: str , optional
            JSON key used to read vertex's z coordinate. Default is "z".
        vertexIDKey : str , optional
            If not None, the vertex dictionary key under which to store the JSON vertex id. Default is "id".
        edgeSourceKey: str , optional
            JSON key used to read edge's start vertex. Default is "source".
        edgeTargetKey: str , optional
            JSON key used to read edge's end vertex. Default is "target".
        edgeIDKey : str , optional
            If not None, the edge dictionary key under which to store the JSON edge id. Default is "id".
        graphPropsKey: str , optional
            JSON key for the graph properties section. Default is "properties".
        verticesKey: str , optional
            JSON key for the vertices section. Default is "vertices".
        edgesKey: str , optional
            JSON key for the edges section. Default is "edges".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, no warnings or error messages are displayed. Default is False.

        Returns
        -------
        topologic_core.Graph
        """
        # --- Imports kept local by request ---
        import json
        import math
        from typing import Any, Iterable

        # TopologicPy imports
        from topologicpy.Graph import Graph
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        # --- Helper functions kept local by request ---
        def _to_plain(value: Any) -> Any:
            "Convert numpy/pandas-ish scalars/arrays and nested containers to plain Python."
            try:
                import numpy as _np  # optional
                if isinstance(value, _np.generic):
                    return value.item()
                if isinstance(value, _np.ndarray):
                    return [_to_plain(v) for v in value.tolist()]
            except Exception:
                pass
            if isinstance(value, (list, tuple)):
                return [_to_plain(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _to_plain(v) for k, v in value.items()}
            if isinstance(value, float):
                if not math.isfinite(value):
                    return 0.0
                # normalize -0.0
                return 0.0 if abs(value) < tolerance else float(value)
            return value

        def _round_num(x: Any, m: int) -> float:
            "Safe float conversion + rounding + tolerance clamp."
            try:
                xf = float(x)
            except Exception:
                return 0.0
            if not math.isfinite(xf):
                return 0.0
            # clamp tiny values to zero to avoid -0.0 drift and floating trash
            if abs(xf) < tolerance:
                xf = 0.0
            return round(xf, max(0, int(m)))

        def _dict_from(obj: dict, drop_keys: Iterable[str] = ()):
            "Create a Topologic Dictionary from a Python dict (optionally dropping some keys)."
            data = {k: _to_plain(v) for k, v in obj.items() if k not in drop_keys}
            if not data:
                return None
            keys = list(map(str, data.keys()))
            vals = list(data.values())
            try:
                return Dictionary.ByKeysValues(keys, vals)
            except Exception:
                # As a last resort, stringify nested types
                import json as _json
                vals2 = [_json.dumps(v) if isinstance(v, (list, dict)) else v for v in vals]
                return Dictionary.ByKeysValues(keys, vals2)

        # --- Load JSON ---
        if not isinstance(jsonDictionary, dict):
            if not silent:
                print(f"Graph.ByJSONDictionary - Error: The input JSON Dictionary parameter is not a valid python dictionary. Returning None.")
            return None

        gprops = jsonDictionary.get(graphPropsKey, {}) or {}
        verts = jsonDictionary.get(verticesKey, {}) or {}
        edges = jsonDictionary.get(edgesKey, {}) or {}

        # --- Build vertices ---
        id_to_vertex = {}
        vertex_list = []
        for v_id, v_rec in verts.items():
            x = _round_num(v_rec.get(xKey, 0.0), mantissa)
            y = _round_num(v_rec.get(yKey, 0.0), mantissa)
            z = _round_num(v_rec.get(zKey, 0.0), mantissa)
            try:
                v = Vertex.ByCoordinates(x, y, z)
            except Exception as e:
                if not silent:
                    print(f"Graph.ByJSONDictionary - Warning: failed to create Vertex {v_id} at ({x},{y},{z}): {e}")
                continue

            # Attach vertex dictionary with all attributes except raw coords
            v_dict_py = dict(v_rec)
            if vertexIDKey:
                v_dict_py[vertexIDKey] = v_id
            v_dict = _dict_from(v_dict_py, drop_keys={xKey, yKey, zKey})
            if v_dict:
                v = Topology.SetDictionary(v, v_dict)

            id_to_vertex[str(v_id)] = v
            vertex_list.append(v)

        # --- Build edges ---
        edge_list = []
        for e_id, e_rec in edges.items():
            s_id = e_rec.get(edgeSourceKey)
            t_id = e_rec.get(edgeTargetKey)
            if s_id is None or t_id is None:
                if not silent:
                    print(f"Graph.ByJSONDictionary - Warning: skipping Edge {e_id}: missing '{edgeSourceKey}' or '{edgeTargetKey}'.")
                continue
            s_id = str(s_id)
            t_id = str(t_id)
            if s_id not in id_to_vertex or t_id not in id_to_vertex:
                if not silent:
                    print(f"Graph.ByJSONDictionary - Warning: skipping Edge {e_id}: unknown endpoint(s) {s_id}->{t_id}.")
                continue
            u = id_to_vertex[s_id]
            v = id_to_vertex[t_id]
            try:
                e = Edge.ByVertices(u, v)
            except Exception as ee:
                if not silent:
                    print(f"Graph.ByJSONDictionary - Warning: failed to create Edge {e_id}: {ee}")
                continue

            # Attach full edge record as dictionary (including source/target keys)
            e_dict = _dict_from(dict(e_rec), drop_keys=())
            if edgeIDKey:
                Dictionary.SetValueAtKey(e_dict, edgeIDKey, e_id)
            if e_dict:
                e = Topology.SetDictionary(e, e_dict)
            edge_list.append(e)

        # --- Assemble graph ---
        try:
            g = Graph.ByVerticesEdges(vertex_list, edge_list)
        except Exception:
            # Fallback: create empty, then add
            g = Graph.ByVerticesEdges([], [])
            for v in vertex_list:
                try:
                    g = Graph.AddVertex(g, v)
                except Exception:
                    pass
            for e in edge_list:
                try:
                    g = Graph.AddEdge(g, e)
                except Exception:
                    pass

        # --- Graph-level dictionary ---
        g_dict = _dict_from(dict(gprops), drop_keys=())
        if g_dict:
            g = Topology.SetDictionary(g, g_dict)

        return g

    @staticmethod
    def ByJSONFile(file,
                   xKey: str = "x",
                   yKey: str = "y",
                   zKey: str = "z",
                   vertexIDKey: str = "id",
                   edgeSourceKey: str = "source",
                   edgeTargetKey: str = "target",
                   edgeIDKey: str = "id",
                   graphPropsKey: str = "properties",
                   verticesKey: str = "vertices",
                   edgesKey: str = "edges",
                   mantissa: int = 6,
                   tolerance: float = 0.0001,
                   silent: bool = False):
        """
        Imports the graph from a JSON file.

        Parameters
        ----------
        file : file object
            The input JSON file.
        xKey: str , optional
            JSON key used to read vertex's x coordinate. Default is "x".
        yKey: str , optional
            JSON key used to read vertex's y coordinate. Default is "y".
        zKey: str , optional
            JSON key used to read vertex's z coordinate. Default is "z".
        vertexIDKey : str , optional
            If not None, the vertex dictionary key under which to store the JSON vertex id. Default is "id".
        edgeSourceKey: str , optional
            JSON key used to read edge's start vertex. Default is "source".
        edgeTargetKey: str , optional
            JSON key used to read edge's end vertex. Default is "target".
        edgeIDKey : str , optional
            If not None, the edge dictionary key under which to store the JSON edge id. Default is "id".
        graphPropsKey: str , optional
            JSON key for the graph properties section. Default is "properties".
        verticesKey: str , optional
            JSON key for the vertices section. Default is "vertices".
        edgesKey: str , optional
            JSON key for the edges section. Default is "edges".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, no warnings or error messages are displayed. Default is False.

        Returns
        -------
        topologic_graph
            the imported graph.

        """
        import json
        if not file:
            if not silent:
                print("Topology.ByJSONFile - Error: the input file parameter is not a valid file. Returning None.")
            return None
        try:
            json_dict = json.load(file)
        except Exception as e:
            if not silent:
                print("Graph.ByJSONFile - Error: Could not load the JSON file: {e}. Returning None.")
            return None
        return Graph.ByJSONDictionary(json_dict,
                                      xKey=xKey,
                                      yKey=yKey,
                                      zKey=zKey,
                                      vertexIDKey=vertexIDKey,
                                      edgeSourceKey=edgeSourceKey,
                                      edgeTargetKey=edgeTargetKey,
                                      edgeIDKey=edgeIDKey,
                                      graphPropsKey=graphPropsKey,
                                      verticesKey=verticesKey,
                                      edgesKey=edgesKey,
                                      mantissa=mantissa,
                                      tolerance=tolerance,
                                      silent=silent)
    
    @staticmethod
    def ByJSONPath(path,
                   xKey: str = "x",
                   yKey: str = "y",
                   zKey: str = "z",
                   vertexIDKey: str = "id",
                   edgeSourceKey: str = "source",
                   edgeTargetKey: str = "target",
                   edgeIDKey: str = "id",
                   graphPropsKey: str = "properties",
                   verticesKey: str = "vertices",
                   edgesKey: str = "edges",
                   mantissa: int = 6,
                   tolerance: float = 0.0001,
                   silent: bool = False):
        """
        Imports the graph from a JSON file.

        Parameters
        ----------
        path : str
            The file path to the json file.
        xKey: str , optional
            JSON key used to read vertex's x coordinate. Default is "x".
        yKey: str , optional
            JSON key used to read vertex's y coordinate. Default is "y".
        zKey: str , optional
            JSON key used to read vertex's z coordinate. Default is "z".
        vertexIDKey : str , optional
            If not None, the vertex dictionary key under which to store the JSON vertex id. Default is "id".
        edgeSourceKey: str , optional
            JSON key used to read edge's start vertex. Default is "source".
        edgeTargetKey: str , optional
            JSON key used to read edge's end vertex. Default is "target".
        edgeIDKey : str , optional
            If not None, the edge dictionary key under which to store the JSON edge id. Default is "id".
        graphPropsKey: str , optional
            JSON key for the graph properties section. Default is "properties".
        verticesKey: str , optional
            JSON key for the vertices section. Default is "vertices".
        edgesKey: str , optional
            JSON key for the edges section. Default is "edges".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, no warnings or error messages are displayed. Default is False.

        Returns
        -------
        list
            The list of imported topologies.

        """
        import json
        if not path:
            if not silent:
                print("Graph.ByJSONPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        try:
            with open(path) as file:
                json_dict = json.load(file)
        except Exception as e:
            if not silent:
                print(f"Graph.ByJSONPath - Error: Could not load file: {e}. Returning None.")
            return None
        return Graph.ByJSONDictionary(json_dict,
                                      xKey=xKey,
                                      yKey=yKey,
                                      zKey=zKey,
                                      vertexIDKey=vertexIDKey,
                                      edgeSourceKey=edgeSourceKey,
                                      edgeTargetKey=edgeTargetKey,
                                      edgeIDKey=edgeIDKey,
                                      graphPropsKey=graphPropsKey,
                                      verticesKey=verticesKey,
                                      edgesKey=edgesKey,
                                      mantissa=mantissa,
                                      tolerance=tolerance,
                                      silent=silent)

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
            The desired tolerance. Default is 0.0001.

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
    def ByNetworkXGraph(nxGraph, vertexID="id", xKey="x", yKey="y", zKey="z", coordsKey='coords', randomRange=(-1, 1), mantissa: int = 6, tolerance: float = 0.0001):
        """
        Converts the input NetworkX graph into a topologic Graph. See http://networkx.org

        Parameters
        ----------
        nxGraph : NetworkX graph
            The input NetworkX graph.
        vertexID : str , optional
            The dictionary key under which the node id should be stored. The default is "id".
        xKey : str , optional
            The dictionary key under which to find the X-Coordinate of the vertex. Default is 'x'.
        yKey : str , optional
            The dictionary key under which to find the Y-Coordinate of the vertex. Default is 'y'.
        zKey : str , optional
            The dictionary key under which to find the Z-Coordinate of the vertex. Default is 'z'.
        coordsKey : str , optional
            The dictionary key under which to find the list of the coordinates vertex. Default is 'coords'.
        randomRange : tuple , optional
            The range to use for random position coordinates if no values are found in the dictionaries. Default is (-1,1)
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologicpy.Graph
            The created topologic graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        import random
        import numpy as np
        import math
        import torch
        from collections.abc import Mapping, Sequence

        def _is_iterable_but_not_str(x):
            return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))

        def _to_python_scalar(x):
            """Return a plain Python scalar if x is a numpy/pandas/Decimal/torch scalar; otherwise return x."""
            # numpy scalar
            if np is not None and isinstance(x, np.generic):
                return x.item()
            # pandas NA
            if pd is not None and x is pd.NA:
                return None
            # pandas Timestamp/Timedelta
            if pd is not None and isinstance(x, (pd.Timestamp, pd.Timedelta)):
                return x.isoformat()
            # torch scalar tensor
            if torch is not None and isinstance(x, torch.Tensor) and x.dim() == 0:
                return _to_python_scalar(x.item())
            # decimal
            try:
                from decimal import Decimal
                if isinstance(x, Decimal):
                    return float(x)
            except Exception:
                pass
            return x

        def _to_python_list(x):
            """Convert arrays/series/tensors/sets/tuples to Python lists (recursively)."""
            if torch is not None and isinstance(x, torch.Tensor):
                x = x.detach().cpu().tolist()
            elif np is not None and isinstance(x, (np.ndarray,)):
                x = x.tolist()
            elif pd is not None and isinstance(x, (pd.Series, pd.Index)):
                x = x.tolist()
            elif isinstance(x, (set, tuple)):
                x = list(x)
            return x

        def _round_number(x, mantissa):
            """Round finite floats; keep ints; sanitize NaNs/Infs to None."""
            if isinstance(x, bool):  # bool is int subclass; keep as bool
                return x
            if isinstance(x, int):
                return x
            # try float conversion
            try:
                xf = float(x)
            except Exception:
                return x  # not a number
            if math.isfinite(xf):
                return round(xf, mantissa)
            return None  # NaN/Inf -> None

        def clean_value(value, mantissa):
            """
            Recursively convert value into TopologicPy-friendly types:
            - numbers rounded to mantissa
            - sequences -> lists (cleaned)
            - mappings -> dicts (cleaned)
            - datetime -> isoformat
            - other objects -> str(value)
            """
            # First, normalize common library wrappers
            value = _to_python_scalar(value)

            # Datetime from stdlib
            import datetime
            if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
                try:
                    return value.isoformat()
                except Exception:
                    return str(value)

            # Mapping (dict-like)
            if isinstance(value, Mapping):
                return {str(k): clean_value(v, mantissa) for k, v in value.items()}

            # Sequences / arrays / tensors -> list
            if _is_iterable_but_not_str(value) or (
                (np is not None and isinstance(value, (np.ndarray,))) or
                (torch is not None and isinstance(value, torch.Tensor)) or
                (pd is not None and isinstance(value, (pd.Series, pd.Index)))
            ):
                value = _to_python_list(value)
                return [clean_value(v, mantissa) for v in value]

            # Strings stay as-is
            if isinstance(value, (str, bytes, bytearray)):
                return value.decode() if isinstance(value, (bytes, bytearray)) else value

            # Numbers (or things that can be safely treated as numbers)
            out = _round_number(value, mantissa)
            # If rounder didn't change type and it's still a weird object, stringify it
            if out is value and not isinstance(out, (type(None), bool, int, float, str)):
                return str(out)
            return out

        def coerce_xyz(val, mantissa, default=0.0):
            """
            Coerce a candidate XYZ value into a float:
            - if mapping with 'x' or 'value' -> try those
            - if sequence -> use first element
            - if string -> try float
            - arrays/tensors -> first element
            - fallback to default
            """
            if val is None:
                return round(float(default), mantissa)
            # library scalars
            val = _to_python_scalar(val)

            # Mapping with common keys
            if isinstance(val, Mapping):
                for k in ("x", "value", "val", "coord", "0"):
                    if k in val:
                        return coerce_xyz(val[k], mantissa, default)
                # otherwise try to take first value
                try:
                    first = next(iter(val.values()))
                    return coerce_xyz(first, mantissa, default)
                except Exception:
                    return round(float(default), mantissa)

            # Sequence / array / tensor
            if _is_iterable_but_not_str(val) or \
            (np is not None and isinstance(val, (np.ndarray,))) or \
            (torch is not None and isinstance(val, torch.Tensor)) or \
            (pd is not None and isinstance(val, (pd.Series, pd.Index))):
                lst = _to_python_list(val)
                if len(lst) == 0:
                    return round(float(default), mantissa)
                return coerce_xyz(lst[0], mantissa, default)

            # String
            if isinstance(val, str):
                try:
                    return round(float(val), mantissa)
                except Exception:
                    return round(float(default), mantissa)

            # Numeric
            try:
                return _round_number(val, mantissa)
            except Exception:
                return round(float(default), mantissa)

                # Create a mapping from NetworkX nodes to TopologicPy vertices
                nx_to_topologic_vertex = {}

        # Create TopologicPy vertices for each node in the NetworkX graph
        vertices = []

        # Create a node to index dictionary
        node_to_index = {}
        i = 0
        for node, data in nxGraph.nodes(data=True):
            # Create a mapping of node -> index based on enumeration
            node_to_index[node] = i
            # Clean the node dictionary
            cleaned_values = []
            cleaned_keys = []
            for k, v in data.items():
                cleaned_keys.append(str(k))
                cleaned_values.append(clean_value(v, mantissa))
            data = dict(zip(cleaned_keys, cleaned_values))
            # Defensive defaults for coordinates
            x_raw = y_raw = z_raw = None
            try:
                x_raw = data.get(xKey, None)
                y_raw = data.get(yKey, None)
                z_raw = data.get(zKey, None)
            except Exception:
                x_raw = y_raw = z_raw = None
            
            if x_raw == None:
                coords = data.get(coordsKey, None)
                if coords:
                    coords = clean_value(coords, mantissa)
                    if isinstance(coords, list):
                        if len(coords) == 2:
                            x_raw = coords[0]
                            y_raw = coords[1]
                            z_raw = 0
                        elif len(coords) == 3:
                            x_raw = coords[0]
                            y_raw = coords[1]
                            z_raw = coords[2]

            # Fall back to random only if missing / invalid
            x = coerce_xyz(x_raw, mantissa, default=random.uniform(*randomRange))
            y = coerce_xyz(y_raw, mantissa, default=random.uniform(*randomRange))
            z = coerce_xyz(z_raw, mantissa, default=0.0)

            # Create vertex
            vertex = Vertex.ByCoordinates(x, y, z)

            # Build and attach TopologicPy dictionary
            node_dict = Dictionary.ByKeysValues(cleaned_keys, cleaned_values)
            node_dict = Dictionary.SetValueAtKey(node_dict, vertexID, node)
            vertex = Topology.SetDictionary(vertex, node_dict)

            #nx_to_topologic_vertex[node] = vertex
            vertices.append(vertex)
            i += 1
        
        # Create TopologicPy edges for each edge in the NetworkX graph
        edges = []
        for u, v, data in nxGraph.edges(data=True):
            u_index = node_to_index[u]
            v_index = node_to_index[v]
            start_vertex = vertices[u_index]
            end_vertex = vertices[v_index]
            # Create a TopologicPy edge with the edge data dictionary
            # Clean the node dictionary
            cleaned_values = []
            cleaned_keys = []
            for k, v in data.items():
                cleaned_keys.append(str(k))
                cleaned_values.append(clean_value(v, mantissa))
            edge_dict = Dictionary.ByKeysValues(cleaned_keys, cleaned_values)
            edge = Edge.ByVertices([start_vertex, end_vertex], tolerance=tolerance)
            edge = Topology.SetDictionary(edge, edge_dict)
            edges.append(edge)

        # Create and return the TopologicPy graph
        topologic_graph = Graph.ByVerticesEdges(vertices, edges)
        return topologic_graph

    @staticmethod
    def BySpatialRelationships(
        *topologies,
        include: list = ["contains", "coveredBy", "covers", "crosses", "disjoint", "equals", "overlaps", "touches","within", "proximity"],
        proximityValues = [1, 5, 10],
        proximityLabels = ["near", "intermediate", "far"],
        useShortestDistance: bool = False,
        useInternalVertex: bool = False,
        vertexIDKey: str = "id",
        edgeKeyFwd: str = "relFwd",
        edgeKeyBwd: str = "relBwd",
        connectsKey:str = "connects",
        storeBREP: bool = False,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False
        ):
        """
        Creates a graph of the spatial relationships of the input topologies according to OGC / ISO 19107 / DE-9IM / RCC-8

        Parameters
        ----------
        *topologies : list
            The list of input topologies
        include : list , optional
            The type(s) of spatial relationships to build. Default is ["contains", "disjoint", "equals", "overlaps", "touches", "within", "covers", "coveredBy"]
        proximityValues: list , optional
            The list of maximum distance values that specify the desired proximityLabel.
            This list must be sorted in ascending order and have the same number of elements as the proximityLabels list.
            Objects that are further than the largest specified distance are not classified and not included.
            An object is considered to fall within the range if it is less than or equal to the value in this list.
            If you wish ALL objects to be classified specifiy the last number in this list to be larger than the
            largest distance that can exist between any two objects in the list. Default is [1, 5, 10]
        proximityLabels: list , optional
            The list of range labels (e.g. "near", "intermediate", "far") that correspond to the proximityValues list.
            The list must have the same number of elements as the proximityValues list. Default is ["near", "intermediate", "far"]
        useShortestDistance: bool , optional
            If set to True, the shortest distance between objects is used. Otherwise, the distance between their centroids is used. Default is False.
        useInternalVertex: bool , optional
            If set to True, an internal vertex of the represented topology will be used as a graph node.
            Otherwise, its centroid will be used. Default is False.
        vertexIDKey: str, optional
            The vertex ID key under which to store a unique numerical and sequential id. Default is "id".
        edgeKeyFwd: str , optional
            The edge key under which to store the forward relationship (from start vertex to end vertex). Default is "relFwd".
        edgeKeyBwd: str , optional
            The edge key under which to store the backward relationship (from end vertex to start vertex). Default is "relBwd".
        connectsKey: str , optional
            The edge key under which to store the indices of the vertices connected by each edge. Default is "connects".
        storeBREP : bool , optional
            If set to True, store the BRep of the topology in its representative vertex. Default is False.
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.graph
            The created graph.

        """
        from topologicpy.Graph import Graph
        from topologicpy.BVH import BVH
        from topologicpy.Cell import Cell
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        # ---------- normalize + filter ----------
        inc = [s.lower() for s in include]  # enable case-insensitive membership
        want_disjoint = "disjoint" in inc
        want_rel = set(inc)  # membership checks

        topologyList = Helper.Flatten(list(topologies))
        topologyList = [t for t in topologyList if Topology.IsInstance(t, "Topology")]
        n = len(topologyList)

        if n == 0:
            if not silent:
                print("Graph.BySpatialRelationships - Error: No valid topologies. Returning None")
            return None
        
        if len(proximityValues) != len(proximityLabels):
            if not silent:
                print("Graph.BySpatialRelationships - Error: the proximityValues and proximityLabels input parameters are not of the same length. Returning None")
            return None
        
        include = [t.lower() for t in include if t.lower() in ["contains", "coveredBy", "covers", "crosses", "disjoint", "equals", "overlaps", "touches","within", "proximity"]]
        
        if len(include) == 0:
            if not silent:
                print("Graph.BySpatialRelationships - Error: The include input parameter does not contain any valid spatial relationship types. Returning None")
            return None

        if n == 1:
            v = Topology.InternalVertex(topologyList[0]) if useInternalVertex else Topology.Centroid(topologyList[0])
            v = Topology.SetDictionary(v, Topology.Dictionary(topologyList[0]))
            return Graph.ByVerticesEdges([v], [])

        # ---------- Calculate Proximity ------------

        def _calc_proximity(topologies: list,
                            ranges: list,
                            labels: list,
                            useShortestDistance: bool = True,
                            tolerance: float = 0.0001,
                            silent: bool = False):
            """
            Creates a proximity graph from a list of topologies using a BVH
            to prune distance checks for large input sets.

            Each topology is represented by a vertex in the graph.
            An edge is created between two vertices if the distance between
            their corresponding topologies is <= max(ranges).

            A BVH is used as a broad-phase accelerator: for each topology,
            a query box centred at its centroid and sized to cover the
            maximum proximity range is used to retrieve only nearby
            candidates. Exact distances are then computed only for those
            candidates.

            Parameters
            ----------
            topologies : list of topologic_core.Topology
                The input topologies to be represented as vertices.
            ranges : list of float
                A list of positive numeric thresholds (e.g. [1.0, 3.0, 5.0]).
                Interpreted as upper bounds of proximity bands.
                Distances larger than max(ranges) are ignored (no edge).
            labels : list of str
                A list of proximity labels, same length as `ranges`.
                For a pair distance d:
                    - d <= ranges[0] -> labels[0]
                    - ranges[0] < d <= ranges[1] -> labels[1]
                    - ...
            useShortestDistance : bool , optional
                If True, use Topology.ShortestDistance(topologyA, topologyB)
                if available. If False (or if that fails), fall back to the
                distance between the centroids of the two topologies.
                Default is True.
            tolerance : float , optional
                A small numeric tolerance used when comparing distances to
                range bounds. Default is 0.0001.
            silent : bool , optional
                If False, basic sanity-check warnings are printed.
                Default is False.

            Returns
            -------
            graph : topologic_core.Graph
                A graph whose vertices correspond to the input topologies
                and whose edges connect topologies that fall within the
                supplied distance ranges. Each edge dictionary contains:
                    - "distance"  : float  (actual distance)
                    - "proximity" : str    (label from `labels`)
                    - "range_max" : float  (upper bound used for the bin)
                    - "source_index" : int
                    - "target_index" : int

            Notes
            -----
            - Complexity is approximately O(n log n + k) where k is the
                number of candidate pairs returned by the BVH.
            - BVH is used only as a broad-phase filter; exact distance
                tests still guarantee correctness with respect to `ranges`.
            """

            # Basic validation
            if not isinstance(topologies, list) or len(topologies) < 2:
                if not silent:
                    print("Graph.BySpatialRelationships - Error: Need a list of at least two topologies.")
                return None

            if not isinstance(ranges, list) or not isinstance(labels, list):
                if not silent:
                    print("Graph.BySpatialRelationships - Error: 'proximityValues' and 'proximityLabels' must be lists.")
                return None

            if len(ranges) == 0 or len(ranges) != len(labels):
                if not silent:
                    print("Graph.BySpatialRelationships - Error: 'proximityValues' must be non-empty and "
                            "have the same length as 'labels'.")
                return None

            # Sort ranges and labels together (ascending by range)
            try:
                rl = sorted(zip(ranges, labels), key=lambda x: x[0])
            except Exception:
                if not silent:
                    print("Graph.BySpatialRelationships - Error: Could not sort ranges; check they are numeric.")
                return None

            sorted_ranges = [r for (r, _) in rl]
            sorted_labels = [lab for (_, lab) in rl]

            max_range = sorted_ranges[-1]

            # Precompute representative vertices (centroids) for each topology
            vertices = []
            n = len(topologies)
            for i, topo in enumerate(topologies):
                if not topo:
                    if not silent:
                        print(f"Graph.BySpatialRelationships - Warning: Ignoring None topology at index {i}.")
                    vertices.append(None)
                    continue
                try:
                    c_vtx = Topology.Centroid(topo)
                except Exception:
                    # Fallback if centroid fails
                    if not silent:
                        print(f"Graph.BySpatialRelationships - Error: Failed to compute centroid for topology {i}, "
                                f"using origin as placeholder.")
                    c_vtx = Vertex.ByCoordinates(0, 0, 0)

                # Attach index dictionary to the vertex (not to the original topology)
                d_keys = ["index"]
                d_vals = [i]
                v_dict = Dictionary.ByKeysValues(d_keys, d_vals)
                c_vtx = Topology.SetDictionary(c_vtx, v_dict)
                vertices.append(c_vtx)

            # Build BVH on the original topologies
            try:
                bvh = BVH.ByTopologies(topologies)
            except Exception as e:
                if not silent:
                    print(f"Graph.BySpatialRelationships - Error: Failed to build BVH, falling back to O(n^2): {e}")
                # Fallback: use the non-BVH variant if you like,
                # or just early-return None. Here we just bail out.
                return None

            # Map from topology identity to index for fast lookup
            id_to_index = {id(topo): i for i, topo in enumerate(topologies)}

            # Helper to compute distance between two topologies
            def _distance(topoA, topoB, vA, vB):
                d_val = None
                if useShortestDistance:
                    try:
                        d_val = Topology.ShortestDistance(topoA, topoB)
                    except Exception:
                        d_val = None
                if d_val is None:
                    try:
                        d_val = Vertex.Distance(vA, vB)
                    except Exception:
                        d_val = None
                return d_val

            edges = []

            # Main loop: for each topology, query BVH for candidates within
            # a bounding box of size 2*max_range around its centroid.
            for i in range(n):
                topo_i = topologies[i]
                v_i = vertices[i]
                if topo_i is None or v_i is None:
                    continue

                # Build a query box centered at the centroid with size 2 * max_range
                try:
                    query_box = Cell.Prism(
                        origin=v_i,
                        width=2 * max_range,
                        length=2 * max_range,
                        height=2 * max_range
                    )
                except Exception as q_err:
                    if not silent:
                        print(f"Graph.BySpatialRelationships - Error: Failed to build query box for {i}: {q_err}")
                    continue

                try:
                    candidates = BVH.Clashes(bvh, query_box)
                except Exception as c_err:
                    if not silent:
                        print(f"Graph.BySpatialRelationships - Error: BVH.Clashes failed for {i}: {c_err}")
                    continue

                if not candidates:
                    continue

                for cand in candidates:
                    j = id_to_index.get(id(cand), None)
                    if j is None:
                        continue
                    # Enforce i < j to avoid duplicate edges
                    if j <= i:
                        continue

                    topo_j = topologies[j]
                    v_j = vertices[j]
                    if topo_j is None or v_j is None:
                        continue

                    # Compute exact distance
                    d = _distance(topo_i, topo_j, v_i, v_j)
                    if d is None:
                        if not silent:
                            print(f"Graph.BySpatialRelationships - Error: Could not compute distance between "
                                    f"{i} and {j}.")
                        continue

                    # Skip if beyond max range (plus tolerance)
                    if d > max_range + tolerance:
                        continue

                    # Bin the distance into the appropriate range/label
                    label = None
                    range_max = None
                    for r, lab in zip(sorted_ranges, sorted_labels):
                        if d <= r + tolerance:
                            label = lab
                            range_max = r
                            break

                    if label is None:
                        continue

                    e_keys = ["distance", "proximity", "range_max", "source_index", "target_index"]
                    e_values = [float(d), str(label), float(range_max), i, j]
                    d = Dictionary.ByKeysValues(e_keys, e_values)
                    edges.append(d)

            return edges

        # ---------- BVH once ----------
        bvh = BVH.ByTopologies(topologyList, silent=True)

        # ---------- O(1) index for objects ----------
        index_of = {id(t): i for i, t in enumerate(topologyList)}

        # ---------- precompute per-topology data once ----------
        vertices_objs = [None] * n
        vertex_dicts = [None] * n
        coords = [None] * n
        for i, t in enumerate(topologyList):
            d = Topology.Dictionary(t)
            d = Dictionary.SetValueAtKey(d, vertexIDKey, i)
            if storeBREP == True:
                d = Dictionary.SetValuesAtKeys(d, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(t), Topology.Type(t), Topology.TypeAsString(t)])
            vertex_dicts[i] = d
            v = Topology.InternalVertex(t) if useInternalVertex else Topology.Centroid(t)
            vertices_objs[i] = v
            coords[i] = Vertex.Coordinates(v, mantissa=mantissa)

        # ---------- optional: build AABBs once for ultra-fast disjoint path ----------
        try:
            from topologicpy.BVH import AABB
            have_aabb = True
            aabbs = []
            for t in topologyList:
                verts = Topology.Vertices(t) or []
                pts = [Vertex.Coordinates(v, mantissa=mantissa) for v in verts]
                # If topology has no vertices (degenerate), fall back to centroid to avoid errors
                if not pts:
                    c = Vertex.Coordinates(Topology.Centroid(t), mantissa=mantissa)
                    pts = [c]
                aabbs.append(AABB.from_points(pts, pad=tolerance))
        except Exception:
            have_aabb = False
            aabbs = None

        # ---------- edges ----------
        edges = []
        edge_dicts = []

        def _add_edge(ai, bj, rel):
            # Map forward/backward labels (no index swapping; keep ai < bj)
            r = rel
            r_lc = r.lower()
            if r_lc == "contains":
                fwd, bwd = "contains", "within"
            elif r_lc == "within":
                fwd, bwd = "within", "contains"
            elif r_lc == "covers":
                fwd, bwd = "covers", "coveredBy"
            elif r_lc == "coveredby":  # tolerate lowercased include
                fwd, bwd = "coveredBy", "covers"
            else:
                # symmetric (disjoint/equals/overlaps/touches) or already directional tag returned
                # keep same tag both ways for symmetric predicates
                fwd = rel
                bwd = rel
            if [ai, bj] in edges:
                i = edges.index([ai, bj])
                edge_dict = Dictionary.ByKeysValues(
                    [edgeKeyFwd, edgeKeyBwd, connectsKey],
                    [fwd, bwd, [ai, bj]])
                edge_dicts[i] = edge_dict
            else:

                edges.append([ai, bj])
                edge_dicts.append(
                    Dictionary.ByKeysValues(
                        [edgeKeyFwd, edgeKeyBwd, connectsKey],
                        [fwd, bwd, [ai, bj]],
                    ))

        # ---------- main loops (each unordered pair once) ----------
        if "proximity" in include:
            prox_dicts = _calc_proximity(topologies = topologyList,
                            ranges = proximityValues,
                            labels = proximityLabels,
                            useShortestDistance = useShortestDistance,
                            tolerance = tolerance,
                            silent = silent)
            for prox_dict in prox_dicts:
                ai = Dictionary.ValueAtKey(prox_dict, "source_index")
                bj = Dictionary.ValueAtKey(prox_dict, "target_index")
                rel = Dictionary.ValueAtKey(prox_dict, "proximity")
                if (rel in proximityLabels):
                    _add_edge(ai, bj, rel)
        
        for i, a in enumerate(topologyList):
            candidates = []
            ai = i
            candidates = BVH.Clashes(bvh, a) or []
            if not candidates:
                # If you want to connect "disjoint" to *all* non-candidates, that would be O(n) per i.
                # We intentionally skip that to keep algorithmic cost bounded by BVH outputs.
                continue

            for b in candidates:
                bj = index_of.get(id(b))
                if bj is None or bj <= ai: 
                    continue  # skip self and already-processed pairs

                # Ultra-fast "disjoint" emit via AABB if requested and boxes do not overlap
                if have_aabb and not aabbs[ai].overlaps(aabbs[bj]):
                    if want_disjoint:
                        _add_edge(ai, bj, "disjoint")
                    continue  # done with this pair
                
                # Otherwise evaluate exact relation (short-circuit inside)
                rel = Topology.SpatialRelationship(
                    a,
                    b,
                    include=include,  # honor user's include filter inside the predicate
                    mantissa=mantissa,
                    tolerance=tolerance,
                    silent=True
                )

                # If the predicate returns a relation in the allowed set, emit the edge
                # Note: include could contain "coveredBy" while we normalized to lower
                rel_ok = (rel is not None) and (
                    rel.lower() in want_rel or rel in include  # accept either case
                )
                if rel_ok:
                    _add_edge(ai, bj, rel)

                # Optional optimization: if disjoint was requested and rel == "disjoint",
                # it will be added by the block above already.

        # ---------- avoid expensive Vertex.Separate unless duplicates exist ----------
        seen = set()
        has_dups = False
        for c in coords:
            key = (round(c[0], mantissa), round(c[1], mantissa), round(c[2], mantissa))
            if key in seen:
                has_dups = True
                break
            seen.add(key)

        if has_dups:
            # Rare, but if present, resolve once
            vertices_objs = Vertex.Separate(vertices_objs, minDistance=tolerance * 10)
            coords = [Vertex.Coordinates(v, mantissa=mantissa) for v in vertices_objs]

        # ---------- build graph ----------
        graph = Graph.ByMeshData(
            vertices=coords,
            edges=edges,
            vertexDictionaries=vertex_dicts,
            edgeDictionaries=edge_dicts,
        )
        return graph

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
                   edgeCategoryKey : str = "category",
                   useInternalVertex: bool = False,
                   storeBREP: bool =False,
                   mantissa: int = 6,
                   tolerance: float = 0.0001,
                   silent: float = False):
        """
        Creates a graph.See https://en.wikipedia.org/wiki/Graph_(discrete_mathematics).

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        direct : bool , optional
            If set to True, connect the subtopologies directly with a single edge. Default is True.
        directApertures : bool , optional
            If set to True, connect the subtopologies directly with a single edge if they share one or more apertures. Default is False.
        viaSharedTopologies : bool , optional
            If set to True, connect the subtopologies via their shared topologies. Default is False.
        viaSharedApertures : bool , optional
            If set to True, connect the subtopologies via their shared apertures. Default is False.
        toExteriorTopologies : bool , optional
            If set to True, connect the subtopologies to their exterior topologies. Default is False.
        toExteriorApertures : bool , optional
            If set to True, connect the subtopologies to their exterior apertures. Default is False.
        toContents : bool , optional
            If set to True, connect the subtopologies to their contents. Default is False.
        toOutposts : bool , optional
            If set to True, connect the topology to the list specified in its outposts. Default is False.
        idKey : str , optional
            The key to use to find outpost by ID. It is case insensitive. Default is "TOPOLOGIC_ID".
        outpostsKey : str , optional
            The key to use to find the list of outposts. It is case insensitive. Default is "outposts".
        vertexCategoryKey : str , optional
            The key under which to store the node type. Node categories are:
            0 : main topology
            1 : shared topology
            2 : shared aperture
            3 : exterior topology
            4 : exterior aperture
            5 : content
            6 : outpost
            The default is "category".
        edgeCategoryKey : str , optional
            The key under which to store the node type. Edge categories are:
            0 : direct
            1 : via shared topology
            2 : via shared aperture
            3 : to exterior topology
            4 : to exterior aperture
            5 : to content
            6 : to outpost
            The default is "category".
        useInternalVertex : bool , optional
            If set to True, use an internal vertex to represent the subtopology. Otherwise, use its centroid. Default is False.
        storeBREP : bool , optional
            If set to True, store the BRep of the subtopology in its representative vertex. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        
        if not Topology.IsInstance(topology, "topology"):
            if not silent:
                print("Graph.ByTopology - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        def _viaSharedTopologies(vt, sharedTops):
            verts = []
            eds = []
            for sharedTopology in sharedTops:
                if useInternalVertex == True:
                    vst = Topology.InternalVertex(sharedTopology, tolerance=tolerance)
                else:
                    vst = Topology.CenterOfMass(sharedTopology)
                d1 = Topology.Dictionary(sharedTopology)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 1) # shared topology
                if storeBREP:
                    d1 = Dictionary.SetValuesAtKeys(d1, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                vst = Topology.SetDictionary(vst, d1, silent=True)
                verts.append(vst)
                tempe = Edge.ByStartVertexEndVertex(vt, vst, tolerance=tolerance)
                tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["Via_Shared_Topologies", 1])
                tempe = Topology.SetDictionary(tempe, tempd, silent=True)
                eds.append(tempe)
            return verts, eds
        
        def _viaSharedApertures(vt, sharedAps):
            verts = []
            eds = []
            for sharedAp in sharedAps:
                if useInternalVertex == True:
                    vsa = Topology.InternalVertex(sharedAp, tolerance=tolerance)
                else:
                    vsa = Topology.CenterOfMass(sharedAp)
                d1 = Topology.Dictionary(sharedAp)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 2) # shared aperture
                vsa = Vertex.ByCoordinates(Vertex.X(vsa, mantissa=mantissa)+(tolerance*100), Vertex.Y(vsa, mantissa=mantissa)+(tolerance*100), Vertex.Z(vsa, mantissa=mantissa)+(tolerance*100))
                if storeBREP:
                    d1 = Dictionary.SetValuesAtKeys(d1, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedAp), Topology.Type(sharedAp), Topology.TypeAsString(sharedAp)])
                vsa = Topology.SetDictionary(vsa, d1, silent=True)
                verts.append(vsa)
                tempe = Edge.ByStartVertexEndVertex(vt, vsa, tolerance=tolerance)
                tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["Via_Shared_Apertures", 2])
                tempe = Topology.SetDictionary(tempe, tempd, silent=True)
                eds.append(tempe)
            return verts, eds
        
        def _toExteriorTopologies(vt, exteriorTops):
            verts = []
            eds = []
            for i, exteriorTop in enumerate(exteriorTops):
                if useInternalVertex == True:
                    vet = Topology.InternalVertex(exteriorTop, tolerance=tolerance)
                else:
                    vet = Topology.CenterOfMass(exteriorTop)
                d1 = Topology.Dictionary(exteriorTop)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 3) # exterior topology
                if storeBREP:
                    d1 = Dictionary.SetValuesAtKeys(d1, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTop), Topology.Type(exteriorTop), Topology.TypeAsString(exteriorTop)])
                vet = Topology.SetDictionary(vet, d1, silent=True)
                verts.append(vet)
                tempe = Edge.ByStartVertexEndVertex(vt, vet, tolerance=tolerance)
                tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["To_Exterior_Topologies", 3])
                tempe = Topology.SetDictionary(tempe, tempd, silent=True)
                eds.append(tempe)
            return verts, eds
        
        def _toExteriorApertures(vt, exteriorAps):
            verts = []
            eds = []
            for exAp in exteriorAps:
                if useInternalVertex == True:
                    vea = Topology.InternalVertex(exAp, tolerance=tolerance)
                else:
                    vea = Topology.CenterOfMass(exAp)
                d1 = Topology.Dictionary(exAp)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 4) # exterior aperture
                vea = Vertex.ByCoordinates(Vertex.X(vea, mantissa=mantissa)+(tolerance*100), Vertex.Y(vea, mantissa=mantissa)+(tolerance*100), Vertex.Z(vea, mantissa=mantissa)+(tolerance*100))
                if storeBREP:
                    d1 = Dictionary.SetValuesAtKeys(d1, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(exAp), Topology.Type(exAp), Topology.TypeAsString(exAp)])
                vea = Topology.SetDictionary(vea, d1, silent=True)
                verts.append(vea)
                tempe = Edge.ByStartVertexEndVertex(vt, vea, tolerance=tolerance)
                tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["To_Exterior_Apertures", 4])
                tempe = Topology.SetDictionary(tempe, tempd, silent=True)
                eds.append(tempe)
            return verts, eds
        
        def _toContents(vt, contents):
            verts = []
            eds = []
            for content in contents:
                if Topology.IsInstance(content, "Aperture"):
                    content = Aperture.Topology(content)
                if useInternalVertex == True:
                    vct = Topology.InternalVertex(content, tolerance=tolerance)
                else:
                    vct = Topology.CenterOfMass(content)
                vct = Vertex.ByCoordinates(Vertex.X(vct, mantissa=mantissa)+(tolerance*100), Vertex.Y(vct, mantissa=mantissa)+(tolerance*100), Vertex.Z(vct, mantissa=mantissa)+(tolerance*100))
                d1 = Topology.Dictionary(content)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                if storeBREP:
                    d1 = Dictionary.SetValuesAtKeys(d1, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                vct = Topology.SetDictionary(vct, d1, silent=True)
                verts.append(vct)
                tempe = Edge.ByStartVertexEndVertex(vt, vct, tolerance=tolerance)
                tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["To_Contents", 5])
                tempe = Topology.SetDictionary(tempe, tempd, silent=True)
                eds.append(tempe)
            return verts, eds

        def _toOutposts(vt, otherTops):
            verts = []
            eds = []
            d = Topology.Dictionary(vt)
            if not d == None:
                keys = Dictionary.Keys(d)
            else:
                keys = []
            k = None
            for key in keys:
                if key.lower() == outpostsKey.lower():
                    k = key
            if k:
                ids = Dictionary.ValueAtKey(d,k)
                outposts = outpostsByID(otherTops, ids, idKey)
            else:
                outposts = []
            for outpost in outposts:
                if useInternalVertex == True:
                    vop = Topology.InternalVertex(outpost, tolerance=tolerance)   
                else:
                    vop = Topology.CenterOfMass(outpost)

                d1 = Topology.Dictionary(vop)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 6) # outpost
                if storeBREP:
                    d1 = Dictionary.SetValuesAtKeys(d1, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(outpost), Topology.Type(outpost), Topology.TypeAsString(outpost)])
                vop = Topology.SetDictionary(vop, d1, silent=True)
                verts.append(vop)
                tempe = Edge.ByStartVertexEndVertex(vt, vop, tolerance=tolerance)
                tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["To_Outposts", 6])
                tempe = Topology.SetDictionary(tempe, tempd, silent=True)
                eds.append(tempe)
            return verts, eds
        
        def mergeDictionaries(sources):
            if isinstance(sources, list) == False:
                sources = [sources]
            sinkKeys = []
            sinkValues = []
            d = Topology.Dictionary(sources[0])
            if d != None:
                stlKeys = d.Keys()
                if len(stlKeys) > 0:
                    sinkKeys = d.Keys()
                    sinkValues = Dictionary.Values(d)
            for i in range(1,len(sources)):
                d = Topology.Dictionary(sources[i])
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
            graph_vertices = []
            graph_edges = []
            cellmat = []
            # Store all the vertices of the cells of the cellComplex
            cells = Topology.Cells(topology)
            for cell in cells:
                if useInternalVertex == True:
                    vCell = Topology.InternalVertex(cell, tolerance=tolerance)
                else:
                    vCell = Topology.CenterOfMass(cell)
                d1 = Topology.Dictionary(cell)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                if storeBREP:
                    d1 = Dictionary.SetValuesAtKeys(d1, ["brep", "brepType", "brepTypeString"], [Topology.BREPString(cell), Topology.Type(cell), Topology.TypeAsString(cell)])
                vCell = Topology.SetDictionary(vCell, d1, silent=True)
                graph_vertices.append(vCell)
            if direct == True:
                cells = Topology.Cells(topology)
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
                                    v1 = Topology.CenterOfMass(cells[i])
                                    v2 = Topology.CenterOfMass(cells[j])
                                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                mDict = mergeDictionaries(sharedt)
                                if not mDict == None:
                                    keys = (Dictionary.Keys(mDict) or [])+["relationship", edgeCategoryKey]
                                    values = (Dictionary.Values(mDict) or [])+["Direct", 0]
                                else:
                                    keys = ["relationship", edgeCategoryKey]
                                    values = ["Direct", 0]
                                mDict = Dictionary.ByKeysValues(keys, values)
                                if mDict:
                                    e = Topology.SetDictionary(e, mDict, silent=True)
                                graph_edges.append(e)
            if directApertures == True:
                cellmat = []
                cells = Topology.Cells(topology)
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
                                        v1 = Topology.CenterOfMass(cells[i])
                                        v2 = Topology.CenterOfMass(cells[j])
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    mDict = mergeDictionaries(apTopList)
                                    if not mDict == None:
                                        keys = (Dictionary.Keys(mDict) or [])+["relationship", edgeCategoryKey]
                                        values = (Dictionary.Values(mDict) or [])+["Direct", 0]
                                    else:
                                        keys = ["relationship", edgeCategoryKey]
                                        values = ["Direct", 0]
                                    mDict = Dictionary.ByKeysValues(keys, values)
                                    if mDict:
                                        e = Topology.SetDictionary(e, mDict, silent=True)
                                    graph_edges.append(e)
            cells = Topology.Cells(topology)
            if any([viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents]):
                for aCell in cells:
                    if useInternalVertex == True:
                        vCell = Topology.InternalVertex(aCell, tolerance=tolerance)
                    else:
                        vCell = Topology.CenterOfMass(aCell)
                    d = Topology.Dictionary(aCell)
                    vCell = Topology.SetDictionary(vCell, d, silent=True)
                    faces = Topology.Faces(aCell)
                    sharedTopologies = []
                    exteriorTopologies = []
                    sharedApertures = []
                    exteriorApertures = []
                    cell_contents = Topology.Contents(aCell)
                    for aFace in faces:
                        cells1 = Topology.SuperTopologies(aFace, topology, topologyType="Cell")
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
                        verts, eds = _viaSharedTopologies(vCell, sharedTopologies)
                        graph_vertices += verts
                        graph_edges += eds
                        for sharedTopology in sharedTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedTopology, tolerance=tolerance)
                            else:
                                vst = Topology.CenterOfMass(sharedTopology)
                            d = Topology.Dictionary(sharedTopology)
                            vst = Topology.SetDictionary(vst, d, silent=True)
                            if toContents:
                                shd_top_contents = Topology.Contents(sharedTopology)
                                verts, eds = _toContents(vst, shd_top_contents)
                                graph_vertices += verts
                                graph_edges += eds
                            if toOutposts and others:
                                verts, eds = _toOutposts(vst, others)
                                graph_vertices += verts
                                graph_edges += eds
                    if viaSharedApertures:
                        verts, eds = _viaSharedApertures(vCell, sharedApertures)
                        graph_vertices += verts
                        graph_edges += eds
                    if toExteriorTopologies:
                        verts, eds = _toExteriorTopologies(vCell, exteriorTopologies)
                        graph_vertices += verts
                        graph_edges += eds
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vet = Topology.InternalVertex(exteriorTopology, tolerance=tolerance)
                            else:
                                vet = Topology.CenterOfMass(exteriorTopology)
                            d = Topology.Dictionary(exteriorTopology)
                            vet = Topology.SetDictionary(vet, d, silent=True)
                            if toContents:
                                ext_top_contents = Topology.Contents(exteriorTopology)
                                verts, eds = _toContents(vet, ext_top_contents)
                                graph_vertices += verts
                                graph_edges += eds
                            if toOutposts and others:
                                verts, eds = _toOutposts(vet, others)
                                graph_vertices += verts
                                graph_edges += eds
                    if toExteriorApertures:
                        verts, eds = _toExteriorApertures(vCell, exteriorApertures)
                        graph_vertices += verts
                        graph_edges += eds
                    if toContents:
                        verts, eds = _toContents(vCell, cell_contents)
                        graph_vertices += verts
                        graph_edges += eds
                    if toOutposts and others:
                        verts, eds = toOutposts(vCell, others)
                        graph_vertices += verts
                        graph_edges += eds
            return [graph_vertices, graph_edges]

        def processCell(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            graph_vertices = []
            graph_edges = []
            if useInternalVertex == True:
                vCell = Topology.InternalVertex(topology, tolerance=tolerance)
            else:
                vCell = Topology.CenterOfMass(topology)
            d1 = Topology.Dictionary(topology)
            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
            if storeBREP:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                vCell = Topology.SetDictionary(vCell, d3, silent=True)
            else:
                vCell = Topology.SetDictionary(vCell, d1, silent=True)
            graph_vertices.append(vCell)
            if any([toExteriorTopologies, toExteriorApertures, toContents, toOutposts]):
                cell_contents = Topology.Contents(topology)
                faces = Topology.Faces(topology)
                exteriorTopologies = []
                exteriorApertures = []
                for aFace in faces:
                    exteriorTopologies.append(aFace)
                    apertures = Topology.Apertures(aFace)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                if toExteriorTopologies:
                    verts, eds = _toExteriorTopologies(vCell, exteriorTopologies)
                    graph_vertices += verts
                    graph_edges += eds
                    for exteriorTopology in exteriorTopologies:
                        if useInternalVertex == True:
                            vet = Topology.InternalVertex(exteriorTopology, tolerance=tolerance)
                        else:
                            vet = Topology.CenterOfMass(exteriorTopology)
                        d = Topology.Dictionary(exteriorTopology)
                        vet = Topology.SetDictionary(vet, d, silent=True)
                        if toContents:
                            ext_top_contents = Topology.Contents(exteriorTopology)
                            verts, eds = _toContents(vet, ext_top_contents)
                            graph_vertices += verts
                            graph_edges += eds
                        if toOutposts and others:
                            verts, eds = _toOutposts(vet, others)
                            graph_vertices += verts
                            graph_edges += eds
                if toExteriorApertures:
                    verts, eds = _toExteriorApertures(vCell, exteriorApertures)
                    graph_vertices += verts
                    graph_edges += eds
                if toContents:
                    verts, eds = _toContents(vCell, cell_contents)
                    graph_vertices += verts
                    graph_edges += eds
                if toOutposts and others:
                    verts, eds = toOutposts(vCell, others)
                    graph_vertices += verts
                    graph_edges += eds
            return [graph_vertices, graph_edges]

        def processShell(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            graph_edges = []
            graph_vertices = []
            facemat = []
            # Store all the vertices of the cells of the cellComplex
            faces = Topology.Faces(topology)
            for face in faces:
                if useInternalVertex == True:
                    vFace = Topology.InternalVertex(face, tolerance=tolerance)
                else:
                    vFace = Topology.CenterOfMass(face)
                d1 = Topology.Dictionary(face)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                if storeBREP:
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(face), Topology.Type(face), Topology.TypeAsString(face)])
                    d3 = mergeDictionaries2([d1, d2])
                    vFace = Topology.SetDictionary(vFace, d3, silent=True)
                else:
                    vFace = Topology.SetDictionary(vFace, d1, silent=True)
                graph_vertices.append(vFace)
            if direct == True:
                faces = Topology.Faces(topology)
                # Create a matrix of zeroes
                for i in range(len(faces)):
                    faceRow = []
                    for j in range(len(faces)):
                        faceRow.append(0)
                    facemat.append(faceRow)
                for i in range(len(faces)):
                    for j in range(len(faces)):
                        if (i != j) and facemat[i][j] == 0:
                            facemat[i][j] = 1
                            facemat[j][i] = 1
                            sharedt = Topology.SharedEdges(faces[i], faces[j])
                            if len(sharedt) > 0:
                                if useInternalVertex == True:
                                    v1 = Topology.InternalVertex(faces[i], tolerance=tolerance)
                                    v2 = Topology.InternalVertex(faces[j], tolerance=tolerance)
                                else:
                                    v1 = Topology.CenterOfMass(faces[i])
                                    v2 = Topology.CenterOfMass(faces[j])
                                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                mDict = mergeDictionaries(sharedt)
                                if not mDict == None:
                                    keys = (Dictionary.Keys(mDict) or [])+["relationship", edgeCategoryKey]
                                    values = (Dictionary.Values(mDict) or [])+["Direct", 0]
                                else:
                                    keys = ["relationship", edgeCategoryKey]
                                    values = ["Direct", 0]
                                mDict = Dictionary.ByKeysValues(keys, values)
                                if mDict:
                                    e = Topology.SetDictionary(e, mDict, silent=True)
                                graph_edges.append(e)
            if directApertures == True:
                facemat = []
                faces = Topology.Faces(topology)
                # Create a matrix of zeroes
                for i in range(len(faces)):
                    faceRow = []
                    for j in range(len(faces)):
                        faceRow.append(0)
                    facemat.append(faceRow)
                for i in range(len(faces)):
                    for j in range(len(faces)):
                        if (i != j) and facemat[i][j] == 0:
                            facemat[i][j] = 1
                            facemat[j][i] = 1
                            sharedt = Topology.SharedEdges(faces[i], faces[j])
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
                                        v1 = Topology.InternalVertex(faces[i], tolerance=tolerance)
                                        v2 = Topology.InternalVertex(faces[j], tolerance=tolerance)
                                    else:
                                        v1 = Topology.CenterOfMass(faces[i])
                                        v2 = Topology.CenterOfMass(faces[j])
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    mDict = mergeDictionaries(apTopList)
                                    if not mDict == None:
                                        keys = (Dictionary.Keys(mDict) or [])+["relationship", edgeCategoryKey]
                                        values = (Dictionary.Values(mDict) or [])+["Direct", 0]
                                    else:
                                        keys = ["relationship", edgeCategoryKey]
                                        values = ["Direct", 0]
                                    mDict = Dictionary.ByKeysValues(keys, values)
                                    if mDict:
                                        e = Topology.SetDictionary(e, mDict, silent=True)
                                    graph_edges.append(e)
            faces = Topology.Faces(topology)
            if any([viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents]):
                for aFace in faces:
                    if useInternalVertex == True:
                        vFace = Topology.InternalVertex(aFace, tolerance=tolerance)
                    else:
                        vFace = Topology.CenterOfMass(aFace)
                    d = Topology.Dictionary(aFace)
                    vFace = Topology.SetDictionary(vFace, d, silent=True)
                    edges = Topology.Edges(aFace)
                    sharedTopologies = []
                    exteriorTopologies = []
                    sharedApertures = []
                    exteriorApertures = []
                    face_contents = Topology.Contents(aFace)
                    for anEdge in edges:
                        faces1 = Topology.SuperTopologies(anEdge, hostTopology=topology, topologyType="Face")
                        if len(faces1) > 1:
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
                        verts, eds = _viaSharedTopologies(vFace, sharedTopologies)
                        graph_vertices += verts
                        graph_edges += eds
                        for sharedTopology in sharedTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedTopology, tolerance=tolerance)
                            else:
                                vst = Topology.CenterOfMass(sharedTopology)
                            d = Topology.Dictionary(sharedTopology)
                            vst = Topology.SetDictionary(vst, d, silent=True)
                            if toContents:
                                shd_top_contents = Topology.Contents(sharedTopology)
                                verts, eds = _toContents(vst, shd_top_contents)
                                graph_vertices += verts
                                graph_edges += eds
                            if toOutposts and others:
                                verts, eds = _toOutposts(vst, others)
                                graph_vertices += verts
                                graph_edges += eds
                    if viaSharedApertures:
                        verts, eds = _viaSharedApertures(vFace, sharedApertures)
                        graph_vertices += verts
                        graph_edges += eds
                    if toExteriorTopologies:
                        verts, eds = _toExteriorTopologies(vFace, exteriorTopologies)
                        graph_vertices += verts
                        graph_edges += eds
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vet = Topology.InternalVertex(exteriorTopology, tolerance=tolerance)
                            else:
                                vet = Topology.CenterOfMass(exteriorTopology)
                            d = Topology.Dictionary(exteriorTopology)
                            vet = Topology.SetDictionary(vet, d, silent=True)
                            if toContents:
                                ext_top_contents = Topology.Contents(exteriorTopology)
                                verts, eds = _toContents(vet, ext_top_contents)
                                graph_vertices += verts
                                graph_edges += eds
                            if toOutposts and others:
                                verts, eds = _toOutposts(vet, others)
                                graph_vertices += verts
                                graph_edges += eds
                    if toExteriorApertures:
                        verts, eds = _toExteriorApertures(vFace, exteriorApertures)
                        graph_vertices += verts
                        graph_edges += eds
                    if toContents:
                        verts, eds = _toContents(vFace, face_contents)
                        graph_vertices += verts
                        graph_edges += eds
                    if toOutposts and others:
                        verts, eds = toOutposts(vFace, others)
                        graph_vertices += verts
                        graph_edges += eds
            return [graph_vertices, graph_edges]

        def processFace(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            graph_vertices = []
            graph_edges = []
            if useInternalVertex == True:
                vFace = Topology.InternalVertex(topology, tolerance=tolerance)
            else:
                vFace = Topology.CenterOfMass(topology)
            d1 = Topology.Dictionary(topology)
            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
            if storeBREP:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                vFace = Topology.SetDictionary(vFace, d3, silent=True)
            else:
                vFace = Topology.SetDictionary(vFace, d1, silent=True)
            graph_vertices.append(vFace)
            if any([toExteriorTopologies, toExteriorApertures, toContents, toOutposts]):
                face_contents = Topology.Contents(topology)
                edges = Topology.Edges(topology)
                exteriorTopologies = []
                exteriorApertures = []
                for anEdge in edges:
                    exteriorTopologies.append(anEdge)
                    apertures = Topology.Apertures(anEdge)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                if toExteriorTopologies:
                    verts, eds = _toExteriorTopologies(vFace, exteriorTopologies)
                    graph_vertices += verts
                    graph_edges += eds
                    for exteriorTopology in exteriorTopologies:
                        if useInternalVertex == True:
                            vet = Topology.InternalVertex(exteriorTopology, tolerance=tolerance)
                        else:
                            vet = Topology.CenterOfMass(exteriorTopology)
                        d = Topology.Dictionary(exteriorTopology)
                        vet = Topology.SetDictionary(vet, d, silent=True)
                        if toContents:
                            ext_top_contents = Topology.Contents(exteriorTopology)
                            verts, eds = _toContents(vet, ext_top_contents)
                            graph_vertices += verts
                            graph_edges += eds
                        if toOutposts and others:
                            verts, eds = _toOutposts(vet, others)
                            graph_vertices += verts
                            graph_edges += eds
                if toExteriorApertures:
                    verts, eds = _toExteriorApertures(vFace, exteriorApertures)
                    graph_vertices += verts
                    graph_edges += eds
                if toContents:
                    verts, eds = _toContents(vFace, face_contents)
                    graph_vertices += verts
                    graph_edges += eds
                if toOutposts and others:
                    verts, eds = toOutposts(vFace, others)
                    graph_vertices += verts
                    graph_edges += eds
            return [graph_vertices, graph_edges]



        def processWire(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            graph_vertices = []
            graph_edges = []
            edgemat = []
            # Store all the vertices of the cells of the cellComplex
            edges = Topology.Edges(topology)
            for edge in edges:
                if useInternalVertex == True:
                    vEdge = Topology.InternalVertex(edge, tolerance=tolerance)
                else:
                    vEdge = Topology.CenterOfMass(edge)
                d1 = Topology.Dictionary(edge)
                d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
                if storeBREP:
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(edge), Topology.Type(edge), Topology.TypeAsString(edge)])
                    d3 = mergeDictionaries2([d1, d2])
                    vEdge = Topology.SetDictionary(vEdge, d3, silent=True)
                else:
                    vEdge = Topology.SetDictionary(vEdge, d1, silent=True)
                graph_vertices.append(vEdge)
            if direct == True:
                edges = Topology.Edges(topology)
                # Create a matrix of zeroes
                for i in range(len(edges)):
                    edgeRow = []
                    for j in range(len(edges)):
                        edgeRow.append(0)
                    edgemat.append(edgeRow)
                for i in range(len(edges)):
                    for j in range(len(edges)):
                        if (i != j) and edgemat[i][j] == 0:
                            edgemat[i][j] = 1
                            edgemat[j][i] = 1
                            sharedt = Topology.SharedVertices(edges[i], edges[j])
                            if len(sharedt) > 0:
                                if useInternalVertex == True:
                                    v1 = Topology.InternalVertex(edges[i], tolerance=tolerance)
                                    v2 = Topology.InternalVertex(edges[j], tolerance=tolerance)
                                else:
                                    v1 = Topology.CenterOfMass(edges[i])
                                    v2 = Topology.CenterOfMass(edges[j])
                                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                mDict = mergeDictionaries(sharedt)
                                if not mDict == None:
                                    keys = (Dictionary.Keys(mDict) or [])+["relationship", edgeCategoryKey]
                                    values = (Dictionary.Values(mDict) or [])+["Direct", 0]
                                else:
                                    keys = ["relationship", edgeCategoryKey]
                                    values = ["Direct", 0]
                                mDict = Dictionary.ByKeysValues(keys, values)
                                if mDict:
                                    e = Topology.SetDictionary(e, mDict, silent=True)
                                graph_edges.append(e)
            if directApertures == True:
                edgemat = []
                edges = Topology.Edges(topology)
                # Create a matrix of zeroes
                for i in range(len(edges)):
                    cellRow = []
                    for j in range(len(edges)):
                        edgeRow.append(0)
                    edgemat.append(edgeRow)
                for i in range(len(edges)):
                    for j in range(len(edges)):
                        if (i != j) and edgemat[i][j] == 0:
                            edgemat[i][j] = 1
                            edgemat[j][i] = 1
                            sharedt = Topology.SharedVertices(edges[i], edges[j])
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
                                        v1 = Topology.InternalVertex(edges[i], tolerance=tolerance)
                                        v2 = Topology.InternalVertex(edges[j], tolerance=tolerance)
                                    else:
                                        v1 = Topology.CenterOfMass(edges[i])
                                        v2 = Topology.CenterOfMass(edges[j])
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    mDict = mergeDictionaries(apTopList)
                                    if not mDict == None:
                                        keys = (Dictionary.Keys(mDict) or [])+["relationship", edgeCategoryKey]
                                        values = (Dictionary.Values(mDict) or [])+["Direct", 0]
                                    else:
                                        keys = ["relationship", edgeCategoryKey]
                                        values = ["Direct", 0]
                                    mDict = Dictionary.ByKeysValues(keys, values)
                                    if mDict:
                                        e = Topology.SetDictionary(e, mDict, silent=True)
                                    graph_edges.append(e)
            edges = Topology.Edges(topology)
            if any([viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents]):
                for anEdge in edges:
                    if useInternalVertex == True:
                        vEdge = Topology.InternalVertex(anEdge, tolerance=tolerance)
                    else:
                        vEdge = Topology.CenterOfMass(anEdge)
                    d = Topology.Dictionary(anEdge)
                    vCell = Topology.SetDictionary(vEdge, d, silent=True)
                    vertices = Topology.Vertices(anEdge)
                    sharedTopologies = []
                    exteriorTopologies = []
                    sharedApertures = []
                    exteriorApertures = []
                    edge_contents = Topology.Contents(anEdge)
                    for aVertex in vertices:
                        edges1 = Topology.SuperTopologies(aVertex, topology, topologyType="Edge")
                        if len(edges1) > 1:
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
                        verts, eds = _viaSharedTopologies(vEdge, sharedTopologies)
                        graph_vertices += verts
                        graph_edges += eds
                        for sharedTopology in sharedTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedTopology, tolerance=tolerance)
                            else:
                                vst = Topology.CenterOfMass(sharedTopology)
                            d = Topology.Dictionary(sharedTopology)
                            vst = Topology.SetDictionary(vst, d, silent=True)
                            if toContents:
                                shd_top_contents = Topology.Contents(sharedTopology)
                                verts, eds = _toContents(vst, shd_top_contents)
                                graph_vertices += verts
                                graph_edges += eds
                            if toOutposts and others:
                                verts, eds = _toOutposts(vst, others)
                                graph_vertices += verts
                                graph_edges += eds
                    if viaSharedApertures:
                        verts, eds = _viaSharedApertures(vEdge, sharedApertures)
                        graph_vertices += verts
                        graph_edges += eds
                    if toExteriorTopologies:
                        verts, eds = _toExteriorTopologies(vEdge, exteriorTopologies)
                        graph_vertices += verts
                        graph_edges += eds
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vet = Topology.InternalVertex(exteriorTopology, tolerance=tolerance)
                            else:
                                vet = Topology.CenterOfMass(exteriorTopology)
                            d = Topology.Dictionary(exteriorTopology)
                            vet = Topology.SetDictionary(vet, d, silent=True)
                            if toContents:
                                ext_top_contents = Topology.Contents(exteriorTopology)
                                verts, eds = _toContents(vet, ext_top_contents)
                                graph_vertices += verts
                                graph_edges += eds
                            if toOutposts and others:
                                verts, eds = _toOutposts(vet, others)
                                graph_vertices += verts
                                graph_edges += eds
                    if toExteriorApertures:
                        verts, eds = _toExteriorApertures(vEdge, exteriorApertures)
                        graph_vertices += verts
                        graph_edges += eds
                    if toContents:
                        verts, eds = _toContents(vEdge, edge_contents)
                        graph_vertices += verts
                        graph_edges += eds
                    if toOutposts and others:
                        verts, eds = toOutposts(vEdge, others)
                        graph_vertices += verts
                        graph_edges += eds
            return [graph_vertices, graph_edges]

        def processEdge(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            graph_vertices = []
            graph_edges = []
            if useInternalVertex == True:
                vEdge = Topology.InternalVertex(topology, tolerance=tolerance)
            else:
                vEdge = Topology.CenterOfMass(topology)
            d1 = Topology.Dictionary(topology)
            d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 0) # main topology
            if storeBREP:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                vEdge = Topology.SetDictionary(vEdge, d3, silent=True)
            else:
                vEdge = Topology.SetDictionary(vEdge, d1, silent=True)
            graph_vertices.append(vEdge)
            if any([toExteriorTopologies, toExteriorApertures, toContents, toOutposts]):
                edge_contents = Topology.Contents(topology)
                vertices = Topology.Vertices(topology)
                exteriorTopologies = []
                exteriorApertures = []
                for aVertex in vertices:
                    exteriorTopologies.append(aVertex)
                    apertures = Topology.Apertures(aVertex)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                if toExteriorTopologies:
                    verts, eds = _toExteriorTopologies(vEdge, exteriorTopologies)
                    graph_vertices += verts
                    graph_edges += eds
                    for exteriorTopology in exteriorTopologies:
                        if useInternalVertex == True:
                            vet = Topology.InternalVertex(exteriorTopology, tolerance=tolerance)
                        else:
                            vet = Topology.CenterOfMass(exteriorTopology)
                        d = Topology.Dictionary(exteriorTopology)
                        vet = Topology.SetDictionary(vet, d, silent=True)
                        if toContents:
                            ext_top_contents = Topology.Contents(exteriorTopology)
                            verts, eds = _toContents(vet, ext_top_contents)
                            graph_vertices += verts
                            graph_edges += eds
                        if toOutposts and others:
                            verts, eds = _toOutposts(vet, others)
                            graph_vertices += verts
                            graph_edges += eds
                if toExteriorApertures:
                    verts, eds = _toExteriorApertures(vEdge, exteriorApertures)
                    graph_vertices += verts
                    graph_edges += eds
                if toContents:
                    verts, eds = _toContents(vEdge, edge_contents)
                    graph_vertices += verts
                    graph_edges += eds
                if toOutposts and others:
                    verts, eds = toOutposts(vEdge, others)
                    graph_vertices += verts
                    graph_edges += eds
            return [graph_vertices, graph_edges]

        def processVertex(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance = item
            vertices = [topology]
            edges = []

            if toContents:
                contents = Topology.Contents(topology)
                for content in contents:
                    if Topology.IsInstance(content, "Aperture"):
                        content = Aperture.Topology(content)
                    if useInternalVertex == True:
                        vst = Topology.InternalVertex(content, tolerance=tolerance)
                    else:
                        vst = Topology.CenterOfMass(content)
                    d1 = Topology.Dictionary(content)
                    d1 = Dictionary.SetValueAtKey(d1, vertexCategoryKey, 5) # content
                    vst = Vertex.ByCoordinates(Vertex.X(vst, mantissa=mantissa)+(tolerance*100), Vertex.Y(vst, mantissa=mantissa)+(tolerance*100), Vertex.Z(vst, mantissa=mantissa)+(tolerance*100))
                    if storeBREP:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                        d3 = mergeDictionaries2([d1, d2])
                        vst = Topology.SetDictionary(vst, d3, silent=True)
                    else:
                        vst = Topology.SetDictionary(vst, d1, silent=True)
                    vertices.append(vst)
                    tempe = Edge.ByStartVertexEndVertex(topology, vst, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["To_Contents", 5])
                    tempe = Topology.SetDictionary(tempe, tempd, silent=True)
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
                        vop = Topology.InternalVertex(outpost, tolerance=tolerance)
                    else:
                        vop = Topology.CenterOfMass(outpost)
                    tempe = Edge.ByStartVertexEndVertex(topology, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship", edgeCategoryKey],["To_Outposts", 6])
                    tempd = Topology.SetDictionary(tempe, tempd, silent=True)
                    edges.append(tempe)
            
            return [vertices, edges]

        if not Topology.IsInstance(topology, "Topology"):
            print("Graph.ByTopology - Error: The input topology is not a valid topology. Returning None.")
            return None
        c_cellComplexes = Topology.CellComplexes(topology, silent=True)
        c_cells = Topology.Cells(topology, silent=True)
        c_shells = Topology.Shells(topology, silent=True)
        c_faces = Topology.Faces(topology, silent=True)
        c_wires = Topology.Wires(topology, silent=True)
        c_edges = Topology.Edges(topology, silent=True)
        c_vertices = Topology.Vertices(topology, silent=True)
        others = c_cellComplexes+c_cells+c_shells+c_faces+c_wires+c_edges+c_vertices
        item = [topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBREP, tolerance]
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
            others = others+c_cellComplexes+c_cells+c_shells+c_faces+c_wires+c_edges+c_vertices
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
    def Choice(graph, method: str = "vertex", weightKey="length", normalize: bool = False, nxCompatible: bool = False, key: str = "choice", colorKey="ch_color", colorScale="viridis", mantissa: int = 6, tolerance: float = 0.001, silent: bool = False):
        """
            This is an alias method for Graph.BetweenessCentrality. Returns the choice (Betweeness Centrality) of the input graph. The order of the returned list is the same as the order of vertices/edges. See https://en.wikipedia.org/wiki/Betweenness_centrality.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        method : str , optional
            The method of computing the betweenness centrality. The options are "vertex" or "edge". Default is "vertex".
        weightKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the weightKey string will be aggregated to calculate
            the shortest path. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead.
            This is used in weighted graphs. if weightKey is set to "Length" or "Distance", the length of the edge will be used as its weight.
        normalize : bool , optional
            If set to True, the values are normalized to be in the range 0 to 1. Otherwise they are not. Default is False.
        nxCompatible : bool , optional
            If set to True, and normalize input parameter is also set to True, the values are set to be identical to NetworkX values. Otherwise, they are normalized between 0 and 1. Default is False.
        key : str , optional
            The desired dictionary key under which to store the betweenness centrality score. Default is "betweenness_centrality".
        colorKey : str , optional
            The desired dictionary key under which to store the betweenness centrality color. Default is "betweenness_centrality".
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
            In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            The choice (betweenness centrality) of the input list of vertices within the input graph. The values are in the range 0 to 1.

        """
        return Graph.BetweennessCentrality(graph,
                                           method=method,
                                           weightKey=weightKey,
                                           normalize=normalize,
                                           nxCompatible=nxCompatible,
                                           key=key,
                                           colorKey=colorKey,
                                           colorScale=colorScale,
                                           mantissa=mantissa,
                                           tolerance=tolerance,
                                           silent=silent)


    @staticmethod
    def ChromaticNumber(graph, maxColors: int = 3, silent: bool = False):
        """
        Returns the chromatic number of the input graph. See https://en.wikipedia.org/wiki/Graph_coloring.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        maxColors : int , optional
            The desired maximum number of colors to test against. Default is 3.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
       
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
            The existing dictionary key to use to read any pre-existing color information. Default is "color".
        key : str , optional
            The new dictionary key to use to write out new color information. Default is "color".
        maxColors : int , optional
            The desired maximum number of colors to use. If set to None, the chromatic number of the graph is used. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def Compare(graphA, graphB,
                weightAccessibilityCentrality: float = 0.0,
                weightAttributes: float = 0.0,
                weightGeometry: float = 0.0,
                weightBetwennessCentrality: float = 0.0,
                weightClosenessCentrality: float = 0.0,
                weightDegreeCentrality: float = 0.0,
                weightDiameter: float = 0.0,
                weightEigenVectorCentrality: float = 0.0,
                weightGlobalClusteringCoefficient: float = 0.0,
                weightHopper: float = 0.0,
                weightJaccard: float = 0.0,
                weightPageRank: float = 0.0,
                weightStructure: float = 0.0,
                weightWeisfeilerLehman: float = 0.0,
                vertexIDKey: str = "id",
                edgeWeightKey: str = None,
                wlKey: str = None,
                hopperKey: str = None,
                iterations: int = 2,
                maxHops: int = 2,
                decay: float = 0.5,
                mantissa: int = 6,
                silent: bool = False):
        """
        Compares two graphs and returns a similarity score based on attributres, geometry, metrics, structure, 
        the Weisfeiler-Lehman graph kernel. See https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test
        , and the weight Jaccard Similarity. See https://www.statology.org/jaccard-similarity/

        Parameters
        ----------
        graphA : topologic Graph
            The first input graph.
        graphB : topologic Graph
            The second input graph.
        weightAccessibilityCentrality : float , optional
            The desired weight for degree accessibility similarity (graph-level and node-level). Default is 0.0.
        weightAttributes : float , optional
            The desired weight for attribute similarity (dictionary key overlap at vertices). Default is 0.0.
        weightBetwennessCentrality : float , optional
            The desired weight for betweenness centrality similarity (graph-level and node-level). Default is 0.0.
        weightClosenessCentrality : float , optional
            The desired weight for closeness centrality similarity (graph-level and node-level). Default is 0.0.
        weightDegreeCentrality : float , optional
            The desired weight for degree centrality similarity (graph-level and node-level). Default is 0.0.
        weightDiameter : float , optional
            The desired weight for diameter similarity (graph-level and node-level). Default is 0.0.
        weightEigenVectorCentrality : float , optional
            The desired weight for eigenvector centrality similarity (graph-level and node-level). Default is 0.0.
        weightGeometry : float , optional
            The desired weight for geometric similarity (vertex positions). Default is 0.0.
        weightGlobalClusteringCoefficient : float , optional
            The desired weight for global clustering coefficient similarity (graph-level and node-level). Default is 0.0.
        weightHopper : float , optional
            The desired weight for Hopper kernel similarity. Default is 0.0.
        weightJaccard: float , optional
            The desired weight for the Weighted Jaccard similarity. Default is 0.0.
        weightPageRank : float , optional
            The desired weight for PageRank similarity (graph-level and node-level). Default is 0.0.
        weightStructure : float , optional
            The desired weight for structural similarity (number of vertices and edges). Default is 0.0.
        weightWeisfeilerLehman : float , optional
            The desired weight for Weisfeiler-Lehman kernel similarity (iterative label propagation). Default is 0.0.
        vertexIDKey: str , optional
            The dictionary key under which to find the unique vertex ID. Default is "id".
        edgeWeightKey: str , optional
            The dictionary key under which to find the weight of the edge for weighted graphs.
            If this parameter is specified as "length" or "distance" then the length of the edge is used as its weight.
            The default is None which means all edges are treated as if they have a weight of 1.
        wlKey: str , optional
            The vertex key to use for the Weifeiler-Lehman kernel initial labels. Default is None which means it will use vertex degree as an initial label.
        hopperKey: str , optional
            The vertex key to use for the Hopper kernel to derive node features. Default is None which means it will use vertex degree as an initial label.
        iterations : int , optional
            WL kernel-specific parameter: The desired number of Weisfeiler-Lehman kernel iterations. Default is 2.
        maxHops : int , optional
            Hopper kernel-specific parameter: The maximum shortest-path hop distance to consider. Default is 3.
        decay : float , optional
            Hopper kernel-specific parameter: A per-hop geometric decay factor in the range (0, 1]. Default is 0.5.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary of similarity scores between 0 (completely dissimilar) and 1 (identical), based on weighted components.
            The keys in the dictionary are:
            "accessibility_centrality"
            "attribute"
            "betwenness_centrality"
            "closeness_centrality"
            "degree_centrality"
            "eigenvector_centrality"
            "geometry"
            "global_clustering_coefficient"
            "hopper"
            "jaccard"
            "pagerank"
            "structure"
            "weisfeiler_lehman"
            "overall"
        """
        
        import hashlib
        from collections import Counter
        from topologicpy.Graph import Graph
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def attribute_similarity(graphA, graphB, mantissa=6):
            v1 = Graph.Vertices(graphA)
            v2 = Graph.Vertices(graphB)
            if len(v1) != len(v2) or len(v1) == 0:
                return 0

            match_score = 0
            for a, b in zip(v1, v2):
                dict_a = Topology.Dictionary(a)
                dict_b = Topology.Dictionary(b)

                keys_a = set(Dictionary.Keys(dict_a)) if dict_a else set()
                keys_b = set(Dictionary.Keys(dict_b)) if dict_b else set()

                if not keys_a and not keys_b:
                    match_score += 1
                else:
                    intersection = len(keys_a & keys_b)
                    union = len(keys_a | keys_b)
                    match_score += intersection / union if union > 0 else 0

            return round(match_score / len(v1), mantissa)
        
        def geometry_similarity(graphA, graphB, mantissa=6):
            v1 = Graph.Vertices(graphA)
            v2 = Graph.Vertices(graphB)
            if len(v1) != len(v2) or len(v1) == 0:
                return 0

            total_dist = 0
            for a, b in zip(v1, v2):  # assumes same order
                p1 = Vertex.Coordinates(a, mantissa=mantissa)
                p2 = Vertex.Coordinates(b, mantissa=mantissa)
                total_dist += sum((i - j) ** 2 for i, j in zip(p1, p2)) ** 0.5
            avg_dist = total_dist / len(v1)
            return round(1 / (1 + avg_dist), mantissa) # Inverse average distance
        
        def weighted_jaccard_similarity(graph1, graph2, vertexIDKey="id", edgeWeightKey=None, mantissa=6):
            """
            Computes weighted Jaccard similarity between two graphs by comparing their edge weights.

            Parameters
            ----------
            graph1 : topologic Graph
                First graph.
            graph2 : topologic Graph
                Second graph.
            vertexIDKey: str , optional
                The dictionary key under which to find the unique vertex ID. Default is "id".
            edgeWeightKey: str , optional
                The dictionary key under which to find the weight of the edge for weighted graphs.
                If this parameter is specified as "length" or "distance" then the length of the edge is used as its weight.
                The default is None which means all edges are treated as if they have a weight of 1.
            mantissa : int , optional
                The number of decimal places to round the result to. Default is 6.

            Returns
            -------
            float
                Similarity score between 0 and 1.
            """
            from topologicpy.Vertex import Vertex
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            from topologicpy.Edge import Edge


            def edge_id(edge, vertexIDKey="id", mantissa=6):
                v1 = Edge.StartVertex(edge)
                v2 = Edge.EndVertex(edge)
                d1 = Topology.Dictionary(v1)
                d2 = Topology.Dictionary(v2)
                v1_id = Dictionary.ValueAtKey(d1, vertexIDKey) if d1 and Dictionary.ValueAtKey(d1, vertexIDKey) is not None else str(sorted(Vertex.Coordinates(v1, mantissa=mantissa)))
                v2_id = Dictionary.ValueAtKey(d2, vertexIDKey) if d2 and Dictionary.ValueAtKey(d2, vertexIDKey) is not None else str(sorted(Vertex.Coordinates(v2, mantissa=mantissa)))

                return tuple(sorted(tuple([v1_id, v2_id])))

            def edge_weights(graph, edgeWeightKey=None, mantissa=6):
                weights = {}
                for edge in Graph.Edges(graph):
                    if edgeWeightKey == None:
                        weight = 1
                    elif edgeWeightKey.lower() == "length" or edgeWeightKey.lower() == "distance":
                        weight = Edge.Length(edge)
                    else:
                        d = Topology.Dictionary(edge)
                        weight = Dictionary.ValueAtKey(d, edgeWeightKey) if d and Dictionary.ValueAtKey(d, edgeWeightKey) is not None else 1.0
                    eid = edge_id(edge, vertexIDKey=vertexIDKey, mantissa=mantissa)
                    weights[eid] = weight
                return weights
            
            w1 = edge_weights(graph1, edgeWeightKey=edgeWeightKey)
            w2 = edge_weights(graph2, edgeWeightKey=edgeWeightKey)
            keys = set(w1.keys()) | set(w2.keys())

            numerator = sum(min(w1.get(k, 0), w2.get(k, 0)) for k in keys)
            denominator = sum(max(w1.get(k, 0), w2.get(k, 0)) for k in keys)

            return numerator / denominator if denominator > 0 else 0.0
        
        def safe_mean(lst):
                return sum(lst)/len(lst) if lst else 0
        
        def accessibility_centrality_similarity(graphA, graphB, mantissa=6):
            v1 = safe_mean(Graph.AccessibilityCentrality(graphA))
            v2 = safe_mean(Graph.AccessibilityCentrality(graphB))
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def betweenness_centrality_similarity(graphA, graphB, mantissa=6):
            v1 = safe_mean(Graph.BetweennessCentrality(graphA))
            v2 = safe_mean(Graph.BetweennessCentrality(graphB))
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def closeness_centrality_similarity(graphA, graphB, mantissa=6):
            v1 = safe_mean(Graph.ClosenessCentrality(graphA))
            v2 = safe_mean(Graph.ClosenessCentrality(graphB))
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def degree_centrality_similarity(graphA, graphB, mantissa=6):
            v1 = safe_mean(Graph.DegreeCentrality(graphA))
            v2 = safe_mean(Graph.DegreeCentrality(graphB))
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def diameter_similarity(graphA, graphB, mantissa=6):
            v1 = Graph.Diameter(graphA)
            v2 = Graph.Diameter(graphB)
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def eigenvector_centrality_similarity(graphA, graphB, mantissa=6):
            v1 = safe_mean(Graph.EigenVectorCentrality(graphA))
            v2 = safe_mean(Graph.EigenVectorCentrality(graphB))
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def global_clustering_coefficient_similarity(graphA, graphB, mantissa=6):
            v1 = Graph.GlobalClusteringCoefficient(graphA)
            v2 = Graph.GlobalClusteringCoefficient(graphB)
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def hopper_similarity(graphA, graphB, key=None, maxHops=3, decay=0.5, mantissa=6, silent=False):
            score = Graph.HopperKernel(graphA, graphB, key=key, maxHops=maxHops, decay=decay, normalize=True, mantissa=mantissa, silent=silent)
            return score
        
        def pagerank_similarity(graphA, graphB, mantissa=6):
            v1 = safe_mean(Graph.PageRank(graphA))
            v2 = safe_mean(Graph.PageRank(graphB))
            if v1 == 0 and v2 == 0:
                return 1
            diff = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            return round((1 - diff), mantissa)
        
        def structure_similarity(graphA, graphB, mantissa=6):
            v1 = Graph.Vertices(graphA)
            v2 = Graph.Vertices(graphB)
            e1 = Graph.Edges(graphA)
            e2 = Graph.Edges(graphB)

            vertex_score = 1 - abs(len(v1) - len(v2)) / max(len(v1), len(v2), 1)
            edge_score = 1 - abs(len(e1) - len(e2)) / max(len(e1), len(e2), 1)

            return round((vertex_score + edge_score) / 2, mantissa)

        def weisfeiler_lehman_similarity(graphA, graphB, key=None, iterations=2, mantissa=6, silent=True):
            score = Graph.WLKernel(graphA, graphB, key=key, iterations=iterations, normalize=True, mantissa=mantissa, silent=silent)
            return score
        
        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.Compare - Error: The graphA input parameter is not a valid topologic graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.Compare - Error: The graphB input parameter is not a valid topologic graph. Returning None.")
            return 
        
        total_weight = sum([weightAccessibilityCentrality,
                            weightAttributes,
                            weightGeometry,
                            weightBetwennessCentrality,
                            weightClosenessCentrality,
                            weightDegreeCentrality,
                            weightDiameter,
                            weightEigenVectorCentrality,
                            weightGlobalClusteringCoefficient,
                            weightHopper,
                            weightPageRank,
                            weightStructure,
                            weightWeisfeilerLehman,
                            weightJaccard])
        accessibility_centrality_score = accessibility_centrality_similarity(graphA, graphB, mantissa=6) if weightAccessibilityCentrality else 0
        attribute_score = attribute_similarity(graphA, graphB, mantissa=6) if weightAttributes else 0
        betweenness_centrality_score = betweenness_centrality_similarity(graphA, graphB, mantissa=6) if weightBetwennessCentrality else 0
        closeness_centrality_score = closeness_centrality_similarity(graphA, graphB, mantissa=6) if weightClosenessCentrality else 0
        degree_centrality_score = degree_centrality_similarity(graphA, graphB, mantissa=6) if weightDegreeCentrality else 0
        diameter_score = diameter_similarity(graphA, graphB, mantissa=6) if weightDiameter else 0
        eigenvector_centrality_score = eigenvector_centrality_similarity(graphA, graphB, mantissa=6) if weightEigenVectorCentrality else 0
        geometry_score = geometry_similarity(graphA, graphB, mantissa=6) if weightGeometry else 0
        global_clustering_coefficient_score = global_clustering_coefficient_similarity(graphA, graphB, mantissa=6) if weightGlobalClusteringCoefficient else 0
        hopper_score = hopper_similarity(graphA, graphB, key=hopperKey, maxHops=maxHops, decay=decay, mantissa=6, silent=silent) if weightHopper else 0
        jaccard_score = weighted_jaccard_similarity(graphA, graphB, vertexIDKey=vertexIDKey, edgeWeightKey=edgeWeightKey, mantissa=6) if weightJaccard else 0
        pagerank_score = pagerank_similarity(graphA, graphB, mantissa=6) if weightPageRank else 0
        structure_score = structure_similarity(graphA, graphB, mantissa=6) if weightStructure else 0
        weisfeiler_lehman_score = weisfeiler_lehman_similarity(graphA, graphB, key=wlKey, iterations=iterations, mantissa=6, silent=silent) if weightWeisfeilerLehman else 0

        weighted_sum = (
            accessibility_centrality_score * weightAccessibilityCentrality +
            attribute_score * weightAttributes +
            betweenness_centrality_score * weightBetwennessCentrality +
            closeness_centrality_score * weightClosenessCentrality +
            degree_centrality_score * weightDegreeCentrality +
            diameter_score * weightDiameter +
            eigenvector_centrality_score * weightEigenVectorCentrality +
            geometry_score * weightGeometry +
            global_clustering_coefficient_score * weightGlobalClusteringCoefficient +
            hopper_score * weightHopper +
            jaccard_score * weightJaccard +
            pagerank_score * weightPageRank +
            structure_score * weightStructure +
            weisfeiler_lehman_score * weightWeisfeilerLehman
        )

        if total_weight <= 0:
            overall_score = 0
        else:
            overall_score = weighted_sum / total_weight
        
        return  {
                 "accessibility_centrality": round(accessibility_centrality_score, mantissa),
                 "attribute": round(attribute_score, mantissa),
                 "betwenness_centrality": round(betweenness_centrality_score, mantissa),
                 "closeness_centrality": round(closeness_centrality_score, mantissa),
                 "degree_centrality": round(degree_centrality_score, mantissa),
                 "eigenvector_centrality": round(eigenvector_centrality_score, mantissa),
                 "geometry": round(geometry_score, mantissa),
                 "global_clustering_coefficient": round(global_clustering_coefficient_score, mantissa),
                 "hopper": round(hopper_score, mantissa),
                 "jaccard": round(jaccard_score, mantissa),
                 "pagerank": round(pagerank_score, mantissa),
                 "structure": round(structure_score, mantissa),
                 "weisfeiler_lehman": round(weisfeiler_lehman_score, mantissa),
                 "overall": round(overall_score, mantissa)
                }

    @staticmethod
    def Complement(graph, tolerance=0.0001, silent=False):
        """
        Creates the complement graph of the input graph. See https://en.wikipedia.org/wiki/Complement_graph

        Parameters
        ----------
        graph : topologicpy.Graph
            The input topologic graph.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Graph
            The created complement topologic graph.

        """
        def complement_graph(adj_dict):
            """
            Creates the complement graph from an input adjacency dictionary.

            Parameters:
                adj_dict (dict): The adjacency dictionary where keys are nodes and 
                                values are lists of connected nodes.

            Returns:
                list of tuples: A list of edge index tuples representing the complement graph.
            """
            # Get all nodes in the graph
            nodes = list(adj_dict.keys())
            # Initialize a set to store edges of the complement graph
            complement_edges = set()
            # Convert adjacency dictionary to a set of existing edges
            existing_edges = set()
            for node, neighbors in adj_dict.items():
                for neighbor in neighbors:
                    # Add the edge as an ordered tuple to ensure no duplicates
                    existing_edges.add(tuple(sorted((node, neighbor))))
            # Generate all possible edges and check if they exist in the original graph
            for i, node1 in enumerate(nodes):
                for j in range(i + 1, len(nodes)):
                    node2 = nodes[j]
                    edge = tuple(sorted((node1, node2)))
                    # Add the edge if it's not in the original graph
                    if edge not in existing_edges:
                        complement_edges.add(edge)
            # Return the complement edges as a sorted list of tuples
            return sorted(complement_edges)
    
        from topologicpy.Graph import Graph
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.Complement - Error: The input graph parameter is not a valid topologic graph. Returning None.")
            return None
        adj_dict = Graph.AdjacencyDictionary(graph)
        py_edges = complement_graph(adj_dict)
        vertices = Graph.Vertices(graph)
        adjusted_vertices = Vertex.Separate(vertices, minDistance=tolerance)
        edges = []
        for py_edge in py_edges:
            start, end = py_edge
            sv = adjusted_vertices[int(start)]
            ev = adjusted_vertices[int(end)]
            edge = Edge.ByVertices(sv, ev, tolerance=tolerance, silent=silent)
            if Topology.IsInstance(edge, "edge"):
                edges.append(edge)    
        return_graph = Graph.ByVerticesEdges(adjusted_vertices, edges)
        return return_graph

    @staticmethod
    def Complete(graph, silent: bool = False):
        """
        Completes the graph by conneting unconnected vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Graph
            the completed graph
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.ConnectedComponents - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)
        visited = set()
        new_edges = []
        for sv in vertices:
            for ev in vertices:
                if sv != ev and not (sv, ev) in visited:
                    visited.add((sv, ev))
                    visited.add((ev,sv))
                    edge = Graph.Edge(graph, sv, ev)
                    if edge == None:
                        new_edges.append(Edge.ByVertices(sv, ev))
        edges += new_edges
        return Graph.ByVerticesEdges(vertices, edges)

    @staticmethod
    def ConnectedComponents(graph, key: str = "component", tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the connected components (islands) of the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        key : str , optional
            The vertex and edge dictionary key under which to store the component number. Default is "component".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of connected components (island graphs).
            The list is sorted by the number of vertices in each component (from highest to lowest).

        """
        def find_connected_components(adjacency_dict):
            visited = set()
            components = []

            for vertex_id in adjacency_dict:
                if vertex_id not in visited:
                    # Perform DFS using a stack
                    stack = [vertex_id]
                    current_island = set()
                    
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            current_island.add(current)
                            stack.extend(set(adjacency_dict[current]) - visited)

                    components.append(current_island)

            return components
        
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.ConnectedComponents - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        labelKey = "__label__"
        lengths = [] #List of lengths to sort the list of components by number of their vertices
        vertices = Graph.Vertices(graph)
        vertex_map = {}
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, labelKey, i)
            v = Topology.SetDictionary(v, d)
            vertex_map[i] = v
        g_dict = Graph.AdjacencyDictionary(graph, vertexLabelKey=labelKey)
        components = find_connected_components(g_dict)
        return_components = []
        for i, component in enumerate(components):
            i_verts = []
            for v in component:
                vert = vertex_map[v]
                d = Topology.Dictionary(vert)
                d = Dictionary.RemoveKey(d, labelKey)
                d = Dictionary.SetValueAtKey(d, key, i+1)
                vert = Topology.SetDictionary(vert, d)
                i_verts.append(vert)
            if len(i_verts) > 0:
                i_edges = Graph.Edges(graph, i_verts)
                for i_edge in i_edges:
                    d = Topology.Dictionary(i_edge)
                    d = Dictionary.SetValueAtKey(d, key, i+1)
                    i_edge = Topology.SetDictionary(i_edge, d)
                lengths.append(len(i_verts))
                g_component = Graph.ByVerticesEdges(i_verts, i_edges)
                return_components.append(g_component)
        if len(return_components) > 0:
            return_components = Helper.Sort(return_components, lengths)
            return_components.reverse()
        return return_components

    @staticmethod
    def ContractEdge(graph, edge, vertex=None, tolerance: float = 0.0001, silent: bool = False):
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
            The vertex to replace the contracted edge. If set to None, the centroid of the edge is chosen. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            if not silent:
                print("Graph.ContractEdge - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
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
        graph = Graph.RemoveEdge(graph, edge, tolerance=tolerance, silent=silent)
        graph = Graph.AddVertex(graph, vertex, tolerance=tolerance, silent=silent)
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
        graph = Graph.RemoveVertex(graph,sv)
        graph = Graph.RemoveVertex(graph,ev)
        return graph
    



    @staticmethod
    def ClosenessCentrality(
        graph,
        weightKey: str = "length",
        normalize: bool = False,
        nxCompatible: bool = True,
        key: str = "closeness_centrality",
        colorKey: str = "cc_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Optimized closeness centrality:
        - Avoids NetworkX and costly per-vertex Topologic calls.
        - Builds integer-index adjacency once from edges (undirected).
        - Unweighted: multi-source BFS (one per node).
        - Weighted: Dijkstra per node (heapq), or SciPy csgraph if available.
        - Supports 'wf_improved' scaling (nxCompatible) and optional normalization.
        """
        from collections import deque
        import math

        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        from topologicpy.Helper import Helper
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        # NOTE: We are inside Graph.*, so Graph.<...> methods are available.

        # Validate graph
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.ClosenessCentrality - Error: The input is not a valid Graph. Returning None.")
            return None

        vertices = Graph.Vertices(graph)
        n = len(vertices)
        if n == 0:
            if not silent:
                print("Graph.ClosenessCentrality - Warning: Graph has no vertices. Returning [].")
            return []

        # Stable vertex key (prefer an 'id' in the vertex dictionary; else rounded coords)
        def vkey(v, r=9):
            d = Topology.Dictionary(v)
            vid = Dictionary.ValueAtKey(d, "id")
            if vid is not None:
                return ("id", vid)
            return ("xyz", round(Vertex.X(v), r), round(Vertex.Y(v), r), round(Vertex.Z(v), r))

        idx_of = {vkey(v): i for i, v in enumerate(vertices)}

        # Normalize weight key
        distance_attr = None
        if isinstance(weightKey, str) and weightKey:
            wl = weightKey.lower()
            if ("len" in wl) or ("dis" in wl):
                weightKey = "length"
            distance_attr = weightKey  # may be "length" or a custom key

        # Build undirected adjacency with minimal weights per edge
        # Use dict-of-dict to collapse multi-edges to minimal weight
        adj = [dict() for _ in range(n)]  # adj[i][j] = weight
        edges = Graph.Edges(graph)

        def edge_weight(e):
            if distance_attr == "length":
                try:
                    return float(Edge.Length(e))
                except Exception:
                    return 1.0
            elif distance_attr:
                try:
                    d = Topology.Dictionary(e)
                    w = Dictionary.ValueAtKey(d, distance_attr)
                    return float(w) if (w is not None) else 1.0
                except Exception:
                    return 1.0
            else:
                return 1.0

        for e in edges:
            try:
                u = Edge.StartVertex(e)
                v = Edge.EndVertex(e)
            except Exception:
                # Fallback in odd cases
                continue
            iu = idx_of.get(vkey(u))
            iv = idx_of.get(vkey(v))
            if iu is None or iv is None or iu == iv:
                continue
            w = edge_weight(e)
            # Keep minimal weight if duplicates
            prev = adj[iu].get(iv)
            if (prev is None) or (w < prev):
                adj[iu][iv] = w
                adj[iv][iu] = w

        # Detect weighted vs unweighted
        weighted = False
        for i in range(n):
            if any(abs(w - 1.0) > 1e-12 for w in adj[i].values()):
                weighted = True
                break

        INF = float("inf")

        # ---- shortest paths helpers ----
        def bfs_sum(i):
            """Sum of unweighted shortest path distances from i; returns (tot, reachable)."""
            dist = [-1] * n
            q = deque([i])
            dist[i] = 0
            reachable = 1
            tot = 0
            pop = q.popleft; push = q.append
            while q:
                u = pop()
                du = dist[u]
                for v in adj[u].keys():
                    if dist[v] == -1:
                        dist[v] = du + 1
                        reachable += 1
                        tot += dist[v]
                        push(v)
            return float(tot), reachable

        def dijkstra_sum(i):
            """Sum of weighted shortest path distances from i; returns (tot, reachable)."""
            import heapq
            dist = [INF] * n
            dist[i] = 0.0
            hq = [(0.0, i)]
            push = heapq.heappush; pop = heapq.heappop
            while hq:
                du, u = pop(hq)
                if du > dist[u]:
                    continue
                for v, w in adj[u].items():
                    nd = du + w
                    if nd < dist[v]:
                        dist[v] = nd
                        push(hq, (nd, v))
            # Exclude self (0.0) and unreachable (INF)
            reachable = 0
            tot = 0.0
            for d in dist:
                if d < INF:
                    reachable += 1
                    tot += d
            # subtract self-distance
            tot -= 0.0
            return float(tot), reachable

        # SciPy acceleration if weighted and available
        use_scipy = False
        if weighted:
            try:
                import numpy as np
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import dijkstra as sp_dijkstra
                use_scipy = True
                # Build CSR once
                rows, cols, data = [], [], []
                for i in range(n):
                    for j, w in adj[i].items():
                        rows.append(i); cols.append(j); data.append(float(w))
                if len(data) == 0:
                    use_scipy = False  # empty graph; fall back
                else:
                    A = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(n, n))
            except Exception:
                use_scipy = False

        # ---- centrality computation ----
        values = [0.0] * n
        if n == 1:
            values[0] = 0.0
        else:
            if not weighted:
                for i in range(n):
                    tot, reachable = bfs_sum(i)
                    s = max(reachable - 1, 0)
                    if tot > 0.0:
                        if nxCompatible:
                            # Wasserman–Faust improved scaling for disconnected graphs
                            values[i] = (s / (n - 1)) * (s / tot)
                        else:
                            values[i] = s / tot
                    else:
                        values[i] = 0.0
            else:
                if use_scipy:
                    # All-pairs from SciPy (fast)
                    import numpy as np
                    D = sp_dijkstra(A, directed=False, return_predecessors=False)
                    for i in range(n):
                        di = D[i]
                        finite = di[np.isfinite(di)]
                        # di includes self at 0; reachable count is len(finite)
                        reachable = int(finite.size)
                        s = max(reachable - 1, 0)
                        tot = float(finite.sum())  # includes self=0
                        if s > 0:
                            if nxCompatible:
                                values[i] = (s / (n - 1)) * (s / tot)
                            else:
                                values[i] = s / tot
                        else:
                            values[i] = 0.0
                else:
                    # Per-source Dijkstra
                    for i in range(n):
                        tot, reachable = dijkstra_sum(i)
                        s = max(reachable - 1, 0)
                        if tot > 0.0:
                            if nxCompatible:
                                values[i] = (s / (n - 1)) * (s / tot)
                            else:
                                values[i] = s / tot
                        else:
                            values[i] = 0.0

        # Optional normalization, round once
        out_vals = Helper.Normalize(values) if normalize else values
        if mantissa is not None and mantissa >= 0:
            out_vals = [round(v, mantissa) for v in out_vals]

        # Color mapping range (use displayed numbers)
        if out_vals:
            min_v, max_v = min(out_vals), max(out_vals)
        else:
            min_v, max_v = 0.0, 1.0
        if abs(max_v - min_v) < tolerance:
            max_v = min_v + tolerance

        # Annotate vertices
        for i, value in enumerate(out_vals):
            d = Topology.Dictionary(vertices[i])
            color_hex = Color.AnyToHex(
                Color.ByValueInRange(value, minValue=min_v, maxValue=max_v, colorScale=colorScale)
            )
            d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color_hex])
            vertices[i] = Topology.SetDictionary(vertices[i], d)

        return out_vals


    # @staticmethod
    # def ClosenessCentrality_old(
    #     graph,
    #     weightKey: str = "length",
    #     normalize: bool = False,
    #     nxCompatible: bool = True,
    #     key: str = "closeness_centrality",
    #     colorKey: str = "cc_color",
    #     colorScale: str = "viridis",
    #     mantissa: int = 6,
    #     tolerance: float = 0.0001,
    #     silent: bool = False
    # ):
    #     """
    #     Returns the closeness centrality of the input graph. The order of the returned
    #     list matches the order of Graph.Vertices(graph).
    #     See: https://en.wikipedia.org/wiki/Closeness_centrality

    #     Parameters
    #     ----------
    #     graph : topologic_core.Graph
    #         The input graph.
    #     weightKey : str , optional
    #         If specified, this edge attribute will be used as the distance weight when
    #         computing shortest paths. If set to a name containing "Length" or "Distance",
    #         it will be mapped to "length".
    #         Note: Graph.NetworkXGraph automatically provides a "length" attribute on all edges.
    #     normalize : bool , optional
    #         If True, the returned values are rescaled to [0, 1]. Otherwise raw values
    #         from NetworkX (optionally using the improved formula) are returned.
    #     nxCompatible : bool , optional
    #         If True, use NetworkX's wf_improved scaling (Wasserman and Faust).
    #         For single-component graphs it matches the original formula.
    #     key : str , optional
    #         The dictionary key under which to store the closeness centrality score.
    #     colorKey : str , optional
    #         The dictionary key under which to store a color derived from the score.
    #     colorScale : str , optional
    #         Plotly color scale name (e.g., "viridis", "plasma").
    #     mantissa : int , optional
    #         The number of decimal places to round the result to. Default is 6.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.
    #     silent : bool , optional
    #         If set to True, error and warning messages are suppressed. Default is False.

    #     Returns
    #     -------
    #     list[float]
    #         Closeness centrality values for vertices in the same order as Graph.Vertices(graph).
    #     """
    #     import warnings
    #     try:
    #         import networkx as nx
    #     except Exception as e:
    #         warnings.warn(
    #             f"Graph.ClosenessCentrality - Error: networkx is required but not installed ({e}). Returning None."
    #         )
    #         return None

    #     from topologicpy.Dictionary import Dictionary
    #     from topologicpy.Color import Color
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Helper import Helper

    #     # Topology.IsInstance is case-insensitive, so a single call is sufficient.
    #     if not Topology.IsInstance(graph, "graph"):
    #         if not silent:
    #             print("Graph.ClosenessCentrality - Error: The input is not a valid Graph. Returning None.")
    #         return None
    #     vertices = Graph.Vertices(graph)
    #     if len(vertices) == 0:
    #         if not silent:
    #             print("Graph.ClosenessCentrality - Warning: Graph has no vertices. Returning [].")
    #         return []

    #     # Normalize the weight key semantics
    #     distance_attr = None
    #     if isinstance(weightKey, str) and weightKey:
    #         if ("len" in weightKey.lower()) or ("dis" in weightKey.lower()):
    #             weightKey = "length"
    #         distance_attr = weightKey

    #     # Build the NX graph
    #     nx_graph = Graph.NetworkXGraph(graph)

    #     # Graph.NetworkXGraph automatically adds "length" to all edges.
    #     # So if distance_attr == "length", we trust it and skip per-edge checks.
    #     if distance_attr and distance_attr != "length":
    #         # For any non-"length" custom attribute, verify presence; else fall back unweighted.
    #         attr_missing = any(
    #             (distance_attr not in data) or (data[distance_attr] is None)
    #             for _, _, data in nx_graph.edges(data=True)
    #         )
    #         if attr_missing:
    #             if not silent:
    #                 print("Graph.ClosenessCentrality - Warning: The specified edge attribute was not found on all edges. Falling back to unweighted closeness.")
    #             distance_arg = None
    #         else:
    #             distance_arg = distance_attr
    #     else:
    #         # Use "length" directly or unweighted if distance_attr is falsy.
    #         distance_arg = distance_attr if distance_attr else None

    #     # Compute centrality (dict keyed by NetworkX nodes)
    #     try:
    #         cc_dict = nx.closeness_centrality(nx_graph, distance=distance_arg, wf_improved=nxCompatible)
    #     except Exception as e:
    #         if not silent:
    #             print(f"Graph.ClosenessCentrality - Error: NetworkX failed to compute centrality ({e}). Returning None.")
    #         return None

    #     # NetworkX vertex ids are in the same numerice order as the list of vertices starting from 0.
    #     raw_values = []
    #     for i, v in enumerate(vertices):
    #         try:
    #             raw_values.append(float(cc_dict.get(i, 0.0)))
    #         except Exception:
    #             if not silent:
    #                 print(f,"Graph.ClosenessCentrality - Warning: Could not retrieve score for vertex {i}. Assigning a Zero (0).")
    #             raw_values.append(0.0)

    #     # Optional normalization ONLY once, then rounding once at the end
    #     values_for_return = Helper.Normalize(raw_values) if normalize else raw_values

    #     # Values for color scaling should reflect the displayed numbers
    #     color_values = values_for_return

    #     # Single rounding at the end for return values
    #     if mantissa is not None and mantissa >= 0:
    #         values_for_return = [round(v, mantissa) for v in values_for_return]

    #     # Prepare color mapping range, guarding equal-range case
    #     if color_values:
    #         min_value = min(color_values)
    #         max_value = max(color_values)
    #     else:
    #         min_value, max_value = 0.0, 1.0

    #     if abs(max_value - min_value) < tolerance:
    #         max_value = min_value + tolerance

    #     # Annotate vertices with score and color
    #     for i, value in enumerate(color_values):
    #         d = Topology.Dictionary(vertices[i])
    #         color_hex = Color.AnyToHex(
    #             Color.ByValueInRange(value, minValue=min_value, maxValue=max_value, colorScale=colorScale)
    #         )
    #         d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [values_for_return[i], color_hex])
    #         vertices[i] = Topology.SetDictionary(vertices[i], d)

    #     return values_for_return

    @staticmethod
    def Community(graph, key: str = "partition", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Computes the best community partition of the input graph based on the Louvain method. See https://en.wikipedia.org/wiki/Louvain_method.

        Parameters
        ----------
        graph : topologicp.Graph
            The input topologic graph.
        key : str , optional
            The dictionary key under which to store the partition number. Default is "partition".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.
        Returns
        -------
        topologicpy.Graph
            The partitioned topologic graph.

        """
        if not silent:
            print("Graph.Community - Warning: This method is deprectated. Please use Graph.CommunityPartition instead.")
        return Graph.CommunityPartition(graph=graph, key=key, mantissa=mantissa, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def CommunityPartition(graph, key: str = "partition", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Computes the best community partition of the input graph based on the Louvain method. See https://en.wikipedia.org/wiki/Louvain_method.

        Parameters
        ----------
        graph : topologicp.Graph
            The input topologic graph.
        key : str , optional
            The dictionary key under which to store the partition number. Default is "partition".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.
        Returns
        -------
        topologicpy.Graph
            The partitioned topologic graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import os
        import warnings
        
        try:
            import igraph as ig
        except:
            print("Graph.Community - Installing required pyhon-igraph library.")
            try:
                os.system("pip install python-igraph")
            except:
                os.system("pip install python-igraph --user")
            try:
                import igraph as ig
                print("Graph.Community - python-igraph library installed correctly.")
            except:
                warnings.warn("Graph.Community - Error: Could not import python-igraph. Please install manually.")
        
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.Community - Error: The input graph parameter is not a valid topologic graph. Returning None")
            return None
        
        mesh_data = Graph.MeshData(graph)
        # Create an igraph graph from the edge list
        ig_graph = ig.Graph(edges=mesh_data['edges'])

        # Detect communities using Louvain method
        communities = ig_graph.community_multilevel()

        # Get the list of communities sorted same as vertices
        partition_list = communities.membership
        vertices = Graph.Vertices(graph)
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, key, partition_list[i]+1)
            v = Topology.SetDictionary(v, d)
        edges = Graph.Edges(graph)
        if not edges == None:
            for edge in edges:
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                status_1 = False
                status_2 = False
                partition_1 = 0
                partition_2 = 0
                for i, v in enumerate(vertices):
                    if Vertex.IsCoincident(sv, v, tolerance=tolerance):
                        status_1 = True
                        partition_1 = Dictionary.ValueAtKey(Topology.Dictionary(v), key)
                        break
                for i, v in enumerate(vertices):
                    if Vertex.IsCoincident(ev, v, tolerance=tolerance):
                        status_2 = True
                        partition_2 = Dictionary.ValueAtKey(Topology.Dictionary(v), key)
                        break
                partition = 0
                if status_1 and status_2:
                    if partition_1 == partition_2:
                        partition = partition_1
                d = Topology.Dictionary(edge)
                d = Dictionary.SetValueAtKey(d, key, partition)
                edge = Topology.SetDictionary(edge, d)
        return graph

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
            The desired tolerance. Default is 0.0001.

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
        _ = graph.Connect(verticesA, verticesB, tolerance) # Hook to Core
        return graph
    
    @staticmethod
    def Connectivity(graph, vertices=None, weightKey: str = None, normalize: bool = False, key: str = "connectivity", colorKey: str = "cn_color", colorScale="Viridis", mantissa: int = 6, tolerance = 0.0001, silent = False):
        """
        This is an alias method for Graph.DegreeCentrality. Return the connectivity measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the connectivity of all the vertices in the input graph is computed. See https://www.spacesyntax.online/term/connectivity/.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. Default is None which means all graph vertices are computed.
        normalize : bool , optional
            If set to True, the values are normalized to be in the range 0 to 1. Otherwise they are not. Default is False.
        weightKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the weightKey string will be aggregated to calculate
            the vertex degree. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead.
            This is used in weighted graphs. if weightKey is set to "Length" or "Distance", the length of the edge will be used as its weight.
        key : str , optional
            The dictionary key under which to store the connectivity score. Default is "connectivity".
        colorKey : str , optional
            The desired dictionary key under which to store the connectivity color. Default is "cn_color".
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
            In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The connectivity score of the input list of vertices within the input graph. The values are in the range 0 to 1 if normalized.

        """
        return Graph.DegreeCentrality(graph=graph,
                                      vertices=vertices,
                                      weightKey=weightKey,
                                      normalize=normalize,
                                      key="connectivity",
                                      colorKey=colorKey,
                                      colorScale=colorScale,
                                      mantissa=mantissa,
                                      tolerance=tolerance,
                                      silent=silent)

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
            The desired tolerance. Default is 0.0001.

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
        return graph.ContainsEdge(edge, tolerance) # Hook to Core
    
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
            Ther desired tolerance. Default is 0.0001.

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
        return graph.ContainsVertex(vertex, tolerance) # Hook to Core

    @staticmethod
    def CutVertices(graph, key: str = "cut", silent: bool = False):
        """
        Returns the list of cut vertices in the input graph. See: https://en.wikipedia.org/wiki/Bridge_(graph_theory)

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        key : str , optional
            The vertex dictionary key under which to store the cut status. 0 means the vertex is NOT a cut vertex. 1 means that the vertex IS a cut vertex. Default is "cut".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The list of bridge edges in the input graph.

        """
        import os
        import warnings
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        try:
            import igraph as ig
        except:
            print("Graph.CutVertices - Installing required pyhon-igraph library.")
            try:
                os.system("pip install python-igraph")
            except:
                os.system("pip install python-igraph --user")
            try:
                import igraph as ig
                print("Graph.CutVertices - python-igraph library installed correctly.")
            except:
                warnings.warn("Graph.CutVertices - Error: Could not import python-igraph. Please install manually.")
        
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.CutVertices - Error: The input graph parameter is not a valid topologic graph. Returning None")
            return None
        
        vertices = Graph.Vertices(graph)
        mesh_data = Graph.MeshData(graph)
        graph_edges = mesh_data['edges']
        ig_graph = ig.Graph(edges=graph_edges)
        articulation_points = ig_graph.vs[ig_graph.articulation_points()]
        articulation_points_list = [v.index for v in articulation_points]
        cut_vertices = []
        for i, vertex in enumerate(vertices):
            d = Topology.Dictionary(vertex)
            if i in articulation_points_list:
                d = Dictionary.SetValueAtKey(d, key, 1)
                cut_vertices.append(vertex)
            else:
                d = Dictionary.SetValueAtKey(d, key, 0)
            vertex = Topology.SetDictionary(vertex, d)
                
        return cut_vertices
    
    @staticmethod
    def DegreeCentrality(graph,
                         vertices: list = None,
                         weightKey: str= None,
                         normalize: bool = False,
                         key: str = "degree_centrality",
                         colorKey="dc_color",
                         colorScale="viridis",
                         mantissa: int = 6,
                         tolerance: float = 0.001,
                         silent: bool = False):
        """
        Returns the degree centrality of the input graph. The order of the returned list is the same as the order of vertices. See https://en.wikipedia.org/wiki/Degree_centrality.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        weightKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the weightKey string will be aggregated to calculate
            the vertex degree. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead.
            This is used in weighted graphs. if weightKey is set to "Length" or "Distance", the length of the edge will be used as its weight.
       normalize : bool , optional
            If set to True, the values are normalized to be in the range 0 to 1. Otherwise they are not. Default is False.
        key : str , optional
            The desired dictionary key under which to store the degree centrality score. Default is "degree_centrality".
        colorKey : str , optional
            The desired dictionary key under which to store the degree centrality color. Default is "dc_color".
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
            In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            The degree centrality of the input list of vertices within the input graph. The values are in the range 0 to 1.

        """
        
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        from topologicpy.Color import Color

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.DegreeCentrality - Error: The input graph is not a valid graph. Returning None.")
            return None
        if vertices == None:
            vertices = Graph.Vertices(graph)
        values = [Graph.VertexDegree(graph, v, weightKey=weightKey, mantissa=mantissa, tolerance=tolerance, silent=silent) for v in vertices]
        if normalize == True:
            if mantissa > 0:
                values = [round(v, mantissa) for v in Helper.Normalize(values)]
            else:
                values = Helper.Normalize(values)
            min_value = 0
            max_value = 1
        else:
            min_value = min(values)
            max_value = max(values)
        
        if abs(max_value - min_value) < tolerance:
            max_value = min_value + tolerance

        for i, value in enumerate(values):
            color = Color.AnyToHex(Color.ByValueInRange(value, minValue=min_value, maxValue=max_value, colorScale=colorScale))
            d = Topology.Dictionary(vertices[i])
            d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color])
            v = Topology.SetDictionary(vertices[i], d)
        return values
    
    @staticmethod
    def DegreeMatrix(graph):
        """
        Returns the degree matrix of the input graph. See https://en.wikipedia.org/wiki/Degree_matrix.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        list
            The degree matrix of the input graph.

        """
        import numpy as np
        adj_matrix = Graph.AdjacencyMatrix(graph)
        np_adj_matrix = np.array(adj_matrix)
        degree_matrix = np.diag(np_adj_matrix.sum(axis=1))
        return degree_matrix.tolist()
    
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
        _ = graph.DegreeSequence(sequence) # Hook to Core
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
        return graph.Density() # Hook to Core
    
    @staticmethod
    def Depth(graph, vertex = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Computes the maximum depth of the input graph rooted at the input vertex.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex , optional
            The input root vertex. If not set, the first vertex in the graph is set as the root vertex. Default is None.
        tolerance : float , optional
                The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int
            The calculated maximum depth of the input graph rooted at the input vertex.

        """
        def dfs(node, depth, visited):
            visited.add(Vertex.Index(node, vertices, tolerance=tolerance))
            max_depth = depth
            for neighbor in Graph.AdjacentVertices(graph, node):
                if Vertex.Index(neighbor, vertices, tolerance=tolerance) not in visited:
                    max_depth = max(max_depth, dfs(neighbor, depth + 1, visited))
            return max_depth

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.Depth - Error: The input graph parameter is not a valid graph. Returning None.")
            return 
        
        vertices = Graph.Vertices(graph)
        if vertex == None:
            v_index = 0
        else:
            if not Topology.IsInstance(vertex, "Vertex"):
                if not silent:
                    print("Graph.Depth - Error: The input rootVertex parameter is not a valid vertex. Returning None.")
                return None  
            v_index = Vertex.Index(vertex, vertices, tolerance=tolerance)
            if v_index == None:
                if not silent:
                    print("Graph.Depth - Error: Could not find the input root vertex in the graph's list of vertices. Returning None.")
                return None
        visited = set()
        return dfs(vertex, 1, visited) - 1

    @staticmethod
    def DepthMap(graph, vertices=None, key: str = "depth", type: str = "topological", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Return the depth map of the input list of vertices within the input graph. The returned list contains the total of the topological distances of each vertex to every other vertex in the input graph. The order of the depth map list is the same as the order of the input list of vertices. If no vertices are specified, the depth map of all the vertices in the input graph is computed.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. Default is None.
        key : str , optional
            The dictionary key under which to store the depth score. Default is "depth".
        type : str , optional
            The type of depth distance to calculate. The options are "topological" or "metric". Default is "topological". See https://www.spacesyntax.online/overview-2/analysis-of-spatial-relations/.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def DetachVertex(graph, *vertices, silent: bool = False):
        """
        Detaches the input vertex from its neigboring vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : *topologic_core.Vertex
            The input vertex or list of vertices.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input vertex removed.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.DetachVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertexList = list(vertices)
        vertexList = Helper.Flatten(vertexList)
        vertexList = [v for v in vertexList if Topology.IsInstance(v, "vertex")]
        if len(vertexList) == 0:
            if not silent:
                print("Graph.DetachVertex - Error: The input vertice parameter does not contain any valid vertices. Returning None.")
            return None
        vertexList = [Graph.NearestVertex(graph, v) for v in vertexList]
        edges = Graph.Edges(graph, vertexList)
        for edge in edges:
            graph = Graph.RemoveEdge(graph, edge)
        return graph

    @staticmethod
    def Diameter(graph, silent: bool = False):
        """
        Returns the diameter of the input (unweighted, undirected) graph.

        The diameter is the maximum, over all pairs of vertices, of the length of a
        shortest path between them. If the graph is disconnected, this returns the
        maximum finite eccentricity across connected components and prints a warning
        unless `silent=True`.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int
            The diameter of the input graph, or None if the graph is empty.
        """
        from collections import deque
        from topologicpy.Topology import Topology

        # Basic checks
        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.Diameter - Error: The input graph is not a valid graph. Returning None.")
            return None

        # Build adjacency dictionary (as sets) and force undirected symmetry
        adj_raw = Graph.AdjacencyDictionary(graph, includeWeights=False)
        if not adj_raw:  # empty graph
            if not silent:
                print("Graph.Diameter - Warning: The graph has no vertices. Returning None.")
            return None

        adj = {u: set(neighbors) for u, neighbors in adj_raw.items()}
        # Ensure symmetry (in case the underlying graph stored directed edges)
        for u, nbrs in list(adj.items()):
            for v in nbrs:
                adj.setdefault(v, set()).add(u)
                adj[u].add(v)

        def bfs_eccentricity(start):
            """Return distances map from start and its eccentricity."""
            dist = {start: 0}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            ecc = max(dist.values()) if dist else 0
            return dist, ecc

        diameter = 0
        n = len(adj)
        disconnected = False

        for s in adj:
            dist, ecc = bfs_eccentricity(s)
            if len(dist) < n:
                disconnected = True
            if ecc > diameter:
                diameter = ecc

        if disconnected and not silent:
            print("Graph.Diameter - Warning: The graph is disconnected. Returning the maximum finite diameter across connected components.")

        return diameter

    
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
        return graph.GetDictionary() # Hook to core library
    
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
            The type of depth distance to calculate. The options are "topological" or "metric". Default is "topological". See https://www.spacesyntax.online/overview-2/analysis-of-spatial-relations/.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            The desired tolerance. Default is 0.0001.

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
        return graph.Edge(vertexA, vertexB, tolerance) # Hook to Core
    
    @staticmethod
    def Edges(
        graph,
        vertices: list = None,
        strict: bool = False,
        sortBy: str = None,
        reverse: bool = False,
        silent: bool = False,
        tolerance: float = 0.0001
    ) -> list:  # list[topologic_core.Edge]
        """
        Returns the list of edges from `graph` whose endpoints match the given `vertices`
        according to the `strict` rule.

        If `strict` is True, both endpoints of an edge must be in `vertices`.
        If `strict` is False, at least one endpoint must be in `vertices`.

        Parameters
        ----------
        graph : topologicpy.Graph
            The input graph.
        vertices : list[topologicpy.Vertex]
            The list of vertices to test membership against.
        strict : bool, optional
            If set to True, require both endpoints to be in `vertices`. Otherwise,
            require at least one endpoint to be in `vertices`. Default is False.
        sortBy : str , optional
            The dictionary key to use for sorting the returned edges. Special strings include "length" and "distance" to sort by the length of the edge. Default is None.
        reverse : bool , optional
            If set to True, the sorted list is reversed. This has no effect if the sortBy parameter is not set. Default is False.
        silent : bool, optional
            Isilent : bool, optional
            If set to True, all errors and warnings are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list[topologic_core.Edge]
            The list of matching edges from the original graph (not recreated).

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Edge import Edge
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary

        def sort_edges(edges, sortBy, reverse):
            if not sortBy is None:
                if "length" in sortBy.lower() or "dist" in sortBy.lower():
                    edge_values = [Edge.Length(e) for e in edges]
                else:
                    edge_values = [Dictionary.ValueAtKey(Topology.Dictionary(e), sortBy, "0") for e in edges]
                edges = Helper.Sort(edges, edge_values)
            if reverse:
                edges.reverse()
            return edges

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.InducedEdges - Error: The input 'graph' is not a valid Graph. Returning [].")
            return []

        graph_edges = []
        _ = graph.Edges(graph_edges, tolerance) # Hook to Core
        graph_edges = list(dict.fromkeys(graph_edges)) # remove duplicates

        if not graph_edges:
            return []
        if not vertices:
            graph_edges = sort_edges(graph_edges, sortBy, reverse)
            return graph_edges
        
        if not isinstance(vertices, list):
            if not silent:
                print("Graph.Edges - Error: The input 'vertices' is not a list. Returning [].")
            return []

        valid_vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if not valid_vertices:
            if not silent:
                print("Graph.Edges - Warning: No valid vertices provided. Returning [].")
            return []

        return_edges = []
        for e in graph_edges:
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)

            in_start = not Vertex.Index(sv, valid_vertices) is None
            in_end   = not Vertex.Index(ev, valid_vertices) is None
            if strict:
                if in_start and in_end:
                    return_edges.append(e)
            else:
                if in_start or in_end:
                    return_edges.append(e)

        return_edges = sort_edges(return_edges, sortBy, reverse)
        return return_edges

    # @staticmethod
    # def Edges(graph, vertices=None, tolerance=0.0001):
    #     """
    #     Returns the edges found in the input graph. If the input list of vertices is specified, this method returns the edges connected to this list of vertices. Otherwise, it returns all graph edges.

    #     Parameters
    #     ----------
    #     graph : topologic_core.Graph
    #         The input graph.
    #     vertices : list , optional
    #         An optional list of vertices to restrict the returned list of edges only to those connected to this list.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.

    #     Returns
    #     -------
    #     list
    #         The list of edges in the graph.

    #     """
    #     from topologicpy.Topology import Topology

    #     if not Topology.IsInstance(graph, "Graph"):
    #         print("Graph.Edges - Error: The input graph is not a valid graph. Returning None.")
    #         return None
    #     if not vertices:
    #         edges = []
    #         _ = graph.Edges(edges, tolerance) # Hook to Core
    #         if not edges:
    #             return []
    #         return list(dict.fromkeys(edges)) # remove duplicates
    #     else:
    #         vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
    #     if len(vertices) < 1:
    #         print("Graph.Edges - Error: The input list of vertices does not contain any valid vertices. Returning None.")
    #         return None
    #     edges = []
    #     _ = graph.Edges(vertices, tolerance, edges) # Hook to Core
    #     return list(dict.fromkeys(edges)) # remove duplicates
    
    @staticmethod
    def EigenVectorCentrality(graph, normalize: bool = False, key: str = "eigen_vector_centrality", colorKey: str = "evc_color", colorScale: str = "viridis", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the eigenvector centrality of the input graph. The order of the returned list is the same as the order of vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        weightKey : str, optional
            Ignored in this implementation. Reserved for future use if weighted adjacency matrix is desired.
        normalize : bool, optional
            If set to True, the centrality values are normalized to be in the range 0 to 1. Default is False.
        key : str, optional
            The desired dictionary key under which to store the eigenvector centrality score. Default is "eigen_vector_centrality".
        colorKey : str, optional
            The desired dictionary key under which to store the eigenvector centrality color. Default is "evc_color".
        colorScale : str, optional
            The desired type of Plotly color scale to use (e.g., "viridis", "plasma"). Default is "viridis".
            For a full list of names, see https://plotly.com/python/builtin-colorscales/.
            Also supports color-blind friendly scales: "protanopia", "deuteranopia", "tritanopia".
        mantissa : int, optional
            The desired length of the mantissa. Default is 6.
        tolerance : float, optional
            The convergence tolerance for the power method. Default is 0.0001.
        silent : bool, optional
            If set to True, suppresses all messaging and warnings. Default is False.

        Returns
        -------
        list
            A list of eigenvector centrality values corresponding to the vertices in the input graph.
        """
        import numpy as np
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.EigenVectorCentrality - Error: The input graph is not a valie Topologic Graph. Returning None.")
            return None
        adjacency_matrix = Graph.AdjacencyMatrix(graph)
        vertices = Graph.Vertices(graph)
        n = len(vertices)
        if n == 0:
            return []

        values = np.ones(n)
        for _ in range(100):
            x_new = np.dot(adjacency_matrix, values)
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new = x_new / norm
            if np.linalg.norm(values - x_new) < tolerance:
                break
            values = x_new
        values = [float(x) for x in values]
        if normalize == True:
            if mantissa > 0: # We cannot round numbers from 0 to 1 with a mantissa = 0.
                values = [round(v, mantissa) for v in Helper.Normalize(values)]
            else:
                values = Helper.Normalize(values)
            min_value = 0
            max_value = 1
        else:
            values = [round(v, mantissa) for v in values]
            min_value = min(values)
            max_value = max(values)

        for i, value in enumerate(values):
            d = Topology.Dictionary(vertices[i])
            color = Color.AnyToHex(Color.ByValueInRange(value, minValue=min_value, maxValue=max_value, colorScale=colorScale))
            d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color])
            vertices[i] = Topology.SetDictionary(vertices[i], d)

        return values

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
            If set to True, any existing file is overwritten. Otherwise, it is not. Default is False.
        bidirectional : bool , optional
            If set to True, reverse relationships are created wherever possible. Otherwise, they are not. Default is False.
        includeAttributes : bool , optional
            If set to True, the attributes associated with vertices in the graph are written out. Otherwise, they are not. Default is False.
        includeLabel : bool , optional
            If set to True, a label is attached to each node. Otherwise, it is not. Default is False.
        includeGeometry : bool , optional
            If set to True, the geometry associated with vertices in the graph are written out. Otherwise, they are not. Default is False.
        siteLabel : str , optional
            The desired site label. Default is "Site_0001".
        siteDictionary : dict , optional
            The dictionary of site attributes to include in the output. Default is None.
        buildingLabel : str , optional
            The desired building label. Default is "Building_0001".
        buildingDictionary : dict , optional
            The dictionary of building attributes to include in the output. Default is None.
        storeyPrefix : str , optional
            The desired prefixed to use for each building storey. Default is "Storey".
        floorLevels : list , optional
            The list of floor levels. This should be a numeric list, sorted from lowest to highest.
            If not provided, floorLevels will be computed automatically based on the vertices' (zKey)) attribute. See below.
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. Default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. Default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "hasX".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "hasY".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "hasZ".
        geometryKey : str , optional
            The desired key name to use for geometry. Default is "brep".
        typeKey : str , optional
            The dictionary key to use to look up the type of the node. Default is "type".
        geometryKey : str , optional
            The dictionary key to use to look up the geometry of the node. Default is "brep".
        spaceType : str , optional
            The dictionary string value to use to look up vertices of type "space". Default is "space".
        wallType : str , optional
            The dictionary string value to use to look up vertices of type "wall". Default is "wall".
        slabType : str , optional
            The dictionary string value to use to look up vertices of type "slab". Default is "slab".
        doorType : str , optional
            The dictionary string value to use to look up vertices of type "door". Default is "door".
        windowType : str , optional
            The dictionary string value to use to look up vertices of type "window". Default is "window".
        contentType : str , optional
            The dictionary string value to use to look up vertices of type "content". Default is "contents".
        namespace : str , optional
            The desired namespace to use in the BOT graph. Default is "http://github.com/wassimj/topologicpy/resources".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        
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
        # Make sure the file extension is .bot
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
            The desired graph ID column header. Default is "graph_id".
        graphLabelHeader : str , optional
            The desired graph label column header. Default is "label".
        graphFeaturesHeader : str , optional
            The desired graph features column header. Default is "feat".
        edgeLabelKey : str , optional
            The edge label dictionary key saved in each graph edge. Default is "label".
        defaultEdgeLabel : int , optional
            The default edge label to use if no edge label is found. Default is 0.
        edgeLabelHeader : str , optional
            The desired edge label column header. Default is "label".
        edgeSRCHeader : str , optional
            The desired edge source column header. Default is "src_id".
        edgeDSTHeader : str , optional
            The desired edge destination column header. Default is "dst_id".
        edgeFeaturesHeader : str , optional
            The desired edge features column header. Default is "feat".
        edgeFeaturesKeys : list , optional
            The list of feature dictionary keys saved in the dicitonaries of edges. Default is [].
        edgeTrainMaskHeader : str , optional
            The desired edge train mask column header. Default is "train_mask".
        edgeValidateMaskHeader : str , optional
            The desired edge validate mask column header. Default is "val_mask".
        edgeTestMaskHeader : str , optional
            The desired edge test mask column header. Default is "test_mask".
        edgeMaskKey : str , optional
            The dictionary key where the edge train, validate, test category is to be found. The value should be 0 for train
            1 for validate, and 2 for test. If no key is found, the ratio of train/validate/test will be used. Default is "mask".
        edgeTrainRatio : float , optional
            The desired ratio of the edge data to use for training. The number must be between 0 and 1. Default is 0.8 which means 80% of the data will be used for training.
            This value is ignored if an edgeMaskKey is foud.
        edgeValidateRatio : float , optional
            The desired ratio of the edge data to use for validation. The number must be between 0 and 1. Default is 0.1 which means 10% of the data will be used for validation.
            This value is ignored if an edgeMaskKey is foud.
        edgeTestRatio : float , optional
            The desired ratio of the edge data to use for testing. The number must be between 0 and 1. Default is 0.1 which means 10% of the data will be used for testing.
            This value is ignored if an edgeMaskKey is foud.
        bidirectional : bool , optional
            If set to True, a reversed edge will also be saved for each edge in the graph. Otherwise, it will not. Default is True.
        nodeFeaturesKeys : list , optional
            The list of features keys saved in the dicitonaries of nodes. Default is [].
        nodeLabelKey : str , optional
            The node label dictionary key saved in each graph vertex. Default is "label".
        defaultNodeLabel : int , optional
            The default node label to use if no node label is found. Default is 0.
        nodeIDHeader : str , optional
            The desired node ID column header. Default is "node_id".
        nodeLabelHeader : str , optional
            The desired node label column header. Default is "label".
        nodeFeaturesHeader : str , optional
            The desired node features column header. Default is "feat".
        nodeTrainMaskHeader : str , optional
            The desired node train mask column header. Default is "train_mask".
        nodeValidateMaskHeader : str , optional
            The desired node validate mask column header. Default is "val_mask".
        nodeTestMaskHeader : str , optional
            The desired node test mask column header. Default is "test_mask".
        nodeMaskKey : str , optional
            The dictionary key where the node train, validate, test category is to be found. The value should be 0 for train
            1 for validate, and 2 for test. If no key is found, the ratio of train/validate/test will be used. Default is "mask".
        nodeTrainRatio : float , optional
            The desired ratio of the node data to use for training. The number must be between 0 and 1. Default is 0.8 which means 80% of the data will be used for training.
            This value is ignored if an nodeMaskKey is found.
        nodeValidateRatio : float , optional
            The desired ratio of the node data to use for validation. The number must be between 0 and 1. Default is 0.1 which means 10% of the data will be used for validation.
            This value is ignored if an nodeMaskKey is found.
        nodeTestRatio : float , optional
            The desired ratio of the node data to use for testing. The number must be between 0 and 1. Default is 0.1 which means 10% of the data will be used for testing.
            This value is ignored if an nodeMaskKey is found.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        overwrite : bool , optional
            If set to True, any existing files are overwritten. Otherwise, the input list of graphs is appended to the end of each file. Default is False.

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
            The desired graph width. Default is 20.
        graphLength : float or int , optional
            The desired graph length. Default is 20.
        graphHeight : float or int , optional
            The desired graph height. Default is 20.
        defaultVertexColor : str , optional
            The desired default vertex color. Default is "black".
        defaultVertexSize : float or int , optional
            The desired default vertex size. Default is 3.
        defaultEdgeColor : str , optional
            The desired default edge color. Default is "black".
        defaultEdgeWeight : float or int , optional
            The desired default edge weight. The edge weight determines the width of the displayed edge. Default is 3.
        defaultEdgeType : str , optional
            The desired default edge type. This can be one of "directed" or "undirected". Default is "undirected".
        vertexLabelKey : str , optional
            If specified, the vertex dictionary is searched for this key to determine the vertex label. If not specified
            the vertex label being is set to "Node X" where is X is a unique number. Default is None.
        vertexColorKey : str , optional
            If specified, the vertex dictionary is searched for this key to determine the vertex color. If not specified
            the vertex color is set to the value defined by defaultVertexColor parameter. Default is None.
        vertexSizeKey : str , optional
            If specified, the vertex dictionary is searched for this key to determine the vertex size. If not specified
            the vertex size is set to the value defined by defaultVertexSize parameter. Default is None.
        edgeLabelKey : str , optional
            If specified, the edge dictionary is searched for this key to determine the edge label. If not specified
            the edge label being is set to "Edge X" where is X is a unique number. Default is None.
        edgeColorKey : str , optional
            If specified, the edge dictionary is searched for this key to determine the edge color. If not specified
            the edge color is set to the value defined by defaultEdgeColor parameter. Default is None.
        edgeWeightKey : str , optional
            If specified, the edge dictionary is searched for this key to determine the edge weight. If not specified
            the edge weight is set to the value defined by defaultEdgeWeight parameter. Default is None.
        overwrite : bool , optional
            If set to True, any existing file is overwritten. Otherwise, it is not. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
        # Make sure the file extension is .gexf
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
            if isinstance(vertex_color, list):
                if len(vertex_color) >= 3:
                    node_dict['color'] = Color.CSSNamedColor(vertex_color)
                    r, g, b = vertex_color
                    node_dict['r'] = r
                    node_dict['g'] = g
                    node_dict['b'] = b
                else:
                    vertex_color = defaultVertexColor
            else:
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
    def ExportToGraphVizGraph(graph,
                              path,
                              device = 'svg_inline', deviceKey=None,
                              scale = 1, scaleKey=None,
                              directed=False, directedKey=None,
        layout = 'dot', # or circo fdp neato nop nop1 nop2 osage patchwork sfdp twopi
        layoutKey=None,
        rankDir='TB', rankDirKey=None, # or LR, RL, BT
        bgColor='white', bgColorKey=None,
        fontName='Arial', fontNameKey=None,
        fontSize= 12, fontSizeKey=None,
        vertexSep= 0.5, vertexSepKey=None,
        rankSep= 0.5, rankSepKey=None,
        splines='True', splinesKey=None,
        showGraphLabel = False,
        graphLabel='', graphLabelKey=None,
        graphLabelLoc='t', graphLabelLocKey=None,
        showVertexLabel = False,
        vertexLabelPrefix='' , vertexLabelKey=None,
        vertexWidth=0.5, vertexWidthKey=None,
        vertexHeight=0.5, vertexHeightKey=None,
        vertexFixedSize=False, vertexFixedSizeKey=None,
        vertexShape='circle', vertexShapeKey=None,
        vertexStyle='filled', vertexStyleKey=None,
        vertexFillColor='lightgray', vertexFillColorKey=None,
        vertexColor='black', vertexColorKey=None,
        vertexFontColor='black', vertexFontColorKey=None,
        showEdgeLabel = False,
        edgeLabelPrefix='', edgeLabelKey=None,
        edgeColor='black', edgeColorKey=None,
        edgeWidth=1, edgeWidthKey=None,
        edgeStyle='solid', edgeStyleKey=None,
        edgeArrowhead='normal', edgeArrowheadKey=None,
        edgeFontColor='black', edgeFontColorKey=None,
        overwrite=False,
        silent=False):
        """
        Exports the input graph to a GraphViz `.gv` (dot) file.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        path : str
            The path to the output file (e.g., "output.gv").
        device : str, optional
            The output format device, such as 'svg_inline', 'pdf', or 'png'. Default is 'svg_inline'.
        deviceKey : str, optional
            Dictionary key to override the `device` value. Default is None.
        scale : float, optional
            Global scaling factor. Default is 1.
        scaleKey : str, optional
            Dictionary key to override the `scale` per-graph. Default is None.
        directed : bool, optional
            Whether to treat the graph as directed. Default is False.
        directedKey : str, optional
            Dictionary key to override the `directed` flag per-graph. Default is None.
        layout : str, optional
            Layout engine to use. Options include 'dot', 'circo', 'fdp', 'neato', 'osage', 'sfdp', etc. Default is 'dot'.
        layoutKey : str, optional
            Dictionary key to override the `layout` per-graph. Default is None.
        rankDir : str, optional
            Direction of graph ranking. Options: 'TB' (top-bottom), 'LR' (left-right), 'RL', 'BT'. Default is 'TB'.
        rankDirKey : str, optional
            Dictionary key to override `rankDir` per-graph. Default is None.
        bgColor : str, optional
            Background color. Default is 'white'.
        bgColorKey : str, optional
            Dictionary key to override `bgColor`. Default is None.
        fontName : str, optional
            Name of the font to use for all text. Default is 'Arial'.
        fontNameKey : str, optional
            Dictionary key to override `fontName`. Default is None.
        fontSize : int or float, optional
            Size of font in points. Default is 12.
        fontSizeKey : str, optional
            Dictionary key to override `fontSize`. Default is None.
        vertexSep : float, optional
            Minimum separation between vertices. Default is 0.5.
        vertexSepKey : str, optional
            Dictionary key to override `vertexSep`. Default is None.
        rankSep : float, optional
            Separation between ranks. Default is 0.5.
        rankSepKey : str, optional
            Dictionary key to override `rankSep`. Default is None.
        splines : str, optional
            Whether to use spline edges. Can be 'true', 'false', or 'polyline'. Default is 'True'.
        splinesKey : str, optional
            Dictionary key to override `splines`. Default is None.
        showGraphLabel : bool, optional
            Whether to show a label for the whole graph. Default is False.
        graphLabel : str, optional
            Text for the graph label. Default is an empty string.
        graphLabelKey : str, optional
            Dictionary key to override `graphLabel`. Default is None.
        graphLabelLoc : str, optional
            Position of the graph label: 't' (top), 'b' (bottom), 'c' (center). Default is 't'.
        graphLabelLocKey : str, optional
            Dictionary key to override `graphLabelLoc`. Default is None.
        showVertexLabel : bool, optional
            Whether to display vertex labels. Default is False.
        vertexLabelPrefix : str, optional
            Text prefix for vertex labels. Default is empty string.
        vertexLabelKey : str, optional
            Dictionary key used to retrieve label text from vertex dictionary. Default is None.
        vertexWidth : float, optional
            Width of each vertex. Default is 0.5.
        vertexWidthKey : str, optional
            Dictionary key to override `vertexWidth`. Default is None.
        vertexHeight : float, optional
            Height of each vertex. Default is 0.5.
        vertexHeightKey : str, optional
            Dictionary key to override `vertexHeight`. Default is None.
        vertexFixedSize : bool, optional
            Whether vertices should be fixed in size. Default is False.
        vertexFixedSizeKey : str, optional
            Dictionary key to override `vertexFixedSize`. Default is None.
        vertexShape : str, optional
            Shape of the vertex ('circle', 'ellipse', 'box', etc.). Default is 'circle'.
        vertexShapeKey : str, optional
            Dictionary key to override `vertexShape`. Default is None.
        vertexStyle : str, optional
            Style of vertex (e.g., 'filled', 'dashed'). Default is 'filled'.
        vertexStyleKey : str, optional
            Dictionary key to override `vertexStyle`. Default is None.
        vertexFillColor : str, optional
            Fill color for vertices. Default is 'lightgray'.
        vertexFillColorKey : str, optional
            Dictionary key to override `vertexFillColor`. Default is None.
        vertexColor : str, optional
            Border color for vertices. Default is 'black'.
        vertexColorKey : str, optional
            Dictionary key to override `vertexColor`. Default is None.
        vertexFontColor : str, optional
            Font color for vertex labels. Default is 'black'.
        vertexFontColorKey : str, optional
            Dictionary key to override `vertexFontColor`. Default is None.
        showEdgeLabel : bool, optional
            Whether to display edge labels. Default is False.
        edgeLabelPrefix : str, optional
            Text prefix for edge labels. Default is empty string.
        edgeLabelKey : str, optional
            Dictionary key used to retrieve label text from edge dictionary. Default is None.
        edgeColor : str, optional
            Color of edges. Default is 'black'.
        edgeColorKey : str, optional
            Dictionary key to override `edgeColor`. Default is None.
        edgeWidth : float, optional
            Width (thickness) of edges. Default is 1.
        edgeWidthKey : str, optional
            Dictionary key to override `edgeWidth`. Default is None.
        edgeStyle : str, optional
            Style of the edge line (e.g., 'solid', 'dashed'). Default is 'solid'.
        edgeStyleKey : str, optional
            Dictionary key to override `edgeStyle`. Default is None.
        edgeArrowhead : str, optional
            Arrowhead style for directed edges. Default is 'normal'.
        edgeArrowheadKey : str, optional
            Dictionary key to override `edgeArrowhead`. Default is None.
        edgeFontColor : str, optional
            Font color for edge labels. Default is 'black'.
        edgeFontColorKey : str, optional
            Dictionary key to override `edgeFontColor`. Default is None.
        overwrite : bool, optional
            If True, overwrites existing files at the given path. Default is False.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the graph was successfully exported. False otherwise.
        """

        from topologicpy.Topology import Topology
        from os.path import exists
        dot = Graph.GraphVizGraph(
        graph,
        device = device, deviceKey = deviceKey,
        scale = scale, scaleKey = scaleKey,
        directed = directed, directedKey = directedKey,
        layout = layout,
        layoutKey = layoutKey,
        rankDir= rankDir, rankDirKey = rankDirKey,
        bgColor=bgColor, bgColorKey=bgColorKey,
        fontName=fontName, fontNameKey=fontNameKey,
        fontSize= fontSize, fontSizeKey=fontSizeKey,
        vertexSep= vertexSep, vertexSepKey=vertexSepKey,
        rankSep= rankSep, rankSepKey=rankSepKey,
        splines=splines, splinesKey=splinesKey,
        showGraphLabel = showGraphLabel,
        graphLabel=graphLabel, graphLabelKey=graphLabelKey,
        graphLabelLoc=graphLabelLoc, graphLabelLocKey=graphLabelLocKey,

        showVertexLabel = showVertexLabel,
        vertexLabelPrefix=vertexLabelPrefix , vertexLabelKey=vertexLabelKey,
        vertexWidth=vertexWidth, vertexWidthKey=vertexWidthKey,
        vertexHeight=vertexHeight, vertexHeightKey=vertexHeightKey,
        vertexFixedSize=vertexFixedSize, vertexFixedSizeKey=vertexFixedSizeKey,
        vertexShape=vertexShape, vertexShapeKey=vertexShapeKey,
        vertexStyle=vertexStyle, vertexStyleKey=vertexStyleKey,
        vertexFillColor=vertexFillColor, vertexFillColorKey=vertexFillColorKey,
        vertexColor=vertexColor, vertexColorKey=vertexColorKey,
        vertexFontColor=vertexFontColor, vertexFontColorKey=vertexFontColorKey,

        showEdgeLabel = showEdgeLabel,
        edgeLabelPrefix=edgeLabelPrefix, edgeLabelKey=edgeLabelKey,
        edgeColor=edgeColor, edgeColorKey=edgeColorKey,
        edgeWidth=edgeWidth, edgeWidthKey=edgeWidthKey,
        edgeStyle=edgeStyle, edgeStyleKey=edgeStyleKey,
        edgeArrowhead=edgeArrowhead, edgeArrowheadKey=edgeArrowheadKey,
        edgeFontColor=edgeFontColor, edgeFontColorKey=edgeFontColorKey,
        silent=silent)

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.ExportToGraphVizGraph - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        if not isinstance(path, str):
            if not silent:
                print("Graph.ExportToGraphVizGraph - Error: the input path parameter is not a valid string. Returning None.")
            return None
        # Make sure the file extension is .gv
        ext = path[len(path)-3:len(path)]
        if ext.lower() != ".gv":
            path = path+".gv"
        if not overwrite and exists(path):
            if not silent:
                print("Graph.ExportToGraphVizGraph - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        try:
            dot.save(filename=path)
            return True
        except:
            return False

    @staticmethod
    def ExportToJSON(graph, path, propertiesKey="properties", verticesKey="vertices", edgesKey="edges", vertexLabelKey="", edgeLabelKey="", xKey="x", yKey="y", zKey="z", indent=4, sortKeys=False, mantissa=6, overwrite=False):
        """
        Exports the input graph to a JSON file.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        path : str
            The path to the JSON file.
        propertiesKey : str , optional
            The desired key name to call graph properties. Default is "properties".
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "z".
        indent : int , optional
            The desired amount of indent spaces to use. Default is 4.
        sortKeys : bool , optional
            If set to True, the keys will be sorted. Otherwise, they won't be. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. Default is False.

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
            jsondata = Graph.JSONData(graph, propertiesKey=propertiesKey, verticesKey=verticesKey, edgesKey=edgesKey, vertexLabelKey=vertexLabelKey, edgeLabelKey=edgeLabelKey, xKey=xKey, yKey=yKey, zKey=zKey, mantissa=mantissa)
            if jsondata != None:
                json.dump(jsondata, f, indent=indent, sort_keys=sortKeys)
                f.close()
                return True
            else:
                f.close()
                return False
        return False

    @staticmethod
    def FiedlerVector(graph, mantissa = 6, silent: bool = False):
        """
        Computes the Fiedler vector of a graph. See https://en.wikipedia.org/wiki/Algebraic_connectivity.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph
        mantissa : int , optional
                The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The Fiedler vector (eigenvector corresponding to the second smallest eigenvalue).
        """
        from topologicpy.Topology import Topology
        from topologicpy.Matrix import Matrix
        import numpy as np

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.FiedlerVector - Error: The input graph parameter is not a valid graph. Returning None.")
            
        laplacian = Graph.Laplacian(graph)
        eigenvalues, eigenvectors = Matrix.EigenvaluesAndVectors(laplacian, mantissa=mantissa)
        return eigenvectors[1]

    @staticmethod
    def FiedlerVectorPartition(graph, key="partition", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Partitions the input graph based on FiedlerVector method. See https://en.wikipedia.org/wiki/Graph_partition.

        Parameters
        ----------
        graph : topologicp.Graph
            The input topologic graph.
        key : str , optional
            The vertex and edge dictionary key under which to store the parition number. Default is "partition".
            Valid partition numbers start from 1. Cut edges receive a partition number of 0.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologicpy.Graph
            The partitioned topologic graph.

        """ 
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge    
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.FiedlerVectorPartition - Error: The input graph parameter is not a valid topologic graph. Returning None.")
            return None

        fiedler_vector = Graph.FiedlerVector(graph, mantissa=mantissa, silent=silent)

        vertices = Graph.Vertices(graph)
        for i, f_v in enumerate(fiedler_vector):
            if f_v >=0:
                partition = 1
            else:
                partition = 2 
            d = Topology.Dictionary(vertices[i])
            d = Dictionary.SetValueAtKey(d, key, partition)
            vertices[i] = Topology.SetDictionary(vertices[i], d)
        edges = Graph.Edges(graph)
        if not edges == None:
            for edge in edges:
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                status_1 = False
                status_2 = False
                partition_1 = 0
                partition_2 = 0
                for i, v in enumerate(vertices):
                    if Vertex.IsCoincident(sv, v, tolerance=tolerance):
                        status_1 = True
                        partition_1 = Dictionary.ValueAtKey(Topology.Dictionary(v), key)
                        break
                for i, v in enumerate(vertices):
                    if Vertex.IsCoincident(ev, v, tolerance=tolerance):
                        status_2 = True
                        partition_2 = Dictionary.ValueAtKey(Topology.Dictionary(v), key)
                        break
                partition = 0
                if status_1 and status_2:
                    if partition_1 == partition_2:
                        partition = partition_1
                d = Topology.Dictionary(edge)
                d = Dictionary.SetValueAtKey(d, key, partition)
                edge = Topology.SetDictionary(edge, d)
        return graph


    @staticmethod
    def InducedSubgraph(graph, vertices: list = None, strict: bool = False, silent: bool = False, tolerance: float = 0.0001):
        """
        Returns the subgraph whose edges are connected to the given `vertices`
        according to the `strict` rule. Isolated vertices are included as-is.

        If `strict` is True, both endpoints of an edge must be in `vertices`.
        If `strict` is False, at least one endpoint must be in `vertices`.

        Parameters
        ----------
        graph : topologicpy.Graph
            The input graph.
        vertices : list[topologicpy.Vertex]
            The list of vertices to test membership against.
        strict : bool, optional
            If set to True, require both endpoints to be in `vertices`. Otherwise,
            require at least one endpoint to be in `vertices`. Default is False.
        silent : bool, optional
            Isilent : bool, optional
            If set to True, all errors and warnings are suppressed. Default is False
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list[topologic_core.Edge]
            The list of matching edges from the original graph (not recreated).

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.InducedSubgraph - Error: The input graph parameter is not a valid graph. Returning None.")
        
        if not isinstance(vertices, list):
            if not silent:
                print("Graph.InducedSubgraph - Error: The input 'vertices' is not a list. Returning None.")
            return None

        valid_vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if not valid_vertices:
            if not silent:
                print("Graph.InducedSubgraph - Warning: No valid vertices provided. Returning None.")
            return None
        connected_vertices = [v for v in valid_vertices if Graph.VertexDegree(graph, v) > 0]
        edges = Graph.Edges(graph, connected_vertices, strict=strict, tolerance=tolerance)
        return Graph.ByVerticesEdges(valid_vertices, edges)

    @staticmethod
    def IsEmpty(graph, silent: bool = False):
        """
        Tests if the input graph is empty (Has no vertices).
        
        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        bool
            True if the two input graphs are isomorphic. False otherwise

        """
        
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.IsEmpty - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        
        return (len(Graph.Vertices(graph)) == 0)
    
    @staticmethod
    def IsIsomorphic(graphA, graphB, maxIterations=10, silent=False):
        """
        Tests if the two input graphs are isomorphic according to the Weisfeiler Lehman graph isomorphism test. See https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test
        
        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        maxIterations : int , optional
            This number limits the number of iterations to prevent the function from running indefinitely, particularly for very large or complex graphs.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        bool
            True if the two input graphs are isomorphic. False otherwise

        """

        from topologicpy.Topology import Topology

        def weisfeiler_lehman_test(graph1, graph2, max_iterations=10):
            """
            Test if two graphs are isomorphic using the Weisfeiler-Leman (WL) algorithm with early stopping.

            Parameters:
            graph1 (dict): Adjacency list representation of the first graph.
            graph2 (dict): Adjacency list representation of the second graph.
            max_iterations (int): Maximum WL iterations allowed (default is 10).

            Returns:
            bool: True if the graphs are WL-isomorphic, False otherwise.
            """

            def wl_iteration(labels, graph):
                """Perform one WL iteration and return updated labels."""
                new_labels = {}
                for node in graph:
                    neighborhood_labels = sorted([labels[neighbor] for neighbor in graph[node]])
                    new_labels[node] = (labels[node], tuple(neighborhood_labels))
                unique_labels = {}
                count = 0
                for node in sorted(new_labels):
                    if new_labels[node] not in unique_labels:
                        unique_labels[new_labels[node]] = count
                        count += 1
                    new_labels[node] = unique_labels[new_labels[node]]
                return new_labels

            # Initialize labels
            labels1 = {node: 1 for node in graph1}
            labels2 = {node: 1 for node in graph2}

            for i in range(max_iterations):
                # Perform WL iteration for both graphs
                new_labels1 = wl_iteration(labels1, graph1)
                new_labels2 = wl_iteration(labels2, graph2)

                # Check if the label distributions match
                if sorted(new_labels1.values()) != sorted(new_labels2.values()):
                    return False

                # Check for stability (early stopping)
                if new_labels1 == labels1 and new_labels2 == labels2:
                    break

                # Update labels for next iteration
                labels1, labels2 = new_labels1, new_labels2

            return True
        
        if not Topology.IsInstance(graphA, "Graph") and not Topology.IsInstance(graphB, "Graph"):
            if not silent:
                print("Graph.IsIsomorphic - Error: The input graph parameters are not valid graphs. Returning None.")
            return None
        if not Topology.IsInstance(graphA, "Graph"):
            if not silent:
                print("Graph.IsIsomorphic - Error: The input graphA parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "Graph"):
            if not silent:
                print("Graph.IsIsomorphic - Error: The input graphB parameter is not a valid graph. Returning None.")
            return None
        if maxIterations <= 0:
            if not silent:
                print("Graph.IsIsomorphic - Error: The input maxIterations parameter is not within a valid range. Returning None.")
            return None
        
        g1 = Graph.AdjacencyDictionary(graphA)
        g2 = Graph.AdjacencyDictionary(graphB)
        return weisfeiler_lehman_test(g1, g2, max_iterations=maxIterations)

    @staticmethod
    def Reshape(graph,
                shape="spring 2D",
                k=0.8, seed=None,
                iterations=50,
                rootVertex=None,
                size=1,
                factor=1,
                sides=16,
                key="",
                tolerance=0.0001,
                silent=False):
        """
        Reshapes the input graph according to the desired input shape parameter.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        shape : str , optional
            The desired shape of the graph.
            ['circle 2D', 'grid 2D', 'line 2D', 'radial 2D', 'spring 2D', 'tree 2D', 'grid 3D', 'sphere 3D', 'tree 3D']
            If set to 'spring 2D' or 'spring_3d', the algorithm uses a simplified version of the Fruchterman-Reingold force-directed algorithm to distribute the vertices.
            If set to 'radial 2D', the nodes will be distributed along concentric circles in the XY plane.
            If set to 'tree 2D' or 'tree 3D', the nodes will be distributed using the Reingold-Tillford layout.
            If set to 'circle 2D', the nodes will be distributed on the cirumference of a segemented circles  in the XY plane, based on the size and sides input parameter (radius=size/2).
            If set to 'line 2D', the nodes will be distributed on a line in the XY plane based on the size input parameter (length=size).
            If set to 'spehere 3D', the nodes will be distributed on the surface of a sphere based on the size input parameter raidus=size/2).
            If set to 'grid 2D', the nodes will be distributed on a grid in the XY plane with size based on the size input parameter (length=width=size).
            If set to 'grid 3D', the nodes will be distributed on a 3D cubic grid/matrix based on the size input parameter(width=length=height=size).
            If set to 'cluster 2D', or 'cluster 3D, the nodes will be clustered according to the 'key' input parameter. The overall radius of the cluster is determined by the size input parameter (radius = size/2)
            The default is 'spring 2D'.
        k : float, optional
            The desired spring constant to use for the attractive and repulsive forces. Default is 0.8.
        seed : int , optional
            The desired random seed to use. Default is None.
        iterations : int , optional
            The desired maximum number of iterations to solve the forces in the 'spring' mode. Default is 50.
        rootVertex : topologic_core.Vertex , optional
            The desired vertex to use as the root of the tree and radial layouts.
        size : float , optional
            The desired overall size of the graph.
        sides : int , optional
            The desired number of sides of the circle layout option. Default is 16
        length : float, optional
            The desired horizontal length for the line layout option. Default is 1.0.
        key : string, optional
            The key under which to find the clustering value for the 'cluster_2d' and 'cluster_3d' options. Default is "".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The reshaped graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Graph import Graph
        from topologicpy.Grid import Grid
        from topologicpy.Helper import Helper
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import numpy as np
        import math
        from collections import defaultdict
        import random


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

            

            # Get the number of nodes from the edge list.
            if len(edge_list) > 0:
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
            else:
                return None

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
        
        def generate_cubic_matrix(size, min_points):
            # Calculate the minimum points per axis to reach or exceed min_points in total
            points_per_axis = int(np.ceil(min_points ** (1/3)))
            
            # Calculate the spacing based on the size and points per axis
            spacing = size / (points_per_axis - 1) if points_per_axis > 1 else 0
            
            # Generate linearly spaced points from -size/2 to size/2 along each axis
            x = np.linspace(-size / 2, size / 2, points_per_axis)
            y = np.linspace(-size / 2, size / 2, points_per_axis)
            z = np.linspace(-size / 2, size / 2, points_per_axis)
            
            # Create a meshgrid and stack them to get XYZ coordinates for each point
            X, Y, Z = np.meshgrid(x, y, z)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            return points
        
        def vertex_max_degree(graph, vertices):
            degrees = [Graph.VertexDegree(graph, vertex) for vertex in vertices]
            i = degrees.index(max(degrees))
            return vertices[i], i
        
        def circle_layout_2d(graph, radius=0.5, sides=16):
            vertices = Graph.Vertices(graph)
            edges = Graph.Edges(graph)
            edge_dict = {}

            for i, edge in enumerate(edges):
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                si = Vertex.Index(sv, vertices)
                ei = Vertex.Index(ev, vertices)
                edge_dict[str(si)+"_"+str(ei)] = i
                edge_dict[str(ei)+"_"+str(si)] = i
            n = len(vertices)
            c = Wire.Circle(radius=radius, sides=sides)
            c_vertices = []
            for i in range(n):
                u = i*(1/n)
                c_vertices.append(Wire.VertexByParameter(c, u))

            for i, c_v in enumerate(c_vertices):
                d = Topology.Dictionary(vertices[i])
                c_v = Topology.SetDictionary(c_v, d, silent=True)
            adj_dict = Graph.AdjacencyDictionary(graph)
            keys = adj_dict.keys()
            
            c_edges = []
            used = [[0] * n for _ in range(n)]
            for key in keys:
                x = int(key)
                adj_vertices = [int(v) for v in adj_dict[key]]
                for y in adj_vertices:
                    if used[x][y] == 0:
                        v1 = Vector.ByCoordinates(Vertex.X(c_vertices[x]), Vertex.Y(c_vertices[x]), Vertex.Z(c_vertices[x]))
                        v2 = Vector.ByCoordinates(Vertex.X(c_vertices[y]), Vertex.Y(c_vertices[y]), Vertex.Z(c_vertices[y]))
                        ang1 = Vector.CompassAngle(v1, [0,1,0])
                        ang2 = Vector.CompassAngle(v2, [0,1,0])
                        if ang2-ang1 < 180:
                            e = Edge.ByVertices(c_vertices[x], c_vertices[y])
                        else:
                            e = Edge.ByVertices(c_vertices[y], c_vertices[x])
                        orig_edge_index = edge_dict.get(str(x)+"_"+str(y), edge_dict.get(str(y)+"_"+str(x), None))
                        if orig_edge_index:
                            d = Topology.Dictionary(edges[orig_edge_index])
                            e = Topology.SetDictionary(e, d, silent=True)
                            c_edges.append(e)
                            used[x][y] = 1
                            used[y][x] = 1
            new_g = Graph.ByVerticesEdges(c_vertices, c_edges)
            return new_g
        
        def cluster_layout_2d(graph, key, radius=0.5):
            
            d = Graph.MeshData(graph)
            edges = d['edges']
            v_dicts = d['vertexDictionaries']
            e_dicts = d['edgeDictionaries']
            vertices = Graph.Vertices(graph)
            # Step 1: Group objects by key value while remembering their original indices
            grouped_objects = defaultdict(list)
            object_indices = []  # Stores original indices of objects in the order they were grouped

            for idx, obj in enumerate(vertices):
                d = Topology.Dictionary(obj)
                value = Dictionary.ValueAtKey(d, key)
                grouped_objects[value].append((obj, idx))
                object_indices.append((value, idx))

            # Step 2: Compute cluster centers on the circumference of a circle
            cluster_centers = {}
            num_clusters = len(grouped_objects)
            
            # Function to generate cluster center on the circle's circumference
            def generate_cluster_center(index, total_clusters, circle_radius):
                # Distribute cluster centers evenly along the circumference of a circle
                angle = (2 * math.pi * index) / total_clusters  # Equal angle separation
                x = circle_radius * math.cos(angle)
                y = circle_radius * math.sin(angle)
                return (x, y)

            # Step 3: Compute vertices for each cluster
            object_positions = [None] * len(vertices)  # Placeholder list for ordered vertices
            cluster_index = 0

            for value, objs in grouped_objects.items():
                intra_cluster_radius = radius*(len(objs)/len(vertices) + 0.1)
                # Determine the center of the current cluster
                if value not in cluster_centers:
                    cluster_center = generate_cluster_center(cluster_index, num_clusters, radius)
                    cluster_centers[value] = cluster_center
                    cluster_index += 1
                else:
                    cluster_center = cluster_centers[value]
                
                # Step 4: Place objects randomly around the cluster center
                for obj, original_index in objs:
                    # Randomly place the object within the intra-cluster circle
                    r = intra_cluster_radius * math.sqrt(random.random())  # Random distance with sqrt for uniform distribution
                    angle = random.uniform(0, 2 * math.pi)  # Random angle

                    # Polar coordinates to Cartesian for local positioning
                    x = cluster_center[0] + r * math.cos(angle)
                    y = cluster_center[1] + r * math.sin(angle)

                    # Save the coordinates in the correct order
                    object_positions[original_index] = [x, y]
            
            positions = [[p[0], p[1], 0] for p in object_positions]
            new_g = Graph.ByMeshData(positions, edges, v_dicts, e_dicts, tolerance=0.001)
            return new_g

        def cluster_layout_3d(graph, key, radius=0.5):
            d = Graph.MeshData(graph)
            edges = d['edges']
            v_dicts = d['vertexDictionaries']
            e_dicts = d['edgeDictionaries']
            vertices = Graph.Vertices(graph)

            # Step 1: Group objects by key value while remembering their original indices
            grouped_objects = defaultdict(list)
            object_indices = []  # Stores original indices of objects in the order they were grouped

            for idx, obj in enumerate(vertices):
                d = Topology.Dictionary(obj)
                value = Dictionary.ValueAtKey(d, key)
                grouped_objects[value].append((obj, idx))
                object_indices.append((value, idx))

            # Step 2: Compute cluster centers on the surface of a sphere
            cluster_centers = {}
            num_clusters = len(grouped_objects)
            
            # Function to generate cluster center on the surface of a sphere
            def generate_cluster_center(index, total_clusters, sphere_radius):
                # Use a spiral algorithm to distribute cluster centers evenly on a sphere's surface
                phi = math.acos(1 - 2 * (index + 0.5) / total_clusters)  # Inclination angle
                theta = math.pi * (1 + 5**0.5) * index  # Azimuthal angle (Golden angle)
                
                x = sphere_radius * math.sin(phi) * math.cos(theta)
                y = sphere_radius * math.sin(phi) * math.sin(theta)
                z = sphere_radius * math.cos(phi)
                return (x, y, z)

            # Step 3: Compute vertices for each cluster
            object_positions = [None] * len(vertices)  # Placeholder list for ordered vertices
            cluster_index = 0

            for value, objs in grouped_objects.items():
                # Determine the center of the current cluster
                if value not in cluster_centers:
                    cluster_center = generate_cluster_center(cluster_index, num_clusters, radius)
                    cluster_centers[value] = cluster_center
                    cluster_index += 1
                else:
                    cluster_center = cluster_centers[value]

                intra_cluster_radius = radius*(len(objs)/len(vertices) + 0.1)

                # Step 4: Place objects randomly within the cluster's spherical volume
                for obj, original_index in objs:
                    # Randomly place the object within the intra-cluster sphere
                    u = random.random()
                    v = random.random()
                    r = intra_cluster_radius * (u ** (1/3))  # Random distance with cube root for uniform distribution

                    theta = 2 * math.pi * v  # Random azimuthal angle
                    phi = math.acos(2 * u - 1)  # Random polar angle

                    # Spherical to Cartesian for local positioning
                    x = cluster_center[0] + r * math.sin(phi) * math.cos(theta)
                    y = cluster_center[1] + r * math.sin(phi) * math.sin(theta)
                    z = cluster_center[2] + r * math.cos(phi)

                    # Save the coordinates in the correct order
                    object_positions[original_index] = [x, y, z]

            positions = [[p[0], p[1], p[2]] for p in object_positions]
            new_g = Graph.ByMeshData(positions, edges, v_dicts, e_dicts, tolerance=0.001)
            return new_g

        def sphere_layout_3d(graph, radius=0.5):
            def points_on_sphere(n, r):
                points = []
                phi = math.pi * (3. - math.sqrt(5.))  # Golden angle in radians

                for i in range(n):
                    y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
                    radius = math.sqrt(1 - y * y)    # radius at y

                    theta = phi * i  # Golden angle increment

                    x = math.cos(theta) * radius * r
                    z = math.sin(theta) * radius * r
                    y *= r

                    points.append([x, y, z])
                return points
            
            vertices = Graph.Vertices(graph)
            edges = Graph.Edges(graph)
            edge_dict = {}

            for i, edge in enumerate(edges):
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                si = Vertex.Index(sv, vertices)
                ei = Vertex.Index(ev, vertices)
                edge_dict[str(si)+"_"+str(ei)] = i
                edge_dict[str(ei)+"_"+str(si)] = i
            n = len(vertices)
            c_points = points_on_sphere(n, r=radius)
            c_vertices = [Vertex.ByCoordinates(coord) for coord in c_points]
            for i, c_v in enumerate(c_vertices):
                d = Topology.Dictionary(vertices[i])
                c_v = Topology.SetDictionary(c_v, d, silent=True)
            adj_dict = Graph.AdjacencyDictionary(graph)
            keys = adj_dict.keys()
            
            c_edges = []
            used = [[0] * n for _ in range(n)]
            for key in keys:
                x = int(key)
                adj_vertices = [int(v) for v in adj_dict[key]]
                for y in adj_vertices:
                    if used[x][y] == 0:
                        v1 = Vector.ByCoordinates(Vertex.X(c_vertices[x]), Vertex.Y(c_vertices[x]), Vertex.Z(c_vertices[x]))
                        v2 = Vector.ByCoordinates(Vertex.X(c_vertices[y]), Vertex.Y(c_vertices[y]), Vertex.Z(c_vertices[y]))
                        ang1 = Vector.CompassAngle(v1, [0,1,0])
                        ang2 = Vector.CompassAngle(v2, [0,1,0])
                        if ang2-ang1 < 180:
                            e = Edge.ByVertices(c_vertices[x], c_vertices[y])
                        else:
                            e = Edge.ByVertices(c_vertices[y], c_vertices[x])
                        orig_edge_index = edge_dict.get(str(x)+"_"+str(y), edge_dict.get(str(y)+"_"+str(x), None))
                        if orig_edge_index:
                            d = Topology.Dictionary(edges[orig_edge_index])
                            e = Topology.SetDictionary(e, d, silent=True)
                            c_edges.append(e)
                            used[x][y] = 1
                            used[y][x] = 1
            new_g = Graph.ByVerticesEdges(c_vertices, c_edges)
            return new_g
        
        def grid_layout_2d(graph, size=1):
            vertices = Graph.Vertices(graph)
            n = len(vertices)
            u = int(math.sqrt(n))
            if u*u < n:
                u += 1
            u_range = [t/(u-1) for t in range(u)]            
            edges = Graph.Edges(graph)
            edge_dict = {}

            for i, edge in enumerate(edges):
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                si = Vertex.Index(sv, vertices)
                ei = Vertex.Index(ev, vertices)
                edge_dict[str(si)+"_"+str(ei)] = i
                edge_dict[str(ei)+"_"+str(si)] = i
            f = Face.Rectangle(width=size, length=size)
            c = Grid.VerticesByParameters(face=f, uRange=u_range, vRange=u_range)
            c_vertices = Topology.Vertices(c)[:len(vertices)]

            for i, c_v in enumerate(c_vertices):
                d = Topology.Dictionary(vertices[i])
                c_v = Topology.SetDictionary(c_v, d, silent=True)
            adj_dict = Graph.AdjacencyDictionary(graph)
            keys = adj_dict.keys()
            
            c_edges = []
            used = [[0] * n for _ in range(n)]
            for key in keys:
                x = int(key)
                adj_vertices = [int(v) for v in adj_dict[key]]
                for y in adj_vertices:
                    if used[x][y] == 0:
                        e = Edge.ByVertices(c_vertices[x], c_vertices[y])
                        orig_edge_index = edge_dict.get(str(x)+"_"+str(y), edge_dict.get(str(y)+"_"+str(x), None))
                        if orig_edge_index:
                            d = Topology.Dictionary(edges[orig_edge_index])
                            e = Topology.SetDictionary(e, d, silent=True)
                            c_edges.append(e)
                            used[x][y] = 1
                            used[y][x] = 1
            new_g = Graph.ByVerticesEdges(c_vertices, c_edges)
            return new_g

        def line_layout_2d(graph, length=1):
            vertices = Graph.Vertices(graph)
            edges = Graph.Edges(graph)
            edge_dict = {}

            for i, edge in enumerate(edges):
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                si = Vertex.Index(sv, vertices)
                ei = Vertex.Index(ev, vertices)
                edge_dict[str(si)+"_"+str(ei)] = i
                edge_dict[str(ei)+"_"+str(si)] = i
            
            n = len(vertices)
            c = Wire.Line(length=length, sides=n-1)
            c_vertices = Topology.Vertices(c)

            for i, c_v in enumerate(c_vertices):
                d = Topology.Dictionary(vertices[i])
                c_v = Topology.SetDictionary(c_v, d, silent=True)
            adj_dict = Graph.AdjacencyDictionary(graph)
            keys = adj_dict.keys()
            c_edges = []
            used = [[0] * n for _ in range(n)]
            for key in keys:
                x = int(key)
                adj_vertices = [int(v) for v in adj_dict[key]]
                for y in adj_vertices:
                    if used[x][y] == 0:
                        if Vertex.X(c_vertices[x]) < Vertex.X(c_vertices[y]):
                            e = Edge.ByVertices(c_vertices[x], c_vertices[y])
                        else:
                            e = Edge.ByVertices(c_vertices[y], c_vertices[x])
                        
                        orig_edge_index = edge_dict[str(x)+"_"+str(y)]
                        d = Topology.Dictionary(edges[orig_edge_index])
                        e = Topology.SetDictionary(e, d, silent=True)
                        c_edges.append(e)
                        used[x][y] = 1
                        used[y][x] = 1
            new_g = Graph.ByVerticesEdges(c_vertices, c_edges)
            return new_g
        
        def grid_layout_3d(graph, size=1):
            vertices = Graph.Vertices(graph)
            n = len(vertices)        
            edges = Graph.Edges(graph)
            edge_dict = {}

            for i, edge in enumerate(edges):
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                si = Vertex.Index(sv, vertices)
                ei = Vertex.Index(ev, vertices)
                edge_dict[str(si)+"_"+str(ei)] = i
                edge_dict[str(ei)+"_"+str(si)] = i
            c_coords = generate_cubic_matrix(size, n)
            c_vertices = [Vertex.ByCoordinates(list(coord)) for coord in c_coords[:n]]

            for i, c_v in enumerate(c_vertices):
                d = Topology.Dictionary(vertices[i])
                c_v = Topology.SetDictionary(c_v, d, silent=True)
            adj_dict = Graph.AdjacencyDictionary(graph)
            keys = adj_dict.keys()
            
            c_edges = []
            used = [[0] * n for _ in range(n)]
            for key in keys:
                x = int(key)
                adj_vertices = [int(v) for v in adj_dict[key]]
                for y in adj_vertices:
                    if used[x][y] == 0:
                        e = Edge.ByVertices(c_vertices[x], c_vertices[y])
                        orig_edge_index = edge_dict.get(str(x)+"_"+str(y), edge_dict.get(str(y)+"_"+str(x), None))
                        if orig_edge_index:
                            d = Topology.Dictionary(edges[orig_edge_index])
                            e = Topology.SetDictionary(e, d, silent=True)
                            c_edges.append(e)
                            used[x][y] = 1
                            used[y][x] = 1
            new_g = Graph.ByVerticesEdges(c_vertices, c_edges)
            return new_g
    
        def spring_layout_2d(edge_list, iterations=500, k=None, seed=None):
            # Compute the layout of a graph using the Fruchterman-Reingold algorithm
            # with a force-directed 
            
            iterations = max(1, iterations)
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

        def spring_layout_3d(edge_list, iterations=500, k=None, seed=None):
            # Compute the layout of a graph using the Fruchterman-Reingold algorithm
            # with a force-directed layout approach.

            iterations = max(1,iterations)

            adj_matrix = edge_list_to_adjacency_matrix(edge_list)
            # Set the random seed
            if seed is not None:
                np.random.seed(seed)

            # Set the optimal distance between nodes
            if k is None or k <= 0:
                k = np.cbrt(1.0 / adj_matrix.shape[0])  # Adjusted for 3D

            # Initialize the positions of the nodes randomly in 3D
            pos = np.random.rand(adj_matrix.shape[0], 3)

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

        def tree_layout_2d(edge_list,  root_index=0):

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
        
        def tree_layout_3d(edge_list, root_index=0, base_radius=1.0, radius_factor=1.5):
            root, num_nodes = tree_from_edge_list(edge_list, root_index)
            dt = buchheim(root)
            pos = np.zeros((num_nodes, 3))  # Initialize 3D positions
            
            pos[int(dt.tree.node), 0] = dt.x
            pos[int(dt.tree.node), 1] = dt.y
            pos[int(dt.tree.node), 2] = 0  # Root at z = 0
            
            old_roots = [dt]
            new_roots = []
            depth = 1  # Start at depth level 1 for children

            while len(old_roots) > 0:
                new_roots = []
                for temp_root in old_roots:
                    children = temp_root.children
                    num_children = len(children)
                    if num_children > 0:
                        # Increase the radius dynamically based on the number of children
                        dynamic_radius = base_radius + (num_children - 1) * radius_factor * depth
                        
                        angle_step = 2 * np.pi / num_children  # Angle between each child
                        for i, child in enumerate(children):
                            angle = i * angle_step
                            pos[int(child.tree.node), 0] = pos[int(temp_root.tree.node), 0] + dynamic_radius * np.cos(angle)  # X position
                            pos[int(child.tree.node), 1] = pos[int(temp_root.tree.node), 1] + dynamic_radius * np.sin(angle)  # Y position
                            pos[int(child.tree.node), 2] = -dynamic_radius*depth  # Z-coordinate based on depth
                            
                    new_roots.extend(children)
                
                old_roots = new_roots
                depth += 1  # Increment depth for the next level

            pos[:, 1] = np.max(pos[:, 1]) - pos[:, 1]  # Flip y-coordinates if necessary

            return pos

        def radial_layout_2d(edge_list, root_index=0):
            import numpy as np
            from collections import deque
            # Build tree and get layout from Buchheim
            root, num_nodes = tree_from_edge_list(edge_list, root_index)
            dt = buchheim(root)

            # Initialize positions array
            pos = np.zeros((num_nodes, 2))
            pos[int(dt.tree.node)] = [dt.x, dt.y]

            # Efficient tree traversal using a queue
            queue = deque([dt])
            while queue:
                current = queue.popleft()
                for child in current.children:
                    pos[int(child.tree.node)] = [child.x, child.y]
                    queue.append(child)

            # Normalize positions
            pos[:, 0] -= np.min(pos[:, 0])
            pos[:, 1] -= np.min(pos[:, 1])
            pos[:, 0] /= np.max(pos[:, 0])
            pos[:, 1] /= np.max(pos[:, 1])

            # Center the root and scale the x-coordinates
            pos[:, 0] -= pos[root_index, 0]
            pos[:, 0] /= (np.max(pos[:, 0]) - np.min(pos[:, 0]))
            pos[:, 0] *= np.pi * 1.98

            # Convert to polar coordinates
            new_pos = np.zeros_like(pos)
            new_pos[:, 0] = pos[:, 1] * np.cos(pos[:, 0])
            new_pos[:, 1] = pos[:, 1] * np.sin(pos[:, 0])

            return new_pos
        
        def dendrimer_layout_2d(graph, root_index=0, base_radius=1, radius_factor=1.5):
            """
            Given a graph as an adjacency dictionary, this function generates a dendrimer layout
            and returns positions of the nodes in 2D space.
            
            :param graph: dict, adjacency dictionary where keys are node ids and values are sets/lists of neighboring nodes
            :param root_index: int, index of the node to start the layout (default: 0)
            :return: list of positions [x, y] for each node, sorted by node id
            """
            import numpy as np
            
            # Initialize variables
            positions = {}
            visited = set()
            layers = {}

            # Helper function to perform a DFS and organize nodes in layers
            def dfs(node, depth):
                visited.add(node)
                if depth not in layers:
                    layers[depth] = []
                layers[depth].append(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        dfs(neighbor, depth + 1)
            
            # Start DFS from the given root node
            starting_node = list(graph.keys())[root_index]
            dfs(starting_node, 0)
            
            # Perform DFS for all nodes to handle disconnected components
            for node in graph.keys():
                if node not in visited:
                    dfs(node, 0)  # Start a new DFS for each unvisited node

            # Compute positions based on layers
            for depth, nodes in layers.items():
                # Place nodes in a circular arrangement at each layer
                num_nodes = len(nodes)
                angle_step = 2 * np.pi / num_nodes if num_nodes > 0 else 0
                for i, node in enumerate(nodes):
                    angle = i * angle_step
                    x = base_radius*depth*np.cos(angle)
                    y = base_radius*depth*np.sin(angle)
                    positions[node] = (x, y)

            # Sort the positions by node id and return them
            keys = list(positions.keys())
            keys.sort()
            return_values = [list(positions[key]) for key in keys]
            return return_values

        def spherical_layout_3d(edge_list, root_index=0):
            root, num_nodes = tree_from_edge_list(edge_list, root_index)
            dt = buchheim(root)

            # Initialize positions with 3 columns for x, y, z coordinates
            pos = np.zeros((num_nodes, 3))

            # Set initial coordinates and depth for the root node
            pos[int(dt.tree.node), 0] = dt.x
            pos[int(dt.tree.node), 1] = dt.y
            depth = np.zeros(num_nodes)  # To track the depth of each node

            old_roots = [(dt, 0)]  # Store nodes with their depth levels
            new_roots = []

            while len(old_roots) > 0:
                new_roots = []
                for temp_root, current_depth in old_roots:
                    children = temp_root.children
                    for child in children:
                        node_index = int(child.tree.node)
                        pos[node_index, 0] = child.x
                        pos[node_index, 1] = child.y
                        depth[node_index] = current_depth + 1  # Increase depth for children
                    new_roots.extend([(child, current_depth + 1) for child in children])

                old_roots = new_roots

            # Normalize x and y coordinates to a [0, 1] range
            pos[:, 0] = pos[:, 0] - np.min(pos[:, 0])
            pos[:, 1] = pos[:, 1] - np.min(pos[:, 1])

            pos[:, 0] = pos[:, 0] / np.max(pos[:, 0])
            pos[:, 0] = pos[:, 0] - pos[:, 0][root_index]
            
            range_ = np.max(pos[:, 0]) - np.min(pos[:, 0])
            pos[:, 0] = pos[:, 0] / range_

            pos[:, 0] = pos[:, 0] * np.pi * 1.98  # Longitude (azimuthal angle)
            pos[:, 1] = pos[:, 1] / np.max(pos[:, 1])  # Latitude (polar angle)

            # Convert the 2D coordinates to 3D spherical coordinates
            new_pos = np.zeros((num_nodes, 3))  # 3D position array
            base_radius = 1  # Base radius for the first sphere
            radius_increment = 1.3  # Increase in radius per depth level

            # pos[:, 0] is the azimuth angle (longitude) in radians
            # pos[:, 1] is the polar angle (latitude) in radians (adjusted to go from 0 to pi)
            polar_angle = pos[:, 1] * np.pi  # Scaling to go from 0 to pi
            azimuth_angle = pos[:, 0]

            # Calculate the 3D Cartesian coordinates based on depth
            for i in range(num_nodes):
                r = base_radius + depth[i] * radius_increment  # Radius grows with depth
                new_pos[i, 0] = r * np.sin(polar_angle[i]) * np.cos(azimuth_angle[i])  # X = r * sin(θ) * cos(φ)
                new_pos[i, 1] = r * np.sin(polar_angle[i]) * np.sin(azimuth_angle[i])  # Y = r * sin(θ) * sin(φ)
                new_pos[i, 2] = r * np.cos(polar_angle[i])  # Z = r * cos(θ)

            return new_pos

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.Reshape - Error: The input graph is not a valid topologic graph. Returning None.")
            return None
        
        vertices = Graph.Vertices(graph)
        if len(vertices) < 2:
            if not silent:
                print("Graph.Reshape - Warning: The graph has less than two vertices. It cannot be rehsaped. Returning the original input graph.")
            return graph
        
        if 'circ' in shape.lower():
            return circle_layout_2d(graph, radius=size/2, sides=sides)
        elif 'lin' in shape.lower():
            return line_layout_2d(graph, length=size)
        elif 'grid' in shape.lower() and '2d' in shape.lower():
            return grid_layout_2d(graph, size=size)
        elif 'sphere' in shape.lower() and '3d' in shape.lower():
            return sphere_layout_3d(graph, radius=size/2)
        elif 'grid' in shape.lower() and '3d' in shape.lower():
            return grid_layout_3d(graph, size=size)
        elif 'cluster' in shape.lower() and '2d' in shape.lower():
            return cluster_layout_2d(graph, radius=size/2, key=key)
        elif 'cluster' in shape.lower() and '3d' in shape.lower():
            return cluster_layout_3d(graph, radius=size/2, key=key)
        else:
            d = Graph.MeshData(graph)
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
            if 'rad' in shape.lower() and '2d' in shape.lower():
                positions = radial_layout_2d(edges, root_index=root_index)
            elif 'spherical' in shape.lower() and '3d' in shape.lower():
                positions = spherical_layout_3d(edges, root_index=root_index)
            elif 'spring' in shape.lower() and "3d" in shape.lower():
                positions = spring_layout_3d(edges, k=k, seed=seed, iterations=iterations)
            elif 'spring' in shape.lower() and '2d' in shape.lower():
                positions = spring_layout_2d(edges, k=k, seed=seed, iterations=iterations)
            elif 'tree' in shape.lower() and '2d' in shape.lower():
                positions = tree_layout_2d(edges, root_index=root_index)
            elif 'tree' in shape.lower() and '3d' in shape.lower():
                positions = tree_layout_3d(edges, root_index=root_index, base_radius=size/2, radius_factor=factor)
            elif 'dendrimer' in shape.lower() and '2d' in shape.lower():
                positions = dendrimer_layout_2d(Graph.AdjacencyDictionary(graph), root_index=root_index, base_radius=size/2, radius_factor=factor)
            else:
                if not silent:
                    print(f"{shape} is not implemented yet. Please choose from ['circle 2D', 'grid 2D', 'line 2D', 'radial 2D', 'spring 2D', 'tree 2D', 'grid 3D', 'sphere 3D', 'tree 3D']. Returning None.")
                return None
            if len(positions[0]) == 3:
                positions = [[p[0], p[1], p[2]] for p in positions]
            else:
                positions = [[p[0], p[1], 0] for p in positions]
            return Graph.ByMeshData(positions, edges, v_dicts, e_dicts, tolerance=tolerance)

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
    def GraphVizGraph(
        graph,
        device = 'svg_inline', deviceKey=None,
        scale = 1, scaleKey=None,
        directed=False, directedKey=None,
        layout = 'dot', # or circo fdp neato nop nop1 nop2 osage patchwork sfdp twopi
        layoutKey=None,
        rankDir='TB', rankDirKey=None, # or LR, RL, BT
        bgColor='white', bgColorKey=None,
        fontName='Arial', fontNameKey=None,
        fontSize= 12, fontSizeKey=None,
        vertexSep= 0.5, vertexSepKey=None,
        rankSep= 0.5, rankSepKey=None,
        splines='true', splinesKey=None,
        showGraphLabel = False,
        graphLabel='', graphLabelKey=None,
        graphLabelLoc='t', graphLabelLocKey=None,

        showVertexLabel = False,
        vertexLabelPrefix='' , vertexLabelKey=None,
        vertexWidth=0.5, vertexWidthKey=None,
        vertexHeight=0.5, vertexHeightKey=None,
        vertexFixedSize=False, vertexFixedSizeKey=None,
        vertexShape='circle', vertexShapeKey=None,
        vertexStyle='filled', vertexStyleKey=None,
        vertexFillColor='lightgray', vertexFillColorKey=None,
        vertexColor='black', vertexColorKey=None,
        vertexFontColor='black', vertexFontColorKey=None,

        showEdgeLabel = False,
        edgeLabelPrefix='', edgeLabelKey=None,
        edgeColor='black', edgeColorKey=None,
        edgeWidth=1, edgeWidthKey=None,
        edgeStyle='solid', edgeStyleKey=None,
        edgeArrowhead='normal', edgeArrowheadKey=None,
        edgeFontColor='black', edgeFontColorKey=None,
        silent=False
    ):
        """
        Converts the input graph to a GraphViz graph. GraphViz should be installed separately, using your system's package manager.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        device : str, optional
            The output format device, such as 'svg_inline', 'pdf', or 'png'. Default is 'svg_inline'.
        deviceKey : str, optional
            Dictionary key to override the `device` value. Default is None.
        scale : float, optional
            Global scaling factor. Default is 1.
        scaleKey : str, optional
            Dictionary key to override the `scale` per-graph. Default is None.
        directed : bool, optional
            Whether to treat the graph as directed. Default is False.
        directedKey : str, optional
            Dictionary key to override the `directed` flag per-graph. Default is None.
        layout : str, optional
            Layout engine to use. Options include 'dot', 'circo', 'fdp', 'neato', 'osage', 'sfdp', etc. Default is 'dot'.
        layoutKey : str, optional
            Dictionary key to override the `layout` per-graph. Default is None.
        rankDir : str, optional
            Direction of graph ranking. Options: 'TB' (top-bottom), 'LR' (left-right), 'RL', 'BT'. Default is 'TB'.
        rankDirKey : str, optional
            Dictionary key to override `rankDir` per-graph. Default is None.
        bgColor : str, optional
            Background color. Default is 'white'.
        bgColorKey : str, optional
            Dictionary key to override `bgColor`. Default is None.
        fontName : str, optional
            Name of the font to use for all text. Default is 'Arial'.
        fontNameKey : str, optional
            Dictionary key to override `fontName`. Default is None.
        fontSize : int or float, optional
            Size of font in points. Default is 12.
        fontSizeKey : str, optional
            Dictionary key to override `fontSize`. Default is None.
        vertexSep : float, optional
            Minimum separation between vertices. Default is 0.5.
        vertexSepKey : str, optional
            Dictionary key to override `vertexSep`. Default is None.
        rankSep : float, optional
            Separation between ranks. Default is 0.5.
        rankSepKey : str, optional
            Dictionary key to override `rankSep`. Default is None.
        splines : str, optional
            Whether to use spline edges. Can be 'true', 'false', or 'polyline'. Default is 'True'.
        splinesKey : str, optional
            Dictionary key to override `splines`. Default is None.
        showGraphLabel : bool, optional
            Whether to show a label for the whole graph. Default is False.
        graphLabel : str, optional
            Text for the graph label. Default is an empty string.
        graphLabelKey : str, optional
            Dictionary key to override `graphLabel`. Default is None.
        graphLabelLoc : str, optional
            Position of the graph label: 't' (top), 'b' (bottom), 'c' (center). Default is 't'.
        graphLabelLocKey : str, optional
            Dictionary key to override `graphLabelLoc`. Default is None.
        showVertexLabel : bool, optional
            Whether to display vertex labels. Default is False.
        vertexLabelPrefix : str, optional
            Text prefix for vertex labels. Default is empty string.
        vertexLabelKey : str, optional
            Dictionary key used to retrieve label text from vertex dictionary. Default is None.
        vertexWidth : float, optional
            Width of each vertex. Default is 0.5.
        vertexWidthKey : str, optional
            Dictionary key to override `vertexWidth`. Default is None.
        vertexHeight : float, optional
            Height of each vertex. Default is 0.5.
        vertexHeightKey : str, optional
            Dictionary key to override `vertexHeight`. Default is None.
        vertexFixedSize : bool, optional
            Whether vertices should be fixed in size. Default is False.
        vertexFixedSizeKey : str, optional
            Dictionary key to override `vertexFixedSize`. Default is None.
        vertexShape : str, optional
            Shape of the vertex ('circle', 'ellipse', 'box', etc.). Default is 'circle'.
        vertexShapeKey : str, optional
            Dictionary key to override `vertexShape`. Default is None.
        vertexStyle : str, optional
            Style of vertex (e.g., 'filled', 'dashed'). Default is 'filled'.
        vertexStyleKey : str, optional
            Dictionary key to override `vertexStyle`. Default is None.
        vertexFillColor : str, optional
            Fill color for vertices. Default is 'lightgray'.
        vertexFillColorKey : str, optional
            Dictionary key to override `vertexFillColor`. Default is None.
        vertexColor : str, optional
            Border color for vertices. Default is 'black'.
        vertexColorKey : str, optional
            Dictionary key to override `vertexColor`. Default is None.
        vertexFontColor : str, optional
            Font color for vertex labels. Default is 'black'.
        vertexFontColorKey : str, optional
            Dictionary key to override `vertexFontColor`. Default is None.
        showEdgeLabel : bool, optional
            Whether to display edge labels. Default is False.
        edgeLabelPrefix : str, optional
            Text prefix for edge labels. Default is empty string.
        edgeLabelKey : str, optional
            Dictionary key used to retrieve label text from edge dictionary. Default is None.
        edgeColor : str, optional
            Color of edges. Default is 'black'.
        edgeColorKey : str, optional
            Dictionary key to override `edgeColor`. Default is None.
        edgeWidth : float, optional
            Width (thickness) of edges. Default is 1.
        edgeWidthKey : str, optional
            Dictionary key to override `edgeWidth`. Default is None.
        edgeStyle : str, optional
            Style of the edge line (e.g., 'solid', 'dashed'). Default is 'solid'.
        edgeStyleKey : str, optional
            Dictionary key to override `edgeStyle`. Default is None.
        edgeArrowhead : str, optional
            Arrowhead style for directed edges. Default is 'normal'.
        edgeArrowheadKey : str, optional
            Dictionary key to override `edgeArrowhead`. Default is None.
        edgeFontColor : str, optional
            Font color for edge labels. Default is 'black'.
        edgeFontColorKey : str, optional
            Dictionary key to override `edgeFontColor`. Default is None.
        overwrite : bool, optional
            If True, overwrites existing files at the given path. Default is False.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        graphviz.graphs.Graph
            The created GraphViz graph.
        """

        import os
        import warnings

        try:
            from graphviz import Digraph
            from graphviz import Graph as Udgraph

        except:
            print("Graph - Installing required graphviz library.")
            try:
                os.system("pip install graphviz")
            except:
                os.system("pip install graphviz --user")
            try:
                from graphviz import Digraph
                from graphviz import Graph as Udgraph
                print("Graph - graphviz library installed correctly.")
            except:
                warnings.warn("Graph - Error: Could not import graphviz.")

        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.GraphVizGraph - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        # Set Graph-level attributes
        def get_attr(dict, keyName, default):
            if keyName:
                return Dictionary.ValueAtKey(dict, keyName, default)
            return default

        graph_dict = Topology.Dictionary(graph)

        is_directed =  get_attr(graph_dict, directedKey, directed)
        if is_directed:
            gv_graph = Digraph()
        else:
            gv_graph = Udgraph()
        
        if showGraphLabel:
            label_value = get_attr(graph_dict, graphLabelKey, graphLabel)
        else:
            label_value = ''
        gv_graph.attr(
            layout = get_attr(graph_dict, layoutKey, layout),
            rankdir=get_attr(graph_dict, rankDirKey, rankDir),
            bgcolor=get_attr(graph_dict, bgColorKey, bgColor),
            fontname=get_attr(graph_dict, fontNameKey, fontName),
            fontsize=get_attr(graph_dict, fontSizeKey, str(fontSize)),
            vertexsep=get_attr(graph_dict, vertexSepKey, str(vertexSep)),
            ranksep=get_attr(graph_dict, rankSepKey, str(rankSep)),
            splines=get_attr(graph_dict, splinesKey, splines),
            label=label_value,
            labelloc=get_attr(graph_dict, graphLabelLocKey, graphLabelLoc),
            device = get_attr(graph_dict, deviceKey, device)
        )

        # Get the Vertices and Edges from the Topologic Graph
        mesh_data = Graph.MeshData(graph)

        # Set Vertex Attributes
        verts = mesh_data['vertices']
        vert_dicts = mesh_data['vertexDictionaries']
        for i, v in enumerate(verts):
            v_dict = vert_dicts[i]
            if showVertexLabel:
                label_value = get_attr(v_dict, vertexLabelKey, f"{vertexLabelPrefix}{i}")
            else:
                label_value = ''
            
            fixed_size_value = get_attr(v_dict, vertexFixedSizeKey, vertexFixedSize)

            if fixed_size_value:
                fixed_size_value = "True"
            else:
                fixed_size_value = "False"
            
            if "nop" in get_attr(graph_dict, layoutKey, layout):
                pos_value = f"{v[0]*scale},{v[1]*scale}!"
            else:
                pos_value = ""
            gv_graph.node(
                str(i),
                pos = pos_value,
                label= label_value,
                shape=get_attr(v_dict, vertexShapeKey, vertexShape),
                width=str(get_attr(v_dict, vertexWidthKey, vertexWidth)),
                height=str(get_attr(v_dict, vertexHeightKey, vertexHeight)),
                fixedSize=fixed_size_value,
                style=get_attr(v_dict, vertexStyleKey, vertexStyle),
                fillcolor=get_attr(v_dict, vertexFillColorKey, vertexFillColor),
                color=get_attr(v_dict, vertexColorKey, vertexColor),
                fontcolor=get_attr(v_dict, vertexFontColorKey, vertexFontColor)
            )
        
        # Set Edge attributes
        edges = mesh_data['edges']
        edge_dicts = mesh_data['edgeDictionaries']
        for i, e in enumerate(edges):
            sid = e[0]
            eid = e[1]

            e_dict = edge_dicts[i]
            if showEdgeLabel:
                label_value = get_attr(e_dict, edgeLabelKey, f"{edgeLabelPrefix}{i}")
            else:
                label_value = ''

            gv_graph.edge(
                str(sid),
                str(eid),
                label= label_value,
                color= get_attr(e_dict, edgeColorKey, edgeColor),
                penwidth=str(get_attr(e_dict, edgeWidthKey, edgeWidth)),
                style= get_attr(e_dict, edgeStyleKey, edgeStyle),
                arrowhead= get_attr(e_dict, edgeArrowheadKey, edgeArrowhead),
                fontcolor= get_attr(e_dict, edgeFontColorKey, edgeFontColor),
            )

        return gv_graph

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
    def HasseDiagram(topology, types=["vertex", "edge", "wire", "face", "shell", "cell", "cellComplex"], topDown: bool = False, minDistance: float=0.1, vertexLabelKey: str="label", vertexTypeKey: str="type", vertexColorKey: str="color", colorScale: str="viridis", storeBREP: bool = False, tolerance: float=0.0001, silent: bool=False):
        """
            Constructs a Hasse diagram from the input topology as a directed graph. See: https://en.wikipedia.org/wiki/Hasse_diagram
            Vertices represent topologies (vertices, edges, wires, faces, shells, cells, cellComplexes).
            Edges represent inclusion (e.g. vertex ⊂ edge, edge ⊂ wire).

            Parameters
            ----------
            topology : topologic_core.Topology
                The input topology
            types : optional, list
                The list of topology types that you wish to encode in the Hasse diagram.
                This list must be ordered according to topologic_core's class hierarchy.
                If you are not interested in representing some topology types. These can be omitted.
                The default is:
                ["vertex", "edge", "wire", "face", "shell", "cell", "cellComplex"].
            topDown : bool , optional
                If set to True, the graph edges are directed from topologies to their subtopologies.
                Otherwise, they are directed from topologies to their supertopologies. Default is False. 
            minDistance : float , optional
                The desired minimum distance between the vertices of the graph. Default is 0.1.
            vertexLabelKey: str , optional
                The desired vertex dictionary key under which to store a unique label (of the form Type_Index). Default is "label".
            vertexTypeKey: str , optional
                The desired vertex dictionary key under which to store the topology type (e.g. "vertex", "edge", "wire"). Default is "type".
            vertexColorKey: str , optional
                The desired vertex dictionary key under which to store the topology color. Default is "color".
            colorScale : str , optional
                The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
                In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
            storeBREP : bool , optional
                If set to True, store the BRep of the topology in its representative vertex. Default is False.
            tolerance : float
                The desired tolerance. Default is 0.0001.
            silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.

            Returns
            -------
            topologic_core.Graph
                The created Hesse diagram graph.

            """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        from topologicpy.Topology import Topology

        def label(topo, index):
            cls = Topology.TypeAsString(topo)
            return f"{cls}_{index}"
        
        def collect_topologies(topology, topo_types):
            """
            Returns a dict of all sub-topologies by dimension.
            """
            topo_by_type = {}
            for sub_type in topo_types:
                topo_by_type[sub_type] = Topology.SubTopologies(topology, subTopologyType=sub_type)
            return topo_by_type
        
        if not Topology.IsInstance(topology, "topology"):
            if not silent:
                print("Graph.HasseDiagram - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if minDistance <= tolerance:
            if not silent:
                print("Graph.HasseDiagram - Error: The input minDistance parameter cannot be less than the input tolerance parameter. Returning None.")
            return None
        types = [t.lower() for t in types]
        for type in types:
            if type not in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex"]:
                if not silent:
                    print("Graph.HasseDiagram - Error: Unknown type found in the types input parameter. Returning None.")
                return None

        topology_type = Topology.TypeAsString(topology).lower()
        try:
            sub_types = types[:types.index(topology_type)]
        except:
            sub_types = types
        topo_by_type = collect_topologies(topology, sub_types)
        all_topos = []
        topo_ids = {}
        index = 0

        # Flatten and assign unique labels
        for sub_type in sub_types:
            color = Color.AnyToHex(Color.ByValueInRange(float(types.index(sub_type)), minValue=0, maxValue=6, colorScale=colorScale))
            lbl_index = 1
            for t in topo_by_type[sub_type]:
                lbl = label(t, lbl_index)
                d = Topology.Dictionary(t)
                d = Dictionary.SetValuesAtKeys(d, [vertexLabelKey, vertexTypeKey, vertexColorKey], [lbl, sub_type, color])
                t = Topology.SetDictionary(t, d)
                all_topos.append(t)
                topo_ids[lbl] = index
                index += 1
                lbl_index += 1

        # Create graph vertices
        graph_vertices = [Topology.Centroid(_) for _ in all_topos]

        # Add dictionaries to each vertex
        for i, t in enumerate(all_topos):
            d = Topology.Dictionary(t)
            if storeBREP == True:
                d = Dictionary.SetValueAtKey(d,"brep", Topology.BREPString(t))
            graph_vertices[i] = Topology.SetDictionary(graph_vertices[i], d)
        
        graph_vertices = Vertex.Separate(graph_vertices, minDistance= minDistance, tolerance=tolerance)
        # Build edges of Hasse diagram
        graph_edges = []
        for parent_type in sub_types[1:]:
            for parent in topo_by_type[parent_type]:
                parent_label = Dictionary.ValueAtKey(Topology.Dictionary(parent), vertexLabelKey)
                children = Topology.SubTopologies(parent, subTopologyType=types[types.index(parent_type) - 1])
                for child in children:
                    child_label = Dictionary.ValueAtKey(Topology.Dictionary(child), vertexLabelKey)
                    child_id = topo_ids.get(child_label)
                    parent_id = topo_ids.get(parent_label)
                    if child_id is not None and parent_id is not None:
                        if topDown:
                            sv = graph_vertices[parent_id]
                            ev = graph_vertices[child_id]
                        else:
                            sv = graph_vertices[child_id]
                            ev = graph_vertices[parent_id]
                        graph_edges.append(Edge.ByVertices(sv, ev, tolerance=tolerance, silent=silent))

        return_graph = Graph.ByVerticesEdges(graph_vertices, graph_edges)
        return_graph = Graph.SetDictionary(return_graph, Topology.Dictionary(topology))
        return return_graph


    @staticmethod
    def HopperKernel(graphA,
                    graphB,
                    key: str = None,
                    maxHops: int = 3,
                    decay: float = 0.5,
                    normalize: bool = True,
                    mantissa: int = 6,
                    silent: bool = False):
        """
        Returns the Graph Hopper kernel between two graphs. This kernel compares hop-wise shortest-path
        frontiers between nodes in two graphs using an automatically selected node-level kernel:
        numeric Radial Basis Function (RBF) if the `key` values are numeric, categorical (delta) if the `key` values are non-numeric,
        or vertex degree if `key` is None or missing. See Vishwanathan et al. (2010) for path-based graph kernels.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        key : str , optional
            The vertex dictionary key used to derive node features. If numeric for most vertices, a numeric
            RBF node kernel is used; if non-numeric, a delta node kernel is used. If None or missing, the
            vertex degree is used as a numeric feature. Default is None.
        maxHops : int , optional
            The maximum shortest-path hop distance to consider. Default is 3.
        decay : float , optional
            A per-hop geometric decay factor in the range (0, 1]. Default is 0.5.
        normalize : bool , optional
            If True, the kernel is cosine-normalized using self-kernel values so that identical graphs score 1.0.
            The default is True.
        mantissa : int , optional
            The number of decimal places to which to round the result. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The (optionally normalized) Graph Hopper kernel value rounded to the specified mantissa.
        """
        from math import sqrt, exp
        from statistics import median
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        # Validate graphs
        if not Topology.IsInstance(graphA, "Graph"):
            if not silent: print("Graph.HopperKernel - Error: The input graphA parameter is not a valid topologic graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "Graph"):
            if not silent: print("Graph.HopperKernel - Error: The input graphB parameter is not a valid topologic graph. Returning None.")
            return None

        # Clamp/clean params
        if maxHops < 0: maxHops = 0
        try:
            decay = float(decay)
        except Exception:
            decay = 0.5
        if decay <= 0: decay = 1.0  # no decay if mis-set

        # Helpers
        def _vertices_and_index(G):
            V = Graph.Vertices(G) or []
            vidx = {v: i for i, v in enumerate(V)}
            return V, vidx

        def _adj_lists(G, vidx):
            adj = {i: [] for i in vidx.values()}
            for v, i in vidx.items():
                nbrs = Graph.AdjacentVertices(G, v) or []
                adj[i] = sorted(vidx[n] for n in nbrs if n in vidx and n is not v)
            return adj

        def _degree_features(adj):
            # one-dimensional numeric feature as a tuple
            return {i: (float(len(neigh)),) for i, neigh in adj.items()}

        def _collect_key_values(G, V, vidx):
            # returns dict idx -> raw value or None
            vals = {}
            if key is None:
                return vals, 0, 0  # empty means fallback later
            found = 0
            nonnull = 0
            for v in V:
                d = Topology.Dictionary(v)
                val = Dictionary.ValueAtKey(d, key)
                if val is not None:
                    nonnull += 1
                vals[vidx[v]] = val
                found += 1
            return vals, found, nonnull

        def _infer_feature_mode(valsA, valsB):
            # Decide numeric vs label from available non-null values
            samples = []
            for d in (valsA, valsB):
                for val in d.values():
                    if val is not None:
                        samples.append(val)
            if not samples:
                return "degree"  # fallback
            numeric = 0
            total = 0
            for x in samples:
                total += 1
                try:
                    float(x)
                    numeric += 1
                except Exception:
                    pass
            frac = (numeric / total) if total else 0.0
            return "numeric" if frac >= 0.8 else "label"

        def _features_from_key(mode, vals, adj):
            if mode == "numeric":
                # coerce to float, missing -> 0.0
                feats = {}
                for i in adj:
                    v = vals.get(i, None)
                    try:
                        feats[i] = (float(v),)
                    except Exception:
                        feats[i] = (0.0,)
                return feats
            if mode == "label":
                labs = {}
                for i in adj:
                    v = vals.get(i, None)
                    labs[i] = str(v) if v is not None else "__MISSING__"
                return labs
            # degree fallback
            return _degree_features(adj)

        def _shells_by_hops(adj, max_hops):
            # For each root, BFS layers up to max_hops. Returns list of lists of sets.
            n = len(adj)
            shells = [ [set() for _ in range(max_hops+1)] for _ in range(n) ]
            for root in range(n):
                visited = set([root])
                shells[root][0].add(root)
                frontier = [root]
                for h in range(1, max_hops+1):
                    new_frontier = []
                    for u in frontier:
                        for w in adj[u]:
                            if w not in visited:
                                visited.add(w)
                                new_frontier.append(w)
                    if not new_frontier:
                        break
                    shells[root][h].update(new_frontier)
                    frontier = new_frontier
            return shells

        def _pairwise_median_sigma(pool):
            # pool is a list of numeric tuples
            vals = []
            m = len(pool)
            for i in range(m):
                xi = pool[i]
                for j in range(i+1, m):
                    xj = pool[j]
                    s = 0.0
                    for a, b in zip(xi, xj):
                        d = a - b
                        s += d*d
                    vals.append(sqrt(s))
            if not vals:
                return 1.0
            med = median(vals)
            return med if med > 0 else 1.0

        # If Key is None, encode the degree centrality
        if key == None:
            _ = Graph.DegreeCentrality(graphA, key="_dc_")
            _ = Graph.DegreeCentrality(graphB, key="_dc_")
            key = "_dc_"

        # Indexing and adjacency
        VA, idxA = _vertices_and_index(graphA)
        VB, idxB = _vertices_and_index(graphB)
        adjA = _adj_lists(graphA, idxA)
        adjB = _adj_lists(graphB, idxB)

        # Build features according to key and data type
        valsA, _, _ = _collect_key_values(graphA, VA, idxA)
        valsB, _, _ = _collect_key_values(graphB, VB, idxB)
        mode = _infer_feature_mode(valsA, valsB) if key is not None else "degree"

        featsA = _features_from_key(mode, valsA, adjA)
        featsB = _features_from_key(mode, valsB, adjB)

        # Prepare shells
        shellsA = _shells_by_hops(adjA, maxHops)
        shellsB = _shells_by_hops(adjB, maxHops)

        # Node kernels
        hopNormalize = True  # fixed to reduce size bias

        if mode == "label":
            def knode_AB(x, y):
                return 1.0 if featsA[x] == featsB[y] else 0.0
            def knode_AA(x, y):
                return 1.0 if featsA[x] == featsA[y] else 0.0
            def knode_BB(x, y):
                return 1.0 if featsB[x] == featsB[y] else 0.0
        else:
            # numeric RBF; estimate sigma from pooled features
            pool = list(featsA.values()) + list(featsB.values())
            sigma = _pairwise_median_sigma(pool)
            denom = 2.0 * (sigma * sigma)
            def _rbf(fX, fY):
                s = 0.0
                for a, b in zip(fX, fY):
                    d = a - b
                    s += d*d
                return exp(-s / denom) if denom > 0 else 0.0
            def knode_AB(x, y):
                return _rbf(featsA[x], featsB[y])
            def knode_AA(x, y):
                return _rbf(featsA[x], featsA[y])
            def knode_BB(x, y):
                return _rbf(featsB[x], featsB[y])

        # Core kernel accumulation
        def _kernel_raw(shellsX, shellsY, knode_xy):
            total = 0.0
            nX = len(shellsX); nY = len(shellsY)
            for h in range(maxHops+1):
                hop_sum = 0.0
                for u in range(nX):
                    Su = shellsX[u][h]
                    if not Su: 
                        continue
                    for v in range(nY):
                        Sv = shellsY[v][h]
                        if not Sv:
                            continue
                        inner = 0.0
                        for x in Su:
                            for y in Sv:
                                inner += knode_xy(x, y)
                        if hopNormalize:
                            denom_local = float(len(Su) * len(Sv))
                            if denom_local > 0:
                                inner /= denom_local
                        hop_sum += inner
                total += (decay ** h) * hop_sum
            return total

        K_ab = _kernel_raw(shellsA, shellsB, knode_AB)

        if not normalize:
            return round(float(K_ab), mantissa)

        # Self-kernels for normalization
        K_aa = _kernel_raw(shellsA, shellsA, knode_AA)
        K_bb = _kernel_raw(shellsB, shellsB, knode_BB)

        denom_norm = (K_aa * K_bb) ** 0.5
        value = float(K_ab) / denom_norm if denom_norm > 0 else 0.0
        # Cleanup keys and values that we encoded:
        if key == "_dc_": 
            _ = [Topology.SetDictionary(v, Dictionary.RemoveKey(Topology.Dictionary(v), "_dc_")) for v in Graph.Vertices(graphA)]
            _ = [Topology.SetDictionary(v, Dictionary.RemoveKey(Topology.Dictionary(v), "_dc_")) for v in Graph.Vertices(graphB)]
        return round(value, mantissa)

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
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            if Vertex.Distance(vertex, ev) <= tolerance:
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
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def Integration(
        graph,
        weightKey: str = "length",
        normalize: bool = False,
        nxCompatible: bool = True,
        key: str = "integration",
        colorKey: str = "in_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        This is an alias method for Graph.ClosenessCentrality. Returns the integration (closeness centrality) of the input graph. The order of the returned
        list matches the order of Graph.Vertices(graph).
        See: https://en.wikipedia.org/wiki/Closeness_centrality

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        weightKey : str , optional
            If specified, this edge attribute will be used as the distance weight when
            computing shortest paths. If set to a name containing "Length" or "Distance",
            it will be mapped to "length".
            Note: Graph.NetworkXGraph automatically provides a "length" attribute on all edges.
        normalize : bool , optional
            If True, the returned values are rescaled to [0, 1]. Otherwise raw values
            from NetworkX (optionally using the improved formula) are returned.
        nxCompatible : bool , optional
            If True, use NetworkX's wf_improved scaling (Wasserman and Faust).
            For single-component graphs it matches the original formula.
        key : str , optional
            The dictionary key under which to store the closeness centrality score.
        colorKey : str , optional
            The dictionary key under which to store a color derived from the score.
        colorScale : str , optional
            Plotly color scale name (e.g., "viridis", "plasma").
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list[float]
            Integration (closeness centrality) values for vertices in the same order as Graph.Vertices(graph).
        """
        return Graph.ClosenessCentrality(
        graph,
        weightKey=weightKey,
        normalize=normalize,
        nxCompatible=nxCompatible,
        key=key,
        colorKey=colorKey,
        colorScale=colorScale,
        mantissa=mantissa,
        tolerance=tolerance,
        silent=silent
        )
    
    @staticmethod
    def IsBipartite(graph, tolerance=0.0001):
        """
        Returns True if the input graph is bipartite. Returns False otherwise. See https://en.wikipedia.org/wiki/Bipartite_graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def IsConnected(graph, vertexA, vertexB, silent: bool = False):
        """
        Returns True if the two input vertices are directly connected by an edge. Returns False otherwise.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the input vertices are connected by an edge. False otherwise.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.IsConnected - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        
        if not Topology.IsInstance(vertexA, "vertex"):
            if not silent:
                print("Graph.IsConnected - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        
        if not Topology.IsInstance(vertexB, "vertex"):
            if not silent:
                print("Graph.IsConnected - Error: The input vertexB parameter is not a valid vertex. Returning None.")
            return None
        
        if vertexA == vertexB:
            if not silent:
                print("Graph.IsConnected - Warrning: The two input vertices are the same vertex. Returning False.")
            return False
        shortest_path = Graph.ShortestPath(graph, vertexA, vertexB)
        if shortest_path == None:
            return False
        else:
            edges = Topology.Edges(shortest_path)
            if len(edges) == 1:
                return True
            else:
                return False
    
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
        _ = graph.IsolatedVertices(vertices) # Hook to Core
        return vertices

    @staticmethod
    def IsTree(graph):
        """
        Returns True if the input graph has a hierarchical tree-like structure. Returns False otherwise.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.

        Returns
        -------
        bool
            True if the input graph has a hierarchical tree-like structure. False otherwise.

        """

        adj_dict = Graph.AdjacencyDictionary(graph, includeWeights=False)
        # Helper function for Depth-First Search (DFS)
        def dfs(node, parent, visited):
            visited.add(node)
            for neighbor in adj_dict[node]:
                if neighbor not in visited:
                    if not dfs(neighbor, node, visited):
                        return False
                elif neighbor != parent:
                    # A cycle is detected
                    return False
            return True

        # Initialize visited set
        visited = set()

        # Start DFS from the first node in the graph
        start_node = next(iter(adj_dict))  # Get an arbitrary starting node
        if not dfs(start_node, None, visited):
            return False

        # Check if all nodes were visited (the graph is connected)
        if len(visited) != len(adj_dict):
            return False

        return True

    @staticmethod
    def JSONLDData(graph, context=None, verticesKey="nodes", edgesKey="edges", labelKey="label", sourceKey="source", targetKey="target", categoryKey="category", xKey="x", yKey="y", zKey="z", mantissa=6):
        """
        Exports the Graph to a JSON-LD representation.

        Parameters
        ----------
        graph : topologic_core.Graph
            The TopologicPy Graph object to export.
        context : dict, optional
            A JSON-LD context mapping TopologicPy keys to IRIs (e.g., schema.org, geo, etc.).
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        labelKey : str , optional
            The desired key name to use for label. Default is "label".
        sourceKey : str , optional
            The desired key name to use for source. Default is "source".
        targetKey : str , optional
            The desired key name to use for target. Default is "target".
        categoryKey : str , optional
            The desired key name to use for lcategoryabel. Default is "category".
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "z".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        
        Returns
        -------
        dict
            A JSON-LD representation of the graph.
        """
        from topologicpy.Graph import Graph
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if context is None:
            context = {
                labelKey: "rdfs:"+labelKey,
                categoryKey: "schema:"+categoryKey,
                xKey: "schema:"+xKey,
                yKey: "schema:"+yKey,
                zKey: "schema:"+zKey,
                "Graph": "https://topologic.app/vocab#Graph",
                "Vertex": "https://topologic.app/vocab#Vertex",
                "Edge": "https://topologic.app/vocab#Edge"
            }

        
        # Helper: Serialize a Vertex
        def serialize_vertex(vertex):
            props = Dictionary.PythonDictionary(Topology.Dictionary(vertex))
            coords = Vertex.Coordinates(vertex, mantissa=mantissa)
            props.update({
                "@type": "Vertex",
                xKey: coords[0],
                yKey: coords[1],
                zKey: coords[2],
            })
            props["@id"] = Topology.UUID(vertex)
            return props

        # Helper: Serialize an Edge
        def serialize_edge(edge, tp_edge, edge_dict, s_vertices):
            sv = edge[0]
            ev = edge[1]
            edge_dict.update({
                "@type": "Edge",
                sourceKey: s_vertices[sv]["@id"],
                targetKey: s_vertices[ev]["@id"]
            })
            edge_dict["@id"] = Topology.UUID(tp_edge)
            return edge_dict

        # Assemble graph
        jsonld = {
            "@context": context,
            "@id": Topology.UUID(graph),
            "@type": "Graph",
            verticesKey: [],
            edgesKey: []
        }

        vertices = Graph.Vertices(graph)
        tp_edges = Graph.Edges(graph)
        mesh_data = Graph.MeshData(graph)
        m_edges = mesh_data['edges']
        edge_dicts = mesh_data['edgeDictionaries']

        s_vertices = []
        for v in vertices:
            sv = serialize_vertex(v)
            s_vertices.append(sv)
            jsonld[verticesKey].append(sv)

        for i, tp_edge in enumerate(tp_edges):
            se = serialize_edge(m_edges[i], tp_edge, edge_dicts[i], s_vertices)
            jsonld[edgesKey].append(se)

        return jsonld

    @staticmethod
    def JSONLDString(graph, context=None, verticesKey="nodes", edgesKey="edges", labelKey="label", sourceKey="source", targetKey="target", categoryKey="category", xKey="x", yKey="y", zKey="z", indent=2, sortKeys=False, mantissa=6):
        """
        Converts the input graph into a JSON-LD string.

        Parameters
        ----------
        graph : topologic_core.Graph
            The TopologicPy Graph object to export.
        context : dict, optional
            A JSON-LD context mapping TopologicPy keys to IRIs (e.g., schema.org, geo, etc.)
        context : dict, optional
            A JSON-LD context mapping TopologicPy keys to IRIs (e.g., schema.org, geo, etc.).
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        labelKey : str , optional
            The desired key name to use for label. Default is "label".
        sourceKey : str , optional
            The desired key name to use for source. Default is "source".
        targetKey : str , optional
            The desired key name to use for target. Default is "target".
        categoryKey : str , optional
            The desired key name to use for lcategoryabel. Default is "category".
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "z".
        indent : int , optional
            The desired indent. Default is 2.
        sortKeys : bool , optional
            If set to True, the keys will be sorted. Otherwise, they won't be. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        Returns
        -------
        dict
            A JSON-LD representation of the graph.
        """
        import json
        jsonld_data = Graph.JSONLDData(graph,
                                context=context,
                                verticesKey=verticesKey,
                                edgesKey=edgesKey,
                                labelKey=labelKey,
                                sourceKey=sourceKey,
                                targetKey=targetKey,
                                categoryKey=categoryKey,
                                xKey=xKey,
                                yKey=yKey,
                                zKey=zKey,
                                mantissa=mantissa)
        return json.dumps(jsonld_data, indent=indent, sort_keys=sortKeys)

    @staticmethod
    def ExportToJSONLD(graph,
                       path,
                       context=None,
                       verticesKey="nodes",
                       edgesKey="edges",
                       labelKey="label",
                       sourceKey="source",
                       targetKey="target",
                       categoryKey="category",
                       xKey="x",
                       yKey="y",
                       zKey="z",
                       indent=2,
                       sortKeys=False,
                       mantissa=6,
                       overwrite=False):
        """
        Exports the input graph to a JSON file.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        path : str
            The path to the JSON file.
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "z".
        indent : int , optional
            The desired amount of indent spaces to use. Default is 4.
        sortKeys : bool , optional
            If set to True, the keys will be sorted. Otherwise, they won't be. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. Default is False.

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
            print("Graph.ExportToJSONLD - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        f = None
        try:
            if overwrite == True:
                f = open(path, "w")
            else:
                f = open(path, "x") # Try to create a new File
        except:
            raise Exception("Graph.ExportToJSONLD - Error: Could not create a new file at the following location: "+path)
        if (f):
            jsonld_data = Graph.JSONLDData(graph,
                                context=context,
                                verticesKey=verticesKey,
                                edgesKey=edgesKey,
                                labelKey=labelKey,
                                sourceKey=sourceKey,
                                targetKey=targetKey,
                                categoryKey=categoryKey,
                                xKey=xKey,
                                yKey=yKey,
                                zKey=zKey,
                                mantissa=mantissa)
            if jsonld_data != None:
                json.dump(jsonld_data, f, indent=indent, sort_keys=sortKeys)
                f.close()
                return True
            else:
                f.close()
                return False
        return False

    @staticmethod
    def JSONData(graph,
                 propertiesKey: str = "properties",
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
        propertiesKey : str , optional
            The desired key name to call the graph properties. Default is "properties".
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        sourceKey : str , optional
            The dictionary key used to store the source vertex. Default is "source".
        targetKey : str , optional
            The dictionary key used to store the target vertex. Default is "target".
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "z".
        geometryKey : str , optional
            The desired key name to use for geometry. Default is "brep".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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

        graph_d = Dictionary.PythonDictionary(Topology.Dictionary(graph))
        vertices = Graph.Vertices(graph)
        j_data = {}
        j_data[propertiesKey] = graph_d
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
        Converts the input graph into a JSON string.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        verticesKey : str , optional
            The desired key name to call vertices. Default is "vertices".
        edgesKey : str , optional
            The desired key name to call edges. Default is "edges".
        vertexLabelKey : str , optional
            If set to a valid string, the vertex label will be set to the value at this key. Otherwise it will be set to Vertex_XXXX where XXXX is a sequential unique number.
            Note: If vertex labels are not unique, they will be forced to be unique.
        edgeLabelKey : str , optional
            If set to a valid string, the edge label will be set to the value at this key. Otherwise it will be set to Edge_XXXX where XXXX is a sequential unique number.
            Note: If edge labels are not unique, they will be forced to be unique.
        xKey : str , optional
            The desired key name to use for x-coordinates. Default is "x".
        yKey : str , optional
            The desired key name to use for y-coordinates. Default is "y".
        zKey : str , optional
            The desired key name to use for z-coordinates. Default is "z".
        indent : int , optional
            The desired amount of indent spaces to use. Default is 4.
        sortKeys : bool , optional
            If set to True, the keys will be sorted. Otherwise, they won't be. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

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
    def Kernel(graphA,
               graphB,
               method: str = "WL",
               key: str = None,
               iterations: int = 2,
               maxHops: int = 3,
               decay: float = 0.5,
               normalize: bool = True,
               mantissa: int = 6,
               silent: bool = False,
               **kwargs):
        """
        Returns a graph-to-graph kernel value using the selected method. This is a
        convenience dispatcher over specific kernel implementations (e.g., WL and Hopper).

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        method : str , optional
            The kernel method to use. Supported values: "WL" (Weisfeiler–Lehman),
            "Hopper" (Graph Hopper). The default is "WL".
        key : str , optional
            A vertex dictionary key used by the selected method to derive node labels/features.
            For "WL", if None the vertex degree is used as the initial label. For "Hopper",
            if None or missing, the vertex degree is used as a numeric feature. Default is None.
        iterations : int , optional
            WL-specific parameter: number of WL iterations. Ignored by other methods. Default is 2.
        maxHops : int , optional
            Hopper-specific parameter: maximum shortest-path hop distance. Ignored by other methods. Default is 3.
        decay : float , optional
            Hopper-specific parameter: per-hop geometric decay in (0, 1]. Ignored by other methods. Default is 0.5.
        normalize : bool , optional
            If True, returns a normalized kernel score in [0, 1] when supported by the method. Default is True.
        mantissa : int , optional
            Number of decimal places for rounding the returned value. Default is 6.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.
        **kwargs
            Reserved for future method-specific options; ignored by current implementations.

        Returns
        -------
        float
            The kernel value computed by the selected method, rounded to `mantissa`,
            or None if inputs are invalid or the method is unsupported.

        Notes
        -----
        - "WL" calls `Graph.WLKernel(graphA, graphB, key=..., iterations=..., normalize=..., mantissa=...)`.
        - "Hopper" calls `Graph.HopperKernel(graphA, graphB, key=..., maxHops=..., decay=..., normalize=..., mantissa=...)`.
        - Method selection is case-insensitive and tolerates common aliases for WL.
        """
        # Normalize method string and map aliases
        m = (method or "WL").strip().lower()
        if m in ("wl", "weisfeiler", "weisfeiler-lehman", "weisfeiler_lehman"):
            return Graph.WLKernel(
                graphA=graphA,
                graphB=graphB,
                key=key,
                iterations=iterations,
                normalize=normalize,
                mantissa=mantissa,
                silent=silent
            )
        elif m in ("hopper", "graphhopper", "graph_hopper"):
            return Graph.HopperKernel(
                graphA=graphA,
                graphB=graphB,
                key=key,
                maxHops=maxHops,
                decay=decay,
                normalize=normalize,
                mantissa=mantissa,
                silent=silent
            )
        else:
            if not silent:
                print(f'Graph.Kernel - Error: Unsupported method "{method}". '
                    f'Supported methods are "WL" and "Hopper". Returning None.')
            return None

    @staticmethod
    def KHopsSubgraph(
        graph,
        vertices: list,
        k: int = 1,
        direction: str = "both",
        silent: bool = False,
    ):
        """
        Returns a subgraph consisting of the k-hop neighborhood around the input list of seed vertices.

        Parameters
        ----------
        graph : topologicpy.Graph
            The input graph.
        vertices : list
            The input list of seed vertices.
        k : int, optional
            Number of hops. Default is 1.
        direction : str, optional
            'both', 'out', or 'in'. Default 'both'.
        silent : bool, optional
            Suppress warnings/errors. Default False.

        Returns
        -------
        topologicpy.Graph or None
            The resulting subgraph, or None on error.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        # ---- validate inputs ----
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.KHopsSubgraph - Error: The input graph parameter is not a valid graph. Returning None.")
            return None

        if not isinstance(vertices, list):
            if not silent:
                print("Graph.KHopsSubgraph - Error: The input vertices parameter is not a valid list. Returning None.")
            return None

        graph_vertices = Graph.Vertices(graph)
        if not graph_vertices:
            if not silent:
                print("Graph.KHopsSubgraph - Error: The input graph does not contain any vertices. Returning None.")
            return None

        # Keep only valid vertex objects
        seed_vertices = [v for v in vertices if Topology.IsInstance(v, "vertex")]
        if not seed_vertices:
            if not silent:
                print("Graph.KHopsSubgraph - Error: The input vertices list does not contain any valid vertices. Returning None.")
            return None

        # ---- map seeds to vertex indices (prefer identity; fallback to list.index) ----
        id_to_index = {Topology.UUID(v): i for i, v in enumerate(graph_vertices)}
        seed_indices = []
        for sv in seed_vertices:
            idx = id_to_index.get(Topology.UUID(sv))
            if idx is None:
                try:
                    idx = graph_vertices.index(sv)  # fallback if same object not used
                except ValueError:
                    idx = None
            if idx is not None:
                seed_indices.append(idx)

        if not seed_indices:
            if not silent:
                print("Graph.KHopsSubgraph - Error: None of the seed vertices are found in the graph. Returning None.")
            return None

        # ---- get mesh data (index-based edge list) ----
        # Expect: mesh_data["vertices"] (list), mesh_data["edges"] (list of [a, b] indices)
        mesh_data = Graph.MeshData(graph)
        edges_idx = mesh_data.get("edges") or []
        # Compute number of vertices robustly
        n_verts = len(mesh_data.get("vertices") or graph_vertices)

        # ---- build adjacency (directed; BFS respects 'direction') ----
        adj_out = {i: set() for i in range(n_verts)}
        adj_in  = {i: set() for i in range(n_verts)}
        for (a, b) in edges_idx:
            if 0 <= a < n_verts and 0 <= b < n_verts:
                adj_out[a].add(b)
                adj_in[b].add(a)

        # ---- BFS up to k hops ----
        dir_norm = (direction or "both").lower()
        if dir_norm not in ("both", "out", "in"):
            dir_norm = "both"

        visited = set(seed_indices)
        frontier = set(seed_indices)
        for _ in range(max(0, int(k))):
            nxt = set()
            for v in frontier:
                if dir_norm in ("both", "out"):
                    nxt |= adj_out.get(v, set())
                if dir_norm in ("both", "in"):
                    nxt |= adj_in.get(v, set())
            nxt -= visited
            if not nxt:
                break
            visited |= nxt
            frontier = nxt

        if not visited:
            if not silent:
                print("Graph.KHopsSubgraph - Warning: No vertices found within the specified k hops. Returning None.")
            return None

        # ---- assemble subgraph ----
        # Vertices: actual TopologicPy Vertex objects
        sub_vertex_indices = sorted(visited)
        sub_vertices = [graph_vertices[i] for i in sub_vertex_indices]

        # Edges: include only those whose endpoints are both in the subgraph
        sub_index_set = set(sub_vertex_indices)
        # Map from global index -> actual Vertex object for edge reconstruction
        idx_to_vertex = {i: graph_vertices[i] for i in sub_vertex_indices}

        sub_edges = []
        for (a, b) in edges_idx:
            if a in sub_index_set and b in sub_index_set:
                # Recreate edge to ensure it references the subgraph vertices
                ea = idx_to_vertex[a]
                eb = idx_to_vertex[b]
                try:
                    e = Edge.ByStartVertexEndVertex(ea, eb)
                except Exception:
                    # If creation fails, skip this edge
                    continue
                # Preserve edge label if present
                try:
                    # Find original edge and copy its dictionary if possible
                    # (best-effort; safe if Graph.Edges aligns with edges_idx order)
                    # Otherwise, leave edge as-is.
                    pass
                except Exception:
                    pass
                sub_edges.append(e)

        try:
            return Graph.ByVerticesEdges(sub_vertices, sub_edges)
        except Exception:
            # As a fallback, some environments accept edges alone
            try:
                return Graph.ByEdges(sub_edges)
            except Exception:
                if not silent:
                    print("Graph.KHopsSubgraph - Error: Failed to construct the subgraph. Returning None.")
                return None

    @staticmethod
    def Laplacian(graph, silent: bool = False, normalized: bool = False):
        """
        Returns the Laplacian matrix of the input graph. See https://en.wikipedia.org/wiki/Laplacian_matrix.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        normalized : bool , optional
            If set to True, the returned Laplacian matrix is normalized. Default is False.

        Returns
        -------
        list
            The Laplacian matrix as a nested list.
        """
        from topologicpy.Topology import Topology
        import numpy as np

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.Laplacian - Error: The input graph parameter is not a valid graph. Returning None.")
            return None

        # Get vertices of the graph
        vertices = Graph.Vertices(graph)
        n = len(vertices)

        # Initialize Laplacian matrix
        laplacian = np.zeros((n, n))

        # Fill Laplacian matrix
        for i, v1 in enumerate(vertices):
            for j, v2 in enumerate(vertices):
                if i == j:
                    laplacian[i][j] = float(Graph.VertexDegree(graph, v1))
                elif Graph.IsConnected(graph, v1, v2):
                    laplacian[i][j] = -1.0
                else:
                    laplacian[i][j] = 0.0

        # Normalize the Laplacian if requested
        if normalized:
            degree_matrix = np.diag(laplacian.diagonal())
            with np.errstate(divide='ignore'):  # Suppress warnings for division by zero
                d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0  # Replace infinities with zero

            normalized_laplacian = d_inv_sqrt @ laplacian @ d_inv_sqrt
            return normalized_laplacian.tolist()

        return laplacian.tolist()

    @staticmethod
    def Leaves(graph, weightKey: str = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a list of all vertices that have a degree of 1, also called leaf nodes.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        weightKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the weightKey string will be aggregated to calculate
            the vertex degree. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead.
            This is used in weighted graphs. if weightKey is set to "Length" or "Distance", the length of the edge will be used as its weight.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The list of leaf nodes

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.Leaves - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        return [v for v in Graph.Vertices(graph) if Graph.VertexDegree(graph, v, weightKey=weightKey, mantissa=mantissa, tolerance=tolerance, silent=silent) == 1]
    
    @staticmethod
    def LineGraph(graph, transferVertexDictionaries=False, transferEdgeDictionaries=False, tolerance=0.0001, silent=False):
        """
        Create a line graph based on the input graph. See https://en.wikipedia.org/wiki/Line_graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        transferVertexDictionaries : bool, optional
            If set to True, the dictionaries of the vertices of the input graph are transferred to the edges of the line graph.
        transferEdgeDictionaries : bool, optional
            If set to True, the dictionaries of the edges of the input graph are transferred to the vertices of the line graph.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Graph
            The created line graph.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.LineGraph - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        
        graph_vertices = Graph.Vertices(graph)
        graph_edges = Graph.Edges(graph)

        # Create line graph vertices (centroids of original graph edges)
        if transferEdgeDictionaries == True:
            lg_vertices = [
                Topology.SetDictionary(Topology.Centroid(edge), Topology.Dictionary(edge), silent=silent)
                for edge in graph_edges
            ]
        else:
            lg_vertices = [Topology.Centroid(edge) for edge in graph_edges]
        
        lg_edges = []
        if transferVertexDictionaries == True:
            for v in graph_vertices:
                edges = Graph.Edges(graph, vertices=[v])
                if len(edges) > 1:
                    d = Topology.Dictionary(v)  # Only need to call Dictionary once
                    visited = set()  # Use a set to track visited pairs of edges
                    centroids = [Topology.Centroid(e) for e in edges]  # Precompute centroids once
                    for i in range(len(edges)):
                        for j in range(i + 1, len(edges)):  # Only loop over pairs (i, j) where i < j
                            if (i, j) not in visited:
                                lg_edge = Edge.ByVertices([centroids[i], centroids[j]], tolerance=tolerance, silent=silent)
                                lg_edge = Topology.SetDictionary(lg_edge, d, silent=silent)
                                lg_edges.append(lg_edge)
                                visited.add((i, j))
                                visited.add((j, i))  # Ensure both directions are marked as visited
        else:
            for v in graph_vertices:
                edges = Graph.Edges(graph, vertices=[v])
                if len(edges) > 1:
                    visited = set()  # Use a set to track visited pairs of edges
                    centroids = [Topology.Centroid(e) for e in edges]  # Precompute centroids once
                    for i in range(len(edges)):
                        for j in range(i + 1, len(edges)):  # Only loop over pairs (i, j) where i < j
                            if (i, j) not in visited:
                                lg_edge = Edge.ByVertices([centroids[i], centroids[j]], tolerance=tolerance, silent=silent)
                                lg_edges.append(lg_edge)
                                visited.add((i, j))
                                visited.add((j, i))  # Ensure both directions are marked as visited

        return Graph.ByVerticesEdges(lg_vertices, lg_edges)

    @staticmethod
    def LocalClusteringCoefficient(graph, vertices: list = None, key: str = "lcc", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the local clustering coefficient of the input list of vertices within the input graph. See https://en.wikipedia.org/wiki/Clustering_coefficient.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. If set to None, the local clustering coefficient of all vertices will be computed. Default is None.
        key : str , optional
            The dictionary key under which to store the local clustering coefficient score. Default is "lcc".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
            The vertex key to maximize. If set the vertices dictionaries will be searched for this key and the associated value will be used to compute the longest path that maximizes the total value. The value must be numeric. Default is None.
        edgeKey : str , optional
            The edge key to maximize. If set the edges dictionaries will be searched for this key and the associated value will be used to compute the longest path that maximizes the total value. The value of the key must be numeric. If set to "length" (case insensitive), the shortest path by length is computed. Default is "length".
        costKey : str , optional
            If not None, the total cost of the longest_path will be stored in its dictionary under this key. Default is None. 
        timeLimit : int , optional
            The time limit in second. Default is 10 seconds.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
        if Vertex.Distance(sv, vertexB) <= tolerance: # Wire is reversed. Re-reverse it
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

    def Match(graphA, graphB, vertexKeys: list = None, edgeKeys: list = None, maxMatches: int = 10, timeLimit: int = 10, tolerance: float = 0.0, silent: bool = False):
        """
        Matches graphA as a subgraph of graphB using structural and semantic similarity.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The smaller graph (subgraph).
        graphB : topologic_core.Graph
            The larger graph (supergraph).
        vertexKeys : str or list of str, optional
            Keys used to semantically compare vertices.
        edgeKeys : str or list of str, optional
            Keys used to semantically compare edges.
        maxMatches : int , optional
            The maximum number of matches to find. Default is 10.
        timeLimit : int , optional
                The time limit in seconds. Default is 10 seconds. Note that this time limit only applies to finding the matches.
        tolerance : float, optional
            Allowed numeric deviation or minimum string similarity (e.g. 0.2 = ≥80% match). Default is 0.
        silent : bool, optional
            If True, suppresses warnings and errors.

        Returns
        -------
        list of dict
            List of mappings from node index in graphA to graphB, sorted by descending similarity.
        """
        import networkx as nx
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        from topologicpy.Helper import Helper
        from difflib import SequenceMatcher
        import time

        def string_similarity(s1, s2):
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

        if not Topology.IsInstance(graphA, "Graph") or not Topology.IsInstance(graphB, "Graph"):
            if not silent:
                print("Graph.Match - Error: One or both inputs are not valid Graphs.")
            return None

        # Normalize keys
        if isinstance(vertexKeys, str):
            vertexKeys = [vertexKeys]
        if isinstance(edgeKeys, str):
            edgeKeys = [edgeKeys]

        nx_ga = Graph.NetworkXGraph(graphA)
        nx_gb = Graph.NetworkXGraph(graphB)

        def similarity_score(val1, val2):
            try:
                v1 = float(val1)
                v2 = float(val2)
                if v1 == 0:
                    return 1.0 if abs(v2) <= tolerance else 0.0
                diff = abs(v1 - v2) / abs(v1)
                return max(0.0, 1.0 - diff)
            except (ValueError, TypeError):
                return string_similarity(str(val1), str(val2))

        def compute_mapping_score(mapping):
            total_score = 0
            count = 0

            # Score vertices
            for i_b, i_a in mapping.items():
                a_attrs = nx_ga.nodes[i_a]
                b_attrs = nx_gb.nodes[i_b]
                if vertexKeys:
                    for key in vertexKeys:
                        if key in a_attrs and key in b_attrs:
                            score = similarity_score(a_attrs[key], b_attrs[key])
                            total_score += score
                            count += 1

            # Score edges
            for (i_a1, i_a2) in nx_ga.edges:
                if i_a1 in mapping and i_a2 in mapping:
                    j_b1 = mapping[i_a1]
                    j_b2 = mapping[i_a2]
                    if nx_gb.has_edge(j_b1, j_b2):
                        a_attrs = nx_ga.get_edge_data(i_a1, i_a2)
                        b_attrs = nx_gb.get_edge_data(j_b1, j_b2)
                        if edgeKeys:
                            for key in edgeKeys:
                                if key in a_attrs and key in b_attrs:
                                    score = similarity_score(a_attrs[key], b_attrs[key])
                                    total_score += score
                                    count += 1

            return total_score / count if count > 0 else 1.0

        def node_match(n1, n2):
            if vertexKeys:
                for key in vertexKeys:
                    if key not in n1 or key not in n2:
                        return False
                    sim = similarity_score(n1[key], n2[key])
                    if sim < (1.0 - tolerance):
                        return False
            return True

        def edge_match(e1, e2):
            if edgeKeys:
                for key in edgeKeys:
                    if key not in e1 or key not in e2:
                        return False
                    sim = similarity_score(e1[key], e2[key])
                    if sim < (1.0 - tolerance):
                        return False
            return True

        matcher = nx.algorithms.isomorphism.GraphMatcher(
            nx_gb, nx_ga,
            node_match=node_match if vertexKeys else None,
            edge_match=edge_match if edgeKeys else None
        )

        start = time.time()
        raw_matches = []
        for i, m in enumerate(matcher.subgraph_isomorphisms_iter()):
            raw_matches.append(m)
            elapsed_time = time.time() - start
            if i + 1 >= maxMatches or elapsed_time >= timeLimit:
                break
        
        if not raw_matches and not silent:
            print("Graph.Match - Warning: No subgraph isomorphisms found.")
            return []

        scores = [compute_mapping_score(m) for m in raw_matches]
        sorted_matches = Helper.Sort(raw_matches, scores, reverseFlags=[True])
        return sorted_matches

    # @staticmethod
    # def Match(graphA, graphB, vertexKeys=None, edgeKeys=None, tolerance: float = 0.0, silent: bool = False):
    #     """
    #     Matches graphA as a subgraph of graphB using structural and semantic similarity.

    #     Parameters
    #     ----------
    #     graphA : topologic_core.Graph
    #         The smaller graph (subgraph).
    #     graphB : topologic_core.Graph
    #         The larger graph (supergraph).
    #     vertexKeys : str or list of str, optional
    #         Keys used to semantically compare vertices.
    #     edgeKeys : str or list of str, optional
    #         Keys used to semantically compare edges.
    #     tolerance : float, optional
    #         Allowed numeric deviation or minimum string similarity (e.g. 0.2 = ≥80% match). Default is 0.
    #     silent : bool, optional
    #         If True, suppresses warnings and errors.

    #     Returns
    #     -------
    #     list of dict
    #         List of mappings from node index in graphA to graphB, sorted by descending similarity.
    #     """
    #     import networkx as nx
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Graph import Graph
    #     from topologicpy.Helper import Helper
    #     from difflib import SequenceMatcher

    #     def string_similarity(s1, s2):
    #         return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    #     if not Topology.IsInstance(graphA, "Graph") or not Topology.IsInstance(graphB, "Graph"):
    #         if not silent:
    #             print("Graph.Match - Error: One or both inputs are not valid Graphs.")
    #         return None

    #     # Normalize keys
    #     if isinstance(vertexKeys, str):
    #         vertexKeys = [vertexKeys]
    #     if isinstance(edgeKeys, str):
    #         edgeKeys = [edgeKeys]

    #     nx_ga = Graph.NetworkXGraph(graphA)
    #     nx_gb = Graph.NetworkXGraph(graphB)

    #     def similarity_score(val1, val2):
    #         try:
    #             v1 = float(val1)
    #             v2 = float(val2)
    #             if v1 == 0:
    #                 return 1.0 if abs(v2) <= tolerance else 0.0
    #             diff = abs(v1 - v2) / abs(v1)
    #             return max(0.0, 1.0 - diff)
    #         except (ValueError, TypeError):
    #             return string_similarity(str(val1), str(val2))

    #     def compute_mapping_score(mapping):
    #         total_score = 0
    #         count = 0

    #         # Score vertices
    #         for i_a, i_b in mapping.items():
    #             a_attrs = nx_ga.nodes[i_a]
    #             b_attrs = nx_gb.nodes[i_b]
    #             if vertexKeys:
    #                 for key in vertexKeys:
    #                     if key in a_attrs and key in b_attrs:
    #                         score = similarity_score(a_attrs[key], b_attrs[key])
    #                         total_score += score
    #                         count += 1

    #         # Score edges
    #         for (i_a1, i_a2) in nx_ga.edges:
    #             if i_a1 in mapping and i_a2 in mapping:
    #                 j_b1 = mapping[i_a1]
    #                 j_b2 = mapping[i_a2]
    #                 if nx_gb.has_edge(j_b1, j_b2):
    #                     a_attrs = nx_ga.get_edge_data(i_a1, i_a2)
    #                     b_attrs = nx_gb.get_edge_data(j_b1, j_b2)
    #                     if edgeKeys:
    #                         for key in edgeKeys:
    #                             if key in a_attrs and key in b_attrs:
    #                                 score = similarity_score(a_attrs[key], b_attrs[key])
    #                                 total_score += score
    #                                 count += 1

    #         return total_score / count if count > 0 else 1.0

    #     def node_match(n1, n2):
    #         if vertexKeys:
    #             for key in vertexKeys:
    #                 if key not in n1 or key not in n2:
    #                     return False
    #                 sim = similarity_score(n1[key], n2[key])
    #                 if sim < (1.0 - tolerance):
    #                     return False
    #         return True

    #     def edge_match(e1, e2):
    #         if edgeKeys:
    #             for key in edgeKeys:
    #                 if key not in e1 or key not in e2:
    #                     return False
    #                 sim = similarity_score(e1[key], e2[key])
    #                 if sim < (1.0 - tolerance):
    #                     return False
    #         return True

    #     matcher = nx.algorithms.isomorphism.GraphMatcher(
    #         nx_gb, nx_ga,
    #         node_match=node_match if vertexKeys else None,
    #         edge_match=edge_match if edgeKeys else None
    #     )

    #     raw_matches = list(matcher.subgraph_isomorphisms_iter())
    #     if not raw_matches and not silent:
    #         print("Graph.Match - Warning: No subgraph isomorphisms found.")
    #         return []

    #     scores = [compute_mapping_score(m) for m in raw_matches]
    #     sorted_matches = Helper.Sort(raw_matches, scores, reverse=True)

    #     return sorted_matches

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
            The edge dictionary key to use to find the value of the forward capacity of the edge. If not set, the length of the edge is used as its capacity. Default is None.
        edgeKeyBwd : str , optional
            The edge dictionary key to use to find the value of the backward capacity of the edge. This is only considered if the edge is set to be bidrectional. Default is None.
        bidirKey : str , optional
            The edge dictionary key to use to determine if the edge is bidrectional. Default is None.
        bidrectional : bool , optional
            If set to True, the whole graph is considered to be bidirectional. Default is False.
        residualKey : str , optional
            The name of the key to use to store the residual value of each edge capacity in the input graph. Default is "residual".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def MergeVertices(graph, *vertices, targetVertex=None, transferDictionaries: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Merges the input vertices into one vertex and reconnects all edges to the new vertex.
        If two of the input vertices are the end points of the same edge, that edge is deleted.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        *vertices : topologic_core.Vertex
            Two or more instances of `topologic_core.Topology` to be processed.
        targetVertex : topologic_core.Vertex, optional
            The target vertex to merge into. If None, a centroid is computed. Default is None.
        transferDictionaries : bool, optional
            If True, the dictionaries of all input vertices (including the target vertex if given) are merged. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            A new graph with the vertices merged and edges updated.
        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        import inspect

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph:", graph)
            if not silent:
                print("Graph.MergeVertices - Error: The input graph is not valid. Returning None.")
            return None

        if len(vertices) == 0:
            if not silent:
                print("Graph.MergeVertices - Error: The input vertices parameter is an empty list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if len(vertices) == 1:
            vertices = vertices[0]
            if isinstance(vertices, list):
                if len(vertices) == 0:
                    if not silent:
                        print("Graph.MergeVertices - Error: The input topologies parameter is an empty list. Returning None.")
                        curframe = inspect.currentframe()
                        calframe = inspect.getouterframes(curframe, 2)
                        print('caller name:', calframe[1][3])
                    return None
                else:
                    vertexList = [x for x in vertices if Topology.IsInstance(x, "Topology")]
                    if len(vertexList) == 0:
                        if not silent:
                            print("Graph.MergeVertices - Error: The input topologies parameter does not contain any valid vertices. Returning None.")
                            curframe = inspect.currentframe()
                            calframe = inspect.getouterframes(curframe, 2)
                            print('caller name:', calframe[1][3])
                        return None
            else:
                if not silent:
                    print("Graph.MergeVertices - Error: The input vertices parameter contains only one vertex. Returning None.")
                    curframe = inspect.currentframe()
                    calframe = inspect.getouterframes(curframe, 2)
                    print('caller name:', calframe[1][3])
                return None
        else:
            vertexList = Helper.Flatten(list(vertices))
            vertexList = [x for x in vertexList if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) == 0:
            if not silent:
                print("Graph.MergeVertices - Error: The input parameters do not contain any valid vertices. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        
        # Step 1: gather all vertices and edges
        all_vertices = Graph.Vertices(graph)
        all_edges = Graph.Edges(graph)

        # Step 2: determine merged vertex
        dictionaries = []
        if targetVertex and Topology.IsInstance(targetVertex, "Vertex"):
            merged_vertex = targetVertex
            if targetVertex not in all_vertices:
                all_vertices.append(targetVertex)
            dictionaries.append(Topology.Dictionary(targetVertex))
        else:
            # Compute centroid
            merged_vertex = Topology.Centroid(Cluster.ByTopologies(vertexList))

        # Step 3: collect dictionaries
        if transferDictionaries:
            for v in vertexList:
                d = Topology.Dictionary(v)
                dictionaries.append(d)
            merged_dict = Dictionary.ByMergedDictionaries(*dictionaries)
            merged_vertex = Topology.SetDictionary(merged_vertex, merged_dict, silent=True)

        # Step 4: remove merged vertices from all_vertices
        for v in vertexList:
            for gv in all_vertices:
                if Topology.IsSame(v, gv):
                    all_vertices.remove(gv)
                    break

        # Step 5: rebuild edge list
        new_edges = []
        seen = set()
        for edge in all_edges:
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)

            sv_merged = any(Topology.IsSame(sv, v) for v in vertexList)
            ev_merged = any(Topology.IsSame(ev, v) for v in vertexList)

            if sv_merged and ev_merged:
                continue  # Remove edges between merged vertices

            new_sv = merged_vertex if sv_merged else sv
            new_ev = merged_vertex if ev_merged else ev

            if Topology.IsSame(new_sv, new_ev):
                continue  # Avoid self-loop

            key = tuple(sorted([Topology.UUID(new_sv), Topology.UUID(new_ev)]))
            if key in seen:
                continue
            seen.add(key)

            new_edge = Edge.ByVertices([new_sv, new_ev])
            if Topology.IsInstance(new_edge, "edge"):
                d = Topology.Dictionary(edge)
                if d:
                    new_edge = Topology.SetDictionary(new_edge, d, silent=True)
                new_edges.append(new_edge)

        all_vertices.append(merged_vertex)
        return Graph.ByVerticesEdges(all_vertices, new_edges)

    @staticmethod
    def MeshData(graph, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the mesh data of the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        g_vertices = Graph.Vertices(graph)
        m_vertices = []
        v_dicts = []
        for g_vertex in g_vertices:
            m_vertices.append(Vertex.Coordinates(g_vertex, mantissa=mantissa))
            d = Dictionary.PythonDictionary(Topology.Dictionary(g_vertex))
            v_dicts.append(d)
        g_edges = Graph.Edges(graph)
        m_edges = []
        e_dicts = []
        for g_edge in g_edges:
            sv = Edge.StartVertex(g_edge)
            ev = Edge.EndVertex(g_edge)
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
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            If set, the value of the edgeKey will be used as the weight and the tree will minimize the weight. The value associated with the edgeKey must be numerical. If the key is not set, the edges will be sorted by their length. Default is None
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
                if Vertex.Distance(v, vertex) <= tolerance:
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
            The desired tolerance. Default is 0.0001.
        numWorkers : int, optional
            Number of workers run in parallel to process. Default is None which sets the number to twice the number of CPU cores.

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
    def NetworkXGraph(graph, xKey='x', yKey='y', zKey='z', mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Converts the input graph into a NetworkX Graph. See http://networkx.org

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        xKey : str , optional
            The dictionary key under which to store the X-Coordinate of the vertex. Default is 'x'.
        yKey : str , optional
            The dictionary key under which to store the Y-Coordinate of the vertex. Default is 'y'.
        zKey : str , optional
            The dictionary key under which to store the Z-Coordinate of the vertex. Default is 'z'.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        networkX Graph
            The created networkX Graph

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import warnings
        import os

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
                print("Graph.NetworkXGraph - Information: networkx library installed correctly.")
            except:
                warnings.warn("Graph - Error: Could not import networkx. Please try to install networkx manually. Returning None.")
                return None

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.NetworkXGraph - Error: The input graph is not a valid graph. Returning None.")
            return None

        nxGraph = nx.Graph()
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)
        mesh_data = Graph.MeshData(graph)

        # Add nodes with attributes
        for i, v in enumerate(vertices):
            d = Topology.Dictionary(v)
            pythonD = Dictionary.PythonDictionary(d) if d else {}
            pythonD[xKey] = Vertex.X(v, mantissa)
            pythonD[yKey] = Vertex.Y(v, mantissa)
            pythonD[zKey] = Vertex.Z(v, mantissa)
            nxGraph.add_node(i, **pythonD)

        # Add edges
        mesh_edges = mesh_data['edges']
        for i, mesh_edge in enumerate(mesh_edges):
            sv_i = mesh_edge[0]
            ev_i = mesh_edge[1]
            sv = vertices[sv_i]
            ev = vertices[ev_i]
            edge_length = Vertex.Distance(sv, ev, mantissa=mantissa)
            edge_dict = Topology.Dictionary(edges[i]) if i < len(edges) else None
            edge_attributes = Dictionary.PythonDictionary(edge_dict) if edge_dict else {}
            edge_attributes['length'] = edge_length
            nxGraph.add_edge(sv_i, ev_i, **edge_attributes)
        
        # Reshape it into a 2D spring layout for future display
        pos = nx.spring_layout(nxGraph, k=0.2)
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
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            if Vertex.Distance(vertex, sv) <= tolerance:
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
            If set to True, the graph is considered to be directed. Otherwise, it will be considered as an unidrected graph. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def PageRank(
        graph,
        alpha: float = 0.85,
        maxIterations: int = 100,
        normalize: bool = True,
        directed: bool = False,
        key: str = "page_rank",
        colorKey: str = "pr_color",
        colorScale: str = "viridis",
        mantissa: int = 6,
        tolerance: float = 1e-4
    ):
        """
        PageRank with stable vertex mapping (by coordinates) so neighbors resolve correctly.
        Handles dangling nodes; uses cached neighbor lists and L1 convergence.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Color import Color
        from topologicpy.Graph import Graph

        vertices = Graph.Vertices(graph)
        n = len(vertices)
        if n < 1:
            print("Graph.PageRank - Error: The input graph parameter has no vertices. Returning None")
            return None

        # ---- stable vertex key (coord-based) ----
        # Use a modest rounding to be robust to tiny numerical noise.
        # If your graphs can have distinct vertices at the exact same coords,
        # switch to a stronger key (e.g., include a unique ID from the vertex dictionary).
        def vkey(v, r=9):
            return (round(Vertex.X(v), r), round(Vertex.Y(v), r), round(Vertex.Z(v), r))

        idx_of = {vkey(v): i for i, v in enumerate(vertices)}

        # Helper that resolves an arbitrary Topologic vertex to our index
        def to_idx(u):
            return idx_of.get(vkey(u), None)

        # ---- build neighbor lists ONCE (by indices) ----
        if directed:
            in_neighbors = [[] for _ in range(n)]
            out_neighbors = [[] for _ in range(n)]

            for i, v in enumerate(vertices):
                inv = Graph.IncomingVertices(graph, v, directed=True)
                onv = Graph.OutgoingVertices(graph, v, directed=True)
                # map to indices, drop misses
                in_neighbors[i] = [j for u in inv if (j := to_idx(u)) is not None]
                out_neighbors[i] = [j for u in onv if (j := to_idx(u)) is not None]
        else:
            in_neighbors = [[] for _ in range(n)]
            out_neighbors = in_neighbors  # same list objects is fine; we set both below
            for i, v in enumerate(vertices):
                nbrs = Graph.AdjacentVertices(graph, v)
                idxs = [j for u in nbrs if (j := to_idx(u)) is not None]
                in_neighbors[i] = idxs
            out_neighbors = in_neighbors  # undirected: in == out

        out_degree = [len(out_neighbors[i]) for i in range(n)]
        dangling = [i for i in range(n) if out_degree[i] == 0]

        # ---- power iteration ----
        pr = [1.0 / n] * n
        base = (1.0 - alpha) / n

        for _ in range(maxIterations):
            # Distribute dangling mass uniformly
            dangling_mass = alpha * (sum(pr[i] for i in dangling) / n) if dangling else 0.0

            new_pr = [base + dangling_mass] * n

            # Sum contributions from incoming neighbors j: alpha * pr[j] / out_degree[j]
            for i in range(n):
                acc = 0.0
                for j in in_neighbors[i]:
                    deg = out_degree[j]
                    if deg > 0:
                        acc += pr[j] / deg
                new_pr[i] += alpha * acc

            # L1 convergence
            if sum(abs(new_pr[i] - pr[i]) for i in range(n)) <= tolerance:
                pr = new_pr
                break
            pr = new_pr

        # ---- normalize & write dictionaries ----
        if normalize:
            pr = Helper.Normalize(pr)
            if mantissa > 0:
                pr = [round(v, mantissa) for v in pr]
            min_v, max_v = 0.0, 1.0
        else:
            min_v, max_v = (min(pr), max(pr)) if n > 0 else (0.0, 0.0)

        for i, value in enumerate(pr):
            d = Topology.Dictionary(vertices[i])
            color = Color.AnyToHex(
                Color.ByValueInRange(value, minValue=min_v, maxValue=max_v, colorScale=colorScale)
            )
            d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color])
            vertices[i] = Topology.SetDictionary(vertices[i], d)

        return pr


    # @staticmethod
    # def PageRank_old(graph, alpha: float = 0.85, maxIterations: int = 100, normalize: bool = True, directed: bool = False, key: str = "page_rank", colorKey="pr_color", colorScale="viridis", mantissa: int = 6, tolerance: float = 0.0001):
    #     """
    #     Calculates PageRank scores for vertices in a directed graph. see https://en.wikipedia.org/wiki/PageRank.

    #     Parameters
    #     ----------
    #     graph : topologic_core.Graph
    #         The input graph.
    #     alpha : float , optional
    #         The damping (dampening) factor. Default is 0.85. See https://en.wikipedia.org/wiki/PageRank.
    #     maxIterations : int , optional
    #         The maximum number of iterations to calculate the page rank. Default is 100.
    #     normalize : bool , optional
    #         If set to True, the results will be normalized from 0 to 1. Otherwise, they won't be. Default is True.
    #     directed : bool , optional
    #         If set to True, the graph is considered as a directed graph. Otherwise, it will be considered as an undirected graph. Default is False.
    #     key : str , optional
    #         The dictionary key under which to store the page_rank score. Default is "page_rank"
    #     colorKey : str , optional
    #         The desired dictionary key under which to store the pagerank color. Default is "pr_color".
    #     colorScale : str , optional
    #         The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
    #         In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
    #     mantissa : int , optional
    #         The desired length of the mantissa.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.

    #     Returns
    #     -------
    #     list
    #         The list of page ranks for the vertices in the graph.
    #     """
    #     from topologicpy.Vertex import Vertex
    #     from topologicpy.Helper import Helper
    #     from topologicpy.Dictionary import Dictionary
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Color import Color

    #     vertices = Graph.Vertices(graph)
    #     num_vertices = len(vertices)
    #     if num_vertices < 1:
    #         print("Graph.PageRank - Error: The input graph parameter has no vertices. Returning None")
    #         return None
    #     initial_score = 1.0 / num_vertices
    #     values = [initial_score for vertex in vertices]
    #     for _ in range(maxIterations):
    #         new_scores = [0 for vertex in vertices]
    #         for i, vertex in enumerate(vertices):
    #             incoming_score = 0
    #             for incoming_vertex in Graph.IncomingVertices(graph, vertex, directed=directed):
    #                 if len(Graph.IncomingVertices(graph, incoming_vertex, directed=directed)) > 0:
    #                     vi = Vertex.Index(incoming_vertex, vertices, tolerance=tolerance)
    #                     if not vi == None:
    #                         incoming_score += values[vi] / len(Graph.IncomingVertices(graph, incoming_vertex, directed=directed))
    #             new_scores[i] = alpha * incoming_score + (1 - alpha) / num_vertices

    #         # Check for convergence
    #         if all(abs(new_scores[i] - values[i]) <= tolerance for i in range(len(vertices))):
    #             break

    #         values = new_scores
    #     if normalize == True:
    #         if mantissa > 0: # We cannot round numbers from 0 to 1 with a mantissa = 0.
    #             values = [round(v, mantissa) for v in Helper.Normalize(values)]
    #         else:
    #             values = Helper.Normalize(values)
    #         min_value = 0
    #         max_value = 1
    #     else:
    #         min_value = min(values)
    #         max_value = max(values)

    #     for i, value in enumerate(values):
    #         d = Topology.Dictionary(vertices[i])
    #         color = Color.AnyToHex(Color.ByValueInRange(value, minValue=min_value, maxValue=max_value, colorScale=colorScale))
    #         d = Dictionary.SetValuesAtKeys(d, [key, colorKey], [value, color])
    #         vertices[i] = Topology.SetDictionary(vertices[i], d)
        
    #     for i, v in enumerate(vertices):
    #         d = Topology.Dictionary(v)
    #         d = Dictionary.SetValueAtKey(d, key, values[i])
    #         v = Topology.SetDictionary(v, d)
    #     return values

    @staticmethod
    def Partition(graph, method: str = "Betweenness", n: int = 2, m: int = 10, key: str ="partition",
                mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Partitions the input graph based on the desired partition method. See https://en.wikipedia.org/wiki/Graph_partition.

        Parameters
        ----------
        graph : topologicp.Graph
            The input topologic graph.
        method : str , optional
            The desired partitioning method. The options are:
            - "Betweenness"
            - "Community" or "Louvain"
            - "Fiedler" or "Eigen"
            It is case insensitive. Default is "Betweenness"
        n : int , optional
            The desired number of partitions when selecting the "Betweenness" method. This parameter is ignored for other methods. Default is 2.
        m : int , optional
            The desired maximum number of tries to partition the graph when selecting the "Betweenness" method. This parameter is ignored for other methods. Default is 10.
        key : str , optional
            The vertex and edge dictionary key under which to store the parition number. Default is "partition".
            Valid partition numbers start from 1. Cut edges receive a partition number of 0.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologicpy.Graph
            The partitioned topologic graph.

        """ 

        m_d = Graph.MeshData(graph)
        new_graph = Graph.ByMeshData(vertices = m_d['vertices'],
                                    edges = m_d['edges'],
                                    vertexDictionaries = m_d['vertexDictionaries'],
                                    edgeDictionaries = m_d['edgeDictionaries'])
        if "between" in method.lower():
            return Graph.BetweennessPartition(new_graph, n=n, m=m, key=key, tolerance=tolerance, silent=silent)
        elif "community" in method.lower() or "louvain" in method.lower():
            return Graph.CommunityPartition(new_graph, key=key, mantissa=mantissa, tolerance=tolerance, silent=silent)
        elif "fied" in method.lower() or "eig" in method.lower():
            return Graph.FiedlerVectorPartition(new_graph, key=key, mantissa=mantissa, tolerance=tolerance, silent=silent)
        else:
            if not silent:
                print("Graph.Partition - Error: The chosen method is not supported. Returning None.")
            return None

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
            The desired tolerance. Default is 0.0001.

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
            The desired figure height in pixels. Default is 900 pixels.
        backgroundColor : str, optional
            The desired background color for the figure. This can be a named color or a hexadecimal value. Default is 'white'.
        fontColor : str , optional
            The desired font color for the figure. This can be a named color or a hexadecimal value. Default is 'black'.
        notebook : bool , optional
            If set to True, the figure will be targeted at a Jupyter Notebook. Note that this is not working well. Pyvis has bugs. Default is False.
        vertexSize : int , optional
            The desired default vertex size. Default is 6.
        vertexSizeKey : str , optional
            If not set to None, the vertex size will be derived from the dictionary value set at this key. If set to "degree", the size of the vertex will be determined by its degree (number of neighbors). Default is None.
        vertexColor : str , optional
            The desired default vertex color. his can be a named color or a hexadecimal value. Default is 'black'.
        vertexColorKey : str , optional
            If not set to None, the vertex color will be derived from the dictionary value set at this key. Default is None.
        vertexLabelKey : str , optional
            If not set to None, the vertex label will be derived from the dictionary value set at this key. Default is None.
        vertexGroupKey : str , optional
            If not set to None, the vertex color will be determined by the group the vertex belongs to as derived from the value set at this key. Default is None.
        vertexGroups : list , optional
            The list of all possible vertex groups. This will help in vertex coloring. Default is None.
        minVertexGroup : int or float , optional
            If the vertex groups are numeric, specify the minimum value you wish to consider for vertex coloring. Default is None.
        maxVertexGroup : int or float , optional
            If the vertex groups are numeric, specify the maximum value you wish to consider for vertex coloring. Default is None.
        
        edgeWeight : int , optional
            The desired default weight of the edge. This determines its thickness. Default is 0.
        edgeWeightKey : str, optional
            If not set to None, the edge weight will be derived from the dictionary value set at this key. If set to "length" or "distance", the weight of the edge will be determined by its geometric length. Default is None.
        edgeLabelKey : str , optional
            If not set to None, the edge label will be derived from the dictionary value set at this key. Default is None.
        showNeighbors : bool , optional
            If set to True, a list of neighbors is shown when you hover over a vertex. Default is True.
        selectMenu : bool , optional
            If set to True, a selection menu will be displayed. Default is True
        filterMenu : bool , optional
            If set to True, a filtering menu will be displayed. Default is True.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
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
                        color = Color.AnyToHex(Color.ByValueInRange(group, minValue=minVertexGroup, maxValue=maxVertexGroup, colorScale=colorScale))
                    else:
                        color = Color.AnyToHex(Color.ByValueInRange(vertexGroups.index(group), minValue=minVertexGroup, maxValue=maxVertexGroup, colorScale=colorScale))
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
    def Quotient(topology,
                topologyType: str = "vertex",
                key: str = None,
                groupLabelKey: str = None,
                groupCountKey: str = "count",
                weighted: bool = False,
                edgeWeightKey: str = "weight",
                idKey: str = None,
                silent: bool = False):
        """
        Construct the quotient graph induced by grouping sub-topologies (Cells/Faces/Edges/Vertices)
        by a dictionary value. Two groups are connected if any member of one is adjacent to any member
        of the other via Topology.AdjacentTopologies. If weighted=True, edge weights count the number
        of distinct member-level adjacencies across groups. See https://en.wikipedia.org/wiki/Quotient_graph

        Parameters
        ----------
        topology : topologic_core.Topology or topologic_core.Graph
            The input topology or graph.
        topologyType : str
            The type of subtopology for which to search. This can be one of "vertex", "edge", "face", "cell". It is case-insensitive.
        key : str , optional
            Dictionary key used to form groups. If None, all items fall into one group.
        groupLabelKey : str , optional
            Vertex-dictionary key storing the group label. Default is "group_label".
        groupCountKey : str , optional
            Vertex-dictionary key storing the group size. Default is "count".
        weighted : bool , optional
            If True, store counts of cross-group adjacencies on edges under edgeWeightKey. Default is False.
        edgeWeightKey : str , optional
            Edge-dictionary key storing the weight when weighted=True. Default "weight".
        idKey : str , optional
            Optional dictionary key that uniquely identifies each sub-topology. If provided and present
            on both members, lookup is O(1) without calling Topology.IsSame. If missing, falls back to Topology.IsSame. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph

        if not Topology.IsInstance(topology, "Topology") and not Topology.IsInstance(topology, "Graph"):
            if not silent:
                print("Graph.Quotient - Error: The input topology parameter is not a valid Topology or Graph. Returning None.")
            return None
        if topologyType.lower() not in {"vertex", "edge", "face", "cell"}:
            if not silent:
                print("Graph.Quotient - Error: topologyType must be one of 'Cell','Face','Edge','Vertex'. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent:
                print("Graph.Quotient - Error: The input key parameter is not a valid string. Returning None.")
            return None
        if groupLabelKey == None:
            groupLabelKey = key
        if not isinstance(groupLabelKey, str):
            if not silent:
                print("Graph.Quotient - Error: The input groupLabelKey parameter is not a valid string. Returning None.")
            return None
        if not isinstance(groupCountKey, str):
            if not silent:
                print("Graph.Quotient - Error: The input groupCountKey parameter is not a valid string. Returning None.")
            return None
        if not isinstance(weighted, bool):
            if not silent:
                print("Graph.Quotient - Error: The input weighted parameter is not a valid boolean. Returning None.")
            return None
        if not isinstance(edgeWeightKey, str):
            if not silent:
                print("Graph.Quotient - Error: The input edgeWeightKey parameter is not a valid string. Returning None.")
            return None
        
        # 1) Collect sub-topologies
        getters = {
            "vertex": Topology.Vertices,
            "edge": Topology.Edges,
            "face": Topology.Faces,
            "cell": Topology.Cells,
        }
        subs = getters[topologyType.lower()](topology)
        if not subs:
            if not silent:
                print("Graph.Quotient - Error: No subtopologies found. Returning None.")
            return None

        # 2) Optional O(1) index via a unique idKey in dictionaries
        def _get_dict(st):
            try:
                return Topology.Dictionary(st)
            except Exception:
                return None

        id_index = {}
        if idKey is not None:
            for i, st in enumerate(subs):
                d = _get_dict(st)
                if d is None:
                    continue
                try:
                    uid = Dictionary.ValueAtKey(d, idKey)
                except Exception:
                    uid = None
                if uid is not None and uid not in id_index:
                    id_index[uid] = i  # first seen wins

        # 3) Labels for grouping
        def _label(st):
            if key is None:
                return None
            d = _get_dict(st)
            if d is None:
                return None
            try:
                return Dictionary.ValueAtKey(d, key)
            except Exception:
                return None

        labels = [_label(st) for st in subs]

        # 4) Partition indices by label
        groups = {}
        for i, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(i)

        group_labels = list(groups.keys())
        label_to_group_idx = {lbl: gi for gi, lbl in enumerate(group_labels)}
        item_to_group_idx = {i: label_to_group_idx[lbl] for lbl, idxs in groups.items() for i in idxs}

        # Helper to resolve neighbor index j for a given neighbor topology nb
        def _neighbor_index(nb):
            # Try idKey shortcut
            if idKey is not None:
                d = _get_dict(nb)
                if d is not None:
                    try:
                        uid = Dictionary.ValueAtKey(d, idKey)
                    except Exception:
                        uid = None
                    if uid is not None:
                        j = id_index.get(uid)
                        if j is not None:
                            return j
            # Fallback: linear scan using Topology.IsSame
            for j, st in enumerate(subs):
                try:
                    if Topology.IsSame(nb, st):
                        return j
                except Exception:
                    continue
            return None

        # 5) Group adjacency with optional weights
        # Use a set to ensure each unordered member-pair is counted once
        seen_member_pairs = set()
        group_edges = {}  # (a,b) -> weight

        for i, st in enumerate(subs):
            try:
                neighs = Topology.AdjacentTopologies(st, topology, topologyType)
            except Exception:
                neighs = []
            gi = item_to_group_idx[i]
            for nb in neighs:
                j = _neighbor_index(nb)
                if j is None or j == i:
                    continue
                aij = (i, j) if i < j else (j, i)
                if aij in seen_member_pairs:
                    continue
                seen_member_pairs.add(aij)

                gj = item_to_group_idx[j]
                if gi == gj:
                    continue
                a, b = (gi, gj) if gi < gj else (gj, gi)
                group_edges[(a, b)] = group_edges.get((a, b), 0) + 1

        # 6) One vertex per group at mean of member centroids
        group_vertices = []
        for lbl in group_labels:
            idxs = groups[lbl]
            pts = []
            for i in idxs:
                try:
                    c = Topology.Centroid(subs[i])
                    pts.append(c)
                except Exception:
                    pass
            if pts:
                try:
                    xs = [Vertex.X(p) for p in pts]
                    ys = [Vertex.Y(p) for p in pts]
                    zs = [Vertex.Z(p) for p in pts]
                    v = Vertex.ByCoordinates(sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
                except Exception:
                    v = Vertex.ByCoordinates(0, 0, 0)
            else:
                v = Vertex.ByCoordinates(0, 0, 0)

            try:
                d = Dictionary.ByKeysValues([groupLabelKey, groupCountKey], [lbl, len(idxs)])
                v = Topology.SetDictionary(v, d)
            except Exception:
                pass

            group_vertices.append(v)

        group_vertices = Vertex.Separate(group_vertices, minDistance = 0.1, strength=0.5, silent=silent)

        # 7) Edges, with optional weights
        edges = []
        for (a, b), w in group_edges.items():
            try:
                e = Edge.ByStartVertexEndVertex(group_vertices[a], group_vertices[b])
                if weighted:
                    try:
                        d = Dictionary.ByKeysValues([edgeWeightKey], [w])
                        e = Topology.SetDictionary(e, d)
                    except Exception:
                        pass
                edges.append(e)
            except Exception:
                continue


        return Graph.ByVerticesEdges(group_vertices, edges)

    @staticmethod
    def RemoveEdge(graph, *edges, tolerance=0.0001, silent: bool = False):
        """
        Removes the input edge from the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        edges : topologic_core.Edge or list of edges
            The input edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input edge removed.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.RemoveEdge - Error: The input graph is not a valid graph. Returning None.")
            return None
        edgeList = list(edges)
        edgeList = [e for e in edgeList if Topology.IsInstance(e, "edge")]
        if len(edgeList) == 0:
            if not silent:
                print("Graph.RemoveEdge - Error: The input edge is not a valid edge. Returning None.")
            return None
        _ = graph.RemoveEdges(edgeList, tolerance) # Hook to Core
        return graph
    
    @staticmethod
    def RemoveIsolatedEdges(graph, removeVertices: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Removes all isolated edges from the input graph.
        Isolated edges are those whose vertices are not connected to any other edges.
        That is, they have a degree of 1.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        removeVertices : bool , optional
            If set to True, the end vertices of the edges are also removed. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with all isolated vertices removed.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Edge import Edge

        
        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.RemoveIsolatedEdges - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        
        edges = Graph.Edges(graph)
        if removeVertices == True:
            for edge in edges:
                va, vb = Edge.Vertices(edge)
                if Graph.VertexDegree(graph, va, tolerance=tolerance, silent=silent) == 1 and Graph.VertexDegree(graph, vb, tolerance=tolerance, silent=silent) == 1:
                    graph = Graph.RemoveEdge(graph, edge, tolerance=tolerance, silent=silent)
                    graph = Graph.RemoveVertex(graph, va, silent=silent)
                    graph = Graph.RemoveVertex(graph, vb, silent=silent)
        else:
            for edge in edges:
                va, vb = Edge.Vertices(edge)
                if Graph.VertexDegree(graph, va, tolerance=tolerance, silent=silent) == 1 and Graph.VertexDegree(graph, vb, tolerance=tolerance, silent=silent) == 1:
                    graph = Graph.RemoveEdge(graph, edge, tolerance=tolerance, silent=silent)
        return graph

    @staticmethod
    def RemoveIsolatedVertices(graph, silent: bool = True):
        """
        Removes all isolated vertices from the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The input graph with all isolated vertices removed.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.RemoveIsolatedVertices - Error: The input graph parameter is not a valid graph. Returning None.")
            return None

        vertices = Graph.Vertices(graph)
        isolated_vertices = [v for v in vertices if Graph.VertexDegree(graph, v) == 0]
        if len(isolated_vertices) > 0:
            _ = graph.RemoveVertices(isolated_vertices) # Hook to Core
        return graph

    @staticmethod
    def RemoveVertex(graph, *vertices, silent: bool = False):
        """
        Removes the input vertex from the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        *vertices : topologic_core.Vertex or list of vertices
            The input vertex.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The input graph with the input vertex removed.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.RemoveVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        vertexList = list(vertices)
        vertexList = [v for v in vertexList if Topology.IsInstance(v, "vertex")]
        if len(vertexList) == 0:
            if not silent:
                print("Graph.RemoveVertex - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
            return None
        vertexList = [Graph.NearestVertex(graph, v) for v in vertexList]
        _ = graph.RemoveVertices(vertexList) # Hook to Core
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
        _ = graph.SetDictionary(dictionary) # Hook to Core
        return graph

    @staticmethod
    def ShortestPath(
        graph,
        vertexA,
        vertexB,
        vertexKey: str = "",
        edgeKey: str = "Length",
        transferDictionaries: bool = False,
        straighten: bool = False,
        host: object = None,
        turnWeight: float = 0.0,
        turnPower: float = 1.0,
        turnKey: str = "",
        directed: bool = False,
        edgeFilter: callable = None,
        vertexFilter: callable = None,
        edgeCostFunc: callable = None,
        vertexCostFunc: callable = None,
        turnCostFunc: callable = None,
        useAStar: bool = False,
        heuristicScale: float = 1.0,
        returnVertices: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):

        """
        Returns the shortest path (as a Wire) between two vertices in a Graph using a
        pure-Python, feature-rich routing algorithm.

        The path cost is computed as a weighted sum of:
        - edge traversal cost (edgeKey or geometric length),
        - optional vertex visitation cost (vertexKey),
        - optional turn/transition cost between consecutive edges (Spread-based).

        This method is backward compatible with the original TopologicPy API, but
        extends it with advanced routing features such as turn penalties, custom
        filters, and A* search, all without relying on the C++ core.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph on which routing is performed.

        vertexA : topologic_core.Vertex
            The start vertex. It is snapped to the nearest vertex in the graph.

        vertexB : topologic_core.Vertex
            The end vertex. It is snapped to the nearest vertex in the graph.

        vertexKey : str, optional
            Name of a numeric key in each vertex dictionary whose value is added
            to the path cost when that vertex is entered. Higher values make routes
            avoid those vertices. If empty, no vertex cost is applied.

        edgeKey : str, optional
            Name of a numeric key in each edge dictionary used as the edge traversal
            cost. If set to "Length" (case-insensitive), geometric edge length is used.
            This is the primary contributor to path length.

        transferDictionaries : bool, optional
            If True, dictionaries from the graph vertices are copied onto the vertices
            of the returned path. This does not affect routing, only the output data.

        straighten : bool, optional
            If True, the resulting path is post-processed to remove unnecessary bends
            while remaining inside the specified face. This does not influence route
            selection, only the final geometry.

        host : topologic_core.Topology, optional
            A host topology within which the path is
            straightened when straighten=True.
        
        obstacles : list, optional
            The list of topologies with which the straightened edges must not intersect.
        
        portals : list, optional
            The list of topologies with which the straightened edges must intersect.
            Portals with which the original wire does NOT intersect are ignored.

        turnWeight : float, optional
            Controls how strongly turning is penalised relative to edge length.
            A value of 0 disables turn cost. Larger values favour straighter routes
            even if they are longer. Default is 1.

        turnPower : float, optional
            Controls how sharply turn penalties increase with turn severity.
            Values greater than 1 strongly penalise right-angle turns while allowing
            gentle bends. Default is 2.

        turnKey : str, optional
            Name of a numeric key in vertex dictionaries that scales turn cost locally.
            Useful for modelling junction complexity or restricted turning areas. Default is None.

        directed : bool, optional
            If True, edges are traversed only from start to end vertex.
            If False, the graph is treated as undirected. Default is False.

        edgeFilter : callable(edge) -> bool, optional
            A function that returns False for edges that must not be traversed.
            This enforces hard constraints such as blocked corridors. Default is None.

        vertexFilter : callable(vertex) -> bool, optional
            A function that returns False for vertices that must not be visited
            (except for start and end vertices). Default is None.

        edgeCostFunc : callable(edge) -> float, optional
            Custom function overriding edgeKey and geometric length to compute
            edge traversal cost.

        vertexCostFunc : callable(vertex) -> float, optional
            Custom function overriding vertexKey to compute vertex visitation cost.

        turnCostFunc : callable(prev, curr, next, inEdge, outEdge, spread) -> float, optional
            Fully custom function to compute turn cost between consecutive edges.
            Overrides turnWeight, turnPower, and turnKey.

        useAStar : bool, optional
            If True, uses A* search instead of Dijkstra when edge costs are geometric,
            improving performance on large graphs.

        heuristicScale : float, optional
            Multiplier for the A* heuristic (must be ≤ 1 for admissibility).
            Lower values make the search more conservative.

        returnVertices : bool, optional
            If True, returns both the Wire and the ordered list of vertices forming
            the path. Useful for debugging or analysis.
        
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.

        silent : bool, optional
            If True, suppresses error and warning messages. Default is False.

        Returns
        -------
        topologic_core.Wire
            A wire representing the shortest path between the two input vertices,
            optionally straightened and with transferred dictionaries.
        """


        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector

        import heapq
        import math

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.ShortestPath - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Graph.ShortestPath - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Graph.ShortestPath - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        if straighten and not Topology.IsInstance(host, "Topology"):
            if not silent:
                print("Graph.ShortestPath - Error: Straighten is True but host is not a valid topology. Returning None.")
            return None

        if isinstance(edgeKey, str) and edgeKey.lower() == "length":
            edgeKey = "Length"
        
        if heuristicScale < 0.0:
            heuristicScale = 0.0
        if heuristicScale > 1.0:
            heuristicScale = 1.0

        # --------------------------------------------------
        # Helpers
        # --------------------------------------------------
        def _num_from_dict(topo, key: str, default: float = 0.0) -> float:
            if not key:
                return default
            try:
                d = Topology.Dictionary(topo)
                if not d:
                    return default
                v = Dictionary.ValueAtKey(d, key)
                if v is None:
                    return default
                if isinstance(v, bool):
                    return float(int(v))
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    vv = v.strip()
                    if vv == "":
                        return default
                    return float(vv)
            except Exception:
                return default
            return default

        def _edge_length(e) -> float:
            try:
                return float(Edge.Length(e))
            except Exception:
                sv, ev = Edge.StartVertex(e), Edge.EndVertex(e)
                ax, ay, az = Vertex.Coordinates(sv, mantissa=15)
                bx, by, bz = Vertex.Coordinates(ev, mantissa=15)
                dx, dy, dz = bx-ax, by-ay, bz-az
                return float(math.sqrt(dx*dx + dy*dy + dz*dz))

        # --------------------------------------------------
        # Extract graph vertices/edges
        # --------------------------------------------------
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)
        if not vertices or not edges:
            return None

        coords = [Vertex.Coordinates(v, mantissa=15) for v in vertices]

        # --------------------------------------------------
        # Robust indexing (no object identity dependency)
        # --------------------------------------------------
        def _vkey(v, m=9):
            return tuple(Vertex.Coordinates(v, mantissa=m))

        key_to_index = {}
        for i, v in enumerate(vertices):
            key_to_index[_vkey(v, m=9)] = i

        def _find_index(v):
            # 1) exact (rounded) coordinate match
            k = _vkey(v, m=9)
            i = key_to_index.get(k, None)
            if i is not None:
                return i

            # 2) nearest fallback by quadrance (only used for endpoints or stray vertices)
            vx, vy, vz = Vertex.Coordinates(v, mantissa=15)
            best_i = None
            best_q = float("inf")
            for j, (x, y, z) in enumerate(coords):
                dx, dy, dz = x - vx, y - vy, z - vz
                q = dx*dx + dy*dy + dz*dz
                if q < best_q:
                    best_q = q
                    best_i = j

            if best_i is None:
                return None

            # Guard against snapping to an unrelated vertex in dense graphs
            if best_q > (tolerance * tolerance):
                return None
            return best_i

        # Snap endpoints to nearest graph vertices (existing behavior)
        try:
            start_v = Graph.NearestVertex(graph, vertexA)
            goal_v = Graph.NearestVertex(graph, vertexB)
        except Exception:
            if not silent:
                print("Graph.ShortestPath - Error: Could not find nearest vertices. Returning None.")
            return None

        s_idx = _find_index(start_v)
        t_idx = _find_index(goal_v)
        if s_idx is None or t_idx is None:
            if not silent:
                print("Graph.ShortestPath - Error: Could not locate start/end vertices in graph. Returning None.")
            return None

        # --------------------------------------------------
        # Directed setting (if not explicitly provided)
        # --------------------------------------------------
        if directed is None:
            try:
                gd = Topology.Dictionary(graph)
                val = Dictionary.ValueAtKey(gd, "directed") if gd else None
                if isinstance(val, bool):
                    directed = val
                elif isinstance(val, (int, float)) and val in (0, 1):
                    directed = bool(val)
            except Exception:
                pass
        if directed is None:
            directed = False

        # --------------------------------------------------
        # Vertex filter (FIXED: now actually used)
        # --------------------------------------------------
        allowed_vertex = [True] * len(vertices)
        if callable(vertexFilter):
            for i, v in enumerate(vertices):
                if i in (s_idx, t_idx):
                    continue  # always allow endpoints
                try:
                    allowed_vertex[i] = bool(vertexFilter(v))
                except Exception:
                    allowed_vertex[i] = True

        if not allowed_vertex[s_idx] or not allowed_vertex[t_idx]:
            # Should not happen because endpoints are forced allowed, but keep safe
            allowed_vertex[s_idx] = True
            allowed_vertex[t_idx] = True

        # --------------------------------------------------
        # Vertex costs
        # --------------------------------------------------
        v_cost = [0.0] * len(vertices)
        if callable(vertexCostFunc):
            for i, v in enumerate(vertices):
                try:
                    v_cost[i] = float(vertexCostFunc(v))
                except Exception:
                    v_cost[i] = 0.0
        elif vertexKey:
            for i, v in enumerate(vertices):
                v_cost[i] = _num_from_dict(v, vertexKey, default=0.0)

        # Conventional: do not charge the start vertex
        v_cost[s_idx] = 0.0

        def _turn_multiplier_at_vertex(i: int) -> float:
            if not turnKey:
                return 1.0
            return _num_from_dict(vertices[i], turnKey, default=1.0)

        # --------------------------------------------------
        # Build adjacency
        #   We build directed traversal "arcs" even for undirected graphs, because
        #   turn costs require direction vectors.
        # --------------------------------------------------
        adj = [[] for _ in range(len(vertices))]  # adj[u] = list of (v, arc_index)

        arc_u = []
        arc_v = []
        arc_dir = []   # direction vector for traversal u->v
        arc_w = []     # traversal cost

        for e in edges:
            # Edge-level filter
            if callable(edgeFilter):
                try:
                    if not bool(edgeFilter(e)):
                        continue
                except Exception:
                    pass

            sv, ev = Edge.StartVertex(e), Edge.EndVertex(e)
            ui, vi = _find_index(sv), _find_index(ev)
            if ui is None or vi is None:
                continue

            # Vertex-level hard constraint (FIXED)
            if not allowed_vertex[ui] or not allowed_vertex[vi]:
                continue

            # Determine edge traversal cost
            if callable(edgeCostFunc):
                try:
                    w = float(edgeCostFunc(e))
                except Exception:
                    w = 0.0
            else:
                if edgeKey == "Length":
                    w = _edge_length(e)
                elif edgeKey:
                    # default to 0.0 if missing, unless edgeKey is Length
                    default = _edge_length(e) if edgeKey == "Length" else 0.0
                    w = _num_from_dict(e, edgeKey, default=default)
                else:
                    w = _edge_length(e)

            ax, ay, az = coords[ui]
            bx, by, bz = coords[vi]

            # forward arc ui->vi
            ai = len(arc_u)
            arc_u.append(ui)
            arc_v.append(vi)
            arc_dir.append([bx-ax, by-ay, bz-az])
            arc_w.append(w)
            adj[ui].append((vi, ai))

            if not directed:
                # reverse arc vi->ui
                aj = len(arc_u)
                arc_u.append(vi)
                arc_v.append(ui)
                arc_dir.append([ax-bx, ay-by, az-bz])
                arc_w.append(w)
                adj[vi].append((ui, aj))

        if s_idx == t_idx:
            # Start and end are the same after snapping
            return None

        # --------------------------------------------------
        # Heuristic (A*) - only safe/admissible if using geometric length-like costs and no custom funcs
        # --------------------------------------------------
        def _heuristic(i: int) -> float:
            if not useAStar:
                return 0.0
            ax, ay, az = coords[i]
            bx, by, bz = coords[t_idx]
            dx, dy, dz = bx-ax, by-ay, bz-az
            return heuristicScale * math.sqrt(dx*dx + dy*dy + dz*dz)

        if useAStar:
            # Disable A* when custom edge cost function is supplied or edgeKey not Length-like.
            if callable(edgeCostFunc) or (edgeKey != "Length"):
                useAStar = False

        # --------------------------------------------------
        # Dijkstra / A* with optional turn costs
        # --------------------------------------------------
        use_turn = (turnWeight != 0.0) or callable(turnCostFunc) or bool(turnKey)

        INF = float("inf")

        if not use_turn:
            # ----------------------------
            # Standard Dijkstra / A* on vertices
            # ----------------------------
            dist = [INF] * len(vertices)
            prev_v = [None] * len(vertices)
            prev_arc = [None] * len(vertices)

            dist[s_idx] = 0.0
            pq = [(0.0 + _heuristic(s_idx), 0.0, s_idx)]  # (f, g, u)

            while pq:
                f, g, u = heapq.heappop(pq)
                if g != dist[u]:
                    continue
                if u == t_idx:
                    break

                for v, aidx in adj[u]:
                    ng = g + arc_w[aidx] + v_cost[v]
                    if ng < dist[v]:
                        dist[v] = ng
                        prev_v[v] = u
                        prev_arc[v] = aidx
                        heapq.heappush(pq, (ng + _heuristic(v), ng, v))

            if dist[t_idx] == INF:
                return None

            # Reconstruct vertex indices
            path_idx = []
            cur = t_idx
            while cur is not None:
                path_idx.append(cur)
                cur = prev_v[cur]
            path_idx.reverse()

        else:
            # ----------------------------
            # Turn-cost routing on states (prev_vertex, curr_vertex)
            # We store the incoming arc index in the state to compute Spread for the next transition.
            # ----------------------------
            start_state = (-1, s_idx)  # (prev_vertex_idx, curr_vertex_idx)

            dist_state = {start_state: 0.0}
            prev_state = {}       # state -> previous state
            in_arc_state = {start_state: None}  # state -> incoming arc index

            pq = [(0.0 + _heuristic(s_idx), 0.0, start_state)]  # (f, g, state)

            goal_state = None

            while pq:
                f, g, state = heapq.heappop(pq)
                if dist_state.get(state, INF) != g:
                    continue

                p, u = state
                if u == t_idx:
                    goal_state = state
                    break

                in_arc = in_arc_state.get(state, None)

                for v, out_arc in adj[u]:
                    ng = g + arc_w[out_arc] + v_cost[v]

                    # Turn penalty only if we have an incoming arc (i.e., not the first move)
                    if in_arc is not None:
                        s = Vector.Spread(arc_dir[in_arc], arc_dir[out_arc], mantissa=15, bracket=False)

                        if callable(turnCostFunc):
                            try:
                                ng += float(turnCostFunc(p, u, v, in_arc, out_arc, s))
                            except Exception:
                                pass
                        else:
                            mult = _turn_multiplier_at_vertex(u)
                            try:
                                ts = (s ** turnPower) if turnPower != 1.0 else s
                            except Exception:
                                ts = s
                            ng += turnWeight * mult * ts

                    next_state = (u, v)
                    if ng < dist_state.get(next_state, INF):
                        dist_state[next_state] = ng
                        prev_state[next_state] = state
                        in_arc_state[next_state] = out_arc
                        heapq.heappush(pq, (ng + _heuristic(v), ng, next_state))

            if goal_state is None:
                return None

            # Reconstruct vertex indices from states
            path_idx_rev = [t_idx]
            st = goal_state
            while st != start_state:
                pst = prev_state.get(st, None)
                if pst is None:
                    break
                # pst is (p, u) where u is the vertex we came from
                path_idx_rev.append(pst[1])
                st = pst
            path_idx_rev.reverse()

            # Ensure start is present and remove any adjacent duplicates only
            if not path_idx_rev or path_idx_rev[0] != s_idx:
                path_idx_rev = [s_idx] + path_idx_rev
            path_idx = []
            for i in path_idx_rev:
                if not path_idx or path_idx[-1] != i:
                    path_idx.append(i)

        # --------------------------------------------------
        # Build output vertices (dictionary transfer is output-only)
        # FIXED: Topology.SetDictionary returns a new topology (do not ignore return value)
        # --------------------------------------------------
        out_vertices = []
        if transferDictionaries:
            for i in path_idx:
                gv = vertices[i]
                x, y, z = coords[i]
                pv = Vertex.ByCoordinates(x, y, z)
                try:
                    pv = Topology.SetDictionary(pv, Topology.Dictionary(gv))
                except Exception:
                    pass
                out_vertices.append(pv)
        else:
            out_vertices = [vertices[i] for i in path_idx]

        if len(out_vertices) < 2:
            return None

        # --------------------------------------------------
        # Build wire
        # --------------------------------------------------
        out_edges = []
        for i in range(len(out_vertices) - 1):
            try:
                e = Edge.ByVertices([out_vertices[i], out_vertices[i+1]], tolerance=tolerance, silent=True)
            except Exception:
                try:
                    e = Edge.ByVertices([out_vertices[i], out_vertices[i+1]])
                except Exception:
                    e = None
            if e is None:
                return None
            out_edges.append(e)

        wire = Wire.ByEdges(out_edges)
        if wire is None:
            return None

        # Orient edges consistently from start
        try:
            wire = Wire.OrientEdges(wire, Wire.StartVertex(wire), tolerance=tolerance)
        except Exception:
            pass

        # Optional straightening (post-process)
        if straighten and Topology.IsInstance(wire, "Wire") and Topology.IsInstance(face, "Face"):
            try:
                wire = Wire.Straighten(wire, host)
            except Exception:
                pass

        if returnVertices:
            return wire, out_vertices
        return wire

    @staticmethod
    def ShortestPath_old(graph,
                     vertexA,
                     vertexB,
                     vertexKey: str = "",
                     edgeKey: str = "Length",
                     transferDictionaries: bool = False,
                     straighten: bool = False,
                     face: Any = None,
                     tolerance: float = 0.0001,
                     silent: bool = False):
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
            The vertex key to minimise. If set the vertices dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value must be numeric. Default is None.
        edgeKey : string , optional
            The edge key to minimise. If set the edges dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value of the key must be numeric. If set to "length" (case insensitive), the shortest path by length is computed. Default is "length".
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the graph vertices will be transferred to the vertices of the shortest path. Otherwise, they won't. Default is False.
            Note: Edge dictionaries are not transferred (In straightened paths, the path edges are no longer the same as
            the original graph edges. Thus, you must implement your own logic to transfer edge dictionaries if needed).
       straighten : bool , optional
            If set to True, the path will be straightened as much as possible while remaining inside the specified face.
            Thus, the face input must be a valid topologic Face that is planar and residing on the XY plane. Default is False.
        face : topologic_core.Face , optional
            The face on which the path resides. This is used for straightening the path. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Wire
            The shortest path between the input vertices.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.ShortestPath - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Graph.ShortestPath - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Graph.ShortestPath - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        if straighten == True:
            if not Topology.IsInstance(face, "face"):
                if not silent:
                    print("Graph.ShortestPath - Error: Straighten is set to True, but the face parameter is not a valid toopologic face. Returning None.")
                return None
        if edgeKey:
            if edgeKey.lower() == "length":
                edgeKey = "Length"
        try:
            gsv = Graph.NearestVertex(graph, vertexA)
            gev = Graph.NearestVertex(graph, vertexB)
            shortest_path = graph.ShortestPath(gsv, gev, vertexKey, edgeKey) # Hook to Core
            if not shortest_path == None:
                if Topology.IsInstance(shortest_path, "Edge"):
                        shortest_path = Wire.ByEdges([shortest_path])
                sv = Topology.Vertices(shortest_path)[0]
                if Vertex.Distance(sv, gev) <= tolerance: # Path is reversed. Correct it.
                    if Topology.IsInstance(shortest_path, "Wire"):
                        shortest_path = Wire.Reverse(shortest_path)
                shortest_path = Wire.OrientEdges(shortest_path, Wire.StartVertex(shortest_path), tolerance=tolerance)
            if Topology.IsInstance(shortest_path, "wire"):
                if straighten == True and Topology.IsInstance(face, "face"):
                    shortest_path = Wire.StraightenInFace(shortest_path, face)
                if transferDictionaries == True:
                    path_verts = Topology.Vertices(shortest_path)
                    for p_v in path_verts:
                        g_v = Graph.NearestVertex(graph, p_v)
                        p_v = Topology.SetDictionary(p_v, Topology.Dictionary(g_v))
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
            The vertex key to minimise. If set the vertices dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value must be numeric. Default is None.
        edgeKey : string , optional
            The edge key to minimise. If set the edges dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value of the key must be numeric. If set to "length" (case insensitive), the shortest path by length is computed. Default is "length".
        timeLimit : int , optional
            The search time limit in seconds. Default is 10 seconds
        pathLimit: int , optional
            The number of found paths limit. Default is 10 paths.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
        Shows the graph using Plotly.

        Parameters
        ----------
        *graphs : topologic_core.Graph
            One or more toplogic_core.graph objects.
        sagitta : float , optional
            The length of the sagitta. In mathematics, the sagitta is the line connecting the center of a chord to the apex (or highest point) of the arc subtended by that chord. Default is 0 which means a straight edge is drawn instead of an arc. Default is 0.
        absolute : bool , optional
            If set to True, the sagitta length is treated as an absolute value. Otherwise, it is treated as a ratio based on the length of the edge. Default is False.
            For example, if the length of the edge is 10, the sagitta is set to 0.5, and absolute is set to False, the sagitta length will be 5. Default is True.
        sides : int , optional
            The number of sides of the arc. Default is 8.
        angle : float, optional
            An additional angle in degrees to rotate arcs (where sagitta is more than 0). Default is 0.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexColorKey : str , optional
            The dictionary key under which to find the vertex color. Default is None.
        vertexSize : float , optional
            The desired size of the vertices. Default is 1.1.
        vertexSizeKey : str , optional
            The dictionary key under which to find the vertex size. Default is None.
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. Default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. Default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. Default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. Default is None.
        vertexMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. Default is None.
        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. Default is True.
        showVertexLabel : bool , optional
            If set to True, the vertex labels are shown permenantely on screen. Otherwise, they are not. Default is False.
        showVertexLegend : bool , optional
            If set to True the vertex legend will be drawn. Otherwise, it will not be drawn. Default is False.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeColorKey : str , optional
            The dictionary key under which to find the edge color. Default is None.
        edgeWidth : float , optional
            The desired thickness of the output edges. Default is 1.
        edgeWidthKey : str , optional
            The dictionary key under which to find the edge width. Default is None.
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. Default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. Default is None.
        edgeGroups : list , optional
            The list of edge groups against which to index the color of the edge. Default is [].
        edgeMinGroup : int or float , optional
            For numeric edgeGroups, edgeMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the minimum value in edgeGroups. Default is None.
        edgeMaxGroup : int or float , optional
            For numeric edgeGroups, edgeMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the maximum value in edgeGroups. Default is None.
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. Default is True.
        showEdgeLabel : bool , optional
            If set to True, the edge labels are shown permenantely on screen. Otherwise, they are not. Default is False.
        showEdgeLegend : bool , optional
            If set to True the edge legend will be drawn. Otherwise, it will not be drawn. Default is False.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). Default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). If set to None, the code will attempt to discover the most suitable renderer. Default is None.
        width : int , optional
            The width in pixels of the figure. The default value is 950.
        height : int , optional
            The height in pixels of the figure. The default value is 950.
        xAxis : bool , optional
            If set to True the x axis is drawn. Otherwise it is not drawn. Default is False.
        yAxis : bool , optional
            If set to True the y axis is drawn. Otherwise it is not drawn. Default is False.
        zAxis : bool , optional
            If set to True the z axis is drawn. Otherwise it is not drawn. Default is False.
        axisSize : float , optional
            The size of the X, Y, Z, axes. Default is 1.
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
            The desired location of the camera). Default is [-1.25, -1.25, 1.25].
        center : list , optional
            The desired center (camera target). Default is [0, 0, 0].
        up : list , optional
            The desired up vector. Default is [0, 0, 1].
        projection : str , optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. Default is "perspective"
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        None

        """
        from topologicpy.Plotly import Plotly
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if isinstance(graphs, tuple):
            graphs = Helper.Flatten(list(graphs))
        if isinstance(graphs, list):
            new_graphs = [t for t in graphs if Topology.IsInstance(t, "Graph")]
        if len(new_graphs) == 0:
            if not silent:
                print("Topology.Show - Error: the input topologies parameter does not contain any valid topology. Returning None.")
            return None
        data = []
        for graph in new_graphs:
            data += Plotly.DataByGraph(graph,
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
    def Subgraph(graph, vertices, vertexKey="cutVertex", edgeKey="cutEdge", tolerance=0.0001, silent: bool = False):
        """
        Returns a subgraph of the input graph as defined by the input vertices.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexKey : str , optional
            The dictionary key under which to store the cut vertex status of each vertex. See https://en.wikipedia.org/wiki/Cut_(graph_theory).
            vertex cuts are indicated with a value of 1. Default is "cutVertex".
        edgeKey : str , optional
            The dictionary key under which to store the cut edge status of each edge. See https://en.wikipedia.org/wiki/Cut_(graph_theory).
            edge cuts are indicated with a value of 1. Default is "cutVertex".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Graph
            The created subgraph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "graph"):
            if not silent:
                print("Graph.Subgraph - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        
        if not isinstance(vertices, list):
            if not silent:
                print("Graph.Subgraph - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        
        vertex_list = [v for v in vertices if Topology.IsInstance(v, "vertex")]
        if len(vertex_list) < 1:
            if not silent:
                print("Graph.Subgraph - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
            return None

        edges = Graph.Edges(graph, vertices=vertex_list)
        # Set the vertexCut status to 0 for all input vertices
        for v in vertex_list:
            d = Topology.Dictionary(v)
            d = Dictionary.SetValueAtKey(d, vertexKey, 0)
            v = Topology.SetDictionary(v, d)

        final_edges = []
        if not edges == None:
            for edge in edges:
                sv = Edge.StartVertex(edge)
                status_1 = any([Vertex.IsCoincident(sv, v, tolerance=tolerance) for v in vertices])                
                ev = Edge.EndVertex(edge)
                status_2 = any([Vertex.IsCoincident(ev, v, tolerance=tolerance) for v in vertices])
                if status_1 and status_2:
                    cutEdge = 0
                else:
                    cutEdge = 1
                d = Topology.Dictionary(edge)
                d = Dictionary.SetValueAtKey(d, edgeKey, cutEdge)
                edge = Topology.SetDictionary(edge, d)
                final_edges.append(edge)
        return_graph = Graph.ByVerticesEdges(vertex_list, final_edges)
        graph_vertices = Graph.Vertices(return_graph)
        # Any vertex in the final graph that does not have a vertexCut of 0 is a new vertex and as such needs to have a vertexCut of 1.
        for v in graph_vertices:
            d = Topology.Dictionary(v)
            value = Dictionary.ValueAtKey(d, vertexKey)
            if not value == 0:
                d = Dictionary.SetValueAtKey(d, vertexKey, 1)
                v = Topology.SetDictionary(v, d)
        return return_graph

    @staticmethod
    def SubGraphMatches(subGraph, superGraph, strict=False, vertexMatcher=None, vertexKey: str = "id", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Finds all subgraph matches from `subgraph` into `supergraph`.
        A match is valid if:
        - Each subgraph vertex maps to a unique supergraph vertex either by the vertexMatcher function or through matching the vertexKey values.
        - Each subgraph edge is represented either by an edge (if strict is set to True) or by an edge or a path (if strict is set to False) in the supergraph.

        Parameters
        ----------
        subGraph : topologic_core.Graph
            The input subgraph.
        superGraph : topologic_core.Graph
            The input supergraph.
        strict : bool , optional
            If set to True, each subgraph edge must be represented by a single edge in the supergraph. Otherwise, an edge in the subgraph can be represented either with an edge or a path in the supergraph. Default is False.
        vertexMatcher : callable, optional
            If specified, this function is called to check if two vertices are matched. The format must be vertex_matcher(sub_vertex, super_vertex, mantissa, tolerance) -> bool.
        vertexKey : str , optional
            The dictionary key to use for vertex matching if the vertexMatcher input parameter is set to None. Default is "id".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            A list of subgraphs matched to the supergraph. Each vertex in the matched subgraph has a dictionary that merges the keys and values from both the subgraph and the supergraph. 
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import itertools

        sub_vertices = Graph.Vertices(subGraph)
        super_vertices = Graph.Vertices(superGraph)

        sub_ids = [Dictionary.ValueAtKey(Topology.Dictionary(v), vertexKey) for v in sub_vertices]

        # Map vertex instance to index in sub_vertices
        sub_vertex_indices = {vid: i for i, vid in enumerate(sub_ids)}

        # Default matcher by dictionary vertexKey
        if vertexMatcher is None:
            def vertexMatcher(v1, v2, mantissa=mantissa, tolerance=tolerance):
                d1 = Topology.Dictionary(v1)
                d2 = Topology.Dictionary(v2)
                id1 = Dictionary.ValueAtKey(d1, vertexKey) if d1 else None
                id2 = Dictionary.ValueAtKey(d2, vertexKey) if d2 else None
                return id1 == id2 and id1 is not None

        # Step 1: Build candidate list for each subgraph vertex (by index)
        candidate_map = {}
        for i, sv in enumerate(sub_vertices):
            candidates = [v for v in super_vertices if vertexMatcher(sv, v, mantissa=mantissa, tolerance=tolerance)]
            if not candidates:
                return []  # No match for this vertex
            candidate_map[i] = candidates

        # Step 2: Generate all injective mappings
        all_matches = []
        sub_indices = list(candidate_map.keys())
        candidate_lists = [candidate_map[i] for i in sub_indices]

        for combo in itertools.product(*candidate_lists):
            if len(set(combo)) < len(combo):
                continue  # Not injective

            mapping = dict(zip(sub_indices, combo))

            # Step 3: Check that each subgraph edge corresponds to a path in supergraph
            valid = True
            for edge in Graph.Edges(subGraph):
                sv1 = Edge.StartVertex(edge)
                sv2 = Edge.EndVertex(edge)
                d1 = Topology.Dictionary(sv1)
                d2 = Topology.Dictionary(sv2)
                id1 = Dictionary.ValueAtKey(d1, vertexKey) if d1 else None
                id2 = Dictionary.ValueAtKey(d2, vertexKey) if d2 else None
                if id1 == None or id2 == None:
                    continue
                else:
                    i1 = sub_vertex_indices[id1]
                    i2 = sub_vertex_indices[id2]
                    gv1 = mapping[i1]
                    gv2 = mapping[i2]

                    path = Graph.ShortestPath(superGraph, gv1, gv2)
                    if not path:
                        valid = False
                        break
                    elif strict:
                        if Topology.IsInstance(path, "Wire"):
                            if len(Topology.Edges(path)) > 1:
                                valid = False
                                break

            if valid:
                all_matches.append(mapping)

        matched_subgraphs = []
        if len(all_matches) > 0:
            vertex_dictionaries = []
            d = Graph.MeshData(subGraph)
            subgraph_edges = d['edges']
            edge_dictionaries = d['edgeDictionaries']
            positions = []
            for i, mapping in enumerate(all_matches, 1):
                for svid, gv in mapping.items():
                    positions.append(Vertex.Coordinates(gv))
                    sd = Topology.Dictionary(sub_vertices[svid])
                    gd = Topology.Dictionary(gv)
                    vertex_dictionaries.append(Dictionary.ByMergedDictionaries(sd, gd))
                matched_subgraphs.append(Graph.ByMeshData(positions, subgraph_edges, vertexDictionaries=vertex_dictionaries, edgeDictionaries=edge_dictionaries))
        return matched_subgraphs

    @staticmethod
    def _topological_distance(g, start, target):
        from collections import deque
        if start == target:
            return 0
        visited = set()
        queue = deque([(start, 0)])  # Each element is a tuple (vertex, distance)

        while queue:
            current, distance = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            for neighbor in g.get(current, []):
                if neighbor == target:
                    return distance + 1
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return None  # Target not reachable

    @staticmethod
    def Tietze(radius: float = 0.5, height: float = 1):
        """
        Creates a Tietze's graph mapped on a mobius strip of the same input radius and height. See https://en.wikipedia.org/wiki/Tietze%27s_graph

        Parameters
        ----------
        radius : float , optional
            The desired radius of the mobius strip on which the graph is mapped. Default is 0.5.
        height : float , optional
            The desired height of the mobius strip on which the graph is mapped. Default is 1.

        Returns
        -------
        topologicpy.Graph
            The created Tietze's graph.

        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Edge import Edge
        from topologicpy.Shell import Shell
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology

        m = Shell.MobiusStrip(radius=radius, height=height, uSides=12, vSides=3)
        eb = Shell.ExternalBoundary(m)
        verts = Topology.Vertices(eb)
        new_verts = []
        for i in range(0, len(verts), 2): #The mobius strip has 24 edges, we need half of that (12).
            new_verts.append(verts[i])

        graph_edges = []
        for r in range(0,6):
            s = (r + 6)
            e = Edge.ByVertices(new_verts[r], new_verts[s])
            if r == 0:
                v1 = Edge.VertexByParameter(e, 2/3)
                v2 = Edge.EndVertex(e)
                e = Edge.ByVertices(v1, v2)
            elif r == 1:
                v3 = Edge.VertexByParameter(e, 1/3)
                v4 = Edge.VertexByParameter(e, 2/3)
                e = Edge.ByVertices(v3, v4)
            elif r == 2:
                v5 = Edge.StartVertex(e)
                v6 = Edge.VertexByParameter(e, 1/3)
                e = Edge.ByVertices(v5, v6)
            elif r == 3:
                v7 = Edge.VertexByParameter(e, 1/3)
                v8 = Edge.VertexByParameter(e, 2/3)
                e = Edge.ByVertices(v7, v8)
            elif r == 4:
                v9 = Edge.VertexByParameter(e, 2/3)
                v10 = Edge.EndVertex(e)
                e = Edge.ByVertices(v9, v10)
            elif r == 5:
                v11 = Edge.VertexByParameter(e, 1/3)
                v12 = Edge.VertexByParameter(e, 2/3)
                e = Edge.ByVertices(v11, v12)
            graph_edges.append(e)

        graph_vertices= [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]
        graph_edges.append(Edge.ByVertices(v10, v2))
        graph_edges.append(Edge.ByVertices(v5, v10))
        graph_edges.append(Edge.ByVertices(v2, v5))
        graph_edges.append(Edge.ByVertices(v1, v4))
        graph_edges.append(Edge.ByVertices(v4, v8))
        graph_edges.append(Edge.ByVertices(v8, v9))
        graph_edges.append(Edge.ByVertices(v9, v12))
        graph_edges.append(Edge.ByVertices(v12, v3))
        graph_edges.append(Edge.ByVertices(v3, v6))
        graph_edges.append(Edge.ByVertices(v6, v7))
        graph_edges.append(Edge.ByVertices(v7, v11))
        graph_edges.append(Edge.ByVertices(v11, v1))
        graph_edges = [Edge.Reverse(e) for e in graph_edges] #This makes them look better when sagitta is applied.
        graph = Graph.ByVerticesEdges(graph_vertices, graph_edges)
        return graph

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
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        int
            The topological distance between the input vertices.

        """
        
        from topologicpy.Vertex import Vertex
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
        
        g = Graph.AdjacencyDictionary(graph)
        vertices = Graph.Vertices(graph)
        keys = list(g.keys())
        index_a = Vertex.Index(vertexA, vertices, tolerance=tolerance)
        if index_a == None:
            return 0
        start = keys[index_a]
        index_b = Vertex.Index(vertexB, vertices, tolerance=tolerance)
        if index_b == None:
            return 0
        target = keys[index_b]
        return Graph._topological_distance(g, start, target)
    
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
            The input root vertex. If not set, the first vertex in the graph is set as the root vertex. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
                    edge = Graph.Edge(graph, parent, vertex, tolerance=tolerance)
                    ev = Edge.EndVertex(edge)
                    if Vertex.Distance(parent, ev) <= tolerance:
                        edge = Edge.Reverse(edge)
                    edges.append(edge)
            if parent == None:
                parent = vertex
            children = getChildren(vertex, parent, graph, vertices)
            dictionary['vertices'] = vertices
            dictionary['edges'] = edges
            for child in children:
                dictionary = buildTree(graph, dictionary, child, vertex, tolerance=tolerance)
            return dictionary
        
        if not Topology.IsInstance(graph, "Graph"):
            print("Graph.Tree - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            vertex = Graph.Vertices(graph)[0]
        else:
            vertex = Graph.NearestVertex(graph, vertex)
        dictionary = {'vertices':[], 'edges':[]}
        dictionary = buildTree(graph, dictionary, vertex, None, tolerance=tolerance)
        return Graph.ByVerticesEdges(dictionary['vertices'], dictionary['edges'])
    
    @staticmethod
    def VertexDegree(graph, vertex, weightKey: str = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the degree of the input vertex. See https://en.wikipedia.org/wiki/Degree_(graph_theory).

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertex : topologic_core.Vertex
            The input vertex.
        weightKey : str , optional
            If specified, the value in the connected edges' dictionary specified by the weightKey string will be aggregated to calculate
            the vertex degree. If a numeric value cannot be retrieved from an edge, a value of 1 is used instead.
            This is used in weighted graphs. if weightKey is set to "Length" or "Distance", the length of the edge will be used as its weight.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int
            The degree of the input vertex.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import numbers

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.VertexDegree - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Graph.VertexDegree - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        edges = Graph.Edges(graph, [vertex], tolerance=tolerance)
        degree = 0
        for edge in edges:
            if weightKey == None:
                value = 1
            elif "len" in weightKey.lower() or "dis" in weightKey.lower():
                value = Edge.Length(edge, mantissa=mantissa)
            else:
                d = Topology.Dictionary(edge)
                value = Dictionary.ValueAtKey(d, weightKey)
                if not isinstance(value, numbers.Number):
                    value = 1
            degree += value
        return round(degree, mantissa)
    
    @staticmethod
    def Vertices(graph, sortBy: str = None, reverse: bool = False, silent: bool = False):
        """
        Returns the list of vertices in the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        sortBy : str , optional
            The dictionary key to use for sorting the returned edges. Special strings include "length" and "distance" to sort by the length of the edge. Default is None.
        reverse : bool , optional
            If set to True, the sorted list is reversed. This has no effect if the sortBy parameter is not set. Default is False.
        silent : bool, optional
            Isilent : bool, optional
            If set to True, all errors and warnings are suppressed. Default is False.
                    
        Returns
        -------
        list
            The list of vertices in the input graph.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.Vertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertices = []
        if graph:
            try:
                _ = graph.Vertices(vertices) # Hook to Core
            except:
                vertices = []
        if not sortBy == None:
            vertex_values = []
            for v in vertices:
                d = Topology.Dictionary(v)
                value = str(Dictionary.ValueAtKey(d, sortBy, "0"))
                vertex_values.append(value)
            vertices = Helper.Sort(vertices, vertex_values)
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
            The first input list of viewpoints (vertices). Visibility edges will connect these veritces to viewpointsB. If set to None, this parameters will be set to all vertices of the input face. Default is None.
        viewpointsB : list , optional
            The input list of viewpoints (vertices). Visibility edges will connect these vertices to viewpointsA. If set to None, this parameters will be set to all vertices of the input face. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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

    @staticmethod
    def WeightedJaccardSimilarity(graphA, graphB, vertexA, vertexB, vertexIDKey="id", edgeWeightKey=None, mantissa=6, silent=False):
        """
        Computes the weighted Jaccard similarity between two vertices based on their neighbors and
        edge weights. Accepts either one graph (both vertices are in the same graph) or two graphs
        (each vertex is in a separate graph).

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first graph
        graphB : topologic_core.Graph
            The second graph (this can be the same as the first graph)
        vertexA : topologic_core.Vertex
            The first vertex.
        vertexB : topologic_core.Vertex
            The second vertex.
        vertexIDKey : str , optional
            The dictionary key under which to find the unique vertex ID. Default is "id".
        edgeWeightKey : str , optional
            The dictionary key under which to find the weight of the edge for weighted graphs.
            If this parameter is specified as "length" or "distance" then the length of the edge is used as its weight.
            The default is None which means all edges are treated as if they have a weight of 1.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        float
            Weighted Jaccard similarity score between 0 (no overlap) and 1 (perfect match).

        """
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge

        if graphB == None:
            graphB = graphA

        def edge_id(edge, vertexIDKey="id", mantissa=6):
            v1 = Edge.StartVertex(edge)
            v2 = Edge.EndVertex(edge)
            d1 = Topology.Dictionary(v1)
            d2 = Topology.Dictionary(v2)
            v1_id = Dictionary.ValueAtKey(d1, vertexIDKey) if d1 and Dictionary.ValueAtKey(d1, vertexIDKey) is not None else str(sorted(Vertex.Coordinates(v1, mantissa=mantissa)))
            v2_id = Dictionary.ValueAtKey(d2, vertexIDKey) if d2 and Dictionary.ValueAtKey(d2, vertexIDKey) is not None else str(sorted(Vertex.Coordinates(v2, mantissa=mantissa)))
            return tuple(sorted([v1_id, v2_id]))
        
        def get_neighbors_with_weights(graph, vertex, vertexIDKey="id", edgeWeightKey=None, mantissa=6):
            weights = {}
            for edge in Graph.Edges(graph, [vertex]):
                eid = edge_id(edge, vertexIDKey=vertexIDKey, mantissa=mantissa)
                if edgeWeightKey == None:
                    weight = 1.0
                elif edgeWeightKey.lower() == "length" or edgeWeightKey.lower() == "distance":
                    weight = Edge.Length(edge, mantissa=mantissa)
                else:
                    d = Topology.Dictionary(edge)
                    d_weight = Dictionary.ValueAtKey(d, edgeWeightKey, silent=silent)
                    if not d or d_weight is None:
                        if not silent:
                            print(f"Graph.WeightedJaccardSimilarity - Warning: The dictionary of edge {eid} is missing '{edgeWeightKey}' key. Defaulting the edge weight to 1.0.")
                        weight = 1.0
                    else:
                        weight = d_weight
                weights[eid] = weight
            return weights

        weights1 = get_neighbors_with_weights(graphA, vertexA, vertexIDKey=vertexIDKey, edgeWeightKey=edgeWeightKey, mantissa=mantissa)
        weights2 = get_neighbors_with_weights(graphB, vertexB, vertexIDKey=vertexIDKey, edgeWeightKey=edgeWeightKey, mantissa=mantissa)

        keys = set(weights1.keys()) | set(weights2.keys())

        numerator = sum(min(weights1.get(k, 0), weights2.get(k, 0)) for k in keys)
        denominator = sum(max(weights1.get(k, 0), weights2.get(k, 0)) for k in keys)

        return round(numerator / denominator, mantissa) if denominator != 0 else 0.0

    @staticmethod
    def _vertex_is_same(v1, v2, keys=None, tolerance=0.0001):
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if keys == None or keys == [] or keys == "":
            return Vertex.Distance(v1, v2) <= tolerance
        
        if isinstance(keys, str):
            if "loc" in keys.lower() or "coord" in keys.lower() or "xyz" in keys.lower():
                return Vertex.Distance(v1, v2) <= tolerance
        if not isinstance(keys, list):
            keys = [keys]
        
        d1 = Topology.Dictionary(v1)
        d2 = Topology.Dictionary(v2)
        a = [Dictionary.ValueAtKey(d1, k, "0") for k in keys]
        b = [Dictionary.ValueAtKey(d2, k, "1") for k in keys]
        return a == b

    @staticmethod
    def _vertex_in_list(vertex, vertex_list, keys=None, tolerance=0.0001):
        for i, v1 in enumerate(vertex_list):
            if Graph._vertex_is_same(vertex, v1, keys=keys, tolerance=tolerance):
                return i+1
        return False

    @staticmethod
    def _edge_in_list(edge, edge_list, vertices_a, vertices_b, keys=None, tolerance=0.0001):
        sv1 = vertices_a[edge[0]]
        ev1 = vertices_a[edge[1]]
        for i, e in enumerate(edge_list):
            sv2 = vertices_b[e[0]]
            ev2 = vertices_b[e[1]]
            if (Graph._vertex_is_same(sv1, sv2, keys=keys, tolerance=tolerance) and Graph._vertex_is_same(ev1, ev2, keys=keys, tolerance=tolerance)) or \
                (Graph._vertex_is_same(sv1, ev2, keys=keys, tolerance=tolerance) and Graph._vertex_is_same(ev1, sv2, keys=keys, tolerance=tolerance)):
                return i+1
        return False

    @staticmethod
    def Union(graphA, graphB, vertexKeys=None, useCentroid: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Union the two input graphs based on the input vertex keys. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        useCentroid : bool , optional
            If set to True, the coordinates of identical vertices from each graph are averaged to located the new merged vertex of the resulting graph.
            Otherwise, the coordinates of the vertex of the first input graph are used. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are merged.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.Union - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.Union - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None
        vertices_a = Graph.Vertices(graphA)
        vertices_a_new = []
        for v in vertices_a:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_a_new.append(v_new)
        vertices_a = vertices_a_new
        vertices_b = Graph.Vertices(graphB)
        vertices_b_new = []
        for v in vertices_b:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_b_new.append(v_new)
        vertices_b = vertices_b_new
        mesh_data_a = Graph.MeshData(graphA)
        mesh_data_b = Graph.MeshData(graphB)
        edges_a = mesh_data_a['edges']
        edges_b = mesh_data_b['edges']
        edges_a_dicts = mesh_data_a['edgeDictionaries']
        edges_b_dicts = mesh_data_b['edgeDictionaries']

        union_vertices = []

        def _add_vertex(v):
            for i, uv in enumerate(union_vertices):
                if Graph._vertex_is_same(v, uv, keys=vertexKeys):
                    d_a = Topology.Dictionary(v)
                    d_b = Topology.Dictionary(uv)
                    d_c = Dictionary.ByMergedDictionaries(d_a, d_b)
                    if useCentroid:
                        c = Topology.Centroid(Cluster.ByTopologies([v, uv]))
                    else:
                        c = uv
                    c = Topology.SetDictionary(c, d_c)
                    union_vertices[i] = c
                    return i
            union_vertices.append(v)
            return len(union_vertices) - 1

        # Map original vertices to indices in union list
        index_map_a = [_add_vertex(v) for v in vertices_a]
        index_map_b = [_add_vertex(v) for v in vertices_b]

        union_edges = []

        def _add_edge(i, j, dictionary):
            vi = union_vertices[i]
            vj = union_vertices[j]
            for k, e in enumerate(union_edges):
                svi = Edge.StartVertex(e)
                evi = Edge.EndVertex(e)
                if (Graph._vertex_is_same(svi, vi, keys=vertexKeys, tolerance=tolerance) and Graph._vertex_is_same(evi, vj, keys=vertexKeys, tolerance=tolerance)) or \
                (Graph._vertex_is_same(svi, vj, keys=vertexKeys, tolerance=tolerance) and Graph._vertex_is_same(evi, vi, keys=vertexKeys, tolerance=tolerance)):
                    # Merge dictionaries
                    d_a = Topology.Dictionary(e)
                    d_c = Dictionary.ByMergedDictionaries([d_a, dictionary], silent=True)
                    new_edge = Edge.ByVertices(vi, vj)
                    new_edge = Topology.SetDictionary(new_edge, d_c, silent=True)
                    union_edges[k] = new_edge
                    return
            # If not found, add new edge
            edge = Edge.ByVertices(vi, vj)
            edge = Topology.SetDictionary(edge, dictionary)
            union_edges.append(edge)

        # Add edges from A
        for idx, e in enumerate(edges_a):
            i = index_map_a[e[0]]
            j = index_map_a[e[1]]
            if not i == j:
                _add_edge(i, j, Dictionary.ByPythonDictionary(edges_a_dicts[idx]))

        # Add edges from B, merging duplicates
        for idx, e in enumerate(edges_b):
            i = index_map_b[e[0]]
            j = index_map_b[e[1]]
            if not i == j:
                _add_edge(i, j, Dictionary.ByPythonDictionary(edges_b_dicts[idx]))

        return Graph.ByVerticesEdges(union_vertices, union_edges)
    
    @staticmethod
    def Impose(graphA, graphB, vertexKeys=None, useCentroid: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Imposes the second input graph on the first input graph based on the input vertex keys. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        useCentroid : bool , optional
            If set to True, the coordinates of identical vertices from each graph are averaged to located the new merged vertex of the resulting graph.
            Otherwise, the coordinates of the vertex of the second input graph are used. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        

        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are merged.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.Impose - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.Impose - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None
        vertices_a = Graph.Vertices(graphA)
        vertices_a_new = []
        for v in vertices_a:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_a_new.append(v_new)
        vertices_a = vertices_a_new
        vertices_b = Graph.Vertices(graphB)
        vertices_b_new = []
        for v in vertices_b:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_b_new.append(v_new)
        vertices_b = vertices_b_new
        mesh_data_a = Graph.MeshData(graphA)
        mesh_data_b = Graph.MeshData(graphB)
        edges_a = mesh_data_a['edges']
        edges_b = mesh_data_b['edges']
        edges_a_dicts = mesh_data_a['edgeDictionaries']
        edges_b_dicts = mesh_data_b['edgeDictionaries']

        union_vertices = []

        def _add_vertex(v):
            for i, uv in enumerate(union_vertices):
                if Graph._vertex_is_same(v, uv, keys=vertexKeys):
                    d_c = Topology.Dictionary(v) # Dictionaries of graphB are imposed.
                    if useCentroid:
                        c = Topology.Centroid(Cluster.ByTopologies([v, uv]))
                    else:
                        c = v
                    c = Topology.SetDictionary(c, d_c)
                    union_vertices[i] = c
                    return i
            union_vertices.append(v)
            return len(union_vertices) - 1

        # Map original vertices to indices in union list
        index_map_a = [_add_vertex(v) for v in vertices_a]
        index_map_b = [_add_vertex(v) for v in vertices_b]

        union_edges = []

        def _add_edge(i, j, dictionary):
            vi = union_vertices[i]
            vj = union_vertices[j]
            for k, e in enumerate(union_edges):
                svi = Edge.StartVertex(e)
                evi = Edge.EndVertex(e)
                if (Graph._vertex_is_same(svi, vi, keys=vertexKeys, tolerance=tolerance) and Graph._vertex_is_same(evi, vj, keys=vertexKeys, tolerance=tolerance)) or \
                (Graph._vertex_is_same(svi, vj, keys=vertexKeys, tolerance=tolerance) and Graph._vertex_is_same(evi, vi, keys=vertexKeys, tolerance=tolerance)):
                    # Impose edge dictionary from graphB
                    new_edge = Edge.ByVertices(vi, vj)
                    new_edge = Topology.SetDictionary(new_edge, dictionary, silent=True)
                    union_edges[k] = new_edge
                    return
            # If not found, add new edge
            edge = Edge.ByVertices(vi, vj)
            edge = Topology.SetDictionary(edge, dictionary)
            union_edges.append(edge)

        # Add edges from A
        for idx, e in enumerate(edges_a):
            i = index_map_a[e[0]]
            j = index_map_a[e[1]]
            if not i == j:
                _add_edge(i, j, Dictionary.ByPythonDictionary(edges_a_dicts[idx]))

        # Add edges from B, merging duplicates
        for idx, e in enumerate(edges_b):
            i = index_map_b[e[0]]
            j = index_map_b[e[1]]
            if not i == j:
                _add_edge(i, j, Dictionary.ByPythonDictionary(edges_b_dicts[idx]))

        return Graph.ByVerticesEdges(union_vertices, union_edges)
    
    @staticmethod
    def Imprint(graphA, graphB, vertexKeys, useCentroid: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Imprints the second input graph on the first input graph based on the input vertex keys. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        useCentroid : bool , optional
            If set to True, the coordinates of identical vertices from each graph are averaged to located the new merged vertex of the resulting graph.
            Otherwise, the coordinates of the vertex of the first input graph are used. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are merged.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.Imprint - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.Imprint - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None

        vertices_a = Graph.Vertices(graphA)
        vertices_a_new = []
        for v in vertices_a:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_a_new.append(v_new)
        vertices_a = vertices_a_new
        vertices_b = Graph.Vertices(graphB)
        vertices_b_new = []
        for v in vertices_b:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_b_new.append(v_new)
        vertices_b = vertices_b_new
        mesh_data_a = Graph.MeshData(graphA)
        mesh_data_b = Graph.MeshData(graphB)
        topo_edges_a = Graph.Edges(graphA)
        edges_a = mesh_data_a['edges']
        edges_b = mesh_data_b['edges']
        edges_b_dicts = mesh_data_b['edgeDictionaries']

        final_vertices = []
        vertex_map = {}
        for i, a in enumerate(vertices_a):
            j = Graph._vertex_in_list(a, vertices_b, keys=vertexKeys, tolerance=tolerance)
            if j:
                b = vertices_b[j-1]
                if useCentroid:
                    c = Topology.Centroid(Cluster.ByTopologies([a, b]))
                else:
                    c = a
                d_c = Topology.Dictionary(b)
                c = Topology.SetDictionary(c, d_c, silent=True)
                vertex_map[i] = c
                final_vertices.append(c)
            else:
                final_vertices.append(a)
        if len(final_vertices) < 1:
            if not silent:
                print("Graph.Imprint - Warning: graphA and graphB do not intersect. Returning None.")
            return None

        final_edges = []

        for i, e in enumerate(edges_a):
            j = Graph._edge_in_list(e, edges_b, vertices_a, vertices_b, keys=vertexKeys, tolerance=tolerance)
            if j:
                # Merge the dictionaries
                d_c = Dictionary.ByPythonDictionary(edges_b_dicts[j-1]) # We added 1 to j to avoid 0 which can be interpreted as False.
                # Create the edge
                #final_edge = Edge.ByVertices(vertices_a[e[0]], vertices_a[e[1]])
                sv = vertex_map[e[0]]
                ev = vertex_map[e[1]]
                final_edge = Edge.ByVertices(sv, ev)
                # Set the edge's dictionary
                final_edge = Topology.SetDictionary(final_edge, d_c, silent=True)
                # Add the final edge to the list
                final_edges.append(final_edge)
            else:
                final_edges.append(topo_edges_a[i])

        return Graph.ByVerticesEdges(final_vertices, final_edges)

    @staticmethod
    def Intersect(graphA, graphB, vertexKeys, vertexColorKey="color", useCentroid: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Intersect the two input graphs based on the input vertex keys. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        useCentroid : bool , optional
            If set to True, the coordinates of identical vertices from each graph are averaged to located the new merged vertex of the resulting graph.
            Otherwise, the coordinates of the vertex of the first input graph are used. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are merged.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.Intersect - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.Intersect - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None

        vertices_a = Graph.Vertices(graphA)
        vertices_a_new = []
        for v in vertices_a:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_a_new.append(v_new)
        vertices_a = vertices_a_new
        vertices_b = Graph.Vertices(graphB)
        vertices_b_new = []
        for v in vertices_b:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_b_new.append(v_new)
        vertices_b = vertices_b_new
        mesh_data_a = Graph.MeshData(graphA)
        mesh_data_b = Graph.MeshData(graphB)
        edges_a = mesh_data_a['edges']
        edges_b = mesh_data_b['edges']
        edges_a_dicts = mesh_data_a['edgeDictionaries']
        edges_b_dicts = mesh_data_b['edgeDictionaries']

        common_vertices = []
        vertex_map = {}
        for i, a in enumerate(vertices_a):
            j = Graph._vertex_in_list(a, vertices_b, keys=vertexKeys, tolerance=tolerance)
            if j:
                b = vertices_b[j-1]
                if useCentroid:
                    c = Topology.Centroid(Cluster.ByTopologies([a, b]))
                else:
                    c = a
                d_a = Topology.Dictionary(a)
                d_b = Topology.Dictionary(b)
                d_c = Dictionary.ByMergedDictionaries([d_a, d_b], silent=True)
                c = Topology.SetDictionary(c, d_c, silent=True)
                vertex_map[i] = c
                common_vertices.append(c)
        if len(common_vertices) < 1:
            if not silent:
                print("Graph.Intersect - Warning: graphA and graphB do not intersect. Returning None.")
            return None

        common_edges = []

        for i, e in enumerate(edges_a):
            j = Graph._edge_in_list(e, edges_b, vertices_a, vertices_b, keys=vertexKeys, tolerance=tolerance)
            if j:
                # Merge the dictionaries
                d_a = Dictionary.ByPythonDictionary(edges_a_dicts[i])
                d_b = Dictionary.ByPythonDictionary(edges_b_dicts[j-1]) # We added 1 to j to avoid 0 which can be interpreted as False.
                d_c = Dictionary.ByMergedDictionaries([d_a, d_b], silent=True)
                # Create the edge
                #final_edge = Edge.ByVertices(vertices_a[e[0]], vertices_a[e[1]])
                sv = vertex_map[e[0]]
                ev = vertex_map[e[1]]
                final_edge = Edge.ByVertices(sv, ev)
                # Set the edge's dictionary
                final_edge = Topology.SetDictionary(final_edge, d_c, silent=True)
                # Add the final edge to the list
                common_edges.append(final_edge)

        return Graph.ByVerticesEdges(common_vertices, common_edges)

    @staticmethod
    def Difference(graphA, graphB, vertexKeys, useCentroid: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Intersect the two input graphs based on the input vertex keys. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        useCentroid : bool , optional
            This is not used here, but included for API consistency for boolean operations.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are not merged.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.Difference - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.Difference - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None

        vertices_a = Graph.Vertices(graphA)
        vertices_a_new = []
        for v in vertices_a:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_a_new.append(v_new)
        vertices_a = vertices_a_new
        vertices_b = Graph.Vertices(graphB)
        vertices_b_new = []
        for v in vertices_b:
            d = Topology.Dictionary(v)
            v_new = Vertex.ByCoordinates(Vertex.Coordinates(v))
            v_new = Topology.SetDictionary(v_new, d)
            vertices_b_new.append(v_new)
        vertices_b = vertices_b_new
        mesh_data_a = Graph.MeshData(graphA)
        mesh_data_b = Graph.MeshData(graphB)
        edges_a = mesh_data_a['edges']
        edges_b = mesh_data_b['edges']
        edges_a_dicts = mesh_data_a['edgeDictionaries']

        diff_vertices = [v for v in vertices_a if not Graph._vertex_in_list(v, vertices_b, keys=vertexKeys, tolerance=tolerance)]
        diff_edges = []

        for i, e in enumerate(edges_a):
            if not Graph._edge_in_list(e, edges_b, vertices_a, vertices_b, keys=vertexKeys, tolerance=tolerance):
                # Create the edge
                if Graph._vertex_in_list(vertices_a[e[0]], diff_vertices, keys=vertexKeys, tolerance=tolerance) and Graph._vertex_in_list(vertices_a[e[1]], diff_vertices, keys=vertexKeys, tolerance=tolerance):
                    final_edge = Edge.ByVertices(vertices_a[e[0]], vertices_a[e[1]])
                    # Set the edge's dictionary
                    final_edge = Topology.SetDictionary(final_edge, Dictionary.ByPythonDictionary(edges_a_dicts[i]), silent=True)
                    # Add the final edge to the list
                    diff_edges.append(final_edge)

        return Graph.ByVerticesEdges(diff_vertices, diff_edges)

    @staticmethod
    def Merge(graphA, graphB, vertexKeys=None, vertexColorKey="color", useCentroid: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Merges the two input graphs based on the input vertex keys. This is an alias for Graph.Union. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        vertexColorKey : str , optional
            The dictionary key that is storing the vertex's color. The final colors will be averaged. Default is "color".
        useCentroid : bool , optional
            If set to True, the coordinates of identical vertices from each graph are averaged to located the new merged vertex of the resulting graph.
            Otherwise, the coordinates of the vertex of the first input graph are used. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are merged.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.Union - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.Union - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None
        return Graph.Union(graphA, graphB, vertexKeys=vertexKeys, vertexColorKey=vertexColorKey, useCentroid=useCentroid, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def SymmetricDifference(graphA, graphB, vertexKeys, useCentroid: bool = False, tolerance: float = 0.001, silent: bool = False):
        """
        Find the symmetric difference (exclusive OR / XOR) of the two input graphs based on the input vertex keys. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        useCentroid : bool , optional
            This is not used here, but included for API consistency for boolean operations.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are not merged.

        """

        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.SymmetricDifference - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.SymmetricDifference - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None
        diffAB = Graph.Difference(graphA, graphB, vertexKeys=vertexKeys, useCentroid=useCentroid, tolerance=tolerance, silent=True)
        diffBA = Graph.Difference(graphB, graphA, vertexKeys=vertexKeys, useCentroid=useCentroid, tolerance=tolerance, silent=True)
        return Graph.Union(diffAB, diffBA, vertexKeys=vertexKeys, useCentroid=useCentroid, tolerance=tolerance, silent=True)
    
    @staticmethod
    def WLFeatures(graph, key: str = None, iterations: int = 2, silent: bool = False):
        """
        Returns a Weisfeiler-Lehman subtree features for a Graph. See https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test

        Parameters
        ----------
        graph : topologic_core.Graph
            The  input graph.
        key : str , optional
            The vertex key to use as an initial label. Default is None which means the vertex degree is used instead.
        iterations : int , optional
            The desired number of WL iterations. (non-negative int). Default is 2.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        dict
            {feature_id: count} where feature_id is an int representing a WL label.
        """

        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from collections import defaultdict
        
        def _neighbors_map(graph):
            """
            Returns:
                vertices: list of vertex objects in a stable order
                vidx: dict mapping vertex -> index
                nbrs: dict index -> sorted list of neighbor indices
            """
            vertices = Graph.Vertices(graph)
            vidx = {v: i for i, v in enumerate(vertices)}
            nbrs = {}
            for v in vertices:
                i = vidx[v]
                adj = Graph.AdjacentVertices(graph, v) or []
                nbrs[i] = sorted(vidx[a] for a in adj if a in vidx and a is not v)
            return vertices, vidx, nbrs

        def _initial_labels(graph, key=None, default="degree"):
            """
            Returns an integer label per node index using either vertex dictionary labels
            or a structural default (degree or constant).
            """
            vertices, vidx, nbrs = _neighbors_map(graph)
            labels = {}
            if key:
                found_any = False
                tmp = {}
                for v in vertices:
                    d = Topology.Dictionary(v)
                    val = Dictionary.ValueAtKey(d, key)
                    if val is not None:
                        found_any = True
                    tmp[vidx[v]] = str(val) if val is not None else None
                if found_any:
                    # fill missing with a sentinel
                    for i, val in tmp.items():
                        labels[i] = val if val is not None else "__MISSING__"
                else:
                    # fall back to structural init if no labels exist
                    if default == "degree":
                        labels = {i: str(len(nbrs[i])) for i in nbrs}
                    else:
                        labels = {i: "0" for i in nbrs}
            else: # Add a vertex degree information.
                _ = Graph.DegreeCentrality(graph, key="_dc_")
                labels, nbrs = _initial_labels(graph, key="_dc_")
                _ = [Topology.SetDictionary(v, Dictionary.RemoveKey(Topology.Dictionary(v), "_dc_")) for v in Graph.Vertices(graph)]
            return labels, nbrs

        def _canonize_string_labels(str_labels):
            """
            Deterministically map arbitrary strings to dense integer ids.
            Returns:
                int_labels: dict node_index -> int label
                vocab: dict string_label -> int id
            """
            # stable order by string to keep mapping deterministic across runs
            unique = sorted(set(str_labels.values()))
            vocab = {lab: k for k, lab in enumerate(unique)}
            return {i: vocab[s] for i, s in str_labels.items()}, vocab

        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Graph.WLFeatures - Error: The input graph parameter is not a valid topologic graph. Returning None.")
            return None
        
        str_labels, nbrs = _initial_labels(graph, key=key)
        features = defaultdict(int)

        # iteration 0
        labels, _ = _canonize_string_labels(str_labels)
        for lab in labels.values():
            features[lab] += 1

        # WL iterations
        cur = labels
        for _ in range(iterations):
            new_str = {}
            for i in nbrs:
                neigh = [cur[j] for j in nbrs[i]]
                neigh.sort()
                new_str[i] = f"{cur[i]}|{','.join(map(str, neigh))}"
            cur, _ = _canonize_string_labels(new_str)
            for lab in cur.values():
                features[lab] += 1

        return dict(features)

    @staticmethod
    def WLKernel(graphA, graphB, key: str = None, iterations: int = 2, normalize: bool = True, mantissa: int = 6, silent: bool = False):
        """
        Returns a cosine-normalized Weisfeiler-Lehman kernel between two graphs. See https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        key : str , optional
            The vertex key to use as an initial label. Default is None which means the vertex degree is used instead.
        iterations : int , optional
            The desired number of WL iterations. (non-negative int). Default is 2.
        normalize : bool , optional
            if set to True, the returned value is normalized between 0 and 1. Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        
        Returns
        -------
        float
            The cosine-normalized Weisfeiler-Lehman kernel
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graphA, "Graph"):
            if not silent:
                print("Graph.WLFeatures - Error: The input graphA parameter is not a valid topologic graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "Graph"):
            if not silent:
                print("Graph.WLFeatures - Error: The input graphB parameter is not a valid topologic graph. Returning None.")
            return None
        f1 = Graph.WLFeatures(graphA, key=key, iterations=iterations)
        f2 = Graph.WLFeatures(graphB, key=key, iterations=iterations)

        # dot product
        keys = set(f1) | set(f2)
        dot = sum(f1.get(k, 0) * f2.get(k, 0) for k in keys)

        if not normalize:
            return round(float(dot), mantissa)

        import math
        n1 = math.sqrt(sum(v*v for v in f1.values()))
        n2 = math.sqrt(sum(v*v for v in f2.values()))
        return_value = float(dot) / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
        return round(return_value, mantissa)

    @staticmethod
    def XOR(graphA, graphB, vertexKeys, useCentroid: bool = False, tolerance: float = 0.001, silent: bool = False):
        """
        Find the symmetric difference (exclusive OR / XOR) of the two input graphs based on an input vertex key. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        graphA : topologic_core.Graph
            The first input graph.
        graphB : topologic_core.Graph
            The second input graph.
        vertexKeys : list or str , optional
            The vertex dictionary key (str) or keys (list of str) to use to determine if two vertices are the same.
            If the vertexKeys are set to None or "loc" or "coord" or "xyz" (case insensitive), the distance between the
            vertices (within the tolerance) will be used to determine sameness. Default is None.
        useCentroid : bool , optional
            This is not used here, but included for API consistency for boolean operations.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            the resultant graph. Vertex and edge dictionaries are not merged.

        """

        from topologicpy.Topology import Topology

        if not Topology.IsInstance(graphA, "graph"):
            if not silent:
                print("Graph.XOR - Error: The graphA input parameter is not a valid graph. Returning None.")
            return None
        if not Topology.IsInstance(graphB, "graph"):
            if not silent:
                print("Graph.XOR - Error: The graphB input parameter is not a valid graph. Returning None.")
            return None
        diffAB = Graph.Difference(graphA, graphB, vertexKeys=vertexKeys, useCentroid=useCentroid, tolerance=tolerance, silent=True)
        diffBA = Graph.Difference(graphB, graphA, vertexKeys=vertexKeys, useCentroid=useCentroid, tolerance=tolerance, silent=True)
        return Graph.Union(diffAB, diffBA, vertexKeys=vertexKeys, useCentroid=useCentroid, tolerance=tolerance, silent=True)