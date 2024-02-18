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

import topologic
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from topologicpy.Aperture import Aperture
from topologicpy.Vertex import Vertex
import random
import time
import os

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
        raise Exception("Graph - Error: Could not import numpy.")

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
        raise Exception("Graph - Error: Could not import pandas.")

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
        raise Exception("Graph - Error: Could not import tqdm.")

try:
    from pyvis.network import Network
except:
    print("Graph - Installing required pyvis library.")
    try:
        os.system("pip install pyvis")
    except:
        os.system("pip install pyvis --user")
    try:
        from pyvis.network import Network
        print("Graph - pyvis library installed correctly.")
    except:
        raise Exception("Graph - Error: Could not import pyvis")

try:
    import networkx as nx
except:
    print("Graph - Installing required networkx library.")
    try:
        os.system("pip install networkx")
    except:
        os.system("pip install networkx --user")
    try:
        import networkx as nx
        print("Graph - networkx library installed correctly.")
    except:
        raise Exception("Graph - Error: Could not import networkx.")

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
    def AdjacencyMatrix(graph, edgeKeyFwd=None, edgeKeyBwd=None, bidirKey=None, bidirectional=True, useEdgeIndex=False, useEdgeLength=False, tolerance=0.0001):
        """
        Returns the adjacency matrix of the input Graph. See https://en.wikipedia.org/wiki/Adjacency_matrix.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        edgeKeyFwd : str , optional
            If set, the value at this key in the connecting edge from start vertex to end verrtex (forward) will be used instead of the value 1. The default is None. useEdgeIndex and useEdgeLength override this setting.
        edgeKeyBwd : str , optional
            If set, the value at this key in the connecting edge from end vertex to start verrtex (backward) will be used instead of the value 1. The default is None. useEdgeIndex and useEdgeLength override this setting.
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

        if not isinstance(graph, topologic.Graph):
            print("Graph.AdjacencyMatrix - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        vertices = Graph.Vertices(graph)
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
            matrix[svi][evi] = valueFwd
            if bidir:
                matrix[evi][svi] = valueBwd
        return matrix
    
    @staticmethod
    def AdjacencyList(graph, tolerance=0.0001):
        """
        Returns the adjacency list of the input Graph. See https://en.wikipedia.org/wiki/Adjacency_list.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The adjacency list.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        if not isinstance(graph, topologic.Graph):
            print("Graph.AdjacencyList - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertices = Graph.Vertices(graph)
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
    def AddEdge(graph, edge, transferVertexDictionaries=False, transferEdgeDictionaries=False, tolerance=0.0001):
        """
        Adds the input edge to the input Graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        edges : topologic.Edge
            The input edge.
        transferDictionaries : bool, optional
            If set to True, the dictionaries of the edge and its vertices are transfered to the graph.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
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
                            _ = Topology.SetDictionary(gv,d)
                    unique = False
                    returnVertex = gv
                    break
            if unique:
                graph_vertices.append(vertex)
            return [graph_vertices, returnVertex]

        if not isinstance(graph, topologic.Graph):
            print("Graph.AddEdge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(edge, topologic.Edge):
            print("Graph.AddEdge - Error: The input edge is not a valid edge. Returning None.")
            return None
        graph_vertices = Graph.Vertices(graph)
        graph_edges = Graph.Edges(graph, graph_vertices, tolerance)
        vertices = Edge.Vertices(edge)
        new_vertices = []
        for vertex in vertices:
            graph_vertices, nv = addIfUnique(graph_vertices, vertex, tolerance)
            new_vertices.append(nv)
        new_edge = Edge.ByVertices([new_vertices[0], new_vertices[1]], tolerance=tolerance)
        if transferEdgeDictionaries == True:
            _ = Topology.SetDictionary(new_edge, Topology.Dictionary(edge))
        graph_edges.append(new_edge)
        new_graph = Graph.ByVerticesEdges(graph_vertices, graph_edges)
        return new_graph
    
    @staticmethod
    def AddVertex(graph, vertex, tolerance=0.0001):
        """
        Adds the input vertex to the input graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The input graph with the input vertex added to it.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.AddVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
            print("Graph.AddVertex - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        _ = graph.AddVertices([vertex], tolerance)
        return graph

    @staticmethod
    def AddVertices(graph, vertices, tolerance=0.0001):
        """
        Adds the input vertex to the input graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertices : list
            The input list of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The input graph with the input vertex added to it.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.AddVertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertices, list):
            print("Graph.AddVertices - Error: The input list of vertices is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) < 1:
            print("Graph.AddVertices - Error: Could not find any valid vertices in the input list of vertices. Returning None.")
            return None
        _ = graph.AddVertices(vertices, tolerance)
        return graph
    
    @staticmethod
    def AdjacentVertices(graph, vertex):
        """
        Returns the list of vertices connected to the input vertex.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertex : topologic.Vertex
            the input vertex.

        Returns
        -------
        list
            The list of adjacent vertices.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.AdjacentVertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
            print("Graph.AdjacentVertices - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        vertices = []
        _ = graph.AdjacentVertices(vertex, vertices)
        return list(vertices)
    
    @staticmethod
    def AllPaths(graph, vertexA, vertexB, timeLimit=10):
        """
        Returns all the paths that connect the input vertices within the allowed time limit in seconds.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
            The second input vertex.
        timeLimit : int , optional
            The time limit in second. The default is 10 seconds.

        Returns
        -------
        list
            The list of all paths (wires) found within the time limit.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.AllPaths - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.AllPaths - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
            print("Graph.AllPaths - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        paths = []
        _ = graph.AllPaths(vertexA, vertexB, True, timeLimit, paths)
        return paths

    @staticmethod
    def AverageClusteringCoefficient(graph, mantissa=6):
        """
        Returns the average clustering coefficient of the input graph. See https://en.wikipedia.org/wiki/Clustering_coefficient.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The average clustering coefficient of the input graph.

        """
        from topologicpy.Vertex import Vertex
        import topologic

        if not isinstance(graph, topologic.Graph):
            print("Graph.LocalClusteringCoefficient - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        vertices = Graph.Vertices(graph)
        if len(vertices) < 1:
            print("Graph.LocalClusteringCoefficient - Error: The input graph parameter is a NULL graph. Returning None.")
            return None
        if len(vertices) == 1:
            return 0.0
        lcc = Graph.LocalClusteringCoefficient(graph, vertices)
        acc = round(sum(lcc)/float(len(lcc)), mantissa)
        return acc

    @staticmethod
    def BetweenessCentrality(graph, vertices=None, sources=None, destinations=None, tolerance=0.001):
        """
            Returns the betweeness centrality measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the betweeness centrality of all the vertices in the input graph is computed. See https://en.wikipedia.org/wiki/Betweenness_centrality.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. The default is None which means all vertices in the input graph are considered.
        sources : list , optional
            The input list of source vertices. The default is None which means all vertices in the input graph are considered.
        destinations : list , optional
            The input list of destination vertices. The default is None which means all vertices in the input graph are considered.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The betweeness centrality of the input list of vertices within the input graph. The values are in the range 0 to 1.

        """
        from topologicpy.Vertex import Vertex
        import sys
        import subprocess

        def betweeness(vertices, topologies, tolerance=0.001):
            returnList = [0] * len(vertices)
            for topology in topologies:
                t_vertices = Topology.Vertices(topology)
                for t_v in t_vertices:
                    index = Vertex.Index(t_v, vertices, strict=False, tolerance=tolerance)
                    if not index == None:
                        returnList[index] = returnList[index]+1
            return returnList

        if not isinstance(graph, topologic.Graph):
            print("Graph.BetweenessCentrality - Error: The input graph is not a valid graph. Returning None.")
            return None
        graphVertices = Graph.Vertices(graph)
        if not isinstance(vertices, list):
            vertices = graphVertices
        else:
            vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) < 1:
            print("Graph.BetweenessCentrality - Error: The input list of vertices does not contain valid vertices. Returning None.")
            return None
        if not isinstance(sources, list):
            sources = graphVertices
        else:
            sources = [v for v in sources if isinstance(v, topologic.Vertex)]
        if len(sources) < 1:
            print("Graph.BetweenessCentrality - Error: The input list of sources does not contain valid vertices. Returning None.")
            return None
        if not isinstance(destinations, list):
            destinations = graphVertices
        else:
            destinations = [v for v in destinations if isinstance(v, topologic.Vertex)]
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

        values = betweeness(vertices, paths, tolerance=tolerance)
        minValue = min(values)
        maxValue = max(values)
        size = maxValue - minValue
        values = [(v-minValue)/size for v in values]
        return values
    
    @staticmethod
    def ByAdjacencyMatrixCSVPath(path):
        """
        Returns graphs according to the input path. This method assumes the CSV files follow an adjacency matrix schema.

        Parameters
        ----------
        path : str
            The file path to the adjacency matrix CSV file.
        
        Returns
        -------
        topologic.Graph
            The created graph.
        
        """

        # Read the adjacency matrix from CSV file using pandas
        adjacency_matrix_df = pd.read_csv(path, header=None)
        
        # Convert DataFrame to a nested list
        adjacency_matrix = adjacency_matrix_df.values.tolist()
        return Graph.ByAdjacencyMatrix(adjacencyMatrix=adjacency_matrix)

    @staticmethod
    def ByAdjacencyMatrix(adjacencyMatrix, xMin=-0.5, yMin=-0.5, zMin=-0.5, xMax=0.5, yMax=0.5, zMax=0.5):
        """
        Returns graphs according to the input folder path. This method assumes the CSV files follow DGL's schema.

        Parameters
        ----------
        adjacencyMatrix : list
            The adjacency matrix expressed as a nested list of 0s and 1s.
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
        topologic.Graph
            The created graph.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        import  random

        if not isinstance(adjacencyMatrix, list):
            print("Graph.BYAdjacencyMatrix - Error: The input adjacencyMatrix parameter is not a valid list. Returning None.")
            return None
        # Add vertices with random coordinates
        vertices = []
        for i in range(len(adjacencyMatrix)):
            x, y, z = random.uniform(xMin,xMax), random.uniform(yMin,yMax), random.uniform(zMin,zMax)
            vertices.append(Vertex.ByCoordinates(x, y, z))

        # Add edges based on the adjacency matrix
        edges = []
        for i in range(len(adjacencyMatrix)):
            for j in range(i + 1, len(adjacencyMatrix[i])):
                if adjacencyMatrix[i][j] == 1:
                    edges.append(Edge.ByVertices([vertices[i], vertices[j]]))

        # Create the graph using vertices and edges
        if len(vertices) == 0:
            print("Graph.BYAdjacencyMatrix - Error: The graph does not contain any vertices. Returning None.")
            return None
        
        return Graph.ByVerticesEdges(vertices, edges)

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
            The list of dicitonary keys to use to index the edge features. The length of this list must match the length of edge features. The default is [].
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
            verts = [] #This is a list of x,y,z tuples to make sure the vertices have unique locations.
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
                while [x,y,z] in verts:
                    x = x + random.randrange(10000,30000,1000)
                    y = y + random.randrange(4000,6000, 100)
                    z = z + random.randrange(70000,90000, 1000)
                verts.append([x,y,z])
                v = Vertex.ByCoordinates(x,y,z)
                if isinstance(v, topologic.Vertex):
                    if len(nodeFeaturesKeys) == 0:
                        values = [node_id, label, mask, features]
                    else:
                        values = [node_id, label, mask]
                        featureList = features.split(",")
                        featureList = [float(s) for s in featureList]
                        values = [node_id, label, mask]+featureList
                    d = Dictionary.ByKeysValues(node_keys, values)
                    if isinstance(d, topologic.Dictionary):
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
            #vertices = vertices_ds[graph_id]
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
                    edge = Edge.ByVertices([vertices[src_id], vertices[dst_id]], tolerance=tolerance)
                    if isinstance(edge, topologic.Edge):
                        d = Dictionary.ByKeysValues(edge_keys, values)
                        if isinstance(d, topologic.Dictionary):
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
            print("2. Length of Vertices:", len(vertices))
            print("2. Length of Edges:", len(edges))
            g = Graph.ByVerticesEdges(vertices, edges)
            temp_v = Graph.Vertices(g)
            temp_e = Graph.Edges(g)
            print("3. Length of Vertices:", len(temp_v))
            print("3. Length of Edges:", len(temp_e))
            if isinstance(g, topologic.Graph):
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
                if isinstance(d, topologic.Dictionary):
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
            graphs.append(topologic.Graph.ByVerticesEdges(vertices, edges))
        return {'graphs':graphs, 'labels':labels}


    @staticmethod
    def ByIFCFile(file, includeTypes=[], excludeTypes=[], includeRels=[], excludeRels=[], xMin=-0.5, yMin=-0.5, zMin=-0.5, xMax=0.5, yMax=0.5, zMax=0.5):
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
        
        Returns
        -------
        topologic.Graph
            The created graph.
        
        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        import ifcopenshell
        import ifcopenshell.util.placement
        import ifcopenshell.util.element
        import ifcopenshell.util.shape
        import ifcopenshell.geom
        import sys
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
            settings.set(settings.USE_BREP_DATA,False)
            settings.set(settings.SEW_SHELLS,True)
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
                    centroid = Vertex.ByCoordinates(x,y,z)
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
                        si = Vertex.Index(sv, vertices)
                        for destination in destinations:
                            if destination == None:
                                continue
                            ev = vertexAtKeyValue(vertices, key="id", value=destination.id())
                            if ev:
                                ei = Vertex.Index(ev, vertices)
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
        topologic.Graph
            The created graph.
        
        """
        import ifcopenshell
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
            The list of [x,y,z] coordinates of the vertices/
        edges : list
            the list of [i,j] indices into the vertices list to signify and edge that connects vertices[i] to vertices[j].
        vertexDictionaries : list , optional
            The python dictionaries of the vertices (in the same order as the list of vertices).
        edgeDictionaries : list , optional
            The python dictionaries of the edges (in the same order as the list of edges).
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The created graph

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
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
    def ByTopology(topology, direct=True, directApertures=False, viaSharedTopologies=False, viaSharedApertures=False, toExteriorTopologies=False, toExteriorApertures=False, toContents=False, toOutposts=False, idKey="TOPOLOGIC_ID", outpostsKey="outposts", useInternalVertex=True, storeBRep=False, tolerance=0.0001):
        """
        Creates a graph.See https://en.wikipedia.org/wiki/Graph_(discrete_mathematics).

        Parameters
        ----------
        topology : topologic.Topology
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
        useInternalVertex : bool , optional
            If set to True, use an internal vertex to represent the subtopology. Otherwise, use its centroid. The default is False.
        storeBRep : bool , optional
            If set to True, store the BRep of the subtopology in its representative vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The created graph.

        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

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
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
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
                                    keys = []
                                    values = []
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
                                    apList = []
                                    _ = x.Apertures(apList)
                                    if len(apList) > 0:
                                        apTopList = []
                                        for ap in apList:
                                            apTopList.append(ap.Topology())
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
                    if storeBRep:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    tempe = Edge.ByStartVertexEndVertex(vcc, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)


            cells = []
            _ = topology.Cells(None, cells)
            if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                for aCell in cells:
                    if useInternalVertex == True:
                        vCell = Topology.InternalVertex(aCell, tolerance=tolerance)
                    else:
                        vCell = aCell.CenterOfMass()
                    d1 = aCell.GetDictionary()
                    if storeBRep:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(aCell), Topology.Type(aCell), Topology.TypeAsString(aCell)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vCell.SetDictionary(d3)
                    else:
                        _ = vCell.SetDictionary(d1)
                    vertices.append(vCell)
                    faces = []
                    _ = aCell.Faces(None, faces)
                    sharedTopologies = []
                    exteriorTopologies = []
                    sharedApertures = []
                    exteriorApertures = []
                    contents = []
                    _ = aCell.Contents(contents)
                    for aFace in faces:
                        cells = []
                        _ = aFace.Cells(topology, cells)
                        if len(cells) > 1:
                            sharedTopologies.append(aFace)
                            apertures = []
                            _ = aFace.Apertures(apertures)
                            for anAperture in apertures:
                                sharedApertures.append(anAperture)
                        else:
                            exteriorTopologies.append(aFace)
                            apertures = []
                            _ = aFace.Apertures(apertures)
                            for anAperture in apertures:
                                exteriorApertures.append(anAperture)
                    if viaSharedTopologies:
                        for sharedTopology in sharedTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedTopology, tolerance)
                            else:
                                vst = sharedTopology.CenterOfMass()
                            d1 = sharedTopology.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via Shared Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = sharedTopology.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    d1 = content.GetDictionary()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X(), vst2.Y(), vst2.Z())
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if viaSharedApertures:
                        for sharedAperture in sharedApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedAperture.Topology(), tolerance)
                            else:
                                vst = sharedAperture.Topology().CenterOfMass()
                            d1 = sharedAperture.Topology().GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedAperture), Topology.Type(sharedAperture), Topology.TypeAsString(sharedAperture)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via Shared Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            _ = vst.SetDictionary(exteriorTopology.GetDictionary())
                            d1 = exteriorTopology.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    d1 = content.GetDictionary()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exteriorAperture in exteriorApertures:
                            extTop = exteriorAperture.Topology()
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(extTop, tolerance)
                            else:
                                vst = exteriorAperture.Topology().CenterOfMass()
                            d1 = exteriorAperture.Topology().GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorAperture), Topology.Type(exteriorAperture), Topology.TypeAsString(exteriorAperture)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = aCell.Contents(contents)
                        for content in contents:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            d1 = content.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)

            for aCell in cells:
                if useInternalVertex == True:
                    vCell = Topology.InternalVertex(aCell, tolerance=tolerance)
                else:
                    vCell = aCell.CenterOfMass()
                d1 = aCell.GetDictionary()
                if storeBRep:
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(aCell), Topology.Type(aCell), Topology.TypeAsString(aCell)])
                    d3 = mergeDictionaries2([d1, d2])
                    _ = vCell.SetDictionary(d3)
                else:
                    _ = vCell.SetDictionary(d1)
                vertices.append(vCell)
            return [vertices,edges]

        def processCell(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
            vertices = []
            edges = []
            if useInternalVertex == True:
                vCell = Topology.InternalVertex(Topology.Copy(topology), tolerance=tolerance)
            else:
                vCell = topology.CenterOfMass()
            d1 = topology.GetDictionary()
            if storeBRep:
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
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
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
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exteriorAperture in exteriorApertures:
                            extTop = exteriorAperture.Topology()
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(extTop, tolerance)
                            else:
                                vst = exteriorAperture.Topology().CenterOfMass()
                            d1 = exteriorAperture.Topology().GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = topology.Contents(contents)
                        for content in contents:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            d1 = content.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vCell, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
            return [vertices, edges]

        def processShell(item):
            from topologicpy.Face import Face
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
            graph = None
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
                                    apList = []
                                    _ = x.Apertures(apList)
                                    if len(apList) > 0:
                                        apertureExists = True
                                        break
                                if apertureExists:
                                    apTopList = []
                                    for ap in apList:
                                        apTopList.append(ap.Topology())
                                    if useInternalVertex == True:
                                        v1 = Topology.InternalVertex(topFaces[i], tolerance=tolerance)
                                        v2 = Topology.InternalVertex(topFaces[j], tolerance=tolerance)
                                    else:
                                        v1 = topFaces[i].CenterOfMass()
                                        v2 = topFaces[j].CenterOfMass()
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    mDict = mergeDictionaries(apTopList)
                                    if mDict:
                                        e.SetDictionary(mDict)
                                    edges.append(e)

            topFaces = []
            _ = topology.Faces(None, topFaces)
            if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                for aFace in topFaces:
                    if useInternalVertex == True:
                        vFace = Topology.InternalVertex(aFace, tolerance=tolerance)
                    else:
                        vFace = aFace.CenterOfMass()
                    _ = vFace.SetDictionary(aFace.GetDictionary())
                    vertices.append(vFace)
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
                            apertures = []
                            _ = anEdge.Apertures(apertures)
                            for anAperture in apertures:
                                sharedApertures.append(anAperture)
                        else:
                            exteriorTopologies.append(anEdge)
                            apertures = []
                            _ = anEdge.Apertures(apertures)
                            for anAperture in apertures:
                                exteriorApertures.append(anAperture)
                    if viaSharedTopologies:
                        for sharedTopology in sharedTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedTopology, tolerance)
                            else:
                                vst = sharedTopology.CenterOfMass()
                            d1 = sharedTopology.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via Shared Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = sharedTopology.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if viaSharedApertures:
                        for sharedAperture in sharedApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedAperture.Topology(), tolerance)
                            else:
                                vst = sharedAperture.Topology().CenterOfMass()
                            d1 = sharedAperture.Topology().GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(Aperture.Topology(sharedAperture)), Topology.Type(Aperture.Topology(sharedAperture)), Topology.TypeAsString(Aperture.Topology(sharedAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via Shared Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            d1 = exteriorTopology.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exteriorAperture in exteriorApertures:
                            extTop = exteriorAperture.Topology()
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(extTop, tolerance)
                            else:
                                vst = exteriorAperture.Topology().CenterOfMass()
                            d1 = exteriorAperture.Topology().GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = aFace.Contents(contents)
                        for content in contents:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            d1 = content.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)

            for aFace in topFaces:
                if useInternalVertex == True:
                    vFace = Topology.InternalVertex(aFace, tolerance)
                else:
                    vFace = aFace.CenterOfMass()
                d1 = aFace.GetDictionary()
                if storeBRep:
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
                    if storeBRep:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    tempe = Edge.ByStartVertexEndVertex(vcc, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            return [vertices, edges]

        def processFace(item):
            from topologic.Face import Face
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
            graph = None
            vertices = []
            edges = []

            if useInternalVertex == True:
                vFace = Topology.InternalVertex(topology, tolerance=tolerance)
            else:
                vFace = topology.CenterOfMass()
            d1 = topology.GetDictionary()
            if storeBRep:
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
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                fEdges = []
                _ = topology.Edges(None, fEdges)
                exteriorTopologies = []
                exteriorApertures = []

                for anEdge in fEdges:
                    exteriorTopologies.append(anEdge)
                    apertures = []
                    _ = anEdge.Apertures(apertures)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            d1 = exteriorTopology.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = exteriorTopology.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exteriorAperture in exteriorApertures:
                            extTop = exteriorAperture.Topology()
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(extTop, tolerance)
                            else:
                                vst = exteriorAperture.Topology().CenterOfMass()
                            d1 = exteriorAperture.Topology().GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = topology.Contents(contents)
                        for content in contents:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            d1 = content.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vFace, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
            return [vertices, edges]

        def processWire(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
            graph = None
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
                                    v1 = topologic.EdgeUtility.PointAtParameter(topEdges[i], 0.5)
                                except:
                                    v1 = topEdges[j].CenterOfMass()
                                try:
                                    v2 = topologic.EdgeUtility.PointAtParameter(topEdges[j], 0.5)
                                except:
                                    v2 = topEdges[j].CenterOfMass()
                                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                mDict = mergeDictionaries(sharedt)
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
                                    apList = []
                                    _ = x.Apertures(apList)
                                    if len(apList) > 0:
                                        apertureExists = True
                                        break
                                if apertureExists:
                                    try:
                                        v1 = topologic.EdgeUtility.PointAtParameter(topEdges[i], 0.5)
                                    except:
                                        v1 = topEdges[j].CenterOfMass()
                                    try:
                                        v2 = topologic.EdgeUtility.PointAtParameter(topEdges[j], 0.5)
                                    except:
                                        v2 = topEdges[j].CenterOfMass()
                                    e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance)
                                    apTopologies = []
                                    for ap in apList:
                                        apTopologies.append(ap.Topology())
                                    mDict = mergeDictionaries(apTopologies)
                                    if mDict:
                                        e.SetDictionary(mDict)
                                    edges.append(e)

            topEdges = []
            _ = topology.Edges(None, topEdges)
            if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                for anEdge in topEdges:
                    try:
                        vEdge = topologic.EdgeUtility.PointAtParameter(anEdge, 0.5)
                    except:
                        vEdge = anEdge.CenterOfMass()
                    d1 = anEdge.GetDictionary()
                    if storeBRep:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(anEdge), Topology.Type(anEdge), Topology.TypeAsString(anEdge)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vEdge.SetDictionary(d3)
                    else:
                        _ = vEdge.SetDictionary(d1)
                    vertices.append(vEdge)
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
                            apertures = []
                            _ = aVertex.Apertures(apertures)
                            for anAperture in apertures:
                                sharedApertures.append(anAperture)
                        else:
                            exteriorTopologies.append(aVertex)
                            apertures = []
                            _ = aVertex.Apertures(apertures)
                            for anAperture in apertures:
                                exteriorApertures.append(anAperture)
                    if viaSharedTopologies:
                        for sharedTopology in sharedTopologies:
                            vst = sharedTopology.CenterOfMass()
                            d1 = sharedTopology.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via Shared Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = sharedTopology.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if viaSharedApertures:
                        for sharedAperture in sharedApertures:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(sharedAperture.Topology(), tolerance)
                            else:
                                vst = sharedAperture.Topology().CenterOfMass()
                            d1 = sharedAperture.Topology().GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(Aperture.Topology(sharedAperture)), Topology.Type(Aperture.Topology(sharedAperture)), Topology.TypeAsString(Aperture.Topology(sharedAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via Shared Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            vst = exteriorTopology
                            vertices.append(exteriorTopology)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = vst.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exteriorAperture in exteriorApertures:
                            extTop = exteriorAperture.Topology()
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(extTop, tolerance)
                            else:
                                vst = extTop.CenterOfMass()
                            d1 = extTop.GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(extTop), Topology.Type(extTop), Topology.TypeAsString(extTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toContents:
                        contents = []
                        _ = anEdge.Contents(contents)
                        for content in contents:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(content, tolerance)
                            else:
                                vst = content.CenterOfMass()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            d1 = content.GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X(), vst.Y(), vst.Z())
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
            for anEdge in topEdges:
                try:
                    vEdge = topologic.EdgeUtility.PointAtParameter(anEdge, 0.5)
                except:
                    vEdge = anEdge.CenterOfMass()
                d1 = anEdge.GetDictionary()
                if storeBRep:
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
                    if storeBRep:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    tempe = Edge.ByStartVertexEndVertex(vcc, vop, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            return [vertices, edges]

        def processEdge(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
            graph = None
            vertices = []
            edges = []

            if useInternalVertex == True:
                try:
                    vEdge = topologic.EdgeUtility.PointAtParameter(topology, 0.5)
                except:
                    vEdge = topology.CenterOfMass()
            else:
                vEdge = topology.CenterOfMass()

            d1 = vEdge.GetDictionary()
            if storeBRep:
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
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            if (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                eVertices = []
                _ = topology.Vertices(None, eVertices)
                exteriorTopologies = []
                exteriorApertures = []
                for aVertex in eVertices:
                    exteriorTopologies.append(aVertex)
                    apertures = []
                    _ = aVertex.Apertures(apertures)
                    for anAperture in apertures:
                        exteriorApertures.append(anAperture)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(exteriorTopology, tolerance)
                            else:
                                vst = exteriorTopology.CenterOfMass()
                            d1 = exteriorTopology.GetDictionary()
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Topologies"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                            if toContents:
                                contents = []
                                _ = vst.Contents(contents)
                                for content in contents:
                                    if useInternalVertex == True:
                                        vst2 = Topology.InternalVertex(content, tolerance)
                                    else:
                                        vst2 = content.CenterOfMass()
                                    vst2 = topologic.Vertex.ByCoordinates(vst2.X()+(tolerance*100), vst2.Y()+(tolerance*100), vst2.Z()+(tolerance*100))
                                    d1 = content.GetDictionary()
                                    if storeBRep:
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = Edge.ByStartVertexEndVertex(vst, vst2, tolerance=tolerance)
                                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                                    _ = tempe.SetDictionary(tempd)
                                    edges.append(tempe)
                    if toExteriorApertures:
                        for exteriorAperture in exteriorApertures:
                            extTop = exteriorAperture.Topology()
                            if useInternalVertex == True:
                                vst = Topology.InternalVertex(extTop, tolerance)
                            else:
                                vst = exteriorAperture.Topology().CenterOfMass()
                            d1 = exteriorAperture.Topology().GetDictionary()
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            if storeBRep:
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            _ = vst.SetDictionary(exteriorAperture.Topology().GetDictionary())
                            vertices.append(vst)
                            tempe = Edge.ByStartVertexEndVertex(vEdge, vst, tolerance=tolerance)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Exterior Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    
            return [vertices, edges]

        def processVertex(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
            vertices = [topology]
            edges = []

            if toContents:
                contents = []
                _ = topology.Contents(contents)
                for content in contents:
                    if useInternalVertex == True:
                        vst = Topology.InternalVertex(content, tolerance)
                    else:
                        vst = content.CenterOfMass()
                    d1 = content.GetDictionary()
                    vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                    if storeBRep:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.BREPString(content), Topology.Type(content), Topology.TypeAsString(content)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vst.SetDictionary(d3)
                    else:
                        _ = vst.SetDictionary(d1)
                    vertices.append(vst)
                    tempe = Edge.ByStartVertexEndVertex(topology, vst, tolerance=tolerance)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
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
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            return [vertices, edges]

        
        if not isinstance(topology, topologic.Topology):
            print("Graph.ByTopology - Error: The input topology is not a valid topology. Returning None.")
            return None
        graph = None
        item = [topology, None, None, None, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, None, useInternalVertex, storeBRep, tolerance]
        vertices = []
        edges = []
        if isinstance(topology, topologic.CellComplex):
            vertices, edges = processCellComplex(item)
        elif isinstance(topology, topologic.Cell):
            vertices, edges = processCell(item)
        elif isinstance(topology, topologic.Shell):
            vertices, edges = processShell(item)
        elif isinstance(topology, topologic.Face):
            vertices, edges = processFace(item)
        elif isinstance(topology, topologic.Wire):
            vertices, edges = processWire(item)
        elif isinstance(topology, topologic.Edge):
            vertices, edges = processEdge(item)
        elif isinstance(topology, topologic.Vertex):
            vertices, edges = processVertex(item)
        elif isinstance(topology, topologic.Cluster):
            c_cellComplexes = Topology.CellComplexes(topology)
            c_cells = Cluster.FreeCells(topology, tolerance=tolerance)
            c_shells = Cluster.FreeShells(topology, tolerance=tolerance)
            c_faces = Cluster.FreeFaces(topology, tolerance=tolerance)
            c_wires = Cluster.FreeWires(topology, tolerance=tolerance)
            c_edges = Cluster.FreeEdges(topology, tolerance=tolerance)
            c_vertices = Cluster.FreeVertices(topology, tolerance=tolerance)
            others = c_cellComplexes+c_cells+c_shells+c_faces+c_wires+c_edges+c_vertices
            parameters = [others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance]

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
        return topologic.Graph.ByVerticesEdges(vertices, edges)
    
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
        topologic.Graph
            The created graph.

        """
        if not isinstance(vertices, list):
            print("Graph.ByVerticesEdges - Error: The input list of vertices is not a valid list. Returning None.")
            return None
        if not isinstance(edges, list):
            print("Graph.ByVerticesEdges - Error: The input list of edges is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        edges = [e for e in edges if isinstance(e, topologic.Edge)]
        return topologic.Graph.ByVerticesEdges(vertices, edges)
    
    @staticmethod
    def Color(graph, vertices=None, key="color", delta=1, tolerance=0.0001):
        """
        Colors the input vertices within the input graph. The saved value is an integer rather than an actual color. See Color.ByValueInRange to convert to an actual color. Any vertices that have been pre-colored will not be affected. See https://en.wikipedia.org/wiki/Graph_coloring.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertices : list , optional
            The input list of graph vertices. If no vertices are specified, all vertices in the input graph are colored. The default is None.
        key : str , optional
            The dictionary key to use to save the color information.
        delta : int , optional
            The desired minimum delta value between the assigned colors.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The colored list of vertices.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import math

        delta = max(abs(delta), 1) # Ensure that delta is never less than 1

        def satisfiesCondition(i, used_colors, delta):
            if delta == 1:
                return i not in used_colors
            else:
                for j in used_colors:
                    if abs(j-i) < delta:
                        return False
                return True
        def color_graph(graph, vertices, key, delta):
            # Create a dictionary to store the colors of each vertex
            colors = {}                
            # Iterate over each vertex in the graph
            for j, vertex in enumerate(vertices):
                d = Topology.Dictionary(vertex)
                color_value = Dictionary.ValueAtKey(d, key)
                if color_value != None:
                    colors[j] = color_value
                # Initialize an empty set of used colors
                used_colors = set()

                # Iterate over each neighbor of the vertex
                for neighbor in Graph.AdjacentVertices(graph, vertex):
                    # If the neighbor has already been colored, add its color to the used colors set
                    index = Vertex.Index(neighbor, vertices)
                    if index in colors:
                        used_colors.add(colors[index])

                if color_value == None:
                    # Choose the smallest unused color for the vertex
                    for i in range(0,int(math.ceil(len(vertices)*int(math.ceil(delta)))), int(math.ceil(delta))):
                        #if i not in used_colors:
                        if satisfiesCondition(i, used_colors, int(math.ceil(delta))):
                            v_d = Topology.Dictionary(vertex)
                            if not v_d == None:
                                keys = Dictionary.Keys(v_d)
                                values = Dictionary.Values(v_d)
                            else:
                                keys = []
                                values = []
                            if len(keys) > 0:
                                keys.append(key)
                                values.append(i)
                            else:
                                keys = [key]
                                values = [i]
                            d = Dictionary.ByKeysValues(keys, values)
                            vertex = Topology.SetDictionary(vertex, d)
                            colors[j] = i
                            break

            return colors

        if not isinstance(graph, topologic.Graph):
            print("Graph.Color - Error: The input graph is not a valid graph. Returning None.")
            return None
        if vertices == None:
            vertices = Graph.Vertices(graph)
        vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) == 0:
            print("Graph.Color - Error: The input list of vertices does not contain any valid vertices. Returning None.")
            return None
        graph_vertices = [Graph.NearestVertex(graph,v) for v in vertices]
        degrees = [Graph.VertexDegree(graph, v) for v in graph_vertices]
        graph_vertices = Helper.Sort(graph_vertices, degrees)
        graph_vertices.reverse()
        _ = color_graph(graph, graph_vertices, key, delta)
        return graph_vertices

    @staticmethod
    def ClosenessCentrality(graph, vertices=None, tolerance = 0.0001):
        """
        Return the closeness centrality measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the closeness centrality of all the vertices in the input graph is computed. See https://en.wikipedia.org/wiki/Closeness_centrality.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The closeness centrality of the input list of vertices within the input graph. The values are in the range 0 to 1.

        """

        if not isinstance(graph, topologic.Graph):
            print("Graph.ClosenessCentrality - Error: The input graph is not a valid graph. Returning None.")
            return None
        graphVertices = Graph.Vertices(graph)
        if not isinstance(vertices, list):
            vertices = graphVertices
        else:
            vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) < 1:
            print("Graph.ClosenessCentrality - Error: The input list of vertices does not contain any valid vertices. Returning None.")
            return None
        n = len(graphVertices)

        returnList = []
        try:
            for va in tqdm(vertices, desc="Computing Closeness Centrality", leave=False):
                top_dist = 0
                for vb in graphVertices:
                    if topologic.Topology.IsSame(va, vb):
                        d = 0
                    else:
                        d = Graph.TopologicalDistance(graph, va, vb, tolerance)
                    top_dist += d
                if top_dist == 0:
                    returnList.append(0)
                else:
                    returnList.append((n-1)/top_dist)
        except:
            print("Could not use tqdm")
            for va in vertices:
                top_dist = 0
                for vb in graphVertices:
                    if topologic.Topology.IsSame(va, vb):
                        d = 0
                    else:
                        d = Graph.TopologicalDistance(graph, va, vb, tolerance)
                    top_dist += d
                if top_dist == 0:
                    returnList.append(0)
                else:
                    returnList.append((n-1)/top_dist)
        return returnList

    @staticmethod
    def Connect(graph, verticesA, verticesB, tolerance=0.0001):
        """
        Connects the two lists of input vertices.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        verticesA : list
            The first list of input vertices.
        verticesB : topologic.Vertex
            The second list of input vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The input graph with the connected input vertices.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Connect - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(verticesA, list):
            print("Graph.Connect - Error: The input list of verticesA is not a valid list. Returning None.")
            return None
        if not isinstance(verticesB, list):
            print("Graph.Connect - Error: The input list of verticesB is not a valid list. Returning None.")
            return None
        verticesA = [v for v in verticesA if isinstance(v, topologic.Vertex)]
        verticesB = [v for v in verticesB if isinstance(v, topologic.Vertex)]
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
        graph : topologic.Graph
            The input graph.
        edge : topologic.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input graph contains the input edge. False otherwise.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.ContainsEdge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(edge, topologic.Edge):
            print("Graph.ContainsEdge - Error: The input edge is not a valid edge. Returning None.")
            return None
        return graph.ContainsEdge(edge, tolerance)
    
    @staticmethod
    def ContainsVertex(graph, vertex, tolerance=0.0001):
        """
        Returns True if the input graph contains the input Vertex. Returns False otherwise.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertex : topologic.Vertex
            The input Vertex.
        tolerance : float , optional
            Ther desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input graph contains the input vertex. False otherwise.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.ContainsVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
            print("Graph.ContainsVertex - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        return graph.ContainsVertex(vertex, tolerance)

    @staticmethod
    def DegreeSequence(graph):
        """
        Returns the degree sequence of the input graph. See https://mathworld.wolfram.com/DegreeSequence.html.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        list
            The degree sequence of the input graph.

        """
        if not isinstance(graph, topologic.Graph):
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
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        float
            The density of the input graph.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Density - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.Density()
    
    @staticmethod
    def DepthMap(graph, vertices=None, tolerance=0.0001):
        """
        Return the depth map of the input list of vertices within the input graph. The returned list contains the total of the topological distances of each vertex to every other vertex in the input graph. The order of the depth map list is the same as the order of the input list of vertices. If no vertices are specified, the depth map of all the vertices in the input graph is computed.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The depth map of the input list of vertices within the input graph.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.DepthMap - Error: The input graph is not a valid graph. Returning None.")
            return None
        graphVertices = Graph.Vertices(graph)
        if not isinstance(vertices, list):
            vertices = graphVertices
        else:
            vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) < 1:
            print("Graph.DepthMap - Error: The input list of vertices does not contain any valid vertices. Returning None.")
            return None
        depthMap = []
        for va in vertices:
            depth = 0
            for vb in graphVertices:
                if topologic.Topology.IsSame(va, vb):
                    dist = 0
                else:
                    dist = Graph.TopologicalDistance(graph, va, vb, tolerance)
                depth = depth + dist
            depthMap.append(depth)
        return depthMap
    
    @staticmethod
    def Diameter(graph):
        """
        Returns the diameter of the input graph. See https://mathworld.wolfram.com/GraphDiameter.html.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        int
            The diameter of the input graph.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Diameter - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.Diameter()
    
    @staticmethod
    def Dictionary(graph):
        """
        Returns the dictionary of the input graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        topologic.Dictionary
            The dictionary of the input graph.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Dictionary - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        return graph.GetDictionary()
    
    @staticmethod
    def Distance(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Returns the shortest-path distance between the input vertices. See https://en.wikipedia.org/wiki/Distance_(graph_theory).

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
            The second input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        int
            The shortest-path distance between the input vertices.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Distance - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.Distance - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
            print("Graph.Distance - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        return graph.TopologicalDistance(vertexA, vertexB, tolerance)

    @staticmethod
    def Edge(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Returns the edge in the input graph that connects in the input vertices.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
            The second input Vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Edge
            The edge in the input graph that connects the input vertices.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Edge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.Edge - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
            print("Graph.Edge - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        return graph.Edge(vertexA, vertexB, tolerance)
    
    @staticmethod
    def Edges(graph, vertices=None, tolerance=0.0001):
        """
        Returns the edges found in the input graph. If the input list of vertices is specified, this method returns the edges connected to this list of vertices. Otherwise, it returns all graph edges.

        Parameters
        ----------
        graph : topologic.Graph
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
        if not isinstance(graph, topologic.Graph):
            print("Graph.Edges - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not vertices:
            edges = []
            _ = graph.Edges(edges, tolerance)
            return edges
        else:
            vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
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
    def ExportToCSV(graph, path, graphLabel, graphFeatures="",  
                       graphIDHeader="graph_id", graphLabelHeader="label", graphFeaturesHeader="feat",
                       
                       edgeLabelKey="label", defaultEdgeLabel=0, edgeFeaturesKeys=[],
                       edgeSRCHeader="src_id", edgeDSTHeader="dst_id",
                       edgeLabelHeader="label", edgeFeaturesHeader="feat",
                       edgeTrainMaskHeader="train_mask", edgeValidateMaskHeader="val_mask", edgeTestMaskHeader="test_mask",
                       edgeMaskKey=None,
                       edgeTrainRatio=0.8, edgeValidateRatio=0.1, edgeTestRatio=0.1,
                       bidirectional=True,

                       nodeLabelKey="label", defaultNodeLabel=0, nodeFeaturesKeys=[],
                       nodeIDHeader="node_id", nodeLabelHeader="label", nodeFeaturesHeader="feat",
                       nodeTrainMaskHeader="train_mask", nodeValidateMaskHeader="val_mask", nodeTestMaskHeader="test_mask",
                       nodeMaskKey=None,
                       nodeTrainRatio=0.8, nodeValidateRatio=0.1, nodeTestRatio=0.1,
                       mantissa=6, overwrite=False):
        """
        Exports the input graph into a set of CSV files compatible with DGL.

        Parameters
        ----------
        graph : topologic.Graph
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
        import os
        from os.path import exists
        
        
        if not isinstance(graph, topologic.Graph):
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
                    print("keys", keys)
                else:
                    keys = []
                    flag = True
                if nodeMaskKey in keys:
                    value = Dictionary.ValueAtKey(nd, nodeMaskKey)
                    if not value in [0,1,2]:
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
            single_node_data = [graph_id, i, vLabel, train_mask, validate_mask, test_mask, node_features, round(float(Vertex.X(v)),mantissa), round(float(Vertex.Y(v)),mantissa), round(float(Vertex.Z(v)),mantissa)]
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
                    if not value in [0,1,2]:
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
            src = Vertex.Index(Edge.StartVertex(edge), vertices)
            dst = Vertex.Index(Edge.EndVertex(edge), vertices)
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
    def Flatten(graph, layout="spring", k=0.8, seed=None, iterations=50, rootVertex=None, tolerance=0.0001):
        """
        Flattens the input graph.

        Parameters
        ----------
        graph : topologic.Graph
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
        rootVertex : topologic.Vertex , optional
            The desired vertex to use as the root of the tree and radial layouts.

        Returns
        -------
        topologic.Graph
            The flattened graph.

        """
        import math
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

            # Get the number of nodes from the edge list.
            num_nodes = max([max(edge) for edge in edge_list]) + 1

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

        if not isinstance(graph, topologic.Graph):
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
            root_index = Vertex.Index(rootVertex, vertices)

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
        graph : topologic.Graph
            The input graph.
        Returns
        -------
        int
            The computed global clustering coefficient.

        """
        import topologic

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
            print("Total Triangles:", total_triangles )
            print("Total Possible Triangles:", total_possible_triangles )
            global_clustering_coeff = 3.0 * total_triangles / total_possible_triangles if total_possible_triangles > 0 else 0.0

            return global_clustering_coeff

        if not isinstance(graph, topologic.Graph):
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
        graph : topologic.Graph
            The input graph.
        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Guid - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        return graph.GetGUID()

    @staticmethod
    def IsBipartite(graph, tolerance=0.0001):
        """
        Returns True if the input graph is bipartite. Returns False otherwise. See https://en.wikipedia.org/wiki/Bipartite_graph.

        Parameters
        ----------
        graph : topologic.Graph
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
        if not isinstance(graph, topologic.Graph):
            print("Graph.IsBipartite - Error: The input graph is not a valid graph. Returning None.")
            return None
        order = Graph.Order(graph)
        adjList = Graph.AdjacencyList(graph, tolerance)
        return isBipartite(order, adjList)

    @staticmethod
    def IsComplete(graph):
        """
        Returns True if the input graph is complete. Returns False otherwise. See https://en.wikipedia.org/wiki/Complete_graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        bool
            True if the input graph is complete. False otherwise

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.IsComplete - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.IsComplete()
    
    @staticmethod
    def IsErdoesGallai(graph, sequence):
        """
        Returns True if the input sequence satisfies the Erdős–Gallai theorem. Returns False otherwise. See https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93Gallai_theorem.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        sequence : list
            The input sequence.

        Returns
        -------
        bool
            True if the input sequence satisfies the Erdős–Gallai theorem. False otherwise.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.IsErdoesGallai - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.IsErdoesGallai(sequence)
    
    @staticmethod
    def IsolatedVertices(graph):
        """
        Returns the list of isolated vertices in the input graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        list
            The list of isolated vertices.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.IsolatedVertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertices = []
        _ = graph.IsolatedVertices(vertices)
        return vertices
    
    @staticmethod
    def LocalClusteringCoefficient(graph, vertices=None, mantissa=6):
        """
        Returns the local clustering coefficient of the input list of vertices within the input graph. See https://en.wikipedia.org/wiki/Clustering_coefficient.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertices : list , optional
            The input list of vertices. If set to None, the local clustering coefficient of all vertices will be computed.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        list
            The list of local clustering coefficient. The order of the list matches the order of the list of input vertices.

        """
        from topologicpy.Vertex import Vertex
        import topologic

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
        if not isinstance(graph, topologic.Graph):
            print("Graph.LocalClusteringCoefficient - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if vertices == None:
            vertices = Graph.Vertices(graph)
        if isinstance(vertices, topologic.Vertex):
            vertices = [vertices]
        vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) < 1:
            print("Graph.LocalClusteringCoefficient - Error: The input vertices parameter does not contain valid vertices. Returning None.")
            return None
        g_vertices = Graph.Vertices(graph)
        adjacency_matrix = Graph.AdjacencyMatrix(graph)
        lcc = []
        for v in vertices:
            i = Vertex.Index(v, g_vertices)
            if not i == None:
                lcc.append(round(local_clustering_coefficient(adjacency_matrix, i), mantissa))
            else:
                lcc.append(None)
        return lcc
    
    @staticmethod
    def LongestPath(graph, vertexA, vertexB, vertexKey=None, edgeKey=None, costKey=None, timeLimit=10, tolerance=0.0001):
        """
        Returns the longest path that connects the input vertices.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
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
        topologic.Wire
            The longest path between the input vertices.

        """
        from topologicpy. Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
    
        if not isinstance(graph, topologic.Graph):
            print("Graph.LongestPath - Error: the input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.LongestPath - Error: the input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
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
                    index = Vertex.Index(vertex, g_vertices)
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
            if isinstance(longest_path, topologic.Edges):
                longest_path = Edge.Reverse(longest_path)
            if isinstance(longest_path, topologic.Wire):
                longest_path = Wire.Reverse(longest_path)
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
        graph : topologic.Graph
            the input graph.

        Returns
        -------
        int
            The maximum delta.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.MaximumDelta - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.MaximumDelta()
    
    @staticmethod
    def MaximumFlow(graph, source, sink, edgeKeyFwd=None, edgeKeyBwd=None, bidirKey=None, bidirectional=False, residualKey="residual", tolerance=0.0001):
        """
        Returns the maximum flow of the input graph. See https://en.wikipedia.org/wiki/Maximum_flow_problem 

        Parameters
        ----------
        graph : topologic.Graph
            The input graph. This is assumed to be a directed graph
        source : topologic.Vertex
            The input source vertex.
        sink : topologic.Vertex
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
        sourceIndex = Vertex.Index(source, vertices)
        sinkIndex = Vertex.Index(sink, vertices)
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
                edge = Topology.SetDictionary(edge,d)
        return max_flow

    @staticmethod
    def MeshData(g):
        """
        Returns the mesh data of the input graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        dict
            The python dictionary of the mesh data of the input graph. The keys in the dictionary are:
            'vertices' : The list of [x,y,z] coordinates of the vertices.
            'edges' : the list of [i,j] indices into the vertices list to signify and edge that connects vertices[i] to vertices[j].
            'vertexDictionaries' : The python dictionaries of the vertices (in the same order as the list of vertices).
            'edgeDictionaries' : The python dictionaries of the edges (in the same order as the list of edges).

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
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
            si = Vertex.Index(sv, g_vertices)
            ei = Vertex.Index(ev, g_vertices)
            m_edges.append([si, ei])
            d = Dictionary.PythonDictionary(Topology.Dictionary(g_edge))
            e_dicts.append(d)
        return {'vertices':m_vertices,
                'edges': m_edges,
                'vertexDictionaries': v_dicts,
                'edgeDictionaries': e_dicts
                }
    
    @staticmethod
    def MinimumDelta(graph):
        """
        Returns the minimum delta of the input graph. The minimum delta of a graph is the minimum degree of a vertex in the graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        int
            The minimum delta.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.MinimumDelta - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.MinimumDelta()
    
    @staticmethod
    def MinimumSpanningTree(graph, edgeKey=None, tolerance=0.0001):
        """
        Returns the minimum spanning tree of the input graph. See https://en.wikipedia.org/wiki/Minimum_spanning_tree.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        edgeKey : string , optional
            If set, the value of the edgeKey will be used as the weight and the tree will minimize the weight. The value associated with the edgeKey must be numerical. If the key is not set, the edges will be sorted by their length. The default is None
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The minimum spanning tree.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        def vertexInList(vertex, vertexList, tolerance=0.0001):
            for v in vertexList:
                if Vertex.Distance(v, vertex) < tolerance:
                    return True
            return False
        
        if not isinstance(graph, topologic.Graph):
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
    def NavigationGraph(face, viewpointsA=None, viewpointsB=None, tolerance=0.0001, progressBar=True):
        """
        Creates a 2D navigation graph.

        Parameters
        ----------
        face : topologic.Face
            The input boundary. View edges will be clipped to this face. The holes in the face are used as the obstacles
        viewpointsA : list
            The first input list of viewpoints (vertices). Visibility edges will connect these veritces to viewpointsB.
        viewpointsB : list
            The input list of viewpoints (vertices). Visibility edges will connect these vertices to viewpointsA.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        tqdm : bool , optional
            If set to True, a tqdm progress bar is shown. Otherwise, it is not. The default is True.

        Returns
        -------
        topologic.Graph
            The visibility graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Graph import Graph
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(face, topologic.Face):
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
        viewpointsA = [v for v in viewpointsA if isinstance(v, topologic.Vertex)]
        if len(viewpointsA) < 1:
            print("Graph.VisibilityGraph - Error: The input viewpointsA parameter does not contain any vertices. Returning None")
            return None
        viewpointsB = [v for v in viewpointsB if isinstance(v, topologic.Vertex)]
        #if len(viewpointsB) < 1: #Nothing to navigate to, so return a graph made of viewpointsA
            #return Graph.ByVerticesEdges(viewpointsA, [])
        
        # Add obstuse angles of external boundary to viewpoints
        e_boundary = Face.ExternalBoundary(face)
        if isinstance(e_boundary, topologic.Wire):
            vertices = Topology.Vertices(e_boundary)
            interior_angles = Wire.InteriorAngles(e_boundary)
            for i, ang in enumerate(interior_angles):
                if ang > 180:
                    viewpointsA.append(vertices[i])
                    viewpointsB.append(vertices[i])
        i_boundaries = Face.InternalBoundaries(face)
        obstacles = []
        for i_boundary in i_boundaries:
            if isinstance(i_boundary, topologic.Wire):
                obstacles.append(Face.ByWire(i_boundary))
                vertices = Topology.Vertices(i_boundary)
                interior_angles = Wire.InteriorAngles(i_boundary)
                for i, ang in enumerate(interior_angles):
                    if ang < 180:
                        viewpointsA.append(vertices[i])
                        viewpointsB.append(vertices[i])
        if len(obstacles) > 0:
            obstacle_cluster = Cluster.ByTopologies(obstacles)
        else:
            obstacle_cluster = None
        used = []
        for i in range(len(viewpointsA)):
            temp_row = []
            for j in range(len(viewpointsB)):
                temp_row.append(0)
            used.append(temp_row)

        final_edges = []
        if progressBar:
            the_range = tqdm(range(len(viewpointsA)))
        else:
            the_range = range(len(viewpointsA))
        for i in the_range:
            va = viewpointsA[i]
            index_b = Vertex.Index(va, viewpointsB)
            for j in range(len(viewpointsB)):
                vb = viewpointsB[j]
                index_a = Vertex.Index(vb, viewpointsA)
                if used[i][j] == 1 or used[j][i] == 1:
                    continue
                if Vertex.Distance(va, vb) > tolerance:
                    edge = Edge.ByVertices([va,vb])
                    if not obstacle_cluster == None:
                        result = Topology.Difference(edge, obstacle_cluster)
                    else:
                        result = edge
                    if not result == None:
                        result2 = Topology.Difference(result, face)
                        if isinstance(result2, topologic.Edge) or isinstance(result2, topologic.Cluster):
                            result = result2
                    if isinstance(result, topologic.Edge):
                        if abs(Edge.Length(result) - Edge.Length(edge)) < tolerance:
                            sv = Edge.StartVertex(result)
                            ev = Edge.EndVertex(result)
                            if (not Vertex.Index(sv, viewpointsA+viewpointsB) == None) and (not Vertex.Index(ev, viewpointsA+viewpointsB) == None):
                                final_edges.append(result)
                used[i][j] = 1
                if not index_a == None and not index_b == None:
                    used[j][i] = 1
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
        graph : topologic.Graph
            The input graph.
        vertex : topologic.Vertex
            The input vertex.

        Returns
        -------
        topologic.Vertex
            The vertex in the input graph that is the nearest to the input vertex.

        """
        from topologicpy.Vertex import Vertex
        if not isinstance(graph, topologic.Graph):
            print("Graph.NearestVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
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
    def NetworkXGraph(graph, tolerance=0.0001):
        """
        converts the input graph into a NetworkX Graph. See http://networkx.org

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        networkX Graph
            The created networkX Graph

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import random
        import sys
        import subprocess
        if not isinstance(graph, topologic.Graph):
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
                values += [Vertex.X(v), Vertex.Y(v), Vertex.Z(v)]
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
                    nxGraph.add_edge(i,adjIndex, length=(Vertex.Distance(v, adjVertex)))

        pos=nx.spring_layout(nxGraph, k=0.2)
        nx.set_node_attributes(nxGraph, pos, "pos")
        return nxGraph

    @staticmethod
    def Order(graph):
        """
        Returns the graph order of the input graph. The graph order is its number of vertices.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        int
            The number of vertices in the input graph

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Order - Error: The input graph is not a valid graph. Returning None.")
            return None
        return len(Graph.Vertices(graph))

    @staticmethod
    def Path(graph, vertexA, vertexB):
        """
        Returns a path (wire) in the input graph that connects the input vertices.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
            The second input vertex.

        Returns
        -------
        topologic.Wire
            The path (wire) in the input graph that connects the input vertices.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Path - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.Path - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
            print("Graph.Path - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        return graph.Path(vertexA, vertexB)
    
    @staticmethod
    def PyvisGraph(graph, path, overwrite=True, height=900, backgroundColor="white", fontColor="black", notebook=False,
                   vertexSize=6, vertexSizeKey=None, vertexColor="black", vertexColorKey=None, vertexLabelKey=None, vertexGroupKey=None, vertexGroups=None, minVertexGroup=None, maxVertexGroup=None, 
                   edgeLabelKey=None, edgeWeight=0, edgeWeightKey=None, showNeighbours=True, selectMenu=True, filterMenu=True, colorScale="viridis"):
        """
        Displays a pyvis graph. See https://pyvis.readthedocs.io/.

        Parameters
        ----------
        graph : topologic.Graph
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
        vertexColor : int , optional
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
            svi = Vertex.Index(sv, vertices)
            evi = Vertex.Index(ev, vertices)
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
        graph : topologic.Graph
            The input graph.
        edge : topologic.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The input graph with the input edge removed.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.RemoveEdge - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(edge, topologic.Edge):
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
        graph : topologic.Graph
            The input graph.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The input graph with the input vertex removed.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.RemoveVertex - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
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
        graph : topologic.Graph
            The input graph.
        dictionary : topologic.Dictionary or dict
            The input dictionary.

        Returns
        -------
        topologic.Graph
            The input graph with the input dictionary set in it.

        """
        from topologicpy.Dictionary import Dictionary

        if not isinstance(graph, topologic.Graph):
            print("Graph.SetDictionary - Error: the input graph parameter is not a valid graph. Returning None.")
            return None
        if isinstance(dictionary, dict):
            dictionary = Dictionary.ByPythonDictionary(dictionary)
        if not isinstance(dictionary, topologic.Dictionary):
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
        Returns the shortest path that connects the input vertices.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
            The second input vertex.
        vertexKey : string , optional
            The vertex key to minimise. If set the vertices dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value must be numeric. The default is None.
        edgeKey : string , optional
            The edge key to minimise. If set the edges dictionaries will be searched for this key and the associated value will be used to compute the shortest path that minimized the total value. The value of the key must be numeric. If set to "length" (case insensitive), the shortest path by length is computed. The default is "length".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.Wire
            The shortest path between the input vertices.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(graph, topologic.Graph):
            print("Graph.ShortestPath - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.ShortestPath - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
            print("Graph.ShortestPath - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        if edgeKey:
            if edgeKey.lower() == "length":
                edgeKey = "Length"
        try:
            gsv = Graph.NearestVertex(graph, vertexA)
            gev = Graph.NearestVertex(graph, vertexB)
            shortest_path = graph.ShortestPath(gsv, gev, vertexKey, edgeKey)
            sv = Topology.Vertices(shortest_path)[0]
            if Vertex.Distance(sv, gev) < tolerance: # Path is reversed. Correct it.
                if isinstance(shortest_path, topologic.Edge):
                    shortest_path = Edge.Reverse(shortest_path)
                if isinstance(shortest_path, topologic.Wire):
                    shortest_path = Wire.Reverse(shortest_path)
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
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
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
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        
        def isUnique(paths, path):
            if path == None:
                return False
            if len(paths) < 1:
                return True
            for aPath in paths:
                copyPath = topologic.Topology.DeepCopy(aPath)
                dif = copyPath.Difference(path, False)
                if dif == None:
                    return False
            return True
        
        if not isinstance(graph, topologic.Graph):
            print("Graph.ShortestPaths - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.ShortestPaths - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
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
    def Show(graph, vertexColor="black", vertexSize=6, vertexLabelKey=None, vertexGroupKey=None, vertexGroups=[], showVertices=True, showVertexLegend=False, edgeColor="black", edgeWidth=1, edgeLabelKey=None, edgeGroupKey=None, edgeGroups=[], showEdges=True, showEdgeLegend=False, colorScale='viridis', renderer="notebook",
             width=950, height=500, xAxis=False, yAxis=False, zAxis=False, axisSize=1, backgroundColor='rgba(0,0,0,0)', marginLeft=0, marginRight=0, marginTop=20, marginBottom=0,
             camera=[-1.25, -1.25, 1.25], center=[0, 0, 0], up=[0, 0, 1], projection="perspective", tolerance=0.0001):
        """
        Shows the graph using Plotly.

        Parameters
        ----------
        graph : topologic.Graph
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
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. The default is True.
        showEdgeLegend : bool , optional
            If set to True the edge legend will be drawn. Otherwise, it will not be drawn. The default is False.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). The default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        renderer : str , optional
            The desired type of renderer. The default is 'notebook'.
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
            The size of the X,Y,Z, axes. The default is 1.
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
            The desired location of the camera). The default is [-1.25,-1.25,1.25].
        center : list , optional
            The desired center (camera target). The default is [0,0,0].
        up : list , optional
            The desired up vector. The default is [0,0,1].
        projection : str , optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. The default is "perspective"

        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        None

        """
        from topologicpy.Plotly import Plotly

        if not isinstance(graph, topologic.Graph):
            print("Graph.Show - Error: The input graph is not a valid graph. Returning None.")
            return None
        
        data= Plotly.DataByGraph(graph, vertexColor=vertexColor, vertexSize=vertexSize, vertexLabelKey=vertexLabelKey, vertexGroupKey=vertexGroupKey, vertexGroups=vertexGroups, showVertices=showVertices, showVertexLegend=showVertexLegend, edgeColor=edgeColor, edgeWidth=edgeWidth, edgeLabelKey=edgeLabelKey, edgeGroupKey=edgeGroupKey, edgeGroups=edgeGroups, showEdges=showEdges, showEdgeLegend=showEdgeLegend, colorScale=colorScale)
        fig = Plotly.FigureByData(data, width=width, height=height, xAxis=xAxis, yAxis=yAxis, zAxis=zAxis, axisSize=axisSize, backgroundColor=backgroundColor,
                                  marginLeft=marginLeft, marginRight=marginRight, marginTop=marginTop, marginBottom=marginBottom, tolerance=tolerance)
        Plotly.Show(fig, renderer=renderer, camera=camera, center=center, up=up, projection=projection)

    @staticmethod
    def Size(graph):
        """
        Returns the graph size of the input graph. The graph size is its number of edges.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        int
            The number of edges in the input graph.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Size - Error: The input graph is not a valid graph. Returning None.")
            return None
        return len(Graph.Edges(graph))

    @staticmethod
    def TopologicalDistance(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Returns the topological distance between the input vertices. See https://en.wikipedia.org/wiki/Distance_(graph_theory).

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertexA : topologic.Vertex
            The first input vertex.
        vertexB : topologic.Vertex
            The second input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        int
            The topological distance between the input vertices.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.TopologicalDistance - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.TopologicalDistance - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
            print("Graph.TopologicalDistance - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        return graph.TopologicalDistance(vertexA, vertexB, tolerance)
    
    @staticmethod
    def Topology(graph):
        """
        Returns the topology (cluster) of the input graph

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        topologic.Cluster
            The topology of the input graph.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Topology - Error: The input graph is not a valid graph. Returning None.")
            return None
        return graph.Topology()
    
    @staticmethod
    def Tree(graph, vertex=None, tolerance=0.0001):
        """
        Creates a tree graph version of the input graph rooted at the input vertex.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertex : topologic.Vertex , optional
            The input root vertex. If not set, the first vertex in the graph is set as the root vertex. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The tree graph version of the input graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        
        def vertexInList(vertex, vertexList):
            if vertex and vertexList:
                if isinstance(vertex, topologic.Vertex) and isinstance(vertexList, list):
                    for i in range(len(vertexList)):
                        if vertexList[i]:
                            if isinstance(vertexList[i], topologic.Vertex):
                                if topologic.Topology.IsSame(vertex, vertexList[i]):
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
        
        if not isinstance(graph, topologic.Graph):
            print("Graph.Tree - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
            vertex = Graph.Vertices(graph)[0]
        else:
            vertex = Graph.NearestVertex(graph, vertex)
        dictionary = {'vertices':[], 'edges':[]}
        dictionary = buildTree(graph, dictionary, vertex, None, tolerance)
        return Graph.ByVerticesEdges(dictionary['vertices'], dictionary['edges'])
    
    @staticmethod
    def VertexDegree(graph, vertex):
        """
        Returns the degree of the input vertex. See https://en.wikipedia.org/wiki/Degree_(graph_theory).

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.
        vertices : topologic.Vertex
            The input vertex.

        Returns
        -------
        int
            The degree of the input vertex.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.VertexDegree - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
            print("Graph.VertexDegree - Error: The input vertex is not a valid vertex. Returning None.")
            return None
        return graph.VertexDegree(vertex)
    
    @staticmethod
    def Vertices(graph):
        """
        Returns the list of vertices in the input graph.

        Parameters
        ----------
        graph : topologic.Graph
            The input graph.

        Returns
        -------
        list
            The list of vertices in the input graph.

        """
        if not isinstance(graph, topologic.Graph):
            print("Graph.Vertices - Error: The input graph is not a valid graph. Returning None.")
            return None
        vertices = []
        if graph:
            try:
                _ = graph.Vertices(vertices)
            except:
                vertices = []
        return vertices

    @staticmethod
    def VisibilityGraph(face, viewpointsA=None, viewpointsB=None, tolerance=0.0001):
        """
        Creates a 2D visibility graph.

        Parameters
        ----------
        face : topologic.Face
            The input boundary. View edges will be clipped to this face. The holes in the face are used as the obstacles
        viewpointsA : list , optional
            The first input list of viewpoints (vertices). Visibility edges will connect these veritces to viewpointsB. If set to None, this parameters will be set to all vertices of the input face. The default is None.
        viewpointsB : list , optional
            The input list of viewpoints (vertices). Visibility edges will connect these vertices to viewpointsA. If set to None, this parameters will be set to all vertices of the input face. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Graph
            The visibility graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Graph import Graph
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(face, topologic.Face):
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
        viewpointsA = [v for v in viewpointsA if isinstance(v, topologic.Vertex)]
        if len(viewpointsA) < 1:
            print("Graph.VisibilityGraph - Error: The input viewpointsA parameter does not contain any vertices. Returning None")
            return None
        viewpointsB = [v for v in viewpointsB if isinstance(v, topologic.Vertex)]
        if len(viewpointsB) < 1: #Nothing to look at, so return a graph made of viewpointsA
            return Graph.ByVerticesEdges(viewpointsA, [])
        
        i_boundaries = Face.InternalBoundaries(face)
        obstacles = []
        for i_boundary in i_boundaries:
            if isinstance(i_boundary, topologic.Wire):
                obstacles.append(Face.ByWire(i_boundary))
        obstacle_cluster = Cluster.ByTopologies(obstacles)

        def intersects_obstacles(edge, obstacle_cluster, tolerance=0.0001):
            result = Topology.Difference(edge, obstacle_cluster)
            if result == None:
                return True
            if isinstance(result, topologic.Cluster):
                return True
            if isinstance(result, topologic.Edge):
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
