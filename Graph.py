import topologicpy
import topologic
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from topologicpy.Aperture import Aperture
from topologicpy.Vertex import Vertex
import random
import time
import sys
import subprocess
try:
    from tqdm.auto import tqdm
except:
    call = [sys.executable, '-m', 'pip', 'install', 'tqdm', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        from tqdm.auto import tqdm
    except:
        print("Graph - Error: Could not import tqdm")

try:
    import numpy as np
except:
    call = [sys.executable, '-m', 'pip', 'install', 'numpy', '-t', sys.path[0]]
    subprocess.run(call)
    try:
        import numpy as np
    except:
        print("Graph - Error: Could not import numpy")

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
        new_edge = Edge.ByVertices([new_vertices[0], new_vertices[1]])
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
    def BetweenessCentrality(graph, vertices=None, sources=None, destinations=None, tolerance=0.001):
        """
            Returns the betweeness centrality measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the betweeness centrality of all the vertices in the input graph is computed. See https://en.wikipedia.org/wiki/Centrality.

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

        try:
            from tqdm.auto import tqdm
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'tqdm', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                from tqdm.auto import tqdm
            except:
                print("Graph.BetweenessCentrality - Error: Could not import tqdm")

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

        #Topology.Show(cluster)
        values = betweeness(vertices, paths, tolerance=tolerance)
        minValue = min(values)
        maxValue = max(values)
        size = maxValue - minValue
        values = [(v-minValue)/size for v in values]
        return values


    @staticmethod
    def ByCSVFile(graphs_file, edges_file, nodes_file,
                  graph_id_header="graph_id", graph_label_header="label", num_nodes_header="num_nodes",
                  src_header="src", dst_header="dst",
                  node_label_header="label", node_X_header="X", node_Y_header="Y", node_Z_header="Z"):
        """
        Returns graphs according to the input CSV files. This method assumes the CSV files follow DGL's schema.

        Parameters
        ----------
        graphs_file : file
            The grpahs CSV file.
        edges_file : file
            The edges CSV file.
        nodes_file : file
            The nodes CSV file.
        graph_id_header : str , optional
            The header string used to specify the graph id. The default is "graph_id".
        graph_label_header : str , optional
            The header string used to specify the graph label. The default is "label".
        num_nodes_header : str , optional
            The header string used to specify the number of nodes. The default is "num_nodes".
        src_header : str , optional
            The header string used to specify the source of edges. The default is "src".
        dst_header : str , optional
            The header string used to specify the destination of edges. The default is "dst".
        node_label_header : str , optional
            The header string used to specify the node label. The default is "label".
        node_X_header : str , optional
            The header string used to specify the node X Coordinate. The default is "X".
        node_Y_header : str , optional
            The header string used to specify the node Y Coordinate. The default is "Y".
        node_Z_header : str , optional
            The header string used to specify the node Y Coordinate. The default is "Z".

        Returns
        -------
        dict
            The dictionary of DGL graphs and labels found in the input CSV files. The keys in the dictionary are "graphs" and "labels"

        """
        if not graphs_file:
            print("Graph.ByCSVFile - Error: The input graphs file is not a valid file. Returning None.")
            return None
        if not edges_file:
            print("Graph.ByCSVFile - Error: The input edges file is not a valid file. Returning None.")
            return None
        if not nodes_file:
            print("Graph.ByCSVFile - Error: The input nodes file is not a valid file. Returning None.")
            return None
        graphs_string = graphs_file.read()
        graphs_file.close()
        edges_string = edges_file.read()
        edges_file.close()
        nodes_string = nodes_file.read()
        nodes_file.close()
        return Graph.ByCSVString(graphs_string, edges_string, nodes_string,
                               graph_id_header=graph_id_header, graph_label_header=graph_label_header, num_nodes_header=num_nodes_header,
                               src_header=src_header, dst_header=dst_header,
                               node_label_header=node_label_header, node_X_header=node_X_header, node_Y_header=node_Y_header, node_Z_header=node_Z_header)
        
    @staticmethod
    def ByCSVPath(graphs_file_path, edges_file_path, nodes_file_path,
                  graph_id_header="graph_id", graph_label_header="label", num_nodes_header="num_nodes",
                  src_header="src", dst_header="dst",
                  node_label_header="label", node_X_header="X", node_Y_header="Y", node_Z_header="Z"):
        """
        Returns graphs according to the input CSV file paths. This method assumes the CSV files follow DGL's schema.

        Parameters
        ----------
        graphs_file_path : str
            The file path to the grpahs CSV file.
        edges_file_path : str
            The file path to the edges CSV file.
        nodes_file_path : str
            The file path to the nodes CSV file.
        graph_id_header : str , optional
            The header string used to specify the graph id. The default is "graph_id".
        graph_label_header : str , optional
            The header string used to specify the graph label. The default is "label".
        num_nodes_header : str , optional
            The header string used to specify the number of nodes. The default is "num_nodes".
        src_header : str , optional
            The header string used to specify the source of edges. The default is "src".
        dst_header : str , optional
            The header string used to specify the destination of edges. The default is "dst".
        node_label_header : str , optional
            The header string used to specify the node label. The default is "label".
        node_X_header : str , optional
            The header string used to specify the node X Coordinate. The default is "X".
        node_Y_header : str , optional
            The header string used to specify the node Y Coordinate. The default is "Y".
        node_Z_header : str , optional
            The header string used to specify the node Y Coordinate. The default is "Z".

        Returns
        -------
        dict
            The dictionary of DGL graphs and labels found in the input CSV files. The keys in the dictionary are "graphs" and "labels"

        """
        if not graphs_file_path:
            print("Graph.ByCSVPath - Error: the input graphs_file_path is not a valid path. Returning None.")
            return None
        if not edges_file_path:
            print("Graph.ByCSVPath - Error: the input edges_file_path is not a valid path. Returning None.")
            return None
        if not nodes_file_path:
            print("Graph.ByCSVPath - Error: the input edges_file_path is not a valid path. Returning None.")
            return None
        try:
            graphs_file = open(graphs_file_path)
        except:
            print("Graph.ByCSVPath - Error: the graphs file is not a valid file. Returning None.")
            return None
        try:
            edges_file = open(edges_file_path)
        except:
            print("Graph.ByCSVPath - Error: the edges file is not a valid file. Returning None.")
            return None
        try:
            nodes_file = open(nodes_file_path)
        except:
            print("Graph.ByCSVPath - Error: the nodes file is not a valid file. Returning None.")
            return None
        return Graph.ByCSVFile(graphs_file, edges_file, nodes_file,
                               graph_id_header=graph_id_header, graph_label_header=graph_label_header, num_nodes_header=num_nodes_header,
                               src_header=src_header, dst_header=dst_header,
                               node_label_header=node_label_header, node_X_header=node_X_header, node_Y_header=node_Y_header, node_Z_header=node_Z_header)
    
    @staticmethod
    def ByCSVString(graphs_string, edges_string, nodes_string,
                    graph_id_header="graph_id", graph_label_header="label", num_nodes_header="num_nodes",
                    src_header="src", dst_header="dst",
                    node_label_header="label", node_X_header="X", node_Y_header="Y", node_Z_header="Z"):
        """
        Returns graphs according to the input CSV strings. This method assumes the CSV strings follow DGL's schema.

        Parameters
        ----------
        graphs_file_path : str
            The file path to the grpahs CSV file.
        edges_file_path : str
            The file path to the edges CSV file.
        nodes_file_path : str
            The file path to the nodes CSV file.
        graph_id_header : str , optional
            The header string used to specify the graph id. The default is "graph_id".
        graph_label_header : str , optional
            The header string used to specify the graph label. The default is "label".
        num_nodes_header : str , optional
            The header string used to specify the number of nodes. The default is "num_nodes".
        src_header : str , optional
            The header string used to specify the source of edges. The default is "src".
        dst_header : str , optional
            The header string used to specify the destination of edges. The default is "dst".
        node_label_header : str , optional
            The header string used to specify the node label. The default is "label".
        node_X_header : str , optional
            The header string used to specify the node X Coordinate. The default is "X".
        node_Y_header : str , optional
            The header string used to specify the node Y Coordinate. The default is "Y".
        node_Z_header : str , optional
            The header string used to specify the node Y Coordinate. The default is "Z".

        Returns
        -------
        dict
            The dictionary of DGL graphs and labels found in the input CSV files. The keys in the dictionary are "graphs" and "labels"

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import pandas as pd
        
        if not graphs_string:
            print("Graph.ByCSVString - Error: the input graphs_string is not a valid string. Returning None.")
            return None
        if not edges_string:
            print("Graph.ByCSVString - Error: the input edges_string is not a valid string. Returning None.")
            return None
        if not nodes_string:
            print("Graph.ByCSVString - Error: the input nodes_string is not a valid string. Returning None.")
            return None
        # Using split by line
        lines = graphs_string.split('\n')[1:-1]
        lines = [l for l in lines if lines != None or lines != ""]
        pd_graphs = pd.DataFrame([row.split(',')[0:3] for row in lines], 
                        columns=[graph_id_header, graph_label_header, num_nodes_header])
        
        lines = edges_string.split('\n')[1:-1]
        lines = [l for l in lines if lines != None or lines != ""]
        edges = pd.DataFrame([row.split(',')[0:3] for row in lines], 
                        columns=[graph_id_header, src_header, dst_header])

        lines = nodes_string.split('\n')[1:-1]
        lines = [l for l in lines if lines[-1] != None or lines[-1] != ""]
        nodes = pd.DataFrame([row.split(',')[0:5] for row in lines], 
                        columns=[graph_id_header, node_label_header, node_X_header, node_Y_header, node_Z_header])

        graphs = []
        labels = []
        graph_ids = []

        # Create a graph for each graph ID from the edges table.
        # First process the graphs table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in pd_graphs.iterrows():
            label_dict[row[graph_id_header]] = row[graph_label_header]
            num_nodes_dict[row[graph_id_header]] = row[num_nodes_header]
        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby(graph_id_header)
        # For the nodes, first group the table by graph IDs.
        nodes_group = nodes.groupby(graph_id_header)
        # For each graph ID...
        for graph_id in tqdm(edges_group.groups, desc="Importing Graphs", leave=False):
            graph_dict = {}
            graph_dict[src_header] = []
            graph_dict[dst_header] = []
            graph_dict[node_label_header] = {}
            graph_dict["node_features"] = []
            num_nodes = num_nodes_dict[graph_id]
            graph_label = label_dict[graph_id]
            if graph_label.isnumeric():
                graph_label = int(graph_label)
            else:
                graph_label = float(graph_label)
            labels.append(graph_label)

            # Find the nodes and their labels and features
            nodes_of_id = nodes_group.get_group(graph_id)
            node_labels = nodes_of_id[node_label_header].values
            node_XCoords = nodes_of_id[node_X_header].values
            node_YCoords = nodes_of_id[node_Y_header].values
            node_ZCoords = nodes_of_id[node_Z_header].values
            vertices = []
            for i in range(len(node_XCoords)):
                v = Vertex.ByCoordinates(float(node_XCoords[i]), float(node_YCoords[i]), float(node_ZCoords[i]))
                d = Dictionary.ByKeysValues([node_label_header], [int(node_labels[i])])
                v = Topology.SetDictionary(v, d)
                vertices.append(v)
            
             # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id[src_header].values
            dst = edges_of_id[dst_header].values
            edges = []
            for i in range(len(src)):
                sv = vertices[int(src[i])]
                ev = vertices[int(dst[i])]
                edges.append(Edge.ByVertices([sv, ev]))
            graphs.append(Graph.ByVerticesEdges(vertices, edges))
            graph_ids.append(int(graph_id))
        graphs.sort(key=dict(zip(graphs, graph_ids)).get)
        labels.sort(key=dict(zip(labels, graph_ids)).get)
        graph_ids.sort()
        return {"graphs":graphs, "labels":labels, "ids":graph_ids}
    
    @staticmethod
    def ByDGCNNFile(file, key="label"):
        """
        Creates a graph from a DGCNN File.

        Parameters
        ----------
        file : file object
            The input file.
        key : str , optional
            The desired key for storing the node label. The default is "label".

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
        return Graph.ByDGCNNString(dgcnn_string, key=key)
    
    @staticmethod
    def ByDGCNNPath(path, key="label"):
        """
        Creates a graph from a DGCNN path.

        Parameters
        ----------
        path : str
            The input file path.
        key : str , optional
            The desired key for storing the node label. The default is "label".

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
        return Graph.ByDGCNNFile(file, key=key)
    
    @staticmethod
    def ByDGCNNString(string, key="label"):
        """
        Creates a graph from a DGCNN string.

        Parameters
        ----------
        string : str
            The input string.
        key : str , optional
            The desired key for storing the node label. The default is "label".

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
                    e = Edge.ByStartVertexEndVertex(sv, ev)
                    edges.append(e)
            index+=n_nodes
            graphs.append(topologic.Graph.ByVerticesEdges(vertices, edges))
        return {'graphs':graphs, 'labels':labels}
    
    @staticmethod
    def ByMeshData(vertices, edges, vertexDictionaries=None, edgeDictionaries=None):
        """
        Creates a graph from the input mesh data

        Parameters
        ----------
        vertices : The list of [x,y,z] coordinates of the vertices/
        edges : the list of [i,j] indices into the vertices list to signify and edge that connects vertices[i] to vertices[j].
        vertexDictionaries : The python dictionaries of the vertices (in the same order as the list of vertices).
        edgeDictionaries : The python dictionaries of the edges (in the same order as the list of edges).

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
                if len(Dictionary.Keys(d)) > 0:
                    g_v = Topology.SetDictionary(g_v, d)
            g_vertices.append(g_v)
            
        g_edges = []
        for i, e in enumerate(edges):
            sv = g_vertices[e[0]]
            ev = g_vertices[e[1]]
            g_e = Edge.ByVertices([sv, ev])
            if not edgeDictionaries == None:
                if isinstance(edgeDictionaries[i], dict):
                    d = Dictionary.ByPythonDictionary(edgeDictionaries[i])
                else:
                    d = edgeDictionaries[i]
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
                keys = Dictionary.Keys(d)
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
                                    v1 = topologic.CellUtility.InternalVertex(cells[i], tolerance)
                                    v2 = topologic.CellUtility.InternalVertex(cells[j], tolerance)
                                else:
                                    v1 = cells[i].CenterOfMass()
                                    v2 = cells[j].CenterOfMass()
                                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                                mDict = mergeDictionaries(sharedt)
                                keys = (Dictionary.Keys(mDict) or [])+["relationship"]
                                values = (Dictionary.Values(mDict) or [])+["Direct"]
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
                                        v1 = topologic.CellUtility.InternalVertex(cells[i], tolerance)
                                        v2 = topologic.CellUtility.InternalVertex(cells[j], tolerance)
                                    else:
                                        v1 = cells[i].CenterOfMass()
                                        v2 = cells[j].CenterOfMass()
                                    e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                                    mDict = mergeDictionaries(apTopList)
                                    if mDict:
                                        e.SetDictionary(mDict)
                                    edges.append(e)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                keys = Dictionary.Keys(d)
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
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    tempe = topologic.Edge.ByStartVertexEndVertex(vcc, vop)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)


            cells = []
            _ = topology.Cells(None, cells)
            if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                for aCell in cells:
                    if useInternalVertex == True:
                        vCell = topologic.CellUtility.InternalVertex(aCell, tolerance)
                    else:
                        vCell = aCell.CenterOfMass()
                    d1 = aCell.GetDictionary()
                    if storeBRep:
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(aCell), Topology.Type(aCell), Topology.TypeAsString(aCell)])
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(sharedAperture), Topology.Type(sharedAperture), Topology.TypeAsString(sharedAperture)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vst = topologic.Vertex.ByCoordinates(vst.X()+(tolerance*100), vst.Y()+(tolerance*100), vst.Z()+(tolerance*100))
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(exteriorAperture), Topology.Type(exteriorAperture), Topology.TypeAsString(exteriorAperture)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)

            for aCell in cells:
                if useInternalVertex == True:
                    vCell = topologic.CellUtility.InternalVertex(aCell, tolerance)
                else:
                    vCell = aCell.CenterOfMass()
                d1 = aCell.GetDictionary()
                if storeBRep:
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(aCell), Topology.Type(aCell), Topology.TypeAsString(aCell)])
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
                vCell = topologic.CellUtility.InternalVertex(Topology.Copy(topology), tolerance)
            else:
                vCell = topology.CenterOfMass()
            d1 = topology.GetDictionary()
            if storeBRep:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                _ = vCell.SetDictionary(d3)
            else:
                _ = vCell.SetDictionary(d1)
            vertices.append(vCell)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                keys = Dictionary.Keys(d)
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
                    tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vop)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vCell, vst)
                            tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
            return [vertices, edges]

        def processShell(item):
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
                                    v1 = topologic.FaceUtility.InternalVertex(topFaces[i], tolerance)
                                    v2 = topologic.FaceUtility.InternalVertex(topFaces[j], tolerance)
                                else:
                                    v1 = topFaces[i].CenterOfMass()
                                    v2 = topFaces[j].CenterOfMass()
                                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
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
                                        v1 = topologic.FaceUtility.InternalVertex(topFaces[i], tolerance)
                                        v2 = topologic.FaceUtility.InternalVertex(topFaces[j], tolerance)
                                    else:
                                        v1 = topFaces[i].CenterOfMass()
                                        v2 = topFaces[j].CenterOfMass()
                                    e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                                    mDict = mergeDictionaries(apTopList)
                                    if mDict:
                                        e.SetDictionary(mDict)
                                    edges.append(e)

            topFaces = []
            _ = topology.Faces(None, topFaces)
            if (viaSharedTopologies == True) or (viaSharedApertures == True) or (toExteriorTopologies == True) or (toExteriorApertures == True) or (toContents == True):
                for aFace in topFaces:
                    if useInternalVertex == True:
                        vFace = topologic.FaceUtility.InternalVertex(aFace, tolerance)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(Aperture.Topology(sharedAperture)), Topology.Type(Aperture.Topology(sharedAperture)), Topology.TypeAsString(Aperture.Topology(sharedAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(aFace), Topology.Type(aFace), Topology.TypeAsString(aFace)])
                    d3 = mergeDictionaries2([d1, d2])
                    _ = vFace.SetDictionary(d3)
                else:
                    _ = vFace.SetDictionary(d1)
                vertices.append(vFace)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                keys = Dictionary.Keys(d)
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
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    tempe = topologic.Edge.ByStartVertexEndVertex(vcc, vop)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Outposts"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            return [vertices, edges]

        def processFace(item):
            topology, others, outpostsKey, idKey, direct, directApertures, viaSharedTopologies, viaSharedApertures, toExteriorTopologies, toExteriorApertures, toContents, toOutposts, useInternalVertex, storeBRep, tolerance = item
            graph = None
            vertices = []
            edges = []

            if useInternalVertex == True:
                vFace = topologic.FaceUtility.InternalVertex(topology, tolerance)
            else:
                vFace = topology.CenterOfMass()
            d1 = topology.GetDictionary()
            if storeBRep:
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                _ = vFace.SetDictionary(d3)
            else:
                _ = vFace.SetDictionary(d1)
            vertices.append(vFace)
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                keys = Dictionary.Keys(d)
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
                    tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vop)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vFace, vst)
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
                                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
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
                                    e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
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
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(anEdge), Topology.Type(anEdge), Topology.TypeAsString(anEdge)])
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(sharedTopology), Topology.Type(sharedTopology), Topology.TypeAsString(sharedTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(Aperture.Topology(sharedAperture)), Topology.Type(Aperture.Topology(sharedAperture)), Topology.TypeAsString(Aperture.Topology(sharedAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
                            tempd = Dictionary.ByKeysValues(["relationship"],["Via Shared Apertures"])
                            _ = tempe.SetDictionary(tempd)
                            edges.append(tempe)
                    if toExteriorTopologies:
                        for exteriorTopology in exteriorTopologies:
                            vst = exteriorTopology
                            vertices.append(exteriorTopology)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(extTop), Topology.Type(extTop), Topology.TypeAsString(extTop)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
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
                    d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(anEdge), Topology.Type(anEdge), Topology.TypeAsString(anEdge)])
                    d3 = mergeDictionaries2([d1, d2])
                    _ = vEdge.SetDictionary(d3)
                else:
                    _ = vEdge.SetDictionary(d1)
                vertices.append(vEdge)
            
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                keys = Dictionary.Keys(d)
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
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vcc.SetDictionary(d3)
                    else:
                        _ = vcc.SetDictionary(d1)
                    vertices.append(vcc)
                    tempe = topologic.Edge.ByStartVertexEndVertex(vcc, vop)
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
                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(topology), Topology.Type(topology), Topology.TypeAsString(topology)])
                d3 = mergeDictionaries2([d1, d2])
                _ = vEdge.SetDictionary(d3)
            else:
                _ = vEdge.SetDictionary(topology.GetDictionary())

            vertices.append(vEdge)

            if toOutposts and others:
                d = Topology.Dictionary(topology)
                keys = Dictionary.Keys(d)
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
                    tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vop)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(exteriorTopology), Topology.Type(exteriorTopology), Topology.TypeAsString(exteriorTopology)])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
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
                                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                                        d3 = mergeDictionaries2([d1, d2])
                                        _ = vst2.SetDictionary(d3)
                                    else:
                                        _ = vst2.SetDictionary(d1)
                                    vertices.append(vst2)
                                    tempe = topologic.Edge.ByStartVertexEndVertex(vst, vst2)
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
                                d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(Aperture.Topology(exteriorAperture)), Topology.Type(Aperture.Topology(exteriorAperture)), Topology.TypeAsString(Aperture.Topology(exteriorAperture))])
                                d3 = mergeDictionaries2([d1, d2])
                                _ = vst.SetDictionary(d3)
                            else:
                                _ = vst.SetDictionary(d1)
                            _ = vst.SetDictionary(exteriorAperture.Topology().GetDictionary())
                            vertices.append(vst)
                            tempe = topologic.Edge.ByStartVertexEndVertex(vEdge, vst)
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
                        d2 = Dictionary.ByKeysValues(["brep", "brepType", "brepTypeString"], [Topology.String(content), Topology.Type(content), Topology.TypeAsString(content)])
                        d3 = mergeDictionaries2([d1, d2])
                        _ = vst.SetDictionary(d3)
                    else:
                        _ = vst.SetDictionary(d1)
                    vertices.append(vst)
                    tempe = topologic.Edge.ByStartVertexEndVertex(topology, vst)
                    tempd = Dictionary.ByKeysValues(["relationship"],["To Contents"])
                    _ = tempe.SetDictionary(tempd)
                    edges.append(tempe)
            
            if toOutposts and others:
                d = Topology.Dictionary(topology)
                keys = Dictionary.Keys(d)
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
                    tempe = topologic.Edge.ByStartVertexEndVertex(topology, vop)
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
            c_cells = Cluster.FreeCells(topology)
            c_shells = Cluster.FreeShells(topology)
            c_faces = Cluster.FreeFaces(topology)
            c_wires = Cluster.FreeWires(topology)
            c_edges = Cluster.FreeEdges(topology)
            c_vertices = Cluster.FreeVertices(topology)
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
                            keys = Dictionary.Keys(v_d)
                            values = Dictionary.Values(v_d)
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
        Return the closeness centrality measure of the input list of vertices within the input graph. The order of the returned list is the same as the order of the input list of vertices. If no vertices are specified, the closeness centrality of all the vertices in the input graph is computed. See https://en.wikipedia.org/wiki/Centrality.

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
        import sys
        import subprocess

        try:
            from tqdm.auto import tqdm
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'tqdm', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                from tqdm.auto import tqdm
            except:
                print("DGL - Error: Could not import tqdm")

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
    def ExportToCSV_GC(graphs, graphLabels, graphsPath, edgesPath, nodesPath, graphIDHeader="graph_id", graphLabelHeader="label",graphNumNodesHeader="num_nodes",edgeSRCHeader="src", edgeDSTHeader="dst", edgeLabelHeader="label", edgeLabelKey="label", defaultEdgeLabel=0, nodeLabelHeader="label", nodeLabelKey="label", defaultNodeLabel=0, overwrite=False):
        """
        Exports the input list of graphs into a set of CSV files compatible with DGL for Graph Classification.

        Parameters
        ----------
        graphs : list
            The input list of graphs.
        graphLabels : list
            The input list of graph labels. This must be list of ints where the minimum must be 0 and the maximum must be n where n is the number of graph categories.
        graphsPaths : str
            The desired path to the output graphs CSV file.
        edgesPath : str
            The desired path to the output edges CSV file.
        nodesPath : str
            The desired path to the output nodes CSV file.
        graphIDHeader : str , optional
            The desired graph ID column header. The default is "graph_id".
        graphLabelHeader : str , optional
            The desired graph label column header. The default is "label".
        graphNumNodesHeader : str , optional
            The desired graph number of nodes column header. The default is "num_nodes".
        edgeSRCHeader : str , optional
            The desired edge source column header. The default is "src".
        edgeDSTHeader : str , optional
            The desired edge destination column header. The default is "dst".
        nodeLabelHeader : str , optional
            The desired node label column header. The default is "label".
        nodeLabelKey : str , optional
            The node label dictionary key saved in each graph vertex. The default is "label".
        defaultNodeLabel : int , optional
            The default node label to use if no node label is found. The default is 0.
        overwrite : bool , optional
            If set to True, any existing files are overwritten. Otherwise, the input list of graphs is appended to the end of each file. The default is False.

        Returns
        -------
        bool
            True if the graphs have been successfully exported. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import random
        import pandas as pd
        from os.path import exists
        
        
        if not isinstance(graphs, list):
            print("Graph.ExportToCSV_GC - Error: The input list of graphs is not a valid list. Returning None.")
            return None
        
        if not isinstance(graphLabels, list):
            print("Graph.ExportToCSV_GC - Error: The input list of labels is not a valid list. Returning None.")
            return None
        
        graphs = [g for g in graphs if isinstance(g, topologic.Graph)]

        if len(graphs) < 1:
            print("Graph.ExportToCSV_GC - Error: The input list of graphs does not contain any valid graphs. Returning None.")
            return None
        if len(graphs) != len(graphLabels):
            print("Graph.ExportToCSV_GC - Error: The input list of graphs and the input list of labels have different lengths. Returning None.")
            return None
        # Make sure the file extension is .csv
        ext = graphsPath[len(graphsPath)-4:len(graphsPath)]
        if ext.lower() != ".csv":
            graphsPath = graphsPath+".csv"
        
        if not overwrite and exists(graphsPath):
            print("DGL.ExportToCSV_GC - Error: a file already exists at the specified graphs path and overwrite is set to False. Returning None.")
            return None
        
        # Make sure the file extension is .csv
        ext = edgesPath[len(edgesPath)-4:len(edgesPath)]
        if ext.lower() != ".csv":
            edgesPath = edgesPath+".csv"
        
        if not overwrite and exists(edgesPath):
            print("DGL.ExportToCSV_GC - Error: a file already exists at the specified edges path and overwrite is set to False. Returning None.")
            return None
        
        # Make sure the file extension is .csv
        ext = nodesPath[len(nodesPath)-4:len(nodesPath)]
        if ext.lower() != ".csv":
            nodesPath = nodesPath+".csv"
        
        if not overwrite and exists(nodesPath):
            print("DGL.ExportToCSV_GC - Error: a file already exists at the specified nodes path and overwrite is set to False. Returning None.")
            return None

        for graph_index, graph in enumerate(graphs):
            graph_label = graphLabels[graph_index]
            # Export Graph Properties
            vertices = Graph.Vertices(graph)
            # Shuffle the vertices
            vertices = random.sample(vertices, len(vertices))
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(graphsPath)
                max_id = max(list(graphs[graphIDHeader]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Helper.Iterate(data)
            data = Helper.Transpose(data)
            df = pd.DataFrame(data, columns= [graphIDHeader, graphLabelHeader, graphNumNodesHeader])
            if overwrite == False:
                df.to_csv(graphsPath, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(graphsPath, mode='w+', index = False, header=True)
                else:
                    df.to_csv(graphsPath, mode='a', index = False, header=False)

            # Export Edge Properties
            edge_src = []
            edge_dst = []
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            node_data = []
            node_columns = [graphIDHeader, nodeLabelHeader, "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = Topology.Dictionary(vertices[0])
            keys = Dictionary.Keys(d)
            for key in keys:
                if key != nodeLabelKey: #We have already saved that in its own column
                    node_columns.append(key)
            for i, v in enumerate(vertices):
                # Might as well get the node labels since we are iterating through the vertices
                d = Topology.Dictionary(v)
                vLabel = Dictionary.ValueAtKey(d, nodeLabelKey)
                if not(vLabel):
                    vLabel = defaultNodeLabel		
                single_node_data = [graph_id, vLabel, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                keys = d.Keys()
                for key in keys:
                    if key != nodeLabelKey and (key in node_columns):
                        value = Dictionary.ValueAtKey(d, key)
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                node_data.append(single_node_data)
                av = Graph.AdjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Vertex.Index(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
            data = [edge_graph_id, edge_src, edge_dst]
            data = Helper.Iterate(data)
            data = Helper.Transpose(data)
            df = pd.DataFrame(data, columns= [graphIDHeader, edgeSRCHeader, edgeDSTHeader])
            if overwrite == False:
                df.to_csv(edgesPath, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(edgesPath, mode='w+', index = False, header=True)
                else:
                    df.to_csv(edgesPath, mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(nodesPath, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(nodesPath, mode='w+', index = False, header=True)
                else:
                    df.to_csv(nodesPath, mode='a', index = False, header=False)
        return True
    
    @staticmethod
    def ExportToCSV_NC(graphs, graphLabels, path, edgeFeaturesKeys=[], nodeFeaturesKeys=[], graphIDHeader="graph_id", graphLabelHeader="label",graphNumNodesHeader="num_nodes", edgeLabelKey="label", defaultEdgeLabel=0, edgeSRCHeader="src_id", edgeDSTHeader="dst_id", edgeLabelHeader="label", edgeFeaturesHeader="feat", nodeLabelKey="label", defaultNodeLabel=0, nodeIDHeader="node_id", nodeLabelHeader="label", nodeFeaturesHeader="feat", trainRatio=0.8, validateRatio=0.1, testRatio=0.1, overwrite=False):
        """
        Exports the input list of graphs into a set of CSV files compatible with DGL for Graph Classification.

        Parameters
        ----------
        graphs : list
            The input list of graphs.
        graphLabels : list
            The input list of graph labels. This must be list of ints where the minimum must be 0 and the maximum must be n where n is the number of graph categories.
        path : str
            The desired path to the output folder where the graphs, edges, and nodes CSV files will be saved.
        edgeFeaturesKeys : list , optional
            The list of features keys saved in the dicitonaries of edges. The default is [].
        nodesFeaturesKeys : list , optional
            The list of features keys saved in the dicitonaries of nodes. The default is [].
        graphIDHeader : str , optional
            The desired graph ID column header. The default is "graph_id".
        graphLabelHeader : str , optional
            The desired graph label column header. The default is "label".
        graphNumNodesHeader : str , optional
            The desired graph number of nodes column header. The default is "num_nodes".
        edgeLabelKey : str , optional
            The edge label dictionary key saved in each graph vertex. The default is "label".
        defaultEdgeLabel : int , optional
            The default nodedge label to use if no edge label is found. The default is 0.
        edgeSRCHeader : str , optional
            The desired edge source column header. The default is "src".
        edgeDSTHeader : str , optional
            The desired edge destination column header. The default is "dst".
        edgeFeaturesHeader : str , optional
            The desired edge features column header. The default is "feat".
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
        trainRatio : float , optional
            The desired ratio of the data to use for training. The number must be between 0 and 1. The default is 0.8 which means 80% of the data will be used for training.
        validateRatio : float , optional
            The desired ratio of the data to use for validation. The number must be between 0 and 1. The default is 0.1 which means 10% of the data will be used for validation.
        testRatio : float , optional
            The desired ratio of the data to use for testing. The number must be between 0 and 1. The default is 0.1 which means 10% of the data will be used for testing.
        overwrite : bool , optional
            If set to True, any existing files are overwritten. Otherwise, the input list of graphs is appended to the end of each file. The default is False.

        Returns
        -------
        bool
            True if the graphs have been successfully exported. False otherwise.
        
        """


        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        import os
        import math
        import random
        import pandas as pd
        from os.path import exists
        
        
        if not isinstance(graphs, list):
            print("Graph.ExportToCSV_GC - Error: The input list of graphs is not a valid list. Returning None.")
            return None
        
        if not isinstance(graphLabels, list):
            print("Graph.ExportToCSV_GC - Error: The input list of labels is not a valid list. Returning None.")
            return None
        
        graphs = [g for g in graphs if isinstance(g, topologic.Graph)]

        if len(graphs) < 1:
            print("Graph.ExportToCSV_GC - Error: The input list of graphs does not contain any valid graphs. Returning None.")
            return None
        if len(graphs) != len(graphLabels):
            print("Graph.ExportToCSV_GC - Error: The input list of graphs and the input list of labels have different lengths. Returning None.")
            return None
        # Make sure the file extension is .csv
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".csv":
            path = path+".csv"
        
        if not overwrite and exists(path):
            print("DGL.ExportToCSV_NC - Error: a file already exists at the specified graphs path and overwrite is set to False. Returning None.")
            return None
        
        if abs(trainRatio  + validateRatio + testRatio - 1) > 0.001:
            print("Graph.ExportToCSV_NC - Error: The train, validate, test ratios do not add up to 1. Returning None")
            return None
        for graph_index, graph in enumerate(graphs):
            graph_label = graphLabels[graph_index]
            # Export Graph Properties
            vertices = Graph.Vertices(graph)
            # Shuffle the vertices
            vertices = random.sample(vertices, len(vertices))
            train_max = math.floor(float(len(vertices))*trainRatio)
            validate_max = math.floor(float(len(vertices))*validateRatio)
            test_max = len(vertices) - train_max - validate_max
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(os.path.join(path,"graphs.csv"))
                max_id = max(list(graphs[graphIDHeader]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Helper.Iterate(data)
            data = Helper.Transpose(data)
            df = pd.DataFrame(data, columns= [graphIDHeader, graphLabelHeader, graphNumNodesHeader])
            if overwrite == False:
                df.to_csv(os.path.join(path, "graphs.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(path, "graphs.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(path, "graphs.csv"), mode='a', index = False, header=False)

            # Export Edge Properties
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            edge_src = []
            edge_dst = []
            edge_lab = []
            edge_feat = []
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = [graphIDHeader, nodeIDHeader, nodeLabelHeader, "train_mask", "val_mask", "test_mask", nodeFeaturesHeader, "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = Topology.Dictionary(vertices[0])
            '''
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            '''
            train = 0
            test = 0
            validate = 0
            
            for i, v in enumerate(vertices):
                if train < train_max:
                    train_mask = True
                    validate_mask = False
                    test_mask = False
                    train = train + 1
                elif validate < validate_max:
                    train_mask = False
                    validate_mask = True
                    test_mask = False
                    validate = validate + 1
                elif test < test_max:
                    train_mask = False
                    validate_mask = False
                    test_mask = True
                    test = test + 1
                else:
                    train_mask = True
                    validate_mask = False
                    test_mask = False
                    train = train + 1
                # Might as well get the node labels since we are iterating through the vertices
                d = Topology.Dictionary(v)
                vLabel = Dictionary.ValueAtKey(d, nodeLabelKey)
                if vLabel == None:
                    vLabel = defaultNodeLabel
                # Might as well get the features since we are iterating through the vertices
                features = ""
                node_features_keys = Helper.Flatten(nodeFeaturesKeys)
                for node_feature_key in node_features_keys:
                    if len(features) > 0:
                        features = features + ","+ str(round(float(Dictionary.ValueAtKey(d, node_feature_key)),5))
                    else:
                        features = str(round(float(Dictionary.ValueAtKey(d, node_feature_key)),5))
                single_node_data = [graph_id, i, vLabel, train_mask, validate_mask, test_mask, features, round(float(Vertex.X(v)),5), round(float(Vertex.Y(v)),5), round(float(Vertex.Z(v)),5)]
                node_data.append(single_node_data)
                av = Graph.AdjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Vertex.Index(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
                    edge = graph.Edge(v, av[k], 0.0001)
                    ed = Topology.Dictionary(edge)
                    edge_label = Dictionary.ValueAtKey(d, edgeLabelKey)
                    if edge_label == None:
                        edge_label = defaultEdgeLabel
                    edge_lab.append(edge_label)
                    edge_features = ""
                    edge_features_keys = Helper.Flatten(edgeFeaturesKeys)
                    for edge_feature_key in edge_features_keys:
                        if len(edge_features) > 0:
                            edge_features = edge_features + ","+ str(round(float(Dictionary.ValueAtKey(ed, edge_feature_key)),5))
                        else:
                            edge_features = str(round(float(Dictionary.ValueAtKey(ed, edge_feature_key)),5))
                    edge_feat.append(edge_features)
            data = [edge_graph_id, edge_src, edge_dst, edge_lab, edge_feat]
            data = Helper.Iterate(data)
            data = Helper.Transpose(data)
            df = pd.DataFrame(data, columns= [graphIDHeader, edgeSRCHeader, edgeDSTHeader, edgeLabelHeader, edgeFeaturesHeader])
            if overwrite == False:
                df.to_csv(os.path.join(path, "edges.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(path, "edges.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(path, "edges.csv"), mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(os.path.join(path, "nodes.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(path, "nodes.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(path, "nodes.csv"), mode='a', index = False, header=False)
        # Write out the meta.yaml file
        yaml_file = open(os.path.join(path,"meta.yaml"), "w")
        yaml_file.write('dataset_name: topologic_dataset\nedge_data:\n- file_name: edges.csv\nnode_data:\n- file_name: nodes.csv\ngraph_data:\n  file_name: graphs.csv')
        yaml_file.close()
        return True
    
    @staticmethod
    def Flatten(graph, layout="spring", k=0.8, seed=None, iterations=50, rootVertex=None):
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
        flat_graph = Graph.ByMeshData(positions, edges, v_dicts, e_dicts)
        return flat_graph

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
    def LongestPath(graph, vertexA, vertexB, vertexKey=None, edgeKey=None, costKey=None, timeLimit=10):
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
        longest_path = Cluster.SelfMerge(Cluster.ByTopologies(new_edges))
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
                keys = Dictionary.Keys(d)
                values = Dictionary.Values(d)
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
                if not value or not isinstance(value, int) or not isinstance(value, float):
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
                    mst = Graph.AddEdge(mst, edge, transferVertexDictionaries=False, transferEdgeDictionaries=True)
        return mst

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
        try:
            import networkx as nx
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'networkx', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                import networkx as nx
            except:
                print("Graph.NetworkXGraph - ERROR: Could not import networkx. Returning None.")
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
    def ShortestPath(graph, vertexA, vertexB, vertexKey="", edgeKey="Length"):
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

        Returns
        -------
        topologic.Wire
            The shortest path between the input vertices.

        """
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
            return graph.ShortestPath(vertexA, vertexB, vertexKey, edgeKey)
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

        Returns
        -------
        topologic.Wire
            The list of shortest paths between the input vertices.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        def nearestVertex(g, v, tolerance):
            vertices = Graph.Vertices(g)
            for aVertex in vertices:
                d = Vertex.Distance(v, aVertex)
                if d < tolerance:
                    return aVertex
            return None
        
        def isUnique(paths, wire):
            if len(paths) < 1:
                return True
            for aPath in paths:
                copyPath = topologic.Topology.DeepCopy(aPath)
                dif = copyPath.Difference(wire, False)
                if dif == None:
                    return False
            return True
        
        if not isinstance(graph, topologic.Graph):
            print("Graph.ShortestPaths - Error: The input graph is not a valid graph. Returning None.")
            return None
        if not isinstance(vertexA, topologic.Vertex):
            print("Graph.ShortestPaths - Error: The input vertexA is not a valid vertex. Returning None.")
            return None
        if not isinstance(vertexB, topologic.Vertex):
            print("Graph.ShortestPaths - Error: The input vertexB is not a valid vertex. Returning None.")
            return None
        shortestPaths = []
        end = time.time() + timeLimit
        while time.time() < end and len(shortestPaths) < pathLimit:
            gsv = nearestVertex(graph, vertexA, tolerance)
            gev = nearestVertex(graph, vertexB, tolerance)
            if (graph != None):
                if edgeKey:
                    if edgeKey.lower() == "length":
                        edgeKey = "Length"
                wire = graph.ShortestPath(gsv,gev,vertexKey,edgeKey) # Find the first shortest path
                wireVertices = []
                flag = False
                try:
                    wireVertices = Wire.Vertices(wire)
                    flag = True
                except:
                    flag = False
                if (flag):
                    if isUnique(shortestPaths, wire):
                        shortestPaths.append(wire)
                vertices = Graph.Vertices(graph)
                random.shuffle(vertices)
                edges = Graph.Edges(graph)
                graph = Graph.ByVerticesEdges(vertices, edges)
        return shortestPaths

    @staticmethod
    def Show(graph, vertexColor="black", vertexSize=6, vertexLabelKey=None, vertexGroupKey=None, vertexGroups=[], showVertices=True, showVertexLegend=False, edgeColor="black", edgeWidth=1, edgeLabelKey=None, edgeGroupKey=None, edgeGroups=[], showEdges=True, showEdgeLegend=False, colorScale='viridis', renderer="notebook",
             width=950, height=500, xAxis=False, yAxis=False, zAxis=False, axisSize=1, backgroundColor='rgba(0,0,0,0)', marginLeft=0, marginRight=0, marginTop=20, marginBottom=0,
             camera=None, target=None, up=None):
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
            The desired location of the camera. The default is [0,0,0].
        target : list , optional
            The desired camera target. The default is [0,0,0].
        up : list , optional
            The desired up vector. The default is [0,0,1].
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
                                  marginLeft=marginLeft, marginRight=marginRight, marginTop=marginTop, marginBottom=marginBottom)
        Plotly.Show(fig, renderer=renderer, camera=camera, target=target, up=up)

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
                    edges.append(Graph.Edge(graph, parent, vertex, tolerance))
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
    def VisibilityGraph(boundary, obstacles=None, viewpointsA=None, viewpointsB=None, tolerance=0.0001):
        """
        Creates a 2D visibility graph.

        Parameters
        ----------
        boundary : topologic.Wire
            The input boundary. View edges will be clipped to this face.
        obstacles : list
            The input list of obstacles (wires).
        viewpointsA : list
            The first input list of viewpoints (vertices). Visibility edges will connect these veritces to viewpointsB.
        viewpointsB : list
            The input list of viewpoints (vertices). Visibility edges will connect these vertices to viewpointsA.
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
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        import sys
        import subprocess
        try:
            from tqdm.auto import tqdm
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'tqdm', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                from tqdm.auto import tqdm
            except:
                print("Graph.VisibilityGraph - Error: Could not import tqdm")
        
        def addEdge(edge, edges, viewpointsA, viewpointsB, tolerance=0.0001):
            # Add edge to edges only if its end points are in vertices
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            con1 = Vertex.Index(sv, viewpointsA, strict=False, tolerance=tolerance)
            con2 = Vertex.Index(ev, viewpointsB, strict=False, tolerance=tolerance)
            if con1 != None and con2 != None:
                edges.append(edge)
            return edges

        if not isinstance(boundary, topologic.Wire):
            print("Graph.VisibilityGraph - Error: The input boundary is not a valid wire. Returning None.")
            return None
        if not obstacles:
            obstacles = []
        obstacles = [x for x in obstacles if isinstance(x, topologic.Wire)]
        if not viewpointsA and not viewpointsB:
            viewpointsA = Wire.Vertices(boundary)
            for obstacle in obstacles:
                obstacleVertices = Wire.Vertices(obstacle)
                viewpointsA += obstacleVertices
        if not isinstance(viewpointsA, list):
            return None
        else:
            viewpointsA = [x for x in viewpointsA if isinstance(x, topologic.Vertex)]
        if not len(viewpointsA) > 0:
            return None
        if not isinstance(viewpointsB, list):
            viewpointsB = []
        else:
            viewpointsB = [x for x in viewpointsB if isinstance(x, topologic.Vertex)]

        boundaryFace = Face.ByWires(boundary, obstacles)
        edges = []
        matrix = []
        if not viewpointsB:
            viewpointsB = viewpointsA
        for i in range(max(len(viewpointsA), len(viewpointsB))):
            tempRow = []
            for j in range(max(len(viewpointsA), len(viewpointsB))):
                tempRow.append(0)
            matrix.append(tempRow)
        try:
            for i in tqdm(range(len(viewpointsA)), desc="Computing Visibility Graph", leave=False):
                for j in range(len(viewpointsB)):
                    if not Topology.IsSame(viewpointsA[i], viewpointsB[j]) and matrix[i][j] == 0:
                        matrix[i][j] = 1
                        matrix[j][i] = 1
                        e = Edge.ByVertices([viewpointsA[i], viewpointsB[j]])
                        if e:
                            e = Topology.Boolean(e, boundaryFace, "intersect", False)
                            if isinstance(e, topologic.Edge):
                                edges = addEdge(e, edges, viewpointsA, viewpointsB, 0.0001)
                            elif isinstance(e, topologic.Cluster):
                                tempEdges = Cluster.Edges(e)
                                if tempEdges:
                                    for tempEdge in tempEdges:
                                        edges = addEdge(tempEdge, edges, viewpointsA, viewpointsB, 0.0001)
        except:
            for i in range(len(viewpointsA)):
                for j in range(len(viewpointsB)):
                    if not Topology.IsSame(viewpointsA[i], viewpointsB[j]) and matrix[i][j] == 0:
                        matrix[i][j] = 1
                        matrix[j][i] = 1
                        e = Edge.ByVertices([viewpointsA[i], viewpointsB[j]])
                        if e:
                            e = Topology.Boolean(e, boundaryFace, "intersect", False)
                            if isinstance(e, topologic.Edge):
                                edges = addEdge(e, edges, viewpointsA, viewpointsB, 0.0001)
                            elif isinstance(e, topologic.Cluster):
                                tempEdges = Cluster.Edges(e)
                                if tempEdges:
                                    for tempEdge in tempEdges:
                                        edges = addEdge(tempEdge, edges, viewpointsA, viewpointsB, 0.0001)
        cluster = Cluster.ByTopologies(viewpointsA+viewpointsB)
        cluster = Cluster.SelfMerge(cluster)
        viewpoints = Cluster.Vertices(cluster)
        return Graph.ByVerticesEdges(viewpoints, edges)
