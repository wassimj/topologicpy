import topologic
import random
import Dictionary
import Topology
import Process
import pandas as pd
import Replication
import math
import os
import time
from topologic import IntAttribute, DoubleAttribute, StringAttribute, ListAttribute
import pyvisgraph as vg
try:
    from py2neo import NodeMatcher,RelationshipMatcher
    from py2neo.data import spatial as sp
except:
    raise Exception("Error: Could not import py2neo.")

class Graph:
    @staticmethod
    def GraphAddEdge(graph, edges, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        edges : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        new_graph : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # edges = item[1]
        # tolerance = item[2]
        def processKeysValues(keys, values):
            if len(keys) != len(values):
                raise Exception("DictionaryByKeysValues - Keys and Values do not have the same length")
            stl_keys = []
            stl_values = []
            for i in range(len(keys)):
                if isinstance(keys[i], str):
                    stl_keys.append(keys[i])
                else:
                    stl_keys.append(str(keys[i]))
                if isinstance(values[i], list) and len(values[i]) == 1:
                    value = values[i][0]
                else:
                    value = values[i]
                if isinstance(value, bool):
                    if value == False:
                        stl_values.append(topologic.IntAttribute(0))
                    else:
                        stl_values.append(topologic.IntAttribute(1))
                elif isinstance(value, int):
                    stl_values.append(topologic.IntAttribute(value))
                elif isinstance(value, float):
                    stl_values.append(topologic.DoubleAttribute(value))
                elif isinstance(value, str):
                    stl_values.append(topologic.StringAttribute(value))
                elif isinstance(value, list):
                    l = []
                    for v in value:
                        if isinstance(v, bool):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, int):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, float):
                            l.append(topologic.DoubleAttribute(v))
                        elif isinstance(v, str):
                            l.append(topologic.StringAttribute(v))
                    stl_values.append(topologic.ListAttribute(l))
                else:
                    raise Exception("Error: Value type is not supported. Supported types are: Boolean, Integer, Double, String, or List.")
            myDict = topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
            return myDict
        
        def listAttributeValues(listAttribute):
            listAttributes = listAttribute.ListValue()
            returnList = []
            for attr in listAttributes:
                if isinstance(attr, topologic.IntAttribute):
                    returnList.append(attr.IntValue())
                elif isinstance(attr, topologic.DoubleAttribute):
                    returnList.append(attr.DoubleValue())
                elif isinstance(attr, topologic.StringAttribute):
                    returnList.append(attr.StringValue())
            return returnList
        
        def getValueAtKey(item, key):
            try:
                attr = item.ValueAtKey(key)
            except:
                raise Exception("Dictionary.ValueAtKey - Error: Could not retrieve a Value at the specified key ("+key+")")
            if isinstance(attr, topologic.IntAttribute):
                return (attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute):
                return (attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute):
                return (attr.StringValue())
            elif isinstance(attr, topologic.ListAttribute):
                return (listAttributeValues(attr))
            else:
                return None
        
        def getValues(item):
            keys = item.Keys()
            returnList = []
            for key in keys:
                try:
                    attr = item.ValueAtKey(key)
                except:
                    raise Exception("Dictionary.Values - Error: Could not retrieve a Value at the specified key ("+key+")")
                if isinstance(attr, topologic.IntAttribute):
                    returnList.append(attr.IntValue())
                elif isinstance(attr, topologic.DoubleAttribute):
                    returnList.append(attr.DoubleValue())
                elif isinstance(attr, topologic.StringAttribute):
                    returnList.append(attr.StringValue())
                elif isinstance(attr, topologic.ListAttribute):
                    returnList.append(listAttributeValues(attr))
                else:
                    returnList.append("")
            return returnList
        
        def mergeDictionaries(sources):
            sinkKeys = []
            sinkValues = []
            d = sources[0]
            if d != None:
                stlKeys = d.Keys()
                if len(stlKeys) > 0:
                    sinkKeys = d.Keys()
                    sinkValues = getValues(d)
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
                            sourceValue = getValueAtKey(d,sourceKeys[i])
                            if sourceValue != None:
                                if sinkValues[index] != "":
                                    if isinstance(sinkValues[index], list):
                                        sinkValues[index].append(sourceValue)
                                    else:
                                        sinkValues[index] = [sinkValues[index], sourceValue]
                                else:
                                    sinkValues[index] = sourceValue
            if len(sinkKeys) > 0 and len(sinkValues) > 0:
                newDict = processKeysValues(sinkKeys, sinkValues)
                return newDict
            return None
        
        def addIfUnique(graph_vertices, vertex, tolerance):
            unique = True
            returnVertex = vertex
            for gv in graph_vertices:
                if (topologic.VertexUtility.Distance(vertex, gv) < tolerance):
                    gd = gv.GetDictionary()
                    vd = vertex.GetDictionary()
                    gk = gd.Keys()
                    vk = vd.Keys()
                    d = None
                    if (len(gk) > 0) and (len(vk) > 0):
                        d = mergeDictionaries([gd, vd])
                    elif (len(gk) > 0) and (len(vk) < 1):
                        d = gd
                    elif (len(gk) < 1) and (len(vk) > 0):
                        d = vd
                    if d:
                        _ = gv.SetDictionary(d)
                    unique = False
                    returnVertex = gv
                    break
            if unique:
                graph_vertices.append(vertex)
            return [graph_vertices, returnVertex]


        graph_edges = []
        graph_vertices = []
        if graph:
            _ = graph.Vertices(graph_vertices)
            _ = graph.Edges(graph_vertices, tolerance, graph_edges)
        if edges:
            if isinstance(edges, list) == False:
                edges = [edges]
            for edge in edges:
                vertices = []
                _ = edge.Vertices(None, vertices)
                new_vertices = []
                for vertex in vertices:
                    graph_vertices, nv = addIfUnique(graph_vertices, vertex, tolerance)
                    new_vertices.append(nv)
                new_edge = topologic.Edge.ByStartVertexEndVertex(new_vertices[0], new_vertices[1])
                _ = new_edge.SetDictionary(edge.GetDictionary())
                graph_edges.append(new_edge)
        new_graph = topologic.Graph.ByVerticesEdges(graph_vertices, graph_edges)
        return new_graph
    
    @staticmethod
    def GraphAddVertex(graph, vertices, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertices : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        graph : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertices = item[1]
        # tolerance = item[2]
        if isinstance(vertices, list) == False:
            vertices = [vertices]
        _ = graph.AddVertices(vertices, tolerance)
        return graph
    
    @staticmethod
    def GraphAdjacentVertices(graph, vertex):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertex : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertex = item[1]
        vertices = []
        _ = graph.AdjacentVertices(vertex, vertices)
        return list(vertices)
    
    @staticmethod
    def GraphAllPaths(graph, vertexA, vertexB, timeLimit):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertexA : TYPE
            DESCRIPTION.
        vertexB : TYPE
            DESCRIPTION.
        timeLimit : TYPE
            DESCRIPTION.

        Returns
        -------
        paths : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertexA = item[1]
        # vertexB = item[2]
        # timeLimit = item[3]
        paths = []
        _ = graph.AllPaths(vertexA, vertexB, True, timeLimit, paths)
        return paths
    
    @staticmethod
    def GraphByImportedDGCNN(file_path, key):
        """
        Parameters
        ----------
        file_path : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # file_path, key = item
        
        def verticesByCoordinates(x_coords, y_coords):
            vertices = []
            for i in range(len(x_coords)):
                vertices.append(topologic.Vertex.ByCoordinates(x_coords[i], y_coords[i], 0))
            return vertices
        
        graphs = []
        labels = []
        file = open(file_path)
        if file:
            lines = file.readlines()
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
                    node_dict = Dictionary.DictionaryByKeysValues([key], [node_label])
                    Topology.TopologySetDictionary(vertices[j], node_dict)
                for j in range(n_nodes):
                    line = lines[index+j].split()
                    sv = vertices[j]
                    adj_vertices = line[2:]
                    for adj_vertex in adj_vertices:
                        ev = vertices[int(adj_vertex)]
                        e = topologic.Edge.ByStartVertexEndVertex(sv, ev)
                        edges.append(e)
                index+=n_nodes
                graphs.append(topologic.Graph.ByVerticesEdges(vertices, edges))
            file.close()
        return [graphs, labels]
    
    @staticmethod
    def GraphByNeo4jGraph(neo4jGraph):
        """
        Parameters
        ----------
        neo4jGraph : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # neo4jGraph = item
        
        def randomVertex(vertices, minDistance):
            print("Creating a Random Vertex!")
            flag = True
            while flag:
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                z = random.uniform(0, 1000)
                v = topologic.Vertex.ByCoordinates(x, y, z)
                test = False
                if len(vertices) < 1:
                    return v
                for vertex in vertices:
                    d = topologic.VertexUtility.Distance(v, vertex)
                    if d < minDistance:
                        test = True
                        break
                if test == False:
                    return v
                else:
                    continue
        
        def processKeysValues(keys, values):
            if len(keys) != len(values):
                raise Exception("DictionaryByKeysValues - Keys and Values do not have the same length")
            stl_keys = []
            stl_values = []
            for i in range(len(keys)):
                if isinstance(keys[i], str):
                    stl_keys.append(keys[i])
                else:
                    stl_keys.append(str(keys[i]))
                if isinstance(values[i], list) and len(values[i]) == 1:
                    value = values[i][0]
                else:
                    value = values[i]
                if isinstance(value, bool):
                    if value == False:
                        stl_values.append(topologic.IntAttribute(0))
                    else:
                        stl_values.append(topologic.IntAttribute(1))
                elif isinstance(value, int):
                    stl_values.append(topologic.IntAttribute(value))
                elif isinstance(value, float):
                    stl_values.append(topologic.DoubleAttribute(value))
                elif isinstance(value, str):
                    stl_values.append(topologic.StringAttribute(value))
                elif isinstance(value, sp.CartesianPoint):
                    value = list(value)
                    l = []
                    for v in value:
                        if isinstance(v, bool):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, int):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, float):
                            l.append(topologic.DoubleAttribute(v))
                        elif isinstance(v, str):
                            l.append(topologic.StringAttribute(v))
                    stl_values.append(topologic.ListAttribute(l))
                elif isinstance(value, list):
                    l = []
                    for v in value:
                        if isinstance(v, bool):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, int):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, float):
                            l.append(topologic.DoubleAttribute(v))
                        elif isinstance(v, str):
                            l.append(topologic.StringAttribute(v))
                    stl_values.append(topologic.ListAttribute(l))
                else:
                    raise Exception("Error: Value type is not supported. Supported types are: Boolean, Integer, Double, String, or List.")
            myDict = topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
            return myDict
        
        node_labels =  neo4jGraph.schema.node_labels
        relationship_types = neo4jGraph.schema.relationship_types
        node_matcher = NodeMatcher(neo4jGraph)
        relationship_matcher = RelationshipMatcher(neo4jGraph)
        vertices = []
        edges = []
        nodes = []
        for node_label in node_labels:
            nodes = nodes + (list(node_matcher.match(node_label)))
        print(nodes)
        for node in nodes:
            #Check if they have X, Y, Z coordinates
            if ('x' in node.keys()) and ('y' in node.keys()) and ('z' in node.keys()) or ('X' in node.keys()) and ('Y' in node.keys()) and ('Z' in node.keys()):
                x = node['x']
                y = node['y']
                z = node['z']
                vertex = topologic.Vertex.ByCoordinates(x, y, z)
            else:
                vertex = randomVertex(vertices, 1)
            keys = list(node.keys())
            values = []
            for key in keys:
                values.append(node[key])
            d = processKeysValues(keys, values)
            _ = vertex.SetDictionary(d)
            vertices.append(vertex)
        for node in nodes:
            for relationship_type in relationship_types:
                relationships = list(relationship_matcher.match([node], r_type=relationship_type))
                for relationship in relationships:
                    print("    ",relationship.start_node['name'], relationship_type, relationship.end_node['name'])
                    print("Nodes Index:",nodes.index(relationship.start_node))
                    sv = vertices[nodes.index(relationship.start_node)]
                    ev = vertices[nodes.index(relationship.end_node)]
                    edge = topologic.Edge.ByStartVertexEndVertex(sv, ev)
                    if relationship.start_node['name']:
                        sv_name = relationship.start_node['name']
                    else:
                        sv_name = 'None'
                    if relationship.end_node['name']:
                        ev_name = relationship.end_node['name']
                    else:
                        ev_name = 'None'
                    d = processKeysValues(["relationship_type", "from", "to"], [relationship_type, sv_name, ev_name])
                    if d:
                        _ = edge.SetDictionary(d)
                    edges.append(edge)

        return topologic.Graph.ByVerticesEdges(vertices,edges)
    
    @staticmethod
    def GraphByTopology(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        graph : TYPE
            DESCRIPTION.

        """
        topology = item[0]
        graph = None
        if topology:
            classType = topology.Type()
            if classType == 64: #CellComplex
                graph = Process.processCellComplex(item)
            elif classType == 32: #Cell
                graph = Process.processCell(item)
            elif classType == 16: #Shell
                graph = Process.processShell(item)
            elif classType == 8: #Face
                graph = Process.processFace(item)
            elif classType == 4: #Wire
                graph = Process.processWire(item)
            elif classType == 2: #Edge
                graph = Process.processEdge(item)
            elif classType == 1: #Vertex
                graph = Process.processVertex(item)
            elif classType == 128: #Cluster
                raise Exception("ERROR: Graph.ByTopology: Cluster is not supported. Decompose into its sub-topologies first.")
        return graph

    
    @staticmethod
    def GraphByVerticesEdges(vertices, edges):
        """
        Parameters
        ----------
        vertices : TYPE
            DESCRIPTION.
        edges : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # vertices = item[0]
        # edges = item[1]
        if isinstance(vertices, list) == False:
            vertices = [vertices]
        if isinstance(edges, list) == False:
            edges = [edges]
        return topologic.Graph.ByVerticesEdges(vertices, edges)
    
    @staticmethod
    def GraphConnect(graph, verticesA, verticesB, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        verticesA : TYPE
            DESCRIPTION.
        verticesB : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        graph : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # verticesA = item[1]
        # verticesB = item[2]
        # tolerance = item[3]
        if isinstance(verticesA, list) == False:
            verticesA = [verticesA]
        if isinstance(verticesB, list) == False:
            verticesB = [verticesB]
        _ = graph.Connect(verticesA, verticesB, tolerance)
        return graph
    
    @staticmethod
    def GraphContainsEdge(graph, edges, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        edges : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # edges = item[1]
        # tolerance = item[2]
        if isinstance(edges, list) == False:
            edges = [edges]
        returnList = []
        for anEdge in edges:
            returnList.append(graph.ContainsEdge(anEdge, tolerance))
        return returnList
    
    @staticmethod
    def GraphContainsVertex(graph, vertices, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertices : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertices = item[1]
        # tolerance = item[2]
        if isinstance(vertices, list) == False:
            vertices = [vertices]
        returnList = []
        for aVertex in vertices:
            returnList.append(graph.ContainsVertex(aVertex, tolerance))
        return returnList
    
    @staticmethod
    def GraphDegreeSequence(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        sequence : TYPE
            DESCRIPTION.

        """
        sequence = []
        _ = item.DegreeSequence(sequence)
        return sequence
    
    @staticmethod
    def GraphDensity(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return item.Density()
    
    @staticmethod
    def GraphDepthMap(graph, vertexList, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertexList : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        depthMap : TYPE
            DESCRIPTION.

        """
        # print(item)
        # graph = item[0]
        # vertexList = item[1]
        # tolerance = item[2]
        graphVertices = []
        _ = graph.Vertices(graphVertices)
        if len(vertexList) == 0:
            vertexList = graphVertices
        depthMap = []
        for va in vertexList:
            depth = 0
            for vb in graphVertices:
                if topologic.Topology.IsSame(va, vb):
                    dist = 0
                else:
                    dist = graph.TopologicalDistance(va, vb, tolerance)
                depth = depth + dist
            depthMap.append(depth)
        return depthMap
    
    @staticmethod
    def GraphDiameter(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return item.Diameter()
    
    @staticmethod
    def GraphEdge(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertexA : TYPE
            DESCRIPTION.
        vertexB : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertexA = item[1]
        # vertexB = item[2]
        # tolerance = item[3]
        return graph.Edge(vertexA, vertexB, tolerance)
    
    @staticmethod
    def GraphEdges(graph, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        edges : TYPE
            DESCRIPTION.

        """
        vertices = []
        _ = graph.Vertices(vertices)
        edges = []
        _ = graph.Edges(vertices, tolerance, edges)
        return edges
    
    @staticmethod
    def graphVertices(graph):
        vertices = []
        if graph:
            try:
                _ = graph.Vertices(vertices)
            except:
                print("ERROR: (Topologic>Graph.Vertices) operation failed.")
                vertices = None
        if vertices:
            return vertices
        else:
            return []
        
    @staticmethod
    def adjacentVertices(graph, vertex):
        vertices = []
        _ = graph.AdjacentVertices(vertex, vertices)
        return list(vertices)
    
    @staticmethod
    def vertexIndex(vertex, vertices):
        for i in range(len(vertices)):
            if topologic.Topology.IsSame(vertex, vertices[i]):
                return i
        return None

    @staticmethod
    def GraphExportToCSV(graph_list, graph_label_list, graphs_file_path, 
                         edges_file_path, nodes_file_path, graph_id_header,
                         graph_label_header, graph_num_nodes_header,
                         edge_src_header, edge_dst_header, node_label_header,
                         node_label_key, default_node_label, overwrite):
        """
        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
        graphs_file_path : TYPE
            DESCRIPTION.
        edges_file_path : TYPE
            DESCRIPTION.
        nodes_file_path : TYPE
            DESCRIPTION.
        graph_id_header : TYPE
            DESCRIPTION.
        graph_label_header : TYPE
            DESCRIPTION.
        graph_num_nodes_header : TYPE
            DESCRIPTION.
        edge_src_header : TYPE
            DESCRIPTION.
        edge_dst_header : TYPE
            DESCRIPTION.
        node_label_header : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_file_path, \
        # edges_file_path, \
        # nodes_file_path, \
        # graph_id_header, \
        # graph_label_header, \
        # graph_num_nodes_header, \
        # edge_src_header, \
        # edge_dst_header, \
        # node_label_header, \
        # node_label_key, \
        # default_node_label, \
        # overwrite = item

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        for graph_index, graph in enumerate(graph_list):
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = Graph.graphVertices(graph)
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(graphs_file_path)
                max_id = max(list(graphs[graph_id_header]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, graph_label_header, graph_num_nodes_header])
            if overwrite == False:
                df.to_csv(graphs_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(graphs_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(graphs_file_path, mode='a', index = False, header=False)

            # Export Edge Properties
            edge_src = []
            edge_dst = []
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = [graph_id_header, node_label_header, "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            for i, v in enumerate(vertices):
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label        
                single_node_data = [graph_id, vLabel, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = Dictionary.DictionaryValueAtKey(d, key)
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
            data = [edge_graph_id, edge_src, edge_dst]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, edge_src_header, edge_dst_header])
            if overwrite == False:
                df.to_csv(edges_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(edges_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(edges_file_path, mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(nodes_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(nodes_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(nodes_file_path, mode='a', index = False, header=False)
        return True

    
    @staticmethod
    def GraphExportToCSV_NC(graph_list, graph_label_list, graphs_folder_path,
                            node_label_key, node_features_keys, default_node_label, edge_label_key,
                            edge_features_keys, default_edge_label,
                            train_ratio, test_ratio, validate_ratio,
                            overwrite):
        """
        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
        graphs_folder_path : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        node_features_keys : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        edge_label_key : TYPE
            DESCRIPTION.
        edge_features_keys : TYPE
            DESCRIPTION.
        default_edge_label : TYPE
            DESCRIPTION.
        train_ratio : TYPE
            DESCRIPTION.
        test_ratio : TYPE
            DESCRIPTION.
        validate_ratio : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_folder_path, \
        # node_label_key, \
        # node_features_keys, \
        # default_node_label, \
        # edge_label_key, \
        # edge_features_keys, \
        # default_edge_label, \
        # train_ratio, \
        # test_ratio, \
        # validate_ratio, \
        # overwrite = item
        
        def graphVertices(graph):
            import random
            vertices = []
            if graph:
                try:
                    _ = graph.Vertices(vertices)
                except:
                    print("ERROR: (Topologic>Graph.Vertices) operation failed.")
                    vertices = None
            if vertices:
                return random.sample(vertices, len(vertices))
            else:
                return []

        assert (train_ratio+test_ratio+validate_ratio > 0.99), "GraphExportToCSV_NC - Error: Train_Test_Validate ratios do not add up to 1."

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        print("GRAPH LIST", graph_list)
        for graph_index, graph in enumerate(graph_list):
            print("GRAPH INDEX", graph_index)
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = graphVertices(graph)
            train_max = math.floor(float(len(vertices))*train_ratio)
            test_max = math.floor(float(len(vertices))*test_ratio)
            validate_max = len(vertices) - train_max - test_max
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(os.path.join(graphs_folder_path,"graphs.csv"))
                max_id = max(list(graphs["graph_id"]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= ["graph_id", "label", "num_nodes"])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)

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
            node_columns = ["graph_id", "node_id","label", "train_mask","val_mask","test_mask","feat", "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
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
                print("VERTEX I", i)
                if train < train_max:
                    train_mask = True
                    test_mask = False
                    validate_mask = False
                    train = train + 1
                elif test < test_max:
                    train_mask = False
                    test_mask = True
                    validate_mask = False
                    test = test + 1
                elif validate < validate_max:
                    train_mask = False
                    test_mask = False
                    validate_mask = True
                    validate = validate + 1
                else:
                    train_mask = True
                    test_mask = False
                    validate_mask = False
                    train = train + 1
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label
                # Might as well get the features since we are iterating through the vertices
                features = ""
                node_features_keys = Replication.flatten(node_features_keys)
                for node_feature_key in node_features_keys:
                    if len(features) > 0:
                        features = features + ","+ str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                    else:
                        features = str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                single_node_data = [graph_id, i, vLabel, train_mask, validate_mask, test_mask, features, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                '''
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = DictionaryValueAtKey.processItem([d, key])
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                '''
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
                    edge = graph.Edge(v, av[k], 0.0001)
                    ed = edge.GetDictionary()
                    edge_label = Dictionary.DictionaryValueAtKey(d, edge_label_key)
                    if not(edge_label):
                        edge_label = default_edge_label
                    edge_lab.append(edge_label)
                    edge_features = ""
                    edge_features_keys = Replication.flatten(edge_features_keys)
                    for edge_feature_key in edge_features_keys:
                        if len(edge_features) > 0:
                            edge_features = edge_features + ","+ str(round(float(Dictionary.DictionaryValueAtKey(ed, edge_feature_key)),5))
                        else:
                            edge_features = str(round(float(Dictionary.DictionaryValueAtKey(ed, edge_feature_key)),5))
                    edge_feat.append(edge_features)
            print("EDGE_GRAPH_ID",edge_graph_id)
            data = [edge_graph_id, edge_src, edge_dst, edge_lab, edge_feat]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= ["graph_id", "src_id", "dst_id", "label", "feat"])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='a', index = False, header=False)
        # Write out the meta.yaml file
        yaml_file = open(os.path.join(graphs_folder_path,"meta.yaml"), "w")
        yaml_file.write('dataset_name: topologic_dataset\nedge_data:\n- file_name: edges.csv\nnode_data:\n- file_name: nodes.csv\ngraph_data:\n  file_name: graphs.csv')
        yaml_file.close()
        return True
    
    @staticmethod
    def GraphExportToCSVGC(graph_list, graph_label_list, graphs_file_path, edges_file_path,
                           nodes_file_path, graph_id_header, graph_label_header, graph_num_nodes_header, 
                           edge_src_header, edge_dst_header, node_label_header, node_label_key, default_node_label, overwrite):
        """
        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
        graphs_file_path : TYPE
            DESCRIPTION.
        edges_file_path : TYPE
            DESCRIPTION.
        nodes_file_path : TYPE
            DESCRIPTION.
        graph_id_header : TYPE
            DESCRIPTION.
        graph_label_header : TYPE
            DESCRIPTION.
        graph_num_nodes_header : TYPE
            DESCRIPTION.
        edge_src_header : TYPE
            DESCRIPTION.
        edge_dst_header : TYPE
            DESCRIPTION.
        node_label_header : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_file_path, \
        # edges_file_path, \
        # nodes_file_path, \
        # graph_id_header, \
        # graph_label_header, \
        # graph_num_nodes_header, \
        # edge_src_header, \
        # edge_dst_header, \
        # node_label_header, \
        # node_label_key, \
        # default_node_label, \
        # overwrite = item

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        for graph_index, graph in enumerate(graph_list):
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = Graph.graphVertices(graph)
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(graphs_file_path)
                max_id = max(list(graphs[graph_id_header]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, graph_label_header, graph_num_nodes_header])
            if overwrite == False:
                df.to_csv(graphs_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(graphs_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(graphs_file_path, mode='a', index = False, header=False)

            # Export Edge Properties
            edge_src = []
            edge_dst = []
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = [graph_id_header, node_label_header, "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            for i, v in enumerate(vertices):
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label        
                single_node_data = [graph_id, vLabel, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = Dictionary.DictionaryValueAtKey(d, key)
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
            data = [edge_graph_id, edge_src, edge_dst]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, edge_src_header, edge_dst_header])
            if overwrite == False:
                df.to_csv(edges_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(edges_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(edges_file_path, mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(nodes_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(nodes_file_path, mode='w+', index = False, header=True)
                else:
                    df.to_csv(nodes_file_path, mode='a', index = False, header=False)
        return True
    
    @staticmethod
    def GraphExportToCSVNC(graph_list, graph_label_list, graphs_folder_path, graph_id_header,
                           graph_label_header, graph_num_nodes_header, edge_src_header, edge_dst_header,
                           node_label_header, node_label_key, node_features_keys, default_node_label, overwrite):
        """
        Parameters
        ----------
        graph_list : TYPE
            DESCRIPTION.
        graph_label_list : TYPE
            DESCRIPTION.
        graphs_folder_path : TYPE
            DESCRIPTION.
        graph_id_header : TYPE
            DESCRIPTION.
        graph_label_header : TYPE
            DESCRIPTION.
        graph_num_nodes_header : TYPE
            DESCRIPTION.
        edge_src_header : TYPE
            DESCRIPTION.
        edge_dst_header : TYPE
            DESCRIPTION.
        node_label_header : TYPE
            DESCRIPTION.
        node_label_key : TYPE
            DESCRIPTION.
        node_features_keys : TYPE
            DESCRIPTION.
        default_node_label : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph_list, \
        # graph_label_list, \
        # graphs_folder_path, \
        # graph_id_header, \
        # graph_label_header, \
        # graph_num_nodes_header, \
        # edge_src_header, \
        # edge_dst_header, \
        # node_label_header, \
        # node_label_key, \
        # node_features_keys, \
        # default_node_label, \
        # overwrite = item

        if not isinstance(graph_list, list):
            graph_list = [graph_list]
        for graph_index, graph in enumerate(graph_list):
            graph_label = graph_label_list[graph_index]
            # Export Graph Properties
            vertices = Graph.graphVertices(graph)
            graph_num_nodes = len(vertices)
            if overwrite == False:
                graphs = pd.read_csv(graphs_folder_path)
                max_id = max(list(graphs[graph_id_header]))
                graph_id = max_id + graph_index + 1
            else:
                graph_id = graph_index
            data = [[graph_id], [graph_label], [graph_num_nodes]]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, graph_label_header, graph_num_nodes_header])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "graphs.csv"), mode='a', index = False, header=False)

            # Export Edge Properties
            edge_src = []
            edge_dst = []
            edge_graph_id = [] #Repetitive list of graph_id for each edge
            node_graph_id = [] #Repetitive list of graph_id for each vertex/node
            node_labels = []
            x_list = []
            y_list = []
            z_list = []
            node_data = []
            node_columns = [graph_id_header, node_label_header, "feat", "X", "Y", "Z"]
            # All keys should be the same for all vertices, so we can get them from the first vertex
            d = vertices[0].GetDictionary()
            keys = d.Keys()
            for key in keys:
                if key != node_label_key: #We have already saved that in its own column
                    node_columns.append(key)
            for i, v in enumerate(vertices):
                # Might as well get the node labels since we are iterating through the vertices
                d = v.GetDictionary()
                vLabel = Dictionary.DictionaryValueAtKey(d, node_label_key)
                if not(vLabel):
                    vLabel = default_node_label
                # Might as well get the features since we are iterating through the vertices
                features = ""
                for node_feature_key in node_features_keys:
                    if len(features) > 0:
                        features = features + ","+ str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                    else:
                        features = str(round(float(Dictionary.DictionaryValueAtKey(d, node_feature_key)),5))
                single_node_data = [graph_id, vLabel, features, round(float(v.X()),5), round(float(v.Y()),5), round(float(v.Z()),5)]
                keys = d.Keys()
                for key in keys:
                    if key != node_label_key and (key in node_columns):
                        value = Dictionary.DictionaryValueAtKey(d, key)
                        if not value:
                            value = 'None'
                        single_node_data.append(value)
                node_data.append(single_node_data)
                av = Graph.adjacentVertices(graph, v)
                for k in range(len(av)):
                    vi = Graph.vertexIndex(av[k], vertices)
                    edge_graph_id.append(graph_id)
                    edge_src.append(i)
                    edge_dst.append(vi)
            data = [edge_graph_id, edge_src, edge_dst]
            data = Replication.iterate(data)
            data = Replication.transposeList(data)
            df = pd.DataFrame(data, columns= [graph_id_header, edge_src_header, edge_dst_header])
            if overwrite == False:
                df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "edges.csv"), mode='a', index = False, header=False)

            # Export Node Properties
            df = pd.DataFrame(node_data, columns= node_columns)

            if overwrite == False:
                df.to_csv(nodes_file_path, mode='a', index = False, header=False)
            else:
                if graph_index == 0:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='w+', index = False, header=True)
                else:
                    df.to_csv(os.path.join(graphs_folder_path, "nodes.csv"), mode='a', index = False, header=False)
        return True
    
    @staticmethod
    def GraphExportToDGCNN(graph, graph_label, key, default_vertex_label, filepath, overwrite):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        graph_label : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        default_vertex_label : TYPE
            DESCRIPTION.
        filepath : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # graph, graph_label, key, default_vertex_label, filepath, overwrite = item
        vertices = Graph.graphVertices(graph)
        new_lines = []
        new_lines.append("\n"+str(len(vertices))+" "+str(graph_label))
        for j in range(len(vertices)):
            d = vertices[j].GetDictionary()
            vLabel = Dictionary.DictionaryValueAtKey(d, key)
            if not(vLabel):
                vLabel = default_vertex_label
            av = Graph.adjacentVertices(graph, vertices[j])
            line = "\n"+str(vLabel)+" "+ str(len(av))+" "
            for k in range(len(av)):
                vi = Graph.vertexIndex(av[k], vertices)
                line = line+str(vi)+" "
            new_lines.append(line)
        # Make sure the file extension is .txt
        ext = filepath[len(filepath)-4:len(filepath)]
        if ext.lower() != ".txt":
            filepath = filepath+".txt"
        old_lines = ["1"]
        if overwrite == False:
            with open(filepath) as f:
                old_lines = f.readlines()
                if len(old_lines):
                    if old_lines[0] != "":
                        old_lines[0] = str(int(old_lines[0])+1)+"\n"
                else:
                    old_lines[0] = "1"
        lines = old_lines+new_lines
        with open(filepath, "w") as f:
            f.writelines(lines)
        return True
    
    @staticmethod
    def GraphIsComplete(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return item.IsComplete()
    
    @staticmethod
    def GraphIsErdoesGallai(graph, sequence):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        sequence : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # sequence = item[1]
        return graph.IsErdoesGallai(sequence)
    
    @staticmethod
    def GraphIsolatedVertices(graph):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.

        Returns
        -------
        vertices : TYPE
            DESCRIPTION.

        """
        # graph = item
        vertices = []
        _ = graph.IsolatedVertices(vertices)
        return vertices
    
    @staticmethod
    def GraphMaximumDelta(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return item.MaximumDelta()
    
    @staticmethod
    def GraphMinimumDelta(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return item.MinimumDelta()
    
    @staticmethod
    def GraphMST(graph, edgeKey, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        edgeKey : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        finalGraph : TYPE
            DESCRIPTION.

        """
        #This code is contributed by Neelam Yadav 
        # graph = item[0]
        # edgeKey = item[1]
        # tolerance = item[2]
        
        def listAttributeValues(listAttribute):
            listAttributes = listAttribute.ListValue()
            returnList = []
            for attr in listAttributes:
                if isinstance(attr, IntAttribute):
                    returnList.append(attr.IntValue())
                elif isinstance(attr, DoubleAttribute):
                    returnList.append(attr.DoubleValue())
                elif isinstance(attr, StringAttribute):
                    returnList.append(attr.StringValue())
            return returnList
        
        def valueAtKey(item, key):
            try:
                attr = item.ValueAtKey(key)
            except:
                raise Exception("Dictionary.ValueAtKey - Error: Could not retrieve a Value at the specified key ("+key+")")
            if isinstance(attr, IntAttribute):
                return (attr.IntValue())
            elif isinstance(attr, DoubleAttribute):
                return (attr.DoubleValue())
            elif isinstance(attr, StringAttribute):
                return (attr.StringValue())
            elif isinstance(attr, ListAttribute):
                return (listAttributeValues(attr))
            else:
                return None
        
        def vertexIndex(v, vertices, tolerance):
            i = 0
            for aVertex in vertices:
                if topologic.VertexUtility.Distance(v, aVertex) < tolerance:
                    return i
                i = i + 1
            return None
        
        vertices = []
        _ = graph.Vertices(vertices)
        edges = []
        _ = graph.Edges(vertices, tolerance, edges)
        g = Graph(len(vertices))
        for anEdge in edges:
            sv = anEdge.StartVertex()
            svi = vertexIndex(sv, vertices, tolerance)
            ev = anEdge.EndVertex()
            evi = vertexIndex(ev, vertices, tolerance)
            edgeDict = anEdge.GetDictionary()
            weight = 1
            if (edgeDict):
                try:
                    weight = valueAtKey(edgeDict,edgeKey)
                except:
                    weight = 1
            g.addEdge(svi, evi, weight) 

        graphEdges = g.KruskalMST() # Get the Minimum Spanning Tree
        # Create an initial Topologic Graph with one Vertex
        sv = vertices[graphEdges[0][0]]
        finalGraph = topologic.Graph.ByTopology(sv, True, False, False, False, False, False, tolerance)
        stl_keys = []
        stl_keys.append(edgeKey)

        eedges = []
        for i in range(len(graphEdges)):
            sv = vertices[graphEdges[i][0]]
            ev = vertices[graphEdges[i][1]]
            tEdge = topologic.Edge.ByStartVertexEndVertex(sv, ev)
            dictValue = graphEdges[i][2]
            stl_values = []
            stl_values.append(topologic.DoubleAttribute(dictValue))
            edgeDict = topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
            _ = tEdge.SetDictionary(edgeDict)
            eedges.append(tEdge)
        finalGraph.AddEdges(eedges, tolerance)
        return finalGraph
    
    @staticmethod
    def GraphNearestVertex(graph, vertex):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertex : TYPE
            DESCRIPTION.

        Returns
        -------
        nearestVertex : TYPE
            DESCRIPTION.

        """
        # graph = input[0]
        # vertex = input[1]

        vertices = []
        _ = graph.Vertices(vertices)
        nearestVertex = vertices[0]
        nearestDistance = topologic.VertexUtility.Distance(vertex, nearestVertex)
        for aGraphVertex in vertices:
            newDistance = topologic.VertexUtility.Distance(vertex, aGraphVertex)
            if newDistance < nearestDistance:
                nearestDistance = newDistance
                nearestVertex = aGraphVertex
        return nearestVertex

    
    @staticmethod
    def GraphPath(graph, vertexA, vertexB):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertexA : TYPE
            DESCRIPTION.
        vertexB : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertexA = item[1]
        # vertexB = item[2]
        return graph.Path(vertexA, vertexB)
    
    @staticmethod
    def GraphRemoveEdge(graph, edges, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        edges : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        graph : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # edges = item[1]
        # tolerance = item[2]
        if isinstance(edges, list) == False:
            edges = [edges]
        _ = graph.RemoveEdges(edges, tolerance)
        return graph
    
    @staticmethod
    def GraphRemoveVertex(graph, vertices, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertices : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        graph : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertices = item[1]
        # tolerance = item[2]
        
        def nearestVertex(graph, vertex):
            vertices = []
            _ = graph.Vertices(vertices)
            nearestVertex = vertices[0]
            nearestDistance = topologic.VertexUtility.Distance(vertex, nearestVertex)
            for aGraphVertex in vertices:
                newDistance = topologic.VertexUtility.Distance(vertex, aGraphVertex)
                if newDistance < nearestDistance:
                    nearestDistance = newDistance
                    nearestVertex = aGraphVertex
            return nearestVertex
        
        if isinstance(vertices, list) == False:
            vertices = [vertices]
        gVertices = []
        for aVertex in vertices:
            gVertices.append(nearestVertex(graph, aVertex))
        _ = graph.RemoveVertices(gVertices)
        return graph
    
    @staticmethod
    def GraphShortestPath(graph, vertexA, vertexB, vertexKey, edgeKey):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertexA : TYPE
            DESCRIPTION.
        vertexB : TYPE
            DESCRIPTION.
        vertexKey : TYPE
            DESCRIPTION.
        edgeKey : TYPE
            DESCRIPTION.

        Returns
        -------
        topology : TYPE
            DESCRIPTION.

        """
        topology = None
        # graph = item[0]
        # vertexA = item[1]
        # vertexB = item[2]
        # vertexKey = item[3]
        # edgeKey = item[4]
        topology = graph.ShortestPath(vertexA, vertexB, vertexKey, edgeKey)
        return topology
    
    @staticmethod
    def GraphShortestPaths(graph, startVertex, endVertex, vertexKey, edgeKey, timeLimit,
                           pathLimit, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        startVertex : TYPE
            DESCRIPTION.
        endVertex : TYPE
            DESCRIPTION.
        vertexKey : TYPE
            DESCRIPTION.
        edgeKey : TYPE
            DESCRIPTION.
        timeLimit : int
            DESCRIPTION.
        pathLimit : int
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        shortestPaths : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # startVertex = item[1]
        # endVertex = item[2]
        # vertexKey = item[3]
        # edgeKey = item[4]
        # timeLimit = int(item[5])
        # pathLimit = int(item[6])
        # tolerance = item[7]
        
        def nearestVertex(g, v, tolerance):
            vertices = []
            _ = g.Vertices(vertices)
            for aVertex in vertices:
                d = topologic.VertexUtility.Distance(v, aVertex)
                if d < tolerance:
                    return aVertex
            return None
        
        def isUnique(paths, wire):
            if len(paths) < 1:
                print("Length of Paths less than 1 so returning True")
                return True
            for aPath in paths:
                print("Checking Path Uniqueness")
                print("aPath: " + str(aPath))
                print(wire)
                copyPath = topologic.Topology.DeepCopy(aPath)
                dif = copyPath.Difference(wire, False)
                if dif == None:
                    return False
            return True
        
        shortestPaths = []
        start = time.time()
        end = time.time() + timeLimit
        while time.time() < end and len(shortestPaths) < pathLimit:
            gsv = nearestVertex(graph, startVertex, tolerance)
            gev = nearestVertex(graph, endVertex, tolerance)
            if (graph != None):
                wire = graph.ShortestPath(gsv,gev,vertexKey,edgeKey) # Find the first shortest path
                wireVertices = []
                flag = False
                try:
                    _ = wire.Vertices(None, wireVertices)
                    flag = True
                except:
                    flag = False
                if (flag):
                    print("Checking if wire is unique")
                    if isUnique(shortestPaths, wire):
                        shortestPaths.append(wire)
                vertices = []
                _ = graph.Vertices(vertices)
                random.shuffle(vertices)
                edges = []
                _ = graph.Edges(edges)
                graph = topologic.Graph.ByVerticesEdges(vertices, edges)
        return shortestPaths
    
    @staticmethod
    def GraphTopologicalDistance(graph, vertexA, vertexB, tolerance=0.0001):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertexA : TYPE
            DESCRIPTION.
        vertexB : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertexA = item[1]
        # vertexB = item[2]
        # tolerance = item[3]
        return graph.TopologicalDistance(vertexA, vertexB, tolerance)
    
    @staticmethod
    def GraphTopology(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return item.Topology()
    
    @staticmethod
    def GraphTree(graph, rootVertex, xSpacing, ySpacing):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        rootVertex : TYPE
            DESCRIPTION.
        xSpacing : TYPE
            DESCRIPTION.
        ySpacing : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # graph, rootVertex, xSpacing, ySpacing = item
        
        def vertexInList(vertex, vertexList):
            if vertex and vertexList:
                if isinstance(vertex, topologic.Vertex) and isinstance(vertexList, list):
                    for i in range(len(vertexList)):
                        if vertexList[i]:
                            if isinstance(vertexList[i], topologic.Vertex):
                                if topologic.Topology.IsSame(vertex, vertexList[i]):
                                    return True
            return False

        def getChildren(vertex, parent, graph, masterVertexList):
            children = []
            adjVertices = []
            if vertex:
                _ = topologic.Graph.AdjacentVertices(graph, vertex, adjVertices)
            if parent == None:
                return adjVertices
            else:
                for aVertex in adjVertices:
                    if (not vertexInList(aVertex, [parent])) and (not vertexInList(aVertex, masterVertexList)):
                        children.append(aVertex)
            return children
        
        def buildTree(vertex, parent, graph, masterVertexList, masterEdgeList, level):
            if not vertexInList(vertex, masterVertexList):
                masterVertexList.append(vertex)
            if parent == None:
                parent = vertex
            children = getChildren(vertex, parent, graph, masterVertexList)
            width = 1
            for aVertex in children:
                v = buildTree(aVertex, vertex, graph, masterVertexList, masterEdgeList, level+1)
                d = v.GetDictionary()
                w = d.ValueAtKey("TOPOLOGIC_width").IntValue()
                width = width + w
                if v:
                    vertex = vertex.AddContents([v], 0)
            top_d = vertex.GetDictionary()
            top_d = Dictionary.DictionarySetValueAtKey(top_d, "TOPOLOGIC_width", width)
            top_d = Dictionary.DictionarySetValueAtKey(top_d, "TOPOLOGIC_depth", level)
            vertex.SetDictionary(top_d)
            return vertex
        
        def buildGraph(vertex, parent, xSpacing, ySpacing, xStart, vertexMasterList, edgeMasterList):
            d = vertex.GetDictionary()
            width = d.ValueAtKey("TOPOLOGIC_width").IntValue()
            depth = d.ValueAtKey("TOPOLOGIC_depth").IntValue()
            
            xLoc = xStart + 0.5*(width-1)*xSpacing
            yLoc = depth*ySpacing
            newVertex = topologic.Vertex.ByCoordinates(xLoc, yLoc, 0)
            newVertex.SetDictionary(vertex.GetDictionary())
            vertexMasterList.append(newVertex)
            if parent:
                e = topologic.Edge.ByStartVertexEndVertex(parent, newVertex)
                edgeMasterList.append(e)
            children = []
            _ = vertex.Contents(children)
            for aChild in children:
                d = aChild.GetDictionary()
                childWidth = d.ValueAtKey("TOPOLOGIC_width").IntValue()
                vertexMasterList, edgeMasterList = buildGraph(aChild, newVertex, xSpacing, ySpacing, xStart, vertexMasterList, edgeMasterList)
                xStart = xStart + childWidth*xSpacing
            return [vertexMasterList, edgeMasterList]
        
        v = None
        if graph != None and rootVertex != None:
            v = buildTree(rootVertex, None, graph, [], 0)
        else:
            return None
        d = v.GetDictionary()
        width = d.ValueAtKey("TOPOLOGIC_width").IntValue()
        xStart = -width*xSpacing
        vList, eList = buildGraph(v, None, xSpacing, ySpacing, xStart, [], [])
        return topologic.Graph.ByVerticesEdges(vList, eList)
    
    @staticmethod
    def GraphVertexDegree(graph, vertices):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        vertices : TYPE
            DESCRIPTION.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
        # graph = item[0]
        # vertices = item[1]
        if isinstance(vertices, list) == False:
            vertices = [vertices]
        returnList = []
        for aVertex in vertices:
            returnList.append(graph.VertexDegree(aVertex))
        return returnList
    
    @staticmethod
    def GraphVertices(graph):
        """
        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vertices = []
        if graph:
            try:
                _ = graph.Vertices(vertices)
            except:
                print("ERROR: (Topologic>Graph.Vertices) operation failed.")
                vertices = None
        if vertices:
            return vertices
        else:
            return []
    
    @staticmethod
    def GraphVerticesAtKeyValue(vertexList, key, value):
        """
        Parameters
        ----------
        vertexList : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.

        Returns
        -------
        returnVertices : TYPE
            DESCRIPTION.

        """
        # key = item[0]
        # value = item[1]
        
        def listAttributeValues(listAttribute):
            listAttributes = listAttribute.ListValue()
            returnList = []
            for attr in listAttributes:
                if isinstance(attr, IntAttribute):
                    returnList.append(attr.IntValue())
                elif isinstance(attr, DoubleAttribute):
                    returnList.append(attr.DoubleValue())
                elif isinstance(attr, StringAttribute):
                    returnList.append(attr.StringValue())
            return returnList

        def valueAtKey(item, key):
            try:
                attr = item.ValueAtKey(key)
            except:
                raise Exception("Dictionary.ValueAtKey - Error: Could not retrieve a Value at the specified key ("+key+")")
            if isinstance(attr, topologic.IntAttribute):
                return (attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute):
                return (attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute):
                return (attr.StringValue())
            elif isinstance(attr, topologic.ListAttribute):
                return (listAttributeValues(attr))
            else:
                return None

        if isinstance(value, list):
            value.sort()
        returnVertices = []
        for aVertex in vertexList:
            d = aVertex.GetDictionary()
            v = valueAtKey(d, key)
            if isinstance(v, list):
                v.sort()
            if str(v) == str(value):
                returnVertices.append(aVertex)
        return returnVertices
    
    @staticmethod
    def GraphVisibilityGraph(cluster):
        """
        Parameters
        ----------
        cluster : TYPE
            DESCRIPTION.

        Returns
        -------
        graph : TYPE
            DESCRIPTION.

        """
        wires = []
        _ = cluster.Wires(None, wires)
        polys = []
        for aWire in wires:
            vertices = []
            _ = aWire.Vertices(None, vertices)
            poly = []
            for v in vertices:
                p = vg.Point(round(v.X(),4),round(v.Y(),4), 0)
                poly.append(p)
            polys.append(poly)
        g = vg.VisGraph()
        g.build(polys)
        tpEdges = []
        vgEdges = g.visgraph.get_edges()
        for vgEdge in vgEdges:
            sv = topologic.Vertex.ByCoordinates(vgEdge.p1.x, vgEdge.p1.y,0)
            ev = topologic.Vertex.ByCoordinates(vgEdge.p2.x, vgEdge.p2.y,0)
            tpEdges.append(topologic.Edge.ByStartVertexEndVertex(sv, ev))
        tpVertices = []
        vgPoints = g.visgraph.get_points()
        for vgPoint in vgPoints:
            v = topologic.Vertex.ByCoordinates(vgPoint.x, vgPoint.y,0)
            tpVertices.append(v)
        graph = topologic.Graph(tpVertices, tpEdges)
        return graph
    