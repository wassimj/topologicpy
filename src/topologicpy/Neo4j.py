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

import time
import random
import os
import warnings

try:
    import py2neo
    from py2neo import NodeMatcher,RelationshipMatcher
    from py2neo.data import spatial as sp
except:
    print("Neo4j - Installing required py2neo library.")
    try:
        os.system("pip install py2neo")
    except:
        os.system("pip install py2neo --user")
    try:
        import py2neo
        from py2neo import NodeMatcher,RelationshipMatcher
        from py2neo.data import spatial as sp
    except:
        warnings.warn("Neo4j - Error: Could not import py2neo")

class Neo4j:

    @staticmethod
    def NodeToVertex(node):
        """
        Converts the input neo4j node to a topologic vertex.

        Parameters
        ----------
        node : Neo4j.Node
            The input neo4j node.

        Returns
        -------
        topologic_core.Vertex
            The output topologic vertex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        
        if ('x' in node.keys()) and ('y' in node.keys()) and ('z' in node.keys()) or ('X' in node.keys()) and ('Y' in node.keys()) and ('Z' in node.keys()):
            x = node['x']
            y = node['y']
            z = node['z']
            vertex = Vertex.ByCoordinates(x, y, z)
        else:
            x = random.uniform(0, 1000)
            y = random.uniform(0, 1000)
            z = random.uniform(0, 1000)
            vertex = Vertex.ByCoordinates(x, y, z)
        keys = list(node.keys())
        values = list(node.values())
        d = Dictionary.ByKeysValues(keys, values)
        vertex = Topology.SetDictionary(vertex, d)
        return vertex
        

    @staticmethod
    def NodesByCypher(neo4jGraph, cypher):
        dataList = neo4jGraph.run(cypher).data()
        nodes = []
        for data in dataList:
            path = data['p']
            nodes += list(path.nodes)
        return nodes

    @staticmethod
    def NodesBySubGraph(subGraph):
        data = subGraph.data()
        nodes = []
        for data in subGraph:
            path = data['p']
            nodes += list(path.nodes)
        return nodes

    @staticmethod
    def SubGraphByCypher(neo4jGraph, cypher):
        return neo4jGraph.run(cypher).to_subgraph()

    @staticmethod
    def SubGraphExportToGraph(subGraph, tolerance=0.0001):
        """
        Exports the input neo4j graph to a topologic graph.

        Parameters
        ----------
        subGraph : Neo4j.SubGraph
            The input neo4j subgraph.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The output topologic graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Graph import Graph

        def randomVertex(vertices, minDistance):
            flag = True
            while flag:
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                z = random.uniform(0, 1000)
                v = Vertex.ByCoordinates(x, y, z)
                test = False
                if len(vertices) < 1:
                    return v
                for vertex in vertices:
                    d = Vertex.Distance(v, vertex)
                    if d < minDistance:
                        test = True
                        break
                if test == False:
                    return v
                else:
                    continue
        
        nodes = subGraph.nodes
        relationships = list(subGraph.relationships)
        vertices = []
        edges = []
        for node in nodes:
            #Check if they have X, Y, Z coordinates
            if ('x' in node.keys()) and ('y' in node.keys()) and ('z' in node.keys()) or ('X' in node.keys()) and ('Y' in node.keys()) and ('Z' in node.keys()):
                x = node['x']
                y = node['y']
                z = node['z']
                vertex = Vertex.ByCoordinates(x, y, z)
            else:
                vertex = randomVertex(vertices, 1)
            keys = list(node.keys())
            keys.append("identity")
            values = [node.identity]
            for key in keys:
                values.append(node[key])
            d = Dictionary.ByKeysValues(keys, values)
            vertex = Topology.SetDictionary(vertex, d)
            vertices.append(vertex)
        for relationship in relationships:
            keys = list(relationship.keys())
            keys.append("identity")
            values = [relationship.identity]
            for key in keys:
                values.append(node[key])
            sv = vertices[nodes.index(relationship.start_node)]
            ev = vertices[nodes.index(relationship.end_node)]
            edge = Edge.ByVertices([sv, ev], tolerance=tolerance)
            if relationship.start_node['name']:
                sv_name = relationship.start_node['name']
            else:
                sv_name = 'None'
            if relationship.end_node['name']:
                ev_name = relationship.end_node['name']
            else:
                ev_name = 'None'
            d = Dictionary.ByKeysValues(["relationship_type", "from", "to"], [relationship.__class__.__name__, sv_name, ev_name])
            if d:
                _ = Topology.SetDictionary(edge, d)
            edges.append(edge)
        return Graph.ByVerticesEdges(vertices,edges)
    
    @staticmethod
    def ExportToGraph(neo4jGraph, tolerance=0.0001):
        """
        Exports the input neo4j graph to a topologic graph.

        Parameters
        ----------
        neo4jGraph : Neo4j.Graph
            The input neo4j graph.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The output topologic graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Graph import Graph

        def randomVertex(vertices, minDistance):
            flag = True
            while flag:
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                z = random.uniform(0, 1000)
                v = Vertex.ByCoordinates(x, y, z)
                test = False
                if len(vertices) < 1:
                    return v
                for vertex in vertices:
                    d = Vertex.Distance(v, vertex)
                    if d < minDistance:
                        test = True
                        break
                if test == False:
                    return v
                else:
                    continue
        
        node_labels =  neo4jGraph.schema.node_labels
        relationship_types = neo4jGraph.schema.relationship_types
        node_matcher = NodeMatcher(neo4jGraph)
        relationship_matcher = RelationshipMatcher(neo4jGraph)
        vertices = []
        edges = []
        nodes = []
        for node_label in node_labels:
            nodes = nodes + (list(node_matcher.match(node_label)))
        for node in nodes:
            #Check if they have X, Y, Z coordinates
            if ('x' in node.keys()) and ('y' in node.keys()) and ('z' in node.keys()) or ('X' in node.keys()) and ('Y' in node.keys()) and ('Z' in node.keys()):
                x = node['x']
                y = node['y']
                z = node['z']
                vertex = Vertex.ByCoordinates(x, y, z)
            else:
                vertex = randomVertex(vertices, 1)
            keys = list(node.keys())
            values = []
            for key in keys:
                values.append(node[key])
            d = Dictionary.ByKeysValues(keys, values)
            vertex = Topology.SetDictionary(vertex, d)
            vertices.append(vertex)
        for node in nodes:
            for relationship_type in relationship_types:
                relationships = list(relationship_matcher.match([node], r_type=relationship_type))
                for relationship in relationships:
                    sv = vertices[nodes.index(relationship.start_node)]
                    ev = vertices[nodes.index(relationship.end_node)]
                    edge = Edge.ByVertices([sv, ev], tolerance=tolerance)
                    if relationship.start_node['name']:
                        sv_name = relationship.start_node['name']
                    else:
                        sv_name = 'None'
                    if relationship.end_node['name']:
                        ev_name = relationship.end_node['name']
                    else:
                        ev_name = 'None'
                    d = Dictionary.ByKeysValues(["relationship_type", "from", "to"], [relationship_type, sv_name, ev_name])
                    if d:
                        _ = Topology.SetDictionary(edge, d)
                    edges.append(edge)
        return Graph.ByVerticesEdges(vertices,edges)
    
    @staticmethod
    def AddGraph(neo4jGraph, graph, labelKey=None, relationshipKey=None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Adds the input topologic graph to the input neo4j graph

        Parameters
        ----------
        neo4jGraph : Neo4j.Graph
            The input neo4j graph.
        graph : topologic_core.Graph
            The input topologic graph.
        labelKey : str , optional
            The label key in the dictionary under which to look for the label value.
        relationshipKey: str , optional
            The relationship key in the dictionary under which to look for the relationship value.
        mantissa : int, optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        Neo4j.Graph
            The input neo4j graph with the input topologic graph added to it.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary

        gmt = time.gmtime()
        timestamp =  str(gmt.tm_zone)+"_"+str(gmt.tm_year)+"_"+str(gmt.tm_mon)+"_"+str(gmt.tm_wday)+"_"+str(gmt.tm_hour)+"_"+str(gmt.tm_min)+"_"+str(gmt.tm_sec)
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)
        tx = neo4jGraph.begin()
        nodes = []
        for  i in range(len(vertices)):
            vDict = Topology.Dictionary(vertices[i])
            keys = Dictionary.Keyus(vDict)
            values = Dictionary.Values(vDict)
            keys.append("x")
            keys.append("y")
            keys.append("z")
            keys.append("timestamp")
            keys.append("location")
            values.append(Vertex.X(vertices[i], mantissa=mantissa))
            values.append(Vertex.Y(vertices[i], mantissa=mantissa))
            values.append(Vertex.Z(vertices[i], mantissa=mantissa))
            values.append(timestamp)
            values.append(sp.CartesianPoint([Vertex.X(vertices[i], mantissa=mantissa), Vertex.Y(vertices[i], mantissa=mantissa), Vertex.Z(vertices[i], mantissa=mantissa)]))
            zip_iterator = zip(keys, values)
            pydict = dict(zip_iterator)
            if labelKey == 'None':
                nodeName = "TopologicGraphVertex"
            else:
                nodeName = str(values[keys.index(labelKey)])
            n = py2neo.Node(nodeName, **pydict)
            neo4jGraph.cypher.execute("CREATE INDEX FOR (n:%s) on (n.name)" %
                    n.nodelabel)
            tx.create(n)
            nodes.append(n)
        for i in range(len(edges)):
            e = edges[i]
            sv = e.StartVertex()
            ev = e.EndVertex()
            sn = nodes[Vertex.Index(sv, vertices, tolerance=tolerance)]
            en = nodes[Vertex.Index(ev, vertices, tolerance=tolerance)]
            relationshipType = Dictionary.ValueAtKey(e, relationshipKey)
            if not (relationshipType):
                relationshipType = "Connected To"
            snen = py2neo.Relationship(sn, relationshipType, en)
            tx.create(snen)
            snen = py2neo.Relationship(en, relationshipType, sn)
            tx.create(snen)
        neo4jGraph.commit(tx)
        return neo4jGraph

    
    @staticmethod
    def ByParameters(url, username, password):
        """
        Returns a Neo4j graph by the input parameters.

        Parameters
        ----------
        url : str
            The URL of the server.
        username : str
            The username to use for logging in.
        password : str
            The password to use for logging in.

        Returns
        -------
        Neo4j.Graph
            The returned Neo4j graph.

        """
        return py2neo.Graph(url, auth=(username, password))
    
    @staticmethod
    def DeleteAll(neo4jGraph):
        """
        Deletes all entities in the input Neo4j graph.

        Parameters
        ----------
        neo4jGraph : Neo4j Graph
            The input Neo4jGraph.

        Returns
        -------
        Neo4J Graph
            The returned empty graph.

        """
        neo4jGraph.delete_all()
        return neo4jGraph
    
    @staticmethod
    def NodeLabels(neo4jGraph):
        """
        Returns all the node labels used in the input neo4j graph.
        
        Parameters
        ----------
        neo4jGraph : Newo4j.Graph
            The input neo4j graph.

        Returns
        -------
        list
            The list of node labels used in the input neo4j graph.

        """
        return list(neo4jGraph.schema.node_labels)
    
    @staticmethod
    def RelationshipTypes(neo4jGraph):
        """
        Returns all the relationship types used in the input neo4j graph.
        
        Parameters
        ----------
        neo4jGraph : Newo4j.Graph
            The input neo4j graph.

        Returns
        -------
        list
            The list of relationship types used in the input neo4j graph.

        """
        return list(neo4jGraph.schema.relationship_types)
    
    @staticmethod
    def SetGraph(neo4jGraph,
                 graph,
                 labelKey: str = None,
                 relationshipKey: str = None,
                 bidirectional: bool = True,
                 deleteAll: bool = True,
                 mantissa: int = 6,
                 tolerance: float = 0.0001):
        """
        Sets the input topologic graph to the input neo4jGraph.

        Parameters
        ----------
        neo4jGraph : Neo4j.Graph
            The input neo4j graph.
        graph : topologic_core.Graph
            The input topologic graph.
        labelKey : str , optional
            The dictionary key under which to find the vertex's label value. The default is None which means the vertex gets the name 'TopologicGraphVertex'.
        relationshipKey : str , optional
            The dictionary key under which to find the edge's relationship value. The default is None which means the edge gets the relationship type 'Connected To'.
        bidirectional : bool , optional
            If set to True, the edges in the neo4j graph are set to be bi-drectional.
        deleteAll : bool , optional
            If set to True, all previous entities are deleted before adding the new entities.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        neo4jGraph : TYPE
            The input neo4j graph with the input topologic graph added to it.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        
        import time
        gmt = time.gmtime()
        timestamp =  str(gmt.tm_zone)+"_"+str(gmt.tm_year)+"_"+str(gmt.tm_mon)+"_"+str(gmt.tm_wday)+"_"+str(gmt.tm_hour)+"_"+str(gmt.tm_min)+"_"+str(gmt.tm_sec)

        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)
        tx = neo4jGraph.begin()
        nodes = []
        for  i in range(len(vertices)):
            vDict = Topology.Dictionary(vertices[i])
            if not vDict:
                keys = []
                values = []
            else:
                keys = Dictionary.Keys(vDict)
                if not keys:
                    keys = []
                values = Dictionary.Values(vDict)
                if not values:
                    values = []
            keys.append("x")
            keys.append("y")
            keys.append("z")
            keys.append("timestamp")
            keys.append("location")
            values.append(Vertex.X(vertices[i], mantissa=mantissa))
            values.append(Vertex.Y(vertices[i], mantissa=mantissa))
            values.append(Vertex.Z(vertices[i], mantissa=mantissa))
            values.append(timestamp)
            values.append(sp.CartesianPoint([Vertex.X(vertices[i], mantissa=mantissa), Vertex.Y(vertices[i], mantissa=mantissa), Vertex.Z(vertices[i], mantissa=mantissa)]))
            zip_iterator = zip(keys, values)
            pydict = dict(zip_iterator)
            if (labelKey == 'None') or (not (labelKey)):
                nodeName = "TopologicGraphVertex"
            else:
                nodeName = str(Dictionary.ValueAtKey(vDict, labelKey))
            n = py2neo.Node(nodeName, **pydict)
            tx.create(n)
            nodes.append(n)
        for i in range(len(edges)):
            e = edges[i]
            sv = e.StartVertex()
            ev = e.EndVertex()
            sn = nodes[Vertex.Index(vertex=sv, vertices=vertices, strict=False, tolerance=tolerance)]
            en = nodes[Vertex.Index(vertex=ev, vertices=vertices, strict=False, tolerance=tolerance)]
            ed = Topology.Dictionary(e)
            if relationshipKey:
                relationshipType = Dictionary.ValueAtKey(ed, relationshipKey)
            else:
                relationshipType = "Connected To"
            if not (relationshipType):
                relationshipType = "Connected To"
            snen = py2neo.Relationship(sn, relationshipType, en)
            tx.create(snen)
            if bidirectional:
                snen = py2neo.Relationship(en, relationshipType, sn)
                tx.create(snen)
        if deleteAll:
            neo4jGraph.delete_all()
        neo4jGraph.commit(tx)
        return neo4jGraph