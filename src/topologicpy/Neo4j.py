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
    import neo4j
    from neo4j import GraphDatabase
except:
    print("Neo4j - Installing required neo4j library.")
    try:
        os.system("pip install neo4j")
    except:
        os.system("pip install neo4j --user")
    try:
        import neo4j
        from neo4j import GraphDatabase
    except:
        warnings.warn("Neo4j - Error: Could not import neo4j")

class Neo4j:    
    @staticmethod
    def ExportToGraph(neo4jGraph, cypher=None, xMin=-0.5, yMin=-0.5, zMin=-0.5, xMax=0.5, yMax=0.5, zMax=0.5, tolerance=0.0001, silent=False):
        """
        Exports the input neo4j graph to a topologic graph.

        Parameters
        ----------
        neo4jGraph : neo4j._sync.driver.BoltDriver or neo4jGraph, neo4j._sync.driver.Neo4jDriver
            The input neo4j driver.
        cypher : str, optional
            If set to a non-empty string, a Cypher query will be run on the neo4j graph database to return a sub-graph. Default is None.
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
        silent : bool, optional
            If set to True, no warnings or error messages are displayed. The default is False.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Graph
            The output Topologic graph.
        """
        import random
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary

        vertices = []
        edges = []
        all_vertices = []
        
        with neo4jGraph.session() as session:
            nodes_result = session.run("MATCH (n) RETURN n")
            # Process nodes
            nodes = [record.get('n') for record in nodes_result]
            for node in nodes:
                if node:
                    properties = dict(node.items())
                    x = properties.get('x', random.uniform(xMin, xMax))
                    y = properties.get('y', random.uniform(yMin, yMax))
                    z = properties.get('z', random.uniform(zMin, zMax))
                    vertex = Vertex.ByCoordinates(x, y, z)  # Create Topologic vertex
                    d = Dictionary.ByPythonDictionary(properties)
                    vertex = Topology.SetDictionary(vertex, d)
                    all_vertices.append(vertex)
            
            if cypher:
                # Run the provided Cypher query
                nodes_result = session.run(cypher)
            else:
                # Fetch all nodes and relationships
                nodes_result = session.run("MATCH (n) RETURN n")
                relationships_result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b")
            
            # Process nodes
            nodes = [record.get('n') for record in nodes_result]
            for node in nodes:
                if node:
                    properties = dict(node.items())
                    x = properties.get('x', random.uniform(xMin, xMax))
                    y = properties.get('y', random.uniform(yMin, yMax))
                    z = properties.get('z', random.uniform(zMin, zMax))
                    vertex = Vertex.ByCoordinates(x, y, z)  # Create Topologic vertex
                    d = Dictionary.ByPythonDictionary(properties)
                    vertex = Topology.SetDictionary(vertex, d)
                    vertices.append(vertex)

            # If a Cypher query was provided, process edges
            if cypher:
                relationships_result = session.run(cypher)
            
            # Process relationships
            for record in relationships_result:
                start_node = record.get('a')
                end_node = record.get('b')
                relationship = record.get('r')

                if start_node and end_node:
                    # Find corresponding vertices
                    #start_vertex = next((v for v in vertices if v.id == start_node.id), None)
                    #end_vertex = next((v for v in vertices if v.id == end_node.id), None)
                    start_filter = Topology.Filter(all_vertices, searchType="equal to", key="id", value=start_node['id'])['filtered']
                    if len(start_filter) > 0:
                        start_vertex = start_filter[0]
                    else:
                        start_vertex = NotImplemented
                    end_filter = Topology.Filter(all_vertices, searchType="equal to", key="id", value=end_node['id'])['filtered']
                    if len(end_filter) > 0:
                        end_vertex = end_filter[0]
                    else:
                        end_vertex = None

                    if not start_vertex == None and not end_vertex == None:
                        edge = Edge.ByVertices(start_vertex, end_vertex)
                        relationship_props = dict(relationship.items())
                        d = Dictionary.ByPythonDictionary(relationship_props)
                        edge = Topology.SetDictionary(edge, d)
                        edges.append(edge)
        return Graph.ByVerticesEdges(vertices, edges)

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
        neo4j._sync.driver.BoltDriver or neo4jGraph, neo4j._sync.driver.Neo4jDriver
            The returned neo4j driver.

        """
        return GraphDatabase.driver(url, auth=(username, password))
    
    @staticmethod
    def Reset(neo4jGraph):
        """
        Resets the database completely.

        Parameters
        ----------
        neo4jGraph : neo4j._sync.driver.BoltDriver or neo4jGraph, neo4j._sync.driver.Neo4jDriver
            The input neo4j driver.

        Returns
        -------
        neo4j._sync.driver.BoltDriver or neo4jGraph, neo4j._sync.driver.Neo4jDriver
            The returned neo4j driver.

        """
        with neo4jGraph.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")

            # Drop all indexes
            indexes = session.run("SHOW INDEXES").data()
            for index in indexes:
                index_name = index['name']
                session.run(f"DROP INDEX {index_name}")

            # Drop all constraints
            constraints = session.run("SHOW CONSTRAINTS").data()
            for constraint in constraints:
                constraint_name = constraint['name']
                session.run(f"DROP CONSTRAINT {constraint_name}")
        
        return neo4jGraph

    def ByGraph(neo4jGraph,
                graph,
                vertexLabelKey: str = "label",
                defaultVertexLabel: str = "NODE",
                vertexCategoryKey: str = "category",
                defaultVertexCategory: str = None,
                edgeLabelKey: str = "label",
                defaultEdgeLabel: str = "CONNECTED_TO",
                edgeCategoryKey: str = "category",
                defaultEdgeCategory: str = None,
                bidirectional: bool = True,
                mantissa: int = 6,
                tolerance: float = 0.0001,
                silent: bool = False):
        """
        Converts a Topologic graph to a Neo4j graph.

        Parameters
        ----------
        neo4jGraph : neo4j._sync.driver.BoltDriver or neo4jGraph, neo4j._sync.driver.Neo4jDriver
            The input neo4j driver.
        vertexLabelKey : str , optional
            The returned vertices are labelled according to the dictionary values stored under this key.
            If the vertexLabelKey does not exist, it will be created and the vertices are labelled numerically using the format defaultVertexLabel_XXX. The default is "label".
        defaultVertexLabel : str , optional
            The default vertex label to use if no value is found under the vertexLabelKey. The default is "NODE".
        vertexCategoryKey : str , optional
            The returned vertices are categorized according to the dictionary values stored under this key. The dfefault is "category".
        defaultVertexCategory : str , optional
            The default vertex category to use if no value is found under the vertexCategoryKey. The default is None.
        edgeLabelKey : str , optional
            The returned edges are labelled according to the dictionary values stored under this key.
            If the edgeLabelKey does not exist, it will be created and the edges are labelled numerically using the format defaultEdgeLabel_XXX. The default is "label".
        defaultEdgeLabel : str , optional
            The default edge label to use if no value is found under the edgeLabelKey. The default is "CONNECTED_TO".
        edgeCategoryKey : str , optional
            The returned edges are categorized according to the dictionary values stored under this key. The dfefault is "category".
        defaultEdgeCategory : str , optional
            The default edge category to use if no value is found under the edgeCategoryKey. The default is None.
        bidirectional : bool , optional
            If set to True, the output Neo4j graph is forced to be bidirectional. The defaul is True.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        neo4j._sync.driver.BoltDriver or neo4jGraph, neo4j._sync.driver.Neo4jDriver
            The returned neo4j driver.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        def sanitize_for_neo4j(identifier):
            """
            Replaces illegal characters in Neo4j labels or relationship types with an underscore ('_').
            Ensures the identifier starts with an alphabetic character and contains only valid characters.
            """
            import re
            # Replace any non-alphanumeric characters with underscores
            sanitized = re.sub(r'[^a-zA-Z0-9]', '_', identifier)

            # Ensure the identifier starts with an alphabetic character
            if not sanitized[0].isalpha():
                sanitized = f"_{sanitized}"

            return sanitized
        
        if not isinstance(neo4jGraph, neo4j._sync.driver.BoltDriver) and not isinstance(neo4jGraph, neo4j._sync.driver.Neo4jDriver):
            if not silent:
                print("Neo4j.ByGraph - Error: The input neo4jGraph is not a valid neo4j graph. Returning None.")
            return None
        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Neo4j.ByGraph - Error: The input graph is not a valid topologic graph. Returning None.")
            return None
        # if not isinstance(vertexLabelKey, str):
        #     if not silent:
        #         print("Neo4j.ByGraph - Error: The input vertexLabelKey is not a valid string. Returning None.")
        #     return None
        # if not isinstance(defaultVertexLabel, str):
        #     if not silent:
        #         print("Neo4j.ByGraph - Error: The input defaultVertexLabel is not a valid string. Returning None.")
        #     return None
        # if not isinstance(vertexCategoryKey, str):
        #     if not silent:
        #         print("Neo4j.ByGraph - Error: The input vertexCategoryKey is not a valid string. Returning None.")
        #     return None
        # if not isinstance(edgeLabelKey, str):
        #     if not silent:
        #         print("Neo4j.ByGraph - Error: The input vertexLabelKey is not a valid string. Returning None.")
        #     return None
        # if not isinstance(defaultEdgeLabel, str):
        #     if not silent:
        #         print("Neo4j.ByGraph - Error: The input defaultEdgeLabel is not a valid string. Returning None.")
        #     return None
        vertices = Graph.Vertices(graph)
        edges = Graph.Edges(graph)

        with neo4jGraph.session() as session:
            # Create vertices (nodes in Neo4j)
            n = max(len(str(len(vertices))), 3)
            for i, vertex in enumerate(vertices):
                vertex_props = Dictionary.PythonDictionary(Topology.Dictionary(vertex))  # Get the dictionary of vertex attributes
                
                # Extract label and category, remove them from the properties
                value = defaultVertexLabel+"_"+str(i+1).zfill(n)
                vertex_label = vertex_props.pop(vertexLabelKey, value)
                vertex_label = sanitize_for_neo4j(vertex_label)
                vertex_category = vertex_props.pop(vertexCategoryKey, defaultVertexCategory)  # Extract category if it exists
                # Add coordinates to the vertex properties
                vertex_props.update({
                    'x': Vertex.X(vertex, mantissa=mantissa),  # X coordinate
                    'y': Vertex.Y(vertex, mantissa=mantissa),  # Y coordinate
                    'z': Vertex.Z(vertex, mantissa=mantissa),  # Z coordinate
                })
                
                if not vertex_category == None:
                    vertex_props['category'] = vertex_category  # Add category to properties if it exists
                if not vertex_label == None:
                    vertex_props[vertexLabelKey] = vertex_label  # Add label to properties if it exists
                
                vertex_props['id'] = i

                # Create a node with dynamic label and properties
                session.run(f"""
                    CREATE (n:{vertex_label} $properties)
                """, properties=vertex_props)

            # Create edges (relationships in Neo4j)
            for edge in edges:
                edge_props = Dictionary.PythonDictionary(Topology.Dictionary(edge))  # Get the dictionary of edge attributes
                
                # Extract label and category for the relationship
                edge_label = edge_props.pop(edgeLabelKey, defaultEdgeLabel)  # Default label is 'CONNECTED_TO'
                edge_label = sanitize_for_neo4j(edge_label)
                edge_category = edge_props.pop(edgeCategoryKey, defaultEdgeCategory)  # Extract category if it exists

                start_vertex = Edge.StartVertex(edge)  # Get the starting vertex of the edge
                start_id = Vertex.Index(vertex=start_vertex, vertices=vertices, strict=False, tolerance=tolerance)
                end_vertex = Edge.EndVertex(edge)      # Get the ending vertex of the edge
                end_id = Vertex.Index(vertex=end_vertex, vertices=vertices, strict=False, tolerance=tolerance)

                # Add category to edge properties if it exists
                if not edge_category == None:
                    edge_props['category'] = edge_category
                if not edge_label == None:
                    edge_props[edgeLabelKey] = edge_label  # Add label to properties if it exists
                # Create the relationship with dynamic label and properties
                session.run(f"""
                    MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
                    CREATE (a)-[r:{edge_label} $properties]->(b)
                """, start_id=start_id, end_id=end_id, properties=edge_props)

                # If the graph is bi-directional, add the reverse edge as well
                if bidirectional:
                    session.run(f"""
                        MATCH (a {{id: $end_id}}), (b {{id: $start_id}})
                        CREATE (a)-[r:{edge_label} $properties]->(b)
                    """, start_id=start_id, end_id=end_id, properties=edge_props)
        
        return neo4jGraph


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