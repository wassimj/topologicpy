import topologic
import time
import random

try:
    import py2neo
    from py2neo import NodeMatcher,RelationshipMatcher
    from py2neo.data import spatial as sp
except:
    raise Exception("Error: Could not import py2neo.")

class Neo4jGraph:

    @staticmethod
    def ExportToGraph(neo4jGraph):
        """
        Description
        -----------
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        neo4jGraph : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

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
            _ = Topology.SetDictionary(vertex, d)
            vertices.append(vertex)
        for node in nodes:
            for relationship_type in relationship_types:
                relationships = list(relationship_matcher.match([node], r_type=relationship_type))
                for relationship in relationships:
                    sv = vertices[nodes.index(relationship.start_node)]
                    ev = vertices[nodes.index(relationship.end_node)]
                    edge = Edge.ByVertices([sv, ev])
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
    def AddTopologicGraph(neo4jGraph, topologicGraph, categoryKey, tolerance):
        """
        Parameters
        ----------
        neo4jGraph : TYPE
            DESCRIPTION.
        topologicGraph : TYPE
            DESCRIPTION.
        categoryKey : TYPE
            DESCRIPTION.
        tolerance : TYPE
            DESCRIPTION.

        Returns
        -------
        neo4jGraph : TYPE
            DESCRIPTION.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        gmt = time.gmtime()
        timestamp =  str(gmt.tm_zone)+"_"+str(gmt.tm_year)+"_"+str(gmt.tm_mon)+"_"+str(gmt.tm_wday)+"_"+str(gmt.tm_hour)+"_"+str(gmt.tm_min)+"_"+str(gmt.tm_sec)
        vertices = Graph.Vertices(topologicGraph)
        edges = Graph.Edges(topologicGraph)
        tx = neo4jGraph.begin()
        nodes = []
        for  i in range(len(vertices)):
            vDict = Topology.GetDictionary(vertices[i])
            keys, values = Neo4jGraph.getKeysAndValues(vDict)
            keys.append("x")
            keys.append("y")
            keys.append("z")
            keys.append("timestamp")
            keys.append("location")
            values.append(vertices[i].X())
            values.append(vertices[i].Y())
            values.append(vertices[i].Z())
            values.append(timestamp)
            values.append(sp.CartesianPoint([vertices[i].X(),vertices[i].Y(),vertices[i].Z()]))
            zip_iterator = zip(keys, values)
            pydict = dict(zip_iterator)
            if categoryKey == 'None':
                nodeName = "TopologicGraphVertex"
            else:
                nodeName = str(values[keys.index(categoryKey)])
            n = py2neo.Node(nodeName, **pydict)
            neo4jGraph.cypher.execute("CREATE INDEX FOR (n:%s) on (n.name)" %
                    n.nodelabel)
            tx.create(n)
            nodes.append(n)
        for i in range(len(edges)):
            e = edges[i]
            sv = e.StartVertex()
            ev = e.EndVertex()
            sn = nodes[Neo4jGraph.vertexIndex(sv, vertices, tolerance)]
            en = nodes[Neo4jGraph.vertexIndex(ev, vertices, tolerance)]
            snen = py2neo.Relationship(sn, "CONNECTEDTO", en)
            tx.create(snen)
            snen = py2neo.Relationship(en, "CONNECTEDTO", sn)
            tx.create(snen)
        neo4jGraph.commit(tx)
        return neo4jGraph

    
    @staticmethod
    def ByParameters(url, username, password, run):
        """
        Parameters
        ----------
        url : TYPE
            DESCRIPTION.
        username : TYPE
            DESCRIPTION.
        password : TYPE
            DESCRIPTION.
        run : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not (run):
            return None
        return py2neo.Graph(url, auth=(username, password))
    
    @staticmethod
    def DeleteAll(neo4jGraph):
        """
        Parameters
        ----------
        neo4jGraph : TYPE
            DESCRIPTION.

        Returns
        -------
        neo4jGraph : TYPE
            DESCRIPTION.

        """
        # neo4jGraph = item
        neo4jGraph.delete_all()
        return neo4jGraph
    
    @staticmethod
    def NodeLabels(neo4jGraph):
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
        return neo4jGraph.schema.node_labels
    
    @staticmethod
    def SetGraph(neo4jGraph, topologicGraph, labelKey, relationshipKey, bidirectional, deleteAll, run, tolerance=0.0001):
        """
        Parameters
        ----------
        neo4jGraph : TYPE
            DESCRIPTION.
        topologicGraph : TYPE
            DESCRIPTION.
        labelKey : TYPE
            DESCRIPTION.
        relationshipKey : TYPE
            DESCRIPTION.
        bidirectional : TYPE
            DESCRIPTION.
        deleteAll : TYPE
            DESCRIPTION.
        run : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        neo4jGraph : TYPE
            DESCRIPTION.

        """
        # neo4jGraph, topologicGraph, labelKey, relationshipKey, bidirectional, deleteAll, tolerance, run = item
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        
        if not (run):
            return None
        import time
        gmt = time.gmtime()
        timestamp =  str(gmt.tm_zone)+"_"+str(gmt.tm_year)+"_"+str(gmt.tm_mon)+"_"+str(gmt.tm_wday)+"_"+str(gmt.tm_hour)+"_"+str(gmt.tm_min)+"_"+str(gmt.tm_sec)

        vertices = Graph.Vertices(topologicGraph)
        edges = Graph.Edges(topologicGraph)
        tx = neo4jGraph.begin()
        nodes = []
        for  i in range(len(vertices)):
            vDict = vertices[i].GetDictionary()
            keys = Dictionary.Keys(vDict)
            values = Dictionary.Values(vDict)
            keys.append("x")
            keys.append("y")
            keys.append("z")
            keys.append("timestamp")
            keys.append("location")
            values.append(vertices[i].X())
            values.append(vertices[i].Y())
            values.append(vertices[i].Z())
            values.append(timestamp)
            values.append(sp.CartesianPoint([vertices[i].X(),vertices[i].Y(),vertices[i].Z()]))
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
            sn = nodes[Neo4jGraph.vertexIndex(sv, vertices, tolerance)]
            en = nodes[Neo4jGraph.vertexIndex(ev, vertices, tolerance)]
            ed = e.GetDictionary()
            relationshipType = Dictionary.ValueAtKey(ed, relationshipKey)
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