import topologic
from py2neo.data import spatial as sp
import time
try:
    import py2neo
except:
    raise Exception("Error: Could not import py2neo.")

class Neo4jGraph:
    @staticmethod
    def listAttributeValues(listAttribute):
        """
        Parameters
        ----------
        listAttribute : TYPE
            DESCRIPTION.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
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
    
    @staticmethod
    def getKeysAndValues(item):
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
        list
            DESCRIPTION.

        """
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
                returnList.append(Neo4jGraph.listAttributeValues(attr))
            else:
                returnList.append("")
        return [keys,returnList]

    @staticmethod
    def vertexIndex(v, vertexList, tolerance):
        for i in range(len(vertexList)):
            d = topologic.VertexUtility.Distance(v, vertexList[i])
            if d < tolerance:
                return i
        return None
    
    @staticmethod
    def Neo4jGraphAddTopologicGraph(neo4jGraph, topologicGraph, categoryKey, tolerance):
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
        gmt = time.gmtime()
        timestamp =  str(gmt.tm_zone)+"_"+str(gmt.tm_year)+"_"+str(gmt.tm_mon)+"_"+str(gmt.tm_wday)+"_"+str(gmt.tm_hour)+"_"+str(gmt.tm_min)+"_"+str(gmt.tm_sec)
        # neo4jGraph, topologicGraph, categoryKey, tolerance = item
        vertices = []
        _ = topologicGraph.Vertices(vertices)
        edges = []
        _ = topologicGraph.Edges(edges)
        notUsed = []
        tx = neo4jGraph.begin()
        nodes = []
        for  i in range(len(vertices)):
            vDict = vertices[i].GetDictionary()
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
            #try:
                #neo4jGraph.cypher.execute("CREATE INDEX FOR (n:%s) on (n.name)" %
                    #n.nodelabel)
            #except:
                #pass
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
    def Neo4jGraphByParameters(url, username, password, run):
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
        # url, username, password, run = item
        if not (run):
            return None
        return py2neo.Graph(url, auth=(username, password))
    
    @staticmethod
    def Neo4jGraphDeleteAll(neo4jGraph):
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
    def Neo4jGraphNodeLabels(neo4jGraph):
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
        return neo4jGraph.schema.node_labels
    
    @staticmethod
    def Neo4jGraphSetGraph(neo4jGraph, topologicGraph, labelKey, relationshipKey, bidirectional, deleteAll, run, tolerance=0.0001):
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
        
        def getValueAtKey(d, searchString):
            keys, values = Neo4jGraph.getKeysAndValues(d)
            for i in range(len(keys)):
                if keys[i].lower() == searchString.lower():
                    return values[i]
            return None
        
        if not (run):
            return None
        import time
        gmt = time.gmtime()
        timestamp =  str(gmt.tm_zone)+"_"+str(gmt.tm_year)+"_"+str(gmt.tm_mon)+"_"+str(gmt.tm_wday)+"_"+str(gmt.tm_hour)+"_"+str(gmt.tm_min)+"_"+str(gmt.tm_sec)

        vertices = []
        _ = topologicGraph.Vertices(vertices)
        edges = []
        _ = topologicGraph.Edges(edges)
        notUsed = []
        tx = neo4jGraph.begin()
        nodes = []
        for  i in range(len(vertices)):
            vDict = vertices[i].GetDictionary()
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
            if (labelKey == 'None') or (not (labelKey)):
                nodeName = "TopologicGraphVertex"
            else:
                nodeName = str(getValueAtKey(vDict, labelKey))
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
            relationshipType = getValueAtKey(ed, relationshipKey)
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