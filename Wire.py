from binascii import a2b_base64
from re import A
import topologicpy
import topologic
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
import math
import itertools
import numpy as np

class Wire(topologic.Wire):
    @staticmethod
    def BoundingRectangle(topology, optimize=0):
        """
        Returns a wire representing a bounding rectangle of the input topology. The returned wire contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting wire will become axis-aligned.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding rectangle. The default is 0.
        
        Returns
        -------
        topologic.Wire
            The bounding rectangle of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from random import sample

        def br(topology):
            vertices = []
            _ = topology.Vertices(None, vertices)
            x = []
            y = []
            for aVertex in vertices:
                x.append(aVertex.X())
                y.append(aVertex.Y())
            minX = min(x)
            minY = min(y)
            maxX = max(x)
            maxY = max(y)
            return [minX, minY, maxX, maxY]

        if not isinstance(topology, topologic.Topology):
            return None

        world_origin = Vertex.ByCoordinates(0,0,0)

        # Create a sample face
        vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
        v = sample(vertices, 3)
        w = Wire.ByVertices(v)
        f = Face.ByWire(w)
        f = Face.Flatten(f)
        dictionary = Topology.Dictionary(f)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")
        
        topology = Topology.Translate(topology, -xTran, -yTran, -zTran)
        topology = Topology.Rotate(topology, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        topology = Topology.Rotate(topology, origin=world_origin, x=0, y=1, z=0, degree=-theta)
        
        boundingRectangle = br(topology)
        minX = boundingRectangle[0]
        minY = boundingRectangle[1]
        maxX = boundingRectangle[2]
        maxY = boundingRectangle[3]
        w = abs(maxX - minX)
        l = abs(maxY - minY)
        best_area = l*w
        orig_area = best_area
        best_z = 0
        best_br = boundingRectangle
        origin = Topology.Centroid(topology)
        optimize = min(max(optimize, 0), 10)
        if optimize > 0:
            factor = (round(((11 - optimize)/30 + 0.57), 2))
            flag = False
            for n in range(10,0,-1):
                if flag:
                    break
                za = n
                zb = 90+n
                zc = n
                for z in range(za,zb,zc):
                    if flag:
                        break
                    t = Topology.Rotate(topology, origin=origin, x=0,y=0,z=1, degree=z)
                    minX, minY, maxX, maxY = br(t)
                    w = abs(maxX - minX)
                    l = abs(maxY - minY)
                    area = l*w
                    if area < orig_area*factor:
                        best_area = area
                        best_z = z
                        best_br = [minX, minY, maxX, maxY]
                        flag = True
                        break
                    if area < best_area:
                        best_area = area
                        best_z = z
                        best_br = [minX, minY, maxX, maxY]
                        
        else:
            best_br = boundingRectangle

        minX, minY, maxX, maxY = best_br
        vb1 = topologic.Vertex.ByCoordinates(minX, minY, 0)
        vb2 = topologic.Vertex.ByCoordinates(maxX, minY, 0)
        vb3 = topologic.Vertex.ByCoordinates(maxX, maxY, 0)
        vb4 = topologic.Vertex.ByCoordinates(minX, maxY, 0)

        boundingRectangle = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=origin, x=0,y=0,z=1, degree=-best_z)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=world_origin, x=0, y=1, z=0, degree=theta)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=world_origin, x=0, y=0, z=1, degree=phi)
        boundingRectangle = Topology.Translate(boundingRectangle, xTran, yTran, zTran)

        dictionary = Dictionary.ByKeysValues(["zrot"], [best_z])
        boundingRectangle = Topology.SetDictionary(boundingRectangle, dictionary)
        return boundingRectangle

    @staticmethod
    def ByEdges(edges):
        """
        Creates a wire from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.

        Returns
        -------
        topologic.Wire
            The created wire.

        """
        if not isinstance(edges, list):
            return None
        edgeList = [x for x in edges if isinstance(x, topologic.Edge)]
        if len(edgeList) < 1:
            return None
        wire = None
        for anEdge in edgeList:
            if anEdge.Type() == 2:
                if wire == None:
                    wire = topologic.Wire.ByEdges([anEdge])
                else:
                    try:
                        wire = wire.Merge(anEdge)
                    except:
                        continue
        if wire.Type() != 4:
            wire = None
        return wire

    @staticmethod
    def ByEdgesCluster(cluster):
        """
        Creates a wire from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of edges.

        Returns
        -------
        topologic.Wire
            The created wire.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        edges = []
        _ = cluster.Edges(None, edges)
        return Wire.ByEdges(edges)

    @staticmethod
    def ByOffset(wire, offset=1.0, miter=False, miterThreshold=None, offsetKey=None, miterThresholdKey=None, step=True):
        """
        Creates an offset wire from the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.
        offset : float , optional
            The desired offset distance. The default is 1.0.
        miter : bool , optional
            if set to True, the corners will be mitered. The default is False.
        miterThreshold : float , optional
            The distance beyond which a miter should be added. The default is None which means the miter threshold is set to the offset distance multiplied by the square root of 2.
        offsetKey : str , optional
            If specified, the dictionary of the edges will be queried for this key to sepcify the desired offset. The default is None.
        miterThresholdKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to sepcify the desired miter threshold distance. The default is None.
        step : bool , optional
            If set to True, The transition between collinear edges with different offsets will be a step. Otherwise, it will be a continous edge. The default is True.

        Returns
        -------
        topologic.Wire
            The created wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector
        from random import randrange, sample

        if not isinstance(wire, topologic.Wire):
            return None
        if not miterThreshold:
            miterThreshold = offset*math.sqrt(2)
        flatFace = Face.ByWire(wire)
        flatFace = Face.Flatten(flatFace)
        
        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        
        
        edges = Wire.Edges(wire)
        vertices = Wire.Vertices(wire)
        flatEdges = []
        flatVertices = []
        newEdges = []
        for i in range(len(vertices)):
            flatVertex = Topology.Translate(vertices[i], -xTran, -yTran, -zTran)
            flatVertex = Topology.Rotate(flatVertex, origin=world_origin, x=0, y=0, z=1, degree=-phi)
            flatVertex = Topology.Rotate(flatVertex, origin=world_origin, x=0, y=1, z=0, degree=-theta)
            flatVertices.append(flatVertex)
        vertices = flatVertices
        for i in range(len(edges)):
            flatEdge = Topology.Translate(edges[i], -xTran, -yTran, -zTran)
            flatEdge = Topology.Rotate(flatEdge, origin=world_origin, x=0, y=0, z=1, degree=-phi)
            flatEdge = Topology.Rotate(flatEdge, origin=world_origin, x=0, y=1, z=0, degree=-theta)
            flatEdges.append(flatEdge)
            if offsetKey:
                d = Topology.Dictionary(edges[i])
                value = Dictionary.ValueAtKey(d, key=offsetKey)
                c = Topology.Centroid(flatEdge)
                if value:
                    finalOffset = value
                else:
                    finalOffset = offset
            else:
                finalOffset = offset
            e1 = Edge.ByOffset(flatEdge,finalOffset)
            newEdges.append(e1)
        edges = flatEdges
        newVertices = []
        dupVertices = []
        if Wire.IsClosed(wire):
            e1 = newEdges[-1]
            e2 = newEdges[0]
            intV = Edge.Intersect2D(e1,e2)
            if intV:
                newVertices.append(intV)
                dupVertices.append(vertices[0])
            elif step:
                edgeVertices= Edge.Vertices(e1)
                newVertices.append(Vertex.NearestVertex(vertices[-1], Cluster.ByTopologies(edgeVertices), useKDTree=False))
                edgeVertices= Edge.Vertices(e2)
                newVertices.append(Vertex.NearestVertex(vertices[0], Cluster.ByTopologies(edgeVertices), useKDTree=False))
                dupVertices.append(vertices[0])
                dupVertices.append(vertices[0])
            else:
                tempEdge1 = Edge.ByVertices([Edge.StartVertex(e1), Edge.EndVertex(e2)])
                normal = Edge.Normal(e1)
                normal = [normal[0]*finalOffset*10, normal[1]*finalOffset*10, normal[2]*finalOffset*10]
                tempV = Vertex.ByCoordinates(vertices[0].X()+normal[0], vertices[0].Y()+normal[1], vertices[0].Z()+normal[2])
                tempEdge2 = Edge.ByVertices([vertices[0], tempV])
                intV = Edge.Intersect2D(tempEdge1,tempEdge2)
                newVertices.append(intV)
                dupVertices.append(vertices[0])


        else:
            newVertices.append(Edge.StartVertex(newEdges[0]))
        
        for i in range(len(newEdges)-1):
            e1 = newEdges[i]
            e2 = newEdges[i+1]
            intV = Edge.Intersect2D(e1,e2)
            if intV:
                newVertices.append(intV)
                dupVertices.append(vertices[i+1])
            elif step:
                newVertices.append(Edge.EndVertex(e1))
                newVertices.append(Edge.StartVertex(e2))
                dupVertices.append(vertices[i+1])
                dupVertices.append(vertices[i+1])
            else:
                tempEdge1 = Edge.ByVertices([Edge.StartVertex(e1), Edge.EndVertex(e2)])
                normal = Edge.Normal(e1)
                normal = [normal[0]*finalOffset*10, normal[1]*finalOffset*10, normal[2]*finalOffset*10]
                tempV = Vertex.ByCoordinates(vertices[i+1].X()+normal[0], vertices[i+1].Y()+normal[1], vertices[i+1].Z()+normal[2])
                tempEdge2 = Edge.ByVertices([vertices[i+1], tempV])
                intV = Edge.Intersect2D(tempEdge1,tempEdge2)
                newVertices.append(intV)
                dupVertices.append(vertices[i+1])

        vertices = dupVertices
        if not Wire.IsClosed(wire):
            newVertices.append(Edge.EndVertex(newEdges[-1]))
        newWire = Wire.ByVertices(newVertices, close=Wire.IsClosed(wire))
        
        newVertices = Wire.Vertices(newWire)
        newEdges = Wire.Edges(newWire)
        miterEdges = []
        cleanMiterEdges = []
        # Handle miter
        if miter:
            for i in range(len(newVertices)):
                if miterThresholdKey:
                    d = Topology.Dictionary(vertices[i])
                    value = Dictionary.ValueAtKey(d, key=miterThresholdKey)
                    if value:
                        finalMiterThreshold = value
                    else:
                        finalMiterThreshold = miterThreshold
                else:
                    finalMiterThreshold = miterThreshold
                if Vertex.Distance(vertices[i], newVertices[i]) > abs(finalMiterThreshold):
                    st = Topology.SuperTopologies(newVertices[i], newWire, topologyType="edge")
                    if len(st) > 1:
                        e1 = st[0]
                        e2 = st[1]
                        if not Edge.IsCollinear(e1, e2):
                            e1 = Edge.Reverse(e1)
                            bisector = Edge.ByVertices([vertices[i], newVertices[i]])
                            nv = Edge.VertexByDistance(bisector, distance=finalMiterThreshold, origin=Edge.StartVertex(bisector), tolerance=0.0001)
                            vec = Edge.Normal(bisector)
                            nv2 = Topology.Translate(nv, vec[0], vec[1], 0)
                            nv3 = Topology.Translate(nv, -vec[0], -vec[1], 0)
                            miterEdge = Edge.ByVertices([nv2,nv3])
                            if miterEdge:
                                miterEdge = Edge.SetLength(miterEdge, abs(offset)*10)
                                msv = Edge.Intersect2D(miterEdge, e1)
                                mev = Edge.Intersect2D(miterEdge, e2)
                                if (Topology.IsInside(e1, msv,tolerance=0.01) and (Topology.IsInside(e2, mev, tolerance=0.01))):
                                    miterEdge = Edge.ByVertices([msv, mev])
                                    if miterEdge:
                                        cleanMiterEdges.append(miterEdge)
                                        miterEdge = Edge.SetLength(miterEdge, Edge.Length(miterEdge)*1.02)
                                        miterEdges.append(miterEdge)

            c = Cluster.SelfMerge(Cluster.ByTopologies(newEdges+miterEdges))
            vertices = Wire.Vertices(c)
            subtractEdges = []
            for v in vertices:
                edges = Topology.SuperTopologies(v, c, topologyType="edge")
                if len(edges) == 2:
                    if not Edge.IsCollinear(edges[0], edges[1]):
                        adjacentVertices = Topology.AdjacentTopologies(v, c)
                        total = 0
                        for adjV in adjacentVertices:
                            tempEdges = Topology.SuperTopologies(adjV, c, topologyType="edge")
                            total += len(tempEdges)
                        if total == 8:
                            subtractEdges = subtractEdges+edges

            if len(subtractEdges) > 0:
                newWire = Topology.Boolean(newWire, Cluster.ByTopologies(subtractEdges), operation="difference")
                if len(cleanMiterEdges) > 0:
                    newWire = Topology.Boolean(newWire, Cluster.ByTopologies(cleanMiterEdges), operation="merge")

        newWire = Topology.Rotate(newWire, origin=world_origin, x=0, y=1, z=0, degree=theta)
        newWire = Topology.Rotate(newWire, origin=world_origin, x=0, y=0, z=1, degree=phi)
        newWire = Topology.Translate(newWire, xTran, yTran, zTran)
        return newWire

    @staticmethod
    def ByVertices(vertices, close=True):
        """
        Creates a wire from the input list of vertices.

        Parameters
        ----------
        vertices : list
            the input list of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.

        Returns
        -------
        topologic.Wire
            The created wire.

        """
        from topologicpy.Cluster import Cluster
        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, topologic.Vertex)]
        if len(vertexList) < 2:
            return None
        edges = []
        for i in range(len(vertexList)-1):
            v1 = vertexList[i]
            v2 = vertexList[i+1]
            try:
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                if e:
                    edges.append(e)
            except:
                continue
        if close:
            v1 = vertexList[-1]
            v2 = vertexList[0]
            try:
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                if e:
                    edges.append(e)
            except:
                pass
        if len(edges) < 1:
            return None
        #return Wire.ByEdges(edges)
        c = Cluster.ByTopologies(edges)
        return Cluster.SelfMerge(c)

    @staticmethod
    def ByVerticesCluster(cluster, close=True):
        """
        Creates a wire from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic.cluster
            the input cluster of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.

        Returns
        -------
        topologic.Wire
            The created wire.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        vertices = []
        _ = cluster.Vertices(None, vertices)
        return Wire.ByVertices(vertices, close)

    @staticmethod
    def Circle(origin=None, radius=0.5, sides=16, fromAngle=0, toAngle=360, close=True, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the circle. The default is 0.5.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Wire
            The created circle.

        """
        if not origin:
            origin = topologic.Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        radius = abs(radius)
        if radius < tolerance:
            return None
        
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle-fromAngle) < tolerance:
            return None
        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*radius + origin.X()
            y = math.cos(angle)*radius + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append(topologic.Vertex.ByCoordinates(x,y,z))

        baseWire = Wire.ByVertices(baseV[::-1], close) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            baseWire = topologic.TopologyUtility.Translate(baseWire, radius, radius, 0)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        baseWire = topologic.TopologyUtility.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = topologic.TopologyUtility.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire

    
    @staticmethod
    def Cycles(wire, maxVertices=4, tolerance=0.0001):
        """
        Returns the closed circuits of wires found within the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.
        maxVertices : int , optional
            The maximum number of vertices of the circuits to be searched. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of circuits (closed wires) found within the input wire.

        """
        
        def vIndex(v, vList, tolerance):
            for i in range(len(vList)):
                if topologic.VertexUtility.Distance(v, vList[i]) < tolerance:
                    return i+1
            return None
        
        #  rotate cycle path such that it begins with the smallest node
        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:]+path[:n]

        def invert(path):
            return rotate_to_smallest(path[::-1])

        def isNew(cycles, path):
            return not path in cycles

        def visited(node, path):
            return node in path

        def findNewCycles(graph, cycles, path, maxVertices):
            if len(path) > maxVertices:
                return
            start_node = path[0]
            next_node= None
            sub = []

            #visit each edge and each node of each edge
            for edge in graph:
                node1, node2 = edge
                if start_node in edge:
                        if node1 == start_node:
                            next_node = node2
                        else:
                            next_node = node1
                        if not visited(next_node, path):
                                # neighbor node not on path yet
                                sub = [next_node]
                                sub.extend(path)
                                # explore extended path
                                findNewCycles(graph, cycles, sub, maxVertices);
                        elif len(path) > 2  and next_node == path[-1]:
                                # cycle found
                                p = rotate_to_smallest(path);
                                inv = invert(p)
                                if isNew(cycles, p) and isNew(cycles, inv):
                                    cycles.append(p)

        def main(graph, cycles, maxVertices):
            returnValue = []
            for edge in graph:
                for node in edge:
                    findNewCycles(graph, cycles, [node], maxVertices)
            for cy in cycles:
                row = []
                for node in cy:
                    row.append(node)
                returnValue.append(row)
            return returnValue

        tEdges = []
        _ = wire.Edges(None, tEdges)
        tVertices = []
        _ = wire.Vertices(None, tVertices)
        tVertices = tVertices

        graph = []
        for anEdge in tEdges:
            graph.append([vIndex(anEdge.StartVertex(), tVertices, tolerance), vIndex(anEdge.EndVertex(), tVertices, tolerance)])

        cycles = []
        resultingCycles = main(graph, cycles, maxVertices)

        result = []
        for aRow in resultingCycles:
            row = []
            for anIndex in aRow:
                row.append(tVertices[anIndex-1])
            result.append(row)

        resultWires = []
        for i in range(len(result)):
            c = result[i]
            resultEdges = []
            for j in range(len(c)-1):
                v1 = c[j]
                v2 = c[j+1]
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                resultEdges.append(e)
            e = topologic.Edge.ByStartVertexEndVertex(c[len(c)-1], c[0])
            resultEdges.append(e)
            resultWire = topologic.Wire.ByEdges(resultEdges)
            resultWires.append(resultWire)
        return resultWires

    @staticmethod
    def Edges(wire):
        """
        Returns the edges of the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.

        Returns
        -------
        list
            The list of edges.

        """
        if not isinstance(wire, topologic.Wire):
            return None
        edges = []
        _ = wire.Edges(None, edges)
        return edges

    @staticmethod
    def Ellipse(origin=None, inputMode=1, width=2.0, length=1.0, focalLength=0.866025, eccentricity=0.866025, majorAxisLength=1.0, minorAxisLength=0.5, sides=32, fromAngle=0, toAngle=360, close=True, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0,0,0).
        inputMode : int , optional
            The method by wich the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Wire
            The created ellipse

        """
        ellipseAll = Wire.EllipseAll(origin=origin, inputMode=inputMode, width=width, length=length, focalLength=focalLength, eccentricity=eccentricity, majorAxisLength=majorAxisLength, minorAxisLength=minorAxisLength, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=close, direction=direction, placement=placement, tolerance=tolerance)
        return ellipseAll["ellipse"]

    @staticmethod
    def EllipseAll(origin=None, inputMode=1, width=2.0, length=1.0, focalLength= 0.866025, eccentricity=0.866025, majorAxisLength=1.0, minorAxisLength=0.5, sides=32, fromAngle=0, toAngle=360, close=True, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0,0,0).
        inputMode : int , optional
            The method by wich the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dictionary
            A dictionary with the following keys and values:
            1. "ellipse" : The ellipse (topologic.Wire)
            2. "foci" : The two focal points (topologic.Cluster containing two vertices)
            3. "a" : The major axis length
            4. "b" : The minor axis length
            5. "c" : The focal length
            6. "e" : The eccentricity
            7. "width" : The width
            8. "length" : The length

        """
        if not origin:
            origin = topologic.Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        if inputMode not in [1,2,3,4]:
            return None
        if placement.lower() not in ["center", "lowerleft"]:
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            return None
        width = abs(width)
        length = abs(length)
        focalLength= abs(focalLength)
        eccentricity=abs(eccentricity)
        majorAxisLength=abs(majorAxisLength)
        minorAxisLength=abs(minorAxisLength)
        sides = abs(sides)
        if width < tolerance or length < tolerance or focalLength < tolerance or eccentricity < tolerance or majorAxisLength < tolerance or minorAxisLength < tolerance or sides < 3:
            return None
        if inputMode == 1:
            w = width
            l = length
            a = width/2
            b = length/2
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
        elif inputMode == 2:
            c = focalLength
            e = eccentricity
            a = c/e
            b = math.sqrt(abs(a**2 - c**2))
            w = a*2
            l = b*2
        elif inputMode == 3:
            c = focalLength
            b = minorAxisLength
            a = math.sqrt(abs(b**2 + c**2))
            e = c/a
            w = a*2
            l = b*2
        elif inputMode == 4:
            a = majorAxisLength
            b = minorAxisLength
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
            w = a*2
            l = b*2
        else:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle - fromAngle) < tolerance:
            return None

        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*a + origin.X()
            y = math.cos(angle)*b + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append(topologic.Vertex.ByCoordinates(x,y,z))

        ellipse = Wire.ByVertices(baseV[::-1], close) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            ellipse = topologic.TopologyUtility.Translate(ellipse, a, b, 0)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        ellipse = topologic.TopologyUtility.Rotate(ellipse, origin, 0, 1, 0, theta)
        ellipse = topologic.TopologyUtility.Rotate(ellipse, origin, 0, 0, 1, phi)

        # Create a Cluster of the two foci
        v1 = topologic.Vertex.ByCoordinates(c+origin.X(), 0+origin.Y(),0)
        v2 = topologic.Vertex.ByCoordinates(-c+origin.X(), 0+origin.Y(),0)
        foci = topologic.Cluster.ByTopologies([v1, v2])
        if placement.lower() == "lowerleft":
            foci = topologic.TopologyUtility.Translate(foci, a, b, 0)
        foci = topologic.TopologyUtility.Rotate(foci, origin, 0, 1, 0, theta)
        foci = topologic.TopologyUtility.Rotate(foci, origin, 0, 0, 1, phi)
        d = {}
        d['ellipse'] = ellipse
        d['foci'] = foci
        d['a'] = a
        d['b'] = b
        d['c'] = c
        d['e'] = e
        d['w'] = w
        d['l'] = l
        return d

    @staticmethod
    def Flatten(wire, oldLocation=None, newLocation=None, direction=None):
        """
        Flattens the input wire such that its center of mass is located at the origin and the specified direction is pointed in the positive Z axis.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.
        oldLocation : topologic.Vertex , optional
            The old location to use as the origin of the movement. If set to None, the center of mass of the input topology is used. The default is None.
        newLocation : topologic.Vertex , optional
            The new location at which to place the topology. If set to None, the world origin (0,0,0) is used. The default is None.
        direction : list , optional
            The direction, expressed as a list of [X,Y,Z] that signifies the direction of the wire. If set to None, the positive ZAxis direction is considered the direction of the wire. The deafult is None.

        Returns
        -------
        topologic.Wire
            The flattened wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector
        if not isinstance(wire, topologic.Wire):
            return None
        if direction == None:
            direction = Vector.ZAxis()
        if not isinstance(oldLocation, topologic.Vertex):
            oldLocation = Topology.CenterOfMass(wire)
        if not isinstance(newLocation, topologic.Vertex):
            newLocation = Vertex.ByCoordinates(0,0,0)
        cm = oldLocation
        world_origin = newLocation

        x1 = Vertex.X(cm)
        y1 = Vertex.Y(cm)
        z1 = Vertex.Z(cm)
        x2 = Vertex.X(cm) + direction[0]
        y2 = Vertex.Y(cm) + direction[1]
        z2 = Vertex.Z(cm) + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        flatWire = Topology.Translate(wire, -cm.X(), -cm.Y(), -cm.Z())
        flatWire = Topology.Rotate(flatWire, world_origin, 0, 0, 1, -phi)
        flatWire = Topology.Rotate(flatWire, world_origin, 0, 1, 0, -theta)
        # Ensure flatness. Force Z to be zero
        edges = Wire.Edges(flatWire)
        flatEdges = []
        for edge in edges:
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            sv1 = Vertex.ByCoordinates(Vertex.X(sv), Vertex.Y(sv), 0)
            ev1 = Vertex.ByCoordinates(Vertex.X(ev), Vertex.Y(ev), 0)
            e1 = Edge.ByVertices([sv1, ev1])
            flatEdges.append(e1)
        flatWire = Topology.SelfMerge(Cluster.ByTopologies(flatEdges))
        dictionary = Dictionary.ByKeysValues(["xTran", "yTran", "zTran", "phi", "theta"], [cm.X(), cm.Y(), cm.Z(), phi, theta])
        flatWire = Topology.SetDictionary(flatWire, dictionary)
        return flatWire
    
    @staticmethod
    def Interpolate(wires: list, n: int = 5, outputType: str = "default", replication: str = "default"):
        """
        Creates *n* number of wires that interpolate between wireA and wireB.

        Parameters
        ----------
        wireA : topologic.Wire
            The first input wire.
        wireB : topologic.Wire
            The second input wire.
        n : int , optional
            The number of intermediate wires to create. The default is 5.
        outputType : str , optional
            The desired type of output. The options are case insensitive. The default is "contour". The options are:
                - "Default" or "Contours" (wires are not connected)
                - "Raster or "Zigzag" or "Toolpath" (the wire ends are connected to create a continous path)
                - "Grid" (the wire ends are connected to create a grid). 
        replication : str , optiona;
            The desired type of replication for wires with different number of vertices. It is case insensitive. The default is "default". The options are:
                - "Default" or "Repeat" which repeats the last vertex of the wire with the least number of vertices
                - "Nearest" which maps the vertices of one wire to the nearest vertex of the next wire creating a list of equal number of vertices.
        Returns
        -------
        toplogic.Topology
            The created interpolated wires as well as the input wires. The return type can be a topologic.Cluster or a topologic.Wire based on options.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Helper import Helper
        
        outputType = outputType.lower()
        if outputType not in ["default", "contours", "raster", "zigzag", "toolpath", "grid"]:
            return None
        if outputType == "default" or outputType == "contours":
            outputType = "contours"
        if outputType == "raster" or outputType == "zigzag" or outputType == "toolpath":
            outputType = "zigzag"
        
        replication = replication.lower()
        if replication not in ["default", "nearest", "repeat"]:
            return None
        
        def nearestVertex(v, vertices):
            distances = [Vertex.Distance(v, vertex) for vertex in vertices]
            return vertices[distances.index(sorted(distances)[0])]
        
        def replicate(vertices, replication="default"):
            vertices = Helper.Repeat(vertices)
            finalList = vertices
            if replication == "nearest":
                finalList = [vertices[0]]
                for i in range(len(vertices)-1):
                    loopA = vertices[i]
                    loopB = vertices[i+1]
                    nearestVertices = []
                    for j in range(len(loopA)):
                        #clusB = Cluster.ByTopologies(loopB)
                        #nv = Vertex.NearestVertex(loopA[j], clusB, useKDTree=False)
                        nv = nearestVertex(loopA[j], loopB)
                        nearestVertices.append(nv)
                    finalList.append(nearestVertices)
            return finalList
        
        def process(verticesA, verticesB, n=5, outputType="contours", replication="repeat"):
            #if outputType == "zigzag" and Wire.IsClosed(wireA):
                #verticesA.append(verticesA[0])
            #verticesA, verticesB = replicate(verticesA=verticesA, verticesB=verticesB, replication=replication)
            
            contours = [verticesA]
            for i in range(1, n+1):
                u = float(i)/float(n+1)
                temp_vertices = []
                for j in range(len(verticesA)):
                    temp_v = Edge.VertexByParameter(Edge.ByVertices([verticesA[j], verticesB[j]]), u)
                    temp_vertices.append(temp_v)
                contours.append(temp_vertices)
            return contours
        
        if len(wires) < 2:
            return None
        
        vertices = []
        for wire in wires:
            vertices.append(Topology.SubTopologies(wire, subTopologyType="vertex"))
        vertices = replicate(vertices, replication=replication)
        contours = []
        
        finalWires = []
        for i in range(len(vertices)-1):
            verticesA = vertices[i]
            verticesB = vertices[i+1]
            contour = process(verticesA=verticesA, verticesB=verticesB, n=n, outputType=outputType, replication=replication)
            contours += contour
            for c in contour:
                finalWires.append(Wire.ByVertices(c, Wire.IsClosed(wires[i])))

        contours.append(vertices[-1])
        finalWires.append(wires[-1])
        ridges = []
        if outputType == "grid" or outputType == "zigzag":
            for i in range(len(contours)-1):
                verticesA = contours[i]
                verticesB = contours[i+1]
                if outputType == "grid":
                    for j in range(len(verticesA)):
                        ridges.append(Edge.ByVertices([verticesA[j], verticesB[j]]))
                elif outputType == "zigzag":
                    if i%2 == 0:
                        sv = verticesA[-1]
                        ev = verticesB[-1]
                        ridges.append(Edge.ByVertices([sv, ev]))
                    else:
                        sv = verticesA[0]
                        ev = verticesB[0]
                        ridges.append(Edge.ByVertices([sv, ev]))

        return Topology.SelfMerge(Cluster.ByTopologies(finalWires+ridges))
    
    @staticmethod
    def Invert(wire):
        """
        Creates a wire that is an inverse (mirror) of the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.

        Returns
        -------
        topologic.Wire
            The inverted wire.

        """
        if not isinstance(wire, topologic.Wire):
            return None
        vertices = Wire.Vertices(wire)
        reversed_vertices = vertices[::-1]
        return Wire.ByVertices(reversed_vertices)

    @staticmethod
    def IsClosed(wire):
        """
        Returns True if the input wire is closed. Returns False otherwise.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.

        Returns
        -------
        bool
            True if the input wire is closed. False otherwise.

        """
        status = None
        if wire:
            if isinstance(wire, topologic.Wire):
                status = wire.IsClosed()
        return status
    
    @staticmethod
    def Isovist(viewPoint, externalBoundary, obstaclesCluster, tolerance=0.0001):
        """
        Returns a list of faces representing the isovist projection from the input viewpoint.

        Parameters
        ----------
        viewPoint : topologic.Vertex
            The vertex representing the location of the viewpoint of the isovist.
        externalBoundary : topologic.Wire
            The wire representing the external boundary (border) of the isovist.
        obstaclesCluster : topologic.Cluster
            A cluster of wires representing the obstacles within the externalBoundary.

        Returns
        -------
        list
            A list of faces representing the isovist projection from the input viewpoint.

        """
        
        def vertexPartofFace(vertex, face, tolerance):
            vertices = []
            _ = face.Vertices(None, vertices)
            for v in vertices:
                if topologic.VertexUtility.Distance(vertex, v) < tolerance:
                    return True
            return False
        
        internalBoundaries = []
        _ = obstaclesCluster.Wires(None, internalBoundaries)
        internalVertices = []
        _ = obstaclesCluster.Vertices(None, internalVertices)
        # 1. Create a Face with external and internal boundaries
        face = topologic.Face.ByExternalInternalBoundaries(externalBoundary, internalBoundaries, False)
        # 2. Draw Rays from viewpoint through each Vertex of the obstacles extending to the External Boundary
        #    2.1 Get the Edges and Vertices of the External Boundary
        exBoundaryEdges = []
        _ = externalBoundary.Edges(None, exBoundaryEdges)
        exBoundaryVertices = []
        _ = externalBoundary.Vertices(None, exBoundaryVertices)
        testTopologies = exBoundaryEdges+exBoundaryVertices
        #    1.2 Find the maximum distance from the viewpoint to the edges and vertices of the external boundary
        distances = []
        for x in testTopologies:
            distances.append(topologic.VertexUtility.Distance(viewPoint, x))
        maxDistance = max(distances)*1.5
        #    1.3 Shoot rays and intersect with the external boundary
        rays = []
        for aVertex in (internalVertices+exBoundaryVertices):
            d = topologic.VertexUtility.Distance(viewPoint, aVertex)
            if d > tolerance:
                scaleFactor = maxDistance/d
                newV = topologic.TopologyUtility.Scale(aVertex, viewPoint, scaleFactor, scaleFactor, scaleFactor)
                try:
                    ray = topologic.Edge.ByStartVertexEndVertex(viewPoint, newV)
                    topologyC = ray.Intersect(externalBoundary, False)
                    vertices = []
                    _ = topologyC.Vertices(None, vertices)
                    if topologyC:
                        try:
                            rays.append(topologic.Edge.ByStartVertexEndVertex(viewPoint, vertices[0]))
                        except:
                            pass
                    try:
                        rays.append(topologic.Edge.ByStartVertexEndVertex(viewPoint, aVertex))
                    except:
                        pass
                except:
                    pass
        rayEdges = []
        for r in rays:
            a = r.Difference(obstaclesCluster, False)
            if a:
                edges = []
                _ = a.Edges(None, edges)
                w = None
                try:
                    w = topologic.Wire.ByEdges(edges)
                    rayEdges = rayEdges + edges
                except:
                    c = topologic.Cluster.ByTopologies(edges)
                    c = c.SelfMerge()
                    wires = []
                    _ = c.Wires(None, wires)
                    if len(wires) > 0:
                        edges = []
                        _ = wires[0].Edges(None, edges)
                        rayEdges = rayEdges + edges
                    else:
                        for e in edges:
                            vertices = []
                            e.Vertices(None, vertices)
                            for v in vertices:
                                if topologic.VertexUtility.Distance(viewPoint, v) < tolerance:
                                    rayEdges.append(e)
        rayCluster = topologic.Cluster.ByTopologies(rayEdges)
        #return rayCluster
        shell = face.Slice(rayCluster, False)
        faces = []
        _ = shell.Faces(None, faces)
        finalFaces = []
        for aFace in faces:
            if vertexPartofFace(viewPoint, aFace, 0.001):
                finalFaces.append(aFace)
        return finalFaces

    
    @staticmethod
    def IsSimilar(wireA, wireB, tolerance=0.0001, angTolerance=0.1):
        """
        Returns True if the input wires are similar. Returns False otherwise. The wires must be closed.

        Parameters
        ----------
        wireA : topologic.Wire
            The first input wire.
        wireB : topologic.Wire
            The second input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.

        Returns
        -------
        bool
            True if the two input wires are similar. False otherwise.

        """
        
        def isCyclicallyEquivalent(u, v, lengthTolerance, angleTolerance):
            n, i, j = len(u), 0, 0
            if n != len(v):
                return False
            while i < n and j < n:
                if (i % 2) == 0:
                    tol = lengthTolerance
                else:
                    tol = angleTolerance
                k = 1
                while k <= n and math.fabs(u[(i + k) % n]- v[(j + k) % n]) <= tol:
                    k += 1
                if k > n:
                    return True
                if math.fabs(u[(i + k) % n]- v[(j + k) % n]) > tol:
                    i += k
                else:
                    j += k
            return False

        def angleBetweenEdges(e1, e2, tolerance):
            a = e1.EndVertex().X() - e1.StartVertex().X()
            b = e1.EndVertex().Y() - e1.StartVertex().Y()
            c = e1.EndVertex().Z() - e1.StartVertex().Z()
            d = topologic.VertexUtility.Distance(e1.EndVertex(), e2.StartVertex())
            if d <= tolerance:
                d = e2.StartVertex().X() - e2.EndVertex().X()
                e = e2.StartVertex().Y() - e2.EndVertex().Y()
                f = e2.StartVertex().Z() - e2.EndVertex().Z()
            else:
                d = e2.EndVertex().X() - e2.StartVertex().X()
                e = e2.EndVertex().Y() - e2.StartVertex().Y()
                f = e2.EndVertex().Z() - e2.StartVertex().Z()
            dotProduct = a*d + b*e + c*f
            modOfVector1 = math.sqrt( a*a + b*b + c*c)*math.sqrt(d*d + e*e + f*f) 
            angle = dotProduct/modOfVector1
            angleInDegrees = math.degrees(math.acos(angle))
            return angleInDegrees

        def getInteriorAngles(edges, tolerance):
            angles = []
            for i in range(len(edges)-1):
                e1 = edges[i]
                e2 = edges[i+1]
                angles.append(angleBetweenEdges(e1, e2, tolerance))
            return angles

        def getRep(edges, tolerance):
            angles = getInteriorAngles(edges, tolerance)
            lengths = []
            for anEdge in edges:
                lengths.append(topologic.EdgeUtility.Length(anEdge))
            minLength = min(lengths)
            normalisedLengths = []
            for aLength in lengths:
                normalisedLengths.append(aLength/minLength)
            return [x for x in itertools.chain(*itertools.zip_longest(normalisedLengths, angles)) if x is not None]
        
        if (wireA.IsClosed() == False):
            return None
        if (wireB.IsClosed() == False):
            return None
        edgesA = []
        _ = wireA.Edges(None, edgesA)
        edgesB = []
        _ = wireB.Edges(None, edgesB)
        if len(edgesA) != len(edgesB):
            return False
        repA = getRep(list(edgesA), tolerance)
        repB = getRep(list(edgesB), tolerance)
        if isCyclicallyEquivalent(repA, repB, tolerance, angTolerance):
            return True
        if isCyclicallyEquivalent(repA, repB[::-1], tolerance, angTolerance):
            return True
        return False

    
    @staticmethod
    def Length(wire, mantissa=4):
        """
        Returns the length of the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The length of the input wire.

        """
        if not wire:
            return None
        if not isinstance(wire, topologic.Wire):
            return None
        totalLength = None
        try:
            edges = []
            _ = wire.Edges(None, edges)
            totalLength = 0
            for anEdge in edges:
                totalLength = totalLength + topologic.EdgeUtility.Length(anEdge)
            totalLength = round(totalLength, mantissa)
        except:
            totalLength = None
        return totalLength

    @staticmethod
    def Planarize(wire):
        """
        Returns a planarized version of the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.

        Returns
        -------
        topologic.Wire
            The planarized wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        if not isinstance(wire, topologic.Wire):
            return None
        verts = []
        _ = wire.Vertices(None, verts)
        w = Wire.ByVertices([verts[0], verts[1], verts[2]], close=True)
        f = topologic.Face.ByExternalBoundary(w)
        f = Topology.Scale(f, f.Centroid(), 500,500,500)
        proj_verts = []
        direction = Face.NormalAtParameters(f)
        for v in verts:
            v = Vertex.ByCoordinates(v.X()+direction[0]*5, v.Y()+direction[1]*5, v.Z()+direction[2]*5)
            proj_verts.append(Vertex.Project(v, f))
        return Wire.ByVertices(proj_verts, close=True)

    @staticmethod
    def Project(wire, face, direction=None, mantissa=4, tolerance=0.0001):
        """
        Creates a projection of the input wire unto the input face.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.
        face : topologic.Face
            The face unto which to project the input wire.
        direction : list, optional
            The vector direction of the projection. If None, the reverse vector of the receiving face normal will be used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Wire
            The projected wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        if not wire:
            return None
        if not isinstance(wire, topologic.Wire):
            return None
        if not face:
            return None
        if not isinstance(face, topologic.Face):
            return None
        if not direction:
            direction = -1*Face.NormalAtParameters(face, 0.5, 0.5, "XYZ", mantissa)
        large_face = Topology.Scale(face, face.CenterOfMass(), 500, 500, 500)
        edges = []
        _ = wire.Edges(None, edges)
        projected_edges = []

        if large_face:
            if (large_face.Type() == Face.Type()):
                for edge in edges:
                    if edge:
                        if (edge.Type() == topologic.Edge.Type()):
                            sv = edge.StartVertex()
                            ev = edge.EndVertex()

                            psv = Vertex.Project(vertex=sv, face=large_face, direction=direction)
                            pev = Vertex.Project(vertex=ev, face=large_face, direction=direction)
                            if psv and pev:
                                try:
                                    pe = Edge.ByVertices([psv, pev])
                                    projected_edges.append(pe)
                                except:
                                    continue
        w = Wire.ByEdges(projected_edges)
        return w

    @staticmethod
    def Rectangle(origin=None, width=1.0, length=1.0, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0,0,0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        direction : list , optional
            The ector representing the up direction of the rectangle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Wire
            The created rectangle.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        if not placement.lower() in ["center", "lowerleft"]:
            return None
        width = abs(width)
        length = abs(length)
        if width < tolerance or length < tolerance:
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5

        vb1 = Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb2 = Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb3 = Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())
        vb4 = Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], True)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        baseWire = Topology.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = Topology.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire
    
    @staticmethod
    def RemoveCollinearEdges(wire, angTolerance=0.1):
        """
        Removes any collinear edges in the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.

        Returns
        -------
        topologic.Wire
            The created wire without any collinear edges.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        def rce(wire, angTolerance=0.1):
            if not isinstance(wire, topologic.Wire):
                return None
            final_wire = None
            vertices = []
            wire_verts = []
            try:
                _ = wire.Vertices(None, vertices)
            except:
                return None
            for aVertex in vertices:
                edges = []
                _ = aVertex.Edges(wire, edges)
                if len(edges) > 1:
                    if not Edge.IsCollinear(edges[0], edges[1], angTolerance=angTolerance):
                        wire_verts.append(aVertex)
                else:
                    wire_verts.append(aVertex)
            if len(wire_verts) > 2:
                if wire.IsClosed():
                    final_wire = Wire.ByVertices(wire_verts, True)
                else:
                    final_wire = Wire.ByVertices(wire_verts, False)
            elif len(wire_verts) == 2:
                final_wire = topologic.Edge.ByStartVertexEndVertex(wire_verts[0], wire_verts[1])
            return final_wire
        
        if not topologic.Topology.IsManifold(wire, wire):
            wires = Wire.Split(wire)
        else:
            wires = [wire]
        returnWires = []
        for aWire in wires:
            if not isinstance(aWire, topologic.Wire):
                returnWires.append(aWire)
            else:
                returnWires.append(rce(aWire, angTolerance=angTolerance))
        if len(returnWires) == 1:
            returnWire = returnWires[0]
            if isinstance(returnWire, topologic.Edge):
                return Wire.ByEdges([returnWire])
            elif isinstance(returnWire, topologic.Wire):
                return returnWire
            else:
                return None
        elif len(returnWires) > 1:
            returnWire = topologic.Cluster.ByTopologies(returnWires).SelfMerge()
            if isinstance(returnWire, topologic.Edge):
                return Wire.ByEdges([returnWire])
            elif isinstance(returnWire, topologic.Wire):
                return returnWire
            else:
                return None
        else:
            return None

    
    @staticmethod
    def Split(wire):
        """
        Splits the input wire into segments at its intersections (i.e. at any vertex where more than two edges meet).

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.

        Returns
        -------
        list
            The list of split wire segments.

        """
        
        def vertexDegree(v, wire):
            edges = []
            _ = v.Edges(wire, edges)
            return len(edges)
        
        def vertexOtherEdge(vertex, edge, wire):
            edges = []
            _ = vertex.Edges(wire, edges)
            if topologic.Topology.IsSame(edges[0], edge):
                return edges[-1]
            else:
                return edges[0]
        
        def edgeOtherVertex(edge, vertex):
            vertices = []
            _ = edge.Vertices(None, vertices)
            if topologic.Topology.IsSame(vertex, vertices[0]):
                return vertices[-1]
            else:
                return vertices[0]
        
        def edgeInList(edge, edgeList):
            for anEdge in edgeList:
                if topologic.Topology.IsSame(anEdge, edge):
                    return True
            return False
        
        vertices = []
        _ = wire.Vertices(None, vertices)
        hubs = []
        for aVertex in vertices:
            if vertexDegree(aVertex, wire) > 2:
                hubs.append(aVertex)
        wires = []
        global_edges = []
        for aVertex in hubs:
            hub_edges = []
            _ = aVertex.Edges(wire, hub_edges)
            wire_edges = []
            for hub_edge in hub_edges:
                if not edgeInList(hub_edge, global_edges):
                    current_edge = hub_edge
                    oe = edgeOtherVertex(current_edge, aVertex)
                    while vertexDegree(oe, wire) == 2:
                        if not edgeInList(current_edge, global_edges):
                            global_edges.append(current_edge)
                            wire_edges.append(current_edge)
                        current_edge = vertexOtherEdge(oe, current_edge, wire)
                        oe = edgeOtherVertex(current_edge, oe)
                    if not edgeInList(current_edge, global_edges):
                        global_edges.append(current_edge)
                        wire_edges.append(current_edge)
                    if len(wire_edges) > 1:
                        wires.append(topologic.Cluster.ByTopologies(wire_edges).SelfMerge())
                    else:
                        wires.append(wire_edges[0])
                    wire_edges = []
        if len(wires) < 1:
            return [wire]
        return wires

    
    @staticmethod
    def Star(origin=None, radiusA=1.0, radiusB=0.4, rays=5, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a star.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the star. The default is None which results in the star being placed at (0,0,0).
        radiusA : float , optional
            The outer radius of the star. The default is 1.0.
        radiusB : float , optional
            The outer radius of the star. The default is 0.4.
        rays : int , optional
            The number of star rays. The default is 5.
        direction : list , optional
            The vector representing the up direction of the star. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Wire
            The created star.

        """

        if not origin:
            origin = topologic.Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        radiusA = abs(radiusA)
        radiusB = abs(radiusB)
        if radiusA < tolerance or radiusB < tolerance:
            return None
        rays = abs(rays)
        if rays < 3:
            return None
        if not placement.lower() in ["center", "lowerleft"]:
            return None
        sides = rays*2 # Sides is double the number of rays
        baseV = []

        xList = []
        yList = []
        for i in range(sides):
            if i%2 == 0:
                radius = radiusA
            else:
                radius = radiusB
            angle = math.radians(360/sides)*i
            x = math.sin(angle)*radius + origin.X()
            y = math.cos(angle)*radius + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append([x,y])

        if placement.lower() == "lowerleft":
            xmin = min(xList)
            ymin = min(yList)
            xOffset = origin.X() - xmin
            yOffset = origin.Y() - ymin
        else:
            xOffset = 0
            yOffset = 0
        tranBase = []
        for coord in baseV:
            tranBase.append(topologic.Vertex.ByCoordinates(coord[0]+xOffset, coord[1]+yOffset, origin.Z()))
        
        baseWire = Wire.ByVertices(tranBase[::-1], True) #reversing the list so that the normal points up in Blender
        
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Z-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Y-Axis
        baseWire = topologic.TopologyUtility.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = topologic.TopologyUtility.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire

    
    @staticmethod
    def Trapezoid(origin=None, widthA=1.0, widthB=0.75, offsetA=0.0, offsetB=0.0, length=1.0, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the trapezoid. The default is None which results in the trapezoid being placed at (0,0,0).
        widthA : float , optional
            The width of the bottom edge of the trapezoid. The default is 1.0.
        widthB : float , optional
            The width of the top edge of the trapezoid. The default is 0.75.
        offsetA : float , optional
            The offset of the bottom edge of the trapezoid. The default is 0.0.
        offsetB : float , optional
            The offset of the top edge of the trapezoid. The default is 0.0.
        length : float , optional
            The length of the trapezoid. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the trapezoid. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Wire
            The created trapezoid.

        """
        if not origin:
            origin = topologic.Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        widthA = abs(widthA)
        widthB = abs(widthB)
        length = abs(length)
        if widthA < tolerance or widthB < tolerance or length < tolerance:
            return None
        if not placement.lower() in ["center", "lowerleft"]:
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "center":
            xOffset = -((-widthA*0.5 + offsetA) + (-widthB*0.5 + offsetB) + (widthA*0.5 + offsetA) + (widthB*0.5 + offsetB))/4.0
            yOffset = 0
        elif placement.lower() == "lowerleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = length*0.5

        vb1 = topologic.Vertex.ByCoordinates(origin.X()-widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb2 = topologic.Vertex.ByCoordinates(origin.X()+widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb3 = topologic.Vertex.ByCoordinates(origin.X()+widthB*0.5+offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())
        vb4 = topologic.Vertex.ByCoordinates(origin.X()-widthB*0.5++offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], True)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        baseWire = topologic.TopologyUtility.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = topologic.TopologyUtility.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire

    @staticmethod
    def Vertices(wire):
        """
        Returns the vertices of the input wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(wire, topologic.Wire):
            return None
        vertices = []
        _ = wire.Vertices(None, vertices)
        return vertices

