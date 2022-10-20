import topologic
import math
import itertools

class Wire(topologic.Wire):
    @staticmethod
    def ByEdges(edges):
        """
        Parameters
        ----------
        edges : list
            The input list of topologic Edges.

        Returns
        -------
        topologic.Wire
            The created topologic Wire.

        """
        if not isinstance(edges, list):
            return None
        edgeList = [x for x in edges if isinstance(x, topologic.Edge)]
        wire = None
        for anEdge in edgeList:
            if anEdge.Type() == 2:
                if wire == None:
                    wire = anEdge
                else:
                    try:
                        wire = wire.Merge(anEdge)
                    except:
                        continue
        if wire.Type() != 4:
            wire = None
        return wire

    
    @staticmethod
    def ByVertices(cluster, close=True):
        """
        Parameters
        ----------
        cluster : topologic.Cluster
            the input topologic Cluster of topologic Vertices.
        close : bool
            Boolean flag to indicate if the topologic Wire should be closed or not.

        Returns
        -------
        topologic.Wire
            The created topologic Wire.

        """
        if isinstance(close, list):
            close = close[0]
        if isinstance(cluster, list):
            if all([isinstance(item, topologic.Vertex) for item in cluster]):
                vertices = cluster
        elif isinstance(cluster, topologic.Cluster):
            vertices = []
            _ = cluster.Vertices(None, vertices)
        else:
            return None
        edges = []
        for i in range(len(vertices)-1):
            v1 = vertices[i]
            v2 = vertices[i+1]
            try:
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                if e:
                    edges.append(e)
            except:
                continue
        if close:
            v1 = vertices[-1]
            v2 = vertices[0]
            try:
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                if e:
                    edges.append(e)
            except:
                pass
        if len(edges) > 0:
            c = topologic.Cluster.ByTopologies(edges, False)
            return Topology.SelfMerge(c)
        else:
            return None

    
    @staticmethod
    def Circle(origin, radius, sides, fromAngle, toAngle, close, dirX,
                   dirY, dirZ, placement):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.
        sides : TYPE
            DESCRIPTION.
        fromAngle : TYPE
            DESCRIPTION.
        toAngle : TYPE
            DESCRIPTION.
        close : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        placement : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        baseWire : TYPE
            DESCRIPTION.

        """
        # origin, \
        # radius, \
        # sides, \
        # fromAngle, \
        # toAngle, \
        # close, \
        # dirX, \
        # dirY, \
        # dirZ, \
        # placement = item
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        elif toAngle == fromAngle:
            raise Exception("Wire.Circle - Error: To angle cannot be equal to the From Angle")
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

        baseWire = Wire.ByVertices(Cluster.ByTopologies(baseV[::-1]), close) #reversing the list so that the normal points up in Blender

        if placement.lower == "lowerleft":
            baseWire = topologic.TopologyUtility.Translate(baseWire, radius, radius, 0)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + dirX
        y2 = origin.Y() + dirY
        z2 = origin.Z() + dirZ
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
    def Cycles(wire, maxVertices, tolerance=0.0001):
        """
        Parameters
        ----------
        wire : TYPE
            DESCRIPTION.
        maxVertices : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        resultWires : TYPE
            DESCRIPTION.

        """
        # wire = item[0]
        # maxVertices = item[1]
        # tolerance = item[2]
        
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
    def Ellipse(origin, w, l, sides, fromAngle, toAngle, close, dirX, dirY, dirZ, placement="center", inputMode="Width and Length"):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        w : TYPE
            Width, Focal Length or Major Axis Length
        l : TYPE
            Length, Eccentricity or Minor Axis length
        sides : TYPE
            DESCRIPTION.
        fromAngle : TYPE
            DESCRIPTION.
        toAngle : TYPE
            DESCRIPTION.
        close : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        placement : TYPE
            DESCRIPTION.
        inputMode : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.
        Exception
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        if inputMode.lower() == "width and length":
            # origin, w, l, sides, fromAngle, toAngle, close, dirX, dirY, dirZ = item
            a = w/2
            b = l/2
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
        elif inputMode.lower() == "focal length and eccentricity":
            # origin, c, e, sides, fromAngle, toAngle, close, dirX, dirY, dirZ = item
            c = w
            e = l
            a = c/e
            b = math.sqrt(abs(a**2 - c**2))
            w = a*2
            l = b*2
        elif inputMode.lower() == "focal length and minor axis":
            # origin, c, b, sides, fromAngle, toAngle, close, dirX, dirY, dirZ = item
            c = w
            b = l
            a = math.sqrt(abs(b**2 + c**2))
            e = c/a
            w = a*2
            l = b*2
        elif inputMode.lower() == "major axis length and minor axis length":
            # origin, a, b, sides, fromAngle, toAngle, close, dirX, dirY, dirZ = item
            a = w
            b = l
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
            w = a*2
            l = b*2
        else:
            raise NotImplementedError
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        elif toAngle == fromAngle:
            raise Exception("Wire.Ellipse - Error: To angle cannot be equal to the From Angle")

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

        ellipse = Wire.ByVertices(Cluster.ByTopologies(baseV[::-1]), close) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            xmin = min(xList)
            ymin = min(yList)
            ellipse = topologic.TopologyUtility.Translate(ellipse, a, b, 0)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + dirX
        y2 = origin.Y() + dirY
        z2 = origin.Z() + dirZ
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
        return [ellipse, foci, a, b, c, e, w, l]

    @staticmethod
    def IsClosed(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        returnItem : TYPE
            DESCRIPTION.

        """
        returnItem = None
        if item:
            if isinstance(item, topologic.Wire):
                returnItem = item.IsClosed()
        return returnItem
    
    @staticmethod
    def Isovist(viewPoint, externalBoundary, obstaclesCluster):
        """
        Parameters
        ----------
        viewPoint : TYPE
            DESCRIPTION.
        externalBoundary : TYPE
            DESCRIPTION.
        obstaclesCluster : TYPE
            DESCRIPTION.

        Returns
        -------
        finalFaces : TYPE
            DESCRIPTION.

        """
        # viewPoint, externalBoundary, obstaclesCluster = item
        
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
            if d > 0:
                scaleFactor = maxDistance/d
                newV = topologic.TopologyUtility.Scale(aVertex, viewPoint, scaleFactor, scaleFactor, scaleFactor)
                ray = topologic.Edge.ByStartVertexEndVertex(viewPoint, newV)
                topologyC = ray.Intersect(externalBoundary, False)
                vertices = []
                _ = topologyC.Vertices(None, vertices)
                if topologyC:
                    rays.append(topologic.Edge.ByStartVertexEndVertex(viewPoint, vertices[0]))
                rays.append(topologic.Edge.ByStartVertexEndVertex(viewPoint, aVertex))
        rayEdges = []
        for r in rays:
            a = r.Difference(obstaclesCluster, False)
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
                            if topologic.VertexUtility.Distance(viewPoint, v) < 0.0001:
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
    def IsSimilar(wireA, wireB, tolerance=0.0001, angTol=0.1):
        """
        Parameters
        ----------
        wireA : TYPE
            DESCRIPTION.
        wireB : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.
        angTol : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # wireA = item[0]
        # wireB = item[1]
        
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
            raise Exception("Error: Wire.IsSimilar - Wire A is not closed.")
        if (wireB.IsClosed() == False):
            raise Exception("Error: Wire.IsSimilar - Wire B is not closed.")
        edgesA = []
        _ = wireA.Edges(None, edgesA)
        edgesB = []
        _ = wireB.Edges(None, edgesB)
        if len(edgesA) != len(edgesB):
            return False
        # lengthTolerance = item[2]
        # angleTolerance = item[3]
        repA = getRep(list(edgesA), tolerance)
        repB = getRep(list(edgesB), tolerance)
        if isCyclicallyEquivalent(repA, repB, tolerance, angTol):
            return True
        if isCyclicallyEquivalent(repA, repB[::-1], tolerance, angTol):
            return True
        return False

    
    @staticmethod
    def Length(wire, mantissa):
        """
        Parameters
        ----------
        wire : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Returns
        -------
        totalLength : TYPE
            DESCRIPTION.

        """
        # wire, mantissa = item
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
    def Rectangle(origin, width, length, dirX, dirY, dirZ, placement):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        width : TYPE
            DESCRIPTION.
        length : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        placement : TYPE
            DESCRIPTION.

        Returns
        -------
        baseWire : TYPE
            DESCRIPTION.

        """
        # origin, width, length, dirX, dirY, dirZ, placement = item

        baseV = []
        xOffset = 0
        yOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5

        vb1 = topologic.Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb2 = topologic.Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb3 = topologic.Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())
        vb4 = topologic.Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())

        baseWire = Wire.ByVertices(topologic.Cluster.ByTopologies([vb1, vb2, vb3, vb4]), True)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + dirX
        y2 = origin.Y() + dirY
        z2 = origin.Z() + dirZ
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
    def Project(wire, face, direction):
        """
        Parameters
        ----------
        wire : TYPE
            DESCRIPTION.
        face : TYPE
            DESCRIPTION.
        direction : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        w : TYPE
            DESCRIPTION.

        """
        
        def projectVertex(vertex, face, vList):
            if topologic.FaceUtility.IsInside(face, vertex, 0.001):
                return vertex
            d = topologic.VertexUtility.Distance(vertex, face)*10
            far_vertex = topologic.TopologyUtility.Translate(vertex, vList[0]*d, vList[1]*d, vList[2]*d)
            if topologic.VertexUtility.Distance(vertex, far_vertex) > 0.001:
                e = topologic.Edge.ByStartVertexEndVertex(vertex, far_vertex)
                pv = face.Intersect(e, False)
                return pv
            else:
                return None
        
        large_face = topologic.TopologyUtility.Scale(face, face.CenterOfMass(), 500, 500, 500)
        try:
            vList = [direction.X(), direction.Y(), direction.Z()]
        except:
            try:
                vList = [direction[0], direction[1], direction[2]]
            except:
                raise Exception("Wire.Project - Error: Could not get the vector from the input direction")
        projected_wire = None
        edges = []
        _ = wire.Edges(None, edges)
        projected_edges = []

        if large_face:
            if (large_face.Type() == topologic.Face.Type()):
                for edge in edges:
                    if edge:
                        if (edge.Type() == topologic.Edge.Type()):
                            sv = edge.StartVertex()
                            ev = edge.EndVertex()

                            psv = projectVertex(sv, large_face, direction)
                            pev = projectVertex(ev, large_face, direction)
                            if psv and pev:
                                try:
                                    pe = topologic.Edge.ByStartVertexEndVertex(psv, pev)
                                    projected_edges.append(pe)
                                except:
                                    continue
        w = topologic.Wire.ByEdges(projected_edges)
        return w

    
    @staticmethod
    def WireRemoveCollinearEdges(wire, angTol=0.1):
        """
        Parameters
        ----------
        wire : TYPE
            DESCRIPTION.
        angTol : float, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # wire, angTol = item
        
        def removeCollinearEdges(wire, angTol):
            final_Wire = None
            vertices = []
            wire_verts = []
            _ = wire.Vertices(None, vertices)
            for aVertex in vertices:
                edges = []
                _ = aVertex.Edges(wire, edges)
                if len(edges) > 1:
                    if not Edge.EdgeIsCollinear(edges[0], edges[1], angTol):
                        wire_verts.append(aVertex)
                else:
                    wire_verts.append(aVertex)
            if len(wire_verts) > 2:
                clus = topologic.Cluster.ByTopologies(wire_verts)
                if wire.IsClosed():
                    final_wire = Wire.ByVertices(clus, True)
                else:
                    final_wire = Wire.ByVertices(clus, False)
            elif len(wire_verts) == 2:
                final_wire = topologic.Edge.ByStartVertexEndVertex(wire_verts[0], wire_verts[1])
            return final_wire
        
        if not topologic.Topology.IsManifold(wire, wire):
            wires = Wire.WireSplit(wire)
        else:
            wires = [wire]
        returnWires = []
        for aWire in wires:
            returnWires.append(removeCollinearEdges(aWire, angTol))
        if len(returnWires) == 1:
            return returnWires[0]
        elif len(returnWires) > 1:
            return topologic.Cluster.ByTopologies(returnWires).SelfMerge()
        else:
            return None

    
    @staticmethod
    def WireSplit(wire):
        """
        Parameters
        ----------
        wire : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

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
            return wire
        return wires

    
    @staticmethod
    def WireStar(origin, radiusA, radiusB, rays, dirX, dirY, dirZ, placement):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        radiusA : TYPE
            DESCRIPTION.
        radiusB : TYPE
            DESCRIPTION.
        rays : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        placement : TYPE
            DESCRIPTION.

        Returns
        -------
        baseWire : TYPE
            DESCRIPTION.

        """
        # origin, radiusA, radiusB, rays, dirX, dirY, dirZ, placement = item
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
        
        baseWire = Wire.ByVertices(topologic.Cluster.ByTopologies(tranBase[::-1]), True) #reversing the list so that the normal points up in Blender
        
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + dirX
        y2 = origin.Y() + dirY
        z2 = origin.Z() + dirZ
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
        centroid = baseWire.Centroid()
        return baseWire

    
    @staticmethod
    def WireTrapezoid(origin, widthA, widthB, offsetA, offsetB, length, dirX, dirY, dirZ, placement):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        widthA : TYPE
            DESCRIPTION.
        widthB : TYPE
            DESCRIPTION.
        offsetA : TYPE
            DESCRIPTION.
        offsetB : TYPE
            DESCRIPTION.
        length : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        placement : TYPE
            DESCRIPTION.

        Returns
        -------
        baseWire : TYPE
            DESCRIPTION.

        """
        # origin, widthA, widthB, offsetA, offsetB, length, dirX, dirY, dirZ, placement = item

        baseV = []
        xOffset = 0
        yOffset = 0
        if placement == "Center":
            xOffset = -((-widthA*0.5 + offsetA) + (-widthB*0.5 + offsetB) + (widthA*0.5 + offsetA) + (widthB*0.5 + offsetB))/4.0
            print("X OFFSET", xOffset)
            yOffset = 0
        elif placement.lower() == "lowerleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = length*0.5

        vb1 = topologic.Vertex.ByCoordinates(origin.X()-widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb2 = topologic.Vertex.ByCoordinates(origin.X()+widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb3 = topologic.Vertex.ByCoordinates(origin.X()+widthB*0.5+offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())
        vb4 = topologic.Vertex.ByCoordinates(origin.X()-widthB*0.5++offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())

        baseWire = Wire.ByVertices(topologic.Cluster.ByTopologies([vb1, vb2, vb3, vb4]), True)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + dirX
        y2 = origin.Y() + dirY
        z2 = origin.Z() + dirZ
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