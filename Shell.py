import topologicpy
import topologic
import math

class Shell(topologic.Shell):
    @staticmethod
    def ByFaces(faces, tolerance=0.0001):
        """
        Description
        -----------
        Creates a shell from the input list of faces.

        Parameters
        ----------
        faces : list
            The input list of faces.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Shell
            The created Shell.

        """
        if not isinstance(faces, list):
            return None
        faceList = [x for x in faces if isinstance(x, topologic.Face)]
        shell = topologic.Shell.ByFaces(faceList, tolerance)
        if not shell:
            result = faceList[0]
            remainder = faceList[1:]
            cluster = topologic.Cluster.ByTopologies(remainder, False)
            result = result.Merge(cluster, False)
            if result.Type() != 16: #16 is the type of a Shell
                if result.Type() > 16:
                    returnShells = []
                    _ = result.Shells(None, returnShells)
                    return returnShells
                else:
                    return None
        else:
            return shell

    def ByFacesCluster(cluster):
        """
        Description
        ----------
        Creates a shell from the input cluster of faces.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of faces.

        Returns
        -------
        topologic.Shell
            The created shell.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        faces = []
        _ = cluster.Faces(None, faces)
        return Shell.ByFaces(faces)

    @staticmethod
    def ByLoft(wires, tolerance=0.0001):
        """
        Description
        ----------
        Creates a shell by lofting through the input wires

        Parameters
        ----------
        wires : list
            The input list of wires.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        topologic.Shell
            The creates shell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        if not isinstance(wires, list):
            return None
        wireList = [x for x in wires if isinstance(x, topologic.Wire)]
        faces = []
        for i in range(len(wireList)-1):
            wire1 = wireList[i]
            wire2 = wireList[i+1]
            if wire1.Type() < topologic.Edge.Type() or wire2.Type() < topologic.Edge.Type():
                raise Exception("Shell.ByLoft - Error: the input topology is not the correct type.")
            if wire1.Type() == topologic.Edge.Type():
                w1_edges = [wire1]
            else:
                w1_edges = []
                _ = wire1.Edges(None, w1_edges)
            if wire2.Type() == topologic.Edge.Type():
                w2_edges = [wire2]
            else:
                w2_edges = []
                _ = wire2.Edges(None, w2_edges)
            if len(w1_edges) != len(w2_edges):
                raise Exception("Shell.ByLoft - Error: The two wires do not have the same number of edges.")
            for j in range (len(w1_edges)):
                e1 = w1_edges[j]
                e2 = w2_edges[j]
                e3 = None
                e4 = None
                try:
                    e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()])
                except:
                    e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                    faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e4])))
                try:
                    e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                except:
                    e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()])
                    faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3])))
                if e3 and e4:
                    e5 = Edge.ByVertices([e1.StartVertex(), e2.EndVertex()])
                    faces.append(Face.ByWire(Wire.ByEdges([e1, e5, e4])))
                    faces.append(Face.ByWire(Wire.ByEdges([e2, e5, e3])))
        return Shell.ByFaces(faces, tolerance)
    
    @staticmethod
    def Edges(shell):
        """
        Description
        __________
        Returns the edges of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of edges.

        """ 
        if not isinstance(shell, topologic.Shell):
            return None
        edges = []
        _ = shell.Edges(None, edges)
        return edges

    @staticmethod
    def ExternalBoundary(shell):
        """
        Description
        ----------
        Returns the external boundary (closed wire) of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        topologic.Wire
            The external boundary (closed wire) of the input shell.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        edges = []
        _ = shell.Edges(None, edges)
        obEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(shell, faces)
            if len(faces) == 1:
                obEdges.append(anEdge)
        returnTopology = None
        try:
            returnTopology = topologic.Wire.ByEdges(obEdges)
        except:
            returnTopology = topologic.Cluster.ByTopologies(obEdges)
            returnTopology = returnTopology.SelfMerge()
        return returnTopology

    @staticmethod
    def Faces(shell):
        """
        Description
        __________
        Returns the faces of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of faces.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        faces = []
        _ = shell.Faces(None, faces)
        return faces

    @staticmethod
    def HyperbolicParaboloidRectangularDomain(origin, llVertex, lrVertex, urVertex, ulVertex, u, v, dirX, dirY, dirZ, originLocation):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        llVertex : TYPE
            DESCRIPTION.
        lrVertex : TYPE
            DESCRIPTION.
        urVertex : TYPE
            DESCRIPTION.
        ulVertex : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.

        Returns
        -------
        returnTopology : TYPE
            DESCRIPTION.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        e1 = Edge.ByVertices([llVertex, lrVertex])
        e2 = Edge.ByVertices([lrVertex, urVertex])
        e3 = Edge.ByVertices([urVertex, ulVertex])
        e4 = Edge.ByVertices([ulVertex, llVertex])
        edges = []
        for i in range(u+1):
            v1 = Edge.VertexByParameter(e1, float(i)/float(u))
            v2 = Edge.Vertex.ByParameter(e3, 1.0 - float(i)/float(u))
            edges.append(Edge.ByVertices([v1, v2]))
        faces = []
        for i in range(u):
            for j in range(v):
                v1 = Edge.VertexByParameter(edges[i], float(j)/float(v))
                v2 = Edge.VertexByParameter(edges[i], float(j+1)/float(v))
                v3 = Edge.VertexByParameter(edges[i+1], float(j+1)/float(v))
                v4 = Edge.VertexByParameter(edges[i+1], float(j)/float(v))
                faces.append(Face.ByVertices([v1, v2, v4]))
                faces.append(Face.ByVertices([v4, v2, v3]))
        returnTopology = Shell.ByFaces(faces)
        if not returnTopology:
            returnTopology = Cluster.ByTopologies(faces)
        zeroOrigin = returnTopology.CenterOfMass()
        xOffset = 0
        yOffset = 0
        zOffset = 0
        minX = min([llVertex.X(), lrVertex.X(), ulVertex.X(), urVertex.X()])
        maxX = max([llVertex.X(), lrVertex.X(), ulVertex.X(), urVertex.X()])
        minY = min([llVertex.Y(), lrVertex.Y(), ulVertex.Y(), urVertex.Y()])
        maxY = max([llVertex.Y(), lrVertex.Y(), ulVertex.Y(), urVertex.Y()])
        minZ = min([llVertex.Z(), lrVertex.Z(), ulVertex.Z(), urVertex.Z()])
        maxZ = max([llVertex.Z(), lrVertex.Z(), ulVertex.Z(), urVertex.Z()])
        if originLocation == "LowerLeft":
            xOffset = -minX
            yOffset = -minY
            zOffset = -minZ
        elif originLocation == "Bottom":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -minZ
        elif originLocation == "Center":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -(minZ + (maxZ - minZ)*0.5)
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + dirX
        y2 = 0 + dirY
        z2 = 0 + dirZ
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        returnTopology = Topology.Rotate(returnTopology, zeroOrigin, 0, 1, 0, theta)
        returnTopology = Topology.Rotate(returnTopology, zeroOrigin, 0, 0, 1, phi)
        returnTopology = Topology.Translate(returnTopology, zeroOrigin.X()+xOffset, zeroOrigin.Y()+yOffset, zeroOrigin.Z()+zOffset)
        return returnTopology
    
    @staticmethod
    def HyperbolicParaboloidCircularDomain(origin, radius, sides, rings, A, B, dirX, dirY, dirZ, originLocation):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.
        sides : TYPE
            DESCRIPTION.
        rings : TYPE
            DESCRIPTION.
        A : TYPE
            DESCRIPTION.
        B : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.

        Returns
        -------
        returnTopology : TYPE
            DESCRIPTION.

        """
        # origin = item[0]
        # radius = item[1]
        # sides = item[2]
        # rings = item[3]
        # A = item[4]
        # B = item[5]
        # dirX = item[6]
        # dirY = item[7]
        # dirZ = item[8]
        from topologicpy.Face import Face
        uOffset = float(360)/float(sides)
        vOffset = float(radius)/float(rings)
        faces = []
        for i in range(rings-1):
            r1 = radius - vOffset*i
            r2 = radius - vOffset*(i+1)
            for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x1 = math.sin(a1)*r1
                y1 = math.cos(a1)*r1
                z1 = A*x1*x1 + B*y1*y1
                x2 = math.sin(a1)*r2
                y2 = math.cos(a1)*r2
                z2 = A*x2*x2 + B*y2*y2
                x3 = math.sin(a2)*r2
                y3 = math.cos(a2)*r2
                z3 = A*x3*x3 + B*y3*y3
                x4 = math.sin(a2)*r1
                y4 = math.cos(a2)*r1
                z4 = A*x4*x4 + B*y4*y4
                v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
                v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
                v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
                v4 = topologic.Vertex.ByCoordinates(x4,y4,z4)
                f1 = Face.ByVertices([v1,v2,v4])
                f2 = Face.ByVertices([v4,v2,v3])
                faces.append(f1)
                faces.append(f2)
            a1 = math.radians(uOffset)*(sides-1)
            a2 = math.radians(360)
            x1 = math.sin(a1)*r1
            y1 = math.cos(a1)*r1
            z1 = A*x1*x1 + B*y1*y1
            x2 = math.sin(a1)*r2
            y2 = math.cos(a1)*r2
            z2 = A*x2*x2 + B*y2*y2
            x3 = math.sin(a2)*r2
            y3 = math.cos(a2)*r2
            z3 = A*x3*x3 + B*y3*y3
            x4 = math.sin(a2)*r1
            y4 = math.cos(a2)*r1
            z4 = A*x4*x4 + B*y4*y4
            v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
            v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
            v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
            v4 = topologic.Vertex.ByCoordinates(x4,y4,z4)
            f1 = Face.ByVertices([v1,v2,v4])
            f2 = Face.ByVertices([v4,v2,v3])
            faces.append(f1)
            faces.append(f2)
        # Special Case: Center triangles
        r = vOffset
        x1 = 0
        y1 = 0
        z1 = 0
        v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
        for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x2 = math.sin(a1)*r
                y2 = math.cos(a1)*r
                z2 = A*x2*x2 + B*y2*y2
                #z2 = 0
                x3 = math.sin(a2)*r
                y3 = math.cos(a2)*r
                z3 = A*x3*x3 + B*y3*y3
                #z3 = 0
                v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
                v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)
        a1 = math.radians(uOffset)*(sides-1)
        a2 = math.radians(360)
        x2 = math.sin(a1)*r
        y2 = math.cos(a1)*r
        z2 = A*x2*x2 + B*y2*y2
        x3 = math.sin(a2)*r
        y3 = math.cos(a2)*r
        z3 = A*x3*x3 + B*y3*y3
        v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
        v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
        f1 = Face.ByVertices([v2,v1,v3])
        faces.append(f1)
        returnTopology = topologic.Shell.ByFaces(faces)
        if not returnTopology:
            returnTopology = topologic.Cluster.ByTopologies(faces)
        vertices = []
        _ = returnTopology.Vertices(None, vertices)
        xList = []
        yList = []
        zList = []
        for aVertex in vertices:
            xList.append(aVertex.X())
            yList.append(aVertex.Y())
            zList.append(aVertex.Z())
        minX = min(xList)
        maxX = max(xList)
        minY = min(yList)
        maxY = max(yList)
        minZ = min(zList)
        maxZ = max(zList)
        zeroOrigin = returnTopology.CenterOfMass()
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if originLocation == "LowerLeft":
            xOffset = -minX
            yOffset = -minY
            zOffset = -minZ
        elif originLocation == "Bottom":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -minZ
        elif originLocation == "Center":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -(minZ + (maxZ - minZ)*0.5)
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + dirX
        y2 = 0 + dirY
        z2 = 0 + dirZ
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        zeroOrigin = topologic.Vertex.ByCoordinates(0,0,0)
        returnTopology = topologic.TopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 1, 0, theta)
        returnTopology = topologic.TopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 0, 1, phi)
        returnTopology = topologic.TopologyUtility.Translate(returnTopology, origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset)
        return returnTopology
    
    @staticmethod
    def InternalBoundaries(shell):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        ibEdges : TYPE
            DESCRIPTION.

        """
        edges = []
        _ = shell.Edges(None, edges)
        ibEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(shell, faces)
            if len(faces) > 1:
                ibEdges.append(anEdge)
        return ibEdges
    
    @staticmethod
    def IsClosed(shell):
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
        return shell.IsClosed()
    @staticmethod
    def Rectangle(origin=None, width=1.0, length=1.0, uSides=2, vSides=2, dirX=0, dirY=0, dirZ=1, placement="center", tolerance=0.0001):
        """
        Description
        ----------
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0,0,0).
        width : float, optional
            The width of the rectangle. The default is 1.0.
        length : float, optional
            The length of the rectangle. The default is 1.0.
        uSides : int, optional
            The number of sides along the width. The default is 2.
        vSides : int, optional
            The number of sides along the length. The default is 2.
        dirX : float, optional
            The X component of the vector representing the up direction of the rectangle. The default is 0.
        dirY : float, optional
            The Y component of the vector representing the up direction of the rectangle. The default is 0.
        dirZ : float, optional
            The Z component of the vector representing the up direction of the rectangle. The default is 1.
        placement : str, optional
            The description of the placement of the origin of the rectangle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created face.

        """
    @staticmethod
    def SelfMerge(shell, angTolerance=0.1):
        """
        Description
        ----------
        Creates a face by merging the faces of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.
        angTolerance : float, optional
            The desired angular tolerance. The default is 0.1.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        if not isinstance(shell, topologic.Shell):
            return None
        ext_boundary = Shell.ShellExternalBoundary(shell)
        if isinstance(ext_boundary, topologic.Wire):
            try:
                return topologic.Face.ByExternalBoundary(Wire.RemoveCollinearEdges(ext_boundary, angTolerance))
            except:
                try:
                    return topologic.Face.ByExternalBoundary(Wire.Planarize(Wire.RemoveCollinearEdges(ext_boundary, angTolerance)))
                except:
                    print("FaceByPlanarShell - Error: The input Wire is not planar and could not be fixed. Returning None.")
                    return None
        elif isinstance(ext_boundary, topologic.Cluster):
            wires = []
            _ = ext_boundary.Wires(None, wires)
            faces = []
            areas = []
            for aWire in wires:
                try:
                    aFace = topologic.Face.ByExternalBoundary(Wire.RemoveCollinearEdges(aWire, angTolerance))
                except:
                    aFace = topologic.Face.ByExternalBoundary(Wire.Planarize(Wire.RemoveCollinearEdges(aWire, angTolerance)))
                anArea = topologic.FaceUtility.Area(aFace)
                faces.append(aFace)
                areas.append(anArea)
            max_index = areas.index(max(areas))
            ext_boundary = faces[max_index]
            int_boundaries = list(set(faces) - set([ext_boundary]))
            int_wires = []
            for int_boundary in int_boundaries:
                temp_wires = []
                _ = int_boundary.Wires(None, temp_wires)
                int_wires.append(Wire.RemoveCollinearEdges(temp_wires[0], angTolerance))
            temp_wires = []
            _ = ext_boundary.Wires(None, temp_wires)
            ext_wire = Wire.RemoveCollinearEdges(temp_wires[0], angTolerance)
            try:
                return topologic.Face.ByExternalInternalBoundaries(ext_wire, int_wires)
            except:
                return topologic.Face.ByExternalInternalBoundaries(Wire.Planarize(ext_wire), planarizeList(int_wires))
        else:
            return None

    @staticmethod
    def TessellatedCircle(origin, radius, height, sides, dirX, dirY, dirZ, originLocation):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.
        height : TYPE
            DESCRIPTION.
        sides : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # origin = item[0]
        # radius = item[1]
        # height = item[2]
        # sides = item[3]
        # dirX = item[4]
        # dirY = item[5]
        # dirZ = item[6]
        
        def wireByVertices(vList):
            edges = []
            for i in range(len(vList)-1):
                edges.append(topologic.Edge.ByStartVertexEndVertex(vList[i], vList[i+1]))
            edges.append(topologic.Edge.ByStartVertexEndVertex(vList[-1], vList[0]))
            return topologic.Wire.ByEdges(edges)
        
        baseV = []
        topV = []
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if originLocation == "Center":
            zOffset = -height*0.5
        elif originLocation == "LowerLeft":
            xOffset = radius
            yOffset = radius

        for i in range(sides):
            angle = math.radians(360/sides)*i
            x = math.sin(angle)*radius + origin.X() + xOffset
            y = math.cos(angle)*radius + origin.Y() + yOffset
            z = origin.Z() + zOffset
            baseV.append(topologic.Vertex.ByCoordinates(x,y,z))
            topV.append(topologic.Vertex.ByCoordinates(x,y,z+height))

        baseWire = wireByVertices(baseV)
        topWire = wireByVertices(topV)
        wires = [baseWire, topWire]
        cyl = topologic.CellUtility.ByLoft(wires)
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
        cyl = topologic.TopologyUtility.Rotate(cyl, origin, 0, 1, 0, theta)
        cyl = topologic.TopologyUtility.Rotate(cyl, origin, 0, 0, 1, phi)
        return topologic.CellUtility.ByLoft(wires)
    
    @staticmethod
    def ShellTessellatedDisk(origin, radius, sides, rings, dirX, dirY, dirZ, originLocation):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.
        sides : TYPE
            DESCRIPTION.
        rings : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.

        Returns
        -------
        shell : TYPE
            DESCRIPTION.

        """
        # origin = item[0]
        # radius = item[1]
        # sides = item[2]
        # rings = item[3]
        # dirX = item[4]
        # dirY = item[5]
        # dirZ = item[6]
        from topologicpy.Face import Face
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if originLocation == "LowerLeft":
            xOffset = radius
            yOffset = radius
        
        uOffset = float(360)/float(sides)
        vOffset = float(radius)/float(rings)
        faces = []
        for i in range(rings-1):
            r1 = radius - vOffset*i
            r2 = radius - vOffset*(i+1)
            for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x1 = math.sin(a1)*r1
                y1 = math.cos(a1)*r1
                z1 = 0
                x2 = math.sin(a1)*r2
                y2 = math.cos(a1)*r2
                z2 = 0
                x3 = math.sin(a2)*r2
                y3 = math.cos(a2)*r2
                z3 = 0
                x4 = math.sin(a2)*r1
                y4 = math.cos(a2)*r1
                z4 = 0
                v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
                v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
                v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
                v4 = topologic.Vertex.ByCoordinates(x4,y4,z4)
                f1 = Face.ByVertices([v1,v2,v4])
                f2 = Face.ByVertices([v4,v2,v3])
                faces.append(f1)
                faces.append(f2)
            a1 = math.radians(uOffset)*(sides-1)
            a2 = math.radians(360)
            x1 = math.sin(a1)*r1
            y1 = math.cos(a1)*r1
            z1 = 0
            x2 = math.sin(a1)*r2
            y2 = math.cos(a1)*r2
            z2 = 0
            x3 = math.sin(a2)*r2
            y3 = math.cos(a2)*r2
            z3 = 0
            x4 = math.sin(a2)*r1
            y4 = math.cos(a2)*r1
            z4 = 0
            v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
            v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
            v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
            v4 = topologic.Vertex.ByCoordinates(x4,y4,z4)
            f1 = Face.ByVertices([v1,v2,v4])
            f2 = Face.ByVertices([v4,v2,v3])
            faces.append(f1)
            faces.append(f2)

        # Special Case: Center triangles
        r = vOffset
        x1 = 0
        y1 = 0
        z1 = 0
        v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
        for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x2 = math.sin(a1)*r
                y2 = math.cos(a1)*r
                z2 = 0
                x3 = math.sin(a2)*r
                y3 = math.cos(a2)*r
                z3 = 0
                v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
                v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)
        a1 = math.radians(uOffset)*(sides-1)
        a2 = math.radians(360)
        x2 = math.sin(a1)*r
        y2 = math.cos(a1)*r
        z2 = 0
        x3 = math.sin(a2)*r
        y3 = math.cos(a2)*r
        z3 = 0
        v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
        v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
        f1 = Face.ByVertices([v2,v1,v3])
        faces.append(f1)

        shell = topologic.Shell.ByFaces(faces)

        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + dirX
        y2 = 0 + dirY
        z2 = 0 + dirZ
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        zeroOrigin = topologic.Vertex.ByCoordinates(0,0,0)
        shell = topologic.TopologyUtility.Rotate(shell, zeroOrigin, 0, 1, 0, theta)
        shell = topologic.TopologyUtility.Rotate(shell, zeroOrigin, 0, 0, 1, phi)
        shell = topologic.TopologyUtility.Translate(shell, origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset)
        return shell

    @staticmethod
    def Vertices(shell):
        """
        Description
        __________
        Returns the vertices of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        vertices = []
        _ = shell.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Wires(shell):
        """
        Description
        __________
        Returns the wires of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        wires = []
        _ = shell.Wires(None, wires)
        return wires

    
    
    
    
    