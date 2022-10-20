import topologic
import math
import warnings

class Shell(topologic.Shell):
    @staticmethod
    def faceByVertices(vList):
        """
        Parameters
        ----------
        vList : TYPE
            DESCRIPTION.

        Returns
        -------
        f : TYPE
            DESCRIPTION.

        """
        edges = []
        for i in range(len(vList)-1):
            edges.append(topologic.Edge.ByStartVertexEndVertex(vList[i], vList[i+1]))
        edges.append(topologic.Edge.ByStartVertexEndVertex(vList[-1], vList[0]))
        w = topologic.Wire.ByEdges(edges)
        f = topologic.Face.ByExternalBoundary(w)
        return f
    
    @staticmethod
    def ShellByFaces(faces, tolerance=0.0001):
        """
        Parameters
        ----------
        faces : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # faces, tol = item
        shell = topologic.Shell.ByFaces(faces, tolerance)
        if not shell:
            warnings.warn("Warning: Default Shell.ByFaces method failed. Attempting to Merge the Faces.", UserWarning)
            result = faces[0]
            remainder = faces[1:]
            cluster = topologic.Cluster.ByTopologies(remainder, False)
            result = result.Merge(cluster, False)
            if result.Type() != 16: #16 is the type of a Shell
                warnings.warn("Warning: Input Faces do not form a Shell", UserWarning)
                if result.Type() > 16:
                    returnShells = []
                    _ = result.Shells(None, returnShells)
                    return returnShells
                else:
                    return None
        else:
            return shell
    
    @staticmethod
    def ShellByLoft(wires, tolerance=0.0001):
        """
        Parameters
        ----------
        wires : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # wires, tolerance = item
        faces = []
        for i in range(len(wires)-1):
            wire1 = wires[i]
            wire2 = wires[i+1]
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
                    e3 = topologic.Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.StartVertex())
                except:
                    e4 = topologic.Edge.ByStartVertexEndVertex(e1.EndVertex(), e2.EndVertex())
                    faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e1, e2, e4])))
                try:
                    e4 = topologic.Edge.ByStartVertexEndVertex(e1.EndVertex(), e2.EndVertex())
                except:
                    e3 = topologic.Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.StartVertex())
                    faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e1, e2, e3])))
                if e3 and e4:
                    e5 = topologic.Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.EndVertex())
                    faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e1, e5, e4])))
                    faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e2, e5, e3])))
        return Shell.ShellByFaces(faces, tolerance)
    
    @staticmethod
    def ShellExternalBoundary(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        returnTopology : TYPE
            DESCRIPTION.

        """
        edges = []
        _ = item.Edges(None, edges)
        obEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(item, faces)
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
    def ShellHyperbolicParaboloidRectangularDomain(origin, llVertex, lrVertex, urVertex, ulVertex, u, v, dirX, dirY, dirZ, originLocation):
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
        # origin, llVertex, lrVertex, urVertex, ulVertex, u, v, dirX, dirY, dirZ = item
        e1 = topologic.Edge.ByStartVertexEndVertex(llVertex, lrVertex)
        e2 = topologic.Edge.ByStartVertexEndVertex(lrVertex, urVertex)
        e3 = topologic.Edge.ByStartVertexEndVertex(urVertex, ulVertex)
        e4 = topologic.Edge.ByStartVertexEndVertex(ulVertex, llVertex)
        edges = []
        for i in range(u+1):
            print("I", i)
            v1 = topologic.EdgeUtility.PointAtParameter(e1, float(i)/float(u))
            v2 = topologic.EdgeUtility.PointAtParameter(e3, 1.0 - float(i)/float(u))
            edges.append(topologic.Edge.ByStartVertexEndVertex(v1, v2))
        faces = []
        for i in range(u):
            for j in range(v):
                v1 = topologic.EdgeUtility.PointAtParameter(edges[i], float(j)/float(v))
                v2 = topologic.EdgeUtility.PointAtParameter(edges[i], float(j+1)/float(v))
                v3 = topologic.EdgeUtility.PointAtParameter(edges[i+1], float(j+1)/float(v))
                v4 = topologic.EdgeUtility.PointAtParameter(edges[i+1], float(j)/float(v))
                faces.append(Shell.faceByVertices([v1, v2, v4]))
                faces.append(Shell.faceByVertices([v4, v2, v3]))
        returnTopology = topologic.Shell.ByFaces(faces)
        if not returnTopology:
            returnTopology = topologic.Cluster.ByTopologies(faces)
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
        returnTopology = topologic.TopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 1, 0, theta)
        returnTopology = topologic.TopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 0, 1, phi)
        returnTopology = topologic.TopologyUtility.Translate(returnTopology, zeroOrigin.X()+xOffset, zeroOrigin.Y()+yOffset, zeroOrigin.Z()+zOffset)
        return returnTopology
    
    @staticmethod
    def ShellHyperbolicParaboloidCircularDomain(origin, radius, sides, rings, A, B, dirX, dirY, dirZ, originLocation):
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
                f1 = Shell.faceByVertices([v1,v2,v4])
                f2 = Shell.faceByVertices([v4,v2,v3])
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
            f1 = Shell.faceByVertices([v1,v2,v4])
            f2 = Shell.faceByVertices([v4,v2,v3])
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
                f1 = Shell.faceByVertices([v2,v1,v3])
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
        f1 = Shell.faceByVertices([v2,v1,v3])
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
    def ShellInternalBoundaries(item):
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
        _ = item.Edges(None, edges)
        ibEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(item, faces)
            if len(faces) > 1:
                ibEdges.append(anEdge)
        return ibEdges
    
    @staticmethod
    def ShellIsClosed(item):
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
        return item.IsClosed()
    
    @staticmethod
    def ShellTessellatedCircle(origin, radius, height, sides, dirX, dirY, dirZ, originLocation):
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
                f1 = Shell.faceByVertices([v1,v2,v4])
                f2 = Shell.faceByVertices([v4,v2,v3])
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
            f1 = Shell.faceByVertices([v1,v2,v4])
            f2 = Shell.faceByVertices([v4,v2,v3])
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
                f1 = Shell.faceByVertices([v2,v1,v3])
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
        f1 = Shell.faceByVertices([v2,v1,v3])
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

    
    
    
    
    