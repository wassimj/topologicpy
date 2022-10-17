from topologicpy import topologic
import math
import time
import Wire
import Topology

class Cell(topologic.Cell):
    @staticmethod
    def CellByFaces(item, tolerance=0.0001):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        cell : TYPE
            DESCRIPTION.

        """
        cell = topologic.Cell.ByFaces(item, tolerance)
        if cell:
            return cell
        else:
            raise Exception("CellByFaces - Could not create a valid Cell")
    
    @staticmethod
    def CellByLoft(wires, tolerance=0.0001):
        """
        Parameters
        ----------
        wires : TYPE
            DESCRIPTION.
        tolerance : float, optional
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
        faces = [topologic.Face.ByExternalBoundary(wires[0])]
        faces.append(topologic.Face.ByExternalBoundary(wires[-1]))
        for i in range(len(wires)-1):
            wire1 = wires[i]
            wire2 = wires[i+1]
            w1_edges = []
            _ = wire1.Edges(None, w1_edges)
            w2_edges = []
            _ = wire2.Edges(None, w2_edges)
            if len(w1_edges) != len(w2_edges):
                raise Exception("Cell.ByLoft - Error: The two wires do not have the same number of edges.")
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
        try:
            return Cell.CellByFaces(faces, tolerance)
        except:
            return topologic.Cluster.ByTopologies(faces)
    
    @staticmethod
    def CellByShell(item):
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
        return topologic.Cell.ByShell(item)
    
    @staticmethod
    def CellByThickenedFace(face, thickness, bothSides, reverse,
                            tolerance=0.0001):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        thickness : TYPE
            DESCRIPTION.
        bothSides : TYPE
            DESCRIPTION.
        reverse : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # face = item[0]
        # thickness = abs(item[1])
        # bothSides = item[2]
        # reverse = item[3]
        # tolerance = item[4]

        if reverse == True and bothSides == False:
            thickness = -thickness
        faceNormal = topologic.FaceUtility.NormalAtParameters(face, 0.5, 0.5)
        if bothSides:
            bottomFace = topologic.TopologyUtility.Translate(face, -faceNormal[0]*0.5*thickness, -faceNormal[1]*0.5*thickness, -faceNormal[2]*0.5*thickness)
            topFace = topologic.TopologyUtility.Translate(face, faceNormal[0]*0.5*thickness, faceNormal[1]*0.5*thickness, faceNormal[2]*0.5*thickness)
        else:
            bottomFace = face
            topFace = topologic.TopologyUtility.Translate(face, faceNormal[0]*thickness, faceNormal[1]*thickness, faceNormal[2]*thickness)

        cellFaces = [bottomFace, topFace]
        bottomEdges = []
        _ = bottomFace.Edges(None, bottomEdges)
        for bottomEdge in bottomEdges:
            topEdge = topologic.TopologyUtility.Translate(bottomEdge, faceNormal[0]*thickness, faceNormal[1]*thickness, faceNormal[2]*thickness)
            sideEdge1 = topologic.Edge.ByStartVertexEndVertex(bottomEdge.StartVertex(), topEdge.StartVertex())
            sideEdge2 = topologic.Edge.ByStartVertexEndVertex(bottomEdge.EndVertex(), topEdge.EndVertex())
            cellWire = topologic.Wire.ByEdges([bottomEdge, sideEdge1, topEdge, sideEdge2])
            cellFaces.append(topologic.Face.ByExternalBoundary(cellWire))
        return topologic.Cell.ByFaces(cellFaces, tolerance)
    
    @staticmethod
    def CellCompactness(item):
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
        compactness : TYPE
            DESCRIPTION.

        """
        faces = []
        _ = item.Faces(None, faces)
        area = 0.0
        for aFace in faces:
            area = area + abs(topologic.FaceUtility.Area(aFace))
        volume = abs(topologic.CellUtility.Volume(item))
        compactness  = 0
        #From https://en.wikipedia.org/wiki/Sphericity
        if area > 0:
            compactness = (((math.pi)**(1/3))*((6*volume)**(2/3)))/area
        else:
            raise Exception("Error: Cell.Compactness: Cell surface area is not positive")
        return compactness
    
    @staticmethod
    def wireByVertices(vList):
        """
        Parameters
        ----------
        vList : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        edges = []
        for i in range(len(vList)-1):
            edges.append(topologic.Edge.ByStartVertexEndVertex(vList[i], vList[i+1]))
        edges.append(topologic.Edge.ByStartVertexEndVertex(vList[-1], vList[0]))
        return topologic.Wire.ByEdges(edges)
    
    @staticmethod
    def CellCone(origin, baseRadius, topRadius, height, sides, dirX, dirY,
                 dirZ, originLocation, tolerance=0.0001):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        baseRadius : TYPE
            DESCRIPTION.
        topRadius : TYPE
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
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        cone : TYPE
            DESCRIPTION.

        """
        # origin = item[0]
        # baseRadius = item[1]
        # topRadius = item[2]
        # height = item[3]
        # sides = item[4]
        # dirX = item[5]
        # dirY = item[6]
        # dirZ = item[7]
        # tol = item[8]
        def createCone(baseWire, topWire, baseVertex, topVertex, tolerance):
            if baseWire == None and topWire == None:
                raise Exception("Cell.Cone - Error: Both radii of the cone cannot be zero at the same time")
            elif baseWire == None:
                apex = baseVertex
                wire = topWire
            elif topWire == None:
                apex = topVertex
                wire = baseWire
            else:
                return topologic.CellUtility.ByLoft([baseWire, topWire])

            vertices = []
            _ = wire.Vertices(None,vertices)
            faces = [topologic.Face.ByExternalBoundary(wire)]
            for i in range(0, len(vertices)-1):
                w = Cell.wireByVertices([apex, vertices[i], vertices[i+1]])
                f = topologic.Face.ByExternalBoundary(w)
                faces.append(f)
            w = Cell.wireByVertices([apex, vertices[-1], vertices[0]])
            f = topologic.Face.ByExternalBoundary(w)
            faces.append(f)
            return topologic.Cell.ByFaces(faces, tolerance)
        
        baseV = []
        topV = []
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if originLocation == "Center":
            zOffset = -height*0.5
        elif originLocation == "LowerLeft":
            xOffset = max(baseRadius, topRadius)
            yOffset = max(baseRadius, topRadius)

        baseZ = origin.Z() + zOffset
        topZ = origin.Z() + zOffset + height
        for i in range(sides):
            angle = math.radians(360/sides)*i
            if baseRadius > 0:
                baseX = math.sin(angle)*baseRadius + origin.X() + xOffset
                baseY = math.cos(angle)*baseRadius + origin.Y() + yOffset
                baseZ = origin.Z() + zOffset
                baseV.append(topologic.Vertex.ByCoordinates(baseX,baseY,baseZ))
            if topRadius > 0:
                topX = math.sin(angle)*topRadius + origin.X() + xOffset
                topY = math.cos(angle)*topRadius + origin.Y() + yOffset
                topV.append(topologic.Vertex.ByCoordinates(topX,topY,topZ))

        if baseRadius > 0:
            baseWire = Cell.wireByVertices(baseV)
        else:
            baseWire = None
        if topRadius > 0:
            topWire = Cell.wireByVertices(topV)
        else:
            topWire = None
        baseVertex = topologic.Vertex.ByCoordinates(origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset)
        topVertex = topologic.Vertex.ByCoordinates(origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset+height)
        cone = createCone(baseWire, topWire, baseVertex, topVertex, tolerance)
        if cone == None:
            raise Exception("Cell.Cone - Error: Could not create cone")
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
        cone = topologic.TopologyUtility.Rotate(cone, origin, 0, 1, 0, theta)
        cone = topologic.TopologyUtility.Rotate(cone, origin, 0, 0, 1, phi)
        return cone
    
    @staticmethod
    def CellCylinder(origin, radius, height, uSides, vSides, dirX, dirY, dirZ,
                     originLocation, tolerance=0.0001):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.
        height : TYPE
            DESCRIPTION.
        uSides : TYPE
            DESCRIPTION.
        vSides : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        cyl : TYPE
            DESCRIPTION.

        """
        # origin, \
        # radius, \
        # height, \
        # uSides, \
        # vSides, \
        # dirX, \
        # dirY, \
        # dirZ, \
        # tolerance = item
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

        for i in range(uSides):
            angle = math.radians(360/uSides)*i
            x = math.sin(angle)*radius + origin.X() + xOffset
            y = math.cos(angle)*radius + origin.Y() + yOffset
            z = origin.Z() + zOffset
            baseV.append(topologic.Vertex.ByCoordinates(x,y,z))
            topV.append(topologic.Vertex.ByCoordinates(x,y,z+height))

        baseWire = Cell.wireByVertices(baseV)
        topologies = []
        for i in range(vSides+1):
            topologies.append(topologic.TopologyUtility.Translate(baseWire, 0, 0, height/float(vSides)*i))
        cyl = Cell.CellByLoft(topologies, tolerance)

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
        return cyl
    
    @staticmethod
    def CellExternalBoundary(item):
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
        return item.ExternalBoundary()
    
    @staticmethod
    def CellHyperboloid(origin, baseRadius, topRadius, height, sides, dirX,
                        dirY, dirZ, twist, originLocation, tolerance=0.0001):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        baseRadius : TYPE
            DESCRIPTION.
        topRadius : TYPE
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
        twist : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.
        tolerance : float, optional
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
        # origin = item[0]
        # baseRadius = item[1]
        # topRadius = item[2]
        # height = item[3]
        # sides = item[4]
        # dirX = item[5]
        # dirY = item[6]
        # dirZ = item[7]
        # twist = item[8]
        # tol = item[9]
        
        def createHyperboloid(baseVertices, topVertices, tolerance):
            baseWire = Cell.wireByVertices(baseVertices)
            topWire = Cell.wireByVertices(topVertices)
            baseFace = topologic.Face.ByExternalBoundary(baseWire)
            topFace = topologic.Face.ByExternalBoundary(topWire)
            faces = [baseFace, topFace]
            for i in range(0, len(baseVertices)-1):
                w = Cell.wireByVertices([baseVertices[i], topVertices[i], topVertices[i+1]])
                f = topologic.Face.ByExternalBoundary(w)
                faces.append(f)
                w = Cell.wireByVertices([baseVertices[i+1], baseVertices[i], topVertices[i+1]])
                f = topologic.Face.ByExternalBoundary(w)
                faces.append(f)
            w = Cell.wireByVertices([baseVertices[-1], topVertices[-1], topVertices[0]])
            f = topologic.Face.ByExternalBoundary(w)
            faces.append(f)
            w = Cell.wireByVertices([baseVertices[0], baseVertices[-1], topVertices[0]])
            f = topologic.Face.ByExternalBoundary(w)
            faces.append(f)
            returnTopology = topologic.Cell.ByFaces(faces, tolerance)
            if returnTopology == None:
                returnTopology = topologic.Cluster.ByTopologies(faces)
            return returnTopology
        
        baseV = []
        topV = []
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if originLocation == "Center":
            zOffset = -height*0.5
        elif originLocation == "LowerLeft":
            xOffset = max(baseRadius, topRadius)
            yOffset = max(baseRadius, topRadius)
        baseZ = origin.Z() + zOffset
        topZ = origin.Z() + zOffset + height
        for i in range(sides):
            angle = math.radians(360/sides)*i
            if baseRadius > 0:
                baseX = math.sin(angle+math.radians(twist))*baseRadius + origin.X() + xOffset
                baseY = math.cos(angle+math.radians(twist))*baseRadius + origin.Y() + yOffset
                baseZ = origin.Z() + zOffset
                baseV.append(topologic.Vertex.ByCoordinates(baseX,baseY,baseZ))
            if topRadius > 0:
                topX = math.sin(angle-math.radians(twist))*topRadius + origin.X() + xOffset
                topY = math.cos(angle-math.radians(twist))*topRadius + origin.Y() + yOffset
                topV.append(topologic.Vertex.ByCoordinates(topX,topY,topZ))

        hyperboloid = createHyperboloid(baseV, topV, tolerance)
        if hyperboloid == None:
            raise Exception("Cell.Hyperboloid - Error: Could not create hyperboloid")
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
        hyperboloid = topologic.TopologyUtility.Rotate(hyperboloid, origin, 0, 1, 0, theta)
        hyperboloid = topologic.TopologyUtility.Rotate(hyperboloid, origin, 0, 0, 1, phi)
        return hyperboloid
    
    @staticmethod
    def CellInternalBoundaries(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        shells : TYPE
            DESCRIPTION.

        """
        shells = []
        _ = item.InternalBoundaries(shells)
        return shells
    
    @staticmethod
    def CellInternalVertex(item, tolerance=0.0001):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        vert : TYPE
            DESCRIPTION.

        """
        vert = None
        if item.Type() == topologic.Cell.Type():
            vert = topologic.CellUtility.InternalVertex(item, tolerance)
        return vert
    
    @staticmethod
    def CellIsInside(topology, vertex, tolerance=0.0001):
        """
        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        vertex : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        status : TYPE
            DESCRIPTION.

        """
        # topology = item[0]
        # vertex = item[1]
        # tolerance = item[2]
        status = False
        if topology.Type() == topologic.Cell.Type():
            status = (topologic.CellUtility.Contains(topology, vertex, tolerance) == 0)
        return status
    
    @staticmethod
    def CellPipe(edge, radius, sides, startOffset, endOffset, endcapA,
                 endcapB):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.
        sides : TYPE
            DESCRIPTION.
        startOffset : TYPE
            DESCRIPTION.
        endOffset : TYPE
            DESCRIPTION.
        endcapA : TYPE
            DESCRIPTION.
        endcapB : TYPE
            DESCRIPTION.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
        # edge = item[0]
        # radius = item[1]
        # sides = item[2]
        # startOffset = item[3]
        # endOffset = item[4]
        # endcapA = item[5]
        # endcapB = item[6]

        length = topologic.EdgeUtility.Length(edge)
        origin = edge.StartVertex()
        startU = startOffset / length
        endU = 1.0 - (endOffset / length)
        sv = topologic.EdgeUtility.PointAtParameter(edge, startU)
        ev = topologic.EdgeUtility.PointAtParameter(edge, endU)
        new_edge = topologic.Edge.ByStartVertexEndVertex(sv, ev)
        x1 = sv.X()
        y1 = sv.Y()
        z1 = sv.Z()
        x2 = ev.X()
        y2 = ev.Y()
        z2 = ev.Z()
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        baseV = []
        topV = []

        for i in range(sides):
            angle = math.radians(360/sides)*i
            x = math.sin(angle)*radius + sv.X()
            y = math.cos(angle)*radius + sv.Y()
            z = sv.Z()
            baseV.append(topologic.Vertex.ByCoordinates(x,y,z))
            topV.append(topologic.Vertex.ByCoordinates(x,y,z+dist))

        baseWire = Cell.wireByVertices(baseV)
        topWire = Cell.wireByVertices(topV)
        wires = [baseWire, topWire]
        cyl = topologic.CellUtility.ByLoft(wires)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        cyl = topologic.TopologyUtility.Rotate(cyl, sv, 0, 1, 0, theta)
        cyl = topologic.TopologyUtility.Rotate(cyl, sv, 0, 0, 1, phi)
        zzz = topologic.Vertex.ByCoordinates(0,0,0)
        returnList = [cyl]
        if endcapA:
            origin = edge.StartVertex()
            x1 = origin.X()
            y1 = origin.Y()
            z1 = origin.Z()
            x2 = edge.EndVertex().X()
            y2 = edge.EndVertex().Y()
            z2 = edge.EndVertex().Z()
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1    
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
            if dist < 0.0001:
                theta = 0
            else:
                theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
            endcapA = topologic.Topology.DeepCopy(endcapA)
            endcapA = topologic.TopologyUtility.Rotate(endcapA, zzz, 0, 1, 0, theta)
            endcapA = topologic.TopologyUtility.Rotate(endcapA, zzz, 0, 0, 1, phi + 180)
            endcapA = topologic.TopologyUtility.Translate(endcapA, origin.X(), origin.Y(), origin.Z())
            returnList.append(endcapA)
        if endcapB:
            origin = edge.EndVertex()
            x1 = origin.X()
            y1 = origin.Y()
            z1 = origin.Z()
            x2 = edge.StartVertex().X()
            y2 = edge.StartVertex().Y()
            z2 = edge.StartVertex().Z()
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1    
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
            if dist < 0.0001:
                theta = 0
            else:
                theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
            endcapB = topologic.Topology.DeepCopy(endcapB)
            endcapB = topologic.TopologyUtility.Rotate(endcapB, zzz, 0, 1, 0, theta)
            endcapB = topologic.TopologyUtility.Rotate(endcapB, zzz, 0, 0, 1, phi + 180)
            endcapB = topologic.TopologyUtility.Translate(endcapB, origin.X(), origin.Y(), origin.Z())
            returnList.append(endcapB)
        return returnList
    
    @staticmethod
    def CellPrism(origin, width, length, height, uSides, vSides, wSides, dirX,
                  dirY, dirZ, placement):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        width : TYPE
            DESCRIPTION.
        length : TYPE
            DESCRIPTION.
        height : TYPE
            DESCRIPTION.
        uSides : TYPE
            DESCRIPTION.
        vSides : TYPE
            DESCRIPTION.
        wSides : TYPE
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
        prism : TYPE
            DESCRIPTION.

        """
        # origin, \
        # width, \
        # length, \
        # height, \
        # uSides, \
        # vSides, \
        # wSides, \
        # dirX, \
        # dirY, \
        # dirZ, \
        # placement = item
        
        def sliceCell(cell, width, length, height, uSides, vSides, wSides):
            origin = cell.Centroid()
            shells = []
            _ = cell.Shells(None, shells)
            shell = shells[0]
            wRect = Wire.WireRectangle(origin, width*1.2, length*1.2, 0, 0, 1, "Center")
            sliceFaces = []
            for i in range(1, wSides):
                sliceFaces.append(topologic.TopologyUtility.Translate(topologic.Face.ByExternalBoundary(wRect), 0, 0, height/wSides*i - height*0.5))
            uRect = Wire.WireRectangle(origin, height*1.2, length*1.2, 1, 0, 0, "Center")
            for i in range(1, uSides):
                sliceFaces.append(topologic.TopologyUtility.Translate(topologic.Face.ByExternalBoundary(uRect), width/uSides*i - width*0.5, 0, 0))
            vRect = Wire.WireRectangle(origin, height*1.2, width*1.2, 0, 1, 0, "Center")
            for i in range(1, vSides):
                sliceFaces.append(topologic.TopologyUtility.Translate(topologic.Face.ByExternalBoundary(vRect), 0, length/vSides*i - length*0.5, 0))
            sliceCluster = topologic.Cluster.ByTopologies(sliceFaces)
            shell = shell.Slice(sliceCluster, False)
            return topologic.Cell.ByShell(shell)
        
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement == "Center":
            zOffset = -height*0.5
        elif placement == "LowerLeft":
            xOffset = width*0.5
            yOffset = length*0.5

        vb1 = topologic.Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z()+zOffset)
        vb2 = topologic.Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z()+zOffset)
        vb3 = topologic.Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z()+zOffset)
        vb4 = topologic.Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z()+zOffset)

        vt1 = topologic.Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z()+height+zOffset)
        vt2 = topologic.Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z()+height+zOffset)
        vt3 = topologic.Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z()+height+zOffset)
        vt4 = topologic.Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z()+height+zOffset)
        baseWire = Cell.wireByVertices([vb1, vb2, vb3, vb4])
        topWire = Cell.wireByVertices([vt1, vt2, vt3, vt4])
        wires = [baseWire, topWire]
        prism =  topologic.CellUtility.ByLoft(wires)
        prism = sliceCell(prism, width, length, height, uSides, vSides, wSides)
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
        prism = topologic.TopologyUtility.Rotate(prism, origin, 0, 1, 0, theta)
        prism = topologic.TopologyUtility.Rotate(prism, origin, 0, 0, 1, phi)
        return prism
    
    @staticmethod
    def CellSets(inputCells, superCells, tolerance=0.0001):
        """
        Parameters
        ----------
        inputCells : TYPE
            DESCRIPTION.
        superCells : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        sets : TYPE
            DESCRIPTION.

        """
        if len(superCells) == 0:
            cluster = inputCells[0]
            for i in range(1, len(inputCells)):
                oldCluster = cluster
                cluster = cluster.Union(inputCells[i])
                del oldCluster
            superCells = []
            _ = cluster.Cells(None, superCells)
        unused = []
        for i in range(len(inputCells)):
            unused.append(True)
        sets = []
        for i in range(len(superCells)):
            sets.append([])
        for i in range(len(inputCells)):
            if unused[i]:
                iv = topologic.CellUtility.InternalVertex(inputCells[i], tolerance)
                for j in range(len(superCells)):
                    if (topologic.CellUtility.Contains(superCells[j], iv, tolerance) == 0):
                        sets[j].append(inputCells[i])
                        unused[i] = False
        return sets
    
    @staticmethod
    def CellSphere(origin, radius, uSides, vSides, dirX, dirY, dirZ,
                   originLocation, tolerance=0.0001):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        radius : TYPE
            DESCRIPTION.
        uSides : TYPE
            DESCRIPTION.
        vSides : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        s : TYPE
            DESCRIPTION.

        """
        # origin, \
        # radius, \
        # uSides, \
        # vSides, \
        # dirX, \
        # dirY, \
        # dirZ, \
        # tolerance = item

        c = Wire.WireCircle(origin, radius, vSides, 90, 270, True, 0, 1, 0, "Center")
        c = topologic.Face.ByExternalBoundary(c)
        s = Topology.TopologySpin(c, origin, 0, 0, 1, 360, uSides, tolerance)
        if s.Type() == topologic.CellComplex.Type():
            s = s.ExternalBoundary()
        if s.Type() == topologic.Shell.Type():
            s = topologic.Cell.ByShell(s)
        if originLocation == "Bottom":
            s = topologic.TopologyUtility.Translate(s, 0, 0, radius)
        elif originLocation == "LowerLeft":
            s = topologic.TopologyUtility.Translate(s, radius, radius, radius)
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
        s = topologic.TopologyUtility.Rotate(s, origin, 0, 1, 0, theta)
        s = topologic.TopologyUtility.Rotate(s, origin, 0, 0, 1, phi)
        return s
    
    @staticmethod
    def CellSuperCells(inputCells, tolerance=0.0001):
        """
        Parameters
        ----------
        inputCells : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        superCells : TYPE
            DESCRIPTION.

        """
        
        def gcClear(item):
            gc = topologic.GlobalCluster.GetInstance()
            subTopologies = []
            _ = gc.SubTopologies(subTopologies)
            for aSubTopology in subTopologies:
                if not (topologic.Topology.IsSame(item, aSubTopology)):
                    gc.RemoveTopology(aSubTopology)
            return
        
        cluster = inputCells[0]
        start = time.time()
        for i in range(1, len(inputCells)):
            oldCluster = cluster
            cluster = cluster.Union(inputCells[i])
            del oldCluster
            if i % 50 == 0:
                end = time.time()
                print("Operation consumed "+str(round(end - start,2))+" seconds")
                start = time.time()
                print(i,"Clearing GlobalCluster")
                gcClear(cluster)
        superCells = []
        _ = cluster.Cells(superCells)
        return superCells
    
    @staticmethod
    def CellSurfaceArea(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        area : TYPE
            DESCRIPTION.

        """
        faces = []
        _ = item.Faces(None, faces)
        area = 0.0
        for aFace in faces:
            area = area + topologic.FaceUtility.Area(aFace)
        return area
    
    @staticmethod
    def CellTorus(origin, majorRadius, minorRadius, uSides, vSides, dirX, dirY,
                  dirZ, originLocation, tolerance=0.0001):
        """
        Parameters
        ----------
        origin : TYPE
            DESCRIPTION.
        majorRadius : TYPE
            DESCRIPTION.
        minorRadius : TYPE
            DESCRIPTION.
        uSides : TYPE
            DESCRIPTION.
        vSides : TYPE
            DESCRIPTION.
        dirX : TYPE
            DESCRIPTION.
        dirY : TYPE
            DESCRIPTION.
        dirZ : TYPE
            DESCRIPTION.
        originLocation : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        s : TYPE
            DESCRIPTION.

        """
        # origin, \
        # majorRadius, \
        # minorRadius, \
        # uSides, \
        # vSides, \
        # dirX, \
        # dirY, \
        # dirZ, \
        # tolerance = item

        c = Wire.WireCircle.processItem(origin, minorRadius, vSides, 0, 360, False, 0, 1, 0, "Center")
        c = topologic.TopologyUtility.Translate(c, majorRadius, 0, 0)
        s = Topology.TopologySpin(c, origin, 0, 0, 1, 360, uSides, tolerance)
        if s.Type() == topologic.Shell.Type():
            s = topologic.Cell.ByShell(s)
        if originLocation == "Bottom":
            s = topologic.TopologyUtility.Translate(s, 0, 0, radius)
        elif originLocation == "LowerLeft":
            s = topologic.TopologyUtility.Translate(s, radius, radius, radius)
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
        s = topologic.TopologyUtility.Rotate(s, origin, 0, 1, 0, theta)
        s = topologic.TopologyUtility.Rotate(s, origin, 0, 0, 1, phi)
        return s
    
    @staticmethod
    def CellVolume(item):
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
        return topologic.CellUtility.Volume(item)


