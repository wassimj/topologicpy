import topologicpy
import topologic
from topologicpy.Vector import Vector
from topologicpy.Wire import Wire
import math

class Face(topologic.Face):
    @staticmethod
    def AddInternalBoundaries(face, wires):
        """
        Adds internal boundaries (closed wires) to the input face. Internal boundaries are considered holes in the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        wires : list
            The input list of internal boundaries (closed wires).

        Returns
        -------
        topologic.Face
            The created face with internal boundaries added to it.

        """
        if not face:
            return None
        if not isinstance(face, topologic.Face):
            return None
        if not wires:
            return face
        if not isinstance(wires, list):
            return face
        wireList = [w for w in wires if isinstance(w, topologic.Wire)]
        if len(wireList) < 1:
            return face
        faceeb = face.ExternalBoundary()
        faceibList = []
        _ = face.InternalBoundaries(faceibList)
        for wire in wires:
            faceibList.append(wire)
        return topologic.Face.ByExternalInternalBoundaries(faceeb, faceibList)

    @staticmethod
    def AddInternalBoundariesCluster(face, cluster):
        """
        Adds the input cluster of internal boundaries (closed wires) to the input face. Internal boundaries are considered holes in the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        cluster : topologic.Cluster
            The input cluster of internal boundaries (topologic wires).

        Returns
        -------
        topologic.Face
            The created face with internal boundaries added to it.

        """
        if not face:
            return None
        if not isinstance(face, topologic.Face):
            return None
        if not cluster:
            return face
        if not isinstance(cluster, topologic.Cluster):
            return face
        wires = []
        _ = cluster.Wires(None, wires)
        return Face.AddInternalBoundaries(face, wires)
    
    @staticmethod
    def Angle(faceA, faceB, mantissa=4):
        """
        Returns the angle in degrees between the two input faces.

        Parameters
        ----------
        faceA : topologic.Face
            The first input face.
        faceB : topologic.Face
            The second input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The angle in degrees between the two input faces.

        """
        from topologicpy.Vector import Vector
        if not faceA or not isinstance(faceA, topologic.Face):
            return None
        if not faceB or not isinstance(faceB, topologic.Face):
            return None
        dirA = Face.NormalAtParameters(faceA, 0.5, 0.5, "xyz", 3)
        dirB = Face.NormalAtParameters(faceB, 0.5, 0.5, "xyz", 3)
        return round((Vector.Angle(dirA, dirB)), mantissa)

    @staticmethod
    def BoundingRectangle(topology, optimize=0):
        """
        Returns a face representing a bounding rectangle of the input topology. The returned face contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting face will become axis-aligned.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding rectangle. The default is 0.
        
        Returns
        -------
        topologic.Face
            The bounding rectangle of the input topology.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        def bb(topology):
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
        vertices = Topology.SubTopologies(topology, subTopologyType="vertex")
        topology = Cluster.ByTopologies(vertices)
        boundingBox = bb(topology)
        minX = boundingBox[0]
        minY = boundingBox[1]
        maxX = boundingBox[2]
        maxY = boundingBox[3]
        w = abs(maxX - minX)
        l = abs(maxY - minY)
        best_area = l*w
        orig_area = best_area
        best_z = 0
        best_bb = boundingBox
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
                    minX, minY, maxX, maxY = bb(t)
                    w = abs(maxX - minX)
                    l = abs(maxY - minY)
                    area = l*w
                    if area < orig_area*factor:
                        best_area = area
                        best_z = z
                        best_bb = [minX, minY, maxX, maxY]
                        flag = True
                        break
                    if area < best_area:
                        best_area = area
                        best_z = z
                        best_bb = [minX, minY, maxX, maxY]
                        
        else:
            best_bb = boundingBox

        minX, minY, maxX, maxY = best_bb
        vb1 = topologic.Vertex.ByCoordinates(minX, minY, 0)
        vb2 = topologic.Vertex.ByCoordinates(maxX, minY, 0)
        vb3 = topologic.Vertex.ByCoordinates(maxX, maxY, 0)
        vb4 = topologic.Vertex.ByCoordinates(minX, maxY, 0)

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire)
        baseFace = Topology.Rotate(baseFace, origin=origin, x=0,y=0,z=1, degree=-best_z)
        dictionary = Dictionary.ByKeysValues(["zrot"], [best_z])
        baseFace = Topology.SetDictionary(baseFace, dictionary)
        return baseFace

    @staticmethod
    def CompassAngle(face, north: list = None, mantissa: int = 4):
        """
        Returns the horizontal compass angle in degrees between the normal vector of the input face and the input vector. The angle is measured in counter-clockwise fashion. Only the first two elements of the vectors are considered.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        north : list , optional
            The second vector representing the north direction. The default is the positive YAxis ([0,1,0]).
        mantissa : int, optional
            The length of the desired mantissa. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        float
            The horizontal compass angle in degrees between the direction of the face and the second input vector.

        """
        from topologicpy.Vector import Vector
        if not isinstance(face, topologic.Face):
            return None
        if not north:
            north = Vector.North()
        dirA = Face.NormalAtParameters(face,mantissa=mantissa)
        return Vector.CompassAngle(vectorA=dirA, vectorB=north, mantissa=mantissa)

    @staticmethod
    def Area(face, mantissa=4):
        """
        Returns the area of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The area of the input face.

        """
        if not isinstance(face, topologic.Face):
            return None
        area = None
        try:
            area = round(topologic.FaceUtility.Area(face), mantissa)
        except:
            area = None
        return area

    @staticmethod
    def BoundingFace(face):
        """
        Returns the bounding face of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        topologic.Face
            The bounding face of the input face.

        """
        if not isinstance(face, topologic.Face):
            return None
        bfv1 = topologic.FaceUtility.VertexAtParameters(face,0,0)
        bfv2 = topologic.FaceUtility.VertexAtParameters(face,1,0)
        bfv3 = topologic.FaceUtility.VertexAtParameters(face,1,1)
        bfv4 = topologic.FaceUtility.VertexAtParameters(face,0,1)
        bfe1 = topologic.Edge.ByStartVertexEndVertex(bfv1,bfv2)
        bfe2 = topologic.Edge.ByStartVertexEndVertex(bfv2,bfv3)
        bfe3 = topologic.Edge.ByStartVertexEndVertex(bfv3,bfv4)
        bfe4 = topologic.Edge.ByStartVertexEndVertex(bfv4,bfv1)
        bfw1 = topologic.Wire.ByEdges([bfe1,bfe2,bfe3,bfe4])
        return topologic.Face.ByExternalBoundary(bfw1)
    
    @staticmethod
    def ByEdges(edges):
        """
        Creates a face from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.

        Returns
        -------
        face : topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        wire = Wire.ByEdges(edges)
        if not wire:
            return None
        if not isinstance(wire, topologic.Wire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def ByEdgesCluster(cluster):
        """
        Creates a face from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of edges.

        Returns
        -------
        face : topologic.Face
            The created face.

        """
        from topologicpy.Cluster import Cluster
        if not isinstance(cluster, topologic.Cluster):
            return None
        edges = Cluster.Edges(cluster)
        return Face.ByEdges(edges)

    @staticmethod
    def ByOffset(face, offset=1, miter=False, miterThreshold=None):
        from topologicpy.Wire import Wire

        external_boundary = face.ExternalBoundary()
        internal_boundaries = []
        _ = face.InternalBoundaries(internal_boundaries)
        offset_external_boundary = Wire.ByOffset(external_boundary, offset=offset, miter=miter, miterThreshold=miterThreshold)
        offset_internal_boundaries = []
        for internal_boundary in internal_boundaries:
            offset_internal_boundaries.append(Wire.ByOffset(internal_boundary, -offset, miter=miter, miterThreshold=miterThreshold))
        return Face.ByWires(offset_external_boundary, offset_internal_boundaries)
    
    @staticmethod
    def ByShell(shell, angTolerance=0.1):
        """
        Creates a face by merging the faces of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        
        ext_boundary = Shell.ExternalBoundary(shell)
        ext_boundary = Wire.RemoveCollinearEdges(ext_boundary, angTolerance)
        if not Topology.IsPlanar(ext_boundary):
            ext_boundary = Wire.Planarize(ext_boundary)

        if isinstance(ext_boundary, topologic.Wire):
            try:
                return topologic.Face.ByExternalBoundary(Wire.RemoveCollinearEdges(ext_boundary, angTolerance))
            except:
                try:
                    w = Wire.Planarize(ext_boundary)
                    f = Face.ByWire(w)
                    return f
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
    def ByVertices(vertices):
        """
        Creates a face from the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire

        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, topologic.Vertex)]
        if len(vertexList) < 3:
            return None
        
        w = Wire.ByVertices(vertexList)
        f = Face.ByExternalBoundary(w)
        return f
        """
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
        v1 = vertices[-1]
        v2 = vertices[0]
        try:
            e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
            if e:
                edges.append(e)
        except:
            pass
        if len(edges) > 0:
            return topologic.Face.ByExternalBoundary(Topology.SelfMerge(topologic.Cluster.ByTopologies(edges, False)))
        else:
            return None
        """
    def ByVerticesCluster(cluster):
        """
        Creates a face from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of vertices.

        Returns
        -------
        topologic.Face
            The crearted face.

        """
        from topologicpy.Cluster import Cluster
        if not isinstance(cluster, topologic.Cluster):
            return None
        vertices = Cluster.Vertices(cluster)
        return Face.ByVertices(vertices)

    @staticmethod
    def ByWire(wire):
        """
        Creates a face from the input closed wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        import random
        if not isinstance(wire, topologic.Wire):
            return None
        if not Wire.IsClosed(wire):
            return None
        f = topologic.Face.ByExternalBoundary(wire)
        area = Face.Area(f)
        if area < 0:
            wire = Wire.Invert(wire)
            f = topologic.Face.ByExternalBoundary(wire)
            return f
        else:
            return f

    @staticmethod
    def ByWires(externalBoundary, internalBoundaries=[]):
        """
        Creates a face from the input external boundary (closed wire) and the input list of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : topologic.Wire
            The input external boundary.
        internalBoundaries : list , optional
            The input list of internal boundaries (closed wires). The default is an empty list.

        Returns
        -------
        topologic.Face
            The created face.

        """
        if not isinstance(externalBoundary, topologic.Wire):
            return None
        if not Wire.IsClosed(externalBoundary):
            return None
        ibList = [x for x in internalBoundaries if isinstance(x, topologic.Wire) and Wire.IsClosed(x)]
        return topologic.Face.ByExternalInternalBoundaries(externalBoundary, ibList)

    @staticmethod
    def ByWiresCluster(externalBoundary, internalBoundariesCluster=None):
        """
        Creates a face from the input external boundary (closed wire) and the input cluster of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : topologic.Wire
            The input external boundary (closed wire).
        internalBoundariesCluster : topologic.Cluster
            The input cluster of internal boundaries (closed wires). The default is None.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        if not isinstance(externalBoundary, topologic.Wire):
            return None
        if not Wire.IsClosed(externalBoundary):
            return None
        if not internalBoundariesCluster:
            internalBoundaries = []
        elif not isinstance(internalBoundariesCluster, topologic.Cluster):
            return None
        else:
            internalBoundaries = Cluster.Wires(internalBoundariesCluster)
        return Face.ByWires(externalBoundary, internalBoundaries)
    
    @staticmethod
    def Circle(origin=None, radius=0.5, sides=16, fromAngle=0, toAngle=360, direction=[0,0,1],
                   placement="center", tolerance=0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created circle.

        """
        from topologicpy.Wire import Wire
        wire = Wire.Circle(origin=origin, radius=radius, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=True, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Compactness(face, mantissa=4):
        """
        Returns the compactness measure of the input face. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        face : topologic.Face
            The input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The compactness measure of the input face.

        """
        exb = face.ExternalBoundary()
        edges = []
        _ = exb.Edges(None, edges)
        perimeter = 0.0
        for anEdge in edges:
            perimeter = perimeter + abs(topologic.EdgeUtility.Length(anEdge))
        area = abs(topologic.FaceUtility.Area(face))
        compactness  = 0
        #From https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        if area <= 0:
            return None
        if perimeter <= 0:
            return None
        compactness = (math.pi*(2*math.sqrt(area/math.pi)))/perimeter
        return round(compactness, mantissa)

    @staticmethod
    def Edges(face):
        """
        Returns the edges of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        list
            The list of edges.

        """
        if not isinstance(face, topologic.Face):
            return None
        edges = []
        _ = face.Edges(None, edges)
        return edges

    def ExternalBoundary(face):
        """
        Returns the external boundary (closed wire) of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        topologic.Wire
            The external boundary of the input face.

        """
        return face.ExternalBoundary()
    
    @staticmethod
    def FacingToward(face, direction=[0,0,-1], asVertex=False, tolerance=0.0001):
        """
        Returns True if the input face is facing toward the input direction.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        direction : list , optional
            The input direction. The default is [0,0,-1].
        asVertex : bool , optional
            If set to True, the direction is treated as an actual vertex in 3D space. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the face is facing toward the direction. False otherwise.

        """
        faceNormal = topologic.FaceUtility.NormalAtParameters(face,0.5, 0.5)
        faceCenter = topologic.FaceUtility.VertexAtParameters(face,0.5,0.5)
        cList = [faceCenter.X(), faceCenter.Y(), faceCenter.Z()]
        try:
            vList = [direction.X(), direction.Y(), direction.Z()]
        except:
            try:
                vList = [direction[0], direction[1], direction[2]]
            except:
                raise Exception("Face.FacingToward - Error: Could not get the vector from the input direction")
        if asVertex:
            dV = [vList[0]-cList[0], vList[1]-cList[1], vList[2]-cList[2]]
        else:
            dV = vList
        uV = Vector.Normalize(dV)
        dot = sum([i*j for (i, j) in zip(uV, faceNormal)])
        if dot < tolerance:
            return False
        return True
    
    @staticmethod
    def Flatten(face, oldLocation=None, newLocation=None, direction=None):
        """
        Flattens the input face such that its center of mass is located at the origin and its normal is pointed in the positive Z axis.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        oldLocation : topologic.Vertex , optional
            The old location to use as the origin of the movement. If set to None, the center of mass of the input topology is used. The default is None.
        newLocation : topologic.Vertex , optional
            The new location at which to place the topology. If set to None, the world origin (0,0,0) is used. The default is None.
        direction : list , optional
            The direction, expressed as a list of [X,Y,Z] that signifies the direction of the face. If set to None, the normal at *u* 0.5 and *v* 0.5 is considered the direction of the face. The deafult is None.

        Returns
        -------
        topologic.Face
            The flattened face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        if not isinstance(face, topologic.Face):
            return None
        if not isinstance(oldLocation, topologic.Vertex):
            oldLocation = Topology.CenterOfMass(face)
        if not isinstance(newLocation, topologic.Vertex):
            newLocation = Vertex.ByCoordinates(0,0,0)
        cm = oldLocation
        world_origin = newLocation
        if not direction or len(direction) < 3:
            direction = Face.NormalAtParameters(face, 0.5, 0.5)
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
        flatFace = Topology.Translate(face, -cm.X(), -cm.Y(), -cm.Z())
        flatFace = Topology.Rotate(flatFace, world_origin, 0, 0, 1, -phi)
        flatFace = Topology.Rotate(flatFace, world_origin, 0, 1, 0, -theta)
        # Ensure flatness. Force Z to be zero
        flatExternalBoundary = Face.ExternalBoundary(flatFace)
        flatFaceVertices = Topology.SubTopologies(flatExternalBoundary, subTopologyType="vertex")
    
        tempVertices = []
        for ffv in flatFaceVertices:
            tempVertices.append(Vertex.ByCoordinates(ffv.X(), ffv.Y(), 0))
        flatExternalBoundary = Wire.ByVertices(tempVertices)

        internalBoundaries = Face.InternalBoundaries(flatFace)
        flatInternalBoundaries = []
        for internalBoundary in internalBoundaries:
            ibVertices = Wire.Vertices(internalBoundary)
            tempVertices = []
            for ibVertex in ibVertices:
                tempVertices.append(Vertex.ByCoordinates(ibVertex.X(), ibVertex.Y(), 0))
            flatInternalBoundaries.append(Wire.ByVertices(tempVertices))
        flatFace = Face.ByWires(flatExternalBoundary, flatInternalBoundaries)
        dictionary = Dictionary.ByKeysValues(["xTran", "yTran", "zTran", "phi", "theta"], [cm.X(), cm.Y(), cm.Z(), phi, theta])
        flatFace = Topology.SetDictionary(flatFace, dictionary)
        return flatFace
    
    @staticmethod
    def Planarize(face, origin=None, direction=None):
        """
        Planarizes the input face such that its center of mass is located at the input origin and its normal is pointed in the input direction.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        origin : topologic.Vertex , optional
            The old location to use as the origin of the movement. If set to None, the center of mass of the input face is used. The default is None.
        direction : list , optional
            The direction, expressed as a list of [X,Y,Z] that signifies the direction of the face. If set to None, the normal at *u* 0.5 and *v* 0.5 is considered the direction of the face. The deafult is None.

        Returns
        -------
        topologic.Face
            The planarized face.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not isinstance(face, topologic.Face):
            return None
        if not isinstance(origin, topologic.Vertex):
            origin = Topology.CenterOfMass(face)
        if not isinstance(direction, list):
            direction = Face.NormalAtParameters(face, 0.5, 0.5)
        flatFace = Face.Flatten(face, oldLocation=origin, direction=direction)

        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        planarizedFace = Topology.Rotate(flatFace, origin=world_origin, x=0, y=1, z=0, degree=theta)
        planarizedFace = Topology.Rotate(planarizedFace, origin=world_origin, x=0, y=0, z=1, degree=phi)
        planarizedFace = Topology.Translate(planarizedFace, xTran, yTran, zTran)
        return planarizedFace

    @staticmethod
    def Harmonize(face):
        """
        Returns a harmonized version of the input face such that the *u* and *v* origins are always in the upperleft corner.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        topologic.Face
            The harmonized face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not isinstance(face, topologic.Face):
            return None
        flatFace = Face.Flatten(face)
        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")
        vertices = Wire.Vertices(Face.ExternalBoundary(flatFace))
        harmonizedEB = Wire.ByVertices(vertices)
        internalBoundaries = Face.InternalBoundaries(flatFace)
        harmonizedIB = []
        for ib in internalBoundaries:
            ibVertices = Wire.Vertices(ib)
            harmonizedIB.append(Wire.ByVertices(ibVertices))
        harmonizedFace = Face.ByWires(harmonizedEB, harmonizedIB)
        harmonizedFace = Topology.Rotate(harmonizedFace, origin=world_origin, x=0, y=1, z=0, degree=theta)
        harmonizedFace = Topology.Rotate(harmonizedFace, origin=world_origin, x=0, y=0, z=1, degree=phi)
        harmonizedFace = Topology.Translate(harmonizedFace, xTran, yTran, zTran)
        return harmonizedFace

    @staticmethod
    def InternalBoundaries(face):
        """
        Returns the internal boundaries (closed wires) of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        list
            The list of internal boundaries (closed wires).

        """
        if not isinstance(face, topologic.Face):
            return None
        wires = []
        _ = face.InternalBoundaries(wires)
        return list(wires)

    @staticmethod
    def InternalVertex(face, tolerance=0.0001):
        """
        Creates a vertex guaranteed to be inside the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            The created vertex.

        """
        if not isinstance(face, topologic.Face):
            return None
        v = topologic.FaceUtility.InternalVertex(face, tolerance)
        return v

    @staticmethod
    def Invert(face):
        """
        Creates a face that is an inverse (mirror) of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        topologic.Face
            The inverted face.

        """
        from topologicpy.Wire import Wire

        if not isinstance(face, topologic.Face):
            return None
        eb = Face.ExternalBoundary(face)
        vertices = Wire.Vertices(eb)
        vertices.reverse()
        inverted_wire = Wire.ByVertices(vertices)
        internal_boundaries = Face.InternalBoundaries(face)
        if not internal_boundaries:
            inverted_face = Face.ByWire(inverted_wire)
        else:
            inverted_face = Face.ByWires(inverted_wire, internal_boundaries)
        return inverted_face

    @staticmethod
    def IsCoplanar(faceA, faceB, tolerance=0.0001):
        """
        Returns True if the two input faces are coplanar. Returns False otherwise.

        Parameters
        ----------
        faceA : topologic.Face
            The first input face.
        faceB : topologic.Face
            The second input face
        tolerance : float , optional
            The desired tolerance. The deafault is 0.0001.

        Raises
        ------
        Exception
            Raises an exception if the angle between the two input faces cannot be determined.

        Returns
        -------
        bool
            True if the two input faces are coplanar. False otherwise.

        """
        if not isinstance(faceA, topologic.Face) or not isinstance(faceB, topologic.Face):
            return None
        dirA = Face.NormalAtParameters(faceA, 0.5, 0.5, "xyz", 3)
        dirB = Face.NormalAtParameters(faceB, 0.5, 0.5, "xyz", 3)
        return Vector.IsCollinear(dirA, dirB, tolerance)
    

    @staticmethod
    def IsInside(face, vertex, tolerance=0.0001):
        """
        Returns True if the input vertex is inside the input face. Returns False otherwise.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertex is inside the input face. False otherwise.

        """

        # Ray tracing from https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
        def ray_tracing_method(x,y,poly):
            n = len(poly)
            inside = False

            p1x,p1y = poly[0]
            for i in range(n+1):
                p2x,p2y = poly[i % n]
                if y > min(p1y,p2y):
                    if y <= max(p1y,p2y):
                        if x <= max(p1x,p2x):
                            if p1y != p2y:
                                xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x,p1y = p2x,p2y

            return inside

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not isinstance(face, topologic.Face):
            return None
        if not isinstance(vertex, topologic.Vertex):
            return None

        world_origin = Vertex.ByCoordinates(0,0,0)
        # Flatten face and vertex
        flatFace = Face.Flatten(face)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        vertex = Topology.Translate(vertex, -xTran, -yTran, -zTran)
        vertex = Topology.Rotate(vertex, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        vertex = Topology.Rotate(vertex, origin=world_origin, x=0, y=1, z=0, degree=-theta)

        # Test if Vertex is hovering above or below face
        if abs(Vertex.Z(vertex)) > tolerance:
            return False

        # Build 2D poly from flat face
        wire = Face.ExternalBoundary(flatFace)
        vertices = Wire.Vertices(wire)
        poly = []
        for v in vertices:
            poly.append([Vertex.X(v), Vertex.Y(v)])

        # Use ray tracing method to test if vertex is inside the face
        status = ray_tracing_method(Vertex.X(vertex), Vertex.Y(vertex), poly)
        # Vertex is not inside
        if not status:
            return status

        # If it is inside, we must check if it is inside a hole in the face
        internal_boundaries = Face.InternalBoundaries(flatFace)
        if len(internal_boundaries) == 0:
            return status
        
        for ib in internal_boundaries:
            vertices = Wire.Vertices(ib)
            poly = []
            for v in vertices:
                poly.append([Vertex.X(v), Vertex.Y(v)])
            status2 = ray_tracing_method(Vertex.X(vertex), Vertex.Y(vertex), poly)
            if status2:
                return False
        return status

    @staticmethod
    def MedialAxis(face, resolution=0, externalVertices=False, internalVertices=False, toLeavesOnly=False, tolerance=0.0001, angTolerance=0.1):
        """
        Returns a wire representing an approximation of the medial axis of the input topology. See https://en.wikipedia.org/wiki/Medial_axis.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        resolution : int , optional
            The desired resolution of the solution (range is 0: standard resolution to 10: high resolution). This determines the density of the sampling along each edge. The default is 0.
        externalVertices : bool , optional
            If set to True, the external vertices of the face will be connected to the nearest vertex on the medial axis. The default is False.
        internalVertices : bool , optional
            If set to True, the internal vertices of the face will be connected to the nearest vertex on the medial axis. The default is False.
        toLeavesOnly : bool , optional
            If set to True, the vertices of the face will be connected to the nearest vertex on the medial axis only if this vertex is a leaf (end point). Otherwise, it will connect to any nearest vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        angTolerance : float , optional
            The desired angular tolerance in degrees for removing collinear edges. The default is 0.1.
        
        Returns
        -------
        topologic.Wire
            The medial axis of the input face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def touchesEdge(vertex,edges, tolerance=0.0001):
            if not isinstance(vertex, topologic.Vertex):
                return False
            for edge in edges:
                u = Edge.ParameterAtVertex(edge, vertex, mantissa=4)
                if not u:
                    continue
                if 0<u<1:
                    return True
            return False

        # Flatten the input face
        flatFace = Face.Flatten(face)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        # Create a Vertex at the world's origin (0,0,0)
        world_origin = Vertex.ByCoordinates(0,0,0)

        faceVertices = Face.Vertices(flatFace)
        faceEdges = Face.Edges(flatFace)
        vertices = []
        resolution = 10 - resolution
        resolution = min(max(resolution, 1), 10)
        for e in faceEdges:
            for n in range(resolution, 100, resolution):
                vertices.append(Edge.VertexByParameter(e,n*0.01))
        
        voronoi = Shell.Voronoi(vertices=vertices, face=flatFace)
        voronoiEdges = Shell.Edges(voronoi)

        medialAxisEdges = []
        for e in voronoiEdges:
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)
            svTouchesEdge = touchesEdge(sv, faceEdges, tolerance=tolerance)
            evTouchesEdge = touchesEdge(ev, faceEdges, tolerance=tolerance)
            #connectsToCorners = (Vertex.Index(sv, faceVertices) != None) or (Vertex.Index(ev, faceVertices) != None)
            #if Face.IsInside(flatFace, sv, tolerance=tolerance) and Face.IsInside(flatFace, ev, tolerance=tolerance):
            if not svTouchesEdge and not evTouchesEdge:
                medialAxisEdges.append(e)

        extBoundary = Face.ExternalBoundary(flatFace)
        extVertices = Wire.Vertices(extBoundary)

        intBoundaries = Face.InternalBoundaries(flatFace)
        intVertices = []
        for ib in intBoundaries:
            intVertices = intVertices+Wire.Vertices(ib)
        
        theVertices = []
        if internalVertices:
            theVertices = theVertices+intVertices
        if externalVertices:
            theVertices = theVertices+extVertices

        tempWire = Cluster.SelfMerge(Cluster.ByTopologies(medialAxisEdges))
        if isinstance(tempWire, topologic.Wire) and angTolerance > 0:
            tempWire = Wire.RemoveCollinearEdges(tempWire, angTolerance=angTolerance)
        medialAxisEdges = Wire.Edges(tempWire)
        for v in theVertices:
            nv = Vertex.NearestVertex(v, tempWire, useKDTree=False)

            if isinstance(nv, topologic.Vertex):
                if toLeavesOnly:
                    adjVertices = Topology.AdjacentTopologies(nv, tempWire)
                    if len(adjVertices) < 2:
                        medialAxisEdges.append(Edge.ByVertices([nv, v]))
                else:
                    medialAxisEdges.append(Edge.ByVertices([nv, v]))
        medialAxis = Cluster.SelfMerge(Cluster.ByTopologies(medialAxisEdges))
        if isinstance(medialAxis, topologic.Wire) and angTolerance > 0:
            medialAxis = Wire.RemoveCollinearEdges(medialAxis, angTolerance=angTolerance)
        medialAxis = Topology.Rotate(medialAxis, origin=world_origin, x=0, y=1, z=0, degree=theta)
        medialAxis = Topology.Rotate(medialAxis, origin=world_origin, x=0, y=0, z=1, degree=phi)
        medialAxis = Topology.Translate(medialAxis, xTran, yTran, zTran)
        return medialAxis

    @staticmethod
    def Normal(face, outputType="xyz", mantissa=4):
        """
        Returns the normal vector to the input face. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The normal vector to the input face. This is computed at the approximate center of the face.

        """
        return Face.NormalAtParameters(face, u=0.5, v=0.5, outputType=outputType, mantissa=mantissa)

    @staticmethod
    def NormalAtParameters(face, u=0.5, v=0.5, outputType="xyz", mantissa=4):
        """
        Returns the normal vector to the input face. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        u : float , optional
            The *u* parameter at which to compute the normal to the input face. The default is 0.5.
        v : float , optional
            The *v* parameter at which to compute the normal to the input face. The default is 0.5.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The normal vector to the input face.

        """
        returnResult = []
        try:
            coords = topologic.FaceUtility.NormalAtParameters(face, u, v)
            x = round(coords[0], mantissa)
            y = round(coords[1], mantissa)
            z = round(coords[2], mantissa)
            outputType = list(outputType.lower())
            for axis in outputType:
                if axis == "x":
                    returnResult.append(x)
                elif axis == "y":
                    returnResult.append(y)
                elif axis == "z":
                    returnResult.append(z)
        except:
            returnResult = None
        return returnResult
    
    @staticmethod
    def NormalEdge(face, length=1):
        """
        Returns the normal vector to the input face as an edge with the desired input length. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        length : float , optional
            The desired length of the normal edge. The default is 1.

        Returns
        -------
        topologic.Edge
            The created normal edge to the input face. This is computed at the approximate center of the face.

        """
        return Face.NormalEdgeAtParameters(face, u=0.5, v=0.5, length=length)

    @staticmethod
    def NormalEdgeAtParameters(face, u=0.5, v=0.5, length=1):
        """
        Returns the normal vector to the input face as an edge with the desired input length. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        u : float , optional
            The *u* parameter at which to compute the normal to the input face. The default is 0.5.
        v : float , optional
            The *v* parameter at which to compute the normal to the input face. The default is 0.5.
        length : float , optional
            The desired length of the normal edge. The default is 1.

        Returns
        -------
        topologic.Edge
            The created normal edge to the input face. This is computed at the approximate center of the face.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        if not isinstance(face, topologic.Face):
            return None
        sv = Face.VertexByParameters(face=face, u=u, v=v)
        vec = Face.NormalAtParameters(face, u=u, v=v)
        ev = Topology.TranslateByDirectionDistance(sv, vec, length)
        return Edge.ByVertices([sv, ev])
    
    @staticmethod
    def Project(faceA, faceB, direction=None, mantissa=4, tolerance=0.0001):
        """
        Creates a projection of the first input face unto the second input face.

        Parameters
        ----------
        faceA : topologic.Face
            The face to be projected.
        faceB : topologic.Face
            The face unto which the first input face will be projected.
        direction : list, optional
            The vector direction of the projection. If None, the reverse vector of the receiving face normal will be used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        topologic.Face
            The projected Face.

        """

        if not faceA:
            return None
        if not isinstance(faceA, topologic.Face):
            return None
        if not faceB:
            return None
        if not isinstance(faceB, topologic.Face):
            return None

        eb = faceA.ExternalBoundary()
        ib_list = []
        _ = faceA.InternalBoundaries(ib_list)
        p_eb = Wire.Project(eb, faceB, direction, mantissa, tolerance)
        p_ib_list = []
        for ib in ib_list:
            temp_ib = Wire.Project(ib, faceB, direction, mantissa, tolerance)
            if temp_ib:
                p_ib_list.append(temp_ib)
        return Face.ByWires(p_eb, p_ib_list)

    @staticmethod
    def Rectangle(origin=None, width=1.0, length=1.0, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0,0,0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created face.

        """
        wire = Wire.Rectangle(origin=origin, width=width, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Star(origin=None, radiusA=1.0, radiusB=0.4, rays=5, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a star.

        Parameters
        ----------
        origin : topologic.Vertex, optional
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
        topologic.Face
            The created face.

        """
        wire = Wire.Star(origin=origin, radiusA=radiusA, radiusB=radiusB, rays=rays, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Trapezoid(origin=None, widthA=1.0, widthB=0.75, offsetA=0.0, offsetB=0.0, length=1.0, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic.Vertex, optional
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
        topologic.Face
            The created trapezoid.

        """
        wire = Wire.Trapezoid(origin=origin, widthA=widthA, widthB=widthB, offsetA=offsetA, offsetB=offsetB, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Triangulate(face):
        """
        Triangulates the input face and returns a list of faces.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        list
            The list of triangles of the input face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        flatFace = Face.Flatten(face)
        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")
    
        faceTriangles = []
        for i in range(0,5,1):
            try:
                _ = topologic.FaceUtility.Triangulate(flatFace, float(i)*0.1, faceTriangles)
                break
            except:
                continue
        if len(faceTriangles) < 1:
            return [face]
        finalFaces = []
        for f in faceTriangles:
            f = Topology.Rotate(f, origin=world_origin, x=0, y=1, z=0, degree=theta)
            f = Topology.Rotate(f, origin=world_origin, x=0, y=0, z=1, degree=phi)
            f = Topology.Translate(f, xTran, yTran, zTran)
            if Face.Angle(face, f) > 90:
                wire = Face.ExternalBoundary(f)
                wire = Wire.Invert(wire)
                f = topologic.Face.ByExternalBoundary(wire)
                finalFaces.append(f)
            else:
                finalFaces.append(f)
        return finalFaces

    @staticmethod
    def TrimByWire(face, wire, reverse = False):
        """
        Trims the input face by the input wire.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        wire : topologic.Wire
            The input wire.
        reverse : bool , optional
            If set to True, the effect of the trim will be reversed. The default is False.

        Returns
        -------
        topologic.Face
            The resulting trimmed face.

        """
        if not isinstance(face, topologic.Face):
            return None
        if not isinstance(wire, topologic.Wire):
            return face
        trimmed_face = topologic.FaceUtility.TrimByWire(face, wire, False)
        if reverse:
            trimmed_face = face.Difference(trimmed_face)
        return trimmed_face
    
    @staticmethod
    def VertexByParameters(face, u=0.5, v=0.5):
        """
        Creates a vertex at the *u* and *v* parameters of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        u : float , optional
            The *u* parameter of the input face. The default is 0.5.
        v : float , optional
            The *v* parameter of the input face. The default is 0.5.

        Returns
        -------
        vertex : topologic vertex
            The created vertex.

        """
        if not isinstance(face, topologic.Face):
            return None
        return topologic.FaceUtility.VertexAtParameters(face, u, v)
    
    @staticmethod
    def VertexParameters(face, vertex, outputType="uv", mantissa=4):
        """
        Returns the *u* and *v* parameters of the input face at the location of the input vertex.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        vertex : topologic.Vertex
            The input vertex.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "uv". It is case insensitive. The default is "uv".
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The list of *u* and/or *v* as specified by the outputType input.

        """
        if not isinstance(face, topologic.Face):
            return None
        if not isinstance(vertex, topologic.Vertex):
            return None
        params = topologic.FaceUtility.ParametersAtVertex(face, vertex)
        u = round(params[0], mantissa)
        v = round(params[1], mantissa)
        outputType = list(outputType.lower())
        returnResult = []
        for param in outputType:
            if param == "u":
                returnResult.append(u)
            elif param == "v":
                returnResult.append(v)
        return returnResult

    @staticmethod
    def Vertices(face):
        """
        Returns the vertices of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(face, topologic.Face):
            return None
        vertices = []
        _ = face.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Wires(face):
        """
        Returns the wires of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(face, topologic.Face):
            return None
        wires = []
        _ = face.Wires(None, wires)
        return wires
