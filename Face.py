import topologicpy
import topologic
from topologicpy.Vector import Vector
from topologicpy.Wire import Wire
import math
import numpy as np
from numpy.linalg import norm

class Face(topologic.Face):
    @staticmethod
    def AddInternalBoundaries(face, wires):
        """
        Description
        ----------
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
        Description
        ----------
        Adds internal the input cluster of internal boundaries (closed wires) to the input face. Internal boundaries are considered holes in the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        cluster : topollogic.Cluster
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
        Description
        ----------
        Returns the angle in degrees between the two input faces.

        Parameters
        ----------
        faceA : topologic.Face
            The first input face.
        faceB : topologic.Face
            The second input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        float
            The angle in degrees between the two input faces.

        """
        
        if not faceA or not isinstance(faceA, topologic.Face):
            return None
        if not faceB or not isinstance(faceB, topologic.Face):
            return None
        dirA = Face.NormalAtParameters(faceA, 0.5, 0.5, "xyz", 3)
        dirB = Face.NormalAtParameters(faceB, 0.5, 0.5, "xyz", 3)
        return round((Vector.Angle(dirA, dirB)), mantissa)
    
    @staticmethod
    def Area(face, mantissa):
        """
        Description
        ----------
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
        Description
        ----------
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
        Description
        ----------
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
        if not isinstance(edges, list):
            return None
        edgeList = [x for x in edges if isinstance(x, topologic.Edge)]
        if len(edgeList) < 1:
            return None
        wire = None
        face = None
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
            return None
        else:
            try:
                face = topologic.Face.ByExternalBoundary(wire)
            except:
                return None
        return face

    @staticmethod
    def ByEdgesCluster(cluster):
        """
        Description
        ----------
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
        if not isinstance(edges, topologic.Cluster):
            return None
        edges = []
        _ = cluster.Edges(None, edges)
        return Face.ByEdges(edges)

    @staticmethod
    def ByOffset(face, offset, reverse, tolerance=0.0001):
        """
        Description
        ----------
        Creates a face by offsetting the edges of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        offset : float , optional
            The desired offset value. The default is 0.
        reverse : bool , optional
            If set to True the offset will be computed to the inside of the input face. Otherwise, it will be computed to the outside of the face. The default is False.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The offsetted face.

        """
        external_boundary = face.ExternalBoundary()
        internal_boundaries = []
        _ = face.InternalBoundaries(internal_boundaries)
        offset_external_boundary = Wire.ByOffset(external_boundary, offset, reverse, tolerance)
        offset_external_face = topologic.Face.ByExternalBoundary(offset_external_boundary)
        if topologic.FaceUtility.Area(offset_external_face) < tolerance:
            raise Exception("ERROR: (Topologic>Face.ByOffset) external boundary area is less than tolerance.")
        offset_internal_boundaries = []
        reverse = not reverse
        area_sum = 0
        for internal_boundary in internal_boundaries:
            internal_wire = Wire.ByOffset(internal_boundary, offset, reverse, tolerance)
            internal_face = topologic.Face.ByExternalBoundary(internal_wire)
            # Check if internal boundary has a trivial area
            if topologic.FaceUtility.Area(internal_face) < tolerance:
                return None
            # Check if area of internal boundary is larger than area of external boundary
            if topologic.FaceUtility.Area(internal_face) > topologic.FaceUtility.Area(offset_external_face):
                return None
            dif_wire = internal_wire.Difference(offset_external_boundary)
            internal_vertices = []
            _ = internal_wire.Vertices(None, internal_vertices)
            dif_vertices = []
            _ = dif_wire.Vertices(None, dif_vertices)
            # Check if internal boundary intersect the outer boundary
            if len(internal_vertices) != len(dif_vertices):
                return None
            offset_internal_boundaries.append(internal_wire)
            area_sum = area_sum + topologic.FaceUtility.Area(internal_face)
        if area_sum > topologic.FaceUtility.Area(offset_external_face):
            return None
        # NOT IMPLEMENTED: Check if internal boundaries intersect each other!
        returnFace = topologic.Face.ByExternalInternalBoundaries(offset_external_boundary, offset_internal_boundaries)
        if returnFace.Type() != 8:
            return None
        if topologic.FaceUtility.Area(returnFace) < tolerance:
            return None
        return returnFace
    
    @staticmethod
    def ByShell(shell, angTolerance=0.1):
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
    def ByVertices(vertices):
        """
        Description
        ----------
        Creates a face from the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.

        Returns
        -------
        topologic.Face
            The crearted face.

        """
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, topologic.Vertex)]
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

    def ByVerticesCluster(cluster):
        """
        Description
        ----------
        Creates a face from the input cluster of vertices.

        Parameters
        ----------
        vertices : topologic.Cluster
            The input cluster of vertices.

        Returns
        -------
        topologic.Face
            The crearted face.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        vertices = []
        _ = cluster.Vertices(None, vertices)
        return Face.ByVertices(vertices)

    @staticmethod
    def ByWire(wire):
        """
        Description
        ----------
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
        if not isinstance(wire, topologic.Wire):
            return None
        if not Wire.IsClosed(wire):
            return None
        return topologic.Face.ByExternalBoundary(wire)

    @staticmethod
    def ByWires(externalBoundary, internalBoundaries=[]):
        """
        Description
        ----------
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
    def ByWiresCluster(externalBoundary, internalBoundariesCluster=[]):
        """
        Description
        ----------
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
        if not isinstance(externalBoundary, topologic.Wire):
            return None
        if not Wire.IsClosed(externalBoundary):
            return None
        internalBoundaries = []
        _ = internalBoundariesCluster.Wires(None, internalBoundaries)
        return Face.ByWires(externalBoundary, internalBoundaries)

    @staticmethod
    def Compactness(face, mantissa=4):
        """
        Description
        ----------
        Returns the compactness value of the input face. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        face : topologic.Face
            The input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The compactness value of the input face.

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
    def ExternalBoundary(face):
        """
        Description
        ----------
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
        Description
        ----------
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
    def Flatten(face):
        """
        Description
        ----------
        Flattens the input face such that its center of mass is located at the origin and its normal is pointed in the positvie Z axis.

        Parameters
        ----------
        face : topologic.Face
            The input face.

        Returns
        -------
        topologic.Face
            The flattened face.

        """
        if not isinstance(face, topologic.Face):
            return None
        origin = topologic.Vertex.ByCoordinates(0,0,0)
        cm = face.CenterOfMass()
        coords = topologic.FaceUtility.NormalAtParameters(face, 0.5, 0.5)
        x1 = cm.X()
        y1 = cm.Y()
        z1 = cm.Z()
        x2 = cm.X() + coords[0]
        y2 = cm.Y() + coords[1]
        z2 = cm.Z() + coords[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        flat_item = topologic.TopologyUtility.Translate(item, -cm.X(), -cm.Y(), -cm.Z())
        flat_item = topologic.TopologyUtility.Rotate(flat_item, origin, 0, 0, 1, -phi)
        flat_item = topologic.TopologyUtility.Rotate(flat_item, origin, 0, 1, 0, -theta)
        return flat_item
    
    @staticmethod
    def InternalBoundaries(face):
        """
        Description
        ----------
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
        Description
        ----------
        Creates a vertex guaranteed to be inside the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            The created vertex.

        """
        if not isinstance(face, topologic.Face):
            return None
        return topologic.FaceUtility.InternalVertex(face, tolerance)
    
    @staticmethod
    def IsCoplanar(faceA, faceB, tolerance=0.0001):
        """
        Description
        ----------
        Returns True if the two input faces are coplanar. Returns False otherwise.

        Parameters
        ----------
        faceA : topologic.Face
            The first input face.
        faceB : topologic.Face
            The second input face
        tolerance : float, optional
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
        Description
        ----------
        Returns True if the input vertex is inside the input face. Returns False otherwise.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertex is inside the input face. False otherwise.

        """
        if not isinstance(face, topologic.Face):
            return None
        if not isinstance(vertex, topologic.Vertex):
            return None
        return (topologic.FaceUtility.IsInside(face, vertex, tolerance))


    @staticmethod
    def NormalAtParameters(face, u=0.5, v=0.5, outputType="xyz", mantissa=4):
        """
        Description
        ----------
        Returns the normal vector to the input face. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        u : float , optional
            THe *u* parameter at which to compute the normal to the input face. The default is 0.5.
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
    def Project(faceA, faceB, direction=None, mantissa=4, tolerance=0.0001):
        """
        Description
        ----------
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
    def TrimByWire(face, wire, reverse = False):
        """
        Description
        ----------
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
        Description
        ----------
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
        Description
        ----------
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
