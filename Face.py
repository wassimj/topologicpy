from topologicpy import topologic
import math
import numpy as np
from numpy import arctan, pi, signbit
from numpy.linalg import norm
import topologic_lib
import Shell
import Wire
import Vertex
import Topology
import Edge

class Face(topologic.Face):
    @staticmethod
    def FaceAddInternalBoundaries(face, cluster):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        cluster : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # face, cluster = item
        assert isinstance(face, topologic.Face), "FaceAddInternalBoundaries - Error: The host face input is not a Face"
        if isinstance(cluster, topologic.Cluster):
            wires = []
            _ = cluster.Wires(None, wires)
        elif isinstance(cluster, topologic.Wire):
            wires = [cluster]
        elif isinstance(cluster, list):
            wires = [w for w in cluster if isinstance(w, topologic.Wire)]
        else:
            return face
        faceeb = face.ExternalBoundary()
        faceibList = []
        _ = face.InternalBoundaries(faceibList)
        for wire in wires:
            faceibList.append(wire)
        return topologic.Face.ByExternalInternalBoundaries(faceeb, faceibList)
    
    @staticmethod
    def FaceAddInternalBoundary(face, cluster):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        cluster : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # face, cluster = item
        assert isinstance(face, topologic.Face), "FaceAddInternalBoundaries - Error: The host face input is not a Face"
        assert isinstance(face, topologic.Cluster), "FaceAddInternalBoundaries - Error: The internal boundaries input is not a Cluster"
        wires = []
        _ = cluster.Wires(None, wires)
        faceeb = face.ExternalBoundary()
        faceibList = []
        _ = face.InternalBoundaries(faceibList)
        for wire in wires:
            faceibList.append(wire)
        return topologic.Face.ByExternalInternalBoundaries(faceeb, faceibList)
    
    @staticmethod
    def angle_between(v1, v2):
        """
        Parameters
        ----------
        v1 : TYPE
            DESCRIPTION.
        v2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        u1 = v1 / norm(v1)
        u2 = v2 / norm(v2)
        y = u1 - u2
        x = u1 + u2
        a0 = 2 * arctan(norm(y) / norm(x))
        if (not signbit(a0)) or signbit(pi - a0):
            return a0
        elif signbit(a0):
            return 0
        else:
            return pi
    
    @staticmethod
    def FaceAngle(faceA, faceB, mantissa):
        """
        Parameters
        ----------
        faceA : TYPE
            DESCRIPTION.
        faceB : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # faceA, faceB, mantissa = item
        
        if not faceA or not isinstance(faceA, topologic.Face):
            raise Exception("Face.Angle - Error: Face A is not valid")
        if not faceB or not isinstance(faceB, topologic.Face):
            raise Exception("Face.Angle - Error: Face B is not valid")
        dirA = Face.FaceNormalAtParameters(faceA, 0.5, 0.5, "XYZ", 3)
        dirB = Face.FaceNormalAtParameters(faceB, 0.5, 0.5, "XYZ", 3)
        return round((Face.angle_between(dirA, dirB) * 180 / pi), mantissa) # convert to degrees
    
    @staticmethod
    def FaceArea(face, mantissa):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Returns
        -------
        area : TYPE
            DESCRIPTION.

        """
        # face, mantissa = item
        area = None
        try:
            area = round(topologic.FaceUtility.Area(face), mantissa)
        except:
            area = None
        return area

    @staticmethod
    def FaceBoundingFace(face):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # face = item[0]
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
    def FaceByEdges(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        face : TYPE
            DESCRIPTION.

        """
        assert isinstance(item, list), "face.ByEdges - Error: Input is not a list"
        edges = [x for x in item if isinstance(x, topologic.Edge)]
        wire = None
        face = None
        for anEdge in edges:
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
    def processWire(wire, offset, reverse):
        """
        Parameters
        ----------
        wire : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.
        reverse : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        face = topologic.Face.ByExternalBoundary(wire)
        if reverse:
            offset = -offset
        external_vertices = []
        _ = wire.Vertices(None, external_vertices)
        offset_vertices = []
        for idx in range(len(external_vertices)-1):
            vrtx = [external_vertices[idx].X(), external_vertices[idx].Y(), external_vertices[idx].Z()]
            vrtx1 = [external_vertices[idx+1].X(), external_vertices[idx+1].Y(), external_vertices[idx+1].Z()]
            vrtx2 = [external_vertices[idx-1].X(), external_vertices[idx-1].Y(), external_vertices[idx-1].Z()]
            u = topologic_lib.normalize([(vrtx1[0] - vrtx[0]), (vrtx1[1] - vrtx[1]),(vrtx1[2] - vrtx[2])])
            v = topologic_lib.normalize([(vrtx2[0] - vrtx[0]), (vrtx2[1] - vrtx[1]),(vrtx2[2] - vrtx[2])])
            ev = external_vertices[idx]
            v3 = vrtx + u
            v4 = vrtx + v
            offset_vertex = ([ev.X(), ev.Y(), ev.Z()] + offset * math.sqrt(2 / (1 - np.dot(u, v))) * topologic_lib.normalize(u + v))
            topologic_offset_vertex = topologic.Vertex.ByCoordinates(offset_vertex[0], offset_vertex[1], offset_vertex[2])
            status = (topologic.FaceUtility.IsInside(face, topologic_offset_vertex, 0.001))
            if reverse:
                status = not status
            if status:
                offset = -offset
                offset_vertex = ([ev.X(), ev.Y(), ev.Z()] + offset * math.sqrt(2 / (1 - np.dot(u, v))) * topologic_lib.normalize(u + v))
            offset_vertices.append([ev.X(), ev.Y(), ev.Z()] + offset * math.sqrt(2 / (1 - np.dot(u, v))) * topologic_lib.normalize(u + v))

        idx = len(external_vertices)-1
        v = [external_vertices[idx].X(), external_vertices[idx].Y(), external_vertices[idx].Z()]
        v1 = [external_vertices[0].X(), external_vertices[0].Y(), external_vertices[0].Z()]
        v2 = [external_vertices[idx-1].X(), external_vertices[idx-1].Y(), external_vertices[idx-1].Z()]
        u = topologic_lib.normalize([(v1[0]-v[0]), (v1[1]-v[1]), (v1[2]-v[2])])
        v = topologic_lib.normalize([(v2[0]-v[0]), (v2[1]-v[1]),(v2[2]-v[2])])
        ev = external_vertices[idx]
        offset_vertex = ([ev.X(), ev.Y(), ev.Z()] + offset * math.sqrt(2 / (1 - np.dot(u, v))) * topologic_lib.normalize(u + v))
        topologic_offset_vertex = topologic.Vertex.ByCoordinates(offset_vertex[0], offset_vertex[1], offset_vertex[2])
        status = (topologic.FaceUtility.IsInside(face, topologic_offset_vertex, 0.001))
        if reverse:
            status = not status
        if status:
            offset = -offset
            offset_vertex = ([ev.X(), ev.Y(), ev.Z()] + offset * math.sqrt(2 / (1 - np.dot(u, v))) * topologic_lib.normalize(u + v))
        offset_vertices.append([ev.X(), ev.Y(), ev.Z()] + offset * math.sqrt(2 / (1 - np.dot(u, v))) * topologic_lib.normalize(u + v))
        edges = []
        for iv, v in enumerate(offset_vertices[:-1]):
            e = topologic.Edge.ByStartVertexEndVertex(topologic.Vertex.ByCoordinates(offset_vertices[iv][0], offset_vertices[iv][1], offset_vertices[iv][2]), topologic.Vertex.ByCoordinates(offset_vertices[iv+1][0], offset_vertices[iv+1][1], offset_vertices[iv+1][2]))
            edges.append(e)
        iv = len(offset_vertices)-1
        e = topologic.Edge.ByStartVertexEndVertex(topologic.Vertex.ByCoordinates(offset_vertices[iv][0], offset_vertices[iv][1], offset_vertices[iv][2]), topologic.Vertex.ByCoordinates(offset_vertices[0][0], offset_vertices[0][1], offset_vertices[0][2]))
        edges.append(e)
        return topologic.Wire.ByEdges(edges)

    @staticmethod
    def FaceByOffset(face, offset, reverse, tolerance=0.0001):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.
        reverse : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        returnFace : TYPE
            DESCRIPTION.

        """
        # face, offset, reverse, tolerance = item
        external_boundary = face.ExternalBoundary()
        internal_boundaries = []
        _ = face.InternalBoundaries(internal_boundaries)
        offset_external_boundary = Face.processWire(external_boundary, offset, reverse)
        offset_external_face = topologic.Face.ByExternalBoundary(offset_external_boundary)
        if topologic.FaceUtility.Area(offset_external_face) < tolerance:
            raise Exception("ERROR: (Topologic>Face.ByOffset) external boundary area is less than tolerance.")
        offset_internal_boundaries = []
        reverse = not reverse
        area_sum = 0
        for internal_boundary in internal_boundaries:
            internal_wire = Face.processWire(internal_boundary, offset, reverse)
            internal_face = topologic.Face.ByExternalBoundary(internal_wire)
            # Check if internal boundary has a trivial area
            if topologic.FaceUtility.Area(internal_face) < tolerance:
                raise Exception("ERROR: (Topologic>Face.ByOffset) internal boundary area is less than tolerance.")
            # Check if area of internal boundary is larger than area of external boundary
            if topologic.FaceUtility.Area(internal_face) > topologic.FaceUtility.Area(offset_external_face):
                raise Exception("ERROR: (Topologic>Face.ByOffset) internal boundary area is larger than the area of the external boundary.")
            dif_wire = internal_wire.Difference(offset_external_boundary)
            internal_vertices = []
            _ = internal_wire.Vertices(None, internal_vertices)
            dif_vertices = []
            _ = dif_wire.Vertices(None, dif_vertices)
            # Check if internal boundary intersect the outer boundary
            if len(internal_vertices) != len(dif_vertices):
                raise Exception("ERROR: (Topologic>Face.ByOffset) internal boundaries intersect outer boundary.")
            offset_internal_boundaries.append(internal_wire)
            area_sum = area_sum + topologic.FaceUtility.Area(internal_face)
        if area_sum > topologic.FaceUtility.Area(offset_external_face):
            raise Exception("ERROR: (Topologic>Face.ByOffset) total area of internal boundaries is larger than the area of the external boundary.")
        # NOT IMPLEMENTED: Check if internal boundaries intersect each other!
        returnFace = topologic.Face.ByExternalInternalBoundaries(offset_external_boundary, offset_internal_boundaries)
        if returnFace.Type() != 8:
            raise Exception("ERROR: (Topologic>Face.ByOffset) invalid resulting face.")
        if topologic.FaceUtility.Area(returnFace) < tolerance:
            raise Exception("ERROR: (Topologic>Face.ByOffset) area of resulting face is smaller than the tolerance.")
        return returnFace
    
    @staticmethod
    def FaceByShell(shell, angTol=0.1):
        """
        Parameters
        ----------
        shell : TYPE
            DESCRIPTION.
        angTol : float, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # shell, angTol = item
        
        def planarize(wire):
            verts = []
            _ = wire.Vertices(None, verts)
            w = Wire.WireByVertices([verts[0], verts[1], verts[2]], True)
            f = topologic.Face.ByExternalBoundary(w)
            proj_verts = []
            for v in verts:
                proj_verts.append(Vertex.VertexProject(v, f))
            new_w = Wire.WireByVertices(proj_verts, True)
            return new_w
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(planarize(aWire))
            return returnList
        
        ext_boundary = Shell.ShellExternalBoundary(shell)
        if isinstance(ext_boundary, topologic.Wire):
            try:
                return topologic.Face.ByExternalBoundary(Wire.WireRemoveCollinearEdges(ext_boundary, angTol))
            except:
                try:
                    return topologic.Face.ByExternalBoundary(planarize(Wire.WireRemoveCollinearEdges(ext_boundary, angTol)))
                except:
                    print("FaceByPlanarShell - Error: The input Wire is not planar and could not be fixed. Returning the planarized Wire.")
                    return planarize(ext_boundary)
        elif isinstance(ext_boundary, topologic.Cluster):
            wires = []
            _ = ext_boundary.Wires(None, wires)
            faces = []
            areas = []
            for aWire in wires:
                try:
                    aFace = topologic.Face.ByExternalBoundary(Wire.WireRemoveCollinearEdges(aWire, angTol))
                except:
                    aFace = topologic.Face.ByExternalBoundary(planarize(Wire.WireRemoveCollinearEdges(aWire, angTol)))
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
                int_wires.append(Wire.WireRemoveCollinearEdges(temp_wires[0], angTol))
            temp_wires = []
            _ = ext_boundary.Wires(None, temp_wires)
            ext_wire = Wire.WireRemoveCollinearEdges(temp_wires[0], angTol)
            try:
                return topologic.Face.ByExternalInternalBoundaries(ext_wire, int_wires)
            except:
                return topologic.Face.ByExternalInternalBoundaries(planarize(ext_wire), planarizeList(int_wires))
        else:
            return None
    
    @staticmethod
    def FaceByVertices(item):
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
        assert isinstance(item, list), "Face.ByVertices - Error: Input is not a list"
        vertices = [x for x in item if isinstance(x, topologic.Vertex)]
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
        v1 = vertices[-1]
        v2 = vertices[0]
        try:
            e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
            if e:
                edges.append(e)
        except:
            pass
        if len(edges) > 0:
            return topologic.Face.ByExternalBoundary(Topology.TopologySelfMerge(topologic.Cluster.ByTopologies(edges, False)))
        else:
            return None
    
    @staticmethod
    def FaceByWire(item):
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
        if isinstance(item, topologic.Wire):
            return topologic.Face.ByExternalBoundary(item)
        return None
    
    @staticmethod
    def FaceByWires(externalBoundary, internalBoundariesCluster):
        """
        Parameters
        ----------
        externalBoundary : TYPE
            DESCRIPTION.
        internalBoundariesCluster : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # externalBoundary, internalBoundariesCluster = item
        assert isinstance(externalBoundary, topologic.Wire), "Face.ByWires - Error: External Boundary Input is not a Wire"
        assert isinstance(internalBoundariesCluster, topologic.Cluster), "Face.ByWires - Error: Internal Boundaries Input is not a Cluster"
        internalBoundaries = []
        _ = internalBoundariesCluster.Wires(None, internalBoundaries)
        return topologic.Face.ByExternalInternalBoundaries(externalBoundary, internalBoundaries)

    @staticmethod
    def FaceCompactness(face, mantissa):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # face, mantissa = item
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
            raise Exception("Error: Face.Compactness: Face area is less than or equal to zero")
        if perimeter <= 0:
            raise Exception("Error: Face.Compactness: Face perimeter is less than or equal to zero")
        compactness = (math.pi*(2*math.sqrt(area/math.pi)))/perimeter
        return round(compactness, mantissa)
    
    @staticmethod
    def FaceExternalBoundary(item):
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
    def FaceFacingToward(face, direction, asVertex, tolerance=0.0001):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        direction : TYPE
            DESCRIPTION.
        asVertex : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # face, direction, asVertex, tol = item

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
        uV = Edge.unitizeVector(dV)
        dot = sum([i*j for (i, j) in zip(uV, faceNormal)])
        ang = math.degrees(math.acos(dot))
        if dot < tolerance:
            return [False, ang]
        return [True, ang]
    
    @staticmethod
    def FaceFlatten(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        flat_item : TYPE
            DESCRIPTION.

        """
        origin = topologic.Vertex.ByCoordinates(0,0,0)
        cm = item.CenterOfMass()
        coords = topologic.FaceUtility.NormalAtParameters(item, 0.5, 0.5)
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
    def FaceGridByDistances(face, uRange, vRange, uOrigin, vOrigin, clip):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        uRange : TYPE
            DESCRIPTION.
        vRange : TYPE
            DESCRIPTION.
        uOrigin : TYPE
            DESCRIPTION.
        vOrigin : TYPE
            DESCRIPTION.
        clip : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # face = item[0]
        # uRange = item[1]
        # vRange = item[2]
        # uOrigin = item[3]
        # vOrigin = item[4]
        # clip = item[5]
        
        def multiplyVector(vector, mag, tol):
            oldMag = 0
            for value in vector:
                oldMag += value ** 2
            oldMag = oldMag ** 0.5
            if oldMag < tol:
                return [0,0,0]
            newVector = []
            for i in range(len(vector)):
                newVector.append(vector[i] * mag / oldMag)
            return newVector
        
        if isinstance(clip, list):
            clip = clip[0]
        uvWireEdges = []
        uCluster = None
        vCluster = None
        uvWire = None
        v1 = topologic.FaceUtility.VertexAtParameters(face, 0, 0)
        v2 = topologic.FaceUtility.VertexAtParameters(face, 1, 0)
        uVector = [v2.X()-v1.X(), v2.Y()-v1.Y(),v2.Z()-v1.Z()]
        v1 = topologic.FaceUtility.VertexAtParameters(face, 0, 0)
        v2 = topologic.FaceUtility.VertexAtParameters(face, 0, 1)
        vVector = [v2.X()-v1.X(), v2.Y()-v1.Y(),v2.Z()-v1.Z()]
        if len(uRange) > 0:
            uRange.sort()
            uRangeEdges = []
            uuVector = Edge.unitizeVector(uVector)
            for u in uRange:
                tempVec = multiplyVector(uuVector, u, 0.0001)
                v1 = topologic.Vertex.ByCoordinates(uOrigin.X()+tempVec[0], uOrigin.Y()+tempVec[1], uOrigin.Z()+tempVec[2])
                v2 = topologic.Vertex.ByCoordinates(v1.X()+vVector[0], v1.Y()+vVector[1], v1.Z()+vVector[2])
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                uRangeEdges.append(e)
                uvWireEdges.append(e)
            if len(uRangeEdges) > 0:
                uCluster = topologic.Cluster.ByTopologies(uRangeEdges)
                if clip:
                    uCluster = uCluster.Intersect(face, False)
        if len(vRange) > 0:
            vRange.sort()
            vRangeEdges = []
            uvVector = Edge.unitizeVector(vVector)
            for v in vRange:
                tempVec = multiplyVector(uvVector, v, 0.0001)
                v1 = topologic.Vertex.ByCoordinates(vOrigin.X()+tempVec[0], vOrigin.Y()+tempVec[1], vOrigin.Z()+tempVec[2])
                v2 = topologic.Vertex.ByCoordinates(v1.X()+uVector[0], v1.Y()+uVector[1], v1.Z()+uVector[2])
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                vRangeEdges.append(e)
                uvWireEdges.append(e)
            if len(vRangeEdges) > 0:
                vCluster = topologic.Cluster.ByTopologies(vRangeEdges)
                if clip:
                    vCluster = vCluster.Intersect(face, False)
        if len(uvWireEdges) > 0 and uCluster and vCluster:
            uvWire = uCluster.Merge(vCluster)
        return [uCluster, vCluster, uvWire]
    
    @staticmethod
    def FaceGridByParameters(face, uRange, vRange, clip):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        uRange : TYPE
            DESCRIPTION.
        vRange : TYPE
            DESCRIPTION.
        clip : TYPE
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
        # face = item[0]
        # uRange = item[1]
        # vRange = item[2]
        # clip = item[3]
        if isinstance(clip, list):
            clip = clip[0]
        uvWireEdges = []
        uCluster = None
        vCluster = None
        uvWire = None
        if len(uRange) > 0:
            if (min(uRange) < 0) or (max(uRange) > 1):
                raise Exception("Face.GridByParameters - Error: uRange input values are outside acceptable range (0,1)")
            uRange.sort()
            uRangeEdges = []
            for u in uRange:
                v1 = topologic.FaceUtility.VertexAtParameters(face, u, 0)
                v2 = topologic.FaceUtility.VertexAtParameters(face, u, 1)
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                uRangeEdges.append(e)
                uvWireEdges.append(e)
            if len(uRangeEdges) > 0:
                uCluster = topologic.Cluster.ByTopologies(uRangeEdges)
                if clip:
                    uCluster = uCluster.Intersect(face, False)
        if len(vRange) > 0:
            if (min(vRange) < 0) or (max(vRange) > 1):
                raise Exception("Face.GridByParameters - Error: vRange input values are outside acceptable range (0,1)")
            vRange.sort()
            vRangeEdges = []
            for v in vRange:
                v1 = topologic.FaceUtility.VertexAtParameters(face, 0, v)
                v2 = topologic.FaceUtility.VertexAtParameters(face, 1, v)
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                vRangeEdges.append(e)
                uvWireEdges.append(e)
            if len(vRangeEdges) > 0:
                vCluster = topologic.Cluster.ByTopologies(vRangeEdges)
                if clip:
                    vCluster = vCluster.Intersect(face, False)
        if len(uvWireEdges) > 0 and uCluster and vCluster:
            uvWire = uCluster.Merge(vCluster)
        return [uCluster, vCluster, uvWire]
    
    @staticmethod
    def FaceInternalBoundaries(item):
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
        wires = []
        _ = item.InternalBoundaries(wires)
        return list(wires)

    @staticmethod
    def FaceInternalVertex(face, tolerance=0.0001):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # face = item[0]
        # tol = item[1]
        return topologic.FaceUtility.InternalVertex(face, tolerance)
    
    @staticmethod
    def FaceIsCoplanar(faceA, faceB, tolerance=0.0001):
        """
        Parameters
        ----------
        faceA : TYPE
            DESCRIPTION.
        faceB : TYPE
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
        # faceA, faceB, tol = item
        
        def collinear(v1, v2, tolerance):
            ang = Face.angle_between(v1, v2)
            if math.isnan(ang) or math.isinf(ang):
                raise Exception("Face.IsCollinear - Error: Could not determine the angle between the input faces")
            elif abs(ang) < tolerance or abs(pi - ang) < tolerance:
                return True
            return False
        
        if not faceA or not isinstance(faceA, topologic.Face):
            raise Exception("Face.IsCoplanar - Error: Face A is not valid")
        if not faceB or not isinstance(faceB, topologic.Face):
            raise Exception("Face.IsCoplanar - Error: Face B is not valid")
        dirA = Face.FaceNormalAtParameters(faceA, 0.5, 0.5, "XYZ", 3)
        dirB = Face.FaceNormalAtParameters(faceB, 0.5, 0.5, "XYZ", 3)
        return collinear(dirA, dirB, tolerance)
    
    @staticmethod
    def FaceIsInside(topology, vertex, tolerance=0.0001):
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
        if topology.Type() == topologic.Face.Type():
            status = (topologic.FaceUtility.IsInside(topology, vertex, tolerance))
        return status

    @staticmethod
    def FaceNormalAtParameters(face, u, v, outputType, decimals):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        outputType : TYPE
            DESCRIPTION.
        decimals : TYPE
            DESCRIPTION.

        Returns
        -------
        returnResult : TYPE
            DESCRIPTION.

        """
        # face, u, v = item
        try:
            coords = topologic.FaceUtility.NormalAtParameters(face, u, v)
            x = round(coords[0], decimals)
            y = round(coords[1], decimals)
            z = round(coords[2], decimals)
            returnResult = []
            if outputType == "XYZ":
                returnResult = [x,y,z]
            elif outputType == "XY":
                returnResult = [x,y]
            elif outputType == "XZ":
                returnResult = [x,z]
            elif outputType == "YZ":
                returnResult = [y,z]
            elif outputType == "X":
                returnResult = x
            elif outputType == "Y":
                returnResult = y
            elif outputType == "Z":
                returnResult = z
        except:
            returnResult = None
        return returnResult
    
    @staticmethod
    def FaceTrimByWire(face, wire, reverseWire):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        wire : TYPE
            DESCRIPTION.
        reverseWire : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # face = item[0]
        # wire = item[1]
        # reverseWire = item[2]
        return topologic.FaceUtility.TrimByWire(face, wire, reverseWire)
    
    @staticmethod
    def FaceVertexByParameters(face, u, v):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.

        Returns
        -------
        vertex : TYPE
            DESCRIPTION.

        """
        # face = item[0]
        # u = item[1]
        # v = item[2]
        vertex = topologic.FaceUtility.VertexAtParameters(face, u, v)
        return vertex
    
    @staticmethod
    def FaceVertexParameters(face, vertex):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        vertex : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # face = item[0]
        # vertex = item[1]
        params = topologic.FaceUtility.ParametersAtVertex(face, vertex)
        return [params[0], params[1]]
