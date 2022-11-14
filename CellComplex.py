import topologic
import warnings
import math

class CellComplex(topologic.CellComplex):
    @staticmethod
    def ByCells(cells, tolerance=0.0001):
        """
        Description
        -----------
        Creates a cellcomplex by merging the input cells.

        Parameters
        ----------
        cells : topologic.Cell
            The input cells.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        if not cells:
            return None
        if not isinstance(cells, list):
            return None
        cells = [x for x in cells if isinstance(x, topologic.Cell)]
        if len(cells) < 1:
            return None
        cellComplex = topologic.CellComplex.ByCells(cells, tolerance)
        if not cellComplex:
            warnings.warn("Warning: Default CellComplex.ByCells method failed. Attempting to Merge the Cells.", UserWarning)
            result = cells[0]
            remainder = cells[1:]
            cluster = topologic.Cluster.ByTopologies(remainder, False)
            result = result.Merge(cluster, False)
            if result.Type() != 64: #64 is the type of a CellComplex
                warnings.warn("Warning: Input Cells do not form a CellComplex", UserWarning)
                if result.Type() > 64:
                    returnCellComplexes = []
                    _ = result.CellComplexes(None, returnCellComplexes)
                    return returnCellComplexes[0]
                else:
                    return None
        else:
            return cellComplex
    @staticmethod
    def ByCellsCluster(cluster, tolerance=0.0001):
        """
        Description
        -----------
        Creates a cellcomplex by merging the cells within the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of cells.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        if not cluster:
            return None
        if not isinstance(cluster, topologic.Cluster):
            return None
        cells = []
        _ = cluster.Cells(None, cells)
        return CellComplex.ByCells(cells, tolerance)

    @staticmethod
    def ByFaces(faces, tolerance=0.0001):
        """
        Description
        -----------
        Creates a cellcomplex by merging the input faces.

        Parameters
        ----------
        faces : topologic.Face
            The input faces.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        if not faces:
            return None
        if not isinstance(faces, list):
            return None
        faces = [x for x in faces if isinstance(x, topologic.Face)]
        if len(faces) < 1:
            return None
        cellComplex = topologic.CellComplex.ByFaces(faces, tolerance, False)
        if not cellComplex:
            warnings.warn("Warning: Default CellComplex.ByFaces method failed. Attempting to Merge the Faces.", UserWarning)
            cellComplex = faces[0]
            for i in range(1,len(faces)):
                newCellComplex = None
                try:
                    newCellComplex = cellComplex.Merge(faces[i], False)
                except:
                    warnings.warn("Warning: Failed to merge Face #"+i+". Skipping.", UserWarning)
                if newCellComplex:
                    cellComplex = newCellComplex
            if cellComplex.Type() != 64: #64 is the type of a CellComplex
                warnings.warn("Warning: Input Faces do not form a CellComplex", UserWarning)
                if cellComplex.Type() > 64:
                    returnCellComplexes = []
                    _ = cellComplex.CellComplexes(None, returnCellComplexes)
                    return returnCellComplexes[0]
                else:
                    return None
        else:
            return cellComplex
    
    @staticmethod
    def ByFacesCluster(cluster, tolerance=0.0001):
        """
        Description
        -----------
        Creates a cellcomplex by merging the faces within the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of faces.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        if not cluster:
            return None
        if not isinstance(cluster, topologic.Cluster):
            return None
        faces = []
        _ = cluster.Faces(None, faces)
        return CellComplex.ByFaces(faces, tolerance)

    @staticmethod
    def ByLoft(wires, tolerance=0.0001):
        """
        Description
        -----------
        Creates a cellcomplex by lofting through the input wires.

        Parameters
        ----------
        wires : topologic.Wire
            The input wires.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        faces = [topologic.Face.ByExternalBoundary(wires[0])]
        for i in range(len(wires)-1):
            wire1 = wires[i]
            wire2 = wires[i+1]
            faces.append(topologic.Face.ByExternalBoundary(wire2))
            w1_edges = []
            _ = wire1.Edges(None, w1_edges)
            w2_edges = []
            _ = wire2.Edges(None, w2_edges)
            if len(w1_edges) != len(w2_edges):
                return None
            for j in range (len(w1_edges)):
                e1 = w1_edges[j]
                e2 = w2_edges[j]
                e3 = None
                e4 = None
                try:
                    e3 = topologic.Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.StartVertex())
                except:
                    try:
                        e4 = topologic.Edge.ByStartVertexEndVertex(e1.EndVertex(), e2.EndVertex())
                        faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e1, e2, e4])))
                    except:
                        pass
                try:
                    e4 = topologic.Edge.ByStartVertexEndVertex(e1.EndVertex(), e2.EndVertex())
                except:
                    try:
                        e3 = topologic.Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.StartVertex())
                        faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e1, e2, e3])))
                    except:
                        pass
                if e3 and e4:
                    e5 = topologic.Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.EndVertex())
                    faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e1, e5, e4])))
                    faces.append(topologic.Face.ByExternalBoundary(topologic.Wire.ByEdges([e2, e5, e3])))
        return CellComplex.ByFaces(faces, tolerance)

    @staticmethod
    def ByLoftCluster(cluster, tolerance=0.0001):
        """
        Description
        -----------
        Creates a cellcomplex by lofting through the wires in the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of wires.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        if not cluster:
            return None
        if not isinstance(cluster, topologic.Cluster):
            return None
        wires = []
        _ = cluster.Wires(None, wires)
        return CellComplex.ByLoft(wires, tolerance)

    @staticmethod
    def Cells(cellComplex):
        """
        Description
        __________
        Returns the cells of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of cells.

        """
        if not isinstance(cellComplex, topologic.CellComplex):
            return None
        cells = []
        _ = cellComplex.Cells(None, cells)
        return cells

    @staticmethod
    def Decompose(cellComplex, tiltAngle=10, tolerance=0.0001):
        """
        Description
        __________
        Decomposes the input cellComplex into its logical components. This method assumes that the positive Z direction is UP.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            the input cellComplex.
        tiltAngle : float , optional
            The threshold tilt angle in degrees to determine if a face is vertical, horizontal, or tilted. The tilt angle is measured from the nearest cardinal direction. The default is 10.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dictionary
            A dictionary with the following keys and values:
            1. "externalVerticalFaces": list of external vertical faces
            2. "internalVerticalFaces": list of internal vertical faces
            3. "topHorizontalFaces": list of top horizontal faces
            4. "bottomHorizontalFaces": list of bottom horizontal faces
            5. "internalHorizontalFaces": list of internal horizontal faces
            6. "externalInclinedFaces": list of external inclined faces
            7. "internalInclinedFaces": list of internal inclined faces
            8. "externalVerticalApertures": list of external vertical apertures
            9. "internalVerticalApertures": list of internal vertical apertures
            10. "topHorizontalApertures": list of top horizontal apertures
            11. "bottomHorizontalApertures": list of bottom horizontal apertures
            12. "internalHorizontalApertures": list of internal horizontal apertures
            13. "externalInclinedApertures": list of external inclined apertures
            14. "internalInclinedApertures": list of internal inclined apertures

        """
        from topologicpy.Face import Face
        from topologicpy.Vector import Vector
        from topologicpy.Aperture import Aperture
        from numpy import arctan, pi, signbit, arctan2, rad2deg

        def angleCode(f, up, tiltAngle):
            dirA = Face.NormalAtParameters(f)
            ang = round(Vector.Angle(dirA, up), 2)
            if abs(ang - 90) < tiltAngle:
                code = 0
            elif abs(ang) < tiltAngle:
                code = 1
            elif abs(ang - 180) < tiltAngle:
                code = 2
            else:
                code = 3
            return code

        def getApertures(topology):
            apertures = []
            apTopologies = []
            apertures = topology.Apertures(apertures)
            for aperture in apertures:
                apTopologies.append(Aperture.Topology(aperture))
            return apTopologies

        if not isinstance(cellComplex, topologic.CellComplex):
            return None
        externalVerticalFaces = []
        internalVerticalFaces = []
        topHorizontalFaces = []
        bottomHorizontalFaces = []
        internalHorizontalFaces = []
        externalInclinedFaces = []
        internalInclinedFaces = []
        externalVerticalApertures = []
        internalVerticalApertures = []
        topHorizontalApertures = []
        bottomHorizontalApertures = []
        internalHorizontalApertures = []
        externalInclinedApertures = []
        internalInclinedApertures = []
        tiltAngle = abs(tiltAngle)
        faces = CellComplex.Faces(cellComplex)
        zList = []
        for f in faces:
            zList.append(f.Centroid().Z())
        zMin = min(zList)
        zMax = max(zList)
        up = [0,0,1]
        for aFace in faces:
            aCode = angleCode(aFace, up, tiltAngle)
            cells = []
            aFace.Cells(cellComplex, cells)
            n = len(cells)
            if aCode == 0:
                if n == 1:
                    externalVerticalFaces.append(aFace)
                    externalVerticalApertures += getApertures(aFace)
                else:
                    internalVerticalFaces.append(aFace)
                    internalVerticalApertures += getApertures(aFace)
            elif aCode == 1:
                if n == 1:
                    if abs(aFace.Centroid().Z() - zMin) < tolerance:
                        bottomHorizontalFaces.append(aFace)
                        bottomHorizontalApertures += getApertures(aFace)
                    else:
                        topHorizontalFaces.append(aFace)
                        topHorizontalApertures += getApertures(aFace)
                else:
                    internalHorizontalFaces.append(aFace)
                    internalHorizontalApertures += getApertures(aFace)
            elif aCode == 2:
                if n == 1:
                    if abs(aFace.Centroid().Z() - zMax) < tolerance:
                        topHorizontalFaces.append(aFace)
                        topHorizontalApertures += getApertures(aFace)
                    else:
                        bottomHorizontalFaces.append(aFace)
                        bottomHorizontalApertures += getApertures(aFace)
                else:
                    internalHorizontalFaces.append(aFace)
                    internalHorizontalApertures += getApertures(aFace)
            elif aCode == 3:
                if n == 1:
                    externalInclinedFaces.append(aFace)
                    externalInclinedApertures += getApertures(aFace)
                else:
                    internalInclinedFaces.append(aFace)
                    internalInclinedApertures += getApertures(aFace)
        
        d = {
            "externalVerticalFaces" : externalVerticalFaces,
            "internalVerticalFaces" : internalVerticalFaces,
            "topHorizontalFaces" : topHorizontalFaces,
            "bottomHorizontalFaces" : bottomHorizontalFaces,
            "internalHorizontalFaces" : internalHorizontalFaces,
            "externalInclinedFaces" : externalInclinedFaces,
            "internalInclinedFaces" : internalInclinedFaces,
            "externalVerticalApertures" : externalVerticalApertures,
            "internalVerticalApertures" : internalVerticalApertures,
            "topHorizontalApertures" : topHorizontalApertures,
            "bottomHorizontalApertures" : bottomHorizontalApertures,
            "internalHorizontalApertures" : internalHorizontalApertures,
            "externalInclinedApertures" : externalInclinedApertures,
            "internalInclinedApertures" : internalInclinedApertures
            }
        return d
    
    @staticmethod
    def Edges(cellComplex):
        """
        Description
        __________
        Returns the edges of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of edges.

        """ 
        if not isinstance(cellComplex, topologic.CellComplex):
            return None
        edges = []
        _ = cellComplex.Edges(None, edges)
        return edges

    @staticmethod
    def ExternalBoundary(cellComplex):
        """
        Description
        __________
        Returns the external boundary (cell) of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        topologic.Cell
            The external boundary of the input cellComplex.

        """
        return cellComplex.ExternalBoundary()

    @staticmethod
    def Faces(cellComplex):
        """
        Description
        __________
        Returns the faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of faces.

        """
        if not isinstance(cellComplex, topologic.CellComplex):
            return None
        faces = []
        _ = cellComplex.Faces(None, faces)
        return faces

    @staticmethod
    def InternalBoundary(cellComplex):
        """
        Description
        __________
        Returns the internal boundaries (faces) of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of internal boundaries (faces) of the input cellComplex.

        """
        faces = []
        _ = cellComplex.InternalBoundaries(faces)
        return faces
    
    @staticmethod
    def NonManifoldFaces(cellComplex):
        """
        Description
        __________
        Returns the non-manifold faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        ist
            The list of non-manifold faces of the input cellComplex.

        """
        faces = []
        _ = cellComplex.NonManifoldFaces(faces)
        return faces
    
    @staticmethod
    def Prism(origin=None, width=1, length=1, height=1, uSides=2, vSides=2, wSides=2,
                         dirX=0, dirY=0, dirZ=1, placement="bottom"):
        """
        Description
        ----------
        Creates a prismatic cellComplex with internal cells.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The origin location of the prism. The default is None which results in the prism being placed at (0,0,0).
        width : float, optional
            The width of the prism. The default is 1.
        length : float, optional
            The length of the prism. The default is 1.
        height : float, optional
            The height of the prism.
        uSides : int, optional
            The number of sides along the width. The default is 1.
        vSides : int, optional
            The number of sides along the length. The default is 1.
        wSides : int, optional
            The number of sides along the height. The default is 1.
        dirX : float, optional
            The X component of the vector representing the up direction of the prism. The default is 0.
        dirY : float, optional
            The Y component of the vector representing the up direction of the prism. The default is 0.
        dirZ : float, optional
            The Z component of the vector representing the up direction of the prism. The default is 1.
        placement : str, optional
            The description of the placement of the origin of the prism. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "bottom".

        Returns
        -------
        topologic.CellComplex
            The created prism.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        if not isinstance(origin, topologic.Vertex):
            origin = Vertex.ByCoordinates(0,0,0)

        uOffset = float(width) / float(uSides)
        vOffset = float(length) / float(vSides)
        wOffset = float(height) / float(wSides)
        if placement.lower() == "center":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "bottom":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = 0
        else:
            xOffset = 0
            yOffset = 0
            zOffset = 0
        cells = []
        for i in range(uSides):
            for j in range(vSides):
                for k in range(wSides):
                    cOrigin = Vertex.ByCoordinates(i*uOffset - xOffset, j*vOffset - yOffset, k*wOffset - zOffset)
                    cells.append(Cell.Prism(cOrigin, width=uOffset, length=vOffset, height=wOffset, placement="lowerleft"))
        prism = CellComplex.ByCells(cells)
        if prism:
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
            prism = Topology.Rotate(prism, origin, 0, 1, 0, theta)
            prism = Topology.Rotate(prism, origin, 0, 0, 1, phi)
            return prism
        else:
            return None

    @staticmethod
    def Shells(cellComplex):
        """
        Description
        __________
        Returns the shells of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of shells.

        """
        if not isinstance(cellComplex, topologic.CellComplex):
            return None
        shells = []
        _ = cellComplex.Shells(None, shells)
        return shells

    @staticmethod
    def Vertices(cellComplex):
        """
        Description
        __________
        Returns the vertices of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(cellComplex, topologic.CellComplex):
            return None
        vertices = []
        _ = cellComplex.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Wires(cellComplex):
        """
        Description
        __________
        Returns the wires of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(cellComplex, topologic.CellComplex):
            return None
        wires = []
        _ = cellComplex.Wires(None, wires)
        return wires

