import topologic
import warnings
import Cell
import math
import Wire

class CellComplex(topologic.CellComplex):
    @staticmethod
    def CellComplexByCells(cells, tolerance=0.0001):
        """
        Parameters
        ----------
        icells : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # cells, tol = item
        assert isinstance(cells, list), "CellComplex.ByCells - Error: Input is not a list"
        cells = [x for x in cells if isinstance(x, topologic.Cell)]
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
                    return returnCellComplexes
                else:
                    return None
        else:
            return cellComplex
    
    @staticmethod
    def CellComplexByFaces(faces, tolerance=0.0001):
        """
        Parameters
        ----------
        faces : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # faces, tol = item
        assert isinstance(faces, list), "CellComplex.ByFaces - Error: Input is not a list"
        faces = [x for x in faces if isinstance(x, topologic.Face)]
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
                    return returnCellComplexes
                else:
                    return None
        else:
            return cellComplex
    
    @staticmethod
    def CellComplexByLoft(wires, tolerance=0.0001):
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
                raise Exception("Shell.ByLoft - Error: The two wires do not have the same number of edges.")
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
        return CellComplex.CellComplexByFaces(faces, tolerance)
    
    @staticmethod
    def CellComplexDecompose(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        
        def getApertures(topology):
            apertures = []
            apTopologies = []
            _ = topology.Apertures(apertures)
            for aperture in apertures:
                apTopologies.append(topologic.Aperture.Topology(aperture))
            return apTopologies
        
        def flatten(element):
            returnList = []
            if isinstance(element, list) == True:
                for anItem in element:
                    returnList = returnList + flatten(anItem)
            else:
                returnList = [element]
            return returnList

        externalVerticalFaces = []
        internalVerticalFaces = []
        topHorizontalFaces = []
        bottomHorizontalFaces = []
        internalHorizontalFaces = []
        externalVerticalApertures = []
        internalVerticalApertures = []
        topHorizontalApertures = []
        bottomHorizontalApertures = []
        internalHorizontalApertures = []

        faces = []
        _ = item.Faces(None, faces)
        for aFace in faces:
            z = topologic.FaceUtility.NormalAtParameters(aFace, 0.5, 0.5)[2]
            cells = []
            aFace.Cells(item, cells)
            n = len(cells)
            if abs(z) < 0.001:
                if n == 1:
                    externalVerticalFaces.append(aFace)
                    externalVerticalApertures.append(getApertures(aFace))
                else:
                    internalVerticalFaces.append(aFace)
                    internalVerticalApertures.append(getApertures(aFace))
            elif n == 1:
                if z > 0.9:
                    topHorizontalFaces.append(aFace)
                    topHorizontalApertures.append(getApertures(aFace))
                elif z < -0.9:
                    bottomHorizontalFaces.append(aFace)
                    bottomHorizontalApertures.append(getApertures(aFace))

            else:
                internalHorizontalFaces.append(aFace)
                internalHorizontalApertures.append(getApertures(aFace))
        return1 = []
        return2 = []
        return3 = []
        return4 = []
        return5 = []
        return6 = []
        return7 = []
        return8 = []
        return9 = []
        return10 = []
        if len(externalVerticalFaces) > 0:
            return1 = flatten(externalVerticalFaces)
        if len(internalVerticalFaces) > 0:
            return2 = flatten(internalVerticalFaces)
        if len(topHorizontalFaces) > 0:
            return3 = flatten(topHorizontalFaces)
        if len(bottomHorizontalFaces) > 0:
            return4 = flatten(bottomHorizontalFaces)
        if len(internalHorizontalFaces) > 0:
            return5 = flatten(internalHorizontalFaces)
        if len(externalVerticalApertures) > 0:
            return6 = flatten(externalVerticalApertures)
        if len(internalVerticalApertures) > 0:
            return7 = flatten(internalVerticalApertures)
        if len(topHorizontalApertures) > 0:
            return8 = flatten(topHorizontalApertures)
        if len(bottomHorizontalApertures) > 0:
            return9 = flatten(bottomHorizontalApertures)
        if len(internalHorizontalApertures) > 0:
            return10 = flatten(internalHorizontalApertures)

        return [return1, return2, return3, return4, return5, return6, return7, return8, return9, return10]
    
    @staticmethod
    def CellComplexExternalBoundary(item):
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
    def CellComplexInternalBoundary(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        faces : TYPE
            DESCRIPTION.

        """
        faces = []
        _ = item.InternalBoundaries(faces)
        return faces
    
    @staticmethod
    def CellComplexNonManifoldFaces(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        faces : TYPE
            DESCRIPTION.

        """
        faces = []
        _ = item.NonManifoldFaces(faces)
        return faces
    
    @staticmethod
    def CellComplexPrism(origin, width, length, height, uSides, vSides, wSides,
                         dirX, dirY, dirZ, originLocation):
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
        originLocation : TYPE
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
        # originLocation = item
        
        def sliceCell(cell, width, length, height, uSides, vSides, wSides):
            origin = cell.Centroid()
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
            cellFaces = []
            _ = cell.Faces(None, cellFaces)
            sliceFaces = sliceFaces + cellFaces
            cellComplex = topologic.CellComplex.ByFaces(sliceFaces, 0.0001)
            return cellComplex
        
        baseV = []
        topV = []
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if originLocation == "Center":
            zOffset = -height*0.5
        elif originLocation == "LowerLeft":
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
    
    
