# Copyright (C) 2024
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import topologic_core as topologic
import math
import os
from topologicpy.Topology import Topology
import warnings

try:
    import numpy as np
except:
    print("CellComplex - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("CellComplex - numpy library installed correctly.")
    except:
        warnings.warn("CellComplex - Error: Could not import numpy.")
try:
    from scipy.spatial import Delaunay
    from scipy.spatial import Voronoi
except:
    print("CellComplex - Install required scipy library.")
    try:
        os.system("pip install scipy")
    except:
        os.system("pip install scipy --user")
    try:
        from scipy.spatial import Delaunay
        from scipy.spatial import Voronoi
    except:
        warnings.warn("CellComplex - Error: Could not import scipy.")

class CellComplex(Topology):
    @staticmethod
    def Box(origin: topologic.Vertex = None,
            width: float = 1.0, length: float = 1.0, height: float = 1.0,
            uSides: int = 2, vSides: int = 2, wSides: int = 2,
            direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Creates a box with internal cells.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin location of the box. The default is None which results in the box being placed at (0, 0, 0).
        width : float , optional
            The width of the box. The default is 1.
        length : float , optional
            The length of the box. The default is 1.
        height : float , optional
            The height of the box.
        uSides : int , optional
            The number of sides along the width. The default is 1.
        vSides : int, optional
            The number of sides along the length. The default is 1.
        wSides : int , optional
            The number of sides along the height. The default is 1.
        direction : list , optional
            The vector representing the up direction of the box. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the box. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.CellComplex
            The created box.

        """
        return CellComplex.Prism(origin=origin,
                                 width=width, length=length, height=height,
                                 uSides=uSides, vSides=vSides, wSides=wSides,
                                 direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def ByCells(cells: list, tolerance: float = 0.0001, silent: bool = False) -> topologic.CellComplex:
        """
        Creates a cellcomplex by merging the input cells.

        Parameters
        ----------
        cells : list
            The list of input cells.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(cells, list):
            if not silent:
                print("CellComplex.ByCells - Error: The input cells parameter is not a valid list. Returning None.")
            return None
        cells = [x for x in cells if isinstance(x, topologic.Cell)]
        if len(cells) < 1:
            if not silent:
                print("CellComplex.ByCells - Error: The input cells parameter does not contain any valid cells. Returning None.")
            return None
        cellComplex = None
        if len(cells) == 1:
            return topologic.CellComplex.ByCells(cells)
        else:
            try:
                cellComplex = topologic.CellComplex.ByCells(cells)
            except:
                topA = cells[0]
                topB = Cluster.ByTopologies(cells[1:])
                cellComplex = Topology.Merge(topA, topB, tranDict=False, tolerance=tolerance)
        
        if not isinstance(cellComplex, topologic.CellComplex):
            if not silent:
                print("CellComplex.ByCells - Warning: Could not create a CellComplex. Returning object of type topologic.Cluster instead of topologic.CellComplex.")
            return Cluster.ByTopologies(cells)
        else:
            temp_cells = CellComplex.Cells(cellComplex)
            if not isinstance(temp_cells, list):
                if not silent:
                    print("CellComplex.ByCells - Error: The resulting object does not contain any cells. Returning None.")
                return None
            elif len(temp_cells) < 1:
                if silent:
                    print("CellComplex.ByCells - Error: Could not create a CellComplex. Returning None.")
                return None
            elif len(temp_cells) == 1:
                if not silent:
                    print("CellComplex.ByCells - Warning: Resulting object contains only one cell. Returning object of type topologic.Cell instead of topologic.CellComplex.")
                return(temp_cells[0])
        return cellComplex
    
    @staticmethod
    def ByCellsCluster(cluster: topologic.Cluster, tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Creates a cellcomplex by merging the cells within the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of cells.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """

        if not isinstance(cluster, topologic.Cluster):
            print("CellComplex.ByCellsCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        cells = []
        _ = cluster.Cells(None, cells)
        return CellComplex.ByCells(cells, tolerance)

    @staticmethod
    def ByFaces(faces: list, tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Creates a cellcomplex by merging the input faces.

        Parameters
        ----------
        faces : topologic.Face
            The input faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """

        if not isinstance(faces, list):
            print("CellComplex.ByFaces - Error: The input faces parameter is not a valid list. Returning None.")
            return None
        faces = [x for x in faces if isinstance(x, topologic.Face)]
        if len(faces) < 1:
            print("CellComplex.ByFaces - Error: The input faces parameter does not contain any valid faces. Returning None.")
            return None
        try:
            cellComplex = topologic.CellComplex.ByFaces(faces, tolerance, False)
        except:
            cellComplex = None
        if not cellComplex:
            print("CellComplex.ByFaces - Warning: The default method failed. Attempting a workaround.")
            cellComplex = faces[0]
            for i in range(1,len(faces)):
                newCellComplex = None
                try:
                    newCellComplex = cellComplex.Merge(faces[i], False, tolerance)
                except:
                    print("CellComplex.ByFaces - Warning: Failed to merge face #"+str(i)+". Skipping.")
                if newCellComplex:
                    cellComplex = newCellComplex
            if cellComplex.Type() != 64: #64 is the type of a CellComplex
                print("CellComplex.ByFaces - Warning: The input faces do not form a cellcomplex")
                if cellComplex.Type() > 64:
                    returnCellComplexes = []
                    _ = cellComplex.CellComplexes(None, returnCellComplexes)
                    if len(returnCellComplexes) > 0:
                        return returnCellComplexes[0]
                    else:
                        print("CellComplex.ByFaces - Error: Could not create a cellcomplex. Returning None.")
                        return None
                else:
                    print("CellComplex.ByFaces - Error: Could not create a cellcomplex. Returning None.")
                    return None
        else:
            return cellComplex
    
    @staticmethod
    def ByFacesCluster(cluster: topologic.Cluster, tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Creates a cellcomplex by merging the faces within the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """

        if not isinstance(cluster, topologic.Cluster):
            print("CellComplex.ByFacesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        faces = []
        _ = cluster.Faces(None, faces)
        return CellComplex.ByFaces(faces, tolerance)

    @staticmethod
    def ByWires(wires: list, triangulate: bool = True, tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Creates a cellcomplex by lofting through the input wires.

        Parameters
        ----------
        wires : list
            The input list of wires. The list should contain a minimum of two wires. All wires must have the same number of edges.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(wires, list):
            print("CellComplex.ByFaces - Error: The input wires parameter is not a valid list. Returning None.")
            return None
        wires = [x for x in wires if isinstance(x, topologic.Wire)]
        if len(wires) < 2:
            print("CellComplex.ByWires - Error: The input wires parameter contains less than two valid wires. Returning None.")
            return None
        faces = [Face.ByWire(wires[0], tolerance=tolerance), Face.ByWire(wires[-1], tolerance=tolerance)]
        if triangulate == True:
            triangles = []
            for face in faces:
                if len(Topology.Vertices(face)) > 3:
                    triangles += Face.Triangulate(face, tolerance=tolerance)
                else:
                    triangles += [face]
            faces = triangles
        for i in range(len(wires)-1):
            wire1 = wires[i]
            wire2 = wires[i+1]
            f = Face.ByWire(wire2, tolerance=tolerance)
            if triangulate == True:
                if len(Topology.Vertices(face)) > 3:
                    triangles = Face.Triangulate(face, tolerance=tolerance)
                else:
                    triangles = [face]
                faces += triangles
            else:
                faces.append(f)
            w1_edges = []
            _ = wire1.Edges(None, w1_edges)
            w2_edges = []
            _ = wire2.Edges(None, w2_edges)
            if len(w1_edges) != len(w2_edges):
                print("CellComplex.ByWires - Error: The input wires parameter contains wires with different number of edges. Returning None.")
                return None
            for j in range (len(w1_edges)):
                e1 = w1_edges[j]
                e2 = w2_edges[j]
                e3 = None
                e4 = None
                try:
                    e3 = Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.StartVertex(), tolerance=tolerance, silent=True)
                except:
                    try:
                        e4 = Edge.ByStartVertexEndVertex(e1.EndVertex(), e2.EndVertex(), tolerance=tolerance, silent=True)
                        f = Face.ByExternalBoundary(Wire.ByEdges([e1, e2, e4], tolerance=tolerance))
                        if triangulate == True:
                            if len(Topology.Vertices(face)) > 3:
                                triangles = Face.Triangulate(face, tolerance=tolerance)
                            else:
                                triangles = [face]
                            faces += triangles
                        else:
                            faces.append(f)
                    except:
                        pass
                try:
                    e4 = Edge.ByStartVertexEndVertex(e1.EndVertex(), e2.EndVertex(), tolerance=tolerance, silent=True)
                except:
                    try:
                        e3 = Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.StartVertex(), tolerance=tolerance, silent=True)
                        f = Face.ByWire(Wire.ByEdges([e1, e2, e3], tolerance=tolerance), tolerance=tolerance)
                        if triangulate == True:
                            if len(Topology.Vertices(face)) > 3:
                                triangles = Face.Triangulate(face, tolerance=tolerance)
                            else:
                                triangles = [face]
                            faces += triangles
                        else:
                            faces.append(f)
                    except:
                        pass
                if e3 and e4:
                    if triangulate == True:
                        e5 = Edge.ByStartVertexEndVertex(e1.StartVertex(), e2.EndVertex(), tolerance=tolerance, silent=True)
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e5, e4], tolerance=tolerance), tolerance=tolerance))
                        faces.append(Face.ByWire(Wire.ByEdges([e2, e5, e3], tolerance=tolerance), tolerance=tolerance))
                    else:
                        f = Face.ByWire(Wire.ByEdges([e1, e4, e2, e3], tolerance=tolerance), tolerance=tolerance) or Face.ByWire(Wire.ByEdges([e1, e3, e2, e4], tolerance=tolerance), tolerance=tolerance)
                        if f:
                            faces.append(f)

                elif e3:
                    faces.append(Face.ByWire(Wire.ByEdges([e1, e3, e2], tolerance=tolerance), tolerance=tolerance))
                elif e4:
                    faces.append(Face.ByWire(Wire.ByEdges([e1, e4, e2], tolerance=tolerance), tolerance=tolerance))
        return CellComplex.ByFaces(faces, tolerance=tolerance)

    @staticmethod
    def ByWiresCluster(cluster: topologic.Cluster, triangulate: bool = True, tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Creates a cellcomplex by lofting through the wires in the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of wires.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellcomplex.

        """

        if not isinstance(cluster, topologic.Cluster):
            print("CellComplex.ByWiresCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        wires = []
        _ = cluster.Wires(None, wires)
        return CellComplex.ByWires(wires, triangulate=triangulate, tolerance=tolerance)

    @staticmethod
    def Cells(cellComplex: topologic.CellComplex) -> list:
        """
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
            print("CellComplex.Cells - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        cells = []
        _ = cellComplex.Cells(None, cells)
        return cells

    @staticmethod
    def Decompose(cellComplex: topologic.CellComplex, tiltAngle: float = 10.0, tolerance: float = 0.0001) -> dict:
        """
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
            1. "cells": list of cells
            2. "externalVerticalFaces": list of external vertical faces
            3. "internalVerticalFaces": list of internal vertical faces
            4. "topHorizontalFaces": list of top horizontal faces
            5. "bottomHorizontalFaces": list of bottom horizontal faces
            6. "internalHorizontalFaces": list of internal horizontal faces
            7. "externalInclinedFaces": list of external inclined faces
            8. "internalInclinedFaces": list of internal inclined faces
            9. "externalVerticalApertures": list of external vertical apertures
            10. "internalVerticalApertures": list of internal vertical apertures
            11. "topHorizontalApertures": list of top horizontal apertures
            12. "bottomHorizontalApertures": list of bottom horizontal apertures
            13. "internalHorizontalApertures": list of internal horizontal apertures
            14. "externalInclinedApertures": list of external inclined apertures
            15. "internalInclinedApertures": list of internal inclined apertures

        """
        from topologicpy.Face import Face
        from topologicpy.Vector import Vector
        from topologicpy.Aperture import Aperture
        from topologicpy.Topology import Topology

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
            apertures = Topology.Apertures(topology)
            for aperture in apertures:
                apTopologies.append(Aperture.Topology(aperture))
            return apTopologies

        if not isinstance(cellComplex, topologic.CellComplex):
            print("CellComplex.Decompose - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
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
        up = [0, 0, 1]
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
        
        cells = Topology.Cells(cellComplex)
        d = {
            "cells" : cells,
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
    def Delaunay(vertices: list = None, tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Triangulates the input vertices based on the Delaunay method. See https://en.wikipedia.org/wiki/Delaunay_triangulation.

        Parameters
        ----------
        vertices: list , optional 
            The input list of vertices to use for delaunay triangulation. If set to None, the algorithm uses the vertices of the input cell parameter.
            if both are set to none, a unit cube centered around the origin is used.
        tolerance : float , optional
            the desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.CellComplex
            The created delaunay cellComplex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from scipy.spatial import Delaunay as SCIDelaunay
        import numpy as np

        if not isinstance(vertices, list):
            cell = Cell.Prism()
            vertices = Topology.Vertices(cell)
        
        vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) < 3:
            print("CellComplex/Delaunay - Error: The input vertices parameter does not contain enough valid vertices. Returning None.")
            return None
        # Get the vertices of the input cell
        points = np.array([Vertex.Coordinates(v) for v in vertices])
        # Compute Delaunay triangulation
        triangulation = SCIDelaunay(points, furthest_site=False)

        faces = []
        for simplex in triangulation.simplices:
            tetrahedron_vertices = points[simplex]
            verts = [Vertex.ByCoordinates(list(coord)) for coord in tetrahedron_vertices]
            tri1 = [verts[0], verts[1], verts[2], verts[0]]
            tri2 = [verts[0], verts[2], verts[3], verts[0]]
            tri3 = [verts[0], verts[1], verts[3], verts[0]]
            tri4 = [verts[1], verts[2], verts[3], verts[1]]
            f1 = Face.ByVertices(tri1)
            f2 = Face.ByVertices(tri2)
            f3 = Face.ByVertices(tri3)
            f4 = Face.ByVertices(tri4)
            faces.append(f1)
            faces.append(f2)
            faces.append(f3)
            faces.append(f4)
        cc = Topology.RemoveCoplanarFaces(CellComplex.ByFaces(faces, tolerance=tolerance))
        faces = [Topology.RemoveCollinearEdges(f) for f in Topology.Faces(cc)]
        cc = CellComplex.ByFaces(faces)
        return cc
    
    @staticmethod
    def Edges(cellComplex: topologic.CellComplex) -> list:
        """
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
            print("CellComplex.Edges - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        edges = []
        _ = cellComplex.Edges(None, edges)
        return edges

    @staticmethod
    def ExternalBoundary(cellComplex: topologic.CellComplex) -> topologic.Cell:
        """
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
    def ExternalFaces(cellComplex: topologic.CellComplex) -> list:
        """
        Returns the external faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of external faces.

        """
        from topologicpy.Cell import Cell
        cell = cellComplex.ExternalBoundary()
        return Cell.Faces(cell)

    @staticmethod
    def Faces(cellComplex: topologic.CellComplex) -> list:
        """
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
            print("CellComplex.Faces - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        faces = []
        _ = cellComplex.Faces(None, faces)
        return faces

    @staticmethod
    def InternalFaces(cellComplex: topologic.CellComplex) -> list:
        """
        Returns the internal boundaries (faces) of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of internal faces of the input cellComplex.

        """
        faces = []
        _ = cellComplex.InternalBoundaries(faces)
        return faces
    
    @staticmethod
    def NonManifoldFaces(cellComplex: topologic.CellComplex) -> list:
        """
        Returns the non-manifold faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of non-manifold faces of the input cellComplex.

        """
        faces = []
        _ = cellComplex.NonManifoldFaces(faces)
        return faces
    
    @staticmethod
    def Octahedron(origin: topologic.Vertex = None, radius: float = 0.5,
                  direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Description
        ----------
        Creates an octahedron. See https://en.wikipedia.org/wiki/Octahedron.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin location of the octahedron. The default is None which results in the octahedron being placed at (0, 0, 0).
        radius : float , optional
            The radius of the octahedron's circumscribed sphere. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the octahedron. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the octahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.CellComplex
            The created octahedron.

        """
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not origin:
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not isinstance(origin, topologic.Vertex):
            print("CellComplex.Octahedron - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        vb1 = Vertex.ByCoordinates(-0.5,0,0)
        vb2 = Vertex.ByCoordinates(0,-0.5,0)
        vb3 = Vertex.ByCoordinates(0.5,0,0)
        vb4 = Vertex.ByCoordinates(0,0.5,0)
        top = Vertex.ByCoordinates(0, 0, 0.5)
        bottom = Vertex.ByCoordinates(0, 0, -0.5)
        f1 = Face.ByVertices([top,vb1,vb2])
        f2 = Face.ByVertices([top,vb2,vb3])
        f3 = Face.ByVertices([top,vb3,vb4])
        f4 = Face.ByVertices([top,vb4,vb1])
        f5 = Face.ByVertices([bottom,vb1,vb2])
        f6 = Face.ByVertices([bottom,vb2,vb3])
        f7 = Face.ByVertices([bottom,vb3,vb4])
        f8 = Face.ByVertices([bottom,vb4,vb1])
        f9 = Face.ByVertices([vb1,vb2,vb3,vb4])

        octahedron = CellComplex.ByFaces([f1,f2,f3,f4,f5,f6,f7,f8,f9], tolerance=tolerance)
        octahedron = Topology.Scale(octahedron, origin=Vertex.Origin(), x=radius/0.5, y=radius/0.5, z=radius/0.5)
        if placement == "bottom":
            octahedron = Topology.Translate(octahedron, 0, 0, radius)
        elif placement == "lowerleft":
            octahedron = Topology.Translate(octahedron, radius, radius, radius)
        octahedron = Topology.Orient(octahedron, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        octahedron = Topology.Place(octahedron, originA=Vertex.Origin(), originB=origin)
        return octahedron
    
    @staticmethod
    def Prism(origin: topologic.Vertex = None,
              width: float = 1.0, length: float = 1.0, height: float = 1.0,
              uSides: int = 2, vSides: int = 2, wSides: int = 2,
              direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001) -> topologic.CellComplex:
        """
        Creates a prismatic cellComplex with internal cells.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin location of the prism. The default is None which results in the prism being placed at (0, 0, 0).
        width : float , optional
            The width of the prism. The default is 1.
        length : float , optional
            The length of the prism. The default is 1.
        height : float , optional
            The height of the prism.
        uSides : int , optional
            The number of sides along the width. The default is 1.
        vSides : int , optional
            The number of sides along the length. The default is 1.
        wSides : int , optional
            The number of sides along the height. The default is 1.
        direction : list , optional
            The vector representing the up direction of the prism. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the prism. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.CellComplex
            The created prism.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        def bb(topology):
            vertices = []
            _ = topology.Vertices(None, vertices)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(aVertex.X())
                y.append(aVertex.Y())
                z.append(aVertex.Z())
            minX = min(x)
            minY = min(y)
            minZ = min(z)
            maxX = max(x)
            maxY = max(y)
            maxZ = max(z)
            return [minX, minY, minZ, maxX, maxY, maxZ]
        
        def slice(topology, uSides, vSides, wSides):
            minX, minY, minZ, maxX, maxY, maxZ = bb(topology)
            centroid = Vertex.ByCoordinates(minX+(maxX-minX)*0.5, minY+(maxY-minY)*0.5, minZ+(maxZ-minZ)*0.5)
            wOrigin = Vertex.ByCoordinates(Vertex.X(centroid), Vertex.Y(centroid), minZ)
            wFace = Face.Rectangle(origin=wOrigin, width=(maxX-minX)*1.1, length=(maxY-minY)*1.1)
            wFaces = []
            wOffset = (maxZ-minZ)/wSides
            for i in range(wSides-1):
                wFaces.append(Topology.Translate(wFace, 0,0,wOffset*(i+1)))
            uOrigin = Vertex.ByCoordinates(minX, Vertex.Y(centroid), Vertex.Z(centroid))
            uFace = Face.Rectangle(origin=uOrigin, width=(maxZ-minZ)*1.1, length=(maxY-minY)*1.1, direction=[1,0,0])
            uFaces = []
            uOffset = (maxX-minX)/uSides
            for i in range(uSides-1):
                uFaces.append(Topology.Translate(uFace, uOffset*(i+1),0,0))
            vOrigin = Vertex.ByCoordinates(Vertex.X(centroid), minY, Vertex.Z(centroid))
            vFace = Face.Rectangle(origin=vOrigin, width=(maxX-minX)*1.1, length=(maxZ-minZ)*1.1, direction=[0,1,0])
            vFaces = []
            vOffset = (maxY-minY)/vSides
            for i in range(vSides-1):
                vFaces.append(Topology.Translate(vFace, 0,vOffset*(i+1),0))
            all_faces = uFaces+vFaces+wFaces
            if len(all_faces) > 0:
                f_clus = Cluster.ByTopologies(uFaces+vFaces+wFaces)
                return Topology.Slice(topology, f_clus, tolerance=tolerance)
            else:
                return topologic.CellComplex.ByCells([topology])
        if not isinstance(origin, topologic.Vertex):
            origin = Vertex.ByCoordinates(0, 0, 0)

        c = Cell.Prism(origin=origin, width=width, length=length, height=height, uSides=1, vSides=1, wSides=1, placement=placement, tolerance=tolerance)
        prism = slice(c, uSides=uSides, vSides=vSides, wSides=wSides)
        if prism:
            prism = Topology.Orient(prism, origin=origin, dirA=[0, 0, 1], dirB=direction)
            return prism
        else:
            print("CellComplex.Prism - Error: Could not create a prism. Returning None.")
            return None

    @staticmethod
    def RemoveCollinearEdges(cellComplex: topologic.CellComplex, angTolerance: float = 0.1, tolerance: float = 0.0001) -> topologic.Wire:
        """
        Removes any collinear edges in the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.CellComplex
            The created cellComplex without any collinear edges.

        """
        from topologicpy.Cell import Cell

        if not isinstance(cellComplex, topologic.CellComplex):
            print("CellComplex.RemoveCollinearEdges - Error: The input cellComplex parameter is not a valid cellComplex. Returning None.")
            return None
        cells = CellComplex.Cells(cellComplex)
        clean_cells = []
        for cell in cells:
            clean_cells.append(Cell.RemoveCollinearEdges(cell, angTolerance=angTolerance, tolerance=tolerance))
        return CellComplex.ByCells(clean_cells, tolerance=tolerance)
    
    @staticmethod
    def Shells(cellComplex: topologic.CellComplex) -> list:
        """
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
            print("CellComplex.Shells - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        shells = []
        _ = cellComplex.Shells(None, shells)
        return shells

    @staticmethod
    def Vertices(cellComplex: topologic.CellComplex) -> list:
        """
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
            print("CellComplex.Vertices - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        vertices = []
        _ = cellComplex.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Volume(cellComplex: topologic.CellComplex, mantissa: int = 6) -> float:
        """
        Returns the volume of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic.CellComplex
            The input cellComplex.
        manitssa: int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The volume of the input cellComplex.

        """
        from topologicpy.Cell import Cell
        if not isinstance(cellComplex, topologic.CellComplex):
            print("CellComplex.Volume - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        cells = CellComplex.Cells(cellComplex)
        volume = 0
        for cell in cells:
            volume = Cell.Volume(cell)
            if not volume == None:
                volume += Cell.Volume(cell)
        return round(volume, mantissa)
    
    @staticmethod
    def Voronoi(vertices: list = None, cell: topologic.Cell = None, tolerance: float = 0.0001):
        """
        Partitions the input cell based on the Voronoi method. See https://en.wikipedia.org/wiki/Voronoi_diagram.

        Parameters
        ----------
        vertices: list , optional 
            The input list of vertices to use for voronoi partitioning. If set to None, the algorithm uses the vertices of the input cell parameter.
            if both are set to none, a unit cube centered around the origin is used.
        cell : topologic.Cell , optional
            The input bounding cell. If set to None, an axes-aligned bounding cell is created from the list of vertices. The default is None.
        tolerance : float , optional
            the desired tolerance. The default is 0.0001.
        

        Returns
        -------
        topologic.CellComplex
            The created voronoi cellComplex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from scipy.spatial import Voronoi as SCIVoronoi
        import numpy as np

        def fracture_with_voronoi(points):
            # Compute Voronoi tessellation
            vor = SCIVoronoi(points)
            verts = []
            faces = []
            for v in vor.vertices:
                verts.append(Vertex.ByCoordinates(list(v)))
            for region in vor.ridge_vertices:
                temp_list = []
                if -1 not in region and len(region) > 0:
                    for item in region:
                        temp_list.append(verts[item])
                    f = Face.ByVertices(temp_list)
                    if isinstance(f, topologic.Face):
                        faces.append(f)
            if len(faces) < 1:
                return None
            return Cluster.ByTopologies(faces)
        
        if cell == None:
            if not isinstance(vertices, list):
                cell = Cell.Prism(uSides=2, vSides=2, wSides=2)
                vertices = Topology.Vertices(cell)
                vertices.append(Vertex.Origin())
            else:
                vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
                if len(vertices) < 1:
                    print("CellComplex.Voronoi - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
                    return None
                cell = Topology.BoundingBox(Cluster.ByTopologies(vertices))
        if not isinstance(vertices, list):
            if not isinstance(cell, topologic.Cell):
                cell = Cell.Prism()
                vertices = Topology.Vertices(cell)
            else:
                vertices = Topology.Vertices(cell)
        else:
            vertices += Topology.Vertices(cell)
        vertices = [v for v in vertices if (Vertex.IsInternal(v, cell) or not Vertex.Index(v, Topology.Vertices(cell)) == None)]
        if len(vertices) < 1:
            print("CellComplex.Voronoi - Error: The input vertices parame ter does not contain any vertices that are inside the input cell parameter. Returning None.")
            return None
        voronoi_points = np.array([Vertex.Coordinates(v) for v in vertices])
        cluster = fracture_with_voronoi(voronoi_points)
        if cluster == None:
            print("CellComplex.Voronoi - Error: the operation failed. Returning None.")
            return None
        cellComplex = Topology.Slice(cell, cluster)
        if not isinstance(cellComplex, topologic.CellComplex):
            print("CellComplex.Voronoi - Error: the operation failed. Returning None.")
            return None
        return cellComplex
    
    @staticmethod
    def Wires(cellComplex: topologic.CellComplex) -> list:
        """
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
            print("CellComplex.Wires - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        wires = []
        _ = cellComplex.Wires(None, wires)
        return wires

