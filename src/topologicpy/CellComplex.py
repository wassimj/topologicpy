# Copyright (C) 2025
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

class CellComplex():
    @staticmethod
    def Box(origin= None,
            width: float = 1.0, length: float = 1.0, height: float = 1.0,
            uSides: int = 2, vSides: int = 2, wSides: int = 2,
            direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a box with internal cells.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
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
        topologic_core.CellComplex
            The created box.

        """
        return CellComplex.Prism(origin=origin,
                                 width=width, length=length, height=height,
                                 uSides=uSides, vSides=vSides, wSides=wSides,
                                 direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def ByCells(cells: list, transferDictionaries = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cellcomplex by merging the input cells.

        Parameters
        ----------
        cells : list
            The list of input cells.
        transferDictionaries : bool , optional
            If set to True, any dictionaries in the cells are transferred to the CellComplex. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
                If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not isinstance(cells, list):
            if not silent:
                print("CellComplex.ByCells - Error: The input cells parameter is not a valid list. Returning None.")
            return None
        cells = [x for x in cells if Topology.IsInstance(x, "Cell")]
        if len(cells) < 1:
            if not silent:
                print("CellComplex.ByCells - Error: The input cells parameter does not contain any valid cells. Returning None.")
            return None
        cluster = Cluster.ByTopologies(cells)
        cellComplex = None
        if len(cells) == 1:
            return topologic.CellComplex.ByCells(cells) # Hook to Core
        else:
            try:
                cellComplex = topologic.CellComplex.ByCells(cells) # Hook to Core
            except:
                topA = cells[0]
                topB = Cluster.ByTopologies(cells[1:])
                cellComplex = Topology.Merge(topA, topB, tranDict=False, tolerance=tolerance)
        
        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.ByCells - Warning: Could not create a CellComplex. Returning object of type topologic_core.Cluster instead of topologic_core.CellComplex.")
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
                    print("CellComplex.ByCells - Warning: Resulting object contains only one cell. Returning object of type topologic_core.Cell instead of topologic_core.CellComplex.")
                return(temp_cells[0])
            if transferDictionaries == True:
                for temp_cell in temp_cells:
                    v = Topology.InternalVertex(temp_cell, tolerance=tolerance)
                    enclosing_cells = Vertex.EnclosingCells(v, cluster)
                    dictionaries = [Topology.Dictionary(ec) for ec in enclosing_cells]
                    d = Dictionary.ByMergedDictionaries(dictionaries, silent=silent)
                    temp_cell = Topology.SetDictionary(temp_cell, d)

        return cellComplex
    
    @staticmethod
    def ByCellsCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a cellcomplex by merging the cells within the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of cells.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """

        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("CellComplex.ByCellsCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        cells = Topology.Cells(cluster)
        return CellComplex.ByCells(cells, tolerance=tolerance)

    @staticmethod
    def ByFaces(faces: list, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cellcomplex by merging the input faces.

        Parameters
        ----------
        faces : list
            The input faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(faces, list):
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter is not a valid list. Returning None.")
            return None
        faces = [x for x in faces if Topology.IsInstance(x, "Face")]
        if len(faces) < 1:
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter does not contain any valid faces. Returning None.")
            return None
        try:
            cellComplex = topologic.CellComplex.ByFaces(faces, tolerance, False) # Hook to Core
        except:
            cellComplex = None
        if not cellComplex:
            if not silent:
                print("CellComplex.ByFaces - Warning: The default method failed. Attempting a workaround.")
            cellComplex = faces[0]
            for i in range(1,len(faces)):
                newCellComplex = None
                try:
                    newCellComplex = cellComplex.Merge(faces[i], False, tolerance) # Hook to Core
                except:
                    if not silent:
                        print("CellComplex.ByFaces - Warning: Failed to merge face #"+str(i)+". Skipping.")
                if newCellComplex:
                    cellComplex = newCellComplex
            if not Topology.Type(cellComplex) == Topology.TypeID("CellComplex"):
                if not silent:
                    print("CellComplex.ByFaces - Warning: The input faces do not form a cellcomplex")
                if Topology.Type(cellComplex) == Topology.TypeID("Cluster"):
                    returnCellComplexes = Cluster.CellComplexes(cellComplex)
                    if len(returnCellComplexes) > 0:
                        return returnCellComplexes[0]
                    else:
                        if not silent:
                            print("CellComplex.ByFaces - Error: Could not create a cellcomplex. Returning None.")
                        return None
                else:
                    if not silent:
                        print("CellComplex.ByFaces - Error: Could not create a cellcomplex. Returning None.")
                    return None
        else:
            return cellComplex
    
    @staticmethod
    def ByFacesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a cellcomplex by merging the faces within the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("CellComplex.ByFacesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        faces = Topology.Faces(cluster)
        return CellComplex.ByFaces(faces, tolerance=tolerance)

    @staticmethod
    def ByWires(wires: list, triangulate: bool = True, tolerance: float = 0.0001):
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
        topologic_core.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(wires, list):
            print("CellComplex.ByFaces - Error: The input wires parameter is not a valid list. Returning None.")
            return None
        wires = [x for x in wires if Topology.IsInstance(x, "Wire")]
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
            w1_edges = Topology.Edges(wire1)
            w2_edges = Topology.Edges(wire2)
            if len(w1_edges) != len(w2_edges):
                print("CellComplex.ByWires - Error: The input wires parameter contains wires with different number of edges. Returning None.")
                return None
            for j in range (len(w1_edges)):
                e1 = w1_edges[j]
                e2 = w2_edges[j]
                e3 = None
                e4 = None
                try:
                    e3 = Edge.ByStartVertexEndVertex(Edge.StartVertex(e1), Edge.StartVertex(e2), tolerance=tolerance, silent=True)
                except:
                    try:
                        e4 = Edge.ByStartVertexEndVertex(Edge.EndVertex(e1), Edge.EndVertex(e2), tolerance=tolerance, silent=True)
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
                    e4 = Edge.ByStartVertexEndVertex(Edge.EndVertex(e1), Edge.EndVertex(e2), tolerance=tolerance, silent=True)
                except:
                    try:
                        e3 = Edge.ByStartVertexEndVertex(Edge.StartVertex(e1), Edge.StartVertex(e2), tolerance=tolerance, silent=True)
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
                        e5 = Edge.ByStartVertexEndVertex(Edge.StartVertex(e1), Edge.EndVertex(e2), tolerance=tolerance, silent=True)
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
    def ByWiresCluster(cluster, triangulate: bool = True, tolerance: float = 0.0001):
        """
        Creates a cellcomplex by lofting through the wires in the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of wires.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("CellComplex.ByWiresCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        wires = Topology.Wires(cluster)
        return CellComplex.ByWires(wires, triangulate=triangulate, tolerance=tolerance)

    @staticmethod
    def Cells(cellComplex) -> list:
        """
        Returns the cells of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of cells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Cells - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        cells = []
        _ = cellComplex.Cells(None, cells) # Hook to Core
        return cells

    @staticmethod
    def Cube(origin= None,
            size: float = 1.0,
            uSides: int = 2, vSides: int = 2, wSides: int = 2,
            direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a cube with internal cells.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the cube. The default is None which results in the cube being placed at (0, 0, 0).
        size : float , optional
            The size of the cube. The default is 1.
        uSides : int , optional
            The number of sides along the width. The default is 1.
        vSides : int, optional
            The number of sides along the length. The default is 1.
        wSides : int , optional
            The number of sides along the height. The default is 1.
        direction : list , optional
            The vector representing the up direction of the cube. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the cube. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.CellComplex
            The created cube.

        """
        return CellComplex.Prism(origin=origin,
                                 width=size, length=size, height=size,
                                 uSides=uSides, vSides=vSides, wSides=wSides,
                                 direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def Decompose(cellComplex, tiltAngle: float = 10.0, tolerance: float = 0.0001) -> dict:
        """
        Decomposes the input cellComplex into its logical components. This method assumes that the positive Z direction is UP.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
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
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Vector import Vector
        from topologicpy.Aperture import Aperture
        from topologicpy.Topology import Topology

        def angleCode(f, up, tiltAngle):
            dirA = Face.Normal(f)
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
            return Topology.Apertures(topology)

        if not Topology.IsInstance(cellComplex, "CellComplex"):
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
            zList.append(Vertex.Z(Topology.Centroid(f)))
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
                    if abs(Vertex.Z(Topology.Centroid(aFace)) - zMin) <= tolerance:
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
                    if abs(Vertex.Z(Topology.Centroid(aFace)) - zMax) <= tolerance:
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
    def Delaunay(vertices: list = None, tolerance: float = 0.0001):
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
        topologic_core.CellComplex
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
        
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
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
    def Edges(cellComplex) -> list:
        """
        Returns the edges of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Edges - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        edges = []
        _ = cellComplex.Edges(None, edges) # Hook to Core
        return edges

    @staticmethod
    def ExternalBoundary(cellComplex):
        """
        Returns the external boundary (cell) of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        topologic_core.Cell
            The external boundary of the input cellComplex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.ExternalBoundary - Error: The input cellComplex parameter is not a valid cellComplex. Returning None.")
            return None
        return cellComplex.ExternalBoundary() # Hook to Core
    
    @staticmethod
    def ExternalFaces(cellComplex) -> list:
        """
        Returns the external faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of external faces.

        """
        from topologicpy.Cell import Cell
        cell = CellComplex.ExternalBoundary(cellComplex)
        return Cell.Faces(cell)

    @staticmethod
    def Faces(cellComplex) -> list:
        """
        Returns the faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of faces.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Faces - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        faces = []
        _ = cellComplex.Faces(None, faces) # Hook to Core
        return faces

    @staticmethod
    def InternalFaces(cellComplex) -> list:
        """
        Returns the internal boundaries (faces) of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of internal faces of the input cellComplex.

        """
        faces = []
        _ = cellComplex.InternalBoundaries(faces) # Hook to Core
        return faces
    
    @staticmethod
    def NonManifoldFaces(cellComplex) -> list:
        """
        Returns the non-manifold faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of non-manifold faces of the input cellComplex.

        """
        faces = []
        _ = cellComplex.NonManifoldFaces(faces) # Hook to Core
        return faces
    
    @staticmethod
    def Octahedron(origin= None, radius: float = 0.5,
                  direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates an octahedron. See https://en.wikipedia.org/wiki/Octahedron.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
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
        topologic_core.CellComplex
            The created octahedron.

        """
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
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
    def Prism(origin= None,
              width: float = 1.0, length: float = 1.0, height: float = 1.0,
              uSides: int = 2, vSides: int = 2, wSides: int = 2,
              direction: list = [0, 0, 1], placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a prismatic cellComplex with internal cells.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
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
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.CellComplex
            The created prism.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        def bb(topology):
            vertices = Topology.Vertices(topology)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(Vertex.X(aVertex, mantissa=mantissa))
                y.append(Vertex.Y(aVertex, mantissa=mantissa))
                z.append(Vertex.Z(aVertex, mantissa=mantissa))
            x_min = min(x)
            y_min = min(y)
            z_min = min(z)
            maxX = max(x)
            maxY = max(y)
            maxZ = max(z)
            return [x_min, y_min, z_min, maxX, maxY, maxZ]
        
        def slice(topology, uSides, vSides, wSides):
            x_min, y_min, z_min, maxX, maxY, maxZ = bb(topology)
            centroid = Vertex.ByCoordinates(x_min+(maxX-x_min)*0.5, y_min+(maxY-y_min)*0.5, z_min+(maxZ-z_min)*0.5)
            wOrigin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), Vertex.Y(centroid, mantissa=mantissa), z_min)
            wFace = Face.Rectangle(origin=wOrigin, width=(maxX-x_min)*1.1, length=(maxY-y_min)*1.1)
            wFaces = []
            wOffset = (maxZ-z_min)/wSides
            for i in range(wSides-1):
                wFaces.append(Topology.Translate(wFace, 0,0,wOffset*(i+1)))
            uOrigin = Vertex.ByCoordinates(x_min, Vertex.Y(centroid, mantissa=mantissa), Vertex.Z(centroid, mantissa=mantissa))
            uFace = Face.Rectangle(origin=uOrigin, width=(maxZ-z_min)*1.1, length=(maxY-y_min)*1.1, direction=[1,0,0])
            uFaces = []
            uOffset = (maxX-x_min)/uSides
            for i in range(uSides-1):
                uFaces.append(Topology.Translate(uFace, uOffset*(i+1),0,0))
            vOrigin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), y_min, Vertex.Z(centroid, mantissa=mantissa))
            vFace = Face.Rectangle(origin=vOrigin, width=(maxX-x_min)*1.1, length=(maxZ-z_min)*1.1, direction=[0,1,0])
            vFaces = []
            vOffset = (maxY-y_min)/vSides
            for i in range(vSides-1):
                vFaces.append(Topology.Translate(vFace, 0,vOffset*(i+1),0))
            all_faces = uFaces+vFaces+wFaces
            if len(all_faces) > 0:
                f_clus = Cluster.ByTopologies(uFaces+vFaces+wFaces)
                return Topology.Slice(topology, f_clus, tolerance=tolerance)
            else:
                return CellComplex.ByCells([topology])
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)

        c = Cell.Prism(origin=origin, width=width, length=length, height=height, uSides=1, vSides=1, wSides=1, placement=placement, mantissa=mantissa, tolerance=tolerance)
        prism = slice(c, uSides=uSides, vSides=vSides, wSides=wSides)
        if prism:
            prism = Topology.Orient(prism, origin=origin, dirA=[0, 0, 1], dirB=direction)
            return prism
        else:
            print("CellComplex.Prism - Error: Could not create a prism. Returning None.")
            return None

    @staticmethod
    def RemoveCollinearEdges(cellComplex, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Removes any collinear edges in the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.CellComplex
            The created cellComplex without any collinear edges.

        """
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        import inspect
        
        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.RemoveCollinearEdges - Error: The input cellComplex parameter is not a valid cellComplex. Returning None.")
            print("CellComplex.RemoveCollinearEdges - Inspection:")
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
            return None
        cells = CellComplex.Cells(cellComplex)
        clean_cells = []
        for cell in cells:
            clean_cells.append(Cell.RemoveCollinearEdges(cell, angTolerance=angTolerance, tolerance=tolerance))
        return CellComplex.ByCells(clean_cells, tolerance=tolerance)
    
    @staticmethod
    def Shells(cellComplex) -> list:
        """
        Returns the shells of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of shells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Shells - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        shells = Topology.Shells(cellComplex)
        return shells

    @staticmethod
    def _grow_connected_group(seed_idx, group_size, adjacency, visited_global):
        """
        Attempts to grow a group of the given size starting from seed_idx using adjacency.
        Returns a list of indices if successful, else None.
        """
        from collections import deque
        import random

        group = [seed_idx]
        visited = set(group)
        queue = deque([seed_idx])

        while queue and len(group) < group_size:
            current = queue.popleft()
            neighbors = adjacency.get(current, [])
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in visited_global:
                    group.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(group) >= group_size:
                        break

        return group if len(group) == group_size else None

    @staticmethod
    def Tetrahedron(origin = None, length: float = 1, depth: int = 1, direction=[0,0,1], placement="center", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a recursive tetrahedron cellComplex with internal cells.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the tetrahedron. The default is None which results in the tetrahedron being placed at (0, 0, 0).
        length : float , optional
            The length of the edge of the tetrahedron. The default is 1.
        depth : int , optional
            The desired maximum number of recrusive subdivision levels.
        direction : list , optional
            The vector representing the up direction of the tetrahedron. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the tetrahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.CellComplex
            The created tetrahedron.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        from math import sqrt

        def subdivide_tetrahedron(tetrahedron, depth):
            """
            Recursively subdivides a tetrahedron into smaller tetrahedra.

            Parameters:
                tetrahedron (Cell): The tetrahedron to subdivide.
                depth (int): Recursion depth for the subdivision.

            Returns:
                list: List of smaller tetrahedral cells.
            """
            if depth == 0:
                return [tetrahedron]

            # Extract the vertices of the tetrahedron
            vertices = Topology.Vertices(tetrahedron)
            v0, v1, v2, v3 = vertices

            # Calculate midpoints of the edges
            m01 = Vertex.ByCoordinates((v0.X() + v1.X()) / 2, (v0.Y() + v1.Y()) / 2, (v0.Z() + v1.Z()) / 2)
            m02 = Vertex.ByCoordinates((v0.X() + v2.X()) / 2, (v0.Y() + v2.Y()) / 2, (v0.Z() + v2.Z()) / 2)
            m03 = Vertex.ByCoordinates((v0.X() + v3.X()) / 2, (v0.Y() + v3.Y()) / 2, (v0.Z() + v3.Z()) / 2)
            m12 = Vertex.ByCoordinates((v1.X() + v2.X()) / 2, (v1.Y() + v2.Y()) / 2, (v1.Z() + v2.Z()) / 2)
            m13 = Vertex.ByCoordinates((v1.X() + v3.X()) / 2, (v1.Y() + v3.Y()) / 2, (v1.Z() + v3.Z()) / 2)
            m23 = Vertex.ByCoordinates((v2.X() + v3.X()) / 2, (v2.Y() + v3.Y()) / 2, (v2.Z() + v3.Z()) / 2)

            # Create smaller tetrahedra
            tetrahedra = [
                Cell.ByFaces([
                    Face.ByVertices([v0, m01, m02]),
                    Face.ByVertices([v0, m01, m03]),
                    Face.ByVertices([v0, m02, m03]),
                    Face.ByVertices([m01, m02, m03])
                ]),
                Cell.ByFaces([
                    Face.ByVertices([m01, v1, m12]),
                    Face.ByVertices([m01, v1, m13]),
                    Face.ByVertices([m01, m12, m13]),
                    Face.ByVertices([v1, m12, m13])
                ]),
                Cell.ByFaces([
                    Face.ByVertices([m02, m12, v2]),
                    Face.ByVertices([m02, m12, m23]),
                    Face.ByVertices([m02, v2, m23]),
                    Face.ByVertices([m12, v2, m23])
                ]),
                Cell.ByFaces([
                    Face.ByVertices([m03, m13, m23]),
                    Face.ByVertices([m03, v3, m13]),
                    Face.ByVertices([m03, v3, m23]),
                    Face.ByVertices([m13, v3, m23])
                ])
            ]

            # Recursively subdivide the smaller tetrahedra
            result = []
            for t in tetrahedra:
                result.extend(subdivide_tetrahedron(t, depth - 1))
            return result

        if not Topology.IsInstance(origin, "vertex"):
            origin = Vertex.Origin()
        
        # Define the four vertices of the tetrahedron
        v0 = Vertex.ByCoordinates(0, 0, 0)
        v1 = Vertex.ByCoordinates(length, 0, 0)
        v2 = Vertex.ByCoordinates(length/2, sqrt(3)/2*length, 0)
        v3 = Vertex.ByCoordinates(length/2, sqrt(3)/2*length/3, sqrt(2/3)*length)

        # Create the initial tetrahedron
        tetrahedron = Cell.ByFaces([
            Face.ByVertices([v0, v1, v2]),
            Face.ByVertices([v0, v1, v3]),
            Face.ByVertices([v1, v2, v3]),
            Face.ByVertices([v2, v0, v3]),
        ])

        bbox = Topology.BoundingBox(tetrahedron)
        d = Topology.Dictionary(bbox)
        bb_width = Dictionary.ValueAtKey(d, "width")
        bb_length = Dictionary.ValueAtKey(d, "length")
        bb_height = Dictionary.ValueAtKey(d, "height")
        
        centroid = Topology.Centroid(tetrahedron)
        c_x, c_y, c_z = Vertex.Coordinates(centroid, mantissa=mantissa)

        if placement.lower() == "center":
            tetrahedron = Topology.Translate(tetrahedron, -c_x, -c_y, -c_z)
        elif placement.lower() == "bottom":
            tetrahedron = Topology.Translate(tetrahedron,-c_x, -c_y, 0)
        elif placement.lower() == "upperleft":
            tetrahedron = Topology.Translate(tetrahedron, 0, 0, -bb_height)
        elif placement.lower() == "upperright":
            tetrahedron = Topology.Translate(tetrahedron, -bb_width, -bb_length, -bb_height)
        elif placement.lower() == "bottomright":
            tetrahedron = Topology.Translate(tetrahedron, -bb_width, -bb_length, 0)
        elif placement.lower() == "top":
            tetrahedron = Topology.Translate(tetrahedron,-c_x, -c_y, -bb_height)
        
        tetrahedron = Topology.Place(tetrahedron, Vertex.Origin(), origin)
        if not direction == [0,0,1]:
            tetrahedron = Topology.Orient(tetrahedron, origin=origin, dirA=[0,0,1], dirB=direction)

        depth = max(depth, 1)
        # Recursively subdivide the tetrahedron
        subdivided_tetrahedra = subdivide_tetrahedron(tetrahedron, depth)
        # Create a cell complex from the subdivided tetrahedra
        return CellComplex.ByCells([tetrahedron]+subdivided_tetrahedra)
    
    @staticmethod
    def Torus(origin= None, majorRadius: float = 0.5, minorRadius: float = 0.125, uSides: int = 16, vSides: int = 8, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a torus.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the torus. The default is None which results in the torus being placed at (0, 0, 0).
        majorRadius : float , optional
            The major radius of the torus. The default is 0.5.
        minorRadius : float , optional
            The minor radius of the torus. The default is 0.1.
        uSides : int , optional
            The number of sides along the longitude of the torus. The default is 16.
        vSides : int , optional
            The number of sides along the latitude of the torus. The default is 8.
        direction : list , optional
            The vector representing the up direction of the torus. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the torus. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created torus.

        """
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Torus - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        c = Wire.Circle(origin=Vertex.Origin(), radius=minorRadius, sides=vSides, fromAngle=0, toAngle=360, close=False, direction=[0, 1, 0], placement="center")
        c = Face.ByWire(c)
        c = Topology.Translate(c, abs(majorRadius-minorRadius), 0, 0)
        torus = Topology.Spin(c, origin=Vertex.Origin(), triangulate=False, direction=[0, 0, 1], angle=360, sides=uSides, tolerance=tolerance)
        if Topology.Type(torus) == Topology.TypeID("Shell"):
            faces = Topology.Faces(torus)
            torus = CellComplex.ByFaces(faces)
        if placement.lower() == "bottom":
            torus = Topology.Translate(torus, 0, 0, minorRadius)
        elif placement.lower() == "lowerleft":
            torus = Topology.Translate(torus, majorRadius, majorRadius, minorRadius)

        torus = Topology.Orient(torus, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        torus = Topology.Place(torus, originA=Vertex.Origin(), originB=origin)
        return torus

    @staticmethod
    def Vertices(cellComplex) -> list:
        """
        Returns the vertices of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Vertices - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        vertices = []
        _ = cellComplex.Vertices(None, vertices) # Hook to Core
        return vertices

    @staticmethod
    def Volume(cellComplex, mantissa: int = 6) -> float:
        """
        Returns the volume of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.
        manitssa: int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The volume of the input cellComplex.

        """
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(cellComplex, "CellComplex"):
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
    def Voronoi(vertices: list = None, cell= None, tolerance: float = 0.0001):
        """
        Partitions the input cell based on the Voronoi method. See https://en.wikipedia.org/wiki/Voronoi_diagram.

        Parameters
        ----------
        vertices: list , optional 
            The input list of vertices to use for voronoi partitioning. If set to None, the algorithm uses the vertices of the input cell parameter.
            if both are set to none, a unit cube centered around the origin is used.
        cell : topologic_core.Cell , optional
            The input bounding cell. If set to None, an axes-aligned bounding cell is created from the list of vertices. The default is None.
        tolerance : float , optional
            the desired tolerance. The default is 0.0001.
        

        Returns
        -------
        topologic_core.CellComplex
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
                    if Topology.IsInstance(f, "Face"):
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
                vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
                if len(vertices) < 1:
                    print("CellComplex.Voronoi - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
                    return None
                cell = Topology.BoundingBox(Cluster.ByTopologies(vertices))
        if not isinstance(vertices, list):
            if not Topology.IsInstance(cell, "Cell"):
                cell = Cell.Prism()
                vertices = Topology.Vertices(cell)
            else:
                vertices = Topology.Vertices(cell)
        else:
            vertices += Topology.Vertices(cell)
        vertices = [v for v in vertices if (Vertex.IsInternal(v, cell) or not Vertex.Index(v, Topology.Vertices(cell), tolerance=tolerance) == None)]
        if len(vertices) < 1:
            print("CellComplex.Voronoi - Error: The input vertices parameter does not contain any vertices that are inside the input cell parameter. Returning None.")
            return None
        voronoi_points = np.array([Vertex.Coordinates(v) for v in vertices])
        cluster = fracture_with_voronoi(voronoi_points)
        if cluster == None:
            print("CellComplex.Voronoi - Error: the operation failed. Returning None.")
            return None
        cellComplex = Topology.Slice(cell, cluster)
        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Voronoi - Error: the operation failed. Returning None.")
            return None
        return cellComplex
    
    @staticmethod
    def Wires(cellComplex) -> list:
        """
        Returns the wires of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of wires.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Wires - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        wires = []
        _ = cellComplex.Wires(None, wires) # Hook to Core
        return wires

