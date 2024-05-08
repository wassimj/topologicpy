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
from topologicpy.Vector import Vector
from topologicpy.Wire import Wire
from topologicpy.Topology import Topology
import math
import os
import warnings

try:
    import numpy as np
except:
    print("Face - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("Face - numpy library installed correctly.")
    except:
        warnings.warn("Face - Error: Could not import numpy.")

class Face(Topology):
    @staticmethod
    def AddInternalBoundaries(face: topologic.Face, wires: list) -> topologic.Face:
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

        if not isinstance(face, topologic.Face):
            print("Face.AddInternalBoundaries - Error: The input face parameter is not a valid topologic face. Returning None.")
            return None
        if not isinstance(wires, list):
            print("Face.AddInternalBoundaries - Warning: The input wires parameter is not a valid list. Returning the input face.")
            return face
        wireList = [w for w in wires if isinstance(w, topologic.Wire)]
        if len(wireList) < 1:
            print("Face.AddInternalBoundaries - Warning: The input wires parameter does not contain any valid wires. Returning the input face.")
            return face
        faceeb = face.ExternalBoundary()
        faceibList = []
        _ = face.InternalBoundaries(faceibList)
        for wire in wires:
            faceibList.append(wire)
        return topologic.Face.ByExternalInternalBoundaries(faceeb, faceibList)

    @staticmethod
    def AddInternalBoundariesCluster(face: topologic.Face, cluster: topologic.Cluster) -> topologic.Face:
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
        if not isinstance(face, topologic.Face):
            print("Face.AddInternalBoundariesCluster - Warning: The input cluster parameter is not a valid cluster. Returning None.")
            return None
        if not cluster:
            return face
        if not isinstance(cluster, topologic.Cluster):
            return face
        wires = []
        _ = cluster.Wires(None, wires)
        return Face.AddInternalBoundaries(face, wires)
    
    @staticmethod
    def Angle(faceA: topologic.Face, faceB: topologic.Face, mantissa: int = 6) -> float:
        """
        Returns the angle in degrees between the two input faces.

        Parameters
        ----------
        faceA : topologic.Face
            The first input face.
        faceB : topologic.Face
            The second input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The angle in degrees between the two input faces.

        """
        from topologicpy.Vector import Vector
        if not isinstance(faceA, topologic.Face):
            print("Face.Angle - Warning: The input faceA parameter is not a valid topologic face. Returning None.")
            return None
        if not isinstance(faceB, topologic.Face):
            print("Face.Angle - Warning: The input faceB parameter is not a valid topologic face. Returning None.")
            return None
        dirA = Face.NormalAtParameters(faceA, 0.5, 0.5, "xyz", 3)
        dirB = Face.NormalAtParameters(faceB, 0.5, 0.5, "xyz", 3)
        return round((Vector.Angle(dirA, dirB)), mantissa)

    @staticmethod
    def Area(face: topologic.Face, mantissa: int = 6) -> float:
        """
        Returns the area of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The area of the input face.

        """
        if not isinstance(face, topologic.Face):
            print("Face.Area - Warning: The input face parameter is not a valid topologic face. Returning None.")
            return None
        area = None
        try:
            area = round(topologic.FaceUtility.Area(face), mantissa)
        except:
            area = None
        return area

    @staticmethod
    def BoundingRectangle(topology: topologic.Topology, optimize: int = 0, tolerance: float = 0.0001) -> topologic.Face:
        """
        Returns a face representing a bounding rectangle of the input topology. The returned face contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting face will become axis-aligned.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding rectangle. The default is 0.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
            print("Face.BoundingRectangle - Warning: The input topology parameter is not a valid topologic topology. Returning None.")
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
                    t = Topology.Rotate(topology, origin=origin, axis=[0, 0, 1], angle=z)
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
        baseFace = Face.ByWire(baseWire, tolerance=tolerance)
        baseFace = Topology.Rotate(baseFace, origin=origin, axis=[0, 0, 1], angle=-best_z)
        dictionary = Dictionary.ByKeysValues(["zrot"], [best_z])
        baseFace = Topology.SetDictionary(baseFace, dictionary)
        return baseFace
    
    @staticmethod
    def ByEdges(edges: list, tolerance : float = 0.0001) -> topologic.Face:
        """
        Creates a face from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        face : topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        if not isinstance(edges, list):
            print("Face.ByEdges - Error: The input edges parameter is not a valid list. Returning None.")
            return None
        edges = [e for e in edges if isinstance(e, topologic.Edge)]
        if len(edges) < 1:
            print("Face.ByEdges - Error: The input edges parameter does not contain any valid edges. Returning None.")
            return None
        wire = Wire.ByEdges(edges, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            print("Face.ByEdges - Error: Could not create the required wire. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)

    @staticmethod
    def ByEdgesCluster(cluster: topologic.Cluster, tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a face from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of edges.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        face : topologic.Face
            The created face.

        """
        from topologicpy.Cluster import Cluster
        if not isinstance(cluster, topologic.Cluster):
            print("Face.ByEdgesCluster - Warning: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = Cluster.Edges(cluster)
        if len(edges) < 1:
            print("Face.ByEdgesCluster - Warning: The input cluster parameter does not contain any valid edges. Returning None.")
            return None
        return Face.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def ByOffset(face: topologic.Face, offset: float = 1.0, miter: bool = False,
                 miterThreshold: float = None, offsetKey: str = None,
                 miterThresholdKey: str = None, step: bool = True, tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates an offset face from the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        offset : float , optional
            The desired offset distance. The default is 1.0.
        miter : bool , optional
            if set to True, the corners will be mitered. The default is False.
        miterThreshold : float , optional
            The distance beyond which a miter should be added. The default is None which means the miter threshold is set to the offset distance multiplied by the square root of 2.
        offsetKey : str , optional
            If specified, the dictionary of the edges will be queried for this key to sepcify the desired offset. The default is None.
        miterThresholdKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to sepcify the desired miter threshold distance. The default is None.
        step : bool , optional
            If set to True, The transition between collinear edges with different offsets will be a step. Otherwise, it will be a continous edge. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        if not isinstance(face, topologic.Face):
            print("Face.ByOffset - Warning: The input face parameter is not a valid toplogic face. Returning None.")
            return None
        eb = Face.Wire(face)
        internal_boundaries = Face.InternalBoundaries(face)
        offset_external_boundary = Wire.ByOffset(wire=eb, offset=offset, miter=miter, miterThreshold=miterThreshold, offsetKey=offsetKey, miterThresholdKey=miterThresholdKey, step=step)
        offset_internal_boundaries = []
        for internal_boundary in internal_boundaries:
            offset_internal_boundaries.append(Wire.ByOffset(wire=internal_boundary, offset=offset, miter=miter, miterThreshold=miterThreshold, offsetKey=offsetKey, miterThresholdKey=miterThresholdKey, step=step))
        return Face.ByWires(offset_external_boundary, offset_internal_boundaries, tolerance=tolerance)
    
    @staticmethod
    def ByShell(shell: topologic.Shell, origin: topologic.Vertex = None, angTolerance: float = 0.1, tolerance: float = 0.0001)-> topologic.Face:
        """
        Creates a face by merging the faces of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        
        if not isinstance(shell, topologic.Shell):
            print("Face.ByShell - Error: The input shell parameter is not a valid toplogic shell. Returning None.")
            return None
        
        if origin == None:
            origin = Topology.Centroid(shell)
        if not isinstance(origin, topologic.Vertex):
            print("Face.ByShell - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        # Try the simple method first
        face = None
        ext_boundary = Wire.RemoveCollinearEdges(Shell.ExternalBoundary(shell))
        if isinstance(ext_boundary, topologic.Wire):
            face = Face.ByWire(ext_boundary)
        elif isinstance(ext_boundary, topologic.Cluster):
            wires = Topology.Wires(ext_boundary)
            faces = [Face.ByWire(w) for w in wires]
            areas = [Face.Area(f) for f in faces]
            wires = Helper.Sort(wires, areas, reverseFlags=[True])
            face = Face.ByWires(wires[0], wires[1:])

        if isinstance(face, topologic.Face):
            return face
        world_origin = Vertex.Origin()
        planar_shell = Shell.Planarize(shell)
        normal = Face.Normal(Topology.Faces(planar_shell)[0])
        planar_shell = Topology.Flatten(planar_shell, origin=origin, direction=normal)
        vertices = Shell.Vertices(planar_shell)
        new_vertices = []
        for v in vertices:
            x, y, z = Vertex.Coordinates(v)
            new_v = Vertex.ByCoordinates(x,y,0)
            new_vertices.append(new_v)
        planar_shell = Topology.SelfMerge(Topology.ReplaceVertices(planar_shell, verticesA=vertices, verticesB=new_vertices), tolerance=tolerance)
        ext_boundary = Shell.ExternalBoundary(planar_shell, tolerance=tolerance)
        ext_boundary = Topology.RemoveCollinearEdges(ext_boundary, angTolerance)
        if not isinstance(ext_boundary, topologic.Topology):
            print("Face.ByShell - Error: Could not derive the external boundary of the input shell parameter. Returning None.")
            return None

        if isinstance(ext_boundary, topologic.Wire):
            if not Topology.IsPlanar(ext_boundary, tolerance=tolerance):
                ext_boundary = Wire.Planarize(ext_boundary, origin=origin, tolerance=tolerance)
            ext_boundary = Topology.RemoveCollinearEdges(ext_boundary, angTolerance)
            try:
                face = Face.ByWire(ext_boundary)
                face = Topology.Unflatten(face, origin=origin, direction=normal)
                return face
            except:
                print("Face.ByShell - Error: The operation failed. Returning None.")
                return None
        elif isinstance(ext_boundary, topologic.Cluster): # The shell has holes.
            wires = []
            _ = ext_boundary.Wires(None, wires)
            faces = []
            areas = []
            for wire in wires:
                aFace = Face.ByWire(wire, tolerance=tolerance)
                if not isinstance(aFace, topologic.Face):
                    print("Face.ByShell - Error: The operation failed. Returning None.")
                    return None
                anArea = abs(Face.Area(aFace))
                faces.append(aFace)
                areas.append(anArea)
            max_index = areas.index(max(areas))
            ext_boundary = faces[max_index]
            int_boundaries = list(set(faces) - set([ext_boundary]))
            int_wires = []
            for int_boundary in int_boundaries:
                temp_wires = []
                _ = int_boundary.Wires(None, temp_wires)
                int_wires.append(Topology.RemoveCollinearEdges(temp_wires[0], angTolerance))
            temp_wires = []
            _ = ext_boundary.Wires(None, temp_wires)
            ext_wire = Topology.RemoveCollinearEdges(temp_wires[0], angTolerance)
            face = Face.ByWires(ext_wire, int_wires)
           
            face = Topology.Unflatten(face, origin=origin, direction=normal)
            return face
        else:
            return None
    
    @staticmethod
    def ByVertices(vertices: list, tolerance: float = 0.0001) -> topologic.Face:
        
        """
        Creates a face from the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
        
        w = Wire.ByVertices(vertexList, tolerance=tolerance)
        f = Face.ByWire(w, tolerance=tolerance)
        return f

    @staticmethod
    def ByVerticesCluster(cluster: topologic.Cluster, tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a face from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The crearted face.

        """
        from topologicpy.Cluster import Cluster
        if not isinstance(cluster, topologic.Cluster):
            return None
        vertices = Cluster.Vertices(cluster)
        return Face.ByVertices(vertices, tolerance=tolerance)

    @staticmethod
    def ByWire(wire: topologic.Wire, tolerance: float = 0.0001, silent=False) -> topologic.Face:
        """
        Creates a face from the input closed wire.

        Parameters
        ----------
        wire : topologic.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to False, error and warning messages are printed. Otherwise, they are not. The default is False.

        Returns
        -------
        topologic.Face or list
            The created face. If the wire is non-planar, the method will attempt to triangulate the wire and return a list of faces.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import random

        def triangulateWire(wire):
            wire = Topology.RemoveCollinearEdges(wire)
            vertices = Topology.Vertices(wire)
            shell = Shell.Delaunay(vertices)
            if isinstance(shell, topologic.Topology):
                return Topology.Faces(shell)
            else:
                return []
        if not isinstance(wire, topologic.Wire):
            if not silent:
                print("Face.ByWire - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if not Wire.IsClosed(wire):
            if not silent:
                print("Face.ByWire - Error: The input wire parameter is not a closed topologic wire. Returning None.")
            return None
        
        edges = Wire.Edges(wire)
        wire = Topology.SelfMerge(Cluster.ByTopologies(edges), tolerance=tolerance)
        vertices = Topology.Vertices(wire)
        fList = []
        if isinstance(wire, topologic.Wire):
            try:
                fList = topologic.Face.ByExternalBoundary(wire)
            except:
                if not silent:
                    print("Face.ByWire - Warning: Could not create face by external boundary. Trying other methods.")
                if len(vertices) > 3:
                    fList = triangulateWire(wire)
                else:
                    fList = []
        
        if not isinstance(fList, list):
            fList = [fList]

        returnList = []
        for f in fList:
            if Face.Area(f) < 0:
                wire = Face.ExternalBoundary(f)
                wire = Wire.Invert(wire)
                try:
                    f = topologic.Face.ByExternalBoundary(wire)
                    returnList.append(f)
                except:
                    pass
            else:
                returnList.append(f)
        if len(returnList) == 0:
            if not silent:
                print("Face.ByWire - Error: Could not build a face from the input wire parameter. Returning None.")
            return None
        elif len(returnList) == 1:
            return returnList[0]
        else:
            if not silent:
                print("Face.ByWire - Warning: Could not build a single face from the input wire parameter. Returning a list of faces.")
            return returnList

    @staticmethod
    def ByWires(externalBoundary: topologic.Wire, internalBoundaries: list = [], tolerance: float = 0.0001, silent: bool = False) -> topologic.Face:
        """
        Creates a face from the input external boundary (closed wire) and the input list of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : topologic.Wire
            The input external boundary.
        internalBoundaries : list , optional
            The input list of internal boundaries (closed wires). The default is an empty list.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to False, error messages are printed. Otherwise, they are not. The default is False.

        Returns
        -------
        topologic.Face
            The created face.

        """
        if not isinstance(externalBoundary, topologic.Wire):
            if not silent:
                print("Face.ByWires - Error: The input externalBoundary parameter is not a valid topologic wire. Returning None.")
            return None
        if not Wire.IsClosed(externalBoundary):
            if not silent:
                print("Face.ByWires - Error: The input externalBoundary parameter is not a closed topologic wire. Returning None.")
            return None
        ibList = [x for x in internalBoundaries if isinstance(x, topologic.Wire) and Wire.IsClosed(x)]
        face = None
        try:
            face = topologic.Face.ByExternalInternalBoundaries(externalBoundary, ibList, tolerance)
        except:
            if not silent:
                print("Face.ByWires - Error: The operation failed. Returning None.")
            face = None
        return face


    @staticmethod
    def ByWiresCluster(externalBoundary: topologic.Wire, internalBoundariesCluster: topologic.Cluster = None, tolerance: float = 0.0001, silent: bool = False) -> topologic.Face:
        """
        Creates a face from the input external boundary (closed wire) and the input cluster of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : topologic.Wire
            The input external boundary (closed wire).
        internalBoundariesCluster : topologic.Cluster
            The input cluster of internal boundaries (closed wires). The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to False, error messages are printed. Otherwise, they are not. The default is False.
        
        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        if not isinstance(externalBoundary, topologic.Wire):
            if not silent:
                print("Face.ByWiresCluster - Error: The input externalBoundary parameter is not a valid topologic wire. Returning None.")
            return None
        if not Wire.IsClosed(externalBoundary):
            if not silent:
                print("Face.ByWiresCluster - Error: The input externalBoundary parameter is not a closed topologic wire. Returning None.")
            return None
        if not internalBoundariesCluster:
            internalBoundaries = []
        elif not isinstance(internalBoundariesCluster, topologic.Cluster):
            if not silent:
                print("Face.ByWiresCluster - Error: The input internalBoundariesCluster parameter is not a valid topologic cluster. Returning None.")
            return None
        else:
            internalBoundaries = Cluster.Wires(internalBoundariesCluster)
        return Face.ByWires(externalBoundary, internalBoundaries, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def NorthArrow(origin: topologic.Vertex = None, radius: float = 0.5, sides: int = 16, direction: list = [0, 0, 1], northAngle: float = 0.0,
                   placement: str = "center", tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a north arrow.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        northAngle : float , optional
            The angular offset in degrees from the positive Y axis direction. The angle is measured in a counter-clockwise fashion where 0 is positive Y, 90 is negative X, 180 is negative Y, and 270 is positive X.
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created circle.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        if not origin:
            origin = Vertex.Origin()
        
        c = Face.Circle(origin=origin, radius=radius, sides=sides, direction=[0, 0, 1], placement="center", tolerance=tolerance)
        r = Face.Rectangle(origin=origin, width=radius*0.01,length=radius*1.2, placement="lowerleft")
        r = Topology.Translate(r, -0.005*radius,0,0)
        arrow = Topology.Difference(c, r, tolerance=tolerance)
        arrow = Topology.Rotate(arrow, origin=Vertex.Origin(), axis=[0, 0, 1], angle=northAngle)
        if placement.lower() == "lowerleft":
            arrow = Topology.Translate(arrow, radius, radius, 0)
        elif placement.lower() == "upperleft":
            arrow = Topology.Translate(arrow, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            arrow = Topology.Translate(arrow, -radius, radius, 0)
        elif placement.lower() == "upperright":
            arrow = Topology.Translate(arrow, -radius, -radius, 0)
        arrow = Topology.Place(arrow, originA=Vertex.Origin(), originB=origin)
        arrow = Topology.Orient(arrow, orign=origin, direction=direction)
        return arrow

    @staticmethod
    def Circle(origin: topologic.Vertex = None, radius: float = 0.5, sides: int = 16, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0, 0, 1],
                   placement: str = "center", tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
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
        return Face.ByWire(wire, tolerance=tolerance)

    @staticmethod
    def Compactness(face: topologic.Face, mantissa: int = 6) -> float:
        """
        Returns the compactness measure of the input face. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        face : topologic.Face
            The input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

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
        area = abs(Face.Area(face))
        compactness  = 0
        #From https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        if area <= 0:
            return None
        if perimeter <= 0:
            return None
        compactness = (math.pi*(2*math.sqrt(area/math.pi)))/perimeter
        return round(compactness, mantissa)

    @staticmethod
    def CompassAngle(face: topologic.Face, north: list = None, mantissa: int = 6) -> float:
        """
        Returns the horizontal compass angle in degrees between the normal vector of the input face and the input vector. The angle is measured in counter-clockwise fashion. Only the first two elements of the vectors are considered.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        north : list , optional
            The second vector representing the north direction. The default is the positive YAxis ([0,1,0]).
        mantissa : int, optional
            The length of the desired mantissa. The default is 6.
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
    def Edges(face: topologic.Face) -> list:
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

    @staticmethod
    def Einstein(origin: topologic.Vertex = None, radius: float = 0.5, direction: list = [0, 0, 1],
                 placement: str = "center", tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates an aperiodic monotile, also called an 'einstein' tile (meaning one tile in German, not the name of the famous physist). See https://arxiv.org/abs/2303.10798

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the tile. The default is None which results in the tiles first vertex being placed at (0, 0, 0).
        radius : float , optional
            The radius of the hexagon determining the size of the tile. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hexagon determining the location of the tile. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        --------
            topologic.Face
                The created Einstein tile.

        """
        from topologicpy.Wire import Wire

        wire = Wire.Einstein(origin=origin, radius=radius, direction=direction, placement=placement)
        if not isinstance(wire, topologic.Wire):
            print("Face.Einstein - Error: Could not create base wire for the Einstein tile. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)
    
    @staticmethod
    def ExteriorAngles(face: topologic.Face, includeInternalBoundaries=False, mantissa: int = 6) -> list:
        """
        Returns the exterior angles of the input face in degrees. The face must be planar.
        
        Parameters
        ----------
        face : topologic.Face
            The input face.
        includeInternalBoundaries : bool , optional
            If set to True and if the face has internal boundaries (holes), the returned list will be a nested list where the first list is the list
            of exterior angles of the external boundary and the second list will contain lists of the exterior angles of each of the
            internal boundaries (holes). For example: [[270,270,270,270], [[270,270,270,270],[300,300,300]]]. If not, the returned list will be
            a simple list of interior angles of the external boundary. For example: [270,270,270,270]. Please note that that the interior angles of the
            internal boundaries are considered to be those interior to the original face. Thus, they are exterior to the internal boundary.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        Returns
        -------
        list
            The list of exterior angles.
        """
        
        from topologicpy.Wire import Wire

        if not isinstance(face, topologic.Face):
            print("Face.ExteriorAngles - Error: The input face parameter is not a valid face. Returning None.")
            return None
        eb = Face.ExternalBoundary(face)
        return_list = Wire.ExteriorAngles(eb, mantissa=mantissa)
        if includeInternalBoundaries:
            internal_boundaries = Face.InternalBoundaries(face)
            ib_i_a_list = []
            if len(internal_boundaries) > 0:
                for ib in internal_boundaries:
                    ib_interior_angles = Wire.InteriorAngles(ib, mantissa=mantissa)
                    ib_i_a_list.append(ib_interior_angles)
            if len(ib_i_a_list) > 0:
                return_list = [return_list]+[ib_i_a_list]
        return return_list

    @staticmethod
    def ExternalBoundary(face: topologic.Face) -> topologic.Wire:
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
    def FacingToward(face: topologic.Face, direction: list = [0,0,-1], asVertex: bool = False, tolerance: float = 0.0001) -> bool:
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
    def Harmonize(face: topologic.Face, tolerance: float = 0.0001) -> topologic.Face:
        """
        Returns a harmonized version of the input face such that the *u* and *v* origins are always in the upperleft corner.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            print("Face.Harmonize - Error: The input face parameter is not a valid face. Returning None.")
            return None
        normal = Face.Normal(face)
        origin = Topology.Centroid(face)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)
        world_origin = Vertex.Origin()
        vertices = Wire.Vertices(Face.ExternalBoundary(flatFace))
        harmonizedEB = Wire.ByVertices(vertices)
        internalBoundaries = Face.InternalBoundaries(flatFace)
        harmonizedIB = []
        for ib in internalBoundaries:
            ibVertices = Wire.Vertices(ib)
            harmonizedIB.append(Wire.ByVertices(ibVertices))
        harmonizedFace = Face.ByWires(harmonizedEB, harmonizedIB, tolerance=tolerance)
        harmonizedFace = Topology.Unflatten(harmonizedFace, origin=origin, direction=normal)
        return harmonizedFace

    @staticmethod
    def InteriorAngles(face: topologic.Face, includeInternalBoundaries: bool = False, mantissa: int = 6) -> list:
        """
        Returns the interior angles of the input face in degrees. The face must be planar.
        
        Parameters
        ----------
        face : topologic.Face
            The input face.
        includeInternalBoundaries : bool , optional
            If set to True and if the face has internal boundaries (holes), the returned list will be a nested list where the first list is the list
            of interior angles of the external boundary and the second list will contain lists of the interior angles of each of the
            internal boundaries (holes). For example: [[90,90,90,90], [[90,90,90,90],[60,60,60]]]. If not, the returned list will be
            a simple list of interior angles of the external boundary. For example: [90,90,90,90]. Please note that that the interior angles of the
            internal boundaries are considered to be those interior to the original face. Thus, they are exterior to the internal boundary.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        Returns
        -------
        list
            The list of interior angles.
        """
        
        from topologicpy.Wire import Wire

        if not isinstance(face, topologic.Face):
            print("Face.InteriorAngles - Error: The input face parameter is not a valid face. Returning None.")
            return None
        eb = Face.ExternalBoundary(face)
        return_list = Wire.InteriorAngles(eb, mantissa=mantissa)
        if includeInternalBoundaries:
            internal_boundaries = Face.InternalBoundaries(face)
            ib_i_a_list = []
            if len(internal_boundaries) > 0:
                for ib in internal_boundaries:
                    ib_interior_angles = Wire.ExteriorAngles(ib, mantissa=mantissa)
                    ib_i_a_list.append(ib_interior_angles)
            if len(ib_i_a_list) > 0:
                return_list = [return_list]+[ib_i_a_list]
        return return_list
    
    @staticmethod
    def InternalBoundaries(face: topologic.Face) -> list:
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
    def InternalVertex(face: topologic.Face, tolerance: float = 0.0001) -> topologic.Vertex:
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
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        if not isinstance(face, topologic.Face):
            return None
        v = Topology.Centroid(face)
        if Vertex.IsInternal(v, face, tolerance=tolerance):
            return v
        l = [0.4,0.6,0.3,0.7,0.2,0.8,0.1,0.9]
        for u in l:
            for v in l:
                v = Face.VertexByParameters(face, u, v)
                if Vertex.IsInternal(v, face, tolerance=tolerance):
                    return v
        v = topologic.FaceUtility.InternalVertex(face, tolerance)
        return v

    @staticmethod
    def Invert(face: topologic.Face, tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a face that is an inverse (mirror) of the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            inverted_face = Face.ByWire(inverted_wire, tolerance=tolerance)
        else:
            inverted_face = Face.ByWires(inverted_wire, internal_boundaries, tolerance=tolerance)
        return inverted_face

    @staticmethod
    def IsCoplanar(faceA: topologic.Face, faceB: topologic.Face, tolerance: float = 0.0001) -> bool:
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
        if not isinstance(faceA, topologic.Face):
            print("Face.IsInide - Error: The input faceA parameter is not a valid topologic face. Returning None.")
            return None
        if not isinstance(faceB, topologic.Face):
            print("Face.IsInide - Error: The input faceB parameter is not a valid topologic face. Returning None.")
            return None
        dirA = Face.NormalAtParameters(faceA, 0.5, 0.5, "xyz", 3)
        dirB = Face.NormalAtParameters(faceB, 0.5, 0.5, "xyz", 3)
        return Vector.IsCollinear(dirA, dirB)
    
    @staticmethod
    def MedialAxis(face: topologic.Face, resolution: int = 0, externalVertices: bool = False, internalVertices: bool = False, toLeavesOnly: bool = False, angTolerance: float = 0.1, tolerance: float = 0.0001) -> topologic.Wire:
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
        angTolerance : float , optional
            The desired angular tolerance in degrees for removing collinear edges. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
                u = Edge.ParameterAtVertex(edge, vertex, mantissa=6)
                if not u:
                    continue
                if 0<u<1:
                    return True
            return False

        # Flatten the input face
        origin = Topology.Centroid(face)
        normal = Face.Normal(face)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)

        # Create a Vertex at the world's origin (0, 0, 0)
        world_origin = Vertex.Origin()

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

        tempWire = Topology.SelfMerge(Cluster.ByTopologies(medialAxisEdges), tolerance=tolerance)
        if isinstance(tempWire, topologic.Wire) and angTolerance > 0:
            tempWire = Wire.RemoveCollinearEdges(tempWire, angTolerance=angTolerance)
        medialAxisEdges = Wire.Edges(tempWire)
        for v in theVertices:
            nv = Vertex.NearestVertex(v, tempWire, useKDTree=False)

            if isinstance(nv, topologic.Vertex):
                if toLeavesOnly:
                    adjVertices = Topology.AdjacentTopologies(nv, tempWire)
                    if len(adjVertices) < 2:
                        medialAxisEdges.append(Edge.ByVertices([nv, v], tolerance=tolerance))
                else:
                    medialAxisEdges.append(Edge.ByVertices([nv, v], tolerance=tolerance))
        medialAxis = Topology.SelfMerge(Cluster.ByTopologies(medialAxisEdges), tolerance=tolerance)
        if isinstance(medialAxis, topologic.Wire) and angTolerance > 0:
            medialAxis = Topology.RemoveCollinearEdges(medialAxis, angTolerance=angTolerance)
        medialAxis = Topology.Unflatten(medialAxis, origin=origin,direction=normal)
        return medialAxis

    @staticmethod
    def Normal(face: topologic.Face, outputType: str = "xyz", mantissa: int = 6) -> list:
        """
        Returns the normal vector to the input face. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        list
            The normal vector to the input face. This is computed at the approximate center of the face.

        """
        return Face.NormalAtParameters(face, u=0.5, v=0.5, outputType=outputType, mantissa=mantissa)

    @staticmethod
    def NormalAtParameters(face: topologic.Face, u: float = 0.5, v: float = 0.5, outputType: str = "xyz", mantissa: int = 6) -> list:
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
            The desired length of the mantissa. The default is 6.

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
    def NormalEdge(face: topologic.Face, length: float = 1.0, tolerance: float = 0.0001, silent: bool = False) -> topologic.Edge:
        """
        Returns the normal vector to the input face as an edge with the desired input length. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        length : float , optional
            The desired length of the normal edge. The default is 1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Edge
            The created normal edge to the input face. This is computed at the approximate center of the face.

        """
        from topologicpy.Edge import Edge

        if not isinstance(face, topologic.Face):
            if not silent:
                print("Face.NormalEdge - Error: The input face parameter is not a valid face. Retuning None.")
            return None
        if length < tolerance:
            if not silent:
                print("Face.NormalEdge - Error: The input length parameter is less than the input tolerance. Retuning None.")
            return None
        iv = Face.InternalVertex(face)
        u, v = Face.VertexParameters(face, iv)
        vec = Face.NormalAtParameters(face, u=u, v=v)
        ev = Topology.TranslateByDirectionDistance(iv, vec, length)
        return Edge.ByVertices([iv, ev], tolerance=tolerance, silent=silent)

    @staticmethod
    def NormalEdgeAtParameters(face: topologic.Face, u: float = 0.5, v: float = 0.5, length: float = 1.0, tolerance: float = 0.0001) -> topologic.Edge:
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
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
        return Edge.ByVertices([sv, ev], tolerance=tolerance, silent=True)

    @staticmethod
    def PlaneEquation(face: topologic.Face, mantissa: int = 6) -> dict:
        """
        Returns the a, b, c, d coefficients of the plane equation of the input face. The input face is assumed to be planar.
        
        Parameters
        ----------
        face : topologic.Face
            The input face.
        
        Returns
        -------
        dict
            The dictionary containing the coefficients of the plane equation. The keys of the dictionary are: ["a", "b", "c", "d"].

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        import random
        import time

        if not isinstance(face, topologic.Face):
            print("Face.PlaneEquation - Error: The input face is not a valid topologic face. Returning None.")
            return None
        vertices = Topology.Vertices(face)
        if len(vertices) < 3:
            print("Face.PlaneEquation - Error: The input face has less than 3 vertices. Returning None.")
            return None
        return Vertex.PlaneEquation(vertices, mantissa=mantissa)
    
    @staticmethod
    def Planarize(face: topologic.Face, origin: topologic.Vertex = None,
                  tolerance: float = 0.0001) -> topologic.Face:
        """
        Planarizes the input face such that its center of mass is located at the input origin and its normal is pointed in the input direction.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        origin : topologic.Vertex , optional
            The desired vertex to use as the origin of the plane to project the face unto. If set to None, the centroidof the input face is used. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.Face
            The planarized face.

        """

        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(face, topologic.Face):
            return None
        if not isinstance(origin, topologic.Vertex):
            origin = Topology.Centroid(face)
        eb = Face.ExternalBoundary(face)
        plan_eb = Wire.Planarize(eb, origin=origin)
        ib_list = Face.InternalBoundaries(face)
        plan_ib_list = []
        for ib in ib_list:
            plan_ib_list.append(Wire.Planarize(ib, origin=origin))
        plan_face = Face.ByWires(plan_eb, plan_ib_list)
        return plan_face

    @staticmethod
    def Project(faceA: topologic.Face, faceB: topologic.Face, direction : list = None,
                mantissa: int = 6, tolerance: float = 0.0001) -> topologic.Face:
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
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The projected Face.

        """

        from topologicpy.Wire import Wire

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
        p_eb = Wire.Project(wire=eb, face = faceB, direction=direction, mantissa=mantissa, tolerance=tolerance)
        p_ib_list = []
        for ib in ib_list:
            temp_ib = Wire.Project(wire=ib, face = faceB, direction=direction, mantissa=mantissa, tolerance=tolerance)
            if temp_ib:
                p_ib_list.append(temp_ib)
        return Face.ByWires(p_eb, p_ib_list, tolerance=tolerance)

    @staticmethod
    def RectangleByPlaneEquation(origin: topologic.Vertex = None, width: float = 1.0, length: float = 1.0, placement: str = "center", equation: dict = None, tolerance: float = 0.0001) -> topologic.Face:
        from topologicpy.Vertex import Vertex
        # Extract coefficients of the plane equation
        a = equation['a']
        b = equation['b']
        c = equation['c']
        d = equation['d']

        # Calculate the normal vector of the plane
        direction = np.array([a, b, c], dtype=float)
        direction /= np.linalg.norm(direction)
        direction = [x for x in direction]

        return Face.Rectangle(origin=origin, width=width, length=length, direction = direction, placement=placement, tolerance=tolerance)

    @staticmethod
    def Rectangle(origin: topologic.Vertex = None, width: float = 1.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        
        wire = Wire.Rectangle(origin=origin, width=width, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            print("Face.Rectangle - Error: Could not create the base wire for the rectangle. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)
    
    @staticmethod
    def RemoveCollinearEdges(face: topologic.Face, angTolerance: float = 0.1, tolerance: float = 0.0001) -> topologic.Wire:
        """
        Removes any collinear edges in the input face.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created face without any collinear edges.

        """
        from topologicpy.Wire import Wire

        if not isinstance(face, topologic.Face):
            print("Face.RemoveCollinearEdges - Error: The input face parameter is not a valid face. Returning None.")
            return None
        eb = Wire.RemoveCollinearEdges(Face.Wire(face), angTolerance=angTolerance, tolerance=tolerance)
        ib = [Wire.RemoveCollinearEdges(w, angTolerance=angTolerance, tolerance=tolerance) for w in Face.InternalBoundaries(face)]
        return Face.ByWires(eb, ib)
    
    @staticmethod
    def Skeleton(face, tolerance=0.001):
        """
            Creates a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number than the usual 0.0001 as it was found to work better)

        Returns
        -------
        topologic.Wire
            The created straight skeleton.

        """
        from topologicpy.Wire import Wire
        if not isinstance(face, topologic.Face):
            print("Face.Skeleton - Error: The input face is not a valid topologic face. Retruning None.")
            return None
        return Wire.Skeleton(face, tolerance=tolerance)
    
    @staticmethod
    def Square(origin: topologic.Vertex = None, size: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a square.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the square. The default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the square. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created square.

        """
        return Face.Rectangle(origin=origin, width=size, length=size, direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def Star(origin: topologic.Vertex = None, radiusA: float = 1.0, radiusB: float = 0.4, rays: int = 5, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a star.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the star. The default is None which results in the star being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the star. The default is 1.0.
        radiusB : float , optional
            The outer radius of the star. The default is 0.4.
        rays : int , optional
            The number of star rays. The default is 5.
        direction : list , optional
            The vector representing the up direction of the star. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        wire = Wire.Star(origin=origin, radiusA=radiusA, radiusB=radiusB, rays=rays, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            print("Face.Rectangle - Error: Could not create the base wire for the star. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)

    @staticmethod
    def Trapezoid(origin: topologic.Vertex = None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001) -> topologic.Face:
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic.Vertex, optional
            The location of the origin of the trapezoid. The default is None which results in the trapezoid being placed at (0, 0, 0).
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
            The vector representing the up direction of the trapezoid. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Face
            The created trapezoid.

        """
        from topologicpy.Wire import Wire
        wire = Wire.Trapezoid(origin=origin, widthA=widthA, widthB=widthB, offsetA=offsetA, offsetB=offsetB, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, topologic.Wire):
            print("Face.Rectangle - Error: Could not create the base wire for the trapezoid. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)

    @staticmethod
    def Triangulate(face:topologic.Face, mode: int = 0, meshSize: float = None, tolerance: float = 0.0001) -> list:
        """
        Triangulates the input face and returns a list of faces.

        Parameters
        ----------
        face : topologic.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        mode : int , optional
            The desired mode of meshing algorithm. Several options are available:
            0: Classic
            1: MeshAdapt
            3: Initial Mesh Only
            5: Delaunay
            6: Frontal-Delaunay
            7: BAMG
            8: Fontal-Delaunay for Quads
            9: Packing of Parallelograms
            All options other than 0 (Classic) use the gmsh library. See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            WARNING: The options that use gmsh can be very time consuming and can create very heavy geometry.
        meshSize : float , optional
            The desired size of the mesh when using the "mesh" option. If set to None, it will be
            calculated automatically and set to 10% of the overall size of the face.

        Returns
        -------
        list
            The list of triangles of the input face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        # This function was contributed by Yidan Xue.
        def generate_gmsh(face, mode="mesh", meshSize = None, tolerance = 0.0001):
            """
            Creates a gmsh of triangular meshes from the input face.

            Parameters
            ----------
            face : topologic.Face
                The input face.
            meshSize : float , optional
                The desired mesh size.
            tolerance : float , optional
                The desired tolerance. The default is 0.0001.
            
            Returns
            ----------
            topologic.Shell
                The shell of triangular meshes.

            """
            import os
            import warnings
            try:
                import numpy as np
            except:
                print("Face.Triangulate - Warning: Installing required numpy library.")
                try:
                    os.system("pip install numpy")
                except:
                    os.system("pip install numpy --user")
                try:
                    import numpy as np
                    print("Face.Triangulate - Warning: numpy library installed correctly.")
                except:
                    warnings.warn("Face.Triangulate - Error: Could not import numpy. Please try to install numpy manually. Returning None.")
                    return None
            try:
                import gmsh
            except:
                print("Face.Triangulate - Warning: Installing required gmsh library.")
                try:
                    os.system("pip install gmsh")
                except:
                    os.system("pip install gmsh --user")
                try:
                    import gmsh
                    print("Face.Triangulate - Warning: gmsh library installed correctly.")
                except:
                    warnings.warn("Face.Triangulate - Error: Could not import gmsh. Please try to install gmsh manually. Returning None.")
                    return None

            import topologic_core as topologic
            from topologicpy.Vertex import Vertex
            from topologicpy.Wire import Wire
            from topologicpy.Face import Face

            if not isinstance(face, topologic.Face):
                print("Shell.ByMeshFace - Error: The input face parameter is not a valid face. Returning None.")
                return None
            if not meshSize:
                bounding_face = Face.BoundingRectangle(face)
                bounding_face_vertices = Face.Vertices(bounding_face)
                bounding_face_vertices_x = [Vertex.X(i) for i in bounding_face_vertices]
                bounding_face_vertices_y = [Vertex.Y(i) for i in bounding_face_vertices]
                width = max(bounding_face_vertices_x)-min(bounding_face_vertices_x)
                length = max(bounding_face_vertices_y)-min(bounding_face_vertices_y)
                meshSize = max([width,length])//10
            
            gmsh.initialize()
            face_external_boundary = Face.ExternalBoundary(face)
            external_vertices = Wire.Vertices(face_external_boundary)
            external_vertex_number = len(external_vertices)
            for i in range(external_vertex_number):
                gmsh.model.geo.addPoint(Vertex.X(external_vertices[i]), Vertex.Y(external_vertices[i]), Vertex.Z(external_vertices[i]), meshSize, i+1)
            for i in range(external_vertex_number):
                if i < external_vertex_number-1:
                    gmsh.model.geo.addLine(i+1, i+2, i+1)
                else:
                    gmsh.model.geo.addLine(i+1, 1, i+1)
            gmsh.model.geo.addCurveLoop([i+1 for i in range(external_vertex_number)], 1)
            current_vertex_number = external_vertex_number
            current_edge_number = external_vertex_number
            current_wire_number = 1

            face_internal_boundaries = Face.InternalBoundaries(face)
            if face_internal_boundaries:
                internal_face_number = len(face_internal_boundaries)
                for i in range(internal_face_number):
                    face_internal_boundary = face_internal_boundaries[i]
                    internal_vertices = Wire.Vertices(face_internal_boundary)
                    internal_vertex_number = len(internal_vertices)
                    for j in range(internal_vertex_number):
                        gmsh.model.geo.addPoint(Vertex.X(internal_vertices[j]), Vertex.Y(internal_vertices[j]), Vertex.Z(internal_vertices[j]), meshSize, current_vertex_number+j+1)
                    for j in range(internal_vertex_number):
                        if j < internal_vertex_number-1:
                            gmsh.model.geo.addLine(current_vertex_number+j+1, current_vertex_number+j+2, current_edge_number+j+1)
                        else:
                            gmsh.model.geo.addLine(current_vertex_number+j+1, current_vertex_number+1, current_edge_number+j+1)
                    gmsh.model.geo.addCurveLoop([current_edge_number+i+1 for i in range(internal_vertex_number)], current_wire_number+1)
                    current_vertex_number = current_vertex_number+internal_vertex_number
                    current_edge_number = current_edge_number+internal_vertex_number
                    current_wire_number = current_wire_number+1

            gmsh.model.geo.addPlaneSurface([i+1 for i in range(current_wire_number)])
            gmsh.model.geo.synchronize()
            if mode not in [1,3,5,6,7,8,9]:
                mode = 6
            gmsh.option.setNumber("Mesh.Algorithm", mode)
            gmsh.model.mesh.generate(2)         # For a 2D mesh
            nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(-1, -1)
            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(-1, -1)
            gmsh.finalize()
            
            vertex_number = len(nodeTags)
            vertices = []
            for i in range(vertex_number):
                vertices.append(Vertex.ByCoordinates(nodeCoords[3*i],nodeCoords[3*i+1],nodeCoords[3*i+2]))

            faces = []
            for n in range(len(elemTypes)):
                vn = elemTypes[n]+1
                et = elemTags[n]
                ent = elemNodeTags[n]
                if vn==3:
                    for i in range(len(et)):
                        face_vertices = []
                        for j in range(vn):
                            face_vertices.append(vertices[np.where(nodeTags==ent[i*vn+j])[0][0]])
                        faces.append(Face.ByVertices(face_vertices))
            return faces

        if not isinstance(face, topologic.Face):
            print("Face.Triangulate - Error: The input face parameter is not a valid face. Returning None.")
            return None
        vertices = Topology.Vertices(face)
        if len(vertices) == 3: # Already a triangle
            return [face]
        origin = Topology.Centroid(face)
        normal = Face.Normal(face)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)

        if mode == 0:
            shell_faces = []
            for i in range(0,5,1):
                try:
                    _ = topologic.FaceUtility.Triangulate(flatFace, float(i)*0.1, shell_faces)
                    break
                except:
                    continue
        else:
            shell_faces = generate_gmsh(flatFace, mode = mode, meshSize = meshSize, tolerance = tolerance)
            
        if len(shell_faces) < 1:
            return []
        finalFaces = []
        for f in shell_faces:
            f = Topology.Unflatten(f, origin=origin, direction=normal)
            if Face.Angle(face, f) > 90:
                wire = Face.ExternalBoundary(f)
                wire = Wire.Invert(wire)
                f = topologic.Face.ByExternalBoundary(wire)
                finalFaces.append(f)
            else:
                finalFaces.append(f)
        return finalFaces

    @staticmethod
    def TrimByWire(face: topologic.Face, wire: topologic.Wire, reverse: bool = False) -> topologic.Face:
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
    def VertexByParameters(face: topologic.Face, u: float = 0.5, v: float = 0.5) -> topologic.Vertex:
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
    def VertexParameters(face: topologic.Face, vertex: topologic.Vertex, outputType: str = "uv", mantissa: int = 6) -> list:
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
            The desired length of the mantissa. The default is 6.

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
    def Vertices(face: topologic.Face) -> list:
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
    def Wire(face: topologic.Face) -> topologic.Wire:
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
    def Wires(face: topologic.Face) -> list:
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
