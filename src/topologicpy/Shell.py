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
import warnings

try:
    from tqdm.auto import tqdm
except:
    print("Shell - Installing required tqdm library.")
    try:
        os.system("pip install tqdm")
    except:
        os.system("pip install tqdm --user")
    try:
        from tqdm.auto import tqdm
        print("Shell - tqdm library installed correctly.")
    except:
        warnings.warn("Shell - Error: Could not import tqdm.")

try:
    from scipy.spatial import Delaunay
    from scipy.spatial import Voronoi
except:
    print("Shell - Install required scipy library.")
    try:
        os.system("pip install scipy")
    except:
        os.system("pip install scipy --user")
    try:
        from scipy.spatial import Delaunay
        from scipy.spatial import Voronoi
    except:
        warnings.warn("Shell - Error: Could not import scipy.")

class Shell():
    @staticmethod
    def ByDisjointFaces(externalBoundary,
                        faces,
                        maximumGap: float = 0.5,
                        mergeJunctions: bool = False,
                        threshold: float = 0.5,
                        uSides: int = 1,
                        vSides: int = 1,
                        transferDictionaries: bool = False,
                        mantissa: int = 6,
                        tolerance: float = 0.0001):
        """
        Creates a shell from an input list of disjointed faces. THIS IS STILL EXPERIMENTAL

        Parameters
        ----------
        externalBoundary : topologic_core.Face
            The input external boundary of the faces. This resembles a ribbon (face with hole) where its interior boundary touches the edges of the input list of faces.
        faces : list
            The input list of faces.
        maximumGap : float , optional
            The length of the maximum gap between the faces. The default is 0.5.
        mergeJunctions : bool , optional
            If set to True, the interior junctions are merged into a single vertex. Otherwise, diagonal edges are added to resolve transitions between different gap distances.
        threshold : float , optional
            The desired threshold under which vertices are merged into a single vertex. The default is 0.5.
        uSides : int , optional
            The desired number of sides along the X axis for the grid that subdivides the input faces to aid in processing. The default is 1.
        vSides : int , optional
            The desired number of sides along the Y axis for the grid that subdivides the input faces to aid in processing. The default is 1.
        transferDictionaries : bool, optional.
            If set to True, the dictionaries in the input list of faces are transfered to the faces of the resulting shell. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created Shell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Helper import Helper
        from topologicpy.Topology import Topology
        from topologicpy.Grid import Grid
        from topologicpy.Dictionary import Dictionary

        def removeShards(edges, hostTopology, maximumGap=0.5):
            returnEdges = []
            for e in tqdm(edges, desc="Removing Shards", leave=False):
                if Edge.Length(e) < maximumGap:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    sEdges = Topology.SuperTopologies(sv, hostTopology, "edge")
                    sn = len(sEdges)
                    eEdges = Topology.SuperTopologies(ev, hostTopology, "edge")
                    en = len(eEdges)
                    if sn >= 2 and en >= 2:
                        returnEdges.append(e)
                else:
                    returnEdges.append(e)
            return returnEdges

        def extendEdges(edges, hostTopology, maximumGap=0.5):
            returnEdges = []
            for e in tqdm(edges, desc="Extending Edges", leave=False):
                sv = Edge.StartVertex(e)
                ev = Edge.EndVertex(e)
                sEdges = Topology.SuperTopologies(sv, hostTopology, "edge")
                sn = len(sEdges)
                eEdges = Topology.SuperTopologies(ev, hostTopology, "edge")
                en = len(eEdges)
                if sn == 1:
                    ee = Edge.Extend(e, distance=maximumGap, bothSides=False, reverse=True)
                    returnEdges.append(ee)
                elif en == 1:
                    ee = Edge.Extend(e, distance=maximumGap, bothSides=False, reverse=False)
                    returnEdges.append(ee)
                else:
                    returnEdges.append(e)
            return returnEdges
        
        facesCluster = Cluster.ByTopologies(faces)
        internalBoundary = Face.ByWire(Face.InternalBoundaries(externalBoundary)[0], tolerance=tolerance)
        bb = Topology.BoundingBox(internalBoundary)
        bb_d = Topology.Dictionary(bb)
        unitU = Dictionary.ValueAtKey(bb_d, 'width') / uSides
        unitV = Dictionary.ValueAtKey(bb_d, 'length') / vSides
        uRange = [u*unitU for u in range(uSides)]
        vRange = [v*unitV for v in range(vSides)]
        grid = Grid.EdgesByDistances(internalBoundary, uRange=uRange, vRange=vRange, clip=True)
        grid = Topology.Slice(internalBoundary, grid, tolerance=tolerance)
        grid_faces = Topology.Faces(grid)
        skeletons = []
        for ib in tqdm(grid_faces, desc="Processing "+str(len(grid_faces))+" tiles", leave=False):
            building_shell = Topology.Slice(ib, facesCluster, tolerance=tolerance)
            wall_faces = Topology.Faces(building_shell)
            walls = []
            for w1 in wall_faces:
                iv = Topology.InternalVertex(w1, tolerance=tolerance)
                flag = False
                for w2 in faces:
                    if Vertex.IsInternal(iv, w2):
                        flag = True
                        break;
                if flag == False:
                    walls.append(w1)
            for wall in walls:
                skeleton = Wire.Skeleton(wall, tolerance=0.001) # This tolerance works better.
                skeleton = Topology.Difference(skeleton, facesCluster, tolerance=tolerance)
                skeleton = Topology.Difference(skeleton, Face.Wire(wall), tolerance=tolerance)
                skeletons.append(skeleton)
        if len(skeletons) > 0:
            skeleton_cluster = Cluster.ByTopologies(skeletons+[internalBoundary])
            skEdges = Topology.SelfMerge(Cluster.ByTopologies(removeShards(Topology.Edges(skeleton_cluster), skeleton_cluster, maximumGap=maximumGap)), tolerance=tolerance)
            if Topology.IsInstance(skEdges, "Edge"):
                skEdges = extendEdges([skEdges], skEdges, maximumGap=maximumGap)
            else:
                skEdges = extendEdges(Topology.Edges(skEdges), skEdges, maximumGap=maximumGap)
            if len(skEdges) < 1:
                print("ShellByDisjointFaces - Warning: No edges were extended.")
            #return Cluster.ByTopologies(skEdges)
        #print("ShellByDisjointFaces - Error: Could not derive central skeleton of interior walls. Returning None.")
        #return None

            shell = Topology.Slice(internalBoundary, skeleton_cluster, tolerance=tolerance)
            if mergeJunctions == True:
                vertices = Topology.Vertices(shell)
                centers = []
                used = []
                for v in vertices:
                    for w in vertices:
                        if not Topology.IsSame(v, w) and not w in used:
                            if Vertex.Distance(v, w, mantissa=mantissa) < threshold:
                                centers.append(v)
                                used.append(w)
                edges = Shell.Edges(shell)
                new_edges = []
                for e in edges:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    for v in centers:
                        if Vertex.Distance(sv, v, mantissa=mantissa) < threshold:
                            sv = v
                        if Vertex.Distance(ev, v, mantissa=mantissa) < threshold:
                            ev = v
                    new_edges.append(Edge.ByVertices([sv,ev], tolerance=tolerance))
                cluster = Cluster.ByTopologies(new_edges)

                vertices = Topology.Vertices(cluster)
                edges = Topology.Edges(shell)

                xList = list(set([Vertex.X(v, mantissa=mantissa) for v in vertices]))
                xList.sort()
                xList = Helper.MergeByThreshold(xList, 0.5)
                yList = list(set([Vertex.Y(v, mantissa=mantissa) for v in vertices]))
                yList.sort()
                yList = Helper.MergeByThreshold(yList, 0.5)
                yList.sort()

                centers = []

                new_edges = []

                for e in edges:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    svx = Vertex.X(sv, mantissa=mantissa)
                    svy = Vertex.Y(sv, mantissa=mantissa)
                    evx = Vertex.X(ev, mantissa=mantissa)
                    evy = Vertex.Y(ev, mantissa=mantissa)
                    for x in xList:
                        if abs(svx-x) < threshold:
                            svx = x
                            break;
                    for y in yList:
                        if abs(svy-y) < threshold:
                            svy = y
                            break;
                    sv = Vertex.ByCoordinates(svx, svy, 0)
                    for x in xList:
                        if abs(evx-x) < threshold:
                            evx = x
                            break;
                    for y in yList:
                        if abs(evy-y) < threshold:
                            evy = y
                            break;
                    sv = Vertex.ByCoordinates(svx, svy, 0)
                    ev = Vertex.ByCoordinates(evx, evy, 0)
                    new_edges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))

                cluster = Cluster.ByTopologies(new_edges)
                eb = Face.ByWire(Shell.ExternalBoundary(shell), tolerance=tolerance)
                shell = Topology.Slice(eb, cluster, tolerance=tolerance)
            if not Topology.IsInstance(shell, "Shell"):
                try:
                    temp_wires = [Wire.RemoveCollinearEdges(w, angTolerance=1.0) for w in Topology.Wires(shell)]
                    temp_faces = [Face.ByWire(w, tolerance=tolerance) for w in temp_wires]
                except:
                    temp_faces = Topology.Faces(shell)
                shell = Shell.ByFaces(temp_faces, tolerance=tolerance)
            if transferDictionaries == True:
                selectors = []
                for f in faces:
                    d = Topology.Dictionary(f)
                    s = Topology.InternalVertex(f, tolerance=tolerance)
                    s = Topology.SetDictionary(s, d)
                    selectors.append(s)
                shell = Topology.TransferDictionariesBySelectors(topology=shell, selectors=selectors, tranFaces=True, tolerance=tolerance)
            return shell
        return None

    @staticmethod
    def ByFaces(faces: list, tolerance: float = 0.0001, silent=False):
        """
        Creates a shell from the input list of faces.

        Parameters
        ----------
        faces : list
            The input list of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Shell
            The created Shell.

        """
        from topologicpy.Topology import Topology

        if not isinstance(faces, list):
            return None
        faceList = [x for x in faces if Topology.IsInstance(x, "Face")]
        if len(faceList) == 0:
            print("Shell.ByFaces - Error: The input faces list does not contain any valid faces. Returning None.")
            return None
        shell = topologic.Shell.ByFaces(faceList, tolerance) # Hook to Core
        if not Topology.IsInstance(shell, "Shell"):
            shell = Topology.SelfMerge(shell, tolerance=tolerance)
            if Topology.IsInstance(shell, "Shell"):
                return shell
            else:
                if not silent:
                    print("Shell.ByFaces - Error: Could not create shell. Returning None.")
                return None
        else:
            return shell

    @staticmethod
    def ByFacesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a shell from the input cluster of faces.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Shell
            The created shell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            return None
        faces = Topology.Faces(cluster)
        return Shell.ByFaces(faces, tolerance=tolerance)

    @staticmethod
    def ByThickenedWire(wire, offsetA: float = 1.0, offsetB: float = 1.0, tolerance: float = 0.0001):
        """
        Creates a shell by thickening the input wire. This method assumes the wire is manifold and planar.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire to be thickened.
        offsetA : float , optional
            The desired offset to the exterior of the wire. The default is 1.0.
        offsetB : float , optional
            The desired offset to the interior of the wire. The default is 1.0.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created shell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Shell.ByThickenedWire - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        
        f = Face.ByThickenedWire(wire, offsetA=offsetA, offsetB=offsetB, tolerance=tolerance)
        outside_wire = Wire.ByOffset(wire, offset=abs(offsetA)*-1, bisectors = False, tolerance=tolerance)
        inside_wire = Wire.ByOffset(wire, offset=abs(offsetB), bisectors = False, tolerance=tolerance)
        border = Topology.Merge(outside_wire, inside_wire)
        outside_wire = Wire.ByOffset(wire, offset=abs(offsetA)*-1, bisectors = True, tolerance=tolerance)
        inside_wire = Wire.ByOffset(wire, offset=abs(offsetB), bisectors = True, tolerance=tolerance)
        grid = Topology.Merge(outside_wire, inside_wire)
        bisectors = Topology.Difference(grid, border)
        return_shell = Topology.Slice(f, bisectors)
        if not Topology.IsInstance(return_shell, "Shell"):
            print("Shell.ByThickenedWire - Error: The operation failed. Returning None.")
            return None
        return return_shell

    @staticmethod
    def ByWires(wires: list, triangulate: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a shell by lofting through the input wires
        
        Parameters
        ----------
        wires : list
            The input list of wires.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
       
        Returns
        -------
        topologic_core.Shell
            The creates shell.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(wires, list):
            return None
        wireList = [x for x in wires if Topology.IsInstance(x, "Wire")]
        faces = []
        for i in range(len(wireList)-1):
            wire1 = wireList[i]
            wire2 = wireList[i+1]
            if Topology.Type(wire1) < Topology.TypeID("Edge") or Topology.Type(wire2) < Topology.TypeID("Edge"):
                return None
            if Topology.Type(wire1) == Topology.TypeID("Edge"):
                w1_edges = [wire1]
            else:
                w1_edges = Topology.Edges(wire1)
            if Topology.Type(wire2) == Topology.TypeID("Edge"):
                w2_edges = [wire2]
            else:
                w2_edges = Topology.Edges(wire2)
            if len(w1_edges) != len(w2_edges):
                return None
            if triangulate == True:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=silent)
                    except:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=silent)
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e4], tolerance=tolerance), tolerance=tolerance))
                    try:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=silent)
                    except:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=silent)
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3],tolerance=tolerance), tolerance=tolerance))
                    if e3 and e4:
                        e5 = Edge.ByVertices([e1.StartVertex(), e2.EndVertex()], tolerance=tolerance, silent=silent)
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e5, e4], tolerance=tolerance), tolerance=tolerance))
                        faces.append(Face.ByWire(Wire.ByEdges([e2, e5, e3], tolerance=tolerance), tolerance=tolerance))
                    elif e3:
                        verts = [Edge.StartVertex(e1), Edge.EndVertex(e1), Edge.StartVertex(e3), Edge.EndVertex(e3), Edge.StartVertex(e2), Edge.EndVertex(e2)]
                        verts = Vertex.Fuse(verts, tolerance=tolerance)
                        w = Wire.ByVertices(verts, close=True)
                        if Topology.IsInstance(w, "Wire"):
                            faces.append(Face.ByWire(w, tolerance=tolerance))
                        else:
                            if not silent:
                                print("Shell.ByWires - Warning: Could not create face.")
                    elif e4:
                        verts = [Edge.StartVertex(e1), Edge.EndVertex(e1), Edge.StartVertex(e4), Edge.EndVertex(e4), Edge.StartVertex(e2), Edge.EndVertex(e2)]
                        verts = Vertex.Fuse(verts, tolerance=tolerance)
                        w = Wire.ByVertices(verts, close=True)
                        if Topology.IsInstance(w, "Wire"):
                            faces.append(Face.ByWire(w, tolerance=tolerance))
                        else:
                            if not silent:
                                print("Shell.ByWires - Warning: Could not create face.")
            else:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=silent)
                    except:
                        try:
                            e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=silent)
                        except:
                            pass
                    try:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=silent)
                    except:
                        try:
                            e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=silent)
                        except:
                            pass
                    if e3 and e4:
                        try:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e4, e2, e3], tolerance=tolerance), tolerance=tolerance))
                        except:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e3, e2, e4], tolerance=tolerance), tolerance=tolerance))
                    elif e3:
                        verts = [Edge.StartVertex(e1), Edge.EndVertex(e1), Edge.StartVertex(e3), Edge.EndVertex(e3), Edge.StartVertex(e2), Edge.EndVertex(e2)]
                        verts = Vertex.Fuse(verts, tolerance=tolerance)
                        w = Wire.ByVertices(verts, close=True)
                        if Topology.IsInstance(w, "Wire"):
                            faces.append(Face.ByWire(w, tolerance=tolerance))
                        else:
                            if not silent:
                                print("Shell.ByWires - Warning: Could not create face.")
                    elif e4:
                        verts = [Edge.StartVertex(e1), Edge.EndVertex(e1), Edge.StartVertex(e4), Edge.EndVertex(e4), Edge.StartVertex(e2), Edge.EndVertex(e2)]
                        verts = Vertex.Fuse(verts, tolerance=tolerance)
                        w = Wire.ByVertices(verts, close=True)
                        if Topology.IsInstance(w, "Wire"):
                            faces.append(Face.ByWire(w, tolerance=tolerance))
                        else:
                            if not silent:
                                print("Shell.ByWires - Warning: Could not create face.")

        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=silent)
        if shell == None:
            if not silent:
                print("Shell.ByWires - Warning: Could not create shell. Returning a cluster of faces instead.")
            return Cluster.ByTopologies(faces)
        return shell

    @staticmethod
    def ByWiresCluster(cluster, triangulate: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a shell by lofting through the input cluster of wires

        Parameters
        ----------
        wires : topologic_core.Cluster
            The input cluster of wires.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Shell
            The creates shell.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not cluster:
            return None
        if not Topology.IsInstance(cluster, "Cluster"):
            return None
        wires = Cluster.Wires(cluster)
        return Shell.ByWires(wires, triangulate=triangulate, tolerance=tolerance, silent=silent)

    @staticmethod
    def Circle(origin= None, radius: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The  radius of the circle. The default is 0.5.
        sides : int , optional
            The number of sides of the circle. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created circle.
        """
        return Shell.Pie(origin=origin, radiusA=radius, radiusB=0, sides=sides, rings=1, fromAngle=fromAngle, toAngle=toAngle, direction=direction, placement=placement, tolerance=tolerance)

    @staticmethod
    def Delaunay(vertices: list, face= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a delaunay partitioning of the input vertices. The vertices must be coplanar. See https://en.wikipedia.org/wiki/Delaunay_triangulation.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic_core.Face , optional
            The input face. If specified, the delaunay triangulation is clipped to the face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        shell
            A shell representing the delaunay triangulation of the input vertices.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from random import sample
        from scipy.spatial import Delaunay as SCIDelaunay
        
        if not isinstance(vertices, list):
            return None
        vertices = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertices) < 3:
            return None

        if Topology.IsInstance(face, "Face"):
            # Flatten the face
            origin = Topology.Centroid(face)
            normal = Face.Normal(face, mantissa=mantissa)
            flatFace = Topology.Flatten(face, origin=origin, direction=normal)
            faceVertices = Topology.Vertices(face)
            vertices += faceVertices

            # Create a cluster of the input vertices
            verticesCluster = Cluster.ByTopologies(vertices)

            # Flatten the cluster using the same transformations
            verticesCluster = Topology.Flatten(verticesCluster, origin=origin, direction=normal)

            vertices = Topology.Vertices(verticesCluster)
        points = []
        for v in vertices:
            points.append([Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa)])
        delaunay = SCIDelaunay(points)
        simplices = delaunay.simplices

        faces = []
        for simplex in simplices:
            tempTriangleVertices = []
            tempTriangleVertices.append(vertices[simplex[0]])
            tempTriangleVertices.append(vertices[simplex[1]])
            tempTriangleVertices.append(vertices[simplex[2]])
            tempFace = Face.ByWire(Wire.ByVertices(tempTriangleVertices), tolerance=tolerance)
            faces.append(tempFace)

        shell = Shell.ByFaces(faces, tolerance=tolerance)
        if shell == None:
            shell = Cluster.ByTopologies(faces)
        
        if Topology.IsInstance(face, "Face"):
            edges = Topology.Edges(shell)
            shell = Topology.Slice(flatFace, Cluster.ByTopologies(edges))
            # Get the internal boundaries of the face
            wires = Face.InternalBoundaries(flatFace)
            ibList = []
            if len(wires) > 0:
                ibList = [Face.ByWire(w) for w in wires]
                cluster = Cluster.ByTopologies(ibList)
                shell = Topology.Difference(shell, cluster)
            shell = Topology.Unflatten(shell, origin=origin, direction=normal)
        return shell

    @staticmethod
    def Edges(shell) -> list:
        """
        Returns the edges of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        edges = []
        _ = shell.Edges(None, edges) # Hook to Core
        return edges

    @staticmethod
    def ExternalBoundary(shell, tolerance: float = 0.0001):
        """
        Returns the external boundary of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire or topologic_core.Cluster
            The external boundary of the input shell. If the shell has holes, the return value will be a cluster of wires.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        edges = Topology.Edges(shell)
        obEdges = []
        for anEdge in edges:
            faces = Topology.SuperTopologies(anEdge, shell, topologyType="face")
            if len(faces) == 1:
                obEdges.append(anEdge)
        return Topology.SelfMerge(Cluster.ByTopologies(obEdges), tolerance=tolerance)

    @staticmethod
    def Faces(shell) -> list:
        """
        Returns the faces of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of faces.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        faces = []
        _ = shell.Faces(None, faces)
        return faces
    
    @staticmethod
    def IsOnBoundary(shell, vertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is on the boundary of the input shell. Returns False otherwise. On the boundary is defined as being on the boundary of one of the shell's external or internal boundaries

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        vertex : topologic_core.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input shell. Returns False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        boundary = Shell.ExternalBoundary(shell, tolerance=tolerance)
        if Vertex.IsInternal(vertex, boundary, tolerance=tolerance):
            return True
        internal_boundaries = Shell.InternalBoundaries(shell, tolerance=tolerance)
        for ib in internal_boundaries:
            if Vertex.IsInternal(vertex, ib, tolerance=tolerance):
                return True
        return False
    
    @staticmethod
    def HyperbolicParaboloidRectangularDomain(origin= None,
                                              llVertex= None,
                                              lrVertex= None,
                                              ulVertex= None,
                                              urVertex= None,
                                              uSides: int = 10,
                                              vSides: int = 10,
                                              direction: list = [0, 0, 1],
                                              placement: str = "center",
                                              mantissa: int = 6,
                                              tolerance: float = 0.0001):
        """
        Creates a hyperbolic paraboloid with a rectangular domain.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin of the hyperbolic paraboloid. If set to None, it will be placed at the (0, 0, 0) origin. The default is None.
        llVertex : topologic_core.Vertex , optional
            The lower left corner of the hyperbolic paraboloid. If set to None, it will be set to (-0.5, -0.5, -0.5).
        lrVertex : topologic_core.Vertex , optional
            The lower right corner of the hyperbolic paraboloid. If set to None, it will be set to (0.5, -0.5, 0.5).
        ulVertex : topologic_core.Vertex , optional
            The upper left corner of the hyperbolic paraboloid. If set to None, it will be set to (-0.5, 0.5, 0.5).
        urVertex : topologic_core.Vertex , optional
            The upper right corner of the hyperbolic paraboloid. If set to None, it will be set to (0.5, 0.5, -0.5).
        uSides : int , optional
            The number of segments along the X axis. The default is 10.
        vSides : int , optional
            The number of segments along the Y axis. The default is 10.
        direction : list , optional
            The vector representing the up direction of the hyperbolic paraboloid. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hyperbolic paraboloid. This can be "center", "lowerleft", "bottom". It is case insensitive. The default is "center".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Shell
            The created hyperbolic paraboloid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(llVertex, "Vertex"):
            llVertex = Vertex.ByCoordinates(-0.5, -0.5, -0.5)
        if not Topology.IsInstance(lrVertex, "Vertex"):
            lrVertex = Vertex.ByCoordinates(0.5, -0.5, 0.5)
        if not Topology.IsInstance(ulVertex, "Vertex"):
            ulVertex = Vertex.ByCoordinates(-0.5, 0.5, 0.5)
        if not Topology.IsInstance(urVertex, "Vertex"):
            urVertex = Vertex.ByCoordinates(0.5, 0.5, -0.5)
        e1 = Edge.ByVertices([llVertex, lrVertex], tolerance=tolerance)
        e3 = Edge.ByVertices([urVertex, ulVertex], tolerance=tolerance)
        edges = []
        for i in range(uSides+1):
            v1 = Edge.VertexByParameter(e1, float(i)/float(uSides))
            v2 = Edge.VertexByParameter(e3, 1.0 - float(i)/float(uSides))
            edges.append(Edge.ByVertices([v1, v2], tolerance=tolerance))
        faces = []
        for i in range(uSides):
            for j in range(vSides):
                v1 = Edge.VertexByParameter(edges[i], float(j)/float(vSides))
                v2 = Edge.VertexByParameter(edges[i], float(j+1)/float(vSides))
                v3 = Edge.VertexByParameter(edges[i+1], float(j+1)/float(vSides))
                v4 = Edge.VertexByParameter(edges[i+1], float(j)/float(vSides))
                faces.append(Face.ByVertices([v1, v2, v4]))
                faces.append(Face.ByVertices([v4, v2, v3]))
        returnTopology = Shell.ByFaces(faces, tolerance=tolerance)
        if not returnTopology:
            returnTopology = None
        xOffset = 0
        yOffset = 0
        zOffset = 0
        x_min = min([Vertex.X(llVertex, mantissa=mantissa), Vertex.X(lrVertex, mantissa=mantissa), Vertex.X(ulVertex, mantissa=mantissa), Vertex.X(urVertex, mantissa=mantissa)])
        maxX = max([Vertex.X(llVertex, mantissa=mantissa), Vertex.X(lrVertex, mantissa=mantissa), Vertex.X(ulVertex, mantissa=mantissa), Vertex.X(urVertex, mantissa=mantissa)])
        y_min = min([Vertex.Y(llVertex, mantissa=mantissa), Vertex.Y(lrVertex, mantissa=mantissa), Vertex.Y(ulVertex, mantissa=mantissa), Vertex.Y(urVertex, mantissa=mantissa)])
        maxY = max([Vertex.Y(llVertex, mantissa=mantissa), Vertex.Y(lrVertex, mantissa=mantissa), Vertex.Y(ulVertex, mantissa=mantissa), Vertex.Y(urVertex, mantissa=mantissa)])
        z_min = min([Vertex.Z(llVertex, mantissa=mantissa), Vertex.Z(lrVertex, mantissa=mantissa), Vertex.Z(ulVertex, mantissa=mantissa), Vertex.Z(urVertex, mantissa=mantissa)])
        maxZ = max([Vertex.Z(llVertex, mantissa=mantissa), Vertex.Z(lrVertex, mantissa=mantissa), Vertex.Z(ulVertex, mantissa=mantissa), Vertex.Z(urVertex, mantissa=mantissa)])

        if placement.lower() == "lowerleft":
            xOffset = -x_min
            yOffset = -y_min
            zOffset = -z_min
        elif placement.lower() == "bottom":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -z_min
        elif placement.lower() == "center":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -(z_min + (maxZ - z_min)*0.5)
        returnTopology = Topology.Translate(returnTopology, xOffset, yOffset, zOffset)
        returnTopology = Topology.Place(returnTopology, originA=Vertex.Origin(), originB=origin)
        returnTopology = Topology.Orient(returnTopology, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return returnTopology
    
    @staticmethod
    def HyperbolicParaboloidCircularDomain(origin= None, radius: float = 0.5, sides: int = 36, rings: int = 10,
                                           A: float = 2.0, B: float = -2.0, direction: list = [0, 0, 1],
                                           placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a hyperbolic paraboloid with a circular domain. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin of the hyperbolic parabolid. If set to None, it will be placed at the (0, 0, 0) origin. The default is None.
        radius : float , optional
            The desired radius of the hyperbolic paraboloid. The default is 0.5.
        sides : int , optional
            The desired number of sides of the hyperbolic parabolid. The default is 36.
        rings : int , optional
            The desired number of concentric rings of the hyperbolic parabolid. The default is 10.
        A : float , optional
            The *A* constant in the equation z = A*x^2^ + B*y^2^. The default is 2.0.
        B : float , optional
            The *B* constant in the equation z = A*x^2^ + B*y^2^. The default is -2.0.
        direction : list , optional
            The  vector representing the up direction of the hyperbolic paraboloid. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "bottom". It is case insensitive. The default is "center".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Shell
            The created hyperbolic paraboloid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
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
                v1 = Vertex.ByCoordinates(x1,y1,z1)
                v2 = Vertex.ByCoordinates(x2,y2,z2)
                v3 = Vertex.ByCoordinates(x3,y3,z3)
                v4 = Vertex.ByCoordinates(x4,y4,z4)
                f1 = Face.ByVertices([v1,v2,v4])
                f2 = Face.ByVertices([v4,v2,v3])
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
            v1 = Vertex.ByCoordinates(x1,y1,z1)
            v2 = Vertex.ByCoordinates(x2,y2,z2)
            v3 = Vertex.ByCoordinates(x3,y3,z3)
            v4 = Vertex.ByCoordinates(x4,y4,z4)
            f1 = Face.ByVertices([v1,v2,v4])
            f2 = Face.ByVertices([v4,v2,v3])
            faces.append(f1)
            faces.append(f2)
        # Special Case: Center triangles
        r = vOffset
        x1 = 0
        y1 = 0
        z1 = 0
        v1 = Vertex.ByCoordinates(x1,y1,z1)
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
                v2 = Vertex.ByCoordinates(x2,y2,z2)
                v3 = Vertex.ByCoordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)
        a1 = math.radians(uOffset)*(sides-1)
        a2 = math.radians(360)
        x2 = math.sin(a1)*r
        y2 = math.cos(a1)*r
        z2 = A*x2*x2 + B*y2*y2
        x3 = math.sin(a2)*r
        y3 = math.cos(a2)*r
        z3 = A*x3*x3 + B*y3*y3
        v2 = Vertex.ByCoordinates(x2,y2,z2)
        v3 = Vertex.ByCoordinates(x3,y3,z3)
        f1 = Face.ByVertices([v2,v1,v3])
        faces.append(f1)
        returnTopology = Shell.ByFaces(faces, tolerance=tolerance)
        if not returnTopology:
            returnTopology = Cluster.ByTopologies(faces)
        vertices = []
        _ = returnTopology.Vertices(None, vertices)
        xList = []
        yList = []
        zList = []
        for aVertex in vertices:
            xList.append(Vertex.X(aVertex, mantissa=mantissa))
            yList.append(Vertex.Y(aVertex, mantissa=mantissa))
            zList.append(Vertex.Z(aVertex, mantissa=mantissa))
        x_min = min(xList)
        maxX = max(xList)
        y_min = min(yList)
        maxY = max(yList)
        z_min = min(zList)
        maxZ = max(zList)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = -x_min
            yOffset = -y_min
            zOffset = -z_min
        elif placement.lower() == "bottom":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -z_min
        elif placement.lower() == "center":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -(z_min + (maxZ - z_min)*0.5)
        returnTopology = Topology.Translate(returnTopology, xOffset, yOffset, zOffset)
        returnTopology = Topology.Place(returnTopology, originA=Vertex.Origin(), originB=origin)
        returnTopology = Topology.Orient(returnTopology, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return returnTopology
    
    @staticmethod
    def InternalBoundaries(shell, tolerance=0.0001):
        """
        Returns the internal boundaries (closed wires) of the input shell. Internal boundaries are considered holes.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of internal boundaries

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        edges = []
        _ = shell.Edges(None, edges)
        ibEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(shell, faces)
            if len(faces) > 1:
                ibEdges.append(anEdge)
        returnTopology = Topology.SelfMerge(Cluster.ByTopologies(ibEdges), tolerance=tolerance)
        wires = Topology.Wires(returnTopology)
        return wires

    
    @staticmethod
    def IsClosed(shell) -> bool:
        """
        Returns True if the input shell is closed. Returns False otherwise.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        bool
            True if the input shell is closed. False otherwise.

        """
        return shell.IsClosed()


    @staticmethod
    def Paraboloid(origin= None, focalLength=0.125, width: float = 1, length: float = 1, uSides: int = 16, vSides: int = 16,
                    direction: list = [0, 0, 1], placement: str ="center", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
            Creates a paraboloid. See https://en.wikipedia.org/wiki/Paraboloid

            Parameters
            ----------
            origin : topologic_core.Vertex , optional
                The origin location of the parabolic surface. The default is None which results in the parabolic surface being placed at (0, 0, 0).
            focalLength : float , optional
                The focal length of the parabola. The default is 1.
            width : float , optional
                The width of the parabolic surface. The default is 1.
            length : float , optional
                The length of the parabolic surface. The default is 1.
            uSides : int , optional
                The number of sides along the width. The default is 16.
            vSides : int , optional
                The number of sides along the length. The default is 16.
            direction : list , optional
                The vector representing the up direction of the parabolic surface. The default is [0, 0, 1].
            placement : str , optional
                The description of the placement of the origin of the parabolic surface. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
            mantissa : int , optional
                The desired length of the mantissa. The default is 6.
            tolerance : float , optional
                The desired tolerance. The default is 0.0001.
            silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
            
            Returns
            -------
            topologic_core.Shell
                The created paraboloid.

            """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        def create_triangulated_mesh(vertices, uSides, vSides):
            faces = []

            # Iterate over the grid of vertices to form triangular faces
            for i in range(uSides - 1):
                for j in range(vSides - 1):
                    # Get the indices of the vertices forming the current grid cell
                    v1 = vertices[i * vSides + j]
                    v2 = vertices[i * vSides + (j + 1)]
                    v3 = vertices[(i + 1) * vSides + j]
                    v4 = vertices[(i + 1) * vSides + (j + 1)]

                    # Create two triangles for each grid cell
                    # Triangle 1: (v1, v2, v3)
                    wire1 = Wire.ByVertices([v1, v2, v3])
                    face1 = Face.ByWire(wire1)
                    faces.append(face1)

                    # Triangle 2: (v3, v2, v4)
                    wire2 = Wire.ByVertices([v3, v2, v4])
                    face2 = Face.ByWire(wire2)
                    faces.append(face2)

            # Create the mesh (Shell) from the list of faces
            mesh = Shell.ByFaces(faces)
            return mesh
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        
        x_range = [-width*0.5, width*0.5]
        y_range = [-length*0.5, length*0.5]
        # Generate x and y values
        x_values = [x_range[0] + i * (x_range[1] - x_range[0]) / (uSides - 1) for i in range(uSides)]
        y_values = [y_range[0] + i * (y_range[1] - y_range[0]) / (vSides - 1) for i in range(vSides)]
        
        # Create the grid and calculate Z values
        vertices = []
        
        for x in x_values:
            for y in y_values:
                z = ((x)**2 + (y)**2) / (4 * focalLength)
                vertices.append(Vertex.ByCoordinates(x, y, z))
        
        mesh = create_triangulated_mesh(vertices=vertices, uSides=uSides, vSides=vSides)
        if not placement.lower() == "bottom":
            x_list = [Vertex.X(v) for v in vertices]
            y_list = [Vertex.Y(v) for v in vertices]
            z_list = [Vertex.Z(v) for v in vertices]
            x_list.sort()
            y_list.sort()
            z_list.sort()
            width = abs(x_list[-1] - x_list[0])
            length = abs(y_list[-1] - y_list[0])
            height = abs(z_list[-1] - z_list[0])
            if placement.lower() == "center":
                mesh = Topology.Translate(mesh, 0, 0, -height*0.5)
            elif placement.lower() == "lowerleft":
                mesh = Topology.Translate(mesh, width*0.5, length*0.5, 0)

        mesh = Topology.Orient(mesh, origin=origin, dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        return mesh
    
    @staticmethod
    def Pie(origin= None, radiusA: float = 0.5, radiusB: float = 0.0, sides: int = 32, rings: int = 1, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a pie shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the pie. The default is None which results in the pie being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the pie. The default is 0.5.
        radiusB : float , optional
            The inner radius of the pie. The default is 0.25.
        sides : int , optional
            The number of sides of the pie. The default is 32.
        rings : int , optional
            The number of rings of the pie. The default is 1.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the pie. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the pie. The default is 360.
        direction : list , optional
            The vector representing the up direction of the pie. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created pie.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle-fromAngle) < tolerance:
            return None
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        angleRange = toAngle - fromAngle
        radiusA = abs(radiusA)
        radiusB = abs(radiusB)
        if radiusB > radiusA:
            temp = radiusA
            radiusA = radiusB
            radiusB = temp
        if abs(radiusA - radiusB) < tolerance or radiusA < tolerance:
            return None
        radiusRange = radiusA - radiusB
        sides = int(abs(math.floor(sides)))
        if sides < 3:
            return None
        rings = int(abs(rings))
        if radiusB < tolerance:
            radiusB = 0
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = radiusA
            yOffset = radiusA
        uOffset = float(angleRange)/float(sides)
        vOffset = float(radiusRange)/float(rings)
        faces = []
        if radiusB > tolerance:
            for i in range(rings):
                r1 = radiusA - vOffset*i
                r2 = radiusA - vOffset*(i+1)
                for j in range(sides):
                    a1 = fromAngle + uOffset*j
                    a2 = fromAngle + uOffset*(j+1)
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
                    v1 = Vertex.ByCoordinates(x1,y1,z1)
                    v2 = Vertex.ByCoordinates(x2,y2,z2)
                    v3 = Vertex.ByCoordinates(x3,y3,z3)
                    v4 = Vertex.ByCoordinates(x4,y4,z4)
                    f1 = Face.ByVertices([v1,v2,v3,v4])
                    faces.append(f1)
        else:
            x1 = 0
            y1 = 0
            z1 = 0
            v1 = Vertex.ByCoordinates(x1,y1,z1)
            for j in range(sides):
                a1 = fromAngle + uOffset*j
                a2 = fromAngle + uOffset*(j+1)
                x2 = math.sin(a1)*radiusA
                y2 = math.cos(a1)*radiusA
                z2 = 0
                x3 = math.sin(a2)*radiusA
                y3 = math.cos(a2)*radiusA
                z3 = 0
                v2 = Vertex.ByCoordinates(x2,y2,z2)
                v3 = Vertex.ByCoordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)

        shell = Shell.ByFaces(faces, tolerance=tolerance)
        if not shell:
            return None
        shell = Topology.Translate(shell, xOffset, yOffset, zOffset)
        shell = Topology.Place(shell, originA=Vertex.Origin(), originB=origin)
        shell = Topology.Orient(shell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return shell

    @staticmethod
    def Planarize(shell, origin= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a planarized version of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        origin : topologic_core.Vertex , optional
            The desired origin of the plane unto which the planar shell will be projected. If set to None, the centroid of the input shell will be chosen. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The planarized shell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            print("Shell.Planarize - Error: The input wire parameter is not a valid topologic shell. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            print("Shell.Planarize - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        vertices = Topology.Vertices(shell)
        faces = Topology.Faces(shell)
        plane_equation = Vertex.PlaneEquation(vertices, mantissa=mantissa)
        rect = Face.RectangleByPlaneEquation(origin=origin , equation=plane_equation, tolerance=tolerance)
        new_vertices = [Vertex.Project(v, rect, mantissa=mantissa) for v in vertices]
        new_shell = Topology.ReplaceVertices(shell, verticesA=vertices, verticesB=new_vertices)
        new_faces = Topology.Faces(new_shell)
        return Topology.SelfMerge(Cluster.ByTopologies(new_faces), tolerance=tolerance)
    
    @staticmethod
    def Rectangle(origin= None, width: float = 1.0, length: float = 1.0,
                  uSides: int = 2, vSides: int = 2, direction: list = [0, 0, 1],
                  placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        uSides : int , optional
            The number of sides along the width. The default is 2.
        vSides : int , optional
            The number of sides along the length. The default is 2.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created shell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        uOffset = float(width)/float(uSides)
        vOffset = float(length)/float(vSides)
        faces = []
        if placement.lower() == "center":
            wOffset = width*0.5
            lOffset = length*0.5
        else:
            wOffset = 0
            lOffset = 0
        for i in range(uSides):
            for j in range(vSides):
                rOrigin = Vertex.ByCoordinates(i*uOffset - wOffset, j*vOffset - lOffset, 0)
                w = Wire.Rectangle(origin=rOrigin, width=uOffset, length=vOffset, direction=[0, 0, 1], placement="lowerleft", tolerance=tolerance)
                f = Face.ByWire(w, tolerance=tolerance)
                faces.append(f)
        shell = Shell.ByFaces(faces, tolerance=tolerance)
        shell = Topology.Place(shell, originA=Vertex.Origin(), originB=origin)
        shell = Topology.Orient(shell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return shell

    @staticmethod
    def RemoveCollinearEdges(shell, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Removes any collinear edges in the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created shell without any collinear edges.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import inspect
        
        if not Topology.IsInstance(shell, "Shell"):
            print("Shell.RemoveCollinearEdges - Error: The input shell parameter is not a valid shell. Returning None.")
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
            return None
        faces = Shell.Faces(shell)
        clean_faces = []
        for face in faces:
            clean_faces.append(Face.RemoveCollinearEdges(face, angTolerance=angTolerance, tolerance=tolerance))
        return Shell.ByFaces(clean_faces, tolerance=tolerance)
    
    @staticmethod
    def Roof(face, angle: float = 45, epsilon: float = 0.01, mantissa: int = 6, tolerance: float = 0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angle : float , optioal
            The desired angle in degrees of the roof. The default is 45.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for distance from plane). The default is 0.01. (This is set to a larger number as it was found to work better)
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Shell
            The created roof.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        def nearest_vertex_2d(v, vertices, tolerance=0.001):
            for vertex in vertices:
                x2 = Vertex.X(vertex, mantissa=mantissa)
                y2 = Vertex.Y(vertex, mantissa=mantissa)
                temp_v = Vertex.ByCoordinates(x2, y2, Vertex.Z(v, mantissa=mantissa))
                if Vertex.Distance(v, temp_v, mantissa=mantissa) <= tolerance:
                    return vertex
            return None
        
        if not Topology.IsInstance(face, "Face"):
            return None
        angle = abs(angle)
        if angle >= 90-tolerance:
            return None
        if angle < tolerance:
            return None
        origin = Topology.Centroid(face)
        normal = Face.Normal(face, mantissa=mantissa)
        flat_face = Topology.Flatten(face, origin=origin, direction=normal)
        roof = Wire.Roof(flat_face, angle=angle, boundary=True, tolerance=tolerance)
        if not roof:
            return None
        shell = Shell.Skeleton(flat_face, tolerance=tolerance)
        faces = Shell.Faces(shell)
        if not faces:
            return None
        triangles = []
        for face in faces:
            internalBoundaries = Face.InternalBoundaries(face)
            if len(internalBoundaries) == 0:
                if len(Topology.Vertices(face)) > 3:
                    triangles += Face.Triangulate(face, tolerance=tolerance)
                else:
                    triangles += [face]

        roof_vertices = Topology.Vertices(roof)
        flat_vertices = []
        for rv in roof_vertices:
            flat_vertices.append(Vertex.ByCoordinates(Vertex.X(rv, mantissa=mantissa), Vertex.Y(rv, mantissa=mantissa), 0))

        final_triangles = []
        for triangle in triangles:
            if len(Topology.Vertices(triangle)) > 3:
                triangles = Face.Triangulate(triangle, tolerance=tolerance)
            else:
                triangles = [triangle]
            final_triangles += triangles

        final_faces = []
        for triangle in final_triangles:
            face_vertices = Topology.Vertices(triangle)
            top_vertices = []
            for sv in face_vertices:
                temp = nearest_vertex_2d(sv, roof_vertices, tolerance=tolerance)
                if temp:
                    top_vertices.append(temp)
                else:
                    top_vertices.append(sv)
            tri_face = Face.ByVertices(top_vertices)
            final_faces.append(tri_face)

        shell = Shell.ByFaces(final_faces, tolerance=tolerance)
        if not shell:
            shell = Cluster.ByTopologies(final_faces)
        try:
            shell = Topology.RemoveCoplanarFaces(shell, epsilon=epsilon, tolerance=tolerance)
        except:
            pass
        shell = Topology.Unflatten(shell, origin=origin, direction=normal)
        return shell
    
    @staticmethod
    def SelfMerge(shell, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Creates a face by merging the faces of the input shell. The shell must be planar within the input angular tolerance.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        if not Topology.IsInstance(shell, "Shell"):
            return None
        ext_boundary = Shell.ExternalBoundary(shell, tolerance=tolerance)
        if Topology.IsInstance(ext_boundary, "Wire"):
            f = Face.ByWire(Topology.RemoveCollinearEdges(ext_boundary, angTolerance), tolerance=tolerance) or Face.ByWire(Wire.Planarize(Topology.RemoveCollinearEdges(ext_boundary, angTolerance), tolerance=tolerance))
            if not f:
                print("FaceByPlanarShell - Error: The input Wire is not planar and could not be fixed. Returning None.")
                return None
            else:
                return f
        elif Topology.IsInstance(ext_boundary, "Cluster"):
            wires = []
            _ = ext_boundary.Wires(None, wires)
            faces = []
            areas = []
            for aWire in wires:
                try:
                    aFace = Face.ByWire(Topology.RemoveCollinearEdges(aWire, angTolerance))
                except:
                    aFace = Face.ByWire(Wire.Planarize(Topology.RemoveCollinearEdges(aWire, angTolerance)))
                anArea = Face.Area(aFace)
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
            try:
                return Face.ByWires(ext_wire, int_wires, tolerance=tolerance)
            except:
                return Face.ByWires(Wire.Planarize(ext_wire), planarizeList(int_wires), tolerance=tolerance)
        else:
            return None

    def Skeleton(face, tolerance: float = 0.001):
        """
            Creates a shell through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Shell
            The created straight skeleton.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import topologic_core as topologic
        import math

        if not Topology.IsInstance(face, "Face"):
            return None
        roof = Wire.Skeleton(face, tolerance=tolerance)
        if not (Topology.IsInstance(roof, "Wire") or Topology.IsInstance(roof, "Cluster")):
            print("Shell.Skeleton - Error: Could not create base skeleton wire. Returning None.")
            return None
        br = Wire.BoundingRectangle(roof) #This works even if it is a Cluster not a Wire
        if not Topology.IsInstance(br, "Wire"):
            print("Shell.Skeleton - Error: Could not create a bounding rectangle wire. Returning None.")
            return None
        br = Topology.Scale(br, Topology.Centroid(br), 1.5, 1.5, 1)
        bf = Face.ByWire(br, tolerance=tolerance)
        if not Topology.IsInstance(bf, "Face"):
            print("Shell.Skeleton - Error: Could not create a bounding rectangle face. Returning None.")
            return None
        large_shell = Topology.Boolean(bf, roof, operation="slice", tolerance=tolerance)
        if not large_shell:
            return None
        faces = Topology.Faces(large_shell)
        if not faces:
            return None
        final_faces = []
        for f in faces:
            internalBoundaries = Face.InternalBoundaries(f)
            if len(internalBoundaries) == 0:
                final_faces.append(f)
        shell = Shell.ByFaces(final_faces, tolerance=tolerance)
        if not Topology.IsInstance(shell, "Shell"):
            print("Shell.Skeleton - Error: Could not create shell. Returning None.")
            return None
        return shell

    @staticmethod
    def Simplify(shell, simplifyBoundary: bool = True, mantissa: int = 6, tolerance: float = 0.0001):
        """
            Simplifies the input shell edges based on the Douglas Peucker algorithm. See https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
            Part of this code was contributed by gaoxipeng. See https://github.com/wassimj/topologicpy/issues/35

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        simplifyBoundary : bool , optional
            If set to True, the external boundary of the shell will be simplified as well. Otherwise, it will not be simplified. The default is True.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001. Edges shorter than this length will be removed.

        Returns
        -------
        topologic_core.Shell
            The simplified shell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        
        def perpendicular_distance(point, line_start, line_end):
            # Calculate the perpendicular distance from a point to a line segment
            x0 = Vertex.X(point, mantissa=mantissa)
            y0 = Vertex.Y(point, mantissa=mantissa)
            x1 = Vertex.X(line_start, mantissa=mantissa)
            y1 = Vertex.Y(line_start, mantissa=mantissa)
            x2 = Vertex.X(line_end, mantissa=mantissa)
            y2 = Vertex.Y(line_end, mantissa=mantissa)

            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = Vertex.Distance(line_start, line_end)

            return numerator / denominator

        def douglas_peucker(wire, tolerance=0.0001):
            if isinstance(wire, list):
                points = wire
            else:
                points = Topology.Vertices(wire)
                # points.insert(0, points.pop())
            if len(points) <= 2:
                return points

            # Use the first and last points in the list as the starting and ending points
            start_point = points[0]
            end_point = points[-1]

            # Find the point with the maximum distance
            max_distance = 0
            max_index = 0

            for i in range(1, len(points) - 1):
                d = perpendicular_distance(points[i], start_point, end_point)
                if d > max_distance:
                    max_distance = d
                    max_index = i

            # If the maximum distance is less than the tolerance, no further simplification is needed
            if max_distance <= tolerance:
                return [start_point, end_point]

            # Recursively simplify
            first_segment = douglas_peucker(points[:max_index + 1], tolerance=tolerance)
            second_segment = douglas_peucker(points[max_index:], tolerance=tolerance)

            # Merge the two simplified segments
            return first_segment[:-1] + second_segment
        if not Topology.IsInstance(shell, "Shell"):
            print("Shell.Simplify - Error: The input shell parameter is not a valid topologic shell. Returning None.")
            return None
        # Get the external boundary of the shell. This can be simplified as well, but might cause issues at the end.
        # At this point, it is assumed to be left as is.
        all_edges = Topology.Edges(shell)
        if simplifyBoundary == False:
            ext_boundary = Face.ByWire(Shell.ExternalBoundary(shell, tolerance=tolerance), tolerance=tolerance)
            
            # Get the internal edges of the shell.
            i_edges = []
            for edge in all_edges:
                faces = Topology.SuperTopologies(edge, shell, topologyType="face")
                if len(faces) > 1: # This means that the edge separates two faces so it is internal.
                    i_edges.append(edge)
            # Creat a Wire from the internal edges
            wire = Topology.SelfMerge(Cluster.ByTopologies(i_edges), tolerance=tolerance)
        else:
            wire = Topology.SelfMerge(Cluster.ByTopologies(all_edges), tolerance=tolerance)
        # Split the wires at its junctions (where more than two edges meet at a vertex)
        components = Wire.Split(wire)
        separators = []
        wires = []
        for component in components:
            if Topology.IsInstance(component, "Cluster"):
                component = Topology.SelfMerge(component, tolerance=tolerance)
                if Topology.IsInstance(component, "Cluster"):
                    separators.append(Cluster.FreeEdges(component, tolerance=tolerance))
                    wires.append(Cluster.FreeWires(component, tolerance=tolerance))
                if Topology.IsInstance(component, "Edge"):
                    separators.append(component)
                if Topology.IsInstance(component, "Wire"):
                    wires.append(component)
            if Topology.IsInstance(component, "Edge"):
                separators.append(component)
            if Topology.IsInstance(component, "Wire"):
                wires.append(component)
        wires = Helper.Flatten(wires)
        separators = Helper.Flatten(separators)
        results = []
        for w in wires:
            temp_wire = Wire.ByVertices(douglas_peucker(w, tolerance=tolerance), close=False)
            results.append(temp_wire)
        # Make a Cluster out of the results
        cluster = Cluster.ByTopologies(results)
        # Get all the edges of the result
        edges = Topology.Edges(cluster)
        # Add them to the final edges
        final_edges = edges + separators
        # Make a Cluster out of the final set of edges
        cluster = Cluster.ByTopologies(final_edges)
        if simplifyBoundary == False:
            # Slice the external boundary of the shell by the cluster
            final_result = Topology.Slice(ext_boundary, cluster, tolerance=tolerance)
        else:
            br = Wire.BoundingRectangle(shell)
            br = Topology.Scale(br, Topology.Centroid(br), 1.5, 1.5, 1.5)
            br = Face.ByWire(br, tolerance=tolerance)
            v = Face.VertexByParameters(br, 0.1, 0.1)
            result = Topology.Slice(br, cluster, tolerance=tolerance)
            faces = Topology.Faces(result)
            final_faces = []
            for face in faces:
                if not Vertex.IsInternal(v, face, tolerance=0.01):
                    final_faces.append(face)
            final_result = Shell.ByFaces(final_faces, tolerance=tolerance)
        return final_result

    @staticmethod
    def Vertices(shell) -> list:
        """
        Returns the vertices of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        vertices = []
        _ = shell.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Voronoi(vertices: list, face= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a voronoi partitioning of the input face based on the input vertices. The vertices must be coplanar and within the face. See https://en.wikipedia.org/wiki/Voronoi_diagram.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic_core.Face , optional
            The input face. If the face is not set an optimised bounding rectangle of the input vertices is used instead. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        shell
            A shell representing the voronoi partitioning of the input face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        
        if not Topology.IsInstance(face, "Face"):
            cluster = Cluster.ByTopologies(vertices)
            br = Wire.BoundingRectangle(cluster, optimize=5)
            face = Face.ByWire(br, tolerance=tolerance)
        if not isinstance(vertices, list):
            return None
        vertices = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertices) < 2:
            return None

        # Flatten the input face
        origin = Topology.Centroid(face)
        normal = Face.Normal(face, mantissa=mantissa)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)
        eb = Face.ExternalBoundary(flatFace)
        ibList = Face.InternalBoundaries(flatFace)
        temp_verts = Topology.Vertices(eb)
        new_verts = [Vertex.ByCoordinates(Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa), 0) for v in temp_verts]
        eb = Wire.ByVertices(new_verts, close=True)
        new_ibList = []
        for ib in ibList:
            temp_verts = Topology.Vertices(ib)
            new_verts = [Vertex.ByCoordinates(Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa), 0) for v in temp_verts]
            new_ibList.append(Wire.ByVertices(new_verts, close=True))
        flatFace = Face.ByWires(eb, new_ibList)

        # Create a cluster of the input vertices
        verticesCluster = Cluster.ByTopologies(vertices)

        # Flatten the cluster using the same transformations
        verticesCluster = Topology.Flatten(verticesCluster, origin=origin, direction=normal)
        flatVertices = Topology.Vertices(verticesCluster)
        flatVertices = [Vertex.ByCoordinates(Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa), 0) for v in flatVertices]
        points = []
        for flatVertex in flatVertices:
            points.append([Vertex.X(flatVertex, mantissa=mantissa), Vertex.Y(flatVertex, mantissa=mantissa)])

        br = Wire.BoundingRectangle(flatFace)
        br_vertices = Topology.Vertices(br)
        br_x = []
        br_y = []
        for br_v in br_vertices:
            x, y = Vertex.Coordinates(br_v, outputType="xy")
            br_x.append(x)
            br_y.append(y)
        min_x = min(br_x)
        max_x = max(br_x)
        min_y = min(br_y)
        max_y = max(br_y)
        br_width = abs(max_x - min_x)
        br_length = abs(max_y - min_y)

        points.append((-br_width*4, -br_length*4))
        points.append((-br_width*4, br_length*4))
        points.append((br_width*4, -br_length*4))
        points.append((br_width*4, br_length*4))

        voronoi = Voronoi(points, furthest_site=False)
        voronoiVertices = []
        for v in voronoi.vertices:
            voronoiVertices.append(Vertex.ByCoordinates(v[0], v[1], 0))

        faces = []
        for region in voronoi.regions:
            tempWire = []
            if len(region) > 1 and not -1 in region:
                for v in region:
                    tempWire.append(Vertex.ByCoordinates(Vertex.X(voronoiVertices[v], mantissa=mantissa), Vertex.Y(voronoiVertices[v], mantissa=mantissa), 0))
                temp_verts = []
                for v in tempWire:
                    if len(temp_verts) == 0:
                        temp_verts.append(v)
                    elif Vertex.Index(v, temp_verts, tolerance=tolerance) == None:
                        temp_verts.append(v)
                tempWire = temp_verts
                temp_w = Wire.ByVertices(tempWire, close=True)
                faces.append(Face.ByWire(Wire.ByVertices(tempWire, close=True), tolerance=tolerance))
        shell = Shell.ByFaces(faces, tolerance=tolerance)
        edges = Shell.Edges(shell)
        edgesCluster = Cluster.ByTopologies(edges)
        shell = Topology.Slice(flatFace,edgesCluster, tolerance=tolerance)
        shell = Topology.Unflatten(shell, origin=origin, direction=normal)
        return shell

    @staticmethod
    def Wires(shell) -> list:
        """
        Returns the wires of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of wires.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        wires = []
        _ = shell.Wires(None, wires)
        return wires

    
    
    
    
    