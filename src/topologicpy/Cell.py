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
    
class Cell():
    @staticmethod
    def Area(cell, mantissa: int = 6):
        """
        Returns the surface area of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The cell.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The surface area of the input cell.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        faces = Topology.Faces(cell)
        area = 0.0
        for aFace in faces:
            area = area + Face.Area(aFace)
        return round(area, mantissa)

    @staticmethod
    def Box(origin = None,
            width: float = 1, length: float = 1, height: float = 1,
            uSides: int = 1, vSides: int = 1, wSides: int = 1,
            direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates a box.

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
        vSides : int , optional
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
        topologic_core.Cell
            The created box.

        """
        return Cell.Prism(origin=origin, width=width, length=length, height=height,
                          uSides=uSides, vSides=vSides, wSides=wSides,
                          direction=direction, placement=placement, tolerance=tolerance)

    @staticmethod
    def ByFaces(faces: list, planarize: bool = False, tolerance: float = 0.0001, silent=False):
        """
        Creates a cell from the input list of faces.

        Parameters
        ----------
        faces : list
            The input list of faces.
        planarize : bool, optional
            If set to True, the input faces are planarized before building the cell. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(faces, list):
            if not silent:
                print("Cell.ByFaces - Error: The input faces parameter is not a valid list. Returning None.")
                return None
        faceList = [x for x in faces if Topology.IsInstance(x, "Face")]
        if len(faceList) < 1:
            if not silent:
                print("Cell.ByFaces - Error: The input faces parameter does not contain valid faces. Returning None.")
                return None
        # Try the default method
        cell = topologic.Cell.ByFaces(faceList, tolerance) # Hook to Core
        if Topology.IsInstance(cell, "Cell"):
            return cell
        
        # Fuse all the vertices first and rebuild the faces
        all_vertices = []
        wires = []
        for f in faceList:
            w = Face.Wire(f)
            wires.append(w)
            all_vertices += Topology.Vertices(w)
        all_vertices = Vertex.Fuse(all_vertices, tolerance=tolerance)
        new_faces = []
        for w in wires:
            face_vertices = []
            for v in Topology.Vertices(w):
                index = Vertex.Index(v, all_vertices, tolerance=tolerance)
                if not index == None:
                    face_vertices.append(all_vertices[index])
            new_w = Wire.ByVertices(face_vertices)
            if Topology.IsInstance(new_w, "Wire"):
                new_f = Face.ByWire(new_w, silent=True)
                if Topology.IsInstance(new_f, "Face"):
                    new_faces.append(new_f)
                elif isinstance(new_f, list):
                    new_faces += new_f
        faceList = new_faces
        planarizedList = []
        enlargedList = []
        if planarize:
            planarizedList = [Face.Planarize(f, tolerance=tolerance) for f in faceList]
            enlargedList = [Face.ByOffset(f, offset=-tolerance*10) for f in planarizedList]
            cell = topologic.Cell.ByFaces(enlargedList, tolerance) # Hook to Core
            faceList = Topology.SubTopologies(cell, subTopologyType="face")
            finalFaces = []
            for f in faceList:
                centroid = Topology.Centroid(f)
                n = Face.Normal(f)
                v = Topology.Translate(centroid, n[0]*0.01, n[1]*0.01, n[2]*0.01)
                if not Vertex.IsInternal(v, cell):
                    finalFaces.append(f)
            finalFinalFaces = []
            for f in finalFaces:
                vertices = Topology.Vertices(f)
                w = Wire.Cycles(Face.ExternalBoundary(f), maxVertices=len(vertices))[0]
                f1 = Face.ByWire(w, tolerance=tolerance, silent=True)
                if Topology.IsInstance(f1, "Face"):
                    finalFinalFaces.append(f1)
                elif isinstance(f1, list):
                    finalFinalFaces += f1
            cell = topologic.Cell.ByFaces(finalFinalFaces, tolerance) # Hook to Core
            if cell == None:
                if not silent:
                    print("Cell.ByFaces - Error: The operation failed. Returning None.")
                    return None
            else:
                return cell
        else:
            cell = topologic.Cell.ByFaces(faces, tolerance) # Hook to Core
            if cell == None:
                if not silent:
                    print("Cell.ByFaces - Error: The operation failed. Returning None.")
                    return None
            else:
                return cell
    @staticmethod
    def ByOffset(cell, offset: float = 1.0, tolerance: float = 0.0001):
        """
        Creates an offset cell from the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        offset : float , optional
            The desired offset distance. The default is 1.0.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        Topology
            The created offset topology. WARNING: This method may fail to create a cell if the offset creates self-intersecting faces. Always check the type being returned by this method.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        vertices = Topology.Vertices(cell)
        new_vertices = []
        for v in vertices:
            faces = Topology.SuperTopologies(v, hostTopology=cell, topologyType="face")
            normals = []
            for face in faces:
                normal = Vector.SetMagnitude(Face.Normal(face), offset)
                normals.append(normal)
            sum_normal = Vector.Sum(normals)
            new_v = Topology.TranslateByDirectionDistance(v, direction=sum_normal, distance=Vector.Magnitude(sum_normal))
            new_vertices.append(new_v)
        new_cell = Topology.SelfMerge(Topology.ReplaceVertices(cell, Topology.Vertices(cell), new_vertices), tolerance=tolerance)
        return new_cell
    
    @staticmethod
    def ByShell(shell, planarize: bool = False, tolerance: float = 0.0001):
        """
        Creates a cell from the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell. The shell must be closed for this method to succeed.
        planarize : bool, optional
            If set to True, the input faces of the input shell are planarized before building the cell. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            print("Cell.ByShell - Error: The input shell parameter is not a valid topologic shell. Returning None.")
            return None
        faces = Topology.SubTopologies(shell, subTopologyType="face")
        return Cell.ByFaces(faces, planarize=planarize, tolerance=tolerance)
    
    
    @staticmethod
    def ByShells(externalBoundary, internalBoundaries: list = [], tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cell from the input external boundary (closed shell) and the input list of internal boundaries (closed shells).

        Parameters
        ----------
        externalBoundary : topologic_core.Shell
            The input external boundary.
        internalBoundaries : list , optional
            The input list of internal boundaries (closed shells). The default is an empty list.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(externalBoundary, "Shell"):
            if not silent:
                print("Cell.ByShells - Error: The input externalBoundary parameter is not a valid topologic shell. Returning None.")
            return None
        if not Shell.IsClosed(externalBoundary):
            if not silent:
                print("Cell.ByShells - Error: The input externalBoundary parameter is not a closed topologic shell. Returning None.")
            return None
        ibList = [Cell.ByShell(s) for s in internalBoundaries if Topology.IsInstance(s, "Shell") and Shell.IsClosed(s)]
        cell = Cell.ByShell(externalBoundary)
        if len(ibList) > 0:
            inner_cluster =Cluster.ByTopologies(ibList)
            cell = Topology.Difference(cell, inner_cluster)
            if not Topology.IsInstance(cell, "Cell"):
                if not silent:
                    print("Cell.ByShells - Error: Could not create cell. Returning None.")
                return None
        return cell

    @staticmethod
    def ByThickenedFace(face, thickness: float = 1.0, bothSides: bool = True, reverse: bool = False,
                            planarize: bool = False, tolerance: float = 0.0001):
        """
        Creates a cell by thickening the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face to be thickened.
        thickness : float , optional
            The desired thickness. The default is 1.0.
        bothSides : bool
            If True, the cell will be lofted to each side of the face. Otherwise, it will be lofted in the direction of the normal to the input face. The default is True.
        reverse : bool
            If True, the cell will be lofted in the opposite direction of the normal to the face. The default is False.
        planarize : bool, optional
            If set to True, the input faces of the input shell are planarized before building the cell. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Cell.ByThickenedFace - Error: The input face parameter is not a valid topologic face. Returning None.")
            return None
        if reverse == True and bothSides == False:
            thickness = -thickness
        faceNormal = Face.Normal(face)
        if bothSides:
            bottomFace = Topology.Translate(face, -faceNormal[0]*0.5*thickness, -faceNormal[1]*0.5*thickness, -faceNormal[2]*0.5*thickness)
            topFace = Topology.Translate(face, faceNormal[0]*0.5*thickness, faceNormal[1]*0.5*thickness, faceNormal[2]*0.5*thickness)
        else:
            bottomFace = face
            topFace = Topology.Translate(face, faceNormal[0]*thickness, faceNormal[1]*thickness, faceNormal[2]*thickness)

        cellFaces = [Face.Invert(bottomFace), topFace]
        bottomEdges = Topology.Edges(bottomFace)
        for bottomEdge in bottomEdges:
            topEdge = Topology.Translate(bottomEdge, faceNormal[0]*thickness, faceNormal[1]*thickness, faceNormal[2]*thickness)
            sideEdge1 = Edge.ByVertices([bottomEdge.StartVertex(), topEdge.StartVertex()], tolerance=tolerance, silent=True)
            sideEdge2 = Edge.ByVertices([bottomEdge.EndVertex(), topEdge.EndVertex()], tolerance=tolerance, silent=True)
            cellWire = Topology.SelfMerge(Cluster.ByTopologies([bottomEdge, sideEdge1, topEdge, sideEdge2]), tolerance=tolerance)
            cellFaces.append(Face.ByWire(cellWire, tolerance=tolerance))
        return Cell.ByFaces(cellFaces, planarize=planarize, tolerance=tolerance)

    @staticmethod
    def ByThickenedShell(shell, direction: list = [0, 0, 1], thickness: float = 1.0, bothSides: bool = True, reverse: bool = False,
                            planarize: bool = False, tolerance: float = 0.0001):
        """
        Creates a cell by thickening the input shell. The shell must be open.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell to be thickened.
        thickness : float , optional
            The desired thickness. The default is 1.0.
        bothSides : bool
            If True, the cell will be lofted to each side of the shell. Otherwise, it will be lofted along the input direction. The default is True.
        reverse : bool
            If True, the cell will be lofted along the opposite of the input direction. The default is False.
        planarize : bool, optional
            If set to True, the input faces of the input shell are planarized before building the cell. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(shell, "Shell"):
            print("Cell.ByThickenedShell - Error: The input shell parameter is not a valid topologic Shell. Returning None.")
            return None
        if reverse == True and bothSides == False:
            thickness = -thickness
        if bothSides:
            bottomShell = Topology.Translate(shell, -direction[0]*0.5*thickness, -direction[1]*0.5*thickness, -direction[2]*0.5*thickness)
            topShell = Topology.Translate(shell, direction[0]*0.5*thickness, direction[1]*0.5*thickness, direction[2]*0.5*thickness)
        else:
            bottomShell = shell
            topShell = Topology.Translate(shell, direction[0]*thickness, direction[1]*thickness, direction[2]*thickness)
        cellFaces = Shell.Faces(bottomShell) + Shell.Faces(topShell)
        bottomWire = Shell.ExternalBoundary(bottomShell, tolerance=tolerance)
        bottomEdges = Wire.Edges(bottomWire)
        for bottomEdge in bottomEdges:
            topEdge = Topology.Translate(bottomEdge, direction[0]*thickness, direction[1]*thickness, direction[2]*thickness)
            sideEdge1 = Edge.ByVertices([Edge.StartVertex(bottomEdge), Edge.StartVertex(topEdge)], tolerance=tolerance, silent=True)
            sideEdge2 = Edge.ByVertices([Edge.EndVertex(bottomEdge), Edge.EndVertex(topEdge)], tolerance=tolerance, silent=True)
            cellWire = Topology.SelfMerge(Cluster.ByTopologies([bottomEdge, sideEdge1, topEdge, sideEdge2]), tolerance=tolerance)
            cellFace = Face.ByWire(cellWire, tolerance=tolerance)
            cellFaces.append(cellFace)
        return Cell.ByFaces(cellFaces, planarize=planarize, tolerance=tolerance)
    
    @staticmethod
    def ByWires(wires: list, close: bool = False, triangulate: bool = True, planarize: bool = False, mantissa: int = 6, tolerance: float = 0.0001, silent=False):
        """
        Creates a cell by lofting through the input list of wires.

        Parameters
        ----------
        wires : list
            The input list of wires.
        close : bool , optional
            If set to True, the last wire in the list of input wires will be connected to the first wire in the list of input wires. The default is False.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        planarize : bool, optional
            If set to True, the created faces are planarized before building the cell. Otherwise, they are not. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Raises
        ------
        Exception
            Raises an exception if the two wires in the list do not have the same number of edges.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology

        if not isinstance(wires, list):
            if not silent:
                print("Cell.ByWires - Error: The input wires parameter is not a valid list. Returning None.")
                return None
        wires = [w for w in wires if Topology.IsInstance(w, "Wire")]
        if len(wires) < 2:
            if not silent:
                print("Cell.ByWires - Error: The input wires parameter contains less than two valid topologic wires. Returning None.")
                return None
        faces = [Face.ByWire(wires[0], tolerance=tolerance), Face.ByWire(wires[-1], tolerance=tolerance)]
        if close == True:
            faces.append(Face.ByWire(wires[0], tolerance=tolerance))
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
            w1_edges = Topology.Edges(wire1)
            w2_edges = Topology.Edges(wire2)
            if len(w1_edges) != len(w2_edges):
                if not silent:
                    print("Cell.ByWires - Error: The input wires parameter contains wires with different number of edges. Returning None.")
                    return None
            if triangulate == True:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=True)
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e4], tolerance=tolerance), tolerance=tolerance))
                        except:
                            pass
                    try:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=True)
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3], tolerance=tolerance), tolerance=tolerance))
                        except:
                            pass
                    if e3 and e4:
                        e5 = Edge.ByVertices([e1.StartVertex(), e2.EndVertex()], tolerance=tolerance, silent=True)
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e5, e4], tolerance=tolerance), tolerance=tolerance))
                        faces.append(Face.ByWire(Wire.ByEdges([e2, e5, e3], tolerance=tolerance), tolerance=tolerance))
            else:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=True)
                        except:
                            pass
                    try:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()], tolerance=tolerance, silent=True)
                        except:
                            pass
                    if e3 and e4:
                        try:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e4, e2, e3], tolerance=tolerance), tolerance=tolerance))
                        except:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e3, e2, e4], tolerance=tolerance), tolerance=tolerance))
                    elif e3:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e3, e2], tolerance=tolerance), tolerance=tolerance))
                    elif e4:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e4, e2], tolerance=tolerance), tolerance=tolerance))
        cell = Cell.ByFaces(faces, planarize=planarize, tolerance=tolerance, silent=silent)
        if not cell:
            shell = Shell.ByFaces(faces, tolerance=tolerance)
            if Topology.IsInstance(shell, "Shell"):
                geom = Topology.Geometry(shell, mantissa=mantissa)
                cell = Topology.ByGeometry(geom['vertices'], geom['edges'], geom['faces'])
            if not Topology.IsInstance(cell, "Cell"):
                print("Cell.ByWires - Error: Could not create a cell. Returning None.")
                return None
        return cell

    @staticmethod
    def ByWiresCluster(cluster, close: bool = False, triangulate: bool = True, planarize: bool = False, tolerance: float = 0.0001):
        """
        Creates a cell by lofting through the input cluster of wires.

        Parameters
        ----------
        cluster : Cluster
            The input Cluster of wires.
        close : bool , optional
            If set to True, the last wire in the cluster of input wires will be connected to the first wire in the cluster of input wires. The default is False.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Raises
        ------
        Exception
            Raises an exception if the two wires in the list do not have the same number of edges.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cell.ByWiresCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        wires = Topology.Wires(cluster)
        return Cell.ByWires(wires, close=close, triangulate=triangulate, planarize=planarize, tolerance=tolerance)

    @staticmethod
    def Capsule(origin = None, radius: float = 0.25, height: float = 1, uSides: int = 16, vSidesEnds:int = 8, vSidesMiddle: int = 1, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a capsule shape. A capsule is a cylinder with hemispherical ends.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the cylinder. The default is None which results in the cylinder being placed at (0, 0, 0).
        radius : float , optional
            The radius of the capsule. The default is 0.25.
        height : float , optional
            The height of the capsule. The default is 1.
        uSides : int , optional
            The number of circle segments of the capsule. The default is 16.
        vSidesEnds : int , optional
            The number of vertical segments of the end hemispheres. The default is 8.
        vSidesMiddle : int , optional
            The number of vertical segments of the middle cylinder. The default is 1.
        direction : list , optional
            The vector representing the up direction of the capsule. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the capsule. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "bottom".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created cell.

            """
        from topologicpy.Topology import Topology
        from topologicpy.Cell import Cell
        from topologicpy.Vertex import Vertex
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Capsule - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        cyl_height = height - radius*2
        if cyl_height <= 0:
            capsule = Cell.Sphere(origin=Vertex.Origin(), radius=radius, uSides= uSides, vSides=vSidesEnds*2)
        else:
            cyl = Cell.Cylinder(origin=Vertex.Origin(),
                                radius=radius,
                                height=cyl_height,
                                uSides=uSides, vSides=vSidesMiddle, direction=[0, 0, 1], placement="center", tolerance=tolerance)
            o1 = Vertex.ByCoordinates(0, 0, cyl_height*0.5)
            o2 = Vertex.ByCoordinates(0, 0, -cyl_height*0.5)
            s1 = Cell.Sphere(origin=o1, radius=radius, uSides=uSides, vSides=vSidesEnds*2, tolerance=tolerance)
            s2 = Cell.Sphere(origin=o2, radius=radius, uSides=uSides, vSides=vSidesEnds*2, tolerance=tolerance)
            capsule = Topology.Union(cyl, s1, tolerance=tolerance)
            capsule = Topology.Union(capsule, s2, tolerance=tolerance)
        if placement == "bottom":
            capsule = Topology.Translate(capsule, 0, 0, height/2)
        if placement == "lowerleft":
            capsule = Topology.Translate(capsule, 0, 0, height/2)
            capsule = Topology.Translate(capsule, radius, radius)
        
        capsule = Topology.Orient(capsule, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        capsule = Topology.Place(capsule, originA=Vertex.Origin(), originB=origin)
        return capsule

    @staticmethod
    def Compactness(cell, reference = "sphere", mantissa: int = 6) -> float:
        """
        Returns the compactness measure of the input cell. If the reference is "sphere", this is also known as 'sphericity' (https://en.wikipedia.org/wiki/Sphericity).

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        reference : str , optional
            The desired reference to which to compare this compactness. The options are "sphere" and "cube". It is case insensitive. The default is "sphere".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The compactness of the input cell.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        faces = Topology.Faces(cell)
        area = 0.0
        for aFace in faces:
            area = area + abs(Face.Area(aFace))
        volume = abs(Cell.Volume(cell, mantissa=mantissa))
        compactness  = 0
        #From https://en.wikipedia.org/wiki/Sphericity
        if area > 0:
            if reference.lower() == "sphere":
                compactness = (((math.pi)**(1/3))*((6*volume)**(2/3)))/area
            else:
                compactness = 6*(volume**(2/3))/area
        else:
            print("Cell.Compactness - Error: cell surface area is not positive. Returning None.")
            return None
        return round(compactness, mantissa)
    
    @staticmethod
    def Cone(origin = None, baseRadius: float = 0.5, topRadius: float = 0, height: float = 1, uSides: int = 16, vSides: int = 1, direction: list = [0, 0, 1],
                 dirZ: float = 1, placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a cone.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the cone. The default is None which results in the cone being placed at (0, 0, 0).
        baseRadius : float , optional
            The radius of the base circle of the cone. The default is 0.5.
        topRadius : float , optional
            The radius of the top circle of the cone. The default is 0.
        height : float , optional
            The height of the cone. The default is 1.
        uSides : int , optional
            The number of circle segments of the cylinder. The default is 16.
        vSides : int , optional
            The number of vertical segments of the cylinder. The default is 1.
        direction : list , optional
            The vector representing the up direction of the cone. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the cone. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created cone.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        import math

        def createCone(baseWire, topWire, baseVertex, topVertex, tolerance=0.0001):
            if baseWire == None and topWire == None:
                raise Exception("Cell.Cone - Error: Both radii of the cone cannot be zero at the same time")
            elif baseWire == None:
                apex = baseVertex
                wire = topWire
            elif topWire == None:
                apex = topVertex
                wire = baseWire
            else:
                return Cell.ByWires([baseWire, topWire])
            vertices = Topology.Vertices(wire)
            faces = [Face.ByWire(wire, tolerance=tolerance)]
            for i in range(0, len(vertices)-1):
                w = Wire.ByVertices([apex, vertices[i], vertices[i+1]])
                f = Face.ByWire(w, tolerance=tolerance)
                faces.append(f)
            w = Wire.ByVertices([apex, vertices[-1], vertices[0]])
            f = Face.ByWire(w, tolerance=tolerance)
            faces.append(f)
            return Cell.ByFaces(faces, tolerance=tolerance)
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Cone - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "center":
            xOffset = 0
            yOffset = 0
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = max(baseRadius, topRadius)
            yOffset = max(baseRadius, topRadius)
            zOffset = 0

        baseZ = origin.Z() + zOffset
        topZ = origin.Z() + zOffset + height
        baseV = []
        topV = []
        for i in range(uSides):
            angle = math.radians(360/uSides)*i
            if baseRadius > 0:
                baseX = math.cos(angle)*baseRadius + Vertex.X(origin, mantissa=mantissa) + xOffset
                baseY = math.sin(angle)*baseRadius + Vertex.Y(origin, mantissa=mantissa) + yOffset
                baseZ = Vertex.Z(origin, mantissa=mantissa) + zOffset
                baseV.append(Vertex.ByCoordinates(baseX,baseY,baseZ))
            if topRadius > 0:
                topX = math.cos(angle)*topRadius + Vertex.X(origin, mantissa=mantissa) + xOffset
                topY = math.sin(angle)*topRadius + Vertex.Y(origin, mantissa=mantissa) + yOffset
                topV.append(Vertex.ByCoordinates(topX,topY,topZ))
        if baseRadius > 0:
            baseWire = Wire.ByVertices(baseV)
        else:
            baseWire = None
        if topRadius > 0:
            topWire = Wire.ByVertices(topV)
        else:
            topWire = None
        baseVertex = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+xOffset, Vertex.Y(origin, mantissa=mantissa)+yOffset, Vertex.Z(origin, mantissa=mantissa)+zOffset)
        topVertex = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+xOffset, Vertex.Y(origin, mantissa=mantissa)+yOffset, Vertex.Z(origin, mantissa=mantissa)+zOffset+height)
        cone = createCone(baseWire, topWire, baseVertex, topVertex, tolerance=tolerance)
        if cone == None:
            print("Cell.Cone - Error: Could not create a cone. Returning None.")
            return None
        
        if vSides > 1:
            cutting_planes = []
            baseX = Vertex.X(origin, mantissa=mantissa) + xOffset
            baseY = Vertex.Y(origin, mantissa=mantissa) + yOffset
            size = max(baseRadius, topRadius)*3
            for i in range(1, vSides):
                baseZ = Vertex.Z(origin, mantissa=mantissa) + zOffset + float(height)/float(vSides)*i
                tool_origin = Vertex.ByCoordinates(baseX, baseY, baseZ)
                cutting_planes.append(Face.ByWire(Wire.Rectangle(origin=tool_origin, width=size, length=size), tolerance=tolerance))
            cutting_planes_cluster = Cluster.ByTopologies(cutting_planes)
            shell = Cell.Shells(cone)[0]
            shell = shell.Slice(cutting_planes_cluster)
            cone = Cell.ByShell(shell)
        cone = Topology.Orient(cone, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return cone
  
    @staticmethod
    def ContainmentStatus(cell, vertex, tolerance: float = 0.0001) -> int:
        """
        Returns the containment status of the input vertex in relationship to the input cell

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        vertex : topologic_core.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        int
            Returns 0 if the vertex is inside, 1 if it is on the boundary of, and 2 if it is outside the input cell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.ContainmentStatus - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Cell.ContainmentStatus - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
            return None
        try:
            status = topologic.CellUtility.Contains(cell, vertex, tolerance) # Hook to Core
            if status == 0:
                return 0
            elif status == 1:
                return 1
            else:
                return 2
        except:
            print("Cell.ContainmentStatus - Error: Could not determine containment status. Returning None.")
            return None
 
    @staticmethod
    def Cylinder(origin = None, radius: float = 0.5, height: float = 1, uSides: int = 16, vSides: int = 1, direction: list = [0, 0, 1],
                     placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a cylinder.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the cylinder. The default is None which results in the cylinder being placed at (0, 0, 0).
        radius : float , optional
            The radius of the cylinder. The default is 0.5.
        height : float , optional
            The height of the cylinder. The default is 1.
        uSides : int , optional
            The number of circle segments of the cylinder. The default is 16.
        vSides : int , optional
            The number of vertical segments of the cylinder. The default is 1.
        direction : list , optional
            The vector representing the up direction of the cylinder. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the cylinder. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "bottom".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Cylinder - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "center":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = radius
            yOffset = radius
        circle_origin = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa) + xOffset, Vertex.Y(origin, mantissa=mantissa) + yOffset, Vertex.Z(origin, mantissa=mantissa) + zOffset)
        
        baseWire = Wire.Circle(origin=circle_origin, radius=radius, sides=uSides, fromAngle=0, toAngle=360, close=True, direction=[0, 0, 1], placement="center", tolerance=tolerance)
        baseFace = Face.ByWire(baseWire, tolerance=tolerance)
        cylinder = Cell.ByThickenedFace(face=baseFace, thickness=height, bothSides=False, tolerance=tolerance)
        if vSides > 1:
            cutting_planes = []
            baseX = Vertex.X(origin, mantissa=mantissa) + xOffset
            baseY = Vertex.Y(origin, mantissa=mantissa) + yOffset
            size = radius*3
            for i in range(1, vSides):
                baseZ = origin.Z() + zOffset + float(height)/float(vSides)*i
                tool_origin = Vertex.ByCoordinates(baseX, baseY, baseZ)
                cutting_planes.append(Face.ByWire(Wire.Rectangle(origin=tool_origin, width=size, length=size), tolerance=tolerance))
            cutting_planes_cluster = Cluster.ByTopologies(cutting_planes)
            cylinder = CellComplex.ExternalBoundary(cylinder.Slice(cutting_planes_cluster))

        cylinder = Topology.Orient(cylinder, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return cylinder
    
    @staticmethod
    def Decompose(cell, tiltAngle: float = 10, tolerance: float = 0.0001) -> dict:
        """
        Decomposes the input cell into its logical components. This method assumes that the positive Z direction is UP.

        Parameters
        ----------
        cell : topologic_core.Cell
            the input cell.
        tiltAngle : float , optional
            The threshold tilt angle in degrees to determine if a face is vertical, horizontal, or tilted. The tilt angle is measured from the nearest cardinal direction. The default is 10.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dictionary
            A dictionary with the following keys and values:
            1. "verticalFaces": list of vertical faces
            2. "topHorizontalFaces": list of top horizontal faces
            3. "bottomHorizontalFaces": list of bottom horizontal faces
            4. "inclinedFaces": list of inclined faces
            5. "verticalApertures": list of vertical apertures
            6. "topHorizontalApertures": list of top horizontal apertures
            7. "bottomHorizontalApertures": list of bottom horizontal apertures
            8. "inclinedApertures": list of inclined apertures

        """
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

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Decompose - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        verticalFaces = []
        topHorizontalFaces = []
        bottomHorizontalFaces = []
        inclinedFaces = []
        verticalApertures = []
        topHorizontalApertures = []
        bottomHorizontalApertures = []
        inclinedApertures = []
        tiltAngle = abs(tiltAngle)
        faces = Cell.Faces(cell)
        zList = []
        for f in faces:
            zList.append(f.Centroid().Z())
        zMin = min(zList)
        zMax = max(zList)
        up = [0, 0, 1]
        for aFace in faces:
            aCode = angleCode(aFace, up, tiltAngle)

            if aCode == 0:
                verticalFaces.append(aFace)
                verticalApertures += getApertures(aFace)
            elif aCode == 1:
                if abs(aFace.Centroid().Z() - zMin) < tolerance:
                    bottomHorizontalFaces.append(aFace)
                    bottomHorizontalApertures += getApertures(aFace)
                else:
                    topHorizontalFaces.append(aFace)
                    topHorizontalApertures += getApertures(aFace)
            elif aCode == 2:
                if abs(aFace.Centroid().Z() - zMax) < tolerance:
                    topHorizontalFaces.append(aFace)
                    topHorizontalApertures += getApertures(aFace)
                else:
                    bottomHorizontalFaces.append(aFace)
                    bottomHorizontalApertures += getApertures(aFace)
            elif aCode == 3:
                inclinedFaces.append(aFace)
                inclinedApertures += getApertures(aFace)
        d = {
            "verticalFaces" : verticalFaces,
            "topHorizontalFaces" : topHorizontalFaces,
            "bottomHorizontalFaces" : bottomHorizontalFaces,
            "inclinedFaces" : inclinedFaces,
            "verticalApertures" : verticalApertures,
            "topHorizontalApertures" : topHorizontalApertures,
            "bottomHorizontalApertures" : bottomHorizontalApertures,
            "inclinedApertures" : inclinedApertures
            }
        return d

    @staticmethod
    def Dodecahedron(origin= None, radius: float = 0.5,
                  direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates a dodecahedron. See https://en.wikipedia.org/wiki/Dodecahedron.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the dodecahedron. The default is None which results in the dodecahedron being placed at (0, 0, 0).
        radius : float , optional
            The radius of the dodecahedron's circumscribed sphere. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the dodecahedron. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the dodecahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Cell
            The created dodecahedron.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Dodecahedron - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        pen = Face.Circle(sides=5, radius=0.5)
        pentagons = [pen]
        edges = Topology.Edges(pen)
        for edge in edges:
            o = Topology.Centroid(edge)
            e_dir = Edge.Direction(edge)
            pentagons.append(Topology.Rotate(pen, origin=o, axis=e_dir, angle=116.565))

        cluster = Cluster.ByTopologies(pentagons)
        
        cluster2 = Topology.Rotate(cluster, origin=Vertex.Origin(), axis=[1, 0, 0], angle=180)
        #cluster2 = Topology.Rotate(cluster2, origin=Vertex.Origin(), axis=[0, 0, 1], angle=36)
        vertices = Topology.Vertices(cluster)
        zList = [Vertex.Z(v) for v in vertices]
        zList = list(set(zList))
        zList.sort()
        zoffset = zList[1] - zList[0]
        total_height = zList[-1] - zList[0]
        cluster2 = Topology.Translate(cluster2, 0, 0, total_height+zoffset)
        pentagons += Topology.Faces(cluster2)
        dodecahedron = Cell.ByFaces(pentagons, tolerance=tolerance)
        centroid = Topology.Centroid(dodecahedron)
        dodecahedron = Topology.Translate(dodecahedron, -Vertex.X(centroid), -Vertex.Y(centroid), -Vertex.Z(centroid))
        vertices = Topology.Vertices(dodecahedron)
        d = Vertex.Distance(Vertex.Origin(), vertices[0])
        dodecahedron = Topology.Scale(dodecahedron, origin=Vertex.Origin(), x=radius/d, y=radius/d, z=radius/d)
        if placement == "bottom":
            dodecahedron = Topology.Translate(dodecahedron, 0, 0, radius)
        elif placement == "lowerleft":
            dodecahedron = Topology.Translate(dodecahedron, radius, radius, radius)
        
        geo = Topology.Geometry(dodecahedron)
        vertices = [Vertex.ByCoordinates(coords) for coords in geo['vertices']]
        vertices = Vertex.Fuse(vertices)
        coords = [Vertex.Coordinates(v) for v in vertices]
        dodecahedron = Topology.RemoveCoplanarFaces(Topology.SelfMerge(Topology.ByGeometry(vertices=coords, faces=geo['faces'])))
        dodecahedron = Topology.Orient(dodecahedron, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        dodecahedron = Topology.Place(dodecahedron, originA=Vertex.Origin(), originB=origin)
        return dodecahedron

    @staticmethod
    def Edges(cell) -> list:
        """
        Returns the edges of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Edges - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        edges = Topology.Edges(cell)
        return edges

    @staticmethod
    def Egg(origin= None, height: float = 1.0, uSides: int = 16, vSides: int = 8, direction: list = [0, 0, 1],
                   placement: str = "center", tolerance: float = 0.0001):
        """
        Creates an egg-shaped cell.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the sphere. The default is None which results in the egg-shaped cell being placed at (0, 0, 0).
        height : float , optional
            The desired height of of the egg-shaped cell. The default is 1.0.
        uSides : int , optional
            The desired number of sides along the longitude of the egg-shaped cell. The default is 16.
        vSides : int , optional
            The desired number of sides along the latitude of the egg-shaped cell. The default is 8.
        direction : list , optional
            The vector representing the up direction of the egg-shaped cell. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the egg-shaped cell. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created egg-shaped cell.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Sphere - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        coords = [[0.0, 0.0, -0.5],
                 [0.074748, 0.0, -0.494015],
                 [0.140819, 0.0, -0.473222],
                 [0.204118, 0.0, -0.438358],
                 [0.259512, 0.0, -0.391913],
                 [0.304837, 0.0, -0.335519],
                 [0.338649, 0.0, -0.271416],
                 [0.361307, 0.0, -0.202039],
                 [0.375678, 0.0, -0.129109],
                 [0.381294, 0.0, -0.053696],
                 [0.377694, 0.0, 0.019874],
                 [0.365135, 0.0, 0.091978],
                 [0.341482, 0.0, 0.173973],
                 [0.300154, 0.0, 0.276001],
                 [0.252928, 0.0, 0.355989],
                 [0.206605, 0.0, 0.405813],
                 [0.157529, 0.0, 0.442299],
                 [0.10604, 0.0, 0.472092],
                 [0.05547, 0.0, 0.491784],
                 [0.0, 0.0, 0.5]]
        verts = [Vertex.ByCoordinates(coord) for coord in coords]
        c = Wire.ByVertices(verts, close=False)
        new_verts = []
        for i in range(vSides+1):
            new_verts.append(Wire.VertexByParameter(c, i/vSides))
        c = Wire.ByVertices(new_verts, close=False)
        egg = Topology.Spin(c, origin=Vertex.Origin(), triangulate=False, direction=[0, 0, 1], angle=360, sides=uSides, tolerance=tolerance)
        if Topology.IsInstance(egg, "CellComplex"):
            egg = CellComplex.ExternalBoundary(egg)
        if Topology.IsInstance(egg, "Shell"):
            egg = Cell.ByShell(egg)
        egg = Topology.Scale(egg, origin=Vertex.Origin(), x=height, y=height, z=height)
        if placement.lower() == "bottom":
            egg = Topology.Translate(egg, 0, 0, height/2)
        elif placement.lower() == "lowerleft":
            bb = Cell.BoundingBox(egg)
            d = Topology.Dictionary(bb)
            width = Dictionary.ValueAtKey(d, 'width')
            length = Dictionary.ValueAtKey(d, 'length')
            egg = Topology.Translate(egg, width*0.5, length*0.5, height*0.5)
        egg = Topology.Orient(egg, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        egg = Topology.Place(egg, originA=Vertex.Origin(), originB=origin)
        return egg
    
    @staticmethod
    def ExternalBoundary(cell):
        """
        Returns the external boundary of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.

        Returns
        -------
        topologic_core.Shell
            The external boundary of the input cell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.ExternalBoundary - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        try:
            return cell.ExternalBoundary() # Hook to Core
        except:
            print("Cell.ExternalBoundary - Error: Could not compute the external boundary. Returning None.")
            return None
    
    @staticmethod
    def Faces(cell) -> list:
        """
        Returns the faces of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.

        Returns
        -------
        list
            The list of faces.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Faces - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        faces = Topology.Faces(cell)
        return faces

    @staticmethod
    def Hyperboloid(origin = None, baseRadius: float = 0.5, topRadius: float = 0.5, height: float = 1, sides: int = 24, direction: list = [0, 0, 1],
                        twist: float = 60, placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a hyperboloid.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the hyperboloid. The default is None which results in the hyperboloid being placed at (0, 0, 0).
        baseRadius : float , optional
            The radius of the base circle of the hyperboloid. The default is 0.5.
        topRadius : float , optional
            The radius of the top circle of the hyperboloid. The default is 0.5.
        height : float , optional
            The height of the cone. The default is 1.
        sides : int , optional
            The number of sides of the cone. The default is 24.
        direction : list , optional
            The vector representing the up direction of the hyperboloid. The default is [0, 0, 1].
        twist : float , optional
            The angle to twist the base cylinder. The default is 60.
        placement : str , optional
            The description of the placement of the origin of the hyperboloid. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created hyperboloid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        import math

        def createHyperboloid(baseVertices, topVertices, tolerance=0.0001):
            baseWire = Wire.ByVertices(baseVertices, close=True)
            topWire = Wire.ByVertices(topVertices, close=True)
            baseFace = Face.ByWire(baseWire, tolerance=tolerance)
            topFace = Face.ByWire(topWire, tolerance=tolerance)
            faces = [baseFace, topFace]
            for i in range(0, len(baseVertices)-1):
                w = Wire.ByVertices([baseVertices[i], topVertices[i], topVertices[i+1]], close=True)
                f = Face.ByWire(w, tolerance=tolerance)
                faces.append(f)
                w = Wire.ByVertices([baseVertices[i+1], baseVertices[i], topVertices[i+1]], close=True)
                f = Face.ByWire(w, tolerance=tolerance)
                faces.append(f)
            w = Wire.ByVertices([baseVertices[-1], topVertices[-1], topVertices[0]], close=True)
            f = Face.ByWire(w, tolerance=tolerance)
            faces.append(f)
            w = Wire.ByVertices([baseVertices[0], baseVertices[-1], topVertices[0]], close=True)
            f = Face.ByWire(w, tolerance=tolerance)
            faces.append(f)
            returnTopology = Cell.ByFaces(faces, tolerance=tolerance)
            if returnTopology == None:
                returnTopology = Cluster.ByTopologies(faces)
            return returnTopology
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Hyperboloid - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        w_origin = Vertex.Origin()
        baseV = []
        topV = []
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "center":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = max(baseRadius, topRadius)
            yOffset = max(baseRadius, topRadius)
        baseZ = w_origin.Z() + zOffset
        topZ = w_origin.Z() + zOffset + height
        for i in range(sides):
            angle = math.radians(360/sides)*i
            if baseRadius > 0:
                baseX = math.sin(angle+math.radians(twist))*baseRadius + Vertex.X(w_origin, mantissa=mantissa) + xOffset
                baseY = math.cos(angle+math.radians(twist))*baseRadius + Vertex.Y(w_origin, mantissa=mantissa) + yOffset
                baseZ = Vertex.Z(w_origin, mantissa=mantissa) + zOffset
                baseV.append(Vertex.ByCoordinates(baseX,baseY,baseZ))
            if topRadius > 0:
                topX = math.sin(angle-math.radians(twist))*topRadius + Vertex.X(w_origin, mantissa=mantissa) + xOffset
                topY = math.cos(angle-math.radians(twist))*topRadius + Vertex.Y(w_origin, mantissa=mantissa) + yOffset
                topV.append(Vertex.ByCoordinates(topX,topY,topZ))

        hyperboloid = createHyperboloid(baseV, topV, tolerance=tolerance)
        if hyperboloid == None:
            print("Cell.Hyperboloid - Error: Could not create a hyperboloid. Returning None.")
            return None
        
        hyperboloid = Topology.Orient(hyperboloid, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        hyperboloid = Topology.Place(hyperboloid, originA=Vertex.Origin(), originB=origin)
        return hyperboloid
    
    @staticmethod
    def Icosahedron(origin= None, radius: float = 0.5,
                  direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates an icosahedron. See https://en.wikipedia.org/wiki/Icosahedron.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the icosahedron. The default is None which results in the icosahedron being placed at (0, 0, 0).
        radius : float , optional
            The radius of the icosahedron's circumscribed sphere. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the icosahedron. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the icosahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Cell
            The created icosahedron.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Dodecahedron - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        rect1 = Wire.Rectangle(width=(1+math.sqrt(5))/2, length=1)
        rect2 = Wire.Rectangle(width=1, length=(1+math.sqrt(5))/2)
        rect2 = Topology.Rotate(rect2, origin=Vertex.Origin(), axis=[1, 0, 0], angle=90)
        rect3 = Wire.Rectangle(width=1, length=(1+math.sqrt(5))/2)
        rect3 = Topology.Rotate(rect3, origin=Vertex.Origin(), axis=[0, 1, 0], angle=90)
        vertices = Topology.Vertices(rect1)
        v1, v2, v3, v4 = vertices
        vertices = Topology.Vertices(rect2)
        v5, v6, v7, v8 = vertices
        vertices = Topology.Vertices(rect3)
        v9, v10, v11, v12 = vertices
        f1 = Face.ByVertices([v1,v8,v4])
        f2 = Face.ByVertices([v1,v4,v5])
        f3 = Face.ByVertices([v3,v2,v6])
        f4 = Face.ByVertices([v2,v3,v7])
        f5 = Face.ByVertices([v10,v9,v2])
        f6 = Face.ByVertices([v10,v9,v1])
        f7 = Face.ByVertices([v12,v11,v4])
        f8 = Face.ByVertices([v12,v11,v3])
        f9 = Face.ByVertices([v8,v7,v9])
        f10 = Face.ByVertices([v8,v7,v12])
        f11 = Face.ByVertices([v5,v6,v10])
        f12 = Face.ByVertices([v5,v6,v11])
        f13 = Face.ByVertices([v8,v1,v9])
        f14 = Face.ByVertices([v9,v2,v7])
        f15 = Face.ByVertices([v7,v3,v12])
        f16 = Face.ByVertices([v8,v12,v4])
        f17 = Face.ByVertices([v1,v5,v10])
        f18 = Face.ByVertices([v10,v2,v6])
        f19 = Face.ByVertices([v6,v3,v11])
        f20 = Face.ByVertices([v11,v4,v5])

        icosahedron = Cell.ByFaces([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,
                        f11,f12,f13,f14,f15,f16,f17,f18,f19,f20], tolerance=tolerance)
        sf = 1.051*0.5 # To inscribe it in a sphere of radius 0.5
        icosahedron = Topology.Scale(icosahedron, origin=Vertex.Origin(), x=sf, y=sf, z=sf)
        sf = radius/0.5
        icosahedron = Topology.Scale(icosahedron, origin=Vertex.Origin(), x=sf, y=sf, z=sf)
        if placement == "bottom":
            icosahedron = Topology.Translate(icosahedron, 0, 0, radius)
        elif placement == "lowerleft":
            icosahedron = Topology.Translate(icosahedron, radius, radius, radius)
        
        icosahedron = Topology.Orient(icosahedron, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        icosahedron = Topology.Place(icosahedron, originA=Vertex.Origin(), originB=origin)
        return icosahedron

    @staticmethod
    def InternalBoundaries(cell) -> list:
        """
        Returns the internal boundaries of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.

        Returns
        -------
        list
            The list of internal boundaries ([topologic_core.Shell]).

        """
        shells = []
        _ = cell.InternalBoundaries(shells) # Hook to Core
        return shells
    
    @staticmethod
    def InternalVertex(cell, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex that is guaranteed to be inside the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Vertex
            The internal vertex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            if not silent:
                print("Cell.InternalVertex - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        try:
            return topologic.CellUtility.InternalVertex(cell, tolerance) # Hook to Core
        except:
            if not silent:
                print("Cell.InternalVertex - Error: Could not create an internal vertex. Returning None.")
            return None
    
    @staticmethod
    def IsOnBoundary(cell, vertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is on the boundary of the input cell. Returns False otherwise.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        vertex : topologic_core.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input cell. Returns False otherwise.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.IsOnBoundary - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Cell.IsOnBoundary - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
            return None
        try:
            return (Cell.ContainmentStatus(cell, vertex, tolerance = tolerance) == 1)
        except:
            print("Cell.IsOnBoundary - Error: Could not determine if the input vertex is on the boundary of the input cell. Returning None.")
            return None
    
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
        topologic_core.Cell
            The created octahedron.

        """
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Octahedron - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        vb1 = Vertex.ByCoordinates(-0.5, 0, 0)
        vb2 = Vertex.ByCoordinates(0, -0.5, 0)
        vb3 = Vertex.ByCoordinates(0.5, 0, 0)
        vb4 = Vertex.ByCoordinates(0, 0.5, 0)
        top = Vertex.ByCoordinates(0, 0, 0.5)
        bottom = Vertex.ByCoordinates(0, 0, -0.5)
        f1 = Face.ByVertices([top, vb1, vb2])
        f2 = Face.ByVertices([top, vb2, vb3])
        f3 = Face.ByVertices([top, vb3, vb4])
        f4 = Face.ByVertices([top, vb4, vb1])
        f5 = Face.ByVertices([bottom, vb1, vb2])
        f6 = Face.ByVertices([bottom, vb2, vb3])
        f7 = Face.ByVertices([bottom, vb3, vb4])
        f8 = Face.ByVertices([bottom, vb4, vb1])

        octahedron = Cell.ByFaces([f1, f2, f3, f4, f5, f6, f7, f8], tolerance=tolerance)
        octahedron = Topology.Scale(octahedron, origin=Vertex.Origin(), x=radius/0.5, y=radius/0.5, z=radius/0.5)
        if placement == "bottom":
            octahedron = Topology.Translate(octahedron, 0, 0, radius)
        elif placement == "lowerleft":
            octahedron = Topology.Translate(octahedron, radius, radius, radius)
        octahedron = Topology.Orient(octahedron, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        octahedron = Topology.Place(octahedron, originA=Vertex.Origin(), originB=origin)
        return octahedron
    
    @staticmethod
    def Paraboloid(origin= None, focalLength=0.125, width: float = 1, length: float = 1, height: float = 0, uSides: int = 16, vSides: int = 16,
                        direction: list = [0, 0, 1], placement: str ="center", mantissa: int = 6, tolerance: float = 0.0001, silent=False):
        """
        Creates a paraboloid cell. See https://en.wikipedia.org/wiki/Paraboloid

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the parabolic surface. The default is None which results in the parabolic surface being placed at (0, 0, 0).
        focalLength : float , optional
            The focal length of the parabola. The default is 0.125.
        width : float , optional
            The width of the parabolic surface. The default is 1.
        length : float , optional
            The length of the parabolic surface. The default is 1.
        height : float , optional
            The additional height of the parabolic surface. Please note this is not the height from the spring point to the apex. It is in addition to that to form a base. The default is 0.
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
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell

        if origin == None:
            origin = Vertex.Origin()
        
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Cell.Paraboloid - Error: The origin input parameter is not a valid vertex. Returning None.")
            return None
        if width <= 0:
            if not silent:
                print("Cell.Paraboloid - Error: The width input parameter cannot be less than or equal to zero. Returning None.")
            return None
        if length <= 0:
            if not silent:
                print("Cell.Paraboloid - Error: The length input parameter cannot be less than or equal to zero. Returning None.")
            return None
        if height < 0:
            if not silent:
                print("Cell.Paraboloid - Error: The height input parameter cannot be negative. Returning None.")
            return None
        
        para = Shell.Paraboloid(focalLength=focalLength, width=width, length=length, uSides=uSides, vSides=vSides,
                        direction=[0,0,1], placement="center", mantissa=mantissa, tolerance=tolerance)
        if not Topology.IsInstance(para, "Shell"):
            if not silent:
                print("Cell.Paraboloid - Error: Could not create paraboloid. Returning None.")
            return None
        eb = Shell.ExternalBoundary(para)
        vertices = Topology.Vertices(eb)
        z_list = [Vertex.Z(v) for v in vertices]
        if focalLength > 0:
            z = max(z_list) + height
        else:
            z = min(z_list) - height
        f = Face.Rectangle(origin=Vertex.ByCoordinates(0,0,z), width=width*1.1, length=length*1.1)
        proj_vertices = []
        for v in vertices:
            proj_vertices.append(Vertex.Project(v, f))
        w = Wire.ByVertices(proj_vertices, close=True)
        sleeve = Shell.ByWires([eb, w], triangulate=False, silent=True)
        if sleeve == None:
            if not silent:
                print("Cell.Paraboloid - Error: Could not create paraboloid. Returning None.")
            return None
        f = Face.ByWire(w, tolerance=tolerance)
        faces = Topology.Faces(sleeve) + [f] + Topology.Faces(para)
        cell = Cell.ByFaces(faces, tolerance=tolerance)
        if cell == None:
            if not silent:
                print("Cell.Paraboloid - Error: Could not create paraboloid. Returning None.")
            return None
        vertices = Topology.Vertices(cell)
        x_list = [Vertex.X(v, mantissa=mantissa) for v in vertices]
        y_list = [Vertex.Y(v, mantissa=mantissa) for v in vertices]
        z_list = [Vertex.Z(v, mantissa=mantissa) for v in vertices]
        min_x = min(x_list)
        max_x = max(x_list)
        mid_x = min_x + (max_x - min_x)/2
        min_y = min(y_list)
        max_y = max(y_list)
        mid_y = min_y + (max_y - min_y)/2
        min_z = min(z_list)
        max_z = max(z_list)
        mid_z = min_z + (max_z - min_z)/2
        if placement.lower() == "center":
            x_tran = -mid_x
            y_tran = -mid_y
            z_tran = -mid_z
        elif placement.lower() == "bottom":
            x_tran = -mid_x
            y_tran = -mid_y
            z_tran = -min_z
        elif placement.lower() == "lowerleft":
            x_tran = -min_x
            y_tran = -min_y
            z_tran = -min_z
        cell = Topology.Translate(cell, x_tran, y_tran, z_tran)
        cell = Topology.Place(cell, originA=Vertex.Origin(), originB=origin)
        if not direction == [0,0,1]:
            cell = Topology.Orient(cell, origin=origin, dirA=[0,0,1], dirB=direction)
        return cell

    @staticmethod
    def Pipe(edge, profile = None, radius: float = 0.5, sides: int = 16, startOffset: float = 0, endOffset: float = 0, endcapA = None, endcapB = None, mantissa: int = 6) -> dict:
        """
        Creates a pipe along the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The centerline of the pipe.
        profile : topologic_core.Wire , optional
            The profile of the pipe. It is assumed that the profile is in the XY plane. If set to None, a circle of radius 0.5 will be used. The default is None.
        radius : float , optional
            The radius of the pipe. The default is 0.5.
        sides : int , optional
            The number of sides of the pipe. The default is 16.
        startOffset : float , optional
            The offset distance from the start vertex of the centerline edge. The default is 0.
        endOffset : float , optional
            The offset distance from the end vertex of the centerline edge. The default is 0.
        endcapA, optional
            The topology to place at the start vertex of the centerline edge. The positive Z direction of the end cap will be oriented in the direction of the centerline edge.
        endcapB, optional
            The topology to place at the end vertex of the centerline edge. The positive Z direction of the end cap will be oriented in the inverse direction of the centerline edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        
        Returns
        -------
        dict
            A dictionary containing the pipe, the start endcap, and the end endcap if they have been specified. The dictionary has the following keys:
            'pipe'
            'endcapA'
            'endcapB'

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(edge, "Edge"):
            print("Cell.Pipe - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        length = Edge.Length(edge)
        origin = Edge.StartVertex(edge)
        startU = startOffset / length
        endU = 1.0 - (endOffset / length)
        sv = Edge.VertexByParameter(edge, startU)
        ev = Edge.VertexByParameter(edge, endU)
        x1 = Vertex.X(sv, mantissa=mantissa)
        y1 = Vertex.Y(sv, mantissa=mantissa)
        z1 = Vertex.Z(sv, mantissa=mantissa)
        x2 = Vertex.X(ev, mantissa=mantissa)
        y2 = Vertex.Y(ev, mantissa=mantissa)
        z2 = Vertex.Z(ev, mantissa=mantissa)
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        baseV = []
        topV = []

        if Topology.IsInstance(profile, "Wire"):
            baseWire = Topology.Translate(profile, 0 , 0, sv.Z())
            topWire = Topology.Translate(profile, 0 , 0, sv.Z()+dist)
        else:
            for i in range(sides):
                angle = math.radians(360/sides)*i
                x = math.sin(angle)*radius + Vertex.X(sv, mantissa=mantissa)
                y = math.cos(angle)*radius + Vertex.Y(sv, mantissa=mantissa)
                z = Vertex.Z(sv, mantissa=mantissa)
                baseV.append(Vertex.ByCoordinates(x, y, z))
                topV.append(Vertex.ByCoordinates(x, y, z+dist))

            baseWire = Wire.ByVertices(baseV)
            topWire = Wire.ByVertices(topV)
        wires = [baseWire, topWire]
        pipe = Cell.ByWires(wires)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        pipe = Topology.Rotate(pipe, origin=sv, axis=[0, 1, 0], angle=theta)
        pipe = Topology.Rotate(pipe, origin=sv, axis=[0, 0, 1], angle=phi)
        zzz = Vertex.ByCoordinates(0, 0, 0)
        if endcapA:
            origin = edge.StartVertex()
            x1 = Vertex.X(origin, mantissa=mantissa)
            y1 = Vertex.Y(origin, mantissa=mantissa)
            z1 = Vertex.Z(origin, mantissa=mantissa)
            x2 = Vertex.X(edge.EndVertex(), mantissa=mantissa)
            y2 = Vertex.Y(edge.EndVertex(), mantissa=mantissa)
            z2 = Vertex.Z(edge.EndVertex(), mantissa=mantissa)
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1    
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
            if dist < 0.0001:
                theta = 0
            else:
                theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
            endcapA = Topology.Copy(endcapA)
            endcapA = Topology.Rotate(endcapA, origin=zzz, axis=[0, 1, 0], angle=theta)
            endcapA = Topology.Rotate(endcapA, origin=zzz, axis=[0, 0, 1], angle=phi+180)
            endcapA = Topology.Translate(endcapA, Vertex.X(origin, mantissa=mantissa), Vertex.Y(origin, mantissa=mantissa), Vertex.Z(origin, mantissa=mantissa))
        if endcapB:
            origin = edge.EndVertex()
            x1 = Vertex.X(origin, mantissa=mantissa)
            y1 = Vertex.Y(origin, mantissa=mantissa)
            z1 = Vertex.Z(origin, mantissa=mantissa)
            x2 = Vertex.X(edge.StartVertex(), mantissa=mantissa)
            y2 = Vertex.Y(edge.StartVertex(), mantissa=mantissa)
            z2 = Vertex.Z(edge.StartVertex(), mantissa=mantissa)
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1    
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
            if dist < 0.0001:
                theta = 0
            else:
                theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
            endcapB = Topology.Copy(endcapB)
            endcapB = Topology.Rotate(endcapB, origin=zzz, axis=[0, 1, 0], angle=theta)
            endcapB = Topology.Rotate(endcapB, origin=zzz, axis=[0, 0, 1], angle=phi+180)
            endcapB = Topology.Translate(endcapB, Vertex.X(origin, mantissa=mantissa), Vertex.Y(origin, mantissa=mantissa), Vertex.Z(origin, mantissa=mantissa))
        return {'pipe': pipe, 'endcapA': endcapA, 'endcapB': endcapB}
    
    @staticmethod
    def Prism(origin= None, width: float = 1, length: float = 1, height: float = 1, uSides: int = 1, vSides: int = 1, wSides: int = 1,
                  direction: list = [0, 0, 1], placement: str ="center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a prism.

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
        topologic_core.Cell
            The created prism.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        def sliceCell(cell, width, length, height, uSides, vSides, wSides):
            origin = Topology.Centroid(cell)
            shells = Topology.Shells(cell)
            shell = shells[0]
            wRect = Wire.Rectangle(origin=origin, width=width*1.2, length=length*1.2, direction=[0, 0, 1], placement="center")
            sliceFaces = []
            for i in range(1, wSides):
                sliceFaces.append(Topology.Translate(Face.ByWire(wRect, tolerance=tolerance), 0, 0, height/wSides*i - height*0.5))
            uRect = Wire.Rectangle(origin=origin, width=height*1.2, length=length*1.2, direction=[1, 0, 0], placement="center")
            for i in range(1, uSides):
                sliceFaces.append(Topology.Translate(Face.ByWire(uRect, tolerance=tolerance), width/uSides*i - width*0.5, 0, 0))
            vRect = Wire.Rectangle(origin=origin, width=height*1.2, length=width*1.2, direction=[0, 1, 0], placement="center")
            for i in range(1, vSides):
                sliceFaces.append(Topology.Translate(Face.ByWire(vRect, tolerance=tolerance), 0, length/vSides*i - length*0.5, 0))
            if len(sliceFaces) > 0:
                sliceCluster = Cluster.ByTopologies(sliceFaces)
                shell = Topology.Slice(topologyA=shell, topologyB=sliceCluster, tranDict=False, tolerance=tolerance)
                return Cell.ByShell(shell)
            return cell
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Prism - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "center":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
        vb1 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)-width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)-length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)
        vb2 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)-length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)
        vb3 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)+length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)
        vb4 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)-width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)+length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire, tolerance=tolerance)

        prism = Cell.ByThickenedFace(baseFace, thickness=height, bothSides = False)

        if uSides > 1 or vSides > 1 or wSides > 1:
            prism = sliceCell(prism, width, length, height, uSides, vSides, wSides)
        prism = Topology.Orient(prism, origin=origin, dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        return prism

    @staticmethod
    def RemoveCollinearEdges(cell, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Removes any collinear edges in the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created cell without any collinear edges.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import inspect
        
        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.RemoveCollinearEdges - Error: The input cell parameter is not a valid cell. Returning None.")
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
            return None
        faces = Cell.Faces(cell)
        clean_faces = []
        for face in faces:
            clean_faces.append(Face.RemoveCollinearEdges(face, angTolerance=angTolerance, tolerance=tolerance))
        return Cell.ByFaces(clean_faces, tolerance=tolerance)
    
    @staticmethod
    def Roof(face, angle: float = 45, epsilon: float = 0.01 , tolerance: float = 0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angle : float , optional
            The desired angle in degrees of the roof. The default is 45.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for distance from plane). The default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        cell
            The created roof.

        """
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        
        shell = Shell.Roof(face=face, angle=angle, epsilon=epsilon, tolerance=tolerance)
        if not Topology.IsInstance(shell, "Shell"):
            print("Cell.Roof - Error: Could not create a roof cell. Returning None.")
            return None
        faces = Topology.Faces(shell) + [face]
        cell = Cell.ByFaces(faces, tolerance=tolerance)
        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Roof - Error: Could not create a roof cell. Returning None.")
            return None
        return cell

    @staticmethod
    def Sets(cells: list, superCells: list, tolerance: float = 0.0001) -> list:
        """
            Classifies the input cells into sets based on their enclosure within the input list of super cells. The order of the sets follows the order of the input list of super cells.

        Parameters
        ----------
        inputCells : list
            The list of input cells.
        superCells : list
            The list of super cells.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The classified list of input cells based on their enclosure within the input list of super cells.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(cells, list):
            print("Cell.Sets - Error: The input cells parameter is not a valid list. Returning None.")
            return None
        if not isinstance(superCells, list):
            print("Cell.Sets - Error: The input superCells parameter is not a valid list. Returning None.")
            return None
        cells = [c for c in cells if Topology.IsInstance(c, "Cell")]
        if len(cells) < 1:
            print("Cell.Sets - Error: The input cells parameter does not contain any valid cells. Returning None.")
            return None
        superCells = [c for c in superCells if Topology.IsInstance(c, "Cell")]
        if len(cells) < 1:
            print("Cell.Sets - Error: The input cells parameter does not contain any valid cells. Returning None.")
            return None
        if len(superCells) == 0:
            cluster = cells[0]
            for i in range(1, len(cells)):
                oldCluster = cluster
                cluster = cluster.Union(cells[i])
                del oldCluster
            superCells = Topology.Cells(cluster)
        unused = []
        for i in range(len(cells)):
            unused.append(True)
        sets = []
        for i in range(len(superCells)):
            sets.append([])
        for i in range(len(cells)):
            if unused[i]:
                iv = Topology.InternalVertex(cells[i], tolerance=tolerance)
                for j in range(len(superCells)):
                    if (Vertex.IsInternal(iv, superCells[j], tolerance=tolerance)):
                        sets[j].append(cells[i])
                        unused[i] = False
        return sets
    
    @staticmethod
    def Shells(cell) -> list:
        """
        Returns the shells of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.

        Returns
        -------
        list
            The list of shells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Shells - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        shells = []
        _ = cell.Shells(None, shells) # Hook to Core
        return shells

    @staticmethod
    def Sphere(origin= None, radius: float = 0.5, uSides: int = 16, vSides: int = 8, direction: list = [0, 0, 1],
                   placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a sphere.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the sphere. The default is None which results in the sphere being placed at (0, 0, 0).
        radius : float , optional
            The radius of the sphere. The default is 0.5.
        uSides : int , optional
            The number of sides along the longitude of the sphere. The default is 16.
        vSides : int , optional
            The number of sides along the latitude of the sphere. The default is 8.
        direction : list , optional
            The vector representing the up direction of the sphere. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the sphere. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created sphere.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Sphere - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        c = Wire.Circle(origin=Vertex.Origin(), radius=radius, sides=vSides, fromAngle=0, toAngle=180, close=False, direction=[0, 0, 1], placement="center")
        c = Topology.Rotate(c, origin=Vertex.Origin(), axis=[1, 0, 0], angle=90)
        sphere = Topology.Spin(c, origin=Vertex.Origin(), triangulate=False, direction=[0, 0, 1], angle=360, sides=uSides, tolerance=tolerance)
        if Topology.Type(sphere) == Topology.TypeID("CellComplex"):
            sphere = CellComplex.ExternalBoundary(sphere)
        if Topology.Type(sphere) == Topology.TypeID("Shell"):
            sphere = Cell.ByShell(sphere)
        if placement.lower() == "bottom":
            sphere = Topology.Translate(sphere, 0, 0, radius)
        elif placement.lower() == "lowerleft":
            sphere = Topology.Translate(sphere, radius, radius, radius)
        sphere = Topology.Orient(sphere, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        sphere = Topology.Place(sphere, originA=Vertex.Origin(), originB=origin)
        return sphere
    
    @staticmethod
    def SurfaceArea(cell, mantissa: int = 6) -> float:
        """
        Returns the surface area of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The cell.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        area : float
            The surface area of the input cell.

        """
        return Cell.Area(cell=cell, mantissa=mantissa)
    
    @staticmethod
    def Tetrahedron(origin= None, radius: float = 0.5,
                  direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates a tetrahedron. See https://en.wikipedia.org/wiki/Tetrahedron.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the tetrahedron. The default is None which results in the tetrahedron being placed at (0, 0, 0).
        radius : float , optional
            The radius of the tetrahedron's circumscribed sphere. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the tetrahedron. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the tetrahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Cell
            The created tetrahedron.

        """
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Tetrahedron - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None

        vb1 = Vertex.ByCoordinates(math.sqrt(8/9), 0, -1/3)
        vb2 = Vertex.ByCoordinates(-math.sqrt(2/9), math.sqrt(2/3), -1/3)
        vb3 = Vertex.ByCoordinates(-math.sqrt(2/9), -math.sqrt(2/3), -1/3)
        vb4 = Vertex.ByCoordinates(0, 0, 1)
        f1 = Face.ByVertices([vb1, vb2, vb3])
        f2 = Face.ByVertices([vb4, vb1, vb2])
        f3 = Face.ByVertices([vb4, vb2, vb3])
        f4 = Face.ByVertices([vb4, vb3, vb1])
        tetrahedron = Cell.ByFaces([f1, f2, f3, f4])
        tetrahedron = Topology.Scale(tetrahedron, origin=Vertex.Origin(), x=0.5, y=0.5, z=0.5)
        tetrahedron = Topology.Scale(tetrahedron, origin=Vertex.Origin(), x=radius/0.5, y=radius/0.5, z=radius/0.5)

        if placement.lower() == "lowerleft":
            tetrahedron = Topology.Translate(tetrahedron, radius, radius, radius)
        elif placement.lower() == "bottom":
            tetrahedron = Topology.Translate(tetrahedron, 0, 0, radius)
        tetrahedron = Topology.Place(tetrahedron, originA=Vertex.Origin(), originB=origin)
        tetrahedron = Topology.Orient(tetrahedron, origin=origin, dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        return tetrahedron
    
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
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Cell.Torus - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        c = Wire.Circle(origin=Vertex.Origin(), radius=minorRadius, sides=vSides, fromAngle=0, toAngle=360, close=False, direction=[0, 1, 0], placement="center")
        c = Topology.Translate(c, abs(majorRadius-minorRadius), 0, 0)
        torus = Topology.Spin(c, origin=Vertex.Origin(), triangulate=False, direction=[0, 0, 1], angle=360, sides=uSides, tolerance=tolerance)
        if Topology.Type(torus) == Topology.TypeID("Shell"):
            torus = Cell.ByShell(torus)
        if placement.lower() == "bottom":
            torus = Topology.Translate(torus, 0, 0, minorRadius)
        elif placement.lower() == "lowerleft":
            torus = Topology.Translate(torus, majorRadius, majorRadius, minorRadius)

        torus = Topology.Orient(torus, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        torus = Topology.Place(torus, originA=Vertex.Origin(), originB=origin)
        return torus
    
    @staticmethod
    def Vertices(cell) -> list:
        """
        Returns the vertices of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Vertices - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        vertices = []
        _ = cell.Vertices(None, vertices) # Hook to Core
        return vertices

    @staticmethod
    def Volume(cell, mantissa: int = 6) -> float:
        """
        Returns the volume of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        manitssa: int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The volume of the input cell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Volume - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        volume = None
        try:
            volume = round(topologic.CellUtility.Volume(cell), mantissa) # Hook to Core
        except:
            print("Cell.Volume - Error: Could not compute the volume of the input cell. Returning None.")
            volume = None
        return volume

    @staticmethod
    def Wires(cell) -> list:
        """
        Returns the wires of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.

        Returns
        -------
        list
            The list of wires.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.Wires - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        wires = []
        _ = cell.Wires(None, wires) # Hook to Core
        return wires

