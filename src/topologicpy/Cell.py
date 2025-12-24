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
            The number of decimal places to round the result to. Default is 6.

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
            The origin location of the box. Default is None which results in the box being placed at (0, 0, 0).
        width : float , optional
            The width of the box. Default is 1.
        length : float , optional
            The length of the box. Default is 1.
        height : float , optional
            The height of the box.
        uSides : int , optional
            The number of sides along the width. Default is 1.
        vSides : int , optional
            The number of sides along the length. Default is 1.
        wSides : int , optional
            The number of sides along the height. Default is 1.
        direction : list , optional
            The vector representing the up direction of the box. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the box. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
            If set to True, the input faces are planarized before building the cell. Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        if len(faceList) < 3:
            if not silent:
                print("Cell.ByFaces - Error: The input faces parameter does not contain valid faces. Returning None.")
                return None
        # Try the default method
        cell = topologic.Cell.ByFaces(faceList, tolerance) # Hook to Core
        if Topology.IsInstance(cell, "Cell"):
            return cell
        
        # Cleanup any non-manifold edges in faces
        clean_faces = []
        for face in faceList:
            eb = Face.ExternalBoundary(face)
            if not Wire.IsClosed(eb):
                continue
            ibList = Face.InternalBoundaries(face)
            closed_ibList = []
            for ib in ibList:
                if Wire.IsClosed(ib):
                    closed_ibList.append(ib)
                else:
                    print("Found an open wire!")
            clean_face = Face.ByWires(eb, closed_ibList)
            if Topology.IsInstance(clean_face, "face"):
                clean_faces.append(clean_face)
        # Try the default method again
        cell = topologic.Cell.ByFaces(clean_faces, tolerance) # Hook to Core
        if Topology.IsInstance(cell, "Cell"):
            return cell
        else:
            if not silent:
                print("Cell.ByFaces - Error: Could not construct cell. Trying other methods.")
        # Fuse all the vertices first and rebuild the faces
        all_vertices = []
        wires = []
        for f in faceList:
            w = Face.Wire(f)
            if Wire.IsClosed(w):
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
                v = Topology.Translate(centroid,
                                       x = n[0]*0.01,
                                       y = n[1]*0.01,
                                       z = n[2]*0.01,
                                       transferDictionaries = False,
                                       silent=True)
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
                    print("Cell.ByFaces 1 - Error: The operation failed. Returning None.")
                    return None
            else:
                return cell
        else:
            cell = topologic.Cell.ByFaces(faces, tolerance) # Hook to Core
            if cell == None:
                if not silent:
                    print("Cell.ByFaces 2 - Error: The operation failed. Returning None.")
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
            The desired offset distance. Default is 1.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
    def ByShell(shell, planarize: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cell from the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell. The shell must be closed for this method to succeed.
        planarize : bool, optional
            If set to True, the input faces of the input shell are planarized before building the cell. Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
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
            The input list of internal boundaries (closed shells). Default is an empty list.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def ByThickenedFace(face, thickness: float = 1.0, bothSides: bool = True, wSides: int = 1,
                        reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cell by thickening the input face.

        Behaviour:
        - Only the bottom and top faces are used as horizontal faces.
        - wSides controls the number of vertical segments along the thickness.
        Intermediate offset layers are used only to build side faces and are
        not included as horizontal faces in Cell.ByFaces.

        Parameters
        ----------
        face : topologic_core.Face
            The input face to be thickened.
        thickness : float , optional
            The desired thickness. Default is 1.0.
        bothSides : bool
            If True, the thickening is symmetric about the original face
            (i.e. from -thickness/2 to +thickness/2).
            If False, the thickening is from 0 to +thickness along the face normal.
            Default is True.
        reverse : bool
            If True, the extrusion direction is flipped (normal is negated).
            Default is False.
        wSides: int, optional
            The number of segments along the thickness direction.
            This is the same definition regardless of bothSides.
            Default is 1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell, or None on failure.
        """
        import math
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cell import Cell

        # -----------------------------
        # Validation
        # -----------------------------
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Cell.ByThickenedFace - Error: Input is not a valid Face. Returning None.")
            return None

        if thickness <= tolerance:
            if not silent:
                print("Cell.ByThickenedFace - Error: Thickness is less than or equal to the tolerance. Returning None.")
            return None
        
        if wSides < 1:
            if not silent:
                print("Cell.ByThickenedFace - Error: wSides is less than 1. Returning None.")
            return None

        if thickness/float(wSides) <= tolerance:
            if not silent:
                print("Cell.ByThickenedFace - Error: The distance between layers is less than or equal to the tolerance. Returning None.")
            return None

        # -----------------------------
        # Face normal (normalized)
        # -----------------------------
        normal = Face.Normal(face)
        if not isinstance(normal, (list, tuple)) or len(normal) != 3:
            if not silent:
                print("Cell.ByThickenedFace - Error: Could not compute face normal.")
            return None

        nx, ny, nz = normal
        L = math.sqrt(nx*nx + ny*ny + nz*nz)
        if L <= tolerance:
            if not silent:
                print("Cell.ByThickenedFace - Error: Degenerate face normal.")
            return None

        nx, ny, nz = nx/L, ny/L, nz/L

        if reverse:
            nx, ny, nz = -nx, -ny, -nz

        # -----------------------------
        # Build offset layers
        # NOTE: We will only keep the min/max offset faces as bottom/top.
        # Intermediate layers are used only for building side faces.
        # -----------------------------
        step = thickness / float(wSides)
        layers = []

        if bothSides:
            # Symmetric: [-thickness/2, ..., +thickness/2]
            start = -0.5 * thickness
            for i in range(wSides + 1):
                offset = start + step * i
                f = Topology.Translate(face, nx*offset, ny*offset, nz*offset)
                layers.append((offset, f))
        else:
            # One-sided: [0, ..., thickness]
            for i in range(wSides + 1):
                offset = step * i
                f = Topology.Translate(face, nx*offset, ny*offset, nz*offset)
                layers.append((offset, f))

        if len(layers) < 2:
            if not silent:
                print("Cell.ByThickenedFace - Error: Not enough layers to form a volume.")
            return None

        layers.sort(key=lambda x: x[0])

        # Bottom and top faces only
        bottom_face = layers[0][1]
        top_face = layers[-1][1]

        faces_all = [bottom_face, top_face]

        # -----------------------------
        # Build side faces between each consecutive pair of layers
        # These are the vertical segmentation faces controlled by wSides.
        # -----------------------------
        for i in range(len(layers) - 1):
            _, faceA = layers[i]
            _, faceB = layers[i + 1]

            edgesA = Topology.Edges(faceA)
            edgesB = Topology.Edges(faceB)

            if not edgesA or not edgesB or len(edgesA) != len(edgesB):
                if not silent:
                    print("Cell.ByThickenedFace - Warning: Edge mismatch between layers. "
                        "Side faces may be incomplete.")
                # We try to continue with min length
            count = min(len(edgesA), len(edgesB))

            for j in range(count):
                eA = edgesA[j]
                eB = edgesB[j]

                vA = Topology.Vertices(eA)
                vB = Topology.Vertices(eB)

                if not vA or not vB or len(vA) != 2 or len(vB) != 2:
                    continue

                vA1, vA2 = vA
                vB1, vB2 = vB

                try:
                    e1 = Edge.ByStartVertexEndVertex(vA1, vA2)
                    e2 = Edge.ByStartVertexEndVertex(vA2, vB2)
                    e3 = Edge.ByStartVertexEndVertex(vB2, vB1)
                    e4 = Edge.ByStartVertexEndVertex(vB1, vA1)

                    if not (e1 and e2 and e3 and e4):
                        continue

                    side_wire = Wire.ByEdges([e1, e2, e3, e4])
                    if not side_wire:
                        continue

                    side_face = Face.ByWire(side_wire)
                    if side_face:
                        faces_all.append(side_face)
                except Exception:
                    # Skip problematic quads but continue
                    continue

        # -----------------------------
        # Build final cell
        # -----------------------------
        try:
            cell = Cell.ByFaces(faces_all, tolerance=tolerance)
        except Exception:
            if not silent:
                print("Cell.ByThickenedFace - Error: Cell.ByFaces failed.")
            return None

        if not Topology.IsInstance(cell, "Cell"):
            if not silent:
                print("Cell.ByThickenedFace - Error: Cell.ByFaces did not return a valid Cell.")
            return None

        return cell

    @staticmethod
    def ByThickenedShell(shell, direction: list = [0, 0, 1], thickness: float = 1.0, bothSides: bool = True, reverse: bool = False,
                            planarize: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cell by thickening the input shell. The shell must be open.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell to be thickened.
        thickness : float , optional
            The desired thickness. Default is 1.0.
        bothSides : bool
            If True, the cell will be lofted to each side of the shell. Otherwise, it will be lofted along the input direction. Default is True.
        reverse : bool
            If True, the cell will be lofted along the opposite of the input direction. Default is False.
        planarize : bool, optional
            If set to True, the input faces of the input shell are planarized before building the cell. Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            if not silent:
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
            sideEdge1 = Edge.ByVertices([Edge.StartVertex(bottomEdge), Edge.StartVertex(topEdge)], tolerance=tolerance, silent=silent)
            sideEdge2 = Edge.ByVertices([Edge.EndVertex(bottomEdge), Edge.EndVertex(topEdge)], tolerance=tolerance, silent=silent)
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
            If set to True, the last wire in the list of input wires will be connected to the first wire in the list of input wires. Default is False.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. Default is True.
        planarize : bool, optional
            If set to True, the created faces are planarized before building the cell. Otherwise, they are not. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
                        e3 = Edge.ByVertices([Edge.StartVertex(e1), Edge.StartVertex(e2)], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e4 = Edge.ByVertices([Edge.EndVertex(e1), Edge.EndVertex(e2)], tolerance=tolerance, silent=True)
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e4], tolerance=tolerance), tolerance=tolerance))
                        except:
                            pass
                    try:
                        e4 = Edge.ByVertices([Edge.EndVertex(e1), Edge.EndVertex(e2)], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e3 = Edge.ByVertices([Edge.StartVertex(e1), Edge.StartVertex(e2)], tolerance=tolerance, silent=True)
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3], tolerance=tolerance), tolerance=tolerance))
                        except:
                            pass
                    if e3 and e4:
                        e5 = Edge.ByVertices([Edge.StartVertex(e1), Edge.EndVertex(e2)], tolerance=tolerance, silent=True)
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e5, e4], tolerance=tolerance), tolerance=tolerance))
                        faces.append(Face.ByWire(Wire.ByEdges([e2, e5, e3], tolerance=tolerance), tolerance=tolerance))
            else:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([Edge.StartVertex(e1), Edge.StartVertex(e2)], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e4 = Edge.ByVertices([Edge.EndVertex(e1), Edge.EndVertex(e2)], tolerance=tolerance, silent=True)
                        except:
                            pass
                    try:
                        e4 = Edge.ByVertices([Edge.EndVertex(e1), Edge.EndVertex(e2)], tolerance=tolerance, silent=True)
                    except:
                        try:
                            e3 = Edge.ByVertices([Edge.StartVertex(e1), Edge.StartVertex(e2)], tolerance=tolerance, silent=True)
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
            If set to True, the last wire in the cluster of input wires will be connected to the first wire in the cluster of input wires. Default is False.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            The location of the origin of the cylinder. Default is None which results in the cylinder being placed at (0, 0, 0).
        radius : float , optional
            The radius of the capsule. Default is 0.25.
        height : float , optional
            The height of the capsule. Default is 1.
        uSides : int , optional
            The number of circle segments of the capsule. Default is 16.
        vSidesEnds : int , optional
            The number of vertical segments of the end hemispheres. Default is 8.
        vSidesMiddle : int , optional
            The number of vertical segments of the middle cylinder. Default is 1.
        direction : list , optional
            The vector representing the up direction of the capsule. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the capsule. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "bottom".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def CHS(origin= None, radius: float = 1.0, height: float = 1.0, thickness: float = 0.25, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a circular hollow section (CHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the CHS. Default is None which results in the CHS being placed at (0, 0, 0).
        radius : float , optional
            The outer radius of the CHS. Default is 1.0.
        thickness : float , optional
            The thickness of the CHS. Default is 0.25.
        height : float , optional
            The height of the CHS. Default is 1.0.
        sides : int , optional
            The desired number of sides of the CSH. Default is 16.
        direction : list , optional
            The vector representing the up direction of the RHS. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the RHS. This can be "center", "bottom", "top", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if 2*thickness >= radius:
            if not silent:
                print("Cell.CHS - Error: Twice the thickness value is larger than or equal to the width value. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        bottom_face = Face.CHS(origin = Vertex.Origin(),radius=radius, thickness=thickness, sides=sides, direction=[0,0,1], placement="center", tolerance=tolerance, silent=silent)
        return_cell = Cell.ByThickenedFace(bottom_face, thickness=height, bothSides=True, reverse=False,
                            planarize = False, tolerance=tolerance, silent=silent)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = height*0.5
        elif placement.lower() == "top":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = radius
            yOffset = radius
            zOffset = height*0.5
        elif placement.lower() == "upperleft":
            xOffset = radius
            yOffset = -radius
            zOffset = -height*0.5
        elif placement.lower() == "lowerright":
            xOffset = -radius
            yOffset = radius
            zOffset = height*0.5
        elif placement.lower() == "upperright":
            xOffset = -radius
            yOffset = -radius
            zOffset = -height*0.5
        return_cell = Topology.Translate(return_cell, x=xOffset, y=yOffset, z=zOffset)
        return_cell = Topology.Place(return_cell, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_cell = Topology.Orient(return_cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_cell

    @staticmethod
    def Compactness(cell, reference = "sphere", mantissa: int = 6) -> float:
        """
        Returns the compactness measure of the input cell. If the reference is "sphere", this is also known as 'sphericity' (https://en.wikipedia.org/wiki/Sphericity).

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        reference : str , optional
            The desired reference to which to compare this compactness. The options are "sphere" and "cube". It is case insensitive. Default is "sphere".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

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
            The location of the origin of the cone. Default is None which results in the cone being placed at (0, 0, 0).
        baseRadius : float , optional
            The radius of the base circle of the cone. Default is 0.5.
        topRadius : float , optional
            The radius of the top circle of the cone. Default is 0.
        height : float , optional
            The height of the cone. Default is 1.
        uSides : int , optional
            The number of circle segments of the cylinder. Default is 16.
        vSides : int , optional
            The number of vertical segments of the cylinder. Default is 1.
        direction : list , optional
            The vector representing the up direction of the cone. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the cone. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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

        baseZ = Vertex.Z(origin) + zOffset
        topZ = Vertex.Z(origin) + zOffset + height
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
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        int
            Returns 0 if the vertex is inside, 1 if it is on the boundary of, and 2 if it is outside the input cell.

        """
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(cell, "Cell"):
            print("Cell.ContainmentStatus - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Cell.ContainmentStatus - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
            return None
        
        # topologic.CellUtility.Contains does not seem to respect the input tolerance. Thus we need to send eight additional vertices
        # to check if any are contained and take the average result.
        
        test_vertices = [vertex]
        if tolerance > 0:
            test_cell = Cell.Prism(origin=vertex, width=tolerance*2, length=tolerance*2, height=tolerance*2, tolerance=tolerance)
            test_vertices.extend(Topology.Vertices(test_cell))
        try:
            av_results = []
            for v in test_vertices:
                
                result  = topologic.CellUtility.Contains(cell, v, tolerance) # Hook to Core
                if result == 0:
                    status = 0
                elif result == 1:
                    status = 1
                else:
                    status = 2
                av_results.append(status)
            return min(av_results)
            
        except:
            print("Cell.ContainmentStatus - Error: Could not determine containment status. Returning None.")
            return None
    
    @staticmethod
    def CrossShape(origin=None,
            width=1,
            length=1,
            height=1,
            a=0.25,
            b=0.25,
            c=None,
            d=None,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a Cross-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the T-shape. Default is None which results in the Cross-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the Cross-shape. Default is 1.0.
        length : float , optional
            The overall length of the Cross-shape. Default is 1.0.
        height : float , optional
            The overall height of the C-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the Cross-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the Cross-shape. Default is 0.25.
        c : float , optional
            The distance of the vertical symmetry axis measured from the left side of the Cross-shape. Default is None which results in the Cross-shape being symmetrical on the Y-axis.
        d : float , optional
            The distance of the horizontal symmetry axis measured from the bottom side of the Cross-shape. Default is None which results in the Cross-shape being symmetrical on the X-axis.
        direction : list , optional
            The vector representing the up direction of the Cross-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the Cross-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created Cross-shape cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Cell.CrossShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Cell.CrossShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Cell.CrossShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Cell.CrossShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if c == None:
            c = width/2
        if d == None:
            d = length/2
        if not isinstance(c, int) and not isinstance(c, float):
            if not silent:
                print("Cell.CrossShape - Error: The c input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(d, int) and not isinstance(d, float):
            if not silent:
                print("Cell.CrossShape - Error: The d input parameter is not a valid number. Returning None.")
        if width <= tolerance:
            if not silent:
                print("Cell.CrossShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Cell.CrossShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Cell.CrossShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Cell.CrossShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if c <= tolerance:
            if not silent:
                print("Cell.CrossShape - Error: The c input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if d <= tolerance:
            if not silent:
                print("Cell.CrossShape - Error: The d input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance*2):
            if not silent:
                print("Cell.CrossShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance*2):
            if not silent:
                print("Cell.CrossShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if c <= (tolerance + a/2):
            if not silent:
                print("Cell.CrossShape - Error: The c input parameter must be more than half the a input parameter. Returning None.")
            return None
        if d <= (tolerance + b/2):
            if not silent:
                print("Cell.CrossShape - Error: The c input parameter must be more than half the b input parameter. Returning None.")
            return None
        if c >= (width - tolerance - a/2):
            if not silent:
                print("Cell.CrossShape - Error: The c input parameter must be less than the width minus half the a input parameter. Returning None.")
            return None
        if d >= (length - tolerance - b/2):
            if not silent:
                print("Cell.CrossShape - Error: The c input parameter must be less than the width minus half the b input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Cell.CrossShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Cell.CrossShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Cell.CrossShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        cross_shape_face = Face.CrossShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   c=c,
                                   d=d,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=direction,
                                   placement=placement,
                                   tolerance=tolerance,
                                   silent=silent)
        return_cell = Cell.ByThickenedFace(cross_shape_face, thickness=height, bothSides=True, reverse=False,
                            tolerance=tolerance, silent=silent)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = height*0.5
        elif placement.lower() == "top":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        return_cell = Topology.Translate(return_cell, x=xOffset, y=yOffset, z=zOffset)
        return_cell = Topology.Place(return_cell, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_cell = Topology.Orient(return_cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_cell
    
    @staticmethod
    def CShape(origin=None,
            width=1,
            length=1,
            height=1,
            wSides=1,
            a=0.25,
            b=0.25,
            c=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            mantissa=6,
            tolerance=0.0001,
            silent=False):
        """
        Creates a C-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the C-shape. Default is None which results in the C-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the C-shape. Default is 1.0.
        length : float , optional
            The overall length of the C-shape. Default is 1.0.
        height : float , optional
            The overall height of the C-shape. Default is 1.0.
        wSides : int , optional
            The desired number of sides along the Z-axis. Default is 1.
        a : float , optional
            The hortizontal thickness of the vertical arm of the C-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the bottom horizontal arm of the C-shape. Default is 0.25.
        c : float , optional
            The vertical thickness of the top horizontal arm of the C-shape. Default is 0.25.
        flipHorizontal : bool , optional
            if set to True, the shape is flipped horizontally. Default is False.
        flipVertical : bool , optional
            if set to True, the shape is flipped vertically. Default is False.
        direction : list , optional
            The vector representing the up direction of the C-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the C-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        mantissa: int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created C-shape cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Cell.CShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Cell.CShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(height, int) and not isinstance(height, float):
            if not silent:
                print("Cell.CShape - Error: The height input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Cell.CShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Cell.CShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Cell.CShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Cell.CShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if height <= tolerance:
            if not silent:
                print("Cell.CShape - Error: The height input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Cell.CShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Cell.CShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if c <= tolerance:
            if not silent:
                print("Cell.CShape - Error: The c input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Cell.CShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b+c >= (length - tolerance):
            if not silent:
                print("Cell.CShape - Error: The b and c input parameters must add to less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Cell.CShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Cell.CShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Cell.CShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None        
        c_shape_wire = Wire.CShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   c=c,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=[0,0,1],
                                   placement="center",
                                   tolerance=tolerance,
                                   silent=silent)
        distance = height/wSides
        wires = [c_shape_wire]
        for i in range(wSides):
            c_shape_wire = Topology.Translate(c_shape_wire, 0, 0, distance)
            wires.append(c_shape_wire)
        return_cell = Cell.ByWires(wires, triangulate=False, mantissa=mantissa, tolerance=tolerance, silent=silent)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = height*0.5
        elif placement.lower() == "top":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        return_cell = Topology.Translate(return_cell, x=xOffset, y=yOffset, z=zOffset)
        return_cell = Topology.Place(return_cell, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_cell = Topology.Orient(return_cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_cell
    
    @staticmethod
    def Cube(origin = None,
            size: float = 1,
            uSides: int = 1, vSides: int = 1, wSides: int = 1,
            direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates a cube.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the cube. Default is None which results in the cube being placed at (0, 0, 0).
        size : float , optional
            The size of the cube. Default is 1.
        uSides : int , optional
            The number of sides along the width. Default is 1.
        vSides : int , optional
            The number of sides along the length. Default is 1.
        wSides : int , optional
            The number of sides along the height. Default is 1.
        direction : list , optional
            The vector representing the up direction of the cube. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the cube. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.Cell
            The created cube.

        """
        return Cell.Prism(origin=origin, width=size, length=size, height=size,
                          uSides=uSides, vSides=vSides, wSides=wSides,
                          direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def Cylinder(origin = None, radius: float = 0.5, height: float = 1, uSides: int = 16, vSides: int = 1, direction: list = [0, 0, 1],
                     placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a cylinder.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the cylinder. Default is None which results in the cylinder being placed at (0, 0, 0).
        radius : float , optional
            The radius of the cylinder. Default is 0.5.
        height : float , optional
            The height of the cylinder. Default is 1.
        uSides : int , optional
            The number of circle segments of the cylinder. Default is 16.
        vSides : int , optional
            The number of vertical segments of the cylinder. Default is 1.
        direction : list , optional
            The vector representing the up direction of the cylinder. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the cylinder. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "bottom".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
        cylinder = Cell.ByThickenedFace(face=baseFace, thickness=height, bothSides=False, reverse=False, tolerance=tolerance)
        if vSides > 1:
            cutting_planes = []
            baseX = Vertex.X(origin, mantissa=mantissa) + xOffset
            baseY = Vertex.Y(origin, mantissa=mantissa) + yOffset
            size = radius*3
            for i in range(1, vSides):
                baseZ = Vertex.Z(origin) + zOffset + float(height)/float(vSides)*i
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
            The threshold tilt angle in degrees to determine if a face is vertical, horizontal, or tilted. The tilt angle is measured from the nearest cardinal direction. Default is 10.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            zList.append(Vertex.Z(Topology.Centroid(f)))
        zMin = min(zList)
        zMax = max(zList)
        up = [0, 0, 1]
        for aFace in faces:
            aCode = angleCode(aFace, up, tiltAngle)

            if aCode == 0:
                verticalFaces.append(aFace)
                verticalApertures += getApertures(aFace)
            elif aCode == 1:
                if abs(Vertex.Z(Topology.Centroid(aFace)) - zMin) <= tolerance:
                    bottomHorizontalFaces.append(aFace)
                    bottomHorizontalApertures += getApertures(aFace)
                else:
                    topHorizontalFaces.append(aFace)
                    topHorizontalApertures += getApertures(aFace)
            elif aCode == 2:
                if abs(Vertex.Z(Topology.Centroid(aFace)) - zMax) <= tolerance:
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
            The origin location of the dodecahedron. Default is None which results in the dodecahedron being placed at (0, 0, 0).
        radius : float , optional
            The radius of the dodecahedron's circumscribed sphere. Default is 0.5.
        direction : list , optional
            The vector representing the up direction of the dodecahedron. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the dodecahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
        # Make sure the distance from the origin to the vertices is equal to the radius.
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
            The origin location of the sphere. Default is None which results in the egg-shaped cell being placed at (0, 0, 0).
        height : float , optional
            The desired height of of the egg-shaped cell. Default is 1.0.
        uSides : int , optional
            The desired number of sides along the longitude of the egg-shaped cell. Default is 16.
        vSides : int , optional
            The desired number of sides along the latitude of the egg-shaped cell. Default is 8.
        direction : list , optional
            The vector representing the up direction of the egg-shaped cell. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the egg-shaped cell. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            bb = Topology.BoundingBox(egg)
            d = Topology.Dictionary(bb)
            width = Dictionary.ValueAtKey(d, 'width')
            length = Dictionary.ValueAtKey(d, 'length')
            egg = Topology.Translate(egg, width*0.5, length*0.5, height*0.5)
        egg = Topology.Orient(egg, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        egg = Topology.Place(egg, originA=Vertex.Origin(), originB=origin)
        return egg
    
    @staticmethod
    def ExternalBoundary(cell, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the external boundary of the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Shell
            The external boundary of the input cell.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        from topologicpy.Shell import Shell

        if not Topology.IsInstance(cell, "Cell"):
            if not silent:
                print("Cell.ExternalBoundary - Error: The input cell parameter is not a valid topologic cell. Returning None.")
            return None
        
        shells = Topology.Shells(cell)
        closed_shells = []
        temp_cells = []
        for shell in shells:
            try:
                temp_cell = Cell.ByShell(shell, silent=True)
                if Topology.IsInstance(temp_cell, "cell"):
                    closed_shells.append(shell)
                    temp_cells.append(temp_cell)
            except:
                pass
        volumes = [Cell.Volume(c) for c in temp_cells]
        closed_shells = Helper.Sort(closed_shells, volumes, silent=silent)
        return closed_shells[-1]
    
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
            The location of the origin of the hyperboloid. Default is None which results in the hyperboloid being placed at (0, 0, 0).
        baseRadius : float , optional
            The radius of the base circle of the hyperboloid. Default is 0.5.
        topRadius : float , optional
            The radius of the top circle of the hyperboloid. Default is 0.5.
        height : float , optional
            The height of the cone. Default is 1.
        sides : int , optional
            The number of sides of the cone. Default is 24.
        direction : list , optional
            The vector representing the up direction of the hyperboloid. Default is [0, 0, 1].
        twist : float , optional
            The angle to twist the base cylinder. Default is 60.
        placement : str , optional
            The description of the placement of the origin of the hyperboloid. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
        baseZ = Vertex.Z(w_origin) + zOffset
        topZ = Vertex.Z(w_origin) + zOffset + height
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
            The origin location of the icosahedron. Default is None which results in the icosahedron being placed at (0, 0, 0).
        radius : float , optional
            The radius of the icosahedron's circumscribed sphere. Default is 0.5.
        direction : list , optional
            The vector representing the up direction of the icosahedron. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the icosahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def IShape(origin=None,
            width: float = 1,
            length: float = 1,
            height: float = 1,
            wSides: int = 1,
            a: float = 0.25,
            b: float = 0.25,
            c: float = 0.25,
            flipHorizontal: bool = False,
            flipVertical: bool = False,
            direction: list = [0,0,1],
            placement: str = "center",
            mantissa: int = 6,
            tolerance: float = 0.0001,
            silent: bool = False):
        """
        Creates an I-shape cell.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the I-shape. Default is None which results in the I-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the I-shape. Default is 1.0.
        length : float , optional
            The overall length of the I-shape. Default is 1.0.
        height : float , optional
            The overall height of the I-shape. Default is 1.0.
        wSides : int , optional
            The desired number of sides along the Z-Axis. Default is 1.
        a : float , optional
            The hortizontal thickness of the central vertical arm of the I-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the bottom horizontal arm of the I-shape. Default is 0.25.
        c : float , optional
            The vertical thickness of the top horizontal arm of the I-shape. Default is 0.25.
        flipHorizontal : bool , optional
            if set to True, the shape is flipped horizontally. Default is False.
        flipVertical : bool , optional
            if set to True, the shape is flipped vertically. Default is False.
        direction : list , optional
            The vector representing the up direction of the I-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the I-shape. This can be "center", "bottom", "top", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created I-shape cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Cell.IShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Cell.IShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Cell.IShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Cell.IShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Cell.IShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Cell.IShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Cell.IShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Cell.IShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Cell.IShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b+c >= (length - tolerance):
            if not silent:
                print("Cell.IShape - Error: The b and c input parameters must add to less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Cell.IShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Cell.IShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Cell.IShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        i_shape_wire = Wire.IShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   c=c,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=[0,0,1],
                                   placement="center",
                                   tolerance=tolerance,
                                   silent=silent)
        distance = height/wSides
        wires = [i_shape_wire]
        for i in range(wSides):
            i_shape_wire = Topology.Translate(i_shape_wire, 0, 0, distance)
            wires.append(i_shape_wire)
        return_cell = Cell.ByWires(wires, triangulate=False, mantissa=mantissa, tolerance=tolerance, silent=silent)
        # move down to center
        return_cell = Topology.Translate(return_cell, 0, 0, -height*0.5)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = height*0.5
        elif placement.lower() == "top":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        return_cell = Topology.Translate(return_cell, x=xOffset, y=yOffset, z=zOffset)
        return_cell = Topology.Place(return_cell, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_cell = Topology.Orient(return_cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_cell
        
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
            The desired tolerance. Default is 0.0001.

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
    def LShape(origin=None,
            width=1,
            length=1,
            height=1,
            wSides=1,
            a=0.25,
            b=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            mantissa=6,
            tolerance=0.0001,
            silent=False):
        """
        Creates an L-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the L-shape. Default is None which results in the L-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the L-shape. Default is 1.0.
        length : float , optional
            The overall length of the L-shape. Default is 1.0.
        height : float , optional
            The overall height of the L-shape. Default is 1.0.
        wSides : int , optional
            The desired number of sides along the Z-axis. Default is 1.
        a : float , optional
            The hortizontal thickness of the vertical arm of the L-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the L-shape. Default is 0.25.
        flipHorizontal : bool , optional
            if set to True, the shape is flipped horizontally. Default is False.
        flipVertical : bool , optional
            if set to True, the shape is flipped vertically. Default is False.
        direction : list , optional
            The vector representing the up direction of the L-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the L-shape. This can be "center", "bottom", "top", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        manitssa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created L-shape cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Cell.LShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Cell.LShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(height, int) and not isinstance(height, float):
            if not silent:
                print("Cell.LShape - Error: The height input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Cell.LShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Cell.LShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Cell.LShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Cell.LShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if height <= tolerance:
            if not silent:
                print("Cell.LShape - Error: The height input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Cell.LShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Cell.LShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Cell.LShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance):
            if not silent:
                print("Cell.LShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Cell.LShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Cell.LShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Cell.LShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        l_shape_wire = Wire.LShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=[0,0,1],
                                   placement="center",
                                   tolerance=tolerance,
                                   silent=silent)
        distance = height/wSides
        wires = [l_shape_wire]
        for i in range(wSides):
            l_shape_wire = Topology.Translate(l_shape_wire, 0, 0, distance)
            wires.append(l_shape_wire)
        return_cell = Cell.ByWires(wires, triangulate=False, mantissa=mantissa, tolerance=tolerance, silent=silent)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = height*0.5
        elif placement.lower() == "top":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        return_cell = Topology.Translate(return_cell, x=xOffset, y=yOffset, z=zOffset)
        return_cell = Topology.Place(return_cell, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_cell = Topology.Orient(return_cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_cell
    
    @staticmethod
    def Noperthedron(origin= None, radius: float = 0.5, direction: list = [0, 0, 1],
                   placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a Noperthedron. A noperthedron is a convex polyhedron without Rupert's property. See: https://arxiv.org/pdf/2508.18475

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the sphere. Default is None which results in the sphere being placed at (0, 0, 0).
        radius : float , optional
            The radius of the sphere. Default is 0.5.
        uSides : int , optional
            The number of sides along the longitude of the sphere. Default is 16.
        vSides : int , optional
            The number of sides along the latitude of the sphere. Default is 8.
        direction : list , optional
            The vector representing the up direction of the sphere. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the sphere. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If set to True, suppresses warning and error messages. Default is False.
        

        Returns
        -------
        topologic_core.Cell
            The created Noperthedron polyhedron.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        from math import cos, sin, pi

        # Validate inputs
        if radius <= 0:
            if not silent:
                print("Cell.Polyhedron - Error: radius must be > 0. Returning None.")
            return None

        # Center
        if origin is None:
            origin = Vertex.ByCoordinates(0, 0, 0)
        
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Cell.Noperthedron - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None
        

    # defaults:
        scale = 1.0
        rotateZ_radians = 0.0
        dedup_dp = 14

        # Correct seeds per paper:
        C1 = (152024884/259375205, 0.0, 210152163/259375205)
        DEN = 10_000_000_000  # 10^10
        C2 = (6632738028/DEN, 6106948881/DEN, 3980949609/DEN)
        C3 = (8193990033/DEN, 5298215096/DEN, 1230614493/DEN)
        seeds = [C1, C2, C3]

        def rotz(p, a):
            x, y, z = p
            ca, sa = cos(a), sin(a)
            return (ca*x - sa*y, sa*x + ca*y, z)

        # Generate 30-element orbit for each seed: (-I)^ℓ · Rz(2πk/15)
        pts = []
        for (x, y, z) in seeds:
            x, y, z = rotz((x, y, z), rotateZ_radians)
            for k in range(15):
                th = 2*pi*k/15
                X, Y, Z = rotz((x, y, z), th)
                pts.append(( X,  Y,  Z))
                pts.append((-X, -Y, -Z))

        # Deduplicate robustly (should keep all 90)
        seen, verts = set(), []
        for p in pts:
            key = (round(p[0], dedup_dp), round(p[1], dedup_dp), round(p[2], dedup_dp))
            if key in seen: 
                continue
            seen.add(key)
            verts.append(Vertex.ByCoordinates(p[0]*scale, p[1]*scale, p[2]*scale))

        hull = Topology.ConvexHull(Cluster.ByTopologies(verts))
        if not Topology.IsInstance(hull, "cell"):
            if not silent:
                print("Cell.Noperthedron - Error: could not create the cell. Returning None.")
            return None

        # Cleanup top and bottom
        cell_dict = Cell.Decompose(hull)
        h_faces = Helper.Flatten([cell_dict['topHorizontalFaces'] + cell_dict['bottomHorizontalFaces']])
        other_faces = [cell_dict['verticalFaces']+ cell_dict['inclinedFaces']]
        caps_cluster = Topology.SelfMerge(Cluster.ByTopologies(h_faces))
        caps_cluster = Topology.RemoveCoplanarFaces(caps_cluster)
        all_faces = other_faces + Topology.Faces(caps_cluster)
        all_faces = Helper.Flatten(all_faces)
        
        # Sew faces into a cell
        try:
            tol = tolerance
            for i in range(3):
                hull = Cell.ByFaces(all_faces, tolerance=tol)
                if Topology.IsInstance(hull, "Cell"):
                    break
                tol = tol*10
            hull = Topology.Scale(hull, origin=Topology.Centroid(hull), x=1/1.949062, y=1/1.949062, z=1/1.949062)
            # Scale by the radius amount:
            hull = Topology.Scale(hull, origin=Topology.Centroid(hull), x=radius*2, y=radius*2, z=radius*2)
        except Exception:
            if not silent:
                print("Cell.Noperthedron - Error: could not create the cell. Returning None.")
            return None
        if placement.lower() != "center":
            bb = Topology.BoundingBox(hull)
            d = Topology.Dictionary(bb)
            width = Dictionary.ValueAtKey(d, "width")
            length = Dictionary.ValueAtKey(d, "length")
            height = Dictionary.ValueAtKey(d, "height")
        if placement.lower() == "bottom":
            hull = Topology.Translate(hull, 0, 0, height*0.5)
        elif placement.lower() == "lowerleft":
            hull = Topology.Translate(hull, width*0.5, length*0.5, height*0.5)
        if direction != [0,0,1]:
            hull = Topology.Orient(hull, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        if Vertex.Coordinates(origin) != [0,0,0]:
            hull = Topology.Place(hull, originA=Vertex.Origin(), originB=origin)
        return hull

    @staticmethod
    def Octahedron(origin= None, radius: float = 0.5,
                  direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates an octahedron. See https://en.wikipedia.org/wiki/Octahedron.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the octahedron. Default is None which results in the octahedron being placed at (0, 0, 0).
        radius : float , optional
            The radius of the octahedron's circumscribed sphere. Default is 0.5.
        direction : list , optional
            The vector representing the up direction of the octahedron. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the octahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
            The origin location of the parabolic surface. Default is None which results in the parabolic surface being placed at (0, 0, 0).
        focalLength : float , optional
            The focal length of the parabola. Default is 0.125.
        width : float , optional
            The width of the parabolic surface. Default is 1.
        length : float , optional
            The length of the parabolic surface. Default is 1.
        height : float , optional
            The additional height of the parabolic surface. Please note this is not the height from the spring point to the apex. It is in addition to that to form a base. Default is 0.
        uSides : int , optional
            The number of sides along the width. Default is 16.
        vSides : int , optional
            The number of sides along the length. Default is 16.
        direction : list , optional
            The vector representing the up direction of the parabolic surface. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the parabolic surface. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
        If set to True, error and warning messages are suppressed. Default is False.
        
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
            The profile of the pipe. It is assumed that the profile is in the XY plane. If set to None, a circle of radius 0.5 will be used. Default is None.
        radius : float , optional
            The radius of the pipe. Default is 0.5.
        sides : int , optional
            The number of sides of the pipe. Default is 16.
        startOffset : float , optional
            The offset distance from the start vertex of the centerline edge. Default is 0.
        endOffset : float , optional
            The offset distance from the end vertex of the centerline edge. Default is 0.
        endcapA, optional
            The topology to place at the start vertex of the centerline edge. The positive Z direction of the end cap will be oriented in the direction of the centerline edge.
        endcapB, optional
            The topology to place at the end vertex of the centerline edge. The positive Z direction of the end cap will be oriented in the inverse direction of the centerline edge.
        mantissa : int , optional
            The desired length of the mantissa. Default is 6
        
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
            baseWire = Topology.Translate(profile, 0 , 0, Vertex.Z(sv))
            topWire = Topology.Translate(profile, 0 , 0, Vertex.Z(sv)+dist)
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
            origin = Edge.StartVertex(edge)
            x1 = Vertex.X(origin, mantissa=mantissa)
            y1 = Vertex.Y(origin, mantissa=mantissa)
            z1 = Vertex.Z(origin, mantissa=mantissa)
            x2 = Vertex.X(Edge.EndVertex(edge), mantissa=mantissa)
            y2 = Vertex.Y(Edge.EndVertex(edge), mantissa=mantissa)
            z2 = Vertex.Z(Edge.EndVertex(edge), mantissa=mantissa)
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
            origin = Edge.EndVertex(edge)
            x1 = Vertex.X(origin, mantissa=mantissa)
            y1 = Vertex.Y(origin, mantissa=mantissa)
            z1 = Vertex.Z(origin, mantissa=mantissa)
            x2 = Vertex.X(Edge.StartVertex(edge), mantissa=mantissa)
            y2 = Vertex.Y(Edge.StartVertex(edge), mantissa=mantissa)
            z2 = Vertex.Z(Edge.StartVertex(edge), mantissa=mantissa)
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
            The origin location of the prism. Default is None which results in the prism being placed at (0, 0, 0).
        width : float , optional
            The width of the prism. Default is 1.
        length : float , optional
            The length of the prism. Default is 1.
        height : float , optional
            The height of the prism.
        uSides : int , optional
            The number of sides along the width. Default is 1.
        vSides : int , optional
            The number of sides along the length. Default is 1.
        wSides : int , optional
            The number of sides along the height. Default is 1.
        direction : list , optional
            The vector representing the up direction of the prism. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the prism. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
            vRect = Wire.Rectangle(origin=origin, length=height*1.2, width=width*1.2, direction=[0, 1, 0], placement="center")
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
        
        vb1 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)-width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)+length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)
        vb2 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)+length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)
        vb3 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)-length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)
        vb4 = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)-width*0.5+xOffset,Vertex.Y(origin, mantissa=mantissa)-length*0.5+yOffset,Vertex.Z(origin, mantissa=mantissa)+zOffset)

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire, tolerance=tolerance)

        prism = Cell.ByThickenedFace(baseFace, thickness=height, bothSides = False, reverse=True)

        if uSides > 1 or vSides > 1 or wSides > 1:
            prism = sliceCell(prism, width, length, height, uSides, vSides, wSides)
        prism = Topology.Orient(prism, origin=origin, dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        return prism

    @staticmethod
    def RemoveCollinearEdges(cell, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Removes any collinear edges in the input cell.

        Parameters
        ----------
        cell : topologic_core.Cell
            The input cell.
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell without any collinear edges.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import inspect
        
        if not Topology.IsInstance(cell, "Cell"):
            if not silent:
                print("Cell.RemoveCollinearEdges - Error: The input cell parameter is not a valid cell. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        faces = Cell.Faces(cell)
        clean_faces = []
        for face in faces:
            clean_faces.append(Face.RemoveCollinearEdges(face, angTolerance=angTolerance, tolerance=tolerance, silent=silent))
        return Cell.ByFaces(clean_faces, tolerance=tolerance)
    
    @staticmethod
    def RHS(origin= None, width: float = 1.0, length: float = 1.0, height: float = 1.0, thickness: float = 0.25, outerFillet: float = 0.0, innerFillet: float = 0.0, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a rectangluar hollow section (RHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the RHS. Default is None which results in the RHS being placed at (0, 0, 0).
        width : float , optional
            The width of the RHS. Default is 1.0.
        length : float , optional
            The length of the RHS. Default is 1.0.
        thickness : float , optional
            The thickness of the RHS. Default is 0.25.
        height : float , optional
            The height of the RHS. Default is 1.0.
        outerFillet : float , optional
            The outer fillet multiplication factor based on the thickness (e.g. 1t). Default is 0.
        innerFillet : float , optional
            The inner fillet multiplication factor based on the thickness (e.g. 1.5t). Default is 0.
        sides : int , optional
            The desired number of sides of the fillets. Default is 16.
        direction : list , optional
            The vector representing the up direction of the RHS. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the RHS. This can be "center", "bottom", "top", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if 2*thickness >= width:
            if not silent:
                print("Cell.RHS - Error: Twice the thickness value is larger than or equal to the width value. Returning None.")
            return None
        if 2*thickness >= width:
            if not silent:
                print("Cell.RHS - Error: Twice the thickness value is larger than or equal to the length value. Returning None.")
            return None
        outer_dimension = min(width, length)
        fillet_dimension = 2*outerFillet*thickness
        if  fillet_dimension > outer_dimension:
            if not silent:
                print("Cell.RHS = Error: The outer fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        inner_dimension = min(width, length) - 2*thickness
        fillet_dimension = 2*innerFillet*thickness
        if fillet_dimension > inner_dimension:
            if not silent:
                print("Cell.RHS = Error: The inner fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        bottom_face = Face.RHS(origin = Vertex.Origin(), width=width, length=length, thickness=thickness, outerFillet=outerFillet, innerFillet=innerFillet, sides=sides, direction=[0,0,1], placement="center", tolerance=tolerance, silent=silent)
        return_cell = Cell.ByThickenedFace(bottom_face, thickness=height, bothSides=True, reverse=False,
                            tolerance=tolerance, silent=silent)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = height*0.5
        elif placement.lower() == "top":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        return_cell = Topology.Translate(return_cell, x=xOffset, y=yOffset, z=zOffset)
        return_cell = Topology.Place(return_cell, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_cell = Topology.Orient(return_cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_cell

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
            The desired angle in degrees of the roof. Default is 45.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for distance from plane). Default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. Default is 0.001. (This is set to a larger number as it was found to work better)

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
            The desired tolerance. Default is 0.0001.

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
    def SHS(origin= None, size: float = 1.0, height: float = 1.0, thickness: float = 0.25, outerFillet: float = 0.0, innerFillet: float = 0.0, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a square hollow section (SHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the SHS. Default is None which results in the SHS being placed at (0, 0, 0).
        size : float , optional
            The size of the SHS. Default is 1.0.
        length : float , optional
            The length of the SHS. Default is 1.0.
        thickness : float , optional
            The thickness of the SHS. Default is 0.25.
        height : float , optional
            The height of the SHS. Default is 1.0.
        outerFillet : float , optional
            The outer fillet multiplication factor based on the thickness (e.g. 1t). Default is 0.
        innerFillet : float , optional
            The inner fillet multiplication factor based on the thickness (e.g. 1.5t). Default is 0.
        sides : int , optional
            The desired number of sides of the fillets. Default is 16.
        direction : list , optional
            The vector representing the up direction of the SHS. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the RHS. This can be "center", "bottom", "top", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if 2*thickness >= size:
            if not silent:
                print("Cell.SHS - Error: Twice the thickness value is larger than or equal to the size value. Returning None.")
            return None
        fillet_dimension = 2*outerFillet*thickness
        if  fillet_dimension > size:
            if not silent:
                print("Cell.SHS = Error: The outer fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        inner_dimension = size - 2*thickness
        fillet_dimension = 2*innerFillet*thickness
        if fillet_dimension > inner_dimension:
            if not silent:
                print("Cell.SHS = Error: The inner fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        return Cell.RHS(origin= origin,
                        width = size,
                        length = size,
                        height = height,
                        thickness = thickness,
                        outerFillet = outerFillet,
                        innerFillet = innerFillet,
                        sides = sides,
                        direction = direction,
                        placement = placement,
                        tolerance = tolerance,
                        silent = silent)
    
    @staticmethod
    def Sphere(origin= None, radius: float = 0.5, uSides: int = 16, vSides: int = 8, direction: list = [0, 0, 1],
                   placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates an approximation of a sphere using a UV grid of triangular faces.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the sphere. Default is None which results in the sphere being placed at (0, 0, 0).
        radius : float , optional
            The radius of the sphere. Default is 0.5.
        uSides : int , optional
            The number of sides along the longitude of the sphere. Default is 16.
        vSides : int , optional
            The number of sides along the latitude of the sphere. Default is 8.
        direction : list , optional
            The vector representing the up direction of the sphere. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the sphere. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If set to True, suppresses warning and error messages. Default is False.
        

        Returns
        -------
        topologic_core.Cell
            The created sphere.

        """
    
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        # Validate inputs
        if radius <= 0 or uSides < 3 or vSides < 2:
            if not silent:
                print("Cell.Sphere - Error: radius must be > 0, uSides >= 3, vSides >= 2. Returning None.")
            return None

        # Center
        if origin is None:
            origin = Vertex.ByCoordinates(0, 0, 0)
        ox = Vertex.X(origin)
        oy = Vertex.Y(origin)
        oz = Vertex.Z(origin)

        # Poles
        top_pole = Vertex.ByCoordinates(ox, oy, oz + radius)
        bottom_pole = Vertex.ByCoordinates(ox, oy, oz - radius)

        # Latitude rings (exclude poles)
        rings = []  # list of list[Vertex]
        for vi in range(1, vSides):
            phi = math.pi * vi / vSides  # 0..pi
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            ring = []
            for ui in range(uSides):
                theta = 2.0 * math.pi * ui / uSides
                x = ox + radius * sin_phi * math.cos(theta)
                y = oy + radius * sin_phi * math.sin(theta)
                z = oz + radius * cos_phi
                ring.append(Vertex.ByCoordinates(x, y, z))
            rings.append(ring)

        faces = []

        # Top cap: triangles from top pole to first ring
        first_ring = rings[0]
        for u in range(uSides):
            v1 = first_ring[u]
            v2 = first_ring[(u + 1) % uSides]
            f = Face.ByVertices([top_pole, v1, v2], tolerance=tolerance)
            if f:
                faces.append(f)

        # Middle bands: split quads into two triangles
        for i in range(len(rings) - 1):
            curr = rings[i]
            nxt = rings[i + 1]
            for u in range(uSides):
                a = curr[u]
                b = nxt[u]
                c = nxt[(u + 1) % uSides]
                d = curr[(u + 1) % uSides]
                f1 = Face.ByVertices([a, b, c], tolerance=tolerance)
                if f1:
                    faces.append(f1)
                f2 = Face.ByVertices([a, c, d], tolerance=tolerance)
                if f2:
                    faces.append(f2)

        # Bottom cap: triangles from last ring to bottom pole
        last_ring = rings[-1]
        for u in range(uSides):
            v1 = last_ring[(u + 1) % uSides]
            v2 = last_ring[u]
            f = Face.ByVertices([bottom_pole, v1, v2], tolerance=tolerance)
            if f:
                faces.append(f)

        # Sew faces into a shell
        sphere = None
        try:
            sphere = Cell.ByFaces(faces, tolerance=tolerance)
        except Exception:
            if not silent:
                print("Cell.Sphere - Error: could not create a sphere. Returning None.")
            return None
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
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        area : float
            The surface area of the input cell.

        """
        return Cell.Area(cell=cell, mantissa=mantissa)
    
    @staticmethod
    def Tetrahedron(origin = None, length: float = 1, depth: int = 1, direction=[0,0,1], placement="center", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a recursive tetrahedron cell.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the tetrahedron. Default is None which results in the tetrahedron being placed at (0, 0, 0).
        length : float , optional
            The length of the edge of the tetrahedron. Default is 1.
        depth : int , optional
            The desired maximum number of recrusive subdivision levels.
        direction : list , optional
            The vector representing the up direction of the tetrahedron. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the tetrahedron. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.CellComplex
            The created tetrahedron.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
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

        depth = max(depth, 0)
        if depth == 0:
            return tetrahedron
        else:
            # Recursively subdivide the tetrahedron
            subdivided_tetrahedra = subdivide_tetrahedron(tetrahedron, depth)
            # Create a cell complex from the subdivided tetrahedra
            return CellComplex.ExternalBoundary(CellComplex.ByCells([tetrahedron]+subdivided_tetrahedra))

    @staticmethod
    def Torus(origin=None, majorRadius: float = 0.5, minorRadius: float = 0.125, uSides: int = 16, vSides: int = 8, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a torus.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the torus. Default is None which results in the torus being placed at (0, 0, 0).
        majorRadius : float , optional
            The major radius of the torus. Default is 0.5.
        minorRadius : float , optional
            The minor radius of the torus. Default is 0.125.
        uSides : int , optional
            The number of sides along the longitude of the torus (around the hole). Default is 16.
        vSides : int , optional
            The number of sides along the latitude of the torus (tube direction). Default is 8.
        direction : list , optional
            The vector representing the up direction of the torus. Default is [0, 0, 1].
        placement : str , optional
            Placement of the input origin relative to the torus. One of:
            - "center": origin is at the torus' geometric center (default)
            - "bottom": origin lies on the lowest point along the up direction
            - "lowerleft": origin is at x/y lower-left and bottom in z of the torus' local bbox
            Comparison is case-insensitive.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Cell
            The created torus.
        """
        # --- Imports kept inside to avoid cyclic dependencies in TopologicPy ---
        from math import cos, sin, pi, sqrt
        try:
            from topologicpy.Vertex import Vertex
            from topologicpy.Face import Face
            from topologicpy.Shell import Shell
            from topologicpy.Cell import Cell
            from topologicpy.Topology import Topology
        except Exception:
            # Fallback class names if imported as core modules in some setups
            from topologic_core import Vertex, Face, Shell, Cell, Topology  # type: ignore

        # --- Validation ---
        if majorRadius <= 0 or minorRadius <= 0:
            raise ValueError("majorRadius and minorRadius must be > 0.")
        if uSides < 3 or vSides < 3:
            raise ValueError("uSides and vSides must be >= 3.")
        if minorRadius >= majorRadius:
            # Geometrically valid but unusual; keep strict to avoid self-intersections at low resolution
            raise ValueError("minorRadius must be smaller than majorRadius for a clean torus.")

        # --- Helpers ---
        def _norm(v):
            x, y, z = v
            m = sqrt(x*x + y*y + z*z)
            if m == 0:
                return (0.0, 0.0, 1.0)
            return (x/m, y/m, z/m)

        def _dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def _cross(a, b):
            return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

        def _rot_matrix_from_z(to_dir):
            """
            Build a rotation matrix that maps +Z to 'to_dir' using Rodrigues' formula.
            """
            z = (0.0, 0.0, 1.0)
            t = _norm(to_dir)
            c = _dot(z, t)  # cos(theta)
            if abs(c - 1.0) < 1e-12:
                # Already aligned
                return ((1.0,0.0,0.0),
                        (0.0,1.0,0.0),
                        (0.0,0.0,1.0))
            if abs(c + 1.0) < 1e-12:
                # 180 degrees: rotate around any axis perpendicular to z (e.g., x-axis)
                return ((1.0, 0.0, 0.0),
                        (0.0,-1.0, 0.0),
                        (0.0, 0.0,-1.0))
            k = _cross(z, t)
            kx, ky, kz = _norm(k)
            s = sqrt(max(0.0, 1.0 - c*c))
            # Skew-symmetric K
            K = ((0.0, -kz,  ky),
                (kz,  0.0, -kx),
                (-ky, kx,  0.0))
            # I + K*s + K^2*(1-c)
            # First compute K^2
            K2 = (
                (K[0][0]*K[0][0] + K[0][1]*K[1][0] + K[0][2]*K[2][0],
                K[0][0]*K[0][1] + K[0][1]*K[1][1] + K[0][2]*K[2][1],
                K[0][0]*K[0][2] + K[0][1]*K[1][2] + K[0][2]*K[2][2]),
                (K[1][0]*K[0][0] + K[1][1]*K[1][0] + K[1][2]*K[2][0],
                K[1][0]*K[0][1] + K[1][1]*K[1][1] + K[1][2]*K[2][1],
                K[1][0]*K[0][2] + K[1][1]*K[1][2] + K[1][2]*K[2][2]),
                (K[2][0]*K[0][0] + K[2][1]*K[1][0] + K[2][2]*K[2][0],
                K[2][0]*K[0][1] + K[2][1]*K[1][1] + K[2][2]*K[2][1],
                K[2][0]*K[0][2] + K[2][1]*K[1][2] + K[2][2]*K[2][2]),
            )
            I = ((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))

            def _madd(A, B, s=1.0):
                return tuple(tuple(A[i][j] + s*B[i][j] for j in range(3)) for i in range(3))

            R = I
            R = _madd(R, K, s)           # I + s*K
            R = _madd(R, K2, (1.0 - c))  # + (1-c)*K^2
            return R

        def _apply_R(p, R):
            return (
                R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2],
                R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2],
                R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2],
            )

        def _add(a, b):
            return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

        def _scale(v, s):
            return (v[0]*s, v[1]*s, v[2]*s)

        # --- Parametric grid in local coordinates (+Z is up) ---
        # u: around the main ring (longitude), v: around the tube (latitude)
        du = 2.0*pi / uSides
        dv = 2.0*pi / vSides

        # Precompute angles to avoid repeated trig
        cosu = [cos(i*du) for i in range(uSides)]
        sinu = [sin(i*du) for i in range(uSides)]
        cosv = [cos(j*dv) for j in range(vSides)]
        sinv = [sin(j*dv) for j in range(vSides)]

        # Vertex grid (uSides x vSides)
        grid = [[None for _ in range(vSides)] for _ in range(uSides)]
        points = [[None for _ in range(vSides)] for _ in range(uSides)]  # store tuples for transforms

        for i in range(uSides):
            cu, su = cosu[i], sinu[i]
            for j in range(vSides):
                cv, sv = cosv[j], sinv[j]
                x = (majorRadius + minorRadius * cv) * cu
                y = (majorRadius + minorRadius * cv) * su
                z =  minorRadius * sv
                points[i][j] = (x, y, z)

        # --- Orientation: rotate local +Z to requested direction ---
        R = _rot_matrix_from_z(direction if isinstance(direction, (list, tuple)) else [0,0,1])
        points = [[_apply_R(points[i][j], R) for j in range(vSides)] for i in range(uSides)]

        # --- Placement: translate relative to the given origin point ---
        # Determine placement offset in *local* frame, then rotate it by R, then add origin.
        placement_lc = placement.lower().strip()
        if placement_lc not in ("center", "bottom", "lowerleft"):
            raise ValueError('placement must be one of: "center", "bottom", "lowerleft".')

        # In local frame, bbox extents are:
        #   x,y in [- (R + r), + (R + r)]
        #   z in [ -r, +r ]
        # So:
        # - "center"   : no extra shift (center at (0,0,0))
        # - "bottom"   : shift up by r along +Z so the lowest point touches z=0 (then move to origin)
        # - "lowerleft": put min x,y at 0 and bottom at z=0, i.e. shift by (R+r, R+r, r)
        if placement_lc == "center":
            placement_local_offset = (0.0, 0.0, 0.0)
        elif placement_lc == "bottom":
            placement_local_offset = (0.0, 0.0, minorRadius)
        else:  # "lowerleft"
            placement_local_offset = (majorRadius + minorRadius, majorRadius + minorRadius, minorRadius)

        # Rotate the local placement offset into world frame
        placement_world_offset = _apply_R(placement_local_offset, R)

        # Determine origin position
        if origin is None:
            ox, oy, oz = (0.0, 0.0, 0.0)
        else:
            try:
                ox = Vertex.X(origin)
                oy = Vertex.Y(origin)
                oz = Vertex.Z(origin)
            except Exception:
                # Accept a plain (x,y,z) tuple/list as a convenience
                ox, oy, oz = origin  # type: ignore

        origin_pt = (ox, oy, oz)
        base_translation = _add(origin_pt, placement_world_offset)

        # Apply final translation
        points = [[_add(points[i][j], base_translation) for j in range(vSides)] for i in range(uSides)]

        # --- Build Topologic vertices (reuse grid references) ---
        for i in range(uSides):
            for j in range(vSides):
                x, y, z = points[i][j]
                grid[i][j] = Vertex.ByCoordinates(x, y, z)

        # --- Triangulate the quad grid into faces (2 triangles per quad) ---
        faces = []
        # Wind triangles so that normals generally point outward
        for i in range(uSides):
            i1 = (i + 1) % uSides
            for j in range(vSides):
                j1 = (j + 1) % vSides
                v00 = grid[i][j]
                v10 = grid[i1][j]
                v11 = grid[i1][j1]
                v01 = grid[i][j1]
                # Two triangles per cell:
                f1 = Face.ByVertices([v00, v10, v11], tolerance)  # triangle
                f2 = Face.ByVertices([v00, v11, v01], tolerance)  # triangle
                if f1 is None or f2 is None:
                    raise RuntimeError("Failed to create torus facets (Face.ByVertices returned None).")
                faces.append(f1)
                faces.append(f2)

        # --- Stitch into a closed shell, then a cell ---
        shell = Shell.ByFaces(faces, tolerance)
        if shell is None:
            # As a fallback, try slight relaxation on tolerance (if environment is finicky)
            shell = Shell.ByFaces(faces, tolerance * 10.0)
        if shell is None:
            raise RuntimeError("Failed to stitch torus shell from facets.")

        # Try common constructors to obtain a solid Cell
        cell = None
        # 1) Common signature: Cell.ByShell(shell)
        try:
            cell = Cell.ByShell(shell)
        except Exception:
            cell = None
        # 2) Sometimes requires tolerance
        if cell is None:
            try:
                cell = Cell.ByShell(shell, tolerance)
            except Exception:
                cell = None
        # 3) Some builds expect a list of shells (external only)
        if cell is None:
            try:
                cell = Cell.ByShells([shell], tolerance)
            except Exception:
                cell = None
        # 4) Rare builds: stitch directly from faces
        if cell is None:
            try:
                cell = Cell.ByFaces(faces, tolerance)
            except Exception:
                cell = None

        if cell is None:
            # As a last resort, return the stitched shell so the caller still gets usable geometry.
            # But the contract says Cell; better to error explicitly so issues are caught early.
            raise RuntimeError("Failed to create a solid Cell from the torus shell. Check tolerances and resolution (uSides/vSides).")

        # Clean up small topological defects if available
        try:
            cell = Topology.Clean(cell, tolerance)  # optional: no-op if not available
        except Exception:
            pass

        return cell

    @staticmethod
    def Torus_old(origin= None, majorRadius: float = 0.5, minorRadius: float = 0.125, uSides: int = 16, vSides: int = 8, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a torus.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the torus. Default is None which results in the torus being placed at (0, 0, 0).
        majorRadius : float , optional
            The major radius of the torus. Default is 0.5.
        minorRadius : float , optional
            The minor radius of the torus. Default is 0.1.
        uSides : int , optional
            The number of sides along the longitude of the torus. Default is 16.
        vSides : int , optional
            The number of sides along the latitude of the torus. Default is 8.
        direction : list , optional
            The vector representing the up direction of the torus. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the torus. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def TShape(origin=None,
            width=1,
            length=1,
            height=1,
            wSides=1,
            a=0.25,
            b=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            mantissa=6,
            tolerance=0.0001,
            silent=False):
        """
        Creates a T-shape cell.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the T-shape. Default is None which results in the T-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the T-shape. Default is 1.0.
        length : float , optional
            The overall length of the T-shape. Default is 1.0.
        height : float , optional
            the overall height of the T-shape. Default is 1.0.
        wSides : int , optional
            The desired number of sides along the Z-axis. Default is 1.
        a : float , optional
            The hortizontal thickness of the vertical arm of the T-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the T-shape. Default is 0.25.
        flipHorizontal : bool , optional
            if set to True, the shape is flipped horizontally. Default is False.
        flipVertical : bool , optional
            if set to True, the shape is flipped vertically. Default is False.
        direction : list , optional
            The vector representing the up direction of the T-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the T-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        mantissa: int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created T-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Cell.TShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Cell.TShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Cell.TShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Cell.LShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Cell.TShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Cell.TShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Cell.TShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Cell.TShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Cell.TShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance):
            if not silent:
                print("Cell.TShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Cell.TShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Cell.TShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Cell.TShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        t_shape_wire = Wire.TShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=[0,0,1],
                                   placement="center",
                                   tolerance=tolerance,
                                   silent=silent)

        distance = height/wSides
        wires = [t_shape_wire]
        for i in range(wSides):
            t_shape_wire = Topology.Translate(t_shape_wire, 0, 0, distance)
            wires.append(t_shape_wire)
        return_cell = Cell.ByWires(wires, triangulate=False, mantissa=mantissa, tolerance=tolerance, silent=silent)
        # move down to center
        return_cell = Topology.Translate(return_cell, 0, 0, -height*0.5)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = height*0.5
        elif placement.lower() == "top":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
            zOffset = height*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5
            zOffset = -height*0.5
        return_cell = Topology.Translate(return_cell, x=xOffset, y=yOffset, z=zOffset)
        return_cell = Topology.Place(return_cell, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_cell = Topology.Orient(return_cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_cell
    
    @staticmethod
    def Tube(origin= None, radius: float = 1.0, height: float = 1.0, thickness: float = 0.25, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a Tube. This method is an alias for the circular hollow section (CHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the CHS. Default is None which results in the CHS being placed at (0, 0, 0).
        radius : float , optional
            The outer radius of the CHS. Default is 1.0.
        thickness : float , optional
            The thickness of the CHS. Default is 0.25.
        height : float , optional
            The height of the CHS. Default is 1.0.
        sides : int , optional
            The desired number of sides of the CSH. Default is 16.
        direction : list , optional
            The vector representing the up direction of the RHS. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the RHS. This can be "center", "bottom", "top", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if thickness >= radius:
            if not silent:
                print("Cell.Tube - Error: The thickness value is larger than or equal to the outer radius value. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        return Cell.CHS(origin=origin,
                        radius=radius,
                        height=height,
                        thickness=thickness,
                        sides=sides,
                        direction=direction,
                        placement=placement,
                        tolerance=tolerance,
                        silent=silent)
        
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
            The number of decimal places to round the result to. Default is 6.

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
    def Wedge(origin=None,
            width=1,
            length=1,
            height=1,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a Wedge.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the Wedge. Default is None which results in the Wedge being placed at (0, 0, 0).
        width : float , optional
            The overall width of the Wedge. Default is 1.0.
        length : float , optional
            The overall length of the Wedge. Default is 1.0.
        height : float , optional
            The overall height of the Wedge. Default is 1.0.
        direction : list , optional
            The vector representing the up direction of the Wedge. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the Wedge. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created Wedge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Cell.Wedge - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Cell.Wedge - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(height, int) and not isinstance(height, float):
            if not silent:
                print("Cell.Wedge - Error: The height input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Cell.Wedge - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Cell.Wedge - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if height <= tolerance:
            if not silent:
                print("Cell.Wedge - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Cell.Wedge - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Cell.Wedge - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Cell.Wedge - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        
        # Define the vertices of the T-shape (counterclockwise)
        v1 = Vertex.ByCoordinates(0,0,0)
        v2 = Vertex.ByCoordinates(width, 0, 0)
        v3 = Vertex.ByCoordinates(width, length, 0)
        v4 = Vertex.ByCoordinates(0, length, 0)
        v5 = Vertex.ByCoordinates(0, length, height)
        v6 = Vertex.ByCoordinates(0, 0, height)

        f1 = Face.ByVertices([v1, v2, v3, v4], tolerance=tolerance)
        f2 = Face.ByVertices([v1, v2, v6], tolerance=tolerance)
        f3 = Face.ByVertices([v4, v5, v3], tolerance=tolerance)
        f4 = Face.ByVertices([v1, v6, v5, v4], tolerance=tolerance)
        f5 = Face.ByVertices([v2, v3, v5, v6], tolerance=tolerance)
        cell = Cell.ByFaces([f1, f2, f3, f4, f5])
        cell = Topology.Translate(cell, -width/2, -length/2, -height/2)
        cell = Topology.Translate(cell, Vertex.X(origin), Vertex.Y(origin), Vertex.Z(origin))
        if flipHorizontal == True:
            xScale = -1
        else:
            xScale = 1
        if flipVertical == True:
            zScale = -1
        else:
            zScale = 1
        if xScale == -1 or zScale == -1:
            cell = Topology.Scale(cell, origin=origin, x=xScale, y=1, z=zScale)
        if placement.lower() == "lowerleft":
            cell = Topology.Translate(cell, origin=origin, x=width/2, y=length/2, z=height/2)
        elif placement.lower() == "upperright":
            cell = Topology.Translate(cell, origin=origin, x=-width/2, y=-length/2, z=-height/2)
        elif placement.lower() == "upperleft":
            cell = Topology.Translate(cell, origin=origin, x=width/2, y=-length/2, z=-height/2)
        elif placement.lower() == "lowerright":
            cell = Topology.Translate(cell, origin=origin, x=-width/2, y=length/2, z=height/2)
        
        if direction != [0, 0, 1]:
            cell = Topology.Orient(cell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        
        return cell

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

