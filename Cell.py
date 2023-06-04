import topologicpy
import topologic
from topologicpy.Wire import Wire
from topologicpy.Topology import Topology
import math

class Cell(Topology):
    @staticmethod
    def Area(cell: topologic.Cell, mantissa: int = 4) -> float:
        """
        Returns the surface area of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The cell.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The surface area of the input cell.

        """
        from topologicpy.Face import Face

        faces = []
        _ = cell.Faces(None, faces)
        area = 0.0
        for aFace in faces:
            area = area + Face.Area(aFace)
        return round(area, mantissa)

    @staticmethod
    def Box(origin: topologic.Vertex = None, width: float = 1, length: float = 1, height: float = 1, uSides: int = 1, vSides:int = 1, wSides:int = 1, direction: list = [0,0,1], placement: str ="center") -> topologic.Cell:
        """
        Creates a box.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin location of the box. The default is None which results in the box being placed at (0,0,0).
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
            The vector representing the up direction of the box. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the box. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".

        Returns
        -------
        topologic.Cell
            The created box.

        """
        return Cell.Prism(origin=origin, width=width, length=length, height=height, uSides=uSides, vSides=vSides, wSides=wSides, direction=direction, placement=placement)

    @staticmethod
    def ByFaces(faces: list, planarize: bool = False, tolerance: float = 0.0001) -> topologic.Cell:
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

        Returns
        -------
        topologic.Cell
            The created cell.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        if not isinstance(faces, list):
            return None
        faceList = [x for x in faces if isinstance(x, topologic.Face)]
        if len(faceList) < 1:
            return None
        planarizedList = []
        enlargedList = []
        if planarize:
            planarizedList = [Face.Planarize(f) for f in faceList]
            enlargedList = [Face.ByOffset(f, offset=-tolerance*10) for f in planarizedList]
            cell = topologic.Cell.ByFaces(enlargedList, tolerance)
            faces = Topology.SubTopologies(cell, subTopologyType="face")
            finalFaces = []
            for f in faces:
                centroid = Topology.Centroid(f)
                n = Face.Normal(f)
                v = Topology.Translate(centroid, n[0]*0.01,n[1]*0.01,n[2]*0.01)
                if not Cell.IsInside(cell, v):
                    finalFaces.append(f)
            finalFinalFaces = []
            for f in finalFaces:
                vertices = Face.Vertices(f)
                w = Wire.Cycles(Face.ExternalBoundary(f), maxVertices=len(vertices))[0]
                finalFinalFaces.append(Face.ByWire(w))
            return topologic.Cell.ByFaces(finalFinalFaces, tolerance)
        else:
            return topologic.Cell.ByFaces(faces, tolerance)

    @staticmethod
    def ByShell(shell: topologic.Shell, planarize: bool = False, tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a cell from the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell. The shell must be closed for this method to succeed.
        planarize : bool, optional
            If set to True, the input faces of the input shell are planarized before building the cell. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cell
            The created cell.

        """
        from topologicpy.Topology import Topology
        if not isinstance(shell, topologic.Shell):
            return None
        faces = Topology.SubTopologies(shell, subTopologyType="face")
        return Cell.ByFaces(faces, planarize=planarize, tolerance=tolerance)
    
    @staticmethod
    def ByThickenedFace(face: topologic.Face, thickness: float = 1.0, bothSides: bool = True, reverse: bool = False,
                            planarize: bool = False, tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a cell by thickening the input face.

        Parameters
        ----------
        face : topologic.Face
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
        topologic.Cell
            The created cell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(face, topologic.Face):
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

        cellFaces = [bottomFace, topFace]
        bottomEdges = []
        _ = bottomFace.Edges(None, bottomEdges)
        for bottomEdge in bottomEdges:
            topEdge = Topology.Translate(bottomEdge, faceNormal[0]*thickness, faceNormal[1]*thickness, faceNormal[2]*thickness)
            sideEdge1 = Edge.ByVertices([bottomEdge.StartVertex(), topEdge.StartVertex()])
            sideEdge2 = Edge.ByVertices([bottomEdge.EndVertex(), topEdge.EndVertex()])
            cellWire = Cluster.SelfMerge(Cluster.ByTopologies([bottomEdge, sideEdge1, topEdge, sideEdge2]))
            cellFaces.append(Face.ByWire(cellWire))
        return Cell.ByFaces(cellFaces, planarize=planarize, tolerance=tolerance)

    @staticmethod
    def ByThickenedShell(shell: topologic.Shell, direction: list = [0,0,1], thickness: float = 1.0, bothSides: bool = True, reverse: bool = False,
                            planarize: bool = False, tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a cell by thickening the input shell. The shell must be open.

        Parameters
        ----------
        shell : topologic.Shell
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
        topologic.Cell
            The created cell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        if not isinstance(shell, topologic.Shell):
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
        bottomWire = Shell.ExternalBoundary(bottomShell)
        bottomEdges = Wire.Edges(bottomWire)
        for bottomEdge in bottomEdges:
            topEdge = Topology.Translate(bottomEdge, direction[0]*thickness, direction[1]*thickness, direction[2]*thickness)
            sideEdge1 = Edge.ByVertices([Edge.StartVertex(bottomEdge), Edge.StartVertex(topEdge)])
            sideEdge2 = Edge.ByVertices([Edge.EndVertex(bottomEdge), Edge.EndVertex(topEdge)])
            cellWire = Cluster.SelfMerge(Cluster.ByTopologies([bottomEdge, sideEdge1, topEdge, sideEdge2]))
            cellFace = Face.ByWire(cellWire)
            cellFaces.append(cellFace)
        return Cell.ByFaces(cellFaces, planarize=planarize, tolerance=tolerance)
    
    @staticmethod
    def ByWires(wires: list, close: bool = False, triangulate: bool = True, planarize: bool = False, tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a cell by lofting through the input list of wires.

        Parameters
        ----------
        wires : topologic.Wire
            The input list of wires.
        close : bool , optional
            If set to True, the last wire in the list of input wires will be connected to the first wire in the list of input wires. The default is False.
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
        topologic.Cell
            The created cell.

        """

        def cleanup(f):
            flatFace = Face.Flatten(f)
            world_origin = Vertex.ByCoordinates(0,0,0)
            # Retrieve the needed transformations
            dictionary = Topology.Dictionary(flatFace)
            xTran = Dictionary.ValueAtKey(dictionary,"xTran")
            yTran = Dictionary.ValueAtKey(dictionary,"yTran")
            zTran = Dictionary.ValueAtKey(dictionary,"zTran")
            phi = Dictionary.ValueAtKey(dictionary,"phi")
            theta = Dictionary.ValueAtKey(dictionary,"theta")

            f = Topology.Rotate(f, origin=world_origin, x=0, y=1, z=0, degree=theta)
            f = Topology.Rotate(f, origin=world_origin, x=0, y=0, z=1, degree=phi)
            f = Topology.Translate(f, xTran, yTran, zTran)
            return f
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        faces = [Face.ByWire(wires[0]), Face.ByWire(wires[-1])]
        if close == True:
            faces.append(Face.ByWire(wires[0]))
        if triangulate == True:
            triangles = []
            for face in faces:
                if len(Topology.Vertices(face)) > 3:
                    triangles += Face.Triangulate(face)
                else:
                    triangles += [face]
            faces = triangles
        for i in range(len(wires)-1):
            wire1 = wires[i]
            wire2 = wires[i+1]
            w1_edges = []
            _ = wire1.Edges(None, w1_edges)
            w2_edges = []
            _ = wire2.Edges(None, w2_edges)
            if len(w1_edges) != len(w2_edges):
                return None
            if triangulate == True:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()])
                    except:
                        try:
                            e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e4])))
                        except:
                            pass
                    try:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                    except:
                        try:
                            e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()])
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3])))
                        except:
                            pass
                    if e3 and e4:
                        e5 = Edge.ByVertices([e1.StartVertex(), e2.EndVertex()])
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e5, e4])))
                        faces.append(Face.ByWire(Wire.ByEdges([e2, e5, e3])))
            else:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()])
                    except:
                        try:
                            e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                        except:
                            pass
                    try:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                    except:
                        try:
                            e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()])
                        except:
                            pass
                    if e3 and e4:
                        try:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e4, e2, e3])))
                        except:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e3, e2, e4])))
                    elif e3:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e3, e2])))
                    elif e4:
                            faces.append(Face.ByWire(Wire.ByEdges([e1, e4, e2])))
        #for f in faces:
            #cleanup(f)
        cell = Cell.ByFaces(faces, planarize=planarize, tolerance=tolerance)
        if not cell:
            cell = Shell.ByFaces(faces)
            if cell:
                geom = Topology.Geometry(cell)
                cell = Topology.ByGeometry(geom['vertices'], geom['edges'], geom['faces'])
            elif not isinstance(cell, topologic.Cell):
                cell = Shell.ByFaces(faces)
                if not cell:
                    cell = Cluster.ByTopologies(faces)
        return cell

    @staticmethod
    def ByWiresCluster(cluster: topologic.Cluster, close: bool = False, triangulate: bool = True, planarize: bool = False, tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a cell by lofting through the input cluster of wires.

        Parameters
        ----------
        cluster : topologic.Cluster
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
        topologic.Cell
            The created cell.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        wires = []
        _ = cluster.Wires(None, wires)
        return Cell.ByWires(wires, close=close, triangulate=triangulate, planarize=planarize, tolerance=tolerance)

    @staticmethod
    def Compactness(cell: topologic.Cell, reference = "sphere", mantissa: int = 4) -> float:
        """
        Returns the compactness measure of the input cell. If the reference is "sphere", this is also known as 'sphericity' (https://en.wikipedia.org/wiki/Sphericity).

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.
        reference : str , optional
            The desired reference to which to compare this compactness. The options are "sphere" and "cube". It is case insensitive. The default is "sphere".
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Raises
        ------
        Exception
            Raises an exception if the resulting surface area is negative. This can occur if the cell is degenerate or has flipped face normals.

        Returns
        -------
        float
            The compactness of the input cell.

        """
        faces = []
        _ = cell.Faces(None, faces)
        area = 0.0
        for aFace in faces:
            area = area + abs(topologic.FaceUtility.Area(aFace))
        volume = abs(topologic.CellUtility.Volume(cell))
        compactness  = 0
        #From https://en.wikipedia.org/wiki/Sphericity
        if area > 0:
            if reference.lower() == "sphere":
                compactness = (((math.pi)**(1/3))*((6*volume)**(2/3)))/area
                print(4.84*volume**(2/3)/area)
            else:
                compactness = 6*(volume**(2/3))/area
        else:
            raise Exception("Error: Cell.Compactness: Cell surface area is not positive")
        return round(compactness, mantissa)
    
    @staticmethod
    def Cone(origin: topologic.Vertex = None, baseRadius: float = 0.5, topRadius: float = 0, height: float = 1, uSides: int = 16, vSides: int = 1, direction: list = [0,0,1],
                 dirZ: float = 1, placement: str = "center", tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a cone.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the cone. The default is None which results in the cone being placed at (0,0,0).
        baseRadius : float , optional
            The radius of the base circle of the cone. The default is 0.5.
        topRadius : float , optional
            The radius of the top circle of the cone. The default is 0.
        height : float , optional
            The height of the cone. The default is 1.
        sides : int , optional
            The number of sides of the cone. The default is 16.
        direction : list , optional
            The vector representing the up direction of the cone. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the cone. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cell
            The created cone.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        def createCone(baseWire, topWire, baseVertex, topVertex, tolerance):
            if baseWire == None and topWire == None:
                raise Exception("Cell.Cone - Error: Both radii of the cone cannot be zero at the same time")
            elif baseWire == None:
                apex = baseVertex
                wire = topWire
            elif topWire == None:
                apex = topVertex
                wire = baseWire
            else:
                return topologic.CellUtility.ByLoft([baseWire, topWire])
            vertices = []
            _ = wire.Vertices(None,vertices)
            faces = [Face.ByWire(wire)]
            for i in range(0, len(vertices)-1):
                w = Wire.ByVertices([apex, vertices[i], vertices[i+1]])
                f = Face.ByWire(w)
                faces.append(f)
            w = Wire.ByVertices([apex, vertices[-1], vertices[0]])
            f = Face.ByWire(w)
            faces.append(f)
            return Cell.ByFaces(faces, tolerance=tolerance)
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
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
                baseX = math.sin(angle)*baseRadius + origin.X() + xOffset
                baseY = math.cos(angle)*baseRadius + origin.Y() + yOffset
                baseZ = origin.Z() + zOffset
                baseV.append(Vertex.ByCoordinates(baseX,baseY,baseZ))
            if topRadius > 0:
                topX = math.sin(angle)*topRadius + origin.X() + xOffset
                topY = math.cos(angle)*topRadius + origin.Y() + yOffset
                topV.append(Vertex.ByCoordinates(topX,topY,topZ))
        if baseRadius > 0:
            baseWire = Wire.ByVertices(baseV)
        else:
            baseWire = None
        if topRadius > 0:
            topWire = Wire.ByVertices(topV)
        else:
            topWire = None
        baseVertex = Vertex.ByCoordinates(origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset)
        topVertex = Vertex.ByCoordinates(origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset+height)
        cone = createCone(baseWire, topWire, baseVertex, topVertex, tolerance)
        if cone == None:
            return None
        
        if vSides > 1:
            cutting_planes = []
            baseX = origin.X() + xOffset
            baseY = origin.Y() + yOffset
            size = max(baseRadius, topRadius)*3
            for i in range(1, vSides):
                baseZ = origin.Z() + zOffset + float(height)/float(vSides)*i
                tool_origin = Vertex.ByCoordinates(baseX, baseY, baseZ)
                cutting_planes.append(Face.ByWire(Wire.Rectangle(origin=tool_origin, width=size, length=size)))
            cutting_planes_cluster = Cluster.ByTopologies(cutting_planes)
            shell = Cell.Shells(cone)[0]
            shell = shell.Slice(cutting_planes_cluster)
            cone = Cell.ByShell(shell)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        cone = Topology.Rotate(cone, origin, 0, 1, 0, theta)
        cone = Topology.Rotate(cone, origin, 0, 0, 1, phi)
        return cone
  
    @staticmethod
    def ContainmentStatus(cell: topologic.Cell, vertex: topologic.Vertex, tolerance: float = 0.0001) -> int:
        """
        Returns the containment status of the input vertex in relationship to the input cell

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        int
            Returns 0 if the vertex is inside, 1 if it is on the boundary of, and 2 if it is outside the input cell.

        """
        if not cell:
            return None
        if not isinstance(cell, topologic.Cell):
            return None
        try:
            status = topologic.CellUtility.Contains(cell, vertex, tolerance)
            if status == 0:
                return 0
            elif status == 1:
                return 1
            else:
                return 2
        except:
            return None
 
    @staticmethod
    def Cylinder(origin: topologic.Vertex = None, radius: float = 0.5, height: float = 1, uSides: int = 16, vSides:int = 1, direction: list = [0,0,1],
                     placement: str = "center", tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a cylinder.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the cylinder. The default is None which results in the cylinder being placed at (0,0,0).
        radius : float , optional
            The radius of the cylinder. The default is 0.5.
        height : float , optional
            The height of the cylinder. The default is 1.
        uSides : int , optional
            The number of circle segments of the cylinder. The default is 16.
        vSides : int , optional
            The number of vertical segments of the cylinder. The default is 1.
        direction : list , optional
            The vector representing the up direction of the cylinder. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the cylinder. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "bottom".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "center":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = radius
            yOffset = radius
        circle_origin = Vertex.ByCoordinates(origin.X() + xOffset, origin.Y() + yOffset, origin.Z() + zOffset)
        
        baseWire = Wire.Circle(origin=circle_origin, radius=radius, sides=uSides, fromAngle=0, toAngle=360, close=True, direction=[0,0,1], placement="center", tolerance=tolerance)
        baseFace = Face.ByWire(baseWire)
        cylinder = Cell.ByThickenedFace(face=baseFace, thickness=height, bothSides=False, reverse=False,
                            tolerance=tolerance)
        if vSides > 1:
            cutting_planes = []
            baseX = origin.X() + xOffset
            baseY = origin.Y() + yOffset
            size = radius*3
            for i in range(1, vSides):
                baseZ = origin.Z() + zOffset + float(height)/float(vSides)*i
                tool_origin = Vertex.ByCoordinates(baseX, baseY, baseZ)
                cutting_planes.append(Face.ByWire(Wire.Rectangle(origin=tool_origin, width=size, length=size)))
            cutting_planes_cluster = Cluster.ByTopologies(cutting_planes)
            cylinder = CellComplex.ExternalBoundary(cylinder.Slice(cutting_planes_cluster))

        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        cylinder = Topology.Rotate(cylinder, origin, 0, 1, 0, theta)
        cylinder = Topology.Rotate(cylinder, origin, 0, 0, 1, phi)
        return cylinder
    
    @staticmethod
    def Decompose(cell: topologic.Cell, tiltAngle: float = 10, tolerance: float = 0.0001) -> dict:
        """
        Decomposes the input cell into its logical components. This method assumes that the positive Z direction is UP.

        Parameters
        ----------
        cell : topologic.Cell
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
            apTopologies = []
            apertures = Topology.Apertures(topology)
            if isinstance(apertures, list):
                for aperture in apertures:
                    apTopologies.append(Aperture.Topology(aperture))
            return apTopologies

        if not isinstance(cell, topologic.Cell):
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
        up = [0,0,1]
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
    def Edges(cell: topologic.Cell) -> list:
        """
        Returns the edges of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.

        Returns
        -------
        list
            The list of edges.

        """ 
        if not isinstance(cell, topologic.Cell):
            return None
        edges = []
        _ = cell.Edges(None, edges)
        return edges

    @staticmethod
    def ExternalBoundary(cell: topologic.Cell) -> topologic.Shell:
        """
        Returns the external boundary of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.

        Returns
        -------
        topologic.Shell
            The external boundary of the input cell.

        """
        if not cell:
            return None
        if not isinstance(cell, topologic.Cell):
            return None
        try:
            return cell.ExternalBoundary()
        except:
            return None
    
    @staticmethod
    def Faces(cell: topologic.Cell) -> list:
        """
        Returns the faces of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.

        Returns
        -------
        list
            The list of faces.

        """
        if not isinstance(cell, topologic.Cell):
            return None
        faces = []
        _ = cell.Faces(None, faces)
        return faces

    @staticmethod
    def Hyperboloid(origin: topologic.Cell = None, baseRadius: float = 0.5, topRadius: float = 0.5, height: float = 1, sides: int = 16, direction: list = [0,0,1],
                        twist: float = 360, placement: str = "center", tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a hyperboloid.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the hyperboloid. The default is None which results in the hyperboloid being placed at (0,0,0).
        baseRadius : float , optional
            The radius of the base circle of the hyperboloid. The default is 0.5.
        topRadius : float , optional
            The radius of the top circle of the hyperboloid. The default is 0.5.
        height : float , optional
            The height of the cone. The default is 1.
        sides : int , optional
            The number of sides of the cone. The default is 16.
        direction : list , optional
            The vector representing the up direction of the hyperboloid. The default is [0,0,1].
        twist : float , optional
            The angle to twist the base cylinder. The default is 360.
        placement : str , optional
            The description of the placement of the origin of the hyperboloid. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cell
            The created hyperboloid.

        """

        def createHyperboloid(baseVertices, topVertices, tolerance):
            baseWire = Wire.ByVertices(baseVertices, close=True)
            topWire = Wire.ByVertices(topVertices, close=True)
            baseFace = Face.ByWire(baseWire)
            topFace = Face.ByWire(topWire)
            faces = [baseFace, topFace]
            for i in range(0, len(baseVertices)-1):
                w = Wire.ByVertices([baseVertices[i], topVertices[i], topVertices[i+1]], close=True)
                f = Face.ByWire(w)
                faces.append(f)
                w = Wire.ByVertices([baseVertices[i+1], baseVertices[i], topVertices[i+1]], close=True)
                f = Face.ByWire(w)
                faces.append(f)
            w = Wire.ByVertices([baseVertices[-1], topVertices[-1], topVertices[0]], close=True)
            f = Face.ByWire(w)
            faces.append(f)
            w = Wire.ByVertices([baseVertices[0], baseVertices[-1], topVertices[0]], close=True)
            f = Face.ByWire(w)
            faces.append(f)
            returnTopology = topologic.Cell.ByFaces(faces, tolerance)
            if returnTopology == None:
                returnTopology = topologic.Cluster.ByTopologies(faces)
            return returnTopology
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
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
        baseZ = origin.Z() + zOffset
        topZ = origin.Z() + zOffset + height
        for i in range(sides):
            angle = math.radians(360/sides)*i
            if baseRadius > 0:
                baseX = math.sin(angle+math.radians(twist))*baseRadius + origin.X() + xOffset
                baseY = math.cos(angle+math.radians(twist))*baseRadius + origin.Y() + yOffset
                baseZ = origin.Z() + zOffset
                baseV.append(Vertex.ByCoordinates(baseX,baseY,baseZ))
            if topRadius > 0:
                topX = math.sin(angle-math.radians(twist))*topRadius + origin.X() + xOffset
                topY = math.cos(angle-math.radians(twist))*topRadius + origin.Y() + yOffset
                topV.append(Vertex.ByCoordinates(topX,topY,topZ))

        hyperboloid = createHyperboloid(baseV, topV, tolerance)
        if hyperboloid == None:
            return None
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        hyperboloid = Topology.Rotate(hyperboloid, origin, 0, 1, 0, theta)
        hyperboloid = Topology.Rotate(hyperboloid, origin, 0, 0, 1, phi)
        return hyperboloid
    
    @staticmethod
    def InternalBoundaries(cell: topologic.Cell) -> list:
        """
        Returns the internal boundaries of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.

        Returns
        -------
        list
            The list of internal boundaries ([topologic.Shell]).

        """
        shells = []
        _ = cell.InternalBoundaries(shells)
        return shells
    
    @staticmethod
    def InternalVertex(cell: topologic.Cell, tolerance: float = 0.0001):
        """
        Creates a vertex that is guaranteed to be inside the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            The internal vertex.

        """
        if not cell:
            return None
        if not isinstance(cell, topologic.Cell):
            return None
        try:
            return topologic.CellUtility.InternalVertex(cell, tolerance)
        except:
            return None
    
    @staticmethod
    def IsInside(cell: topologic.Cell, vertex: topologic.Vertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is inside the input cell. Returns False otherwise.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input cell. Returns False otherwise.

        """
        if not isinstance(cell, topologic.Cell):
            return None
        try:
            return (topologic.CellUtility.Contains(cell, vertex, tolerance) == 0)
        except:
            return None
    
    @staticmethod
    def IsOnBoundary(cell: topologic.Cell, vertex: topologic.Vertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is on the boundary of the input cell. Returns False otherwise.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input cell. Returns False otherwise.

        """
        if not cell:
            return None
        if not isinstance(cell, topologic.Cell):
            return None
        try:
            return (topologic.CellUtility.Contains(cell, vertex, tolerance) == 1)
        except:
            return None
 
    @staticmethod
    def IsOutside(cell: topologic.Cell, vertex: topologic.Vertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is outisde the input cell. Returns False otherwise.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.
        vertex : topologic.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input cell. Returns False otherwise.

        """
        if not cell:
            return None
        if not isinstance(cell, topologic.Cell):
            return None
        try:
            return (topologic.CellUtility.Contains(cell, vertex, tolerance) == 2)
        except:
            return None
   
    @staticmethod
    def Pipe(edge: topologic.Edge, profile: topologic.Wire = None, radius: float = 0.5, sides: int = 16, startOffset: float = 0, endOffset: float = 0, endcapA: topologic.Topology = None, endcapB: topologic.Topology = None) -> dict:
        """
        Description
        ----------
        Creates a pipe along the input edge.

        Parameters
        ----------
        edge : topologic.Edge
            The centerline of the pipe.
        profile : topologic.Wire , optional
            The profile of the pipe. It is assumed that the profile is in the XY plane. If set to None, a circle of radius 0.5 will be used. The default is None.
        radius : float , optional
            The radius of the pipe. The default is 0.5.
        sides : int , optional
            The number of sides of the pipe. The default is 16.
        startOffset : float , optional
            The offset distance from the start vertex of the centerline edge. The default is 0.
        endOffset : float , optional
            The offset distance from the end vertex of the centerline edge. The default is 0.
        endcapA : topologic.Topology, optional
            The topology to place at the start vertex of the centerline edge. The positive Z direction of the end cap will be oriented in the direction of the centerline edge.
        endcapB : topologic.Topology, optional
            The topology to place at the end vertex of the centerline edge. The positive Z direction of the end cap will be oriented in the inverse direction of the centerline edge.

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
        from topologicpy.Topology import Topology

        if not edge:
            return None
        if not isinstance(edge, topologic.Edge):
            return None
        length = Edge.Length(edge)
        origin = Edge.StartVertex(edge)
        startU = startOffset / length
        endU = 1.0 - (endOffset / length)
        sv = Edge.VertexByParameter(edge, startU)
        ev = Edge.VertexByParameter(edge, endU)
        x1 = sv.X()
        y1 = sv.Y()
        z1 = sv.Z()
        x2 = ev.X()
        y2 = ev.Y()
        z2 = ev.Z()
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        baseV = []
        topV = []

        if isinstance(profile, topologic.Wire):
            baseWire = Topology.Translate(profile, 0 , 0, sv.Z())
            topWire = Topology.Translate(profile, 0 , 0, sv.Z()+dist)
        else:
            for i in range(sides):
                angle = math.radians(360/sides)*i
                x = math.sin(angle)*radius + sv.X()
                y = math.cos(angle)*radius + sv.Y()
                z = sv.Z()
                baseV.append(Vertex.ByCoordinates(x,y,z))
                topV.append(Vertex.ByCoordinates(x,y,z+dist))

            baseWire = Wire.ByVertices(baseV)
            topWire = Wire.ByVertices(topV)
        wires = [baseWire, topWire]
        pipe = Cell.ByWires(wires)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        pipe = Topology.Rotate(pipe, sv, 0, 1, 0, theta)
        pipe = Topology.Rotate(pipe, sv, 0, 0, 1, phi)
        zzz = Vertex.ByCoordinates(0,0,0)
        if endcapA:
            origin = edge.StartVertex()
            x1 = origin.X()
            y1 = origin.Y()
            z1 = origin.Z()
            x2 = edge.EndVertex().X()
            y2 = edge.EndVertex().Y()
            z2 = edge.EndVertex().Z()
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
            endcapA = Topology.Rotate(endcapA, zzz, 0, 1, 0, theta)
            endcapA = Topology.Rotate(endcapA, zzz, 0, 0, 1, phi + 180)
            endcapA = Topology.Translate(endcapA, origin.X(), origin.Y(), origin.Z())
        if endcapB:
            origin = edge.EndVertex()
            x1 = origin.X()
            y1 = origin.Y()
            z1 = origin.Z()
            x2 = edge.StartVertex().X()
            y2 = edge.StartVertex().Y()
            z2 = edge.StartVertex().Z()
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
            endcapB = Topology.Rotate(endcapB, zzz, 0, 1, 0, theta)
            endcapB = Topology.Rotate(endcapB, zzz, 0, 0, 1, phi + 180)
            endcapB = Topology.Translate(endcapB, origin.X(), origin.Y(), origin.Z())
        return {'pipe': pipe, 'endcapA': endcapA, 'endcapB': endcapB}
    
    @staticmethod
    def Prism(origin: topologic.Vertex = None, width: float = 1, length: float = 1, height: float = 1, uSides: int = 1, vSides: int = 1, wSides: int = 1,
                  direction: list = [0,0,1], placement: str ="center") -> topologic.Cell:
        """
        Description
        ----------
        Creates a prism.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin location of the prism. The default is None which results in the prism being placed at (0,0,0).
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
            The vector representing the up direction of the prism. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the prism. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".

        Returns
        -------
        topologic.Cell
            The created prism.

        """
        def sliceCell(cell, width, length, height, uSides, vSides, wSides):
            origin = cell.Centroid()
            shells = []
            _ = cell.Shells(None, shells)
            shell = shells[0]
            wRect = Wire.Rectangle(origin=origin, width=width*1.2, length=length*1.2, direction=[0, 0, 1], placement="center")
            sliceFaces = []
            for i in range(1, wSides):
                sliceFaces.append(Topology.Translate(Face.ByWire(wRect), 0, 0, height/wSides*i - height*0.5))
            uRect = Wire.Rectangle(origin=origin, width=height*1.2, length=length*1.2, direction=[1, 0, 0], placement="center")
            for i in range(1, uSides):
                sliceFaces.append(Topology.Translate(Face.ByWire(uRect), width/uSides*i - width*0.5, 0, 0))
            vRect = Wire.Rectangle(origin=origin, width=height*1.2, length=width*1.2, direction=[0, 1, 0], placement="center")
            for i in range(1, vSides):
                sliceFaces.append(Topology.Translate(Face.ByWire(vRect), 0, length/vSides*i - length*0.5, 0))
            if len(sliceFaces) > 0:
                sliceCluster = topologic.Cluster.ByTopologies(sliceFaces)
                shell = Topology.Slice(topologyA=shell, topologyB=sliceCluster, tranDict=False)
                return Cell.ByShell(shell)
            return cell
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "center":
            zOffset = -height*0.5
        elif placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
        vb1 = Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z()+zOffset)
        vb2 = Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z()+zOffset)
        vb3 = Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z()+zOffset)
        vb4 = Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z()+zOffset)

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire)

        prism = Cell.ByThickenedFace(baseFace, thickness=height, bothSides = False)

        if uSides > 1 or vSides > 1 or wSides > 1:
            prism = sliceCell(prism, width, length, height, uSides, vSides, wSides)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
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
    
    def Roof(face, degree=45, angTolerance=2, tolerance=0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic.Face
            The input face.
        degree : float , optioal
            The desired angle in degrees of the roof. The default is 45.
        angTolerance : float , optional
            The desired angular tolerance. The default is 2. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        cell
            The created roof.

        """
        from topologicpy import Polyskel
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        import topologic
        import math
        '''
        def nearest_vertex_2d(v, vertices, tolerance=0.001):
            for vertex in vertices:
                x2 = Vertex.X(vertex)
                y2 = Vertex.Y(vertex)
                temp_v = Vertex.ByCoordinates(x2, y2, Vertex.Z(v))
                if Vertex.Distance(v, temp_v) <= tolerance:
                    return vertex
            return None
        
        if not isinstance(face, topologic.Face):
            return None
        degree = abs(degree)
        if degree >= 90-tolerance:
            return None
        if degree < tolerance:
            return None
        flat_face = Face.Flatten(face)
        d = Topology.Dictionary(flat_face)
        roof = Wire.Roof(flat_face, degree)
        if not roof:
            return None
        shell = Shell.Skeleton(flat_face)
        faces = Shell.Faces(shell)
        
        if not faces:
            return None
        triangles = []
        for face in faces:
            internalBoundaries = Face.InternalBoundaries(face)
            if len(internalBoundaries) == 0:
                if len(Topology.Vertices(face)) > 3:
                    triangles += Face.Triangulate(face)
                else:
                    triangles += [face]

        roof_vertices = Topology.Vertices(roof)
        flat_vertices = []
        for rv in roof_vertices:
            flat_vertices.append(Vertex.ByCoordinates(Vertex.X(rv), Vertex.Y(rv), 0))

        final_triangles = []
        for triangle in triangles:
            if len(Topology.Vertices(triangle)) > 3:
                triangles = Face.Triangulate(triangle)
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
        '''
        shell = Shell.Roof(face=face, degree=degree, angTolerance=angTolerance, tolerance=tolerance)
        faces = Topology.Faces(shell) + [face]
        cell = Cell.ByFaces(faces, tolerance=tolerance)
        if not cell:
            return None
        return cell
        '''
        if not cell:
            cell = Shell.ByFaces(final_faces, tolerance=tolerance)
            if not cell:
                cell = Cluster.ByTopologies(final_faces)
        cell = Topology.RemoveCoplanarFaces(cell, angTolerance=angTolerance)
        xTran = Dictionary.ValueAtKey(d,"xTran")
        yTran = Dictionary.ValueAtKey(d,"yTran")
        zTran = Dictionary.ValueAtKey(d,"zTran")
        phi = Dictionary.ValueAtKey(d,"phi")
        theta = Dictionary.ValueAtKey(d,"theta")
        cell = Topology.Rotate(cell, origin=Vertex.Origin(), x=0, y=1, z=0, degree=theta)
        cell = Topology.Rotate(cell, origin=Vertex.Origin(), x=0, y=0, z=1, degree=phi)
        cell = Topology.Translate(cell, xTran, yTran, zTran)
        return cell
        '''
    @staticmethod
    def Sets(inputCells: list, superCells: list, tolerance: float = 0.0001) -> list:
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
            The classified list of input cells based on their encolsure within the input list of super cells.

        """
        if len(superCells) == 0:
            cluster = inputCells[0]
            for i in range(1, len(inputCells)):
                oldCluster = cluster
                cluster = cluster.Union(inputCells[i])
                del oldCluster
            superCells = []
            _ = cluster.Cells(None, superCells)
        unused = []
        for i in range(len(inputCells)):
            unused.append(True)
        sets = []
        for i in range(len(superCells)):
            sets.append([])
        for i in range(len(inputCells)):
            if unused[i]:
                iv = topologic.CellUtility.InternalVertex(inputCells[i], tolerance)
                for j in range(len(superCells)):
                    if (topologic.CellUtility.Contains(superCells[j], iv, tolerance) == 0):
                        sets[j].append(inputCells[i])
                        unused[i] = False
        return sets
    
    @staticmethod
    def Shells(cell: topologic.Cell) -> list:
        """
        Returns the shells of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.

        Returns
        -------
        list
            The list of shells.

        """
        if not isinstance(cell, topologic.Cell):
            return None
        shells = []
        _ = cell.Shells(None, shells)
        return shells

    @staticmethod
    def Sphere(origin: topologic.Vertex = None, radius: float = 0.5, uSides: int = 16, vSides: int = 8, direction: list = [0,0,1],
                   placement: str = "center", tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a sphere.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin location of the sphere. The default is None which results in the sphere being placed at (0,0,0).
        radius : float , optional
            The radius of the sphere. The default is 0.5.
        uSides : int , optional
            The number of sides along the longitude of the sphere. The default is 16.
        vSides : int , optional
            The number of sides along the latitude of the sphere. The default is 8.
        direction : list , optional
            The vector representing the up direction of the sphere. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the sphere. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cell
            The created sphere.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        c = Wire.Circle(origin=origin, radius=radius, sides=vSides, fromAngle=90, toAngle=270, close=False, direction=[0, 1, 0], placement="center")
        s = Topology.Spin(c, origin=origin, triangulate=False, direction=[0,0,1], degree=360, sides=uSides, tolerance=tolerance)
        if s.Type() == topologic.CellComplex.Type():
            s = s.ExternalBoundary()
        if s.Type() == topologic.Shell.Type():
            s = topologic.Cell.ByShell(s)
        if placement.lower() == "bottom":
            s = Topology.Translate(s, 0, 0, radius)
        elif placement.lower() == "lowerleft":
            s = Topology.Translate(s, radius, radius, radius)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        s = Topology.Rotate(s, origin, 0, 1, 0, theta)
        s = Topology.Rotate(s, origin, 0, 0, 1, phi)
        return s
    
    @staticmethod
    def SurfaceArea(cell: topologic.Cell, mantissa: int = 4) -> float:
        """
        Returns the surface area of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The cell.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        area : float
            The surface area of the input cell.

        """
        return Cell.Area(cell=cell, mantissa=mantissa)

    @staticmethod
    def Torus(origin: topologic.Vertex = None, majorRadius: float = 0.5, minorRadius: float = 0.125, uSides: int = 16, vSides: int = 8, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> topologic.Cell:
        """
        Creates a torus.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin location of the torus. The default is None which results in the torus being placed at (0,0,0).
        majorRadius : float , optional
            The major radius of the torus. The default is 0.5.
        minorRadius : float , optional
            The minor radius of the torus. The default is 0.1.
        uSides : int , optional
            The number of sides along the longitude of the torus. The default is 16.
        vSides : int , optional
            The number of sides along the latitude of the torus. The default is 8.
        direction : list , optional
            The vector representing the up direction of the torus. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the torus. This can be "bottom", "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cell
            The created torus.

        """
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        c = Wire.Circle(origin=origin, radius=minorRadius, sides=vSides, fromAngle=0, toAngle=360, close=False, direction=[0, 1, 0], placement="center")
        c = Topology.Translate(c, abs(majorRadius-minorRadius), 0, 0)
        s = Topology.Spin(c, origin=origin, triangulate=False, direction=[0,0,1], degree=360, sides=uSides, tolerance=tolerance)
        if s.Type() == topologic.Shell.Type():
            s = topologic.Cell.ByShell(s)
        if placement.lower() == "bottom":
            s = Topology.Translate(s, 0, 0, majorRadius)
        elif placement.lower() == "lowerleft":
            s = Topology.Translate(s, majorRadius, majorRadius, minorRadius)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        s = Topology.Rotate(s, origin, 0, 1, 0, theta)
        s = Topology.Rotate(s, origin, 0, 0, 1, phi)
        return s
    
    @staticmethod
    def Vertices(cell: topologic.Cell) -> list:
        """
        Returns the vertices of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(cell, topologic.Cell):
            return None
        vertices = []
        _ = cell.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Volume(cell: topologic.Cell, mantissa: int = 4) -> float:
        """
        Returns the volume of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.
        manitssa: int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The volume of the input cell.

        """
        if not cell:
            return None
        return round(topologic.CellUtility.Volume(cell), mantissa)

    @staticmethod
    def Wires(cell: topologic.Cell) -> list:
        """
        Returns the wires of the input cell.

        Parameters
        ----------
        cell : topologic.Cell
            The input cell.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(cell, topologic.Cell):
            return None
        wires = []
        _ = cell.Wires(None, wires)
        return wires

