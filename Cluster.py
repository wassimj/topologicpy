import topologicpy
import topologic
class Cluster(topologic.Cluster):
    @staticmethod
    def ByTopologies(topologies):
        """
        Creates a topologic Cluster from the input list of topologies.

        Parameters
        ----------
        topologies : list
            The list of topologies.

        Returns
        -------
        topologic.Cluster
            The created topologic Cluster.

        """
        assert isinstance(topologies, list), "Cluster.ByTopologies - Error: Input is not a list"
        topologyList = [x for x in topologies if isinstance(x, topologic.Topology)]
        return topologic.Cluster.ByTopologies(topologyList, False)

    @staticmethod
    def CellComplexes(cluster):
        """
        Returns the cellComplexes of the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of cellComplexes.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        cellComplexes = []
        _ = cluster.CellComplexes(None, cellComplexes)
        return cellComplexes

    @staticmethod
    def Cells(cluster):
        """
        Returns the cells of the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of cells.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        cells = []
        _ = cluster.Cells(None, cells)
        return cells

    @staticmethod
    def Edges(cluster):
        """
        Returns the edges of the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of edges.

        """ 
        if not isinstance(cluster, topologic.Cluster):
            return None
        edges = []
        _ = cluster.Edges(None, edges)
        return edges

    @staticmethod
    def Faces(cluster):
        """
        Returns the faces of the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of faces.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        faces = []
        _ = cluster.Faces(None, faces)
        return faces

    @staticmethod
    def FreeCells(cluster):
        """
        Returns the free cells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of free cells.

        """
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology

        if not isinstance(cluster, topologic.Cluster):
            return None
        allCells = []
        _ = cluster.Cells(None, allCells)
        allCellsCluster = Cluster.ByTopologies(allCells)
        freeCells = []
        cellComplexes = []
        _ = cluster.CellComplexes(None, cellComplexes)
        cellComplexesCells = []
        for cellComplex in cellComplexes:
            tempCells = CellComplex.Cells(cellComplex)
            cellComplexesCells += tempCells
        if len(cellComplexesCells) == 0:
            return allCells
        cellComplexesCluster = Cluster.ByTopologies(cellComplexesCells)
        resultingCluster = Topology.Boolean(allCellsCluster, cellComplexesCluster, operation="Difference")
        if isinstance(resultingCluster, topologic.Cell):
            return [resultingCluster]
        return Topology.SubTopologies(resultingCluster, subTopologyType="cell")
    
    @staticmethod
    def FreeShells(cluster):
        """
        Returns the free shells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of free shells.

        """
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        if not isinstance(cluster, topologic.Cluster):
            return None
        allShells = []
        _ = cluster.Shells(None, allShells)
        allShellsCluster = Cluster.ByTopologies(allShells)
        cells = []
        _ = cluster.Cells(None, cells)
        cellsShells = []
        for cell in cells:
            tempShells = Cell.Shells(cell)
            cellsShells += tempShells
        if len(cellsShells) == 0:
            return allShells
        cellsCluster = Cluster.ByTopologies(cellsShells)
        resultingCluster = Topology.Boolean(allShellsCluster, cellsCluster, operation="Difference")
        if isinstance(resultingCluster, topologic.Shell):
            return [resultingCluster]
        return Topology.SubTopologies(resultingCluster, subTopologyType="shell")
    
    @staticmethod
    def FreeFaces(cluster):
        """
        Returns the free faces of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of free faces.

        """
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology
        if not isinstance(cluster, topologic.Cluster):
            return None
        allFaces = []
        _ = cluster.Faces(None, allFaces)
        allFacesCluster = Cluster.ByTopologies(allFaces)
        shells = []
        _ = cluster.Shells(None, shells)
        shellFaces = []
        for shell in shells:
            tempFaces = Shell.Faces(shell)
            shellFaces += tempFaces
        if len(shellFaces) == 0:
            return allFaces
        shellCluster = Cluster.ByTopologies(shellFaces)
        resultingCluster = Topology.Boolean(allFacesCluster, shellCluster, operation="Difference")
        if isinstance(resultingCluster, topologic.Face):
            return [resultingCluster]
        return Topology.SubTopologies(resultingCluster, subTopologyType="face")

    @staticmethod
    def FreeWires(cluster):
        """
        Returns the free wires of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of free wires.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(cluster, topologic.Cluster):
            return None
        allWires = []
        _ = cluster.Wires(None, allWires)
        allWiresCluster = Cluster.ByTopologies(allWires)
        faces = []
        _ = cluster.Faces(None, faces)
        facesWires = []
        for face in faces:
            tempWires = Face.Wires(face)
            facesWires += tempWires
        if len(facesWires) == 0:
            return allWires
        facesCluster = Cluster.ByTopologies(facesWires)
        resultingCluster = Topology.Boolean(allWiresCluster, facesCluster, operation="Difference")
        if isinstance(resultingCluster, topologic.Wire):
            return [resultingCluster]
        return Topology.SubTopologies(resultingCluster, subTopologyType="wire")
    
    @staticmethod
    def FreeEdges(cluster):
        """
        Returns the free edges of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of free edges.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(cluster, topologic.Cluster):
            return None
        allEdges = []
        _ = cluster.Edges(None, allEdges)
        allEdgesCluster = Cluster.ByTopologies(allEdges)
        wires = []
        _ = cluster.Wires(None, wires)
        wireEdges = []
        for wire in wires:
            tempEdges = Wire.Edges(wire)
            wireEdges += tempEdges
        if len(wireEdges) == 0:
            return allEdges
        wireCluster = Cluster.ByTopologies(wireEdges)
        resultingCluster = Topology.Boolean(allEdgesCluster, wireCluster, operation="Difference")
        if isinstance(resultingCluster, topologic.Edge):
            return [resultingCluster]
        return Topology.SubTopologies(resultingCluster, subTopologyType="edge")
    
    @staticmethod
    def FreeVertices(cluster):
        """
        Returns the free vertices of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of free vertices.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not isinstance(cluster, topologic.Cluster):
            return None
        allVertices = []
        _ = cluster.Vertices(None, allVertices)
        allVerticesCluster = Cluster.ByTopologies(allVertices)
        edges = []
        _ = cluster.Edges(None, edges)
        edgesVertices = []
        for edge in edges:
            tempVertices = Edge.Vertices(edge)
            edgesVertices += tempVertices
        if len(edgesVertices) == 0:
            return allVertices
        edgesCluster = Cluster.ByTopologies(edgesVertices)
        resultingCluster = Topology.Boolean(allVerticesCluster, edgesCluster, operation="Difference")
        if isinstance(resultingCluster, topologic.Vertex):
            return [resultingCluster]
        return Topology.SubTopologies(resultingCluster, subTopologyType="vertex")
    
    @staticmethod
    def HighestType(cluster):
        """
        Returns the type of the highest dimension subtopology found in the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        int
            The type of the highest dimension subtopology found in the input cluster.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        cellComplexes = Cluster.CellComplexes(cluster)
        if len(cellComplexes) > 0:
            return topologic.CellComplex.Type()
        cells = Cluster.Cells(cluster)
        if len(cells) > 0:
            return topologic.Cell.Type()
        shells = Cluster.Shells(cluster)
        if len(shells) > 0:
            return topologic.Shell.Type()
        faces = Cluster.Faces(cluster)
        if len(faces) > 0:
            return topologic.Face.Type()
        wires = Cluster.Wires(cluster)
        if len(wires) > 0:
            return topologic.Wire.Type()
        edges = Cluster.Edges(cluster)
        if len(edges) > 0:
            return topologic.Edge.Type()
        vertices = Cluster.Vertices(cluster)
        if len(vertices) > 0:
            return topologic.Vertex.Type()

    @staticmethod
    def MysticRose(wire=None, origin=None, radius=1, sides=16, perimeter=True, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a mystic rose.

        Parameters
        ----------
        wire : topologic.Wire , optional
            The input Wire. if set to None, a circle with the input parameters is created. Otherwise, the input parameters are ignored.
        origin : topologic.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the mystic rose. The default is 1.
        sides : int, optional
            The number of sides of the mystic rose. The default is 16.
        perimeter : bool, optional
            If True, the perimeter edges are included in the output. The default is True.
        direction : list , optional
            The vector representing the up direction of the mystic rose. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the mystic rose. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.cluster
            The created mystic rose (cluster of edges).

        """
        import topologicpy
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from itertools import combinations

        if not wire:
            wire = Wire.Circle(origin=origin, radius=radius, sides=sides, fromAngle=0, toAngle=360, close=True, direction=direction, placement=placement, tolerance=tolerance)
        if not Wire.IsClosed(wire):
            return None
        vertices = Wire.Vertices(wire)
        indices = list(range(len(vertices)))
        combs = [[comb[0],comb[1]] for comb in combinations(indices, 2) if not (abs(comb[0]-comb[1]) == 1) and not (abs(comb[0]-comb[1]) == len(indices)-1)]
        edges = []
        if perimeter:
            edges = Wire.Edges(wire)
        for comb in combs:
            edges.append(Edge.ByVertices([vertices[comb[0]], vertices[comb[1]]]))
        return Cluster.ByTopologies(edges)

    @staticmethod
    def Shells(cluster):
        """
        Returns the shells of the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of shells.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        shells = []
        _ = cluster.Shells(None, shells)
        return shells

    @staticmethod
    def Simplify(cluster):
        """
        Simplifies the input cluster if possible. For example, if the cluster contains only one cell, that cell is returned.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        topologic.Topology or list
            The simplification of the cluster.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        resultingTopologies = []
        topCC = []
        _ = cluster.CellComplexes(None, topCC)
        topCells = []
        _ = cluster.Cells(None, topCells)
        topShells = []
        _ = cluster.Shells(None, topShells)
        topFaces = []
        _ = cluster.Faces(None, topFaces)
        topWires = []
        _ = cluster.Wires(None, topWires)
        topEdges = []
        _ = cluster.Edges(None, topEdges)
        topVertices = []
        _ = cluster.Vertices(None, topVertices)
        if len(topCC) == 1:
            cc = topCC[0]
            ccVertices = []
            _ = cc.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cc)
        if len(topCC) == 0 and len(topCells) == 1:
            cell = topCells[0]
            ccVertices = []
            _ = cell.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 1:
            shell = topShells[0]
            ccVertices = []
            _ = shell.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(shell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 1:
            face = topFaces[0]
            ccVertices = []
            _ = face.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(face)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 1:
            wire = topWires[0]
            ccVertices = []
            _ = wire.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(wire)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 1:
            edge = topEdges[0]
            ccVertices = []
            _ = wire.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(edge)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 0 and len(topVertices) == 1:
            vertex = topVertices[0]
            resultingTopologies.append(vertex)
        if len(resultingTopologies) == 1:
            return resultingTopologies[0]
        return cluster

    @staticmethod
    def Vertices(cluster):
        """
        Returns the vertices of the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        vertices = []
        _ = cluster.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Wires(cluster):
        """
        Returns the wires of the input cluster.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        wires = []
        _ = cluster.Wires(None, wires)
        return wires

    