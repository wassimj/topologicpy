import topologicpy
import topologic
class Cluster(topologic.Cluster):
    @staticmethod
    def ByTopologies(topologies):
        """
        Description
        -----------
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
        Description
        __________
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
        _ = cluster.Cells(None, cellComplexes)
        return cellComplexes

    @staticmethod
    def Cells(cluster):
        """
        Description
        __________
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
        Description
        __________
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
        Description
        __________
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
    def MysticRose(wire=None, origin=None, radius=1, sides=16, perimeter=True, dirX=0, dirY=0, dirZ=1, placement="center", tolerance=0.0001):
        """
        Description
        ----------
        Creates a mystic rose.

        Parameters
        ----------
        wire : topologic.Wire , optional
            The input Wire. if set to None, a circle with the input parameters is created. Otherwise, the input parameters are ignored.
        origin : topologic.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int, optional
            The number of sides of the circle. The default is 16.
        perimeter : bool, optional
            If True, the perimeter edges are included in the output. The default is True.
        dirX : float , optional
            The X component of the vector representing the up direction of the circle. The default is 0.
        dirY : float , optional
            The Y component of the vector representing the up direction of the circle. The default is 0.
        dirZ : float , optional
            The Z component of the vector representing the up direction of the circle. The default is 1.
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
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
            wire = Wire.Circle(origin=origin, radius=radius, sides=sides, fromAngle=0, toAngle=360, close=True, dirX=dirX,dirY=dirY, dirZ=dirZ, placement=placement, tolerance=tolerance)
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
        Description
        __________
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
    def Vertices(cluster):
        """
        Description
        __________
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
        Description
        __________
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

    