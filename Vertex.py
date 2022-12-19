import topologicpy
import topologic
from topologicpy.Face import Face
from topologicpy.Topology import Topology
import collections

class Vertex(Topology):
    @staticmethod
    def ByCoordinates(x, y, z):
        """
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        x : float
            The X coordinate.
        y : float
            The Y coordinate.
        z : float
            The Z coodinate.

        Returns
        -------
        topologic.Vertex
            The created vertex.

        """
        vertex = None
        try:
            vertex = topologic.Vertex.ByCoordinates(x, y, z)
        except:
            vertex = None
        return vertex
    
    @staticmethod
    def Coordinates(vertex, outputType="xyz", mantissa=4):
        """
        Returns the coordinates of the input vertex.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        outputType : string, optional
            The desired output type. Could be any permutation or substring of "xyz" or the string "matrix". The default is "xyz". The input is case insensitive and the coordinates will be returned in the specified order.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The coordinates of the input vertex.

        """
        if not isinstance(vertex, topologic.Vertex):
            return None
        x = round(vertex.X(), mantissa)
        y = round(vertex.Y(), mantissa)
        z = round(vertex.Z(), mantissa)
        matrix = [[1,0,0,x],
                [0,1,0,y],
                [0,0,1,z],
                [0,0,0,1]]
        output = []
        outputType = outputType.lower()
        if outputType == "matrix":
            return matrix
        else:
            outputType = list(outputType)
            for axis in outputType:
                if axis == "x":
                    output.append(x)
                elif axis == "y":
                    output.append(y)
                elif axis == "z":
                    output.append(z)
        return output

    
    @staticmethod
    def Distance(vertex, topology, mantissa=4):
        """
        Returns the distance between the input vertex and the input topology.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        topology : topologic.Topology
            The input topology.
        mantissa: int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The distance between the input vertex and the input topology.

        """
        if not isinstance(vertex, topologic.Vertex) or not isinstance(topology, topologic.Topology):
            return None
        return round(topologic.VertexUtility.Distance(vertex, topology), mantissa)
    
    @staticmethod
    def EnclosingCell(vertex, topology, exclusive=True, tolerance=0.0001):
        """
        Returns the list of Cells found in the input topology that enclose the input vertex.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        topology : topologic.Topology
            The input topology.
        exclusive : bool , optional
            If set to True, return only the first found enclosing cell. The default is True.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a cell. The default is 0.0001.

        Returns
        -------
        list
            The list of enclosing cells.

        """
        
        def boundingBox(cell):
            vertices = []
            _ = cell.Vertices(None, vertices)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(aVertex.X())
                y.append(aVertex.Y())
                z.append(aVertex.Z())
            return ([min(x), min(y), min(z), max(x), max(y), max(z)])
        
        if isinstance(topology, topologic.Cell):
            cells = [topology]
        elif isinstance(topology, topologic.Cluster) or isinstance(topology, topologic.CellComplex):
            cells = []
            _ = topology.Cells(None, cells)
        else:
            return None
        if len(cells) < 1:
            return None
        enclosingCells = []
        for i in range(len(cells)):
            bbox = boundingBox(cells[i])
            if ((vertex.X() < bbox[0]) or (vertex.Y() < bbox[1]) or (vertex.Z() < bbox[2]) or (vertex.X() > bbox[3]) or (vertex.Y() > bbox[4]) or (vertex.Z() > bbox[5])) == False:
                if topologic.CellUtility.Contains(cells[i], vertex, tolerance) == 0:
                    if exclusive:
                        return([cells[i]])
                    else:
                        enclosingCells.append(cells[i])
        return enclosingCells

    @staticmethod
    def Index(vertex, vertices, strict=False, tolerance=0.0001):
        """
        Returns index of the input vertex in the input list of vertices

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        vertices : list
            The input list of vertices.
        strict : bool , optional
            If set to True, the vertex must be strictly identical to the one found in the list. Otherwise, a distance comparison is used. The default is False.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a cell. The default is 0.0001.

        Returns
        -------
        int
            The index of the input vertex in the input list of vertices.

        """
        from topologicpy.Topology import Topology
        if not isinstance(vertex, topologic.Vertex):
            return None
        if not isinstance(vertices, list):
            return None
        new_vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if not len(new_vertices) == len(vertices):
            return None
        for i in range(len(vertices)):
            if strict:
                if Topology.IsSame(vertex, vertices[i]):
                    return i
            else:
                if Vertex.Distance(vertex, vertices[i]) < tolerance:
                    return i
        return None

    @staticmethod
    def IsInside(vertex, topology, tolerance=0.0001):
        """
        Returns True if the input vertex is inside the input topology. Returns False otherwise.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        topology : topologic.Topology
            The input topology.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a cell. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertex is inside the input topology. False otherwise.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        if not isinstance(vertex, topologic.Vertex):
            return None
        if not isinstance(topology, topologic.Topology):
            return None

        if isinstance(topology, topologic.Vertex):
            return topologic.VertexUtility.Distance(vertex, topology) < tolerance
        elif isinstance(topology, topologic.Edge):
            try:
                parameter = topologic.EdgeUtility.ParameterAtPoint(topology, vertex)
            except:
                parameter = 400 #aribtrary large number greater than 1
            return 0 <= parameter <= 1
        elif isinstance(topology, topologic.Wire):
            edges = Wire.Edges(topology)
            for edge in edges:
                if Vertex.IsInside(vertex, edge, tolerance):
                    return True
            return False
        elif isinstance(topology, topologic.Face):
            return topologic.FaceUtility.IsInside(topology, vertex, tolerance)
        elif isinstance(topology, topologic.Shell):
            faces = Shell.Faces(topology)
            for face in faces:
                if Vertex.IsInside(vertex, face, tolerance):
                    return True
            return False
        elif isinstance(topology, topologic.Cell):
            return topologic.CellUtility.Contains(topology, vertex, tolerance) == 0
        elif isinstance(topology, topologic.CellComplex):
            cells = CellComplex.Cells(topology)
            faces = CellComplex.Faces(topology)
            edges = CellComplex.Edges(topology)
            vertices = CellComplex.Vertices(topology)
            subtopologies = cells + faces + edges + vertices
            for subtopology in subtopologies:
                if Vertex.IsInside(vertex, subtopology, tolerance):
                    return True
            return False
        elif isinstance(topology, topologic.Cluster):
            cells = Cluster.Cells(topology)
            faces = Cluster.Faces(topology)
            edges = Cluster.Edges(topology)
            vertices = Cluster.Vertices(topology)
            subtopologies = cells + faces + edges + vertices
            for subtopology in subtopologies:
                if Vertex.IsInside(vertex, subtopology, tolerance):
                    return True
            return False
        return False
    
    @staticmethod
    def NearestVertex(vertex, topology, useKDTree=True):
        """
        Returns the vertex found in the input topology that is the nearest to the input vertex.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        topology : topologic.Topology
            The input topology to be searched for the nearest vertex.
        useKDTree : bool , optional
            if set to True, the algorithm will use a KDTree method to search for the nearest vertex. The default is True.

        Returns
        -------
        topologic.Vertex
            The nearest vertex.

        """        
        def SED(a, b):
            """Compute the squared Euclidean distance between X and Y."""
            p1 = (a.X(), a.Y(), a.Z())
            p2 = (b.X(), b.Y(), b.Z())
            return sum((i-j)**2 for i, j in zip(p1, p2))
        
        BT = collections.namedtuple("BT", ["value", "left", "right"])
        BT.__doc__ = """
        A Binary Tree (BT) with a node value, and left- and
        right-subtrees.
        """
        def firstItem(v):
            return v.X()
        def secondItem(v):
            return v.Y()
        def thirdItem(v):
            return v.Z()

        def itemAtIndex(v, index):
            if index == 0:
                return v.X()
            elif index == 1:
                return v.Y()
            elif index == 2:
                return v.Z()

        def sortList(vertices, index):
            if index == 0:
                vertices.sort(key=firstItem)
            elif index == 1:
                vertices.sort(key=secondItem)
            elif index == 2:
                vertices.sort(key=thirdItem)
            return vertices
        
        def kdtree(topology):
            assert isinstance(topology, topologic.Topology), "Vertex.NearestVertex: The input is not a Topology."
            vertices = []
            _ = topology.Vertices(None, vertices)
            assert (len(vertices) > 0), "Vertex.NearestVertex: Could not find any vertices in the input Topology"

            """Construct a k-d tree from an iterable of vertices.

            This algorithm is taken from Wikipedia. For more details,

            > https://en.wikipedia.org/wiki/K-d_tree#Construction

            """
            # k = len(points[0])
            k = 3

            def build(*, vertices, depth):
                if len(vertices) == 0:
                    return None
                #points.sort(key=operator.itemgetter(depth % k))
                vertices = sortList(vertices, (depth % k))

                middle = len(vertices) // 2
                
                return BT(
                    value = vertices[middle],
                    left = build(
                        vertices=vertices[:middle],
                        depth=depth+1,
                    ),
                    right = build(
                        vertices=vertices[middle+1:],
                        depth=depth+1,
                    ),
                )

            return build(vertices=list(vertices), depth=0)
        
        NNRecord = collections.namedtuple("NNRecord", ["vertex", "distance"])
        NNRecord.__doc__ = """
        Used to keep track of the current best guess during a nearest
        neighbor search.
        """

        def find_nearest_neighbor(*, tree, vertex):
            """Find the nearest neighbor in a k-d tree for a given vertex.
            """
            k = 3 # Forcing k to be 3 dimensional
            best = None
            def search(*, tree, depth):
                """Recursively search through the k-d tree to find the nearest neighbor.
                """
                nonlocal best

                if tree is None:
                    return
                distance = SED(tree.value, vertex)
                if best is None or distance < best.distance:
                    best = NNRecord(vertex=tree.value, distance=distance)

                axis = depth % k
                diff = itemAtIndex(vertex,axis) - itemAtIndex(tree.value,axis)
                if diff <= 0:
                    close, away = tree.left, tree.right
                else:
                    close, away = tree.right, tree.left

                search(tree=close, depth=depth+1)
                if diff**2 < best.distance:
                    search(tree=away, depth=depth+1)

            search(tree=tree, depth=0)
            return best.vertex
        
        if useKDTree:
            tree = kdtree(topology)
            return find_nearest_neighbor(tree=tree, vertex=vertex)
        else:
            vertices = []
            _ = topology.Vertices(None, vertices)
            distances = []
            indices = []
            for i in range(len(vertices)):
                distances.append(SED(vertex, vertices[i]))
                indices.append(i)
            sorted_indices = [x for _, x in sorted(zip(distances, indices))]
        return vertices[sorted_indices[0]]

    
    @staticmethod
    def Project(vertex, face, direction=None, mantissa=4, tolerance=0.0001):
        """
        Returns a vertex that is the projection of the input vertex unto the input face.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex to project unto the input face.
        face : topologic.Face
            The input face that receives the projection of the input vertex.
        direction : vector, optional
            The direction in which to project the input vertex unto the input face. If not specified, the direction of the projection is the normal of the input face. The default is None.
        mantissa : int , optional
            The length of the desired mantissa. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            The projected vertex.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if not isinstance(vertex, topologic.Vertex):
            return None
        if not isinstance(face, topologic.Face):
            return None
        if not direction:
            direction = Vector.Reverse(Face.NormalAtParameters(face, 0.5, 0.5, "XYZ", mantissa))
        if topologic.FaceUtility.IsInside(face, vertex, tolerance):
            return vertex
        d = topologic.VertexUtility.Distance(vertex, face)*10
        far_vertex = topologic.TopologyUtility.Translate(vertex, direction[0]*d, direction[1]*d, direction[2]*d)
        if topologic.VertexUtility.Distance(vertex, far_vertex) > tolerance:
            e = topologic.Edge.ByStartVertexEndVertex(vertex, far_vertex)
            pv = face.Intersect(e, False)
            if not pv:
                far_vertex = topologic.TopologyUtility.Translate(vertex, -direction[0]*d, -direction[1]*d, -direction[2]*d)
                if topologic.VertexUtility.Distance(vertex, far_vertex) > tolerance:
                    e = topologic.Edge.ByStartVertexEndVertex(vertex, far_vertex)
                    pv = face.Intersect(e, False)
            return pv
        else:
            return None

    @staticmethod
    def X(vertex, mantissa=4):
        """
        Returns the X coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The X coordinate of the input vertex.

        """
        if not isinstance(vertex, topologic.Vertex):
            return None
        return round(vertex.X(), mantissa)

    @staticmethod
    def Y(vertex, mantissa=4):
        """
        Returns the Y coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The Y coordinate of the input vertex.

        """
        if not isinstance(vertex, topologic.Vertex):
            return None
        return round(vertex.Y(), mantissa)

    @staticmethod
    def Z(vertex, mantissa=4):
        """
        Returns the Z coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The Z coordinate of the input vertex.

        """
        if not isinstance(vertex, topologic.Vertex):
            return None
        return round(vertex.Z(), mantissa)
           