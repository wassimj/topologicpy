from topologicpy import topologic
# from mathutils import Matrix This is Blender-specific. We need a workaround.
import collections

class Vertex(topologic.Vertex):
    @staticmethod
    def ByCoordinates(x, y, z):
        """
        Description
        -----------
        This method creates a topologic Vertex located at the coordinates specified by the x, y, z input parameters.

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
        vert : topologic.Vertex
            The created topologic Vertex.

        """
        vert = None
        try:
            vert = topologic.Vertex.ByCoordinates(x, y, z)
        except:
            vert = None
        return vert
    
    @staticmethod
    def Coordinates(vertex, outputType="XYZ", mantissa=3):
        """
        Description
        -----------
        This method returns the coordinates of the input topologic Vertex in the format specified by the outputType input parameter and with a mantissa specified by the mantissa input parameter.

        Parameters
        ----------
        vertex : topologic.Vertex.
            The topologic Vertex.
        outputType : string, optional
            The desired output type. Could be one of "XYZ", "XY", "XZ", "YZ", "X", "Y", "Z", "Matrix". The default is "XYZ".
        mantissa : int, optional
            The desired mantissa. The default is 3.

        Returns
        -------
        float, list, or matrix
            The coordinates of the input topologic Vertex.

        """
        if vertex:
            output = None
            x = round(vertex.X(), mantissa)
            y = round(vertex.Y(), mantissa)
            z = round(vertex.Z(), mantissa)
            matrix = [[1,0,0,x],
                    [0,1,0,y],
                    [0,0,1,z],
                    [0,0,0,1]]
            if outputType == "XYZ":
                output = [x,y,z]
            elif outputType == "XY":
                output = [x,y]
            elif outputType == "XZ":
                output = [x,z]
            elif outputType == "YZ":
                output = [y,z]
            elif outputType == "X":
                output = x
            elif outputType == "Y":
                output = y
            elif outputType == "Z":
                output = z
            elif outputType == "Matrix":
                output = matrix
            return output
        else:
            return None

    
    @staticmethod
    def Distance(vertex, topology, mantissa=3):
        """
        Description
        -----------
        This method returns the distance between the input topologic Vertex and the input topology specified by the mantissa input parameter.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic Vertex.
        ttopology : topologic.Topology
            The topologic Topology.
        mantissa: int, optional.
            The desired mantissa. The default is 3.

        Returns
        -------
        dist : float
            The distance between the topologic Vertex and the topologic Topology.

        """
        assert isinstance(vertex, topologic.Vertex), "Vertex.Distance: input is not a Topologic Vertex"
        assert isinstance(topology, topologic.Topology), "Vertex.Distance: input is not a Topologic Topology"
        if vertex and topology:
            dist = round(topologic.VertexUtility.Distance(vertex, topology), mantissa)
        else:
            dist = None
        return dist
    
    @staticmethod
    def EnclosingCell(vertex, topology, exclusive=True, tolerance=0.0001):
        """
        Description
        -----------
        This method returns the list of Cells found in the input topology parameter that enclose the input vertex parameter. If exclusive is set to True, only a list of a maximum of one topologic Cell is returned.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic Vertex.
        topology : topologic.Topology
            The topologic Topology.
        exclusive : boolean, optional
            If set to True, return only the first found enclosing topologic Cell. The default is True.
        tolerance : float, optional
            The tolerance for finding if the vertex is enclosed in a Cell. The default is 0.0001.

        Raises
        ------
        Exception
            Raises an error if the input Topology does not contain any Cells.

        Returns
        -------
        enclosingCells : [topologic.Cell]
            The list of enclosing topologic Cells.

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
            raise Exception("Vertex.EnclosingCell - Error: Input topology does not contain any Cells.")
        if len(cells) < 1:
            raise Exception("Vertex.EnclosingCell - Error: Input topology does not contain any Cells.")
        enclosingCells = []
        for i in range(len(cells)):
            bbox = boundingBox(cells[i])
            minX = bbox[0]
            if ((vertex.X() < bbox[0]) or (vertex.Y() < bbox[1]) or (vertex.Z() < bbox[2]) or (vertex.X() > bbox[3]) or (vertex.Y() > bbox[4]) or (vertex.Z() > bbox[5])) == False:
                if topologic.CellUtility.Contains(cells[i], vertex, tolerance) == 0:
                    if exclusive:
                        return([cells[i]])
                    else:
                        enclosingCells.append(cells[i])
        return enclosingCells

    
    @staticmethod
    def NearestVertex(vertex, topology, useKDTree=True):
        """
        Description
        -----------
        This method returns the topologic Vertex found in the input topology parameter that is the nearest to the input vertex parameter. If useKDTree is set to True, the algorithm uses the KDTree search method.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic Vertex.
        topology : topologic.Topology
            THe topologic Topology to search for the nearest topologic Vertex.
        useKDTree : boolean, optional
            if set to True, the algorithm will use a KDTree method to search for the nearest topologic Vertex.

        Returns
        -------
        topologic.Vertex
            The nearest topologic Vertex.

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
    def Project(vertex, face):
        """
        Description
        -----------
        This method projects the input vertex parameter unto the input face parameter.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic Vertex.
        face : topologic.Face
            the topologic Face.

        Returns
        -------
        projected_vertex : topologic.Vertex
            The projected topologic Vertex of the input topologic Vertex unto the input topologic Face.

        """
        projected_vertex = None
        if vertex and face:
            if (face.Type() == topologic.Face.Type()) and (vertex.Type() == topologic.Vertex.Type()):
                projected_vertex = (topologic.FaceUtility.ProjectToSurface(face, vertex))
        return projected_vertex