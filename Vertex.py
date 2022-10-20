import topologic
import collections

class Vertex(topologic.Vertex):
    @staticmethod
    def ByCoordinates(x, y, z):
        """
        Description
        -----------
        Creates a topologic vertex at the coordinates specified by the x, y, z inputs.

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
            The created topologic vertex.

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
        Returns the coordinates of the input topologic vertex in the format specified by the outputType input parameter. Accuracy is specified by the mantissa input parameter.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic vertex.
        outputType : string, optional
            The desired output type. Could be any permutation or substring of "XYZ" or the string "Matrix". The default is "XYZ". The input is case insensitive and the coordinates will be returned in the specified order.
        mantissa : int, optional
            The desired mantissa. The default is 3.

        Returns
        -------
        list
            The coordinates of the input topologic vertex.

        """
        if vertex:
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
                '''
                if outputType == "xyz":
                    output = [x,y,z]
                elif outputType == "xy":
                    output = [x,y]
                elif outputType == "xz":
                    output = [x,z]
                elif outputType == "yz":
                    output = [y,z]
                elif outputType == "x":
                    output = x
                elif outputType == "y":
                    output = y
                elif outputType == "z":
                    output = z
                '''
            return output
        else:
            return None

    
    @staticmethod
    def Distance(vertex, topology, mantissa=3):
        """
        Description
        -----------
        Returns the distance between the input topologic vertex and the input topologic topology. Accuracy is specified by the mantissa input parameter.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic vertex.
        topology : topologic.Topology
            The topologic topology.
        mantissa: int, optional
            The desired mantissa. The default is 3.

        Returns
        -------
        float
            The distance between the input vertex and the input topology.

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
        Returns the list of Cells found in the input topologic topology that enclose the input topologic vertex. If exclusive is set to True, only a list of a maximum of one topologic cell is returned.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic vertex.
        topology : topologic.Topology
            The topologic topology.
        exclusive : boolean, optional
            If set to True, return only the first found enclosing topologic cell. The default is True.
        tolerance : float, optional
            The tolerance for computing if the input vertex is enclosed in a cell. The default is 0.0001.

        Raises
        ------
        Exception
            Raises an exception if the input topology does not contain any cells.

        Returns
        -------
        [topologic.Cell]
            The list of enclosing topologic cells.

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
        Returns the topologic vertex found in the input topologic topology that is the nearest to the input topologic vertex. If useKDTree is set to True, the algorithm uses the KDTree search method.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic vertex.
        topology : topologic.Topology
            The topologic topology to search within for the nearest topologic vertex.
        useKDTree : boolean, optional
            if set to True, the algorithm will use a KDTree method to search for the nearest topologic vertex. The default is True.

        Returns
        -------
        topologic.Vertex
            The nearest topologic vertex.

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
        Returns a topologic vertex that is the projection of the input vertex unto the input face. The direction of the projection is the normal of the input face.

        Parameters
        ----------
        vertex : topologic.Vertex
            The topologic vertex.
        face : topologic.Face
            the topologic face.

        Returns
        -------
        topologic.Vertex
            The projected topologic vertex of the input topologic vertex unto the input topologic face.

        """
        projected_vertex = None
        if vertex and face:
            if (face.Type() == topologic.Face.Type()) and (vertex.Type() == topologic.Vertex.Type()):
                projected_vertex = (topologic.FaceUtility.ProjectToSurface(face, vertex))
        return projected_vertex