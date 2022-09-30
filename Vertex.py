import topologic
from mathutils import Matrix
import collections

class Vertex(topologic.Vertex):
    @staticmethod
    def VertexByCoordinates(x, y, z):
        """
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.

        Returns
        -------
        vert : TYPE
            DESCRIPTION.

        """
        # x = item[0]
        # y = item[1]
        # z = item[2]
        vert = None
        try:
            vert = topologic.Vertex.ByCoordinates(x, y, z)
        except:
            vert = None
        return vert
    
    @staticmethod
    def VertexCoordinates(vertex, outputType, mantissa):
        """
        Parameters
        ----------
        vertex : TYPE
            DESCRIPTION.
        outputType : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # vertex, outputType, mantissa = item
        if vertex:
            output = None
            x = round(vertex.X(), mantissa)
            y = round(vertex.Y(), mantissa)
            z = round(vertex.Z(), mantissa)
            matrix = Matrix([[1,0,0,x],
                    [0,1,0,y],
                    [0,0,1,z],
                    [0,0,0,1]])
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
            if outputType == "Matrix":
                return Matrix()
            else:
                return None

    
    @staticmethod
    def VertexDistance(v, t):
        """
        Parameters
        ----------
        v : TYPE
            DESCRIPTION.
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        dist : TYPE
            DESCRIPTION.

        """
        # v, t = item
        assert isinstance(v, topologic.Vertex), "Vertex.Distance: input is not a Topologic Vertex"
        assert isinstance(t, topologic.Topology), "Vertex.Distance: input is not a Topologic Topology"
        if v and t:
            dist = topologic.VertexUtility.Distance(v, t)
        else:
            dist = None
        return dist
    
    @staticmethod
    def VertexEnclosingCell(vertex, topology, exclusive, tolerance=0.0001):
        """
        Parameters
        ----------
        vertex : TYPE
            DESCRIPTION.
        topology : TYPE
            DESCRIPTION.
        exclusive : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        enclosingCells : TYPE
            DESCRIPTION.

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
        
        # vertex, topology, exclusive, tolerance = input
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
    def VertexNearestVertex(vertex, topology, useKDTree):
        """
        Parameters
        ----------
        vertex : TYPE
            DESCRIPTION.
        topology : TYPE
            DESCRIPTION.
        useKDTree : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # vertex, topology, useKDTree = input
        
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
    def VertexProject(vertex, face):
        """
        Parameters
        ----------
        vertex : TYPE
            DESCRIPTION.
        face : TYPE
            DESCRIPTION.

        Returns
        -------
        projected_vertex : TYPE
            DESCRIPTION.

        """
        # vertex, face = item
        projected_vertex = None
        if vertex and face:
            if (face.Type() == topologic.Face.Type()) and (vertex.Type() == topologic.Vertex.Type()):
                projected_vertex = (topologic.FaceUtility.ProjectToSurface(face, vertex))
        return projected_vertex