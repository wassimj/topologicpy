#from types import NoneType
import topologicpy
import topologic
from topologicpy.Topology import Topology
import math

class Shell(Topology):
    @staticmethod
    def ByFaces(faces, tolerance=0.0001):
        """
        Creates a shell from the input list of faces.

        Parameters
        ----------
        faces : list
            The input list of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Shell
            The created Shell.

        """
        if not isinstance(faces, list):
            return None
        faceList = [x for x in faces if isinstance(x, topologic.Face)]
        shell = topologic.Shell.ByFaces(faceList, tolerance)
        if not shell:
            result = faceList[0]
            remainder = faceList[1:]
            cluster = topologic.Cluster.ByTopologies(remainder, False)
            result = result.Merge(cluster, False)
            if result.Type() > 16:
                returnShells = []
                _ = result.Shells(None, returnShells)
                return returnShells
            else:
                return None
        else:
            return shell

    @staticmethod
    def ByFacesCluster(cluster):
        """
        Creates a shell from the input cluster of faces.

        Parameters
        ----------
        cluster : topologic.Cluster
            The input cluster of faces.

        Returns
        -------
        topologic.Shell
            The created shell.

        """
        if not isinstance(cluster, topologic.Cluster):
            return None
        faces = []
        _ = cluster.Faces(None, faces)
        return Shell.ByFaces(faces)

    @staticmethod
    def ByWires(wires, triangulate=True, tolerance=0.0001):
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
        Returns
        -------
        topologic.Shell
            The creates shell.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        if not isinstance(wires, list):
            return None
        wireList = [x for x in wires if isinstance(x, topologic.Wire)]
        faces = []
        for i in range(len(wireList)-1):
            wire1 = wireList[i]
            wire2 = wireList[i+1]
            if wire1.Type() < topologic.Edge.Type() or wire2.Type() < topologic.Edge.Type():
                return None
            if wire1.Type() == topologic.Edge.Type():
                w1_edges = [wire1]
            else:
                w1_edges = []
                _ = wire1.Edges(None, w1_edges)
            if wire2.Type() == topologic.Edge.Type():
                w2_edges = [wire2]
            else:
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
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e4])))
                    try:
                        e4 = Edge.ByVertices([e1.EndVertex(), e2.EndVertex()])
                    except:
                        e3 = Edge.ByVertices([e1.StartVertex(), e2.StartVertex()])
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3])))
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
                            faces.append(Face.ByWire(topologic.Wire.ByEdges([e1, e4, e2, e3])))
                        except:
                            faces.append(Face.ByWire(topologic.Wire.ByEdges([e1, e3, e2, e4])))
                    elif e3:
                            faces.append(Face.ByWire(topologic.Wire.ByEdges([e1, e3, e2])))
                    elif e4:
                            faces.append(Face.ByWire(topologic.Wire.ByEdges([e1, e4, e2])))
        return Shell.ByFaces(faces, tolerance)

    @staticmethod
    def ByWiresCluster(cluster, triangulate=True, tolerance=0.0001):
        """
        Creates a shell by lofting through the input cluster of wires

        Parameters
        ----------
        wires : topologic.Cluster
            The input cluster of wires.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Shell
            The creates shell.

        """
        from topologicpy.Cluster import Cluster
        if not cluster:
            return None
        if not isinstance(cluster, topologic.Cluster):
            return None
        wires = Cluster.Wires(cluster)
        return Shell.ByWires(wires, triangulate=triangulate, tolerance=tolerance)

    @staticmethod
    def Circle(origin=None, radius=0.5, sides=32, fromAngle=0, toAngle=360, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The  radius of the circle. The default is 0.5.
        sides : int , optional
            The number of sides of the circle. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Shell
            The created circle.
        """
        return Shell.Pie(origin=origin, radiusA=radius, radiusB=0, sides=sides, rings=1, fromAngle=fromAngle, toAngle=toAngle, direction=direction, placement=placement, tolerance=tolerance)

    @staticmethod
    def Delaunay(vertices, face=None):
        """
        Returns a delaunay partitioning of the input vertices. The vertices must be coplanar. See https://en.wikipedia.org/wiki/Delaunay_triangulation.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic.Face , optional
            The input face. If specified, the delaunay triangulation is clipped to the face.

        Returns
        -------
        shell
            A shell representing the delaunay triangulation of the input vertices.

        """
        import scipy
        from scipy.spatial import Delaunay
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from random import sample

        if not isinstance(vertices, list):
            return None
        vertices = [x for x in vertices if isinstance(x, topologic.Vertex)]
        if len(vertices) < 2:
            return None

        if not isinstance(face, topologic.Face):
            face_vertices = sample(vertices,3)
            tempFace = Face.ByWire(Wire.ByVertices(face_vertices))
            # Flatten the input face
            flatFace = Face.Flatten(tempFace)
        else:
            flatFace = Face.Flatten(face)
            faceVertices = Face.Vertices(face)
            vertices += faceVertices
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        # Create a Vertex at the world's origin (0,0,0)
        world_origin = Vertex.ByCoordinates(0,0,0)

        # Create a cluster of the input vertices
        verticesCluster = Cluster.ByTopologies(vertices)

        # Flatten the cluster using the same transformations
        verticesCluster = Topology.Translate(verticesCluster, -xTran, -yTran, -zTran)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=1, z=0, degree=-theta)

        flatVertices = Cluster.Vertices(verticesCluster)
        tempFlatVertices = []
        points = []
        for flatVertex in flatVertices:
            tempFlatVertices.append(Vertex.ByCoordinates(flatVertex.X(), flatVertex.Y(), 0))
            points.append([flatVertex.X(), flatVertex.Y()])
        flatVertices = tempFlatVertices
        delaunay = Delaunay(points)
        simplices = delaunay.simplices

        faces = []
        for simplex in simplices:
            tempTriangleVertices = []
            tempTriangleVertices.append(flatVertices[simplex[0]])
            tempTriangleVertices.append(flatVertices[simplex[1]])
            tempTriangleVertices.append(flatVertices[simplex[2]])
            faces.append(Face.ByWire(Wire.ByVertices(tempTriangleVertices)))

        shell = Shell.ByFaces(faces)
        if isinstance(face, topologic.Face):
            edges = Shell.Edges(shell)
            edgesCluster = Cluster.ByTopologies(edges)
            shell = Topology.Boolean(flatFace,edgesCluster, operation="slice")
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=1, z=0, degree=theta)
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=0, z=1, degree=phi)
        shell = Topology.Translate(shell, xTran, yTran, zTran)
        return shell


    @staticmethod
    def Edges(shell):
        """
        Returns the edges of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of edges.

        """ 
        if not isinstance(shell, topologic.Shell):
            return None
        edges = []
        _ = shell.Edges(None, edges)
        return edges

    @staticmethod
    def ExternalBoundary(shell):
        """
        Returns the external boundary (closed wire) of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        topologic.Wire
            The external boundary (closed wire) of the input shell.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        edges = []
        _ = shell.Edges(None, edges)
        obEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(shell, faces)
            if len(faces) == 1:
                obEdges.append(anEdge)
        returnTopology = None
        try:
            returnTopology = topologic.Wire.ByEdges(obEdges)
        except:
            returnTopology = topologic.Cluster.ByTopologies(obEdges)
            returnTopology = returnTopology.SelfMerge()
        return returnTopology

    @staticmethod
    def Faces(shell):
        """
        Returns the faces of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of faces.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        faces = []
        _ = shell.Faces(None, faces)
        return faces

    @staticmethod
    def HyperbolicParaboloidRectangularDomain(origin=None, llVertex=None, lrVertex=None, ulVertex=None, urVertex=None, u=10, v=10, direction=[0,0,1], placement="bottom"):
        """
        Creates a hyperbolic paraboloid with a rectangular domain.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin of the hyperbolic parabolid. If set to None, it will be placed at the (0,0,0) origin. The default is None.
        llVertex : topologic.Vertex , optional
            The lower left corner of the hyperbolic parabolid. If set to None, it will be set to (-0.5,-0.5,-0.5).
        lrVertex : topologic.Vertex , optional
            The lower right corner of the hyperbolic parabolid. If set to None, it will be set to (0.5,-0.5,0.5).
        ulVertex : topologic.Vertex , optional
            The upper left corner of the hyperbolic parabolid. If set to None, it will be set to (-0.5,0.5,0.5).
        urVertex : topologic.Vertex , optional
            The upper right corner of the hyperbolic parabolid. If set to None, it will be set to (0.5,0.5,-0.5).
        u : int , optional
            The number of segments along the X axis. The default is 10.
        v : int , optional
            The number of segments along the Y axis. The default is 10.
        direction : list , optional
            The vector representing the up direction of the hyperbolic parabolid. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the hyperbolic parabolid. This can be "center", "lowerleft", "bottom". It is case insensitive. The default is "center".

        Returns
        -------
        topologic.Shell
            The created hyperbolic paraboloid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        if not isinstance(origin, topologic.Vertex):
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(llVertex, topologic.Vertex):
            llVertex = Vertex.ByCoordinates(-0.5,-0.5,-0.5)
        if not isinstance(lrVertex, topologic.Vertex):
            lrVertex = Vertex.ByCoordinates(0.5,-0.5,0.5)
        if not isinstance(ulVertex, topologic.Vertex):
            ulVertex = Vertex.ByCoordinates(-0.5,0.5,0.5)
        if not isinstance(urVertex, topologic.Vertex):
            urVertex = Vertex.ByCoordinates(0.5,0.5,-0.5)
        e1 = Edge.ByVertices([llVertex, lrVertex])
        e3 = Edge.ByVertices([urVertex, ulVertex])
        edges = []
        for i in range(u+1):
            v1 = Edge.VertexByParameter(e1, float(i)/float(u))
            v2 = Edge.VertexByParameter(e3, 1.0 - float(i)/float(u))
            edges.append(Edge.ByVertices([v1, v2]))
        faces = []
        for i in range(u):
            for j in range(v):
                v1 = Edge.VertexByParameter(edges[i], float(j)/float(v))
                v2 = Edge.VertexByParameter(edges[i], float(j+1)/float(v))
                v3 = Edge.VertexByParameter(edges[i+1], float(j+1)/float(v))
                v4 = Edge.VertexByParameter(edges[i+1], float(j)/float(v))
                faces.append(Face.ByVertices([v1, v2, v4]))
                faces.append(Face.ByVertices([v4, v2, v3]))
        returnTopology = Shell.ByFaces(faces)
        if not returnTopology:
            returnTopology = None
        zeroOrigin = returnTopology.CenterOfMass()
        xOffset = 0
        yOffset = 0
        zOffset = 0
        minX = min([llVertex.X(), lrVertex.X(), ulVertex.X(), urVertex.X()])
        maxX = max([llVertex.X(), lrVertex.X(), ulVertex.X(), urVertex.X()])
        minY = min([llVertex.Y(), lrVertex.Y(), ulVertex.Y(), urVertex.Y()])
        maxY = max([llVertex.Y(), lrVertex.Y(), ulVertex.Y(), urVertex.Y()])
        minZ = min([llVertex.Z(), lrVertex.Z(), ulVertex.Z(), urVertex.Z()])
        maxZ = max([llVertex.Z(), lrVertex.Z(), ulVertex.Z(), urVertex.Z()])
        if placement.lower() == "lowerleft":
            xOffset = -minX
            yOffset = -minY
            zOffset = -minZ
        elif placement.lower() == "bottom":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -minZ
        elif placement.lower() == "center":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -(minZ + (maxZ - minZ)*0.5)
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + direction[0]
        y2 = 0 + direction[1]
        z2 = 0 + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        returnTopology = Topology.Rotate(returnTopology, zeroOrigin, 0, 1, 0, theta)
        returnTopology = Topology.Rotate(returnTopology, zeroOrigin, 0, 0, 1, phi)
        returnTopology = Topology.Translate(returnTopology, zeroOrigin.X()+xOffset, zeroOrigin.Y()+yOffset, zeroOrigin.Z()+zOffset)
        return returnTopology
    
    @staticmethod
    def HyperbolicParaboloidCircularDomain(origin=None, radius=0.5, sides=36, rings=10, A=1.0, B=-1.0, direction=[0,0,1], placement="bottom"):
        """
        Creates a hyperbolic paraboloid with a circular domain. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The origin of the hyperbolic parabolid. If set to None, it will be placed at the (0,0,0) origin. The default is None.
        radius : float , optional
            The desired radius of the hyperbolic paraboloid. The default is 0.5.
        sides : int , optional
            The desired number of sides of the hyperbolic parabolid. The default is 36.
        rings : int , optional
            The desired number of concentric rings of the hyperbolic parabolid. The default is 10.
        A : float , optional
            The *A* constant in the equation z = A*x^2^ + B*y^2^. The default is 1.0.
        B : float , optional
            The *B* constant in the equation z = A*x^2^ + B*y^2^. The default is -1.0.
        direction : list , optional
            The  vector representing the up direction of the hyperbolic paraboloid. The default is [0,0,1.
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "bottom". It is case insensitive. The default is "center".

        Returns
        -------
        topologic.Shell
            The created hyperboloic paraboloid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        if not isinstance(origin, topologic.Vertex):
            origin = Vertex.ByCoordinates(0,0,0)
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
                v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
                v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
                v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
                v4 = topologic.Vertex.ByCoordinates(x4,y4,z4)
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
            v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
            v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
            v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
            v4 = topologic.Vertex.ByCoordinates(x4,y4,z4)
            f1 = Face.ByVertices([v1,v2,v4])
            f2 = Face.ByVertices([v4,v2,v3])
            faces.append(f1)
            faces.append(f2)
        # Special Case: Center triangles
        r = vOffset
        x1 = 0
        y1 = 0
        z1 = 0
        v1 = topologic.Vertex.ByCoordinates(x1,y1,z1)
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
                v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
                v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
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
        v2 = topologic.Vertex.ByCoordinates(x2,y2,z2)
        v3 = topologic.Vertex.ByCoordinates(x3,y3,z3)
        f1 = Face.ByVertices([v2,v1,v3])
        faces.append(f1)
        returnTopology = topologic.Shell.ByFaces(faces)
        if not returnTopology:
            returnTopology = topologic.Cluster.ByTopologies(faces)
        vertices = []
        _ = returnTopology.Vertices(None, vertices)
        xList = []
        yList = []
        zList = []
        for aVertex in vertices:
            xList.append(aVertex.X())
            yList.append(aVertex.Y())
            zList.append(aVertex.Z())
        minX = min(xList)
        maxX = max(xList)
        minY = min(yList)
        maxY = max(yList)
        minZ = min(zList)
        maxZ = max(zList)
        zeroOrigin = returnTopology.CenterOfMass()
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = -minX
            yOffset = -minY
            zOffset = -minZ
        elif placement.lower() == "bottom":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -minZ
        elif placement.lower() == "center":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -(minZ + (maxZ - minZ)*0.5)
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + direction[0]
        y2 = 0 + direction[1]
        z2 = 0 + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        zeroOrigin = topologic.Vertex.ByCoordinates(0,0,0)
        returnTopology = topologic.TopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 1, 0, theta)
        returnTopology = topologic.TopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 0, 1, phi)
        returnTopology = topologic.TopologyUtility.Translate(returnTopology, origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset)
        return returnTopology
    
    @staticmethod
    def InternalBoundaries(shell):
        """
        Returns the internal boundaries (closed wires) of the input shell. Internal boundaries are considered holes.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        topologic.Topology
            The wire if a single hole or a cluster of wires if more than one hole.

        """
        from topologicpy.Cluster import Cluster
        edges = []
        _ = shell.Edges(None, edges)
        ibEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(shell, faces)
            if len(faces) > 1:
                ibEdges.append(anEdge)
        return Cluster.SelfMerge(Cluster.ByTopologies(ibEdges))
    
    @staticmethod
    def IsClosed(shell):
        """
        Returns True if the input shell is closed. Returns False otherwise.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        bool
            True if the input shell is closed. False otherwise.

        """
        return shell.IsClosed()

    @staticmethod
    def Pie(origin=None, radiusA=0.5, radiusB=0, sides=32, rings=1, fromAngle=0, toAngle=360, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a pie shape.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the pie. The default is None which results in the pie being placed at (0,0,0).
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
            The vector representing the up direction of the pie. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Shell
            The created pie.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
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

        shell = Shell.ByFaces(faces, tolerance)
        if not shell:
            return None
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + direction[0]
        y2 = 0 + direction[1]
        z2 = 0 + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < tolerance:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        shell = Topology.Rotate(shell, origin, 0, 1, 0, theta)
        shell = Topology.Rotate(shell, origin, 0, 0, 1, phi)
        shell = Topology.Translate(shell, origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset)
        return shell

    @staticmethod
    def Rectangle(origin=None, width=1.0, length=1.0, uSides=2, vSides=2, direction=[0,0,1], placement="center", tolerance=0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic.Vertex , optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0,0,0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        uSides : int , optional
            The number of sides along the width. The default is 2.
        vSides : int , optional
            The number of sides along the length. The default is 2.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Shell
            The created shell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
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
                w = Wire.Rectangle(origin=rOrigin, width=uOffset, length=vOffset, direction=[0,0,1], placement="lowerleft", tolerance=tolerance)
                f = Face.ByWire(w)
                faces.append(f)
        shell = Shell.ByFaces(faces)
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
        shell = Topology.Rotate(shell, origin, 0, 1, 0, theta)
        shell = Topology.Rotate(shell, origin, 0, 0, 1, phi)
        return shell


        
    @staticmethod
    def SelfMerge(shell, angTolerance=0.1):
        """
        Creates a face by merging the faces of the input shell. The shell must be planar within the input angular tolerance.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.

        Returns
        -------
        topologic.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        if not isinstance(shell, topologic.Shell):
            return None
        ext_boundary = Shell.ExternalBoundary(shell)
        if isinstance(ext_boundary, topologic.Wire):
            try:
                return topologic.Face.ByExternalBoundary(Wire.RemoveCollinearEdges(ext_boundary, angTolerance))
            except:
                try:
                    return topologic.Face.ByExternalBoundary(Wire.Planarize(Wire.RemoveCollinearEdges(ext_boundary, angTolerance)))
                except:
                    print("FaceByPlanarShell - Error: The input Wire is not planar and could not be fixed. Returning None.")
                    return None
        elif isinstance(ext_boundary, topologic.Cluster):
            wires = []
            _ = ext_boundary.Wires(None, wires)
            faces = []
            areas = []
            for aWire in wires:
                try:
                    aFace = topologic.Face.ByExternalBoundary(Wire.RemoveCollinearEdges(aWire, angTolerance))
                except:
                    aFace = topologic.Face.ByExternalBoundary(Wire.Planarize(Wire.RemoveCollinearEdges(aWire, angTolerance)))
                anArea = topologic.FaceUtility.Area(aFace)
                faces.append(aFace)
                areas.append(anArea)
            max_index = areas.index(max(areas))
            ext_boundary = faces[max_index]
            int_boundaries = list(set(faces) - set([ext_boundary]))
            int_wires = []
            for int_boundary in int_boundaries:
                temp_wires = []
                _ = int_boundary.Wires(None, temp_wires)
                int_wires.append(Wire.RemoveCollinearEdges(temp_wires[0], angTolerance))
            temp_wires = []
            _ = ext_boundary.Wires(None, temp_wires)
            ext_wire = Wire.RemoveCollinearEdges(temp_wires[0], angTolerance)
            try:
                return Face.ByWires(ext_wire, int_wires)
            except:
                return Face.ByWires(Wire.Planarize(ext_wire), planarizeList(int_wires))
        else:
            return None

    @staticmethod
    def Vertices(shell):
        """
        Returns the vertices of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        vertices = []
        _ = shell.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Voronoi(vertices, face=None):
        """
        Returns a voronoi partitioning of the input face based on the input vertices. The vertices must be coplanar and within the face. See https://en.wikipedia.org/wiki/Voronoi_diagram.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic.Face , optional
            The input face. If the face is not set an optimised bounding rectangle of the input vertices is used instead. The default is None.

        Returns
        -------
        shell
            A shell representing the voronoi partitioning of the input face.

        """
        from scipy.spatial import Voronoi
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not isinstance(face, topologic.Face):
            cluster = Cluster.ByTopologies(vertices)
            br = Wire.BoundingRectangle(cluster, optimize=5)
            face = Face.ByWire(br)
        if not isinstance(vertices, list):
            return None
        vertices = [x for x in vertices if isinstance(x, topologic.Vertex)]
        if len(vertices) < 2:
            return None

        # Flatten the input face
        flatFace = Face.Flatten(face)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        # Create a Vertex at the world's origin (0,0,0)
        world_origin = Vertex.ByCoordinates(0,0,0)

        # Create a cluster of the input vertices
        verticesCluster = Cluster.ByTopologies(vertices)

        # Flatten the cluster using the same transformations
        verticesCluster = Topology.Translate(verticesCluster, -xTran, -yTran, -zTran)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=1, z=0, degree=-theta)

        flatVertices = Cluster.Vertices(verticesCluster)
        points = []
        for flatVertex in flatVertices:
            points.append([flatVertex.X(), flatVertex.Y()])

        br = Wire.BoundingRectangle(flatFace)
        br_vertices = Wire.Vertices(br)
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
                    tempWire.append(Vertex.ByCoordinates(voronoiVertices[v].X(), voronoiVertices[v].Y(),0))
                faces.append(Face.ByWire(Wire.ByVertices(tempWire, close=True)))
        shell = Shell.ByFaces(faces)
        edges = Shell.Edges(shell)
        edgesCluster = Cluster.ByTopologies(edges)
        shell = Topology.Boolean(flatFace,edgesCluster, operation="slice")
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=1, z=0, degree=theta)
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=0, z=1, degree=phi)
        shell = Topology.Translate(shell, xTran, yTran, zTran)
        return shell

    @staticmethod
    def Wires(shell):
        """
        Returns the wires of the input shell.

        Parameters
        ----------
        shell : topologic.Shell
            The input shell.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(shell, topologic.Shell):
            return None
        wires = []
        _ = shell.Wires(None, wires)
        return wires

    
    
    
    
    