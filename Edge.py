import topologicpy
import topologic
from topologicpy.Vector import Vector

class Edge():
    @staticmethod
    def Angle(edgeA, edgeB, mantissa=4, bracket=False):
        """
        Description
        -----------
        Returns the angle in degrees between the two input edges.

        Parameters
        ----------
        edgeA : topologic.Edge
            The first input edge.
        edgeB : topologic Edge
            The second input edge.
        mantissa : int, optional
            The desired length of the mantissa. The default is 4.
        bracket : bool
            If set to True, the returned angle is bracketed between 0 and 180. The default is False.

        Returns
        -------
        float
            The angle in degrees between the two input edges.

        """

        if not isinstance(edgeA, topologic.Edge) or not isinstance(edgeB, topologic.Edge):
            return None
        dirA = Edge.Direction(edgeA, mantissa)
        dirB = Edge.Direction(edgeB, mantissa)
        ang = Vector.Angle(dirA, dirB)
        if bracket:
            if ang > 90:
                ang = 180 - ang
        return round(ang, mantissa)

    @staticmethod
    def Bisect(edgeA, edgeB, length=1.0, placement=0, tolerance=0.0001):
        """
        Description
        -----------
        Creates a bisecting edge between edgeA and edgeB.

        Parameters
        ----------
        edgeA : topologic.Edge
            The first topologic Edge.
        edgeB : topologic Edge
            The second topologic Edge.
        length : float, optional
            The desired length of the bisecting edge. The default is 1.0.
        placement : int, optional
            The desired placement of the bisecting edge.
            If set to 0, the bisecting edge centroid will be placed at the end vertex of the first edge.
            If set to 1, the bisecting edge start vertex will be placed at the end vertex of the first edge.
            If set to 2, the bisecting edge end vertex will be placed at the end vertex of the first edge.
            If set to any number other than 0, 1, or 2, the bisecting edge centroid will be placed at the end vertex of the first edge.
        tolerance : float, optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.

        Returns
        -------
        topologic.Edge
            The created bisecting edge.

        """
        if not isinstance(edgeA, topologic.Edge) or not isinstance(edgeB, topologic.Edge):
            return None
        if Edge.Length(edgeA) < tolerance or Edge.Length(edgeB) < tolerance:
            return None
        from topologicpy.Topology import Topology
        v1 = Edge.VertexByDistance(edgeA, -1, edgeA.EndVertex(), tolerance=0.0001)
        newEdgeA = Edge.ByVertices([v1, edgeA.EndVertex()])
        v1 = Edge.VertexByDistance(edgeB, 1, edgeB.StartVertex(), tolerance=0.0001)
        newEdgeB = Edge.ByVertices([edgeB.StartVertex(), v1])
        newEdgeB = Topology.Place(newEdgeB, newEdgeB.StartVertex(), newEdgeA.StartVertex())
        bisectingEdge = Edge.ByVertices([newEdgeA.EndVertex(), newEdgeB.EndVertex()])
        bEdgeLength = Edge.Length(bisectingEdge)
        bisectingEdge = Topology.Scale(bisectingEdge, bisectingEdge.StartVertex(), 1/bEdgeLength, 1/bEdgeLength, 1/bEdgeLength)
        if length != 1.0 and length > tolerance:
            bisectingEdge = Topology.Scale(bisectingEdge, bisectingEdge.StartVertex(), length, length, length)
        newLocation = edgeA.EndVertex()
        if placement == 2:
            oldLocation = bisectingEdge.EndVertex()
        elif placement == 1:
            oldLocation = bisectingEdge.StartVertex()
        else:
            oldLocation = bisectingEdge.Centroid()
        bisectingEdge = Topology.Place(bisectingEdge, oldLocation, newLocation)
        return bisectingEdge

    @staticmethod
    def ByStartVertexEndVertex(vertexA, vertexB, tolerance=0.0001):
        """
        Description
        -----------
        Creates a straight edge that connects the input vertices.

        Parameters
        ----------
        vertexA : topologic.Vertex
            The first input vertex. This is considered the start vertex.
        vertexB : toopologic.Vertex
            The second input vertex. This is considered the end vertex.
        tolerance : float, optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.

        Returns
        -------
        edge : topologic.Edge
            The created edge.

        """
        edge = None
        if not isinstance(vertexA, topologic.Vertex):
            return None
        if not isinstance(vertexB, topologic.Vertex):
            return None
        if topologic.Topology.IsSame(vertexA, vertexB):
            return None
        if topologic.VertexUtility.Distance(vertexA, vertexB) < tolerance:
            return None
        try:
            edge = topologic.Edge.ByStartVertexEndVertex(vertexA, vertexB)
        except:
            edge = None
        return edge
    
    @staticmethod
    def ByVertices(vertices, tolerance=0.0001):
        """
        Description
        -----------
        Creates a straight edge that connects the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices. The first item is considered the start vertex and the last item is considered the end vertex.
        tolerance : float, optional
            The desired tolerance to decide if an edge can be created. The default is 0.0001.

        Returns
        -------
        topologic.Edge
            The created edge.

        """
        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, topologic.Vertex)]
        if len(vertexList) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertexList[0], vertexList[-1], tolerance)
    
    @staticmethod
    def Direction(edge, mantissa=4):
        """
        Description
        -----------
        Returns the direction of the input edge.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.
        mantissa : int, optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The direction of the input edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        ev = edge.EndVertex()
        sv = edge.StartVertex()
        x = ev.X() - sv.X()
        y = ev.Y() - sv.Y()
        z = ev.Z() - sv.Z()
        uvec = Vector.Normalize([x,y,z])
        x = round(uvec[0], mantissa)
        y = round(uvec[1], mantissa)
        z = round(uvec[2], mantissa)
        return [x, y, z]
    
    @staticmethod
    def EndVertex(edge):
        """
        Description
        -----------
        Returns the end vertex of the input edge.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.

        Returns
        -------
        topologic.Vertex
            The end vertex of the input edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        vert = None
        try:
            vert = edge.EndVertex()
        except:
            vert = None
        return vert
    
    @staticmethod
    def IsCollinear(edgeA, edgeB, angTolerance=0.1):
        """
        Description
        -----------
        Tests if the two input edges are collinear.

        Parameters
        ----------
        edgeA : topologic.Edge
            The first input edge.
        edgeB : topologic.Edge
            The second input edge.
        angTolerance : float, optional
            The angular tolerance used for the test. The default is 0.1.

        Returns
        -------
        bool
            True if the two edges are collinear. False otherwise.

        """
        if not isinstance(edgeA, topologic.Edge) or not isinstance(edgeB, topologic.Edge):
            return None
        ang = Edge.Angle(edgeA, edgeB, mantissa=len(str(angTolerance).split(".")[1]), bracket=True)
        if abs(ang) < angTolerance:
            return True
        return False
    
    @staticmethod
    def Length(edge, mantissa=4):
        """
        Description
        -----------
        Returns the length of the input edge.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.
        mantissa : int, optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The length of the input edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        length = None
        try:
            length = round(topologic.EdgeUtility.Length(edge), mantissa)
        except:
            length = None
        return length

    @staticmethod
    def Normalize(edge, useEndVertex=False):
        """
        Description
        -----------
        Creates a normalized edge that has the same direction as the input edge, but a length of 1.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.
        useEndVertex : bool, optional
            If True the normalized edge end vertex will be placed at the end vertex of the input edge. Otherwise, the normalized edge start vertex will be placed at the start vertex of the input edge. The default is False.

        Returns
        -------
        topologic.Edge
            The normalized edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        if not useEndVertex:
            sv = edge.StartVertex()
            ev = Edge.VertexByDistance(edge, 1.0, edge.StartVertex())
        else:
            sv = Edge.VertexByDistance(edge, 1.0, edge.StartVertex())
            ev = edge.EndVertex()
        return Edge.ByVertices([sv, ev])

    @staticmethod
    def ParameterAtVertex(edge, vertex, mantissa=4):
        """
        Description
        -----------
        Returns the *u* parameter along the input edge based on the location of the input vertex.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.
        vertex : topologic.Vertex
            The input vertex.
        mantissa : int, optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The *u* parameter along the input edge based on the location of the input vertex.

        """
        if not isinstance(edge, topologic.Edge) or not isinstance(vertex, topologic.Vertex):
            return None
        parameter = None
        try:
            parameter = topologic.EdgeUtility.ParameterAtPoint(edge, vertex)
        except:
            parameter = None
        return round(parameter, mantissa)

    @staticmethod
    def Reverse(edge):
        """
        Description
        -----------
        Creates an edge that has the reverse direction of the input edge.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.

        Returns
        -------
        topologic.Edge
            The reversed edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        return Edge.ByVertices([edge.EndVertex(), edge.StartVertex()])
    
    @staticmethod
    def StartVertex(edge):
        """
        Description
        -----------
        Returns the start vertex of the input edge.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.

        Returns
        -------
        topologic.Vertex
            The start vertex of the input edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        vert = None
        try:
            vert = edge.StartVertex()
        except:
            vert = None
        return vert
    
    @staticmethod
    def VertexByDistance(edge, distance, origin, tolerance=0.0001):
        """
        Description
        -----------
        Creates a vertex along the input edge offset by the input distance from the input origin.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.
        distance : float
            The offset distance.
        origin : topologic.Vertex
            The origin of the offset distance.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            The created vertex.

        """

        if not isinstance(edge, topologic.Edge):
            return None
        if not origin:
            origin = edge.StartVertex()
        if not isinstance(origin, topologic.Vertex):
            return None
        sv = edge.StartVertex()
        ev = edge.EndVertex()
        vx = ev.X() - sv.X()
        vy = ev.Y() - sv.Y()
        vz = ev.Z() - sv.Z()
        vector = Vector.Normalize([vx, vy, vz])
        vector = Vector.Multiply(vector, distance, tolerance)
        return topologic.Vertex.ByCoordinates(origin.X()+vector[0], origin.Y()+vector[1], origin.Z()+vector[2])
    
    @staticmethod
    def VertexByParameter(edge, parameter=0):
        """
        Description
        -----------
        Creates a vertex along the input edge offset by the input *u* parameter.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.
        parameter : float , optional
            The *u* parameter along the input topologic Edge. A parameter of 0 returns the start vertex. A parameter of 1 returns the end vertex. The default is 0.

        Returns
        -------
        topologic.Vertex
            The created vertex.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        vertex = None
        if parameter == 0:
            vertex = edge.StartVertex()
        elif parameter == 1:
            vertex = edge.EndVertex()
        else:
            try:
                vertex = topologic.EdgeUtility.PointAtParameter(edge, parameter)
            except:
                vertex = None
        return vertex

    @staticmethod
    def Vertices(edge):
        """
        Description
        __________
        Returns the vertices of the input edge.

        Parameters
        ----------
        edge : topologic.Edge
            The input edge.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        vertices = []
        _ = edge.Vertices(None, vertices)
        return vertices