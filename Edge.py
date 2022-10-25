import topologicpy
import topologic
import numpy as np
import numpy.linalg as la
from numpy import pi

class Edge(topologic.Edge):
    @staticmethod
    def Angle(edgeA, edgeB, mantissa=3, bracket=False):
        """
        Description
        -----------
        Returns the angle in degrees between the two input topologic Edges.

        Parameters
        ----------
        edgeA : topologic.Edge
            The first topologic Edge.
        edgeB : topologic Edge
            The second topologic Edge.
        mantissa : int, optional
            The desired mantissa. The default is 3.
        bracket : bool
            If set to True, the returned angle is bracketed between 0 and 180. The default is False.

        Returns
        -------
        float
            The angle in degrees between the two input topologic Edges.

        """

        if not isinstance(edgeA, topologic.Edge) or not isinstance(edgeB, topologic.Edge):
            return None
        dirA = Edge.Direction(edgeA, mantissa)
        dirB = Edge.Direction(edgeB, mantissa)
        ang = topologic_lib.angle_between(dirA, dirB) * 180 / pi # convert to degrees
        if bracket:
            if ang > 90:
                ang = 180 - ang
        return round(ang, mantissa)

    @staticmethod
    def ByStartVertexEndVertex(vertexA, vertexB, tolerance=0.0001):
        """
        Description
        -----------
        Creates a straight topologic Edge that connects the input vertices.

        Parameters
        ----------
        vertexA : topologic.Vertex
            The start topologic Vertex.
        vertexB : toopologic.Vertex
            The end topologic Vertex.
        tolerance : float, optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.

        Returns
        -------
        edge : topologic.Edge
            The created topologic Edge.

        """
        edge = None
        if not vertexA or not vertexB:
            return None
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
        Creates a straight topologic Edge that connects the input topologic Vertices.

        Parameters
        ----------
        vertices : list
            The list of topologic Vertices.
        tolerance : float, optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.

        Returns
        -------
        topologic.Edge
            The created topologic Edge.

        """
        if not vertices:
            return None
        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, topologic.Vertex)]
        if len(vertexList) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertexList[0], vertexList[-1], tolerance)
    
    @staticmethod
    def Direction(edge, mantissa=3):
        """
        Description
        -----------
        Returns the direction of the input topologic Edge.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.
        mantissa : int, optional
            The desired mantissa. The default is 3.

        Returns
        -------
        list
            The direction of the topologic Edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        ev = edge.EndVertex()
        sv = edge.StartVertex()
        x = ev.X() - sv.X()
        y = ev.Y() - sv.Y()
        z = ev.Z() - sv.Z()
        uvec = Helper.unitizeVector([x,y,z])
        x = round(uvec[0], mantissa)
        y = round(uvec[1], mantissa)
        z = round(uvec[2], mantissa)
        return [x, y, z]
    
    @staticmethod
    def EndVertex(edge):
        """
        Description
        -----------
        Returns the end topologic Vertex of the input topologic Edge.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.

        Returns
        -------
        topologic.Vertex
            The end topologic Vertex of the input topologic Edge.

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
    def IsCollinear(edgeA, edgeB, tolerance=0.0001):
        """
        Description
        -----------
        Tests if the two input topologic Edges are collinear.

        Parameters
        ----------
        edgeA : topologic.Edge
            The first topologic Edge.
        edgeB : topologic.Edge
            The second topologic Edge.
        tolerance : float, optional
            The tolerance used for the test. The default is 0.0001.

        Returns
        -------
        bool
            True if the two edges are collinear. False otherwise.

        """
        if not isinstance(edgeA, topologic.Edge):
            return None
        if not isinstance(edgeB, topologic.Edge):
            return None
        ang = Edge.Angle(edgeA, edgeB, mantissa=len(str(tolerance).split(".")[1]), bracket=True)
        if abs(ang) < tolerance:
            return True
        return False
    
    @staticmethod
    def Length(edge, mantissa=3):
        """
        Description
        -----------
        Returns the length of the input topologic Edge.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.
        mantissa : int, optional
            The desired mantissa. The default is 3.

        Returns
        -------
        float
            The length of the input topologic Edge.

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
    def ParameterAtVertex(edge, vertex, mantissa=3):
        """
        Description
        -----------
        Returns the *u* parameter along the input topologic Edge based on the location of the input topologic Vertex.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.
        vertex : topologic.Vertex
            The topologic Vertex.
        mantissa : int, optional
            The desired mantissa. The default is 3.

        Returns
        -------
        float
            The *u* parameter along the input topologic Edge based on the location of the input topologic Vertex.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        if not isinstance(vertex, topologic.Vertex):
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
        Creates a topologic Edge that has the reverse direction of the input topologic Edge.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.

        Returns
        -------
        topologic.Edge
            The reversed topologic Edge.

        """
        if not isinstance(edge, topologic.Edge):
            return None
        return Edge.ByVertices([edge.EndVertex(), edge.StartVertex()])
    
    @staticmethod
    def StartVertex(edge):
        """
        Description
        -----------
        Returns the start topologic Vertex of the input topologic Edge.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.

        Returns
        -------
        topologic.Vertex
            The start topologic Vertex of the input topologic Edge.

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
        creates a topologic Vertex along the input topologic Edge offset by the input distance from the input topologic Vertex.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.
        distance : float
            The offset distance.
        origin : topologic.Vertex
            The origin of the offset distance.
        tolerance : float, optional
            The tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            The created topologic Vertex.

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
        vector = topologic_lib.unitizeVector([vx, vy, vz])
        vector = topologic_lib.multiplyVector(vector, distance, tolerance)
        return topologic.Vertex.ByCoordinates(origin.X()+vector[0], origin.Y()+vector[1], origin.Z()+vector[2])
    
    @staticmethod
    def VertexByParameter(edge, parameter):
        """
        Description
        -----------
        creates a topologic Vertex along the input topologic Edge offset by the input u parameter. A parameter of 0 returns the start Vertex, a parameter of 1 returns the end Vertex, and a parameter of 0.5 returns the midpoint of the Edge.

        Parameters
        ----------
        edge : topologic.Edge
            The topologic Edge.
        parameter : float
            The u parameter along the input topologic Edge.

        Returns
        -------
        topologic.Vertex
            The created topologic Vertex.

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