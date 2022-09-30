import topologic
import numpy as np
import numpy.linalg as la
from numpy import pi

class Edge(topologic.Edge):
    @staticmethod
    def EdgeAngle(edgeA, edgeB, mantissa, bracket):
        """
        Parameters
        ----------
        edgeA : TYPE
            DESCRIPTION.
        edgeB : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.
        bracket : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # edgeA, edgeB, mantissa, bracket = item
        
        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2' """
            n_v1=la.norm(v1)
            n_v2=la.norm(v2)
            if (abs(np.log10(n_v1/n_v2)) > 10):
                v1 = v1/n_v1
                v2 = v2/n_v2
            cosang = np.dot(v1, v2)
            sinang = la.norm(np.cross(v1, v2))
            return np.arctan2(sinang, cosang)
        
        if not isinstance(edgeA, topologic.Edge) or not isinstance(edgeB, topologic.Edge):
            return None
        dirA = Edge.EdgeDirection(edgeA, mantissa)
        dirB = Edge.EdgeDirection(edgeB, mantissa)
        ang = angle_between(dirA, dirB) * 180 / pi # convert to degrees
        if bracket:
            if ang > 90:
                ang = 180 - ang
        return round(ang, mantissa)
    
    @staticmethod
    def unitizeVector(vector):
        """
        Parameters
        ----------
        vector : TYPE
            DESCRIPTION.

        Returns
        -------
        unitVector : TYPE
            DESCRIPTION.

        """
        mag = 0
        for value in vector:
            mag += value ** 2
        mag = mag ** 0.5
        unitVector = []
        for i in range(len(vector)):
            unitVector.append(vector[i] / mag)
        return unitVector
    
    @staticmethod
    def EdgeByStartVertexEndVertex(sv, ev, tolerance=0.0001):
        """
        Parameters
        ----------
        sv : TYPE
            DESCRIPTION.
        ev : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        edge : TYPE
            DESCRIPTION.

        """
        # sv = item[0]
        # ev = item[1]
        # tol = item[2]
        edge = None
        if not sv or not ev:
            return None
        if topologic.Topology.IsSame(sv, ev):
            return None
        if topologic.VertexUtility.Distance(sv, ev) < tolerance:
            return None
        try:
            edge = topologic.Edge.ByStartVertexEndVertex(sv, ev)
        except:
            edge = None
        return edge
    
    @staticmethod
    def EdgeByVertices(item):
        """
        Parameters
        ----------
        item : list
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        assert isinstance(item, list), "Edge.ByVertices - Error: Input is not a list"
        vertices = [x for x in item if isinstance(x, topologic.Vertex)]
        if len(vertices) < 2:
            return None
        elif len(vertices) == 2:
            return topologic.Edge.ByStartVertexEndVertex(vertices[0], vertices[-1])
        else:
            edges = []
            for i in range(len(vertices)-1):
                edges.append(topologic.Edge.ByStartVertexEndVertex(vertices[i], vertices[i+1]))
            return edges
    
    @staticmethod
    def EdgeDirection(edge, mantissa):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # edge, mantissa = item
        assert isinstance(edge, topologic.Edge), "Edge.Direction - Error: Input is not an Edge"
        ev = edge.EndVertex()
        sv = edge.StartVertex()
        x = ev.X() - sv.X()
        y = ev.Y() - sv.Y()
        z = ev.Z() - sv.Z()
        uvec = Edge.unitizeVector([x,y,z])
        x = round(uvec[0], mantissa)
        y = round(uvec[1], mantissa)
        z = round(uvec[2], mantissa)
        return [x, y, z]
    
    @staticmethod
    def EdgeEndVertex(edge):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.

        Returns
        -------
        vert : TYPE
            DESCRIPTION.

        """
        # edge = item[0]
        vert = None
        try:
            vert = edge.EndVertex()
        except:
            vert = None
        return vert
    
    @staticmethod
    def EdgeIsCollinear(edgeA, edgeB, tolerance=0.0001):
        """
        Parameters
        ----------
        edgeA : TYPE
            DESCRIPTION.
        edgeB : TYPE
            DESCRIPTION.
        tolerance : float, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        # edgeA, edgeB, tol = item
        assert isinstance(edgeA, topologic.Edge), "Edge.Angle - Error: Edge A is not a Topologic Edge."
        assert isinstance(edgeB, topologic.Edge), "Edge.Angle - Error: Edge B is not a Topologic Edge."
        ang = Edge.EdgeAngle(edgeA, edgeB, 8, True)
        if abs(ang) < tolerance:
            return True
        return False
    
    @staticmethod
    def EdgeLength(edge, mantissa):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Returns
        -------
        length : TYPE
            DESCRIPTION.

        """
        # edge, mantissa = item
        length = None
        try:
            length = round(topologic.EdgeUtility.Length(edge), mantissa)
        except:
            length = None
        return length
    
    @staticmethod
    def EdgeParameterAtVertex(edge, vertex, mantissa):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.
        vertex : TYPE
            DESCRIPTION.
        mantissa : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # edge, vertex, mantissa = item
        parameter = None
        try:
            parameter = topologic.EdgeUtility.ParameterAtPoint(edge, vertex)
        except:
            parameter = None
        return round(parameter, mantissa)
    
    @staticmethod
    def EdgeStartVertex(edge):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.

        Returns
        -------
        vert : TYPE
            DESCRIPTION.

        """
        # edge = item[0]
        vert = None
        try:
            vert = edge.StartVertex()
        except:
            vert = None
        return vert
    
    @staticmethod
    def EdgeVertexByDistance(edge, distance, vertex, tolerance=0.0001):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.
        distance : TYPE
            DESCRIPTION.
        vertex : TYPE
            DESCRIPTION.
        tolerance : TYPE, optional
            DESCRIPTION. The default is 0.0001.

        Returns
        -------
        rv : TYPE
            DESCRIPTION.

        """
        # edge, distance, vertex, tol = item
        
        def multiplyVector(vector, mag, tolerance):
            oldMag = 0
            for value in vector:
                oldMag += value ** 2
            oldMag = oldMag ** 0.5
            if oldMag < tolerance:
                return [0,0,0]
            newVector = []
            for i in range(len(vector)):
                newVector.append(vector[i] * mag / oldMag)
            return newVector
        
        if not isinstance(edge, topologic.Edge):
            return None
        if (not vertex) or (vertex == 0):
            vertex = edge.StartVertex()
        rv = None
        sv = edge.StartVertex()
        ev = edge.EndVertex()
        vx = ev.X() - sv.X()
        vy = ev.Y() - sv.Y()
        vz = ev.Z() - sv.Z()
        vector = Edge.unitizeVector([vx, vy, vz])
        vector = multiplyVector(vector, distance, tolerance)
        if vertex == None:
            vertex = sv
        rv = topologic.Vertex.ByCoordinates(vertex.X()+vector[0], vertex.Y()+vector[1], vertex.Z()+vector[2])
        return rv
    
    @staticmethod
    def EdgeVertexByParameter(edge, parameter):
        """
        Parameters
        ----------
        edge : TYPE
            DESCRIPTION.
        parameter : TYPE
            DESCRIPTION.

        Returns
        -------
        vertex : TYPE
            DESCRIPTION.

        """
        # edge, parameter = item
        if not isinstance(edge, topologic.Edge):
            return None
        vertex = None
        try:
            vertex = topologic.EdgeUtility.PointAtParameter(edge, parameter)
        except:
            vertex = None
        return vertex