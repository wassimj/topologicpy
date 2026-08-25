# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from topologicpy.Core import Core

class Edge():

    @staticmethod
    def Align2D(edgeA, edgeB, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a 4x4 transformation matrix that aligns the first input edge to the second input edge in 2D.

        The transformation translates the centroid of edgeA to the centroid of edgeB, uniformly scales edgeA to
        the length of edgeB, and rotates it about the global Z-axis to match the chord direction of edgeB.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The source edge.
        edgeB : topologic_core.Edge
            The target edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The 4x4 transformation matrix.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Matrix import Matrix
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Align2D - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Align2D - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None

        lengthA = Edge.Length(edgeA, mantissa=15, tolerance=tolerance, silent=True)
        lengthB = Edge.Length(edgeB, mantissa=15, tolerance=tolerance, silent=True)
        if lengthA is None or lengthB is None or lengthA <= tolerance or lengthB <= tolerance:
            if not silent:
                print("Edge.Align2D - Error: One or both input edges are degenerate. Returning None.")
            return None

        centroidA = Topology.Centroid(edgeA)
        centroidB = Topology.Centroid(edgeB)
        if not Topology.IsInstance(centroidA, "Vertex") or not Topology.IsInstance(centroidB, "Vertex"):
            if not silent:
                print("Edge.Align2D - Error: Could not compute valid edge centroids. Returning None.")
            return None

        x1, y1, z1 = Vertex.Coordinates(centroidA, mantissa=None)
        x2, y2, z2 = Vertex.Coordinates(centroidB, mantissa=None)
        directionA = Edge.Direction(edgeA, mantissa=15, tolerance=tolerance, silent=True)
        directionB = Edge.Direction(edgeB, mantissa=15, tolerance=tolerance, silent=True)
        if directionA is None or directionB is None:
            if not silent:
                print("Edge.Align2D - Error: Could not compute valid edge directions. Returning None.")
            return None

        move_to_origin = Matrix.ByTranslation(-x1, -y1, -z1)
        move_to_target = Matrix.ByTranslation(x2, y2, z2)
        scale_factor = lengthB / lengthA
        scaling_matrix = Matrix.ByScaling(scale_factor, scale_factor, 1.0)
        angleA = Vector.CompassAngle(directionA, [1, 0, 0])
        angleB = Vector.CompassAngle(directionB, [1, 0, 0])
        if angleA is None or angleB is None:
            if not silent:
                print("Edge.Align2D - Error: Could not compute the required rotation. Returning None.")
            return None
        rotation_matrix = Matrix.ByRotation(0, 0, angleB - angleA, order="xyz")
        matrix = Matrix.Multiply(scaling_matrix, move_to_origin)
        matrix = Matrix.Multiply(rotation_matrix, matrix)
        return Matrix.Multiply(move_to_target, matrix)

    @staticmethod
    def Angle(edgeA, edgeB, mantissa: int = 6, bracket: bool = False, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the angle in degrees between the two input edges.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        bracket : bool , optional
            If set to True, the returned angle is bracketed to the range 0 to 90 degrees. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The angle in degrees between the two input edges.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Angle - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Angle - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        directionA = Edge.Direction(edgeA, mantissa=15, tolerance=tolerance, silent=True)
        directionB = Edge.Direction(edgeB, mantissa=15, tolerance=tolerance, silent=True)
        if directionA is None or directionB is None:
            return None
        angle = Vector.Angle(directionA, directionB)
        if angle is None:
            return None
        if bracket and angle > 90:
            angle = 180 - angle
        return round(float(angle), mantissa)

    @staticmethod
    def Bisect(edgeA, edgeB, length: float = 1.0, placement: int = 0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a bisecting edge between two input edges that share an endpoint.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        length : float , optional
            The desired length of the bisecting edge. Default is 1.0.
        placement : int , optional
            The desired placement of the bisecting edge. If set to 0, its centroid is placed at the shared vertex.
            If set to 1, its start vertex is placed at the shared vertex. If set to 2, its end vertex is placed at
            the shared vertex. Other values are treated as 0. Default is 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created bisecting edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Bisect - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Bisect - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            length = float(length)
        except Exception:
            if not silent:
                print("Edge.Bisect - Error: The input length parameter is not numerical. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Edge.Bisect - Error: The input length parameter must be greater than the input tolerance. Returning None.")
            return None
        if Edge.Length(edgeA, mantissa=15, tolerance=tolerance, silent=True) <= tolerance:
            return None
        if Edge.Length(edgeB, mantissa=15, tolerance=tolerance, silent=True) <= tolerance:
            return None

        a0, a1 = Edge.StartVertex(edgeA, silent=True), Edge.EndVertex(edgeA, silent=True)
        b0, b1 = Edge.StartVertex(edgeB, silent=True), Edge.EndVertex(edgeB, silent=True)
        shared = None
        if Vertex.IsCoincident(a0, b0, tolerance=tolerance, silent=True):
            shared = a0
            edge1, edge2 = edgeA, edgeB
        elif Vertex.IsCoincident(a0, b1, tolerance=tolerance, silent=True):
            shared = a0
            edge1, edge2 = edgeA, Edge.Reverse(edgeB, tolerance=tolerance, silent=True)
        elif Vertex.IsCoincident(a1, b0, tolerance=tolerance, silent=True):
            shared = a1
            edge1, edge2 = Edge.Reverse(edgeA, tolerance=tolerance, silent=True), edgeB
        elif Vertex.IsCoincident(a1, b1, tolerance=tolerance, silent=True):
            shared = a1
            edge1 = Edge.Reverse(edgeA, tolerance=tolerance, silent=True)
            edge2 = Edge.Reverse(edgeB, tolerance=tolerance, silent=True)
        else:
            if not silent:
                print("Edge.Bisect - Error: The input edges do not share a vertex. Returning None.")
            return None

        direction1 = Edge.Direction(edge1, mantissa=15, tolerance=tolerance, silent=True)
        direction2 = Edge.Direction(edge2, mantissa=15, tolerance=tolerance, silent=True)
        if direction1 is None or direction2 is None:
            return None
        bisector = Vector.Bisect(direction1, direction2)
        if not isinstance(bisector, list) or len(bisector) != 3:
            return None
        end = Topology.TranslateByDirectionDistance(shared, bisector, length)
        result = Edge.ByVertices(shared, end, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(result, "Edge"):
            return None
        if placement == 0 or placement not in [0, 1, 2]:
            result = Topology.TranslateByDirectionDistance(result, Vector.Reverse(bisector), 0.5*length)
        elif placement == 2:
            result = Topology.TranslateByDirectionDistance(result, Vector.Reverse(bisector), length)
        return result

    @staticmethod
    def ByFaceNormal(face, origin=None, length: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge representing the normal to the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        origin : topologic_core.Vertex , optional
            The desired origin of the edge. If set to None, the centroid of the face is used. Default is None.
        length : float , optional
            The desired length of the edge. Default is 1.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Edge.ByFaceNormal - Error: The input face parameter is not a valid topologic face. Returning None.")
            return None
        if not isinstance(length, (int, float)) or float(length) <= tolerance:
            if not silent:
                print("Edge.ByFaceNormal - Error: The input length parameter must be greater than the input tolerance. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(face)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.ByFaceNormal - Error: Could not determine a valid origin. Returning None.")
            return None
        normal = Face.Normal(face)
        if not isinstance(normal, (list, tuple)) or len(normal) != 3:
            if not silent:
                print("Edge.ByFaceNormal - Error: Could not compute a valid face normal. Returning None.")
            return None
        return Edge.ByOriginDirectionLength(
            origin=origin,
            direction=list(normal),
            length=float(length),
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def ByOffset2D(edge, offset: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge offset to the left of the input edge in the XY plane.

        The start-to-end chord of the input edge defines the 2D direction. The returned edge lies on Z = 0,
        matching the historical behaviour of this method.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        offset : float , optional
            The signed offset distance. Positive values offset to the left of the start-to-end direction.
            Default is 1.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The offset edge.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.ByOffset2D - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)
        if not Topology.IsInstance(sv, "Vertex") or not Topology.IsInstance(ev, "Vertex"):
            return None
        x1, y1, _ = Vertex.Coordinates(sv, mantissa=None)
        x2, y2, _ = Vertex.Coordinates(ev, mantissa=None)
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length <= tolerance:
            if not silent:
                print("Edge.ByOffset2D - Error: The XY projection of the input edge is degenerate. Returning None.")
            return None
        nx = -dy / length
        ny = dx / length
        new_sv = Vertex.ByCoordinates(x1 + nx * offset, y1 + ny * offset, 0)
        new_ev = Vertex.ByCoordinates(x2 + nx * offset, y2 + ny * offset, 0)
        return Edge.ByVertices(new_sv, new_ev, tolerance=tolerance, silent=silent)
    # @staticmethod
    # def ByOffset2D(edge, offset: float = 1.0, tolerance: float = 0.0001):
    #     """
    #     Creates and edge offset from the input edge. This method is intended for edges that are in the XY plane.

    #     Parameters
    #     ----------
    #     edge : topologic_core.Edge
    #         The input edge.
    #     offset : float , optional
    #         The desired offset. Default is 1.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.

    #     Returns
    #     -------
    #     topologic_core.Edge
    #         An edge offset from the input edge.

    #     """
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Vector import Vector

    #     n = Edge.Normal(edge)
    #     n = Vector.Normalize(n)
    #     n = Vector.Multiply(n, offset, tolerance=tolerance)
    #     edge = Topology.Translate(edge, n[0], n[1], n[2])
    #     return edge

    @staticmethod
    def ByStartVertexEndVertex(vertexA, vertexB, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge that connects the input vertices.

        Parameters
        ----------
        vertexA : topologic_core.Vertex
            The first input vertex. This is considered the start vertex.
        vertexB : topologic_core.Vertex
            The second input vertex. This is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: The input vertexA parameter is not a valid topologic vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: The input vertexB parameter is not a valid topologic vertex. Returning None.")
            return None
        try:
            distance = Vertex.Distance(vertexA, vertexB, mantissa=None, tolerance=tolerance, silent=True)
        except Exception:
            distance = None
        if distance is None or distance <= tolerance:
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: The distance between the input vertices is less than or equal to the input tolerance. Returning None.")
            return None
        try:
            edge = Core.Edge.ByStartVertexEndVertex(vertexA, vertexB)
        except Exception:
            edge = None
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: Could not create an edge. Returning None.")
            return None
        return edge
    
    @staticmethod
    def ByOriginDirectionLength(origin=None, direction=[0, 0, 1], length: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge from an origin, direction, and length.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin (start vertex) of the edge. If None, the global origin is used. Default is None.
        direction : list , optional
            The desired direction vector of the edge. Default is [0, 0, 1].
        length : float , optional
            The desired length of the edge. Default is 1.0.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        try:
            length = float(length)
        except Exception:
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input length parameter is not numerical. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input length parameter must be greater than the input tolerance. Returning None.")
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) < 3:
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None
        try:
            dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
        except Exception:
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input direction parameter is not numerical. Returning None.")
            return None
        magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
        if magnitude <= tolerance:
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input direction vector has zero magnitude. Returning None.")
            return None
        direction = [dx/magnitude, dy/magnitude, dz/magnitude]
        endVertex = Topology.TranslateByDirectionDistance(origin, direction=direction, distance=length)
        return Edge.ByVertices(origin, endVertex, tolerance=tolerance, silent=silent)

    @staticmethod
    def ByVertices(*vertices, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge that connects the first and last valid input vertices.

        Parameters
        ----------
        vertices : list
            The input vertices. Nested lists and positional vertex arguments are accepted. The first valid vertex is
            considered the start vertex and the last valid vertex is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Topology import Topology

        vertexList = Helper.Flatten(list(vertices))
        vertexList = [vertex for vertex in vertexList if Topology.IsInstance(vertex, "Vertex")]
        if len(vertexList) < 2:
            if not silent:
                print("Edge.ByVertices - Error: The input vertices parameter contains fewer than two valid vertices. Returning None.")
            return None
        return Edge.ByStartVertexEndVertex(vertexList[0], vertexList[-1], tolerance=tolerance, silent=silent)
    
    @staticmethod
    def ByVerticesCluster(cluster, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge that connects the first and last vertices of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of vertices.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Edge.ByVerticesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        vertices = Topology.Vertices(cluster) or []
        vertices = [vertex for vertex in vertices if Topology.IsInstance(vertex, "Vertex")]
        if len(vertices) < 2:
            if not silent:
                print("Edge.ByVerticesCluster - Error: The input cluster parameter contains fewer than two vertices. Returning None.")
            return None
        return Edge.ByStartVertexEndVertex(vertices[0], vertices[-1], tolerance=tolerance, silent=silent)

    @staticmethod
    def Connection(edgeA, edgeB, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the shortest straight edge connecting endpoint vertices of the two input edges.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The connecting edge, or None if a valid connecting edge cannot be created.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Connection - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Connection - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None

        endpointsA = [Edge.StartVertex(edgeA, silent=True), Edge.EndVertex(edgeA, silent=True)]
        endpointsB = [Edge.StartVertex(edgeB, silent=True), Edge.EndVertex(edgeB, silent=True)]
        pairs = [(a, b) for a in endpointsA for b in endpointsB]
        best_pair = None
        best_distance = None
        for vertexA, vertexB in pairs:
            if not Topology.IsInstance(vertexA, "Vertex") or not Topology.IsInstance(vertexB, "Vertex"):
                continue
            distance = Vertex.Distance(vertexA, vertexB, mantissa=None, tolerance=tolerance, silent=True)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_pair = [vertexA, vertexB]

        if best_pair is None or best_distance is None or best_distance <= tolerance:
            if not silent:
                print("Edge.Connection - Warning: Could not create a non-degenerate connecting edge. Returning None.")
            return None
        result = Edge.ByVertices(best_pair, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.Connection - Warning: Could not connect the two edges. Returning None.")
            return None
        return result
    
    @staticmethod
    def Direction(edge, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the unit chord direction of the input edge.

        For a curved edge, the direction is defined by the vector from its oriented start vertex to its oriented
        end vertex. It is therefore a global chord direction rather than a local curve tangent.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance used to detect a degenerate chord. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The unit chord direction of the input edge.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Direction - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)
        if not Topology.IsInstance(sv, "Vertex") or not Topology.IsInstance(ev, "Vertex"):
            return None
        x1, y1, z1 = Vertex.Coordinates(sv, mantissa=None)
        x2, y2, z2 = Vertex.Coordinates(ev, mantissa=None)
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
        if magnitude <= tolerance:
            if not silent:
                print("Edge.Direction - Error: The input edge has a degenerate start-to-end chord. Returning None.")
            return None
        return [round(dx / magnitude, mantissa), round(dy / magnitude, mantissa), round(dz / magnitude, mantissa)]
    
    @staticmethod
    def EndVertex(edge, silent: bool = False):
        """
        Returns the end vertex of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The end vertex of the input edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.EndVertex - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            vertex = Core.InstanceCall(edge, "EndVertex")
        except Exception:
            vertex = None
        return vertex if Topology.IsInstance(vertex, "Vertex") else None
    
    @staticmethod
    def Equation2D(edge, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the 2D line equation of the start-to-end chord of the input edge in the XY plane.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance used to identify a vertical line. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary with keys ``slope``, ``x_intercept``, and ``y_intercept``.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Equation2D - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)
        if not Topology.IsInstance(sv, "Vertex") or not Topology.IsInstance(ev, "Vertex"):
            return None
        x1, y1, _ = Vertex.Coordinates(sv, mantissa=None)
        x2, y2, _ = Vertex.Coordinates(ev, mantissa=None)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) <= tolerance:
            return {"slope": float("inf"), "x_intercept": round(x1, mantissa), "y_intercept": None}
        slope = dy / dx
        intercept = y1 - slope * x1
        return {"slope": round(slope, mantissa), "x_intercept": None, "y_intercept": round(intercept, mantissa)}


    @staticmethod
    def Extend(edge, distance: float = 1.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Extends the input edge by the input distance.

        For curved edges under the PythonOCC backend, extension follows the local curve tangent when evaluation beyond
        the trimmed curve is not supported directly by OCCT. TopologicCore retains its historical pathway.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The extension distance. Default is 1.0.
        bothSides : bool , optional
            If set to True, half the distance is added at each end. Default is True.
        reverse : bool , optional
            If set to True and bothSides is False, the start side is extended. Otherwise, the end side is extended.
            Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The extended edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Extend - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            distance = abs(float(distance))
        except Exception:
            if not silent:
                print("Edge.Extend - Error: The input distance parameter is not a valid number. Returning None.")
            return None
        if distance <= tolerance:
            return edge

        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)
        if bothSides:
            new_start = Edge.VertexByDistance(edge, distance=-0.5 * distance, origin=start, tolerance=tolerance, silent=True)
            new_end = Edge.VertexByDistance(edge, distance=0.5 * distance, origin=end, tolerance=tolerance, silent=True)
        elif reverse:
            new_start = Edge.VertexByDistance(edge, distance=-distance, origin=start, tolerance=tolerance, silent=True)
            new_end = end
        else:
            new_start = start
            new_end = Edge.VertexByDistance(edge, distance=distance, origin=end, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(new_start, "Vertex") or not Topology.IsInstance(new_end, "Vertex"):
            if not silent:
                print("Edge.Extend - Error: Could not compute valid extension vertices. Returning None.")
            return None
        return Edge.ByVertices([new_start, new_end], tolerance=tolerance, silent=silent)

    @staticmethod
    def ExtendToEdge(edgeA, edgeB, mantissa: int = 6, step: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Extends the first input edge to meet the second input edge.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge. This edge will be extended to meet edgeB.
        edgeB : topologic_core.Edge
            The second input edge. This edge will be used as the target.
        mantissa : int , optional
            The number of decimal places to round intermediate scalar results to. Default is 6.
        step : bool , optional
            Retained for API compatibility. The current geometric construction does not require iterative stepping.
            Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The extended edge. If the input edges are collinear, parallel, or extension fails, the shortest endpoint
            connection is returned when one can be created.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        if not Edge.IsCoplanar(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edges are not coplanar. Returning the original edge.")
            return edgeA
        if Edge.IsCollinear(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Warning: The input edges are collinear. Returning their shortest endpoint connection.")
            return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)
        if Edge.IsParallel(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Warning: The input edges are parallel. Returning their shortest endpoint connection.")
            return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)

        startA = Edge.StartVertex(edgeA, silent=True)
        endA = Edge.EndVertex(edgeA, silent=True)
        distance_start = Vertex.Distance(startA, edgeB, mantissa=None, tolerance=tolerance, silent=True)
        distance_end = Vertex.Distance(endA, edgeB, mantissa=None, tolerance=tolerance, silent=True)
        direction = Edge.Direction(edgeA, mantissa=15, tolerance=tolerance, silent=True)
        if distance_start is None or distance_end is None or direction is None:
            return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)

        if distance_start < distance_end:
            fixed = endA
            moving = startA
            direction = Vector.Reverse(direction)
        else:
            fixed = startA
            moving = endA

        extension_distance = max(distance_start, distance_end) * 2.0
        moved = Topology.TranslateByDirectionDistance(moving, direction=direction, distance=extension_distance)
        candidate = Edge.ByVertices([fixed, moved], tolerance=tolerance, silent=True)
        if Topology.IsInstance(candidate, "Edge"):
            intersection = Topology.Intersect(candidate, edgeB, tolerance=tolerance)
            if Topology.IsInstance(intersection, "Vertex"):
                return Edge.ByVertices([fixed, intersection], tolerance=tolerance, silent=silent)

        if not silent:
            print("Edge.ExtendToEdge - Warning: The extension operation failed. Returning the shortest endpoint connection.")
        return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def ExternalBoundary(edge, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the external boundary (cluster of end vertices) of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            The external boundary of the input edge. This is a cluster of the edge's end vertices.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster


        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.ExternalBoundary - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        return Cluster.ByTopologies([Edge.StartVertex(edge), Edge.EndVertex(edge)])
    
    @staticmethod
    def Index(edge, edges: list, strict: bool = False, tolerance: float = 0.0001, silent: bool = False) -> int:
        """
        Returns the index of the input edge in the input list of edges.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        edges : list
            The input list of edges.
        strict : bool , optional
            If set to True, the edge must be topologically identical. Otherwise, endpoint coincidence within
            tolerance is used and reversed orientation is accepted. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int
            The index of the input edge, or None if no matching edge is found.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Index - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        if not isinstance(edges, list):
            if not silent:
                print("Edge.Index - Error: The input edges parameter is not a valid list. Returning None.")
            return None
        candidates = [candidate for candidate in edges if Topology.IsInstance(candidate, "Edge")]
        if len(candidates) < 1:
            if not silent:
                print("Edge.Index - Error: The input edges parameter contains no valid edges. Returning None.")
            return None
        startA = Edge.StartVertex(edge, silent=True)
        endA = Edge.EndVertex(edge, silent=True)
        for index, candidate in enumerate(candidates):
            if strict:
                if Topology.IsSame(edge, candidate):
                    return index
                continue
            startB = Edge.StartVertex(candidate, silent=True)
            endB = Edge.EndVertex(candidate, silent=True)
            direct = Vertex.IsCoincident(startA, startB, tolerance=tolerance, silent=True) and Vertex.IsCoincident(endA, endB, tolerance=tolerance, silent=True)
            reverse = Vertex.IsCoincident(startA, endB, tolerance=tolerance, silent=True) and Vertex.IsCoincident(endA, startB, tolerance=tolerance, silent=True)
            if direct or reverse:
                return index
        return None

    @staticmethod
    def Intersect2D(edgeA, edgeB, silent: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the intersection of the infinite 2D lines defined by the two input edge chords.

        The input edges are assumed to be represented in the XY plane. The returned intersection need not lie within
        the trimmed extents of either edge. Curved edges are interpreted by their start-to-end chords.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        mantissa : int , optional
            The number of decimal places to round the returned coordinates to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Vertex
            The intersection vertex, or None if the lines are parallel, collinear, or invalid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Intersect2D - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Intersect2D - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None

        a0 = Edge.StartVertex(edgeA, silent=True)
        a1 = Edge.EndVertex(edgeA, silent=True)
        b0 = Edge.StartVertex(edgeB, silent=True)
        b1 = Edge.EndVertex(edgeB, silent=True)
        vertices = [a0, a1, b0, b1]
        if not all(Topology.IsInstance(vertex, "Vertex") for vertex in vertices):
            return None

        for vertexA in (a0, a1):
            for vertexB in (b0, b1):
                if Vertex.IsCoincident(vertexA, vertexB, tolerance=tolerance, silent=True):
                    x, y, _ = Vertex.Coordinates(vertexA, mantissa=None)
                    return Vertex.ByCoordinates(round(x, mantissa), round(y, mantissa), 0)

        ax0, ay0, _ = Vertex.Coordinates(a0, mantissa=None)
        ax1, ay1, _ = Vertex.Coordinates(a1, mantissa=None)
        bx0, by0, _ = Vertex.Coordinates(b0, mantissa=None)
        bx1, by1, _ = Vertex.Coordinates(b1, mantissa=None)
        rx, ry = ax1 - ax0, ay1 - ay0
        sx, sy = bx1 - bx0, by1 - by0
        r_length = (rx * rx + ry * ry) ** 0.5
        s_length = (sx * sx + sy * sy) ** 0.5
        if r_length <= tolerance or s_length <= tolerance:
            if not silent:
                print("Edge.Intersect2D - Error: One or both input edge chords are degenerate in XY. Returning None.")
            return None

        denominator = rx * sy - ry * sx
        sine = abs(denominator) / (r_length * s_length)
        if sine <= tolerance:
            qpx, qpy = bx0 - ax0, by0 - ay0
            collinear_distance = abs(qpx * ry - qpy * rx) / r_length
            if not silent:
                if collinear_distance <= tolerance:
                    print("Edge.Intersect2D - Error: The input edge chords are collinear. An intersection vertex cannot be uniquely determined. Returning None.")
                else:
                    print("Edge.Intersect2D - Error: The input edge chords are parallel. Returning None.")
            return None

        qpx, qpy = bx0 - ax0, by0 - ay0
        t = (qpx * sy - qpy * sx) / denominator
        x = ax0 + t * rx
        y = ay0 + t * ry
        return Vertex.ByCoordinates(round(x, mantissa), round(y, mantissa), 0)

    @staticmethod
    def IsCollinear(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns True if the start-to-end chords of the two input edges are collinear.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            Retained for API compatibility. Geometric calculations are performed at full precision. Default is 6.
        tolerance : float , optional
            The desired distance tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the edge chords are collinear. False otherwise.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge") or not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.IsCollinear - Error: One or both input parameters are not valid topologic edges. Returning None.")
            return None
        a0 = Vertex.Coordinates(Edge.StartVertex(edgeA, silent=True), mantissa=None)
        a1 = Vertex.Coordinates(Edge.EndVertex(edgeA, silent=True), mantissa=None)
        b0 = Vertex.Coordinates(Edge.StartVertex(edgeB, silent=True), mantissa=None)
        b1 = Vertex.Coordinates(Edge.EndVertex(edgeB, silent=True), mantissa=None)
        ax, ay, az = a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]
        lengthA = math.sqrt(ax*ax + ay*ay + az*az)
        bx, by, bz = b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2]
        lengthB = math.sqrt(bx*bx + by*by + bz*bz)
        if lengthA <= tolerance or lengthB <= tolerance:
            if not silent:
                print("Edge.IsCollinear - Error: One or both input edges are degenerate. Returning None.")
            return None

        def point_line_distance(point):
            px, py, pz = point[0]-a0[0], point[1]-a0[1], point[2]-a0[2]
            cx = py*az - pz*ay
            cy = pz*ax - px*az
            cz = px*ay - py*ax
            return math.sqrt(cx*cx + cy*cy + cz*cz) / lengthA

        return bool(point_line_distance(b0) <= tolerance and point_line_distance(b1) <= tolerance)
    
    @staticmethod
    def IsCoplanar(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns True if the infinite lines defined by the two input edge chords are coplanar.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            Retained for API compatibility. Geometric calculations are performed at full precision. Default is 6.
        tolerance : float , optional
            The desired distance tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the two edge chords are coplanar. False otherwise.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge") or not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.IsCoplanar - Error: One or both input parameters are not valid topologic edges. Returning None.")
            return None
        a0 = Vertex.Coordinates(Edge.StartVertex(edgeA, silent=True), mantissa=None)
        a1 = Vertex.Coordinates(Edge.EndVertex(edgeA, silent=True), mantissa=None)
        b0 = Vertex.Coordinates(Edge.StartVertex(edgeB, silent=True), mantissa=None)
        b1 = Vertex.Coordinates(Edge.EndVertex(edgeB, silent=True), mantissa=None)
        ax, ay, az = a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]
        bx, by, bz = b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2]
        lengthA = math.sqrt(ax*ax + ay*ay + az*az)
        lengthB = math.sqrt(bx*bx + by*by + bz*bz)
        if lengthA <= tolerance or lengthB <= tolerance:
            if not silent:
                print("Edge.IsCoplanar - Error: One or both input edges are degenerate. Returning None.")
            return None
        nx = ay*bz - az*by
        ny = az*bx - ax*bz
        nz = ax*by - ay*bx
        normal_length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if normal_length <= tolerance * lengthA * lengthB:
            return True  # Any two non-degenerate parallel lines are coplanar.
        rx, ry, rz = b0[0]-a0[0], b0[1]-a0[1], b0[2]-a0[2]
        separation = abs(rx*nx + ry*ny + rz*nz) / normal_length
        return bool(separation <= tolerance)

    @staticmethod
    def IsParallel(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns True if the start-to-end chords of the two input edges are parallel.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            Retained for API compatibility. Geometric calculations are performed at full precision. Default is 6.
        tolerance : float , optional
            The desired tolerance applied to the sine of the angle between the chord directions. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the edge chords are parallel. False otherwise.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge") or not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.IsParallel - Error: One or both input parameters are not valid topologic edges. Returning None.")
            return None
        a0 = Vertex.Coordinates(Edge.StartVertex(edgeA, silent=True), mantissa=None)
        a1 = Vertex.Coordinates(Edge.EndVertex(edgeA, silent=True), mantissa=None)
        b0 = Vertex.Coordinates(Edge.StartVertex(edgeB, silent=True), mantissa=None)
        b1 = Vertex.Coordinates(Edge.EndVertex(edgeB, silent=True), mantissa=None)
        ax, ay, az = a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]
        bx, by, bz = b1[0]-b0[0], b1[1]-b0[1], b1[2]-b0[2]
        lengthA = math.sqrt(ax*ax + ay*ay + az*az)
        lengthB = math.sqrt(bx*bx + by*by + bz*bz)
        if lengthA <= tolerance or lengthB <= tolerance:
            if not silent:
                print("Edge.IsParallel - Error: One or both input edges are degenerate. Returning None.")
            return None
        cx = ay*bz - az*by
        cy = az*bx - ax*bz
        cz = ax*by - ay*bx
        sine = math.sqrt(cx*cx + cy*cy + cz*cz) / (lengthA * lengthB)
        return bool(sine <= tolerance)

    @staticmethod
    def Length(edge, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the geometric length of the input edge.

        For the PythonOCC backend, the length is evaluated from the underlying OCCT curve rather than from the
        endpoint chord. The TopologicCore pathway continues to use its native EdgeUtility implementation.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The geometric length of the input edge.

        """
        import math
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Length - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance"):
                length = Core.EdgeUtility.Length(edge, tolerance)
            else:
                length = Core.EdgeUtility.Length(edge)
        except Exception:
            length = None
        if not isinstance(length, (int, float)) or not math.isfinite(float(length)):
            if not silent:
                print("Edge.Length - Error: Could not compute the length of the input edge. Returning None.")
            return None
        return round(float(length), mantissa)

    @staticmethod
    def Line(origin=None, length: float = 1.0, direction: list = [1, 0, 0], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge using the input parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin. If None, the global origin is used. Default is None.
        length : float , optional
            The desired length of the edge. Default is 1.0.
        direction : list , optional
            The desired direction vector. Default is [1, 0, 0].
        placement : str , optional
            The placement of the origin relative to the edge: "center", "start", or "end". Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.Line - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(length, (int, float)) or float(length) <= tolerance:
            if not silent:
                print("Edge.Line - Error: The input length must be greater than the input tolerance. Returning None.")
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Edge.Line - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None
        try:
            dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
        except Exception:
            if not silent:
                print("Edge.Line - Error: The input direction parameter is not numerical. Returning None.")
            return None
        magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
        if magnitude <= tolerance:
            if not silent:
                print("Edge.Line - Error: The input direction vector has zero magnitude. Returning None.")
            return None
        direction = [dx/magnitude, dy/magnitude, dz/magnitude]
        if not isinstance(placement, str):
            if not silent:
                print("Edge.Line - Error: The input placement parameter is not a valid string. Returning None.")
            return None
        placement = placement.lower()
        ox, oy, oz = Vertex.Coordinates(origin, mantissa=None)
        half = 0.5 * float(length)
        if placement == "center":
            start_distance, end_distance = -half, half
        elif placement == "start":
            start_distance, end_distance = 0.0, float(length)
        elif placement == "end":
            start_distance, end_distance = -float(length), 0.0
        else:
            if not silent:
                print("Edge.Line - Error: The input placement string is not one of center, start, or end. Returning None.")
            return None
        sv = Vertex.ByCoordinates(ox + direction[0]*start_distance, oy + direction[1]*start_distance, oz + direction[2]*start_distance)
        ev = Vertex.ByCoordinates(ox + direction[0]*end_distance, oy + direction[1]*end_distance, oz + direction[2]*end_distance)
        return Edge.ByVertices(sv, ev, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def Normal(edge, angle: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a unit normal vector to the input edge.

        For a curved PythonOCC edge, the normal is evaluated at normalized parameter 0.5 from the local OCCT
        tangent. For TopologicCore, the historical start-to-end chord convention is preserved.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        angle : float , optional
            The rotational offset in degrees around the edge direction. Default is 0.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The unit normal vector.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Normal - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        normal_edge = Edge.NormalEdge(edge, length=1.0, u=0.5, angle=angle, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(normal_edge, "Edge"):
            return None
        return Edge.Direction(normal_edge, tolerance=tolerance, silent=silent)

    @staticmethod
    def NormalEdge(edge, length: float = 1.0, u: float = 0.5, angle: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge normal to the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        length : float , optional
            The desired length of the normal edge. Default is 1.0.
        u : float , optional
            The normalized parameter at which to place the normal edge. Default is 0.5.
        angle : float , optional
            The rotational offset in degrees around the local edge direction. Default is 0.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The normal edge.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.NormalEdge - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        if not isinstance(length, (int, float)) or float(length) <= tolerance:
            if not silent:
                print("Edge.NormalEdge - Error: The input length parameter must be greater than the input tolerance. Returning None.")
            return None
        try:
            u = float(u)
            angle = float(angle)
        except Exception:
            if not silent:
                print("Edge.NormalEdge - Error: The input u or angle parameter is not numerical. Returning None.")
            return None

        # PythonOCC-native local curve frame.
        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance") and Core.HasAttribute("EdgeUtility", "TangentAtParameter"):
                origin = Edge.VertexByParameter(edge, u=u, tolerance=tolerance, silent=True)
                tangent = Core.EdgeUtility.TangentAtParameter(edge, u)
                normal = Core.EdgeUtility.NormalAtParameter(edge, u)
                if Topology.IsInstance(origin, "Vertex") and isinstance(tangent, (list, tuple)) and isinstance(normal, (list, tuple)):
                    tx, ty, tz = [float(value) for value in tangent[:3]]
                    nx, ny, nz = [float(value) for value in normal[:3]]
                    radians = math.radians(angle)
                    c = math.cos(radians)
                    s = math.sin(radians)
                    dot = tx*nx + ty*ny + tz*nz
                    rx = nx*c + (ty*nz - tz*ny)*s + tx*dot*(1-c)
                    ry = ny*c + (tz*nx - tx*nz)*s + ty*dot*(1-c)
                    rz = nz*c + (tx*ny - ty*nx)*s + tz*dot*(1-c)
                    magnitude = math.sqrt(rx*rx + ry*ry + rz*rz)
                    if magnitude > tolerance:
                        rx, ry, rz = rx/magnitude, ry/magnitude, rz/magnitude
                        x, y, z = Vertex.Coordinates(origin, mantissa=None)
                        end = Vertex.ByCoordinates(x + rx*length, y + ry*length, z + rz*length)
                        result = Edge.ByVertices(origin, end, tolerance=tolerance, silent=silent)
                        if Topology.IsInstance(result, "Edge"):
                            return result
        except Exception:
            pass

        # Historical TopologicCore-compatible chord pathway.
        direction = Edge.Direction(edge, mantissa=15, tolerance=tolerance, silent=True)
        if direction is None:
            return None
        dx, dy, dz = direction
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            normal = [1.0, 0.0, 0.0]
        elif abs(dz) <= tolerance:
            normal = [-dy, dx, 0.0]
        else:
            normal = [dy, -dx, 0.0]
        magnitude = math.sqrt(sum(value*value for value in normal))
        if magnitude <= tolerance:
            return None
        normal = [value/magnitude for value in normal]
        origin = Edge.StartVertex(edge, silent=True)
        x, y, z = Vertex.Coordinates(origin, mantissa=None)
        unit_end = Vertex.ByCoordinates(x + normal[0], y + normal[1], z + normal[2])
        normal_edge = Edge.ByVertices(origin, unit_end, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(normal_edge, "Edge"):
            return None
        normal_edge = Edge.SetLength(normal_edge, length, bothSides=False, tolerance=tolerance, silent=True)
        normal_edge = Topology.Rotate(normal_edge, origin=Edge.StartVertex(normal_edge, silent=True), axis=direction, angle=angle)
        distance = Edge.Length(edge, mantissa=15, tolerance=tolerance, silent=True) * u
        return Topology.TranslateByDirectionDistance(normal_edge, direction, distance)

    @staticmethod
    def Normalize(edge, useEndVertex: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a normalized straight edge with the same chord direction as the input edge and a length of 1.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        useEndVertex : bool , optional
            If True, the normalized edge end vertex is placed at the end vertex of the input edge. Otherwise,
            its start vertex is placed at the start vertex of the input edge. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The normalized edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Normalize - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        direction = Edge.Direction(edge, mantissa=15, tolerance=tolerance, silent=True)
        if direction is None:
            if not silent:
                print("Edge.Normalize - Error: Could not compute a valid edge direction. Returning None.")
            return None
        if not useEndVertex:
            sv = Edge.StartVertex(edge, silent=True)
            if not Topology.IsInstance(sv, "Vertex"):
                return None
            x, y, z = Vertex.Coordinates(sv, mantissa=None)
            ev = Vertex.ByCoordinates(x + direction[0], y + direction[1], z + direction[2])
        else:
            ev = Edge.EndVertex(edge, silent=True)
            if not Topology.IsInstance(ev, "Vertex"):
                return None
            x, y, z = Vertex.Coordinates(ev, mantissa=None)
            sv = Vertex.ByCoordinates(x - direction[0], y - direction[1], z - direction[2])
        return Edge.ByVertices(sv, ev, tolerance=tolerance, silent=silent)

    @staticmethod
    def ParameterAtVertex(edge, vertex, mantissa: int = 6, silent: bool = False, tolerance: float = 0.0001) -> float:
        """
        Returns the normalized parameter of a vertex located on the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        vertex : topologic_core.Vertex
            The input vertex.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired distance tolerance used to determine whether the vertex lies on the edge. Default is 0.0001.

        Returns
        -------
        float
            The normalized parameter in the range 0 to 1, or None if the vertex is not on the edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.ParameterAtVertex - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Edge.ParameterAtVertex - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
            return None
        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance"):
                parameter = Core.EdgeUtility.ParameterAtPoint(edge, vertex, tolerance)
            else:
                parameter = Core.EdgeUtility.ParameterAtPoint(edge, vertex)
        except Exception:
            return None
        if parameter is None:
            return None
        try:
            parameter = float(parameter)
        except Exception:
            return None
        if abs(parameter) <= tolerance:
            parameter = 0.0
        elif abs(parameter - 1.0) <= tolerance:
            parameter = 1.0
        return round(parameter, mantissa)


    @staticmethod
    def Quadrance(edge, mantissa: int = 6, silent: bool = False) -> float:
        """
        Returns the squared Euclidean distance between the start and end vertices of the input edge.

        For a curved edge this is the squared chord length, not the square of the curve length.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The quadrance of the edge chord.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Quadrance - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)
        a = Vertex.Coordinates(sv, mantissa=None)
        b = Vertex.Coordinates(ev, mantissa=None)
        value = sum((a[i] - b[i]) ** 2 for i in range(3))
        return round(float(value), mantissa)


    @staticmethod
    def Reverse(edge, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates an edge with the reverse orientation of the input edge.

        The PythonOCC pathway reverses the OCCT edge itself and therefore preserves curved geometry. The
        TopologicCore pathway retains the historical reconstruction from swapped endpoints.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The reversed edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Reverse - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance"):
                result = Core.Edge.Reverse(edge, tolerance, silent)
                if Topology.IsInstance(result, "Edge"):
                    return result
        except Exception:
            pass
        return Edge.ByVertices(Edge.EndVertex(edge, silent=True), Edge.StartVertex(edge, silent=True), tolerance=tolerance, silent=silent)
    
    @staticmethod
    def SetLength(edge, length: float = 1.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns an edge with the requested length in the same chord direction as the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        length : float , optional
            The desired length of the edge. Default is 1.0.
        bothSides : bool , optional
            If set to True, the length change is distributed symmetrically between both ends. Default is True.
        reverse : bool , optional
            If bothSides is False, set to True to modify the start side instead of the end side. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The resized edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.SetLength - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        if not isinstance(length, (int, float)) or float(length) <= tolerance:
            if not silent:
                print("Edge.SetLength - Error: The input length parameter must be greater than the input tolerance. Returning None.")
            return None
        current_length = Edge.Length(edge, mantissa=15, tolerance=tolerance, silent=True)
        if current_length is None:
            return None
        delta = float(length) - current_length
        if abs(delta) <= tolerance:
            return edge
        if delta > 0:
            return Edge.Extend(edge=edge, distance=delta, bothSides=bothSides, reverse=reverse, tolerance=tolerance, silent=silent)
        return Edge.Trim(edge=edge, distance=-delta, bothSides=bothSides, reverse=reverse, tolerance=tolerance, silent=silent)

    @staticmethod
    def Spread(edgeA, edgeB, mantissa: int = 6, bracket: bool = False, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the rational-trigonometry spread between the two input edge chord directions.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        bracket : bool , optional
            If set to True, the spread is invariant under edge reversal. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The spread between the two input edges.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Spread - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Spread - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        u = Edge.Direction(edgeA, mantissa=15, tolerance=tolerance, silent=True)
        v = Edge.Direction(edgeB, mantissa=15, tolerance=tolerance, silent=True)
        if u is None or v is None:
            return None
        return Vector.Spread(u, v, mantissa=mantissa, bracket=bracket)

    @staticmethod
    def StartVertex(edge, silent: bool = False):
        """
        Returns the start vertex of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The start vertex of the input edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.StartVertex - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            vertex = Core.InstanceCall(edge, "StartVertex")
        except Exception:
            vertex = None
        return vertex if Topology.IsInstance(vertex, "Vertex") else None

    @staticmethod
    def Trim(edge, distance: float = 0.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Trims the input edge by the input distance.

        For PythonOCC curved edges, the method preserves the underlying OCCT curve whenever the trimmed endpoints
        can be represented as parameters of the original edge. The TopologicCore pathway retains the historical
        endpoint reconstruction.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The total trim distance. Default is 0.0.
        bothSides : bool , optional
            If set to True, half the distance is trimmed from each end. Default is True.
        reverse : bool , optional
            If bothSides is False, set to True to trim from the start rather than the end. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The trimmed edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Trim - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            distance = abs(float(distance))
        except Exception:
            if not silent:
                print("Edge.Trim - Error: The input distance parameter is not numerical. Returning None.")
            return None
        if distance == 0:
            return edge
        if distance <= tolerance:
            if not silent:
                print("Edge.Trim - Warning: The input distance parameter is less than or equal to the input tolerance. Returning the input edge.")
            return edge
        edge_length = Edge.Length(edge, mantissa=15, tolerance=tolerance, silent=True)
        if edge_length is None or distance >= edge_length - tolerance:
            if not silent:
                print("Edge.Trim - Error: The input distance is too large for the input edge. Returning None.")
            return None
        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)
        if bothSides:
            sve = Edge.VertexByDistance(edge, distance=distance*0.5, origin=sv, tolerance=tolerance, silent=True)
            eve = Edge.VertexByDistance(edge, distance=-distance*0.5, origin=ev, tolerance=tolerance, silent=True)
        elif reverse:
            sve = Edge.VertexByDistance(edge, distance=distance, origin=sv, tolerance=tolerance, silent=True)
            eve = ev
        else:
            sve = sv
            eve = Edge.VertexByDistance(edge, distance=-distance, origin=ev, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(sve, "Vertex") or not Topology.IsInstance(eve, "Vertex"):
            return None

        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance"):
                uA = Edge.ParameterAtVertex(edge, sve, mantissa=15, silent=True, tolerance=tolerance)
                uB = Edge.ParameterAtVertex(edge, eve, mantissa=15, silent=True, tolerance=tolerance)
                if uA is not None and uB is not None:
                    result = Core.EdgeUtility.Trim(edge, uA, uB)
                    if Topology.IsInstance(result, "Edge"):
                        return result
        except Exception:
            pass
        return Edge.ByVertices(sve, eve, tolerance=tolerance, silent=silent)

    @staticmethod
    def TrimByEdge(edgeA, edgeB, reverse: bool = False, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Trims the first input edge by the second input edge.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge. This edge will be trimmed by edgeB.
        edgeB : topologic_core.Edge
            The second input edge. This edge will be used to trim edgeA.
        reverse : bool , optional
            If set to True, the segment adjacent to the end vertex of edgeA is preserved. Otherwise, the segment
            adjacent to the start vertex is preserved. Default is False.
        mantissa : int , optional
            The number of decimal places to round scalar geometric comparisons to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The trimmed edge, the original edge when no valid trim is found, or None for a collinear case with no
            usable trimming vertex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        if not Edge.IsCoplanar(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edges are not coplanar. Returning the original edge.")
            return edgeA
        if Edge.IsParallel(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edges are parallel. Returning the original edge.")
            return edgeA

        startA = Edge.StartVertex(edgeA, silent=True)
        endA = Edge.EndVertex(edgeA, silent=True)
        startB = Edge.StartVertex(edgeB, silent=True)
        endB = Edge.EndVertex(edgeB, silent=True)

        if Edge.IsCollinear(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            intersection = None
            if Vertex.IsInternal(startB, edgeA, tolerance=tolerance, silent=True):
                intersection = startB
            elif Vertex.IsInternal(endB, edgeA, tolerance=tolerance, silent=True):
                intersection = endB
            if not Topology.IsInstance(intersection, "Vertex"):
                return None
            anchor = endA if reverse else startA
            return Edge.ByVertices([anchor, intersection], tolerance=tolerance, silent=silent)

        intersection = Topology.Intersect(edgeA, edgeB, tolerance=tolerance)
        if Topology.IsInstance(intersection, "Vertex") and Vertex.IsInternal(intersection, edgeA, tolerance=tolerance, silent=True):
            anchor = endA if reverse else startA
            return Edge.ByVertices([anchor, intersection], tolerance=tolerance, silent=silent)
        return edgeA

    @staticmethod
    def VertexByDistance(edge, distance: float = 0.0, origin=None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex offset by the input distance from an origin along the input edge direction.

        For the PythonOCC backend, an origin that lies on the edge is advanced by true curvilinear distance along
        the underlying OCCT curve. If that native operation is unavailable or unsuitable, the historical chord
        direction behaviour is used.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The signed offset distance. Default is 0.0.
        origin : topologic_core.Vertex , optional
            The origin of the offset. If None, the start vertex is used. Default is None.
        mantissa : int , optional
            The number of decimal places to round returned coordinates to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.VertexByDistance - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Edge.StartVertex(edge, silent=True)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.VertexByDistance - Error: Could not determine a valid origin vertex. Returning None.")
            return None
        try:
            distance = float(distance)
        except Exception:
            if not silent:
                print("Edge.VertexByDistance - Error: The input distance parameter is not numerical. Returning None.")
            return None
        if abs(distance) <= tolerance:
            return origin

        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance"):
                vertex = Core.EdgeUtility.PointAtDistance(edge, distance, origin, tolerance)
                if Topology.IsInstance(vertex, "Vertex"):
                    coordinates = Vertex.Coordinates(vertex, mantissa=None)
                    return Vertex.ByCoordinates([round(value, mantissa) for value in coordinates])
        except Exception:
            pass

        # Historical TopologicCore-compatible chord pathway.
        direction = Edge.Direction(edge, mantissa=15, tolerance=tolerance, silent=True)
        if direction is None:
            return None
        x, y, z = Vertex.Coordinates(origin, mantissa=None)
        return Vertex.ByCoordinates(
            round(x + direction[0]*distance, mantissa),
            round(y + direction[1]*distance, mantissa),
            round(z + direction[2]*distance, mantissa),
        )
    
    @staticmethod
    def VertexByParameter(edge, u: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex at normalized parameter u along the input edge.

        Under the PythonOCC backend, the vertex is evaluated on the underlying OCCT curve. Under TopologicCore,
        the historical TopologicPy chord-based pathway is retained for compatibility.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        u : float , optional
            The normalized edge parameter. A value of 0 returns the start vertex and 1 returns the end vertex.
            Default is 0.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.VertexByParameter - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            u = float(u)
        except Exception:
            if not silent:
                print("Edge.VertexByParameter - Error: The input u parameter is not numerical. Returning None.")
            return None
        if u == 0.0:
            return Edge.StartVertex(edge, silent=True)
        if u == 1.0:
            return Edge.EndVertex(edge, silent=True)

        # New PythonOCC capability: true OCCT curve evaluation.
        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance"):
                vertex = Core.EdgeUtility.PointAtParameter(edge, u)
                if Topology.IsInstance(vertex, "Vertex"):
                    return vertex
                return None
        except Exception:
            return None

        # Historical TopologicCore pathway: move from the start vertex along
        # the start-to-end chord by curve_length * u.
        direction = Edge.Direction(edge, tolerance=tolerance, silent=True)
        edge_length = Edge.Length(edge, mantissa=15, tolerance=tolerance, silent=True)
        start = Edge.StartVertex(edge, silent=True)
        if direction is None or edge_length is None or not Topology.IsInstance(start, "Vertex"):
            return None
        return Topology.TranslateByDirectionDistance(start, direction=direction, distance=edge_length*u)

    @staticmethod
    def Vertices(edge, silent: bool = False) -> list:
        """
        Returns the vertices of the input edge in start-to-end order.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list containing the start and end vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Vertices - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        vertices = []
        try:
            Core.InstanceCall(edge, "Vertices", None, vertices)
        except Exception:
            vertices = []
        vertices = [vertex for vertex in vertices if Topology.IsInstance(vertex, "Vertex")]
        if len(vertices) >= 2:
            return vertices
        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)
        return [vertex for vertex in [start, end] if Topology.IsInstance(vertex, "Vertex")]
