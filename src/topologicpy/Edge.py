# Copyright (C) 2025
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import topologic_core as topologic

class Edge():

    @staticmethod
    def Align2D(edgeA, edgeB):
        """
        Compute the 4x4 transformation matrix to fully align edgeA to edgeB.

        Parameters:
            edge1 (Edge): The source 2D edge to transform.
            edge2 (Edge): The target 2D edge.

        Returns:
            list: A 4x4 transformation matrix.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Matrix import Matrix
        from topologicpy.Vector import Vector

        centroid1 = Topology.Centroid(edgeA)
        centroid2 = Topology.Centroid(edgeB)
        # Extract coordinates
        x1, y1, z1 = Vertex.Coordinates(centroid1)
        x2, y2, z2 = Vertex.Coordinates(centroid2)

        # Translation to move edge1 to the origin
        move_to_origin = Matrix.ByTranslation(-x1, -y1, -z1)
        
        # Translation to move edge1 from the origin to the start of edge2
        move_to_target = Matrix.ByTranslation(x2, y2, z2)

        # Lengths of the edges
        length1 = Edge.Length(edgeA)
        length2 = Edge.Length(edgeB)

        if length1 == 0 or length2 == 0:
            raise ValueError("Edges must have non-zero length.")

        # Calculate scaling factor
        scale_factor = length2 / length1
        # Scaling matrix
        scaling_matrix = Matrix.ByScaling(scale_factor, scale_factor, 1.0)

        # Calculate angles of the edges relative to the X-axis
        angle1 = Vector.CompassAngle(Edge.Direction(edgeA), [1,0,0])
        angle2 = Vector.CompassAngle(Edge.Direction(edgeB), [1,0,0])
        # Rotation angle
        rotation_angle = angle2 - angle1
        # Rotation matrix (about Z-axis for 2D alignment)
        rotation_matrix = Matrix.ByRotation(0, 0, rotation_angle, order="xyz")

        # Combine transformations: Move to origin -> Scale -> Rotate -> Move to target
        transformation_matrix = Matrix.Multiply(scaling_matrix, move_to_origin)
        transformation_matrix = Matrix.Multiply(rotation_matrix, transformation_matrix)
        transformation_matrix = Matrix.Multiply(move_to_target, transformation_matrix)
        return transformation_matrix

    @staticmethod
    def Angle(edgeA, edgeB, mantissa: int = 6, bracket: bool = False) -> float:
        """
        Returns the angle in degrees between the two input edges.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic Edge
            The second input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        bracket : bool
            If set to True, the returned angle is bracketed between 0 and 180. The default is False.

        Returns
        -------
        float
            The angle in degrees between the two input edges.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(edgeA, "Edge"):
            print("Edge.Angle - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            print("Edge.Angle - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        dirA = Edge.Direction(edgeA, mantissa)
        dirB = Edge.Direction(edgeB, mantissa)
        ang = Vector.Angle(dirA, dirB)
        if bracket:
            if ang > 90:
                ang = 180 - ang
        return round(ang, mantissa)

    @staticmethod
    def Bisect(edgeA, edgeB, length: float = 1.0, placement: int = 0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a bisecting edge between edgeA and edgeB.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first topologic Edge.
        edgeB : topologic Edge
            The second topologic Edge.
        length : float , optional
            The desired length of the bisecting edge. The default is 1.0.
        placement : int , optional
            The desired placement of the bisecting edge.
            If set to 0, the bisecting edge centroid will be placed at the end vertex of the first edge.
            If set to 1, the bisecting edge start vertex will be placed at the end vertex of the first edge.
            If set to 2, the bisecting edge end vertex will be placed at the end vertex of the first edge.
            If set to any number other than 0, 1, or 2, the bisecting edge centroid will be placed at the end vertex of the first edge. The default is 0.
        tolerance : float , optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Edge
            The created bisecting edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        def process_edges(edge1, edge2, tolerance=0.0001):
            start1 = Edge.StartVertex(edge1)
            end1 = Edge.EndVertex(edge1)
            start2 = Edge.StartVertex(edge2)
            end2 = Edge.EndVertex(edge2)
            
            shared_vertex = None
            
            if Vertex.Distance(start1, start2) <= tolerance:
                shared_vertex = start1
            elif Vertex.Distance(start1, end2) <= tolerance:
                shared_vertex = start1
                edge2 = Edge.Reverse(edge2)
            elif Vertex.Distance(end1, start2) <= tolerance:
                shared_vertex = start2
                edge1 = Edge.Reverse(edge1)
            elif Vertex.Distance(end1, end2) <= tolerance:
                shared_vertex = end1
                edge1 = Edge.Reverse(edge1)
                edge2 = Edge.Reverse(edge2)
            
            if shared_vertex is None:
                return [None, None]
            return edge1, edge2
        
        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Bisect - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Bisect - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        if Edge.Length(edgeA) <= tolerance:
            if not silent:
                print("Edge.Bisect - Error: The input edgeA parameter is shorter than the input tolerance parameter. Returning None.")
            return None
        if Edge.Length(edgeB) <= tolerance:
            if not silent:
                print("Edge.Bisect - Error: The input edgeB parameter is shorter than the input tolerance parameter. Returning None.")
            return None
        
        
        edge1, edge2 = process_edges(edgeA, edgeB, tolerance=tolerance)
        if edge1 == None or edge2 == None:
            if not silent:
                print("Edge.Bisect - Error: The input edgeA and edgeB parameters do not share a vertex and thus cannot be bisected. Returning None.")
            return None
        sv = Edge.StartVertex(edge1)
        dir1 = Edge.Direction(edge1)
        dir2 = Edge.Direction(edge2)
        bisecting_vector = Vector.Bisect(dir1, dir2)
        ev = Topology.TranslateByDirectionDistance(sv, bisecting_vector, length)
        bisecting_edge = Edge.ByVertices([sv, ev], tolerance=tolerance, silent=silent)
        if placement == 0:
            bisecting_edge = Topology.TranslateByDirectionDistance(bisecting_edge, Vector.Reverse(bisecting_vector), length*0.5)
        elif placement == 2:
            bisecting_edge = Topology.TranslateByDirectionDistance(bisecting_edge, Vector.Reverse(bisecting_vector), length)
        return bisecting_edge

    @staticmethod
    def ByFaceNormal(face, origin= None, length: float = 1.0, tolerance: float = 0.0001):
        """
        Creates a straight edge representing the normal to the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face
        origin : topologic_core.Vertex , optional
            The desired origin of the edge. If set to None, the centroid of the face is chosen as the origin of the edge. The default is None.
        length : float , optional
            The desired length of the edge. The default is 1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        edge : topologic_core.Edge
            The created edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        edge = None
        if not Topology.IsInstance(face, "Face"):
            print("Edge.ByFaceNormal - Error: The input face parameter is not a valid topologic face. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(face)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Edge.ByFaceNormal - Error: The input origin parameter is not a valid topologic origin. Returning None.")
            return None
        n = Face.Normal(face)
        v2 = Topology.Translate(origin, n[0], n[1], n[2])
        edge = Edge.ByStartVertexEndVertex(origin, v2, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.ByFaceNormal - Error: Could not create an edge. Returning None.")
            return None
        edge = Edge.SetLength(edge, length, bothSides=False)
        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.ByFaceNormal - Error: Could not create an edge. Returning None.")
            return None
        return edge

    @staticmethod
    def ByOffset2D(edge, offset: float = 1.0, tolerance: float = 0.0001):
        """
        Creates and edge offset from the input edge. This method is intended for edges that are in the XY plane.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        offset : float , optional
            The desired offset. The default is 1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Edge
            An edge offset from the input edge.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        n = Edge.Normal(edge)
        n = Vector.Normalize(n)
        n = Vector.Multiply(n, offset, tolerance=tolerance)
        edge = Topology.Translate(edge, n[0], n[1], n[2])
        return edge

    @staticmethod
    def ByStartVertexEndVertex(vertexA, vertexB, tolerance: float = 0.0001, silent=False):
        """
        Creates a straight edge that connects the input vertices.

        Parameters
        ----------
        vertexA : topologic_core.Vertex
            The first input vertex. This is considered the start vertex.
        vertexB : topologic_core.Vertex
            The second input vertex. This is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        edge : topologic_core.Edge
            The created edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import inspect
        
        edge = None
        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: The input vertexA parameter is not a valid topologic vertex. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: The input vertexB parameter is not a valid topologic vertex. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if Topology.IsSame(vertexA, vertexB):
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: The input vertexA and vertexB parameters are the same vertex. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if Vertex.Distance(vertexA, vertexB) <= tolerance:
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: The distance between the input vertexA and vertexB parameters is less than the input tolerance. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        try:
            edge = topologic.Edge.ByStartVertexEndVertex(vertexA, vertexB)  # Hook to Core
        except:
            if not silent:
                print("Edge.ByStartVertexEndVertex - Error: Could not create an edge. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            edge = None
        return edge
    
    @staticmethod
    def ByOriginDirectionLength(origin = None, direction=[0,0,1], length: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge from the input parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex
            The origin (start vertex) of the edge.
        direction : list , optional
            The desired direction vector of the edge. The default is [0,0,1] (pointing up in the Z direction)
        length: float , optional
            The desired length of edge. The default is 1.0.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin == None:
            origin = Vertex.Origin()
        
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Edge.ByVertexDirectionLength - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        
        if length <= tolerance:
            if not silent:
                print("Edge.ByVertexDirectionLength - Error: The input edge parameter must not be less than the input tolerance parameter. Returning None.")
            return None

        endVertex = Topology.TranslateByDirectionDistance(origin, direction=direction[:3], distance=length)
        edge = Edge.ByVertices(origin, endVertex, tolerance=tolerance, silent=silent)
        return edge

    @staticmethod
    def ByVertices(*args, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge that connects the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices. The first item is considered the start vertex and the last item is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Topology import Topology
        import inspect

        if len(args) == 0:
            if not silent:
                print("Edge.ByVertices - Error: The input vertices parameter is an empty list. Returning None.")
            return None
        if len(args) == 1:
            vertices = args[0]
            if isinstance(vertices, list):
                if len(vertices) == 0:
                    if not silent:
                        print("Edge.ByVertices - Error: The input vertices parameter is an empty list. Returning None.")
                    return None
                else:
                    vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
                    if len(vertexList) == 0:
                        if not silent:
                            print("Edge.ByVertices - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
                        return None
            else:
                if not silent:
                    print("Edge.ByVertices - Warning: The input vertices parameter contains only one vertex. Returning None.")
                return None
        else:
            vertexList = Helper.Flatten(list(args))
            vertexList = [x for x in vertexList if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 2:
            if not silent:
                print("Edge.ByVertices - Error: The input vertices parameter has less than two vertices. Returning None.")
            return None
        edge = Edge.ByStartVertexEndVertex(vertexList[0], vertexList[-1], tolerance=tolerance, silent=silent)
        if not edge:
            if not silent:
                print("Edge.ByVertices - Error: Could not create an edge. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
        return edge
    
    @staticmethod
    def ByVerticesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a straight edge that connects the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of vertices. The first item is considered the start vertex and the last item is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. The default is 0.0001.

        Returns
        -------
        topologic_core.Edge
            The created edge.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Edge.ByVerticesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        vertices = Topology.Vertices(cluster)
        vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 2:
            print("Edge.ByVerticesCluster - Error: The input cluster parameter contains less than two vertices. Returning None.")
            return None
        return Edge.ByStartVertexEndVertex(vertexList[0], vertexList[-1], tolerance=tolerance)

    @staticmethod
    def Connection(edgeA, edgeB, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the edge representing the connection between the first input edge to the second input edge using the two closest vertices.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge. This edge will be extended to meet edgeB.
        edgeB : topologic_core.Edge
            The second input edge. This edge will be used to extend edgeA.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Edge or topologic_core.Wire
            The connected edge. Since it is made of two edges, this method returns a Wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper

        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        svb = Edge.StartVertex(edgeB)
        evb = Edge.EndVertex(edgeB)
        v_list = [[sva, svb], [sva, evb], [eva, svb], [eva, evb]]
        distances = []
        for pair in v_list:
            distances.append(Vertex.Distance(pair[0], pair[1]))
        v_list = Helper.Sort(v_list, distances)
        closest_pair = v_list[0]
        return_edge = Edge.ByVertices(closest_pair, tolerance=tolerance, silent=silent)
        if return_edge == None:
            if not silent:
                print("Edge.ConnectToEdge - Warning: Could not connect the two edges. Returning None.")
            return None
        return return_edge
    
    @staticmethod
    def Direction(edge, mantissa: int = 6) -> list:
        """
        Returns the direction of the input edge expressed as a list of three numbers.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        list
            The direction of the input edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Direction - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        x = Vertex.X(ev, mantissa=mantissa) - Vertex.X(sv, mantissa=mantissa)
        y = Vertex.Y(ev, mantissa=mantissa) - Vertex.Y(sv, mantissa=mantissa)
        z = Vertex.Z(ev, mantissa=mantissa) - Vertex.Z(sv, mantissa=mantissa)
        uvec = Vector.Normalize([x,y,z])
        x = round(uvec[0], mantissa)
        y = round(uvec[1], mantissa)
        z = round(uvec[2], mantissa)
        return [x, y, z]
    
    @staticmethod
    def EndVertex(edge):
        """
        Returns the end vertex of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.

        Returns
        -------
        topologic_core.Vertex
            The end vertex of the input edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.EndVertex - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        vert = None
        try:
            vert = edge.EndVertex() # Hook to core
        except:
            vert = None
        return vert
    
    @staticmethod
    def Equation2D(edge, mantissa=6):
        """
        Returns the 2D equation of the input edge. This is assumed to be in the XY plane.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        dict
            The equation of the edge stored in a dictionary. The dictionary has the following keys:
            "slope": The slope of the line. This can be float('inf')
            "x_intercept": The X axis intercept. This can be None.
            "y_intercept": The Y axis intercept. This can be None.

        """
        from topologicpy.Vertex import Vertex

        # Extract the start and end vertices
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        
        # Extract coordinates of the vertices
        x1, y1 = Vertex.X(sv, mantissa=mantissa), Vertex.Y(sv, mantissa=mantissa)
        x2, y2 = Vertex.X(ev, mantissa=mantissa), Vertex.Y(ev, mantissa=mantissa)
        
        # Calculate the slope (m) and y-intercept (c)
        if x2 - x1 != 0:
            m = round((y2 - y1) / (x2 - x1), mantissa)
            c = round(y1 - m * x1, mantissa)
            return {
                "slope": m,
                "x_intercept": None,
                "y_intercept": c
            }
        else:
            # The line is vertical, slope is undefined
            return {
                "slope": float('inf'),
                "x_intercept": x1,
                "y_intercept": None
            }


    @staticmethod
    def Extend(edge, distance: float = 1.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Extends the input edge by the input distance.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The offset distance. The default is 1.
        bothSides : bool , optional
            If set to True, the edge will be extended by half the distance at each end. The default is False.
        reverse : bool , optional
            If set to True, the edge will be extended from its start vertex. Otherwise, it will be extended from its end vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Edge
            The extended edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Extend - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        distance = abs(distance)
        if distance <= tolerance:
            return edge
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        if bothSides:
            sve = Edge.VertexByDistance(edge, distance=-distance*0.5, origin=sv, tolerance=tolerance)
            eve = Edge.VertexByDistance(edge, distance=distance*0.5, origin=ev, tolerance=tolerance)
        elif reverse:
            sve = Edge.VertexByDistance(edge, distance=-distance, origin=sv, tolerance=tolerance)
            eve = Edge.EndVertex(edge)
        else:
            sve = Edge.StartVertex(edge)
            eve = Edge.VertexByDistance(edge, distance=distance, origin=ev, tolerance=tolerance)
        return Edge.ByVertices([sve, eve], tolerance=tolerance, silent=silent)

    @staticmethod
    def ExtendToEdge(edgeA, edgeB, mantissa: int = 6, step: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Extends the first input edge to meet the second input edge.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge. This edge will be extended to meet edgeB.
        edgeB : topologic_core.Edge
            The second input edge. This edge will be used to extend edgeA.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        topologic_core.Edge
            The extended edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        if not Edge.IsCoplanar(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edges are not coplanar. Returning the original edge.")
            return edgeA
        if Edge.IsCollinear(edgeA, edgeB, tolerance=tolerance):
            if not silent:
                print("Edge.ExtendToEdge - Warning: The input edges are collinear. Connecting the edges instead. Check return value.")
            return Edge.ConnectToEdge(edgeA, edgeB, tolerance=tolerance)
        if Edge.IsParallel(edgeA, edgeB, tolerance=tolerance):
            if not silent:
                print("Edge.ExtendToEdge - Warning: The input edges are parallel. Connecting the edges instead. Returning a Wire.")
            return Edge.ConnectToEdge(edgeA, edgeB, tolerance=tolerance)
        
        
        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        d1 = Vertex.Distance(sva, edgeB)
        d2 = Vertex.Distance(eva, edgeB)
        edge_direction = Edge.Direction(edgeA)
        if d1 < d2:
            v1 = eva
            v2 = sva
            edge_direction = Vector.Reverse(edge_direction)
        else:
            v1 = sva
            v2 = eva
        
        d = max(d1, d2)*2
        v2 = Topology.TranslateByDirectionDistance(v2, direction=edge_direction, distance=d)
        new_edge = Edge.ByVertices([v1, v2], tolerance=tolerance, silent=silent)
        
        svb = Edge.StartVertex(edgeB)
        evb = Edge.EndVertex(edgeB)

        intVertex = Topology.Intersect(new_edge, edgeB, tolerance=tolerance)
        if intVertex:
            return Edge.ByVertices([v1, intVertex], tolerance=tolerance, silent=silent)
        if not silent:
            print("Edge.ExtendToEdge - Warning: The operation failed. Connecting the edges instead. Returning a Wire.")
        return Edge.ConnectToEdge(edgeA, edgeB, tolerance=tolerance)
    
    @staticmethod
    def ExternalBoundary(edge):
        """
        Returns the external boundary (cluster of end vertices) of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.

        Returns
        -------
        topologic_core.Cluster
            The external boundary of the input edge. This is a cluster of the edge's end vertices.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.ExternalBoundary - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        return Cluster.ByTopologies([Edge.StartVertex(edge), Edge.EndVertex(edge)])
    
    @staticmethod
    def Index(edge, edges: list, strict: bool = False, tolerance: float = 0.0001) -> int:
        """
        Returns index of the input edge in the input list of edges

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        edges : list
            The input list of edges.
        strict : bool , optional
            If set to True, the edge must be strictly identical to the one found in the list. Otherwise, a distance comparison is used. The default is False.
        tolerance : float , optional
            The tolerance for computing if the input edge is identical to an edge from the list. The default is 0.0001.

        Returns
        -------
        int
            The index of the input edge in the input list of edges.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Index - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        if not isinstance(edges, list):
            print("Edge.Index - Error: The input edges parameter is not a valid list. Returning None.")
            return None
        edges = [e for e in edges if Topology.IsInstance(e, "Edge")]
        if len(edges) < 1:
            print("Edge.Index - Error: The input edges parameter contains no valid edges. Returning None.")
            return None
        sva = Edge.StartVertex(edge)
        eva = Edge.EndVertex(edge)
        for i in range(len(edges)):
            if strict:
                if Topology.IsSame(edge, edges[i]):
                    return i
            else:
                svb = Edge.StartVertex(edges[i])
                evb = Edge.EndVertex(edges[i])
                dsvsv = Vertex.Distance(sva, svb)
                devev = Vertex.Distance(eva, evb)
                if dsvsv <= tolerance and devev <= tolerance:
                    return i
                dsvev = Vertex.Distance(sva, evb)
                devsv = Vertex.Distance(eva, svb)
                if dsvev <= tolerance and devsv <= tolerance:
                    return i
        return None

    @staticmethod
    def Intersect2D(edgeA, edgeB, silent: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the intersection vertex of the two input edges. This is assumed to be in the XY plane.
        The intersection vertex does not necessarily fall within the extents of either edge.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        topologic_core.Vertex
            The intersection vertex or None if the edges are parallel or collinear.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        svb = Edge.StartVertex(edgeB)
        evb = Edge.EndVertex(edgeB)
        v_list = [[sva, svb], [sva, evb], [eva, svb], [eva, evb]]
        distances = []
        for pair in v_list:
            distances.append(Vertex.Distance(pair[0], pair[1]))
        v_list = Helper.Sort(v_list, distances)
        closest_pair = v_list[0]
        if Vertex.Distance(closest_pair[0], closest_pair[1]) <= tolerance:
            return Topology.Centroid(Cluster.ByTopologies(closest_pair))
        
        if Edge.IsCollinear(edgeA, edgeB, tolerance=tolerance):
            if not silent:
                print("Edge.Intersect2D - Error: The input edges are collinear and overlapping. An intersection vertex cannot be found. Returning None.")
            return None
        if Edge.IsParallel(edgeA, edgeB, tolerance=tolerance):
            if not silent:
                print("Edge.Intersect2D - Error: The input edges are parallel. An intersection vertex cannot be found. Returning None.")
            return None
        
        eq1 = Edge.Equation2D(edgeA, mantissa=mantissa)
        eq2 = Edge.Equation2D(edgeB, mantissa=mantissa)
        if eq1["slope"] == float('inf'):
            x = eq1["x_intercept"]
            y = eq2["slope"] * x + eq2["y_intercept"]
        elif eq2["slope"] == float('inf'):
            x = eq2["x_intercept"]
            y = eq1["slope"] * x + eq1["y_intercept"]
        else:
            x = (eq2["y_intercept"] - eq1["y_intercept"]) / (eq1["slope"] - eq2["slope"])
            y = eq1["slope"] * x + eq1["y_intercept"]
        
        return Vertex.ByCoordinates(x,y,0)

    @staticmethod
    def IsCollinear(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001) -> bool:
        """
        Return True if the two input edges are collinear. Returns False otherwise.
        This code is based on a contribution by https://github.com/gaoxipeng

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the two edges are collinear. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import numpy as np

        if not Topology.IsInstance(edgeA, "Edge"):
            print("Edge.IsCollinear - Error: The input parameter edgeA is not a valid edge. Returning None")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            print("Edge.IsCollinear - Error: The input parameter edgeB is not a valid edge. Returning None")
            return None
        if Edge.Length(edgeA) <= tolerance:
            print("Edge.IsCollinear - Error: The length of edgeA is less than or equal the tolerance. Returning None")
            return None
        if Edge.Length(edgeB) <= tolerance:
            print("Edge.IsCollinear - Error: The length of edgeB is less than or equal to the tolerance. Returning None")
            return None
        
        # Get start and end points of the first edge
        start_a = Edge.StartVertex(edgeA)
        end_a = Edge.EndVertex(edgeA)
        start_a_coords = np.array([Vertex.X(start_a, mantissa=mantissa), Vertex.Y(start_a, mantissa=mantissa), Vertex.Z(start_a, mantissa=mantissa)])
        end_a_coords = np.array(
            [Vertex.X(end_a, mantissa=mantissa), Vertex.Y(end_a, mantissa=mantissa), Vertex.Z(end_a, mantissa=mantissa)])

        # Calculate the direction vector of the first edge
        direction_a = end_a_coords - start_a_coords

        # Normalize the direction vector
        norm_a = np.linalg.norm(direction_a)
        if norm_a == 0:
            print("Edge.IsCollinear - Error: Division by zero. Returning None.")
            return None
        direction_a /= norm_a

        # Function to calculate perpendicular distance from a point to the line defined by a point and direction vector
        def distance_from_line(point, line_point, line_dir):
            point = np.array([Vertex.X(point, mantissa=mantissa), Vertex.Y(point, mantissa=mantissa),
                            Vertex.Z(point, mantissa=mantissa)])
            line_point = np.array(line_point)
            diff = point - line_point
            cross_product = np.cross(diff, line_dir)
            line_dir_norm = np.linalg.norm(line_dir)
            if line_dir_norm == 0:
                print("Edge.IsCollinear - Error: Division by zero. Returning None.")
                return None
            distance = np.linalg.norm(cross_product) / np.linalg.norm(line_dir)
            return distance

        # Get start and end points of the second edge
        start_b = Edge.StartVertex(edgeB)
        end_b = Edge.EndVertex(edgeB)

        # Calculate distances for start and end vertices of the second edge to the line defined by the first edge
        distance_start = distance_from_line(start_b, start_a_coords, direction_a)
        distance_end = distance_from_line(end_b, start_a_coords, direction_a)

        # Check if both distances are within tolerance
        return bool(distance_start <= tolerance) and bool(distance_end <= tolerance)
    
    @staticmethod
    def IsCoplanar(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Return True if the two input edges are coplanar. Returns False otherwise.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the two edges are coplanar. False otherwise.

        """
        import numpy as np
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            print("Edge.IsCoplanar - Error: The input parameter edgeA is not a valid edge. Returning None")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            print("Edge.IsCoplanar - Error: The input parameter edgeB is not a valid edge. Returning None")
            return None
        if Edge.Length(edgeA) <= tolerance:
            print("Edge.IsCoplanar - Error: The length of edgeA is less than or equal to the tolerance. Returning None")
            return None
        if Edge.Length(edgeB) <= tolerance:
            print("Edge.IsCoplanar - Error: The length of edgeB is less than or equal to the tolerance. Returning None")
            return None
        
        # Extract points
        sva, eva = [Topology.Vertices(edgeA)[0], Topology.Vertices(edgeA)[-1]]
        p1 = Vertex.Coordinates(sva, mantissa=mantissa)
        p2 = Vertex.Coordinates(eva, mantissa=mantissa)
        svb, evb = [Topology.Vertices(edgeB)[0], Topology.Vertices(edgeB)[-1]]
        p3 = Vertex.Coordinates(svb, mantissa=mantissa)
        p4 = Vertex.Coordinates(evb, mantissa=mantissa)

        # Create vectors
        v1 = np.subtract(p2, p1)
        v2 = np.subtract(p4, p3)
        v3 = np.subtract(p3, p1)
        
        # Calculate the scalar triple product
        scalar_triple_product = np.dot(np.cross(v1, v2), v3)
        
        # Check for coplanarity
        return np.isclose(scalar_triple_product, 0, atol=tolerance)

    @staticmethod
    def IsParallel(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001) -> bool:
        """
        Return True if the two input edges are parallel. Returns False otherwise.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the two edges are collinear. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import numpy as np

        if not Topology.IsInstance(edgeA, "Edge"):
            print("Edge.IsParallel - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            print("Edge.IsParallel - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        
        def are_lines_parallel(line1, line2, tolerance=0.0001):
            """
            Determines if two lines in 3D space are parallel.
            
            Parameters:
            line1 (tuple): A tuple of two points defining the first line. Each point is a tuple of (x, y, z).
            line2 (tuple): A tuple of two points defining the second line. Each point is a tuple of (x, y, z).
            
            Returns:
            bool: True if the lines are parallel, False otherwise.
            """
            def vector_from_points(p1, p2):
                return np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
            
            # Get direction vectors for both lines
            vec1 = vector_from_points(line1[0], line1[1])
            vec2 = vector_from_points(line2[0], line2[1])
            
            # Compute the cross product of the direction vectors
            cross_product = np.cross(vec1, vec2)
            
            # Two vectors are parallel if their cross product is a zero vector
            return np.allclose(cross_product, 0, atol=tolerance)

        x1, y1, z1 = Vertex.Coordinates(Edge.StartVertex(edgeA), mantissa=mantissa)
        x2, y2, z2 = Vertex.Coordinates(Edge.EndVertex(edgeA), mantissa=mantissa)
        x3, y3, z3 = Vertex.Coordinates(Edge.StartVertex(edgeB), mantissa=mantissa)
        x4, y4, z4 = Vertex.Coordinates(Edge.EndVertex(edgeB), mantissa=mantissa)
        line1 = ((x1, y1, z1), (x2, y2, z2))
        line2 = ((x3, y3, z3), (x4, y4, z4))
        return are_lines_parallel(line1, line2, tolerance=tolerance)

    @staticmethod
    def Length(edge, mantissa: int = 6) -> float:
        """
        Returns the length of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The length of the input edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Length - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        length = None
        try:
            length = round(topologic.EdgeUtility.Length(edge), mantissa)  # Hook to Core
        except:
            length = None
        if length == None:
            print("Edge.Length - Error: Could not compute the length of the input edge parameter. Returning None.")
        return length

    @staticmethod
    def Line(origin= None, length: float = 1, direction: list = [1,0,0], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates a straight edge (line) using the input parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the box. The default is None which results in the edge being placed at (0, 0, 0).
        length : float , optional
            The desired length of the edge. The default is 1.0.
        direction : list , optional
            The desired direction (vector) of the edge. The default is [1,0,0] (along the X-axis).
        placement : str , optional
            The desired placement of the edge. The options are:
            1. "center" which places the center of the edge at the origin.
            2. "start" which places the start of the edge at the origin.
            3. "end" which places the end of the edge at the origin.
            The default is "center". It is case insensitive.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        Returns
        -------
        topologic_core.Edge
            The created edge
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            print("Edge.Line - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if length <= 0:
            print("Edge.Line - Error: The input length is less than or equal to zero. Returning None.")
            return None
        if not isinstance(direction, list):
            print("Edge.Line - Error: The input direction parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            print("Edge.Line - Error: The length of the input direction parameter is not equal to three. Returning None.")
            return None
        direction = Vector.Normalize(direction)
        if "center" in placement.lower():
            sv = Topology.TranslateByDirectionDistance(origin, direction=Vector.Reverse(direction), distance=length*0.5)
            ev = Topology.TranslateByDirectionDistance(sv, direction=direction, distance=length)
            return Edge.ByVertices([sv,ev], tolerance=tolerance, silent=True)
        if "start" in placement.lower():
            sv = origin
            ev = Topology.TranslateByDirectionDistance(sv, direction=direction, distance=length)
            return Edge.ByVertices([sv,ev], tolerance=tolerance, silent=True)
        if "end" in placement.lower():
            sv = Topology.TranslateByDirectionDistance(origin, direction=Vector.Reverse(direction), distance=length)
            ev = Topology.TranslateByDirectionDistance(sv, direction=direction, distance=length)
            return Edge.ByVertices([sv,ev], tolerance=tolerance, silent=True)
        else:
            print("Edge.Line - Error: The input placement string is not one of center, start, or end. Returning None.")
            return None
    
    @staticmethod
    def Normal(edge, angle: float = 0.0):
        """
        Returns the normal (perpendicular) vector to the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        angle : float , optional
            The desired rotational offset angle in degrees for the normal edge. This rotates the normal edge
            by the angle value around the axis defined by the input edge. The default is 0.0.

        Returns
        -------
        list
            The normal (perpendicular ) vector to the input edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Normal - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        normal_edge = Edge.NormalEdge(edge, length=1.0, u=0.5, angle=angle)
        return Edge.Direction(normal_edge)

    @staticmethod
    def NormalEdge(edge, length: float = 1.0, u: float = 0.5, angle: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the normal (perpendicular) vector to the input edge as an edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        length : float , optional
            The desired length of the normal edge. The default is 1.0.
        u : float , optional
            The desired u parameter placement of the normal edge. A value of 0.0 places the normal edge
            at the start vertex of the input edge, a value of 0.5 places the normal edge
            at the midpoint of the input edge, and a value of 1.0 places the normal edge
            at the end vertex of the input edge. The default is 0.5
        angle : float , optional
            The desired rotational offset angle in degrees for the normal edge. This rotates the normal edge
            by the angle value around the axis defined by the input edge. The default is 0.0.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        topologic_core.Edge
            The normal (perpendicular) vector to the input edge as an edge.

        """
        import numpy as np
        from numpy.linalg import norm
        import topologic_core as topologic
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology



        def calculate_normal(start_vertex, end_vertex):
            start_vertex = np.array([float(x) for x in start_vertex])
            end_vertex = np.array([float(x) for x in end_vertex])

            # Calculate the direction vector of the edge
            direction_vector = end_vertex - start_vertex
            
            # Check if the edge is vertical (only Z component)
            if np.isclose(direction_vector[0], 0) and np.isclose(direction_vector[1], 0):
                # Choose an arbitrary perpendicular vector in the X-Y plane, e.g., [1, 0, 0]
                normal_vector = np.array([1.0, 0.0, 0.0])
            elif np.isclose(direction_vector[2], 0):
                # The edge lies in the X-Y plane; compute a perpendicular in the X-Y plane
                normal_vector = np.array([-direction_vector[1], direction_vector[0], 0.0])
            else:
                # Otherwise, calculate the normal by crossing with the Z-axis
                z_axis = np.array([0, 0, 1])
                normal_vector = np.cross(direction_vector, z_axis)

            # Check if the normal vector is effectively zero before normalization
            if np.isclose(norm(normal_vector), 0):
                return normal_vector
    
            # Normalize the normal vector
            normal_vector /= norm(normal_vector)  
            return normal_vector

        def calculate_normal_line(start_vertex, end_vertex):
            # Calculate the normal vector of the line
            normal_vector = calculate_normal(start_vertex, end_vertex)

            # Calculate the new end vertex for the normal line to have a length of 1
            normal_end_vertex = np.array(start_vertex) + normal_vector

            # Return the start and end vertices of the normal line
            return start_vertex, normal_end_vertex

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.NormalEdge - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        if length <= 0.0:
            if not silent:
                print("Edge.NormalEdge - Error: The input length parameter is not a positive number greater than zero. Returning None.")
            return None

        # Get start and end vertex coordinates
        start_vertex = Vertex.Coordinates(Edge.StartVertex(edge))
        end_vertex = Vertex.Coordinates(Edge.EndVertex(edge))

        # Calculate the normal line
        normal_line_start, normal_line_end = calculate_normal_line(start_vertex, end_vertex)
        # Create the normal edge in Topologic
        sv = Vertex.ByCoordinates(list(normal_line_start))
        ev = Vertex.ByCoordinates(list(normal_line_end))
        
        # Create an edge from the start to the end of the normal vector
        normal_edge = Edge.ByVertices([sv, ev], tolerance=tolerance, silent=silent)
        if normal_edge == None:
            if not silent:
                print("Edge.NormalEdge - Error: Could not create edge. Returning None.")
            return None
        # Set the length of the normal edge
        normal_edge = Edge.SetLength(normal_edge, length, bothSides=False, tolerance=tolerance)

        # Rotate the normal edge around the input edge by the specified angle
        edge_direction = Edge.Direction(edge)
        x, y, z = edge_direction
        normal_edge = Topology.Rotate(normal_edge, origin=Edge.StartVertex(normal_edge), axis=[x, y, z], angle=angle)

        # Translate the normal edge along the edge direction according to the u parameter
        dist = Edge.Length(edge) * u
        normal_edge = Topology.TranslateByDirectionDistance(normal_edge, edge_direction, dist)

        return normal_edge

    @staticmethod
    def Normalize(edge, useEndVertex: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a normalized edge that has the same direction as the input edge, but a length of 1.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        useEndVertex : bool , optional
            If True the normalized edge end vertex will be placed at the end vertex of the input edge. Otherwise, the normalized edge start vertex will be placed at the start vertex of the input edge. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        topologic_core.Edge
            The normalized edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Normalize - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        if not useEndVertex:
            sv = Edge.StartVertex(edge)
            ev = Edge.VertexByDistance(edge, 1.0, Edge.StartVertex(edge))
        else:
            sv = Edge.VertexByDistance(edge, 1.0, Edge.StartVertex(edge))
            ev = Edge.EndVertex(edge)
        return Edge.ByVertices([sv, ev], tolerance=tolerance, silent=silent)

    @staticmethod
    def ParameterAtVertex(edge, vertex, mantissa: int = 6, silent: bool = False) -> float:
        """
        Returns the *u* parameter along the input edge based on the location of the input vertex.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        vertex : topologic_core.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        float
            The *u* parameter along the input edge based on the location of the input vertex.

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
        parameter = None
        try:
            parameter = topologic.EdgeUtility.ParameterAtPoint(edge, vertex)  # Hook to Core
        except:
            return None #Return silently because topologic C++ returns a runtime error if point is not on curve.
        return round(parameter, mantissa)

    @staticmethod
    def Reverse(edge, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates an edge that has the reverse direction of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

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
        return Edge.ByVertices([Edge.EndVertex(edge), Edge.StartVertex(edge)], tolerance=tolerance, silent=silent)
    
    @staticmethod
    def SetLength(edge , length: float = 1.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001):
        """
        Returns an edge with the new length in the same direction as the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        length : float , optional
            The desired length of the edge. The default is 1.
        bothSides : bool , optional
            If set to True, the edge will be offset symmetrically from each end. The default is True.
        reverse : bool , optional
            If set to True, the edge will be offset from its start vertex. Otherwise, it will be offset from its end vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Edge
            The extended edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.SetLength - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        distance = (length - Edge.Length(edge))
        if distance > 0:
            return Edge.Extend(edge=edge, distance=distance, bothSides=bothSides, reverse=reverse, tolerance=tolerance)
        return Edge.Trim(edge=edge, distance=distance, bothSides=bothSides, reverse=reverse, tolerance=tolerance)

    @staticmethod
    def StartVertex(edge):
        """
        Returns the start vertex of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.

        Returns
        -------
        topologic_core.Vertex
            The start vertex of the input edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.StartVertex - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        vert = None
        try:
            vert = edge.StartVertex() # Hook to core
        except:
            vert = None
        return vert

    @staticmethod
    def Trim(edge, distance: float = 0.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Trims the input edge by the input distance.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The offset distance. The default is 0.
        bothSides : bool , optional
            If set to True, the edge will be trimmed by half the distance at each end. The default is False.
        reverse : bool , optional
            If set to True, the edge will be trimmed from its start vertex. Otherwise, it will be trimmed from its end vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Edge
            The trimmed edge.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Trim - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        distance = abs(distance)
        if distance == 0:
            return edge
        if distance <= tolerance:
            print("Edge.Trim - Warning: The input distance parameter is less than or equal to the input tolerance parameter. Returning the input edge.")
            return edge
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        if bothSides:
            sve = Edge.VertexByDistance(edge, distance=distance*0.5, origin=sv, tolerance=tolerance)
            eve = Edge.VertexByDistance(edge, distance=-distance*0.5, origin=ev, tolerance=tolerance)
        elif reverse:
            sve = Edge.VertexByDistance(edge, distance=distance, origin=sv, tolerance=tolerance)
            eve = Edge.EndVertex(edge)
        else:
            sve = Edge.StartVertex(edge)
            eve = Edge.VertexByDistance(edge, distance=-distance, origin=ev, tolerance=tolerance)
        return Edge.ByVertices([sve, eve], tolerance=tolerance, silent=silent)

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
            If set to True, which segment is preserved is reversed. Otherwise, it is not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        topologic_core.Edge
            The trimmed edge.

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
        if not Edge.IsCoplanar(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edges are not coplanar. Returning the original edge.")
            return edgeA
        if Edge.IsParallel(edgeA, edgeB, tolerance=tolerance):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edges are parallel. Returning the original edge.")
            return edgeA
        
        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        svb = Edge.StartVertex(edgeB)
        evb = Edge.EndVertex(edgeB)
        intVertex = None
        if Edge.IsCollinear(edgeA, edgeB, tolerance=tolerance):
            if Vertex.IsInternal(svb, edgeA):
                intVertex = svb
            elif Vertex.IsInternal(evb, edgeA):
                intVertex = evb
            else:
                intVertex = None
            if intVertex:
                if reverse:
                        return Edge.ByVertices([eva, intVertex], tolerance=tolerance, silent=silent)
                else:
                    return Edge.ByVertices([sva, intVertex], tolerance=tolerance, silent=silent)
            else:
                return None
        
        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        intVertex = Topology.Intersect(edgeA, edgeB)
        if intVertex and (Vertex.IsInternal(intVertex, edgeA)):
            if reverse:
                return Edge.ByVertices([eva, intVertex], tolerance=tolerance, silent=slient)
            else:
                return Edge.ByVertices([sva, intVertex], tolerance=tolerance, silent=silent)
        return edgeA

    @staticmethod
    def VertexByDistance(edge, distance: float = 0.0, origin= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a vertex along the input edge offset by the input distance from the input origin.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The offset distance. The default is 0.
        origin : topologic_core.Vertex , optional
            The origin of the offset distance. If set to None, the origin will be set to the start vertex of the input edge. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.TrimByEdge2D - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Edge.StartVertex(edge)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Edge.TrimByEdge2D - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        vx = Vertex.X(ev, mantissa=mantissa) - Vertex.X(sv, mantissa=mantissa)
        vy = Vertex.Y(ev, mantissa=mantissa) - Vertex.Y(sv, mantissa=mantissa)
        vz = Vertex.Z(ev, mantissa=mantissa) - Vertex.Z(sv, mantissa=mantissa)
        vector = Vector.Normalize([vx, vy, vz])
        vector = Vector.Multiply(vector, distance, tolerance=tolerance)
        return Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+vector[0], Vertex.Y(origin, mantissa=mantissa)+vector[1], Vertex.Z(origin, mantissa=mantissa)+vector[2])
    
    @staticmethod
    def VertexByParameter(edge, u: float = 0.0):
        """
        Creates a vertex along the input edge offset by the input *u* parameter.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        u : float , optional
            The *u* parameter along the input topologic Edge. A parameter of 0 returns the start vertex. A parameter of 1 returns the end vertex. The default is 0.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.VertexByParameter - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        vertex = None
        if u == 0:
            vertex = Edge.StartVertex(edge)
        elif u == 1:
            vertex = Edge.EndVertex(edge)
        else:
            dir = Edge.Direction(edge)
            edge_length = Edge.Length(edge)
            dist = edge_length*u
            vertex = Topology.TranslateByDirectionDistance(Edge.StartVertex(edge), direction=dir, distance=dist)
        return vertex

    @staticmethod
    def Vertices(edge) -> list:
        """
        Returns the list of vertices of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Vertices - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        vertices = []
        _ = edge.Vertices(None, vertices) # Hook to Core
        return vertices