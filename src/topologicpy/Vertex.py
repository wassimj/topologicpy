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
import collections
import os
import warnings

try:
    import numpy as np
except:
    print("Vertex - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("Vertex - numpy library installed successfully.")
    except:
        warnings.warn("Vertex - Error: Could not import numpy.")

class Vertex():
    @staticmethod
    def AlignCoordinates(vertex, xList: list = None, yList: list = None, zList: list = None, xEpsilon: float = 0.0001, yEpsilon: float = 0.0001, zEpsilon: float = 0.0001, transferDictionary: bool = False, mantissa: int = 6, silent: bool = False):
        """
            Aligns the coordinates of the input vertex with the list of x,y, and z coordinates.

            Parameters
            ----------
            vertex : topologic_core.Vertex
                The input vertex.
            xList : list , optional
                The input numerical list of x-coordinates. The default is None.
            yList : list , optional
                The input numerical list of y-coordinates. The default is None.
            zList : list , optional
                The input numerical list of z-coordinates. The default is None.
            xEpsilon : float , optional
                The desired tolerance for the x coordinates. The default is 0.0001.
            yEpsilon : float , optional
                The desired tolerance for the y coordinates. The default is 0.0001. 
            zEpsilon : float , optional
                The desired tolerance for the z coordinates. The default is 0.0001. 
            transferDictionary : bool , optional
                if set to True, the dictionary of the input vertex is transferred to the new vertex.
            mantissa : int , optional
                The desired length of the mantissa. The default is 6.
            silent : bool , optional
                If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
            
            Returns
            -------
            topologic_core.Vertex
                The created vertex aligned to the input list of x,y, and z coordinates.
        
        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(vertex, "vertex"):
            if not silent:
                print("Vertex.AlignCoordinates - Error: The input vertex parameter is not a topologic vertex. Returning None.")
            return None
        
        x, y, z = Vertex.Coordinates(vertex, mantissa=mantissa)
        if isinstance(xList, list):
            if len(xList) > 0:
                closest_x = round(xList[Helper.ClosestMatch(x, xList)], mantissa)
        if isinstance(yList, list):
            if len(yList) > 0:
                closest_y = round(yList[Helper.ClosestMatch(y, yList)], mantissa)
        if isinstance(zList, list):
            if len(zList) > 0:
                closest_z = round(zList[Helper.ClosestMatch(z, zList)], mantissa)
        
        if abs(x - closest_x) < xEpsilon:
            x = closest_x
        if abs(y - closest_y) < yEpsilon:
            y = closest_y
        if abs(z - closest_z) < zEpsilon:
            z = closest_z
        return_vertex = Vertex.ByCoordinates(x, y, z)
        if transferDictionary == True:
            return_vertex = Topology.SetDictionary(return_vertex, Topology.Dictionary(vertex), silent=silent)
        return return_vertex

    @staticmethod
    def AreCollinear(vertices: list, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns True if the input list of vertices form a straight line. Returns False otherwise.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        
        def areCollinear(vertices, tolerance=0.0001):
            point1 = Vertex.Coordinates(vertices[0], mantissa=mantissa)
            point2 = Vertex.Coordinates(vertices[1], mantissa=mantissa)
            point3 = Vertex.Coordinates(vertices[2], mantissa=mantissa)

            vector1 = [point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]]
            vector2 = [point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2]]

            cross_product_result = Vector.Cross(vector1, vector2, tolerance=tolerance)
            return cross_product_result == None
        
        if not isinstance(vertices, list):
            print("Vertex.AreCollinear - Error: The input list of vertices is not a valid list. Returning None.")
            return None
        vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 2:
            print("Vertex.AreCollinear - Error: The input list of vertices does not contain sufficient valid vertices. Returning None.")
            return None
        if len(vertexList) < 3:
            return True # Any two vertices can form a line!
        cluster = Topology.SelfMerge(Cluster.ByTopologies(vertexList), tolerance=tolerance)
        vertexList = Topology.Vertices(cluster)
        slices = []
        for i in range(2,len(vertexList)):
            slices.append([vertexList[0], vertexList[1], vertexList[i]])
        for slice in slices:
            if not areCollinear(slice, tolerance=tolerance):
                return False
        return True
    
    @staticmethod
    def AreCoplanar(vertices: list, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input list of vertices are coplanar. Returns False otherwise.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001
        silent : bool , optional
            If set to True, no warnings or errors are printed. The default is False.

        Returns
        -------
        bool
            True if the input vertices are coplanar.

        """
        from topologicpy.Topology import Topology
        from numpy.linalg import svd

        if not isinstance(vertices, list):
            if not silent:
                print("Vertex.Normal - Error: The vertices input parameter is not a valid list. Returning None.")
            return None
        
        verts = [v for v in vertices if Topology.IsInstance(v, "vertex")]

        if len(verts) < 3:
            if not silent:
                print("Vertex.AreCoplanar - Error: The list of vertices contains less than 3 valid topologic vertices. Returning None.")
            return None  # At least 3 vertices are needed
        
        coords = np.array([Vertex.Coordinates(v, mantissa=mantissa) for v in vertices])

        # Check if points are coplanar using SVD
        _, s, vh = svd(coords - coords.mean(axis=0))
        rank = (s > tolerance).sum()
        return (rank <= 2)
    
    @staticmethod
    def AreIpsilateral(vertices: list, face) -> bool:
        """
        Returns True if the input list of vertices are on one side of a face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic_core.Face
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        def check(dot_productA, pointB, pointC, normal):
            # Calculate the dot products of the vectors from the surface point to each of the input points.
            dot_productB = (pointB[0] - pointC[0]) * normal[0] + \
                        (pointB[1] - pointC[1]) * normal[1] + \
                        (pointB[2] - pointC[2]) * normal[2]

            # Check if both points are on the same side of the surface.
            if dot_productA * dot_productB > 0:
                return True

            # Check if both points are on opposite sides of the surface.
            elif dot_productA * dot_productB < 0:
                return False

            # Otherwise, at least one point is on the surface.
            else:
                return True
    
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 2:
            return None
        pointA = Vertex.Coordinates(vertexList[0])
        pointC = Vertex.Coordinates(Face.VertexByParameters(face, 0.5, 0.5))
        normal = Face.Normal(face)
        dot_productA = (pointA[0] - pointC[0]) * normal[0] + \
                        (pointA[1] - pointC[1]) * normal[1] + \
                        (pointA[2] - pointC[2]) * normal[2]
        for i in range(1, len(vertexList)):
            pointB = Vertex.Coordinates(vertexList[i])
            if not check(dot_productA, pointB, pointC, normal):
                return False
        return True
    
    @staticmethod
    def AreIpsilateralCluster(cluster, face) -> bool:
        """
        Returns True if the two input vertices are on the same side of the input face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input list of vertices.
        face : topologic_core.Face
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Topology"):
            return None
        vertices = Topology.SubTopologies(cluster, subTopologyType="vertex")
        return Vertex.AreIpsilateral(vertices, face)
    
    @staticmethod
    def AreOnSameSide(vertices: list, face) -> bool:
        """
        Returns True if the two input vertices are on the same side of the input face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic_core.Face
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        return Vertex.AreIpsilateral(vertices, face)

    @staticmethod
    def AreOnSameSideCluster(cluster, face) -> bool:
        """
        Returns True if the two input vertices are on the same side of the input face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input list of vertices.
        face : topologic_core.Face
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Topology"):
            return None
        vertices = Topology.SubTopologies(cluster, subTopologyType="vertex")
        return Vertex.AreIpsilateral(vertices, face)

    @staticmethod
    def ByCoordinates(*args, **kwargs):
        """
        Creates a vertex at the coordinates specified by the x, y, z inputs. You can call this method using a list of coordinates or individually.
        Examples:
        v = Vertex.ByCoordinates(3.4, 5.7, 2.8)
        v = Vertex.ByCoordinates([3.4, 5.7, 2.8])
        v = Vertex.ByCoordinates(x=3.4, y=5.7, z=2.8)

        Parameters
        ----------
        x : float , optional
            The X coordinate. The default is 0.
        y : float , optional
            The Y coordinate. The default is 0.
        z : float , optional
            The Z coordinate. The defaults is 0.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        import numbers

        x = None
        y = None
        z = None
        if len(args) > 3 or len(kwargs.items()) > 3:
            print("Vertex.ByCoordinates - Error: Input parameters are greater than 3. Returning None.")
            return None
        if len(args) > 0:
            value = args[0]
            if (isinstance(value, list) or isinstance(value, tuple)) and len(value) > 3:
                print("Vertex.ByCoordinates - Error: Input parameters are greater than 3. Returning None.")
                return None
            elif (isinstance(value, list) or isinstance(value, tuple)) and len(value) == 3:
                x = value[0]
                y = value[1]
                z = value[2]
            elif (isinstance(value, list) or isinstance(value, tuple)) and len(value) == 2:
                x = value[0]
                y = value[1]
            elif (isinstance(value, list) or isinstance(value, tuple)) and len(value) == 1:
                x = value[0]
            elif len(args) == 3:
                x = args[0]
                y = args[1]
                z = args[2]
            elif len(args) == 2:
                x = args[0]
                y = args[1]
            elif len(args) == 1:
                x = args[0]
        for key, value in kwargs.items():
            if "x" in key.lower():
                if not x == None:
                    print("Vertex.ByCoordinates - Error: Input parameters are not formed properly. Returning None.")
                    return None
                x = value
            elif "y" in key.lower():
                if not y == None:
                    print("Vertex.ByCoordinates - Error: Input parameters are not formed properly. Returning None.")
                    return None
                y = value
            elif "z" in key.lower():
                if not z == None:
                    print("Vertex.ByCoordinates - Error: Input parameters are not formed properly. Returning None.")
                    return None
                z = value
        if x == None:
            x = 0
        if y == None:
            y = 0
        if z == None:
            z = 0
        if not isinstance(x, numbers.Number):
            print("Vertex.ByCoordinates - Error: The x value is not a valid number. Returning None.")
            return None
        if not isinstance(y, numbers.Number):
            print("Vertex.ByCoordinates - Error: The y value is not a valid number. Returning None.")
            return None
        if not isinstance(z, numbers.Number):
            print("Vertex.ByCoordinates - Error: The z value is not a valid number. Returning None.")
            return None
        
        vertex = None
        try:
            vertex = topologic.Vertex.ByCoordinates(x, y, z) # Hook to Core
        except:
            vertex = None
            print("Vertex.ByCoordinates - Error: Could not create a topologic vertex. Returning None.")
        return vertex
    
    @staticmethod
    def Centroid(vertices: list, mantissa: int = 6):
        """
        Returns the centroid of the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        topologic_core.Vertex
            The computed centroid of the input list of vertices
        
        """
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            print("Vertex.Centroid - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Vertex.Centroid - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
            return None
        if len(vertices) == 1:
            return vertices[0]
        cx = sum(Vertex.X(v, mantissa=mantissa) for v in vertices) / len(vertices)
        cy = sum(Vertex.Y(v, mantissa=mantissa) for v in vertices) / len(vertices)
        cz = sum(Vertex.Z(v, mantissa=mantissa) for v in vertices) / len(vertices)
        return Vertex.ByCoordinates(cx, cy, cz)
    
    @staticmethod
    def Clockwise2D(vertices):
        """
        Sorts the input list of vertices in a clockwise fashion. This method assumes that the vertices are on the XY plane. The Z coordinate is ignored.

        Parameters
        ----------
        vertices : list
            The input list of vertices

        Returns
        -------
        list
            The input list of vertices sorted in a counter clockwise fashion
        
        """
        
        return list(reversed(Vertex.CounterClockwise2D(vertices)))
    
    @staticmethod
    def Coordinates(vertex, outputType: str = "xyz", mantissa: int = 6) -> list:
        """
        Returns the coordinates of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        outputType : string, optional
            The desired output type. Could be any permutation or substring of "xyz" or the string "matrix". The default is "xyz". The input is case insensitive and the coordinates will be returned in the specified order.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        list
            The coordinates of the input vertex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        x = Vertex.X(vertex, mantissa)
        y = Vertex.Y(vertex, mantissa)
        z = Vertex.Z(vertex, mantissa)
        matrix = [[1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]]
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
    def CounterClockwise2D(vertices: list, mantissa: int = 6):
        """
        Sorts the input list of vertices in a counterclockwise fashion. This method assumes that the vertices are on the XY plane. The Z coordinate is ignored.

        Parameters
        ----------
        vertices : list
            The input list of vertices
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        list
            The input list of vertices sorted in a counter clockwise fashion
        
        """
        from topologicpy.Topology import Topology
        import math

        if not isinstance(vertices, list):
            print("Vertex.CounterClockwise2D - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 1:
            print("Vertex.CounterClockwise2D - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
            return None
        if len(vertices) == 1:
            return vertices[0]
        
        # find the centroid of the points
        cx = sum(Vertex.X(v, mantissa=mantissa) for v in vertices) / len(vertices)
        cy = sum(Vertex.Y(v, mantissa=mantissa) for v in vertices) / len(vertices)

        # sort the points based on their angle with respect to the centroid
        vertices.sort(key=lambda v: (math.atan2(Vertex.Y(v) - cy, Vertex.X(v) - cx) + 2 * math.pi) % (2 * math.pi))
        return vertices

    @staticmethod
    def Degree(vertex, hostTopology, topologyType: str = "edge"):
        """
        Returns the vertex degree (the number of super topologies connected to it). See https://en.wikipedia.org/wiki/Degree_(graph_theory).

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        hostTopology : topologic_core.Topology
            The input host topology in which to search for the connected super topologies.
        topologyType : str , optional
            The topology type to search for. This can be any of "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. If set to None, the immediate supertopology type is searched for. The default is None.

        Returns
        -------
        int
            The number of super topologies connected to this vertex

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            print("Vertex.Degree - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
        if not Topology.IsInstance(hostTopology, "Topology"):
            print("Vertex.Degree - Error: The input hostTopology parameter is not a valid topologic topology. Returning None.")
        superTopologies = Topology.SuperTopologies(topology=vertex, hostTopology=hostTopology, topologyType=topologyType)
        return len(superTopologies)

    @staticmethod
    def Distance(vertex, topology, includeCentroid: bool =True, mantissa: int = 6) -> float:
        """
        Returns the distance between the input vertex and the input topology. This method returns the distance to the closest sub-topology in the input topology, optionally including its centroid.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        includeCentroid : bool
            If set to True, the centroid of the input topology will be considered in finding the nearest subTopology to the input vertex. The default is True.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        float
            The distance between the input vertex and the input topology.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        def distance_point_to_point(point1, point2):
            # Convert input points to NumPy arrays
            point1 = np.array(point1)
            point2 = np.array(point2)
            
            # Calculate the Euclidean distance
            distance = np.linalg.norm(point1 - point2)
            
            return float(distance)

        def distance_point_to_line(point, line_start, line_end):
            # Convert input points to NumPy arrays for vector operations
            point = np.array(point)
            line_start = np.array(line_start)
            line_end = np.array(line_end)
            
            # Calculate the direction vector of the edge
            line_direction = line_end - line_start
            
            # Vector from the edge's starting point to the point
            point_to_start = point - line_start
            
            # Calculate the parameter 't' where the projection of the point onto the edge occurs
            if np.dot(line_direction, line_direction) == 0:
                t = 0
            else:
                t = np.dot(point_to_start, line_direction) / np.dot(line_direction, line_direction)
            
            # Check if 't' is outside the range [0, 1], and if so, calculate distance to closest endpoint
            if t < 0:
                return np.linalg.norm(point - line_start)
            elif t > 1:
                return np.linalg.norm(point - line_end)
            
            # Calculate the closest point on the edge to the given point
            closest_point = line_start + t * line_direction
            
            # Calculate the distance between the closest point and the given point
            distance = np.linalg.norm(point - closest_point)
            
            return distance
        
        def distance_to_vertex(vertexA, vertexB):
            a = (Vertex.X(vertexA, mantissa=mantissa), Vertex.Y(vertexA, mantissa=mantissa), Vertex.Z(vertexA, mantissa=mantissa))
            b = (Vertex.X(vertexB, mantissa=mantissa), Vertex.Y(vertexB, mantissa=mantissa), Vertex.Z(vertexB, mantissa=mantissa))
            return distance_point_to_point(a, b)
        
        def distance_to_edge(vertex, edge):
            a = (Vertex.X(vertex, mantissa=mantissa), Vertex.Y(vertex, mantissa=mantissa), Vertex.Z(vertex, mantissa=mantissa))
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            svp = (Vertex.X(sv, mantissa=mantissa), Vertex.Y(sv, mantissa=mantissa), Vertex.Z(sv, mantissa=mantissa))
            evp = (Vertex.X(ev, mantissa=mantissa), Vertex.Y(ev, mantissa=mantissa), Vertex.Z(ev, mantissa=mantissa))
            return distance_point_to_line(a,svp, evp)
        
        def distance_to_face(vertex, face, includeCentroid):
            v_proj = Vertex.Project(vertex, face, mantissa=mantissa)
            if not Vertex.IsInternal(v_proj, face):
                vertices = Topology.Vertices(topology)
                distances = [distance_to_vertex(vertex, v) for v in vertices]
                edges = Topology.Edges(topology)
                distances += [distance_to_edge(vertex, e) for e in edges]
                if includeCentroid:
                    distances.append(distance_to_vertex(vertex, Topology.Centroid(topology)))
                return min(distances)
            dic = Face.PlaneEquation(face)
            a = dic["a"]
            b = dic["b"]
            c = dic["c"]
            d = dic["d"]
            x1, y1, z1 = Vertex.Coordinates(vertex, mantissa=mantissa)
            d = abs((a * x1 + b * y1 + c * z1 + d))
            e = (math.sqrt(a * a + b * b + c * c))
            if e == 0:
                return 0
            return d/e
        if not Topology.IsInstance(vertex, "Vertex") or not Topology.IsInstance(topology, "Topology"):
            return None
        if Topology.IsInstance(topology, "Vertex"):
            return round(distance_to_vertex(vertex,topology), mantissa)
        elif Topology.IsInstance(topology, "Edge"):
            return round(distance_to_edge(vertex,topology), mantissa)
        elif Topology.IsInstance(topology, "Wire"):
            vertices = Topology.Vertices(topology)
            distances = [distance_to_vertex(vertex, v) for v in vertices]
            edges = Topology.Edges(topology)
            distances += [distance_to_edge(vertex, e) for e in edges]
            if includeCentroid:
                distances.append(distance_to_vertex(vertex, Topology.Centroid(topology)))
            return round(min(distances), mantissa)
        elif Topology.IsInstance(topology, "Face"):
            vertices = Topology.Vertices(topology)
            distances = [distance_to_vertex(vertex, v) for v in vertices]
            edges = Topology.Edges(topology)
            distances += [distance_to_edge(vertex, e) for e in edges]
            distances.append(distance_to_face(vertex,topology, includeCentroid))
            if includeCentroid:
                distances.append(distance_to_vertex(vertex, Topology.Centroid(topology)))
            return round(min(distances), mantissa)
        elif Topology.IsInstance(topology, "Shell") or Topology.IsInstance(topology, "Cell") or Topology.IsInstance(topology, "CellComplex") or Topology.IsInstance(topology, "Cluster"):
            vertices = Topology.Vertices(topology)
            distances = [distance_to_vertex(vertex, v) for v in vertices]
            edges = Topology.Edges(topology)
            distances += [distance_to_edge(vertex, e) for e in edges]
            faces = Topology.Faces(topology)
            distances += [distance_to_face(vertex, f, includeCentroid) for f in faces]
            if includeCentroid:
                distances.append(distance_to_vertex(vertex, Topology.Centroid(topology)))
            return round(min(distances), mantissa)
        else:
            print("Vertex.Distance - Error: Could not recognize the input topology. Returning None.")
            return None
    
    @staticmethod
    def EnclosingCells(vertex, topology, exclusive: bool = True, mantissa: int = 6, tolerance: float = 0.0001) -> list:
        """
        Returns the list of Cells found in the input topology that enclose the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        exclusive : bool , optional
            If set to True, return only the first found enclosing cell. The default is True.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a cell. The default is 0.0001.

        Returns
        -------
        list
            The list of enclosing cells.

        """
        from topologicpy.Topology import Topology
        
        def boundingBox(cell):
            vertices = Topology.Vertices(cell)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(Vertex.X(aVertex, mantissa=mantissa))
                y.append(Vertex.Y(aVertex, mantissa=mantissa))
                z.append(Vertex.Z(aVertex, mantissa=mantissa))
            return ([min(x), min(y), min(z), max(x), max(y), max(z)])
        
        if Topology.IsInstance(topology, "Cell"):
            cells = [topology]
        elif Topology.IsInstance(topology, "Cluster") or Topology.IsInstance(topology, "CellComplex"):
            cells = Topology.Cells(topology)
        else:
            return None
        if len(cells) < 1:
            return None
        enclosingCells = []
        for i in range(len(cells)):
            bbox = boundingBox(cells[i])
            if ((Vertex.X(vertex, mantissa=mantissa) < bbox[0]) or (Vertex.Y(vertex, mantissa=mantissa) < bbox[1]) or (Vertex.Z(vertex, mantissa=mantissa) < bbox[2]) or (Vertex.X(vertex, mantissa=mantissa) > bbox[3]) or (Vertex.Y(vertex, mantissa=mantissa) > bbox[4]) or (Vertex.Z(vertex, mantissa=mantissa) > bbox[5])) == False:
                if Vertex.IsInternal(vertex, cells[i], tolerance=tolerance):
                    if exclusive:
                        return([cells[i]])
                    else:
                        enclosingCells.append(cells[i])
        return enclosingCells

    @staticmethod
    def ExternalBoundary(vertex):
        """
        Returns the external boundary (self) of the input vertex. This method is trivial, but included for completeness.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.

        Returns
        -------
        topologic_core.Vertex
            The external boundary of the input vertex. This is the input vertex itself.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            print("Vertex.ExternalBoundary - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        return vertex
    
    @staticmethod
    def Fuse(vertices: list, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a list of vertices where vertices within a specified tolerance distance are fused while retaining duplicates, ensuring that vertices with nearly identical coordinates are replaced by a single shared coordinate.

        Parameters
        ----------
        vertices : list
            The input list of topologic vertices.
        mantissa : int , optional
            The desired length of the mantissa for retrieving vertex coordinates. The default is 6.
        tolerance : float , optional
            The desired tolerance for computing if vertices need to be fused. Any vertices that are closer to each other than this tolerance will be fused. The default is 0.0001.

        Returns
        -------
        list
            The list of fused vertices. This list contains the same number of vertices and in the same order as the input list of vertices. However, the coordinates
            of these vertices have now been modified so that they are exactly the same with other vertices that are within the tolerance distance.
        
        """
        from topologicpy.Topology import Topology
        import numpy as np

        def fuse_vertices(vertices, tolerance=0.0001):
            fused_vertices = []
            merged_indices = {}

            for idx, vertex in enumerate(vertices):
                if idx in merged_indices:
                    fused_vertices.append(fused_vertices[merged_indices[idx]])
                    continue

                merged_indices[idx] = len(fused_vertices)
                fused_vertex = vertex
                for i in range(idx + 1, len(vertices)):
                    if i in merged_indices:
                        continue

                    other_vertex = vertices[i]
                    distance = np.linalg.norm(np.array(vertex) - np.array(other_vertex))
                    if distance <= tolerance:
                        # Choose the coordinate with the least amount of decimal points
                        if count_decimal_points(other_vertex) < count_decimal_points(fused_vertex):
                            fused_vertex = other_vertex

                        merged_indices[i] = len(fused_vertices)

                fused_vertices.append(fused_vertex)

            return fused_vertices
        def count_decimal_points(vertex):
            # Count the number of decimal points in the coordinates
            decimals_list = []
            for coord in vertex:
                coord_str = str(coord)
                if '.' in coord_str:
                    decimals_list.append(len(coord_str.split('.')[1]))
                elif 'e' in coord_str:
                    decimals_list.append(int(coord_str.split('e')[1].replace('-','')))
            return max(decimals_list)

        if not isinstance(vertices, list):
            print("Vertex.Fuse - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) == 0:
            print("Vertex.Fuse - Error: The input vertices parameter does not contain any valid topologic vertices. Returning None.")
            return None
        
        vertices = [(Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa), Vertex.Z(v, mantissa=mantissa)) for v in vertices]
        fused_vertices = fuse_vertices(vertices, tolerance=tolerance)
        return_vertices = [Vertex.ByCoordinates(list(coord)) for coord in fused_vertices]
        return return_vertices

    @staticmethod
    def IncomingEdges(vertex, hostTopology, tolerance: float = 0.0001) -> list:
        """
        Returns the incoming edges connected to a vertex. An edge is considered incoming if its end vertex is
        coincident with the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        hostTopology : topologic_core.Topology
            The input host topology to which the vertex belongs.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of incoming edges

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            print("Vertex.IncomingEdges - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            print("Vertex.IncomingEdges - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        
        edges = Topology.SuperTopologies(vertex, hostTopology=hostTopology, topologyType="Edge")
        incoming_edges = []
        for edge in edges:
            ev = Edge.EndVertex(edge)
            if Vertex.Distance(vertex, ev) <= tolerance:
                incoming_edges.append(edge)
        return incoming_edges
    
    @staticmethod
    def Index(vertex, vertices: list, strict: bool = False, tolerance: float = 0.0001) -> int:
        """
        Returns index of the input vertex in the input list of vertices

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        vertices : list
            The input list of vertices.
        strict : bool , optional
            If set to True, the vertex must be strictly identical to the one found in the list. Otherwise, a distance comparison is used. The default is False.
        tolerance : float , optional
            The tolerance for computing if the input vertex is identical to a vertex from the list. The default is 0.0001.

        Returns
        -------
        int
            The index of the input vertex in the input list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        if not isinstance(vertices, list):
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) == 0:
            return None
        for i in range(len(vertices)):
            if strict:
                if Topology.IsSame(vertex, vertices[i]):
                    return i
            else:
                d = Vertex.Distance(vertex, vertices[i])
                if d <= tolerance:
                    return i
        return None

    @staticmethod
    def InterpolateValue(vertex, vertices: list, n: int = 3, key: str = "intensity", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Interpolates the value of the input vertex based on the values of the *n* nearest vertices.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        vertices : list
            The input list of vertices.
        n : int , optional
            The maximum number of nearest vertices to consider. The default is 3.
        key : str , optional
            The key that holds the value to be interpolated in the dictionaries of the vertices. The default is "intensity".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The tolerance for computing if the input vertex is coincident with another vertex in the input list of vertices. The default is 0.0001.

        Returns
        -------
        topologic_core.vertex
            The input vertex with the interpolated value stored in its dictionary at the key specified by the input key. Other keys and values in the dictionary are preserved.

        """

        def interpolate_value(point, data_points, n, tolerance=0.0001):
            """
            Interpolates the value associated with a point in 3D by averaging the values of the n nearest points.
            The influence of the adjacent points is inversely proportional to their distance from the input point.

            Args:
                data_points (list): A list of tuples, each representing a data point in 3D space as (x, y, z, value).
                                    The 'value' represents the value associated with that data point.
                point (tuple): A tuple representing the point in 3D space as (x, y, z) for which we want to interpolate a value.
                n (int): The number of nearest points to consider for interpolation.

            Returns:
                The interpolated value for the input point.
            """
            # Calculate the distances between the input point and all data points
            distances = [(distance(p[:3], point), p[3]) for p in data_points]

            # Sort the distances in ascending order
            sorted_distances = sorted(distances, key=lambda x: x[0])

            # Take the n nearest points
            nearest_points = sorted_distances[:n]

            n_p = nearest_points[0]
            n_d = n_p[0]
            if n_d <= tolerance:
                return n_p[1]

            # Calculate the weights for each nearest point based on inverse distance

            weights = [(1/d[0], d[1]) for d in nearest_points]

            # Normalize the weights so they sum to 1
            total_weight = sum(w[0] for w in weights)
            normalized_weights = [(w[0]/total_weight, w[1]) for w in weights]

            # Interpolate the value as the weighted average of the nearest points
            interpolated_value = sum(w[0]*w[1] for w in normalized_weights)

            return interpolated_value

        def distance(point1, point2):
            """
            Calculates the Euclidean distance between two points in 3D space.

            Args:
                point1 (tuple): A tuple representing a point in 3D space as (x, y, z).
                point2 (tuple): A tuple representing a point in 3D space as (x, y, z).

            Returns:
                The Euclidean distance between the two points.
            """
            return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)**0.5
        
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        if not isinstance(vertices, list):
            return None
        
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) == 0:
            return None
        
        point = (Vertex.X(vertex, mantissa=mantissa), Vertex.Y(vertex, mantissa=mantissa), Vertex.Z(vertex, mantissa=mantissa))
        data_points = []
        for v in vertices:
            d = Topology.Dictionary(v)
            value = Dictionary.ValueAtKey(d, key)
            if not value == None:
                if type(value) == int or type(value) == float:
                    data_points.append((Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa), Vertex.Z(v, mantissa=mantissa), value))
        if len(data_points) == 0:
            return None
        if n > len(data_points):
            n = len(data_points)
        value = interpolate_value(point, data_points, n, tolerance=0.0001)
        d = Topology.Dictionary(vertex)
        d = Dictionary.SetValueAtKey(d, key, value)
        vertex = Topology.SetDictionary(vertex, d)
        return vertex

    @staticmethod
    def IsCoincident(vertexA, vertexB, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input vertexA is coincident with the input vertexB. Returns False otherwise.

        Parameters
        ----------
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        tolerance : float , optional
            The tolerance for computing if the input vertexA is coincident with the input vertexB. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertexA is coincident with the input vertexB. False otherwise.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Vertex.IsCoincident - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Vertex.IsICoincident - Error: The input vertexB parameter is not a valid vertex. Returning None.")
            return None
        return Vertex.IsInternal(vertexA, vertexB, tolerance=tolerance, silent=silent)

    @staticmethod
    def IsExternal(vertex, topology, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input vertex is external to the input topology. Returns False otherwise.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        tolerance : float , optional
            The tolerance for computing if the input vertex is external to the input topology. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        bool
            True if the input vertex is external to the input topology. False otherwise.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.IsExternal - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Vertex.IsExternal - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        return not (Vertex.IsPeripheral(vertex, topology, tolerance=tolerance, silent=silent) or Vertex.IsInternal(vertex, topology, tolerance=tolerance, silent=silent))
    
    @staticmethod
    def IsInternal(vertex, topology, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input vertex is inside the input topology. Returns False otherwise.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        tolerance : float , optional
            The tolerance for computing if the input vertex is internal to the input topology. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        bool
            True if the input vertex is internal to the input topology. False otherwise.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.IsInternal - Error: The input vertex parameter is not a valid vertex. Returning None.", vertex, topology)
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Vertex.IsInternal - Error: The input topology parameter is not a valid topology. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

        if Topology.IsInstance(topology, "Vertex"):
            return Vertex.Distance(vertex, topology) <= tolerance
        elif Topology.IsInstance(topology, "Edge"):
            try:
                parameter = Edge.ParameterAtVertex(topology, vertex)
            except:
                parameter = 400 #arbitrary large number greater than 1
            if parameter == None:
                return False
            return 0 <= parameter <= 1
        elif Topology.IsInstance(topology, "Wire"):
            vertices = [v for v in Topology.Vertices(topology) if Vertex.Degree(v, topology) > 1]
            edges = Wire.Edges(topology)
            sub_list = vertices + edges
            for sub in sub_list:
                if Vertex.IsInternal(vertex, sub, tolerance=tolerance, silent=silent):
                    return True
            return False
        elif Topology.IsInstance(topology, "Face"):
            # Test the distance first
            if Vertex.PerpendicularDistance(vertex, topology) > tolerance:
                return False
            if Vertex.IsPeripheral(vertex, topology):
                return False
            normal = Face.Normal(topology)
            proj_v = Vertex.Project(vertex, topology)
            v1 = Topology.TranslateByDirectionDistance(proj_v, normal, 1)
            v2 = Topology.TranslateByDirectionDistance(proj_v, normal, -1)
            edge = Edge.ByVertices(v1, v2)
            intersect = edge.Intersect(topology)
            if intersect == None:
                return False
            return True
        elif Topology.IsInstance(topology, "Shell"):
            if Vertex.IsPeripheral(vertex, topology, tolerance=tolerance, silent=silent):
                return False
            else:
                edges = Topology.Edges(topology)
                for edge in edges:
                    if Vertex.IsInternal(vertex, edge, tolerance=tolerance, silent=silent):
                        return True
                faces = Topology.Faces(topology)
                for face in faces:
                    if Vertex.IsInternal(vertex, face, tolerance=tolerance, silent=silent):
                        return True
                return False
        elif Topology.IsInstance(topology, "Cell"):
            return topologic.CellUtility.Contains(topology, vertex, tolerance) == 0 # Hook to Core
        elif Topology.IsInstance(topology, "CellComplex"):
            ext_boundary = CellComplex.ExternalBoundary(topology)
            return Vertex.IsInternal(vertex, ext_boundary, tolerance=tolerance, silent=silent)
        elif Topology.IsInstance(topology, "Cluster"):
            sub_list = Cluster.FreeTopologies(topology)
            for sub in sub_list:
                if Vertex.IsInternal(vertex, sub, tolerance=tolerance, silent=silent):
                    return True
            return False
        return False
    
    @staticmethod
    def IsPeripheral(vertex, topology, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input vertex is peripheral to the input topology. Returns False otherwise.
        A vertex is said to be peripheral to the input topology if:
        01. Vertex: If it is internal to it (i.e. coincident with it).
        02. Edge: If it is internal to its start or end vertices.
        03. Manifold open wire: If it is internal to its start or end vertices.
        04. Manifold closed wire: If it is internal to any of its vertices.
        05. Non-manifold wire: If it is internal to any of its vertices that has a vertex degree of 1.
        06. Face: If it is internal to any of its edges or vertices.
        07. Shell: If it is internal to external boundary
        08. Cell: If it is internal to any of its faces, edges, or vertices.
        09. CellComplex: If it is peripheral to its external boundary.
        10. Cluster: If it is peripheral to any of its free topologies. (See Cluster.FreeTopologies)

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        tolerance : float , optional
            The tolerance for computing if the input vertex is peripheral to the input topology. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        bool
            True if the input vertex is peripheral to the input topology. False otherwise.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.IsPeripheral - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Vertex.IsPeripheral - Error: The input topology parameter is not a valid topology. Returning None.")
            return None

        if Topology.IsInstance(topology, "Vertex"):
            return Vertex.IsInternal(vertex, topology, tolerance=tolerance, silent=silent)
        elif Topology.IsInstance(topology, "Edge"):
            sv = Edge.StartVertex(topology)
            ev = Edge.EndVertex(topology)
            f1 = Vertex.IsInternal(vertex, sv, tolerance=tolerance, silent=silent)
            f2 = Vertex.IsInternal(vertex, ev, tolerance=tolerance, silent=silent)
            return f1 or f2
        elif Topology.IsInstance(topology, "Wire"):
            if Wire.IsManifold(topology):
                if not Wire.IsClosed(topology):
                    sv = Wire.StartVertex(topology)
                    ev = Wire.EndVertex(topology)
                    f1 = Vertex.IsInternal(vertex, sv, tolerance=tolerance, silent=silent)
                    f2 = Vertex.IsInternal(vertex, ev, tolerance=tolerance, silent=silent)
                    return f1 or f2
                else:
                    sub_list = [v for v in Topology.Vertices(topology)]
                    for sub in sub_list:
                        if Vertex.IsPeripheral(vertex, sub, tolerance=tolerance, silent=silent):
                            return True
                    return False
            else:
                sub_list = [v for v in Topology.Vertices(topology) if Vertex.Degree(v, topology) == 1]
                for sub in sub_list:
                    if Vertex.IsPeripheral(vertex, sub, tolerance=tolerance, silent=silent):
                        return True
                return False
        elif Topology.IsInstance(topology, "Face"):
            sub_list = Topology.Vertices(topology) + Topology.Edges(topology)
            for sub in sub_list:
                if Vertex.IsInternal(vertex, sub, tolerance=tolerance, silent=silent):
                    return True
            return False
        elif Topology.IsInstance(topology, "Shell"):
            ext_boundary = Shell.ExternalBoundary(topology)
            sub_list = Topology.Vertices(ext_boundary) + Topology.Edges(ext_boundary)
            for sub in sub_list:
                if Vertex.IsInternal(vertex, sub, tolerance=tolerance, silent=silent):
                    return True
            return False
        elif Topology.IsInstance(topology, "Cell"):
            sub_list = Topology.Vertices(topology) + Topology.Edges(topology) + Topology.Faces(topology)
            for sub in sub_list:
                if Vertex.IsInternal(vertex, sub, tolerance=tolerance, silent=silent):
                    return True
            return False
        elif Topology.IsInstance(topology, "CellComplex"):
            ext_boundary = CellComplex.ExternalBoundary(topology)
            sub_list = Topology.Vertices(ext_boundary) + Topology.Edges(ext_boundary) + Topology.Faces(ext_boundary)
            for sub in sub_list:
                if Vertex.IsInternal(vertex, sub, tolerance=tolerance, silent=silent):
                    return True
            return False
        elif Topology.IsInstance(topology, "Cluster"):
            sub_list = Cluster.FreeTopologies(topology)
            for sub in sub_list:
                if Vertex.IsPeripheral(vertex, sub, tolerance=tolerance, silent=silent):
                    return True
            return False
        return False
    
    @staticmethod
    def NearestVertex(vertex, topology, useKDTree: bool = True, mantissa: int = 6):
        """
        Returns the vertex found in the input topology that is the nearest to the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology to be searched for the nearest vertex.
        useKDTree : bool , optional
            if set to True, the algorithm will use a KDTree method to search for the nearest vertex. The default is True.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        topologic_core.Vertex
            The nearest vertex.

        """
        from topologicpy.Topology import Topology

        def SED(a, b):
            """Compute the squared Euclidean distance between X and Y."""
            p1 = (Vertex.X(a, mantissa=mantissa), Vertex.Y(a, mantissa=mantissa), Vertex.Z(a, mantissa=mantissa))
            p2 = (Vertex.X(b, mantissa=mantissa), Vertex.Y(b, mantissa=mantissa), Vertex.Z(b, mantissa=mantissa))
            return sum((i-j)**2 for i, j in zip(p1, p2))
        
        BT = collections.namedtuple("BT", ["value", "left", "right"])
        BT.__doc__ = """
        A Binary Tree (BT) with a node value, and left- and
        right-subtrees.
        """
        def firstItem(v):
            return Vertex.X(v, mantissa=mantissa)
        def secondItem(v):
            return Vertex.Y(v, mantissa=mantissa)
        def thirdItem(v):
            return Vertex.Z(v, mantissa=mantissa)

        def itemAtIndex(v, index):
            if index == 0:
                return Vertex.X(v, mantissa=mantissa)
            elif index == 1:
                return Vertex.Y(v, mantissa=mantissa)
            elif index == 2:
                return Vertex.Z(v, mantissa=mantissa)

        def sortList(vertices, index):
            if index == 0:
                vertices.sort(key=firstItem)
            elif index == 1:
                vertices.sort(key=secondItem)
            elif index == 2:
                vertices.sort(key=thirdItem)
            return vertices
        
        def kdtree(topology):
            assert Topology.IsInstance(topology, "Topology"), "Vertex.NearestVertex: The input is not a Topology."
            vertices = Topology.Vertices(topology)
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
            vertices = Topology.Vertices(topology)
            distances = []
            indices = []
            for i in range(len(vertices)):
                distances.append(SED(vertex, vertices[i]))
                indices.append(i)
            sorted_indices = [x for _, x in sorted(zip(distances, indices))]
        return vertices[sorted_indices[0]]

    @staticmethod
    def Normal(vertices, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Computes the normal vector of a list of co-planar Topologic vertices.
        Depending on the order of the vertices, the normal can be flipped 180 degrees.

        Parameters
        ----------
        vertices : list
            A list of Topologic Vertex objects that are assumed to be co-planar.
        mantissa : int, optional
            The desired length of the mantissa. The default is 6.
        tolerance : float, optional
           The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True no warnings or errors are printed. The default is False.

        Returns
        -------
        list
            A unit normal vector [x, y, z] of the plane defined by the vertices, or None if invalid.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import numpy as np

        if not isinstance(vertices, list):
            if not silent:
                print("Vertex.Normal - Error: The vertices input parameter is not a valid list. Returning None.")
            return None
        
        verts = [v for v in vertices if Topology.IsInstance(v, "vertex")]

        if len(verts) < 3:
            if not silent:
                print("Vertex.Normal - Error: The list of vertices contains less than 3 valid topologic vertices. Returning None.")
            return None  # At least 3 vertices are needed

        coords = np.array([Vertex.Coordinates(v, mantissa=mantissa) for v in verts])
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid

        # Use SVD to find the normal as the vector corresponding to the smallest singular value
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # The last row is the normal of the best-fit plane

        norm = np.linalg.norm(normal)
        if norm < tolerance:
            return None  # Degenerate normal
        return list(np.round(normal / norm, mantissa))
    
    @staticmethod
    def Origin():
        """
        Returns a vertex with coordinates (0, 0, 0)

        Parameters
        ----------

        Returns
        -------
        topologic_core.Vertex
        """
        return Vertex.ByCoordinates(0, 0, 0)
    
    @staticmethod
    def OutgoingEdges(vertex, hostTopology, tolerance: float = 0.0001) -> list:
        """
        Returns the outgoing edges connected to a vertex. An edge is considered incoming if its start vertex is
        coincident with the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        hostTopology : topologic_core.Topology
            The input host topology to which the vertex belongs.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of outgoing edges

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            print("Vertex.OutgoingEdges - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            print("Vertex.OutgoingEdges - Error: The input graph parameter is not a valid graph. Returning None.")
            return None
        
        edges = Topology.SuperTopologies(vertex, hostTopology=hostTopology, topologyType="Edge")
        outgoing_edges = []
        for edge in edges:
            sv = Edge.StartVertex(edge)
            if Vertex.Distance(vertex, sv) <= tolerance:
                outgoing_edges.append(edge)
        return outgoing_edges
    
    @staticmethod
    def PerpendicularDistance(vertex, face, mantissa: int = 6):
        """
        Returns the perpendicular distance between the input vertex and the input face. The face is considered to be infinite.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        face : topologic_core.Face
            The input face.
        mantissa: int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The distance between the input vertex and the input topology.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        def distance_point_to_line(point, line_start, line_end):
            # Convert input points to NumPy arrays for vector operations
            point = np.array(point)
            line_start = np.array(line_start)
            line_end = np.array(line_end)
            
            # Calculate the direction vector of the edge
            line_direction = line_end - line_start
            
            # Vector from the edge's starting point to the point
            point_to_start = point - line_start
            
            # Calculate the parameter 't' where the projection of the point onto the edge occurs
            if np.dot(line_direction, line_direction) == 0:
                t = 0
            else:
                t = np.dot(point_to_start, line_direction) / np.dot(line_direction, line_direction)
            
            # Check if 't' is outside the range [0, 1], and if so, calculate distance to closest endpoint
            if t < 0:
                return np.linalg.norm(point - line_start)
            elif t > 1:
                return np.linalg.norm(point - line_end)
            
            # Calculate the closest point on the edge to the given point
            closest_point = line_start + t * line_direction
            
            # Calculate the distance between the closest point and the given point
            distance = np.linalg.norm(point - closest_point)
            return float(distance)
        
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Vertex.PerpendicularDistance - Error: The input vertex is not a valid topologic vertex. Returning None.")
            return None
        if not Topology.IsInstance(face, "Face"):
            print("Vertex.PerpendicularDistance - Error: The input face is not a valid topologic face. Returning None.")
            return None
        dic = Face.PlaneEquation(face)
        if dic == None: # The face is degenerate. Try to treat as an edge.
            point = Vertex.Coordinates(vertex)
            face_vertices = Topology.Vertices(face)
            line_start = Vertex.Coordinates(face_vertices[0])
            line_end = Vertex.Coordinates(face_vertices[1])
            return round(distance_point_to_line(point, line_start, line_end), mantissa)
        a = dic["a"]
        b = dic["b"]
        c = dic["c"]
        d = dic["d"]
        x1, y1, z1 = Vertex.Coordinates(vertex)
        d = abs((a * x1 + b * y1 + c * z1 + d))
        e = (math.sqrt(a * a + b * b + c * c))
        if e == 0:
            return 0
        return round(d/e, mantissa)
    
    @staticmethod
    def PlaneEquation(vertices, mantissa: int = 6):
        """
        Returns the equation of the average plane passing through a list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        dict
            The dictionary containing the values of a, b, c, d for the plane equation in the form of ax+by+cz+d=0.
            The keys in the dictionary are ["a", "b", "c". "d"]
        """

        vertices = [Vertex.Coordinates(v) for v in vertices]
        # Convert vertices to a NumPy array for easier calculations
        vertices = np.array(vertices)

        # Calculate the centroid of the vertices
        centroid = np.mean(vertices, axis=0)

        # Center the vertices by subtracting the centroid
        centered_vertices = vertices - centroid

        # Calculate the covariance matrix
        covariance_matrix = np.dot(centered_vertices.T, centered_vertices)

        # Find the normal vector by computing the eigenvector of the smallest eigenvalue
        _, eigen_vectors = np.linalg.eigh(covariance_matrix)
        normal_vector = eigen_vectors[:, 0]

        # Normalize the normal vector
        normal_vector /= np.linalg.norm(normal_vector)

        # Calculate the constant D using the centroid and the normal vector
        d = -np.dot(normal_vector, centroid)
        d = round(d, mantissa)

        # Create the plane equation in the form Ax + By + Cz + D = 0
        a, b, c = normal_vector
        a = round(a, mantissa)
        b = round(b, mantissa)
        c = round(c, mantissa)

        return {"a":a, "b":b, "c":c, "d":d}
    
    @staticmethod
    def Point(x=0, y=0, z=0):
        """
        Creates a point (vertex) using the input parameters

        Parameters
        ----------
        x : float , optional.
            The desired x coordinate. The default is 0.
        y : float , optional.
            The desired y coordinate. The default is 0.
        z : float , optional.
            The desired z coordinate. The default is 0.

        Returns
        -------
        topologic_core.Vertex
        """
        
        return Vertex.ByCoordinates(x, y, z)

    @staticmethod
    def Project(vertex, face, direction: bool = None, mantissa: int = 6):
        """
        Returns a vertex that is the projection of the input vertex unto the input face.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex to project unto the input face.
        face : topologic_core.Face
            The input face that receives the projection of the input vertex.
        direction : vector, optional
            The direction in which to project the input vertex unto the input face. If not specified, the direction of the projection is the normal of the input face. The default is None.
        mantissa : int , optional
            The length of the desired mantissa. The default is 6.

        Returns
        -------
        topologic_core.Vertex
            The projected vertex.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        
        def project_point_onto_plane(point, plane_coeffs, direction_vector):
            """
            Project a 3D point onto a plane defined by its coefficients and using a direction vector.

            Parameters:
                point (tuple or list): The 3D point coordinates (x, y, z).
                plane_coeffs (tuple or list): The coefficients of the plane equation (a, b, c, d).
                direction_vector (tuple or list): The direction vector (vx, vy, vz).

            Returns:
                tuple: The projected point coordinates (x_proj, y_proj, z_proj).
            """
            # Unpack point coordinates
            x, y, z = point

            # Unpack plane coefficients
            a, b, c, d = plane_coeffs

            # Unpack direction vector
            vx, vy, vz = direction_vector

            # Calculate the distance from the point to the plane
            distance = (a * x + b * y + c * z + d) / (a * vx + b * vy + c * vz)

            # Calculate the projected point coordinates
            x_proj = x - distance * vx
            y_proj = y - distance * vy
            z_proj = z - distance * vz

            return [x_proj, y_proj, z_proj]

        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        if not Topology.IsInstance(face, "Face"):
            return None
        eq = Face.PlaneEquation(face, mantissa= mantissa)
        if direction == None or direction == []:
            direction = Face.Normal(face)
        pt = project_point_onto_plane(Vertex.Coordinates(vertex), [eq["a"], eq["b"], eq["c"], eq["d"]], direction)
        return Vertex.ByCoordinates(pt[0], pt[1], pt[2])

    @staticmethod
    def Separate(*vertices, minDistance: float = 0.0001, silent: bool = False):
        """
        Separates the input vertices such that no two vertices are within the input minimum distance.

        Parameters
        ----------
        vertices : *topologicpy.Vertex
            One or more instances of a topologic vertex to be processed.
        minDistance : float , optional
            The desired minimum distance. The default is 0.0001.
        silent : bool , optional
                If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The list of vertices with adjusted positions

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from math import sqrt
        from scipy.spatial import KDTree
        import numpy as np

        if len(vertices) == 0:
            if not silent:
                print("Vertex.Separate - Error: The input vertices parameter is an empty list. Returning None.")
            return None
        if len(vertices) == 1:
            vertices = vertices[0]
            if isinstance(vertices, list):
                if len(vertices) == 0:
                    if not silent:
                        print("Vertex.Separate - Error: The input vertices parameter is an empty list. Returning None.")
                    return None
                else:
                    vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
                    if len(vertexList) == 0:
                        if not silent:
                            print("Vertex.Separate - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
                        return None
            else:
                if not silent:
                    print("Vertex.Separate - Warning: The input vertices parameter contains only one vertex. Returning the same vertex.")
                return vertices
        else:
            vertexList = Helper.Flatten(list(vertices))
            vertexList = [x for x in vertexList if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) == 0:
            if not silent:
                print("Vertex.Separate - Error: The input parameters do not contain any valid vertices. Returning None.")
            return None
    
        coords = np.array([[v.X(), v.Y(), v.Z()] for v in vertexList])  # Extract coordinates
        tree = KDTree(coords)  # Build k-d tree for efficient neighbor search
        
        for i, vertex in enumerate(coords):
            neighbors = tree.query_ball_point(vertex, minDistance)
            for neighbor_index in neighbors:
                if neighbor_index != i:  # Avoid self-comparison
                    direction = coords[neighbor_index] - vertex
                    distance = np.linalg.norm(direction)
                    if distance < minDistance:
                        # Move current vertex away from its neighbor
                        adjustment = (minDistance - distance) / 2
                        unit_vector = direction / distance if distance != 0 else np.random.rand(3)
                        coords[i] -= unit_vector * adjustment
                        coords[neighbor_index] += unit_vector * adjustment

            # Rebuild the k-d tree after adjustment
            tree = KDTree(coords)

        # Convert adjusted coordinates back to Vertex objects
        separated_vertices = [Vertex.ByCoordinates(x, y, z) for x, y, z in coords]
        for i, vertex in enumerate(vertexList):
            d = Topology.Dictionary(vertex)
            if len(Dictionary.Keys(d)) > 0:
                separated_vertices[i] = Topology.SetDictionary(separated_vertices[i], d)
        return separated_vertices

    @staticmethod
    def Transform(vertex, matrix, mantissa: int = 6, silent: bool = False):
        """
        Transforms a 3D vertex using a 4x4 transformation matrix.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex
        matrix : list
            The 4x4 transformation matrix.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Vertex
            The transformed vertex.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "vertex"):
            if not silent:
                print("Vertex.Transform - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        
        if not isinstance(matrix, list):
            if not silent:
                print("Vertex.Transform - Error: The input matrix parameter is not a valid 4X4 matrix. Returning None.")
            return None
        
        matrix = np.array(matrix)  # Convert list to numpy array if necessary
    
        # Check if the shape of the matrix is (4, 4)
        if not matrix.shape == (4, 4):
            if not silent:
                print("Vertex.Transform - Error: The input matrix parameter is not a valid 4X4 matrix. Returning None.")
            return None
        
        # Convert the vertex to a 4D homogeneous coordinate
        coords = Vertex.Coordinates(vertex, mantissa=mantissa)
        homogeneous_coords= np.array([coords[0], coords[1], coords[2], 1.0])
        
        # Perform matrix multiplication
        transformed_coords = np.dot(matrix, homogeneous_coords)
        
        # Convert back to 3D by dividing by w
        if not np.isclose(transformed_coords[3], 0):  # Avoid division by zero
            transformed_coords /= transformed_coords[3]
        
        # Return the transformed (x', y', z') as a list
        coords = transformed_coords[:3].tolist()
        coords = [round(v, mantissa) for v in coords]
        return Vertex.ByCoordinates(coords)
    
    @staticmethod
    def X(vertex, mantissa: int = 6) -> float:
        """
        Returns the X coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The X coordinate of the input vertex.

        """
        try:
            return round(vertex.X(), mantissa) # Hook to Core
        except:
            try:
                return round(vertex.x(), mantissa) # Hook to Core
            except:
                return None

    @staticmethod
    def Y(vertex, mantissa: int = 6) -> float:
        """
        Returns the Y coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The Y coordinate of the input vertex.

        """
        try:
            return round(vertex.Y(), mantissa) # Hook to Core
        except:
            try:
                return round(vertex.y(), mantissa) # Hook to Core
            except:
                return None

    @staticmethod
    def Z(vertex, mantissa: int = 6) -> float:
        """
        Returns the Z coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The Z coordinate of the input vertex.

        """
        try:
            return round(vertex.Z(), mantissa) # Hook to Core
        except:
            try:
                return round(vertex.z(), mantissa) # Hook to Core
            except:
                return None
           