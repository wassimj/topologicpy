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
    def AlignCoordinates(vertex,
                        xList: list = None,
                        yList: list = None,
                        zList: list = None,
                        xEpsilon: float = 0.0001,
                        yEpsilon: float = 0.0001,
                        zEpsilon: float = 0.0001,
                        transferDictionary: bool = False,
                        mantissa: int = 6,
                        silent: bool = False):
        """
        Aligns the coordinates of the input vertex with the closest values in the
        supplied x, y, and z coordinate lists.
        Any coordinate list may be omitted. If a list is omitted, empty, or contains
        no valid numeric values, the corresponding coordinate is left unchanged.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        xList : list , optional
            The input numerical list of x-coordinates. Default is None.
        yList : list , optional
            The input numerical list of y-coordinates. Default is None.
        zList : list , optional
            The input numerical list of z-coordinates. Default is None.
        xEpsilon : float , optional
            The tolerance within which the x-coordinate will be snapped. Default is 0.0001.
        yEpsilon : float , optional
            The tolerance within which the y-coordinate will be snapped. Default is 0.0001.
        zEpsilon : float , optional
            The tolerance within which the z-coordinate will be snapped. Default is 0.0001.
        transferDictionary : bool , optional
            If set to True, the dictionary of the input vertex is transferred to the new vertex.
            Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The created vertex aligned to the input coordinate lists.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        import math

        def _is_number(value):
            return (
                isinstance(value, (int, float)) and
                not isinstance(value, bool) and
                math.isfinite(value)
            )

        def _clean_numeric_list(values):
            if not isinstance(values, list):
                return []
            return [float(v) for v in values if _is_number(v)]

        def _aligned_coordinate(value, values, epsilon):
            values = _clean_numeric_list(values)
            if len(values) < 1:
                return value
            try:
                closest_value = round(values[Helper.ClosestMatch(value, values)], mantissa)
            except Exception:
                return value
            if abs(value - closest_value) <= epsilon:
                return closest_value
            return value

        if not Topology.IsInstance(vertex, "vertex"):
            if not silent:
                print("Vertex.AlignCoordinates - Error: The input vertex parameter is not a topologic vertex. Returning None.")
            return None

        x, y, z = Vertex.Coordinates(vertex, mantissa=mantissa)
        x = _aligned_coordinate(x, xList, xEpsilon)
        y = _aligned_coordinate(y, yList, yEpsilon)
        z = _aligned_coordinate(z, zList, zEpsilon)

        return_vertex = Vertex.ByCoordinates(x, y, z)
        if not Topology.IsInstance(return_vertex, "vertex"):
            if not silent:
                print("Vertex.AlignCoordinates - Error: Could not create the aligned vertex. Returning None.")
            return None

        if transferDictionary == True:
            return_vertex = Topology.SetDictionary(
                return_vertex,
                Topology.Dictionary(vertex),
                silent=silent
            )
        return return_vertex

    @staticmethod
    def AreCollinear(vertices: list, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input list of vertices is collinear. Returns False otherwise.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        mantissa : int , optional
            The number of decimal places used for compatibility with the TopologicPy API. The geometric test itself is performed at full coordinate precision. Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the valid input vertices are collinear. False otherwise.
        """
        from topologicpy.Topology import Topology
        import math

        if not isinstance(vertices, list):
            if not silent:
                print("Vertex.AreCollinear - Error: The input vertices parameter is not a valid list. Returning None.")
            return None

        vertexList = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertexList) < 2:
            if not silent:
                print("Vertex.AreCollinear - Error: The input list of vertices does not contain sufficient valid vertices. Returning None.")
            return None
        if len(vertexList) < 3:
            return True

        tol = abs(float(tolerance))
        points = [Vertex.Coordinates(v, mantissa=None) for v in vertexList]
        points = [p for p in points if isinstance(p, (list, tuple)) and len(p) >= 3]
        if len(points) < 2:
            if not silent:
                print("Vertex.AreCollinear - Error: Could not extract sufficient valid coordinates. Returning None.")
            return None
        if len(points) < 3:
            return True

        def _distance(a, b):
            return math.sqrt(sum((float(b[i]) - float(a[i])) ** 2 for i in range(3)))

        unique_points = []
        for p in points:
            p = [float(p[0]), float(p[1]), float(p[2])]
            if not any(_distance(p, q) <= tol for q in unique_points):
                unique_points.append(p)

        if len(unique_points) < 3:
            return True

        p0 = unique_points[0]
        p1 = next((p for p in unique_points[1:] if _distance(p0, p) > tol), None)
        if p1 is None:
            return True

        bx, by, bz = p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]
        base_length = math.sqrt(bx*bx + by*by + bz*bz)
        if base_length <= tol:
            return True

        for p in unique_points[1:]:
            tx, ty, tz = p[0]-p0[0], p[1]-p0[1], p[2]-p0[2]
            cx = by*tz - bz*ty
            cy = bz*tx - bx*tz
            cz = bx*ty - by*tx
            perpendicular_distance = math.sqrt(cx*cx + cy*cy + cz*cz) / base_length
            if perpendicular_distance > tol:
                return False
        return True
    
    @staticmethod
    def AreCoplanar(vertices: list, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input list of vertices is coplanar. Returns False otherwise.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        mantissa : int , optional
            The number of decimal places used for compatibility with the TopologicPy API. The geometric test itself is performed at full coordinate precision. Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the valid input vertices are coplanar. False otherwise.
        """
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            if not silent:
                print("Vertex.AreCoplanar - Error: The vertices input parameter is not a valid list. Returning None.")
            return None

        verts = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(verts) < 3:
            if not silent:
                print("Vertex.AreCoplanar - Error: The list of vertices contains less than 3 valid topologic vertices. Returning None.")
            return None

        try:
            coords = np.asarray([Vertex.Coordinates(v, mantissa=None) for v in verts], dtype=float)
            centroid = coords.mean(axis=0)
            centered = coords - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            norm = float(np.linalg.norm(normal))
            if norm <= max(abs(float(tolerance)), 1.0e-15):
                return True
            normal = normal / norm
            distances = np.abs(centered @ normal)
            return bool(float(np.max(distances)) <= abs(float(tolerance)))
        except Exception:
            if not silent:
                print("Vertex.AreCoplanar - Error: Could not determine coplanarity. Returning None.")
            return None
    
    @staticmethod
    def AreIpsilateral(vertices: list, face, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input vertices lie on the same side of a planar face.
        If at least one vertex lies on the face's supporting plane within tolerance,
        this method returns True, preserving the historical TopologicPy behaviour.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic_core.Face
            The input planar face.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the vertices are on the same side of the face. False otherwise.
            Returns None if the inputs are invalid or the face cannot define one
            global supporting plane.
        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            if not silent:
                print("Vertex.AreIpsilateral - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Vertex.AreIpsilateral - Error: The input face parameter is not a valid face. Returning None.")
            return None

        vertexList = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertexList) < 2:
            if not silent:
                print("Vertex.AreIpsilateral - Error: The input vertices parameter contains less than two valid vertices. Returning None.")
            return None

        tol = abs(float(tolerance))
        signed_distances = None

        if Core.HasAttribute("VertexUtility", "SignedDistanceToFace"):
            try:
                signed_distances = [
                    Core.VertexUtility.SignedDistanceToFace(v, face, tol)
                    for v in vertexList
                ]
                if any(value is None for value in signed_distances):
                    if not silent:
                        print("Vertex.AreIpsilateral - Error: The input face does not define a single planar supporting surface. Returning None.")
                    return None
            except Exception:
                signed_distances = None

        if signed_distances is None:
            # Legacy TopologicCore compatibility path.
            try:
                reference_vertex = Face.VertexByParameters(face, 0.5, 0.5)
                reference = Vertex.Coordinates(reference_vertex, mantissa=None)
                try:
                    normal = Face.Normal(face, mantissa=12)
                except TypeError:
                    normal = Face.Normal(face)
                if reference is None or normal is None:
                    return None
                signed_distances = []
                for v in vertexList:
                    point = Vertex.Coordinates(v, mantissa=None)
                    signed_distances.append(
                        (point[0] - reference[0]) * normal[0]
                        + (point[1] - reference[1]) * normal[1]
                        + (point[2] - reference[2]) * normal[2]
                    )
            except Exception:
                if not silent:
                    print("Vertex.AreIpsilateral - Error: Could not determine the side of the input face. Returning None.")
                return None

        signs = []
        for value in signed_distances:
            value = float(value)
            if abs(value) <= tol:
                continue
            signs.append(1 if value > 0.0 else -1)
        return len(set(signs)) <= 1
    
    @staticmethod
    def AreIpsilateralCluster(cluster, face, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the vertices of the input cluster lie on the same side of a planar face.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster containing vertices.
        face : topologic_core.Face
            The input planar face.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the cluster vertices are on the same side of the face. False otherwise.
        """
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(cluster, "Topology"):
            if not silent:
                print("Vertex.AreIpsilateralCluster - Error: The input cluster parameter is not a valid topology. Returning None.")
            return None
        vertices = Topology.SubTopologies(cluster, subTopologyType="vertex")
        return Vertex.AreIpsilateral(vertices, face, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def AreOnSameSide(vertices: list, face, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input vertices lie on the same side of a planar face.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : topologic_core.Face
            The input planar face.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the vertices lie on the same side of the face. False otherwise.
        """
        return Vertex.AreIpsilateral(vertices, face, tolerance=tolerance, silent=silent)

    @staticmethod
    def AreOnSameSideCluster(cluster, face, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the vertices of the input cluster lie on the same side of a planar face.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster containing vertices.
        face : topologic_core.Face
            The input planar face.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the cluster vertices lie on the same side of the face. False otherwise.
        """
        return Vertex.AreIpsilateralCluster(cluster, face, tolerance=tolerance, silent=silent)

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
            The X coordinate. Default is 0.
        y : float , optional
            The Y coordinate. Default is 0.
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
            vertex = Core.Vertex.ByCoordinates(x, y, z)
        except:
            vertex = None
            print("Vertex.ByCoordinates - Error: Could not create a topologic vertex. Returning None.")
        return vertex

    @staticmethod
    def ByOffset2DRelativeToEdge(vertex, edge, offset: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a new vertex offset from the input vertex in the XY plane using
        the left-hand normal of the chord from the input edge's start vertex to
        its end vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The vertex to offset.
        edge : topologic_core.Edge
            The reference edge. For a curved edge, its start-to-end chord defines
            the 2D reference direction.
        offset : float , optional
            The offset distance. Default is 1.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The offset vertex.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.ByOffset2DRelativeToEdge - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Vertex.ByOffset2DRelativeToEdge - Error: The input edge parameter is not a valid edge. Returning None.")
            return None

        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)
        p1 = Vertex.Coordinates(sv, mantissa=None)
        p2 = Vertex.Coordinates(ev, mantissa=None)
        point = Vertex.Coordinates(vertex, mantissa=None)
        if p1 is None or p2 is None or point is None:
            return None

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length <= abs(float(tolerance)):
            if not silent:
                print("Vertex.ByOffset2DRelativeToEdge - Error: The input edge has no usable XY chord direction. Returning None.")
            return None

        nx = -dy / length
        ny = dx / length
        return Vertex.ByCoordinates(
            point[0] + nx * float(offset),
            point[1] + ny * float(offset),
            point[2],
        )
    
    @staticmethod
    def Centroid(vertices: list, mantissa: int = 6):
        """
        Returns the centroid of the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

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
    def Coordinates(vertex, outputType: str = "xyz", mantissa: int = None) -> list:
        """
        Returns the coordinates of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        outputType : string, optional
            The desired output type. Could be any permutation or substring of "xyz" or the string "matrix". Default is "xyz". The input is case insensitive and the coordinates will be returned in the specified order.
        mantissa : int , optional
            The number of decimal places to round the result to. None means no rounding. Default is None.

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
            The number of decimal places to round the result to. Default is 6.

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
            The topology type to search for. This can be any of "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. If set to None, the immediate supertopology type is searched for. Default is None.

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
    def Distance(vertex, topology, includeCentroid: bool = True, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the shortest distance between the input vertex and the input topology.
        The distance is measured to the topology's geometric boundary and constituent
        subtopologies, optionally including its centroid. On the PythonOCC backend,
        primitive geometric distances are delegated to OCCT.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        includeCentroid : bool , optional
            If set to True, the centroid of the input topology is also considered.
            Default is True.
        mantissa : int , optional
            The number of decimal places to round the returned distance to. Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The shortest distance between the input vertex and the considered
            geometry of the input topology.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.Distance - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Vertex.Distance - Error: The input topology parameter is not a valid topology. Returning None.")
            return None

        tol = abs(float(tolerance))

        def _round(value):
            if value is None:
                return None
            return round(float(value), mantissa) if mantissa is not None else float(value)

        def _native_distance(target):
            if not Core.HasAttribute("VertexUtility", "DistanceToTopology"):
                return None
            try:
                value = Core.VertexUtility.DistanceToTopology(vertex, target, tol)
                return None if value is None else float(value)
            except Exception:
                return None

        def _point_distance(a, b):
            pa = Vertex.Coordinates(a, mantissa=None)
            pb = Vertex.Coordinates(b, mantissa=None)
            if pa is None or pb is None:
                return None
            return math.sqrt(
                (pa[0] - pb[0]) ** 2
                + (pa[1] - pb[1]) ** 2
                + (pa[2] - pb[2]) ** 2
            )

        def _vertex_distance(target):
            value = _native_distance(target)
            if value is not None:
                return value
            try:
                if Core.HasAttribute("VertexUtility", "Distance"):
                    value = Core.VertexUtility.Distance(vertex, target)
                    if value is not None:
                        return float(value)
            except Exception:
                pass
            return _point_distance(vertex, target)

        def _edge_distance(edge):
            value = _native_distance(edge)
            if value is not None:
                return value

            # Legacy TopologicCore compatibility fallback. This is used only when
            # the active backend does not expose a native Vertex-to-topology distance.
            sv = Edge.StartVertex(edge, silent=True)
            ev = Edge.EndVertex(edge, silent=True)
            p = Vertex.Coordinates(vertex, mantissa=None)
            a = Vertex.Coordinates(sv, mantissa=None)
            b = Vertex.Coordinates(ev, mantissa=None)
            if p is None or a is None or b is None:
                return None
            ab = [b[i] - a[i] for i in range(3)]
            ap = [p[i] - a[i] for i in range(3)]
            denominator = sum(v * v for v in ab)
            if denominator <= tol * tol:
                return math.sqrt(sum((p[i] - a[i]) ** 2 for i in range(3)))
            parameter = sum(ap[i] * ab[i] for i in range(3)) / denominator
            parameter = max(0.0, min(1.0, parameter))
            q = [a[i] + parameter * ab[i] for i in range(3)]
            return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(3)))

        def _face_distance(face):
            value = _native_distance(face)
            if value is not None:
                return value

            # Legacy TopologicCore compatibility fallback.
            projected = Vertex.Project(vertex, face, mantissa=None, tolerance=tol, silent=True)
            if projected is not None and Vertex.IsInternal(projected, face, tolerance=tol, silent=True):
                value = Vertex.PerpendicularDistance(vertex, face, mantissa=None, tolerance=tol, silent=True)
                if value is not None:
                    return float(value)

            distances = []
            for edge in Topology.Edges(face, silent=True) or []:
                value = _edge_distance(edge)
                if value is not None:
                    distances.append(value)
            for v in Topology.Vertices(face, silent=True) or []:
                value = _vertex_distance(v)
                if value is not None:
                    distances.append(value)
            return min(distances) if distances else None

        def _centroid_distance(target):
            if not includeCentroid:
                return None
            try:
                centroid = Topology.Centroid(target, silent=True)
            except TypeError:
                try:
                    centroid = Topology.Centroid(target)
                except Exception:
                    centroid = None
            except Exception:
                centroid = None
            if not Topology.IsInstance(centroid, "Vertex"):
                return None
            return _vertex_distance(centroid)

        if Topology.IsInstance(topology, "Vertex"):
            return _round(_vertex_distance(topology))

        if Topology.IsInstance(topology, "Edge"):
            return _round(_edge_distance(topology))

        if Topology.IsInstance(topology, "Wire"):
            distances = []
            native = _native_distance(topology)
            if native is not None:
                distances.append(native)
            else:
                for edge in Topology.Edges(topology, silent=True) or []:
                    value = _edge_distance(edge)
                    if value is not None:
                        distances.append(value)
            centroid_distance = _centroid_distance(topology)
            if centroid_distance is not None:
                distances.append(centroid_distance)
            return _round(min(distances)) if distances else None

        if Topology.IsInstance(topology, "Face"):
            distances = []
            value = _face_distance(topology)
            if value is not None:
                distances.append(value)
            centroid_distance = _centroid_distance(topology)
            if centroid_distance is not None:
                distances.append(centroid_distance)
            return _round(min(distances)) if distances else None

        if (
            Topology.IsInstance(topology, "Shell")
            or Topology.IsInstance(topology, "Cell")
            or Topology.IsInstance(topology, "CellComplex")
            or Topology.IsInstance(topology, "Cluster")
        ):
            distances = []
            is_cluster = Topology.IsInstance(topology, "Cluster")

            # A Shell is itself a boundary, so its native shape distance is valid.
            # For Cells and CellComplexes we deliberately measure to constituent
            # boundary geometry rather than to the solid, preserving historical
            # Vertex.Distance semantics for points inside a volume. Heterogeneous
            # Clusters must consider every represented dimensionality.
            if Topology.IsInstance(topology, "Shell"):
                native = _native_distance(topology)
                if native is not None:
                    distances.append(native)

            faces = Topology.Faces(topology, silent=True) or []
            for face in faces:
                value = _face_distance(face)
                if value is not None:
                    distances.append(value)

            if is_cluster or not faces:
                edges = Topology.Edges(topology, silent=True) or []
                for edge in edges:
                    value = _edge_distance(edge)
                    if value is not None:
                        distances.append(value)

                if is_cluster or not edges:
                    for v in Topology.Vertices(topology, silent=True) or []:
                        value = _vertex_distance(v)
                        if value is not None:
                            distances.append(value)

            centroid_distance = _centroid_distance(topology)
            if centroid_distance is not None:
                distances.append(centroid_distance)

            return _round(min(distances)) if distances else None

        if not silent:
            print("Vertex.Distance - Error: Could not recognize the input topology. Returning None.")
        return None
    
    @staticmethod
    def EnclosingCells(vertex, topology, exclusive: bool = True, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the list of Cells found in the input topology that enclose the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        exclusive : bool , optional
            If set to True, return only the first found enclosing cell. Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a cell. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of enclosing cells.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Cell import Cell
        from topologicpy.BVH import BVH
        
        if Topology.IsInstance(topology, "Cell"):
            cells = [topology]
        elif Topology.IsInstance(topology, "Cluster") or Topology.IsInstance(topology, "CellComplex"):
            cells = Topology.Cells(topology)
        else:
            if not silent:
                print("Vertex.EnclosingCells - Error: The input topology does not contain any cells. Returning None.")
            return None
        if len(cells) < 1:
            if not silent:
                print("Vertex.EnclosingCells - Error: The input topology does not contain any cells. Returning None.")
            return None
        
        bvh = BVH.ByTopologies(cells, tolerance=tolerance, silent=True)
        candidates = BVH.Clashes(bvh, vertex, tolerance=tolerance)
        enclosingCells = []
        for i in range(len(candidates)):
            if Vertex.IsInternal(vertex, candidates[i], tolerance=tolerance):
                if exclusive:
                    return([candidates[i]])
                else:
                    enclosingCells.append(candidates[i])
        return enclosingCells
    
    @staticmethod
    def EnclosingEdges(vertex, topology, exclusive: bool = True, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the list of Edges found in the input topology that enclose the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        exclusive : bool , optional
            If set to True, return only the first found enclosing face. Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a face. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of enclosing faces.

        """
        from topologicpy.Topology import Topology
        from topologicpy.BVH import BVH

        if not Topology.IsInstance(vertex, "Vertex"):
            return None

        if Topology.IsInstance(topology, "Edge"):
            edges = [topology]
        elif Topology.IsInstance(topology, "Cluster") or \
             Topology.IsInstance(topology, "Wire") or \
            Topology.IsInstance(topology, "Shell") or \
            Topology.IsInstance(topology, "Cell") or \
            Topology.IsInstance(topology, "CellComplex"):
            edges = Topology.Edges(topology)
        else:
            if not silent:
                print("Vertex.EnclosingEdges - Error: The input topology does not contain any edges. Returning None.")
            return None

        if len(edges) < 1:
            if not silent:
                print("Vertex.EnclosingEdges - Error: The input topology does not contain any edges. Returning None.")
            return None

        bvh = BVH.ByTopologies(edges, tolerance=tolerance, silent=True)
        candidates = BVH.Clashes(bvh, vertex, tolerance=tolerance)

        enclosingEdges = []
        for i in range(len(candidates)):
            if Vertex.IsInternal(vertex, candidates[i], tolerance=tolerance):
                if exclusive:
                    return [candidates[i]]
                else:
                    enclosingEdges.append(candidates[i])

        return enclosingEdges
 
    @staticmethod
    def EnclosingFaces(vertex, topology, exclusive: bool = True, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the list of Faces found in the input topology that enclose the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        exclusive : bool , optional
            If set to True, return only the first found enclosing face. Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a face. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of enclosing faces.

        """
        from topologicpy.Topology import Topology
        from topologicpy.BVH import BVH

        if not Topology.IsInstance(vertex, "Vertex"):
            return None

        if Topology.IsInstance(topology, "Face"):
            faces = [topology]
        elif Topology.IsInstance(topology, "Cluster") or \
            Topology.IsInstance(topology, "Shell") or \
            Topology.IsInstance(topology, "Cell") or \
            Topology.IsInstance(topology, "CellComplex"):
            faces = Topology.Faces(topology)
        else:
            return None

        if len(faces) < 1:
            return None

        bvh = BVH.ByTopologies(faces, tolerance=tolerance, silent=True)
        candidates = BVH.Clashes(bvh, vertex, tolerance=tolerance)

        enclosingFaces = []
        for i in range(len(candidates)):
            if Vertex.IsInternal(vertex, candidates[i], tolerance=tolerance):
                if exclusive:
                    return [candidates[i]]
                else:
                    enclosingFaces.append(candidates[i])

        return enclosingFaces

    @staticmethod
    def ExternalBoundary(vertex, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the external boundary of the input vertex (None according OGC / ISO / DE-9IM). This method is trivial, but included for completeness.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The external boundary of the input vertex. This is the input vertex itself.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.ExternalBoundary - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        return None
    
    @staticmethod
    def Fuse(vertices: list, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a list of vertices where vertices within a specified tolerance distance are fused while retaining duplicates, ensuring that vertices with nearly identical coordinates are replaced by a single shared coordinate.

        Parameters
        ----------
        vertices : list
            The input list of topologic vertices.
        mantissa : int , optional
            The desired length of the mantissa for retrieving vertex coordinates. Default is 6.
        tolerance : float , optional
            The desired tolerance for computing if vertices need to be fused. Any vertices that are closer to each other than this tolerance will be fused. Default is 0.0001.

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
    def IncomingEdges(vertex, hostTopology, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the incoming edges connected to a vertex. An edge is incoming if
        its end vertex is coincident with the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        hostTopology : topologic_core.Topology
            The input host topology to which the vertex belongs.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of incoming edges.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.IncomingEdges - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            if not silent:
                print("Vertex.IncomingEdges - Error: The input hostTopology parameter is not a valid topology. Returning None.")
            return None

        edges = Topology.SuperTopologies(vertex, hostTopology=hostTopology, topologyType="Edge") or []
        return [
            edge for edge in edges
            if Vertex.IsCoincident(vertex, Edge.EndVertex(edge, silent=True), tolerance=tolerance, silent=True)
        ]
    
    @staticmethod
    def Index(vertex, vertices: list, strict: bool = False, tolerance: float = 0.0001) -> int:
        """
        Returns the index of the input vertex in the input list of vertices.

        This implementation avoids rebuilding the input list and uses a fast
        coordinate pre-check before falling back to Topology.IsSame or
        Vertex.IsCoincident.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        vertices : list
            The input list of vertices.
        strict : bool , optional
            If set to True, the vertex must be strictly identical to the one found
            in the list. Otherwise, a distance comparison is used. Default is False.
        tolerance : float , optional
            The tolerance for computing if the input vertex is identical to a vertex
            from the list. Default is 0.0001.

        Returns
        -------
        int
            The index of the input vertex in the input list of vertices.
        """

        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            return None

        if not isinstance(vertices, list) or len(vertices) == 0:
            return None

        try:
            x = vertex.X()
            y = vertex.Y()
            z = vertex.Z()
        except Exception:
            return None

        tol = abs(tolerance) if isinstance(tolerance, (int, float)) else 0.0001
        tol2 = tol * tol

        for i, v in enumerate(vertices):
            if v is None:
                continue

            # Fast coordinate path first. This is much cheaper than Vertex.Distance.
            try:
                dx = x - v.X()
                if abs(dx) > tol:
                    continue

                dy = y - v.Y()
                if abs(dy) > tol:
                    continue

                dz = z - v.Z()
                if abs(dz) > tol:
                    continue

                if strict:
                    if Topology.IsSame(vertex, v):
                        return i
                else:
                    if (dx * dx + dy * dy + dz * dz) <= tol2:
                        return i

            except Exception:
                # Fallback for unusual vertex objects.
                try:
                    if not Topology.IsInstance(v, "Vertex"):
                        continue

                    if strict:
                        if Topology.IsSame(vertex, v):
                            return i
                    else:
                        if Vertex.IsCoincident(vertex, v, tolerance=tol, silent=True):
                            return i
                except Exception:
                    continue

        return None

    @staticmethod
    def InterpolateValue(vertex, vertices: list, n: int = 3, key: str = "intensity", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Interpolates the value of the input vertex based on the values of the n nearest vertices.

        The input vertex and the vertices in the input list can be either:
        - topologic_core.Vertex objects, or
        - TGraph vertex records of the form:
        {"index": ..., "dictionary": {"x": ..., "y": ..., "z": ..., key: ...}, ...}

        Parameters
        ----------
        vertex : topologic_core.Vertex or dict
            The input vertex. This can be a Topologic vertex or a TGraph vertex record.
        vertices : list
            The input list of vertices. The list can contain Topologic vertices, TGraph
            vertex records, or a mixture of both.
        n : int , optional
            The maximum number of nearest vertices to consider. Default is 3.
        key : str , optional
            The key that holds the value to be interpolated in the dictionaries of the
            vertices. Default is "intensity".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The tolerance for computing if the input vertex is coincident with another
            vertex in the input list. Default is 0.0001.

        Returns
        -------
        topologic_core.Vertex or dict or None
            The input vertex with the interpolated value stored in its dictionary at
            the specified key. Other keys and values are preserved. If the input vertex
            is a TGraph vertex record, the record is updated in place and returned.
        """

        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        import numbers

        def _is_topologic_vertex(v):
            try:
                return Topology.IsInstance(v, "Vertex")
            except Exception:
                return False

        def _is_tgraph_vertex_record(v):
            if not isinstance(v, dict):
                return False
            if "dictionary" in v and isinstance(v.get("dictionary"), dict):
                d = v.get("dictionary", {})
                return any(k in d for k in ["x", "y", "z"]) or "representation" in v
            return all(k in v for k in ["x", "y", "z"])

        def _unwrap(value):
            if isinstance(value, list) and len(value) == 1:
                return value[0]
            return value

        def _numeric(value, default=None):
            value = _unwrap(value)
            if isinstance(value, numbers.Number):
                return float(value)
            try:
                return float(value)
            except Exception:
                return default

        def _coordinates(v):
            if _is_topologic_vertex(v):
                try:
                    return (
                        Vertex.X(v, mantissa=mantissa),
                        Vertex.Y(v, mantissa=mantissa),
                        Vertex.Z(v, mantissa=mantissa),
                    )
                except Exception:
                    return None

            if isinstance(v, dict):
                d = v.get("dictionary", v)
                if isinstance(d, dict):
                    x = _numeric(d.get("x", None), None)
                    y = _numeric(d.get("y", None), None)
                    z = _numeric(d.get("z", None), None)

                    if x is not None and y is not None and z is not None:
                        if mantissa is not None and mantissa >= 0:
                            return (round(x, mantissa), round(y, mantissa), round(z, mantissa))
                        return (x, y, z)

                # Fallback: a TGraph vertex record may carry a Topologic vertex as its representation.
                rep = v.get("representation", None)
                if _is_topologic_vertex(rep):
                    try:
                        return (
                            Vertex.X(rep, mantissa=mantissa),
                            Vertex.Y(rep, mantissa=mantissa),
                            Vertex.Z(rep, mantissa=mantissa),
                        )
                    except Exception:
                        return None

            return None

        def _dictionary(v):
            if _is_topologic_vertex(v):
                try:
                    return Topology.Dictionary(v)
                except Exception:
                    return None

            if isinstance(v, dict):
                d = v.get("dictionary", None)
                if isinstance(d, dict):
                    return d

            return None

        def _value_at_key(v, k):
            d = _dictionary(v)

            if d is None:
                return None

            if isinstance(d, dict):
                return d.get(k, None)

            try:
                return Dictionary.ValueAtKey(d, k, None)
            except TypeError:
                try:
                    return Dictionary.ValueAtKey(d, k)
                except Exception:
                    return None
            except Exception:
                return None

        def _set_value_at_key(v, k, value):
            if _is_topologic_vertex(v):
                try:
                    d = Topology.Dictionary(v)
                    if d is None:
                        d = Dictionary.ByKeysValues([k], [value])
                    else:
                        d = Dictionary.SetValueAtKey(d, k, value)
                    return Topology.SetDictionary(v, d)
                except Exception:
                    return None

            if isinstance(v, dict):
                d = v.setdefault("dictionary", {})
                if isinstance(d, dict):
                    d[k] = value
                    return v

            return None

        def _distance(point1, point2):
            return (
                (point1[0] - point2[0]) ** 2 +
                (point1[1] - point2[1]) ** 2 +
                (point1[2] - point2[2]) ** 2
            ) ** 0.5

        def _interpolate_value(point, data_points, n, tolerance=0.0001):
            distances = [(_distance(p[:3], point), p[3]) for p in data_points]
            sorted_distances = sorted(distances, key=lambda x: x[0])
            nearest_points = sorted_distances[:n]

            nearest_distance, nearest_value = nearest_points[0]

            if nearest_distance <= tolerance:
                return nearest_value

            weights = [(1.0 / d, value) for d, value in nearest_points if d > tolerance]

            if not weights:
                return nearest_value

            total_weight = sum(w[0] for w in weights)

            if total_weight <= 0:
                return nearest_value

            interpolated_value = sum((w / total_weight) * value for w, value in weights)

            return interpolated_value

        if not (_is_topologic_vertex(vertex) or _is_tgraph_vertex_record(vertex)):
            return None

        if not isinstance(vertices, list):
            return None

        point = _coordinates(vertex)

        if point is None:
            return None

        data_points = []

        for v in vertices:
            if not (_is_topologic_vertex(v) or _is_tgraph_vertex_record(v)):
                continue

            coords = _coordinates(v)

            if coords is None:
                continue

            value = _numeric(_value_at_key(v, key), None)

            if value is None:
                continue

            data_points.append((coords[0], coords[1], coords[2], value))

        if len(data_points) == 0:
            return None

        try:
            n = int(n)
        except Exception:
            n = 3

        n = max(1, min(n, len(data_points)))

        value = _interpolate_value(point, data_points, n, tolerance=tolerance)

        if mantissa is not None and mantissa >= 0:
            try:
                value = round(float(value), mantissa)
            except Exception:
                pass

        return _set_value_at_key(vertex, key, value)

    @staticmethod
    def IsCoincident(vertexA, vertexB, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the two input vertices are coincident within tolerance.

        Parameters
        ----------
        vertexA : topologic_core.Vertex
            The first input vertex.
        vertexB : topologic_core.Vertex
            The second input vertex.
        tolerance : float , optional
            The desired coincidence tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the input vertices are coincident. False otherwise.
        """
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Vertex.IsCoincident - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Vertex.IsCoincident - Error: The input vertexB parameter is not a valid vertex. Returning None.")
            return None

        tol = abs(float(tolerance))
        if Core.HasAttribute("VertexUtility", "IsCoincident"):
            try:
                return bool(Core.VertexUtility.IsCoincident(vertexA, vertexB, tol))
            except Exception:
                pass

        try:
            if Core.HasAttribute("VertexUtility", "Distance"):
                distance = Core.VertexUtility.Distance(vertexA, vertexB)
                if distance is not None:
                    return float(distance) <= tol
        except Exception:
            pass

        a = Vertex.Coordinates(vertexA, mantissa=None)
        b = Vertex.Coordinates(vertexB, mantissa=None)
        if a is None or b is None:
            return False
        distance = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
        return distance <= tol

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
            The tolerance for computing if the input vertex is external to the input topology. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def IsInternal(
        vertex,
        topology,
        maxLeafSize: int = 4,
        identify: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns True if the input vertex lies inside or on the represented
        geometry of the input topology according to TopologicPy containment
        semantics. Primitive geometric classification is delegated to the active
        backend whenever an appropriate native operation is available.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        maxLeafSize : int , optional
            Retained for backward compatibility. No temporary BVH is constructed
            for a single-vertex query. Default is 4.
        identify : bool , optional
            If set to True, returns ``(status, subTopology)`` and identifies the
            lowest-dimensional matching constituent when applicable. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool or tuple
            True/False, or ``(True/False, topology)`` when ``identify`` is True.
        """
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        def _return(status, item=None):
            return (bool(status), item) if identify else bool(status)

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.IsInternal - Error: The input vertex parameter is not a valid vertex. Returning False.")
            return _return(False, None)
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Vertex.IsInternal - Error: The input topology parameter is not a valid topology. Returning False.")
            return _return(False, None)

        tol = abs(float(tolerance))

        # Preserve the established TopologicCore pathway exactly. The native
        # classification below is enabled only for non-TopologicCore backends.
        try:
            if Topology._IsTopologicCoreBackend():
                return Vertex._IsInternalTopologicCore(
                    vertex,
                    topology,
                    maxLeafSize=maxLeafSize,
                    identify=identify,
                    tolerance=tol,
                    silent=silent,
                )
        except Exception:
            pass

        def _backend_internal(candidate):
            if not Core.HasAttribute("VertexUtility", "DistanceToTopology"):
                return None
            if not Core.HasAttribute("VertexUtility", "IsInternal"):
                return None
            try:
                return bool(Core.VertexUtility.IsInternal(vertex, candidate, tol))
            except Exception:
                return None

        def _point_in_vertex(candidate):
            return bool(Vertex.IsCoincident(vertex, candidate, tolerance=tol, silent=True))

        def _point_in_edge(candidate):
            status = _backend_internal(candidate)
            if status is not None:
                return status
            distance = Vertex.Distance(vertex, candidate, includeCentroid=False, mantissa=None, tolerance=tol, silent=True)
            return distance is not None and distance <= tol

        def _point_in_face(candidate):
            status = _backend_internal(candidate)
            if status is not None:
                return status
            try:
                if Core.HasAttribute("FaceUtility", "IsInside"):
                    return bool(Core.FaceUtility.IsInside(candidate, vertex, tol))
            except Exception:
                pass
            distance = Vertex.Distance(vertex, candidate, includeCentroid=False, mantissa=None, tolerance=tol, silent=True)
            return distance is not None and distance <= tol

        def _point_in_cell(candidate):
            # Keep the established Cell.ContainmentStatus policy because it
            # provides the TopologicCore tolerance workaround and preserves
            # cross-backend behaviour.
            try:
                return Cell.ContainmentStatus(candidate, vertex, tolerance=tol) == 0
            except TypeError:
                try:
                    return Cell.ContainmentStatus(candidate, vertex) == 0
                except Exception:
                    return False
            except Exception:
                return False

        # Fast primitive paths.
        if Topology.IsInstance(topology, "Vertex"):
            status = _point_in_vertex(topology)
            return _return(status, topology if status else None)
        if Topology.IsInstance(topology, "Edge"):
            status = _point_in_edge(topology)
            return _return(status, topology if status else None)
        if Topology.IsInstance(topology, "Face"):
            status = _point_in_face(topology)
            return _return(status, topology if status else None)
        if Topology.IsInstance(topology, "Cell"):
            status = _point_in_cell(topology)
            return _return(status, topology if status else None)
        if Topology.IsInstance(topology, "Wire") or Topology.IsInstance(topology, "Shell"):
            status = _backend_internal(topology)
            if status is None:
                status = Vertex.Distance(vertex, topology, includeCentroid=False, mantissa=None, tolerance=tol, silent=True)
                status = status is not None and status <= tol
            return _return(status, topology if status else None)

        def _subtopologies(kind):
            try:
                method = getattr(Topology, kind)
                return method(topology, silent=True) or []
            except TypeError:
                try:
                    return method(topology) or []
                except Exception:
                    return []
            except Exception:
                return []

        # Cluster semantics consider all represented dimensionalities. Other
        # aggregate topology types consider their highest-dimensional direct
        # geometric constituents, preserving the current TopologicPy policy.
        if Topology.IsInstance(topology, "Cluster"):
            vertices = _subtopologies("Vertices")
            edges = _subtopologies("Edges")
            faces = _subtopologies("Faces")
            cells = _subtopologies("Cells")
        else:
            cells = _subtopologies("Cells")
            if cells:
                vertices, edges, faces = [], [], []
            else:
                faces = _subtopologies("Faces")
                if faces:
                    vertices, edges = [], []
                else:
                    edges = _subtopologies("Edges")
                    vertices = [] if edges else _subtopologies("Vertices")

        # Lower-dimensional matches have priority when identify=True.
        for candidate in vertices:
            if _point_in_vertex(candidate):
                return _return(True, candidate)
        for candidate in edges:
            if _point_in_edge(candidate):
                return _return(True, candidate)
        for candidate in faces:
            if _point_in_face(candidate):
                return _return(True, candidate)
        for candidate in cells:
            if _point_in_cell(candidate):
                return _return(True, candidate)

        return _return(False, None)






    @staticmethod
    def _IsInternalTopologicCore(
        vertex,
        topology,
        maxLeafSize: int = 4,
        identify: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Preserves the current TopologicCore-specific Vertex.IsInternal implementation.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        maxLeafSize: int , optional
            Retained for backward compatibility. This implementation avoids building
            a BVH for every call because that is expensive for single-point queries.
            Default is 4.
        identify: bool, optional
            If set to True, a tuple is returned where the identified subTopology is
            returned (e.g. (True, edge)). Default is False.
        tolerance : float, optional
            The desired tolerance. Default 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool or tuple
            True/False, or (True/False, topology) if identify is True.
        """

        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell

        def _return(status, item=None):
            if identify:
                return (status, item)
            return status

        def _warn(message):
            if not silent:
                print("Vertex.IsInternal - Warning:", message)

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.IsInternal - Error: The input vertex is not a valid vertex. Returning False.")
            return _return(False, None)

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Vertex.IsInternal - Error: The input topology is not a valid topology. Returning False.")
            return _return(False, None)

        try:
            vertex_coords = Vertex.Coordinates(vertex)
        except Exception:
            vertex_coords = None

        if vertex_coords is None:
            return _return(False, None)

        # ------------------------------------------------------------------
        # Small vector helpers
        # ------------------------------------------------------------------

        def _dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def _cross(a, b):
            return (
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            )

        def _sub(a, b):
            return (
                a[0] - b[0],
                a[1] - b[1],
                a[2] - b[2],
            )

        def _length(a):
            return (_dot(a, a))**0.5

        def _dominant_axis(n):
            ax = abs(n[0])
            ay = abs(n[1])
            az = abs(n[2])

            if ax >= ay and ax >= az:
                return 0
            if ay >= ax and ay >= az:
                return 1
            return 2

        def _project_2d(p, drop_axis):
            if drop_axis == 0:
                return (p[1], p[2])
            if drop_axis == 1:
                return (p[0], p[2])
            return (p[0], p[1])

        def _ring_area_2d(ring):
            if not ring or len(ring) < 3:
                return 0.0

            area = 0.0
            n = len(ring)

            for i in range(n):
                x1, y1 = ring[i]
                x2, y2 = ring[(i + 1) % n]
                area += x1*y2 - x2*y1

            return 0.5 * area

        def _point_on_segment_2d(p, a, b, tol):
            px, py = p
            ax, ay = a
            bx, by = b

            minx = min(ax, bx) - tol
            maxx = max(ax, bx) + tol
            miny = min(ay, by) - tol
            maxy = max(ay, by) + tol

            if px < minx or px > maxx or py < miny or py > maxy:
                return False

            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay

            cross = apx*aby - apy*abx
            if abs(cross) > tol:
                return False

            return True

        def _point_in_ring_2d(point, ring, tol):
            if not ring or len(ring) < 3:
                return False

            x, y = point
            inside = False
            n = len(ring)

            for i in range(n):
                a = ring[i]
                b = ring[(i + 1) % n]

                if _point_on_segment_2d(point, a, b, tol):
                    return True

                x1, y1 = a
                x2, y2 = b

                if (y1 > y) != (y2 > y):
                    xinters = ((x2 - x1) * (y - y1) / ((y2 - y1) + 1e-300)) + x1
                    if x <= xinters + tol:
                        inside = not inside

            return inside

        def _topology_vertices(topo):
            try:
                return Topology.Vertices(topo, silent=True) or []
            except TypeError:
                try:
                    return Topology.Vertices(topo) or []
                except Exception:
                    return []
            except Exception:
                return []

        def _topology_edges(topo):
            try:
                return Topology.Edges(topo, silent=True) or []
            except TypeError:
                try:
                    return Topology.Edges(topo) or []
                except Exception:
                    return []
            except Exception:
                return []

        def _topology_faces(topo):
            try:
                return Topology.Faces(topo, silent=True) or []
            except TypeError:
                try:
                    return Topology.Faces(topo) or []
                except Exception:
                    return []
            except Exception:
                return []

        def _topology_cells(topo):
            try:
                return Topology.Cells(topo, silent=True) or []
            except TypeError:
                try:
                    return Topology.Cells(topo) or []
                except Exception:
                    return []
            except Exception:
                return []

        # ------------------------------------------------------------------
        # Primitive containment tests
        # ------------------------------------------------------------------

        def _point_in_vertex(vtx, other_vertex):
            try:
                return Vertex.Distance(vtx, other_vertex) <= tolerance
            except Exception:
                return False

        def _point_in_edge(vtx, edge):
            try:
                return Vertex.Distance(vtx, edge) <= tolerance
            except Exception:
                return False

        def _point_in_cell(vtx, cell):
            try:
                return Cell.ContainmentStatus(cell, vtx, tolerance=tolerance) == 0
            except TypeError:
                try:
                    return Cell.ContainmentStatus(cell, vtx) == 0
                except Exception:
                    return False
            except Exception:
                return False

        def _face_rings(face):
            wires = []

            try:
                external_boundary = Face.ExternalBoundary(face)
                if external_boundary is not None:
                    wires.append(external_boundary)
            except Exception:
                pass

            try:
                internal_boundaries = Face.InternalBoundaries(face)
                if internal_boundaries:
                    wires.extend([w for w in internal_boundaries if w is not None])
            except Exception:
                pass

            if not wires:
                try:
                    wires = Topology.Wires(face, silent=True) or []
                except TypeError:
                    try:
                        wires = Topology.Wires(face) or []
                    except Exception:
                        wires = []
                except Exception:
                    wires = []

            rings = []

            for wire in wires:
                verts = _topology_vertices(wire)
                if len(verts) < 3:
                    continue

                coords = []

                for v in verts:
                    try:
                        c = Vertex.Coordinates(v)
                        if c is not None:
                            coords.append(c)
                    except Exception:
                        continue

                if len(coords) >= 3:
                    rings.append(coords)

            return rings

        def _point_in_face_fast(vtx, face):
            # First reject non-coplanar points.
            try:
                if Vertex.PerpendicularDistance(vtx, face) > tolerance:
                    return False
            except Exception:
                pass

            try:
                projected_vertex = Vertex.Project(vtx, face)
                if projected_vertex is not None:
                    point_3d = Vertex.Coordinates(projected_vertex)
                else:
                    point_3d = Vertex.Coordinates(vtx)
            except Exception:
                point_3d = Vertex.Coordinates(vtx)

            if point_3d is None:
                return False

            try:
                normal = Face.Normal(face)
            except Exception:
                normal = None

            if normal is None or len(normal) < 3 or _length(normal) <= tolerance:
                return _point_in_face_fallback(vtx, face)

            drop_axis = _dominant_axis(normal)
            point_2d = _project_2d(point_3d, drop_axis)

            rings_3d = _face_rings(face)
            if not rings_3d:
                return _point_in_face_fallback(vtx, face)

            rings_2d = []

            for ring_3d in rings_3d:
                ring_2d = [_project_2d(p, drop_axis) for p in ring_3d]
                area = abs(_ring_area_2d(ring_2d))

                if area > tolerance * tolerance:
                    rings_2d.append((area, ring_2d))

            if not rings_2d:
                return _point_in_face_fallback(vtx, face)

            # Largest ring is treated as the external boundary.
            rings_2d.sort(key=lambda item: item[0], reverse=True)

            outer = rings_2d[0][1]
            holes = [item[1] for item in rings_2d[1:]]

            if not _point_in_ring_2d(point_2d, outer, tolerance):
                return False

            for hole in holes:
                if _point_in_ring_2d(point_2d, hole, tolerance):
                    return False

            return True

        def _point_in_face_fallback(vtx, face):
            # Original transform-based method retained as a safety fallback.
            try:
                from topologicpy.Vector import Vector

                v = Vertex.ByCoordinates(Vertex.Coordinates(vtx))

                if Vertex.PerpendicularDistance(v, face) > tolerance:
                    return False

                v = Vertex.Project(v, face)
                centroid = Topology.Centroid(face)

                x_tran = -Vertex.X(centroid)
                y_tran = -Vertex.Y(centroid)
                z_tran = -Vertex.Z(centroid)

                face_2 = Topology.Translate(face, x_tran, y_tran, z_tran)
                vertex_2 = Topology.Translate(v, x_tran, y_tran, z_tran)

                face_normal = Face.Normal(face_2)
                up = [0, 0, 1]
                tran_mat = Vector.TransformationMatrix(face_normal, up)

                flat_face = Topology.Transform(face_2, tran_mat, transferDictionaries=False)
                flat_vertex = Topology.Transform(vertex_2, tran_mat)
                flat_vertex = Topology.Translate(flat_vertex, 0, 0, -Vertex.Z(flat_vertex))

                return Vertex.IsInternal2D(flat_vertex, flat_face)
            except Exception:
                return False

        # ------------------------------------------------------------------
        # Fast direct paths.
        # These avoid AABB construction, primitive collection, BVH construction,
        # BVH querying, and candidate sorting for simple topologies.
        # ------------------------------------------------------------------

        if Topology.IsInstance(topology, "Vertex"):
            return _return(_point_in_vertex(vertex, topology), topology if _point_in_vertex(vertex, topology) else None)

        if Topology.IsInstance(topology, "Edge"):
            status = _point_in_edge(vertex, topology)
            return _return(status, topology if status else None)

        if Topology.IsInstance(topology, "Face"):
            status = _point_in_face_fast(vertex, topology)
            return _return(status, topology if status else None)

        if Topology.IsInstance(topology, "Cell"):
            status = _point_in_cell(vertex, topology)
            return _return(status, topology if status else None)

        # ------------------------------------------------------------------
        # Composite topology path.
        # Avoid building a BVH per call. For a single vertex query, direct
        # iteration is usually faster than constructing a temporary BVH.
        # ------------------------------------------------------------------

        if Topology.IsInstance(topology, "Cluster"):
            vertices = _topology_vertices(topology)
            edges = _topology_edges(topology)
            faces = _topology_faces(topology)
            cells = _topology_cells(topology)
        else:
            cells = _topology_cells(topology)

            if cells:
                vertices = []
                edges = []
                faces = []
            else:
                faces = _topology_faces(topology)

                if faces:
                    vertices = []
                    edges = []
                else:
                    edges = _topology_edges(topology)

                    if edges:
                        vertices = []
                    else:
                        vertices = _topology_vertices(topology)

        if not vertices and not edges and not faces and not cells:
            return _return(False, None)

        # Optional coarse AABB rejection for composite topologies only.
        # This is much cheaper than building a BVH and helps reject obvious misses.
        all_vertices = []

        if vertices:
            all_vertices = vertices
        else:
            all_vertices = _topology_vertices(topology)

        if all_vertices:
            try:
                xs = []
                ys = []
                zs = []

                for v in all_vertices:
                    c = Vertex.Coordinates(v)
                    if c is None:
                        continue
                    xs.append(c[0])
                    ys.append(c[1])
                    zs.append(c[2])

                if xs and ys and zs:
                    x, y, z = vertex_coords

                    if (
                        x < min(xs) - tolerance or x > max(xs) + tolerance or
                        y < min(ys) - tolerance or y > max(ys) + tolerance or
                        z < min(zs) - tolerance or z > max(zs) + tolerance
                    ):
                        return _return(False, None)
            except Exception:
                pass

        # Priority: vertices, edges, faces, cells.
        # This preserves the intent of the original sorted candidate loop without
        # calling Helper.Sort or repeatedly querying Topology.Type.
        for candidate in vertices:
            if _point_in_vertex(vertex, candidate):
                return _return(True, candidate)

        for candidate in edges:
            if _point_in_edge(vertex, candidate):
                return _return(True, candidate)

        for candidate in faces:
            if _point_in_face_fast(vertex, candidate):
                return _return(True, candidate)

        for candidate in cells:
            if _point_in_cell(vertex, candidate):
                return _return(True, candidate)

        return _return(False, None)

    @staticmethod
    def IsInternal_old(
        vertex,
        topology,
        maxLeafSize: int = 4,
        identify: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns True if the input vertex lies inside the input topology.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        maxLeafSize: int , optional
            The maximum number of primitives (topologies) that can be stored in a single leaf node of the BVH.
            Smaller values result in deeper trees with finer spatial subdivision (potentially faster queries but slower build times),
            while larger values produce shallower trees with coarser spatial grouping (faster builds but less precise queries).
            Default is 4.
        identify: bool, optional
            If set to True, a tuple is returned where the identified subTopology is returned (e.g. (True, edge)). Default is False.
        tolerance : float, optional
            The desired tolerance. Default 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
        """
        # --- Local imports (TopologicPy) ---
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Shell import Shell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Wire import Wire
        from topologicpy.Edge import Edge
        from topologicpy.BVH import BVH
        from topologicpy.BVH import AABB
        from topologicpy.Helper import Helper

        # --------------------------
        # Utilities
        # --------------------------

        def vec_dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def vec_cross(a, b):
            return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

        def vec_len(a):
            return (a[0]*a[0]+a[1]*a[1]+a[2]*a[2])**0.5

        def vec_norm(a):
            l = vec_len(a)
            if l == 0:
                return (0.0, 0.0, 0.0)
            return (a[0]/l, a[1]/l, a[2]/l)

        def dominant_axis(n):
            # Return axis to drop when projecting to 2D (index 0=x,1=y,2=z)
            ax = abs(n[0]); ay = abs(n[1]); az = abs(n[2])
            if ax >= ay and ax >= az:
                return 0
            if ay >= ax and ay >= az:
                return 1
            return 2

        def project_point(p, drop_axis):
            if drop_axis == 0:
                return (p[1], p[2])
            elif drop_axis == 1:
                return (p[0], p[2])
            else:
                return (p[0], p[1])


        # 2D point-in-polygon (ray crossing). Polygon is list of 2D points (closed or open).
        def pip_ray_cross_2d(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            if n < 3:
                return False
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i+1) % n]
                # Check if point is on edge (within tolerance)
                # Project distance to segment
                # (cheap check first)
                minx = min(x1, x2) - 1e-15
                maxx = max(x1, x2) + 1e-15
                miny = min(y1, y2) - 1e-15
                maxy = max(y1, y2) + 1e-15
                if minx <= x <= maxx and miny <= y <= maxy:
                    # Cross product close to zero?
                    dx1, dy1 = x - x1, y - y1
                    dx2, dy2 = x2 - x1, y2 - y1
                    cross = dx1 * dy2 - dy1 * dx2
                    if abs(cross) <= 1e-12:
                        return True  # on boundary
                # Ray crossing
                cond1 = (y1 > y) != (y2 > y)
                if cond1:
                    xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
                    if x <= xinters + 1e-15:
                        inside = not inside
            return inside

        def polygon_with_holes_contains_2d(pt, outer, holes):
            if not pip_ray_cross_2d(pt, outer):
                return False
            for hole in holes:
                if pip_ray_cross_2d(pt, hole):
                    return False
            return True

        # 2D containment in an Edge
        def point_in_vertex(vtx, vertex, tol):
            # Boundary snap first
            if Vertex.Distance(vtx, vertex) <= tol:
                return True
            return False
        
        # 2D containment in an Edge
        def point_in_edge(vtx, edge, tol):
            # Boundary snap first
            if Vertex.Distance(vtx, edge) <= tol:
                return True
            return False
        
        # 2D containment in a Face (vertex assumed coplanar or nearly so)
        def point_in_face(vtx, face, tol):
            from topologicpy.Vector import Vector
            v = Vertex.ByCoordinates(Vertex.Coordinates(vtx))
            if Vertex.PerpendicularDistance(v, face) > tol:
                return False
            else:
                v = Vertex.Project(v, face)
            centroid = Topology.Centroid(face)
            x_tran = -Vertex.X(centroid)
            y_tran = -Vertex.Y(centroid)
            z_tran = -Vertex.Z(centroid)
            face_2 = Topology.Translate(face, x_tran, y_tran, z_tran)
            vertex_2 = Topology.Translate(v, x_tran, y_tran, z_tran)

            face_normal = Face.Normal(face_2)
            up = [0,0,1]
            tran_mat = Vector.TransformationMatrix(face_normal, up)
            flat_face = Topology.Transform(face_2, tran_mat, transferDictionaries=False)
            flat_vertex = Topology.Transform(vertex_2, tran_mat)
            flat_vertex = Topology.Translate(flat_vertex, 0, 0, -Vertex.Z(flat_vertex))
            return Vertex.IsInternal2D(flat_vertex, flat_face)
            # from topologicpy.Vector import Vector
            # face_normal = Face.Normal(face)
            # up = [0,0,1]
            # tran_mat = Vector.TransformationMatrix(face_normal, up)
            # flat_face = Topology.Transform(face, tran_mat)
            # flat_vertex = Topology.Transform(vtx, tran_mat)
            # dist = Vertex.PerpendicularDistance(vtx, face)
            # if dist <= tol:
            #     vtx2 = Vertex.Project(vtx, face)
            #     status = Core.FaceUtility.IsInside(face, vtx2, tol)
            # else:
            #     status = False
            # return status

        # 3D containment in a Cell via ray casting (+X direction)
        def point_in_cell(vtx, cell, tol):
            status = Cell.ContainmentStatus(cell, vtx, tolerance = tol)
            return status == 0

        # --------------------------
        # Check if inside AABB
        # --------------------------
        points = [Vertex.Coordinates(v) for v in Topology.Vertices(topology)]
        aabb = AABB.from_points(points, pad=tolerance)
        if(Vertex.Coordinates(vertex) is None): 
            if identify:
                return (False, None)
            return False
        if not aabb.contains_point(Vertex.Coordinates(vertex)):
            if identify:
                return (False, None)
            return False

        # --------------------------
        # Collect primitives
        # --------------------------
        def collect_cells(topo):
            if Topology.IsInstance(topo, "cell"):
                return [topo]
            else:
                return Topology.Cells(topo, silent=True)

        def collect_faces(topo):
            if Topology.IsInstance(topo, "face"):
                return [topo]
            else:
                return Topology.Faces(topo, silent=True)

        def collect_edges(topo):
            if Topology.IsInstance(topo, "edge"):
                return [topo]
            else:
                return Topology.Edges(topo, silent=True)
        def collect_vertices(topo):
            if Topology.IsInstance(topo, "vertex"):
                return [topo]
            else:
                return Topology.Vertices(topo, silent=True)

        if Topology.IsInstance(topology, "cluster"):
            cells = collect_cells(topology)
            faces = collect_faces(topology)
            edges = collect_edges(topology)
            vertices = collect_vertices(topology)
        else:
            cells = collect_cells(topology)
            faces = [] if cells else collect_faces(topology)
            edges = [] if faces or cells else collect_edges(topology)
            vertices = [] if edges or faces or cells else collect_vertices(topology)
        if not cells and not faces and not edges and not vertices:
            if identify:
                return (False, None)
            return False

        # --------------------------
        # Build BVH and fetch candidates
        # --------------------------
        primitives = []
        primitives.extend(vertices)
        primitives.extend(edges)
        primitives.extend(faces)
        primitives.extend(cells)
        bvh = BVH.ByTopologies(primitives, maxLeafSize=maxLeafSize, tolerance=tolerance, silent=True)
        try:
            candidates = BVH.Clashes(bvh, vertex, tolerance=tolerance) or []
        except Exception:
            # Fallback if your BVH needs a non-degenerate query
            candidates = primitives

        if not candidates:
            if identify:
                return (False, None)
            return False

        # sort by types so that priority is given to lower dimensional types (e.g. vertices, then edges, then faces, then cells)
        types = [Topology.Type(c) for c in candidates]
        candidates = Helper.Sort(candidates, types)
        # --------------------------
        # Narrow phase
        # --------------------------
        for c in candidates:
            if Topology.IsInstance(c, "cell"):
                # Exact geometric test
                try:
                    if point_in_cell(vertex, c, tolerance):
                        if identify:
                            return (True, c)
                        else:
                            return True
                except Exception:
                    if not silent:
                        print("Warning: point_in_cell failed on a candidate.")
            elif Topology.IsInstance(c, "face"):
                try:
                    if point_in_face(vertex, c, tolerance):
                        if identify:
                            return (True, c)
                        else:
                            return True
                        return True
                except Exception:
                    if not silent:
                        print("Warning: point_in_face failed on a candidate.")
            elif Topology.IsInstance(c, "edge"):
                try:
                    if point_in_edge(vertex, c, tolerance):
                        if identify:
                            return (True, c)
                        else:
                            return True
                except Exception:
                    if not silent:
                        print("Warning: point_in_edge failed on a candidate.")
            elif Topology.IsInstance(c, "vertex"):
                try:
                    if point_in_vertex(vertex, c, tolerance):
                        if identify:
                            return (True, c)
                        else:
                            return True
                except Exception:
                    if not silent:
                        print("Warning: point_in_vertex failed on a candidate.")
        if identify:
            return (False, None)
        return False
    
    @staticmethod
    def IsInternal2D(vertices, face, includeBoundary: bool = True,
                    mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Fast, batch point-in-face test (supports holes) using NumPy vectorized ray casting.

        Parameters
        ----------
        face : topologic_core.Face
            Input face (may have holes). Assumes planar and evaluated in XY.
        vertices : topologic_core.Vertex or list[topologic_core.Vertex]
            Query vertex/vertices.
        includeBoundary : bool, optional
            If True, points on the *outer* boundary are counted as inside.
            Points on hole boundaries are always treated as outside. Default is True.
        mantissa : int, optional
            Rounding precision for XY conversion. Default is 6.
        tolerance : float, optional
            The desired tolerance. Default 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool or list[bool]
            If a single vertex is supplied, returns a bool.
            If a list is supplied, returns a list of bools of the same length.
        """
        import numpy as np
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Wire import Wire

        # -----------------------------
        # Local helpers
        # -----------------------------
        def _as_list(vs):
            if vs is None:
                return []
            if isinstance(vs, (list, tuple)):
                return list(vs)
            return [vs]

        def _to_xy(vs):
            # (M,2) float64
            return np.array(
                [[round(Vertex.X(v), mantissa), round(Vertex.Y(v), mantissa)] for v in vs],
                dtype=np.float64
            )

        def _wire_xy(wire):
            wv = Wire.Vertices(wire) or []
            return _to_xy(wv)

        def _points_in_polygon(P, V):
            """
            Vectorized ray casting (crossing number). P: (M,2), V: (N,2) open ring.
            Returns (M,) bool.
            """
            P = np.asarray(P, dtype=np.float64)
            V = np.asarray(V, dtype=np.float64)
            m = P.shape[0]
            n = V.shape[0]
            if m == 0 or n < 3:
                return np.zeros((m,), dtype=bool)

            x = P[:, 0]
            y = P[:, 1]

            x0 = V[:, 0]
            y0 = V[:, 1]
            x1 = np.roll(x0, -1)
            y1 = np.roll(y0, -1)

            # bbox reject
            minx, maxx = x0.min(), x0.max()
            miny, maxy = y0.min(), y0.max()
            cand = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
            idx = np.nonzero(cand)[0]
            if idx.size == 0:
                return np.zeros((m,), dtype=bool)

            xx = x[idx][:, None]  # (K,1)
            yy = y[idx][:, None]  # (K,1)

            # straddle test
            cond = ((y0 > yy) != (y1 > yy))
            xinters = (x1 - x0) * (yy - y0) / (y1 - y0 + 1e-300) + x0
            crossings = cond & (xx < xinters)

            out = np.zeros((m,), dtype=bool)
            out[idx] = (np.count_nonzero(crossings, axis=1) & 1) == 1
            return out

        def _points_on_edges(P, V, tol):
            """
            Vectorized 'point on any segment' test.
            Loops over edges (usually modest), vectorizes over points.
            Returns (M,) bool.
            """
            P = np.asarray(P, dtype=np.float64)
            V = np.asarray(V, dtype=np.float64)
            m = P.shape[0]
            n = V.shape[0]
            if m == 0 or n < 2:
                return np.zeros((m,), dtype=bool)

            tol2 = float(tol) * float(tol)
            on = np.zeros((m,), dtype=bool)

            x = P[:, 0]
            y = P[:, 1]

            x0 = V[:, 0]
            y0 = V[:, 1]
            x1 = np.roll(x0, -1)
            y1 = np.roll(y0, -1)

            for i in range(n):
                ax, ay = x0[i], y0[i]
                bx, by = x1[i], y1[i]

                # segment bbox prune (+tol)
                minx, maxx = (ax, bx) if ax <= bx else (bx, ax)
                miny, maxy = (ay, by) if ay <= by else (by, ay)
                cand = (x >= (minx - tol)) & (x <= (maxx + tol)) & (y >= (miny - tol)) & (y <= (maxy + tol))
                if not np.any(cand):
                    continue

                dx = bx - ax
                dy = by - ay
                seg_len2 = dx*dx + dy*dy
                if seg_len2 <= 1e-300:
                    # degenerate edge: treat as point
                    ddx = x[cand] - ax
                    ddy = y[cand] - ay
                    on[cand] |= (ddx*ddx + ddy*ddy) <= tol2
                    continue

                # projection t onto segment [0,1]
                px = x[cand] - ax
                py = y[cand] - ay
                t = (px*dx + py*dy) / seg_len2
                t = np.clip(t, 0.0, 1.0)

                cx = ax + t*dx
                cy = ay + t*dy

                ddx = x[cand] - cx
                ddy = y[cand] - cy
                on[cand] |= (ddx*ddx + ddy*ddy) <= tol2

                if np.all(on):
                    break

            return on

        
        # -----------------------------
        # Inputs
        # -----------------------------
        vs = _as_list(vertices)
        if len(vs) == 0:
            return [] if isinstance(vertices, (list, tuple)) else False

        P = _to_xy(vs)

        # Face rings (XY)
        outer_wire = Face.ExternalBoundary(face)
        outer = _wire_xy(outer_wire)

        holes = []
        ib = Face.InternalBoundaries(face)
        if ib:
            for w in ib:
                holes.append(_wire_xy(w))

        # -----------------------------
        # Inside / boundary logic
        # -----------------------------
        inside_outer = _points_in_polygon(P, outer)

        if includeBoundary:
            on_outer = _points_on_edges(P, outer, tolerance)
            inside = inside_outer | on_outer
        else:
            inside = inside_outer

        if holes:
            for h in holes:
                if h.shape[0] < 3:
                    continue
                inside_h = _points_in_polygon(P, h)
                if includeBoundary:
                    on_h = _points_on_edges(P, h, tolerance)
                    # holes remove interior AND boundary
                    inside &= ~(inside_h | on_h)
                else:
                    inside &= ~inside_h

        # Return type matches input
        if isinstance(vertices, (list, tuple)):
            return inside.tolist()
        return bool(inside[0])

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
            The tolerance for computing if the input vertex is peripheral to the input topology. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            if set to True, the algorithm will use a KDTree method to search for the nearest vertex. Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        
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
        Computes the unit normal vector of the best-fit plane through a list of vertices.
        Depending on the order and numerical configuration of the vertices, the normal
        can be flipped by 180 degrees.

        Parameters
        ----------
        vertices : list
            A list of topologic vertices.
        mantissa : int , optional
            The number of decimal places to round the returned normal to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A unit normal vector ``[x, y, z]``, or None if a normal cannot be determined.
        """
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            if not silent:
                print("Vertex.Normal - Error: The vertices input parameter is not a valid list. Returning None.")
            return None

        verts = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(verts) < 3:
            if not silent:
                print("Vertex.Normal - Error: The list of vertices contains less than 3 valid topologic vertices. Returning None.")
            return None

        try:
            coords = np.asarray([Vertex.Coordinates(v, mantissa=None) for v in verts], dtype=float)
            centered = coords - coords.mean(axis=0)
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
            tol = max(abs(float(tolerance)), 1.0e-15)
            if len(singular_values) < 2 or float(singular_values[1]) <= tol:
                if not silent:
                    print("Vertex.Normal - Error: The input vertices do not define a unique plane. Returning None.")
                return None
            normal = vh[-1]
            norm = float(np.linalg.norm(normal))
            if norm <= 1.0e-15:
                return None
            normal = normal / norm
            values = normal.tolist()
            return [round(float(value), mantissa) for value in values] if mantissa is not None else [float(value) for value in values]
        except Exception:
            if not silent:
                print("Vertex.Normal - Error: Could not compute a normal. Returning None.")
            return None
    
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
    def OutgoingEdges(vertex, hostTopology, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the outgoing edges connected to a vertex. An edge is outgoing if
        its start vertex is coincident with the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        hostTopology : topologic_core.Topology
            The input host topology to which the vertex belongs.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of outgoing edges.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.OutgoingEdges - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            if not silent:
                print("Vertex.OutgoingEdges - Error: The input hostTopology parameter is not a valid topology. Returning None.")
            return None

        edges = Topology.SuperTopologies(vertex, hostTopology=hostTopology, topologyType="Edge") or []
        return [
            edge for edge in edges
            if Vertex.IsCoincident(vertex, Edge.StartVertex(edge, silent=True), tolerance=tolerance, silent=True)
        ]
    
    @staticmethod
    def PerpendicularDistance(vertex, face, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the perpendicular distance from the input vertex to the infinite
        supporting plane of the input planar face.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        face : topologic_core.Face
            The input planar face.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The perpendicular distance to the supporting plane, or None if the
            face does not define a single planar supporting surface.
        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.PerpendicularDistance - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Vertex.PerpendicularDistance - Error: The input face parameter is not a valid face. Returning None.")
            return None

        tol = abs(float(tolerance))
        if (
            Core.HasAttribute("VertexUtility", "SignedDistanceToFace")
            and Core.HasAttribute("VertexUtility", "PerpendicularDistance")
        ):
            try:
                value = Core.VertexUtility.PerpendicularDistance(vertex, face, tol)
                if value is not None:
                    return round(float(value), mantissa) if mantissa is not None else float(value)
                # A native backend returning None means the Face is not planar.
                if Core.HasAttribute("VertexUtility", "SignedDistanceToFace"):
                    if not silent:
                        print("Vertex.PerpendicularDistance - Error: The input face is not planar. Returning None.")
                    return None
            except Exception:
                pass

        # Legacy TopologicCore compatibility path.
        try:
            precision = max(12, int(mantissa) if mantissa is not None else 12)
            equation = Face.PlaneEquation(face, mantissa=precision)
        except Exception:
            equation = None
        if not isinstance(equation, dict):
            if not silent:
                print("Vertex.PerpendicularDistance - Error: Could not determine the supporting plane of the input face. Returning None.")
            return None

        try:
            a = float(equation["a"])
            b = float(equation["b"])
            c = float(equation["c"])
            d = float(equation["d"])
            x, y, z = Vertex.Coordinates(vertex, mantissa=None)
            denominator = math.sqrt(a*a + b*b + c*c)
            if denominator <= max(tol, 1.0e-15):
                return None
            value = abs(a*x + b*y + c*z + d) / denominator
            return round(float(value), mantissa) if mantissa is not None else float(value)
        except Exception:
            if not silent:
                print("Vertex.PerpendicularDistance - Error: Could not compute the perpendicular distance. Returning None.")
            return None
    
    @staticmethod
    def PlaneEquation(vertices, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the equation of the best-fit plane passing through a list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        mantissa : int , optional
            The number of decimal places to round the returned coefficients to. Default is 6.
        tolerance : float , optional
            The desired tolerance used to reject a degenerate normal. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary containing ``a``, ``b``, ``c`` and ``d`` for the plane
            equation ``ax + by + cz + d = 0``.
        """
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            if not silent:
                print("Vertex.PlaneEquation - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        verts = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(verts) < 3:
            if not silent:
                print("Vertex.PlaneEquation - Error: The input list contains less than 3 valid vertices. Returning None.")
            return None

        try:
            coords = np.asarray([Vertex.Coordinates(v, mantissa=None) for v in verts], dtype=float)
            centroid = coords.mean(axis=0)
            centered = coords - centroid
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
            tol = max(abs(float(tolerance)), 1.0e-15)
            if len(singular_values) < 2 or float(singular_values[1]) <= tol:
                if not silent:
                    print("Vertex.PlaneEquation - Error: The input vertices do not define a unique plane. Returning None.")
                return None
            normal = vh[-1]
            norm = float(np.linalg.norm(normal))
            if norm <= 1.0e-15:
                return None
            normal = normal / norm
            d = -float(np.dot(normal, centroid))
            values = [float(normal[0]), float(normal[1]), float(normal[2]), d]
            if mantissa is not None:
                values = [round(value, mantissa) for value in values]
            return {"a": values[0], "b": values[1], "c": values[2], "d": values[3]}
        except Exception:
            if not silent:
                print("Vertex.PlaneEquation - Error: Could not determine the best-fit plane. Returning None.")
            return None
    
    @staticmethod
    def Point(x=0, y=0, z=0):
        """
        Creates a point (vertex) using the input parameters

        Parameters
        ----------
        x : float , optional.
            The desired x coordinate. Default is 0.
        y : float , optional.
            The desired y coordinate. Default is 0.
        z : float , optional.
            The desired z coordinate. Default is 0.

        Returns
        -------
        topologic_core.Vertex
        """
        
        return Vertex.ByCoordinates(x, y, z)

    @staticmethod
    def Project(vertex, face, direction: list = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the projection of the input vertex onto the supporting geometry of
        the input face. On the PythonOCC backend this operation is delegated to OCCT.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex to project.
        face : topologic_core.Face
            The input face receiving the projection.
        direction : list , optional
            The projection direction. If None, normal/nearest-surface projection is
            used. For a planar face an explicit direction intersects the infinite
            supporting plane. Default is None.
        mantissa : int , optional
            The number of decimal places to round the returned coordinates to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The projected vertex, or None if the projection cannot be computed.
        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.Project - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Vertex.Project - Error: The input face parameter is not a valid face. Returning None.")
            return None

        tol = abs(float(tolerance))
        if direction is not None:
            if not isinstance(direction, (list, tuple)) or len(direction) != 3:
                if not silent:
                    print("Vertex.Project - Error: The input direction parameter is not a valid 3D vector. Returning None.")
                return None
            try:
                direction = [float(direction[0]), float(direction[1]), float(direction[2])]
                if math.sqrt(sum(value * value for value in direction)) <= tol:
                    if not silent:
                        print("Vertex.Project - Error: The input direction vector has zero magnitude. Returning None.")
                    return None
            except Exception:
                if not silent:
                    print("Vertex.Project - Error: The input direction parameter is not a valid numerical vector. Returning None.")
                return None

        native_projection = Core.HasAttribute("VertexUtility", "DistanceToTopology")
        if native_projection and Core.HasAttribute("Vertex", "Project"):
            try:
                projected = Core.Vertex.Project(vertex, face, direction, tol)
            except TypeError:
                try:
                    projected = Core.Vertex.Project(vertex, face, direction)
                except Exception:
                    projected = None
            except Exception:
                projected = None

            if Topology.IsInstance(projected, "Vertex"):
                coords = Vertex.Coordinates(projected, mantissa=mantissa)
                return Vertex.ByCoordinates(coords) if coords is not None else projected

            # If the active backend exposes the native Project implementation,
            # failure is authoritative; do not substitute a different numerical
            # geometry model.
            if Core.HasAttribute("VertexUtility", "DistanceToTopology"):
                if not silent:
                    print("Vertex.Project - Warning: The native backend could not project the vertex. Returning None.")
                return None

        # Legacy TopologicCore compatibility path.
        if direction is None:
            try:
                direction = Face.Normal(face)
            except Exception:
                direction = None
        if direction is None or len(direction) != 3:
            if not silent:
                print("Vertex.Project - Error: Could not determine a valid projection direction. Returning None.")
            return None

        try:
            equation = Face.PlaneEquation(face, mantissa=max(12, mantissa if mantissa is not None else 12))
        except Exception:
            equation = None
        if not isinstance(equation, dict):
            if not silent:
                print("Vertex.Project - Error: Could not determine the supporting plane of the input face. Returning None.")
            return None

        try:
            a = float(equation["a"])
            b = float(equation["b"])
            c = float(equation["c"])
            d = float(equation["d"])
            dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
            denominator = a*dx + b*dy + c*dz
            if abs(denominator) <= tol:
                if not silent:
                    print("Vertex.Project - Warning: The projection direction is parallel to the face. Returning None.")
                return None
            x, y, z = Vertex.Coordinates(vertex, mantissa=None)
            parameter = -(a*x + b*y + c*z + d) / denominator
            coords = [x + parameter*dx, y + parameter*dy, z + parameter*dz]
            if mantissa is not None:
                coords = [round(float(value), mantissa) for value in coords]
            return Vertex.ByCoordinates(coords)
        except Exception:
            if not silent:
                print("Vertex.Project - Error: Could not project the input vertex. Returning None.")
            return None


    @staticmethod
    def Quadrance(vertex, topology, includeCentroid: bool = True, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the quadrance between the input vertex and the input topology.
        Quadrance is the squared Euclidean distance.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        topology : topologic_core.Topology
            The input topology.
        includeCentroid : bool , optional
            If set to True, the centroid of the input topology is also considered.
            Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The squared distance between the input vertex and topology.
        """
        distance = Vertex.Distance(
            vertex,
            topology,
            includeCentroid=includeCentroid,
            mantissa=None,
            tolerance=tolerance,
            silent=silent,
        )
        if distance is None:
            return None
        value = float(distance) * float(distance)
        return round(value, mantissa) if mantissa is not None else value


    @staticmethod
    def RandomVertex(vertices, maxTries: int = 1000, pad: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a random vertex within the bounding box of the input list of vertices,
        ensuring that it is not coincident with any input vertex.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        pad : float , optional
            The desired additional distance to use outside the bounding box of the input list of vertices. Default is 0.0.
        tolerance : float , optional
            The desired tolerance for coincidence checking. The default is 0.0001.
        maxTries : int , optional
            The maximum number of attempts to generate a non-coincident random vertex.
            The default is 1000.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Vertex or None
            A random vertex within the bounding box of the input vertices that is not
            coincident with any of them. Returns None on failure.

        """
        import random
        from topologicpy.BVH import AABB
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            return None

        vertex_list = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertex_list) < 1:
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            z = random.uniform(0, 100)
            return Vertex.ByCoordinates(x, y, z)

        if len(vertex_list) == 1:
            existing_vert = vertex_list[0]
            min_value = max(tolerance, 1)
            max_value = min_value*100
            x_offset = random.uniform(min_value, max_value)
            y_offset = random.uniform(min_value, max_value)
            z_offset = random.uniform(min_value, max_value)
            return_vert = Topology.Translate(existing_vert, x_offset, y_offset, z_offset)
            return return_vert    
        
        pts = [Vertex.Coordinates(v) for v in vertex_list]
        aabb = AABB.from_points(pts=pts, pad=pad)

        # Degenerate case: bounding box collapses to a single point
        if abs(aabb.maxx - aabb.minx) <= tolerance and abs(aabb.maxy - aabb.miny) <= tolerance and abs(aabb.maxz - aabb.minz) <= tolerance:
            if not silent:
                print("Vertex.RandomVertex - Error: Degenerate bounding box. Returning None.")
            return None

        for _ in range(maxTries):
            x = random.uniform(aabb.minx, aabb.maxx)
            y = random.uniform(aabb.miny, aabb.maxy)
            z = random.uniform(aabb.minz, aabb.maxz)
            candidate = Vertex.ByCoordinates(x, y, z)

            is_coincident = False
            for v in vertex_list:
                if Vertex.Distance(candidate, v) <= tolerance:
                    is_coincident = True
                    break

            if not is_coincident:
                return candidate
        if not silent:
            print("Vertex.RandomVertex - Warning: Could not generate a vertex within the allocated number of tries. Try increasing it. Returning None.")
        return None
    
    @staticmethod
    def Separate(*vertices, minDistance: float = 0.0001, iterations: int = 100, strength: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Separates the input vertices such that no two vertices are within the input minimum distance.

        Parameters
        ----------
        vertices : *topologicpy.Vertex
            One or more instances of a topologic vertex to be processed.
        minDistance : float , optional
            The desired minimum distance. Default is 0.0001.
        iterations : int
            The number of iterations to run the repulsion simulation. Default is 100.
        strength : float
            The force multiplier controlling how strongly vertices repel each other. Default is 0.1.
        tolerance : float
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of vertices with adjusted positions

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        from topologicpy.Vertex import Vertex
        import math
        from collections import defaultdict

        # --- Gather & validate inputs ---
        if len(vertices) == 0:
            if not silent:
                print("Vertex.Separate - Error: The input vertices parameter is an empty list. Returning None.")
            return None

        # Allow either a single list or varargs
        if len(vertices) == 1 and isinstance(vertices[0], list):
            raw_list = vertices[0]
        else:
            raw_list = Helper.Flatten(list(vertices))

        vertexList = [v for v in raw_list if Topology.IsInstance(v, "Vertex")]
        if len(vertexList) == 0:
            if not silent:
                print("Vertex.Separate - Error: The input parameters do not contain any valid vertices. Returning None.")
            return None
        if len(vertexList) == 1:
            if not silent:
                print("Vertex.Separate - Warning: Only one vertex supplied. Returning it unchanged.")
            return vertexList

        minDistance = float(minDistance) + float(tolerance)  # safety margin
        n = len(vertexList)

        # Mutable coordinates
        coords = [[Vertex.X(vertexList[i]), Vertex.Y(vertexList[i]), Vertex.Z(vertexList[i])] for i in range(n)]
        dicts  = [Topology.Dictionary(v) for v in vertexList]

        # --- Pre-seed coincident vertices so they can start moving ---
        # Cluster indices by quantized coordinate to catch exact (or near-exact) duplicates
        key_scale = max(tolerance, 1e-12)
        clusters = defaultdict(list)
        for idx, (x, y, z) in enumerate(coords):
            key = (round(x / key_scale), round(y / key_scale), round(z / key_scale))
            clusters[key].append(idx)

        # For any cluster with >1 vertex, spread them on a small circle in XY
        for idxs in clusters.values():
            k = len(idxs)
            if k > 1:
                r = minDistance * 0.5  # small initial spread; repulsion will take it from here
                for m, idx in enumerate(idxs):
                    ang = (2.0 * math.pi * m) / k
                    coords[idx][0] += r * math.cos(ang)
                    coords[idx][1] += r * math.sin(ang)
                    # leave Z unchanged to avoid unintended vertical drift

        # --- Repulsion simulation ---
        eps = 1e-12
        for _ in range(int(iterations)):
            all_ok = True
            for i in range(n):
                xi, yi, zi = coords[i]
                for j in range(i + 1, n):
                    xj, yj, zj = coords[j]
                    dx = xj - xi
                    dy = yj - yi
                    dz = zj - zi
                    dist_sq = dx*dx + dy*dy + dz*dz
                    if dist_sq <= 0.0:
                        # still coincident: nudge with a tiny deterministic push along x
                        dx, dy, dz = (eps, 0.0, 0.0)
                        dist_sq = eps*eps
                    dist = math.sqrt(dist_sq)

                    if dist < minDistance:
                        all_ok = False
                        # Repulsion magnitude; clamp denominator to avoid blow-ups
                        repel = (minDistance - dist) / max(dist, eps) * float(strength)
                        # Split the move equally
                        sx = 0.5 * dx * repel
                        sy = 0.5 * dy * repel
                        sz = 0.5 * dz * repel
                        coords[i][0] -= sx; coords[i][1] -= sy; coords[i][2] -= sz
                        coords[j][0] += sx; coords[j][1] += sy; coords[j][2] += sz
            if all_ok:
                break  # everything already at least minDistance apart

        # --- Rebuild vertices & restore dictionaries ---
        new_vertices = [Vertex.ByCoordinates(x, y, z) for (x, y, z) in coords]
        for i in range(n):
            new_vertices[i] = Topology.SetDictionary(new_vertices[i], dicts[i])

        return new_vertices


    @staticmethod
    def Transform(vertex, matrix, mantissa: int = 6, silent: bool = False, tolerance: float = 0.0001):
        """
        Transforms a 3D vertex using a 4x4 affine transformation matrix.
        The affine transformation is delegated to ``Topology.Transform``, which
        uses the active backend's native transform implementation and preserves
        the legacy TopologicCore fallback.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex.
        matrix : list
            The 4x4 affine transformation matrix.
        mantissa : int , optional
            The number of decimal places to round the returned coordinates to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance for validating the affine matrix. Default is 0.0001.

        Returns
        -------
        topologic_core.Vertex
            The transformed vertex.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.Transform - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None

        transformed = Topology.Transform(
            vertex,
            matrix,
            transferDictionaries=True,
            tolerance=tolerance,
            silent=silent,
        )
        if not Topology.IsInstance(transformed, "Vertex"):
            return None

        if mantissa is None:
            return transformed

        coords = Vertex.Coordinates(transformed, mantissa=mantissa)
        if coords is None:
            return transformed
        rounded_vertex = Vertex.ByCoordinates(coords)
        if not Topology.IsInstance(rounded_vertex, "Vertex"):
            return transformed
        try:
            dictionary = Topology.Dictionary(transformed)
            if dictionary is not None:
                rounded_vertex = Topology.SetDictionary(rounded_vertex, dictionary, silent=True)
        except Exception:
            pass
        return rounded_vertex
    
    @staticmethod
    def Weld(vertices: list, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a list of vertices where vertices within a specified tolerance distance are fused while retaining duplicates, ensuring that vertices with nearly identical coordinates are replaced by a single shared coordinate.

        Parameters
        ----------
        vertices : list
            The input list of topologic vertices.
        mantissa : int , optional
            The desired length of the mantissa for retrieving vertex coordinates. Default is 6.
        tolerance : float , optional
            The desired tolerance for computing if vertices need to be fused. Any vertices that are closer to each other than this tolerance will be fused. Default is 0.0001.

        Returns
        -------
        list
            The list of fused vertices. This list contains the same number of vertices and in the same order as the input list of vertices. However, the coordinates
            of these vertices have now been modified so that they are exactly the same with other vertices that are within the tolerance distance.
        
        """
        return Vertex.Fuse(vertices = vertices, mantissa = mantissa, tolerance = tolerance)
    
    @staticmethod
    def X(vertex, mantissa: int = None, silent: bool = False):
        """
        Returns the X coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex
        mantissa : int , optional
            The desired length of the mantissa for retrieving vertex coordinates. None means no rounding. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        float
            The X coordinate of the input vertex.
        
        """
        from topologicpy.Core import Core
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.X - Error: The input vertex parameter is not a valid vertex. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

        try:
            if mantissa is None:
                return Core.InstanceCall(vertex, "X")
            else:
                return round(Core.InstanceCall(vertex, "X"), mantissa)
        except Exception:
            if not silent:
                print("Vertex.X - Error: Could not retrieve the X coordinate. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

    @staticmethod
    def Y(vertex, mantissa: int = None, silent: bool = False):
        """
        Returns the Y coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex
        mantissa : int , optional
            The desired length of the mantissa for retrieving vertex coordinates. None means no rounding. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        float
            The Y coordinate of the input vertex.
        
        """
        from topologicpy.Core import Core
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.Y - Error: The input vertex parameter is not a valid vertex. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

        try:
            if mantissa is None:
                return Core.InstanceCall(vertex, "Y")
            else:
                return round(Core.InstanceCall(vertex, "Y"), mantissa)
        except Exception:
            if not silent:
                print("Vertex.Y - Error: Could not retrieve the Y coordinate. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

    @staticmethod
    def Z(vertex, mantissa: int = None, silent: bool = False):
        """
        Returns the Z coordinate of the input vertex.

        Parameters
        ----------
        vertex : topologic_core.Vertex
            The input vertex
        mantissa : int , optional
            The desired length of the mantissa for retrieving vertex coordinates. None means no rounding. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        float
            The Z coordinate of the input vertex.
        
        """
        from topologicpy.Core import Core
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Vertex.Z - Error: The input vertex parameter is not a valid vertex. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

        try:
            if mantissa is None:
                return Core.InstanceCall(vertex, "Z")
            else:
                return round(Core.InstanceCall(vertex, "Z"), mantissa)
        except Exception:
            if not silent:
                print("Vertex.Z - Error: Could not retrieve the Y coordinate. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
           
